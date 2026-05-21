"""Network airline revenue management under bid-price controls.

Implements the bid-price control policy analysed in:

    K. Talluri and G. van Ryzin, "An Analysis of Bid-Price Controls for
    Network Revenue Management", *Management Science* 44(11):1577-1593, 1998.

**Model overview**

An airline network is described by:

* *L* flight legs with seat capacities ``c = (c_1, ..., c_L)``;
* *n* origin-destination-fare (ODF) products, each using a subset
  ``A(j) ⊆ {1, ..., L}`` of legs encoded by a binary matrix
  ``a[j, l] ∈ {0, 1}``;
* fares ``f = (f_1, ..., f_n)`` with ``f_j > 0``.

The booking horizon is split into ``T`` discrete periods.  Demand for
product *j* in period *t* is generated from a Gamma-Poisson
("Pólya") process:

    Λ_j     ~ Gamma(α_j, β_j)                     [latent total intensity]
    D_{j,t} ~ Poisson(Λ_j · p_j(t))               [period arrivals]

where ``p_j(t) = (1/T) · Beta((t+0.5)/T ; a_j, b_j)`` is the midpoint
quadrature of a Beta arrival profile -- the same temporal demand
machinery used by :mod:`simopt.models.bertsimas_airline_revenue` and
:mod:`simopt.models.vanryzin_airline_revenue`.

**Bid-price control (Talluri & van Ryzin 1998)**

The control consists of a vector of *bid prices* per period,
``π_t = (π_{1,t}, ..., π_{L,t}) ∈ R_+^L``.  When a request for product *j*
arrives in period *t* it is accepted *iff*

    (i)  every leg ``l ∈ A(j)`` has at least one unit of remaining
         capacity, **and**
    (ii) ``f_j ≥ Σ_{l ∈ A(j)} π_{l,t}`` (the fare clears the sum of
         bid prices on the requested resources).

Otherwise the request is rejected.  This is the canonical *static*
bid-price form analysed by Talluri & van Ryzin (their §3 / Eq. (1));
re-solving ``π_t`` at the start of every period yields the dynamic
form (§4) studied here as the multistage variant.

**Two concrete model classes are provided:**

* :class:`TalluriBidPriceSingleStage` -- a standard :class:`Model` that
  simulates the entire booking horizon in one call to ``replicate()``
  under a stationary bid-price vector ``π ∈ R_+^L``.
* :class:`TalluriBidPriceMultistage` -- a :class:`MultistageModel` where
  each booking period is a separate decision stage.  The bid-price
  vector for the current period is chosen by the solver; Bayesian
  Gamma-Poisson posterior updates inform the cost-to-go through
  posterior-sampling lookahead (cf. Bertsimas & de Boer 2005,
  Van Ryzin & Vulcano 2008).

**Default network (for direct comparison with AIRLINE / VANRYZIN)**

To keep the new family directly comparable to the existing single-leg
parallel implementation in :mod:`simopt.models.bertsimas_airline_revenue`,
the default configuration uses the same 2-leg, 6-product hub-and-spoke
network: legs ``(100, 200)``, products with leg masks
``(L1, L1, L2, L2, L1+L2, L1+L2)`` and fares
``(300, 100, 150, 50, 100, 25)``.

**References**

Talluri, K. and van Ryzin, G. (1998).  An analysis of bid-price controls
for network revenue management.  *Management Science* 44(11), 1577-1593.

Talluri, K. and van Ryzin, G. (2004).  *The Theory and Practice of
Revenue Management*.  Springer.

Bertsimas, D. and de Boer, S. (2005).  Simulation-based booking limits
for airline revenue management.  *Operations Research* 53(1), 90-106.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Self, cast

import numpy as np
from pydantic import BaseModel, Field, model_validator
from scipy.special import gamma as gamma_fn

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import (
    ConstraintType,
    Model,
    MultistageModel,
    MultistageProblem,
    Objective,
    Problem,
    RepResult,
    VariableType,
)

# ======================================================================
# Configuration (Pydantic)
# ======================================================================


class TalluriBidPriceConfigBase(BaseModel):
    """Shared configuration for the Talluri & van Ryzin (1998) bid-price model.

    The network has ``n_classes`` ODF products over ``n_legs`` flight
    legs (inferred from ``ODF_leg_matrix``), simulated across
    ``n_periods`` booking periods.  Default network mirrors the small
    2-leg / 6-product hub-and-spoke example used by the existing
    :mod:`simopt.models.bertsimas_airline_revenue` module so the two
    control families are directly comparable.
    """

    n_classes: Annotated[
        int,
        Field(
            default=6,
            description="number of ODF (origin-destination-fare) products",
            gt=0,
        ),
    ]
    ODF_leg_matrix: Annotated[
        list[list[int]],
        Field(
            # Default: same network as AIRLINE / AIRLINE-ML.  Two legs; products
            # 0-1 use leg 0 only, 2-3 use leg 1 only, 4-5 are connecting
            # itineraries that use both legs.
            default_factory=lambda: [
                [1, 0],
                [1, 0],
                [0, 1],
                [0, 1],
                [1, 1],
                [1, 1],
            ],
            description=(
                "Binary matrix of shape (n_classes, n_legs).  Entry [j, l] "
                "is 1 if product j uses leg l, else 0."
            ),
        ),
    ]
    capacity: Annotated[
        tuple[int, ...],
        Field(
            default=(100, 200),
            description="integer seat capacity for each flight leg",
        ),
    ]
    fares: Annotated[
        tuple[float, ...],
        Field(
            default=(300.0, 100.0, 150.0, 50.0, 100.0, 25.0),
            description="fare (revenue) for each ODF product",
        ),
    ]
    bid_prices: Annotated[
        tuple[float, ...],
        Field(
            # Default: zeros (accept everything subject to capacity).  Solvers
            # override this through the problem's decision vector.
            default=(0.0, 0.0),
            description=(
                "default per-leg bid price vector π ∈ R_+^L applied at every "
                "period (single-stage model) or as the rollout policy "
                "(multistage model).  Length must equal n_legs."
            ),
        ),
    ]
    n_periods: Annotated[
        int,
        Field(
            default=10,
            description="number of booking periods in the horizon (T)",
            gt=0,
        ),
    ]
    booking_period_length: Annotated[
        float,
        Field(
            default=1.0,
            description=(
                "physical length τ of the booking horizon.  The Beta "
                "density is standardised over [0, τ]."
            ),
            gt=0.0,
        ),
    ]
    gamma_shape: Annotated[
        tuple[float, ...],
        Field(
            default=(2.0, 2.0, 2.0, 2.0, 2.0, 2.0),
            description="Gamma shape α_j for total demand Λ_j per product",
        ),
    ]
    gamma_scale: Annotated[
        tuple[float, ...],
        Field(
            default=(50.0, 50.0, 50.0, 50.0, 50.0, 50.0),
            description="Gamma scale β_j for total demand Λ_j per product",
        ),
    ]
    beta_alpha: Annotated[
        tuple[float, ...],
        Field(
            default=(2.0, 2.0, 2.0, 2.0, 2.0, 2.0),
            description="Beta α governing temporal demand shape per product",
        ),
    ]
    beta_beta: Annotated[
        tuple[float, ...],
        Field(
            default=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            description="Beta β governing temporal demand shape per product",
        ),
    ]

    model_config = {"populate_by_name": True}

    # -- helpers -------------------------------------------------------

    def _check_len(
        self, field_name: str, expected: int, label: str = "n_classes"
    ) -> None:
        val = getattr(self, field_name)
        if len(val) != expected:
            raise ValueError(
                f"Length of {field_name} ({len(val)}) must equal {label} ({expected})."
            )

    def _check_nonneg(self, field_name: str) -> None:
        val = getattr(self, field_name)
        if any(v < 0 for v in val):
            raise ValueError(f"All elements in {field_name} must be >= 0.")

    def _check_positive(self, field_name: str) -> None:
        val = getattr(self, field_name)
        if any(v <= 0 for v in val):
            raise ValueError(f"All elements in {field_name} must be > 0.")

    # -- validator -----------------------------------------------------

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        n = self.n_classes
        odf_mat = np.array(self.ODF_leg_matrix)

        if odf_mat.ndim != 2 or odf_mat.shape[0] == 0 or odf_mat.shape[1] == 0:
            raise ValueError("ODF_leg_matrix must be a non-empty 2-D array.")
        if not np.all(np.isin(odf_mat, [0, 1])):
            raise ValueError("ODF_leg_matrix must contain only 0s and 1s.")
        if odf_mat.shape[0] != n:
            raise ValueError(
                f"ODF_leg_matrix has {odf_mat.shape[0]} rows but n_classes is {n}."
            )

        n_legs = odf_mat.shape[1]

        for name in ("fares", "gamma_shape", "gamma_scale", "beta_alpha", "beta_beta"):
            self._check_len(name, n)
            self._check_positive(name)

        self._check_len("capacity", n_legs, label="n_legs")
        self._check_positive("capacity")

        # Bid prices: length = n_legs, non-negative
        self._check_len("bid_prices", n_legs, label="n_legs")
        self._check_nonneg("bid_prices")

        for j in range(n):
            if int(odf_mat[j].sum()) == 0:
                raise ValueError(
                    f"Product {j} does not use any leg "
                    "(row of ODF_leg_matrix is all zeros)."
                )

        return self


class TalluriBidPriceSingleStageConfig(TalluriBidPriceConfigBase):
    """Config for the single-stage (full-horizon) bid-price model."""

    pass


class TalluriBidPriceMultistageConfig(TalluriBidPriceConfigBase):
    """Config for the multistage bid-price model (one period per stage)."""

    pass


# ======================================================================
# Demand helpers (Gamma-Poisson Pólya process)
# ======================================================================


def _beta_density(t: int, n_periods: int, alpha: float, beta: float) -> float:
    """Midpoint Beta-density proportion p_j(t) for period *t*.

    See Talluri & van Ryzin (2004) §9.4 and Bertsimas & de Boer (2005)
    for the temporal-demand-shape derivation.
    """
    T = n_periods  # noqa: N806
    mid = (t + 0.5) / T
    log_beta_fn = (
        np.log(gamma_fn(alpha))
        + np.log(gamma_fn(beta))
        - np.log(gamma_fn(alpha + beta))
    )
    log_density = (alpha - 1) * np.log(mid) + (beta - 1) * np.log(1 - mid) - log_beta_fn
    return float(np.exp(log_density) / T)


def _period_proportions(
    n_periods: int,
    beta_alpha: tuple[float, ...],
    beta_beta: tuple[float, ...],
) -> np.ndarray:
    """Pre-compute demand proportions ``p_j(t)`` for all products and periods."""
    n_classes = len(beta_alpha)
    props = np.zeros((n_classes, n_periods))
    for j in range(n_classes):
        for t in range(n_periods):
            props[j, t] = _beta_density(t, n_periods, beta_alpha[j], beta_beta[j])
        total = props[j].sum()
        if total > 0:
            props[j] /= total
    return props


def _sample_expected_demand(
    rng: MRG32k3a,
    gamma_shape: tuple,
    gamma_scale: tuple,
) -> list[float]:
    """Sample total expected demand Λ_j for each product from its Gamma prior.

    ``MRG32k3a.gammavariate(alpha, beta)`` uses the *scale* parametrisation:
    ``E[X] = alpha * beta``.
    """
    return [
        rng.gammavariate(alpha=s, beta=sc)
        for s, sc in zip(gamma_shape, gamma_scale, strict=True)
    ]


def _generate_period_demand_poisson(
    rng: MRG32k3a,
    expected_demand: list[float],
    proportions: np.ndarray,
    stage: int,
) -> list[int]:
    """Generate integer Poisson demand counts for one booking period.

    ``D_{j,t} ~ Poisson(Λ_j · p_j(t))`` independently across products.
    """
    n_classes = len(expected_demand)
    out: list[int] = []
    for j in range(n_classes):
        mean = float(expected_demand[j] * proportions[j, stage])
        if mean <= 0.0:
            out.append(0)
        else:
            out.append(int(rng.poissonvariate(mean)))
    return out


# ======================================================================
# Bid-price acceptance logic
# ======================================================================


@dataclass
class BidPriceBookingResult:
    """Outcome of processing one booking period under bid-price control.

    Attributes:
        seats_sold: Updated cumulative units sold per product.
        remaining_capacity: Updated remaining seats per leg.
        revenue: Revenue earned in this period.
        accepted: Accepted requests per product in this period.
        rejected_capacity: Requests rejected because some required leg
            was at zero remaining capacity.
        rejected_bidprice: Requests rejected because the fare did not
            clear the sum of leg bid prices (Talluri & van Ryzin Eq. 1).
    """

    seats_sold: list[int]
    remaining_capacity: list[int]
    revenue: float
    accepted: list[int]
    rejected_capacity: list[int]
    rejected_bidprice: list[int]


def _process_bidprice_period(
    selling_rng: MRG32k3a,
    demand: list[int],
    bid_prices: list[float] | tuple[float, ...],
    odf_leg_matrix: np.ndarray,
    seats_sold: list[int],
    remaining_capacity: list[int],
    fares: tuple[float, ...],
) -> BidPriceBookingResult:
    """Process one period's arrivals under bid-price control.

    Implements the Talluri & van Ryzin (1998) acceptance rule.  Within
    each period the *individual* arrivals (one per unit of Poisson demand)
    are processed in a uniformly random order across products so that
    no product is privileged when capacity becomes binding -- consistent
    with the i.i.d. arrival assumption of the underlying NHPP.

    The arguments ``seats_sold`` and ``remaining_capacity`` are mutated
    in place; copies are also returned on the result dataclass.

    Args:
        selling_rng: RNG used to shuffle the arrival order.
        demand: Integer Poisson arrival counts per product for this period.
        bid_prices: Per-leg bid prices π ∈ R_+^L for this period.
        odf_leg_matrix: Binary product-leg adjacency, shape (n, L).
        seats_sold: Cumulative units sold per product (mutated in place).
        remaining_capacity: Remaining seats per leg (mutated in place).
        fares: Per-product fare vector.

    Returns:
        :class:`BidPriceBookingResult` summarising the period.
    """
    n_classes = len(demand)
    n_legs = len(remaining_capacity)
    if len(bid_prices) != n_legs:
        raise ValueError(
            f"bid_prices has length {len(bid_prices)}; expected {n_legs}."
        )

    # Solver-injected states (e.g., the ADP value-function fitter samples
    # remaining-capacity from a continuous LHS / heuristic distribution that
    # was designed for the fluid Van Ryzin & Vulcano (2008) model) may pass
    # in fractional capacities.  Talluri & van Ryzin (1998) is an integer-
    # seat model: enforce that contract here by flooring on entry.  This
    # is a no-op when capacities are already integer and prevents the
    # downstream non-negativity check from firing on harmless fractional
    # drift between sampled and decremented capacity.
    for l_idx in range(n_legs):
        remaining_capacity[l_idx] = int(max(0, int(remaining_capacity[l_idx])))

    # Pre-compute per-product sum of bid prices over the legs the product uses
    legs_per_product: list[list[int]] = [
        np.nonzero(odf_leg_matrix[j])[0].tolist() for j in range(n_classes)
    ]
    sum_pi: list[float] = [
        sum(float(bid_prices[l]) for l in legs_per_product[j]) for j in range(n_classes)
    ]

    # Build an interleaved arrival stream: each integer arrival becomes one
    # discrete decision in the period.  We then shuffle the stream.
    total = int(sum(int(d) for d in demand))
    stream: list[int] = []
    for j in range(n_classes):
        stream.extend([j] * int(demand[j]))

    # Fisher-Yates shuffle (MRG32k3a-driven) so identical seeds reproduce.
    for i in range(total - 1, 0, -1):
        swap = int(selling_rng.uniform(0, i + 1)) % (i + 1)
        stream[i], stream[swap] = stream[swap], stream[i]

    accepted = [0] * n_classes
    rej_cap = [0] * n_classes
    rej_bp = [0] * n_classes
    period_revenue = 0.0

    for j in stream:
        # (i) capacity check on every leg used by product j
        capacity_ok = True
        for l in legs_per_product[j]:  # noqa: E741
            if remaining_capacity[l] <= 0:
                capacity_ok = False
                break
        if not capacity_ok:
            rej_cap[j] += 1
            continue

        # (ii) bid-price (revenue) test
        if fares[j] < sum_pi[j]:
            rej_bp[j] += 1
            continue

        # Accept: decrement capacity on each leg, record sale and revenue.
        for l in legs_per_product[j]:  # noqa: E741
            remaining_capacity[l] -= 1
        seats_sold[j] += 1
        accepted[j] += 1
        period_revenue += float(fares[j])

    # Defensive non-negativity check
    for l_idx, cap in enumerate(remaining_capacity):
        if cap < 0:
            raise ValueError(
                "Bid-price acceptance produced negative remaining_capacity"
                f"[{l_idx}]={cap}."
            )

    return BidPriceBookingResult(
        seats_sold=seats_sold,
        remaining_capacity=remaining_capacity,
        revenue=period_revenue,
        accepted=accepted,
        rejected_capacity=rej_cap,
        rejected_bidprice=rej_bp,
    )


# ======================================================================
# Single-stage model (entire booking horizon in one call)
# ======================================================================


class TalluriBidPriceSingleStage(Model):
    """Talluri & van Ryzin (1998) bid-price control -- full horizon in one call.

    Simulates the full booking horizon of ``n_periods`` periods under a
    *stationary* per-leg bid-price vector ``π ∈ R_+^L``.  The decision
    variables for the corresponding :class:`TalluriBidPriceSingleStageProblem`
    are the components of π.
    """

    class_name_abbr: ClassVar[str] = "TALLURI-SL"
    class_name: ClassVar[str] = "Bid-Price Airline Revenue (Single Stage)"
    config_class: ClassVar[type[BaseModel]] = TalluriBidPriceSingleStageConfig
    n_rngs: ClassVar[int] = 2  # demand RNG + selling-order RNG
    n_responses: ClassVar[int] = 1

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize instance."""
        super().__init__(fixed_factors)
        self._demand_rng: MRG32k3a | None = None
        self._selling_rng: MRG32k3a | None = None
        self._odf_leg_matrix = np.array(self.factors["ODF_leg_matrix"])

    def before_replicate(self, rng_list: list[MRG32k3a]) -> None:  # noqa: D102
        self._demand_rng = rng_list[0]
        self._selling_rng = rng_list[1]

    def replicate(self) -> tuple[dict, dict]:
        """Simulate the entire booking horizon under stationary bid prices.

        Returns:
            tuple:
                - dict: Performance measures.  Contains ``revenue`` (total
                  network revenue), plus per-product / per-leg diagnostics
                  ``seats_sold``, ``remaining_capacity``, ``accepted``,
                  ``rejected_capacity``, ``rejected_bidprice``.
                - dict: Empty gradient dict.
        """
        if self._demand_rng is None or self._selling_rng is None:
            raise RuntimeError(
                "before_replicate() must be called before replicate()."
            )

        n_classes: int = self.factors["n_classes"]
        capacity: tuple = self.factors["capacity"]
        fares: tuple = self.factors["fares"]
        bid_prices: tuple = self.factors["bid_prices"]
        n_periods: int = self.factors["n_periods"]
        gamma_shape: tuple = self.factors["gamma_shape"]
        gamma_scale: tuple = self.factors["gamma_scale"]
        beta_alpha: tuple = self.factors["beta_alpha"]
        beta_beta: tuple = self.factors["beta_beta"]

        proportions = _period_proportions(n_periods, beta_alpha, beta_beta)

        # Sample the episode's "true" Λ once.
        expected_demand = _sample_expected_demand(
            self._demand_rng, gamma_shape, gamma_scale
        )

        seats_sold = [0] * n_classes
        remaining_capacity = list(capacity)
        total_accepted = [0] * n_classes
        total_rej_cap = [0] * n_classes
        total_rej_bp = [0] * n_classes
        total_revenue = 0.0

        for t in range(n_periods):
            demand = _generate_period_demand_poisson(
                self._demand_rng, expected_demand, proportions, t
            )
            result = _process_bidprice_period(
                self._selling_rng,
                demand,
                bid_prices,
                self._odf_leg_matrix,
                seats_sold,
                remaining_capacity,
                fares,
            )
            seats_sold = result.seats_sold
            remaining_capacity = result.remaining_capacity
            total_revenue += result.revenue
            for j in range(n_classes):
                total_accepted[j] += result.accepted[j]
                total_rej_cap[j] += result.rejected_capacity[j]
                total_rej_bp[j] += result.rejected_bidprice[j]

        responses = {
            "revenue": total_revenue,
            "seats_sold": dict(enumerate(seats_sold)),
            "remaining_capacity": dict(enumerate(remaining_capacity)),
            "accepted": dict(enumerate(total_accepted)),
            "rejected_capacity": dict(enumerate(total_rej_cap)),
            "rejected_bidprice": dict(enumerate(total_rej_bp)),
        }
        return responses, {}


# ======================================================================
# Multistage model (one booking period per decision stage)
# ======================================================================


class TalluriBidPriceMultistage(MultistageModel):
    """Talluri & van Ryzin (1998) bid-price control -- one period per stage.

    Each stage corresponds to one booking period.  The decision at stage
    *t* is the per-leg bid-price vector ``π_t ∈ R_+^L`` to apply for the
    arrivals in that period.

    * **State**: ``dict`` with ``remaining_capacity`` (list[int], L),
      ``seats_sold`` (list[int], n), ``expected_demand`` (list[float], n;
      empty signals "draw a fresh Λ"), Gamma posterior parameters
      (``gamma_shape_posterior``, ``gamma_rate_posterior``, each list[float]
      of length n), and ``total_requests`` (list[int], n).
    * **Transition**: generate Poisson arrivals for the period, process
      them under the current bid prices, update the Gamma posterior.
    * **Stage reward**: revenue earned in that period.
    * **Default policy**: returns the configured stationary ``bid_prices``
      vector at every stage.
    * **Lookahead**: posterior-sampling Pólya rollout -- each Monte Carlo
      replication draws a fresh Λ from the current Gamma posterior and
      uses it for the remaining stages (cf. Van Ryzin & Vulcano 2008
      §3.3 and Bertsimas & de Boer 2005).
    """

    class_name_abbr: ClassVar[str] = "TALLURI-ML"
    class_name: ClassVar[str] = "Bid-Price Airline Revenue (Multistage)"
    config_class: ClassVar[type[BaseModel]] = TalluriBidPriceMultistageConfig
    n_rngs: ClassVar[int] = 2  # demand RNG + selling-order RNG
    n_responses: ClassVar[int] = 1
    n_stages: ClassVar[int] = 10  # overridden in __init__ from config

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize instance."""
        super().__init__(fixed_factors)
        # Instance-scope override so different instances can have different T.
        self.__dict__["n_stages"] = self.factors["n_periods"]
        self._proportions = _period_proportions(
            self.factors["n_periods"],
            self.factors["beta_alpha"],
            self.factors["beta_beta"],
        )
        self._odf_leg_matrix = np.array(self.factors["ODF_leg_matrix"])
        self._demand_rng: MRG32k3a | None = None
        self._selling_rng: MRG32k3a | None = None
        self._episode_expected_demand: list[float] = []
        self._last_booking_result: BidPriceBookingResult | None = None

    # ---- MultistageModel abstract interface --------------------------

    def get_initial_state(self) -> dict[str, Any]:  # noqa: D102
        n_classes: int = self.factors["n_classes"]
        gamma_shape: tuple = self.factors["gamma_shape"]
        gamma_scale: tuple = self.factors["gamma_scale"]
        return {
            "remaining_capacity": list(self.factors["capacity"]),
            "seats_sold": [0] * n_classes,
            "expected_demand": [],
            "gamma_shape_posterior": list(gamma_shape),
            "gamma_rate_posterior": [1.0 / sc for sc in gamma_scale],
            "total_requests": [0] * n_classes,
        }

    def before_replication(self, rng_list: list[MRG32k3a]) -> None:
        """Store RNGs and sample the episode's true Λ."""
        super().before_replication(rng_list)
        self._demand_rng = rng_list[0]
        self._selling_rng = rng_list[1]
        self._episode_expected_demand = _sample_expected_demand(
            self._demand_rng,
            self.factors["gamma_shape"],
            self.factors["gamma_scale"],
        )

    def transition(
        self,
        state: object,
        decision: tuple,
        stage: int,
        rng_list: list[MRG32k3a],
    ) -> object:
        """Simulate one booking period under bid-price control.

        Args:
            state: Current state dict.
            decision: Per-leg bid-price vector for this period (length L).
            stage: Period index t ∈ {0, ..., T-1}.
            rng_list: ``[demand_rng, selling_rng]``.

        Returns:
            Next-state dict (deep copy of the in/out fields).
        """
        if not isinstance(state, dict):
            raise TypeError("state must be a dict.")
        state_dict = cast(dict[str, Any], state)

        demand_rng = rng_list[0]
        selling_rng = rng_list[1]

        n_classes: int = self.factors["n_classes"]

        # Choose Λ to drive demand for this period: the state's
        # ``expected_demand`` overrides the episode default when set (used
        # by posterior-sampling lookahead).
        state_expected = cast(list[float], state_dict.get("expected_demand", []))
        expected_demand: list[float] = (
            state_expected if state_expected else list(self._episode_expected_demand)
        )

        demand = _generate_period_demand_poisson(
            demand_rng, expected_demand, self._proportions, stage
        )

        new_seats_sold = list(cast(list[int], state_dict["seats_sold"]))
        new_remaining_capacity = list(
            cast(list[int], state_dict["remaining_capacity"])
        )

        result = _process_bidprice_period(
            selling_rng,
            demand,
            list(decision),
            self._odf_leg_matrix,
            new_seats_sold,
            new_remaining_capacity,
            self.factors["fares"],
        )
        self._last_booking_result = result

        # ------------------------------------------------------------
        # Bayesian Gamma-Poisson conjugate update.
        # Prior: Λ_j ~ Gamma(α_j, β_j)  (rate parametrisation).
        # Observation: D_{j,t} ~ Poisson(Λ_j · p_j(t)).
        # Posterior: α_j ← α_j + D_{j,t};  β_j ← β_j + p_j(t).
        # ------------------------------------------------------------
        state_gamma_shape = cast(
            list[float], state_dict.get("gamma_shape_posterior", [])
        )
        state_gamma_rate = cast(
            list[float], state_dict.get("gamma_rate_posterior", [])
        )
        state_total_requests = cast(list[int], state_dict.get("total_requests", []))
        new_gamma_shape = (
            list(state_gamma_shape)
            if state_gamma_shape
            else list(self.factors["gamma_shape"])
        )
        new_gamma_rate = (
            list(state_gamma_rate)
            if state_gamma_rate
            else [1.0 / sc for sc in self.factors["gamma_scale"]]
        )
        new_total_requests = (
            list(state_total_requests) if state_total_requests else [0] * n_classes
        )

        for j in range(n_classes):
            new_gamma_shape[j] += float(demand[j])
            new_gamma_rate[j] += float(self._proportions[j, stage])
            new_total_requests[j] += int(demand[j])

        return {
            "remaining_capacity": result.remaining_capacity,
            "seats_sold": result.seats_sold,
            "expected_demand": list(expected_demand),
            "gamma_shape_posterior": new_gamma_shape,
            "gamma_rate_posterior": new_gamma_rate,
            "total_requests": new_total_requests,
        }

    def stage_reward(  # noqa: D102
        self,
        state: object,
        decision: tuple,  # noqa: ARG002
        next_state: object,
        stage: int,  # noqa: ARG002
    ) -> dict[str, float]:
        if not isinstance(state, dict) or not isinstance(next_state, dict):
            raise TypeError("state and next_state must be dicts.")
        state_dict = cast(dict[str, Any], state)
        next_state_dict = cast(dict[str, Any], next_state)

        n_classes = self.factors["n_classes"]
        fares = self.factors["fares"]
        new_sales = [
            int(cast(list[int], next_state_dict["seats_sold"])[j])
            - int(cast(list[int], state_dict["seats_sold"])[j])
            for j in range(n_classes)
        ]
        revenue = float(sum(f * s for f, s in zip(fares, new_sales, strict=True)))
        return {"revenue": revenue}

    def get_default_policy(self) -> Callable[[dict[str, Any], int], tuple]:
        """Return a stationary policy that uses ``factors['bid_prices']`` each stage."""
        default_decision = tuple(float(p) for p in self.factors["bid_prices"])

        def _policy(state: dict[str, Any], stage: int) -> tuple:  # noqa: ARG001
            return default_decision

        return _policy

    # ---- Replicate override: attach per-period diagnostics ----------

    def replicate(
        self,
        state: Any,  # noqa: ANN401
        decision: tuple,
        stage: int,
        policy: Callable[[Any, int], tuple] | None = None,
        n_lookahead_reps: int = 30,
        future_responses: dict[str, float] | None = None,
    ) -> tuple[dict[str, float], dict]:
        """Replicate the current stage and attach booking diagnostics."""
        responses, gradients = super().replicate(
            state,
            decision,
            stage,
            policy,
            n_lookahead_reps,
            future_responses=future_responses,
        )
        rich = cast(dict[str, Any], responses)
        n_classes = self.factors["n_classes"]
        br = self._last_booking_result
        if br is not None:
            rich["seats_sold"] = dict(enumerate(br.seats_sold))
            rich["remaining_capacity"] = dict(enumerate(br.remaining_capacity))
            rich["accepted"] = dict(enumerate(br.accepted))
            rich["rejected_capacity"] = dict(enumerate(br.rejected_capacity))
            rich["rejected_bidprice"] = dict(enumerate(br.rejected_bidprice))
        else:
            rich["seats_sold"] = dict(enumerate([0] * n_classes))
            rich["remaining_capacity"] = dict(
                enumerate(list(self.factors["capacity"]))
            )
            rich["accepted"] = dict(enumerate([0] * n_classes))
            rich["rejected_capacity"] = dict(enumerate([0] * n_classes))
            rich["rejected_bidprice"] = dict(enumerate([0] * n_classes))
        return cast(dict[str, float], rich), gradients

    # ---- Posterior-sampling lookahead -------------------------------

    def simulate_lookahead(
        self,
        state: object,
        start_stage: int,
        policy: Callable[[Any, int], tuple] | None = None,
        n_reps: int = 30,
    ) -> dict[str, float]:
        """Estimate cost-to-go with Λ drawn from the Gamma posterior.

        Each MC replication draws a fresh ``Λ`` from the current Gamma
        posterior in *state* and uses it as ``expected_demand`` for all
        remaining stages.  This is the Pólya-process predictive scheme
        used by the Bertsimas / Van Ryzin multistage variants.
        """
        if start_stage >= self.n_stages:
            return {}
        if not isinstance(state, dict):
            raise TypeError("state must be a dict.")
        state_dict = cast(dict[str, Any], state)

        policy_fn = cast(
            Callable[[Any, int], tuple],
            policy or self.get_default_policy(),
        )
        if self._demand_rng is None:
            raise RuntimeError(
                "before_replication() must be called before simulate_lookahead()."
            )

        n_classes: int = self.factors["n_classes"]
        total_responses: dict[str, float] = {}

        for _ in range(n_reps):
            lookahead_lambda = [
                self._demand_rng.gammavariate(
                    alpha=cast(list[float], state_dict["gamma_shape_posterior"])[j],
                    beta=1.0
                    / cast(list[float], state_dict["gamma_rate_posterior"])[j],
                )
                for j in range(n_classes)
            ]

            sim_state = deepcopy(state_dict)
            sim_state["expected_demand"] = lookahead_lambda

            rep_responses: dict[str, float] = {}
            for t in range(start_stage, self.n_stages):
                decision = policy_fn(sim_state, t)
                stage_resp, sim_state = self.replicate_stage(
                    sim_state, decision, t, self._rng_list
                )
                for key, val in stage_resp.items():
                    rep_responses[key] = rep_responses.get(key, 0.0) + val

            for key, val in rep_responses.items():
                total_responses[key] = total_responses.get(key, 0.0) + val

        if n_reps > 0:
            for key in total_responses:
                total_responses[key] /= n_reps

        return total_responses


# ======================================================================
# Problem configurations
# ======================================================================


class TalluriBidPriceSingleStageProblemConfig(BaseModel):
    """Config for the single-stage bid-price optimisation problem.

    Decision variables: stationary per-leg bid prices ``π ∈ R_+^L``.
    """

    initial_solution: Annotated[
        tuple[float, ...],
        Field(
            default=(0.0, 0.0),
            description=(
                "initial per-leg bid prices π (length n_legs); zeros mean "
                "'accept everything subject to capacity'."
            ),
        ),
    ]
    budget: Annotated[
        int,
        Field(
            default=10000,
            description="maximum replications for the solver",
            gt=0,
            json_schema_extra={"isDatafarmable": False},
        ),
    ]


class TalluriBidPriceMultistageProblemConfig(BaseModel):
    """Config for the multistage bid-price optimisation problem.

    Decision variables at every stage: per-leg bid prices ``π_t ∈ R_+^L``.
    """

    initial_solution: Annotated[
        tuple[float, ...],
        Field(
            default=(0.0, 0.0),
            description=(
                "initial per-leg bid prices π applied at every stage "
                "(length n_legs)."
            ),
        ),
    ]
    budget: Annotated[
        int,
        Field(
            default=10000,
            description="maximum replications for the solver",
            gt=0,
            json_schema_extra={"isDatafarmable": False},
        ),
    ]
    n_lookahead_reps: Annotated[
        int,
        Field(
            default=30,
            description="MC replications for cost-to-go estimation",
            gt=0,
        ),
    ]


# ======================================================================
# Single-stage problem
# ======================================================================


class TalluriBidPriceSingleStageProblem(Problem):
    """Maximise expected total revenue by choosing stationary bid prices.

    Decision vector: ``(π_1, ..., π_L) ∈ R_+^L``.  The same bid-price
    vector is applied at every period of the booking horizon.

    Bounds: each π_l ∈ [0, max(fares)].  Bid prices strictly above the
    maximum fare on any product using leg *l* trivially block all sales
    on that leg, so no useful policy exceeds ``max(fares)``.
    """

    class_name_abbr: ClassVar[str] = "TALLURI-1"
    class_name: ClassVar[str] = "Max Revenue Bid-Price Airline (Single Stage)"
    config_class: ClassVar[type[BaseModel]] = TalluriBidPriceSingleStageProblemConfig
    model_class: ClassVar[type[Model]] = TalluriBidPriceSingleStage
    n_objectives: ClassVar[int] = 1
    n_stochastic_constraints: ClassVar[int] = 0
    minmax: ClassVar[tuple[int, ...]] = (1,)  # maximise revenue
    constraint_type: ClassVar[ConstraintType] = ConstraintType.BOX
    variable_type: ClassVar[VariableType] = VariableType.CONTINUOUS
    gradient_available: ClassVar[bool] = False
    optimal_solution: tuple | None = None
    model_default_factors: ClassVar[dict] = {}
    model_decision_factors: ClassVar[set[str]] = {"bid_prices"}

    @property
    def optimal_value(self) -> float | None:  # noqa: D102
        return None

    @property
    def dim(self) -> int:  # noqa: D102
        return len(self.model.factors["capacity"])

    @property
    def lower_bounds(self) -> tuple:  # noqa: D102
        return (0.0,) * self.dim

    @property
    def upper_bounds(self) -> tuple:  # noqa: D102
        max_fare = float(max(self.model.factors["fares"]))
        return (max_fare,) * self.dim

    def vector_to_factor_dict(self, vector: tuple) -> dict:  # noqa: D102
        return {"bid_prices": tuple(float(v) for v in vector)}

    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:  # noqa: D102
        return tuple(float(v) for v in factor_dict["bid_prices"])

    def replicate(self, x: tuple) -> RepResult:  # noqa: ARG002, D102
        responses, _ = self.model.replicate()
        objectives = [Objective(stochastic=float(responses["revenue"]))]
        return RepResult(objectives=objectives)

    def check_deterministic_constraints(self, x: tuple) -> bool:
        """Non-negative bid prices within the box bounds."""
        for v, lb, ub in zip(x, self.lower_bounds, self.upper_bounds, strict=False):
            if v < lb or v > ub:
                return False
        return True

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:  # noqa: D102
        max_fare = float(max(self.model.factors["fares"]))
        return tuple(rand_sol_rng.uniform(0.0, max_fare) for _ in range(self.dim))


# ======================================================================
# Multistage problem
# ======================================================================


class TalluriBidPriceMultistageProblem(MultistageProblem):
    """Maximise expected total revenue by choosing per-period bid prices.

    At each booking period the solver selects a per-leg bid-price vector
    ``π_t``.  The objective is the immediate period revenue plus the
    Monte-Carlo estimated remaining-horizon revenue under the default
    (stationary) rollout policy (posterior-sampling lookahead).
    """

    class_name_abbr: ClassVar[str] = "TALLURI-2"
    class_name: ClassVar[str] = "Max Revenue Bid-Price Airline (Multistage)"
    config_class: ClassVar[type[BaseModel]] = TalluriBidPriceMultistageProblemConfig
    model_class: ClassVar[type[MultistageModel]] = TalluriBidPriceMultistage
    n_objectives: ClassVar[int] = 1
    n_stochastic_constraints: ClassVar[int] = 0
    minmax: ClassVar[tuple[int, ...]] = (1,)  # maximise
    constraint_type: ClassVar[ConstraintType] = ConstraintType.BOX
    variable_type: ClassVar[VariableType] = VariableType.CONTINUOUS
    gradient_available: ClassVar[bool] = False
    model_default_factors: ClassVar[dict] = {}
    model_decision_factors: ClassVar[set[str]] = {"bid_prices"}
    n_lookahead_reps: ClassVar[int] = 30

    @property
    def optimal_value(self) -> float | None:  # noqa: D102
        return None

    @property
    def dim(self) -> int:  # noqa: D102
        return len(self.model.factors["capacity"])

    @property
    def lower_bounds(self) -> tuple:  # noqa: D102
        return (0.0,) * self.dim

    @property
    def upper_bounds(self) -> tuple:  # noqa: D102
        max_fare = float(max(self.model.factors["fares"]))
        return (max_fare,) * self.dim

    def vector_to_factor_dict(self, vector: tuple) -> dict:  # noqa: D102
        return {"bid_prices": tuple(float(v) for v in vector)}

    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:  # noqa: D102
        return tuple(float(v) for v in factor_dict["bid_prices"])

    def replicate(self, x: tuple) -> RepResult:  # noqa: D102
        responses, _ = self.model.replicate(
            state=self.current_state,
            decision=x,
            stage=self.current_stage,
            n_lookahead_reps=self.n_lookahead_reps,
        )
        objectives = [Objective(stochastic=float(responses["total_revenue"]))]
        return RepResult(objectives=objectives)

    def check_deterministic_constraints(self, x: tuple) -> bool:
        """Non-negative bid prices within the box bounds."""
        for v, lb, ub in zip(x, self.lower_bounds, self.upper_bounds, strict=False):
            if v < lb or v > ub:
                return False
        return True

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:  # noqa: D102
        max_fare = float(max(self.model.factors["fares"]))
        return tuple(rand_sol_rng.uniform(0.0, max_fare) for _ in range(self.dim))
