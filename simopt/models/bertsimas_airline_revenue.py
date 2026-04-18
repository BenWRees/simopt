"""Network airline revenue management from Bertsimas & de Boer (2005).

Implements the network model described in:

    D. Bertsimas and S. de Boer, "Simulation-Based Booking Limits for
    Airline Revenue Management", *Operations Research* 53(1):90-106, 2005.

**Model overview**

An airline network consists of one or more flight *legs*, each with its own
seat capacity.  Passengers are grouped into *ODF* (origin-destination-fare)
classes; each ODF class may use one or more legs, as described by a binary
ODF-leg adjacency matrix.  Over a booking horizon of *T* discrete time
periods, the airline controls revenue through per-ODF *booking limits*:
class *j* is available for sale only when ``seats_sold[j] < booking_limit[j]``
and every leg it uses still has remaining capacity.

**Demand model (Pólya / Gamma-Poisson)**

Demand for each ODF class follows a Pólya (Gamma-Poisson) process.  The
total expected demand for class *j* over the full booking period is a random
variable Λ_j drawn from a Gamma prior:

    Λ_j ~ Gamma(alpha_j, beta_j)

The booking period has physical length tau (configurable via the
``booking_period_length`` factor).  Within the horizon the arrival intensity
at continuous time *s ∈ [0, tau]* is

    lambda_j(s) = Λ_j * f(s/tau; a_j, b_j) / tau

where *f(*; a, b)* is the standard Beta density with shape parameters
*(a_j, b_j)*.  After discretisation into *T* periods, the number of
requests for class *j* in period *t* is

    D_{j,t} ~ Poisson(Λ_j * p_j(t))

where p_j(t) = (1/T) * f((t + 0.5)/T; a_j, b_j) is the midpoint-
quadrature proportion of demand in period *t*.

**Bayesian Gamma-Poisson conjugate updates (multistage)**

In the multistage formulation, the airline's belief about Λ_j is updated
after each period via the Gamma-Poisson conjugate relationship.  Let the
current posterior be Λ_j ~ Gamma(alpha_j, beta_j) using the *rate*
parametrisation.  After observing D_{j,t} booking requests in period *t*:

    alpha_j  ←  alpha_j + D_{j,t}
    beta_j  ←  beta_j + p_j(t)

The posterior mean (MMSE estimate) of Λ_j is alpha_j / beta_j.  For lookahead
(Monte Carlo cost-to-go estimation), a fresh Λ is drawn from the current
posterior at the start of each replication.

Two concrete classes are provided:

* :class:`AirlineRevenueSingleStage` -- a standard :class:`Model` that
  simulates the entire booking horizon in one call to ``replicate()``.
* :class:`AirlineRevenueMultistage` -- a :class:`MultistageModel` where each
  time period is a separate decision stage, with Bayesian demand updates
  and posterior-based lookahead.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field, fields
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
# State dataclass used by the multistage model
# ======================================================================


@dataclass
class AirlineState:
    """State of the network booking process at the start of a period.

    Supports both attribute access (``state.remaining_capacity``) and
    dict-style access (``state["remaining_capacity"]``, ``"key" in state``)
    so that solvers written against a dict-based state interface work
    transparently.

    Attributes:
        remaining_capacity: Unsold seats per flight leg.
        seats_sold: Cumulative seats sold per ODF class.
        expected_demand: Sampled Λ_j values used for demand generation.
            In the main episode this is the "true" Λ drawn at the start;
            in a lookahead replication it is freshly sampled from the
            posterior.  Empty list signals that a new Λ should be drawn.
        gamma_shape_posterior: Current Gamma posterior shape parameter per
            ODF class (initialised to the prior shape alpha_j).
        gamma_rate_posterior: Current Gamma posterior *rate* parameter per
            ODF class (initialised to 1 / scale_j).  Rate parametrisation
            is used because the Gamma-Poisson conjugate update simply adds
            observed counts to alpha and cumulative intensity proportions to beta.
        total_requests: Total booking *requests* (arrivals) observed per
            ODF class up to the current period (includes rejected requests).
    """

    remaining_capacity: list[int] = field(default_factory=list)
    seats_sold: list[int] = field(default_factory=list)
    expected_demand: list[float] = field(default_factory=list)
    gamma_shape_posterior: list[float] = field(default_factory=list)
    gamma_rate_posterior: list[float] = field(default_factory=list)
    total_requests: list[int] = field(default_factory=list)

    # -- Dict-like access for solver compatibility --

    def __contains__(self, key: str) -> bool:
        """Support ``"key" in state`` checks."""
        return hasattr(self, key)

    def __getitem__(self, key: str) -> object:
        """Support ``state["key"]`` reads."""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def __setitem__(self, key: str, value: object) -> None:
        """Support ``state["key"] = value`` writes."""
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise KeyError(f"{type(self).__name__} has no field '{key}'")

    def keys(self) -> list[str]:
        """Return field names, matching the dict state interface."""
        return [f.name for f in fields(self)]

    def get(self, key: str, default: object = None) -> object:
        """Support ``state.get("key", default)``."""
        return getattr(self, key, default)


# ======================================================================
# Pydantic configuration models
# ======================================================================


class AirlineRevenueConfigBase(BaseModel):
    """Shared configuration for the network airline revenue model.

    The model is defined over a network of flight *legs*.  Each
    origin-destination-fare (ODF) class may use one or more legs, as
    specified by the binary ``ODF_leg_matrix``.
    """

    n_classes: Annotated[
        int,
        Field(
            default=6,
            description="number of ODF (origin-destination-fare) classes",
            gt=0,
        ),
    ]
    ODF_leg_matrix: Annotated[
        list[list[int]],
        Field(
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
                "is 1 if ODF class j uses leg l, else 0."
            ),
        ),
    ]
    capacity: Annotated[
        tuple[int, ...],
        Field(
            default=(100, 200),
            description="seat capacity for each flight leg",
        ),
    ]
    fares: Annotated[
        tuple[float, ...],
        Field(
            default=(300.0, 100.0, 150.0, 50.0, 100.0, 25.0),
            description="fare (price) for each ODF class",
        ),
    ]
    booking_limits: Annotated[
        tuple[int, ...],
        Field(
            default=(10, 10, 10, 10, 10, 10),
            description="booking limit for each ODF class",
        ),
    ]
    n_periods: Annotated[
        int,
        Field(
            default=10,
            description="number of time periods in the booking horizon",
            gt=0,
        ),
    ]
    booking_period_length: Annotated[
        float,
        Field(
            default=1.0,
            description=(
                "physical length tau of the booking period.  The Beta "
                "density that shapes temporal demand is standardised "
                "over [0, tau]."
            ),
            gt=0.0,
        ),
    ]
    gamma_shape: Annotated[
        tuple[float, ...],
        Field(
            default=(2.0, 2.0, 2.0, 2.0, 2.0, 2.0),
            description="Gamma shape parameter for total demand per ODF class",
        ),
    ]
    gamma_scale: Annotated[
        tuple[float, ...],
        Field(
            default=(50.0, 50.0, 50.0, 50.0, 50.0, 50.0),
            description="Gamma scale parameter for total demand per ODF class",
        ),
    ]
    beta_alpha: Annotated[
        tuple[float, ...],
        Field(
            default=(2.0, 2.0, 2.0, 2.0, 2.0, 2.0),
            description="Beta alpha parameter governing temporal demand shape per ODF class",  # noqa: E501
        ),
    ]
    beta_beta: Annotated[
        tuple[float, ...],
        Field(
            default=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            description="Beta beta parameter governing temporal demand shape per ODF class",  # noqa: E501
        ),
    ]

    model_config = {"populate_by_name": True}

    # -- Validators --

    def _check_len(
        self, field_name: str, expected: int, label: str = "n_classes"
    ) -> None:
        val = getattr(self, field_name)
        if len(val) != expected:
            raise ValueError(
                f"Length of {field_name} ({len(val)}) must equal {label} ({expected})."
            )

    def _check_positive(self, field_name: str) -> None:
        val = getattr(self, field_name)
        if any(v <= 0 for v in val):
            raise ValueError(f"All elements in {field_name} must be > 0.")

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        n = self.n_classes
        odf_mat = np.array(self.ODF_leg_matrix)

        # ODF matrix shape checks
        if odf_mat.ndim != 2 or odf_mat.shape[0] == 0 or odf_mat.shape[1] == 0:
            raise ValueError("ODF_leg_matrix must be a non-empty 2-D array.")
        if not np.all(np.isin(odf_mat, [0, 1])):
            raise ValueError("ODF_leg_matrix must contain only 0s and 1s.")
        if odf_mat.shape[0] != n:
            raise ValueError(
                f"ODF_leg_matrix has {odf_mat.shape[0]} rows but n_classes is {n}."
            )

        n_legs = odf_mat.shape[1]

        # Per-ODF-class arrays
        for name in (
            "fares",
            "booking_limits",
            "gamma_shape",
            "gamma_scale",
            "beta_alpha",
            "beta_beta",
        ):
            self._check_len(name, n)
        for name in ("fares", "gamma_shape", "gamma_scale", "beta_alpha", "beta_beta"):
            self._check_positive(name)

        # Capacity must match number of legs
        self._check_len("capacity", n_legs, label="n_legs")
        self._check_positive("capacity")

        # Each ODF's booking limit must not exceed the minimum capacity
        # across the legs it uses.
        for j in range(n):
            legs_used = np.nonzero(odf_mat[j])[0]
            if len(legs_used) == 0:
                raise ValueError(
                    f"ODF class {j} does not use any leg "
                    "(row of ODF_leg_matrix is all zeros)."
                )
            min_cap = min(self.capacity[int(l)] for l in legs_used)  # noqa: E741
            if self.booking_limits[j] > min_cap:
                raise ValueError(
                    f"Booking limit for ODF class {j} "
                    f"({self.booking_limits[j]}) exceeds the minimum "
                    f"capacity ({min_cap}) of the legs it uses."
                )

        return self


class AirlineRevenueSingleStageConfig(AirlineRevenueConfigBase):
    """Config for the single-stage (full-horizon) model."""

    pass


class AirlineRevenueMultistageConfig(AirlineRevenueConfigBase):
    """Config for the multistage model (one period per stage)."""

    pass


# ======================================================================
# Demand helpers
# ======================================================================


def _beta_density(t: int, n_periods: int, alpha: float, beta: float) -> float:
    """Beta-density proportion for period *t* (standardised by booking period).

    The arrival intensity for class *j* at continuous time *s ∈ [0, tau]* is

        lambda_j(s) = Λ_j * f(s/tau; alpha_j, beta_j) / tau

    where *f* is the standard Beta density.  Discretising into *n_periods*
    equal-width bins of width Δ = tau / n_periods gives an expected count in
    period *t* of

        Λ_j * p_j(t),   where  p_j(t) = (1/T) * f((t + 0.5)/T; alpha, beta)

    (midpoint quadrature, *T* = ``n_periods``).  Note that *tau* cancels when
    computing proportions, so the function needs only *n_periods*.

    Returns:
        The proportion p_j(t) of total demand expected in period *t*.
    """
    T = n_periods  # noqa: N806
    mid = (t + 0.5) / T
    # Beta density: f(x) = x^(a-1)(1-x)^(b-1) / B(a,b)
    log_beta_fn = (
        np.log(gamma_fn(alpha))
        + np.log(gamma_fn(beta))
        - np.log(gamma_fn(alpha + beta))
    )
    log_density = (alpha - 1) * np.log(mid) + (beta - 1) * np.log(1 - mid) - log_beta_fn
    # Multiply by bin width = 1/T to get probability mass for this period
    return float(np.exp(log_density) / T)


def _period_proportions(
    n_periods: int, beta_alpha: tuple[float, ...], beta_beta: tuple[float, ...]
) -> np.ndarray:
    """Pre-compute demand proportions p_j(t) for all classes and periods.

    Returns:
        Array of shape (n_classes, n_periods) where entry [j, t] is the
        proportion of class-j total demand expected in period t.
    """
    n_classes = len(beta_alpha)
    props = np.zeros((n_classes, n_periods))
    for j in range(n_classes):
        for t in range(n_periods):
            props[j, t] = _beta_density(t, n_periods, beta_alpha[j], beta_beta[j])
        # Normalise so proportions sum to 1 for each class
        total = props[j].sum()
        if total > 0:
            props[j] /= total
    return props


def _sample_expected_demand(
    rng: MRG32k3a, gamma_shape: tuple, gamma_scale: tuple
) -> list[float]:
    """Sample total expected demand Λ_j for each class from Gamma priors.

    MRG32k3a.gammavariate(alpha, beta) uses the *scale* parametrisation:
    E[X] = alpha * beta.
    """
    return [
        rng.gammavariate(alpha=s, beta=sc)
        for s, sc in zip(gamma_shape, gamma_scale, strict=True)
    ]


def _generate_period_demand(
    rng: MRG32k3a,
    expected_demand: list[float],
    proportions: np.ndarray,
    stage: int,
) -> list[int]:
    """Generate demand for each class in a single period.

    Args:
        rng: RNG for Poisson sampling.
        expected_demand: Sampled Λ_j for each class.
        proportions: Pre-computed proportions array (n_classes x n_periods).
        stage: Current period index.

    Returns:
        List of demand counts per class.
    """
    n_classes = len(expected_demand)
    demand = []
    for j in range(n_classes):
        lam = expected_demand[j] * proportions[j, stage]
        demand.append(rng.poissonvariate(lam))
    return demand


def _legs_for_odf(odf_leg_matrix: np.ndarray, j: int) -> list[int]:
    """Return the leg indices used by ODF class *j*."""
    return np.nonzero(odf_leg_matrix[j])[0].tolist()


@dataclass
class BookingResult:
    """Outcome of processing booking requests for one period.

    Attributes:
        seats_sold: Updated cumulative seats sold per ODF class.
        remaining_capacity: Updated remaining capacity per leg.
        revenue: Revenue earned in this period.
        refused_booking_limit: Requests refused per ODF class because the
            booking limit was reached.
        refused_capacity: Requests refused per ODF class because at least
            one leg was full.
    """

    seats_sold: list[int]
    remaining_capacity: list[int]
    revenue: float
    refused_booking_limit: list[int]
    refused_capacity: list[int]


def _process_bookings(
    selling_rng: MRG32k3a,
    demand: list[int],
    booking_limits: tuple[int, ...],
    seats_sold: list[int],
    remaining_capacity: list[int],
    fares: tuple[float, ...],
    odf_leg_matrix: np.ndarray,
) -> BookingResult:
    """Process booking requests on a network with per-ODF booking limits.

    Requests arrive in a random order (shuffled).  A request for ODF class
    *j* is accepted iff:
      - ``seats_sold[j] < booking_limits[j]``, **and**
      - every leg that ODF *j* uses has ``remaining_capacity[l] > 0``.

    When a seat is sold, ``seats_sold[j]`` is incremented and capacity is
    decremented on every leg that ODF *j* uses.

    Args:
        selling_rng: RNG used to shuffle the arrival order.
        demand: Number of requests per ODF class in this period.
        booking_limits: Per-ODF booking limits.
        seats_sold: Previous cumulative bookings per ODF class
            (**mutated in-place**).
        remaining_capacity: Seats available on each leg
            (**mutated in-place**).
        fares: Fare per ODF class.
        odf_leg_matrix: Binary matrix (n_classes x n_legs).

    Returns:
        :class:`BookingResult` with updated state and refusal counts.
    """
    n_classes = len(demand)
    refused_booking_limit = [0] * n_classes
    refused_capacity = [0] * n_classes

    # Build a list of individual arrivals (ODF index repeated by demand)
    arrivals: list[int] = []
    for j, d in enumerate(demand):
        arrivals.extend([j] * d)

    # Shuffle arrivals to randomise the order
    n = len(arrivals)
    for i in range(n - 1, 0, -1):
        swap = int(selling_rng.uniform(0, i + 1)) % (i + 1)
        arrivals[i], arrivals[swap] = arrivals[swap], arrivals[i]

    period_revenue = 0.0
    for j in arrivals:
        # Per-ODF booking limit check
        if seats_sold[j] >= booking_limits[j]:
            refused_booking_limit[j] += 1
            continue

        # Check remaining capacity on every leg this ODF uses
        legs = _legs_for_odf(odf_leg_matrix, j)
        if any(remaining_capacity[l] <= 0 for l in legs):  # noqa: E741
            refused_capacity[j] += 1
            continue

        # Accept the booking
        seats_sold[j] += 1
        for l in legs:  # noqa: E741
            remaining_capacity[l] -= 1
        period_revenue += fares[j]

    return BookingResult(
        seats_sold=seats_sold,
        remaining_capacity=remaining_capacity,
        revenue=period_revenue,
        refused_booking_limit=refused_booking_limit,
        refused_capacity=refused_capacity,
    )


# ======================================================================
# Single-stage model (full horizon in one replicate)
# ======================================================================


class AirlineRevenueSingleStage(Model):
    """Network airline revenue model -- full horizon in one ``replicate()``.

    Simulates the entire booking horizon of *T* periods in a single call,
    using fixed booking limits. Returns the total revenue.
    """

    class_name_abbr: ClassVar[str] = "AIRLINE-SL"
    class_name: ClassVar[str] = "Network Airline Revenue (Single Stage)"
    config_class: ClassVar[type[BaseModel]] = AirlineRevenueSingleStageConfig
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
        """Simulate the full booking horizon and return total revenue.

        Returns:
            tuple: (responses dict, gradients dict).
        """
        assert self._demand_rng is not None and self._selling_rng is not None

        n_classes = self.factors["n_classes"]
        capacity: tuple = self.factors["capacity"]
        fares: tuple = self.factors["fares"]
        booking_limits: tuple = self.factors["booking_limits"]
        n_periods: int = self.factors["n_periods"]
        gamma_shape: tuple = self.factors["gamma_shape"]
        gamma_scale: tuple = self.factors["gamma_scale"]
        beta_alpha: tuple = self.factors["beta_alpha"]
        beta_beta: tuple = self.factors["beta_beta"]

        # Pre-compute period proportions
        proportions = _period_proportions(n_periods, beta_alpha, beta_beta)

        # Sample total expected demand for each ODF class (once per episode)
        expected_demand = _sample_expected_demand(
            self._demand_rng, gamma_shape, gamma_scale
        )

        seats_sold = [0] * n_classes
        remaining_capacity = list(capacity)
        total_revenue = 0.0
        total_refused_bl = [0] * n_classes
        total_refused_cap = [0] * n_classes

        for t in range(n_periods):
            demand = _generate_period_demand(
                self._demand_rng, expected_demand, proportions, t
            )
            result = _process_bookings(
                self._selling_rng,
                demand,
                booking_limits,
                seats_sold,
                remaining_capacity,
                fares,
                self._odf_leg_matrix,
            )
            seats_sold = result.seats_sold
            remaining_capacity = result.remaining_capacity
            total_revenue += result.revenue
            for j in range(n_classes):
                total_refused_bl[j] += result.refused_booking_limit[j]
                total_refused_cap[j] += result.refused_capacity[j]

        responses = {
            "revenue": total_revenue,
            "seats_sold": dict(enumerate(seats_sold)),
            "remaining_capacity": dict(enumerate(remaining_capacity)),
            "refused_booking_limit": dict(enumerate(total_refused_bl)),
            "refused_capacity": dict(enumerate(total_refused_cap)),
        }
        return responses, {}


# ======================================================================
# Multistage model (one period per stage)
# ======================================================================


class AirlineRevenueMultistage(MultistageModel):
    """Network airline revenue model -- one booking period per stage.

    Each stage corresponds to one time period in the booking horizon.

    * **State**: :class:`AirlineState` (per-leg remaining capacity,
      per-ODF cumulative bookings, sampled expected demand, Gamma
      posterior parameters).
    * **Decision**: tuple of per-ODF booking limits ``(b_1, ..., b_n)``.
    * **Transition**: generate demand, process bookings on the network.
    * **Stage reward**: revenue earned in that period.

    The default rollout policy keeps the initial booking limits unchanged
    across all future stages.
    """

    class_name_abbr: ClassVar[str] = "AIRLINE-ML"
    class_name: ClassVar[str] = "Network Airline Revenue (Multistage)"
    config_class: ClassVar[type[BaseModel]] = AirlineRevenueMultistageConfig
    n_rngs: ClassVar[int] = 2  # demand RNG + selling-order RNG
    n_responses: ClassVar[int] = 1
    n_stages: ClassVar[int] = 10  # overridden at __init__ from config

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize instance."""
        super().__init__(fixed_factors)
        # Override class-level n_stages from the config
        self.__class__.n_stages = self.factors["n_periods"]

        # Pre-compute temporal demand proportions
        self._proportions = _period_proportions(
            self.factors["n_periods"],
            self.factors["beta_alpha"],
            self.factors["beta_beta"],
        )

        # Cache the ODF-leg matrix as numpy array
        self._odf_leg_matrix = np.array(self.factors["ODF_leg_matrix"])

    # ---- MultistageModel abstract methods ----

    def get_initial_state(self) -> AirlineState:  # noqa: D102
        n_classes = self.factors["n_classes"]
        gamma_shape: tuple = self.factors["gamma_shape"]
        gamma_scale: tuple = self.factors["gamma_scale"]
        return AirlineState(
            remaining_capacity=list(self.factors["capacity"]),
            seats_sold=[0] * n_classes,
            expected_demand=[],  # populated after before_replication
            gamma_shape_posterior=list(gamma_shape),
            gamma_rate_posterior=[1.0 / sc for sc in gamma_scale],
            total_requests=[0] * n_classes,
        )

    def before_replication(self, rng_list: list[MRG32k3a]) -> None:
        """Store RNGs and sample the episode's true expected demand Λ_j."""
        super().before_replication(rng_list)
        self._demand_rng = rng_list[0]
        self._selling_rng = rng_list[1]

        # Sample the "true" Λ_j for this episode from the Gamma prior
        self._episode_expected_demand = _sample_expected_demand(
            self._demand_rng,
            self.factors["gamma_shape"],
            self.factors["gamma_scale"],
        )

    def transition(  # noqa: D102
        self,
        state: object,
        decision: tuple,
        stage: int,
        rng_list: list[MRG32k3a],
    ) -> object:
        if not isinstance(state, AirlineState):
            raise TypeError("state must be an AirlineState")

        demand_rng = rng_list[0]
        selling_rng = rng_list[1]

        n_classes: int = self.factors["n_classes"]

        # ----------------------------------------------------------
        # Determine which Λ to use for demand generation.
        # ----------------------------------------------------------
        expected_demand = (
            state.expected_demand
            if state.expected_demand
            else self._episode_expected_demand
        )

        # Generate demand (total booking requests) for this period
        demand = _generate_period_demand(
            demand_rng, expected_demand, self._proportions, stage
        )

        # Process bookings on the network (work on copies)
        new_seats_sold = list(state.seats_sold)
        new_remaining_capacity = list(state.remaining_capacity)
        result = _process_bookings(
            selling_rng,
            demand,
            tuple(int(b) for b in decision),
            new_seats_sold,
            new_remaining_capacity,
            self.factors["fares"],
            self._odf_leg_matrix,
        )
        new_seats_sold = result.seats_sold
        new_remaining_capacity = result.remaining_capacity
        self._last_booking_result = result

        # ----------------------------------------------------------
        # Bayesian Gamma-Poisson conjugate update.
        #
        # Prior / current posterior:
        #   Λ_j ~ Gamma(alpha_j, beta_j)        [rate parametrisation]
        #
        # Observation in period *t*:
        #   D_{j,t} ~ Poisson(Λ_j * p_j(t))
        #
        # Posterior update:
        #   alpha_j  ←  alpha_j + D_{j,t}
        #   beta_j  ←  beta_j + p_j(t)
        #
        # The posterior mean (MMSE estimate) of Λ_j is then alpha_j / beta_j.
        # ----------------------------------------------------------
        new_gamma_shape = (
            list(state.gamma_shape_posterior)
            if state.gamma_shape_posterior
            else list(self.factors["gamma_shape"])
        )
        new_gamma_rate = (
            list(state.gamma_rate_posterior)
            if state.gamma_rate_posterior
            else [1.0 / sc for sc in self.factors["gamma_scale"]]
        )
        new_total_requests = (
            list(state.total_requests) if state.total_requests else [0] * n_classes
        )

        for j in range(n_classes):
            new_gamma_shape[j] += demand[j]
            new_gamma_rate[j] += self._proportions[j, stage]
            new_total_requests[j] += demand[j]

        return AirlineState(
            remaining_capacity=new_remaining_capacity,
            seats_sold=new_seats_sold,
            expected_demand=list(expected_demand),
            gamma_shape_posterior=new_gamma_shape,
            gamma_rate_posterior=new_gamma_rate,
            total_requests=new_total_requests,
        )

    def stage_reward(  # noqa: D102
        self,
        state: object,
        decision: tuple,  # noqa: ARG002
        next_state: object,
        stage: int,  # noqa: ARG002
    ) -> dict[str, float]:
        if not isinstance(state, AirlineState) or not isinstance(
            next_state, AirlineState
        ):
            raise TypeError("state and next_state must be AirlineState instances")

        n_classes = self.factors["n_classes"]
        fares = self.factors["fares"]
        new_sales = [
            next_state.seats_sold[j] - state.seats_sold[j] for j in range(n_classes)
        ]
        revenue = sum(f * s for f, s in zip(fares, new_sales, strict=True))
        # Only return float-valued responses so that the base-class
        # accumulation in simulate_lookahead / replicate works.
        return {"revenue": revenue}

    def get_default_policy(self) -> Callable[[AirlineState, int], tuple]:
        """Return a policy that uses the configured booking limits at every stage."""
        default_limits = tuple(self.factors["booking_limits"])

        def _policy(state: AirlineState, stage: int) -> tuple:  # noqa: ARG001
            return default_limits

        return _policy

    # ---- Override replicate to add rich current-stage data ----

    def replicate(
        self,
        state: Any,  # noqa: ANN401
        decision: tuple,
        stage: int,
        policy: Callable[[Any, int], tuple] | None = None,
        n_lookahead_reps: int = 30,
        future_responses: dict[str, float] | None = None,
    ) -> tuple[dict[str, float], dict]:
        """Replicate with rich per-ODF/per-leg detail for the current stage.

        Calls the base-class implementation (which accumulates *revenue*
        across the lookahead horizon), then appends the current stage's
        ``seats_sold``, ``remaining_capacity``, ``refused_booking_limit``
        and ``refused_capacity`` dicts keyed by ODF/leg index.
        """
        responses, gradients = super().replicate(
            state,
            decision,
            stage,
            policy,
            n_lookahead_reps,
            future_responses=future_responses,
        )
        rich_responses = cast(dict[str, Any], responses)

        n_classes = self.factors["n_classes"]
        br = getattr(self, "_last_booking_result", None)

        # Current-stage transition has already been run by the base class;
        # self._last_booking_result holds the result from the current
        # stage's transition call (the most recent one, which is the
        # one for the current stage because replicate_stage calls
        # transition first, then stage_reward).
        if br is not None:
            rich_responses["seats_sold"] = dict(enumerate(br.seats_sold))
            rich_responses["remaining_capacity"] = dict(
                enumerate(br.remaining_capacity)
            )
            rich_responses["refused_booking_limit"] = dict(
                enumerate(br.refused_booking_limit)
            )
            rich_responses["refused_capacity"] = dict(enumerate(br.refused_capacity))
        else:
            rich_responses["seats_sold"] = dict(enumerate([0] * n_classes))
            rich_responses["remaining_capacity"] = dict(
                enumerate(list(self.factors["capacity"]))
            )
            rich_responses["refused_booking_limit"] = dict(enumerate([0] * n_classes))
            rich_responses["refused_capacity"] = dict(enumerate([0] * n_classes))

        return cast(dict[str, float], rich_responses), gradients

    # ---- Lookahead override for Pólya posterior sampling ----

    def simulate_lookahead(
        self,
        state: object,
        start_stage: int,
        policy: Callable[[Any, int], tuple] | None = None,
        n_reps: int = 30,
    ) -> dict[str, float]:
        """Estimate cost-to-go with Λ sampled from the Gamma posterior.

        For each Monte Carlo replication, a fresh Λ vector is drawn from
        the current Gamma posterior held in *state*, then used as the
        ``expected_demand`` for all remaining stages in that replication.
        This implements the Pólya-process predictive distribution.
        """
        if start_stage >= self.n_stages:
            return {}

        if not isinstance(state, AirlineState):
            raise TypeError("state must be an AirlineState")

        policy = policy or self.get_default_policy()
        n_classes: int = self.factors["n_classes"]

        total_responses: dict[str, float] = {}
        for _ in range(n_reps):
            # --- Sample Λ from the posterior for this replication ---
            lookahead_lambda = [
                self._demand_rng.gammavariate(
                    alpha=state.gamma_shape_posterior[j],
                    beta=1.0 / state.gamma_rate_posterior[j],
                )
                for j in range(n_classes)
            ]

            sim_state = deepcopy(state)
            sim_state.expected_demand = lookahead_lambda

            rep_responses: dict[str, float] = {}
            for t in range(start_stage, self.n_stages):
                decision = policy(sim_state, t)
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
# Problem configuration
# ======================================================================


class AirlineRevenueSingleStageProblemConfig(BaseModel):
    """Config for the single-stage booking-limit optimisation problem."""

    initial_solution: Annotated[
        tuple[int, ...],
        Field(
            default=(10, 10, 10, 10, 10, 10),
            description="initial per-ODF booking limits",
        ),
    ]
    budget: Annotated[
        int,
        Field(
            default=10000,
            description="max replications for the solver",
            gt=0,
            json_schema_extra={"isDatafarmable": False},
        ),
    ]


class AirlineRevenueMultistageProblemConfig(BaseModel):
    """Config for the multistage booking-limit optimisation problem."""

    initial_solution: Annotated[
        tuple[int, ...],
        Field(
            default=(10, 10, 10, 10, 10, 10),
            description="initial per-ODF booking limits",
        ),
    ]
    budget: Annotated[
        int,
        Field(
            default=10000,
            description="max replications for the solver",
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


class AirlineRevenueSingleStageProblem(Problem):
    """Maximise expected total revenue by choosing per-ODF booking limits.

    Full booking horizon simulated in one shot (standard ``Problem``).
    """

    class_name_abbr: ClassVar[str] = "AIRLINE-1"
    class_name: ClassVar[str] = "Max Revenue Network Airline (Single Stage)"
    config_class: ClassVar[type[BaseModel]] = AirlineRevenueSingleStageProblemConfig
    model_class: ClassVar[type[Model]] = AirlineRevenueSingleStage
    n_objectives: ClassVar[int] = 1
    n_stochastic_constraints: ClassVar[int] = 0
    minmax: ClassVar[tuple[int, ...]] = (1,)  # maximise
    constraint_type: ClassVar[ConstraintType] = ConstraintType.DETERMINISTIC
    variable_type: ClassVar[VariableType] = VariableType.DISCRETE
    gradient_available: ClassVar[bool] = False
    optimal_value: ClassVar[float | None] = None
    optimal_solution: tuple | None = None
    model_default_factors: ClassVar[dict] = {}
    model_decision_factors: ClassVar[set[str]] = {"booking_limits"}

    @property
    def dim(self) -> int:  # noqa: D102
        return self.model.factors["n_classes"]

    @property
    def lower_bounds(self) -> tuple:  # noqa: D102
        return (0,) * self.dim

    @property
    def upper_bounds(self) -> tuple:  # noqa: D102
        odf_mat = np.array(self.model.factors["ODF_leg_matrix"])
        capacity = self.model.factors["capacity"]
        ub = []
        for j in range(self.dim):
            legs = np.nonzero(odf_mat[j])[0]
            ub.append(min(capacity[int(l)] for l in legs))  # noqa: E741
        return tuple(ub)

    def vector_to_factor_dict(self, vector: tuple) -> dict:  # noqa: D102
        return {"booking_limits": tuple(int(v) for v in vector)}

    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:  # noqa: D102
        return tuple(factor_dict["booking_limits"])

    def replicate(self, x: tuple) -> RepResult:  # noqa: ARG002, D102
        responses, _ = self.model.replicate()
        objectives = [Objective(stochastic=responses["revenue"])]
        return RepResult(objectives=objectives)

    def check_deterministic_constraints(self, x: tuple) -> bool:  # noqa: D102
        if not all(v >= 0 for v in x):
            return False
        ub = self.upper_bounds
        return all(x[j] <= ub[j] for j in range(len(x)))

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:  # noqa: D102
        ub = self.upper_bounds
        n = self.dim
        while True:
            vals = tuple(int(rand_sol_rng.uniform(0, ub[j] + 1)) for j in range(n))
            if self.check_deterministic_constraints(vals):
                return vals


# ======================================================================
# Multistage problem
# ======================================================================


class AirlineRevenueMultistageProblem(MultistageProblem):
    """Maximise expected total revenue by choosing booking limits at each stage.

    At each period the solver selects per-ODF booking limits for the
    current period.  Revenue is evaluated as the immediate period revenue
    plus Monte Carlo estimated future revenue under a default rollout
    policy.
    """

    class_name_abbr: ClassVar[str] = "AIRLINE-2"
    class_name: ClassVar[str] = "Max Revenue Network Airline (Multistage)"
    config_class: ClassVar[type[BaseModel]] = AirlineRevenueMultistageProblemConfig
    model_class: ClassVar[type[MultistageModel]] = AirlineRevenueMultistage
    n_objectives: ClassVar[int] = 1
    n_stochastic_constraints: ClassVar[int] = 0
    minmax: ClassVar[tuple[int, ...]] = (1,)  # maximise
    constraint_type: ClassVar[ConstraintType] = ConstraintType.DETERMINISTIC
    variable_type: ClassVar[VariableType] = VariableType.DISCRETE
    gradient_available: ClassVar[bool] = False
    model_default_factors: ClassVar[dict] = {}
    model_decision_factors: ClassVar[set[str]] = {"booking_limits"}
    n_lookahead_reps: ClassVar[int] = 30

    @property
    def dim(self) -> int:  # noqa: D102
        return self.model.factors["n_classes"]

    @property
    def lower_bounds(self) -> tuple:  # noqa: D102
        return (0,) * self.dim

    @property
    def upper_bounds(self) -> tuple:  # noqa: D102
        # Each ODF's limit is capped by min remaining capacity on its legs
        odf_mat = np.array(self.model.factors["ODF_leg_matrix"])
        cap = (
            self.current_state.remaining_capacity
            if hasattr(self, "current_state")
            and hasattr(self.current_state, "remaining_capacity")
            else list(self.model.factors["capacity"])
        )
        ub = []
        for j in range(self.dim):
            legs = np.nonzero(odf_mat[j])[0]
            ub.append(min(cap[int(l)] for l in legs))  # noqa: E741
        return tuple(ub)

    def vector_to_factor_dict(self, vector: tuple) -> dict:  # noqa: D102
        return {"booking_limits": tuple(int(v) for v in vector)}

    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:  # noqa: D102
        return tuple(factor_dict["booking_limits"])

    def replicate(self, x: tuple) -> RepResult:  # noqa: D102
        responses, _ = self.model.replicate(
            state=self.current_state,
            decision=x,
            stage=self.current_stage,
            n_lookahead_reps=self.n_lookahead_reps,
        )
        # Use the total revenue (stage + future) as the objective
        objectives = [Objective(stochastic=responses["total_revenue"])]
        return RepResult(objectives=objectives)

    def check_deterministic_constraints(self, x: tuple) -> bool:  # noqa: D102
        if not all(v >= 0 for v in x):
            return False
        ub = self.upper_bounds
        return all(x[j] <= ub[j] for j in range(len(x)))

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:  # noqa: D102
        ub = self.upper_bounds
        n = self.dim
        while True:
            vals = tuple(int(rand_sol_rng.uniform(0, ub[j] + 1)) for j in range(n))
            if self.check_deterministic_constraints(vals):
                return vals
