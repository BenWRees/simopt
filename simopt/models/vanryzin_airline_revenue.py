"""Network airline revenue management with virtual nesting from Van Ryzin & Vulcano.

(2008).

Implements the *continuous fluid* virtual nesting control model described in:

    G. van Ryzin and G. Vulcano, "Simulation-Based Optimization of Virtual
    Nesting Controls for Network Revenue Management", *Operations Research*
    56(4):865-880, 2008.

**Model overview**

An airline network consists of one or more flight *legs*, each with its own
(continuous) capacity.  Passengers are grouped into *products*
(origin-destination-fare classes); each product may use one or more legs, as
described by a binary product-leg adjacency matrix.

In *virtual nesting*, products are mapped ("indexed") into a relatively small
number of *virtual classes* on each leg they use.  Nested protection levels
control the availability of these virtual classes.

Capacity and demand are **continuous quantities** (the "fluid" model).
Requests can be **partially accepted**: when a product's demand in a period
exceeds the remaining available capacity under the nesting control, the
accepted amount equals the minimum available capacity across all required
legs rather than zero.

**Virtual Nesting Control**

There are *K* virtual classes on each leg, numbered 1 (highest priority) to
*K* (lowest priority).  The protection levels on leg *l* satisfy

    y_{l,1} ≤ y_{l,2} ≤ ... ≤ y_{l,K} ≤ C_l

where *C_l* is the (continuous) capacity of leg *l*.  Here *y_{l,k}* acts as
a cumulative booking limit -- the maximum total amount of capacity that may be
held by virtual classes 1 through *k* on leg *l*.

The available capacity for virtual class *k* on leg *l* is:

    avail_{l,k} = max(0, y_{l,k} - ∑_{k'=1}^{k} bookings_{l,k'})

A request for product *j* with continuous demand quantity *q* is processed
according to the paper's fluid acceptance rule:

    u_j(x, y, q) = min(q,  min_{l ∈ L(j)} avail_{l, sigma_l(j)})

i.e. the accepted amount is the minimum available capacity among all
resources required by *j*, or *q* if there is at least *q* units of
capacity on all these resources.  This means requests can be (and
typically are) **partially** accepted.

Each product *j* is assigned to virtual class sigma_l(j) on each leg *l* it uses
via the ``virtual_class_indexing`` matrix.

**Demand model (continuous fluid with Gamma prior)**

Demand for each product follows a continuous fluid model.  The total expected
demand for product *j* over the full booking period is a random variable
Λ_j drawn from a Gamma prior:

    Λ_j ~ Gamma(alpha_j, beta_j)

The booking period has physical length tau (configurable via the
``booking_period_length`` factor).  Within the horizon the arrival intensity
at continuous time *s ∈ [0, tau]* is

    lambda_j(s) = Λ_j * f(s/tau; a_j, b_j) / tau

where *f(*; a, b)* is the standard Beta density with shape parameters
*(a_j, b_j)*.  After discretisation into *T* periods, the continuous
demand quantity for product *j* in period *t* is the fluid rate:

    q_{j,t} = Λ_j * p_j(t)

where p_j(t) = (1/T) * f((t + 0.5)/T; a_j, b_j) is the midpoint-quadrature
proportion of demand in period *t*.  This is a **deterministic** function of
the (random) total demand Λ_j; there is no additional Poisson noise in the
fluid model.

**Bayesian Gamma updates (multistage)**

In the multistage formulation, the airline's belief about Λ_j is updated
after each period.  In the fluid model, the observed continuous demand
quantity q_{j,t} substitutes for the discrete Poisson count in the
conjugate update.  See :mod:`simopt.models.bertsimas_airline_revenue`
for the original Gamma-Poisson conjugate relationship.

**Two concrete classes are provided:**

* :class:`VanRyzinRevenueSingleStage` -- a standard :class:`Model` that
  simulates the entire booking horizon in one call to ``replicate()``.
* :class:`VanRyzinRevenueMultistage` -- a :class:`MultistageModel` where each
  time period is a separate decision stage, with Bayesian demand updates
  and posterior-based lookahead.
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
    Objective,
    Problem,
    RepResult,
    VariableType,
)
from simopt.multistage_model import MultistageModel
from simopt.multistage_problem import MultistageProblem
from simopt.problem import Solution

# ======================================================================
# State dataclass used by the multistage model
# ======================================================================


# @dataclass
# class VanRyzinState:
#     """State of the virtual nesting booking process at the start of a period.

#     All capacity, booking, and demand quantities are **continuous** (float)
#     to support the fluid model with partial acceptance.

#     Attributes:
#         remaining_capacity: Remaining (continuous) capacity per flight leg.
#         virtual_class_bookings: Cumulative bookings per (leg, virtual class).
#             Nested list of shape ``(n_legs, n_virtual_classes)``.
#         seats_sold: Cumulative units sold per product (ODF class).
#         expected_demand: Sampled Λ_j values used for demand generation.
#             In the main episode this is the "true" Λ drawn at the start;
#             in a lookahead replication it is freshly sampled from the
#             posterior.  Empty list signals that a new Λ should be drawn.
#         gamma_shape_posterior: Current Gamma posterior shape parameter per
#             product (initialised to the prior shape alpha_j).
#         gamma_rate_posterior: Current Gamma posterior *rate* parameter per
#             product (initialised to 1 / scale_j).  Rate parametrisation
#             is used because the conjugate update simply adds
#             observed demand to alpha and cumulative intensity proportions to beta.
#         total_requests: Total booking *demand* (continuous fluid quantity)
#             observed per product up to the current period.
#     """

#     remaining_capacity: list[float] = field(default_factory=list)
#     virtual_class_bookings: list[list[float]] = field(default_factory=list)
#     seats_sold: list[float] = field(default_factory=list)
#     expected_demand: list[float] = field(default_factory=list)
#     gamma_shape_posterior: list[float] = field(default_factory=list)
#     gamma_rate_posterior: list[float] = field(default_factory=list)
#     total_requests: list[float] = field(default_factory=list)


# ======================================================================
# Pydantic configuration models
# ======================================================================


class VanRyzinRevenueConfigBase(BaseModel):
    """Shared configuration for the virtual nesting airline revenue model.

    The model is defined over a network of flight *legs*.  Each product
    (origin-destination-fare class) may use one or more legs, as specified
    by the binary ``ODF_leg_matrix``.  Products are mapped to virtual
    classes on each leg via ``virtual_class_indexing``, and capacity is
    controlled through nested ``protection_levels``.
    """

    n_classes: Annotated[
        int,
        Field(
            # Example 2 (§4.2): five-airport hub-and-spoke network (Williamson 1992).
            # 10 round-trip O-D markets x 4 fare classes (Y,M,B,Q) = 80 products.
            default=80,
            description="number of products (origin-destination-fare classes)",
            gt=0,
        ),
    ]
    ODF_leg_matrix: Annotated[
        list[list[int]],
        Field(
            # Example 2: five-airport hub-and-spoke network with ATL as hub.
            # 8 directed legs: 0=ATL→LAX, 1=LAX→ATL, 2=ATL→BOS, 3=BOS→ATL,
            #                  4=ATL→SAV, 5=SAV→ATL, 6=ATL→MIA, 7=MIA→ATL.
            # Products 0-31 are direct (single-leg); 32-79 are connecting (two-leg).
            # Each itinerary has 4 products in Y,M,B,Q order.
            default_factory=lambda: [
                row
                for itinerary_legs in [
                    # --- Direct one-leg itineraries (groups 0-7) ---
                    [0, 0, 1, 0, 0, 0, 0, 0],  # ATLBOS
                    [0, 0, 0, 1, 0, 0, 0, 0],  # BOSATL
                    [1, 0, 0, 0, 0, 0, 0, 0],  # ATLLAX
                    [0, 1, 0, 0, 0, 0, 0, 0],  # LAXATL
                    [0, 0, 0, 0, 0, 0, 1, 0],  # ATLMIA
                    [0, 0, 0, 0, 0, 0, 0, 1],  # MIAATL
                    [0, 0, 0, 0, 1, 0, 0, 0],  # ATLSAV
                    [0, 0, 0, 0, 0, 1, 0, 0],  # SAVATL
                    # --- Connecting two-leg itineraries via ATL (groups 8-19) ---
                    [1, 0, 0, 1, 0, 0, 0, 0],  # BOSLAX  (BOS→ATL + ATL→LAX)
                    [0, 1, 1, 0, 0, 0, 0, 0],  # LAXBOS  (LAX→ATL + ATL→BOS)
                    [0, 0, 0, 1, 0, 0, 1, 0],  # BOSMIA  (BOS→ATL + ATL→MIA)
                    [0, 0, 1, 0, 0, 0, 0, 1],  # MIABOS  (MIA→ATL + ATL→BOS)
                    [0, 0, 0, 1, 1, 0, 0, 0],  # BOSSAV  (BOS→ATL + ATL→SAV)
                    [0, 0, 1, 0, 0, 1, 0, 0],  # SAVBOS  (SAV→ATL + ATL→BOS)
                    [0, 1, 0, 0, 0, 0, 1, 0],  # LAXMIA  (LAX→ATL + ATL→MIA)
                    [1, 0, 0, 0, 0, 0, 0, 1],  # MIALAX  (MIA→ATL + ATL→LAX)
                    [0, 1, 0, 0, 1, 0, 0, 0],  # LAXSAV  (LAX→ATL + ATL→SAV)
                    [1, 0, 0, 0, 0, 1, 0, 0],  # SAVLAX  (SAV→ATL + ATL→LAX)
                    [0, 0, 0, 0, 1, 0, 0, 1],  # MIASAV  (MIA→ATL + ATL→SAV)
                    [0, 0, 0, 0, 0, 1, 1, 0],  # SAVMIA  (SAV→ATL + ATL→MIA)
                ]
                # 4 products (Y, M, B, Q) per itinerary share the same leg pattern
                for row in [list(itinerary_legs)] * 4
            ],
            description=(
                "Binary matrix of shape (n_classes, n_legs).  Entry [j, l] "
                "is 1 if product j uses leg l, else 0."
            ),
        ),
    ]
    capacity: Annotated[
        tuple[float, ...],
        Field(
            # Example 2: eight legs, capacity 180 per leg (Table 4 base case).
            default=(180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0),
            description="(continuous) capacity for each flight leg",
        ),
    ]
    fares: Annotated[
        tuple[float, ...],
        Field(
            # Example 2: revenues from Table 3 (§4.2), Y/M/B/Q per O-D market.
            # Products ordered: ATLBOS, BOSATL, ATLLAX, LAXATL, ATLMIA, MIAATL,
            #   ATLSAV, SAVATL, BOSLAX, LAXBOS, BOSMIA, MIABOS, BOSSAV, SAVBOS,
            #   LAXMIA, MIALAX, LAXSAV, SAVLAX, MIASAV, SAVMIA  (4 classes each).
            default=(
                310.0,
                290.0,
                95.0,
                69.0,  # ATLBOS  Y M B Q
                310.0,
                290.0,
                95.0,
                69.0,  # BOSATL
                455.0,
                391.0,
                142.0,
                122.0,  # ATLLAX
                455.0,
                391.0,
                142.0,
                122.0,  # LAXATL
                280.0,
                209.0,
                94.0,
                59.0,  # ATLMIA
                280.0,
                209.0,
                94.0,
                59.0,  # MIAATL
                159.0,
                140.0,
                64.0,
                49.0,  # ATLSAV
                159.0,
                140.0,
                64.0,
                49.0,  # SAVATL
                575.0,
                380.0,
                159.0,
                139.0,  # BOSLAX
                575.0,
                380.0,
                159.0,
                139.0,  # LAXBOS
                403.0,
                314.0,
                124.0,
                89.0,  # BOSMIA
                403.0,
                314.0,
                124.0,
                89.0,  # MIABOS
                319.0,
                250.0,
                109.0,
                69.0,  # BOSSAV
                319.0,
                250.0,
                109.0,
                69.0,  # SAVBOS
                477.0,
                239.0,
                139.0,
                119.0,  # LAXMIA
                477.0,
                239.0,
                139.0,
                119.0,  # MIALAX
                502.0,
                450.0,
                154.0,
                134.0,  # LAXSAV
                502.0,
                450.0,
                154.0,
                134.0,  # SAVLAX
                226.0,
                168.0,
                84.0,
                59.0,  # MIASAV
                226.0,
                168.0,
                84.0,
                59.0,  # SAVMIA
            ),
            description="fare (price) for each product",
        ),
    ]
    n_virtual_classes: Annotated[
        int,
        Field(
            # Example 2: c̄ = 9 protection levels → K = 10 virtual classes per leg.
            default=10,
            description="number of virtual classes per leg (K)",
            gt=0,
        ),
    ]
    virtual_class_indexing: Annotated[
        list[list[int]],
        Field(
            # Example 2: simplified DAVN-style indexing.
            # Direct itineraries (groups 0-7) → VCs 1-4  (Y=1, M=2, B=3, Q=4).
            # Connecting itineraries (groups 8-19) → VCs 5-8 (Y=5, M=6, B=7, Q=8).
            # Products not using a leg receive 0 (unused).
            default_factory=lambda: [
                [vc * e for e in legs]
                for vc, legs in (
                    # Direct itineraries: VC offset 1 (class index 0-3 → VCs 1-4)
                    [
                        (c + 1, list(pat))
                        for pat in [
                            [0, 0, 1, 0, 0, 0, 0, 0],  # ATLBOS
                            [0, 0, 0, 1, 0, 0, 0, 0],  # BOSATL
                            [1, 0, 0, 0, 0, 0, 0, 0],  # ATLLAX
                            [0, 1, 0, 0, 0, 0, 0, 0],  # LAXATL
                            [0, 0, 0, 0, 0, 0, 1, 0],  # ATLMIA
                            [0, 0, 0, 0, 0, 0, 0, 1],  # MIAATL
                            [0, 0, 0, 0, 1, 0, 0, 0],  # ATLSAV
                            [0, 0, 0, 0, 0, 1, 0, 0],  # SAVATL
                        ]
                        for c in range(4)
                    ]
                    +
                    # Connecting itineraries: VC offset 5 (class index 0-3 → VCs 5-8)
                    [
                        (c + 5, list(pat))
                        for pat in [
                            [1, 0, 0, 1, 0, 0, 0, 0],  # BOSLAX
                            [0, 1, 1, 0, 0, 0, 0, 0],  # LAXBOS
                            [0, 0, 0, 1, 0, 0, 1, 0],  # BOSMIA
                            [0, 0, 1, 0, 0, 0, 0, 1],  # MIABOS
                            [0, 0, 0, 1, 1, 0, 0, 0],  # BOSSAV
                            [0, 0, 1, 0, 0, 1, 0, 0],  # SAVBOS
                            [0, 1, 0, 0, 0, 0, 1, 0],  # LAXMIA
                            [1, 0, 0, 0, 0, 0, 0, 1],  # MIALAX
                            [0, 1, 0, 0, 1, 0, 0, 0],  # LAXSAV
                            [1, 0, 0, 0, 0, 1, 0, 0],  # SAVLAX
                            [0, 0, 0, 0, 1, 0, 0, 1],  # MIASAV
                            [0, 0, 0, 0, 0, 1, 1, 0],  # SAVMIA
                        ]
                        for c in range(4)
                    ]
                )
            ],
            description=(
                "Virtual class indexing matrix of shape (n_classes, n_legs). "
                "Entry [j, l] is the 1-based virtual class index for "
                "product j on leg l, or 0 if product j does not use leg l."
            ),
        ),
    ]
    protection_levels: Annotated[
        list[list[float]],
        Field(
            # Example 2: initial DAVN protection levels y^(0) from Table 5 (§4.2),
            # x_i = 180 for all legs, first booking period.  Shape (8, 10).
            # Entries from the paper padded to K=10 with leg capacity (180.0).
            # Legs: 0=ATL→LAX, 1=LAX→ATL, 2=ATL→BOS, 3=BOS→ATL,
            #       4=ATL→SAV, 5=SAV→ATL, 6=ATL→MIA, 7=MIA→ATL.
            default_factory=lambda: [
                [
                    6.0,
                    34.0,
                    42.0,
                    49.0,
                    61.0,
                    75.0,
                    88.0,
                    180.0,
                    180.0,
                    180.0,
                ],  # Leg 0 (resource 1)
                [
                    6.0,
                    34.0,
                    42.0,
                    49.0,
                    61.0,
                    75.0,
                    88.0,
                    180.0,
                    180.0,
                    180.0,
                ],  # Leg 1 (resource 2)
                [
                    7.0,
                    18.0,
                    30.0,
                    59.0,
                    73.0,
                    103.0,
                    180.0,
                    180.0,
                    180.0,
                    180.0,
                ],  # Leg 2 (resource 3)
                [
                    7.0,
                    18.0,
                    30.0,
                    59.0,
                    73.0,
                    103.0,
                    180.0,
                    180.0,
                    180.0,
                    180.0,
                ],  # Leg 3 (resource 4)
                [
                    3.0,
                    11.0,
                    18.0,
                    38.0,
                    63.0,
                    81.0,
                    96.0,
                    180.0,
                    180.0,
                    180.0,
                ],  # Leg 4 (resource 5)
                [
                    3.0,
                    11.0,
                    18.0,
                    38.0,
                    63.0,
                    81.0,
                    96.0,
                    180.0,
                    180.0,
                    180.0,
                ],  # Leg 5 (resource 6)
                [
                    25.0,
                    52.0,
                    62.0,
                    79.0,
                    98.0,
                    107.0,
                    141.0,
                    180.0,
                    180.0,
                    180.0,
                ],  # Leg 6 (resource 7)
                [
                    25.0,
                    52.0,
                    62.0,
                    79.0,
                    98.0,
                    107.0,
                    141.0,
                    180.0,
                    180.0,
                    180.0,
                ],  # Leg 7 (resource 8)
            ],
            description=(
                "Nested protection levels of shape (n_legs, n_virtual_classes). "
                "Entry [l, k] = y_{l, k+1}: the amount of remaining capacity on "
                "leg l that is reserved for virtual classes 1 through k+1 "
                "(paper Eq. 2).  Must be monotonically non-decreasing within "
                "each leg, with the final entry equal to leg capacity."
            ),
        ),
    ]
    n_periods: Annotated[
        int,
        Field(
            # Example 2: booking horizon split evenly into 3 periods; protection
            # levels are reoptimised at the start of each period (§4.2).
            default=3,
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
            # Example 2: mean demands mu_j from Table 3 (§4.2).  The paper uses
            # truncated Normal with mean mu_j and std sigma_j = √mu_j (Var = mu_j).
            # The moment-matched Gamma has shape alpha = mu_j and scale beta = 1.
            default=(
                12.0,
                9.0,
                11.0,
                17.0,  # ATLBOS  Y M B Q
                12.0,
                9.0,
                11.0,
                17.0,  # BOSATL
                8.0,
                4.0,
                11.0,
                27.0,  # ATLLAX
                8.0,
                4.0,
                11.0,
                27.0,  # LAXATL
                20.0,
                9.0,
                7.0,
                14.0,  # ATLMIA
                20.0,
                9.0,
                7.0,
                14.0,  # MIAATL
                25.0,
                7.0,
                5.0,
                13.0,  # ATLSAV
                25.0,
                7.0,
                5.0,
                13.0,  # SAVATL
                9.0,
                7.0,
                11.0,
                24.0,  # BOSLAX
                9.0,
                7.0,
                11.0,
                24.0,  # LAXBOS
                11.0,
                5.0,
                14.0,
                20.0,  # BOSMIA
                11.0,
                5.0,
                14.0,
                20.0,  # MIABOS
                5.0,
                7.0,
                11.0,
                27.0,  # BOSSAV
                5.0,
                7.0,
                11.0,
                27.0,  # SAVBOS
                17.0,
                11.0,
                7.0,
                14.0,  # LAXMIA
                17.0,
                11.0,
                7.0,
                14.0,  # MIALAX
                5.0,
                7.0,
                7.0,
                32.0,  # LAXSAV
                5.0,
                7.0,
                7.0,
                32.0,  # SAVLAX
                13.0,
                4.0,
                7.0,
                25.0,  # MIASAV
                13.0,
                4.0,
                7.0,
                25.0,  # SAVMIA
            ),
            description="Gamma shape parameter alpha for total demand Λ_j per product",
        ),
    ]
    gamma_scale: Annotated[
        tuple[float, ...],
        Field(
            # See gamma_shape note: scale beta = 1 for moment-matched Gamma (80
            # products).
            default=tuple([1.0] * 80),
            description="Gamma scale parameter beta for total demand Λ_j per product",
        ),
    ]
    beta_alpha: Annotated[
        tuple[float, ...],
        Field(
            # Example 2: demand arrives low-to-high fare (§4.2): Q first, then B,
            # then M, then Y.  Within each group of 4 products (Y,M,B,Q):
            #   Y (index 0): Beta(5,2) peaks late   -- highest fare, last to arrive.
            #   M (index 1): Beta(3,2) moderate-late.
            #   B (index 2): Beta(2,3) moderate-early.
            #   Q (index 3): Beta(2,5) peaks early  -- lowest fare, first to arrive.
            default=tuple([5.0, 3.0, 2.0, 2.0] * 20),
            description="Beta alpha parameter governing temporal demand shape per product",  # noqa: E501
        ),
    ]
    beta_beta: Annotated[
        tuple[float, ...],
        Field(
            # Paired with beta_alpha above: [2,2,3,5] x 20 groups (80 products).
            default=tuple([2.0, 2.0, 3.0, 5.0] * 20),
            description="Beta beta parameter governing temporal demand shape per product",  # noqa: E501
        ),
    ]

    model_config = {"populate_by_name": True}

    # -- Helpers --

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

    # -- Validators --

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        n = self.n_classes
        K = self.n_virtual_classes  # noqa: N806
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

        # Per-product arrays
        for name in (
            "fares",
            "gamma_shape",
            "gamma_scale",
            "beta_alpha",
            "beta_beta",
        ):
            self._check_len(name, n)
        for name in (
            "fares",
            "gamma_shape",
            "gamma_scale",
            "beta_alpha",
            "beta_beta",
        ):
            self._check_positive(name)

        # Capacity must match number of legs
        self._check_len("capacity", n_legs, label="n_legs")
        self._check_positive("capacity")

        # Each product must use at least one leg
        for j in range(n):
            if np.sum(odf_mat[j]) == 0:
                raise ValueError(
                    f"Product {j} does not use any leg "
                    "(row of ODF_leg_matrix is all zeros)."
                )

        # ---- Virtual class indexing checks ----
        idx_mat = np.array(self.virtual_class_indexing)
        if idx_mat.shape != (n, n_legs):
            raise ValueError(
                f"virtual_class_indexing must have shape ({n}, {n_legs}), "
                f"got {idx_mat.shape}."
            )
        for j in range(n):
            for l_idx in range(n_legs):
                odf_val = odf_mat[j, l_idx]
                vc_val = idx_mat[j, l_idx]
                if odf_val == 1 and vc_val == 0:
                    raise ValueError(
                        f"Product {j} uses leg {l_idx} "
                        f"(ODF_leg_matrix[{j}][{l_idx}]=1) but "
                        f"virtual_class_indexing[{j}][{l_idx}]=0."
                    )
                if odf_val == 0 and vc_val != 0:
                    raise ValueError(
                        f"Product {j} does not use leg {l_idx} but "
                        f"virtual_class_indexing[{j}][{l_idx}]={vc_val} != 0."
                    )
                if vc_val < 0 or vc_val > K:
                    raise ValueError(
                        f"virtual_class_indexing[{j}][{l_idx}]={vc_val} "
                        f"must be in [0, {K}]."
                    )

        # ---- Protection levels checks ----
        prot = np.array(self.protection_levels)
        if prot.shape != (n_legs, K):
            raise ValueError(
                f"protection_levels must have shape ({n_legs}, {K}), got {prot.shape}."
            )
        for l_idx in range(n_legs):
            for k in range(K):
                if prot[l_idx, k] < 0:
                    raise ValueError(
                        f"protection_levels[{l_idx}][{k}]={prot[l_idx, k]} "
                        f"must be >= 0."
                    )
                if prot[l_idx, k] > self.capacity[l_idx]:
                    raise ValueError(
                        f"protection_levels[{l_idx}][{k}]={prot[l_idx, k]} "
                        f"exceeds capacity[{l_idx}]={self.capacity[l_idx]}."
                    )
            # Monotonicity
            for k in range(1, K):
                if prot[l_idx, k] < prot[l_idx, k - 1]:
                    raise ValueError(
                        f"protection_levels[{l_idx}] must be monotonically "
                        f"non-decreasing, but [{l_idx}][{k - 1}]="
                        f"{prot[l_idx, k - 1]} > [{l_idx}][{k}]="
                        f"{prot[l_idx, k]}."
                    )

        return self


class VanRyzinRevenueSingleStageConfig(VanRyzinRevenueConfigBase):
    """Config for the single-stage (full-horizon) virtual nesting model."""

    pass


class VanRyzinRevenueMultistageConfig(VanRyzinRevenueConfigBase):
    """Config for the multistage virtual nesting model (one period per stage)."""

    pass


# ======================================================================
# Demand helpers
# ======================================================================


#! DEMAND NEEDS REMODELLING, THE DEMAND FOR ODF j AT STAGE t is given as:
#! the minumum between the amount requested for the ODF class and the capacity -
# ! the protection class for the virtual class associated to the ODF class for classes
# below the virtual class.
def _beta_density(t: int, n_periods: int, alpha: float, beta: float) -> float:
    """Beta-density proportion for period *t* (standardised by booking period).

    Returns the proportion p_j(t) of total demand expected in period *t*.
    See :mod:`simopt.models.bertsimas_airline_revenue` for derivation.
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
    """Pre-compute demand proportions p_j(t) for all products and periods.

    Returns:
        Array of shape ``(n_classes, n_periods)`` where entry ``[j, t]``
        is the proportion of product-*j* total demand expected in period *t*.
    """
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
    """Sample total expected demand Λ_j for each product from Gamma priors.

    ``MRG32k3a.gammavariate(alpha, beta)`` uses the *scale* parametrisation:
    ``E[X] = alpha * beta``.
    """
    return [
        rng.gammavariate(alpha=s, beta=sc)
        for s, sc in zip(gamma_shape, gamma_scale, strict=True)
    ]


def _generate_period_demand(
    rng: MRG32k3a,  # noqa: ARG001
    expected_demand: list[float],
    proportions: np.ndarray,
    stage: int,
) -> list[float]:
    """Generate continuous fluid demand for each product in a single period.

    In the fluid model, the demand quantity for product *j* in period *t* is
    the deterministic fluid rate:

        q_{j,t} = Λ_j * p_j(t)

    where Λ_j is the (random) total expected demand sampled from the Gamma
    prior and p_j(t) is the Beta-shaped temporal proportion for this period.

    Args:
        rng: RNG (unused in the fluid model; retained for API compatibility).
        expected_demand: Sampled Λ_j for each product.
        proportions: Pre-computed proportions array (n_classes x n_periods).
        stage: Current period index.

    Returns:
        List of continuous demand quantities per product.
    """
    n_classes = len(expected_demand)
    demand: list[float] = []
    for j in range(n_classes):
        q = expected_demand[j] * proportions[j, stage]
        demand.append(q)
    return demand


# ======================================================================
# Virtual nesting booking helpers
# ======================================================================


@dataclass
class VNBookingResult:
    """Outcome of processing booking requests for one period under virtual nesting.

    All quantities are continuous (float) in the fluid model.

    Attributes:
        seats_sold: Updated cumulative units sold per product.
        remaining_capacity: Updated remaining capacity per leg.
        virtual_class_bookings: Updated cumulative bookings per (leg, VC).
        revenue: Revenue earned in this period.
        refused: Demand refused per product (sum of nesting and capacity
            refusals).  In the fluid model a product may be *partially*
            accepted, so ``refused[j]`` is the unmet demand quantity.
    """

    seats_sold: list[float]
    remaining_capacity: list[float]
    virtual_class_bookings: list[list[float]]
    revenue: float
    refused: list[float]
    # Gradient data (populated only when record_gradient_data=True)
    product_order: list[int] | None = None
    accepted_amounts: list[float] | None = None
    capacity_snapshots: list[list[float]] | None = None


def _process_vn_bookings(
    selling_rng: MRG32k3a,
    demand: list[float],
    protection_levels: list[list[float]],
    virtual_class_indexing: np.ndarray,
    odf_leg_matrix: np.ndarray,
    seats_sold: list[float],
    remaining_capacity: list[float],
    virtual_class_bookings: list[list[float]],
    fares: tuple[float, ...],
    record_gradient_data: bool = False,
    smoothing_perturbation: list[float] | None = None,
) -> VNBookingResult:
    """Process continuous fluid demand under virtual nesting with partial acceptance.

    Products are processed in a random order (shuffled).  For each product *j*
    with continuous demand quantity *q_j*, the amount accepted is:

        u_j = min(q_j,  min_{l ∈ L(j)}  avail_{l, sigma_l(j)})

    where:

        avail_{l,k} = max(0, y_{l,k} - Σ_{k'=1}^{k} bookings_{l,k'})

    is the remaining capacity available to virtual class *k* on leg *l* under
    the nested protection levels.  This implements the paper's fluid
    acceptance rule: a request is partially accepted up to the minimum
    available capacity across all required legs.

    When ``u_j > 0``:
    - ``seats_sold[j]`` is increased by ``u_j``.
    - ``remaining_capacity[l]`` is decreased by ``u_j`` for every leg *l*.
    - ``virtual_class_bookings[l][k-1]`` is increased by ``u_j`` for every
      leg *l*, where *k = sigma_l(j)*.

    Args:
        selling_rng: RNG used to shuffle the product processing order.
        demand: Continuous demand quantity per product in this period.
        protection_levels: Nested protection levels, shape
            ``(n_legs, n_virtual_classes)``.
        virtual_class_indexing: VC indexing matrix, shape
            ``(n_classes, n_legs)``.  1-based VC indices; 0 = not used.
        odf_leg_matrix: Binary adjacency, shape ``(n_classes, n_legs)``.
        seats_sold: Previous cumulative sales per product
            (**mutated in-place**).
        remaining_capacity: Capacity available on each leg
            (**mutated in-place**).
        virtual_class_bookings: Cumulative bookings per (leg, VC)
            (**mutated in-place**).
        fares: Fare per product.
        record_gradient_data: Whether to record gradient data.
        smoothing_perturbation: Optional perturbation for smoothing.

    Returns:
        :class:`VNBookingResult` with updated state and refusal quantities.
    """
    n_classes = len(demand)
    n_legs = len(remaining_capacity)
    refused = [0.0] * n_classes

    # Build a product-index list and shuffle to randomise processing order
    product_order = list(range(n_classes))
    for i in range(n_classes - 1, 0, -1):
        swap = int(selling_rng.uniform(0, i + 1)) % (i + 1)
        product_order[i], product_order[swap] = (
            product_order[swap],
            product_order[i],
        )

    # Gradient data storage
    grad_accepted: list[float] | None = None
    grad_cap_snapshots: list[list[float]] | None = None
    if record_gradient_data:
        grad_accepted = [0.0] * n_classes
        grad_cap_snapshots = [[0.0] * n_legs for _ in range(n_classes)]

    # Smoothing perturbation ξ (paper §2.1): subtract from capacity to ensure
    # differentiability a.s.  Only applied when computing gradients.
    if smoothing_perturbation is not None:
        effective_capacity = [
            remaining_capacity[i] - smoothing_perturbation[i] for i in range(n_legs)
        ]
    else:
        # Keep a distinct buffer so per-acceptance updates do not subtract
        # twice when no smoothing perturbation is used.
        effective_capacity = list(remaining_capacity)

    period_revenue = 0.0
    for j in product_order:
        q_j = demand[j]
        if q_j <= 0.0:
            continue

        legs = np.nonzero(odf_leg_matrix[j])[0].tolist()

        # Record capacity snapshot before this product's acceptance
        if record_gradient_data:
            if grad_cap_snapshots is None:
                raise RuntimeError("Gradient snapshots were not initialized.")
            for i in range(n_legs):
                grad_cap_snapshots[j][i] = effective_capacity[i]

        # Paper Eq. (2) - fluid acceptance rule under theft nesting:
        #   u_j(x(t), y, q) = min{ q,  (x_i(t) - y_{i, c_i(j)-1})^+  for i in A_j }
        available = q_j
        for leg in legs:
            k = int(virtual_class_indexing[j, leg])  # 1-based VC index = c_i(j)
            y_protection = 0.0 if k == 1 else protection_levels[leg][k - 2]
            avail_l_k = max(0.0, effective_capacity[leg] - y_protection)
            available = min(available, avail_l_k)

        if available <= 0.0:
            refused[j] += q_j
            continue

        # Accept ``available`` units (may be < q_j: partial acceptance)
        accepted = available
        if record_gradient_data:
            if grad_accepted is None:
                raise RuntimeError("Gradient accepted amounts were not initialized.")
            grad_accepted[j] = accepted

        seats_sold[j] += accepted
        for leg in legs:
            remaining_capacity[leg] -= accepted
            effective_capacity[leg] -= accepted
            k = int(virtual_class_indexing[j, leg])
            virtual_class_bookings[leg][k - 1] += accepted
        period_revenue += fares[j] * accepted
        refused[j] += q_j - accepted

    # Enforce the physical-state invariant: remaining capacity must be >= 0.
    # Snap tiny floating-point negatives to zero, but fail loudly on material
    # violations so model bugs are surfaced at the source.
    neg_tol = 1e-8
    for l_idx, cap in enumerate(remaining_capacity):
        if cap < -neg_tol:
            raise ValueError(
                "_process_vn_bookings produced materially negative "
                f"remaining_capacity[{l_idx}]={cap:.6g}."
            )
        if cap < 0.0:
            remaining_capacity[l_idx] = 0.0

    return VNBookingResult(
        seats_sold=seats_sold,
        remaining_capacity=remaining_capacity,
        virtual_class_bookings=virtual_class_bookings,
        revenue=period_revenue,
        refused=refused,
        product_order=product_order if record_gradient_data else None,
        accepted_amounts=grad_accepted,
        capacity_snapshots=grad_cap_snapshots,
    )


def _compute_sample_path_gradient(
    period_data: list[VNBookingResult],
    period_demands: list[list[float]],
    period_protection_levels: list[list[list[float]]],
    odf_leg_matrix: np.ndarray,
    virtual_class_indexing: np.ndarray,
    fares: tuple[float, ...],
    n_legs: int,
    n_virtual_classes: int,
) -> np.ndarray:
    """Compute the sample-path gradient via the backward recursion (Eqs 7-8, 10-11).

    Implements the Van Ryzin & Vulcano (2008) backward pass over the full
    sample path.  Each period's products (processed in shuffled order) act as
    individual "customers" in the recursion.

    Args:
        period_data: List of VNBookingResult per period (with gradient data).
        period_demands: Demand quantities per product per period.
        period_protection_levels: Protection levels used at each period,
            shape ``[n_periods][n_legs][n_virtual_classes]``.
        odf_leg_matrix: Binary adjacency (n_classes, n_legs).
        virtual_class_indexing: VC indexing (n_classes, n_legs), 1-based.
        fares: Fare per product.
        n_legs: Number of legs.
        n_virtual_classes: Number of virtual classes per leg.

    Returns:
        Flat gradient array of shape ``(n_periods * n_legs * n_virtual_classes,)``
        giving ∂R/∂y for each period's protection levels.
    """
    n_periods = len(period_data)
    n_legs * n_virtual_classes

    # dR/dx_i: marginal value of extra capacity on leg i (Eq 8)
    dR_dx = np.zeros(n_legs)  # noqa: N806
    # dR/dy_{stage, i, c}: gradient w.r.t. protection level at each stage (Eq 7)
    dR_dy = np.zeros((n_periods, n_legs, n_virtual_classes))  # noqa: N806

    # Backward pass: iterate periods in reverse
    for t in range(n_periods - 1, -1, -1):
        result = period_data[t]
        demands = period_demands[t]
        prot = period_protection_levels[t]
        order = result.product_order
        accepted = result.accepted_amounts
        cap_snap = result.capacity_snapshots

        if order is None or accepted is None or cap_snap is None:
            continue

        # Within each period, iterate products in reverse of processing order
        for j in reversed(order):
            q_j = demands[j]
            u_j = accepted[j]

            if q_j <= 0.0 or u_j <= 0.0:
                continue

            legs = np.nonzero(odf_leg_matrix[j])[0].tolist()

            # Determine if acceptance was constrained (0 < u_j < q_j)
            # This is the case where some leg is binding (Eq 10, conditions i-iv)
            constrained = u_j < q_j - 1e-12

            if not constrained:
                # Fully accepted: no binding protection level, derivatives are 0
                continue

            # Find the binding leg: the leg with minimum slack
            # slack_l = x_l - y_{l, c_l(j)-1} (the available capacity on leg l)
            binding_leg = -1
            min_slack = float("inf")
            for leg in legs:
                k = int(virtual_class_indexing[j, leg])  # 1-based VC index
                y_prot = 0.0 if k == 1 else prot[leg][k - 2]
                slack = cap_snap[j][leg] - y_prot
                if slack < min_slack:
                    min_slack = slack
                    binding_leg = leg

            if binding_leg < 0:
                continue

            # Marginal value of accepting one more unit of product j
            # = fare_j - displacement cost on all legs used by j
            # Displacement cost = sum of dR_{t+1}/dx_{i'} for i' in A_j
            marginal_value = fares[j] - sum(dR_dx[leg] for leg in legs)

            # Update dR/dx for the binding leg (Eq 8):
            # ∂u_j/∂x_{binding_leg} = +1 when binding
            dR_dx[binding_leg] += marginal_value * 1.0

            # Update dR/dy for the binding leg (Eq 7):
            # ∂u_j/∂y_{i,c} = -1 for all c < c_i(j) on the binding leg
            k_binding = int(virtual_class_indexing[j, binding_leg])  # 1-based
            for c in range(k_binding - 1):  # c < c_i(j), 0-indexed in array
                dR_dy[t, binding_leg, c] += marginal_value * (-1.0)

    # Flatten to (n_periods * dim_per_stage,)
    return dR_dy.reshape(-1)


# ======================================================================
# Single-stage model (full horizon in one replicate)
# ======================================================================


class VanRyzinRevenueSingleStage(Model):
    """Virtual nesting airline revenue model -- full horizon in one ``replicate()``.

    Simulates the entire booking horizon of *T* periods in a single call,
    using fixed nested protection levels under virtual nesting control.
    Returns the total revenue.
    """

    class_name_abbr: ClassVar[str] = "VANRYZIN-SL"
    class_name: ClassVar[str] = "Virtual Nesting Airline Revenue (Single Stage)"
    config_class: ClassVar[type[BaseModel]] = VanRyzinRevenueSingleStageConfig
    n_rngs: ClassVar[int] = 2  # demand RNG + selling-order RNG
    n_responses: ClassVar[int] = 1

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize instance."""
        super().__init__(fixed_factors)
        self._demand_rng: MRG32k3a | None = None
        self._selling_rng: MRG32k3a | None = None
        self._odf_leg_matrix = np.array(self.factors["ODF_leg_matrix"])
        self._vc_indexing = np.array(self.factors["virtual_class_indexing"])

    def before_replicate(self, rng_list: list[MRG32k3a]) -> None:  # noqa: D102
        self._demand_rng = rng_list[0]
        self._selling_rng = rng_list[1]

    def replicate(self) -> tuple[dict, dict]:
        """Simulate the full booking horizon and return total revenue.

        Returns:
            tuple: (responses dict, gradients dict).
        """
        assert self._demand_rng is not None and self._selling_rng is not None

        n_classes: int = self.factors["n_classes"]
        capacity: tuple = self.factors["capacity"]
        fares: tuple = self.factors["fares"]
        n_virtual_classes: int = self.factors["n_virtual_classes"]
        protection_levels: list[list[float]] = self.factors["protection_levels"]
        n_periods: int = self.factors["n_periods"]
        gamma_shape: tuple = self.factors["gamma_shape"]
        gamma_scale: tuple = self.factors["gamma_scale"]
        beta_alpha: tuple = self.factors["beta_alpha"]
        beta_beta: tuple = self.factors["beta_beta"]

        n_legs = len(capacity)

        # Pre-compute period proportions
        proportions = _period_proportions(n_periods, beta_alpha, beta_beta)

        # Sample total expected demand for each product (once per episode)
        expected_demand = _sample_expected_demand(
            self._demand_rng, gamma_shape, gamma_scale
        )

        seats_sold = [0.0] * n_classes
        remaining_capacity = list(capacity)
        virtual_class_bookings = [[0.0] * n_virtual_classes for _ in range(n_legs)]
        total_revenue = 0.0
        total_refused = [0.0] * n_classes

        for t in range(n_periods):
            demand = _generate_period_demand(
                self._demand_rng, expected_demand, proportions, t
            )
            result = _process_vn_bookings(
                self._selling_rng,
                demand,
                protection_levels,
                self._vc_indexing,
                self._odf_leg_matrix,
                seats_sold,
                remaining_capacity,
                virtual_class_bookings,
                fares,
            )
            seats_sold = result.seats_sold
            remaining_capacity = result.remaining_capacity
            virtual_class_bookings = result.virtual_class_bookings
            total_revenue += result.revenue
            for j in range(n_classes):
                total_refused[j] += result.refused[j]

        responses = {
            "revenue": total_revenue,
            "seats_sold": dict(enumerate(seats_sold)),
            "remaining_capacity": dict(enumerate(remaining_capacity)),
            "refused": dict(enumerate(total_refused)),
        }
        return responses, {}


# ======================================================================
# Multistage model (one period per stage)
# ======================================================================


class VanRyzinRevenueMultistage(MultistageModel):
    """Virtual nesting airline revenue model -- one booking period per stage.

    Each stage corresponds to one time period in the booking horizon.

    * **State**: :class:`VanRyzinState` (per-leg remaining capacity,
      per-(leg, VC) bookings, per-product cumulative sales, Gamma
      posterior parameters, sampled expected demand).
    * **Decision**: flattened tuple of nested protection levels for all
      ``(leg, virtual_class)`` pairs in leg-major order:
      ``(y_{0,1}, y_{0,2}, ..., y_{0,K}, y_{1,1}, ..., y_{L-1,K})``.
    * **Transition**: generate demand, process bookings under virtual
      nesting, update Bayesian demand posterior.
    * **Stage reward**: revenue earned in that period.

    The default rollout policy keeps the initially configured protection
    levels unchanged across all future stages.
    """

    class_name_abbr: ClassVar[str] = "VANRYZIN-ML"
    class_name: ClassVar[str] = "Virtual Nesting Airline Revenue (Multistage)"
    config_class: ClassVar[type[BaseModel]] = VanRyzinRevenueMultistageConfig
    n_rngs: ClassVar[int] = 2  # demand RNG + selling-order RNG
    n_responses: ClassVar[int] = 1
    n_stages: ClassVar[int] = 3  # overridden at __init__ from config

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize instance."""
        super().__init__(fixed_factors)
        # Keep n_stages instance-scoped to avoid cross-instance bleed
        self.__dict__["n_stages"] = self.factors["n_periods"]

        # Pre-compute temporal demand proportions
        self._proportions = _period_proportions(
            self.factors["n_periods"],
            self.factors["beta_alpha"],
            self.factors["beta_beta"],
        )

        # Cache numpy arrays
        self._odf_leg_matrix = np.array(self.factors["ODF_leg_matrix"])
        self._vc_indexing = np.array(self.factors["virtual_class_indexing"])

    # ---- Helper to convert flat decision tuple → nested protection levels ----

    def _decision_to_protection_levels(self, decision: tuple) -> list[list[float]]:
        """Convert a flat decision tuple into a protection levels matrix.

        The decision tuple is ordered leg-major:
        ``(y_{0,1}, ..., y_{0,K}, y_{1,1}, ..., y_{L-1,K})``.
        """
        n_legs = len(self.factors["capacity"])
        K: int = self.factors["n_virtual_classes"]  # noqa: N806
        prot: list[list[float]] = []
        idx = 0
        for _l in range(n_legs):
            row: list[float] = []
            for _k in range(K):
                row.append(float(decision[idx]))
                idx += 1
            prot.append(row)
        return prot

    # ---- MultistageModel abstract methods ----

    def get_initial_state(self) -> dict[str, list]:  # noqa: D102
        n_classes: int = self.factors["n_classes"]
        n_legs = len(self.factors["capacity"])
        K: int = self.factors["n_virtual_classes"]  # noqa: N806
        gamma_shape: tuple = self.factors["gamma_shape"]
        gamma_scale: tuple = self.factors["gamma_scale"]
        return {
            "remaining_capacity": list(self.factors["capacity"]),
            "virtual_class_bookings": [[0.0] * K for _ in range(n_legs)],
            "seats_sold": [0.0] * n_classes,
            "expected_demand": [],  # populated after before_replication
            "gamma_shape_posterior": list(gamma_shape),
            "gamma_rate_posterior": [1.0 / sc for sc in gamma_scale],
            "total_requests": [0.0] * n_classes,
        }

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

    def transition(
        self,
        state: object,
        decision: tuple,
        stage: int,
        rng_list: list[MRG32k3a],
    ) -> object:
        """The transition function simulates one booking period under virtual nesting.

        control,.

            generating demand, processing bookings, and updating the Bayesian posterior.

        Args:
            state (dict[str, list]): The current state of the system, including
            remaining capacity,
            decision (tuple): A flat tuple of nested protection levels for all (leg,
            virtual_class) pairs in leg-major order,
            stage (int): The current stage index in the booking horizon,
            rng_list (list[MRG32k3a]): A list of random number generators, where
            rng_list[0] is used for demand generation and rng_list[1] is used for
            shuffling the order of product processing.

        Returns:
            dict[str, list]: The next state of the system after processing the demand
            and updating the Bayesian posterior.
        """
        if not isinstance(state, dict):
            raise TypeError("state must be a dict[str, list].")
        state_dict = cast(dict[str, Any], state)

        demand_rng = rng_list[0]
        selling_rng = rng_list[1]

        n_classes: int = self.factors["n_classes"]

        # ----------------------------------------------------------
        # Determine which Λ to use for demand generation.
        # ----------------------------------------------------------
        state_expected_demand = cast(list[float], state_dict.get("expected_demand", []))
        expected_demand = (
            state_expected_demand
            if state_expected_demand
            else self._episode_expected_demand
        )

        # Generate demand (total booking requests) for this period
        demand = _generate_period_demand(
            demand_rng, expected_demand, self._proportions, stage
        )

        # Convert flat decision into protection levels matrix
        protection_levels = self._decision_to_protection_levels(decision)

        # Process bookings on the network (work on copies)
        new_seats_sold = list(cast(list[float], state_dict["seats_sold"]))
        new_remaining_capacity = list(
            cast(list[float], state_dict["remaining_capacity"])
        )
        new_vc_bookings = [
            list(row)
            for row in cast(list[list[float]], state_dict["virtual_class_bookings"])
        ]

        result = _process_vn_bookings(
            selling_rng,
            demand,
            protection_levels,
            self._vc_indexing,
            self._odf_leg_matrix,
            new_seats_sold,
            new_remaining_capacity,
            new_vc_bookings,
            self.factors["fares"],
        )
        self._last_booking_result = result

        # ----------------------------------------------------------
        # Bayesian Gamma conjugate update (fluid model).
        #
        # Prior / current posterior:
        #   Λ_j ~ Gamma(alpha_j, beta_j)        [rate parametrisation]
        #
        # Observation in period *t*:
        #   q_{j,t} = Λ_j * p_j(t)   (continuous fluid demand)
        #
        # Posterior update:
        #   alpha_j  ←  alpha_j + q_{j,t}
        #   beta_j  ←  beta_j + p_j(t)
        # ----------------------------------------------------------
        state_gamma_shape = cast(
            list[float],
            state_dict.get("gamma_shape_posterior", []),
        )
        state_gamma_rate = cast(list[float], state_dict.get("gamma_rate_posterior", []))
        state_total_requests = cast(list[float], state_dict.get("total_requests", []))
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
            list(state_total_requests) if state_total_requests else [0.0] * n_classes
        )

        for j in range(n_classes):
            new_gamma_shape[j] += demand[j]
            new_gamma_rate[j] += self._proportions[j, stage]
            new_total_requests[j] += demand[j]

        return {
            "remaining_capacity": result.remaining_capacity,
            "virtual_class_bookings": result.virtual_class_bookings,
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
            raise TypeError("state and next_state must be dict[str, list].")

        state_dict = cast(dict[str, Any], state)
        next_state_dict = cast(dict[str, Any], next_state)

        n_classes = self.factors["n_classes"]
        fares = self.factors["fares"]
        new_sales = [
            cast(list[float], next_state_dict["seats_sold"])[j]
            - cast(list[float], state_dict["seats_sold"])[j]
            for j in range(n_classes)
        ]
        revenue = sum(f * s for f, s in zip(fares, new_sales, strict=True))
        return {"revenue": revenue}

    def get_default_policy(self) -> Callable[[dict[str, list], int], tuple]:
        """Return a policy that uses the configured protection levels at every stage."""
        prot = self.factors["protection_levels"]
        # Flatten protection levels into a tuple (leg-major order)
        default_decision = tuple(
            prot[leg][v_class]
            for leg in range(len(prot))
            for v_class in range(len(prot[leg]))
        )

        def _policy(state: dict[str, list], stage: int) -> tuple:  # noqa: ARG001
            return default_decision

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
        """Replicate with rich per-product/per-leg detail for the current stage.

        Calls the base-class implementation (which accumulates *revenue*
        across the lookahead horizon), then appends the current stage's
        ``seats_sold``, ``remaining_capacity``, and ``refused``
        dicts keyed by product/leg index.
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

        if br is not None:
            rich_responses["seats_sold"] = dict(enumerate(br.seats_sold))
            rich_responses["remaining_capacity"] = dict(
                enumerate(br.remaining_capacity)
            )
            rich_responses["refused"] = dict(enumerate(br.refused))
        else:
            rich_responses["seats_sold"] = dict(enumerate([0.0] * n_classes))
            rich_responses["remaining_capacity"] = dict(
                enumerate(list(self.factors["capacity"]))
            )
            rich_responses["refused"] = dict(enumerate([0.0] * n_classes))

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

        if not isinstance(state, dict):
            raise TypeError("state must be a dict[str, list].")

        state_dict = cast(dict[str, Any], state)

        policy_fn = cast(
            Callable[[Any, int], tuple],
            policy or self.get_default_policy(),
        )
        n_classes: int = self.factors["n_classes"]

        total_responses: dict[str, float] = {}
        for _ in range(n_reps):
            # --- Sample Λ from the posterior for this replication ---
            lookahead_lambda = [
                self._demand_rng.gammavariate(
                    alpha=cast(list[float], state_dict["gamma_shape_posterior"])[j],
                    beta=1.0 / cast(list[float], state_dict["gamma_rate_posterior"])[j],
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

    # ---- Full-trajectory simulation with analytical gradient ----

    def simulate_with_gradient(
        self,
        decisions: tuple[tuple, ...],
        rng_list: list[MRG32k3a],
        smoothing_epsilon: float = 0.01,
    ) -> tuple[float, np.ndarray]:
        """Simulate full trajectory and compute analytical gradient.

        Based on Van Ryzin & Vulcano (2008).

        Performs a forward pass through all stages recording gradient data,
        then a backward pass to compute ∂R/∂y via the paper's recursion.

        Args:
            decisions: Per-stage decision tuples (protection levels, flat).
            rng_list: RNGs for demand and selling order.
            smoothing_epsilon: Perturbation magnitude for capacity smoothing.

        Returns:
            (total_revenue, grad_flat) where grad_flat has shape
            ``(n_stages * n_legs * n_virtual_classes,)``.
        """
        demand_rng = rng_list[0]
        selling_rng = rng_list[1]

        n_classes: int = self.factors["n_classes"]
        capacity: tuple = self.factors["capacity"]
        fares: tuple = self.factors["fares"]
        n_virtual_classes: int = self.factors["n_virtual_classes"]
        n_legs = len(capacity)

        # Sample episode's true expected demand Λ_j
        expected_demand = _sample_expected_demand(
            demand_rng,
            self.factors["gamma_shape"],
            self.factors["gamma_scale"],
        )

        # Forward pass: simulate all stages, recording gradient data
        period_data: list[VNBookingResult] = []
        period_demands: list[list[float]] = []
        period_protection_levels: list[list[list[float]]] = []

        seats_sold = [0.0] * n_classes
        remaining_capacity = list(capacity)
        virtual_class_bookings = [[0.0] * n_virtual_classes for _ in range(n_legs)]
        total_revenue = 0.0

        for t in range(self.n_stages):
            # Generate demand for this period
            demand = _generate_period_demand(
                demand_rng, expected_demand, self._proportions, t
            )
            period_demands.append(demand)

            # Convert flat decision to protection levels matrix
            prot = self._decision_to_protection_levels(decisions[t])
            period_protection_levels.append(prot)

            # Generate smoothing perturbation ξ ~ Uniform[0, epsilon]^m
            smoothing_pert = [
                selling_rng.uniform(0, smoothing_epsilon) for _ in range(n_legs)
            ]

            # Process bookings with gradient recording
            result = _process_vn_bookings(
                selling_rng,
                demand,
                prot,
                self._vc_indexing,
                self._odf_leg_matrix,
                seats_sold,
                remaining_capacity,
                virtual_class_bookings,
                fares,
                record_gradient_data=True,
                smoothing_perturbation=smoothing_pert,
            )

            period_data.append(result)
            seats_sold = result.seats_sold
            remaining_capacity = result.remaining_capacity
            virtual_class_bookings = result.virtual_class_bookings
            total_revenue += result.revenue

        # Backward pass: compute analytical gradient
        grad_flat = _compute_sample_path_gradient(
            period_data=period_data,
            period_demands=period_demands,
            period_protection_levels=period_protection_levels,
            odf_leg_matrix=self._odf_leg_matrix,
            virtual_class_indexing=self._vc_indexing,
            fares=fares,
            n_legs=n_legs,
            n_virtual_classes=n_virtual_classes,
        )

        return total_revenue, grad_flat


# ======================================================================
# Problem configuration
# ======================================================================


class VanRyzinRevenueSingleStageProblemConfig(BaseModel):
    """Config for the single-stage virtual nesting optimisation problem."""

    initial_solution: Annotated[
        tuple[float, ...],
        Field(
            default=(
                # Leg 0 (ATL→LAX, resource 1): Table 5, x_i=180
                6.0,
                34.0,
                42.0,
                49.0,
                61.0,
                75.0,
                88.0,
                180.0,
                180.0,
                180.0,
                # Leg 1 (LAX→ATL, resource 2)
                6.0,
                34.0,
                42.0,
                49.0,
                61.0,
                75.0,
                88.0,
                180.0,
                180.0,
                180.0,
                # Leg 2 (ATL→BOS, resource 3)
                7.0,
                18.0,
                30.0,
                59.0,
                73.0,
                103.0,
                180.0,
                180.0,
                180.0,
                180.0,
                # Leg 3 (BOS→ATL, resource 4)
                7.0,
                18.0,
                30.0,
                59.0,
                73.0,
                103.0,
                180.0,
                180.0,
                180.0,
                180.0,
                # Leg 4 (ATL→SAV, resource 5)
                3.0,
                11.0,
                18.0,
                38.0,
                63.0,
                81.0,
                96.0,
                180.0,
                180.0,
                180.0,
                # Leg 5 (SAV→ATL, resource 6)
                3.0,
                11.0,
                18.0,
                38.0,
                63.0,
                81.0,
                96.0,
                180.0,
                180.0,
                180.0,
                # Leg 6 (ATL→MIA, resource 7)
                25.0,
                52.0,
                62.0,
                79.0,
                98.0,
                107.0,
                141.0,
                180.0,
                180.0,
                180.0,
                # Leg 7 (MIA→ATL, resource 8)
                25.0,
                52.0,
                62.0,
                79.0,
                98.0,
                107.0,
                141.0,
                180.0,
                180.0,
                180.0,
            ),
            description=(
                "initial flattened protection levels in leg-major order: "
                "(y_{0,1}, ..., y_{0,K}, y_{1,1}, ..., y_{L-1,K})"
            ),
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


class VanRyzinRevenueMultistageProblemConfig(BaseModel):
    """Config for the multistage virtual nesting optimisation problem."""

    initial_solution: Annotated[
        tuple[float, ...],
        Field(
            default=(
                # Leg 0 (ATL→LAX, resource 1): Table 5, x_i=180
                6.0,
                34.0,
                42.0,
                49.0,
                61.0,
                75.0,
                88.0,
                180.0,
                180.0,
                180.0,
                # Leg 1 (LAX→ATL, resource 2)
                6.0,
                34.0,
                42.0,
                49.0,
                61.0,
                75.0,
                88.0,
                180.0,
                180.0,
                180.0,
                # Leg 2 (ATL→BOS, resource 3)
                7.0,
                18.0,
                30.0,
                59.0,
                73.0,
                103.0,
                180.0,
                180.0,
                180.0,
                180.0,
                # Leg 3 (BOS→ATL, resource 4)
                7.0,
                18.0,
                30.0,
                59.0,
                73.0,
                103.0,
                180.0,
                180.0,
                180.0,
                180.0,
                # Leg 4 (ATL→SAV, resource 5)
                3.0,
                11.0,
                18.0,
                38.0,
                63.0,
                81.0,
                96.0,
                180.0,
                180.0,
                180.0,
                # Leg 5 (SAV→ATL, resource 6)
                3.0,
                11.0,
                18.0,
                38.0,
                63.0,
                81.0,
                96.0,
                180.0,
                180.0,
                180.0,
                # Leg 6 (ATL→MIA, resource 7)
                25.0,
                52.0,
                62.0,
                79.0,
                98.0,
                107.0,
                141.0,
                180.0,
                180.0,
                180.0,
                # Leg 7 (MIA→ATL, resource 8)
                25.0,
                52.0,
                62.0,
                79.0,
                98.0,
                107.0,
                141.0,
                180.0,
                180.0,
                180.0,
            ),
            description=(
                "initial flattened protection levels in leg-major order: "
                "(y_{0,1}, ..., y_{0,K}, y_{1,1}, ..., y_{L-1,K})"
            ),
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


class VanRyzinRevenueSingleStageProblem(Problem):
    """Maximise expected total revenue by choosing nested protection levels.

    Full booking horizon simulated in one shot under virtual nesting
    control (standard ``Problem``).

    Decision variables are the flattened protection levels in leg-major
    order: ``(y_{0,1}, ..., y_{0,K}, y_{1,1}, ..., y_{L-1,K})``, subject
    to the constraint that protection levels are monotonically non-
    decreasing within each leg and bounded by the leg capacity.
    """

    class_name_abbr: ClassVar[str] = "VANRYZIN-1"
    class_name: ClassVar[str] = "Max Revenue Virtual Nesting Airline (Single Stage)"
    config_class: ClassVar[type[BaseModel]] = VanRyzinRevenueSingleStageProblemConfig
    model_class: ClassVar[type[Model]] = VanRyzinRevenueSingleStage
    n_objectives: ClassVar[int] = 1
    n_stochastic_constraints: ClassVar[int] = 0
    minmax: ClassVar[tuple[int, ...]] = (1,)  # maximise
    constraint_type: ClassVar[ConstraintType] = ConstraintType.DETERMINISTIC
    variable_type: ClassVar[VariableType] = VariableType.CONTINUOUS
    gradient_available: ClassVar[bool] = False
    optimal_solution: tuple | None = None
    model_default_factors: ClassVar[dict] = {}
    model_decision_factors: ClassVar[set[str]] = {"protection_levels"}

    def __init__(
        self,
        name: str = "",
        fixed_factors: dict | None = None,
        model_fixed_factors: dict | None = None,
    ) -> None:
        """Initialize instance."""
        super().__init__(name, fixed_factors, model_fixed_factors)

    @property
    def optimal_value(self) -> float | None:  # noqa: D102
        return None

    @property
    def dim(self) -> int:  # noqa: D102
        n_legs = len(self.model.factors["capacity"])
        K: int = self.model.factors["n_virtual_classes"]  # noqa: N806
        return n_legs * K

    @dim.setter
    def dim(self, value: int) -> None:
        self._dim = value

    @property
    def lower_bounds(self) -> tuple:  # noqa: D102
        return (0.0,) * self.dim

    @property
    def upper_bounds(self) -> tuple:  # noqa: D102
        capacity = self.model.factors["capacity"]
        K: int = self.model.factors["n_virtual_classes"]  # noqa: N806
        ub: list[float] = []
        for c in capacity:
            for _ in range(K):
                ub.append(float(c))
        return tuple(ub)

    def vector_to_factor_dict(self, vector: tuple) -> dict:  # noqa: D102
        K: int = self.model.factors["n_virtual_classes"]  # noqa: N806
        n_legs = len(self.model.factors["capacity"])
        prot: list[list[float]] = []
        idx = 0
        for _l in range(n_legs):
            row: list[float] = []
            for _k in range(K):
                row.append(float(vector[idx]))
                idx += 1
            prot.append(row)
        return {"protection_levels": prot}

    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:  # noqa: D102
        prot = factor_dict["protection_levels"]
        return tuple(
            prot[leg][v_class]
            for leg in range(len(prot))
            for v_class in range(len(prot[leg]))
        )

    def replicate(self, x: tuple) -> RepResult:  # noqa: ARG002, D102
        responses, _ = self.model.replicate()
        objectives = [Objective(stochastic=responses["revenue"])]
        return RepResult(objectives=objectives)

    def check_deterministic_constraints(self, x: tuple) -> bool:
        """Check non-negativity, monotonicity per leg, and capacity bounds."""
        if not all(v >= 0.0 for v in x):
            return False
        capacity = self.model.factors["capacity"]
        K: int = self.model.factors["n_virtual_classes"]  # noqa: N806
        idx = 0
        for c in capacity:
            prev = 0.0
            for _ in range(K):
                if x[idx] < prev or x[idx] > c:
                    return False
                prev = x[idx]
                idx += 1
        return True

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:  # noqa: D102
        capacity = self.model.factors["capacity"]
        K: int = self.model.factors["n_virtual_classes"]  # noqa: N806
        while True:
            vals: list[float] = []
            for c in capacity:
                # Generate K random continuous values and sort for monotonicity
                raw = sorted(rand_sol_rng.uniform(0.0, c) for _ in range(K))
                vals.extend(raw)
            result = tuple(vals)
            if self.check_deterministic_constraints(result):
                return result


# ======================================================================
# Multistage problem
# ======================================================================


class VanRyzinRevenueMultistageProblem(MultistageProblem):
    """Maximise expected total revenue by choosing protection levels at each stage.

    At each period the solver selects nested protection levels for all
    legs.  Revenue is evaluated as the immediate period revenue plus
    Monte Carlo estimated future revenue under a default rollout policy.
    """

    class_name_abbr: ClassVar[str] = "VANRYZIN-2"
    class_name: ClassVar[str] = "Max Revenue Virtual Nesting Airline (Multistage)"
    config_class: ClassVar[type[BaseModel]] = VanRyzinRevenueMultistageProblemConfig
    model_class: ClassVar[type[MultistageModel]] = VanRyzinRevenueMultistage
    n_objectives: ClassVar[int] = 1
    n_stochastic_constraints: ClassVar[int] = 0
    minmax: ClassVar[tuple[int, ...]] = (1,)  # maximise
    constraint_type: ClassVar[ConstraintType] = ConstraintType.DETERMINISTIC
    variable_type: ClassVar[VariableType] = VariableType.CONTINUOUS
    gradient_available: ClassVar[bool] = False
    model_default_factors: ClassVar[dict] = {}
    model_decision_factors: ClassVar[set[str]] = {"protection_levels"}
    n_lookahead_reps: ClassVar[int] = 30

    def __init__(
        self,
        name: str = "",
        fixed_factors: dict | None = None,
        model_fixed_factors: dict | None = None,
    ) -> None:
        """Initialize instance."""
        super().__init__(name, fixed_factors, model_fixed_factors)
        self.compute_gradients: bool = False

    @property
    def optimal_value(self) -> float | None:  # noqa: D102
        return None

    @property
    def dim(self) -> int:  # noqa: D102
        n_legs = len(self.model.factors["capacity"])
        K: int = self.model.factors["n_virtual_classes"]  # noqa: N806
        return n_legs * K

    @dim.setter
    def dim(self, value: int) -> None:
        self._dim = value

    @property
    def lower_bounds(self) -> tuple:  # noqa: D102
        return (0.0,) * self.dim

    @property
    def upper_bounds(self) -> tuple:  # noqa: D102
        # Use original capacity as the upper bound (protection levels are
        # cumulative limits, not relative to remaining capacity).
        capacity = self.model.factors["capacity"]
        K: int = self.model.factors["n_virtual_classes"]  # noqa: N806
        ub: list[float] = []
        for c in capacity:
            for _ in range(K):
                ub.append(float(c))
        return tuple(ub)

    def vector_to_factor_dict(self, vector: tuple) -> dict:  # noqa: D102
        K: int = self.model.factors["n_virtual_classes"]  # noqa: N806
        n_legs = len(self.model.factors["capacity"])
        prot: list[list[float]] = []
        idx = 0
        for _l in range(n_legs):
            row: list[float] = []
            for _k in range(K):
                row.append(float(vector[idx]))
                idx += 1
            prot.append(row)
        return {"protection_levels": prot}

    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:  # noqa: D102
        prot = factor_dict["protection_levels"]
        return tuple(
            prot[leg][v_class]
            for leg in range(len(prot))
            for v_class in range(len(prot[leg]))
        )

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

    def check_deterministic_constraints(self, x: tuple) -> bool:
        """Check non-negativity, monotonicity per leg, and capacity bounds."""
        if not all(v >= 0.0 for v in x):
            return False
        capacity = self.model.factors["capacity"]
        K: int = self.model.factors["n_virtual_classes"]  # noqa: N806
        idx = 0
        for c in capacity:
            prev = 0.0
            for _ in range(K):
                if x[idx] < prev or x[idx] > c:
                    return False
                prev = x[idx]
                idx += 1
        return True

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:  # noqa: D102
        capacity = self.model.factors["capacity"]
        K: int = self.model.factors["n_virtual_classes"]  # noqa: N806
        while True:
            vals: list[float] = []
            for c in capacity:
                raw = sorted(rand_sol_rng.uniform(0.0, c) for _ in range(K))
                vals.extend(raw)
            result = tuple(vals)
            if self.check_deterministic_constraints(result):
                return result

    def simulate_policy(
        self,
        solution: Solution,
        num_macroreps: int = 1,
    ) -> None:
        """Evaluate a policy over the full trajectory.

        When ``self.compute_gradients`` is True, uses the analytical gradient
        from Van Ryzin & Vulcano (2008) instead of the default policy rollout.
        Otherwise falls back to the base class implementation.
        """
        if not self.compute_gradients:
            super().simulate_policy(solution, num_macroreps)
            return

        # Extract open-loop decisions from solution
        if solution.policy is not None:
            policy = solution.policy
        else:

            def policy(state: dict, stage: int) -> float:  # noqa: ARG001
                return solution.x[stage]

        for _ in range(num_macroreps):
            self.model.before_replication(solution.rng_list)
            if self.before_replicate_override is not None:
                self.before_replicate_override(self.model, solution.rng_list)

            # Build per-stage decisions tuple
            n_stages = self.model.n_stages
            if hasattr(solution, "policy") and solution.policy is not None:
                state = self.model.get_initial_state()
                decisions = []
                for s in range(n_stages):
                    d = policy(state, s)
                    decisions.append(d)
                decisions = tuple(decisions)
            else:
                decisions = tuple(solution.x[s] for s in range(n_stages))

            # Call gradient-enabled simulation
            if not hasattr(self.model, "simulate_with_gradient"):
                raise TypeError(
                    "compute_gradients=True requires a model with "
                    "simulate_with_gradient()."
                )
            simulate_with_gradient = cast(
                Callable[..., tuple[float, np.ndarray]],
                self.model.simulate_with_gradient,
            )
            total_revenue, grad_flat = simulate_with_gradient(
                decisions=decisions,
                rng_list=self.model._rng_list,
            )

            # Build objective with gradient
            objectives = [
                Objective(
                    stochastic=total_revenue,
                    stochastic_gradients=tuple(grad_flat.tolist()),
                )
            ]
            solution.add_replicate_result(RepResult(objectives=objectives))

            for rng in solution.rng_list:
                rng.advance_subsubstream()
