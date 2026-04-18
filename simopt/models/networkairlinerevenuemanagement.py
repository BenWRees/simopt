"""Defines a Network Airline Revenue Management Model and Problem as presented in.

'Simulation-Based Booking Limits for Airline Revenue Management'
by Dimitri Bertsimas and Sanne de Boer.

The simulation model captures the dynamics of booking requests across multiple fare
classes,
for a single time window
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, ClassVar, Self

import numpy as np
from pydantic import BaseModel, Field, model_validator
from scipy.special import gamma

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import (
    ConstraintType,
    Model,
    Objective,
    Problem,
    RepResult,
    VariableType,
)


class AirlineRevenueModelConfig(BaseModel):
    """Configuration model for Airline Revenue Management simulation.

    A model for airline revenue management with multiple fare classes and stochastic
    demand.
    """

    num_classes: Annotated[
        int,
        Field(
            default=2,
            description="number of fare classes",
            gt=0,
        ),
    ]
    ODF_leg_matrix: Annotated[
        list[list[int]],
        Field(
            default_factory=lambda: [[1, 0], [1, 0], [0, 1], [0, 1], [1, 1], [1, 1]],
            description="Matrix of odf classes served by each leg",
        ),
    ]
    prices: Annotated[
        tuple[float, ...],
        Field(
            default=(300.0, 100.0, 150.0, 50.0, 100.0, 25.0),
            description="prices for each origin-destination fare class",
        ),
    ]
    capacity: Annotated[
        tuple[int, ...],
        Field(
            default=(100, 200),
            description="total capacity for the flight",
        ),
    ]
    booking_limits: Annotated[
        tuple[int, ...],
        Field(
            default=(10, 10, 10, 10, 10, 10),
            description="booking limits for each fare class and each origin-destination pair",  # noqa: E501
            alias="booking limits",
        ),
    ]
    alpha: Annotated[
        tuple[float, ...],
        Field(
            default=(2.0, 2.0, 2.0, 2.0, 2.0, 2.0),
            description="parameters for the Polya process governing booking requests for each odf",  # noqa: E501
        ),
    ]
    beta: Annotated[
        tuple[float, ...],
        Field(
            default=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            description="parameters for the Polya process governing booking requests for each odf",  # noqa: E501
        ),
    ]
    gamma_shape: Annotated[
        tuple[float, ...],
        Field(
            default=(2.0, 2.0, 2.0, 2.0, 2.0, 2.0),
            description="shape parameters for the gamma distribution governing expected demand for each odf",  # noqa: E501
        ),
    ]
    gamma_scale: Annotated[
        tuple[float, ...],
        Field(
            default=(50.0, 50.0, 50.0, 50.0, 50.0, 50.0),
            description="scale parameters for the gamma distribution governing expected demand for each odf",  # noqa: E501
        ),
    ]
    time_steps: Annotated[
        int,
        Field(
            default=1,
            description="number of time steps in the booking horizon",
            gt=0,
            alias="time steps",
        ),
    ]
    tau: Annotated[
        tuple[int, ...],
        Field(
            default=(1,),
            description="length of booking period for each time step",
        ),
    ]
    deterministic_flag: Annotated[
        bool,
        Field(
            default=True,
            description="flag to test Phillips Airline Revenue Management Problem",
            alias="deterministic flag",
        ),
    ]

    model_config = {"populate_by_name": True}

    def _check_odf_leg_matrix(self) -> None:
        adjacency_matrix = np.array(self.ODF_leg_matrix)
        if (
            len(adjacency_matrix.shape) != 2
            or adjacency_matrix.shape[0] == 0
            or adjacency_matrix.shape[1] == 0
        ):
            raise ValueError("ODF_leg_matrix must be a non-empty 2D list or array.")
        if not np.all(np.isin(adjacency_matrix, [0, 1])):
            raise ValueError("ODF_leg_matrix must contain only 0s and 1s.")
        if not np.any(np.sum(adjacency_matrix, axis=1) > 0):
            raise ValueError("Each leg must serve at least one fare class.")

    def _check_prices(self) -> None:
        odf, _ = np.array(self.ODF_leg_matrix).shape
        prices = list(self.prices)
        if len(prices) != odf:
            raise ValueError(
                "Length of prices must equal number of origin-destination fare classes."
            )
        if np.any(np.array(prices) <= 0):
            raise ValueError("All prices must be positive.")

    def _check_capacity(self) -> None:
        _, legs = np.array(self.ODF_leg_matrix).shape
        if len(self.capacity) != legs:
            raise ValueError("Length of capacity must equal number of legs.")
        if np.any(np.array(self.capacity) <= 0):
            raise ValueError("All capacity values must be positive.")

    def _check_booking_limits(self) -> None:
        odf_matrix = np.array(self.ODF_leg_matrix)
        odf, _ = odf_matrix.shape
        booking_limits = np.array(self.booking_limits).reshape(-1, 1)
        booking_limit_per_fare = (booking_limits.T @ odf_matrix).flatten()

        if len(booking_limits) != odf:
            raise ValueError(
                "The length of the booking limits should be equal to the number of odf classes."  # noqa: E501
            )
        if np.all(np.subtract(booking_limit_per_fare, np.array(self.capacity)) > 0):
            raise ValueError(
                "The booking limits for all fare classes across a flight must not exceed the capacity of that flight."  # noqa: E501
            )

    def _check_alpha(self) -> None:
        odf, _ = np.array(self.ODF_leg_matrix).shape
        if len(self.alpha) != odf:
            raise ValueError("Length of alpha must equal number of odf classes.")
        if np.any(np.array(self.alpha) <= 0):
            raise ValueError("All alpha values must be positive.")

    def _check_beta(self) -> None:
        odf, _ = np.array(self.ODF_leg_matrix).shape
        if len(self.beta) != odf:
            raise ValueError("Length of beta must equal number of odf classes.")
        if np.any(np.array(self.beta) <= 0):
            raise ValueError("All beta values must be positive.")

    def _check_gamma_shape(self) -> None:
        odf, _ = np.array(self.ODF_leg_matrix).shape
        if len(self.gamma_shape) != odf:
            raise ValueError("Length of gamma_shape must equal number of odf classes.")
        if np.any(np.array(self.gamma_shape) <= 0):
            raise ValueError("All gamma_shape values must be positive.")

    def _check_gamma_scale(self) -> None:
        odf, _ = np.array(self.ODF_leg_matrix).shape
        if len(self.gamma_scale) != odf:
            raise ValueError("Length of gamma_scale must equal number of odf classes.")
        if np.any(np.array(self.gamma_scale) <= 0):
            raise ValueError("All gamma_scale values must be positive.")

    def _check_tau(self) -> None:
        if len(self.tau) != self.time_steps:
            raise ValueError("Length of tau must equal number of time steps.")
        if np.any(np.array(self.tau) <= 0):
            raise ValueError("All tau values must be positive.")

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        self._check_odf_leg_matrix()
        self._check_prices()
        self._check_capacity()
        self._check_booking_limits()
        self._check_alpha()
        self._check_beta()
        self._check_gamma_shape()
        self._check_gamma_scale()
        self._check_tau()
        return self


class AirlineRevenueModel(Model):
    """A model for airline revenue management with multiple fare classes and stochastic.

    demand.
    """

    class_name_abbr: ClassVar[str] = "AIRLINE_REV"
    class_name: ClassVar[str] = "Airline Revenue Management"
    config_class: ClassVar[type[BaseModel]] = AirlineRevenueModelConfig
    n_rngs: ClassVar[int] = 2
    n_responses: ClassVar[int] = 2

    def __init__(self, fixed_factors: dict | None = None) -> None:  # noqa: D107
        super().__init__(fixed_factors)

    def compute_leg_number(self, odf_class: int) -> list[int]:
        """Compute the legs involved for the given odf class.

        Args:
            odf_class (int): Origin-destination fare class.

        Returns:
            list[int]: List of leg numbers that the odf class travels on.
        """
        adjacency_matrix = np.array(self.factors["ODF_leg_matrix"])
        odf_vector = np.zeros((1, adjacency_matrix.shape[0]))
        odf_vector[:, odf_class] = 1
        leg_numbers = odf_vector @ adjacency_matrix
        return np.nonzero(leg_numbers)[1].tolist()

    def beta_distribution(self, t: int, tau: int, alpha: float, beta: float) -> float:
        """Compute the cumulative distribution function (CDF) of the Beta distribution.

        Args:
            t (int): Current time step.
            tau (int): Total time steps.
            alpha (float): Alpha parameter of the Beta distribution.
            beta (float): Beta parameter of the Beta distribution.

        Returns:
            float: CDF value at time t.
        """
        t = t + 1  # Adjust for 1-based indexing
        first_term = (1 / tau) * (t / tau) ** (alpha - 1) * (1 - t / tau) ** (beta - 1)
        second_term = gamma(alpha + beta) / (gamma(alpha) * gamma(beta))
        return first_term * second_term

    def gamma_variate(self, rand: MRG32k3a, shape: float, scale: float) -> float:  # noqa: D417
        """Generate a random value that obeys the Gamma distribution.

        Args:
            shape (float): Shape parameter of the Gamma distribution.
            scale (float): Scale parameter of the Gamma distribution.

        Returns:
            float: Random variate from the Gamma distribution.
        """
        # Warning: a few older sources define the gamma distribution in terms
        # of alpha > -1.0
        if shape <= 0.0 or scale <= 0.0:
            raise ValueError("gammavariate: shape and scale must be > 0.0")

        random = rand.random
        if shape > 1.0:
            # Uses R.C.H. Cheng, "The generation of Gamma
            # variables with non-integral shape parameters",
            # Applied Statistics, (1977), 26, No. 1, p71-74

            ainv = np.sqrt(2.0 * shape - 1.0)
            bbb = shape - np.log(4.0)
            ccc = shape + ainv

            while True:
                u1 = random()
                if not 1e-7 < u1 < 0.9999999:
                    continue
                u2 = 1.0 - random()
                v = np.log(u1 / (1.0 - u1)) / ainv
                x = shape * np.exp(v)
                z = u1 * u1 * u2
                r = bbb + ccc * v - x
                if r + (1.0 + np.log(4.5)) - 4.5 * z >= 0.0 or r >= np.log(z):
                    return x * scale

        elif shape == 1.0:
            # expovariate(1/beta)
            return -np.log(1.0 - random()) * scale

        else:
            # alpha is between 0 and 1 (exclusive)
            # Uses ALGORITHM GS of Statistical Computing - Kennedy & Gentle
            while True:
                u = random()
                b = (np.e + shape) / np.e
                p = b * u
                x = p ** (1.0 / shape) if p <= 1.0 else -np.log((b - p) / shape)
                u1 = random()
                if p > 1.0:
                    if u1 <= x ** (shape - 1.0):
                        break
                elif u1 <= np.exp(-x):
                    break
            return x * scale

    def compute_arrivals(self, rand: MRG32k3a, t: int) -> list[int]:
        """Compute the number of arrivals for a given fare class at time t.

        Args:
            rand (MRG32k3a): Random number generator.
            t (int): Current time step.
            remaining_capacity (list[int]): Remaining capacity for each leg of the
            network.

        Returns:
            list[int]: Number of arrivals for each odf class at time t.
        """
        booking_limits = list(self.factors["booking limits"])
        odf_classes = len(booking_limits)
        arrivals = [0] * odf_classes

        tau = self.factors["tau"][t]
        alpha = list(self.factors["alpha"])
        beta = list(self.factors["beta"])
        shape = list(self.factors["gamma_shape"])
        scale = list(self.factors["gamma_scale"])

        for x in range(odf_classes):
            # Compute the expected demand using the Gamma distribution
            if t == 0:
                # For the first time step, we sample from the gamma distribution to get the expected demand  # noqa: E501
                expected_demand = self.gamma_variate(rand, shape[x], scale[x])
            # TODO: Work out parameters for gamma variable as they depend on number of booking requests up to time t-1  # noqa: E501
            else:
                # For subsequent time steps, we use the gamma distribution with values from the number of booking requests up to time t-1  # noqa: E501
                expected_demand = self.gamma_variate(rand, shape[x], scale[x])

            # Compute the probability of arrival using the Beta CDF
            prob_arrival = self.beta_distribution(t, tau, alpha[x], beta[x])

            # Compute the number of arrivals as a Poisson random variable
            arrivals[x] = np.random.poisson(expected_demand * prob_arrival)

        return arrivals

    def sell_seats(  # noqa: D102
        self,
        selling_rng: MRG32k3a,
        arrivals: Sequence[int],
        seats_sold: list[int],
        remaining_capacity: list[int],
    ) -> tuple[list[int], list[int]]:
        booking_limits = list(self.factors["booking limits"])
        arrivals_left = list(arrivals)
        len(booking_limits)

        # Randomly generate an odf class to arrive, check if they have any more arrivals
        while any(a > 0 for a in arrivals_left):
            # Get indices of OD classes that still have arrivals left
            available_odfs = [i for i, a in enumerate(arrivals_left) if a > 0]

            # Pick one uniformly at random
            odf_to_arrive = int(selling_rng.uniform(0, len(available_odfs)))
            odf_to_arrive = available_odfs[odf_to_arrive]

            # else assume that this arrival has occurred
            arrivals_left[odf_to_arrive] -= 1
            leg_nos = self.compute_leg_number(odf_to_arrive)

            # check the remaining capacity for the legs being flown
            current_caps = [remaining_capacity[leg_no] for leg_no in leg_nos]

            # if any leg that the odf class flies on is at capacity then skip selling
            if any(cap <= 0 for cap in current_caps):
                print(f"The capacity for the odf {odf_to_arrive} is full")
                continue
            # if booking limit is reached for this capacity then skip selling
            if seats_sold[odf_to_arrive] >= booking_limits[odf_to_arrive]:
                print(
                    f"The booking limit for the odf {odf_to_arrive} has been reached with seats_sold \n {seats_sold}"  # noqa: E501
                )
                arrivals_left[odf_to_arrive] = 0
                continue
            # Sell a seat for the odf class and reduce capacity across legs the odf class flies on  # noqa: E501
            seats_sold[odf_to_arrive] += 1
            for leg_no in leg_nos:
                remaining_capacity[leg_no] = remaining_capacity[leg_no] - 1

        print(
            f"The remaining capacity after a round of selling seats is {remaining_capacity}"  # noqa: E501
        )
        print(f"The seats that were sold are: {seats_sold}")
        return seats_sold, remaining_capacity

    # def sell_seats(self, selling_rng: MRG32k3a, arrivals: list[int], seats_sold: list[int], remaining_capacity: list[int]) -> tuple[list[int], list[int]]:  # noqa: E501
    #     """
    #     For a single time step, sell as many seats as possible given the arrivals,
    #     booking limits, and remaining capacity.
    #     """
    #     booking_limits = list(self.factors['booking limits'])
    #     odf_classes = len(booking_limits)

    #     print(f'arrivals: {arrivals}')

    #     for odf in range(odf_classes):
    #         arrival_no = arrivals[odf]
    #         leg_nos = self.compute_leg_number(odf)

    #         for _ in range(arrival_no):
    #             # check the remaining capacity for the legs being flown
    #             current_caps = [remaining_capacity[leg_no] for leg_no in leg_nos]

    #             # if any leg is full, can't sell this itinerary or if booking limit is reached, stop selling this ODF  # noqa: E501
    #             if all(cap > 0 for cap in current_caps) and seats_sold[odf] < booking_limits[odf]:  # noqa: E501
    #                 # sell one seat
    #                 seats_sold[odf] += 1

    #                 # reduce capacity across all legs this ODF flies on
    #                 for leg_no in leg_nos:
    #                     remaining_capacity[leg_no] =  remaining_capacity[leg_no] - 1

    #     print(f'The remaining capacity after a round of selling seats is {remaining_capacity}')  # noqa: E501
    #     print(f'The seats that were sold are: {seats_sold}')

    #     return seats_sold, remaining_capacity

    def compute_revenue(self, seats_sold: list[list[int]]) -> list[int | float]:
        """Compute the total revenue for all the seats_sold across fare classes for a.

        single time step.

        Args:
            seats_sold (list[list[int]]): Number of seats sold for each odf class at
            each time step.


        Returns:
            list[float]: Revenue generated from each odf class indexed at each time
            step.
        """
        # total_revenue = 0.0
        odfs = len(seats_sold[0])
        prices = list(self.factors["prices"])
        revenue_over_time_steps = [0.0] * odfs

        for seat_sold_per_time_step in seats_sold:
            # First calculate the revenue made for this time step
            revenue_per_time_step = [
                p * x for x, p in zip(seat_sold_per_time_step, prices, strict=False)
            ]
            # then add on the revenue made in this time step to the total revenue made in the previous time steps  # noqa: E501
            revenue_over_time_steps = [
                total_rev + new_rev
                for new_rev, total_rev in zip(
                    revenue_per_time_step, revenue_over_time_steps, strict=False
                )
            ]

        return revenue_over_time_steps

    def before_replicate(self, rng_list: list[MRG32k3a]) -> None:  # noqa: D102
        self.arrival_rng = rng_list[0]
        self.selling_rng = rng_list[1]

    def replicate(self) -> tuple[dict, dict]:
        """Simulate a single replication of the airline revenue management model.

        Returns:
            tuple[dict, dict]: A tuple containing:
                - responses (dict): Performance measures of interest, including:
                    - "revenue": Total revenue from the booking period.
                - gradients (dict): A dictionary of gradient estimates for
                    each response (None if not available).
        """
        #! Extract model parameters
        capacity = self.factors["capacity"]
        booking_limits = list(self.factors["booking limits"])
        odf = len(booking_limits)  # number of origin-destination fare classes
        time_steps = self.factors["time steps"]
        total_revenue = 0

        np.array(booking_limits)  # Remaining booking limits for

        if self.factors["deterministic flag"]:
            self.factors["capacity"] = (100, 120)

        remaining_capacity = list(capacity)  # Remaining capacity for each leg

        #! Designate sources of randomness
        arrival_rng = self.arrival_rng
        selling_rng = self.selling_rng
        """
            For each time step, we compute the number of arrivals for each odf class and
            then compute how many
            seats can be sold on the networrk based on the booking limits and remaining
            capacity.
            We then update the remaining capacity and booking limits accordingly.
        """
        seats_sold_all_time_steps = []

        for time_step in range(time_steps):
            seats_sold = [0] * odf
            if self.factors["deterministic flag"]:
                arrivals = (30, 60, 20, 80, 30, 40)
                self.factors["prices"] = (150, 100, 120, 80, 250, 170)
                self.factors["ODF_leg_matrix"] = [
                    [1, 0],
                    [1, 0],
                    [0, 1],
                    [0, 1],
                    [1, 1],
                    [1, 1],
                ]
                seats_sold, remaining_capacity = self.sell_seats(
                    selling_rng, arrivals, seats_sold, remaining_capacity
                )
                seats_sold_all_time_steps.append(seats_sold)

            else:
                arrivals = self.compute_arrivals(arrival_rng, time_step)

                seats_sold, remaining_capacity = self.sell_seats(
                    selling_rng, arrivals, seats_sold, remaining_capacity
                )

                seats_sold_all_time_steps.append(seats_sold)

        # Compute total revenue

        list_revenues_each_odf = self.compute_revenue(seats_sold_all_time_steps)

        # get the total revenues
        total_revenue = sum([np.sum(a) for a in list_revenues_each_odf])

        responses = {
            "revenue": total_revenue,
            "revenue per odf": list_revenues_each_odf,
        }

        return responses, {}


class AirlineRevenueBookingLimitProblemConfig(BaseModel):
    """Configuration model for Airline Revenue Booking Limit Problem."""

    initial_solution: Annotated[
        tuple[int, ...],
        Field(
            default=(10, 10, 10, 10, 10, 10),
            description="initial booking limits for each fare class",
        ),
    ]
    budget: Annotated[
        int,
        Field(
            default=5000,
            description="total capacity available",
            gt=0,
            json_schema_extra={"isDatafarmable": False},
        ),
    ]


class AirlineRevenueBookingLimitProblem(Problem):
    """Finds the optimal set of booking limits to maximize expected revenue."""

    class_name_abbr: ClassVar[str] = "AIRLINE-1"
    class_name: ClassVar[str] = "Airline Revenue Management Booking Limit Problem"
    config_class: ClassVar[type[BaseModel]] = AirlineRevenueBookingLimitProblemConfig
    model_class: ClassVar[type[Model]] = AirlineRevenueModel
    n_objectives: ClassVar[int] = 1
    n_stochastic_constraints: ClassVar[int] = 0
    minmax: ClassVar[tuple[int, ...]] = (1,)  # Maximize revenue
    constraint_type: ClassVar[ConstraintType] = ConstraintType.DETERMINISTIC
    variable_type: ClassVar[VariableType] = VariableType.DISCRETE
    gradient_available: ClassVar[bool] = False
    optimal_value: ClassVar[float | None] = None
    optimal_solution: tuple | None = None
    model_default_factors: ClassVar[dict] = {}
    model_decision_factors: ClassVar[set[str]] = {"booking limits"}

    @property
    def dim(self) -> int:  # noqa: D102
        return len(self.model.factors["booking limits"])

    @property
    def lower_bounds(self) -> tuple:  # noqa: D102
        return (0,) * self.dim

    @property
    def upper_bounds(self) -> tuple:  # noqa: D102
        odf_matrix = np.array(self.model.factors["ODF_leg_matrix"])
        capacity = list(self.model.factors["capacity"])
        odfs, _legs = odf_matrix.shape
        x = []
        for odf in range(odfs):
            # Find the legs the odf flies on
            odf_vector = np.zeros((1, odf_matrix.shape[0]))
            odf_vector[:, odf] = 1
            leg_numbers = odf_vector @ odf_matrix
            leg_numbers_for_odf = np.nonzero(leg_numbers)[1].tolist()

            # Find the capacity
            capacities_leg_flies_on = [capacity[a] for a in leg_numbers_for_odf]
            x.append(np.min(capacities_leg_flies_on))

        return tuple(x)

    def vector_to_factor_dict(self, vector: tuple) -> dict:  # noqa: D102
        return {"booking limits": vector[:]}

    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:  # noqa: D102
        return tuple(factor_dict["booking limits"])

    def replicate(self, _x: tuple) -> RepResult:  # noqa: D102
        responses, _ = self.model.replicate()
        objectives = [Objective(stochastic=responses["revenue"])]
        return RepResult(objectives=objectives)

    def check_deterministic_constraints(self, x: tuple) -> bool:  # noqa: D102
        return all(x[j] > 0 for j in range(self.dim))

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:  # noqa: D102
        return tuple(
            rand_sol_rng.mvnormalvariate(
                mean_vec=[0] * self.dim,
                cov=np.eye(self.dim),
                factorized=False,
            )
        )


class AirlineRevenueBidPriceProblemConfig(BaseModel):
    """Configuration model for Airline Revenue Bid Price Problem."""

    initial_solution: Annotated[
        tuple[float, ...],
        Field(
            default=(300.0, 100.0, 150.0, 50.0, 100.0, 25.0),
            description="initial prices for each fare class",
        ),
    ]
    budget: Annotated[
        int,
        Field(
            default=10,
            description="total capacity available",
            gt=0,
            json_schema_extra={"isDatafarmable": False},
        ),
    ]


class AirlineRevenueBidPriceProblem(Problem):
    """Finds the optimal set of fare prices to maximize expected revenue."""

    class_name_abbr: ClassVar[str] = "AIRLINE-2"
    class_name: ClassVar[str] = "Airline Revenue Management Bid Price Problem"
    config_class: ClassVar[type[BaseModel]] = AirlineRevenueBidPriceProblemConfig
    model_class: ClassVar[type[Model]] = AirlineRevenueModel
    n_objectives: ClassVar[int] = 1
    n_stochastic_constraints: ClassVar[int] = 0
    minmax: ClassVar[tuple[int, ...]] = (1,)  # Maximize revenue
    constraint_type: ClassVar[ConstraintType] = ConstraintType.DETERMINISTIC
    variable_type: ClassVar[VariableType] = VariableType.CONTINUOUS
    gradient_available: ClassVar[bool] = False
    optimal_value: ClassVar[float | None] = None
    optimal_solution: tuple | None = None
    model_default_factors: ClassVar[dict] = {}
    model_decision_factors: ClassVar[set[str]] = {"prices"}

    @property
    def dim(self) -> int:  # noqa: D102
        return len(self.model.factors["prices"])

    @property
    def lower_bounds(self) -> tuple:  # noqa: D102
        return (0,) * self.dim

    @property
    def upper_bounds(self) -> tuple:  # noqa: D102
        # TODO: Needs to be changed - has to do with adjacency matrix of flight?
        # Booking limits cannot exceed the capacity of the flight
        return (np.inf,) * self.dim

    def vector_to_factor_dict(self, vector: tuple) -> dict:  # noqa: D102
        return {"prices": vector[:]}

    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:  # noqa: D102
        return tuple(factor_dict["prices"])

    def replicate(self, _x: tuple) -> RepResult:  # noqa: D102
        responses, _ = self.model.replicate()
        objectives = [Objective(stochastic=responses["revenue"])]
        return RepResult(objectives=objectives)

    def check_deterministic_constraints(self, x: tuple) -> bool:  # noqa: D102
        return all(x[j] > 0 for j in range(self.dim))

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:  # noqa: D102
        return tuple(
            rand_sol_rng.mvnormalvariate(
                mean_vec=[0] * self.dim,
                cov=np.eye(self.dim),
                factorized=False,
            )
        )
