import warnings  # noqa: D100, N999
from math import ceil, log

warnings.filterwarnings("ignore")

from math import floor  # noqa: E402

import numpy as np  # noqa: E402

from simopt.base import Problem  # noqa: E402

__all__ = [
    "ASTROMoRFSampling",
    "AdaptiveSampling",
    "BasicSampling",
    "OriginalAdaptiveSampling",
    "SamplingRule",
]


class SamplingRule:  # noqa: D101
    def __init__(self, tr_instance, sampling_rule) -> None:  # noqa: ANN001, D107
        self.tr_instance = tr_instance
        self.sampling_rule = sampling_rule
        self.kappa = None

    # When sampling_rule is called is the behaviour of the sampling rule
    def __call__(self, problem, k, current_solution, delta_k, sample_after=True):  # noqa: ANN001, ANN204, D102
        return self.sampling_rule(problem, k, current_solution, delta_k, sample_after)

    # def __call__(self, *params) :
    # 	current_solution, budget = self.sampling_instance(*params)
    # 	return current_solution, budget


# This is a basic dynamic sampling rule - samples the objective fuction more


class BasicSampling(SamplingRule):  # noqa: D101
    def __init__(self, tr_instance) -> None:  # noqa: ANN001, D107
        self.tr_instance = tr_instance
        super().__init__(tr_instance, self.sampling)

    def sampling(self, problem, k, current_solution, delta_k, sample_after=True):  # noqa: ANN001, ANN201, ARG002, D102
        # sample 10 times
        sample_number = 10
        problem.simulate(current_solution, sample_number)
        self.tr_instance.budget.request(sample_number)
        return current_solution


# ASTRODF originial adaptive sampling rule
class OriginalAdaptiveSampling(SamplingRule):  # noqa: D101
    def __init__(self, tr_instance) -> None:  # noqa: ANN001, D107
        self.kappa = 1
        self.tr_instance = tr_instance
        super().__init__(tr_instance, self.sampling)

    def sample_size(self, k, sig, delta_k):  # noqa: ANN001, ANN201, D102
        alpha_k = 1
        lambda_k = 10 * log(k, 10) ** 1.5
        kappa = 10**2
        return floor(
            max(
                5,
                lambda_k,
                (lambda_k * sig) / ((kappa ^ 2) * delta_k ** (2 * (1 + 1 / alpha_k))),
            )
        )
        # S_k = math.floor(max(lambda_k,(lambda_k*sig)/((kappa^2)*delta**(2*(1+1/alpha_k)))))  # noqa: E501

    def sampling(self, problem, k, current_solution, delta_k, sample_after=True):  # noqa: ANN001, ANN201, ARG002, D102
        # need to check there is existing result
        problem.simulate(current_solution, 1)
        self.tr_instance.budget.request(1)
        sample_size = 1

        # Adaptive sampling
        while True:
            problem.simulate(current_solution, 1)
            self.tr_instance.budget.request(1)
            sample_size += 1
            sig = current_solution.objectives_var
            if sample_size >= self.samplesize(k, sig, delta_k):
                break

        return current_solution


class AdaptiveSampling(SamplingRule):  # noqa: D101
    def __init__(self, tr_instance) -> None:  # noqa: ANN001, D107
        self.kappa = 1
        # self.pilot_run = None
        super().__init__(tr_instance, self.sampling_strategy)
        # self.tr_instance = tr_instance
        # self.expended_budget = 0
        self.delta_power = 2 if tr_instance.factors["crn_across_solns"] else 4

    def calculate_pilot_run(self, k, problem):  # noqa: ANN001, ANN201, D102
        lambda_min = self.tr_instance.factors["lambda_min"]
        lambda_max = self.tr_instance.budget.remaining
        return ceil(
            max(lambda_min * log(10 + k, 10) ** 1.1, min(0.5 * problem.dim, lambda_max))
            - 1
        )

    def calculate_kappa(self, k, problem, current_solution, delta_k):  # noqa: ANN001, ANN201, D102
        lambda_max = self.tr_instance.budget.remaining
        pilot_run = self.calculate_pilot_run(k, problem)

        # calculate kappa
        problem.simulate(current_solution, pilot_run)
        self.tr_instance.budget.request(pilot_run)

        # current_solution, expended_budget = self.__calculate_kappa(problem, current_solution, delta_k, expended_budget)  # noqa: E501
        sample_size = pilot_run

        while True:
            rhs_for_kappa = current_solution.objectives_mean
            sig2 = current_solution.objectives_var[0]

            self.kappa = (
                rhs_for_kappa * np.sqrt(pilot_run) / (delta_k ** (self.delta_power / 2))
            )
            stopping = self.get_stopping_time(sig2, delta_k, k, problem)
            if (
                sample_size >= min(stopping, lambda_max)
                or self.tr_instance.budget.remaining <= 0
            ):
                # calculate kappa
                self.kappa = (
                    rhs_for_kappa
                    * np.sqrt(pilot_run)
                    / (delta_k ** (self.delta_power / 2))
                )
                # print("kappa "+str(kappa))
                break
            problem.simulate(current_solution, 1)
            self.tr_instance.budget.request(1)
            sample_size += 1

        return current_solution

    def get_stopping_time(
        self, sig2: float, delta: float, k: int, problem: Problem
    ) -> int:
        """Compute the sample size based on adaptive sampling stopping rule using the.

        optimality gap.
        """
        pilot_run = self.calculate_pilot_run(k, problem)
        if self.kappa == 0:
            self.kappa = 1

        # compute sample size
        raw_sample_size = pilot_run * max(
            1, sig2 / (self.kappa**2 * delta**self.delta_power)
        )
        # Convert out of ndarray if it is
        if isinstance(raw_sample_size, np.ndarray):
            raw_sample_size = raw_sample_size[0]
        # round up to the nearest integer
        sample_size: int = ceil(raw_sample_size)
        return sample_size

    # this is sample_type = 'conditional after'
    def adaptive_sampling_1(self, problem, k, new_solution, delta_k):  # noqa: ANN001, ANN201, D102
        lambda_max = self.tr_instance.budget.remaining
        pilot_run = self.calculate_pilot_run(k, problem)

        problem.simulate(new_solution, pilot_run)
        self.tr_instance.budget.request(pilot_run)
        sample_size = pilot_run

        # adaptive sampling
        while True:
            sig2 = new_solution.objectives_var[0]
            stopping = self.get_stopping_time(sig2, delta_k, k, problem)
            if (
                sample_size >= min(stopping, lambda_max)
                or self.tr_instance.budget.remaining <= 0
            ):
                break
            problem.simulate(new_solution, 1)
            self.tr_instance.budget.request(1)
            sample_size += 1

        return new_solution

    # this is sample_type = 'conditional before'
    def adaptive_sampling_2(self, problem, k, new_solution, delta_k):  # noqa: ANN001, ANN201, D102
        lambda_max = self.tr_instance.budget.remaining
        sample_size = new_solution.n_reps
        sig2 = new_solution.objectives_var[0]

        while True:
            stopping = self.get_stopping_time(sig2, delta_k, k, problem)
            if (
                sample_size >= min(stopping, lambda_max)
                or self.tr_instance.budget.remaining <= 0
            ):
                break
            problem.simulate(new_solution, 1)
            self.tr_instance.budget.request(1)
            sample_size += 1
            sig2 = new_solution.objectives_var[0]
        return new_solution

    def sampling_strategy(self, problem, k, new_solution, delta_k, sample_after=True):  # noqa: ANN001, ANN201, D102
        if sample_after:
            return self.adaptive_sampling_1(problem, k, new_solution, delta_k)

        return self.adaptive_sampling_2(problem, k, new_solution, delta_k)


class ASTROMoRFSampling(AdaptiveSampling):
    """ASTRO-MoRF sampling rule."""

    def __init__(self, tr_instance) -> None:  # noqa: ANN001, D107
        self.kappa = 1
        super().__init__(tr_instance)
        self.delta_power = 2 if tr_instance.factors["crn_across_solns"] else 4
        self.tr_instance = tr_instance
