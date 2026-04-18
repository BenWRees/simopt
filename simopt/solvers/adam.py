"""First-order gradient-based optimization of stochastic objective functions.

An algorithm for first-order gradient-based optimization of
stochastic objective functions, based on adaptive estimates of lower-order moments.
"""

from __future__ import annotations

from typing import Annotated, ClassVar

import numpy as np
from pydantic import Field

from simopt.base import (
    ConstraintType,
    ObjectiveType,
    ProblemLike,
    Solver,
    SolverConfig,
    VariableType,
)
from simopt.solver import BudgetExhaustedError
from simopt.solvers.utils import finite_diff


class ADAMConfig(SolverConfig):
    """Configuration for ADAM solver."""

    r: Annotated[
        int,
        Field(
            default=30,
            gt=0,
            description="number of replications taken at each solution",
        ),
    ]
    beta_1: Annotated[
        float,
        Field(
            default=0.9,
            gt=0,
            lt=1,
            description="exponential decay of the rate for the first moment estimates",
        ),
    ]
    beta_2: Annotated[
        float,
        Field(
            default=0.999,
            lt=1,
            description="exponential decay rate for the second-moment estimates",
        ),
    ]
    alpha: Annotated[float, Field(default=0.5, gt=0, description="step size")]
    epsilon: Annotated[
        float,
        Field(default=1e-8, gt=0, description="a small value to prevent zero-division"),
    ]
    sensitivity: Annotated[
        float,
        Field(default=1e-7, gt=0, description="shrinking scale for variable bounds"),
    ]


class ADAM(Solver):
    """First-order gradient-based optimization of stochastic objective functions.

    An algorithm for first-order gradient-based optimization of
    stochastic objective functions, based on adaptive estimates of lower-order moments.
    """

    name: str = "ADAM"
    config_class: ClassVar[type[SolverConfig]] = ADAMConfig
    class_name_abbr: ClassVar[str] = "ADAM"
    class_name: ClassVar[str] = "ADAM"
    objective_type: ClassVar[ObjectiveType] = ObjectiveType.SINGLE
    constraint_type: ClassVar[ConstraintType] = ConstraintType.BOX
    variable_type: ClassVar[VariableType] = VariableType.CONTINUOUS
    gradient_needed: ClassVar[bool] = False

    def solve(self, problem: ProblemLike) -> None:  # noqa: D102
        # Default values.
        self.iteration_count = 0
        self.record_count = 0
        r: int = self.factors["r"]
        beta_1: float = self.factors["beta_1"]
        beta_2: float = self.factors["beta_2"]
        alpha: float = self.factors["alpha"]
        epsilon: float = self.factors["epsilon"]

        # Upper bound and lower bound.
        lower_bound = np.array(problem.lower_bounds)
        upper_bound = np.array(problem.upper_bounds)

        # Start with the initial solution.
        new_solution = self.create_new_solution(
            problem.factors["initial_solution"], problem
        )
        self.recommended_solns.append(new_solution)
        self.intermediate_budgets.append(self.budget.used)

        self.budget.request(r)
        problem.simulate(new_solution, r)

        self.budget_history.append(self.budget.used)
        self.fn_estimates.append(new_solution.objectives_mean.item())
        self.iterations.append(self.iteration_count)
        self.record_count += 1

        best_solution = new_solution

        # Initialize the first moment vector, the second moment vector,
        # and the timestep.
        m = np.zeros(problem.dim)
        v = np.zeros(problem.dim)
        t = 0
        try:
            while True:
                # Update timestep.
                t += 1
                # Check variable bounds.
                forward = np.isclose(
                    new_solution.x, lower_bound, atol=self.factors["sensitivity"]
                ).astype(int)
                backward = np.isclose(
                    new_solution.x, upper_bound, atol=self.factors["sensitivity"]
                ).astype(int)
                # 1 stands for forward, -1 stands for backward, 0 means central diff.
                bounds_check = np.subtract(forward, backward)

                # Use finite difference to estimate gradient if IPA gradient is
                # not available.
                finite_diff_budget = (
                    2 * problem.dim - np.count_nonzero(bounds_check)
                ) * r
                self.budget.request(int(finite_diff_budget))
                grad = finite_diff(
                    solver=self,
                    new_solution=new_solution,
                    bounds_check=bounds_check,
                    problem=problem,
                    stepsize=self.factors["alpha"],
                    r=self.factors["r"],
                )

                # Update biased first moment estimate.
                m = beta_1 * m + (1 - beta_1) * grad
                # Update biased second raw moment estimate.
                v = beta_2 * v + (1 - beta_2) * grad**2
                # Compute bias-corrected first moment estimate.
                mhat = m / (1 - beta_1**t)
                # Compute bias-corrected second raw moment estimate.
                vhat = v / (1 - beta_2**t)
                # Update new_x (vectorized) and apply box constraints
                new_x = new_solution.x - alpha * mhat / (np.sqrt(vhat) + epsilon)
                new_x = np.clip(new_x, lower_bound, upper_bound)

                for i in range(len(new_x)):
                    # Check variable bounds.
                    if new_x[i] > upper_bound[i]:
                        new_x[i] = upper_bound[i] - 10 ** (-7)
                    if new_x[i] < lower_bound[i]:
                        new_x[i] = lower_bound[i] + 10 ** (-7)

                # Create new solution based on new x
                new_solution = self.create_new_solution(tuple(new_x), problem)
                # Use r simulated observations to estimate the objective value.
                self.budget.request(r)
                problem.simulate(new_solution, r)

                if (new_solution.objectives_mean > best_solution.objectives_mean) ^ (
                    problem.minmax[0] < 0
                ):
                    best_solution = new_solution
                    self.recommended_solns.append(new_solution)
                    self.intermediate_budgets.append(self.budget.used)

                self.iteration_count += 1
                self.budget_history.append(self.budget.used)
                self.fn_estimates.append(new_solution.objectives_mean.item())
                self.iterations.append(self.iteration_count)
                self.record_count += 1
        except BudgetExhaustedError:
            if self.iterations[-1] != self.iteration_count:
                self.fn_estimates.append(best_solution.objectives_mean.item())
                self.budget_history.append(self.budget.used)
                self.iterations.append(
                    self.iteration_count
                    if self.record_count > 0
                    else self.iteration_count + 1
                )
