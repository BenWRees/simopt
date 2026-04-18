#!/usr/bin/env python3
"""Robbins-Monro Solver.

The Robbins-Monro algorithm finds the root of a stochastic function
using a predetermined target level (alpha).
"""

from __future__ import annotations

from typing import Annotated, ClassVar

import numpy as np
from pydantic import Field

from simopt.base import (
    ConstraintType,
    ObjectiveType,
    Problem,
    Solution,
    Solver,
    SolverConfig,
    VariableType,
)


class RobbinsMonroConfig(SolverConfig):
    """Configuration for Robbins-Monro solver."""

    stepsize_coeff: Annotated[
        float,
        Field(
            default=1 / 3,
            gt=0,
            description="coefficient for stepsize function: a_n = coeff/n",
        ),
    ]
    alpha: Annotated[
        float,
        Field(
            default=0.0,
            description="the target value of the function at the root",
        ),
    ]
    replication_size: Annotated[
        int,
        Field(
            default=5,
            gt=0,
            description="number of replications taken at each solution",
        ),
    ]


class RobbinsMonro(Solver):
    """The Robbins-Monro solver.

    The Robbins-Monro algorithm finds the root of a stochastic function
    using a predetermined target level (alpha).

    Attributes:
    ----------
    name : str
            Name of solver.
    objective_type : ObjectiveType
            Type of objective (single or multi).
    constraint_type : ConstraintType
            Type of constraints supported.
    variable_type : VariableType
            Type of decision variables.
    gradient_needed : bool
            Whether gradient of objective function is needed.
    """

    name: str = "ROBBINSMONRO"
    config_class: ClassVar[type[SolverConfig]] = RobbinsMonroConfig
    class_name_abbr: ClassVar[str] = "ROBBINSMONRO"
    class_name: ClassVar[str] = "Robbins-Monro"
    objective_type: ClassVar[ObjectiveType] = ObjectiveType.SINGLE
    constraint_type: ClassVar[ConstraintType] = ConstraintType.UNCONSTRAINED
    variable_type: ClassVar[VariableType] = VariableType.CONTINUOUS
    gradient_needed: ClassVar[bool] = False

    @property
    def iteration_count(self) -> int:
        """Get the current iteration count."""
        return self._iteration_count

    @iteration_count.setter
    def iteration_count(self, value: int) -> None:
        """Set the current iteration count."""
        self._iteration_count = value

    @property
    def incumbent_x(self) -> tuple[float, ...]:
        """Get the incumbent solution."""
        return self._incumbent_x

    @incumbent_x.setter
    def incumbent_x(self, value: tuple[float, ...]) -> None:
        """Set the incumbent solution."""
        self._incumbent_x = value

    @property
    def incumbent_solution(self) -> Solution:
        """Get the incumbent solution."""
        return self._incumbent_solution

    @incumbent_solution.setter
    def incumbent_solution(self, value: Solution) -> None:
        """Set the incumbent solution."""
        self._incumbent_solution = value

    def stepsize(self, n: int) -> float:
        """Calculate stepsize for iteration n.

        Args:
                n: Current iteration number.

        Returns:
                Stepsize value a_n = coeff/n.
        """
        return self.factors["stepsize_coeff"] / n

    def solve(self, problem: Problem) -> None:  # ty: ignore[invalid-method-override]
        """Run a single macroreplication of the solver on a problem.

        Args:
                problem: Simulation-optimization problem to solve.
        """
        # Store reference to problem for use in other methods
        self.problem: Problem = problem

        # Use problem's optimal value as alpha if available, otherwise use config
        alpha = getattr(self.problem, "optimal_value", self.factors["alpha"])
        replication_size = getattr(
            self.problem, "replication_size", self.factors["replication_size"]
        )

        # Initialize iteration counter (avoid division by zero)
        self.iteration_count = 1

        # Create initial solution
        self.incumbent_x = tuple(self.problem.factors["initial_solution"])
        self.incumbent_solution = self.create_new_solution(
            self.incumbent_x, self.problem
        )

        self.problem.simulate(self.incumbent_solution, replication_size)
        self.current_fn_estimate = self.incumbent_solution.objectives_mean.item()

        self.recommended_solns.append(self.incumbent_solution)
        self.intermediate_budgets.append(self.budget.used)
        self.fn_estimates.append(self.current_fn_estimate)
        self.budget_history.append(self.budget.used)
        self.iterations.append(self.iteration_count)

        # Record initial solution
        # self.recommended_solns = []
        # self.intermediate_budgets = []
        # self.fn_estimates = []
        # self.budget_history = []
        # self.iterations = []

        while self.budget.remaining > 0:
            # Simulate current solution
            # self.problem.simulate(self.incumbent_solution, replication_size)
            observation = self.incumbent_solution.objectives_mean[0]
            self.budget.request(replication_size)

            # Vectorized update: new_x = current_x + stepsize * (alpha - observation)
            new_x = np.array(self.incumbent_x) + self.stepsize(self.iteration_count) * (
                alpha - observation
            )
            new_x = new_x.tolist()

            # Clip to bounds using clamp_with_epsilon
            new_x = [
                clamp_with_epsilon(
                    val, self.problem.lower_bounds[j], self.problem.upper_bounds[j]
                )
                for j, val in enumerate(new_x)
            ]

            # Update incumbent
            self.incumbent_x = tuple(new_x)
            self.incumbent_solution = self.create_new_solution(
                self.incumbent_x, self.problem
            )
            self.problem.simulate(self.incumbent_solution, replication_size)
            self.current_fn_estimate = self.incumbent_solution.objectives_mean.item()

            self.iteration_count += 1

            # Record solution
            self.intermediate_budgets.append(self.budget.used)
            self.recommended_solns.append(self.incumbent_solution)
            self.fn_estimates.append(self.current_fn_estimate)
            self.budget_history.append(self.budget.used)
            self.iterations.append(self.iteration_count)


def clamp_with_epsilon(
    val: float, lower_bound: float, upper_bound: float, epsilon: float = 0.01
) -> float:
    """Clamp a value within bounds while avoiding exact boundary values.

    Adds a small epsilon to the lower bound or subtracts it from the upper bound
    if `val` lies outside the specified range.

    Args:
            val (float): The value to clamp.
            lower_bound (float): Minimum acceptable value.
            upper_bound (float): Maximum acceptable value.
            epsilon (float, optional): Small margin to avoid returning exact boundary
                    values. Defaults to 0.01.

    Returns:
            float: The adjusted value, guaranteed to lie strictly within the bounds.
    """
    if val <= lower_bound:
        return lower_bound + epsilon
    if val >= upper_bound:
        return upper_bound - epsilon
    return val
