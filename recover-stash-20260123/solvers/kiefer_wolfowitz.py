#!/usr/bin/env python3
"""Kiefer-Wolfowitz (FDSA) Solver.

The Kiefer-Wolfowitz algorithm (FDSA) finds the optimal value of a stochastic
regression function using finite difference gradient approximations.
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
from simopt.solvers.utils import finite_diff


class KieferWolfowitzConfig(SolverConfig):
    """Configuration for Kiefer-Wolfowitz solver."""

    stepsize_a_coeff: Annotated[
        float,
        Field(
            default=0.25,
            gt=0,
            description="coefficient for stepsize function a: a_n = coeff/n",
        ),
    ]
    stepsize_c_coeff: Annotated[
        float,
        Field(
            default=2.0,
            gt=0,
            description="coefficient for stepsize function c: c_n = coeff/n^(1/3)",
        ),
    ]
    gradient_clipping_enabled: Annotated[
        bool,
        Field(
            default=True,
            description="whether gradient clipping is enabled",
        ),
    ]
    gradient_clipping_value: Annotated[
        float,
        Field(
            default=5.0,
            gt=0,
            description="maximum gradient norm when clipping is enabled",
        ),
    ]


class KieferWolfowitz(Solver):
    """The Kiefer-Wolfowitz (FDSA) solver.

    The Kiefer-Wolfowitz algorithm finds the optimal value of a stochastic
    regression function using finite difference gradient approximations.

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

    name: str = "KIEFERWOLFOWITZ"
    config_class: ClassVar[type[SolverConfig]] = KieferWolfowitzConfig
    class_name_abbr: ClassVar[str] = "KIEFERWOLFOWITZ"
    class_name: ClassVar[str] = "Kiefer-Wolfowitz"
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

    def stepsize_a(self, n: int) -> float:
        """Calculate stepsize a for iteration n.

        Args:
                n: Current iteration number.

        Returns:
                Stepsize value a_n = coeff/n.
        """
        return self.factors["stepsize_a_coeff"] / n

    def stepsize_c(self, n: int) -> float:
        """Calculate stepsize c for iteration n.

        Args:
                n: Current iteration number.

        Returns:
                Stepsize value c_n = coeff/n^(1/3).
        """
        return self.factors["stepsize_c_coeff"] / (n ** (1 / 3))

    def solve(self, problem: Problem) -> None:
        """Run a single macroreplication of the solver on a problem.

        Args:
                problem: Simulation-optimization problem to solve.
        """
        # Store reference to problem for use in other methods
        self.problem: Problem = problem

        grad_clip_value = self.factors["gradient_clipping_value"]
        grad_clip_enabled = self.factors["gradient_clipping_enabled"]

        # Initialize iteration counter (avoid division by zero)
        self.iteration_count = 1

        # Create initial solution
        self.incumbent_x = tuple(self.problem.factors["initial_solution"])
        self.incumbent_solution = self.create_new_solution(
            self.incumbent_x, self.problem
        )

        # simulate incumbent solution to get a first estimate
        self.problem.simulate(self.incumbent_solution, 10)
        self.current_fn_estimate = (self.incumbent_solution.objectives_mean.item())

        self.recommended_solns.append(self.incumbent_solution)
        self.intermediate_budgets.append(self.budget.used)
        self.budget_history.append(self.budget.used)   
        self.fn_estimates.append(self.current_fn_estimate)
        self.iterations.append(self.iteration_count)

        # intialise trackers for output
        # self.recommended_solns = []
        # self.intermediate_budgets = []
        # self.iterations = []
        # self.budget_history = []
        # self.fn_estimates = []

        # Get bounds
        lower_bound = np.array(self.problem.lower_bounds)
        upper_bound = np.array(self.problem.upper_bounds)

        while self.budget.remaining > 0:

            # Check proximity to bounds for finite difference direction
            # Check variable bounds.
            forward = np.isclose(
                self.incumbent_x, lower_bound, atol=self.factors["sensitivity"]
            ).astype(int)
            backward = np.isclose(
                self.incumbent_x, upper_bound, atol=self.factors["sensitivity"]
            ).astype(int)
            # 1 stands for forward, -1 stands for backward, 0 means central diff.
            bounds_check = np.subtract(forward, backward)

            # Calculate finite difference gradient approximation
            # diff = self.finite_diff(self.incumbent_solution, bounds_check)
            diff = finite_diff(self, self.incumbent_solution, bounds_check, problem, self.stepsize_c(self.iteration_count), 1)
            self.budget.request(2 * self.problem.dim)

            # Apply gradient clipping if enabled
            if grad_clip_enabled and np.linalg.norm(diff) >= grad_clip_value:
                diff = grad_clip_value * (diff / np.linalg.norm(diff))

            # Calculate new point using gradient step
            new_x = (
                np.array(self.incumbent_x).flatten()
                + self.problem.minmax[0] * self.stepsize_a(self.iteration_count) * diff
            )
            new_x = new_x.tolist()

            # Clip to bounds using clamp_with_epsilon
            new_x = [
                clamp_with_epsilon(
                    val, self.problem.lower_bounds[j], self.problem.upper_bounds[j]
                )
                for j, val in enumerate(new_x)
            ]

            # Create new solution and update incumbent
            self.incumbent_x = tuple(new_x)
            self.incumbent_solution = self.create_new_solution(
                self.incumbent_x, self.problem
            )

            self.problem.simulate(self.incumbent_solution, 10)
            self.current_fn_estimate = (self.incumbent_solution.objectives_mean.item())

            self.iteration_count += 1

            # Record solution
            self.recommended_solns.append(self.incumbent_solution)
            self.intermediate_budgets.append(self.budget.used)
            self.fn_estimates.append(self.current_fn_estimate)
            self.iterations.append(self.iteration_count)
            self.budget_history.append(self.budget.used)


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
