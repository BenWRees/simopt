#!/usr/bin/env python3
"""Mini-batch Stochastic Gradient Descent (SGD) Solver.

The mini-batch SGD solver uses finite difference gradient approximations
with multiple replications to find optimal solutions.
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


class SGDConfig(SolverConfig):
    """Configuration for SGD solver."""

    r: Annotated[
        int,
        Field(
            default=10,
            gt=0,
            description="number of replications taken at each solution",
        ),
    ]
    alpha: Annotated[
        float,
        Field(
            default=0.9,
            gt=0,
            description="step size",
        ),
    ]
    gradient_clipping_enabled: Annotated[
        bool,
        Field(
            default=False,
            description="whether gradient clipping is enabled",
        ),
    ]
    gradient_clipping_value: Annotated[
        float,
        Field(
            default=20.0,
            gt=0,
            description="maximum gradient norm when clipping is enabled",
        ),
    ]
    spsa_gradient: Annotated[
        bool,
        Field(
            default=False,
            description="flag for using an SPSA-like gradient",
        ),
    ]


class SGD(Solver):
    """The mini-batch Stochastic Gradient Descent (SGD) solver.

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

    name: str = "SGD"
    config_class: ClassVar[type[SolverConfig]] = SGDConfig
    class_name_abbr: ClassVar[str] = "SGD"
    class_name: ClassVar[str] = "Stochastic Gradient Descent"
    objective_type: ClassVar[ObjectiveType] = ObjectiveType.SINGLE
    constraint_type: ClassVar[ConstraintType] = ConstraintType.BOX
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

    def solve(self, problem: Problem) -> None:  # ty: ignore[invalid-method-override]
        """Run a single macroreplication of the solver on a problem.

        Args:
                problem: Simulation-optimization problem to solve.
        """
        # Store reference to problem for use in other methods
        self.problem: Problem = problem

        # Get solver parameters
        r = self.factors["r"]
        alpha = self.factors["alpha"]
        grad_clip_value = self.factors["gradient_clipping_value"]
        grad_clip_enabled = self.factors["gradient_clipping_enabled"]
        spsa_enabled = self.factors["spsa_gradient"]

        # Get bounds as numpy arrays for vectorized operations
        lower_bound = np.array(self.problem.lower_bounds)
        upper_bound = np.array(self.problem.upper_bounds)

        # Initialize iteration counter
        self.iteration_count = 1

        # Create initial solution
        self.incumbent_x = tuple(self.problem.factors["initial_solution"])
        self.incumbent_solution = self.create_new_solution(
            self.incumbent_x, self.problem
        )

        self.problem.simulate(self.incumbent_solution, r)
        self.current_fn_estimate = self.incumbent_solution.objectives_mean.item()

        self.recommended_solns.append(self.incumbent_solution)
        self.intermediate_budgets.append(self.budget.used)
        self.fn_estimates.append(self.current_fn_estimate)
        self.budget_history.append(self.budget.used)
        self.iterations.append(self.iteration_count)

        # self.intermediate_budgets = []
        # self.recommended_solns = []
        # self.fn_estimates = []
        # self.budget_history = []
        # self.iterations = []

        while self.budget.remaining > 0:
            # Check proximity to bounds for finite difference direction
            forward = np.isclose(self.incumbent_x, lower_bound, atol=1e-7).astype(int)
            backward = np.isclose(self.incumbent_x, upper_bound, atol=1e-7).astype(int)
            # BdsCheck: 1 stands for forward, -1 stands for backward, 0 means central
            # diff
            bounds_check = np.subtract(forward, backward)

            # Compute gradient approximation
            if spsa_enabled:
                grad = self.finite_diff_spsa(self.incumbent_solution, bounds_check)
                self.budget.request(2 * r)
            else:
                # grad = self.finite_diff(self.incumbent_solution, bounds_check)
                grad = finite_diff(
                    self, self.incumbent_solution, bounds_check, problem, 1e-8, r
                )
                self.budget.request(2 * self.problem.dim * r)

            # Apply gradient clipping if enabled
            if grad_clip_enabled and np.linalg.norm(grad) >= grad_clip_value:
                grad = grad_clip_value * (grad / np.linalg.norm(grad))

            # Vectorized update: new_x = current_x - alpha * grad
            new_x = np.array(self.incumbent_x).flatten() - alpha * grad
            new_x = new_x.tolist()

            # Clip to bounds using clip_with_epsilon
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
            self.problem.simulate(self.incumbent_solution, r)
            self.current_fn_estimate = self.incumbent_solution.objectives_mean.item()

            self.iteration_count += 1

            # Record solution
            self.intermediate_budgets.append(self.budget.used)
            self.recommended_solns.append(self.incumbent_solution)
            self.fn_estimates.append(self.current_fn_estimate)
            self.budget_history.append(self.budget.used)
            self.iterations.append(self.iteration_count)

    # def finite_diff(
    #     self, new_solution: Solution, bounds_check: np.ndarray
    # ) -> np.ndarray:
    #     """Compute finite difference approximation of the gradient.

    #     Uses multiple replications and averages the gradient estimates.

    #     Args:
    #             new_solution: The current iteration's solution.
    # bounds_check: Array indicating boundary check for finite difference type.
    #                     1 for forward, -1 for backward, 0 for central difference.

    #     Returns:
    #             The averaged gradient approximation at the current solution.
    #     """
    #     r = self.factors["r"]
    #     alpha = self.factors["alpha"]
    #     grads = np.zeros((self.problem.dim, r))

    #     for batch in range(r):
    #         grad = finite_difference_gradient(
    #             new_solution, self.problem, alpha=alpha, BdsCheck=bounds_check
    #         )
    #         grads[:, batch] = grad

    #     return np.mean(grads, axis=1)

    def finite_diff_spsa(
        self,
        new_solution: Solution,
        bounds_check: np.ndarray,  # noqa: ARG002
    ) -> np.ndarray:
        """Compute SPSA-like finite difference approximation of the gradient.

        Args:
                new_solution: The current iteration's solution.
                bounds_check: Array indicating boundary check for finite difference
                type.

        Returns:
                The averaged gradient approximation from SPSA-style estimates.
        """
        r = self.factors["r"]
        x_k = np.array(new_solution.x)
        c_k = self.factors["alpha"] / (self.iteration_count + 1) ** 0.101
        lower_bound = np.array(self.problem.lower_bounds)
        upper_bound = np.array(self.problem.upper_bounds)

        gbar = []
        for _ in range(r):
            delta = np.array(
                self.rng_list[2].choices([-1, 1], [0.5, 0.5], k=self.problem.dim)
            )

            thetaplus = x_k + c_k * delta
            thetaminus = x_k - c_k * delta

            # Check bounds
            if np.any(c_k * delta < lower_bound):
                new_displacement = np.abs(lower_bound - x_k)
                thetaminus = x_k - new_displacement

            if np.any(c_k * delta > upper_bound):
                new_displacement = np.abs(upper_bound - x_k)
                thetaplus = x_k + new_displacement

            thetaplus_sol = self.create_new_solution(tuple(thetaplus), self.problem)
            thetaminus_sol = self.create_new_solution(tuple(thetaminus), self.problem)

            self.problem.simulate(thetaplus_sol, 1)
            self.problem.simulate(thetaminus_sol, 1)

            finite_diff = (
                thetaplus_sol.objectives_mean - thetaminus_sol.objectives_mean
            ) / (2 * c_k)
            ghat = -1 * self.problem.minmax[0] * finite_diff * delta
            gbar.append(ghat)

        return np.mean(gbar, axis=0)


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
