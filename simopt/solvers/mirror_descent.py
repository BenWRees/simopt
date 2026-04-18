#!/usr/bin/env python3
"""Stochastic Mirror Descent (SMD) Solver.

The Stochastic Mirror Descent solver uses a Bregman divergence with a convex
potential function as a regularisation term to gradient descent.
"""

from __future__ import annotations

from typing import Annotated, ClassVar

import numpy as np
from pydantic import Field
from scipy.optimize import minimize

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


def clamp_with_epsilon(
    val: float, lower_bound: float, upper_bound: float, epsilon: float = 0.01
) -> float:
    """Clamp a value within bounds while avoiding exact boundary values.

    Args:
            val: The value to clamp.
            lower_bound: Minimum acceptable value.
            upper_bound: Maximum acceptable value.
            epsilon: Small margin to avoid returning exact boundary values.

    Returns:
            The adjusted value, guaranteed to lie strictly within the bounds.
    """
    if val <= lower_bound:
        return lower_bound + epsilon
    if val >= upper_bound:
        return upper_bound - epsilon
    return val


class MirrorDescentConfig(SolverConfig):
    """Configuration for Mirror Descent solver."""

    r: Annotated[
        int,
        Field(
            default=30,
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
            default=True,
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


class MirrorDescent(Solver):
    """The Stochastic Mirror Descent (SMD) solver.

    Uses a Bregman divergence with a convex potential function as a
    regularisation term to gradient descent.

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

    name: str = "MIRRORDESCENT"
    config_class: ClassVar[type[SolverConfig]] = MirrorDescentConfig
    class_name_abbr: ClassVar[str] = "MIRRORDESCENT"
    class_name: ClassVar[str] = "Mirror Descent"
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

    def mirror_map(self, x: np.ndarray) -> float:
        """Compute the mirror map (potential function).

        Args:
                x: Input vector.

        Returns:
                The value of the mirror map at x.
        """
        return float(0.5 * np.linalg.norm(x, ord=2) ** 2)

    def bregman_divergence(
        self, value_1: np.ndarray, value_2: np.ndarray, gradient: np.ndarray
    ) -> float:
        """Compute the Bregman divergence.

        Args:
                value_1: First point.
                value_2: Second point.
                gradient: Gradient of mirror map at value_2.

        Returns:
                The Bregman divergence D(value_1, value_2).
        """
        diff = value_1 - value_2
        taylor_exp = self.mirror_map(value_2) + np.dot(gradient, diff)
        return self.mirror_map(value_1) - taylor_exp

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

        # Initialize tracking lists
        # self.intermediate_budgets: list[int] = []
        # self.recommended_solns: list[Solution] = []
        # self.fn_estimates: list[float] = []
        # self.budget_history: list[int] = []
        # self.iterations: list[int] = []

        while self.budget.remaining > 0:
            curr_x = np.array(self.incumbent_x)

            # Check proximity to bounds for finite difference direction
            forward = np.isclose(curr_x, lower_bound, atol=1e-7).astype(int)
            backward = np.isclose(curr_x, upper_bound, atol=1e-7).astype(int)
            # BdsCheck: 1 stands for forward, -1 stands for backward, 0 means central
            # diff
            bounds_check = np.subtract(forward, backward)

            # Compute gradient of mirror map at current point
            grad_mirror = self.mirror_finite_diff(self.incumbent_solution, bounds_check)

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

            # Define Bregman divergence function for optimization
            def breg_div(x):  # noqa: ANN001, ANN202
                return self.bregman_divergence(x, curr_x, grad_mirror)  # noqa: B023

            # Solve the optimization problem
            def obj_fn(x):  # noqa: ANN001, ANN202
                return alpha * np.dot(x, grad) + breg_div(x)  # noqa: B023

            obj_new_x = minimize(obj_fn, curr_x)
            new_x = obj_new_x.x

            # Clip to bounds using clamp_with_epsilon
            new_x = [
                clamp_with_epsilon(val, lower_bound[j], upper_bound[j])
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

    def mirror_finite_diff(
        self, new_solution: Solution, bounds_check: np.ndarray
    ) -> np.ndarray:
        """Compute gradient approximation of the potential function.

        Args:
                new_solution: The current iteration's solution.
                bounds_check: Array indicating boundary check for finite difference
                type.
                        1 for forward, -1 for backward, 0 for central difference.

        Returns:
                The gradient approximation of the mirror descent function.
        """
        alpha = 0.001
        new_x = np.array(new_solution.x)
        lower_bound = np.array(self.problem.lower_bounds)
        upper_bound = np.array(self.problem.upper_bounds)
        dim = len(new_x)
        FnPlusMinus = np.zeros((dim, 3))  # noqa: N806
        grad = np.zeros(dim)

        for i in range(dim):
            x1 = new_x.copy()
            x2 = new_x.copy()
            steph1 = alpha
            steph2 = alpha

            # Check variable bounds
            if x1[i] + steph1 > upper_bound[i]:
                steph1 = np.abs(upper_bound[i] - x1[i])
            if x2[i] - steph2 < lower_bound[i]:
                steph2 = np.abs(x2[i] - lower_bound[i])

            # Decide stepsize based on bounds check
            if bounds_check[i] == 0:
                # Central diff
                FnPlusMinus[i, 2] = min(steph1, steph2)
                x1[i] = x1[i] + FnPlusMinus[i, 2]
                x2[i] = x2[i] - FnPlusMinus[i, 2]
            elif bounds_check[i] == 1:
                # Forward diff
                FnPlusMinus[i, 2] = steph1
                x1[i] = x1[i] + FnPlusMinus[i, 2]
            else:
                # Backward diff
                FnPlusMinus[i, 2] = steph2
                x2[i] = x2[i] - FnPlusMinus[i, 2]

            fn1, fn2 = 0.0, 0.0
            if bounds_check[i] != -1:
                fn1 = self.mirror_map(x1)
                FnPlusMinus[i, 0] = fn1

            if bounds_check[i] != 1:
                fn2 = self.mirror_map(x2)
                FnPlusMinus[i, 1] = fn2

            # Calculate gradient
            if bounds_check[i] == 0:
                grad[i] = (fn1 - fn2) / (2 * FnPlusMinus[i, 2])
            elif bounds_check[i] == 1:
                grad[i] = (fn1 - self.mirror_map(new_x)) / FnPlusMinus[i, 2]
            elif bounds_check[i] == -1:
                grad[i] = (self.mirror_map(new_x) - fn2) / FnPlusMinus[i, 2]

        return grad

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


# Keep backwards compatibility alias
Mirror_Descent = MirrorDescent
