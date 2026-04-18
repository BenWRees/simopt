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
    MultistageProblem,
    ObjectiveType,
    Solution,
    Solver,
    SolverConfig,
    VariableType,
)
from simopt.problem_types import SolverProblemType
from simopt.solvers.utils import finite_diff


def _finite_diff_policy(
    solver: VanRyzinSolver,
    flat_x: np.ndarray,
    fn: float,
    bounds_check: np.ndarray,
    problem: MultistageProblem,
    stepsize: float,
    r: int,
) -> np.ndarray:
    """Finite-difference gradient over a flat open-loop policy vector.

    Works like :func:`~simopt.solvers.utils.finite_diff` but evaluates
    each perturbation via :meth:`MultistageProblem.simulate_policy`.
    """
    full_dim = len(flat_x)
    lower_bound = np.array(solver._policy_lower)
    upper_bound = np.array(solver._policy_upper)

    function_diff = np.zeros((full_dim, 3))

    step_forward = np.minimum(stepsize, upper_bound - flat_x)
    step_backward = np.minimum(stepsize, flat_x - lower_bound)

    central_mask = bounds_check == 0
    forward_mask = bounds_check == 1
    backward_mask = bounds_check == -1

    function_diff[:, 2] = np.where(
        central_mask,
        np.minimum(step_forward, step_backward),
        np.where(forward_mask, step_forward, step_backward),
    )

    x1 = np.tile(flat_x, (full_dim, 1))
    x2 = np.tile(flat_x, (full_dim, 1))
    np.fill_diagonal(x1, flat_x + function_diff[:, 2])
    np.fill_diagonal(x2, flat_x - function_diff[:, 2])

    x1_indices = np.where(bounds_check != -1)[0]
    x2_indices = np.where(bounds_check != 1)[0]

    minmax = -1 * problem.minmax[0]

    for i in x1_indices:
        sol = solver._simulate_policy_vec(problem, x1[i], r)
        fn1 = minmax * sol.objectives_mean
        function_diff[i, 0] = fn1[0] if isinstance(fn1, np.ndarray) else fn1

    for i in x2_indices:
        sol = solver._simulate_policy_vec(problem, x2[i], r)
        fn2 = minmax * sol.objectives_mean
        function_diff[i, 1] = fn2[0] if isinstance(fn2, np.ndarray) else fn2

    fn_divisor = function_diff[:, 2].copy()
    fn_divisor[central_mask] *= 2

    fn_diff = np.zeros(full_dim)
    if np.any(central_mask):
        fn_diff[central_mask] = (
            function_diff[central_mask, 0] - function_diff[central_mask, 1]
        )
    if np.any(forward_mask):
        fn_diff[forward_mask] = function_diff[forward_mask, 0] - fn
    if np.any(backward_mask):
        fn_diff[backward_mask] = fn - function_diff[backward_mask, 1]

    return fn_diff / fn_divisor


class VanRyzinConfig(SolverConfig):
    """Configuration for Van Ryzin solver."""

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
    not_use_adp_solver: Annotated[
        bool,
        Field(
            default=True,
            description="flag for not using ADP solver for multistage problems, which changes the budget accounting for gradient estimation",  # noqa: E501
        ),
    ]
    use_direct_gradients: Annotated[
        bool,
        Field(
            default=True,
            description="use analytical gradients from the model instead of finite differences (requires VanRyzin model)",  # noqa: E501
        ),
    ]


class VanRyzinSolver(Solver):
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

    name: str = "VanRyzinSolver"
    config_class: ClassVar[type[SolverConfig]] = VanRyzinConfig
    class_name_abbr: ClassVar[str] = "VANRYZIN_SGD"
    class_name: ClassVar[str] = "Stochastic Gradient Descent Van Ryzin Variant"
    supported_problem_type: ClassVar[SolverProblemType] = SolverProblemType.MULTISTAGE
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
        """Incumbent x."""
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

    # ------------------------------------------------------------------
    # MultistageProblem helpers
    # ------------------------------------------------------------------

    def _simulate_policy_vec(
        self,
        problem: MultistageProblem,
        flat_x: np.ndarray,
        n_reps: int,
    ) -> Solution:
        """Evaluate an open-loop policy given a flat decision vector.

        Reshapes *flat_x* into per-stage tuples, creates a policy solution,
        and evaluates via :meth:`MultistageProblem.simulate_policy`.
        """
        decisions = self._flat_x_to_decisions(flat_x, problem)
        sol = problem.create_policy_solution(decisions)
        sol.attach_rngs(rng_list=self.solution_progenitor_rngs, copy=True)
        problem.simulate_policy(sol, num_macroreps=n_reps)
        # Enforce budget accounting for multistage policy evaluation
        if hasattr(self, "budget") and self.budget is not None:
            if getattr(self.config, "not_use_adp_solver", False) and isinstance(
                problem, MultistageProblem
            ):
                self.budget.request(n_reps * problem.model.n_stages)
            else:
                self.budget.request(n_reps)
        return sol

    def _flat_x_to_decisions(
        self, flat_x: np.ndarray, problem: MultistageProblem
    ) -> tuple[tuple, ...]:
        """Reshape a flat vector into per-stage decision tuples."""
        d = problem.dim
        n = problem.model.n_stages
        flat = np.asarray(flat_x, dtype=float).ravel()

        # Allow a stage-0 vector and broadcast it to every stage.
        if flat.size == d:
            flat = np.tile(flat, n)
        elif flat.size != d * n:
            raise ValueError(
                "Policy vector length mismatch: expected either "
                f"{d} (single stage) or {d * n} (all stages), got {flat.size}."
            )

        return tuple(tuple(flat[s * d : (s + 1) * d].tolist()) for s in range(n))

    def _record_policy_solution(
        self, flat_x: np.ndarray, problem: MultistageProblem
    ) -> None:
        """Append a policy solution snapshot for the current checkpoint."""
        decisions = self._flat_x_to_decisions(flat_x, problem)
        ps = problem.create_policy_solution(decisions)
        ps.attach_rngs(rng_list=self.solution_progenitor_rngs, copy=True)
        self.policy_solutions.append(ps)

    # ------------------------------------------------------------------
    # Main solve
    # ------------------------------------------------------------------

    def stepsize(self, iteration: int) -> float:
        """Compute the step size for a given iteration."""
        alpha = self.factors["alpha"]
        return alpha / iteration

    def solve(self, problem: MultistageProblem) -> None:  # ty: ignore[invalid-method-override]
        """Run a single macroreplication of the solver on a problem.

        Args:
                problem: Simulation-optimization problem to solve.
        """
        # Store reference to problem for use in other methods
        self.problem = problem  # type: ignore[assignment]

        # Get solver parameters
        r = self.factors["r"]
        use_direct_grad = self.factors["use_direct_gradients"]

        # Detect whether we should optimise the full open-loop policy or
        # just the current stage (used by wrapped/ADP sub-problems).
        is_multistage = isinstance(problem, MultistageProblem)
        stage0_only = is_multistage and not getattr(
            problem, "_solver_lookahead_enabled", True
        )

        # Enable analytical gradients only for full-policy optimisation.
        if (
            use_direct_grad
            and is_multistage
            and not stage0_only
            and hasattr(problem, "compute_gradients")
        ):
            problem.compute_gradients = True  # ty: ignore[invalid-assignment]

        if is_multistage and not stage0_only:
            n_stages = problem.model.n_stages
            stage_dim = problem.dim
            full_dim = stage_dim * n_stages
            lower_bound = np.tile(np.array(problem.lower_bounds), n_stages)
            upper_bound = np.tile(np.array(problem.upper_bounds), n_stages)
        else:
            full_dim = problem.dim
            lower_bound = np.array(self.problem.lower_bounds)
            upper_bound = np.array(self.problem.upper_bounds)

        # Used by _finite_diff_policy.
        self._policy_lower = lower_bound
        self._policy_upper = upper_bound

        # Initialize iteration counter
        self.iteration_count = 1

        # Create initial solution and flat decision vector
        if is_multistage and not stage0_only:
            stage0_x = np.array(self.problem.factors["initial_solution"], dtype=float)
            # Open-loop start: repeat stage-0 controls across all stages.
            flat_x = np.tile(stage0_x, problem.model.n_stages)
            self.incumbent_x = tuple(flat_x.tolist())
            self.incumbent_solution = self._simulate_policy_vec(problem, flat_x, r)
        else:
            self.incumbent_x = tuple(self.problem.factors["initial_solution"])
            self.incumbent_solution = self.create_new_solution(
                self.incumbent_x, self.problem
            )
            flat_x = np.array(self.incumbent_x, dtype=float)
            self.problem.simulate(self.incumbent_solution, r)

        print(
            "[VANRYZIN] Starting solve: "
            f"budget.total={self.budget.total} "
            f"dim={full_dim} "
            f"direct_gradients={use_direct_grad and getattr(problem, 'compute_gradients', False)}"  # noqa: E501
        )

        self.current_fn_estimate = self.incumbent_solution.objectives_mean.item()

        self.recommended_solns.append(self.incumbent_solution)
        self.intermediate_budgets.append(self.budget.used)
        self.fn_estimates.append(self.current_fn_estimate)
        self.budget_history.append(self.budget.used)
        self.iterations.append(self.iteration_count)

        if is_multistage and not stage0_only:
            self._record_policy_solution(
                np.array(self.incumbent_x, dtype=float), problem
            )

        while self.budget.remaining > 0:
            print(
                "[SGD] Iteration start: "
                f"k={self.iteration_count} budget.used={self.budget.used} "
                f"budget.remaining={self.budget.remaining}"
            )

            bounds_check = np.sign(
                np.minimum(flat_x - lower_bound, upper_bound - flat_x)
            )

            if (
                use_direct_grad
                and is_multistage
                and not stage0_only
                and getattr(problem, "compute_gradients", False)
            ):
                # Use analytical gradient from Van Ryzin & Vulcano (2008)
                sol = self._simulate_policy_vec(problem, flat_x, r)
                # objectives_gradients_mean[0] is the gradient of the objective
                # minmax[0] = +1 for maximisation; SGD minimises, so negate
                grad = (
                    -1 * problem.minmax[0] * np.array(sol.objectives_gradients_mean[0])
                )
            elif is_multistage and not stage0_only:
                # Fall back to finite differences over the full open-loop policy.
                fn_val = -1 * problem.minmax[0] * self.current_fn_estimate
                grad = _finite_diff_policy(
                    solver=self,
                    flat_x=flat_x,
                    fn=fn_val,
                    bounds_check=bounds_check,
                    problem=problem,
                    stepsize=self.stepsize(self.iteration_count),
                    r=r,
                )
            else:
                # Stage-0 mode: standard finite-difference gradient.
                grad = finite_diff(
                    self,
                    self.incumbent_solution,
                    bounds_check,
                    problem=problem,
                    stepsize=1e-8,
                    r=r,
                )
                self.budget.request(2 * self.problem.dim * r)

            # Gradient descent step
            flat_x = flat_x - self.stepsize(self.iteration_count) * grad

            # Clip to bounds
            flat_x = np.array(
                [
                    clamp_with_epsilon(val, lower_bound[j], upper_bound[j])
                    for j, val in enumerate(flat_x)
                ]
            )

            # Evaluate candidate solution
            candidate_x = tuple(flat_x.tolist())

            if is_multistage and not stage0_only:
                candidate_solution = self._simulate_policy_vec(problem, flat_x, r)
            else:
                candidate_solution = self.create_new_solution(candidate_x, self.problem)
                self.problem.simulate(candidate_solution, r)
                self.budget.request(r)

            candidate_fn_estimate = candidate_solution.objectives_mean.item()

            # Update incumbent if objective improved
            if (
                problem.minmax[0] * candidate_fn_estimate
                > problem.minmax[0] * self.current_fn_estimate
            ):
                print(
                    f"[SGD] Incumbent updated: {self.current_fn_estimate:.6g} -> {candidate_fn_estimate:.6g}"  # noqa: E501
                )
                self.incumbent_x = candidate_x
                self.incumbent_solution = candidate_solution
                self.current_fn_estimate = candidate_fn_estimate

            self.iteration_count += 1

            # Record solution
            self.intermediate_budgets.append(self.budget.used)
            self.recommended_solns.append(self.incumbent_solution)
            self.fn_estimates.append(self.current_fn_estimate)
            self.budget_history.append(self.budget.used)
            self.iterations.append(self.iteration_count)

            if is_multistage and not stage0_only:
                self._record_policy_solution(
                    np.array(self.incumbent_x, dtype=float), problem
                )

            print(
                "[VANRYZIN] Iteration complete: "
                f"k={self.iteration_count} fn_est={self.current_fn_estimate:.6g} "
                f"budget.used={self.budget.used} budget.remaining={self.budget.remaining}"  # noqa: E501
            )

    def finite_diff_gradient(
        self,
        problem: MultistageProblem,
        decision: Solution,
    ) -> np.ndarray:
        """Compute a finite-difference gradient estimate of the revenue function based on VANRYZIN."""  # noqa: E501
        raise NotImplementedError


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
