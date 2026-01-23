#!/usr/bin/env python3
"""Complete Enumeration Rank and Selection Solver.

A rank and selection solver that evaluates all solutions from a provided list
of solution tuples. Each solution is simulated with a specified number of replications,
and the best solution is determined by comparing their objective function values.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from traceback import format_exc
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


class CompleteEnumerationConfig(SolverConfig):
    """Configuration for Complete Enumeration solver."""

    sample_size: Annotated[
        int,
        Field(
            default=1,
            gt=0,
            description="sample size (replications) per solution",
        ),
    ]
    solution_list: Annotated[
        list[tuple],
        Field(
            default_factory=list,
            description="list of solution tuples to evaluate",
        ),
    ]


class CompleteEnumeration(Solver):
    """Complete Enumeration Rank and Selection Solver.

    A solver that evaluates all solutions from a provided list of solution tuples.
    Each solution is simulated with a fixed number of replications, and solutions
    are compared to identify the best one. Supports both minimization and maximization
    problems, and can handle stochastic constraints.

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

    name: str = "ENUMERATION"
    config_class: ClassVar[type[SolverConfig]] = CompleteEnumerationConfig
    class_name_abbr: ClassVar[str] = "RANDS"
    class_name: ClassVar[str] = "Complete Enumeration"
    objective_type: ClassVar[ObjectiveType] = ObjectiveType.SINGLE
    constraint_type: ClassVar[ConstraintType] = ConstraintType.BOX
    variable_type: ClassVar[VariableType] = VariableType.DISCRETE
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

    def solve(self, problem: Problem) -> None:
        """Solve the given problem using Complete Enumeration.

        Args:
                problem: Simulation-optimization problem to solve.

        Raises:
                ValueError: if budget is insufficient for the number of solutions
                        and sample size.
        """
        # Store reference to problem for use in other methods
        self.problem: Problem = problem

        sample_size = self.factors["sample_size"]
        solution_list = self.factors["solution_list"]

        # Get problem constraints
        lower_bound = np.array(self.problem.lower_bounds)
        upper_bound = np.array(self.problem.upper_bounds)

        # Initialize iteration counter
        self.iteration_count = 1

        # Initialize tracking lists
        # self.intermediate_budgets: list[int] = []
        # self.recommended_solns: list[Solution] = []
        # self.fn_estimates: list[float] = []
        # self.budget_history: list[int] = []
        # self.iterations: list[int] = []

        # Validate solution list
        if len(solution_list) == 0:
            raise ValueError("solution_list must contain at least one solution tuple.")

        # Check for sufficiently large budget
        required_budget = sample_size * len(solution_list)
        if self.budget.total < required_budget:
            raise ValueError(
                f"Budget ({self.budget.total}) is insufficient. "
                f"Need at least {required_budget} replications "
                f"({sample_size} per solution × {len(solution_list)} solutions)."
            )

        try:
            # Clip solution tuples to bounds
            clipped_solution_list = []
            for soln_tuple in solution_list:
                clipped_soln = tuple(
                    clamp_with_epsilon(val, lower_bound[j], upper_bound[j])
                    for j, val in enumerate(soln_tuple)
                )
                clipped_solution_list.append(clipped_soln)

            # Create all the solutions to test
            all_solutions = [
                self.create_new_solution(soln_tuple, self.problem)
                for soln_tuple in clipped_solution_list
            ]

            # Request budget for all solutions
            self.budget.request(required_budget)

            # Simulate all solutions in parallel
            print(
                f"\nStarting Complete Enumeration of {len(all_solutions)} solutions "
                f"with {sample_size} replications each..."
            )
            n_jobs = 20
            if n_jobs == 1:
                # Sequential execution
                for sol in all_solutions:
                    self.problem.simulate(sol, sample_size)
            else:
                # Parallel execution of simulations
                def simulate_solution(sol):
                    print(f"    Simulating solution: {sol.x}...")
                    self.problem.simulate(sol, sample_size)
                    return sol

                with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                    futures = [
                        executor.submit(simulate_solution, sol) for sol in all_solutions
                    ]
                    for future in as_completed(futures):
                        sol = future.result()
                        print(f"    Completed simulation for solution: {sol.x}.")

            # Determine the best solution
            self.incumbent_solution = None
            best_is_feasible = False
            best_objective = None
            print("\nEvaluating solutions to identify the best one...")

            for sol in all_solutions:
                if self.incumbent_solution is None:
                    # First solution - set as best regardless
                    self.incumbent_solution = sol
                    self.incumbent_x = sol.x
                    best_is_feasible = all(
                        sol.x[i] <= upper_bound[i] and sol.x[i] >= lower_bound[i]
                        for i in range(len(sol.x))
                    )
                    best_objective = -1 * self.problem.minmax[0] * sol.objectives_mean
                    self.current_fn_estimate = best_objective[0]

                    # Record initial solution
                    self.recommended_solns.append(self.incumbent_solution)
                    self.intermediate_budgets.append(self.budget.used)

                    self.fn_estimates.append(self.incumbent_solution.objectives_mean.item())
                    self.budget_history.append(self.budget.used)
                    self.iterations.append(self.iteration_count)
                else:
                    # Compare solutions
                    is_feasible = all(
                        sol.x[i] <= upper_bound[i] and sol.x[i] >= lower_bound[i]
                        for i in range(len(sol.x))
                    )
                    should_update = False

                    if is_feasible and not best_is_feasible:
                        # New solution is feasible and current best is not
                        should_update = True
                    elif is_feasible == best_is_feasible:
                        # Both feasible or both infeasible - compare objectives
                        objective_diff = (
                            -1 * self.problem.minmax[0] * sol.objectives_mean
                            - best_objective
                        )

                        # Check if new solution is better (accounting for min/max)
                        if all(objective_diff < 0):
                            should_update = True

                    if should_update:
                        self.incumbent_solution = sol
                        self.incumbent_x = sol.x
                        best_is_feasible = is_feasible
                        best_objective = (
                            -1 * self.problem.minmax[0] * sol.objectives_mean
                        )
                        self.current_fn_estimate = best_objective[0]

                        # Record updated solution
                        self.recommended_solns.append(self.incumbent_solution)
                        self.intermediate_budgets.append(self.budget.used)
                        print(
                            f"    *** New best solution found! "
                            f"Objective: {best_objective[0]:.4f}"
                        )

                self.iteration_count += 1

                self.fn_estimates.append(sol.objectives_mean.item())
                self.budget_history.append(self.budget.used)
                self.iterations.append(self.iteration_count)

        except Exception as e:
            print(
                f"An error occurred during Complete Enumeration: "
                f"{e.__class__.__name__!s}: {e!s}"
            )
            print(format_exc())
