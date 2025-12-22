"""Complete Enumeration Rank and Selection Solver.

A rank and selection solver that evaluates all solutions from a provided list
of solution tuples. Each solution is simulated with a specified number of replications,
and the best solution is determined by comparing their objective function values.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from traceback import format_exc

from simopt.base import (
	ConstraintType,
	ObjectiveType,
	Problem,
	Solver,
	VariableType,
)
from simopt.utils import classproperty, override


class CompleteEnumeration(Solver):
	"""Complete Enumeration Rank and Selection Solver.

	A solver that evaluates all solutions from a provided list of solution tuples.
	Each solution is simulated with a fixed number of replications, and solutions
	are compared to identify the best one. Supports both minimization and maximization
	problems, and can handle stochastic constraints.
	"""

	@classproperty
	@override
	def class_name_abbr(cls) -> str:
		return "RANDS"

	@classproperty
	@override
	def class_name(cls) -> str:
		return "Complete Enumeration"

	@classproperty
	@override
	def objective_type(cls) -> ObjectiveType:
		return ObjectiveType.SINGLE

	@classproperty
	@override
	def constraint_type(cls) -> ConstraintType:
		return ConstraintType.BOX

	@classproperty
	@override
	def variable_type(cls) -> VariableType:
		return VariableType.DISCRETE

	@classproperty
	@override
	def gradient_needed(cls) -> bool:
		return False

	@classproperty
	@override
	def specifications(cls) -> dict[str, dict]:
		return {
			"crn_across_solns": {
				"description": "use CRN across solutions?",
				"datatype": bool,
				"default": True,
			},
			"sample_size": {
				"description": "sample size (replications) per solution",
				"datatype": int,
				"default": 1,
			},
			"solution_list": {
				"description": "list of solution tuples to evaluate",
				"datatype": list,
				"default": [],
			},
		}

	@property
	@override
	def check_factor_list(self) -> dict[str, Callable]:
		return {
			"crn_across_solns": self.check_crn_across_solns,
			"sample_size": self._check_sample_size,
			"solution_list": self._check_solution_list,
		}

	def __init__(
		self, name: str = "ENUMERATION", fixed_factors: dict | None = None
	) -> None:
		"""Initialize Complete Enumeration solver.

		Args:
			name (str): user-specified name for solver
			fixed_factors (dict, optional): fixed_factors of the solver.
				Defaults to None.
		"""
		# Let the base class handle default arguments.
		super().__init__(name, fixed_factors)

	def _check_sample_size(self) -> None:
		if self.factors["sample_size"] <= 0:
			raise ValueError("Sample size must be greater than 0.")

	def _check_solution_list(self) -> None:
		if not isinstance(self.factors["solution_list"], list):
			raise TypeError("solution_list must be a list.")
		if len(self.factors["solution_list"]) == 0:
			raise ValueError(
				"solution_list must contain at least one solution tuple."
			)
		# Check that all elements are tuples
		for i, soln in enumerate(self.factors["solution_list"]):
			if not isinstance(soln, tuple):
				raise TypeError(
					f"Element {i} of solution_list must be a tuple, "
					f"got {type(soln).__name__}."
				)

	@override 
	def solve(self, problem: Problem) -> None:
		"""Solve the given problem using Complete Enumeration.

		Args:
			problem (Problem): problem to solve

		Raises:
			ValueError: if budget is insufficient for the number of solutions
				and sample size.
		"""
		sample_size = self.factors["sample_size"]
		solution_list = self.factors["solution_list"]

		# Get problem constraints
		problem_lb = problem.lower_bounds
		problem_ub = problem.upper_bounds

		# Check for sufficiently large budget
		required_budget = sample_size * len(solution_list)
		if self.budget.total < required_budget:
			raise ValueError(
				f"Budget ({self.budget.total}) is insufficient. "
				f"Need at least {required_budget} replications "
				f"({sample_size} per solution × {len(solution_list)} solutions)."
			)
		
		try :
			# Create all the solutions to test 
			all_solutions = [
				self.create_new_solution(soln_tuple, problem) for soln_tuple in solution_list
			]

			# Request budget for all solutions
			self.budget.request(required_budget)

			# Simulate all solutions in parallel
			print(f"\nStarting Complete Enumeration of {len(all_solutions)} solutions with {sample_size} replications each...")
			n_jobs = 20
			if n_jobs == 1:
				# Sequential execution
				for sol in all_solutions:
					problem.simulate(sol, sample_size)
			else:
				# Parallel execution of simulations 
				def simulate_solution(sol):
					print(f"    Simulating solution: {sol.x}...")
					problem.simulate(sol, sample_size)
					return sol
				
				with ThreadPoolExecutor(max_workers=n_jobs) as executor:
					futures = [executor.submit(simulate_solution, sol) for sol in all_solutions]
					for future in as_completed(futures):
						sol = future.result()
						print(f"    Completed simulation for solution: {sol.x}.")

			# Determine the best solution
			best_solution = None
			best_is_feasible = False
			best_objective = None
			print("\nEvaluating solutions to identify the best one...")
			for sol in all_solutions : 
				if best_solution is None :
					# First solution - set as best regardless
					best_solution = sol
					best_is_feasible = all(
						sol.x[i] <= problem_ub[i] and sol.x[i] >= problem_lb[i]
						for i in range(len(sol.x))
					)
					best_objective = -1 * problem.minmax[0] * sol.objectives_mean
					self.recommended_solns.append(best_solution)
					self.intermediate_budgets.append(self.budget.used)
				else :
					# Compare solutions
					is_feasible = all(
						sol.x[i] <= problem_ub[i] and sol.x[i] >= problem_lb[i]
						for i in range(len(sol.x))
					)
					should_update = False

					if is_feasible and not best_is_feasible:
						# New solution is feasible and current best is not
						should_update = True
					elif is_feasible == best_is_feasible:
						# Both feasible or both infeasible - compare objectives
						objective_diff = -1 * problem.minmax[0] * sol.objectives_mean - best_objective

						# Check if new solution is better (accounting for min/max)
						if all(objective_diff < 0):
							should_update = True

					if should_update:
						best_solution = sol
						best_is_feasible = is_feasible
						best_objective = -1 * problem.minmax[0] * sol.objectives_mean
						self.recommended_solns.append(best_solution)
						self.intermediate_budgets.append(self.budget.used)
						print(
							f"    *** New best solution found! "
							f"Objective: {best_objective[0]:.4f}"
						)
		
		except Exception as e :
			print(f"An error occurred during Complete Enumeration: {str(e.__class__.__name__)}: {str(e)}")
			print(format_exc())

