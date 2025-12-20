

from __future__ import annotations

import logging
from math import ceil, log
from copy import deepcopy
from typing import Callable, Iterable

import numpy as np
from numpy.linalg import LinAlgError, inv, norm, pinv
from scipy.optimize import NonlinearConstraint, minimize

from simopt.base import (
	ConstraintType,
	ObjectiveType,
	Problem,
	Solution,
	Solver,
	VariableType,
)
from simopt.utils import classproperty, override


class MixedIntTRSolver(Solver):
	"""Mixed-integer Trust-Region solver with integer-aware enhancements.

	Key enhancements:
	  - integer-aware minimum trust-region radius so integer variables don't freeze
	  - integer-stagnation detection + forced integer shake
	  - integer-lattice-aware interpolation generation (ensures ±1 moves for ints)
	  - lightweight MADS-style integer poll (periodic discrete local search)
	  - duplicate removal and padding to ensure 2*d+1 interpolation points
	"""

	@classproperty
	@override
	def class_name(cls) -> str:
		return "MIXINTTR"

	@classproperty
	@override
	def objective_type(cls) -> ObjectiveType:
		return ObjectiveType.SINGLE

	@classproperty
	@override
	def constraint_type(cls) -> ConstraintType:
		return ConstraintType.DETERMINISTIC

	@classproperty
	@override
	def variable_type(cls) -> VariableType:
		return VariableType.MIXED

	@classproperty
	@override
	def gradient_needed(cls) -> bool:
		return False

	@classproperty
	@override
	def specifications(cls) -> dict[str, dict]:
		# Added integer-related control knobs
		return {
			"crn_across_solns": {
				"description": "use CRN across solutions",
				"datatype": bool,
				"default": False
			},
			"eta_1": {
				"description": "threshold for a successful iteration",
				"datatype": float,
				"default": 0.1
			},
			"eta_2": {
				"description": "threshold for a very successful iteration",
				"datatype": float,
				"default": 0.8
			},
			"gamma_1": {
				"description": "trust-region radius increase after very successful iteration",
				"datatype": float,
				"default": 2.5
			},	
			"gamma_2": {
				"description": "trust-region radius decrease after unsuccessful iteration",
				"datatype": float,
				"default": 0.5
			},
			"lambda_min": {
				"description": "minimum sample size",
				"datatype": int, 
				"default": 5
			},
			"easy_solve": {
				"description": "solve subproblem approximately with Cauchy point",
				"datatype": bool,
				"default": False
			},
			"reuse_points": {
				"description": "reuse previously visited points",
				"datatype": bool,
				"default": True
			},
			"ps_sufficient_reduction": {
				"description": "pattern-search sufficient reduction factor",
				"datatype": float,
				"default": 0.1
			},
			"use_gradients": {
				"description": "use gradient observations if available",
				"datatype": bool,
				"default": False
			},
			# new knobs
			"min_int_radius": {
				"description": "minimum trust-region radius to allow integer moves (>=1)",
				"datatype": float,
				"default": 1.0
			},
			"int_poll_frequency": {
				"description": "perform MADS-like integer poll every N iterations (0 disables)",
				"datatype": int,
				"default": 5
			},
			"max_int_stall": {
				"description": "max consecutive iterations without integer change before forcing shake",
				"datatype": int,
				"default": 4
			},
		}

	@property
	@override
	def check_factor_list(self) -> dict[str, Callable]:
		return {
			"crn_across_solns": self.check_crn_across_solns,
			"eta_1": self._check_eta_1,
			"eta_2": self._check_eta_2,
			"gamma_1": self._check_gamma_1,
			"gamma_2": self._check_gamma_2,
			"lambda_min": self._check_lambda_min,
			"ps_sufficient_reduction": self._check_ps_sufficient_reduction,
			"min_int_radius": self._check_min_int_radius,
			"int_poll_frequency": self._check_int_poll_frequency,
			"max_int_stall": self._check_max_int_stall,
		}

	def __init__(self, name: str = "MIXINTTR", fixed_factors: dict | None = None) -> None:
		super().__init__(name, fixed_factors)

	@property
	def iteration_count(self) -> int:
		"""Get the current iteration count."""
		return self._iteration_count

	@iteration_count.setter
	def iteration_count(self, value: int) -> None:
		"""Set the current iteration count."""
		self._iteration_count = value

	@property
	def delta_k(self) -> float:
		"""Get the current delta_k value."""
		return self._delta_k

	@delta_k.setter
	def delta_k(self, value: float) -> None:
		"""Set the current delta_k value."""
		self._delta_k = value

	@property
	def delta_max(self) -> float:
		"""Get the current delta_max value."""
		return self._delta_max

	@delta_max.setter
	def delta_max(self, value: float) -> None:
		"""Set the current delta_max value."""
		self._delta_max = value

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

	@property
	def h_k(self) -> np.ndarray:
		"""Get the Hessian approximation."""
		return self._h_k

	@h_k.setter
	def h_k(self, value: np.ndarray) -> None:
		"""Set the Hessian approximation."""
		self._h_k = value

	# ---------------------- basic helpers ----------------------
	def _check_eta_1(self) -> None:
		if self.factors["eta_1"] <= 0:
			raise ValueError("eta_1 must be > 0")

	def _check_eta_2(self) -> None:
		if self.factors["eta_2"] <= self.factors["eta_1"]:
			raise ValueError("eta_2 must be > eta_1")

	def _check_gamma_1(self) -> None:
		if self.factors["gamma_1"] <= 1:
			raise ValueError("gamma_1 must be > 1")

	def _check_gamma_2(self) -> None:
		if self.factors["gamma_2"] <= 0 or self.factors["gamma_2"] >= 1:
			raise ValueError("gamma_2 must be in (0,1)")

	def _check_lambda_min(self) -> None:
		if self.factors["lambda_min"] <= 2:
			raise ValueError("lambda_min must be > 2")

	def _check_ps_sufficient_reduction(self) -> None:
		if self.factors["ps_sufficient_reduction"] < 0:
			raise ValueError("ps_sufficient_reduction must be >= 0")
		
	def _check_min_int_radius(self) -> None:
		if self.factors["min_int_radius"] < 1.0:
			raise ValueError("min_int_radius must be >= 1.0")
	
	def _check_int_poll_frequency(self) -> None:	
		if self.factors["int_poll_frequency"] < 0:
			raise ValueError("int_poll_frequency must be >= 0")
		
	def _check_max_int_stall(self) -> None:
		if self.factors["max_int_stall"] < 0:
			raise ValueError("max_int_stall must be >= 0")
		

	def get_coordinate_vector(self, size: int, v_no: int) -> np.ndarray:
		arr = np.zeros(size)
		arr[v_no] = 1.0
		return arr

	def get_rotated_basis(self, first_basis: np.ndarray, rotate_index: np.ndarray) -> np.ndarray:
		rotate_matrix = np.array(first_basis)
		rotation = np.zeros((2, 2), dtype=int)
		rotation[0][1] = -1
		rotation[1][0] = 1
		for i in range(1, len(rotate_index)):
			v1 = np.array([[first_basis[rotate_index[0]]], [first_basis[rotate_index[i]]]])
			v2 = np.dot(rotation, v1)
			rotated_basis = np.copy(first_basis)
			rotated_basis[rotate_index[0]] = v2[0][0]
			rotated_basis[rotate_index[i]] = v2[1][0]
			rotate_matrix = np.vstack((rotate_matrix, rotated_basis))
		return rotate_matrix

	def evaluate_model(self, x_k: np.ndarray, q: np.ndarray) -> float:
		xk_arr = np.array(x_k).flatten()
		x_val = np.hstack(([1], xk_arr, xk_arr ** 2))
		return np.matmul(x_val, q).item()

	def get_stopping_time(self, pilot_run: int, sig2: float, delta: float, kappa: float) -> int:
		if kappa == 0:
			kappa = 1
		raw_sample_size = pilot_run * max(1, sig2 / (kappa ** 2 * delta ** self.delta_power))
		return ceil(float(raw_sample_size))

	# ---------------------- integer utilities ----------------------
	def _is_integer_index(self, idx: int) -> bool:
		lb = self.problem.lower_bounds[idx]
		ub = self.problem.upper_bounds[idx]
		return isinstance(lb, int) and isinstance(ub, int)

	def _int_indices(self) -> list[int]:
		return [i for i in range(self.problem.dim) if self._is_integer_index(i)]

	def force_integer_move(self) -> None:
		"""Force a small integer perturbation to escape integer plateaus."""
		if not hasattr(self, "int_indices") or len(self.int_indices) == 0:
			return
		new = list(self.incumbent_x)
		rng = np.random.default_rng()
		for i in self.int_indices:
			lb = int(self.problem.lower_bounds[i])
			ub = int(self.problem.upper_bounds[i])
			step = int(rng.choice([-1, 1]))
			cand = int(round(new[i])) + step
			cand = max(lb, min(ub, cand))
			new[i] = cand
		new = tuple(new)

		# Evaluate with a short pilot to check improvement
		new = self.ensure_feasible(new, self.problem)
		sol = self.create_new_solution(new, self.problem)
		pilot = max(1, min(self.budget.remaining, ceil(self.lambda_min / 2)))
		self.budget.request(pilot)
		self.problem.simulate(sol, pilot)

		# Replace if better
		if -1 * self.problem.minmax[0] * sol.objectives_mean < -1 * self.problem.minmax[0] * self.incumbent_solution.objectives_mean:
			self.incumbent_x = new
			self.incumbent_solution = sol
			self.recommended_solns.append(sol)
		
	def ensure_feasible(self, x: Iterable[float | int], problem: Problem) -> tuple[float | int, ...]:
		"""
			Ensure that the value is within the feasible region of the problem and satisfies all constraints.
			The function should convert the input to the appropriate types (int or float) based on the problem's variable types
			and check all constraints of the problem, clamping the values as necessary to ensure feasibility.
		Args:
			x (Iterable[float  |  int]): The solution to be checked before being simulated 
			problem (Problem): The SO problem instance

		Returns:
			tuple[float | int, ...]: tuple that is in the feasible region of the problem
		"""
		# First, convert to the problem's typed vector (handles ints/floats and basic bounds clipping)
		converted = self.convert_types(x, problem)

		# Ensure we have a mutable list
		try:
			vals = list(converted)
		except Exception:
			vals = list(x)

		# Enforce box bounds strictly (but keep integer types as ints)
		for i, (v, lb, ub) in enumerate(zip(vals, problem.lower_bounds, problem.upper_bounds)):
			# Use numeric comparison; convert to float for comparison but preserve ints
			if isinstance(lb, int) or isinstance(ub, int):
				v_i = int(round(float(v)))
				if v_i < lb:
					v_i = lb
				elif v_i > ub:
					v_i = ub
				vals[i] = v_i
			else:
				v_f = float(v)
				if v_f < lb:
					v_f = lb
				elif v_f > ub:
					v_f = ub
				vals[i] = v_f

		# If the problem has deterministic constraints, attempt a simple repair procedure.
		# We'll try greedy coordinate projection: for any infeasible point, move each variable
		# towards its nearest bound (already clamped) and check constraint satisfaction.
		# If the Problem exposes a method `satisfies_constraints` or `evaluate_constraints`, use it.
		satisfied = True
		# Prefer boolean checker if available
		if hasattr(problem, "satisfies_constraints") and callable(getattr(problem, "satisfies_constraints")):
			satisfied = problem.satisfies_constraints(tuple(vals))
		elif hasattr(problem, "evaluate_constraints") and callable(getattr(problem, "evaluate_constraints")):
			# assume evaluate_constraints returns array-like where all <=0 indicate feasibility
			try:
				cons = problem.evaluate_constraints(tuple(vals))
				# treat empty or all finite non-positive as satisfied
				satisfied = True if cons is None else all(c <= 0 for c in np.atleast_1d(cons))
			except Exception:
				satisfied = True

		if not satisfied:
			# Try simple repairs: iterate variables and nudge them towards bounds or small random nearby values
			max_attempts = 5 * problem.dim if getattr(problem, "dim", None) is not None else 50
			attempt = 0
			rng = np.random.default_rng()
			current = vals.copy()
			while attempt < max_attempts:
				# check again
				if hasattr(problem, "satisfies_constraints") and callable(getattr(problem, "satisfies_constraints")):
					if problem.satisfies_constraints(tuple(current)):
						break
					else:
						# try moving each coord to its bound then check
						changed = False
						for i in range(len(current)):
							lb = problem.lower_bounds[i]
							ub = problem.upper_bounds[i]
							# try lower bound
							trial = current.copy()
							trial[i] = lb if not (isinstance(lb, float) and np.isinf(lb)) else current[i]
							if problem.satisfies_constraints(tuple(trial)):
								current = trial
								changed = True
								break
							# try upper bound
							trial = current.copy()
							trial[i] = ub if not (isinstance(ub, float) and np.isinf(ub)) else current[i]
							if problem.satisfies_constraints(tuple(trial)):
								current = trial
								changed = True
								break
						# try small random perturbation within bounds for this coordinate
						trial = current.copy()
						if isinstance(lb, int) or isinstance(ub, int):
							trial[i] = int(round(rng.integers(max(lb, int(lb)), min(ub, int(ub)) + 1)))
						else:
							trial[i] = float(rng.uniform(lb, ub))
						if problem.satisfies_constraints(tuple(trial)):
							current = trial
							changed = True
							break
					if not changed:
						# if none of the single-coordinate repairs worked, try a small random full-vector trial
						trial = []
						for i in range(len(current)):
							lb = problem.lower_bounds[i]
							ub = problem.upper_bounds[i]
							if isinstance(lb, int) or isinstance(ub, int):
								trial.append(int(round(rng.integers(max(lb, int(lb)), min(ub, int(ub)) + 1))))
							else:
								trial.append(float(rng.uniform(lb, ub)))
						if problem.satisfies_constraints(tuple(trial)):
							current = trial
							changed = True
				attempt += 1
				if changed:
					# update satisfied flag for while check
					if hasattr(problem, "satisfies_constraints") and callable(getattr(problem, "satisfies_constraints")):
						if problem.satisfies_constraints(tuple(current)):
							break
			# adopt repaired point if feasible
			if hasattr(problem, "satisfies_constraints") and callable(getattr(problem, "satisfies_constraints")) and problem.satisfies_constraints(tuple(current)):
				vals = current
			# otherwise fall back to the box-clamped values

		# Final type normalization: ensure integers are ints and floats are floats
		for i, (lb, ub) in enumerate(zip(problem.lower_bounds, problem.upper_bounds)):
			if isinstance(lb, int) or isinstance(ub, int):
				vals[i] = int(round(float(vals[i])))
			else:
				vals[i] = float(vals[i])

		return tuple(vals)

	def mads_integer_poll(self) -> None:
		"""Lightweight MADS poll exploring ±1 on each integer index."""
		if not hasattr(self, "int_indices") or len(self.int_indices) == 0:
			return
		best_x = tuple(self.incumbent_x)
		best_val = -1 * self.problem.minmax[0] * self.incumbent_solution.objectives_mean[0]
		best_sol = None

		for i in self.int_indices:
			for step in (-1, 1):
				cand = list(self.incumbent_x)
				cand[i] = int(round(cand[i])) + step
				lb = int(self.problem.lower_bounds[i])
				ub = int(self.problem.upper_bounds[i])
				cand[i] = max(lb, min(ub, cand[i]))
				cand = self.ensure_feasible(cand, self.problem)
				cand_tuple = tuple(cand)
				sol = self.create_new_solution(cand_tuple, self.problem)
				pilot = max(1, min(self.budget.remaining, self.lambda_min))
				self.budget.request(pilot)
				self.problem.simulate(sol, pilot)
				val = -1 * self.problem.minmax[0] * sol.objectives_mean[0]
				if val < best_val:
					best_val = val
					best_x = cand_tuple
					best_sol = sol

		if best_sol is not None:
			self.incumbent_x = best_x
			self.incumbent_solution = best_sol
			self.recommended_solns.append(best_sol)
			# encourage local exploration
			self.delta_k = min(max(self.delta_k, self.factors.get("min_int_radius", 1.0)), self.delta_max)

	# ---------------------- interpolation construction (integer-aware) ----------------------
	def select_interpolation_points(self, delta_k: float, f_index: int) -> tuple[list, list]:
		if self.incumbent_x is None:
			raise ValueError("incumbent_x should be initialized before use")

		# choose coordinate or rotated basis depending on reuse logic
		if (not self.reuse_points
			or (norm(np.array(self.incumbent_x) - np.array(self.visited_pts_list[f_index].x)) == 0)
			or self.iteration_count == 1):
			var_y = self.get_coordinate_basis_interpolation_points(self.incumbent_x, delta_k, self.problem)
			var_z = self.get_coordinate_basis_interpolation_points(tuple(np.zeros(self.problem.dim)), delta_k, self.problem)
		else:
			visited_pts_array = np.array(self.visited_pts_list[f_index].x)
			diff_array = visited_pts_array - np.array(self.incumbent_x)
			if norm(diff_array) == 0:
				first_basis = diff_array
			else:
				first_basis = (diff_array) / norm(diff_array)
			rotate_list = np.nonzero(first_basis)[0]
			rotate_matrix = self.get_rotated_basis(first_basis, rotate_list)

			for i in range(self.problem.dim):
				if first_basis[i] == 0:
					coord_vector = self.get_coordinate_vector(self.problem.dim, i)
					rotate_matrix = np.vstack((rotate_matrix, coord_vector))

			var_y = self.get_rotated_basis_interpolation_points(np.array(self.incumbent_x), delta_k, self.problem, rotate_matrix, self.visited_pts_list[f_index].x)
			var_z = self.get_rotated_basis_interpolation_points(np.zeros(self.problem.dim), delta_k, self.problem, rotate_matrix, np.array(self.visited_pts_list[f_index].x) - np.array(self.incumbent_x))

		# Remove duplicates (after conversion) and ensure we have exactly 2d+1 points
		unique = []
		seen = set()
		for p in var_y:
			tup = tuple(np.round(np.array(p[0], dtype=float), 12))
			if tup not in seen:
				seen.add(tup)
				unique.append(p)
		var_y = unique

		# If duplicates removed, pad using integer-aware neighbours
		required = 2 * self.problem.dim + 1
		if len(var_y) < required:
			add_points = []
			c = list(self.incumbent_x)
			for i in range(self.problem.dim):
				plus = c.copy()
				minus = c.copy()
				if i in self.int_indices:
					plus[i] = int(round(plus[i])) + 1
					minus[i] = int(round(minus[i])) - 1
				else:
					plus[i] = plus[i] + delta_k
					minus[i] = minus[i] - delta_k
				plus = [max(lb, min(ub, v)) for v, lb, ub in zip(plus, self.problem.lower_bounds, self.problem.upper_bounds)]
				minus = [max(lb, min(ub, v)) for v, lb, ub in zip(minus, self.problem.lower_bounds, self.problem.upper_bounds)]
				add_points.append([self.convert_types(plus, self.problem)])
				add_points.append([self.convert_types(minus, self.problem)])
			idx = 0
			while len(var_y) < required and idx < len(add_points):
				t = tuple(np.round(np.array(add_points[idx][0], dtype=float), 12))
				if t not in seen:
					var_y.append(add_points[idx])
					seen.add(t)
				idx += 1

		# Safety fallback (shouldn't happen)
		while len(var_y) < required:
			var_y.append([self.incumbent_x])

		var_z = var_y.copy()
		return var_y, var_z

	def get_coordinate_basis_interpolation_points(self, x_k: tuple[int | float, ...], delta: float, problem: Problem) -> list[list[list[int | float]]]:
		y_var = [[list(x_k)]]
		is_block_constraint = sum(x_k) != 0
		num_decision_vars = problem.dim

		lower_bounds = problem.lower_bounds
		upper_bounds = problem.upper_bounds

		for var_idx in range(num_decision_vars):
			coord_vector = self.get_coordinate_vector(num_decision_vars, var_idx)
			coord_diff = delta * coord_vector

			minus = [x - d for x, d in zip(x_k, coord_diff)]
			plus = [x + d for x, d in zip(x_k, coord_diff)]

			# if integer index, ensure movement at least 1 in integer lattice
			if var_idx in self.int_indices:
				plus = list(x_k)
				minus = list(x_k)
				plus[var_idx] = int(round(x_k[var_idx])) + 1
				minus[var_idx] = int(round(x_k[var_idx])) - 1

			if is_block_constraint:
				minus = [clamp_with_epsilon(val, lower_bounds[j], upper_bounds[j]) for j, val in enumerate(minus)]
				plus = [clamp_with_epsilon(val, lower_bounds[j], upper_bounds[j]) for j, val in enumerate(plus)]

			y_var.append([self.convert_types(plus, self.problem)])
			y_var.append([self.convert_types(minus, self.problem)])

		return y_var

	def get_rotated_basis_interpolation_points(self, x_k: np.ndarray, delta: float, problem: Problem, rotate_matrix: np.ndarray, reused_x: np.ndarray) -> list[list[np.ndarray]]:
		y_var = [[x_k]]
		is_block_constraint = np.sum(x_k) != 0
		num_decision_vars = problem.dim

		lower_bounds = np.array(problem.lower_bounds)
		upper_bounds = np.array(problem.upper_bounds)

		for i in range(num_decision_vars):
			rotate_matrix_delta: np.ndarray = delta * rotate_matrix[i]

			# avoid replacing critical basis direction with reused point for i==0
			plus = x_k + rotate_matrix_delta if i != 0 else reused_x
			minus = x_k - rotate_matrix_delta

			# enforce integer lattice moves where rotation touches integer dimensions
			for idx in range(num_decision_vars):
				if idx in self.int_indices:
					if i != 0:
						if abs(rotate_matrix_delta[idx]) > 0:
							plus = np.array(plus, dtype=float)
							minus = np.array(minus, dtype=float)
							plus[idx] = int(round(x_k[idx])) + 1
							minus[idx] = int(round(x_k[idx])) - 1

			if is_block_constraint:
				minus = np.array([clamp_with_epsilon(val, lower_bounds[j], upper_bounds[j]) for j, val in enumerate(minus)])
				plus = np.array([clamp_with_epsilon(val, lower_bounds[j], upper_bounds[j]) for j, val in enumerate(plus)])

			y_var.append([self.convert_types(plus, self.problem)])
			y_var.append([self.convert_types(minus, self.problem)])

		return y_var

	# ---------------------- model fitting ----------------------
	def get_model_coefficients(self, y_var: list, fval: list, problem: Problem) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		num_design_points = 2 * problem.dim + 1
		m_var = np.array([np.hstack(([1], np.ravel(y_var[i]), np.ravel(y_var[i]) ** 2)) for i in range(num_design_points)])
		try:
			matrix_inverse = inv(m_var)
		except LinAlgError:
			matrix_inverse = pinv(m_var)
		inverse_mult = np.matmul(matrix_inverse, fval)
		decision_var_idx = problem.dim + 1
		grad = inverse_mult[1:decision_var_idx].reshape(problem.dim)
		hessian = inverse_mult[decision_var_idx:num_design_points].reshape(problem.dim)
		return inverse_mult, grad, hessian

	# ---------------------- adaptive sampling ----------------------
	def perform_adaptive_sampling(self, solution: Solution, pilot_run: int, delta_k: float, compute_kappa: bool = False) -> None:
		sample_size = solution.n_reps if solution.n_reps > 0 else pilot_run
		lambda_max = self.budget.remaining

		if solution.n_reps == 0:
			self.budget.request(pilot_run)
			self.problem.simulate(solution, pilot_run)
			sample_size = pilot_run

		while True:
			sig2 = solution.objectives_var[0]
			if self.delta_power == 0:
				sig2 = max(sig2, np.trace(solution.objectives_gradients_var))

			kappa = None
			if compute_kappa:
				if self.enable_gradient:
					rhs_for_kappa = norm(solution.objectives_gradients_mean[0])
				else:
					rhs_for_kappa = solution.objectives_mean
				kappa = (rhs_for_kappa * np.sqrt(pilot_run) / (delta_k ** (self.delta_power / 2)))

			if kappa is not None:
				k = kappa
			elif self.kappa is not None:
				k = self.kappa
			else:
				logging.warning("kappa is not set. Using default value 0.")
				k = 0

			stopping = self.get_stopping_time(pilot_run, sig2, delta_k, k)

			if sample_size >= min(stopping, lambda_max) or self.budget.remaining <= 0:
				if compute_kappa:
					self.kappa = kappa
				break

			self.budget.request(1)
			self.problem.simulate(solution, 1)
			sample_size += 1

	# ---------------------- model construction ----------------------
	def construct_model(self) -> tuple[list[float], list, np.ndarray, np.ndarray, np.ndarray, list[Solution]]:
		if self.delta_k is None:
			raise ValueError("delta_k should be initialized")
		if self.incumbent_x is None:
			raise ValueError("incumbent_x should be initialized")
		if self.incumbent_solution is None:
			raise ValueError("incumbent_solution should be initialized")

		interpolation_solns: list[Solution] = []
		lambda_max = self.budget.remaining
		pilot_run = ceil(max(self.lambda_min * log(10 + self.iteration_count, 10) ** 1.1, min(0.5 * self.problem.dim, lambda_max)) - 1)

		delta = self.delta_k
		model_iterations = 0
		w = 0.85
		mu = 1000
		beta = 10

		while True:
			delta_k = delta * w ** model_iterations
			model_iterations += 1

			# compute reuse candidate index
			distance_array = []
			for point in self.visited_pts_list:
				dist_diff = np.array(point.x) - np.array(self.incumbent_x)
				distance = norm(dist_diff) - delta_k
				dist_to_append = -delta_k * 10000 if distance > 0 else distance
				distance_array.append(dist_to_append)
			if len(distance_array) == 0:
				f_index = 0
			else:
				f_index = distance_array.index(max(distance_array))

			var_y, var_z = self.select_interpolation_points(delta_k, f_index)

			# evaluate estimates for interpolation points
			fval = []
			double_dim = 2 * self.problem.dim + 1
			for i in range(double_dim):
				if i == 0:
					adapt_soln = self.incumbent_solution
				elif (
					i == 1
					and self.reuse_points
					and len(self.visited_pts_list) > 0
					and norm(np.array(self.incumbent_x) - np.array(self.visited_pts_list[f_index].x)) != 0
				):
					adapt_soln = self.visited_pts_list[f_index]
				else:
					decision_vars = tuple(var_y[i][0])
					decision_vars = self.ensure_feasible(decision_vars, self.problem)
					new_solution = self.create_new_solution(decision_vars, self.problem)
					self.visited_pts_list.append(new_solution)
					self.budget.request(pilot_run)
					self.problem.simulate(new_solution, pilot_run)
					adapt_soln = new_solution

				# adaptive sampling for each design point (skip first at iteration 0)
				if not (i == 0 and self.iteration_count == 0):
					self.perform_adaptive_sampling(adapt_soln, pilot_run, delta_k)

				# Ensure we store a plain Python float for model fitting (handle numpy scalar/array)
				try:
					val0 = float(np.atleast_1d(adapt_soln.objectives_mean)[0])
				except Exception:
					val0 = float(adapt_soln.objectives_mean)
				fval.append(-1 * self.problem.minmax[0] * val0)
				interpolation_solns.append(adapt_soln)

			q, grad, hessian = self.get_model_coefficients(var_z, fval, self.problem)

			norm_grad = norm(grad)
			if delta_k <= mu * norm_grad or norm_grad == 0:
				break

		beta_n_grad = float(beta * norm_grad)
		self.delta_k = min(max(beta_n_grad, delta_k), delta)

		return fval, var_y, q, grad, hessian, interpolation_solns

	# ---------------------- Hessian update ----------------------
	def update_hessian(self, candidate_solution: Solution, grad: np.ndarray, s: np.ndarray) -> None:
		epsilon = 1e-15
		if not hasattr(self, "hessian_skip_count"):
			self.hessian_skip_count = 0

		def handle_hessian_skip(variable: str, value: float | np.ndarray) -> None:
			self.hessian_skip_count += 1
			logging.debug(f"{variable} near zero ({value}); skipping Hessian update. ({self.hessian_skip_count} skips)")

		candidate_grad = -1 * self.problem.minmax[0] * candidate_solution.objectives_gradients_mean[0]
		y_k = candidate_grad - grad
		y_ks = y_k @ s
		if np.isclose(y_ks, 0, atol=epsilon):
			handle_hessian_skip("y_ks", y_ks)
			return
		r_k = 1.0 / y_ks
		h_s_k = self.h_k @ s
		s_h_s_k = s @ h_s_k
		if np.all(np.isclose(s_h_s_k, 0, atol=epsilon)):
			handle_hessian_skip("s_h_s_k", s_h_s_k)
			return
		self.h_k += np.outer(y_k, y_k) * r_k - np.outer(h_s_k, h_s_k) / s_h_s_k
		self.hessian_skip_count = 0

	# ---------------------- main iterate / solve ----------------------
	def iterate(self) -> None:
		self.iteration_count += 1
		neg_minmax = -self.problem.minmax[0]

		pilot_run = ceil(max(self.lambda_min * log(10 + self.iteration_count, 10) ** 1.1, min(0.5 * self.problem.dim, self.budget.total)) - 1)

		if self.iteration_count == 1:
			self.incumbent_solution = self.create_new_solution(self.incumbent_x, self.problem)
			self.visited_pts_list.append(self.incumbent_solution)
			self.perform_adaptive_sampling(self.incumbent_solution, pilot_run, self.delta_k, compute_kappa=True)
			self.recommended_solns.append(self.incumbent_solution)
			self.intermediate_budgets.append(self.budget.used)
		elif self.factors["crn_across_solns"]:
			self.perform_adaptive_sampling(self.incumbent_solution, pilot_run, self.delta_k)

		if self.iteration_count == 1:
			self.iterations.append(self.iteration_count)
			self.budget_history.append(self.budget.used)
			self.fn_estimates.append(neg_minmax * self.incumbent_solution.objectives_mean[0])
			
		print("Constructing model...")
		if self.enable_gradient:
			fval = (np.ones(2 * self.problem.dim + 1) * neg_minmax * self.incumbent_solution.objectives_mean)
			grad = neg_minmax * self.incumbent_solution.objectives_gradients_mean[0]
			hessian = self.h_k
			q = np.array([])
			y_var = [[]]
			interpolation_solns: list[Solution] = []
		else:
			fval, y_var, q, grad, hessian, interpolation_solns = self.construct_model()

		# Solve the local model (subproblem)
		print("Solving subproblem...")
		if self.easy_solve:
			dot_a = np.dot(grad, hessian) if self.enable_gradient else grad * hessian
			check_positive_definite: float = np.dot(dot_a, grad)
			if check_positive_definite <= 0:
				tau = 1.0
			else:
				norm_ratio = norm(grad) ** 3 / (self.delta_k * check_positive_definite)
				tau = min(1.0, float(norm_ratio))
			grad: np.ndarray = np.reshape(grad, (1, self.problem.dim))[0]
			grad_norm = norm(grad)
			if grad_norm == 0:
				candidate_x = self.incumbent_x
			else:
				product = tau * self.delta_k * grad
				adjustment = product / grad_norm
				candidate_x = self.incumbent_x - adjustment
		else:
			def subproblem(s: np.ndarray) -> float:
				s_grad_dot: np.ndarray = np.dot(s, grad)
				s_hessian_dot: np.ndarray = np.dot(np.multiply(s, hessian), s)
				result = fval[0] + s_grad_dot + s_hessian_dot
				# `result` should be a scalar; avoid indexing into a 0-d array
				return float(result)

			def con_f(s: np.ndarray) -> float:
				return float(norm(s))

			nlc = NonlinearConstraint(con_f, 0, self.delta_k)
			solve_subproblem = minimize(subproblem, np.zeros(self.problem.dim), method="trust-constr", constraints=nlc)
			candidate_x = self.incumbent_x + solve_subproblem.x

		# enforce box and integer conversions
		candidate_x = tuple(clamp_with_epsilon(float(candidate_x[i]), self.problem.lower_bounds[i], self.problem.upper_bounds[i]) for i in range(self.problem.dim))
		candidate_x = self.ensure_feasible(candidate_x, self.problem)

		candidate_solution = self.create_new_solution(candidate_x, self.problem)
		self.visited_pts_list.append(candidate_solution)
		
		print("Evaluating candidate solution...")
		if self.factors["crn_across_solns"]:
			num_sims = self.incumbent_solution.n_reps
			self.budget.request(num_sims)
			self.problem.simulate(candidate_solution, num_sims)
		else:
			self.perform_adaptive_sampling(candidate_solution, pilot_run, self.delta_k)

		fval_tilde = -1 * self.problem.minmax[0] * candidate_solution.objectives_mean[0] if not isinstance(candidate_solution.objectives_mean, (list, np.ndarray)) else -1 * self.problem.minmax[0] * candidate_solution.objectives_mean

		if not self.enable_gradient:
			min_fval = min(fval)
			sufficient_reduction = (fval[0] - min_fval) >= self.factors["ps_sufficient_reduction"] * self.delta_k ** 2
			condition_met = min_fval < fval_tilde and sufficient_reduction
			high_variance = False
			if not condition_met:
				if candidate_solution.objectives_mean[0] == 0:
					logging.debug("Candidate solution objectives_mean is zero, skipping variance check.")
				else:
					high_variance = (candidate_solution.objectives_var[0] / (candidate_solution.n_reps * candidate_solution.objectives_mean[0] ** 2)) > 0.75
			if condition_met or high_variance:
				fval_tilde = min_fval
				min_idx = fval.index(min_fval)
				candidate_x = y_var[min_idx][0]
				candidate_solution = interpolation_solns[min_idx]

		candidate_x_arr = np.array(candidate_x)
		incumbent_x_arr = np.array(self.incumbent_x)
		s = np.subtract(candidate_x_arr, incumbent_x_arr)
		if self.enable_gradient:
			model_reduction = -np.dot(s, grad) - 0.5 * np.dot(np.dot(s, hessian), s)
		else:
			model_reduction = self.evaluate_model(np.zeros(self.problem.dim), q) - self.evaluate_model(s, q)
		rho = 0 if model_reduction <= 0 else (fval[0] - fval_tilde) / model_reduction

		successful = rho >= self.eta_1
		print(f"Iteration {self.iteration_count}: rho = {rho}, successful = {successful}")
		if successful:
			self.successful_iterations.append(candidate_x)
			previous_integers = tuple(int(round(v)) for v in self.incumbent_x)
			self.incumbent_x = candidate_x
			self.incumbent_solution = candidate_solution
			self.recommended_solns.append(candidate_solution)
			self.intermediate_budgets.append(self.budget.used)
			current_integers = tuple(int(round(v)) for v in self.incumbent_x)
			if hasattr(self, "int_stall_count") and self.int_indices:
				if previous_integers == current_integers:
					self.int_stall_count += 1
				else:
					self.int_stall_count = 0
			else:
				self.int_stall_count = 0

			self.delta_k = min(self.delta_k, self.delta_max)
			if rho >= self.eta_2:
				self.delta_k = min(self.gamma_1 * self.delta_k, self.delta_max)
			if self.enable_gradient:
				self.update_hessian(candidate_solution, grad, s)
		else:
			self.unsuccessful_iterations.append(candidate_x)
			min_int_radius = self.factors.get("min_int_radius", 1.0)
			new_delta = min(self.gamma_2 * self.delta_k, self.delta_max)
			if len(self.int_indices) > 0:
				self.delta_k = max(new_delta, min_int_radius)
			else:
				self.delta_k = new_delta

		# integer stall handling and MADS poll
		if not hasattr(self, "int_stall_count"):
			self.int_stall_count = 0
		if self.int_stall_count >= self.factors.get("max_int_stall", 4):
			logging.info("Integer stall detected: forcing integer move")
			self.force_integer_move()
			self.int_stall_count = 0

		int_poll_freq = int(self.factors.get("int_poll_frequency", 5))
		if int_poll_freq > 0 and (self.iteration_count % int_poll_freq == 0):
			logging.debug("Running MADS-style integer poll")
			self.mads_integer_poll()

	def _initialize_solving(self) -> None:
		self.eta_1: float = self.factors["eta_1"]
		self.eta_2: float = self.factors["eta_2"]
		self.gamma_1: float = self.factors["gamma_1"]
		self.gamma_2: float = self.factors["gamma_2"]
		self.easy_solve: bool = self.factors["easy_solve"]
		self.reuse_points: bool = self.factors["reuse_points"]
		self.lambda_min: int = self.factors["lambda_min"]

		rng = self.rng_list[1]
		dummy_solns = [self.problem.get_random_solution(rng) for _ in range(100 * max(1, self.problem.dim))]

		delta_max_candidates: list[float | int] = []
		for i in range(self.problem.dim):
			sol_values = [sol[i] for sol in dummy_solns]
			min_soln, max_soln = min(sol_values), max(sol_values)
			bound_range = self.problem.upper_bounds[i] - self.problem.lower_bounds[i]
			delta_max_candidates.append(min(max_soln - min_soln, bound_range))

		self.delta_max = max(delta_max_candidates) if len(delta_max_candidates) > 0 else 1.0
		self.delta_k = max(1.0, 10 ** (ceil(log(self.delta_max * 2, 10) - 1) / max(1, self.problem.dim)))

		if "initial_solution" in self.problem.factors:
			self.incumbent_x = tuple(self.problem.factors["initial_solution"])
			self.incumbent_x = self.ensure_feasible(self.incumbent_x, self.problem)
		else:
			self.incumbent_x = tuple(self.problem.get_random_solution(rng))
			self.incumbent_x = self.ensure_feasible(self.incumbent_x, self.problem)

		self.incumbent_solution = self.create_new_solution(self.incumbent_x, self.problem)
		self.h_k = np.identity(self.problem.dim)

		self.enable_gradient = (self.problem.gradient_available and self.factors["use_gradients"])

		self.int_indices = self._int_indices()
		self.factors.setdefault("min_int_radius", 1.0)
		self.min_int_radius = max(1.0, float(self.factors.get("min_int_radius", 1.0)))

		if self.factors["crn_across_solns"]:
			self.delta_power = 0 if self.enable_gradient else 2
		else:
			self.delta_power = 4

		self.iteration_count = 0
		self.recommended_solns = []
		self.intermediate_budgets = []
		self.visited_pts_list = []
		self.kappa = None

	@override
	def solve(self, problem: Problem) -> None:
		self.problem = problem
		self._initialize_solving()
		self.successful_iterations = []
		self.unsuccessful_iterations = []
		self.iterations = []
		self.budget_history = []
		self.fn_estimates = []

		k = 1
		while self.budget.remaining > 0:
			print(f'Starting iteration {k} with budget remaining {self.budget.remaining}...')
			self.iterate()
			self.iterations.append(self.iteration_count)
			self.budget_history.append(self.budget.used)
			self.fn_estimates.append(-1 * problem.minmax[0] * self.incumbent_solution.objectives_mean[0])
			k += 1

	def convert_types(self, value: Iterable, problem: Problem) -> list:
		"""Convert a candidate (list/tuple/array) into the problem's typed vector."""
		try:
			converted_value = []
			for val, ub, lb in zip(value, problem.upper_bounds, problem.lower_bounds):
				if isinstance(lb, int) or isinstance(ub, int):
					val_to_add = int(round(float(val)))
					if val_to_add < lb:
						val_to_add = lb
					elif val_to_add > ub:
						val_to_add = ub
					converted_value.append(val_to_add)
				else:
					val_to_add = float(val)
					if val_to_add < lb:
						val_to_add = lb
					elif val_to_add > ub:
						val_to_add = ub
					converted_value.append(val_to_add)

			# preserve container type where possible
			try:
				if isinstance(value, np.ndarray):
					return list(np.array(converted_value, dtype=float))
				return type(value)(converted_value)
			except Exception:
				return converted_value
		except Exception:
			return list(value)


def clamp_with_epsilon(val: float, lower_bound: float, upper_bound: float, epsilon: float = 0.01) -> float:
	"""Clamp a float strictly inside [lower, upper] with a small epsilon margin."""
	if val <= lower_bound:
		return lower_bound + epsilon
	if val >= upper_bound:
		return upper_bound - epsilon
	return val
