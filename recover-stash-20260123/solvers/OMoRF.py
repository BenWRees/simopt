"""OMoRF Solver.

The OMoRF (Optimisation by Moving Ridge Functions) solver by Gross and Parks
progressively builds local models using interpolation on a reduced subspace
constructed through Active Subspace dimensionality reduction.
"""

from __future__ import annotations

import importlib
import traceback
import warnings
from copy import deepcopy
from math import comb
from typing import Annotated, ClassVar, Self

import numpy as np
from numpy.linalg import norm, pinv
from pydantic import Field, model_validator
from scipy.optimize import NonlinearConstraint, minimize
from scipy.stats import linregress

from simopt.base import (
	ConstraintType,
	ObjectiveType,
	Problem,
	Solution,
	Solver,
	SolverConfig,
	VariableType,
)
from simopt.linear_algebra_base import finite_difference_gradient
from simopt.solvers.active_subspaces.index_set import IndexSet, CARD_LIMIT_HARD
from simopt.solver import BudgetExhaustedError

warnings.filterwarnings("ignore")


class OMoRFConfig(SolverConfig):
	"""Configuration for OMoRF solver."""

	crn_across_solns: Annotated[
		bool,
		Field(default=True, description="use CRN across solutions?"),
	]
	interpolation_update_tol: Annotated[
		tuple[float, float],
		Field(
			default=(0.01, 0.1),
			description="tolerance values to check for updating the interpolation model",
			alias="interpolation update tol",
		),
	]
	delta: Annotated[
		float,
		Field(default=0.5, gt=0, description="initial trust-region radius"),
	]
	gamma_1: Annotated[
		float,
		Field(
			default=0.5,
			lt=1,
			gt=0,
			description="trust-region radius increase rate after unsuccessful iteration",
		),
	]
	gamma_2: Annotated[
		float,
		Field(
			default=1.2,
			gt=1,
			description="trust-region radius decrease rate after successful iteration",
		),
	]
	gamma_3: Annotated[
		float,
		Field(
			default=2.5,
			gt=0,
			description="trust-region radius increase rate after very successful iteration",
		),
	]
	gamma_shrinking: Annotated[
		float,
		Field(
			default=0.5,
			gt=0,
			description="constant to make upper bound for delta in contraction loop",
		),
	]
	omega_shrinking: Annotated[
		float,
		Field(
			default=0.5,
			gt=0,
			description="factor to shrink the trust-region radius",
		),
	]
	eta_1: Annotated[
		float,
		Field(default=0.1, gt=0, description="threshold for a successful iteration"),
	]
	eta_2: Annotated[
		float,
		Field(
			default=0.8,
			description="threshold for a very successful iteration",
		),
	]
	initial_subspace_dimension: Annotated[
		int,
		Field(
			default=1,
			ge=1,
			description="dimension size of the active subspace",
			alias="initial subspace dimension",
		),
	]
	random_directions: Annotated[
		bool,
		Field(
			default=False,
			description="determine to take random directions in set construction",
			alias="random directions",
		),
	]
	alpha_1: Annotated[
		float,
		Field(
			default=0.1,
			gt=0,
			description="scale factor to shrink the stepsize check",
		),
	]
	alpha_2: Annotated[
		float,
		Field(
			default=0.5,
			gt=0,
			description="scale factor to shrink the trust-region radius",
		),
	]
	rho_min: Annotated[
		float,
		Field(
			default=1.0e-8,
			gt=0,
			description="initial rho when shrinking",
		),
	]
	easy_solve: Annotated[
		bool,
		Field(
			default=False,
			description="solve the subproblem approximately with Cauchy point",
		),
	]
	polynomial_basis: Annotated[
		str,
		Field(
			default="AstroDFBasis",
			description="polynomial basis to use in model construction",
			alias="polynomial basis",
		),
	]
	polynomial_degree: Annotated[
		int,
		Field(
			default=2,
			ge=1,
			description="degree of the polynomial",
			alias="polynomial degree",
		),
	]
	model_type: Annotated[
		str,
		Field(
			default="RandomModel",
			description="the type of random model used",
			alias="model type",
		),
	]

	@model_validator(mode="after")
	def _validate_eta_2_greater_than_eta_1(self) -> Self:
		if self.eta_2 <= self.eta_1:
			raise ValueError("Eta 2 must be greater than Eta 1.")
		return self

	@model_validator(mode="after")
	def _validate_gamma_2_greater_than_gamma_1(self) -> Self:
		if self.gamma_2 < self.gamma_1:
			raise ValueError("Gamma 1 must be greater than or equal to Gamma 2.")
		return self


class OMoRF(Solver):
	"""The OMoRF solver (Optimisation by Moving Ridge Functions)."""

	name: str = "OMoRF"
	config_class: ClassVar[type[SolverConfig]] = OMoRFConfig
	class_name_abbr: ClassVar[str] = "OMoRF"
	class_name: ClassVar[str] = "OMoRF"
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
	def delta_k(self) -> float:
		"""Get the current trust-region radius."""
		return self._delta_k

	@delta_k.setter
	def delta_k(self, value: float) -> None:
		"""Set the current trust-region radius."""
		self._delta_k = value

	@property
	def rho_k(self) -> float:
		"""Get the current rho value."""
		return self._rho_k

	@rho_k.setter
	def rho_k(self, value: float) -> None:
		"""Set the current rho value."""
		self._rho_k = value

	@property
	def ratio(self) -> float:
		"""Get the current ratio."""
		return self._ratio

	@ratio.setter
	def ratio(self, value: float) -> None:
		"""Set the current ratio."""
		if isinstance(value, (list, np.ndarray)):
			value = value[0]
		self._ratio = value

	@property
	def unsuccessful_iteration_counter(self) -> int:
		"""Get the unsuccessful iteration counter."""
		return self._unsuccessful_iteration_counter

	@unsuccessful_iteration_counter.setter
	def unsuccessful_iteration_counter(self, value: int) -> None:
		"""Set the unsuccessful iteration counter."""
		self._unsuccessful_iteration_counter = value

	@property
	def s_old(self) -> np.ndarray:
		"""Get the previous solution."""
		return self._s_old

	@s_old.setter
	def s_old(self, value: np.ndarray) -> None:
		"""Set the previous solution."""
		self._s_old = value

	@property
	def f_old(self) -> float:
		"""Get the previous function value."""
		return self._f_old

	@f_old.setter
	def f_old(self, value: float) -> None:
		"""Set the previous function value."""
		self._f_old = value

	@property
	def U(self) -> np.ndarray:
		"""Get the active subspace matrix."""
		return self._U

	@U.setter
	def U(self, value: np.ndarray) -> None:
		"""Set the active subspace matrix."""
		self._U = value

	def polynomial_basis_instantiation(self) -> type:
		"""Get the polynomial basis class.

		Returns:
				The polynomial basis class from active_subspaces.basis module.
		"""
		class_name = self.factors["polynomial basis"].strip()
		module = importlib.import_module("simopt.solvers.active_subspaces.basis")
		return getattr(module, class_name)

	def geometry_type_instantiation(self) -> type:
		"""Get the geometry type class.

		Returns:
				The geometry class from active_subspaces.geometry module.
		"""
		module = importlib.import_module("simopt.solvers.active_subspaces.geometry")
		return module.Geometry

	def _set_delta(self, val: float) -> None:
		"""Set the trust region radius."""
		self._delta_k = val

	def _set_counter(self, count: int) -> None:
		"""Set the unsuccessful iteration counter."""
		self._unsuccessful_iteration_counter = count

	def _set_ratio(self, ratio: float | list | np.ndarray) -> None:
		"""Set the ratio value."""
		if isinstance(ratio, (list, np.ndarray)):
			ratio = ratio[0]
		self._ratio = ratio

	def _set_rho_k(self, value: float) -> None:
		"""Set the rho_k value."""
		self._rho_k = value

	def blackbox_evaluation(self, s: np.ndarray, problem: Problem) -> float:
		"""Evaluate the point s for the problem.

		Args:
				s: The point to evaluate.
				problem: The problem to evaluate on.

		Returns:
				The function value at point s.

		Note:
				self.S is array-like of all the x values.
				self.f is a 1-d array of function values.
		"""
		# Ensure s is a row-vector
		s = np.asarray(s).reshape(1, -1)

		# If S has 1 or more points and s matches an existing point, return stored value
		if getattr(self, "S", np.array([])).size > 0 and np.unique(
			np.vstack((self.S, s)), axis=0
		).shape[0] == self.S.shape[0]:
			ind_repeat = int(np.argmin(norm(self.S - s, ord=np.inf, axis=1)))
			# extract scalar and account for storage shape
			f_val = float(np.atleast_2d(self.f)[ind_repeat].flatten()[0])
			self.budget.request(1)
			return f_val

		# Otherwise evaluate the black-box and store as a 2D column
		new_solution = self.create_new_solution(tuple(s.flatten()), problem)
		problem.simulate(new_solution, 1)
		f_scalar = -1 * problem.minmax[0] * new_solution.objectives_mean.item()
		self.budget.request(1)

		f_arr = np.array([[f_scalar]])

		if getattr(self, "f", np.array([])).size == 0:
			self.S = s
			self.f = f_arr
		else:
			# ensure self.f is a column 2D array
			if self.f.ndim == 1:
				self.f = self.f.reshape(-1, 1)
			self.S = np.vstack((self.S, s))
			self.f = np.vstack((self.f, f_arr))

		return f_scalar

	def calculate_subspace(
		self, S: np.ndarray, f: np.ndarray, delta_k: float
	) -> np.ndarray:
		"""Calculate the Active Subspace

		Args:
				S (np.ndarray): A matrix of shape (M,n) of sample points
				f (np.ndarray): A column vector of shape (M,1)
				problem (simopt.Problem): The optimisation problem to solve on

		Returns:
				np.ndarray: The active subspace matrix of shape (n,d)
		"""
		# construct covariance matrix
		M, n = S.shape
		num_grad_lb = 2.0 * self.d * np.log(n)

		if num_grad_lb > M:
			warnings.warn(
				"Number of gradient evaluation points is likely to be insufficient. Consider resampling!",
				UserWarning,
			)

		covar = self._get_grads(S, f, delta_k)
		weights = np.ones((M, 1)) / M
		R = covar * weights
		C = np.dot(covar.T, R)

		# Compute eigendecomposition!
		e, W = np.linalg.eigh(C)
		idx = e.argsort()[::-1]
		eigs = e[idx]
		eigVecs = W[:, idx]
		if hasattr(self, "data_scaler"):
			Xmax, Xmin = np.max(S, axis=0), np.min(S, axis=0)
			scale_factors = 2.0 / (Xmax - Xmin)
			eigVecs = scale_factors[:, np.newaxis] * eigVecs
			eigVecs = np.linalg.qr(eigVecs)[0]

		subspace = eigVecs
		eigenvalues = eigs

		U0 = subspace[:, 0].reshape(-1, 1)  # this is a column vector
		U1 = subspace[:, 1:]

		# Add the other d-1 columns to U0 by selecting the columns of U1 with the largest coefficients of determination
		for i in range(self.d - 1):
			R = []
			# loop through each column
			for j in range(U1.shape[1]):
				# stack U with the AS and the jth column of the orthogonal complement
				U = np.hstack((U0, U1[:, j].reshape(-1, 1)))
				Y = np.dot(S, U)  # map the sample points to the reduced subspace

				# print(f'shape of Y: {Y.shape}')

				coeff = self.construct_model(Y, f, self.poly_basis_subspace, U=U)
				sample_pts = []
				for pt in Y:
					pt = np.array(pt).reshape(-1, 1)
					sample_pts.append(
						self.eval_model(pt, self.poly_basis_subspace, coeff=coeff)
					)

				# r = linregress(sample_pts, f).rvalue
				sample_pts = np.array(sample_pts).reshape(
					-1,
				)
				_, _, r, _, _ = linregress(sample_pts, f.flatten())

				R.append(r**2)  # coefficient of determination
			index = np.argmax(R)
			U0 = np.hstack(
				(U0, U1[:, index].reshape(-1, 1))
			)  # add the column corresponding to the largest coefficient of determination in the orthogonal complement to AS
			U1 = np.delete(U1, index, 1)  # remove feature that was added

		# store U as shape (n, d) (columns = reduced basis vectors)
		self.U = U0
		return self.U

	def construct_model(self, S, f, poly_basis, U=None) -> list[float]:
		"""Construct the interpolation model by solving the linear solution V(Y)coeff = fvals

		Args:
				S (np.ndarray): A (M,d) array of sample points
				f (np.ndarray): A (M,1) column vector of function evaluations
				poly_basis (Basis): The poly_basis being used
				U (np.ndarray): The active subspace matrix

		Returns:
				list[float]: The coefficients of the model
		"""
		# S = S.T
		if U is None:
			U = self.U
			
		n,d = U.shape

		# Map full-space samples S (M, n) to reduced-space Y (M, d)
		if S.shape[1] == n :
			Y = S @ U  # (M, d)
		else : 
			Y = S  # (M, d)

		if not isinstance(Y, np.ndarray):
			Y = np.vstack(Y)  # reshape Y to be a matrix of (M,d)
		M = poly_basis.V(Y,d)  # now constructs M based on the polynomial basis being used
		# print(f'shape of M: {M.shape}')
		q = np.matmul(pinv(M), f)

		return q

	def eval_model(self, x_k, poly_basis, coeff=None):
		# x_k can be full-space (n,) or reduced (d,) or a 2D row (1,dim)
		if coeff is None:
			coeff = self.coefficients

		interpolation_set = np.atleast_2d(x_k).reshape(1, -1)
		input_dim = interpolation_set.shape[1]

		poly_dim = getattr(poly_basis, "dim", None)

		# If input is full-space but the polynomial basis expects reduced dimension,
		# project using the active subspace `U` so the input given to the basis
		# matches the reduced-space samples used to construct `coeff`.
		if poly_dim is not None and input_dim != poly_dim:
			if hasattr(self, "U") and input_dim == getattr(self, "n", input_dim) and poly_dim < input_dim:
				interpolation_set = (self.U.T @ interpolation_set.T).T
				input_dim = interpolation_set.shape[1]
			# If dimensions still differ (e.g., basis was constructed for a different reduced
			# dimension), prefer using the actual input dimension when building V so the
			# Vandermonde matches the shape used to compute `coeff`.

		dim_used = interpolation_set.shape[1]
		X = poly_basis.V(interpolation_set, dim_used)
		evaluation = X @ coeff
		return float(np.atleast_1d(evaluation).item())

	def finite_difference_gradient(
		self, new_solution: Solution, problem: Problem
	) -> np.ndarray:
		"""Calculate the finite difference gradient of the problem at new_solution.

		Args:
				new_solution (Solution): The solution at which to calculate the gradient.
				problem (Problem): The problem`that contains the function to differentiate.

		Returns:
				np.ndarray: The solution value of the gradient

				int: The expended budget
		"""
		lower_bound = problem.lower_bounds
		upper_bound = problem.upper_bounds
		# problem.simulate(new_solution,1)
		# fn = -1 * problem.minmax[0] * new_solution.objectives_mean
		# self.expended_budget += 1
		# new_solution = self.create_new_solution(tuple(x), problem)

		new_x = new_solution.x
		forward = np.isclose(new_x, lower_bound, atol=10 ** (-7)).astype(int)
		backward = np.isclose(new_x, upper_bound, atol=10 ** (-7)).astype(int)
		# BdsCheck: 1 stands for forward, -1 stands for backward, 0 means central diff.
		BdsCheck = np.subtract(forward, backward)

		self.budget.request((2 * problem.dim) + 1)
		return finite_difference_gradient(new_solution, problem, BdsCheck=BdsCheck)

	def finite_differencing(
		self, x_val: np.ndarray, model_coeff: list[float], delta_k: float
	):
		lower_bound = x_val - delta_k
		upper_bound = x_val + delta_k

		fn = self.eval_model(x_val, self.poly_basis_subspace, coeff=model_coeff)

		BdsCheck = np.zeros(self.n)

		FnPlusMinus = np.zeros((self.n, 3))
		grad = np.zeros(self.n)
		for i in range(self.n):
			# Initialization.
			x1 = deepcopy(x_val.tolist())
			x2 = deepcopy(x_val.tolist())
			# Forward stepsize.
			steph1 = 1.0e-8
			# Backward stepsize.
			steph2 = 1.0e-8

			# Check variable bounds.
			if x1[i] + steph1 > upper_bound[i]:
				steph1 = np.abs(upper_bound[i] - x1[i])
			if x2[i] - steph2 < lower_bound[i]:
				steph2 = np.abs(x2[i] - lower_bound[i])

			# Decide stepsize.
			# Central diff.
			if BdsCheck[i] == 0:
				FnPlusMinus[i, 2] = min(steph1, steph2)
				x1[i] = x1[i] + FnPlusMinus[i, 2]
				x2[i] = x2[i] - FnPlusMinus[i, 2]
			# Forward diff.
			elif BdsCheck[i] == 1:
				FnPlusMinus[i, 2] = steph1
				x1[i] = x1[i] + FnPlusMinus[i, 2]
			# Backward diff.
			else:
				FnPlusMinus[i, 2] = steph2
				x2[i] = x2[i] - FnPlusMinus[i, 2]

			fn1, fn2 = 0, 0
			x1 = np.array(x1)
			if BdsCheck[i] != -1:
				fn1 = self.eval_model(x1, self.poly_basis_subspace, coeff=model_coeff)
				# First column is f(x+h,y).
				FnPlusMinus[i, 0] = fn1
			x2 = np.array(x2)
			if BdsCheck[i] != 1:
				fn2 = self.eval_model(x2, self.poly_basis_subspace, coeff=model_coeff)
				# Second column is f(x-h,y).
				FnPlusMinus[i, 1] = fn2

			# Calculate gradient.
			if BdsCheck[i] == 0:
				grad[i] = (fn1 - fn2) / (2 * FnPlusMinus[i, 2])
			elif BdsCheck[i] == 1:
				grad[i] = (fn1 - fn) / FnPlusMinus[i, 2]
			elif BdsCheck[i] == -1:
				grad[i] = (fn - fn2) / FnPlusMinus[i, 2]

		return grad

	def _get_grads(self, X: np.ndarray, f: np.ndarray, delta_k: float) -> np.ndarray:
		"""Calculate gradients

		Args:
				X (np.ndarray): (N,m) matrix of N x-vals to be evaluated
				problem (Problem):

		Returns:
				np.ndarray: (N,m) matrix of N gradients evaluated at each row of X
		"""
		grads = np.zeros(X.shape)

		# Construct a local model over the space of subspace interpolation points
		coeff = self.construct_model(X, f, self.poly_basis_subspace)

		for idx, x_val in enumerate(X):
			grads[idx, :] = self.finite_differencing(x_val, coeff, delta_k)

		return grads

	def solve(self, problem):
		# initialise factors:
		# self.recommended_solns = []
		# self.intermediate_budgets = []

		self.S = np.array([])
		self.f = np.array([])
		self.g = np.array([])
		self.d = self.factors["initial subspace dimension"]

		# self.delta_k = self.factors['delta']
		self._set_delta(self.factors["delta"])
		# self.rho_k = self.delta_k
		self._set_rho_k(self.delta_k)
		self._set_counter(0)
		self.rhos = []
		self.n = problem.dim
		self.deg = self.factors["polynomial degree"]
		self.q = (
			comb(self.d + self.deg, self.deg) + self.n * self.d
		)  # int(0.5*(self.d+1)*(self.d+2))
		self.p = self.n + 1
		self.epsilon_1, self.epsilon_2 = self.factors[
			"interpolation update tol"
		]  # epsilon is the tolerance in the interpolation set update
		self.random_initial = self.factors["random directions"]
		self.alpha_1 = self.factors[
			"alpha_1"
		]  # shrink the trust region radius in set improvement
		self.alpha_2 = self.factors["alpha_2"]  # shrink the stepsize reduction
		self.rho_min = self.factors["rho_min"]

		# set up initial Solution
		current_x = problem.factors["initial_solution"]
		# print(f'initial solution: {current_x}')
		self.current_solution = self.create_new_solution(current_x, problem)
		self.recommended_solns.append(self.current_solution)
		self.intermediate_budgets.append(self.budget.used)

		""" 
		self.s_old = self._apply_scaling(s_old) #shifts the old solution into the unit ball
		"""

		self.iteration_count = 1
		self._set_counter(0)

		geo_factors = {
			"random_directions": self.random_initial,
			"epsilon_1": self.epsilon_1,
			"epsilon_2": self.epsilon_2,
			"rho_min": self.rho_min,
			"alpha_1": self.alpha_1,
			"alpha_2": self.alpha_2,
			"n": self.n,
			"d": self.d,
			"q": self.q,
			"p": self.p,
		}

		# self.fn_estimates = []

		# basis construction
		# Clamp the index-set dimension used to build a total-order basis so the
		# cardinality does not exceed CARD_LIMIT_HARD (prevents huge arrays).
		q_requested = int(self.q)
		highest_order = 2
		# find largest q such that comb(highest_order+q, highest_order) < CARD_LIMIT_HARD
		q_test = 0
		q_max = 0
		while True:
			L = comb(highest_order + q_test, highest_order)
			if L >= CARD_LIMIT_HARD:
				q_max = max(0, q_test - 1)
				break
			q_test += 1
			if q_test > q_requested:
				q_max = q_requested
				break
		if q_max < 1:
			q_max = 1
		if q_requested > q_max:
			warnings.warn(
				f"Requested index-set dimension q={q_requested} would produce cardinality >= {CARD_LIMIT_HARD}. Truncating to q={q_max}."
			)
		q_use = min(q_requested, q_max)
		index_set = IndexSet("total-order", orders=np.tile([2], q_use))
		self.index_set = index_set.get_basis()[:, range(self.d - 1, -1, -1)]

		self.poly_basis_model = self.polynomial_basis_instantiation()(
			self.factors["polynomial degree"],
			problem,
			dim=self.factors["initial subspace dimension"],
		)
		self.poly_basis_subspace = self.polynomial_basis_instantiation()(
			self.factors["polynomial degree"], problem, dim=self.n
		)
		self.geometry_instance = self.geometry_type_instantiation()(
			problem, self, self.index_set, **geo_factors
		)

		self.s_old = np.array(current_x)
		self.f_old = self.blackbox_evaluation(self.s_old, problem)

		# store data on initial solution			
		self.fn_estimates.append(self.f_old)
		self.iterations.append(self.iteration_count)
		self.budget_history.append(self.budget.used)

		# Construct the sample set for the subspace
		S_full = self.geometry_instance.generate_set(
			self.n + 1, self.s_old, self.delta_k
		)
		f_full = np.zeros((self.n + 1, 1))
		f_full[0, :] = self.f_old  # first row gets the old function values

		# get the rest of the function evaluations - write as a function
		try :
			for i in range(1, self.n + 1):
				# simulate the problem at each component of f_
				new_solution = self.create_new_solution(S_full[i, :], problem)
				problem.simulate(new_solution, 1)
				self.budget.request(1)
				f_full[i, :] = -1 * problem.minmax[0] * new_solution.objectives_mean
				# self.expended_budget = reset_budget
				# case where we use up our whole budget getting the function values
		except BudgetExhaustedError :
			print("Budget exhausted during initial sample set evaluations.")
			self.recommended_solns.append(self.current_solution)
			self.intermediate_budgets.append(self.budget.used)
			return

		# This is needed to ensure that model construction in the subspace works
		self.U = np.eye(self.n, self.n)

		# initial subspace calculation - requires gradients of f_full
		self.calculate_subspace(S_full, f_full, self.delta_k)

		# This constructs the sample set for the model construction
		S_red, f_red = self.geometry_instance.sample_set(
			"new",
			self.s_old,
			self.delta_k,
			self.rho_k,
			self.f_old,
			self.U,
			full_space=False,
		)
		try :
			while self.budget.remaining > 0:

				print(
					f"\niteration: {self.iteration_count} \t expended budget {self.budget.used} \t current objective function value: {self.f_old}"
				)
				# if rho has been decreased too much we end the algorithm
				if self.rho_k <= self.rho_min:
					break

				# BUILD MODEL
				try:
					self.coefficients = self.construct_model(
						S_red, f_red, self.poly_basis_model
					)  # this should be the model instance construct model
				except:  # thrown if the sample set is not defined properly
					print(traceback.format_exc())
					S_red, f_red = self.geometry_instance.sample_set(
						"improve",
						self.s_old,
						self.delta_k,
						self.rho_k,
						self.f_old,
						self.U,
						S_red,
						f_red,
						full_space=False,
					)
					self.intermediate_budgets.append(self.budget.used)
					continue

				# SOLVE THE SUBPROBLEM
				candidate_solution, S_full, S_red, f_full, f_red, reset_flag = (
					self.solve_subproblem(problem, S_full, S_red, f_full, f_red)
				)
				candidate_fval = self.blackbox_evaluation(
					np.array(candidate_solution.x), problem
				)

				if reset_flag:
					self.recommended_solns.append(self.current_solution)
					self.intermediate_budgets.append(self.budget.used)
					self.iteration_count += 1
					break

				# EVALUATE THE CANDIDATE SOLUTION
				S_red, S_full, f_red, f_full = self.evaluate_candidate_solution(
					problem,
					candidate_fval,
					candidate_solution,
					S_red,
					S_full,
					f_red,
					f_full,
				)

				# print(f'EXPENDED BUDGET: {self.expended_budget}')

				self.iteration_count += 1

				self.iterations.append(self.iteration_count)
				self.budget_history.append(self.budget.used)
				self.fn_estimates.append(self.f_old)
		except BudgetExhaustedError :
			print("Budget exhausted during OMoRF iterations.")
			self.recommended_solns.append(self.current_solution)
			self.intermediate_budgets.append(self.budget.used)
			
		finally:
			print("finished OMoRF solver")

	def solve_subproblem(
		self,
		problem: Problem,
		S_full: np.ndarray,
		S_red: np.ndarray,
		f_full: float,
		f_red: float,
	):
		"""Solves the trust-region subproblem for ``trust-region`` or ``omorf`` methods"""
		omega_s = self.factors["omega_shrinking"]
		reset_flag = False

		cons = NonlinearConstraint(lambda x: norm(x), 0, self.delta_k)

		def obj(x) :
			x_arr = np.array(x).reshape(-1,1)
			proj_x = self.U.T @ x_arr
			res = self.eval_model(
				proj_x.reshape(1,-1), self.poly_basis_model
			)
			return res

		res = minimize(
			obj,
			np.zeros(problem.dim),
			method="trust-constr",
			constraints=cons,
			options={"disp": False},
		)
		# Ensure optimizer succeeded before trusting result
		if not getattr(res, "success", True):
			print("Warning: subproblem solver did not converge; using zero step.")
			step_dist = np.zeros(problem.dim)
		else:
			step_dist = res.x
		print(f"stepsize: {step_dist}")
		s_new = self.s_old + step_dist
		# m_new = res.fun

		print(f"CANDIDATE SOLUTION: {s_new}")

		for i in range(problem.dim):
			if s_new[i] <= problem.lower_bounds[i]:
				s_new[i] = problem.lower_bounds[i] + 0.01
			elif s_new[i] >= problem.upper_bounds[i]:
				s_new[i] = problem.upper_bounds[i] - 0.01

		candidate_solution = self.create_new_solution(tuple(s_new), problem)

		step_dist = norm(s_new - self.s_old, ord=np.inf)

		# Safety step implemented in BOBYQA
		if step_dist < omega_s * self.rho_k:
			self.ratio = -0.1
			self._set_counter(3)
			# self.delta_k = max(0.5*self.delta_k, self.rho_k)
			self._set_delta(max(0.5 * self.delta_k, self.rho_k))
			# self.d += 1 #increase the dimension
			S_full, f_full, S_red, f_red, delta_k, rho_k, U = (
				self.geometry_instance.update_geometry_omorf(
					self.s_old,
					self.f_old,
					self.delta_k,
					self.rho_k,
					self.U,
					S_full,
					f_full,
					S_red,
					f_red,
					self.unsuccessful_iteration_counter,
					self.ratio,
				)
			)
			self.U = U
			self._set_delta(delta_k)
			self._set_rho_k(rho_k)

		# #this is a breaking condition
		if self.rho_k <= self.rho_min:
			reset_flag = True

		return candidate_solution, S_full, S_red, f_full, f_red, reset_flag

	def evaluate_candidate_solution(
		self,
		problem: Problem,
		fval_tilde: float,
		candidate_solution: Solution,
		S_red: np.ndarray,
		S_full: np.ndarray,
		f_red: np.ndarray,
		f_full: np.ndarray,
	):
		gamma_1 = self.factors["gamma_1"]
		gamma_2 = self.factors["gamma_2"]
		gamma_3 = self.factors["gamma_3"]
		eta_1 = self.factors["eta_1"]
		eta_2 = self.factors["eta_2"]
		s_new = np.array(candidate_solution.x)

		s_old_proj = self.U.T @ self.s_old.reshape(-1, 1)
		s_new_proj = self.U.T @ s_new.reshape(-1, 1)

		model_eval_old = self.eval_model(
			 s_old_proj.reshape(1,-1), self.poly_basis_model
		)
		model_eval_new = self.eval_model(
			s_new_proj.reshape(1,-1), self.poly_basis_model
		)

		print(
			f"DIFFERENCE IN CANDIDATE EVALUATION AT MODEL AND FUNCTION: {abs(fval_tilde - model_eval_new)}"
		)

		#! ONLY ISSUE IS THAT IT ACCEPTS GROWING VALUES - DUE TO BIG DIFFERENCE BETWEEN model_eval_new AND fval_tilde
		del_f = self.f_old - fval_tilde  # self.f_old - f_new
		# del_m = np.asscalar(my_poly.get_polyfit(np.dot(self.s_old,self.U))) - m_new
		del_m = model_eval_old - model_eval_new

		print(
			f"The model evaluation for the old value is {model_eval_old} and for the candidate value it is {model_eval_new}"
		)
		print(
			f"The old function value is {self.f_old} and the new function value is {fval_tilde}"
		)

		print(f"numerator of ratio is {del_f} and the denominator is {del_m}")

		step_dist = norm(np.array(candidate_solution.x) - self.s_old, ord=np.inf)

		# in the case that the denominator is very small
		# if abs(del_m) < 100*np.finfo(float).eps :
		# # if del_m <= 0:
		# 	self._set_ratio(1.0)

		#! Need to handle the case where the model evaluation is increasing when it should be decreasing - should reject this!
		# elif norm(model_eval_new - fval_tilde) > abs(self.f_old - fval_tilde) :
		if del_f < 0:
			# candidate made objective worse
			self._set_ratio(0.0)
		else:
			# guard against very small denominator
			if abs(del_m) < 1e-12:
				print("Warning: model denominator near zero; setting ratio to 0.")
				self._set_ratio(0.0)
			else:
				self._set_ratio(del_f / del_m)

		self.rhos.append(self.ratio)

		# self._set_iterate(problem)

		"""ind_min = np.argmin(self.f) #get the index of the smallest function value 
		self.s_old = self.S[ind_min,:] #get the x-val that corresponds to the smallest function value 
		self.f_old = np.asscalar(self.f[ind_min]) #get the smallest function value"""

		# add candidate value and corresponding fn val to interpolation sets
		S_red, f_red = self.geometry_instance.sample_set(
			"replace",
			self.s_old,
			self.delta_k,
			self.rho_k,
			self.f_old,
			self.U,
			S_red,
			f_red,
			s_new,
			fval_tilde,
			full_space=False,
		)
		S_full, f_full = self.geometry_instance.sample_set(
			"replace",
			self.s_old,
			self.delta_k,
			self.rho_k,
			self.f_old,
			self.U,
			S_full,
			f_full,
			s_new,
			fval_tilde,
		)

		print(f"RATIO COMPARISON VALUE: {self.ratio}")

		if self.ratio >= eta_2:
			print("VERY SUCCESSFUL ITERATION")
			self._set_counter(0)
			# self.delta_k = max(gamma_2*self.delta_k, gamma_3*step_dist)
			self._set_delta(max(gamma_1 * self.delta_k, gamma_3 * step_dist))
			self.current_solution = candidate_solution
			self.recommended_solns.append(self.current_solution)
			self.intermediate_budgets.append(self.budget.used)
			self.f_old = fval_tilde
			self.s_old = s_new

		elif self.ratio >= eta_1:
			print("SUCCESSFUL ITERATION")
			self._set_counter(0)
			# self.delta_k = max(gamma_1*self.delta_k, step_dist, self.rho_k)
			self._set_delta(max(gamma_2 * self.delta_k, step_dist, self.rho_k))
			self.current_solution = candidate_solution
			self.f_old = fval_tilde
			self.recommended_solns.append(self.current_solution)
			self.intermediate_budgets.append(self.budget.used)
			self.s_old = s_new

		else:
			print("UNSUCCESSFUL ITERATION")
			self._set_counter(self.unsuccessful_iteration_counter + 1)
			# self.delta_k = max(min(gamma_1*self.delta_k, step_dist), self.rho_k)
			self._set_delta(max(min(gamma_1 * self.delta_k, step_dist), self.rho_k))
			S_full, f_full, S_red, f_red, delta_k, rho_k, U = (
				self.geometry_instance.update_geometry_omorf(
					self.s_old,
					self.f_old,
					self.delta_k,
					self.rho_k,
					self.U,
					S_full,
					f_full,
					S_red,
					f_red,
					self.unsuccessful_iteration_counter,
					self.ratio,
				)
			)
			self.U = U
			self._set_delta(delta_k)
			self._set_rho_k(rho_k)

		return S_red, S_full, f_red, f_full
