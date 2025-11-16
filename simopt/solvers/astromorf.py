"""
Summary
-------
The ASTROMoRF (Adaptive Sampling for Trust-Region Optimisation by Moving Ridge Functions) progressively builds local models using
interpolation on a reduced subspace constructed through Active Subspace dimensionality reduction.
The use of Active Subspace reduction allows for a reduced number of interpolation points to be evaluated in the model construction.
	
TODO: FIX CRITICALITY STEP IN CONSTRUCT MODEL

TODO: ADD FUNCTIONALITY FOR DIMENSION 1 MODELS 

"""

from __future__ import annotations
import warnings

warnings.filterwarnings("ignore")
from typing import Callable
import os
import time
from math import ceil, isnan, isinf, comb, log, floor
import importlib
from copy import deepcopy
import inspect
import random 
import csv
from enum import Enum

from numpy.linalg import norm, pinv, qr
from numpy.polynomial.hermite_e import hermevander, hermeder
from numpy.polynomial.polynomial import polyvander, polyder, polyroots
from numpy.polynomial.legendre import legvander, legder
from numpy.polynomial.chebyshev import chebvander, chebder
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, linprog
from scipy import optimize
from scipy.special import factorial
import scipy
import math
import sympy as sp



from simopt.base import (
	ConstraintType,
	ObjectiveType,
	Problem,
	Solution,
	Solver,
	VariableType,
)
from simopt.utils import classproperty, override
from simopt.solvers.active_subspaces.compute_optimal_dim import find_optimal_d, compute_optimal_polynomial_degree


#! === POLYNOMIAL BASIS ADAPTERS ===

class PolyBasisType(Enum):
	HERMITE = "hermite"
	LEGENDRE = "legendre"
	CHEBYSHEV = "chebyshev"
	MONOMIAL = "monomial"

class PolynomialBasisAdapter:
	def __init__(self, vander, deriv):
		self.vander = vander
		self.deriv = deriv
	def scale(self, X):
		return X
	def dscale(self, X):
		return np.ones_like(X)

class BoxScalingAdapter(PolynomialBasisAdapter):
	def __init__(self, vander, deriv, lo, hi):
		super().__init__(vander, deriv)
		self.lo = lo
		self.hi = hi

	def scale_to_box(self, X: np.ndarray, lo: float, hi: float) -> np.ndarray:
		if self.scale_factor is None:
			self.scale_factor = np.full(X.shape[1], (hi - lo) / np.maximum(np.ptp(X, axis=0), 1e-10))
		offset = np.mean(X, axis=0, keepdims=True)
		return (X - offset) * self.scale_factor + lo

	def scale(self, X):
		return self.scale_to_box(X, self.lo, self.hi)
	
	def dscale(self, X):
		return np.broadcast_to(self.scale_factor, X.shape)

POLY_BASIS_LOOKUP: dict[PolyBasisType, PolynomialBasisAdapter] = {
	PolyBasisType.HERMITE:    PolynomialBasisAdapter(hermevander, hermeder),
	PolyBasisType.LEGENDRE:   BoxScalingAdapter(legvander, legder, -1.0, 1.0),
	PolyBasisType.CHEBYSHEV:  BoxScalingAdapter(chebvander, chebder, -1.0, 1.0),
	PolyBasisType.MONOMIAL:   PolynomialBasisAdapter(polyvander, polyder),
}


class ASTROMORF(Solver):
	"""
	The ASTROMoRF Solver

	Attributes
	----------
	name : string
		name of solver
	objective_type : string
		description of objective types:
			"single" or "multi"
	constraint_type : string
		description of constraints types:
			"unconstrained", "box", "deterministic", "stochastic"
	variable_type : string
		description of variable types:
			"discrete", "continuous", "mixed"
	gradient_needed : bool
		indicates if gradient of objective function is needed
	factors : dict
		changeable factors (i.e., parameters) of the solver
	specifications : dict
		details of each factor (for GUI, data validation, and defaults)
	rng_list : list of mrg32k3a.mrg32k3a.MRG32k3a objects
		list of RNGs used for the solver's internal purposes

	Arguments
	---------
	name : str
		user-specified name for solver
	fixed_factors : dict
		fixed_factors of the solver
	See also
	--------
	base.Solver
	"""

	@classproperty
	@override 
	def class_name(cls) -> str:
		return "ASTROMORF"

	@classproperty
	@override 
	def objective_type(cls) -> ObjectiveType:
		return ObjectiveType.SINGLE

	@classproperty
	@override 
	def constraint_type(cls) -> ConstraintType:
		return ConstraintType.UNCONSTRAINED

	@classproperty
	@override 
	def variable_type(cls) -> VariableType:
		return VariableType.CONTINUOUS

	@classproperty
	@override 
	def gradient_needed(cls) -> bool:
		return False
	

	@classproperty
	@override
	def specifications(cls) -> dict[str, dict] : 
		return {
			"crn_across_solns": {
				"description": "CRN across solutions?",
				"datatype": bool,
				"default": False
			},
			"mu" : {
				"description": "dampening of the criticality step",
				"datatype": float,
				"default": 1000.0
			},
			"eta_1": {
				"description": "threshhold for a successful iteration",
				"datatype": float,
				"default": 0.05
			},
			"eta_2": {
				"description": "threshhold for a very successful iteration",
				"datatype": float,
				"default": 0.5
			},
			"gamma_1": {
				"description": "trust-region radius increase rate after a very successful iteration",
				"datatype": float,
				"default": 1.5
			},
			"gamma_2": {
				"description": "trust-region radius increase rate after a successful iteration",
				"datatype": float,
				"default": 1.2
			},
			"gamma_3": {
				"description": "trust-region radius decrease rate after an unsuccessful iteration",
				"datatype": float,
				"default": 0.5
			},
			"lambda_min": {
				"description": "minimum sample size",
				"datatype": int,
				"default": 5
			},
			"ps_sufficient_reduction": {
				"description": "use pattern search if with sufficient reduction, 0 always allows it, large value never does",
				"datatype": float,
				"default": 100.0,
			},
			'initial subspace dimension': {
				"description": "dimension size of the active subspace",
				"datatype": int, 
				"default": 5
			}, 
			"polynomial degree": {
				"description": "The degree of the local model", 
				"datatype": int, 
				"default": 2
			}, 
			"polynomial basis": {
				"description": "The polynomial basis type for the local model",
				"datatype": PolyBasisType,
				"default": PolyBasisType.HERMITE
			}
		}
	
	@property
	def check_factor_list(self) -> dict[str, Callable] : 
		return {
			"crn_across_solns": self.check_crn_across_solns,
			"mu": self._check_mu,
			"eta_1": self._check_eta_1,
			"eta_2": self._check_eta_2,
			"gamma_1": self._check_gamma_1,
			"gamma_2": self._check_gamma_2,
			"gamma_3": self._check_gamma_3,
			"lambda_min": self._check_lambda_min,
			"ps_sufficient_reduction": self._check_ps_sufficient_reduction,
			'initial subspace dimension': self._check_dimension_reduction,
			'polynomial degree': self._check_degree,
			'polynomial basis': self._check_polynomial_basis,
		}
	
	def _check_eta_1(self) -> None : 
		if self.factors["eta_1"] <= 0 :
			raise ValueError(
				"The threshold for a 'successful' iteration needs to be positive"
				)
		
	def _check_mu(self) -> None : 
		if self.factors['mu'] <= 0 : 
			raise ValueError(
				"The dampening on the criticality step must be positive"
			)

	def _check_eta_2(self) -> None : 
		if self.factors["eta_2"] <= self.factors["eta_1"] :
			raise ValueError(
				"A 'very successful' iteration threshold needs to be greater than a 'successful' iteration threshold"
				)

	def _check_gamma_1(self) -> None : 
		if self.factors["gamma_1"] < 1 or self.factors["gamma_1"] <= self.factors["gamma_2"] :
			raise ValueError(
				'The trust region radius increase after a very successful iteration ' \
				'must be greater than 1 and gamma 2'
				) 

	def _check_gamma_2(self) -> None : 
		if self.factors["gamma_2"] < 1 :
			raise ValueError(
				'The trust region radius increase after a successful iteration ' \
				'must be greater than 1'
				)
		
	def _check_gamma_3(self) -> None : 
		if self.factors["gamma_3"] >= 1 or self.factors["gamma_3"] <= 0:
			raise ValueError(
				"Gamma 3 must be between 0 and 1."
				)

	def _check_lambda_min(self) -> None : 
		if self.factors["lambda_min"] <= 2:
			raise ValueError(
				"The minimum sample size must be greater than 2."
				)

	def _check_ps_sufficient_reduction(self) -> None : 
		if self.factors["ps_sufficient_reduction"] < 0:
			raise ValueError(
				"ps_sufficient reduction must be greater than or equal to 0."
			)

	def _check_dimension_reduction(self) -> None : 
		if self.factors['initial subspace dimension'] < 1 : 
			raise ValueError(
				"The initial subspace dimension needs to be greater than 1."
			)
		
	def _check_degree(self) -> None : 
		if self.factors['polynomial degree'] < 1 :
			raise ValueError(
				'The degree of the local model should be at least 1.'
			)
		

	def _check_polynomial_basis(self) -> None :
		if not isinstance(self.factors['polynomial basis'], PolyBasisType):
			raise ValueError(
				'The polynomial basis must be an instance of PolyBasisType Enum.'
			)
		

	def __init__(self, name: str = 'ASTROMORF', fixed_factors: dict | None = None) -> None : 
		super().__init__(name, fixed_factors)


	def _set_basis(self, basis: PolyBasisType) -> None:
		"""
			Set the polynomial basis for the local model.
		Args:
			basis (PolyBasisType): The polynomial basis type to set
		"""
		adapter = POLY_BASIS_LOOKUP[basis]
		self.basis_adapter = adapter
		self.vander = adapter.vander
		self.polyder = adapter.deriv

	def set_basis(self, basis: PolyBasisType) -> None:
		"""
			Set the polynomial basis for the local model.
		Args:
			basis (PolyBasisType): The polynomial basis type to set
		"""
		if isinstance(basis, str):
			basis = PolyBasisType(basis.lower())
		self._set_basis(basis)


	def write_diagnostics_to_txt(self, content: str, mode: str = 'a', problem: 'Problem' = None):
		"""
			Write diagnostic content to a text file.
			Automatically creates file path and writes header if file is empty.
			
		Args:
			content (str): The content to write
			mode (str): File mode - 'w' for write (overwrite), 'a' for append
			problem (Problem): The problem instance (needed for file creation)
		"""
		# Create diagnostics file path if it doesn't exist
		if self.diagnostics_file_path is None:
			if problem is None:
				return
			import os
			from datetime import datetime
			from pathlib import Path
			self._timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
			
			# Create Diagnostics directory if it doesn't exist
			diagnostics_dir = Path(__file__).parent.parent.parent / "Diagnostics"
			diagnostics_dir.mkdir(parents=True, exist_ok=True)
			
			self.diagnostics_file_path = str(diagnostics_dir / f"astromorf_diagnostics_{problem.name}_{self._timestamp}.txt")
		
		# Check if file is empty and write header if needed
		import os
		file_is_empty = not os.path.exists(self.diagnostics_file_path) or os.path.getsize(self.diagnostics_file_path) == 0
		
		if file_is_empty and hasattr(self, '_timestamp') and problem is not None:
			with open(self.diagnostics_file_path, 'w', encoding='utf-8') as f:
				header = f'==== Starting ASTROMoRF Solver on problem: {problem.name} ====\n'
				header += f'Timestamp: {self._timestamp}\n'
				header += f'Initial budget: {self.budget.remaining}\n\n'
				f.write(header)
		
		with open(self.diagnostics_file_path, mode, encoding='utf-8') as f:
			f.write(content)


	def initialise_solver_factors(self, problem: Problem) -> None: 
		"""
			Initialise all the solver factors needed for the run of the algorithm
		Args:
			problem (Problem): The simulation problem to be solved
		"""		
		#For creating all the class members needed for the run of the algorithm
		self.d: int = self.factors['initial subspace dimension'] #max(1, find_optimal_d(problem))
		
		# Compute optimal polynomial degree based on subspace dimension
		# This ensures well-conditioned interpolation (points/terms ratio >= 0.60)
		optimal_degree = compute_optimal_polynomial_degree(self.d) #max(1, find_optimal_d(problem)) 
		
		user_degree = self.factors['polynomial degree']
		# Warn if user-specified degree will cause poor conditioning
		num_points = 2 * self.d + 1
		num_terms = comb(self.d + user_degree, self.d)
		ratio = num_points / num_terms
		if ratio < 0.60:
			self.degree = optimal_degree
		else :
			self.degree = user_degree
		
		self.eta_1: float = self.factors["eta_1"]
		self.eta_2: float = self.factors["eta_2"]
		self.gamma_1: float = self.factors["gamma_1"]
		self.gamma_2: float = self.factors["gamma_2"]
		self.gamma_3: float = self.factors["gamma_3"]
		self.mu : float = self.factors["mu"]
		self.lambda_min: int = self.factors["lambda_min"]

		self.set_basis(self.factors['polynomial basis'])

		self.delta_max: float = self.calculate_max_radius(problem)
		# Start with VERY conservative trust region to ensure excellent interpolation coverage
		self.delta: float = min(10 ** (ceil(log(self.delta_max * 2, 10) - 1) / problem.dim), 0.25 * self.delta_max)
		self.delta_initial: float = self.delta
		self.delta_min: float = 0.01 * self.delta_max

		self.delta_power: int = 0 if self.factors['crn_across_solns'] else 4

		rng = self.rng_list[1]

		if "initial_solution" in problem.factors:
			self.incumbent_x: tuple = tuple(problem.factors["initial_solution"])
		else:
			self.incumbent_x: tuple = tuple(problem.get_random_solution(rng))

		self.incumbent_solution: Solution = self.create_new_solution(
			self.incumbent_x, problem
		)


		# Reset iteration count and data storage
		self.iteration_count: int = 1
		self.unsuccessful_iterations: list = []
		self.successful_iterations: list = []
		self.recommended_solns: list = []
		self.intermediate_budgets: list = []
		self.visited_points: list = []
		self.kappa: float | None = None
		self.prev_r_squared: float | None = None  # Track R² across iterations
		
		# Track prediction quality for TR expansion dampening
		self.recent_prediction_errors: list = []  # Store last few relative errors
		
		# Track prediction quality across iterations for analysis
		self.prediction_error_history: list = []  # Track all prediction errors

	def write_final_diagnostics(self, problem: Problem):
		"""
			Write final prediction error analysis and solver summary to diagnostics file.
		Args:
			problem (Problem): The simulation problem being solved
		"""
		# Write prediction error trend analysis to file
		if len(self.prediction_error_history) > 0:
			output = "\n" + "="*70 + "\n"
			output += "PREDICTION ERROR ANALYSIS\n"
			output += "="*70 + "\n"
			errors = [e['relative_error'] for e in self.prediction_error_history]
			output += f"Total iterations with predictions: {len(errors)}\n"
			output += f"Mean prediction error: {np.mean(errors)*100:.1f}%\n"
			output += f"Median prediction error: {np.median(errors)*100:.1f}%\n"
			output += f"Min prediction error: {np.min(errors)*100:.1f}%\n"
			output += f"Max prediction error: {np.max(errors)*100:.1f}%\n"
			
			# Show trend over time
			if len(errors) >= 5:
				early_errors = np.mean(errors[:len(errors)//3])
				late_errors = np.mean(errors[-len(errors)//3:])
				output += f"\nTrend: Early errors: {early_errors*100:.1f}% \u2192 Late errors: {late_errors*100:.1f}%\n"
				if late_errors > 1.5 * early_errors:
					output += "  \u26a0\ufe0f  WARNING: Prediction quality degraded significantly over iterations\n"
					output += "  Possible causes:\n"
					output += "  - Trust region too large for polynomial model\n"
					output += "  - Subspace dimension too small\n"
					output += "  - Problem has complex local structure\n"
					output += "  - Polynomial degree insufficient\n"
			output += "="*70 + "\n\n"
			self.write_diagnostics_to_txt(output, problem=problem)
		
		# Write final summary to file
		output = "\n" + "="*70 + "\n"
		output += '---- ASTROMoRF Solver Finished ----\n'
		output += f"ASTROMoRF completed after {self.iteration_count} iterations and {self.budget.used} total function evaluations.\n"
		
		total_iterations = len(self.successful_iterations) + len(self.unsuccessful_iterations)
		if total_iterations > 0:
			output += f'The number of successful iterations was {len(self.successful_iterations)} and the number of unsuccessful iterations was {len(self.unsuccessful_iterations)}.\n'
			output += f'The percentage of successful iterations was {len(self.successful_iterations)/total_iterations*100:.2f}%.\n'
		else:
			output += 'No iterations were completed (solver may have failed early).\n'
		
		if hasattr(self, 'incumbent_solution') and self.incumbent_solution is not None:
			output += f'The best solution found has an objective value of {self.incumbent_solution.objectives_mean.item():.6f} with a sample size of {self.incumbent_solution.n_reps}.\n'
		else:
			output += 'No solution was found.\n'
		
		output += "="*70 + "\n\n"
		self.write_diagnostics_to_txt(output, problem=problem)
		
		# Print simple completion message to console
		print(f"\n\u2713 ASTROMoRF Solver completed. Diagnostics written to: {self.diagnostics_file_path}")

	@override
	def solve(self, problem: Problem) -> None:
		"""
			Main solver method for ASTROMoRF
		Args:
			problem (Problem): The simulation problem to be solved
		"""
		self.diagnostics_file_path = None
		self.initialise_solver_factors(problem)

		self.iterations = []
		self.budget_history = []
		self.fn_estimates = []

		try :
			while self.budget.remaining > 0: 
				self.iterations.append(self.iteration_count)
				self.budget_history.append(self.budget.used)

				if self.iteration_count == 1 :
					self.incumbent_solution: Solution = self.create_new_solution(self.incumbent_x, problem)
					# current_solution = self.create_new_solution(current_solution.x, problem)
					if len(self.visited_points) == 0:
						self.visited_points.append(self.incumbent_solution)
					
					self.calculate_kappa(problem)
				
					self.recommended_solns.append(self.incumbent_solution)
					self.intermediate_budgets.append(self.budget.used)
					self.fn_estimates.append(-1 * problem.minmax[0] * self.incumbent_solution.objectives_mean.item())

				elif self.factors['crn_across_solns'] :
					# since incument was only evaluated with the sample size of previous incumbent, here we compute its adaptive sample size
					# adaptive sampling
					lambda_max = self.budget.remaining
					sample_size = self.incumbent_solution.n_reps
					while True:
						sig2 = self.incumbent_solution.objectives_var[0]
						stopping = self.get_stopping_time(sig2, problem)
						if (sample_size >= min(stopping, lambda_max) or self.budget.remaining <= 0):
							break
						problem.simulate(self.incumbent_solution, 1)
						self.budget.request(1)
					sample_size += 1

				if self.iteration_count > 1 :
					self.fn_estimates.append(-1 * problem.minmax[0] * self.incumbent_solution.objectives_mean.item())
				

				#build random model 
				model, model_grad, U, fval, interpolation_solns, X, fX = self.construct_model(problem)

				# Diagnose model quality 
				diagnostics = self.diagnose_model_quality(model, model_grad, X, fX, U)
				self.print_model_diagnostics(diagnostics, problem)

				#solve random model 
				candidate_solution = self.solve_subproblem(model, model_grad, problem, U)

				
				#sample candidate solution
				candidate_solution, fval_tilde, = self.simulate_candidate_soln(problem, candidate_solution)

				
				# Diagnose candidate solution quality
				candidate_diagnostics = self.diagnose_candidate_solution(candidate_solution, model, problem, X)
				self.print_candidate_diagnostics(candidate_diagnostics, problem)
				
				# Store relative error for TR expansion dampening (keep last 5)
				self.recent_prediction_errors.append(candidate_diagnostics['relative_error'])
				if len(self.recent_prediction_errors) > 5:
					self.recent_prediction_errors.pop(0)  # Keep only last 5
				
				# Track prediction error for analysis
				self.prediction_error_history.append({
					'iteration': self.iteration_count,
					'relative_error': candidate_diagnostics['relative_error'],
					'prediction_error': candidate_diagnostics['prediction_error'],
					'trust_region_radius': self.delta,
					'min_dist_to_design': candidate_diagnostics['min_dist_to_design']
				})
				
				#evaluate model (adaptive trust region shrinkage now handled in update_parameters)
				self.evaluate_candidate_solution(problem, model, fval, fval_tilde, interpolation_solns, candidate_solution, X)

			
				self.iteration_count += 1

		finally :
			self.write_final_diagnostics(problem)
		# return None

	#! === TRUST-REGION METHODS === 

	def diagnose_candidate_solution(self, candidate_solution: Solution, model: callable, problem: Problem, X: np.ndarray) -> dict:
		"""
			Diagnose candidate solution quality by comparing to design points and model predictions.
			
		Args:
			candidate_solution (Solution): The candidate solution to diagnose
			model (callable): The surrogate model function
			problem (Problem): The current simulation model
			X (np.ndarray): Design points (M, n)
			
		Returns:
			dict: Diagnostic statistics
		"""
		x_candidate = np.array(candidate_solution.x).reshape(-1, 1)
		x_current = np.array(self.incumbent_x).reshape(-1, 1)
		
		# Distance metrics (computed here for diagnostics, also computed in update_parameters)
		distances_to_design = [norm(x_candidate.flatten() - X[i,:]) for i in range(X.shape[0])]
		min_dist_to_design = min(distances_to_design)
		mean_dist_to_design = np.mean(distances_to_design)
		
		# Full-space step size
		full_space_step = norm(x_candidate - x_current)
		
		# Model prediction vs actual
		x_candidate_arr = np.array(candidate_solution.x).reshape(1, -1)
		model_prediction = model(x_candidate_arr)
		actual_value = -1 * problem.minmax[0] * candidate_solution.objectives_mean.item()
		prediction_error = abs(model_prediction - actual_value)
		relative_error = prediction_error / (abs(actual_value) + 1e-10)
		
		# Current objective value
		current_value = -1 * problem.minmax[0] * self.incumbent_solution.objectives_mean.item()
		actual_improvement = current_value - actual_value
		predicted_improvement = current_value - model_prediction
		
		diagnostics = {
			'min_dist_to_design': min_dist_to_design,
			'mean_dist_to_design': mean_dist_to_design,
			'full_space_step': full_space_step,
			'trust_region_radius': self.delta,
			'step_to_radius_ratio': full_space_step / self.delta if self.delta > 0 else 0,
			'model_prediction': model_prediction,
			'actual_value': actual_value,
			'prediction_error': prediction_error,
			'relative_error': relative_error,
			'current_value': current_value,
			'actual_improvement': actual_improvement,
			'predicted_improvement': predicted_improvement
		}
		
		return diagnostics
	
	def print_candidate_diagnostics(self, diagnostics: dict, problem: Problem):
		"""Write candidate solution diagnostics to file.
		Args:
			diagnostics (dict): Diagnostic statistics
			problem (Problem): The simulation problem being solved
		"""
		output = "\n" + "="*70 + "\n"
		output += "CANDIDATE SOLUTION DIAGNOSTICS\n"
		output += "="*70 + "\n"
		output += f"Distance to nearest design point:  {diagnostics['min_dist_to_design']:.4f}\n"
		output += f"Mean distance to design points:     {diagnostics['mean_dist_to_design']:.4f}\n"
		output += f"Full-space step size:               {diagnostics['full_space_step']:.4f}\n"
		output += f"Trust region radius:                {diagnostics['trust_region_radius']:.4f}\n"
		output += f"Step to radius ratio:               {diagnostics['step_to_radius_ratio']:.1%}\n"
		output += "\nPrediction Quality:\n"
		output += f"  Model prediction:                 {diagnostics['model_prediction']:.6f}\n"
		output += f"  Actual value:                     {diagnostics['actual_value']:.6f}\n"
		output += f"  Prediction error:                 {diagnostics['prediction_error']:.6f} ({diagnostics['relative_error']*100:.1f}%)\n"
		output += "\nImprovement:\n"
		output += f"  Current objective:                {diagnostics['current_value']:.6f}\n"
		output += f"  Predicted improvement:            {diagnostics['predicted_improvement']:.6f}\n"
		output += f"  Actual improvement:               {diagnostics['actual_improvement']:.6f}\n"
		
		# Analysis of why prediction might be poor
		if diagnostics['relative_error'] > 0.15:
			output += "\n⚠️  WARNING: High prediction error detected!\n"
			if diagnostics['min_dist_to_design'] > 0.5 * diagnostics['trust_region_radius']:
				output += "  → Candidate is far from design points (interpolation issue)\n"
			if diagnostics['step_to_radius_ratio'] > 0.95:
				output += "  → Step at trust region boundary (extrapolation issue)\n"
			if abs(diagnostics['predicted_improvement']) < 1e-4:
				output += "  → Model predicts very small change (model may be too flat)\n"
		
		output += "="*70 + "\n\n"
		self.write_diagnostics_to_txt(output, problem=problem)

	def solve_subproblem(self, model: callable, model_grad: callable, problem: Problem, U: np.ndarray) -> Solution :
		"""
			Solves the trust-region subproblem within the reduced subspace with regularization.
			Objective: min m_k(U@z) + lambda * ||U@z||^2 / ||z||^2
			Constraint: ||x_k + U@z - x_k|| <= delta (full-space trust region)
			
			The regularization term penalizes small full-space steps for given reduced-space steps,
			encouraging solutions that make substantial progress in full space.

			NEW APPROACH: Penalize based on ratio of norms ||U·s|| / ||s|| to encourage large full-space steps
			while keeping the optimization stable. The norm is L-1 for better numerical behavior.


		Args:
			model (callable): The surrogate model function
			model_grad (callable): The surrogate model gradient function
			problem (Problem): The current simulation model
			U (np.ndarray): The (n,d) active subsapce matrix

		Returns:
			Solution: The candidate solution in the full space
		"""
		# Get current solution in full space
		x_current = np.array(self.incumbent_x).reshape(-1, 1)  # shape (n, 1)
		
		# Regularization weight: prevent null-space drift
		# Very small value to avoid interfering with optimization convergence
		lambda_reg = 0.001  # Reduced from 0.01 to be even less aggressive

		def obj_fn(z):
			# z is the step in reduced space (d,)
			z_col = np.array(z).reshape(-1, 1)  # shape (d, 1)
			z_full = U @ z_col  # Full-space step, shape (n, 1)
			
			# Model value at reduced-space step: m(s)
			model_val = model(z_col.reshape(1, -1))
			
			# Regularization: penalize when full-space step is small relative to reduced-space step
			# This encourages steps that make substantial progress in full space
			full_space_step_norm = norm(z_full, ord=1)
			reduced_space_step_norm = norm(z_col, ord=1)
			
			# if reduced_space_step_norm > 1e-10:
			# Penalty is low when full_space_step ≈ reduced_space_step (good)
			# Penalty is high when full_space_step << reduced_space_step (bad - null space drift)
			ratio_penalty = float(reduced_space_step_norm / (full_space_step_norm + 1e-10)) 
				# ratio_penalty = max(0.0, ratio_penalty)  # Only penalize when ratio > 1
			# else:
				# ratio_penalty = 0.0
			
			res = float(model_val + lambda_reg * ratio_penalty)
			return res 

		def obj_grad(z):
			# get gradient of objective function
			z_col = np.array(z).reshape(-1, 1)  # shape (d, 1)
			z_full = U @ z_col  # Full-space step, shape (n, 1)
			
			# Get gradient in reduced space (d-dimensional)
			# model_grad accepts reduced-space coordinates directly
			model_grad_reduced = model_grad(z_col.reshape(1, -1)).flatten()  # shape (d,)
			
			# Gradient of penalty term: ∂/∂s[||s|| / ||U·s||]
			# = s/(||s||·||U·s||) - (||s||/||U·s||³)·U^T·(U·s)
			full_space_step_norm = norm(z_full, ord=1)  # ||U·s||
			reduced_space_step_norm = norm(z_col, ord=1)  # ||s||
			
			if reduced_space_step_norm > 1e-10 and full_space_step_norm > 1e-10:
				term1 = z_col / (reduced_space_step_norm * full_space_step_norm)  # s/(||s||·||U·s||)
				term2 = (reduced_space_step_norm / (full_space_step_norm ** 3)) * (U.T @ z_full)  # (||s||/||U·s||³)·U^T·(U·s)
				penalty_grad = (term1 - term2).flatten()  # shape (d,)
			else:
				penalty_grad = np.zeros(self.d)  # Avoid division by zero at origin

			# if reduced_space_step_norm > 1e-10 and full_space_step_norm > 1e-10:
			# 	ratio = reduced_space_step_norm / full_space_step_norm
			# 	if ratio > 1.0:  # Only add gradient when penalizing
			# 		# Gradient of penalty term (approximation for numerical stability)
			# 		grad_penalty = z / (reduced_space_step_norm + 1e-10)
			# 		model_grad_reduced = model_grad_reduced + lambda_reg * grad_penalty
			model_grad_reduced += lambda_reg * penalty_grad.flatten()
			
			return model_grad_reduced
		
		def cons_fn(z):
			# Constraint: ||s|| <= delta (step size in REDUCED space)
			# Penalty term encourages large ||U·s|| relative to ||s|| for algorithmic progress
			z_col = np.array(z).reshape(-1, 1)
			return norm(z_col)

		cons = NonlinearConstraint(cons_fn, 0, self.delta)
		# Check model gradient at current point to diagnose flat models
		grad_at_current = obj_grad(np.zeros(self.d)) 


		if norm(grad_at_current) < 1e-6:
			warning = f'⚠️ WARNING: Model gradient is very small - model may be too flat!\n'
			self.write_diagnostics_to_txt(warning, problem=problem)
		
		# Solve trust region subproblem
		# Start from origin (no step) in reduced space
		# Relax tolerances to prevent premature xtol termination
		res = minimize(obj_fn, np.zeros(self.d), method='trust-constr', jac=obj_grad, 
					   constraints=cons, options={'disp': False, 'verbose': 0, 
					   'xtol': 1e-8, 'gtol': 1e-6})
		
		if not res.success:
			warning = f'⚠️ WARNING: Optimizer did not fully converge: {res.message}\n'
			self.write_diagnostics_to_txt(warning, problem=problem)
		

		# Ensure the reduced-space step respects the trust region (guard against optimizer drift)
		step_norm_reduced = norm(res.x)
		if step_norm_reduced > self.delta * 1.001:  # 0.1% tolerance
			# print(f'WARNING: Step violated trust region: {step_norm_reduced:.4f} > {self.delta:.4f}')
			res.x = res.x * (self.delta / step_norm_reduced)
			step_norm_reduced = self.delta
		
		# Check if step hit trust region boundary (indicates model gradient is strong)
		boundary_tolerance = 0.9 * self.delta

		if step_norm_reduced < boundary_tolerance :	
			# If step is very small, try Cauchy point (steepest descent to boundary)
			if step_norm_reduced < 0.5 * self.delta:
				# print(f'  -> Step is small, trying Cauchy point...')
				# Get gradient at current point in reduced space
				grad_reduced = obj_grad(np.zeros(self.d))
				grad_norm = norm(grad_reduced)
				
				if grad_norm > 1e-10:
					# Cauchy point: step in steepest descent direction to trust region boundary
					cauchy_step = -(self.delta / grad_norm) * grad_reduced
					cauchy_obj = obj_fn(cauchy_step)
					current_obj = obj_fn(np.zeros(self.d))
					
					# Use Cauchy point if it improves over the optimizer's solution
					if cauchy_obj < obj_fn(res.x):
						# print(f'  -> Using Cauchy point (obj: {cauchy_obj:.6f} vs {obj_fn(res.x):.6f})')
						res.x = cauchy_step
						step_norm_reduced = self.delta
				

		# Compute full space point directly (already done in obj_fn, but needed here)
		s_new = (x_current + U @ res.x.reshape(-1, 1)).flatten()

		

		for i in range(problem.dim):
			if s_new[i] <= problem.lower_bounds[i]:
				s_new[i] = problem.lower_bounds[i] + 0.01
			elif s_new[i] >= problem.upper_bounds[i]:
				s_new[i] = problem.upper_bounds[i] - 0.01



		candidate_solution = self.create_new_solution(tuple(s_new), problem)

		self.visited_points.append(candidate_solution) 

		return candidate_solution

	

	def evaluate_candidate_solution(self, problem: Problem, model: callable, fval: list[float], fval_tilde: float,
								   interpolation_solns: list[Solution], candidate_solution: Solution, X: np.ndarray) -> None :
		"""
			Evaluates the candidate solution and updates the trust-region radius accordingly
		Args:
			problem (Problem): The current simulation model
			model (callable): The surrogate model function
			fval (list[float]): The list of objective function values at interpolation points
			fval_tilde (float): The predicted objective function value at the candidate solution
			U (np.ndarray): The (n,d) active subsapce matrix
			interpolation_solns (list[Solution]): The list of interpolation solutions
			candidate_solution (Solution): The candidate solution to be evaluated
		"""
		#pattern search
		if ((min(fval) < fval_tilde) and ((fval[0] - min(fval))>= self.factors["ps_sufficient_reduction"] * self.delta**2)) or ((candidate_solution.objectives_var[0]/ (candidate_solution.n_reps * candidate_solution.objectives_mean[0]**2)) > 0.75):
			fval_tilde = min(fval)
			candidate_solution = interpolation_solns[fval.index(min(fval))]  # type: ignore

		#compute ratio
		rho = self.compute_ratio(problem, model, candidate_solution, fval_tilde)


		#update parameters
		# Check if rho is the sentinel value for cautious acceptance
		cautious_accept = (rho == -999.0)
		self.update_parameters(rho, candidate_solution, X, cautious_accept=cautious_accept)

			

	def update_parameters(self, rho: float, candidate_solution: Solution, X: np.ndarray, cautious_accept: bool = False) -> None :
		"""
			Update the trust-region radius and current solution based on the ratio rho.
			Also performs adaptive trust region shrinkage based on interpolation quality.
		Args:
			rho (float): The ratio of actual reduction to predicted reduction
			candidate_solution (Solution): The candidate solution being considered
			X (np.ndarray): Design points (M, n) for computing interpolation quality
			cautious_accept (bool): If True, accept solution but keep trust-region radius unchanged
		Returns:
			tuple[Solution, float]: The updated current solution and trust-region radius
		"""

		# Adaptive trust region based on interpolation quality
		# Compute distance from candidate to nearest design point
		x_candidate = np.array(candidate_solution.x).reshape(-1, 1)
		distances_to_design = [norm(x_candidate.flatten() - X[i,:]) for i in range(X.shape[0])]
		min_dist_to_design = min(distances_to_design)
		
		# If candidate is consistently far from design points, shrink trust region MORE aggressively
		if min_dist_to_design > 0.6 * self.delta:
			old_delta = self.delta
			self.delta = max(0.5 * self.delta, self.delta_min)
			# print(f"\n⚠️  Candidate far from design points. Shrinking trust region aggressively: {old_delta:.4f} → {self.delta:.4f}\n")


		if cautious_accept:
			# Accept the solution because it shows actual improvement, but don't change trust region
			# The model is unreliable, so we don't reward with radius increase
			# print(f'Cautious acceptance with actual improvement despite poor model prediction')
			self.incumbent_solution: Solution = candidate_solution
			self.incumbent_x: tuple = candidate_solution.x
			
			self.recommended_solns.append(candidate_solution)
			self.successful_iterations.append(candidate_solution)
			self.intermediate_budgets.append(self.budget.used)
			
			# Keep delta unchanged (no increase or decrease)
			# Optionally: could apply modest shrinkage like: self.delta = max(0.9 * self.delta, self.delta_min)
			# print(f'Trust-region radius unchanged at {self.delta:.4f}')

		elif rho >= self.eta_1:
			# print(f'Successful iteration with rho={rho:.4f}')
			# print(f'changing incumbent solution from f={self.incumbent_solution.objectives_mean[0]:.4f} to f={candidate_solution.objectives_mean[0]:.4f}')
			self.incumbent_solution: Solution = candidate_solution
			self.incumbent_x: tuple = candidate_solution.x
			
			self.recommended_solns.append(candidate_solution)
			self.successful_iterations.append(candidate_solution)
			self.intermediate_budgets.append(self.budget.used)
			
			old_delta = self.delta
			
			# Check recent prediction quality to inform TR expansion
			# If recent predictions are poor, be more conservative with expansion
			avg_recent_error = np.mean(self.recent_prediction_errors) if len(self.recent_prediction_errors) > 0 else 0.0
			
			# Dampen expansion if prediction quality is degrading
			if avg_recent_error > 0.20:  # More than 20% average error
				expansion_factor = 0.8  # Dampen expansion by 20%
			elif avg_recent_error > 0.15:  # More than 15% average error
				expansion_factor = 0.9  # Dampen expansion by 10%
			else:
				expansion_factor = 1.0  # Full expansion allowed
			
			if rho >= self.eta_2:
				# Very successful: use gamma_1 (larger increase) but dampen if predictions poor
				new_delta = self.gamma_1 * self.delta
				self.delta: float = max(min(old_delta + expansion_factor * (new_delta - old_delta), self.delta_max), self.delta_min)
				# print(f'Very successful! Trust-region radius increased from {old_delta:.4f} to {self.delta:.4f}')
			else:
				# Moderately successful: use gamma_2 (smaller increase) with dampening
				new_delta = self.gamma_2 * self.delta
				self.delta: float = max(min(old_delta + expansion_factor * (new_delta - old_delta), self.delta_max), self.delta_min)
				# print(f'Moderately successful! Trust-region radius increased from {old_delta:.4f} to {self.delta:.4f}')
	
		else:
			# print(f'Unsuccessful iteration with rho={rho:.4f} (threshold eta_1={self.eta_1})')
			old_delta = self.delta
			self.delta: float = max(self.gamma_3 * self.delta, self.delta_min)
			# print(f'Trust-region radius decreased from {old_delta:.4f} to {self.delta:.4f}')

			self.unsuccessful_iterations.append(self.incumbent_solution)
		


	#! THIS NEEDS REWRITING 
	def compute_ratio(self, 
				   problem: Problem, 
				   model: callable,
				   candidate_solution: Solution, fval_tilde: float) -> float:
		"""
			Compute the ratio of actual reduction to predicted reduction 
			we produce two values here: rho_effective, which is the ratio used to update the trust-region 
			radius and force_accept, which indicates whether to force acceptance of the candidate solution.
			force_accept is true if the candidate solution shows statistically significant improvement over 
			the current solution
		Args:
			problem (Problem): The current simulation model
			model (callable): The surrogate model used for prediction
			candidate_solution (Solution): The candidate solution being evaluated
			fval_tilde (float): The predicted objective function value at the candidate solution
			
		Returns:
			float: The effective ratio used to update the trust-region radius
											whether the step is feasible
		"""

		current_f = -1 * problem.minmax[0] * self.incumbent_solution.objectives_mean.item()
		candidate_f = fval_tilde
		actual_improvement = current_f - candidate_f
		# print(f'Actual f at current solution: {current_f:.4f}, Actual f at candidate solution: {candidate_f:.4f}')

		current_m = model(np.array(self.incumbent_x).reshape(1,-1))
		candidate_m = model(np.array(candidate_solution.x).reshape(1,-1))
		predicted_improvement = current_m - candidate_m
		# print(f'Predicted f at current solution: {current_m:.4f}, Predicted f at candidate solution: {candidate_m:.4f}')
		# print(f'Actual improvement: {actual_improvement:.6f}, Predicted improvement: {predicted_improvement:.6f}')

		# Safeguard against very small predicted improvements
		abs_tolerance = 1e-10 * max(1.0, abs(current_m), abs(candidate_m))
		
		# Handle different cases
		if predicted_improvement <= abs_tolerance:
			# Model predicts no improvement or worsening
			if actual_improvement > 0:
				# Actual improvement despite poor model prediction - accept cautiously
				# print(f'Cautious acceptance: actual improvement {actual_improvement:.6f} despite predicted improvement {predicted_improvement:.6f}')
				rho = -999.0  # Special sentinel for cautious acceptance
			else:
				# Both model and reality show no improvement
				rho = -1e6  # Reject
		else:
			# Normal case: model predicts improvement
			rho = actual_improvement / predicted_improvement
			
			# Clamp extreme ratios that might arise from numerical issues
			rho = max(-100.0, min(100.0, rho))
			
			# Additional check: if we have actual improvement but rho < eta_1, 
			# be more lenient if the actual improvement is significant
			if actual_improvement > abs_tolerance and rho < self.eta_1:
				# Check if actual improvement is substantial relative to current value
				relative_improvement = abs(actual_improvement) / max(abs(current_f), 1e-10)
				if relative_improvement > 0.001:  # 0.1% relative improvement
					# print(f'Forcing acceptance due to significant actual improvement: {actual_improvement:.6f} ({relative_improvement*100:.2f}%)')
					rho = self.eta_1  # Bump up to minimum acceptance threshold

		# print(f'Final rho: {rho:.6f}')
		return rho
	
	def diagnose_model_quality(self, model: callable, model_grad: callable, X: np.ndarray, fX: np.ndarray, U: np.ndarray) -> dict:
		"""
		Compute comprehensive model quality diagnostics to understand poor success rates.
		
		Args:
			model (callable): The polynomial model
			model_grad (callable): The gradient of the polynomial model
			X (np.ndarray): Design points (M, n)
			fX (np.ndarray): True function values at design points (M, 1)
			U (np.ndarray): Active subspace (n, d)
		
		Returns:
			dict: Diagnostic statistics
		"""
		M = X.shape[0]
		n = X.shape[1]
		
		# 1. Model fit quality on design points
		predictions = np.array([model(X[i, :].reshape(-1, 1)) for i in range(M)])
		residuals = fX.flatten() - predictions
		
		# Compute R² (coefficient of determination)
		ss_res = np.sum(residuals**2)
		ss_tot = np.sum((fX.flatten() - np.mean(fX))**2)
		r_squared = 1 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
		
		# Root mean squared error
		rmse = np.sqrt(ss_res / M)
		
		# Maximum absolute error
		max_error = np.max(np.abs(residuals))
		
		# 2. Active subspace quality
		# Project design points to subspace
		X_proj = X @ U  # (M, d)
		
		# Compute spread in full space vs subspace
		full_space_variance = np.sum(np.var(X, axis=0))
		subspace_variance = np.sum(np.var(X_proj, axis=0))
		variance_captured = subspace_variance / full_space_variance if full_space_variance > 1e-12 else 0.0
		
		# Check orthonormality of U
		should_be_identity = U.T @ U
		orthonormality_error = np.max(np.abs(should_be_identity - np.eye(self.d)))
		
		# 3. Design set geometry
		# Compute condition number of Vandermonde matrix
		V_matrix = self.V(X_proj)
		cond_V = np.linalg.cond(V_matrix)
		
		# Compute pairwise distances (sample 100 pairs if M is large)
		sample_size = min(M, 100)
		indices = np.random.choice(M, size=sample_size, replace=False) if M > 100 else np.arange(M)
		distances = []
		for i in range(len(indices)):
			for j in range(i+1, len(indices)):
				distances.append(norm(X[indices[i], :] - X[indices[j], :]))
		
		min_dist = np.min(distances) if distances else 0.0
		max_dist = np.max(distances) if distances else 0.0
		mean_dist = np.mean(distances) if distances else 0.0
		
		# 4. Gradient information
		# Compute model gradient at incumbent (first point in X)
		x_current = np.array(self.incumbent_x).reshape(1,-1)
		grad_norm = norm(model_grad(x_current))  # Will use internal coef
		
		diagnostics = {
			'r_squared': r_squared,
			'rmse': rmse,
			'max_error': max_error,
			'variance_captured': variance_captured,
			'orthonormality_error': orthonormality_error,
			'vandermonde_condition': cond_V,
			'min_point_distance': min_dist,
			'max_point_distance': max_dist,
			'mean_point_distance': mean_dist,
			'gradient_norm': grad_norm,
			'num_design_points': M,
			'dimension': n,
			'subspace_dim': self.d,
			'delta': self.delta
		}
		
		return diagnostics
	
	def print_model_diagnostics(self, diagnostics: dict, problem: Problem):
		"""Write model quality diagnostics to file.
		Args:
			diagnostics (dict): Model quality statistics
			problem (Problem): The simulation problem being solved
		"""
		output = "\n" + "="*70 + "\n"
		output += "MODEL QUALITY DIAGNOSTICS\n"
		output += "="*70 + "\n"
		
		output += "\n📊 MODEL FIT QUALITY:\n"
		r2_status = '⚠️ POOR' if diagnostics['r_squared'] < 0.7 else '✓ Good' if diagnostics['r_squared'] > 0.9 else '⚠️ Moderate'
		
		# Check R² trend if we have previous value
		if self.prev_r_squared is not None:
			r2_change = diagnostics['r_squared'] - self.prev_r_squared
			trend_arrow = "📈" if r2_change > 0.05 else "📉" if r2_change < -0.05 else "→"
			output += f"  R² (goodness of fit):     {diagnostics['r_squared']:.4f}  {r2_status}  {trend_arrow} (Δ={r2_change:+.3f})\n"
			
			if r2_change < -0.1:
				output += f"    ⚠️  R² dropped significantly! Previous: {self.prev_r_squared:.4f}\n"
		else:
			output += f"  R² (goodness of fit):     {diagnostics['r_squared']:.4f}  {r2_status}\n"
		
		# Store for next iteration
		self.prev_r_squared = diagnostics['r_squared']
		
		output += f"  RMSE:                     {diagnostics['rmse']:.4e}\n"
		output += f"  Max absolute error:       {diagnostics['max_error']:.4e}\n"
		
		output += "\n🎯 ACTIVE SUBSPACE QUALITY:\n"
		output += f"  Variance captured:        {diagnostics['variance_captured']:.4f}  {'⚠️ POOR' if diagnostics['variance_captured'] < 0.5 else '✓ Good' if diagnostics['variance_captured'] > 0.8 else '⚠️ Moderate'}\n"
		output += f"  Orthonormality error:     {diagnostics['orthonormality_error']:.4e}  {'⚠️ ISSUE' if diagnostics['orthonormality_error'] > 1e-6 else '✓ Good'}\n"
		output += f"  Subspace dimension:       {diagnostics['subspace_dim']}/{diagnostics['dimension']}\n"
		
		output += "\n📐 DESIGN SET GEOMETRY:\n"
		output += f"  Condition number:         {diagnostics['vandermonde_condition']:.4e}  {'⚠️ ILL-CONDITIONED' if diagnostics['vandermonde_condition'] > 1e3 else '⚠️ Poor' if diagnostics['vandermonde_condition'] > 1e2 else '✓ Good'}\n"
		output += f"  Number of points:         {diagnostics['num_design_points']}\n"
		output += f"  Min point distance:       {diagnostics['min_point_distance']:.4e}\n"
		output += f"  Max point distance:       {diagnostics['max_point_distance']:.4e}\n"
		output += f"  Mean point distance:      {diagnostics['mean_point_distance']:.4e}\n"
		
		output += "\n🔍 OPTIMIZATION INFO:\n"
		output += f"  Gradient norm:            {diagnostics['gradient_norm']:.4e}\n"
		output += f"  Trust-region radius:      {diagnostics['delta']:.4e}\n"
		
		# Overall assessment
		output += "\n💡 ASSESSMENT:\n"
		issues = []
		if diagnostics['r_squared'] < 0.7:
			issues.append("  ⚠️  Low R² - model doesn't fit training data well")
		if diagnostics['variance_captured'] < 0.5:
			issues.append("  ⚠️  Low variance captured - active subspace may be missing important directions")
		if diagnostics['vandermonde_condition'] > 1e10:
			issues.append("  ⚠️  Ill-conditioned Vandermonde - design points may be too close or poorly distributed")
		if diagnostics['min_point_distance'] < 1e-8:
			issues.append("  ⚠️  Points too close together - numerical instability likely")
		if diagnostics['orthonormality_error'] > 1e-6:
			issues.append("  ⚠️  Active subspace not orthonormal - projection may be inaccurate")
		
		if issues:
			for issue in issues:
				output += issue + "\n"
		else:
			output += "  ✓ All metrics look reasonable\n"
		
		output += "="*70 + "\n\n"
		self.write_diagnostics_to_txt(output, problem=problem)


	#! === SAMPLING METHODS === 

	def evaluate_interpolation_points(self, problem: Problem, visited_index: int, X:np.ndarray) -> tuple[np.ndarray, list[Solution]]:
		"""
			Run adaptive sampling on the model construction design points to obtain a sample 
			average of their responses.
		Args:
			problem (Problem): The Simulation Problem being run.
			current_solution (Solution): The incumbent solution of the solver
			visited_index (int): The index of the current solution in the visited points list
			X (np.ndarray): The design points for model construction
			visited_pts (list[Solution]): The list of previously visited solutions
			delta (float): The current trust-region radius

		Returns:
			tuple[np.ndarray, list[Solution]]:
												The array of sample average objective function values at the design points,
												The list of interpolation solutions,
		"""	
		fX = []	 
		interpolation_solutions = []
		for idx,x in enumerate(X) : 
			#for the current solution, we don't need to simulate
			if (idx == 0) and (self.iteration_count==1) :
				fX.append(-1 * problem.minmax[0] * self.incumbent_solution.objectives_mean.item()) 
				interpolation_solutions.append(self.incumbent_solution)
			
			#reuse the replications for x_k
			elif idx == 0: 
				self.incumbent_solution = self.adaptive_sampling_before(problem, self.incumbent_solution)
				fX.append(-1 * problem.minmax[0] * self.incumbent_solution.objectives_mean.item())
				interpolation_solutions.append(self.incumbent_solution)

			elif (idx==1) and ((norm(np.array(self.incumbent_x)-np.array(self.visited_points[visited_index].x)) != 0) and self.visited_points is not None) :
				reuse_solution = self.visited_points[visited_index]
				reuse_solution = self.adaptive_sampling_before(problem, reuse_solution)
				fX.append(-1 * problem.minmax[0] * reuse_solution.objectives_mean.item())
				interpolation_solutions.append(reuse_solution)
			#For new points, run the simulation with pilot run
			else :
				solution = self.create_new_solution(tuple(x.ravel()), problem)
				solution = self.adaptive_sampling_after(problem, solution)
				fX.append(-1 * problem.minmax[0] * solution.objectives_mean.item())
				interpolation_solutions.append(solution)


		return np.array(fX).reshape(-1,1), interpolation_solutions

	def simulate_candidate_soln(self, problem: Problem, candidate_solution: Solution) -> tuple[Solution, float] :
		"""
			Run adaptive sampling on the candidate solution to obtain a sample average of the 
			response to the candidate solution.

		Args:
			problem (Problem): The Simulation Problem being run.
			candidate_solution (Solution): The candidate solution to be evaluated
			current_solution (Solution): The incumbent solution of the solver
			delta (float): The current trust-region radius

		Returns:
			tuple[Solution, float]: 
										The updated candidate solution with simulation results,
										The sample average objective function value at the candidate solution,
		"""
		if self.factors['crn_across_solns'] :
			problem.simulate(candidate_solution, self.incumbent_solution.n_reps) 
			self.budget.request(self.incumbent_solution.n_reps) 
			fval_tilde = -1 * problem.minmax[0] * candidate_solution.objectives_mean.item()
			return candidate_solution, fval_tilde
			
		else :
			candidate_solution = self.adaptive_sampling_after(problem, candidate_solution)
			fval_tilde = -1 * problem.minmax[0] * candidate_solution.objectives_mean.item()
			return candidate_solution, fval_tilde

	def calculate_pilot_run(self, problem: Problem) -> int : 
		"""
			Calculate the pilot run sample size based on the current iteration number k
		Args:
			problem (Problem): The Simulation Problem being run.

		Returns:
			int: The calculated pilot run sample size
		"""
		lambda_max = self.budget.remaining
		pilot_run = ceil(max(self.lambda_min * log(10 + self.iteration_count, 10) ** 1.1, min(0.5 * problem.dim, lambda_max))-1)
		
		return pilot_run

	def calculate_kappa(self, problem: Problem) -> None :
		"""
			Calculate kappa and run adaptive sampling on the incumbent solution to obtain a sample average of its response.
		Args:
			problem (Problem): The Simulation Problem being run.
		"""
		lambda_max = self.budget.remaining
		pilot_run = self.calculate_pilot_run(problem)

		if self.kappa is None:
			self.kappa = 0.0

		#calculate kappa
		problem.simulate(self.incumbent_solution, pilot_run)
		self.budget.request(pilot_run)
		sample_size = pilot_run
		while True:
			rhs_for_kappa = self.incumbent_solution.objectives_mean
			sig2 = self.incumbent_solution.objectives_var[0]

			self.kappa = rhs_for_kappa * np.sqrt(pilot_run) / (self.delta ** (self.delta_power / 2))
			stopping = self.get_stopping_time(sig2, problem)
			
			if (sample_size >= min(stopping, lambda_max) or self.budget.remaining <= 0):
				# calculate kappa
				self.kappa = (rhs_for_kappa * np.sqrt(pilot_run)/ (self.delta ** (self.delta_power / 2)))
				break
			
			problem.simulate(self.incumbent_solution, 1)
			self.budget.request(1)
			sample_size += 1

	
	def get_stopping_time(self, sig2: float, problem: Problem) -> float :
		"""
			Calculate the stopping time (sample size) based on current variance estimate, delta, and kappa
		Args:
			sig2 (float): Current variance estimate
			problem (Problem): The Simulation Problem being run.

		Returns:
			float: The calculated stopping time (sample size)
		"""
		pilot_run = self.calculate_pilot_run(problem)
		if self.kappa == 0.0:
			self.kappa = 1.0

		raw_sample_size = pilot_run * max(1, sig2 / (self.kappa**2 * self.delta**self.delta_power))
		if isinstance(raw_sample_size, np.ndarray):
			raw_sample_size = raw_sample_size.item()
		# round up to the nearest integer
		sample_size: int = ceil(raw_sample_size)
		return sample_size
	
	def samplesize(self, sig: float) -> int:
		"""
			Calculate the sample size S_k based on current variance estimate, delta, and kappa
		Args:
			sig (float): Current variance estimate

		Returns:
			int: The calculated sample size S_k
		"""
		lambda_k: int = 1 if self.iteration_count == 1 else floor(self.lambda_min*math.log(self.iteration_count)**1.5)
		S_k = ceil((lambda_k*sig**2)/(self.kappa**2*self.delta**4))
		return S_k
	
	def adaptive_sampling_after(self, problem: Problem, new_solution: Solution) -> Solution  :
		"""
			Run adaptive sampling on a new solution to obtain a sample average of its response.
		Args:
			problem (Problem): The Simulation Problem being run.
			new_solution (Solution): The new solution to be sampled

		Returns:
			Solution: The updated solution with simulation results
		"""
		# adaptive sampling
		lambda_max = self.budget.remaining
		pilot_run = self.calculate_pilot_run(problem)

		problem.simulate(new_solution, pilot_run)
		self.budget.request(pilot_run)
		sample_size = pilot_run
		sig_init = new_solution.objectives_var[0]

		while True:
			sig2 = new_solution.objectives_var[0]
			stopping = self.get_stopping_time(sig2, problem)
			if ((sample_size >= min(stopping, lambda_max)) or self.budget.remaining <= 0):
				return new_solution
			
			problem.simulate(new_solution, 1)
			self.budget.request(1)
			sample_size += 1

	
	#this is sample_type = 'conditional before'	
	def adaptive_sampling_before(self, problem: Problem, new_solution: Solution) -> Solution : 
		"""
			Run adaptive sampling on an existing solution to obtain a sample average of its response.
		Args:
			problem (Problem): The Simulation Problem being run.
			new_solution (Solution): The existing solution to be sampled

		Returns:
			Solution: The updated solution with simulation results
		"""

		lambda_max = self.budget.remaining

		if new_solution.n_reps < 2 :
			sim_no = 2-new_solution.n_reps
			problem.simulate(new_solution,sim_no)
			self.budget.request(sim_no)
			sample_size = sim_no

		sample_size = new_solution.n_reps 
		sig2 = new_solution.objectives_var[0]

		while True:
			stopping = self.get_stopping_time(sig2, problem)
			if (sample_size >= min(stopping, lambda_max) or self.budget.remaining <= 0):
				return new_solution
			
			problem.simulate(new_solution, 1)
			self.budget.request(1)
			sample_size += 1
			sig2 = new_solution.objectives_var[0]


	#! === PRELIMINARY FUNCTIONS ===
	# delta = 10 ** (ceil(log(self.delta_max * 2, 10) - 1) / problem.dim)
	def calculate_max_radius(self, problem: Problem) -> float : 
		"""
			Calculate the maximum trust-region radius based on the problem's variable bounds and random sampling.
		Args:
			problem (Problem): The Simulation Problem being run.

		Returns:
			float: The calculated maximum trust-region radius
		"""
		find_next_soln_rng = self.rng_list[1]

		dummy_solns: list[tuple[int, ...]] = []
		for _ in range(1000 * problem.dim):
			random_soln = problem.get_random_solution(find_next_soln_rng)
			dummy_solns.append(random_soln)
		delta_max_arr: list[float | int] = []
		for i in range(problem.dim):
			delta_max_arr += [
				min(
					max([sol[i] for sol in dummy_solns])
					- min([sol[i] for sol in dummy_solns]),
					problem.upper_bounds[0] - problem.lower_bounds[0],
				)
			]
		delta_max = max(delta_max_arr)
		return delta_max 
	

	def calculate_min_radius(self, problem: Problem) -> float :
		"""
			Calculate the minimum trust-region radius based on the problem's variable bounds.
		Args:
			problem (Problem): The Simulation Problem being run.

		Returns:
			float: The calculated minimum trust-region radius
		"""
		finite_spans = []
		for j in range(problem.dim):
			lb = problem.lower_bounds[j]
			ub = problem.upper_bounds[j]
			if np.isfinite(lb) and np.isfinite(ub):
				span = max(ub - lb, 0.0)
				if span > 0.0:
					finite_spans.append(span)

		if finite_spans:
			base = min(finite_spans)
		else:
			base = self.calculate_max_radius(problem)

		return max(1e-8, 1e-3 * base)

	#! === DESIGN SET CONSTRUCTION === 


	def column_vectors_U(self, index: int, U: np.ndarray) -> np.ndarray:
		"""
		Get the index column vector of U. The column vectors are orthonormal basis vectors that span the active subspace. 

		Args:
			problem (Problem): The SO problem 
			index (int): The index of the column vector
			U (np.ndarray): The active subspace matrix

		Returns:
			np.ndarray: The n-dimensional column vector at the given index
		"""
		col_vector = U[:, index].reshape(-1,1)
		return col_vector

	def compute_adaptive_interpolation_radius_fraction(self, problem: Problem, U: np.ndarray) -> float:
		"""
		Compute interpolation radius as a fraction of trust-region radius.
		
		KEY PRINCIPLE: Interpolation radius should grow MORE SLOWLY than TR radius
		to maintain model quality as TR expands.
		
		PROBLEM: When TR expands (e.g., 0.5 → 1.0 after success):
		- Linear scaling: interpolation radius also doubles (0.4 → 0.8)
		- Model becomes less accurate in center where candidates are generated
		- Leads to high prediction errors after TR expansion
		
		SOLUTION: Use sublinear scaling - cap the effective interpolation radius
		- For small TR: use ~0.8δ (good coverage)
		- For large TR: use smaller fraction to keep points tighter (better accuracy)
		- This prevents model degradation after TR expansion
		
		Args:
			problem (Problem): The simulation problem
			U (np.ndarray): The (n,d) active subspace matrix
			
		Returns:
			float: Fraction of trust-region radius to use
		"""
		# Compute reference scale: average distance between bounds
		lower = np.array(problem.lower_bounds).reshape(-1, 1)
		upper = np.array(problem.upper_bounds).reshape(-1, 1)
		domain_scale = np.mean(upper - lower)
		
		# Base fraction depends on TR size relative to domain
		tr_relative_size = self.delta / domain_scale
		
		# Use adaptive scaling based on TR size:
		# - Small TR (< 10% of domain): use 0.75-0.80 (broader coverage)
		# - Medium TR (10-30% of domain): use 0.65-0.75 (balanced)
		# - Large TR (> 30% of domain): use 0.55-0.65 (tighter for accuracy)
		if tr_relative_size < 0.1:
			base_fraction = 0.80
		elif tr_relative_size < 0.2:
			base_fraction = 0.75
		elif tr_relative_size < 0.3:
			base_fraction = 0.70
		else:
			# For very large TR, use even smaller fraction to maintain accuracy
			base_fraction = max(0.55, 0.80 - 1.0 * tr_relative_size)
		
		# Adjust for subspace dimension
		if self.d >= 8:
			base_fraction = min(base_fraction + 0.05, 0.85)  # Slightly wider for high-D
		
		# Case: Trust region hitting domain bounds
		x_k = np.array(self.incumbent_x).reshape(-1, 1)
		dist_to_lower = x_k - lower
		dist_to_upper = upper - x_k
		min_clearance = min(np.min(dist_to_lower), np.min(dist_to_upper))
		
		# If trust region would place points outside bounds, reduce radius
		if min_clearance < 0.9 * self.delta:
			safe_fraction = 0.9 * min_clearance / self.delta
			base_fraction = min(base_fraction, max(safe_fraction, 0.5))
		
		return base_fraction

	def interpolation_points_without_reuse(self, problem: Problem, U: np.ndarray) -> list[np.ndarray]:
		"""
		Constructs a 2d+1 interpolation set without reusing points.
		Points placed at adaptively computed radius for optimal coverage of typical candidate locations.
		
		Args:
			problem (Problem): The current simulation model
			U (np.ndarray): The (n,d) active subspace matrix
		
		Returns:
			[np.array]: A list of 2d+1 n-dimensional design points for interpolation
		"""
		x_k = np.array(self.incumbent_x).reshape(-1,1)
		Y = [x_k]
		epsilon = 0.01
		
		# Adaptively compute interpolation radius based on problem characteristics
		radius_fraction = self.compute_adaptive_interpolation_radius_fraction(problem, U)
		interpolation_radius = radius_fraction * self.delta
		
		for i in range(0, self.d):
			plus = Y[0] + interpolation_radius * self.column_vectors_U(i, U)
			minus = Y[0] - interpolation_radius * self.column_vectors_U(i, U)

			if sum(x_k) != 0:
				# block constraints
				if minus[i] <= problem.lower_bounds[i]:
					minus[i] = problem.lower_bounds[i] + epsilon
				if plus[i] >= problem.upper_bounds[i]:
					plus[i] = problem.upper_bounds[i] - epsilon

			Y.append(plus)
			Y.append(minus)

		return Y 


	# generate the mutually orthonormal rotated basis using A_k1 as the first basis vector
	def get_rotated_basis(self, A_k1: np.ndarray, U: np.ndarray) -> list[np.ndarray]:
		"""
		Generate the other d-1 rotated coordinate basis using A_k1 as the first basis vector. 
		We use Gram-Schmidt process to generate the orthonormal basis.
		Args:
			A_k1 (np.ndarray): The first direction vector for the reused design point
			d (int): The subspace dimension and the number of vectors to have 
			U (np.ndarray): The (n,d) active subspace matrix

		Returns:
			list[np.ndarray]: A list of d d-dimensional rotated basis vectors each with shape (d,1)
		"""

		# Start with A_normalized as first vector
		basis = [A_k1]
		# Generate candidate vectors from the standard basis
		I = np.eye(self.d)
		candidates = [I[:,i].reshape(-1,1) for i in range(1,self.d)]		



		#Build successive orthononormal basis using Gram-Schmidt process from A_k1
		for c in candidates:
			v = c.copy()
			#calculate gram-schmidt projection
			for b in basis:
				dot_prod = v.T @ b
				v -= dot_prod.item() * b 

			#Normalize v
			v = v / np.linalg.norm(v)
			basis.append(v.reshape(-1,1))

			if len(basis) == self.d:
				break

		return basis

	# compute the interpolation points (2d+1) using the rotated coordinate basis (reuse one design point)
	def interpolation_points_with_reuse(self, problem: Problem, reused_x: np.ndarray, rotation_vectors: list[np.ndarray], U: np.ndarray) -> list[np.ndarray]:
		"""
			Constructs a 2d+1 interpolation set with reusing one design point.
			Points placed at adaptively computed radius for optimal coverage of typical candidate locations.
		Args:
			problem (Problem): The current simulation model
			x_k (np.ndarray): The current incumbent solution
			reused_x (np.ndarray): The design point to be reused
			delta (float): The current trust-region radius
			rotation_vectors (list[np.ndarray]): The rotated coordinate basis vectors
			U (np.ndarray): The (n,d) active subspace matrix

		Returns:
			list[np.ndarray]: A list of 2d+1 n-dimensional design points for interpolation
		"""
		x_k = np.array(self.incumbent_x).reshape(-1,1)
		Y = [x_k, reused_x]
		epsilon = 0.01

		# Adaptively compute interpolation radius based on problem characteristics
		radius_fraction = self.compute_adaptive_interpolation_radius_fraction(problem, U)
		interpolation_radius = radius_fraction * self.delta
		
		for i in range(1, self.d):
			plus = Y[0] + interpolation_radius * (U @ rotation_vectors[i])

			#block constraints
			for j in range(problem.dim) :
				if plus[j] <= problem.lower_bounds[j]:
					plus[j] = problem.lower_bounds[j] + epsilon
				if plus[j] >= problem.upper_bounds[j]:
					plus[j] = problem.upper_bounds[j] - epsilon

			Y.append(plus)

		for i in range(self.d):
			minus = Y[0] - interpolation_radius * (U @ rotation_vectors[i])
			
			# block constraints
			for j in range(problem.dim):
				if minus[j] <= problem.lower_bounds[j]:
					minus[j] = problem.lower_bounds[j] + epsilon
				if minus[j] >= problem.upper_bounds[j]:
					minus[j] = problem.upper_bounds[j] - epsilon
			Y.append(minus)

		return Y 


	def construct_interpolation_set(self, problem: Problem, U: np.ndarray) -> tuple[list[np.ndarray], int] : 
		"""
			Constructs the interpolation set either by reusing one design point from the visited points list or not reusing any design points.
			This is the only method that is called to build the interpolation set.
		Args:
			current_solution (Solution): The current incumbent solution
			problem (Problem): The current simulation model
			U (np.ndarray): The (n,d) active subspace matrix
			delta (float): The current trust-region radius
			k (int): The current iteration number
			visited_pts_list (list[Solution]): The list of previously visited solutions

		Returns:
			tuple[list[np.ndarray], int]: A tuple containing the list of interpolation points and the index of the reused point
		"""
		x_k = np.array(self.incumbent_x).reshape(-1,1) #current solution as n-dim vector
		Dist = []
		for i in range(len(self.visited_points)):
			dist_of_pt = norm(U.T @ (np.array(self.visited_points[i].x).reshape(-1,1) - x_k))
			
			# If the design point is outside the trust region, then make sure it isn't considered for reuse
			if dist_of_pt <= self.delta:
				Dist.append(dist_of_pt) 
			else : 
				Dist.append(-1)

		# Find the index of visited design points list for reusing points
		# The reused point will be the farthest point from the center point among the design points within the trust region
		f_index = Dist.index(max(Dist))

		# If it is the first iteration or there is no design point we can reuse within the trust region, use the coordinate basis
		if (self.iteration_count == 1) or norm(x_k - np.array(self.visited_points[f_index].x).reshape(-1,1))==0 :
			Y = self.interpolation_points_without_reuse(problem, U)

		# Else if we will reuse one design point
		elif self.iteration_count > 1 :
			reused_pt = np.array(self.visited_points[f_index].x).reshape(-1,1)
			diff_array = U.T @ (reused_pt - x_k) #has shape (d,1)
			A_k1 = (diff_array) / norm(diff_array) #has shape (d,1)
			P_k = norm(diff_array)

			rotate_matrix: list[np.ndarray] = self.get_rotated_basis(A_k1, U)

			# construct the interpolation set
			Y = self.interpolation_points_with_reuse(problem, reused_pt, rotate_matrix, U)
		return np.vstack([v.ravel() for v in Y]), f_index

	#! === GEOMETRY IMPROVEMENT === 
	def generate_set(self, problem: Problem, num: int) -> np.ndarray:
		"""
			Generates a set of points around the current solution within the trust region
		Args:
			problem (Problem): The current simulation model
			num (int): The number of points to generate
			current_solution (Solution): The current incumbent solution
			delta (float): The current trust-region radius

		Returns:
			np.ndarray: A set of points around the current solution within the trust region
		"""
		x_k = np.array(self.incumbent_x).reshape(-1,1)


		bounds_l = np.maximum(np.array(problem.lower_bounds).reshape(x_k.shape), x_k-self.delta)
		bounds_u = np.minimum(np.array(problem.upper_bounds).reshape(x_k.shape), x_k+self.delta)
		direcs = self.coordinate_directions(problem, num, bounds_l-x_k, bounds_u-x_k)

		S = np.zeros((num, problem.dim))
		S[0, :] = x_k.flatten()
		bounds_l_flat = bounds_l.flatten()
		bounds_u_flat = bounds_u.flatten()
		x_k_flat = x_k.flatten()
		for i in range(1, num):
			S[i, :] = x_k_flat + np.minimum(np.maximum(bounds_l_flat-x_k_flat, direcs[i, :]), bounds_u_flat-x_k_flat)

		return S #shape (num, n)
	
	def get_scale(self, dirn: list[float], lower: np.ndarray, upper: np.ndarray, scale: float | None = None) -> float:
		"""
		Calculates the scaling factor for a direction vector to ensure it stays within bounds
		Args:
			dirn (list[float]): The direction vector
			lower (np.ndarray): The lower bounds
			upper (np.ndarray): The upper bounds
			scale (float, optional): An initial scaling factor. Defaults to None.

		Returns:
			float: The scaling factor
		"""
		scale = self.delta if scale is None else scale
		for j in range(len(dirn)):
			if dirn[j] < 0.0:
				scale = min(scale, lower[j] / dirn[j])
			elif dirn[j] > 0.0:
				scale = min(scale, upper[j] / dirn[j])
		return scale
	
	def coordinate_directions(self, problem: Problem, num_pnts: int, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
		"""
			Generates coordinate directions for the given problem
		Args:
			problem (Problem): The current simulation model
			num_pnts (int): The number of points to generate
			lower (np.ndarray): The lower bounds
			upper (np.ndarray): The upper bounds
		Returns:
			np.ndarray: Coordinate directions within the trust region
		"""
		n = problem.dim
		at_lower_boundary = (lower > -1.e-8 * self.delta)
		at_upper_boundary = (upper < 1.e-8 * self.delta)
		direcs = np.zeros((num_pnts, n))
		for i in range(1, num_pnts):
			if 1 <= i < n + 1:
				dirn = i - 1
				step = self.delta if not at_upper_boundary[dirn] else - self.delta
				direcs[i, dirn] = step
			elif n + 1 <= i < 2*n + 1:
				dirn = i - n - 1
				step = - self.delta
				if at_lower_boundary[dirn]:
					step = min(2.0* self.delta, upper[dirn])
				if at_upper_boundary[dirn]:
					step = max(-2.0* self.delta, lower[dirn])
				direcs[i, dirn] = step
			else:
				itemp = (i - n - 1) // n
				q = i - itemp*n - n
				p = q + itemp
				if p > n:
					p, q = q, p - n
				direcs[i, p-1] = direcs[p, p-1]
				direcs[i, q-1] = direcs[q, q-1]
		return direcs #shape (num_pnts, n)

	def random_directions(self, problem: Problem, num_pnts: int, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
		"""
			Generates random directions for the given problem
		Args:
			problem (Problem): The current simulation model
			num_pnts (int): The number of points to generate
			lower (np.ndarray): The lower bounds
			upper (np.ndarray): The upper bounds
			delta (float): The current trust-region radius

		Returns:
			np.ndarray: Random directions within the trust region
		"""
		n = problem.dim
		direcs = np.zeros((n, max(2*n+1, num_pnts)))
		idx_l = (lower == 0)
		idx_u = (upper == 0)
		active = np.logical_or(idx_l, idx_u)
		inactive = np.logical_not(active)
		nactive = np.sum(active)
		ninactive = n - nactive
		if ninactive > 0:
			A = np.random.normal(size=(ninactive, ninactive))
			Qred = qr(A)[0]
			Q = np.zeros((n, ninactive))
			Q[inactive, :] = Qred
			for i in range(ninactive):
				scale = self.get_scale(Q[:,i], lower, upper) 
				direcs[:, i] = scale * Q[:,i]
				scale = self.get_scale(-Q[:,i], lower, upper)
				direcs[:, n+i] = -scale * Q[:,i]
		idx_active = np.where(active)[0]
		for i in range(nactive):
			idx = idx_active[i]
			direcs[idx, ninactive+i] = 1.0 if idx_l[idx] else -1.0
			direcs[:, ninactive+i] = self.get_scale(direcs[:, ninactive+i], lower, upper) * direcs[:, ninactive+i]
			sign = 1.0 if idx_l[idx] else -1.0
			if upper[idx] - lower[idx] > self.delta:
				direcs[idx, n+ninactive+i] = 2.0*sign*self.delta
			else:
				direcs[idx, n+ninactive+i] = 0.5*sign*(upper[idx] - lower[idx])
			direcs[:, n+ninactive+i] = self.get_scale(direcs[:, n+ninactive+i], lower, upper, 1.0)*direcs[:, n+ninactive+i]
		for i in range(num_pnts - 2*n):
			dirn = np.random.normal(size=(n,))
			for j in range(nactive):
				idx = idx_active[j]
				sign = 1.0 if idx_l[idx] else -1.0
				if dirn[idx]*sign < 0.0:
					dirn[idx] *= -1.0
			dirn = dirn / norm(dirn)
			scale = self.get_scale(dirn, lower, upper)
			direcs[:, 2*n+i] = dirn * scale
		return np.vstack((np.zeros(n), direcs[:, :num_pnts].T)) #shape (num_pnts, n)
	

	def improve_geometry(self, problem: Problem, U: np.ndarray,  X: np.ndarray, fX: np.ndarray, interpolation_solutions: list[int]) -> tuple[np.ndarray, np.ndarray, list[Solution]]:
		"""
			Improves the geometry of the interpolation set by generating a sample set and performing LU pivoting.
			Works on the projected design set X @ U but returns the original design set X.
		Args:
			problem (Problem): The current simulation model
			U (np.ndarray): The current active subspace matrix (shape (n,d))
			X (np.ndarray): The current interpolation points (shape (M, n))
			fX (np.ndarray): The function values at the interpolation points (shape (M, 1))
			interpolation_solutions (list[int]): The list of interpolation solutions 

		Returns:
			tuple[np.ndarray, np.ndarray, list[Solution]]: 
															Updated interpolation points of shape (M, n),
															function values of shape (M, 1),, 
															interpolation solutions, 
		"""
		epsilon_1 = 0.5  
		dist = epsilon_1*self.delta
		x_k = np.array(self.incumbent_x).reshape(-1,1)
		
		# Project X to subspace for geometry check
		X_projected = X @ U  # shape (M, d)
		x_k_projected = x_k.T @ U  # shape (1, d)
		
		if max(norm(X_projected - x_k_projected, axis=1, ord=np.inf)) > dist:
			X, fX, interpolation_solutions = self.sample_set(problem, U, X, fX, interpolation_solutions)

		#build interpolation_sols 
		interpolation_solutions = [self.create_new_solution(tuple(x.ravel()), problem) for x in X]

		#add in visited points from the sample set function
		existing_x = set([obj.x for obj in self.visited_points])
		for obj in interpolation_solutions:
			if obj.x not in existing_x:
				self.visited_points.append(obj)

		return X, fX, interpolation_solutions
	
	def sample_set(self, problem: Problem, U: np.ndarray,
				X: np.ndarray, fX: np.ndarray, interpolation_solutions: list[Solution]) -> tuple[np.ndarray, np.ndarray, list[Solution]]:
		"""
			Generates a sample set around the current solution and performs LU pivoting to improve the interpolation set.
			Works on projected coordinates X @ U but maintains original full-dimensional points.
		Args:
			problem (Problem): The current simulation model
			current_f (float): The current objective function value
			U (np.ndarray): The current active subspace matrix (shape (n,d))
			X (np.ndarray): The current interpolation points (shape (M, n))
			fX (np.ndarray): The function values at the interpolation points (shape (M, 1))
			interpolation_solutions (list[Solution]): The list of interpolation solutions

		Returns:
			tuple[np.ndarray, np.ndarray, list[Solution]]: 
															Updated interpolation points of shape (M, n),
															function values of shape (M, 1),, 
															interpolation solutions, 
		"""
		epsilon_1 = 0.5  
		q = len(self.index_set(self.degree, self.d).astype(int)) 
		
		x_k = np.array(self.incumbent_x).reshape(-1,1)
		current_f = -1 * problem.minmax[0] * self.incumbent_solution.objectives_mean.item()
		dist = epsilon_1*self.delta

		# Work with original points but check distances in projected space
		X_improved = np.copy(X)
		f_hat = np.copy(fX)
		X_improved_projected = X_improved @ U
		x_k_projected = x_k.T @ U
		
		if max(norm(X_improved_projected - x_k_projected, axis=1, ord=np.inf)) > dist:
			X_improved, f_hat = self.remove_furthest_point_projected(X_improved, f_hat, x_k, U)
		X_improved, f_hat = self.remove_point_from_set_projected(X_improved, f_hat, x_k, U)
		
		X = np.zeros((q, problem.dim))
		fX = np.zeros((q, 1))
		X[0, :] = x_k.flatten()
		fX[0, :] = current_f
		X, fX, interpolation_solutions = self.LU_pivoting(problem, X, fX, X_improved, f_hat, U, interpolation_solutions)

		return X, fX, interpolation_solutions

	def LU_pivoting(self, problem: Problem, X: np.ndarray, fX: np.ndarray, X_improved: np.ndarray, fX_improved: np.ndarray, U: np.ndarray,
				interpolation_solutions: list[Solution]) -> tuple[np.ndarray, np.ndarray, list[Solution]] :
		"""
			Improves the interpolation set using LU pivoting.
		Args:
			problem (Problem): The current simulation model
			X (np.ndarray): The current interpolation points (shape (M, n))
			fX (np.ndarray): The function values at the interpolation points (shape (M, 1))
			X_improved (np.ndarray): The current sample set (shape (M, n))
			fX_improved (np.ndarray): The function values at the sample set points (shape (M, 1))
			U (np.ndarray): The current active subspace matrix (shape (n,d))
			interpolation_solutions (list[Solution]): The list of interpolation solutions

		Returns:
			tuple[np.ndarray, np.ndarray, list[Solution]]: 
															Updated interpolation points of shape (M, n),
															function values of shape (M, 1),
															interpolation solutions, 
		"""
		# Less aggressive pivot thresholds - prefer reusing good existing points
		psi_1 = 0.05  # Accept pivots >= 0.05 (was 0.1)
		psi_2 = 0.5   # Last pivot >= 0.5 (was 1.0)
		x_k = np.array(self.incumbent_x).reshape(-1,1)

		phi_function, phi_function_deriv = self.get_phi_function_and_derivative(U)
		q = len(self.index_set(self.degree, self.d).astype(int))
		p = X.shape[0]

		#Initialise R matrix of LU factorisation of M matrix (see Conn et al.)
		R = np.zeros((p,q))
		R[0,:] = phi_function(x_k)

		#Perform the LU factorisation algorithm for the rest of the points
		for k in range(1, q):
			flag = True
			v = np.zeros(q)
			for j in range(k):
				v[j] = -R[j,k] / R[j,j]
			v[k] = 1.0

			#If there are still points to choose from, find if points meet criterion. If so, use the index to choose 
			#point with given index to be next point in regression/interpolation set
			if fX_improved.size > 0:
				Phi_X_improved = np.vstack([phi_function(X_improved[i, :].reshape(-1, 1)) for i in range(X_improved.shape[0])])
				M = np.absolute(Phi_X_improved @ v)
				index = np.argmax(M)
				if M[index] < psi_1:
					flag = False
				elif (k == q - 1 and M[index] < psi_2):
					flag = False
			else:
				flag = False
			
			#If index exists, choose the point with that index and delete it from possible choices
			if flag:
				s = X_improved[index,:]
				X[k, :] = s
				fX[k, :] = fX_improved[index]
				X_improved = np.delete(X_improved, index, 0)
				fX_improved = np.delete(fX_improved, index, 0)

			#If index doesn't exist, solve an optimisation problem to find the point in the range which best satisfies criterion
			else:
				try:
					s = self.find_new_point(problem, v, phi_function, phi_function_deriv)
					# Check for duplicates in projected space with tolerance
					X_proj_k = X[:k, :] @ U
					s_proj = (s.reshape(1, -1) @ U).ravel()
					# Check if s_proj is too close to any existing point in projected space
					min_dist = np.min(norm(X_proj_k - s_proj, axis=1)) if k > 0 else np.inf
					# Points should be separated by at least 1% of delta in projected space
					if min_dist < 0.01 * self.delta:
						s = self.find_new_point_alternative(problem, v, phi_function, X[:k, :], U)
				except:
					s = self.find_new_point_alternative(problem, v, phi_function, X[:k, :], U)
				if fX_improved.size > 0 and M[index] >= abs(np.dot(v, phi_function(s))):
					s = X_improved[index,:]
					X[k, :] = s
					fX[k, 0] = float(np.asarray(fX_improved[index]).item())
					X_improved = np.delete(X_improved, index, 0)
					fX_improved = np.delete(fX_improved, index, 0)
				else:
					X[k, :] = s
					soln_at_s = self.create_new_solution(tuple(s.ravel()), problem)
					# Sample the newly generated interpolation point without re-evaluating the full design set
					soln_at_s = self.adaptive_sampling_after(problem, soln_at_s)
					f_value = -1 * problem.minmax[0] * soln_at_s.objectives_mean.item()
					fX[k, 0] = f_value
					if all(tuple(pt.x) != tuple(soln_at_s.x) for pt in self.visited_points):
						self.visited_points.append(soln_at_s)
					interpolation_solutions.append(soln_at_s)
			
			#Update R factorisation in LU algorithm
			phi = phi_function(s)
			R[k,k] = np.dot(v, phi)
			
			# Check if pivot is too small (would cause poor conditioning)
			# Require full psi_1 for all points except last which needs psi_2
			min_pivot = psi_2 if k == q-1 else psi_1
			if abs(R[k,k]) < min_pivot:
				# Try to find a better point if pivot is too small
				try:
					s_backup = self.find_new_point_alternative(problem, v, phi_function, X[:k, :], U)
					phi_backup = phi_function(s_backup)
					R_backup = np.dot(v, phi_backup)
					if abs(R_backup) > abs(R[k,k]):
						s = s_backup
						phi = phi_backup
						R[k,k] = R_backup
						# Update X[k] if we changed the point
						X[k, :] = s
						soln_at_s = self.create_new_solution(tuple(s.ravel()), problem)
						soln_at_s = self.adaptive_sampling_after(problem, soln_at_s)
						f_value = -1 * problem.minmax[0] * soln_at_s.objectives_mean.item()
						fX[k, 0] = f_value
						if all(tuple(pt.x) != tuple(soln_at_s.x) for pt in self.visited_points):
							self.visited_points.append(soln_at_s)
						if k < len(interpolation_solutions):
							interpolation_solutions[k] = soln_at_s
						else:
							interpolation_solutions.append(soln_at_s)
				except:
					pass  # Keep original point if backup fails
			
			for i in range(k+1,q):
				R[k,i] += phi[i]
				for j in range(k):
					R[k,i] -= (phi[j]*R[j,i]) / R[j,j]

		return X, fX, interpolation_solutions
	
	def getTotalOrderBasisRecursion(self, highest_order: int, dimensions: int) -> np.ndarray:
		"""
			Generates the total order basis recursively.
		Args:
			highest_order (int): The highest polynomial order
			dimensions (int): The number of dimensions

		Returns:
			np.ndarray: The total order basis of shape (L, dimensions) where L is the cardinality
		"""
		if dimensions == 1:
			I = np.zeros((1,1))
			I[0,0] = highest_order
		else:
			for j in range(0, highest_order + 1):
				U = self.getTotalOrderBasisRecursion(highest_order - j, dimensions - 1)
				rows, cols = U.shape
				T = np.zeros((rows, cols + 1) ) # allocate space!
				T[:,0] = j * np.ones((1, rows))
				T[:, 1: cols+1] = U
				if j == 0:
					I = T
				elif j >= 0:
					rows_I, cols_I = I.shape
					rows_T, cols_T = T.shape
					Itemp = np.zeros((rows_I + rows_T, cols_I))
					Itemp[0:rows_I,:] = I
					Itemp[rows_I : rows_I + rows_T, :] = T
					I = Itemp
				del T
		return I
	
	def get_basis(self, orders: np.ndarray) -> np.ndarray: 
		"""
			Generates the total order basis for the given orders.
		Args:
			orders (np.ndarray): The orders for each dimension

		Raises:
			Exception: If the cardinality is too large

		Returns:
			np.ndarray: The total order basis of shape (L, dimensions) where L is the cardinality
		"""
		dimensions = len(orders)
		highest_order = np.max(orders)
		# Check what the cardinality will be, stop if too large!
		L = int(math.factorial(highest_order+dimensions)/(math.factorial(highest_order)*math.factorial(dimensions)))
		# Check cardinality
		if L >= int(1e6):
			raise Exception('Cardinality %.1e is >= hard cardinality limit %.1e' %(L,int(1e6)))
		# Generate basis
		total_order = np.zeros((1, dimensions))
		for i in range(1, highest_order+1):
			R = self.getTotalOrderBasisRecursion(i, dimensions)
			total_order = np.vstack((total_order, R))
		return total_order 
	
	def get_phi_function_and_derivative(self, U: np.ndarray) -> tuple[callable, callable]:
		"""
			Generates the phi function and its derivative for the given sample set.
		Args:
			U (np.ndarray): The active subspace matrix (shape (n,d))

		Returns:
			tuple[callable, callable]: The phi function and its derivative
		"""
		q = len(self.index_set(self.degree, self.d).astype(int))
		x_k = np.asarray(self.incumbent_x).ravel()   

		total_order_index_set = self.get_basis(np.tile([2], q))[:, range(self.d-1, -1, -1)]

		# if S_hat.size > 0:
		# 	delta = max(norm(np.dot(S_hat - s_old, U), axis=1))

		def phi_function(s: np.ndarray) -> np.ndarray:
			s = s.ravel()    # shape (d,)
			u = np.dot(s - x_k, U) / self.delta
			u = np.atleast_2d(u)  
			m = u.shape[0]

			phi = np.zeros((m, q))
			for k in range(q):
				exponents = total_order_index_set[k, :]
				numerator = np.power(u, exponents)
				denom = np.array([factorial(int(e)) for e in exponents])
				phi[:, k] = np.prod(numerator / denom, axis=1)
			if phi.shape[0] == 1:
				return phi.ravel()
			return phi               

		def phi_function_deriv(s: np.ndarray) -> np.ndarray:
			s = s.ravel()  
			u = np.dot(s - x_k, U) / self.delta
			phi_deriv = np.zeros((self.d, q))
			for i in range(self.d):
				for k in range(1, q):  
					exponent = total_order_index_set[k, i]
					if exponent != 0:
						tmp = np.zeros(self.d, dtype=int)
						tmp[i] = 1
						exps_minus_tmp = total_order_index_set[k, :] - tmp
						numerator = np.prod(np.divide(np.power(u, exps_minus_tmp), [factorial(int(e)) for e in total_order_index_set[k, :]]))
						phi_deriv[i, k] = exponent * numerator
			phi_deriv = phi_deriv / self.delta   
			return np.dot(U, phi_deriv)

		return phi_function, phi_function_deriv
	
	def find_new_point(self, problem: Problem, v: np.ndarray, phi_function: callable, phi_function_deriv: callable) -> np.ndarray:
		"""
			Finds a new point in the trust region that maximizes the absolute value of the dot product with the phi function.
		Args:
			problem (Problem): The current simulation model
			v (np.ndarray): The direction vector of shape (q,1)
			phi_function (callable): The phi function
			phi_function_deriv (callable): The derivative of the phi function

		Returns:
			np.ndarray: The new point in the trust region
		"""
		x_k = np.array(self.incumbent_x).reshape(-1,1)	
		bounds_l = np.maximum(np.array(problem.lower_bounds).reshape(x_k.shape), x_k-self.delta)
		bounds_u = np.minimum(np.array(problem.upper_bounds).reshape(x_k.shape), x_k+self.delta)

		bounds = []
		for i in range(problem.dim):
			bounds.append((bounds_l[i], bounds_u[i])) 
		
		obj1 = lambda s: np.dot(v, phi_function(s))
		jac1 = lambda s: np.dot(phi_function_deriv(s), v)
		obj2 = lambda s: -np.dot(v, phi_function(s))
		jac2 = lambda s: -np.dot(phi_function_deriv(s), v)
		res1 = minimize(obj1, x_k, method='TNC', jac=jac1, \
				bounds=bounds, options={'disp': False})
		res2 = minimize(obj2, x_k, method='TNC', jac=jac2, \
				bounds=bounds, options={'disp': False})
		if abs(res1['fun']) > abs(res2['fun']):
			s = res1['x']
		else:
			s = res2['x']
		return s
	
	def generate_set_in_subspace(self, problem: Problem, U: np.ndarray, num: int) -> np.ndarray:
		"""
			Generates a set of points with good geometry in the projected subspace.
			Points are generated in the active subspace and then lifted to full space.
			Uses coordinate directions first, then random directions for better coverage.
		Args:
			problem (Problem): The current simulation model
			U (np.ndarray): The active subspace matrix (shape (n, d))
			num (int): The number of points to generate

		Returns:
			np.ndarray: A set of points with good geometry in projected space (shape (num, n))
		"""
		x_k = np.array(self.incumbent_x).reshape(-1, 1)
		d = U.shape[1]  # Get subspace dimension from U
		
		# Generate directions in the d-dimensional subspace
		S = np.zeros((num, problem.dim))
		S[0, :] = x_k.flatten()
		
		idx = 1
		# First, add coordinate directions in the subspace (2d points)
		for j in range(min(d, num-1)):
			for sign in [1, -1]:
				if idx >= num:
					break
				y = np.zeros(d)
				y[j] = sign * self.delta
				
				# Lift to full space: x_k + U @ y
				s = x_k.flatten() + (U @ y)
				
				# Project back to feasible region
				bounds_l = np.array(problem.lower_bounds)
				bounds_u = np.array(problem.upper_bounds)
				s = np.maximum(bounds_l, np.minimum(bounds_u, s))
				
				S[idx, :] = s
				idx += 1
		
		# Fill remaining with random directions for better coverage
		for i in range(idx, num):
			# Generate random direction in subspace
			y = np.random.randn(d)
			y = y / norm(y)  # Normalize
			
			# Scale by delta (use varying scales for diversity)
			scale = self.delta * (0.5 + 0.5 * np.random.rand())
			y = y * scale
			
			# Lift to full space: x_k + U @ y
			s = x_k.flatten() + (U @ y)
			
			# Project back to feasible region
			bounds_l = np.array(problem.lower_bounds)
			bounds_u = np.array(problem.upper_bounds)
			s = np.maximum(bounds_l, np.minimum(bounds_u, s))
			
			S[i, :] = s
		
		return S

	def find_new_point_alternative(self, problem: Problem, v: np.ndarray, phi_function: callable, X: np.ndarray, U: np.ndarray) -> np.ndarray:
		"""
			Finds a new point in the trust region by generating a sample set and selecting the point that maximizes the 
			absolute value of the dot product with the phi function.
			Checks for duplicates in the projected space to ensure good geometry.
		Args:
			problem (Problem): The current simulation model
			v (np.ndarray): The direction vector
			phi_function (callable): The phi function
			X (np.ndarray): The current sample set (shape (k, n))
			U (np.ndarray): The active subspace matrix (shape (n, d))

		Returns:
			np.ndarray: The new point in the trust region
		"""
		no_pts = max(int(0.5*self.d*(self.d+2)), 2*self.d+1, 20)  # Generate enough points in subspace
		# d = U.shape[1]
		# no_pts = max(int(d*(d+2)), 4*d+1, 50)  # At least 50 points, more for higher dimensions
		# Generate points with good geometry in the projected subspace
		X_tmp = self.generate_set_in_subspace(problem, U, no_pts)
		Phi_X_improved = np.vstack([phi_function(X_tmp[i, :].reshape(-1, 1)) for i in range(X_tmp.shape[0])])
		M = np.absolute(Phi_X_improved @ v)
		indices = np.argsort(M)[::-1][:len(M)]
		# Check for duplicates in projected space with tolerance
		X_proj = X @ U
		for index in indices:
			s = X_tmp[index,:]
			s_proj = (s.reshape(1, -1) @ U).ravel()
			# Check if s_proj is sufficiently far from all existing points
			# Points should be separated by at least 1% of delta in projected space
			min_dist = np.min(norm(X_proj - s_proj, axis=1)) if X_proj.shape[0] > 0 else np.inf
			if min_dist >= 0.01 * self.delta:
				return s
		return X_tmp[indices[0], :]
	
	def remove_point_from_set(self, X: np.ndarray, fX: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		"""
			Removes the current solution from the sample set.
		Args:
			X (np.ndarray): The current sample set
			f (np.ndarray): The function values corresponding to the sample set
			x (np.ndarray): The current solution to be removed

		Returns:
			tuple[np.ndarray, np.ndarray]: The updated sample set and function values after removal
		"""
		ind_current = np.where(norm(X-x.ravel(), axis=1, ord=np.inf) == 0.0)[0]
		X = np.delete(X, ind_current, 0)
		fX = np.delete(fX, ind_current, 0)
		return X, fX

	def remove_point_from_set_projected(self, X: np.ndarray, fX: np.ndarray, x: np.ndarray, U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		"""
			Removes the current solution from the sample set based on projected coordinates.
		Args:
			X (np.ndarray): The current sample set (shape (M, n))
			fX (np.ndarray): The function values corresponding to the sample set
			x (np.ndarray): The current solution to be removed (shape (n, 1))
			U (np.ndarray): The active subspace matrix (shape (n, d))

		Returns:
			tuple[np.ndarray, np.ndarray]: The updated sample set and function values after removal
		"""
		X_projected = X @ U
		x_projected = x.T @ U
		ind_current = np.where(norm(X_projected - x_projected, axis=1, ord=np.inf) == 0.0)[0]
		X = np.delete(X, ind_current, 0)
		fX = np.delete(fX, ind_current, 0)
		return X, fX

	
	def remove_furthest_point(self, X: np.ndarray, fX: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		"""
			Removes the furthest point from the current solution in the sample set.
		Args:
			X (np.ndarray): The current sample set
			fX (np.ndarray): The function values corresponding to the sample set
			x (np.ndarray): The current solution to be removed
		Returns:
			tuple[np.ndarray, np.ndarray]: The updated sample set and function values after removal
		"""
		ind_distant = np.argmax(norm(X-x.ravel(), axis=1, ord=np.inf))
		X = np.delete(X, ind_distant, 0)
		fX = np.delete(fX, ind_distant, 0)
		return X, fX

	def remove_furthest_point_projected(self, X: np.ndarray, fX: np.ndarray, x: np.ndarray, U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		"""
			Removes the furthest point from the current solution in the sample set based on projected coordinates.
		Args:
			X (np.ndarray): The current sample set (shape (M, n))
			fX (np.ndarray): The function values corresponding to the sample set
			x (np.ndarray): The current solution to be removed (shape (n, 1))
			U (np.ndarray): The active subspace matrix (shape (n, d))
		Returns:
			tuple[np.ndarray, np.ndarray]: The updated sample set and function values after removal
		"""
		X_projected = X @ U
		x_projected = x.T @ U
		ind_distant = np.argmax(norm(X_projected - x_projected, axis=1, ord=np.inf))
		X = np.delete(X, ind_distant, 0)
		fX = np.delete(fX, ind_distant, 0)
		return X, fX



	#! === MODEL CONSTRUCTION === 

	def construct_model(
			self, problem: Problem) -> tuple[
				callable, callable, np.ndarray, list[float], list[Solution], np.ndarray, np.ndarray]: 
		"""
			Builds a local approximation of the response surface within the current trust region (defined as ||x-x_k||<=delta).
			The method fit recovers the local approximation given a design set of 2d+1 design points and a corresponding active subspace U of shape (n,d)
			That projects the n-dimensional design points to a d-dimensional subspace.

		Args:
			problem (Problem): The SO problem with dimension n

		Returns:
			tuple[callable, callable, np.ndarray, list[float], list[Solution]]:
				- The local model as a function that takes a numpy vector of shape (n,1) and returns a float
				- The local model gradient as a function that takes a numpy vector of shape (n,1) and returns gradient
				- The final computed active subspace of the iteration of shape (n,d)
				- A list of the function estimates of the objective function at each of the final design points
				- The list of solutions of the final design points 
		"""
		init_S_full = self.generate_set(problem, self.d)
		U, _ = np.linalg.qr(init_S_full.T)

		X, f_index = self.construct_interpolation_set(problem, U)

		fX, interpolation_solutions = self.evaluate_interpolation_points(problem, f_index, X)
		interpolation_solutions = [self.create_new_solution(tuple(x.ravel()), problem) for x in X]

		fval = fX.flatten().tolist()


		U, model, model_grad, X, fX, interpolation_solutions = self.fit(problem, X, fX, interpolation_solutions, U) 

		fval = fX.flatten().tolist()

		return model, model_grad, U, fval, interpolation_solutions, X, fX
	
	def model_evaluate(self, x_proj: np.ndarray, coef: np.ndarray, U: np.ndarray) -> float :
		"""
			Evaluates the local approximated model at a given design point.

		Args:
			x (np.ndarray): Design point to evaluate of shape (d,1)
			delta (float): radius of the current trust-region
			coef (np.ndarray): The coefficients of the local model of shape (q,1)
			U (np.ndarray): The active subspace matrix of shape (n,d)

		Returns:
			float: The evaluation of the model at x, given as m(U^Tx)
		"""    

		if len(x_proj.shape) != 2 or x_proj.shape[1] != 1 : 
			x_proj = x_proj.reshape(-1,1)

		if x_proj.shape[0] == U.shape[0] :
			#project x to active subspace 
			x_proj = U.T @ x_proj #(d,1)

		if len(coef.shape) != 2 : 
			coef = coef.reshape(-1,1)
		

		#build vandermonde matrix of shape (1,q)
		V_matrix = self.V(x_proj.T) #(1,q)


		#find evaluation: 
		res = V_matrix @ coef 

		return res.item()
	
	def fit(
		self, problem:Problem, X:np.ndarray, fX:np.ndarray, interpolation_solutions:list[Solution], U0: np.ndarray
		) -> tuple[
			np.ndarray, callable, callable, np.ndarray, np.ndarray, list[Solution]
			] : 
		"""
			Fits the design set and evaluated points to a local model with recovered active subspace. 
			It undergoes a loop until the active subspace converges. 
			First, it improves the design set X and constructing an interpolation model until it can ensure the criticality step is satisfied. 
			Second, after fixing the model coefficients, it updates the active subspace through a variable projection scheme
		Args:
			problem (Problem): The SO problem with dimension n
			X (np.ndarray): design set of shape (M,n)
			fX (np.ndarray): corresponding function estimates of design points of shape (M,1)
			interpolation_solutions (list[Solution]): A list of the design points in the 
			U0 (np.ndarray): The initial estimate for the active subspace of shape (n,d)

		Returns:
			tuple[np.ndarray, callable, callable, np.ndarray, np.ndarray, list[Solution]]:  
				- The final computed active subspace of the iteration of shape (n,d)
				- The local model as a function that takes a numpy vector of shape (n,1) and returns a float (this is given as model = lambda x : self.model_evaluate(x,coef, U))
				- The local model gradient as a function that takes a numpy vector of shape (n,1) and returns a float (this is given as model_grad = lambda x : self.model_evaluate(x,U))
				- The design set after going through fitting of shape (M,n)
				- The function estimates of the objective function at each of the final design points of shape (M,1)
				- The design points as solution objects 
		"""
		#Algorithmic Parameters 
		beta = 10

		# Orthogonalize just to make sure the starting value satisfies constraints	
		U0, R = np.linalg.qr(U0, mode = 'reduced') 
		U0 = np.dot(U0, np.diag(np.sign(np.diag(R))))

		prev_U = np.zeros(U0.shape)
		U = np.copy(U0)
		model_delta = float(self.delta)

		if self.degree == 1 and U.shape[1] == 1 : 
			V_matrix = np.hstack((np.ones((X.shape[0],1)), X)) #(M, n+1)
			fn_coef = pinv(V_matrix) @ fX #(n+1,1)
			fn_grad = fn_coef[1:, :].reshape(-1,1) #(n,1)
			U = fn_grad / norm(fn_grad)


		else : 
			i = 0
			while True : #not self.converged_subspace_check(prev_U, U) : 
				subspace_tol = min(1e-2, 0.5*max(model_delta, 1e-8))
				if self.converged_subspace_check(prev_U, U, tol=1e-3) :
					break
				#* Construct model and Criticality step 
				coef, model_delta, X, fX, interpolation_solutions, = self.criticality_check(problem, X, fX, U, interpolation_solutions)

				#set the old U and update
				prev_U = np.copy(U)
				U = self.fit_varpro(X, fX, U)
				i += 1

		coef = self.fit_coef(X,fX,U)

		#final fitting of the coefficients and rotating the final U 
		U = self.rotate_U(X, fX, coef, U)
		coef = self.fit_coef(X,fX,U)
		
		model = lambda x : self.model_evaluate(x, coef, U) #returns a float
		model_grad = lambda x : self.grad(x, coef, U) # returns np.ndarray of shape (d,1)

		if self.delta != model_delta :
			self.delta = min( max(self.delta, beta*norm(self.grad(X,coef,U))), model_delta)

		return  U, model, model_grad, X, fX, interpolation_solutions

	def criticality_check(self, problem: Problem,  X: np.ndarray, fX: np.ndarray, U: np.ndarray, interpolation_solutions: list[Solution]) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, list[Solution]]:
		"""
			Performs the criticality step of the trust-region method. 
			It fits a local model to the design set and checks whether the criticality condition is satisfied.
		Args:
			problem (Problem): The SO Problem
			X (np.ndarray): The design set of shape (M,n)
			fX (np.ndarray): The corresponding function estimates of shape (M,1)
			U (np.ndarray): The current active subspace matrix
			interpolation_solutions (list[Solution]): The list of solutions in the interpolation set

		Returns:
			tuple[np.ndarray, float, np.ndarray, np.ndarray, list[Solution]]: 
				- The model coefficients of shape (q,1)
				- The trust-region radius after criticality check
				- The design set after criticality check of shape (M,n)
				- The function estimates of the objective function at each of the final design points of shape (M,1)
				- The design points as solution objects 
		"""
		w: float = 0.85
		tol: float = 1e-6
		kappa_f: float = 10.0
		kappa_g: float = 10.0
		coef: np.ndarray | None = None 

		model_delta = float(self.delta)

		#* Construct model and Criticality step 
		fitting_iter = 0
		while True:
			coef = self.fit_coef(X, fX, U)
			grad =  self.grad(X, coef, U) @ U.T  #(M,n)
			gnorm = norm(grad)

			if gnorm <= tol:
				if self.fully_linear_test(X, fX, coef, U, kappa_f, kappa_g):
					break
				X, fX, interpolation_solutions = self.improve_geometry(problem, U, X, fX, interpolation_solutions)
				model_delta = self.delta * w**fitting_iter
				fitting_iter += 1
				continue

			elif model_delta >  max(self.mu * gnorm, 1e-12):
				model_delta = min(model_delta, max(self.delta_min, self.mu * gnorm, 1e-12))
				X, fX, interpolation_solutions = self.improve_geometry(problem, U, X, fX, interpolation_solutions)
				fitting_iter += 1
				continue
			else:
				break

		return coef, model_delta, X, fX, interpolation_solutions

	def converged_subspace_check(self, prev_U: np.ndarray, U: np.ndarray, tol: float) -> bool :
		"""
			Check whether the active subspace has converged by computing the subspace 
			distance between previous and current subspace estimates

		Args:
			prev_U (np.ndarray): Active subspace matrix from previous iteration (shape (n,d))
			U (np.ndarray): Active subspace from current iteration (shape (n,d))
			tol (float, optional): Convergence tolerance for subspace distance.

		Returns:
			bool: Returns True is subspace change is below tolerance and False otherwise
		"""	
		if np.all(prev_U == 0) :
			return False

		C = prev_U.T @ U  # shape (d, d)
		# Singular values of C are cos(theta_i)
		sigma = np.linalg.svd(C, compute_uv=False)
		sigma = np.clip(sigma, -1.0, 1.0)
		# Compute principal angles and distance
		sin_theta = np.sqrt(1.0 - sigma**2)
		subspace_dist = float(np.max(sin_theta))  # operator norm of projector difference

		converged = subspace_dist <= tol
		return converged

	def fully_linear_test(self, X: np.ndarray, fX: np.ndarray, coef: np.ndarray, U: np.ndarray, kappa_f: float, kappa_g: float) -> bool : 
		"""
			Check whether a model is fully linear in a trust region, using function residuals and model gradient consistency
		Args:
			X (np.ndarray): Design set of points (shape (M,n))
			fX (np.ndarray): Corresponding function estimation of design points (shape (M,1))
			coef (np.ndarray): Model coefficients (shape (q,1))
			U (np.ndarray): Active subspace matrix (shape (n,d))
			kappa_f (float): Tolerance of zeroth-order fully-linear bound
			kappa_g (float): Tolerance of first-order fully-linear bound

		Returns:
			bool: True if the model is fully-linear and False otherwise.
		"""
		M, n = X.shape

		# --- 1. Value-based condition ---
		mX = np.array([self.model_evaluate(U.T @ np.array(x).reshape(-1,1), coef, U) for x in X]).reshape(-1,1)
		residuals = np.abs(fX - mX)
		value_condition = np.max(residuals) <= kappa_f * self.delta**2

		# --- 2. Gradient consistency condition ---
		m_grads = self.grad(X, coef, U) #shape (M, d)
		consistent = True

		for i in range(M):
			for j in range(i + 1, M):
				dx = X[i, :] - X[j, :] #shape (n,)
				dm = mX[i, 0] - mX[j, 0]
				g_j = (U @ m_grads[j, :].reshape(-1,1)).flatten() #shape (n,)
				linearized_diff = np.dot(g_j, dx)
				model_error = np.abs(dm - linearized_diff)
				if model_error > kappa_g * np.linalg.norm(dx)**2:
					consistent = False
					break
			if not consistent:
				break

		# --- Fully linear if both conditions hold ---
		return bool(value_condition and consistent)

	def fit_coef(self, X:np.ndarray, fX: np.ndarray, U: np.ndarray) -> np.ndarray : 
		"""
			Finds the coefficients of the interpolation model by solving the system of equations:
					V(U^TX)coeff = fX
		Args:
			X (np.ndarray): The design set of design points of shape (M,n)
			fX (np.ndarray): corresponding function estimates of design points of shape (M,1)
			U (np.ndarray): The active subspace of shape (n,d)
			delta (float): radius of the current trust-region

		Returns:
			np.ndarray: A list of the coefficients of shape (q,1)
		"""
		Y = X @ U
		V_matrix = self.V(Y) #shape (M,q)
		coef = pinv(V_matrix) @ fX  #(q,1)

		if len(coef.shape) != 2 : 
			coef = coef.reshape(-1,1)

		return coef 
	
	def grassmann_trajectory(self, U: np.ndarray, Delta: np.ndarray, t: float) -> np.ndarray:
		"""
			Calculates the geodesic along the Grassmann manifold 
		Args:
			U (np.ndarray): The active subspace matrix of shape (n,d)
			Delta (np.ndarray): The search direction along the Grassmann manifold with shape (n,d)
			t (float): Independent parameter in the line equation takes values between (0,infty) and is selected to ensure convergence. 

		Returns:
			np.ndarray: The new candidate for the active subspace based on the step made of shape (n,d)
		"""
		Y, sig, ZT = scipy.linalg.svd(Delta, full_matrices = False, lapack_driver = 'gesvd')

		UZ = np.dot(U, ZT.T)
		U_new = np.dot(UZ, np.diag(np.cos(sig*t))) + np.dot(Y, np.diag(np.sin(sig*t)))

		#Correct the new step U by ensuring it is orthonormal with consistent sign on the elements
		U_new, R = np.linalg.qr(U_new, mode = 'reduced') 
		U_new = np.dot(U_new, np.diag(np.sign(np.diag(R))))
		return U_new
	

	def residual(self, X: np.ndarray, fX: np.ndarray, U: np.ndarray) -> np.ndarray:
		"""
			Construct the Residual of the model fitting, such that 
			r = fX - V(U^TX)coeff
		Args:
			X (np.ndarray): The design set of design points of shape (M,n)
			fX (np.ndarray): The corresponding function estimates of design points of shape (M,1) 
			U (np.ndarray): The active subspace of shape (n,d)
			delta (float): radius of the current trust-region

		Returns:
			np.ndarray: The residual error for each design point on the local model of shape (M,1)
		"""
		c = self.fit_coef(X,fX,U) #shape(q,1)
		model_fX = np.array([self.model_evaluate(U.T @ np.array(x).reshape(-1,1), c, U) for x in X]).reshape(-1,1) #A list of length M with float elements
		r = fX - model_fX
		return r

	#! THIS NEEDS CHECKING OVER
	def jacobian(self, X: np.ndarray, fX:np.ndarray, U:np.ndarray) -> np.ndarray : 
		"""
			Constructs the Jacobian of the residual with respect to the active subspace
		Args:
			X (np.ndarray): The design set of design points of shape (M,n)
			fX (np.ndarray): The corresponding function estimates of design points of shape (M,1) 
			U (np.ndarray): The active subspace of shape (n,d)
			delta (float): radius of the current trust-region

		Returns:
			np.ndarray: A tensor of shape (M,n,d) where each element is the partial derivative of the i-th residual component with respect to the (j,k)th entry of the active subspace 
		"""
		
		#FIRST ENSURE THAT THE ARGUMENTS HAVE DIMENSIONS THAT MATCH
		assert X.shape[1] == U.shape[0], "X should have columns equal to the number of rows in U"
		assert X.shape[0] == fX.shape[0], "The number of samples in the design set X should match the number of function estimations in fX" 
		assert fX.shape[1] == 1, "The function estimates of the design set should be a column vector"
		
		#get dimensions
		M, n = X.shape

		#find the residual 
		Y = X @ U

		c = self.fit_coef(X,fX,U) #shape(q,1)
		r = self.residual(X,fX,U) #(M,1)

		#! FROM HERE THE FUNCTION NEEDS CHECKING
		#find the vandermonde matrix and derivative of the vandermonde matrix of the projected design set 
		V_matrix = self.V(Y) #shape (M,q)
		DV_matrix = self.DV(Y) #shape (M,q,n)
		

		M,q = V_matrix.shape

		Y, s, ZT = scipy.linalg.svd(V_matrix, full_matrices = False)
		# s = np.array([np.inf if x == 0.0 else x for x in s]) 
		with np.errstate(divide='ignore', invalid='ignore'):
			D = np.diag(1.0 / s)
			D[np.isinf(D)] = 0  # convert inf to 0 if desired

		J1 = np.zeros((M, n, self.d))
		J2 = np.zeros((q, n, self.d))

		# populate the Jacobian
		for k in range(self.d):
			for j in range(n):
				
				#This is the derivative of U  
				DVDU_k =  X[:,j,None]*DV_matrix[:,:,k] #shape (M,q)

				#first term in the Jacobian 
				J1[:, j, k] = DVDU_k.dot(c).flatten() #shape (M,)
				
				#second term of the Jacobian before V(U)^-
				J2[:, j, k] = DVDU_k.T.dot(r).flatten() #shape of (M,)

		# project J1 against the range of V
		J1 -= np.tensordot(Y, np.tensordot(Y.T, J1, axes=(1,0)), axes=(1,0))  # shape: (M,)

		# apply pseudo-inverse via SVD components
		J2_projected = np.tensordot(D, np.tensordot(ZT, J2, axes=(1,0)), axes=(1,0))  # shape: (q, n, d)

		# combine terms to get full Jacobian
		Jac = -(J1 + np.tensordot(Y, J2_projected, axes=(1,0)))  # shape: (M, n, d)

		
		return Jac
	

	def fit_varpro(self, X: np.ndarray, fX: np.ndarray, U: np.ndarray) -> np.ndarray : 
		"""
			Runs a Gauss-Newton

		Args:
			X (np.ndarray): design set of shape (M,n)
			fX (np.ndarray): corresponding function estimates of design points of shape (M,1)
			U (np.ndarray): The active subspace of shape (n,d)
			delta (float): radius of the current trust-region
			

		Returns:
			np.ndarray: The active subspace of shape (n,d)
		"""

		def gn_solver(Jac: np.ndarray, residual: np.ndarray) -> np.ndarray :
			"""
				An anonymous function to compute the Gauss-Newton step to find a descent direction
			Args:
				Jac (np.ndarray): The Jacobian of the residual with respect to the active subspace. It has shape (M,n,d)
				residual (np.ndarray): The residual of the current model approximation with shape (M,1)

			Returns:
				np.ndarray: A vectorised form of the descent direction with shape (nd,). The full descent direction has shape (n,d)
			""" 

			# Handle edge cases where residual or Jacobian are zero
			if np.all(residual == 0) and np.all(Jac == 0) :
				return np.zeros(Jac.shape[1]*Jac.shape[2])
			
			if np.all(Jac == 0) :
				raise ValueError("Jacobian is zero, cannot compute Gauss-Newton step.")
			
			if np.all(residual == 0) :
				return np.zeros(Jac.shape[1]*Jac.shape[2])
			
			M,n,d = Jac.shape
			Jac_vec = Jac.reshape(X.shape[0],-1) #reshapes (M,n,d) to (M,nd) 	
			
			#compute short form SVD
			Y, sig, ZT = scipy.linalg.svd(Jac_vec, full_matrices = False, lapack_driver = 'gesvd') #Y has shape (M,M), sig has shape (M,nd), and ZT has shape (nd,nd)
			
			# Find descent direction
			# Extract singular values from s (the diagonal)
			# sing_vals = np.diag(sig)  # shape (M,)
			tol = np.max(sig) * np.finfo(float).eps
			s_inv = np.where(sig > tol, 1.0 / sig, 0.0)

			# Compute Y^T r
			YTr = (Y.T @ residual).flatten()  # shape (M,)

			s_inv = 1.0 / sig  # shape (M,)

			Delta_vec = -ZT.T @ (s_inv * YTr)  # shape (n*d,)

			return Delta_vec

		jacobian_variable_U = lambda U : self.jacobian(X,fX,U)
		residual_variable_U = lambda U : self.residual(X,fX,U) 

		U = self.gauss_newton_solver(residual_variable_U, jacobian_variable_U, U, gn_solver) 

		return U 
	

	def gauss_newton_solver(self, residual: callable, jacobian: callable, U: np.ndarray, gn_solver: callable) -> np.ndarray : 
		"""
			Solves the Gauss_newton problem on the Grassmann manifold:
				vec(Delta) = -vec(Jac(U))^{+}r(U)

		Args:
			residual (callable): Function that takes the active subspace U of shape (n,d) and calculates the residual of the predicted model under a fixed design set. Returns a matrix of shape (M,1)
			jacobian (callable): Function that takes the active subspace U of shape (n,d) and calculates the Jacobian of the residual with respect to U. Returns a matrix of shape (M,n,d)
			U (np.ndarray): The subspace matrix with shape (n,d)
			gn_solver (callable): The Gauss-Newton step that returns the vectorised descent direction of shape (nd,)

		Returns:
			np.ndarray: A new active subspace matrix U_+ of shape (n,d)
		"""

		#initial values for res and Jac and Grad
		max_iter = 100
		res = residual(U) #shape (M,1)
		Jac = jacobian(U) #shape (M,n,d)
		Grad = np.tensordot(res.ravel(), Jac, axes=(0, 0))  # (n,d)

		if np.all(Jac == 0) and np.all(res == 0) :
			return U

		if np.all(Jac == 0) :
			raise ValueError("Jacobian is zero, cannot compute Gauss-Newton step.")

		if np.all(res == 0) :
			return U

		#Compute tolerances 
		Grad_norm = norm(Grad)
		tol = max(1e-10*Grad_norm, 1e-14)
		tol_Delta_norm = 1e-12

		#loop over linesearch until the norm of the gauss-newton step, the norm of the Grad or the norm of Res(U) increases
		for _ in range(max_iter) :
			residual_increased = False

			Jac_vec = Jac.reshape(Jac.shape[0], -1) #shape (M, nd)
			Delta_vec = gn_solver(Jac, res) #shape (nd,)
			Delta = Delta_vec.reshape(Jac.shape[1], Jac.shape[2]) #shape (n,d)

			# backtracking: find acceptable step gamma (t) along geodesic trajectory
			U_new, step = self.backtracking(residual, Grad, Delta, U)

			res_candidate = residual(U_new)
			Jac_candidate = jacobian(U_new)
			Grad_candidate = np.tensordot(res_candidate.ravel(), Jac_candidate, axes=(0, 0))


			if norm(res_candidate) >= norm(res) : 
				residual_increased = True 
			else :
				#Update the residual, jacobian, Gradient, and active subspace
				res = res_candidate 
				Jac = Jac_candidate
				Grad = Grad_candidate
				U = U_new 
				

			#Termination Conditions 
			if Grad_norm < tol or norm(Delta) < tol_Delta_norm or residual_increased : 
				return U_new 
			
		return U
	

	def backtracking(self, residual: callable, Grad: np.ndarray, delta: np.ndarray, U:np.ndarray) -> tuple[np.ndarray, float]  :
		"""
			Backtracking line search to satisfy the Armijo Condition:
				residual(U + alpha*delta) < residual(U) + alpha*beta*gamma
				where: 
					- alpha is <Grad, delta>
					- beta is a control parameter in (0,1)
					- gamma is the backtracking coefficient

		Args:
			residual (callable): Function that takes the active subspace U of shape (n,d) and calculates the residual of the predicted model under a fixed design set 
			Grad (np.ndarray): Gradient of the active subspace matrix on the Grassmann manifold of shape (n,d)
			delta (np.ndarray): The Gauss-Newton step of shaoe (n,d)
			U (np.ndarray): The active subspace matrix with shape (n,d)

		Returns:
			tuple[np.ndarray, float]: 
				- The new active subspace matrix U of shape (n,d)
				- The backtracking coefficient gamma  (gamma=1 implies no backtracking)
		"""	 
		#initialise control parameter, step shrink factor, and max iterations
		beta = 1e-4
		rho = 0.5
		max_iter = 100

		# directional derivative
		alpha = np.inner(Grad.reshape(-1,), delta.reshape(-1,))  # vecGrad^T vec(delta) in matrix form

		# If direction is not a descent direction, flip to negative gradient
		if alpha >= 0:
			delta = -Grad
			alpha = np.inner(Grad.reshape(-1,), delta.reshape(-1,))

		# starting objective and residual
		init_res = residual(U)

		step_size = 1.0
		for _ in range(max_iter):
			U_candidate = self.grassmann_trajectory(U, delta, step_size)
			res_candidate = residual(U_candidate)

			# Armijo condition: f(U + t delta) <= f(U) + t * beta * alpha
			if norm(res_candidate) <= norm(init_res) + step_size * beta * alpha:
				# success
				#Make sure U_new is orthonormal
				U_candidate, _ = np.linalg.qr(U_candidate)
				U_candidate = np.sign(np.diag(_)) * U_candidate  # ensure consistent orientation
				return U_candidate, step_size

			# otherwise shrink step
			step_size *= rho

		# if not found, return the best we have (the last candidate)
		U_candidate = self.grassmann_trajectory(U, delta, step_size)
		#Make sure U_new is orthonormal
		U_candidate, _ = np.linalg.qr(U_candidate)
		U_candidate = np.sign(np.diag(_)) * U_candidate  # ensure consistent orientation

		return U_candidate, step_size
	
	def rotate_U(self, X:np.ndarray, fX: np.ndarray, coef:np.ndarray, U:np.ndarray) -> np.ndarray : 
		"""
			Rotates the active subspace matrix onto the most important direction of 
		Args:
			X (np.ndarray): design set of shape (M,n)
			fX (np.ndarray): corresponding function estimates of design points of shape (M,1)
			coef (np.ndarray): The coefficients of the local model of shape (q,1)
			U (np.ndarray): The active subspace of shape (n,d)

		Returns:
			np.ndarray: The rotated active subspace matrix of shape (n,d)
		"""

		# Step 1: Apply active subspaces to the profile function at samples X
				# to rotate onto the most important directions
		if U.shape[1] > 1 :   
			grads = self.grad(X, coef, U)
			active_grads = grads 
			# We only need the short-form SVD
			Ur = scipy.linalg.svd(active_grads.T, full_matrices = False)[0]
			U = U @ Ur

		# Step 2: Flip signs such that average slope is positive in the coordinate directions
		coef = self.fit_coef(X, fX, U)
		grads = self.grad(X, coef, U)
		active_grads = grads #shape (M,d)
		U = U.dot(np.diag(np.sign(np.mean(active_grads, axis=0))))

		return U

	def grad(self, X: np.ndarray, coef: np.ndarray, U:np.ndarray) -> np.ndarray:
		"""
			Computes the gradients of the local model at each design point of the design set X 
		Args:
			X (np.ndarray): design set of shape (M,n) or (M,d)
			coef (np.ndarray): The coefficients of the local model of shape (q,1)
			U (np.ndarray): The active subspace of shape (n,d)
			delta (float): radius of the current trust-region

		Returns:
			np.ndarray: The gradients of the model at each design point X shape (M,d)
		"""
		if len(X.shape) == 1:
			one_d = True
			X = X.reshape(1,-1)	
		else:
			one_d = False	
		
		# Check if X is full-space (n dimensions) or reduced-space (d dimensions)
		if X.shape[1] == U.shape[0]:
			# Full-space input: project to reduced space
			Y = X @ U
		else:
			# Already in reduced space
			Y = X
			
		DV_matrix = self.DV(Y) #shape (M,q,d)
		# Compute gradient on projected space
		Df = np.tensordot(DV_matrix, coef, axes = (1,0)) #shape (M,d,1)
		# Inflate back to whole space
		Df = np.squeeze(Df, axis=-1) #shape (M,d)

		if one_d:
			return Df.reshape(Y.shape[1]) #shape (d,)
		else:
			return Df #shape (M,d)

	#! === VANDERMONDE CONSTRUCTION === 
	
	def scale(self, X: np.ndarray) -> np.ndarray:
		"""
			Scale the design points using the basis adapter if provided
		Args:
			X (np.ndarray): The design points to be scaled	

		Returns:
			np.ndarray: The scaled design points
		"""
		if self.basis_adapter is None:
			return X
		return self.basis_adapter.scale(X)

	def dscale(self, X: np.ndarray) -> np.ndarray:
		"""
			Scale the derivatives of the design points using the basis adapter if provided
		Args:
			X (np.ndarray): The design points whose derivatives are to be scaled

		Returns:
			np.ndarray: The scaled derivatives of the design points
		"""
		if self.basis_adapter is None:
			return np.ones_like(X)
		return self.basis_adapter.dscale(X)

	def V(self, X: np.ndarray) -> np.ndarray :
		"""
			Generate the Vandermonde Matrix
		Args:
			X (np.ndarray): The design set of shape (M,n) or (M,d) where n is the dimension of the original problem and d is the subspace dimension.
		Returns:
			np.ndarray: A vandermonde matrix of shape (M,q) where q is the length of the polynomial basis 
		"""
		M, d = X.shape
		X = self.scale(X)
		indices = self.index_set(self.degree, d).astype(int)
		M = X.shape[0]
		assert X.shape[1] == d, "Expected %d dimensions, got %d" % (d, X.shape[1])
		V_coordinate = [self.vander(X[:,k], self.degree) for k in range(d)]

		
		V = np.ones((M, len(indices)), dtype = X.dtype)
		
		for j, alpha in enumerate(indices):
			for k in range(d):
				V[:,j] *= V_coordinate[k][:,alpha[k]]
		return V
	
	def DV(self, X: np.ndarray) -> np.ndarray : 
		"""
			Column-wise derivative of the Vandermonde matrix

			Given design points this creates the Vandermonde-like matrix whose entries
			correspond to the derivatives of each of basis elements
		Args:
			X (np.ndarray): The design set of shape (M,d) where d is the subspace dimension.
		Returns:
			np.ndarray: Derivative of Vandermonde matrix  of shape (M,q,d) where DV[i,j,:] is the gradient of the
			partial derivative of the j-th basis function with respect to the x_k component of the d-dimensional vector 
			and evaluated at i-th design point
		"""
		M, d = X.shape
		X = self.scale(X)


		Dmat = self.build_Dmat()
		indices = self.index_set(self.degree, d).astype(int)

		V_coordinate = [self.vander(X[:,k], self.degree) for k in range(d)]
		
		N = len(indices)
		DV = np.ones((M, N, d), dtype = X.dtype)


		dscale = self.dscale(X)


		for k in range(d):
			for j, alpha in enumerate(indices):
				for q in range(d):
					if q == k:
						DV[:,j,k] *= np.dot(V_coordinate[q][:,0:-1], Dmat[alpha[q],:])
					else:
						DV[:,j,k] *= V_coordinate[q][:,alpha[q]]
				DV[:,j,k] *= dscale[:,k]

		return DV
	
	def build_Dmat(self) -> np.ndarray :
		"""
			Constructs the (scalar) derivative matrix for polynomial basis up to specified degree
		Returns:
			np.ndarray: The derivative matrix
		"""
		Dmat = np.zeros( (self.degree+1, self.degree))
		I = np.eye(self.degree + 1)
		for j in range(self.degree + 1):
			Dmat[j,:] = self.polyder(I[:,j])

		return Dmat 

	def full_index_set(self, n: int, d: int) -> np.ndarray:
		"""
			Enumerate multi-indices for a total degree of exactly `n` in `d` variables.
		Args:
			n (int): The total degree
			d (int): The number of variables

		Returns:
			np.ndarray: The multi-indices for the given total degree and number of variables
		"""
		if d == 1:
			I = np.array([[n]])
		else:
			II = self.full_index_set(n, d-1)
			m = II.shape[0]
			I = np.hstack((np.zeros((m, 1)), II))
			for i in range(1, n+1):
				II = self.full_index_set(n-i, d-1)
				m = II.shape[0]
				T = np.hstack((i*np.ones((m, 1)), II))
				I = np.vstack((I, T))
		return I

	def index_set(self, n: int, d: int) -> np.ndarray:
		"""
			Enumerate multi-indices for a total degree of up to `n` in `d` variables.
		Args:
			n (int): The maximum total degree
			d (int): The number of variables

		Returns:
			np.ndarray: The multi-indices for the given maximum total degree and number of variables
		"""
		I = np.zeros((1, d), dtype = np.integer)
		for i in range(1, n+1):
			II = self.full_index_set(i, d)
			I = np.vstack((I, II))
		return I[:,::-1].astype(int) #this has length
