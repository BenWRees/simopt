from __future__ import annotations
from typing import Callable

from numpy.linalg import norm, pinv, qr, lstsq
import numpy as np
from math import ceil, isnan, isinf, comb, factorial, log
import warnings
from scipy.optimize import minimize, NonlinearConstraint, linprog
from scipy import optimize
from scipy.stats import linregress
warnings.filterwarnings("ignore")
import importlib
from copy import deepcopy
import inspect
import traceback
import matplotlib.pyplot as plt 


from simopt.linear_algebra_base import finite_difference_gradient


from simopt.base import (
	ConstraintType,
	ObjectiveType,
	Problem,
	Solution,
	Solver,
	VariableType,
)

from simopt.solvers.active_subspaces.basis import *
# from simopt.solvers.active_subspaces.polyridge import *
# from simopt.solvers.active_subspaces.subspace import ActiveSubspace
from simopt.solvers.active_subspaces.index_set import IndexSet


from .Sampling import SamplingRule
from .TrustRegion import * 
from .Sampling import * 
from .Geometry import *

__all__ = ['TrustRegion', 'OMoRF']



class TrustRegionBase(Solver) :
	
	@property
	def objective_type(self) -> ObjectiveType:
		return ObjectiveType.SINGLE

	@property
	def constraint_type(self) -> ConstraintType:
		return ConstraintType.BOX

	@property
	def variable_type(self) -> VariableType:
		return VariableType.CONTINUOUS

	@property
	def gradient_needed(self) -> bool:
		return False
	
	@property
	def specifications(self) -> dict[str, dict] :
		return {
			"crn_across_solns": {
				"description": "CRN across solutions?",
				"datatype": bool,
				"default": True
			},
			"eta_1": {
				"description": "threshhold for a successful iteration",
				"datatype": float,
				"default": 0.1
			},
			"eta_2": {
				"description": "threshhold for a very successful iteration",
				"datatype": float,
				"default": 0.8
			},
			"gamma_1": {
				"description": "trust-region radius increase rate after a very successful iteration",
				"datatype": float,
				"default": 2.5
			},
			"gamma_2": {
				"description": "trust-region radius decrease rate after an unsuccessful iteration",
				"datatype": float,
				"default": 0.5
			},
			"easy_solve": {
				"description": "solve the subproblem approximately with Cauchy point",
				"datatype": bool,
				"default": False
			},
			"sampling rule" : {
				"description": "An instance of the sampling rule being used",
				"datatype": str,
				"default": 'BasicSampling' #just returns 10 every time
			}, 
			"lambda_min": {
				"description": "minimum sample size",
				"datatype": int,
				"default": 5
			},
			"geometry instance": {
				"description": "Instance of the geometric behaviours of the space where trust region values are sampled from",
				"datatype": str,
				"default": "TrustRegionGeometry"
			},
			"polynomial basis": {
				"description": "Polynomial basis to use in model construction",
				"datatype": str, 
				"default": "AstroDFBasis"
			}, 
			"polynomial degree": {
				"description": "degree of the polynomial",
				"datatype": int, 
				"default": 2
			},
			"model type" : {
				"description": "The type of random model used",
				"datatype": str,
				"default": "RandomModel" 
			},
			"ps_sufficient_reduction": {
				"description": "use pattern search if with sufficient reduction, 0 always allows it, large value never does",
				"datatype": float,
				"default": 0.1,
			},
			"model construction parameters": {
				"description": "List of initial parameters for the model construction",
				"datatype": dict, 
				"default": {'w': 0.85, 
					'mu':1000,
					'beta':10, 
					'criticality_threshold': 0.1, 
					'skip_criticality': False,
				}
			}
		}
	
	@property
	def check_factor_list(self) -> dict[str, Callable] : 
		return {
			"crn_across_solns": self.check_crn_across_solns,
			"eta_1": self.check_eta_1,
			"eta_2": self.check_eta_2,
			"gamma_1": self.check_gamma_1,
			"gamma_2": self.check_gamma_2,
			"lambda_min": self.check_lambda_min,
			"geometry instance": self.check_geometry_instance, 
			"polynomial basis": self.check_poly_basis, 
			"model type": self.check_random_model_type,
			"model construction parameters": self.check_model_construction_parameters,
			"sampling rule": self.check_sampling_rule,
			"polynomial degree": self.check_poly_degree,
			"ps_sufficient_reduction": self.check_ps_sufficient_reduction,
		}
	
	def __init__(self, name="TRUSTREGION_BASE", fixed_factors: dict | None = None) -> None :
		super().__init__(name, fixed_factors)
		self.rho = []

	def check_eta_1(self):
		if self.factors["eta_1"] <= 0 :
			raise ValueError("The threshold for a 'successful' iteration needs to be positive")


	def check_eta_2(self):
		if self.factors["eta_2"] <= self.factors["eta_1"] :
			raise ValueError("A 'very successful' iteration threshold needs to be greater than a 'successful' iteration threshold")

	def check_gamma_1(self):
		if self.factors["gamma_1"] <= 1 :
			raise ValueError('The trust region radius increase must be greater than 1')

	def check_gamma_2(self):
		if (self.factors["gamma_2"] >= 1 or self.factors["gamma_2"] <= 0) : 
			raise ValueError('The trust region radius decrease must be between 1 and 0 (exclusive)')
	
		
	def check_lambda_min(self):
		if self.factors["lambda_min"] <= 2 :
			raise ValueError('The smallest sample size must be greater than 2')
	
	def check_geometry_instance(self) -> bool:
		if self.factors['geometry instance'] is None : 
			raise ValueError('Geometry Instance Needs to be Implemented')
	
	def check_poly_basis(self) -> bool:
		if self.factors['polynomial basis'] is None : 
			raise ValueError('Polynomial Basis Needs to be Implemented') 
	
	def check_poly_degree(self) -> bool : 
		if self.factors['polynomial degree'] < 1 :
			raise ValueError('Local Model Polynomial Degree must be at least 1')
	
	def check_random_model_type(self) -> bool:
		if self.factors['model type'] is None : 
			raise ValueError('random model type is not implemented') 
	
	def check_model_construction_parameters(self) -> None :
		if not isinstance(self.factors['model construction parameters'], dict) : 
			raise ValueError('The model construction parameters must be a dictionary')
	
	def check_sampling_rule(self) -> None:
		if self.factors['sampling rule'] is None : 
			raise ValueError('sampling rule is not implemented')
	
	def check_ps_sufficient_reduction(self) -> None:
		if self.factors["ps_sufficient_reduction"] < 0:
			raise ValueError(
				"ps_sufficient reduction must be greater than or equal to 0."
			)

	#nice way to allow for different types of random models
	def model_instantiation(self) :
		class_name = self.factors['model type'].strip()
		module = importlib.import_module('simopt.solvers.TrustRegion.Models')
		return getattr(module, class_name)
	
	def polynomial_basis_instantiation(self) :
		class_name = self.factors['polynomial basis'].strip()
		module = importlib.import_module('simopt.solvers.active_subspaces.basis')
		return getattr(module, class_name)

	def sample_instantiation(self) :
		class_name = self.factors['sampling rule'].strip()
		module = importlib.import_module('simopt.solvers.TrustRegion.Sampling')
		sampling_instance = getattr(module, class_name)(self)
		return sampling_instance

	def geometry_type_instantiation(self) :
		class_name = self.factors['geometry instance'].strip()
		module = importlib.import_module('simopt.solvers.TrustRegion.Geometry')
		return getattr(module, class_name)


	def solve_subproblem(self, delta, model, problem, solution, visited_pts_list) :
		"""
		Solve the Trust-Region subproblem either using Cauchy reduction or a black-box optimisation solver
		
		Args:
			model (random_model): the locally constucted model
			problem (base.Problem): the simulation-optimisation Problem
			solution (base.Solution): the current iteration's solution

		Returns:
			base.Solution - the candidate solution from the subproblem
		"""
		raise NotImplementedError


	def evaluate_candidate_solution(self, model, problem, fval_tilde, delta_k, interpolation_solns, current_solution, candidate_solution, recommended_solns) :
		"""
		Evaluate the candidate solution, by looking at the ratio comparison 
		
		Args:
			model (random_model): the local model
			delta_k (float): the current trust-region radius size
			candidate_solution (base.Solution): the current iterations candidate solution
			recommended_solns ([]): Description
		"""
		raise NotImplementedError

	#solve the problem - inherited from base.Solver
	def solve(self, problem: Problem) -> tuple[list[Solution], list[int]] :
		raise NotImplementedError


class TrustRegion(TrustRegionBase) :

	# @property
	# def specifications(self) -> dict[str, dict] :
	# 	return {a:b for a,b in zip(list(super().specifications.keys()), list(super().specifications.values()))}
	
	def __init__(self, name="TRUSTREGION", fixed_factors: dict | None = None) -> None :
		super().__init__(name, fixed_factors)
		self.rho = []
	
	def construct_symmetric_matrix(self, column_vector) :
		flat_vector = np.array(column_vector).flatten()
		# Create the symmetric matrix
		n = len(flat_vector)
		symmetric_matrix = np.zeros((n, n), dtype=flat_vector.dtype)

		for i in range(n):
			for j in range(n):
				symmetric_matrix[i, j] = flat_vector[abs(i - j)]
		
		return symmetric_matrix


	def solve_subproblem(self, delta, model, problem, solution, visited_pts_list) :
		"""
		Solve the Trust-Region subproblem either using Cauchy reduction or a black-box optimisation solver
		
		Args:
			model (random_model): the locally constucted model
			problem (base.Problem): the simulation-optimisation Problem
			solution (base.Solution): the current iteration's solution

		Returns:
			base.Solution - the candidate solution from the subproblem
		"""
		new_x = solution.x

		q, grad, Hessian = model.coefficients
		fval = model.fval

		if self.factors['easy_solve'] :
			# Cauchy reduction
			if np.dot(np.multiply(grad, Hessian), grad) <= 0:
				tau = 1
			else:
				tau = min(1, norm(grad) ** 3 / (delta * np.dot(np.multiply(grad, Hessian), grad)))
			grad = np.reshape(grad, (1, problem.dim))[0]
			candidate_x = new_x - tau * delta * grad / norm(grad)

		
		else:
			def subproblem(s) : 
				Hessian_matrix = np.diag(Hessian)
				result = fval[0] + np.dot(s,grad) + np.dot(np.matmul(s,Hessian_matrix),s)
				return float(result[0])
			
			con_f = lambda s: float(norm(s))
			nlc = NonlinearConstraint(con_f, 0, delta)
			solve_subproblem = minimize(subproblem, np.zeros(problem.dim), method='trust-constr', constraints=nlc)
			candidate_x =  new_x + solve_subproblem.x



		# handle the box constraints
		new_candidate_list = []
		for i in range(problem.dim):
			if candidate_x[i] <= problem.lower_bounds[i]:
				new_candidate_list.append(problem.lower_bounds[i] + 0.01)
			elif candidate_x[i] >= problem.upper_bounds[i]:
				new_candidate_list.append(problem.upper_bounds[i] - 0.01)
			else:
				new_candidate_list.append(candidate_x[i])
		candidate_x = tuple(new_candidate_list)
		
		candidate_solution = self.create_new_solution(candidate_x, problem)
		if self.factors['model type'] == 'RandomModelReuse' :
			#we only append to the visited points list if we care about reusing points
			visited_pts_list.append(candidate_solution) 

		return candidate_solution, visited_pts_list

	def evaluate_candidate_solution(self, model, problem, fval_tilde, delta_k, interpolation_solns, current_solution, candidate_solution, recommended_solns, expended_budget, intermediate_budgets) :
		"""
		Evaluate the candidate solution, by looking at the ratio comparison 
		
		Args:
			model (random_model): the local model
			delta_k (float): the current trust-region radius size
			candidate_solution (base.Solution): the current iterations candidate solution
			recommended_solns ([]): Description
		"""
		fval = model.fval

		#pattern search
		if ((min(fval) < fval_tilde) and ((fval[0] - min(fval))>= self.factors["ps_sufficient_reduction"] * delta_k**2)) or ((candidate_solution.objectives_var/ (candidate_solution.n_reps * candidate_solution.objectives_mean**2))[0]> 0.75):
			fval_tilde = min(fval)
			candidate_solution = interpolation_solns[fval.index(min(fval))]  # type: ignore

		stepsize = np.subtract(np.array(candidate_solution.x), np.array(current_solution.x))
		model_reduction = model.local_model_evaluate(np.zeros(problem.dim)) - model.local_model_evaluate(stepsize)
		if model_reduction <= 0:
			rho = 0
		else:
			# difference = np.subtract(candidate_solution.x, current_solution.x)
			rho = (fval[0] - fval_tilde) / model_reduction

		self.rho.append(rho)

		# successful: accept
		if rho >= self.factors['eta_1']:
			# new_x = candidate_x
			current_solution = candidate_solution
			# final_ob = candidate_solution.objectives_mean
			recommended_solns.append(candidate_solution)
			intermediate_budgets.append(expended_budget)
			delta_k = min(delta_k, self.delta_max)
			
			# very successful: expand and accept
			if rho >= self.factors['eta_2'] :
				delta_k = min(self.factors['gamma_1'] * delta_k, self.delta_max)
			
		# unsuccessful: shrink and reject
		else:
			delta_k = min(self.factors['gamma_2'] * delta_k, self.delta_max)
			# new_solution = current_solution
			# recommended_solns.append(current_solution)
			# final_ob = fval[0]

		return current_solution, delta_k, recommended_solns, expended_budget, intermediate_budgets

	#solve the problem - inherited from base.Solver
	def solve(self, problem: Problem) -> tuple[list[Solution], list[int]] :
		recommended_solns = []
		intermediate_budgets = []
		expended_budget = 0

		self.delta_max = self.calculate_max_radius(problem)
		delta_k = 10 ** (ceil(log(self.delta_max * 2, 10) - 1) / problem.dim)

		visited_pts_list = []

		new_x = problem.factors["initial_solution"]
		current_solution = self.create_new_solution(new_x, problem)
		model_construction_parameters = self.factors['model construction parameters']


		#Dynamically load in different sampling rule, geometry type, and random model
		sampling_instance = self.sample_instantiation()

		geometry_instance = self.geometry_type_instantiation()(problem)
		poly_basis_instance = self.polynomial_basis_instantiation()(self.factors['polynomial degree'], None, problem.dim)
		model = self.model_instantiation()(geometry_instance, self, poly_basis_instance, problem, sampling_instance, model_construction_parameters)
		

		k=0

		while expended_budget < problem.factors["budget"]:
			k += 1 
			print(f'iteration: {k} \t expended budget {expended_budget} \t current objective function value: {current_solution.x}')

			#get pilot run and calculate kappa 
			# if hasattr(sampling_instance.__class__, 'calculate_kappa') :
			if k == 1 and hasattr(sampling_instance.__class__, 'calculate_kappa'):
				current_solution = self.create_new_solution(current_solution.x, problem)
				# current_solution = self.create_new_solution(current_solution.x, problem)
				if len(visited_pts_list) == 0:
					visited_pts_list.append(current_solution)
				
				current_solution, expended_budget = sampling_instance.calculate_kappa(k, problem, expended_budget, current_solution, delta_k)
			
				recommended_solns.append(current_solution)
				intermediate_budgets.append(expended_budget)
			elif self.factors['crn_across_solns'] and hasattr(sampling_instance.__class__, 'calculate_kappa'):
				# since incument was only evaluated with the sample size of previous incumbent, here we compute its adaptive sample size
				# adaptive sampling
				# current_solution, expended_budget = sampling_instance(problem, k, current_solution, delta_k, expended_budget, False)
				lambda_max = problem.factors['budget'] - expended_budget
				sample_size = current_solution.n_reps
				while True:
					sig2 = current_solution.objectives_var[0]
					stopping = sampling_instance.get_stopping_time(sig2, delta_k, k, problem, expended_budget)
					if (sample_size >= min(stopping, lambda_max) or expended_budget >= problem.factors['budget']):
						break
					problem.simulate(current_solution, 1)
					expended_budget += 1
					sample_size += 1

			#need to simulation the current solution before evaluating objective variance of the candidate solution later on
			elif k == 1 and not hasattr(sampling_instance.__class__, 'calculate_kappa'):
				current_solution = self.create_new_solution(current_solution.x,problem)
				# current_solution, expended_budget = sampling_instance(problem,k,current_solution,delta_k,expended_budget)
				problem.simulate(current_solution, 1)
				expended_budget += 1
				recommended_solns.append(current_solution)
				intermediate_budgets.append(expended_budget)

				

			#build random model 
			current_solution, delta_k, expended_budget, interpolation_solns, visited_pts_list = model.construct_model(current_solution, delta_k, k, expended_budget, visited_pts_list)
			# expended_budget = construction_budget # the additions to the expended budget is done in model.construct_model

			#solve random model 
			candidate_solution, visited_pts_list = self.solve_subproblem(delta_k, model, problem, current_solution, visited_pts_list)
			
			candidate_solution, fval_tilde, expended_budget = self.sample_candidate_solution(sampling_instance, candidate_solution, current_solution, problem, k, delta_k, expended_budget)

			#evaluate model
			# model, problem, fval_tilde, delta_k, interpolation_solns, candidate_solution, recommended_solns
			current_solution, delta_k, recommended_solns, expended_budget, intermediate_budgets = self.evaluate_candidate_solution(model, problem, fval_tilde, delta_k, interpolation_solns, current_solution,\
																			   candidate_solution, recommended_solns, expended_budget, intermediate_budgets)	
			

			# print('new solution: ', new_solution.x)

		return recommended_solns, intermediate_budgets
	

	def sample_candidate_solution(self, sampling_instance, candidate_solution, current_solution, problem, k, delta_k, expended_budget) : 
			if self.factors['crn_across_solns'] :
				problem.simulate(candidate_solution, current_solution.n_reps) 
				expended_budget += current_solution.n_reps 
				fval_tilde = -1 * problem.minmax[0] * candidate_solution.objectives_mean
				return candidate_solution, fval_tilde, expended_budget
			else :
				candidate_solution, expended_budget = sampling_instance(problem, k, candidate_solution, delta_k, expended_budget)
				fval_tilde = -1 * problem.minmax[0] * candidate_solution.objectives_mean
				return candidate_solution, fval_tilde, expended_budget

	def calculate_max_radius(self, problem) : 
		# Designate random number generator for random sampling
		find_next_soln_rng = self.rng_list[1]

		# Generate many dummy solutions without replication only to find a reasonable maximum radius
		dummy_solns: list[tuple[int, ...]] = []
		for _ in range(1000 * problem.dim):
			random_soln = problem.get_random_solution(find_next_soln_rng)
			dummy_solns.append(random_soln)
		# Range for each dimension is calculated and compared with box constraints range if given
		# TODO: just use box constraints range if given
		# delta_max = min(self.factors["delta_max"], problem.upper_bounds[0] - problem.lower_bounds[0])
		delta_max_arr: list[float | int] = []
		for i in range(problem.dim):
			delta_max_arr += [
				min(
					max([sol[i] for sol in dummy_solns])
					- min([sol[i] for sol in dummy_solns]),
					problem.upper_bounds[0] - problem.lower_bounds[0],
				)
			]
		# TODO: update this so that it could be used for problems with decision variables at varying scales!
		delta_max = max(delta_max_arr)
		return delta_max 

class OMoRF(TrustRegionBase):
	"""
		Optimisation by Moving Ridge Functions solver by Gross and Parks
	
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
	check_factor_list : dict 
		functions to check each fixed factor is performing
	"""

	@property
	def objective_type(self) -> ObjectiveType:
		return ObjectiveType.SINGLE

	@property
	def constraint_type(self) -> ConstraintType:
		return ConstraintType.BOX

	@property
	def variable_type(self) -> VariableType:
		return VariableType.CONTINUOUS

	@property
	def gradient_needed(self) -> bool:
		return False
	
	@property
	def specifications(self) -> dict[str, dict] :
		new_specifications = {
			"interpolation update tol":{
				"description": "tolerance values to check for updating the interpolation model",
				"datatype": tuple, 
				"default": (2.0,10.0)
			},
			"delta": {
				"description": "initial trust-region radius",
				"datatype": float, 
				"default": 0.5
			}, 
			"gamma_3": {
				"description": "trust_region radius increase rate after a very successful iteration",
				"datatype": float, 
				"default": 2.5
			},
			"gamma_shrinking": {
				"description": "the constant to make upper bound for delta in contraction loop",
				"datatype": float, 
				"default": 0.5
			}, 
			"omega_shrinking": {
				"description": "factor to shrink the trust-region radius",
				"datatype": float,
				"default": 0.5
			}, 
			"subspace dimension": {
				"description": "dimension size of the active subspace",
				"datatype": int, 
				"default": 1
			}, 
			"random directions": {
				"description": "Determine to take random directions in set construction",
				"datatype": bool, 
				"default": False 
			},
			"alpha_1" : {
				"description": "Scale factor to shrink the stepsize check",
				"datatype": float, 
				"default": 0.1
			},
			"alpha_2": {
				"description": "Scale factor to shrink the trust-region radius",
				"datatype": float,
				"default": 0.5
			},
			"rho_min": {
				"description": "initial rho when shrinking", 
				"datatype": float, 
				"default": 1.0e-8
			},
		}
		super().factors['eta_1'] = 0.1
		super().factors['eta_2'] = 0.7
		super().factors['geometry instance'] = 'OMoRFGeometry'
		super().factors['polynomial basis'] = 'NaturalPolynomialBasis'
		return {**super().specifications, **new_specifications}
	
	@property
	def check_factor_list(self) -> dict[str, Callable] : 
		new_check_list = {
			"interpolation update tol":self.check_tolerance,
			"delta": self.check_initial_radius,
			"gamma_3": self.check_gamma_3,
			"gamma_shrinking": self.check_gamma_shrinking,
			"omega_shrinking": self.check_omega_shrinking,
			"subspace dimension": self.check_dimension_reduction,
			"random directions": self.check_random_directions,
			"alpha_1": self.check_alpha_1,
			"alpha_2": self.check_alpha_2,
			"rho_min": self.check_rho_min,
			}
		return {**super().check_factor_list, **new_check_list}

	
	def check_tolerance(self) -> bool:
		return self.factors['interpolation update tol'] >(0,0) and self.factors['interpolation update tol'] <= (1,1)
	
	def check_initial_radius(self) -> bool:
		return self.factors['delta'] > 0
	
	def check_gamma_3(self) -> bool:
		return self.factors['gamma_3'] > 0
	
	def check_gamma_shrinking(self) -> bool:
		return self.factors['gamma_shrinking'] > 0
	
	def check_omega_shrinking(self) -> bool:
		return self.factors['omega_shrinking'] > 0
	
	def check_dimension_reduction(self) -> bool:
		return self.factors['subspace dimension'] >= 1
	
	def check_random_directions(self) -> bool : 
		return isinstance(self.factors['random directions'], bool)
	
	def check_alpha_1(self) -> bool :
		return self.factors['alpha_1'] > 0 
	
	def check_alpha_2(self) -> bool :
		return self.factors['alpha_2'] > 0 
	
	def check_rho_min(self) -> bool : 
		return self.factors['rho_min'] < self.factors['delta']

	def __init__(self, name="OMoRF", fixed_factors: dict | None = None) -> None:
		"""
			Initialisation of the OMoRF solver see base.Solver 
		
		Parameters
		----------
		name : str, optional
			user-specified name for solver
		fixed_factors : None, optional
			fixed_factors of the solver
		"""
		# self.check_factor_list = {**super().check_factor_list, **self.check_factor_list()}
		super().__init__(name, fixed_factors)

	#! Do not use this. Will reset the function value if it is wrong
	def _set_iterate(self, problem):
		if problem.minmax[0] == -1 :
			ind_min = np.argmin(self.f) #get the index of the smallest function value 
			self.s_old = self.S[ind_min,:] #get the x-val that corresponds to the smallest function value 
			self.f_old = self.f[ind_min] #get the smallest function value 
		else : 
			ind_min = np.argmax(self.f) #get the index of the largest function value 
			self.s_old = self.S[ind_min,:] #get the x-val that corresponds to the largest function value 
			self.f_old = self.f[ind_min] #get the largest function value 

	
	"""	
	def _get_scale(dirn, delta, lower, upper):
		scale = delta
		for j in range(len(dirn)):
			if dirn[j] < 0.0:
				scale = min(scale, lower[j] / dirn[j])
			elif dirn[j] > 0.0:
				scale = min(scale, upper[j] / dirn[j])
		return scale

	def _apply_scaling(self, S):
		if self.bounds is not None and self.scale_bounds:
			shift = self.bounds[0].copy()
			scale = self.bounds[1] - self.bounds[0]
			return np.divide((S - shift), scale)
		else:
			return S

	def _remove_scaling(self, S):
		if self.bounds is not None and self.scale_bounds:
			shift = self.bounds[0].copy()
			scale = self.bounds[1] - self.bounds[0]
			return shift + np.multiply(S, scale)
		else:
			return S
	"""

	def _set_delta(self, val) : 
		self.delta_k = val

	def _set_counter(self, count) :
		self.unsuccessful_iteration_counter = count 

	def _set_ratio(self, ratio) : 
		if type(ratio) == list or isinstance(ratio, np.ndarray): 
			ratio = ratio[0]
		self.ratio = ratio 

	def _set_rho_k(self, value):
		self.rho_k = value 


	def blackbox_evaluation(self, s, problem):
		"""
		Evaluates the point s for the problem

		self.S is array-like of all the x values 
		self.f is a 1-d array of function values 
		"""
		# If S has 1 or more points in it and the point being evaluated is not unique. Just grab the existing fn. value
		if self.S.size > 0 and np.unique(np.vstack((self.S, s)), axis=0).shape[0] == self.S.shape[0]:
			s = s.reshape(1,-1)
			ind_repeat = np.argmin(norm(self.S - s, ord=np.inf, axis=1))
			f = self.f[ind_repeat]
		else :
			new_solution = self.create_new_solution(tuple(s), problem) 
			problem.simulate(new_solution, 1)
			f = -1 * problem.minmax[0] * new_solution.objectives_mean
			self.expended_budget += 1
			
			s = s.reshape(1,-1)

			#update S and f
			if self.f.size == 0:
				self.S = s
				self.f = f
			else:
				self.S = np.vstack((self.S, s))
				self.f = np.vstack((self.f, f))
		

		return f[0]
	


	def calculate_subspace(self, S, f, delta_k) -> np.ndarray:
		""" Calculate the Active Subspace

		Args:
			S (np.ndarray): A matrix of shape (M,n) of sample points
			f (np.ndarray): A column vector of shape (M,1)
			problem (simopt.Problem): The optimisation problem to solve on

		Returns:
			np.ndarray: The active subspace matrix of shape (n,d)
		"""				

		#construct covariance matrix
		M, n = S.shape
		num_grad_lb = 2.0 * self.d * np.log(n)

		if M < num_grad_lb:
			warnings.warn('Number of gradient evaluation points is likely to be insufficient. Consider resampling!', UserWarning)
		
		covar = self._get_grads(S, f, delta_k) 
		weights = np.ones((M, 1)) / M
		R = covar * weights
		C = np.dot(covar.T, R)


		# Compute eigendecomposition!
		e, W = np.linalg.eigh(C)
		idx = e.argsort()[::-1]
		eigs = e[idx]
		eigVecs = W[:, idx]
		if hasattr(self, 'data_scaler'):
			Xmax, Xmin = np.max(S,axis=0), np.min(S,axis=0)
			scale_factors = 2.0 / (Xmax - Xmin)
			eigVecs = scale_factors[:, np.newaxis] * eigVecs
			eigVecs = np.linalg.qr(eigVecs)[0]

		subspace = eigVecs
		eigenvalues = eigs

		U0 = subspace[:,0].reshape(-1,1) #this is a column vector
		U1 = subspace[:,1:]
	   

	   #Add the other d-1 columns to U0 by selecting the columns of U1 with the largest coefficients of determination
		for i in range(self.d-1):
			R = []
			#loop through each column
			for j in range(U1.shape[1]):
				#stack U with the AS and the jth column of the orthogonal complement
				U = np.hstack((U0, U1[:, j].reshape(-1,1)))
				Y = np.dot(S, U) #map the sample points to the reduced subspace 
				
				# print(f'shape of Y: {Y.shape}')
				
				coeff = self.construct_model(Y, f, self.poly_basis_subspace, U=U)
				sample_pts = [] 
				for pt in Y : 
					pt = np.array(pt).reshape(-1,1)
					sample_pts.append(self.eval_model(pt, self.poly_basis_subspace, coeff=coeff))

				# r = linregress(sample_pts, f).rvalue
				sample_pts = np.array(sample_pts).reshape(-1,)
				_, _, r, _, _ = linregress(sample_pts, f.flatten())

				R.append(r**2) #coefficient of determination
			index = np.argmax(R)
			U0 = np.hstack((U0, U1[:, index].reshape(-1,1))) #add the column corresponding to the largest coefficient of determination in the orthogonal complement to AS
			U1 = np.delete(U1, index, 1) #remove feature that was added 
	   

		self.U = U0.T
		return self.U


	
	def construct_model(self, S, f, poly_basis, U=None) -> list[float] : 
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
		if U is None : 
			U = self.U

		Y = np.matmul(S, U.T) #This is throwing an exception

		if not isinstance(Y, np.ndarray) :
			Y = np.row_stack(Y) #reshape Y to be a matrix of (M,d)
		M = poly_basis.V(Y) # now constructs M based on the polynomial basis being used
		# print(f'shape of M: {M.shape}')
		q = np.matmul(pinv(M), f)
				
		return q
	

	def eval_model(self, x_k, poly_basis, coeff=None) : 
		if coeff is None : 
			coeff = self.coefficients

		if len(x_k) != poly_basis.dim : 
			x_k = list(x_k.flatten())
			diff = poly_basis.dim - len(x_k)
			x_k += [0] * diff
			x_k = np.array(x_k)
		interpolation_set = x_k.reshape((1,len(x_k)))
		# interpolation_set = np.row_stack(interpolation_set)
		X = poly_basis.V(interpolation_set)[0]
		evaluation = np.dot(X,coeff)
		return evaluation


	def finite_difference_gradient(self, new_solution:Solution, problem: Problem) -> np.ndarray : 
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
		forward = np.isclose(new_x, lower_bound, atol = 10**(-7)).astype(int)
		backward = np.isclose(new_x, upper_bound, atol = 10**(-7)).astype(int)
		# BdsCheck: 1 stands for forward, -1 stands for backward, 0 means central diff.
		BdsCheck = np.subtract(forward, backward)

		self.expended_budget += (2*problem.dim) + 1
		return finite_difference_gradient(new_solution, problem, BdsCheck=BdsCheck)

	def finite_differencing(self,x_val: np.ndarray, model_coeff: list[float], delta_k: float) : 
		lower_bound = x_val - delta_k
		upper_bound = x_val + delta_k


		fn = self.eval_model(x_val, self.poly_basis_subspace, coeff=model_coeff)

		
		BdsCheck =  np.zeros(self.n)
		
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

			fn1, fn2 = 0,0 
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


	def _get_grads(self, X: np.ndarray, f: np.ndarray, delta_k: float) -> np.ndarray : 
		"""Calculate gradients 

		Args:
			X (np.ndarray): (N,m) matrix of N x-vals to be evaluated
			problem (Problem): 

		Returns:
			np.ndarray: (N,m) matrix of N gradients evaluated at each row of X
		"""
		grads = np.zeros(X.shape)

		# #fill out each row 
		# for idx,x_val in enumerate(X) : 
		# 	x_solution = self.create_new_solution(x_val, problem)
		# 	grads[idx, :] = self.finite_difference_gradient(x_solution, problem)

		#Construct a local model over the space of subspace interpolation points
		coeff = self.construct_model(X, f, self.poly_basis_subspace)

		for idx, x_val in enumerate(X) : 
			grads[idx, :] = self.finite_differencing(x_val, coeff, delta_k)


		return grads 


	def solve(self, problem):
		#initialise factors: 
		self.recommended_solns = []
		self.intermediate_budgets = []
		
		self.S = np.array([])
		self.f = np.array([])
		self.g = np.array([])
		self.d = self.factors['subspace dimension']

		self.expended_budget = 0
		# self.delta_k = self.factors['delta']
		self._set_delta(self.factors['delta'])
		# self.rho_k = self.delta_k
		self._set_rho_k(self.delta_k) 
		self._set_counter(0)
		self.rhos = []
		self.budgets = [0]
		self.deltas = [self.delta_k]
		self.n = problem.dim
		self.deg = self.factors['polynomial degree'] 
		self.q = comb(self.d + self.deg, self.deg) +  self.n * self.d #int(0.5*(self.d+1)*(self.d+2))
		self.p = self.n+1
		self.epsilon_1, self.epsilon_2 = self.factors['interpolation update tol'] #epsilon is the tolerance in the interpolation set update 
		self.random_initial = self.factors['random directions']
		self.alpha_1 = self.factors['alpha_1'] #shrink the trust region radius in set improvement 
		self.alpha_2 = self.factors['alpha_2'] #shrink the stepsize reduction  
		self.rho_min = self.factors['rho_min']


		#set up initial Solution
		current_x = problem.factors["initial_solution"]
		# print(f'initial solution: {current_x}')
		self.current_solution = self.create_new_solution(current_x, problem)
		self.recommended_solns.append(self.current_solution)
		self.intermediate_budgets.append(self.expended_budget)

		
		""" 
		self.s_old = self._apply_scaling(s_old) #shifts the old solution into the unit ball
		"""

		self.k = 0
		self._set_counter(0)

		geo_factors = {
			'random_directions': self.random_initial,
			'epsilon_1': self.epsilon_1,
			'epsilon_2': self.epsilon_2,
			'rho_min': self.rho_min,
			'alpha_1': self.alpha_1,
			'alpha_2': self.alpha_2,
			'n': self.n,
			'd': self.d,
			'q': self.q,
			'p': self.p
			
		}


		#basis construction
		index_set = IndexSet('total-order', orders=np.tile([2], self.q))
		self.index_set = index_set.get_basis()[:,range(self.d-1, -1, -1)]
		
		self.poly_basis_model = self.polynomial_basis_instantiation()(self.factors['polynomial degree'], dim=self.factors['subspace dimension'])
		self.poly_basis_subspace = self.polynomial_basis_instantiation()(self.factors['polynomial degree'], dim=self.n)
		self.geometry_instance = self.geometry_type_instantiation()(problem, self, self.index_set, **geo_factors)

		self.s_old = np.array(current_x)
		self.f_old = self.blackbox_evaluation(self.s_old,problem)

		# Construct the sample set for the subspace 
		S_full = self.geometry_instance.generate_set(self.n + 1, self.s_old, self.delta_k)
		f_full = np.zeros((self.n + 1, 1))
		f_full[0, :] = self.f_old #first row gets the old function values 


		# get the rest of the function evaluations - write as a function
		for i in range(1, self.n+1):
			#simulate the problem at each component of f_
			new_solution = self.create_new_solution(S_full[i, :], problem)
			problem.simulate(new_solution, 1)
			self.expended_budget += 1
			f_full[i, :] = -1 * problem.minmax[0] * new_solution.objectives_mean
			# self.expended_budget = reset_budget 
			#case where we use up our whole budget getting the function values 
			if self.expended_budget > problem.factors['budget'] :
				return self.recommended_solns, self.intermediate_budget


		
		#This is needed to ensure that model construction in the subspace works
		self.U = np.eye(self.n, self.n)

		#initial subspace calculation - requires gradients of f_full 
		self.calculate_subspace(S_full, f_full, self.delta_k)

		#This constructs the sample set for the model construction
		S_red, f_red = self.geometry_instance.sample_set('new', self.s_old, self.delta_k, self.rho_k, self.f_old, self.U, full_space=False)
		
		while self.expended_budget < problem.factors['budget'] :
			print(f'\niteration: {self.k} \t expended budget {self.expended_budget} \t current objective function value: {self.f_old}')
			#if rho has been decreased too much we end the algorithm  
			if self.rho_k <= self.rho_min:
				break
			
			#BUILD MODEL
			try: 
				self.coefficients = self.construct_model(S_red, f_red, self.poly_basis_model) #this should be the model instance construct model
			except : #thrown if the sample set is not defined properly 
				print(traceback.format_exc())
				S_red, f_red = self.geometry_instance.sample_set('improve', self.s_old, self.delta_k, self.rho_k, self.f_old, self.U, S_red, f_red, full_space=False)
				self.intermediate_budgets.append(self.expended_budget)
				continue 

			#SOLVE THE SUBPROBLEM
			candidate_solution, S_full, S_red, f_full, f_red, reset_flag = self.solve_subproblem(problem, S_full, S_red, f_full, f_red)
			candidate_fval = self.blackbox_evaluation(np.array(candidate_solution.x),problem)

			if reset_flag :
				self.recommended_solns.append(self.current_solution)
				self.intermediate_budgets.append(self.expended_budget) 
				self.k +=1
				break 
			
			#EVALUATE THE CANDIDATE SOLUTION
			S_red, S_full, f_red, f_full = self.evaluate_candidate_solution(problem, candidate_fval, candidate_solution, S_red, S_full, f_red, f_full)

			# print(f'EXPENDED BUDGET: {self.expended_budget}')

			self.k+=1

		self.plot_deltas_against_budget()
		return self.recommended_solns, self.intermediate_budgets

	def plot_deltas_against_budget(self) :
		plt.plot(self.budgets, self.deltas)
		plt.title('trust region radius against expended budget')
		plt.show()

		plt.scatter(self.budgets[1:], self.rhos,marker='x')
		plt.title('ratios against expended budget')
		plt.show()

	def solve_subproblem(self, problem: Problem, S_full:np.ndarray, S_red: np.ndarray, f_full: float, f_red: float) :
		"""
			Solves the trust-region subproblem for ``trust-region`` or ``omorf`` methods
		"""
		
		omega_s = self.factors['omega_shrinking']
		reset_flag = False

		cons =  NonlinearConstraint(lambda x : norm(x), 0, self.delta_k)
		
		
		obj = lambda x: self.eval_model(np.dot(x,self.U.T), self.poly_basis_model).item()
		res = minimize(obj, np.zeros(problem.dim), method='trust-constr', constraints=cons, options={'disp': False})
		step_dist = res.x
		print(f'stepsize: {step_dist}')
		s_new = self.s_old + step_dist
		# m_new = res.fun 

		print(f'CANDIDATE SOLUTION: {s_new}')

		for i in range(problem.dim):
			if s_new[i] <= problem.lower_bounds[i]:
				s_new[i] = problem.lower_bounds[i] + 0.01
			elif s_new[i] >= problem.upper_bounds[i]:
				s_new[i] = problem.upper_bounds[i] - 0.01


		candidate_solution = self.create_new_solution(tuple(s_new), problem)

		step_dist = norm(s_new - self.s_old, ord=np.inf)

		# Safety step implemented in BOBYQA
		if step_dist < omega_s*self.rho_k:
			self.ratio= -0.1
			self._set_counter(3)
			# self.delta_k = max(0.5*self.delta_k, self.rho_k)
			self._set_delta(max(0.5*self.delta_k, self.rho_k))
			# self.d += 1 #increase the dimension
			S_full, f_full, S_red, f_red, delta_k, rho_k, U = self.geometry_instance.update_geometry_omorf(self.s_old, self.f_old, self.delta_k, self.rho_k, self.U, S_full, f_full, S_red, f_red, self.unsuccessful_iteration_counter, self.ratio)
			self.U = U
			self._set_delta(delta_k)
			self._set_rho_k(rho_k)
		
		# #this is a breaking condition
		if self.rho_k <= self.rho_min:
			reset_flag=True

		return candidate_solution, S_full, S_red, f_full, f_red, reset_flag

	def evaluate_candidate_solution(self, problem: Problem, fval_tilde: float, candidate_solution: Solution, S_red: np.ndarray, S_full: np.ndarray, f_red: np.ndarray, f_full: np.ndarray) :
		
		gamma_1 = self.factors['gamma_1']
		gamma_2 = self.factors['gamma_2']
		gamma_3 = self.factors['gamma_3']
		eta_1 = self.factors['eta_1']
		eta_2 = self.factors['eta_2']
		s_new = np.array(candidate_solution.x)
		
		model_eval_old = self.eval_model(np.dot(self.s_old, self.U.T), self.poly_basis_model).item()
		model_eval_new = self.eval_model(np.dot(s_new, self.U.T), self.poly_basis_model).item()

		print(f'DIFFERENCE IN CANDIDATE EVALUATION AT MODEL AND FUNCTION: {abs(fval_tilde - model_eval_new)}')

		#! ONLY ISSUE IS THAT IT ACCEPTS GROWING VALUES - DUE TO BIG DIFFERENCE BETWEEN model_eval_new AND fval_tilde
		del_f =  self.f_old - fval_tilde #self.f_old - f_new 
		# del_m = np.asscalar(my_poly.get_polyfit(np.dot(self.s_old,self.U))) - m_new
		del_m = model_eval_old - model_eval_new

		print(f'The model evaluation for the old value is {model_eval_old} and for the candidate value it is {model_eval_new}')
		print(f'The old function value is {self.f_old} and the new function value is {fval_tilde}')

		print(f'numerator of ratio is {del_f} and the denominator is {del_m}')

		step_dist = norm(np.array(candidate_solution.x) - self.s_old, ord=np.inf)

		#in the case that the denominator is very small 
		# if abs(del_m) < 100*np.finfo(float).eps :
		# # if del_m <= 0:
		# 	self._set_ratio(1.0)

		#! Need to handle the case where the model evaluation is increasing when it should be decreasing - should reject this!
		# elif norm(model_eval_new - fval_tilde) > abs(self.f_old - fval_tilde) :
		if del_f < 0 :
		# elif ((self.f_old < fval_tilde ) and problem.minmax[0] == 1) or ((self.f_old > fval_tilde ) and problem.minmax[0] == -1) : 
			self._set_ratio(0.0)
		else:
			self._set_ratio((del_f/del_m))

		self.rhos.append(self.ratio)

		# self._set_iterate(problem)

		"""ind_min = np.argmin(self.f) #get the index of the smallest function value 
		self.s_old = self.S[ind_min,:] #get the x-val that corresponds to the smallest function value 
		self.f_old = np.asscalar(self.f[ind_min]) #get the smallest function value""" 

		#add candidate value and corresponding fn val to interpolation sets
		S_red, f_red = self.geometry_instance.sample_set('replace', self.s_old, self.delta_k, self.rho_k, self.f_old, self.U, S_red, f_red, s_new, fval_tilde, full_space=False)
		S_full, f_full = self.geometry_instance.sample_set('replace', self.s_old, self.delta_k, self.rho_k, self.f_old, self.U, S_full, f_full, s_new, fval_tilde)

		print(f'RATIO COMPARISON VALUE: {self.ratio}')

		if self.ratio >= eta_2:
			print('VERY SUCCESSFUL ITERATION')
			self._set_counter(0)
			# self.delta_k = max(gamma_2*self.delta_k, gamma_3*step_dist)
			self._set_delta(max(gamma_1*self.delta_k, gamma_3*step_dist))
			self.current_solution = candidate_solution
			self.recommended_solns.append(self.current_solution) 
			self.intermediate_budgets.append(self.expended_budget)
			self.f_old = fval_tilde
			self.s_old = s_new
			
			
		
		elif self.ratio >= eta_1:
			print('SUCCESSFUL ITERATION')
			self._set_counter(0)
			# self.delta_k = max(gamma_1*self.delta_k, step_dist, self.rho_k)
			self._set_delta(max(gamma_2*self.delta_k, step_dist, self.rho_k))
			self.current_solution = candidate_solution
			self.f_old = fval_tilde
			self.recommended_solns.append(self.current_solution) 
			self.intermediate_budgets.append(self.expended_budget)
			self.s_old = s_new

		else:
			print('UNSUCCESSFUL ITERATION')
			self._set_counter(self.unsuccessful_iteration_counter + 1)
			# self.delta_k = max(min(gamma_1*self.delta_k, step_dist), self.rho_k)
			self._set_delta(max(min(gamma_1*self.delta_k, step_dist), self.rho_k))
			S_full, f_full, S_red, f_red, delta_k, rho_k, U = self.geometry_instance.update_geometry_omorf(self.s_old, self.f_old, self.delta_k, self.rho_k, self.U, S_full, f_full, S_red, f_red, self.unsuccessful_iteration_counter, self.ratio)
			self.U = U 
			self._set_delta(delta_k)
			self._set_rho_k(rho_k)

		self.deltas.append(self.delta_k)
		self.budgets.append(self.expended_budget)
		return S_red, S_full, f_red, f_full
	