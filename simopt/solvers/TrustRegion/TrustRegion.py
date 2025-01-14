from __future__ import annotations
from typing import Callable

from numpy.linalg import norm, pinv
import numpy as np
from math import ceil, isnan, isinf, comb, factorial
import warnings
from scipy.optimize import minimize, NonlinearConstraint
from scipy import optimize
warnings.filterwarnings("ignore")
import importlib
from copy import deepcopy
import inspect


from simopt.base import (
	ConstraintType,
	ObjectiveType,
	Problem,
	Solution,
	Solver,
	VariableType,
)

from simopt.solvers.active_subspaces.basis import *
from simopt.solvers.active_subspaces.polyridge import *
from simopt.solvers.active_subspaces.subspace import ActiveSubspace
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
				"default": 0.7
			},
			"gamma_1": {
				"description": "trust-region radius increase rate after a very successful iteration",
				"datatype": float,
				"default": 1.5
			},
			"gamma_2": {
				"description": "trust-region radius decrease rate after an unsuccessful iteration",
				"datatype": float,
				"default": 0.5
			},
			"delta": {
				"description": "size of the trust-region radius",
				"datatype": float,
				"default": 5.0
			}, 
			"delta_max": {
				"description": "maximum size of the trust-region radius",
				"datatype": float,
				"default": 200.0
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
				"default": 4
			},
			"geometry instance": {
				"description": "Instance of the geometric behaviours of the space where trust region values are sampled from",
				"datatype": str,
				"default": "TrustRegionGeometry"
			},
			"polynomial basis": {
				"description": "Polynomial basis to use in model construction",
				"datatype": str, 
				"default": "NaturalPolynomialBasis"
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
			"model construction parameters": {
				"description": "List of initial parameters for the model construction",
				"datatype": dict, 
				"default": {'w': 0.85, 
					'mu':1000,
					'beta':10, 
					'criticality_threshold': 0.1, 
					'skip_criticality': True,
					'lambda_min': 4
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
			"delta_max": self.check_delta_max,
			"delta": self.check_delta,
			"lambda_min": self.check_lambda_min,
			"geometry instance": self.check_geometry_instance, 
			"polynomial basis": self.check_poly_basis, 
			"model type": self.check_random_model_type,
			"model construction parameters": self.check_model_construction_parameters,
			"sampling rule": self.check_sampling_rule,
			"polynomial degree": self.check_poly_degree
		}
	
	def __init__(self, name="TRUSTREGION_BASE", fixed_factors: dict | None = None) -> None :
		super().__init__(name, fixed_factors)
		self.rho = []

	def check_eta_1(self):
		return self.factors["eta_1"] > 0

	def check_eta_2(self):
		return self.factors["eta_2"] > self.factors["eta_1"]

	def check_gamma_1(self):
		return self.factors["gamma_1"] > 1

	def check_gamma_2(self):
		return (self.factors["gamma_2"] < 1 and self.factors["gamma_2"] > 0)
	
	def check_delta_max(self):
		return self.factors["delta_max"] > 0
	
	def check_delta(self):
		return self.factors["delta_max"] > 0

	def check_lambda_min(self):
		return self.factors["lambda_min"] > 2
	
	def check_geometry_instance(self) -> bool:
		return True 
	
	def check_poly_basis(self) -> bool:
		return True 
	
	def check_poly_degree(self) -> bool : 
		self.factors['polynomial degree'] >= 1
	
	def check_random_model_type(self) -> bool:
		return True 
	
	def check_model_construction_parameters(self) -> bool :
		return isinstance(self.factors['model construction parameters'], dict)
	
	def check_sampling_rule(self) -> bool:
		return True 

	#nice way to allow for different types of random models
	def model_instantiation(self) :
		class_name = self.factors['model type']
		module = importlib.import_module('simopt.solvers.TrustRegion.Models')
		return getattr(module, class_name)
	
	def polynomial_basis_instantiation(self) :
		class_name = self.factors['polynomial basis']
		module = importlib.import_module('simopt.solvers.active_subspaces.basis')
		return getattr(module, class_name)

	def sample_instantiation(self) :
		class_name = self.factors['sampling rule']
		module = importlib.import_module('simopt.solvers.TrustRegion.Sampling')
		sampling_instance = getattr(module, class_name)(self)
		return sampling_instance

	def geometry_type_instantiation(self) :
		class_name = self.factors['geometry instance']
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
		fval = model.fval
		# #If using the GP model we do expected improvement
		# if self.factors['model type'] == 'GPModel' : 
		# 	def subproblem(s) : 
		# 		return model.prediction(s) 

		# 	con_f = lambda s: norm(s)
		# 	nlc = NonlinearConstraint(con_f, 0, delta)
		# 	solve_subproblem = minimize(subproblem, np.zeros(problem.dim), method='trust-constr', constraints=nlc)
		# 	candidate_x =  new_x + solve_subproblem.x

		q, grad, Hessian = model.coefficients

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
				Hessian_matrix = self.construct_symmetric_matrix(Hessian)
				return q[0] + np.dot(s,grad) + np.dot(np.matmul(s,Hessian_matrix),s)
			
			con_f = lambda s: norm(s)
			nlc = NonlinearConstraint(con_f, 0, delta)
			solve_subproblem = minimize(subproblem, np.zeros(problem.dim), method='trust-constr', constraints=nlc)
			candidate_x =  new_x + solve_subproblem.x



		# handle the box constraints
		for i in range(problem.dim):
			if candidate_x[i] <= problem.lower_bounds[i]:
				candidate_x[i] = problem.lower_bounds[i] + 0.01
			elif candidate_x[i] >= problem.upper_bounds[i]:
				candidate_x[i] = problem.upper_bounds[i] - 0.01
		
		candidate_solution = self.create_new_solution(tuple(candidate_x), problem)
		if self.factors['model type'] == 'RandomModelReuse' :
			#we only append to the visited points list if we care about reusing points
			visited_pts_list.append(candidate_solution) 

		return candidate_solution, visited_pts_list

	def evaluate_candidate_solution(self, model, problem, fval_tilde, delta_k, interpolation_solns, current_solution, candidate_solution, recommended_solns) :
		"""
		Evaluate the candidate solution, by looking at the ratio comparison 
		
		Args:
			model (random_model): the local model
			delta_k (float): the current trust-region radius size
			candidate_solution (base.Solution): the current iterations candidate solution
			recommended_solns ([]): Description
		"""
		fval = model.fval
		stepsize = np.subtract(candidate_solution.x, current_solution.x)
		model_reduction = model.local_model_evaluate(np.zeros(problem.dim)) - model.local_model_evaluate(stepsize)
		if model_reduction <= 0:
			rho = 0
		else:
			# difference = np.subtract(candidate_solution.x, current_solution.x)
			rho = (np.array(fval[0]) - np.array(fval_tilde)) / model_reduction

		self.rho.append(rho)

		# successful: accept
		if rho >= self.factors['eta_1']:
			# new_x = candidate_x
			current_solution = candidate_solution
			# final_ob = candidate_solution.objectives_mean
			recommended_solns.append(candidate_solution)
			# intermediate_budgets.append(expended_budget)
			delta_k = min(delta_k, self.factors['delta_max'])
			
			# very successful: expand and accept
			if rho >= self.factors['eta_2'] :
				# new_x = candidate_x
				# current_solution = candidate_solution
				# final_ob = candidate_solution.objectives_mean
				# recommended_solns.append(candidate_solution)
				# intermediate_budgets.append(expended_budget)
				delta_k = min(self.factors['gamma_1'] * delta_k, self.factors['delta_max'])
			
		# unsuccessful: shrink and reject
		else:
			delta_k = min(self.factors['gamma_2'] * delta_k, self.factors['delta_max'])
			# new_solution = current_solution
			recommended_solns.append(current_solution)
			# final_ob = fval[0]

		return current_solution, delta_k, recommended_solns

	#solve the problem - inherited from base.Solver
	def solve(self, problem: Problem) -> tuple[list[Solution], list[int]] :
		recommended_solns = []
		intermediate_budgets = []
		expended_budget = 0
		delta_k = self.factors['delta']
		visited_pts_list = []

		new_x = problem.factors["initial_solution"]
		new_solution = self.create_new_solution(new_x, problem)
		recommended_solns.append(new_solution)
		intermediate_budgets.append(expended_budget)
		# model_construction_parameters = self.factors['model construction parameters']
		model_construction_parameters = {
			'structure': 'const', 
			'degree': self.factors['polynomial degree'],
			'nugget': 5, 
			'Lfixed': None, 
			'n_init': 1,
			'mu': 1000
		}
		# model_construction_parameters = {
		# 'w': 0.85, 
		# 'mu':1000,
		# 'beta':10, 
		# 'criticality_threshold': 0.1, 
		# 'skip_criticality': True,
		# 'lambda_min': self.factors['lambda_min']
		# }

		#Dynamically load in different sampling rule, geometry type, and random model
		sampling_instance = self.sample_instantiation()

		geometry_instance = self.geometry_type_instantiation()(problem)
		poly_basis_instance = self.polynomial_basis_instantiation()(self.factors['polynomial degree'], None, problem.dim)
		model = self.model_instantiation()(geometry_instance, self, poly_basis_instance, problem, sampling_instance, model_construction_parameters)
		

		k=0

		while expended_budget < problem.factors["budget"]:
			k += 1 

			#build random model 
			current_solution, delta_k, construction_budget, interpolation_solns, visited_pts_list, sample_size = model.construct_model(new_solution, delta_k, k, expended_budget, visited_pts_list)
			expended_budget = construction_budget # the additions to the expended budget is done in model.construct_model

			#solve random model 
			candidate_solution, visited_pts_list = self.solve_subproblem(delta_k, model, problem, current_solution, visited_pts_list)
			#adaptive sampling - need way to include additional parameters 
			if sampling_instance.sampling_rule.__class__.__name__ == 'adaptive_sampling' :
				problem.simulate(candidate_solution, 1)
				expended_budget += 1
				sample_size = 1

			candidate_solution, sampling_budget = sampling_instance(problem, candidate_solution, k, delta_k, expended_budget, sample_size, 0)
			fval_tilde = -1 * problem.minmax[0] * candidate_solution.objectives_mean
			expended_budget = sampling_budget

			#evaluate model
			# model, problem, fval_tilde, delta_k, interpolation_solns, candidate_solution, recommended_solns
			new_solution, delta_k, recommended_solns = self.evaluate_candidate_solution(model, problem, fval_tilde, delta_k, interpolation_solns, current_solution,\
																			   candidate_solution, recommended_solns)	
			

			intermediate_budgets.append(expended_budget)


			# print('new solution: ', new_solution.x)

		return recommended_solns, intermediate_budgets
	

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
				"description": "tolerance value to check for updating the interpolation model",
				"datatype": float, 
				"default": 0.01
			},
			"initial radius": {
				"description": "initial trust-region radius",
				"datatype": float, 
				"default": 0.0
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
				"default": 2
			}
		}
		super().factors['geometry instance'] = 'OMoRFGeometry'
		super().factors['polynomial basis'] = 'NaturalPolynomialBasis'
		return {**super().specifications, **new_specifications}
	
	@property
	def check_factor_list(self) -> dict[str, Callable] : 
		new_check_list = {
				"interpolation update tol":self.check_tolerance,
				"initial radius": self.check_initial_radius,
				"gamma_3": self.check_gamma_3,
				"gamma_shrinking": self.check_gamma_shrinking,
				"omega_shrinking": self.check_omega_shrinking,
				"subspace dimension": self.check_dimension_reduction,
			}
		return {**super().check_factor_list, **new_check_list}

	
	def check_tolerance(self) -> bool:
		return self.factors['interpolation update tol'] >0 and self.factors['interpolation update tol'] <= 1
	
	def check_initial_radius(self) -> bool:
		return self.factors['initial radius'] > 0
	
	def check_gamma_3(self) -> bool:
		return self.factors['gamma_3'] > 0
	
	def check_gamma_shrinking(self) -> bool:
		return self.factors['gamma_shrinking'] > 0
	
	def check_omega_shrinking(self) -> bool:
		return self.factors['omega_shrinking'] > 0
	
	def check_dimension_reduction(self) -> bool:
		return self.factors['subspace dimension'] >= 1

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

	def solve_subproblem(self, delta: float, rho_k: float, expended_budget:int, problem: Problem, solution: Solution, subspace_int_set:list[np.ndarray], model_int_set: list[np.ndarray], fvals: list[float], grad_fval: list[float]) -> tuple[Solution, float, float, int, list, np.ndarray, list, list, list, bool]:
		"""
		Solve the Trust-Region subproblem either using Cauchy reduction or a black-box optimisation solver
		
		Args:
			model (PolynomialRidgeApproximation): the polynomial ridge approximation model
			problem (base.Problem): the simulation-optimisation Problem
			solution (base.Solution): the current iteration's solution

		Returns:
			base.Solution - the candidate solution from the subproblem
		"""
		new_x = np.array(solution.x)
		grad, Hessian = self.local_model.grad(new_x), self.local_model.hessian(new_x)
		# grad, Hessian = np.zeros(new_x.shape),np.zeros(new_x.shape)
		gamma_s = self.factors['gamma_shrinking']
		omega_s = self.factors['omega_shrinking']
		reset_flag = False


		if self.factors['easy_solve'] :
			# Cauchy reduction
			if np.dot(np.multiply(grad, Hessian), grad) <= 0:
				tau = 1
			else:
				tau = min(1, norm(grad) ** 3 / (delta * np.dot(np.multiply(grad, Hessian), grad)))
			grad = np.reshape(grad, (1, problem.dim))[0]
			candidate_x = new_x - tau * delta * grad / norm(grad)

		
		else:
			# Search engine - solve subproblem
			def subproblem(s) : 
				res =  self.local_model.eval(new_x + s)
				return res
			
			con_f = lambda s: norm(s)
			nlc = NonlinearConstraint(con_f, 0, delta)

			solve_subproblem = minimize(subproblem, np.zeros(problem.dim), method='trust-constr', constraints=nlc)
			candidate_x =  new_x + np.array(solve_subproblem.x) 


		# handle the box constraints
		for i in range(problem.dim):
			if candidate_x[i] <= problem.lower_bounds[i]:
				candidate_x[i] = problem.lower_bounds[i] + 0.01
			elif candidate_x[i] >= problem.upper_bounds[i]:
				candidate_x[i] = problem.upper_bounds[i] - 0.01

		candidate_solution = self.create_new_solution(tuple(candidate_x), problem) 
		
		# check if the stepsize is too small
		if norm(solve_subproblem.x) < rho_k * gamma_s :
			delta = max(delta*omega_s, rho_k)
			model_int_set, subspace_int_set, fvals, grad_fval, rho_k, delta, expended_budget = self.interpolation_update(solution, problem, expended_budget, model_int_set, subspace_int_set, fvals, grad_fval, delta, rho_k, True)
			reset_flag = True
			candidate_solution = solution
			self.deltas.append(delta)


		return candidate_solution, rho_k, delta, expended_budget, subspace_int_set, model_int_set, fvals, grad_fval, reset_flag
	
	def evaluate_candidate_solution(self, problem: Problem, current_fval: float, fval_tilde: float, delta_k: float, rho_k: float, current_solution: Solution, candidate_solution: Solution, recommended_solns: list[Solution], model_int_set: list[np.ndarray], subspace_int_set: list[np.ndarray], fvals: list[float], grad_fval: list[np.ndarray]) -> tuple[Solution, float, float, list[Solution], list[np.ndarray], list[np.ndarray], np.ndarray, list[float], list[np.ndarray], float, int]:
		# fval = model.fval
		expended_budget = 0
		current_x = np.array(current_solution.x)
		candidate_x = np.array(candidate_solution.x)
		step_size = np.subtract(candidate_x,current_x)

		print(f'candidate value: {candidate_x}')
		print(f'incumbent value: {current_x}')

		print(f'current best function value: {current_fval}')
		print(f'candidate function value: {fval_tilde}')



		numerator = current_fval - fval_tilde
		denominator = self.local_model.eval(current_x) - self.local_model.eval(candidate_x)
		# denominator = self.local_model.eval(np.zeros(problem.dim)) - self.local_model.eval(step_size)
		print(f'numerator: {numerator}')
		print(f'denominator: {denominator}')
		if denominator > 0 : 
			ratioComparison = numerator / denominator
		else : 
			ratioComparison = 0
		
		self.rhos.append(ratioComparison)
		if ratioComparison >= self.factors['eta_1']:
			current_solution = candidate_solution
			current_fval = fval_tilde
			recommended_solns.append(candidate_solution)
			delta_k = max(self.factors['gamma_1']*delta_k, norm(step_size),rho_k)
			
			# very successful: expand and accept
			if ratioComparison >= self.factors['eta_2'] :
				delta_k = max(self.factors['gamma_2'] * delta_k, self.factors['gamma_3']*norm(step_size))
			
		# unsuccessful: shrink and reject
		else:
			delta_k = max(min(self.factors['gamma_1']*delta_k, norm(step_size)), rho_k)
			recommended_solns.append(current_solution)

		#append the candidate solution to the interpolation set and active subspace set
		model_int_set = np.vstack((model_int_set, candidate_x))
		subspace_int_set = np.vstack((subspace_int_set, candidate_x))

		#append candidate_x to fvals
		fvals = np.vstack((fvals,fval_tilde))
		
		#calculate the gradient at fval_tilde and 
		grad_append, budget = self.finite_difference_gradient(candidate_solution, problem)
		grad_fval = np.vstack((grad_fval, grad_append))
		expended_budget += budget



		if ratioComparison >= self.factors['eta_1'] :
			print('ITERATION WAS SUCCESSFUL')
			model_int_set, budget = self.geometry_improvement(model_int_set, current_solution, delta_k, False)
			expended_budget += budget
			subspace_int_set, budget =  self.geometry_improvement(subspace_int_set, current_solution, delta_k, False)
			expended_budget += budget

			#update the fvals for the new subspace interpolation set 
			fvals, expended_budget = self.get_fvals(model_int_set,problem, expended_budget) 
			grad_fval, expended_budget = self.get_grad_fvals(subspace_int_set, problem, expended_budget)

		else :
			print('ITERATION WAS UNSUCCESSFUL')
			model_int_set, subspace_int_set, fvals, grad_fval, rho_k, delta_k, budget = self.interpolation_update(current_solution, problem, expended_budget, model_int_set, subspace_int_set, fvals, grad_fval, delta_k, rho_k, True)
			expended_budget += budget

		
		self.deltas.append(delta_k)

		return current_solution, current_fval, delta_k, recommended_solns, model_int_set, subspace_int_set, fvals, grad_fval, rho_k, expended_budget


	def finite_difference_gradient(self, new_solution: Solution, problem: Problem) -> tuple[np.ndarray, int] :
		"""Calculate the finite difference gradient of the problem at new_solution.

		Args:
			new_solution (Solution): The solution at which to calculate the gradient.
			problem (Problem): The problem`that contains the function to differentiate.

		Returns:
			np.ndarray: The solution value of the gradient 

			int: The expended budget 
		"""
		budget = 0
		alpha = 1e-3		
		lower_bound = problem.lower_bounds
		upper_bound = problem.upper_bounds
		# new_solution = self.create_new_solution(tuple(x), problem)

		new_x = new_solution.x
		FnPlusMinus = np.zeros((problem.dim, 3))
		grad = np.zeros(problem.dim)
		for i in range(problem.dim):
			# Initialization.
			x1 = list(new_x)
			x2 = list(new_x)
			steph1 = alpha
			steph2 = alpha

			# Check variable bounds.
			if x1[i] + steph1 > upper_bound[i]:
				steph1 = np.abs(upper_bound[i] - x1[i])
			if x2[i] - steph2 < lower_bound[i]:
				steph2 = np.abs(x2[i] - lower_bound[i])

			# Decide stepsize.
			FnPlusMinus[i, 2] = min(steph1, steph2)
			x1[i] = x1[i] + FnPlusMinus[i, 2]
			x2[i] = x2[i] - FnPlusMinus[i, 2]

			fn1, fn2 = 0,0 
			x1_solution = self.create_new_solution(tuple(x1), problem)
			problem.simulate_up_to([x1_solution], 1)
			budget += 1
			fn1 = -1 * problem.minmax[0] * x1_solution.objectives_mean
			# First column is f(x+h,y).
			FnPlusMinus[i, 0] = fn1
			
			x2_solution = self.create_new_solution(tuple(x2), problem)
			problem.simulate_up_to([x2_solution], 1)
			budget += 1
			fn2 = -1 * problem.minmax[0] * x2_solution.objectives_mean
			# Second column is f(x-h,y).
			FnPlusMinus[i, 1] = fn2

			# Calculate gradient.
			grad[i] = (fn1 - fn2) / (2 * FnPlusMinus[i, 2])

		return grad, budget

	def get_fvals(self, samples: list[np.ndarray], problem: Problem, budget: int) -> tuple[np.ndarray, int] : 
		fvals = []
		for s in samples :
			s = [a for a in s.flatten()]
			new_solution = self.create_new_solution(tuple(s), problem)
			problem.simulate(new_solution,1)
			budget += 1
			fvals.append(-1 * problem.minmax[0] * new_solution.objectives_mean)
			

		# print("grads: ", np.array(grads).shape)
			
		return np.array(fvals).reshape((-1,1)), budget
	

	def get_grad_fvals(self, samples: list[np.ndarray], problem:Problem, budget:int) -> tuple[np.ndarray, int] :
		grads = []
		for s in samples :
			s = [a for a in s.flatten()]
			new_solution = self.create_new_solution(tuple(s), problem)
			grad, expended_budget = self.finite_difference_gradient(new_solution, problem)
			grads.append(grad)
			budget += expended_budget
	
		return np.array(grads).reshape((-1,problem.dim)), budget
	



	def generate_samples(self, lower_bound, upper_bound, new_x, problem) : 
		samples = [] 
		# degree = self.factors['polynomial degree']
		# subspace_dim = self.factors['subspace dimension']
		dim = self.factors['subspace dimension']
		no_samples = dim  #comb(subspace_dim + degree, degree) + dim*subspace_dim - (subspace_dim*(subspace_dim+1))//2

		while len(samples) < no_samples:
			sample = np.random.normal(size=(len(new_x),)) 
			#check bounds if inside then append 
			if np.all(sample >= lower_bound) and np.all(sample <= upper_bound) : 
				samples.append(sample)

		return np.array(samples)


	def solve(self, problem):
		"""
		Run a single macroreplication of a solver on a problem.
		
		Arguments
		---------
		problem : Problem object
			simulation-optimization problem to solve
		
		Returns
		-------
		recommended_solns : list of Solution objects
			list of solutions recommended throughout the budget
		intermediate_budgets : list of ints
			list of intermediate budgets when recommended solutions changes
		
		Deleted Parameters
		------------------
		crn_across_solns : bool
			indicates if CRN are used when simulating different solutions
		"""
		recommended_solns = []
		intermediate_budgets = []
		expended_budget = 0
		delta_k = self.factors['delta']
		rho_k = delta_k 
		self.rhos = []
		self.deltas = []

		current_x = problem.factors["initial_solution"]
		print(f'initial solution: {current_x}')
		current_solution = self.create_new_solution(current_x, problem)
		recommended_solns.append(current_solution)
		intermediate_budgets.append(expended_budget)

		#evaluate the current solution
		problem.simulate(current_solution,1)
		expended_budget += 1
		current_fval = -1 * problem.minmax[0] * current_solution.objectives_mean

		sampling_instance = self.sample_instantiation()

		self.poly_basis = self.polynomial_basis_instantiation()(self.factors['polynomial degree'], dim=self.factors['subspace dimension'])
		geometry_instance = self.geometry_type_instantiation()(problem)

		#instantiate ActiveSubspace and use it to construct the active subspace matrix 
		self.active_subspace_matrix = ActiveSubspace()



		subspace_int_set = self.generate_samples(problem.lower_bounds, problem.upper_bounds, current_x, problem)


		#get function values of the interpolation set
		grad_fval, expended_budget = self.get_grad_fvals(subspace_int_set, problem, expended_budget)
		print(f'gradient of the function values of the subspace interpolation set shape: {grad_fval.shape}')
		self.active_subspace_matrix.fit(grad_fval)

		#construct an initial subspace matrix using subspace_int_set

		print('subspace matrix shape: ', self.active_subspace_matrix._U.shape)

		#build an initial set of interpolation points
		model_int_set = geometry_instance.interpolation_points(np.array(current_x),delta_k, self.factors['polynomial degree'], self.factors['subspace dimension'] )
		model_int_set = np.array(model_int_set).reshape(-1, problem.dim)


		fvals, expended_budget = self.get_fvals(model_int_set, problem, expended_budget)

		#initialise the polyridge model 
		self.local_model = PolynomialRidgeApproximation(self.factors['polynomial degree'], self.factors['subspace dimension'], problem, self.poly_basis)
		print(f'model_int_set shape: {model_int_set.shape}')
		print(f'fvals shape: {fvals.shape}')
		self.local_model.fit(model_int_set, fvals, U0=self.active_subspace_matrix._U)


		k=0
		reset_counter = 0

		while expended_budget < problem.factors['budget']:
			print(f'iteration number {k} ')
			
			if k == 0 :
				print('shape of subspace_int_set: ', subspace_int_set.shape)
				print('shape of model int set: ', model_int_set.shape)
				print('shape of fvals: ', fvals.shape)
				print(f'model interpolation set: {model_int_set}')
			
			if k > 0 :
				#construct the model using the interpolation set
				print('shape of model_int_set: ', model_int_set.shape)
				print('subspace matrix: ', self.active_subspace_matrix._U)
				print('shape of fvals: ', fvals.shape)
				print(f'model_int_set: {model_int_set}')
				print(f'fvals: {fvals}')
				self.local_model.fit(model_int_set, fvals, U0=self.active_subspace_matrix._U)

			#solve the subproblem

			candidate_solution, rho_k, delta_k, expended_budget, subspace_int_set, model_int_set, fvals, grad_fval, reset_flag = self.solve_subproblem(delta_k, rho_k, expended_budget, problem, current_solution, subspace_int_set, model_int_set, fvals, grad_fval) 
			#if the stepsize is too small, do not evaluate and restart the loop
			if reset_flag :
				print('RESETTING')
				reset_counter += 1 					
				print(f'candidate solution: {candidate_solution.x}')
				print(f'current solution: {current_solution.x}')
				print(f'model_int_set shape: {model_int_set.shape}')
				fvals, expended_budget = self.get_fvals(model_int_set, problem, expended_budget)
				grad_fval, expended_budget = self.get_grad_fvals(subspace_int_set, problem, expended_budget)
				print(f'fvals shape: {fvals.shape}')
				print(f'grad_fvals shape: {grad_fval.shape}')
				recommended_solns.append(current_solution)
				intermediate_budgets.append(expended_budget) 
				if reset_counter >= 4 : 
					print('END PROCESS')
					break

				continue
			else : 
				#evaluate the candidate solution
				candidate_solution, sampling_budget = sampling_instance(problem, candidate_solution, k, delta_k, expended_budget, 1, 0)
				fval_tilde = -1 * problem.minmax[0] * candidate_solution.objectives_mean
				expended_budget = sampling_budget

				#evaluate the candidate solution
				current_solution, current_fval, delta_k, recommended_solns, model_int_set, subspace_int_set, fvals, grad_fval, rho_k, budget = self.evaluate_candidate_solution(problem, current_fval, fval_tilde, delta_k, rho_k, current_solution, candidate_solution, recommended_solns, model_int_set, subspace_int_set, fvals, grad_fval)
				expended_budget += budget

				self.active_subspace_matrix.fit(grad_fval)
				
				intermediate_budgets.append(expended_budget)
			k += 1
			print('rhos: \n', self.rhos)
			print('deltas: \n', self.deltas)
			
		return recommended_solns, intermediate_budgets
		

	"""
		p - 
		d - subspace dimension
		q -  number of interpolation points in model interpolation
		n - number of interpolation points in subspace interpolation
		S - 
		delta_k - trust_region_radius
		eta_1 - successful iteration thresehold
		eta_2 - very successful iteration threshold 
		epsilon_1 - 
		epsilon_2 - 
		rho_k - ratio comparison value 
		f - 
		s_old - 
		f_old - 


	"""

	def _update_geometry_omorf(self, S_full, f_full, S_red, f_red):
		dist = max(self.epsilon_1*self.del_k, self.epsilon_2*self.rho_k)
		if max(np.linalg.norm(S_full-self.s_old, axis=1, ord=np.inf)) > dist:
			S_full, f_full = self._sample_set('improve', S_full, f_full)
			try:
				pass 
				# grad_full = self.get_grad_fvals(f_full, problem=, budget=)
				# self.active_subspace_matrix(grad_full)
			except:
				pass
		elif max(np.linalg.norm(S_red-self.s_old, axis=1, ord=np.inf)) > dist:
			S_red, f_red = self._sample_set('improve', S_red, f_red, full_space=False)
		elif self.del_k == self.ratioComparison:
			self._set_del_k(self.alpha_2*self.ratioComparison)
			if self.count >= 3 and self.r_k < 0:
				if self.ratioComparison >= 250*self.rho_min:
					self.ratioComparison = (self.alpha_1*self.rho_k)
				elif 16*self.rho_min < self.rho_k < 250*self.rho_min:
					self.ratioComparison = np.sqrt(self.rho_k*self.rho_min)
				else:
					self.ratioComparison = self.rho_min
		return S_full, f_full, S_red, f_red
	

	def _sample_set(self, method, S=None, f=None, s_new=None, f_new=None, full_space=True):
		if full_space:
			q = self.p
		else:
			q = self.q
		dist = max(self.epsilon_1*self.del_k, self.epsilon_2*self.rho_k)
		if method == 'replace':
			S_hat = np.vstack((S, s_new))
			f_hat = np.vstack((f, f_new))
			if S_hat.shape != np.unique(S_hat, axis=0).shape:
				S_hat, indices = np.unique(S_hat, axis=0, return_index=True)
				f_hat = f_hat[indices]
			elif f_hat.size > q and max(np.linalg.norm(S_hat-self.s_old, axis=1, ord=np.inf)) > dist:
				S_hat, f_hat = self._remove_furthest_point(S_hat, f_hat, self.s_old)
			S_hat, f_hat = self._remove_point_from_set(S_hat, f_hat, self.s_old)
			S = np.zeros((q, self.n))
			f = np.zeros((q, 1))
			S[0, :] = self.s_old
			f[0, :] = self.f_old
			S, f = self._LU_pivoting(S, f, S_hat, f_hat, full_space)
		elif method == 'improve':
			S_hat = np.copy(S)
			f_hat = np.copy(f)
			if max(np.linalg.norm(S_hat-self.s_old, axis=1, ord=np.inf)) > dist:
				S_hat, f_hat = self._remove_furthest_point(S_hat, f_hat, self.s_old)
			S_hat, f_hat = self._remove_point_from_set(S_hat, f_hat, self.s_old)
			S = np.zeros((q, self.n))
			f = np.zeros((q, 1))
			S[0, :] = self.s_old
			f[0, :] = self.f_old
			S, f = self._LU_pivoting(S, f, S_hat, f_hat, full_space, 'improve')
		elif method == 'new':
			S_hat = f_hat = np.array([])
			S = np.zeros((q, self.n))
			f = np.zeros((q, 1))
			S[0, :] = self.s_old
			f[0, :] = self.f_old
			S, f = self._LU_pivoting(S, f, S_hat, f_hat, full_space, 'new')
		return S, f
	
	@staticmethod
	def _remove_point_from_set(S, f, s):
		ind_current = np.where(np.linalg.norm(S-s, axis=1, ord=np.inf) == 0.0)[0]
		S = np.delete(S, ind_current, 0)
		f = np.delete(f, ind_current, 0)
		return S, f

	@staticmethod
	def _remove_furthest_point(S, f, s):
		ind_distant = np.argmax(np.linalg.norm(S-s, axis=1, ord=np.inf))
		S = np.delete(S, ind_distant, 0)
		f = np.delete(f, ind_distant, 0)
		return S, f
	
	def _blackbox_evaluation(self, s):
		"""
		Evaluates the point s for ``trust-region`` or ``omorf`` methods
		"""
		s = s.reshape(1,-1)
		if self.S.size > 0 and np.unique(np.vstack((self.S, s)), axis=0).shape[0] == self.S.shape[0]:
			ind_repeat = np.argmin(np.linalg.norm(self.S - s, ord=np.inf, axis=1))
			f = self.f[ind_repeat]
		else:
			f = np.array([[self.objective['function'](self._remove_scaling(s.flatten()))]])
			self.num_evals += 1
			if self.f.size == 0:
				self.S = s
				self.f = f
			else:
				self.S = np.vstack((self.S, s))
				self.f = np.vstack((self.f, f))
		return np.asscalar(f)
	
	def _LU_pivoting(self, S, f, S_hat, f_hat, full_space, method=None):
		psi_1 = 1.0e-4
		if full_space:
			psi_2 = 1.0
		else:
			psi_2 = 0.25
		phi_function, phi_function_deriv = self._get_phi_function_and_derivative(S_hat, full_space)
		if full_space:
			q = self.p
		else:
			q = self.q
#       Initialise U matrix of LU factorisation of M matrix (see Conn et al.)
		U = np.zeros((q,q))
		U[0,:] = phi_function(self.s_old)
#       Perform the LU factorisation algorithm for the rest of the points
		for k in range(1, q):
			flag = True
			v = np.zeros(q)
			for j in range(k):
				v[j] = -U[j,k] / U[j,j]
			v[k] = 1.0
#           If there are still points to choose from, find if points meet criterion. If so, use the index to choose 
#           point with given index to be next point in regression/interpolation set
			if f_hat.size > 0:
				M = np.absolute(np.dot(phi_function(S_hat),v).flatten())
				index = np.argmax(M)
				if M[index] < psi_1:
					flag = False
				elif method == 'improve' and (k == q - 1 and M[index] < psi_2):
					flag = False
				elif method == 'new' and M[index] < psi_2:
					flag = False
			else:
				flag = False
#           If index exists, choose the point with that index and delete it from possible choices
			if flag:
				s = S_hat[index,:]
				S[k, :] = s
				f[k, :] = f_hat[index]
				S_hat = np.delete(S_hat, index, 0)
				f_hat = np.delete(f_hat, index, 0)
#           If index doesn't exist, solve an optimisation problem to find the point in the range which best satisfies criterion
			else:
				try:
					s = self._find_new_point(v, phi_function, phi_function_deriv, full_space)
					if np.unique(np.vstack((S[:k, :], s)), axis=0).shape[0] != k+1:
						s = self._find_new_point_alternative(v, phi_function, S[:k, :])
				except:
					s = self._find_new_point_alternative(v, phi_function, S[:k, :])
				if f_hat.size > 0 and M[index] >= abs(np.dot(v, phi_function(s))):
					s = S_hat[index,:]
					S[k, :] = s
					f[k, :] = f_hat[index]
					S_hat = np.delete(S_hat, index, 0)
					f_hat = np.delete(f_hat, index, 0)
				else:
					S[k, :] = s
					f[k, :] = self._blackbox_evaluation(s)
#           Update U factorisation in LU algorithm
			phi = phi_function(s)
			U[k,k] = np.dot(v, phi)
			for i in range(k+1,q):
				U[k,i] += phi[i]
				for j in range(k):
					U[k,i] -= (phi[j]*U[j,i]) / U[j,j]
		return S, f

	def _find_new_point(self, v, phi_function, phi_function_deriv, full_space=False):
		bounds = []
		for i in range(self.n):
			bounds.append((self.bounds_l[i], self.bounds_u[i])) 
		if full_space:
			c = v[1:]
			res1 = optimize.linprog(c, bounds=bounds)
			res2 = optimize.linprog(-c, bounds=bounds)
			if abs(np.dot(v, phi_function(res1['x']))) > abs(np.dot(v, phi_function(res2['x']))):
				s = res1['x']
			else:
				s = res2['x']
		else:
			obj1 = lambda s: np.dot(v, phi_function(s))
			jac1 = lambda s: np.dot(phi_function_deriv(s), v)
			obj2 = lambda s: -np.dot(v, phi_function(s))
			jac2 = lambda s: -np.dot(phi_function_deriv(s), v)
			res1 = optimize.minimize(obj1, self.s_old, method='TNC', jac=jac1, \
					bounds=bounds, options={'disp': False})
			res2 = optimize.minimize(obj2, self.s_old, method='TNC', jac=jac2, \
					bounds=bounds, options={'disp': False})
			if abs(res1['fun']) > abs(res2['fun']):
				s = res1['x']
			else:
				s = res2['x']
		return s

	def _find_new_point_alternative(self, v, phi_function, S):
		S_tmp = self._generate_set(int(0.5*(self.n+1)*(self.n+2)))
		M = np.absolute(np.dot(phi_function(S_tmp), v).flatten())
		indices = np.argsort(M)[::-1][:len(M)]
		for index in indices:
			s = S_tmp[index,:]
			if np.unique(np.vstack((S, s)), axis=0).shape[0] == S.shape[0]+1:
				return s
		return S_tmp[indices[0], :]
	
	def _get_phi_function_and_derivative(self, S_hat, full_space):
		Del_S = self.del_k
		if full_space:
			if S_hat.size > 0:
				Del_S = max(np.linalg.norm(S_hat-self.s_old, axis=1, ord=np.inf))
			def phi_function(s):
				s_tilde = np.divide((s - self.s_old), Del_S)
				try:
					m,n = s_tilde.shape
				except:
					m = 1
					s_tilde = s_tilde.reshape(1,-1)
				phi = np.zeros((m, self.p))
				phi[:, 0] = 1.0
				phi[:, 1:] = s_tilde
				if m == 1:
					return phi.flatten()
				else:
					return phi
			phi_function_deriv = None
		else :
			if S_hat.size > 0:
				Del_S = max(np.linalg.norm(np.dot(S_hat-self.s_old,self.U), axis=1))
			def phi_function(s):
				u = np.divide(np.dot((s - self.s_old), self.U), Del_S)
				try:
					m,n = u.shape
				except:
					m = 1
					u = u.reshape(1,-1)
				phi = np.zeros((m, self.q))
				for k in range(self.q):
					phi[:,k] = np.prod(np.divide(np.power(u, self.basis[k,:]), factorial(self.basis[k,:])), axis=1)
				if m == 1:
					return phi.flatten()
				else:
					return phi
			def phi_function_deriv(s):
				u = np.divide(np.dot((s - self.s_old), self.U), Del_S)
				phi_deriv = np.zeros((self.d, self.q))
				for i in range(self.d):
					for k in range(1, self.q):
						if self.basis[k, i] != 0.0:
							tmp = np.zeros(self.d)
							tmp[i] = 1
							phi_deriv[i,k] = self.basis[k, i] * np.prod(np.divide(np.power(u, self.basis[k,:]-tmp), \
									factorial(self.basis[k,:])))
				phi_deriv = np.divide(phi_deriv.T, Del_S).T
				return np.dot(self.U, phi_deriv)
		return phi_function, phi_function_deriv

