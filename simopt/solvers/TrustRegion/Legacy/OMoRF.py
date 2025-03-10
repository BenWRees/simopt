#type: ignore 
"""mini-batch SGD 

TODO: add bounds in

"""
from __future__ import annotations
from typing import Callable

import warnings

#import ridge function approximation from PSDR-Master
import sys
import importlib
import inspect
import cvxpy as cp
import numpy as np 
from numpy.linalg import norm
from scipy.optimize import minimize, NonlinearConstraint
from math import comb
from copy import deepcopy

sys.path.append('~/Desktop/Simopt')

warnings.filterwarnings("ignore")

from simopt.base import (
	Solution, 
	Problem,
	ConstraintType,
	ObjectiveType,
	Solver,
	VariableType,
)
from simopt.solvers.active_subspaces.polyridge import *
from simopt.solvers.active_subspaces.basis import *
from simopt.solvers.active_subspaces.subspace import ActiveSubspace
from simopt.solvers.trust_region_class import sampling_rule

class OMoRF(Solver):
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
		return {
			"crn_across_solns": {
				"description": "use CRN across solutions?",
				"datatype": bool,
				"default": True
			},
			"easy_solve": {
				"description": "solve the subproblem approximately with Cauchy point",
				"datatype": bool,
				"default": False
			},
			"interpolation update tol":{
				"description": "tolerance value to check for updating the interpolation model",
				"datatype": float, 
				"default": 0.01
			},
			"eta_1" : {
				"description": "threshold for successful iteration",
				"datatype": float, 
				"default": 0.1
			}, 
			"eta_2": {
				"description": "threshold for very successful iteration",
				"datatype": float, 
				"default": 0.7

			},
			"initial radius": {
				"description": "initial trust-region radius",
				"datatype": float, 
				"default": 0.0
			}, 
			"delta": {
				"description": "size of the trust-region radius",
				"datatype": float,
				"default": 5.0
			}, 
			"delta_max": {
				"description": "largest trust-region radius",
				"datatype": float, 
				"default": 0.0
			},
			"gamma_1": {
				"description": "trust-region radius increase rate after a successful iteration",
				"datatype": float,
				"default": 1.5
			},
			"gamma_2": {
				"description": "trust-region radius decrease rate after an unsuccessful iteration",
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
				"default": 2
			}, 
			"polynomial degree": {
				"description": "degree of the polynomial",
				"datatype": int, 
				"default": 2
			},
			"geometry_instance": {
				"description": "Instance of the geometric behaviours of the space where trust region values are sampled from",
				"datatype": str,
				"default": "simopt.solvers.OMoRF:trust_region_interpolation_points"
			},
			"poly_basis": {
				"description": "Polynomial basis to use in model construction",
				"datatype": str, 
				"default": "MonomialPolynomialBasis"
			}, 
			"sampling_rule" : {
				"description": "An instance of the sampling rule being used",
				"datatype": str,
				"default": 'simopt.solvers.trust_region_class:basic_sampling' #just returns 10 every time
			},
		}
	
	@property
	def check_factor_list(self) -> dict[str, Callable] : 
		return {
			"crn_across_solns": self.check_crn_across_solns,
			"easy_solve": self.check_easy_solve,
			"interpolation update tol":self.check_tolerance,
			"eta_1": self.check_eta_1,
			"eta_2": self.check_eta_2,
			"initial radius": self.check_initial_radius,
			"delta": self.check_delta,
			"delta_max": self.check_delta_max,
			"gamma_1": self.check_gamma_1,
			"gamma_2": self.check_gamma_2,
			"gamma_3": self.check_gamma_3,
			"gamma_shrinking": self.check_gamma_shrinking,
			"omega_shrinking": self.check_omega_shrinking,
			"subspace dimension": self.check_dimension_reduction,
			"polynomial degree": self.check_polynomial_degree,
			"geometry_instance": self.check_geometry_instance,
			"poly_basis": self.check_poly_basis,
			"sampling_rule": self.check_sampling_rule
		}
	
	def check_crn_across_solns(self) -> bool:
		return True
	
	def check_easy_solve(self) -> bool:
		return True
	
	def check_tolerance(self) -> bool:
		return self.factors['interpolation update tol'] >0 and self.factors['interpolation update tol'] <= 1
	
	def check_eta_1(self) -> bool:
		return self.factors['eta_1'] > 0
	
	def check_eta_2(self) -> bool:
		return self.factors['eta_2'] > self.factors['eta_1'] and self.factors['eta_2'] > 0
	
	def check_initial_radius(self) -> bool:
		return self.factors['initial radius'] > 0
	
	def check_delta(self) -> bool:
		return self.factors['delta'] > 0
	
	def check_delta_max(self) -> bool:
		return self.factors['delta_max'] > self.factors['delta'] and self.factors['delta_max'] > 0
	
	def check_gamma_1(self) -> bool:
		return self.factors['gamma_1'] > 0
	
	def check_gamma_2(self) -> bool:
		return self.factors['gamma_2'] > 0
	
	def check_gamma_3(self) -> bool:
		return self.factors['gamma_3'] > 0
	
	def check_gamma_shrinking(self) -> bool:
		return self.factors['gamma_shrinking'] > 0
	
	def check_omega_shrinking(self) -> bool:
		return self.factors['omega_shrinking'] > 0
	
	def check_dimension_reduction(self) -> bool:
		return self.factors['subspace dimension'] >= 1
	
	def check_polynomial_degree(self) -> bool:
		return self.factors['polynomial degree'] >= 1
	
	def check_geometry_instance(self) -> bool:
		try:
			module_name, _ = self.factors['geometry_instance'].split(':')
			module = importlib.import_module(module_name)
			if hasattr(module, self.factors['geometry_instance']):
				attr = getattr(module, self.factors['geometry_instance'])
				return inspect.isclass(attr)
			return False
		except ModuleNotFoundError:
			return False
	
	def check_poly_basis(self) -> bool:
		try :
			module = importlib.import_module('simopt.solvers.active_subspaces.basis')
			if hasattr(module, self.factors['poly_basis']):
				attr = getattr(module, self.factors['poly_basis'])
				return inspect.isclass(attr)
			return False
		except ModuleNotFoundError:
			return False
		
	def check_sampling_rule(self) -> bool:
		try:
			module_name, _ = self.factors['sampling_rule'].split(':')
			module = importlib.import_module(module_name)
			if hasattr(module, self.factors['sampling_rule']):
				attr = getattr(module, self.factors['sampling_rule'])
				return inspect.isclass(attr)
			return False
		except ModuleNotFoundError:
			return False

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
		super().__init__(name, fixed_factors)
	
	def polynomial_basis_instantiation(self) -> Callable:
		class_name = self.factors['poly_basis']
		module = importlib.import_module('simopt.solvers.active_subspaces.basis')
		return getattr(module, class_name)

	def geometry_type_instantiation(self) -> Callable:
		module_name, class_name = self.factors['geometry_instance'].split(':')
		module = importlib.import_module(module_name)
		return getattr(module, class_name)
	
	def sample_instantiation(self) -> sampling_rule:
		module_name, class_name = self.factors['sampling_rule'].split(':')
		module = importlib.import_module(module_name)
		sampling_instance = getattr(module, class_name)(self)
		return sampling_rule(self, sampling_instance)

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


	def finite_difference_gradient(self, new_solution: Solution, problem: Problem) -> tuple(np.ndarray, int) :
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

	def get_fvals(self, samples: list[np.ndarray], problem: Problem, budget: int) -> tuple(np.ndarray, int) : 
		fvals = []
		for s in samples :
			s = [a for a in s.flatten()]
			new_solution = self.create_new_solution(tuple(s), problem)
			problem.simulate(new_solution,1)
			budget += 1
			fvals.append(-1 * problem.minmax[0] * new_solution.objectives_mean)
			

		# print("grads: ", np.array(grads).shape)
			
		return np.array(fvals).reshape((-1,1)), budget
	

	def get_grad_fvals(self, samples: list[np.ndarray], problem:Problem, budget:int) -> tuple(np.ndarray, int) :
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


	#This is the update for the interpolation sets as defined in the paper
	def interpolation_update(self, current_solution, problem, expended_budget, interpolation_set, active_subspace_set, fvals, grad_fval, delta_k, rho_k, geometry_improving_flag) : 
		
		tol = min(2*delta_k, 10*rho_k) #!!should be a specification parameter
		# tol = self.factors['interpolation update tolerance']

		current_x = np.array(current_solution.x)
		# for elem_int, elem_as in zip(interpolation_set, active_subspace_set)  : 
		interpolation_points_dist_current_x = [norm(current_x - elem_int) for elem_int in interpolation_set]
		subspace_points_dist_current_x = [norm(current_x - elem_as) for elem_as in active_subspace_set]

		if max(interpolation_points_dist_current_x) > tol : 
			#geometry inmproving algorithm on interpolation_set
			print('GEOMETRY IMPROVING ON MODEL INTERPOLATION SET')
			interpolation_set, budget = self.geometry_improvement(interpolation_set, current_solution, delta_k, geometry_improving_flag)
			expended_budget += budget

			fvals, _ = self.get_fvals(interpolation_set, problem, expended_budget)
			expended_budget += 1

		
		elif max(subspace_points_dist_current_x) > tol : 
			#geometry improving algorithm on active_subspace_set
			print('GEOMETRY IMPROVING ON SUBSPACE INTERPOLATION SET')
			active_subspace_set, budget = self.geometry_improvement(active_subspace_set, current_solution, delta_k, geometry_improving_flag)
			expended_budget += budget
			
			#update the function values of the active subspace set
			grad_fval, _ = self.get_grad_fvals(active_subspace_set, problem, expended_budget)
			expended_budget += 1 + (2 * problem.dim) 

			#update the subspace matrix
			self.active_subspace_matrix.fit(grad_fval)

		else :		
			print('NO IMPROVEMENT')
			if delta_k==rho_k : 
				#shrink rho_k+1 and delta_k+1
				print('SHRINKING DELTA_K and RHO_K')
				rho_k = 0.1 * rho_k 
				delta_k = 0.5 * delta_k


		print('interpolation_set shape: ', interpolation_set.shape)
		print('active subspace set shape: ', active_subspace_set.shape)
		
		return interpolation_set, active_subspace_set, fvals, grad_fval, rho_k, delta_k, expended_budget
		

	#the set parameter is of the form {x_k,x^2,...,x^q,...}
	def geometry_improvement(self, set:np.ndarray, current_solution: Solution, delta_k: float, geometry_improving_flag: bool) -> tuple(np.ndarray, int) : 
		q = len(set)
		#convert set into a list of numpy vectors where each row is an element
		set = [set[i,:] for i in range(q)]
		budget = 0
		current_x = np.array(current_solution.x)

		# print('current_x: ', current_x.shape)

		#Build list of polynomial basis functions that can be called at any value 
		# print('degree of basis: ', basis.degree)
		polynomial_basis = lambda x : self.poly_basis.vander(x,q-1)

		vander_current_x = polynomial_basis(current_x) 
		# print('current_x: ', current_x)
		#construct pivot polynomials 
		def pivot_polynomial_basis(x, j) :
			# print('x: ', x)
			# vand_at_x = polynomial_basis(x)
			# frac = vander_current_x[:,j]/vander_current_x[:,0]
			# res = vand_at_x[:,j] - frac * vand_at_x[:,0]
			vand_at_x = polynomial_basis(x)
			frac = np.divide(vander_current_x[0,j],vander_current_x[0,0])
			res = vand_at_x[0,j] - frac * vand_at_x[0,0]
			# print(f'pivot polynomial at {j} is {res}')
			return res 
	
		
		# pivot_polynomials = [vander_current_x[:,0]]
		pivot_polynomials = [lambda x : vander_current_x[0,0]]
		for i in range(1, q) : 
			pivot_polynomials.append(lambda x : pivot_polynomial_basis(x, i))


		new_set = [set.pop(0)] 
		for i in range(1, q) :
			x_t = np.zeros(current_x.shape)

			if geometry_improving_flag :

				obj_fn_gi = lambda x : -np.abs(pivot_polynomials[i](x))
				nlc = NonlinearConstraint(lambda x : norm(x-current_x), 0, delta_k)
				x_t = minimize(obj_fn_gi, current_x, method='SLSQP', constraints=nlc).x

			else : 
				print('NOT IMPROVING GEOMETRY')
				#TODO: Fix the problem of the numerator being NaN - Possibly affects how the model is constructed and leads to lack of updates
				def obj_fn_non_geo(x) : 
					denom = max(norm(x-x_t)**4/delta_k**4,1)
					numerator = pivot_polynomials[i](x_t) #!!This is returning NAN
					# print('i: ', i)
					# print('length of pivot_polynomial: ', len(pivot_polynomials))
					# print('x_t: ', x_t)
					# print('numerator: ', numerator)
					res = np.abs(numerator)/denom
					return res 
				# print('set: ',set)
				obj_val_set = [(i,obj_fn_non_geo(i)) for i in set]
				# print('obj_val_set: ', obj_val_set)
				value_to_find_xt = max([a[1] for a in obj_val_set], key=np.linalg.norm)

				
				for x_val, f_i in obj_val_set : 
					if f_i == value_to_find_xt : 
						x_t = x_val 
				# print('x_to_remove: ',x_t)
				# removes x_t from the set 
				set = [v for v in set if not np.array_equal(v, x_t)]
			
			new_set.append(x_t)

			# [print(a(x_t).shape) for a in pivot_polynomials]
			updated_pivot_polynomials = self.update_pivot_polynomials(pivot_polynomials, x_t, i)
			pivot_polynomials = updated_pivot_polynomials


		new_set = np.array(new_set).reshape(-1,len(current_x))

		return new_set, budget
	

	def update_pivot_polynomials(self,old_pivot_polynomials: list[Callable], x_t: np.ndarray, idx: int) -> list[Callable] :

		q = len(old_pivot_polynomials)
		# print('length of old pivot polynomials: ',len(old_pivot_polynomials))
		# print(f'old_pivot_polynomials up to {idx}: {old_pivot_polynomials[:idx]}')
		# print(f'type of old_pivot_polynomials: {type(old_pivot_polynomials[:idx])}')
		new_pivot_poly = deepcopy(old_pivot_polynomials[:idx])
		# print('x_t: ', x_t)
		fn_eval_idx_xt = old_pivot_polynomials[idx](x_t)
		new_pivot_poly.append(lambda x : fn_eval_idx_xt)

		for j in range(idx+1,q) : 
			# print('j: ', j)
			fn_eval_j_xt = old_pivot_polynomials[j](x_t)
			new_poly = lambda x : old_pivot_polynomials[j](x) - (fn_eval_j_xt/fn_eval_idx_xt) * old_pivot_polynomials[idx](x)
			new_pivot_poly.append(new_poly)

		return new_pivot_poly

	
	
"""
	Class that represents the geometry of the solution space. It is able to construct an interpolation set and handle geometric behaviours of the space
"""
class trust_region_interpolation_points :
	def __init__(self, problem):
		self.problem = problem
		# self.current_val = current_val

	def assign_current_val(self, current_val) :
		"""
			Assigns the current iteration solution to the class member current_val

		Args:
			current_val (np.array): the current iteration solution 
		"""
		self.current_val = current_val

	def interpolation_points(self, current_val, delta, degree, sub_dim):
		"""
		Samples 0.5*(d+1)*(d+2) points within the trust region
		Args:
			delta (float): the trust-region radius
		
		Returns:
			[np.array]: the trust_region set
		"""

		# size = int(0.5*(self.problem.dim + 1)*(self.problem.dim + 2))
		# size = comb(sub_dim + degree, degree)
		dim = len(current_val)
		size = comb(sub_dim + degree, degree) + dim*sub_dim #- (sub_dim*(sub_dim+1))//2
		print('number of interpolation points: ', size)
		x_k = current_val 
		Y = [x_k]
		for i in range(1,size) : 
			random_vector = np.random.normal(size=len(x_k))
			random_vector /= norm(random_vector) #normalise the vector
			random_vector *= delta #scale the vector to lie on the surface of the trust region 
			random_vector += x_k #translate vector to be in trust region
			Y.append(random_vector)

		#maybe make half of the vectors stored in Y reflections by -rand_vect?
		return Y 
	
