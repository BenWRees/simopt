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
from simopt.solvers.trust_region_class import sampling_rule
import numpy as np 
from numpy.linalg import norm
from scipy.optimize import minimize, NonlinearConstraint
from math import comb

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
			"dimension reduction": {
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
				"default": "simopt.solvers.trust_region_class:trust_region_geometry"
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
			"dimension reduction": self.check_dimension_reduction,
			"polynomial degree": self.check_polynomial_degree,
			"geometry_instance": self.check_geometry_instance,
			"poly_basis": self.check_poly_basis,
			"sampling_rule": self.check_sampling_rule
		}
	
	def check_crn_across_solns(self) -> bool:
		return True
	
	def check_easy_solve(self) -> bool:
		return True
	
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
		return self.factors['dimension reduction'] >= 1
	
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

	def solve_subproblem(self, delta: float, rho_k: float, expended_budget:int, model: PolynomialRidgeApproximation, subspace_matrix:np.ndarray, problem: Problem, solution: Solution, polynomial_basis: Basis, subspace_int_set:list[np.ndarray], model_int_set: list[np.ndarray], fvals: list[float], grad_fval: list[float]) -> tuple[Solution, float, float, int, list, np.ndarray, list, list, list, bool]:
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
		grad, Hessian = model.grad(new_x), model.hessian(new_x)
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
				return model.eval(new_x + s)
			
			con_f = lambda s: norm(new_x)
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
			subspace_int_set, model_int_set, subspace_matrix, fvals, grad_fval, rho_k, delta, expended_budget = self.interpolation_update(solution, problem, expended_budget, polynomial_basis, model_int_set, subspace_int_set, subspace_matrix, fvals, grad_fval, delta, rho_k, True)
			reset_flag = True
			candidate_solution = solution

		 

		return candidate_solution, rho_k, delta, expended_budget, subspace_int_set, subspace_matrix, model_int_set, fvals, grad_fval, reset_flag
	
	def evaluate_candidate_solution(self, model:PolynomialRidgeApproximation, subspace_matrix: np.ndarray, problem: Problem, current_fval: float, fval_tilde: float, delta_k: float, rho_k: float, current_solution: Solution, candidate_solution: Solution, recommended_solns: list[Solution], polynomial_basis: Basis, model_int_set: list[np.ndarray], subspace_int_set: list[np.ndarray], fvals: list[float]) -> tuple[Solution, float, float, list[Solution], list[np.ndarray], list[np.ndarray], np.ndarray, list[float], float, int]:
		# fval = model.fval
		expended_budget = 0
		current_x = np.array(current_solution.x)
		candidate_x = np.array(candidate_solution.x)
		step_size = np.subtract(candidate_x,current_x)


		if (model.eval(np.zeros(problem.dim)) - model.eval(np.subtract(candidate_x, current_x))) <= 0:
			rho = 0
		else:
			numerator = current_fval - fval_tilde
			denominator = model.eval(current_x) - model.eval(candidate_x)
			rho = numerator / denominator
		
		if rho >= self.factors['eta_1']:
			current_solution = candidate_solution
			current_fval = fval_tilde
			recommended_solns.append(candidate_solution)
			delta_k = max(self.factors['gamma_1']*delta_k, norm(step_size),rho_k)
			
			# very successful: expand and accept
			if rho >= self.factors['eta_2'] :
				delta_k = max(self.factors['gamma_2'] * delta_k, self.factors['gamma_3']*norm(step_size))
			
		# unsuccessful: shrink and reject
		else:
			delta_k = max(min(self.factors['gamma_1']*delta_k, norm(step_size)), rho_k)
			recommended_solns.append(current_solution)

		#append the candidate solution to the interpolation set and active subspace set
		print('model_int_set in evaluate: ', model_int_set)
		print('subspace_int_set in evaluate: ', subspace_int_set)
		model_int_set = np.vstack((model_int_set, candidate_x))
		subspace_int_set = np.vstack((subspace_int_set, candidate_x))

		if rho_k >= self.factors['eta_1'] :
			model_int_set, budget = self.geometry_improvement(model_int_set, current_solution, polynomial_basis, delta_k, False)
			expended_budget += budget
			subspace_int_set, budget =  self.geometry_improvement(subspace_int_set, current_solution, polynomial_basis, delta_k, False)
			expended_budget += budget
		else :
			model_int_set, subspace_int_set, subspace_matrix, fvals, grad_fval, rho_k, delta_k, budget = self.interpolation_update(current_solution, problem, polynomial_basis, model_int_set, subspace_int_set, subspace_matrix, fvals, grad_fval, delta_k, rho_k, True)
			expended_budget += budget

		return current_solution, current_fval, delta_k, recommended_solns, model_int_set, subspace_int_set, subspace_matrix, fvals, rho_k, expended_budget


	def finite_difference_gradient(self, new_solution: Solution, problem: Problem) -> np.ndarray :
		"""Calculate the finite difference gradient of the problem at new_solution.

		Args:
			new_solution (Solution): The solution at which to calculate the gradient.
			problem (Problem): The problem`that contains the function to differentiate.

		Returns:
			np.ndarray: The solution value of the gradient 
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

	def get_fvals(self, samples, problem, budget, get_grads=True) : 
		fvals = []
		grads = []
		for s in samples :
			s = [a for a in s.flatten()]
			new_solution = self.create_new_solution(tuple(s), problem)
			problem.simulate(new_solution,1)
			budget += 1
			fvals.append(-1 * problem.minmax[0] * new_solution.objectives_mean)
			
			if get_grads :
				grad, expended_budget = self.finite_difference_gradient(new_solution, problem)
				grads.append(grad)
				budget += expended_budget

		# print("grads: ", np.array(grads).shape)
			
		return np.array(fvals).reshape((len(fvals),1)), np.array(grads).reshape((len(grads),problem.dim)), budget


	def generate_samples(self, lower_bound, upper_bound, new_x, problem) : 
		samples = [] 
		degree = self.factors['polynomial degree']
		subspace_dim = self.factors['dimension reduction']
		dim = problem.dim
		no_samples = comb(subspace_dim + degree, degree) + dim*subspace_dim - (subspace_dim*(subspace_dim+1))//2

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

		current_x = problem.factors["initial_solution"]
		current_solution = self.create_new_solution(current_x, problem)
		recommended_solns.append(current_solution)
		intermediate_budgets.append(expended_budget)

		#evaluate the current solution
		problem.simulate(current_solution,1)
		expended_budget += 1
		current_fval = -1 * problem.minmax[0] * current_solution.objectives_mean

		sampling_instance = self.sample_instantiation()

		poly_basis = self.polynomial_basis_instantiation()(self.factors['polynomial degree'], dim=problem.dim)
		geometry_instance = self.geometry_type_instantiation()(problem)

		#build an initial set of interpolation points of n+1 points
		subspace_int_set = self.generate_samples(problem.lower_bounds, problem.upper_bounds, current_x, problem)

		#get function values of the interpolation set
		fvals, grad_fval, expended_budget = self.get_fvals(subspace_int_set, problem, expended_budget)

		#construct an initial subspace matrix using subspace_int_set
		subspace_matrix = initialize_subspace(subspace_int_set, fvals, grad_fval)

		#build an initial set of interpolation points
		model_int_set = geometry_instance.interpolation_points(np.array(current_x),delta_k)
		model_int_set = np.array(model_int_set).reshape(len(model_int_set), problem.dim)

		#initialise the polyridge model 
		local_model = PolynomialRidgeApproximation(self.factors['polynomial degree'], self.factors['dimension reduction'], problem, poly_basis)
		local_model.fit(subspace_int_set, fvals, U0=subspace_matrix)


		k=0

		while expended_budget < problem.factors['budget']:
			print('k: ', k)
			if k > 0 :
				#construct the model using the interpolation set
				local_model.fit(subspace_int_set, fvals, U0=subspace_matrix)

			#solve the subproblem

			candidate_solution, rho_k, delta_k, expended_budget, subspace_int_set, subspace_matrix, model_int_set, fvals, grad_fval, reset_flag = self.solve_subproblem(delta_k, rho_k, expended_budget, local_model, subspace_matrix, problem, current_solution, poly_basis, subspace_int_set, model_int_set, fvals, grad_fval) 

			#if the stepsize is too small, do not evaluate and restart the loop
			if reset_flag :
				recommended_solns.append(current_solution)
				intermediate_budgets.append(expended_budget) 
				continue
			else : 
				#evaluate the candidate solution
				candidate_solution, sampling_budget = sampling_instance(problem, candidate_solution, k, delta_k, expended_budget, 1, 0)
				fval_tilde = -1 * problem.minmax[0] * candidate_solution.objectives_mean
				expended_budget = sampling_budget

				#evaluate the candidate solution
				current_solution, current_fval, delta_k, recommended_solns, model_int_set, subspace_int_set, subspace_matrix, fvals, rho_k, budget = self.evaluate_candidate_solution(local_model, subspace_matrix, problem, current_fval, fval_tilde, delta_k, rho_k, current_solution, candidate_solution, recommended_solns, poly_basis, subspace_int_set, model_int_set, fvals)
				expended_budget += budget
				
				intermediate_budgets.append(expended_budget)
			k += 1
			
		return recommended_solns, intermediate_budgets


	#This is the update for the interpolation sets as defined in the paper
	def interpolation_update(self, current_solution, problem, expended_budget, polynomial_basis, interpolation_set, active_subspace_set, subspace_matrix, fvals, grad_fval, delta_k, rho_k, geometry_improving_flag) : 
		tol = max(2*delta_k, 10*rho_k)
		for elem_int,elem_as in zip(interpolation_set, active_subspace_set)  : 
			
			if max(norm(current_solution.x - elem_int)) > tol : 
				#geometry inmproving algorithm on interpolation_set
				interpolation_set, budget = self.geometry_improvement(interpolation_set, problem, current_solution, polynomial_basis, delta_k, geometry_improving_flag)
				expended_budget += budget

				#update the function values of the interpolation set
				fvals, grad_fval, budget = self.get_fvals(interpolation_set, problem, expended_budget)
				expended_budget += 2

				#set new subspace matrix, rho_k+1, delta_k+1
				return interpolation_set, active_subspace_set, subspace_matrix, fvals, grad_fval, rho_k, delta_k, expended_budget
			
			elif max(norm(current_solution.x - elem_as)) > tol : 
				#geometry improving algorithm on active_subspace_set
				active_subspace_set, budget = self.geometry_improvement(active_subspace_set, current_solution, polynomial_basis, delta_k, geometry_improving_flag)
				expended_budget += budget
				
				#update the function values of the active subspace set
				fvals, grad_fval, budget = self.get_fvals(active_subspace_set, problem, expended_budget)
				expended_budget += 2 

				#update the subspace matrix
				subspace_matrix = initialize_subspace_matrix(active_subspace_set, fvals, grad_fval)


				#reconstruct basis matrix of active subspace 
				#set rho_k+1 and delta_k+1
				return interpolation_set, active_subspace_set, subspace_matrix, fvals, grad_fval, rho_k, delta_k, expended_budget
			
		if delta_k==rho_k : 
			#shrink rho_k+1 and delta_k+1
			rho_k = 0.1 * rho_k 
			delta_k = 0.5 * delta_k

			return interpolation_set, active_subspace_set, subspace_matrix, fvals, grad_fval, rho_k, delta_k, expended_budget
		
		return interpolation_set, active_subspace_set, subspace_matrix, fvals, grad_fval, rho_k, delta_k, expended_budget
		

	#the set parameter is of the form {x_k,x^2,...,x^q,...}
	#TODO: Needs fixing 
	def geometry_improvement(self, set:np.ndarray, current_solution: Solution, basis: Basis, delta_k: float, geometry_improving_flag: bool) -> tuple(np.ndarray, int) : 
		q = len(set)
		#convert set into a list of numpy vectors where each row is an element
		set = [set[i,:] for i in range(q)]
		budget = 0
		current_x = np.array(current_solution.x)
		print('current_x: ', current_x.shape)

		#Build list of polynomial basis functions that can be called at any value 
		polynomial_basis = lambda x : basis.vander(x,q)

		# polynomial_basis = []
		# for i in range(len(set)) : 
		# 	basis_fn_at_i = lambda x : basis.vander(x, i)
		# 	polynomial_basis.append(basis_fn_at_i)

		vander_current_x = polynomial_basis(current_x) 
		print('vander at current_x: ', vander_current_x)
		#construct pivot polynomials 
		def pivot_polynomial_basis(x, j) :
			vand_at_x = polynomial_basis(x)
			frac = vander_current_x[:,j]/vander_current_x[:,0]
			res = vand_at_x[:,j] - frac * vand_at_x[:,0]
			return res 
		
		pivot_polynomials = [vander_current_x[:,0]]
		for i in range(1, q) : 
			pivot_polynomials.append(lambda x : pivot_polynomial_basis(x, i))


		new_set = [set.pop(0)] 
		for i in range(1, q) :
			x_t = np.zeros(current_x.shape)

			if geometry_improving_flag :

				obj_fn_gi = lambda x : np.abs(pivot_polynomials[i](x))
				nlc = NonlinearConstraint(lambda x : norm(x-current_x), 0, delta_k)
				x_t = minimize(obj_fn_gi, current_x, method='trust-constr', constraints=nlc).x

			else : 
				obj_fn = lambda x : np.abs(pivot_polynomials[i](x_t))/(max(norm(x-x_t)**4/delta_k**4,1))
				obj_val_set = [obj_fn(i) for i in set]
				print('obj_val_set: ', obj_val_set)
				x_t = max(obj_val_set, key=np.linalg.norm)
				
				# set.remove(x_t)
				# x_t = np.array(x_t)
				set = [v for v in set if not np.array_equal(v, x_t)]
			
			print('TEST')
			new_set.append(x_t)

			#update pivot polynomials
			#FIX: after a few iterations this has a recursion error 
			old_fn_i = pivot_polynomials[i]
			old_fn_i_at_x_t = old_fn_i(x_t)
			res = lambda x : old_fn_i_at_x_t
			pivot_polynomials[i] = res 
			for j in range(i+1, q) :
				old_fn_j = pivot_polynomials[j]
				old_fn_j_at_x_t = old_fn_j(x_t)
				# res = lambda x : pivot_polynomials[j](x) - (pivot_polynomials[j](x_t)/pivot_polynomials[j](x_t))* pivot_polynomials[i](x)
				res = lambda x : old_fn_j(x) - (old_fn_j_at_x_t/old_fn_i_at_x_t)* old_fn_i(x)
				pivot_polynomials[j] = res

			print('length of pivot polynomials: ', len(pivot_polynomials))

		return new_set, budget
	
	


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

	def interpolation_points(self, current_val, delta):
		"""
		Samples 0.5*(d+1)*(d+2) points within the trust region
		Args:
			delta (float): the trust-region radius
		
		Returns:
			[np.array]: the trust_region set
		"""

		size = 0.5*(self.problem.dim + 1)*(self.problem.dim + 2)
		x_k = self.current_val 
		Y = [x_k]
		for i in range(1,size) : 
			random_vector = np.random.normal(size=len(x_k))
			random_vector /= norm(random_vector) #normalise the vector
			random_vector *= delta #scale the vector to lie on the surface of the trust region 
			random_vector += x_k #translate vector to be in trust region
			Y.append(random_vector)

		#maybe make half of the vectors stored in Y reflections by -rand_vect?
		return Y 
	
