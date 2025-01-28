"""
Summary
-------
The ASTROMoRF (Adaptive Sampling for Trust-Region Optimisation by Moving Ridge Functions) progressively builds local models using
interpolation on a reduced subspace constructed through Active Subspace dimensionality reduction.
The use of Active Subspace reduction allows for a reduced number of interpolation points to be evaluated in the model construction.

"""
from __future__ import annotations
from typing import Callable

from numpy.linalg import norm, pinv, qr
import numpy as np
from math import ceil, isnan, isinf, comb, factorial, log
import warnings
from scipy.optimize import minimize, NonlinearConstraint, linprog
from scipy import optimize
warnings.filterwarnings("ignore")
import importlib
from copy import deepcopy
import inspect


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
from simopt.solvers.active_subspaces.polyridge import *
from simopt.solvers.active_subspaces.subspace import ActiveSubspace
from simopt.solvers.active_subspaces.index_set import IndexSet

from simopt.solvers.TrustRegion.Sampling import SamplingRule
from simopt.solvers.TrustRegion.TrustRegion import * 
from simopt.solvers.TrustRegion.Sampling import * 
from simopt.solvers.TrustRegion.Geometry import *

#TODO: Add Orthogonal Geometry into Getting Directions 
#TODO: Rewrite model construction with adaptive sampling 
class ASTROMoRF(Solver):
	"""Combining the ASTRO-DF solver and the OMoRF solver

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
			"lambda_min": {
				"description": "minimum sample size",
				"datatype": int,
				"default": 5
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
			},
			"interpolation update tol":{
				"description": "tolerance values to check for updating the interpolation model",
				"datatype": tuple, 
				"default": (2.0,10.0)
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
				"default": 7
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
			"polynomial basis": self.check_poly_basis, 
			"model construction parameters": self.check_model_construction_parameters,
			"polynomial degree": self.check_poly_degree,
			"ps_sufficient_reduction": self.check_ps_sufficient_reduction,
			"interpolation update tol":self.check_tolerance,
			"initial radius": self.check_initial_radius,
			"gamma_3": self.check_gamma_3,
			"gamma_shrinking": self.check_gamma_shrinking,
			"omega_shrinking": self.check_omega_shrinking,
			"subspace dimension": self.check_dimension_reduction,
			"random directions": self.check_random_directions,
			"alpha_1": self.check_alpha_1,
			"alpha_2": self.check_alpha_2,
			"rho_min": self.check_rho_min
			}
		

	
	def check_tolerance(self) -> bool:
		return self.factors['interpolation update tol'] >(0,0) and self.factors['interpolation update tol'] <= (1,1)
	
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
	
	def check_random_directions(self) -> bool : 
		return isinstance(self.factors['random directions'], bool)
	
	def check_alpha_1(self) -> bool :
		return self.factors['alpha_1'] > 0 
	
	def check_alpha_2(self) -> bool :
		return self.factors['alpha_2'] > 0 
	
	def check_rho_min(self) -> bool : 
		return self.factors['rho_min'] < self.factors['delta']

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
	
	def check_poly_basis(self) -> bool:
		if self.factors['polynomial basis'] is None : 
			raise ValueError('Polynomial Basis Needs to be Implemented') 
	
	def check_poly_degree(self) -> bool : 
		if self.factors['polynomial degree'] < 1 :
			raise ValueError('Local Model Polynomial Degree must be at least 1')
	
	def check_model_construction_parameters(self) -> None :
		if not isinstance(self.factors['model construction parameters'], dict) : 
			raise ValueError('The model construction parameters must be a dictionary')
	
	def check_ps_sufficient_reduction(self) -> None:
		if self.factors["ps_sufficient_reduction"] < 0:
			raise ValueError(
				"ps_sufficient reduction must be greater than or equal to 0."
			)

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
	
	def polynomial_basis_instantiation(self) :
		class_name = self.factors['polynomial basis'].strip()
		module = importlib.import_module('simopt.solvers.active_subspaces.basis')
		return getattr(module, class_name)

	"""def _set_iterate(self):
		ind_min = np.argmin(self.f) #get the index of the smallest function value 
		self.s_old = self.S[ind_min,:] #get the x-val that corresponds to the smallest function value 
		self.f_old = self.f[ind_min] #get the smallest function value """


	def _set_delta(self, val) : 
		self.delta_k = val

	def _set_counter(self, count) :
		self.unsuccessful_iteration_counter = count 

	def _set_ratio(self, ratio) : 
		self.ratio = ratio 

	def _set_rho_k(self, value):
		self.rho_k = value 


	def blackbox_evaluation(self, s, problem):
		"""
		Evaluates the point s for the problem

		self.S is array-like of all the x values 
		self.f is a 1-d array of function values 
		"""
		#If S has 1 or more points in it and the point being evaluated is not unique. Just grab the existing fn. value
		if self.S.size > 0 and np.unique(np.vstack((self.S, s)), axis=0).shape[0] == self.S.shape[0]:
			s = s.reshape(1,-1)
			ind_repeat = np.argmin(norm(self.S - s, ord=np.inf, axis=1))
			f = self.f[ind_repeat]
		else :
			#TODO: Add Sampling method to the blackbox evaluation
			new_solution = self.create_new_solution(tuple(s), problem) 

			"""if self.factors['crn_across_solns'] :
				problem.simulate(candidate_solution, current_solution.n_reps) 
				expended_budget += current_solution.n_reps 
			else :
				candidate_solution, expended_budget = sampling_instance(problem, candidate_solution, k, delta_k, expended_budget, 0, 0)"""

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
	

	def finite_difference_gradient_OMoRF(self, new_solution:Solution, problem: Problem) -> np.ndarray : 
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

		new_x = new_solution.x
		forward = np.isclose(new_x, lower_bound, atol = 10**(-7)).astype(int)
		backward = np.isclose(new_x, upper_bound, atol = 10**(-7)).astype(int)
		# BdsCheck: 1 stands for forward, -1 stands for backward, 0 means central diff.
		BdsCheck = np.subtract(forward, backward)

		self.expended_budget += (2*problem.dim) + 1
		return finite_difference_gradient(new_solution, problem, BdsCheck=BdsCheck)
		


	def _get_grads(self, X: np.ndarray, problem: Problem) -> np.ndarray : 
		"""Calculate gradients 

		Args:
			X (np.ndarray): (N,m) matrix of N x-vals to be evaluated
			problem (Problem): 

		Returns:
			np.ndarray: (N,m) matrix of N gradients evaluated at each row of X
		"""
		grads = np.zeros(X.shape)

		# print(f'X: {X}')

		#fill out each row 
		for idx,x_val in enumerate(X) : 
			x_solution = self.create_new_solution(x_val, problem)
			grads[idx, :] = self.finite_difference_gradient_OMoRF(x_solution, problem)

		# print(f'grads: {grads}')
		return grads 

	def fit_subspace(self, X:np.ndarray, problem: Problem) -> None:
		"""
		Takes the active subspace and fits

		Args:
			X (np.ndarray): (N,m) matrix of N x-values
			problem (Problem)
		"""
		grads = self._get_grads(X, problem) 
		self.U.fit(grads, self.d) 


	def solve(self, problem):
		#initialise factors: 
		self.recommended_solns = []
		self.intermediate_budgets = []

		self.kappa = 1
		self.pilot_run = 2
		self.delta_power = 2 if self.factors['crn_across_solns'] else 4

		self.expended_budget = 0

		self.S = np.array([])
		self.f = np.array([])
		self.g = np.array([])
		self.d = self.factors['subspace dimension']

		self.delta_max = self.calculate_max_radius(problem)

		# self.delta_k = self.factors['delta']
		self._set_delta(10 ** (ceil(log(self.delta_max * 2, 10) - 1) / problem.dim))
		# self.rho_k = self.delta_k
		self._set_rho_k(self.delta_k) 
		self.rhos = []
		self.deltas = []
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
		 
		index_set = IndexSet('total-order', orders=np.tile([2], self.q))
		self.index_set = index_set.get_basis()[:,range(self.d-1, -1, -1)]
		self.poly_basis = self.polynomial_basis_instantiation()(self.factors['polynomial degree'], dim=self.factors['subspace dimension'])

		#instantiate ActiveSubspace and use it to construct the active subspace matrix 
		self.U = ActiveSubspace()

		self.s_old = np.array(current_x)
		self.f_old = self.blackbox_evaluation(self.s_old,problem)

		# Construct the sample set for the subspace 
		S_full = self.generate_set(self.d, self.s_old, self.delta_k, problem)
		f_full = np.zeros((self.d, 1))
		f_full[0, :] = self.f_old #first row gets the old function values 
		

		#initial subspace calculation - requires gradients of f_full 
		self.fit_subspace(S_full, problem)
		
		#This constructs the sample set for the model construction
		S_red, f_red = self.sample_set('new', self.s_old, self.delta_k, self.rho_k, self.f_old, self.U.U, problem, full_space=False)
		self.local_model = PolynomialRidgeApproximation(self.deg, self.d, problem, self.poly_basis)
		
		while self.expended_budget < problem.factors['budget'] :
			
			#GET KAPPA AND PILOT RUN
			self.calculate_pilot_run(problem)
			if self.k == 1 : 
				self.recommended_solns.append(self.current_solution)
				self.intermediate_budgets.append(self.expended_budget)
		
			#if rho has been decreased too much we end the algorithm  
			if self.rho_k <= self.rho_min:
				break
			
			#BUILD MODEL
			try: 
				self.local_model.fit(S_red, f_red, U0=self.U.U) #this should be the model instance construct model
			except: #thrown if the sample set is not defined properly 
				S_red, f_red = self.sample_set('improve', self.s_old, self.delta_k, self.rho_k, self.f_old, self.U.U, problem, S_red, f_red, full_space=False)
				self.intermediate_budgets.append(self.expended_budget)
				continue 

			#SOLVE THE SUBPROBLEM
			candidate_solution, S_full, S_red, f_full, f_red, reset_flag = self.solve_subproblem(problem, S_full, S_red, f_full, f_red)

			if reset_flag :
				self.recommended_solns.append(self.current_solution)
				self.intermediate_budgets.append(self.expended_budget) 
				self.k +=1
				break 

			#! Adaptive sampling of the candidate solution
			if self.factors['crn_across_solns'] :
				problem.simulate(candidate_solution, self.current_solution.n_reps) 
				self.expended_budget += self.current_solution.n_reps 
			else :
				candidate_solution = self.adaptive_sampling(problem, candidate_solution, 0, 0)
			
			candidate_fval = -1 * problem.minmax[0] * candidate_solution.objectives_mean

			
			#EVALUATE THE CANDIDATE SOLUTION
			S_red, S_full, f_red, f_full = self.evaluate_candidate_solution(problem, candidate_fval, candidate_solution, S_red, S_full, f_red, f_full)
		

			self.recommended_solns.append(self.current_solution) 
			self.intermediate_budgets.append(self.expended_budget)
			self.k+=1

		return self.recommended_solns, self.intermediate_budgets



	def solve_subproblem(self, problem: Problem, S_full:np.ndarray, S_red: np.ndarray, f_full: float, f_red: float) :
		"""
			Solves the trust-region subproblem
		"""
		
		omega_s = self.factors['omega_shrinking']
		reset_flag = False


		cons =  NonlinearConstraint(lambda x : norm(self.s_old - x), 0, self.delta_k)
		
		obj = lambda x: self.local_model.eval(x)[0]
		res = minimize(obj, self.s_old, method='trust-constr', constraints=cons, options={'disp': False})
		s_new = res.x


		for i in range(problem.dim):
			if s_new[i] <= problem.lower_bounds[i]:
				s_new[i] = problem.lower_bounds[i] + 0.01
			elif s_new[i] >= problem.upper_bounds[i]:
				s_new[i] = problem.upper_bounds[i] - 0.01


		candidate_solution = self.create_new_solution(tuple(s_new), problem)

		step_dist = norm(s_new - self.s_old, ord=np.inf)

		# Safety step implemented in BOBYQA
		if step_dist < omega_s*self.rho_k:
			# self.ratio= -0.1
			self._set_ratio(-0.1)
			self._set_counter(3)
			# self.delta_k = max(0.5*self.delta_k, self.rho_k)
			self._set_delta(max(0.5*self.delta_k, self.rho_k))
			S_full, f_full, S_red, f_red, delta_k, rho_k, U = self.update_geometry_omorf(problem,self.s_old, self.f_old, self.delta_k, self.rho_k, self.U.U, S_full, f_full, S_red, f_red, self.unsuccessful_iteration_counter, self.ratio)
			self.U.set_U(U)
			self._set_delta(delta_k)
			self._set_rho_k(rho_k)
		
		#this is a breaking condition
		if self.rho_k <= self.rho_min:
			reset_flag=True

		return candidate_solution, S_full, S_red, f_full, f_red, reset_flag

	def evaluate_candidate_solution(self, problem: Problem, fval_tilde: float, candidate_solution: Solution, S_red: np.ndarray, S_full: np.ndarray, f_red: np.ndarray, f_full: np.ndarray) :
		
		#add candidate value and corresponding fn val to interpolation sets
		S_red, f_red = self.sample_set('replace', self.s_old, self.delta_k, self.rho_k, self.f_old, self.U.U, problem, S_red, f_red, s_new, fval_tilde, full_space=False)
		S_full, f_full = self.sample_set('replace', self.s_old, self.delta_k, self.rho_k, self.f_old, self.U.U, problem, S_full, f_full, s_new, fval_tilde)
		
		ind_min = np.argmin(self.f) 
		min_fval = np.asscalar(self.f[ind_min]) 

		#Pattern Search
		if ((min_fval < fval_tilde) and ((self.f_old - min_fval)>= self.factors["ps_sufficient_reduction"] * delta_k**2)) or ((candidate_solution.objectives_var/ (candidate_solution.n_reps * candidate_solution.objectives_mean**2))[0]> 0.75):
			# self.s_old = self.S[ind_min,:] 
			# self.f_old = min_fval 
			candidate_solution = self.create_new_solution(tuple(min_fval), problem) #type: ignore
			

		s_new = np.array(candidate_solution.x)
		
		del_f = self.f_old - fval_tilde #self.f_old - f_new 
		# del_m = np.asscalar(my_poly.get_polyfit(np.dot(self.s_old,self.U))) - m_new
		del_m = self.local_model.eval(self.s_old)[0] - self.local_model.eval(s_new)[0]


		#in the case that the denominator is very small 
		if del_m <= 0 : 
			# self.ratio = 1.0
			self._set_ratio(0.0)
		else:
			# self.ratio = del_f/del_m
			self._set_ratio(del_f/del_m)


		# successful: accept
		if self.ratio >= self.factors['eta_1']:
			self.current_solution = candidate_solution
			self._set_counter(0)
			self.f_old = fval_tilde
			self.s_old = s_new	
			self.recommended_solns.append(candidate_solution)
			self._set_delta(min(delta_k, self.delta_max))
			
			# very successful: expand and accept
			if self.ratio >= self.factors['eta_2'] :
				self._set_delta(min(self.factors['gamma_1'] * delta_k, self.delta_max))
			
		# unsuccessful: shrink and reject
		else:
			self._set_counter(1)
			self._set_delta(min(self.factors['gamma_2'] * delta_k, self.delta_max))
			S_full, f_full, S_red, f_red, delta_k, rho_k, U = self.update_geometry_omorf(problem, self.s_old, self.f_old, self.delta_k, self.rho_k, self.U.U, S_full, f_full, S_red, f_red, self.unsuccessful_iteration_counter, self.ratio)
			self.U.set_U(U)
			self._set_delta(delta_k)
			self._set_rho_k(rho_k)
			self.recommended_solns.append(self.current_solution)


		"""if self.ratio >= eta_2:
			# print('VERY SUCCESSFUL ITERATION')
			self._set_counter(0)
			# self.delta_k = max(gamma_2*self.delta_k, gamma_3*step_dist)
			self._set_delta(max(gamma_2*self.delta_k, gamma_3*step_dist))
			self.current_solution = candidate_solution
			self.f_old = fval_tilde
			self.s_old = s_new
		
		elif self.ratio >= eta_1:
			# print('SUCCESSFUL ITERATION')
			self._set_counter(0)
			# self.delta_k = max(gamma_1*self.delta_k, step_dist, self.rho_k)
			self._set_delta(max(gamma_1*self.delta_k, step_dist, self.rho_k))
			self.current_solution = candidate_solution
			self.f_old = fval_tilde
			self.s_old = s_new

		else:
			# print('UNSUCCESSFUL ITERATION')
			self._set_counter(1)
			# self.delta_k = max(min(gamma_1*self.delta_k, step_dist), self.rho_k)
			self._set_delta(max(min(gamma_1*self.delta_k, step_dist), self.rho_k))
			S_full, f_full, S_red, f_red, delta_k, rho_k, U = self.update_geometry_omorf(problem, self.s_old, self.f_old, self.delta_k, self.rho_k, self.U.U, S_full, f_full, S_red, f_red, self.unsuccessful_iteration_counter, self.ratio)
			self.U.set_U(U)
			self._set_delta(delta_k)
			self._set_rho_k(rho_k)"""

		return S_red, S_full, f_red, f_full
	
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

	"""
		GEOMETRY FUNCTIONS - HOW CAN ADAPTIVE SAMPLING BE ADDED INTO THIS STAGE?
	"""
	def generate_set(self, num, s_old, delta_k, problem):
		"""
		Generates an initial set of samples using either coordinate directions or orthogonal, random directions
		"""
		bounds_l = np.maximum(np.array(problem.lower_bounds).reshape(s_old.shape), s_old-delta_k)
		bounds_u = np.minimum(np.array(problem.upper_bounds).reshape(s_old.shape), s_old+delta_k)

		if self.random_initial:
			direcs = self.random_directions(num, bounds_l-s_old, bounds_u-s_old, delta_k)
		else:
			direcs = self.coordinate_directions(num, bounds_l-s_old, bounds_u-s_old, delta_k)
		S = np.zeros((num, self.n))
		S[0, :] = s_old
		for i in range(1, num):
			S[i, :] = s_old + np.minimum(np.maximum(bounds_l-s_old, direcs[i, :]), bounds_u-s_old)
		return S

	def coordinate_directions(self, num_pnts, lower, upper, delta_k):
		"""
		Generates coordinate directions
		"""
		at_lower_boundary = (lower > -1.e-8 * delta_k)
		at_upper_boundary = (upper < 1.e-8 * delta_k)
		direcs = np.zeros((num_pnts, self.n))
		for i in range(1, num_pnts):
			if 1 <= i < self.n + 1:
				dirn = i - 1
				step = delta_k if not at_upper_boundary[dirn] else - delta_k
				direcs[i, dirn] = step
			elif self.n + 1 <= i < 2*self.n + 1:
				dirn = i - self.n - 1
				step = - delta_k
				if at_lower_boundary[dirn]:
					step = min(2.0* delta_k, upper[dirn])
				if at_upper_boundary[dirn]:
					step = max(-2.0* delta_k, lower[dirn])
				direcs[i, dirn] = step
			else:
				itemp = (i - self.n - 1) // self.n
				q = i - itemp*self.n - self.n
				p = q + itemp
				if p > self.n:
					p, q = q, p - self.n
				direcs[i, p-1] = direcs[p, p-1]
				direcs[i, q-1] = direcs[q, q-1]
		return direcs
	
	def get_scale(self, dirn, delta, lower, upper):
		scale = delta
		for j in range(len(dirn)):
			if dirn[j] < 0.0:
				scale = min(scale, lower[j] / dirn[j])
			elif dirn[j] > 0.0:
				scale = min(scale, upper[j] / dirn[j])
		return scale

	#TODO: Add ASTRO-DF orthogonal directions into this function
	def random_directions(self, num_pnts, lower, upper, delta_k):
		"""
		Generates orthogonal, random directions
		"""
		direcs = np.zeros((self.n, max(2*self.n+1, num_pnts)))
		idx_l = (lower == 0)
		idx_u = (upper == 0)
		active = np.logical_or(idx_l, idx_u)
		inactive = np.logical_not(active)
		nactive = np.sum(active)
		ninactive = self.n - nactive
		if ninactive > 0:
			A = np.random.normal(size=(ninactive, ninactive))
			Qred = qr(A)[0]
			Q = np.zeros((self.n, ninactive))
			Q[inactive, :] = Qred
			for i in range(ninactive):
				# scale = self._get_scale(Q[:,i], self.delta_k, lower, upper) 
				direcs[:, i] = scale * Q[:,i]
				scale = self.get_scale(-Q[:,i], delta_k, lower, upper)
				direcs[:, self.n+i] = -scale * Q[:,i]
		idx_active = np.where(active)[0]
		for i in range(nactive):
			idx = idx_active[i]
			direcs[idx, ninactive+i] = 1.0 if idx_l[idx] else -1.0
			direcs[:, ninactive+i] = self.get_scale(direcs[:, ninactive+i], delta_k, lower, upper) * direcs[:, ninactive+i]
			sign = 1.0 if idx_l[idx] else -1.0
			if upper[idx] - lower[idx] > delta_k:
				direcs[idx, self.n+ninactive+i] = 2.0*sign*delta_k
			else:
				direcs[idx, self.n+ninactive+i] = 0.5*sign*(upper[idx] - lower[idx])
			direcs[:, self.n+ninactive+i] = self.get_scale(direcs[:, self.n+ninactive+i], 1.0, lower, upper)*direcs[:, self.n+ninactive+i]
		for i in range(num_pnts - 2*self.n):
			dirn = np.random.normal(size=(self.n,))
			for j in range(nactive):
				idx = idx_active[j]
				sign = 1.0 if idx_l[idx] else -1.0
				if dirn[idx]*sign < 0.0:
					dirn[idx] *= -1.0
			dirn = dirn / norm(dirn)
			scale = self.get_scale(dirn, delta_k, lower, upper)
			direcs[:, 2*self.n+i] = dirn * scale
		return np.vstack((np.zeros(self.n), direcs[:, :num_pnts].T))


	def update_geometry_omorf(self, problem, s_old, f_old, delta_k, rho_k, U, S_full, f_full, S_red, f_red, unsuccessful_iteration_counter, ratio):
		dist = max(self.epsilon_1*delta_k, self.epsilon_2*rho_k)

		as_matrix = U

		if max(norm(S_full-s_old, axis=1, ord=np.inf)) > dist:
			S_full, f_full = self.sample_set('improve', s_old, delta_k, rho_k, f_old, U, problem, S=S_full, f=f_full) #f_full is not needed to be evaluated
			try:
				# print('UPDATING GEOMETRY')
				self.fit_subspace(S_full, problem) 
				as_matrix = self.U.U
			except:
				pass
		
		elif max(norm(S_red-s_old, axis=1, ord=np.inf)) > dist:
			S_red, f_red = self.sample_set('improve', s_old, delta_k, rho_k, f_old, U, problem, S=S_red, f=f_red, full_space=False)
		
		elif delta_k == rho_k:
			# self.delta_k = self.alpha_2*self.rho_k
			# self._set_delta(self.alpha_2*self.rho_k)
			delta_k = self.alpha_2* rho_k


			# print(f'unsuccessful counter: {self.unsuccessful_iteration_counter}')
			# print(f'ratio: {self.ratio}')
			
			if unsuccessful_iteration_counter >= 3 and ratio < 0:
				# print('UNSUCCESSFUL 3 OR MORE TIMES')
				if rho_k >= 250*self.rho_min:
					# self.rho_k = self.alpha_1*self.rho_k
					# self._set_rho_k(self.alpha_1*self.rho_k)
					rho_k = self.alpha_1*rho_k
				
				elif 16*self.rho_min < rho_k < 250*self.rho_min:
					# self.rho_k = np.sqrt(self.rho_k*self.rho_min)
					# self._set_rho_k(np.sqrt(self.rho_k*self.rho_min))
					rho_k = np.sqrt(rho_k*self.rho_min)
				
				else:
					# self.rho_k = self.rho_min
					# self._set_rho_k(self.rho_min)
					rho_k = self.rho_min
		
		return S_full, f_full, S_red, f_red, delta_k, rho_k, as_matrix
	
	def sample_set(self, method, s_old, delta_k, rho_k, f_old, U, problem, S=None, f=None, s_new=None, f_new=None, full_space=True):
		q = self.p if full_space else self.q

		dist = max(self.epsilon_1*delta_k, self.epsilon_2*rho_k)
		
		if method == 'replace':
			S_hat = np.vstack((S, s_new))
			f_hat = np.vstack((f, f_new))
			if S_hat.shape != np.unique(S_hat, axis=0).shape:
				S_hat, indices = np.unique(S_hat, axis=0, return_index=True)
				f_hat = f_hat[indices]
			elif f_hat.size > q and max(norm(S_hat-s_old, axis=1, ord=np.inf)) > dist:
				S_hat, f_hat = self.remove_furthest_point(S_hat, f_hat, s_old)
			S_hat, f_hat = self.remove_point_from_set(S_hat, f_hat, s_old)
			S = np.zeros((q, self.n))
			f = np.zeros((q, 1))
			S[0, :] = s_old
			f[0, :] = f_old
			S, f = self.LU_pivoting(S, f, s_old, delta_k, S_hat, f_hat, full_space, U, problem, evaluate_f_flag=full_space)
		
		elif method == 'improve':
			S_hat = np.copy(S)
			f_hat = np.copy(f)
			if max(norm(S_hat-s_old, axis=1, ord=np.inf)) > dist:
				S_hat, f_hat = self.remove_furthest_point(S_hat, f_hat, s_old)
			S_hat, f_hat = self.remove_point_from_set(S_hat, f_hat, s_old)
			S = np.zeros((q, self.n))
			f = np.zeros((q, 1))
			S[0, :] = s_old
			f[0, :] = f_old
			S, f = self.LU_pivoting(S, f, s_old, delta_k, S_hat, f_hat, full_space, U, problem, evaluate_f_flag=full_space, method='improve')
		
		elif method == 'new':
			S_hat = f_hat = np.array([])
			S = np.zeros((q, self.n))
			f = np.zeros((q, 1))
			S[0, :] = s_old
			f[0, :] = f_old
			S, f = self.LU_pivoting(S, f, s_old, delta_k, S_hat, f_hat, full_space, U, problem, evaluate_f_flag=full_space, method='new')

		return S, f
	
	def LU_pivoting(self, S, f, s_old, delta_k, S_hat, f_hat, full_space, active_subspace, problem, evaluate_f_flag=True, method=None):
		psi_1 = 1.0e-4
		psi_2 = 1.0 if full_space else 0.25

		phi_function, phi_function_deriv = self.get_phi_function_and_derivative(S_hat, s_old, delta_k, full_space, active_subspace)
		q = self.p if full_space else self.q

		#Initialise U matrix of LU factorisation of M matrix (see Conn et al.)
		U = np.zeros((q,q))
		U[0,:] = phi_function(s_old)
  
		#Perform the LU factorisation algorithm for the rest of the points
		for k in range(1, q):
			flag = True
			v = np.zeros(q)
			for j in range(k):
				v[j] = -U[j,k] / U[j,j]
			v[k] = 1.0
	
			#If there are still points to choose from, find if points meet criterion. If so, use the index to choose 
			#point with given index to be next point in regression/interpolation set
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
			
			#If index exists, choose the point with that index and delete it from possible choices
			if flag:
				s = S_hat[index,:]
				S[k, :] = s
				f[k, :] = f_hat[index]
				S_hat = np.delete(S_hat, index, 0)
				f_hat = np.delete(f_hat, index, 0)

			#If index doesn't exist, solve an optimisation problem to find the point in the range which best satisfies criterion
			else:
				try:
					s = self.find_new_point(v, phi_function, phi_function_deriv, problem, full_space)
					if np.unique(np.vstack((S[:k, :], s)), axis=0).shape[0] != k+1:
						s = self.find_new_point_alternative(v, phi_function, S[:k, :], s_old, delta_k, problem)
				except:
					s = self.find_new_point_alternative(v, phi_function, S[:k, :], s_old, delta_k, problem)
				if f_hat.size > 0 and M[index] >= abs(np.dot(v, phi_function(s))):
					s = S_hat[index,:]
					S[k, :] = s
					f[k, :] = f_hat[index]
					S_hat = np.delete(S_hat, index, 0)
					f_hat = np.delete(f_hat, index, 0)
				else:
					S[k, :] = s
					if not evaluate_f_flag : #If full_space is True then we don't want to evaluate 
						f[k, :] = self.blackbox_evaluation(s, problem) #!This should only be evaluated if its f_red 
			
			#Update U factorisation in LU algorithm
			phi = phi_function(s)
			U[k,k] = np.dot(v, phi)
			for i in range(k+1,q):
				U[k,i] += phi[i]
				for j in range(k):
					U[k,i] -= (phi[j]*U[j,i]) / U[j,j]
		return S, f

	def get_phi_function_and_derivative(self, S_hat, s_old, delta_k, full_space, active_subspace):
		Del_S = delta_k

		if full_space:
			if S_hat.size > 0:
				Del_S = max(norm(S_hat-s_old, axis=1, ord=np.inf))
			
			def phi_function(s):
				s_tilde = np.divide((s - s_old), Del_S)
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
				Del_S = max(norm(np.dot(S_hat-s_old, active_subspace), axis=1))
			
			def phi_function(s):
				u = np.divide(np.dot((s - s_old), active_subspace), Del_S)
				# print(f'shape of u: {u.shape}')
				try:
					m,n = u.shape
				except:
					m = 1
					u = u.reshape(1,-1)
				phi = np.zeros((m, self.q))
				for k in range(self.q):
					phi[:,k] = np.prod(np.divide(np.power(u, self.index_set[k,:]), factorial(self.index_set[k,:])), axis=1)
				if m == 1:
					return phi.flatten()
				else:
					return phi
			
			def phi_function_deriv(s):
				u = np.divide(np.dot((s - s_old), active_subspace), Del_S)
				phi_deriv = np.zeros((self.d, self.q))
				for i in range(self.d):
					for k in range(1, self.q):
						if self.index_set[k, i] != 0.0: 
							tmp = np.zeros(self.d)
							tmp[i] = 1
							phi_deriv[i,k] = self.index_set[k, i] * np.prod(np.divide(np.power(u, self.index_set[k,:]-tmp), factorial(self.index_set[k,:]))) 
				phi_deriv = np.divide(phi_deriv.T, Del_S).T
				return np.dot(active_subspace.U, phi_deriv)
		
		return phi_function, phi_function_deriv
	
	def find_new_point(self, v, phi_function, phi_function_deriv, s_old, delta_k, problem, full_space=False):
		#change bounds to be defined using the problem and delta_k
		bounds_l = np.maximum(np.array(problem.lower_bounds).reshape(s_old.shape), s_old-delta_k)
		bounds_u = np.minimum(np.array(problem.upper_bounds).reshape(s_old.shape), s_old+delta_k)

		bounds = []
		for i in range(self.n):
			bounds.append((bounds_l[i], bounds_u[i])) 
		
		if full_space:
			c = v[1:]
			res1 = linprog(c, bounds=bounds)
			res2 = linprog(-c, bounds=bounds)
			if abs(np.dot(v, phi_function(res1['x']))) > abs(np.dot(v, phi_function(res2['x']))):
				s = res1['x']
			else:
				s = res2['x']
		else:
			obj1 = lambda s: np.dot(v, phi_function(s))
			jac1 = lambda s: np.dot(phi_function_deriv(s), v)
			obj2 = lambda s: -np.dot(v, phi_function(s))
			jac2 = lambda s: -np.dot(phi_function_deriv(s), v)
			res1 = minimize(obj1, s_old, method='TNC', jac=jac1, \
					bounds=bounds, options={'disp': False})
			res2 = minimize(obj2, s_old, method='TNC', jac=jac2, \
					bounds=bounds, options={'disp': False})
			#FIX: Don't want to rely on this (possibly)
			if abs(res1['fun']) > abs(res2['fun']):
				s = res1['x']
			else:
				s = res2['x']
		return s

	def find_new_point_alternative(self, v, phi_function, S, s_old, delta_k, problem):
		S_tmp = self.generate_set(int(0.5*(self.n+1)*(self.n+2)), s_old, delta_k, problem)
		M = np.absolute(np.dot(phi_function(S_tmp), v).flatten())
		indices = np.argsort(M)[::-1][:len(M)]
		for index in indices:
			s = S_tmp[index,:]
			if np.unique(np.vstack((S, s)), axis=0).shape[0] == S.shape[0]+1:
				return s
		return S_tmp[indices[0], :]

	@staticmethod
	def remove_point_from_set(S, f, s):
		ind_current = np.where(norm(S-s, axis=1, ord=np.inf) == 0.0)[0]
		S = np.delete(S, ind_current, 0)
		f = np.delete(f, ind_current, 0)
		return S, f

	@staticmethod
	def remove_furthest_point(S, f, s):
		ind_distant = np.argmax(norm(S-s, axis=1, ord=np.inf))
		S = np.delete(S, ind_distant, 0)
		f = np.delete(f, ind_distant, 0)
		return S, f

	def remove_points_outside_limits(self, S, s_old, delta_k, rho_k):
		ind_inside = np.where(norm(S-s_old, axis=1, ord=np.inf) <= max(self.epsilon_1*delta_k, self.epsilon_2*rho_k))[0]
		S = S[ind_inside, :]
		f = f[ind_inside]
		return S, f
	
	""" 
		SAMPLING METHODS  
	"""
	def calculate_pilot_run(self, problem) :
		lambda_min = self.factors['lambda_min']
		lambda_max = problem.factors['budget'] - self.expended_budget
		self.pilot_run = ceil(max(lambda_min * log(10 + self.k, 10) ** 1.1, min(0.5 * problem.dim, lambda_max))-1)

		if self.k == 1 :
			problem.simulate(self.current_solution, self.pilot_run)
			self.expended_budget += self.pilot_run
			sample_size = self.pilot_run

			self.__calculate_kappa(problem)
		elif self.factors['crn_across_solns'] :
			# since incument was only evaluated with the sample size of previous incumbent, here we compute its adaptive sample size
			sample_size = self.current_solution.n_reps
			sig2 = self.current_solution.objectives_var[0]
			# adaptive sampling
			self.current_solution = self.adaptive_sampling_1(problem, self.current_solution, sample_size, sig2)
			

		# return current_solution, expended_budget

	def get_stopping_time(self, sig2: float, dim: int,) -> int:
		"""
		Compute the sample size based on adaptive sampling stopping rule using the optimality gap
		"""
		if self.kappa == 0:
			self.kappa = 1
		# lambda_k = max(
		#     self.factors["lambda_min"], 2 * log(dim + 0.5, 10)
		# ) * max(log(k + 0.1, 10) ** (1.01), 1)

		# compute sample size
		raw_sample_size = self.pilot_run * max(1, sig2 / (self.kappa**2 * self.delta_k**self.delta_power))
		# Convert out of ndarray if it is
		if isinstance(raw_sample_size, np.ndarray):
			raw_sample_size = raw_sample_size[0]
		# round up to the nearest integer
		sample_size: int = ceil(raw_sample_size)
		return sample_size
	
	def __calculate_kappa(self, problem) :
		
		lambda_max = problem.factors['budget'] - self.expended_budget
		while True:
			rhs_for_kappa = self.current_solution.objectives_mean
			sig2 = self.current_solution.objectives_var[0]
			if self.delta_power == 0:
				sig2 = max(sig2, np.trace(self.current_solution.objectives_gradients_var),)
			self.kappa = rhs_for_kappa * np.sqrt(self.pilot_run) / (self.delta_k ** (self.delta_power / 2))
			stopping = self.get_stopping_time(sig2, problem.dim,)
			if (sample_size >= min(stopping, lambda_max) or self.expended_budget >= problem.factors['budget']):
				# calculate kappa
				# self.kappa = (rhs_for_kappa * np.sqrt(self.pilot_run)/ (delta_k ** (self.delta_power / 2)))
				# print("kappa "+str(kappa))
				break
			problem.simulate(self.current_solution, 1)
			self.expended_budget += 1
			sample_size += 1

		#return current_solution, used_budget


	def get_sig_2(self, solution) :
		return solution.objectives_var[0]
	
	#this is sample_type = 'conditional after'
	def adaptive_sampling_1(self, problem, solution, sample_size, sig2) :
		lambda_max = problem.factors['budget'] - self.expended_budget
		
		problem.simulate(solution, self.pilot_run)
		self.expended_budget += self.pilot_run
		sample_size = self.pilot_run

		# adaptive sampling
		while True:
			sig2 = solution.objectives_var[0]
			stopping = self.get_stopping_time(sig2, problem.dim,)
			if (sample_size >= min(stopping, lambda_max) or self.expended_budget >= problem.factors['budget']):
				break
			problem.simulate(solution, 1)
			self.expended_budget += 1
			sample_size += 1
		return solution
		# return new_solution, used_budget
	
	#this is sample_type = 'conditional before'	
	def adaptive_sampling_2(self, problem, solution, sample_size, sig2) : 
		lambda_max = problem.factors['budget'] - self.expended_budget
		while True:
			stopping = self.get_stopping_time(sig2, problem.dim,)
			if (sample_size >= min(stopping, lambda_max) or self.expended_budget >= problem.factors['budget']):
				break
			problem.simulate(solution, 1)
			self.expended_budget += 1
			sample_size += 1
			sig2 = solution.objectives_var[0]
		return solution

	#!This is the only function for sampling that should be called 
	def adaptive_sampling(self, problem, solution, sample_size, sig2, sample_after=True) :
		if sample_after :
			return self.adaptive_sampling_1(problem, solution, 0, sig2)
		
		return self.adaptive_sampling_2(problem, solution, sample_size, sig2)