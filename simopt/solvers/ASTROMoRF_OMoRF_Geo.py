"""
Summary
-------
The ASTROMoRF (Adaptive Sampling for Trust-Region Optimisation by Moving Ridge Functions) progressively builds local models using
interpolation on a reduced subspace constructed through Active Subspace dimensionality reduction.
The use of Active Subspace reduction allows for a reduced number of interpolation points to be evaluated in the model construction.

"""

#TODO: FIX how subspace dimension 1 problems are handled 
#TODO: Fix how rho_k is handlded - look at updating trust region radius updates
#TODO: FIX GEOMETRY IMPROVEMENT 
#?: SEE IF WE CAN REDUCE THE NUMBER OF TIMES WE HAVE TO SAMPLE - may reduce the interpolation set size? 
#?: SEE IF WE CAN REUSE MORE THAN ONE DESIGN POINT - Not possible as then we won't have a span of the region
from __future__ import annotations
from typing import Callable

from numpy.linalg import norm, pinv, qr
import numpy as np
from math import ceil, isnan, isinf, comb, log
import warnings
from scipy.optimize import minimize, NonlinearConstraint, linprog
from scipy import optimize
from scipy.special import factorial
import scipy
warnings.filterwarnings("ignore")
import importlib
from copy import deepcopy
import inspect


from simopt.linear_algebra_base import finite_difference_gradient
from simopt.utils import classproperty, override


from simopt.base import (
	ConstraintType,
	ObjectiveType,
	Problem,
	Solution,
	Solver,
	VariableType,
)

from simopt.solvers.active_subspaces.basis import *
from simopt.solvers.active_subspaces.index_set import IndexSet


class BadStep(Exception):
	pass

#TODO: Rewrite model construction with adaptive sampling 
class ASTROMoRF_OMoRFGeo(Solver):
	"""Combining the ASTRO-DF solver and the OMoRF solver - uses the OMoRF Geometry 

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
		return "ASTROMoRF-GEO"

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
				"default": "LegendreTensorBasis"
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
				"default": 3
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
			"max iterations": {
				"description": "maximum number of iterations for the gauss newton",
				"datatype": int, 
				"default": 100
			}, 
			"omega shrinking": {
				"description": "factor to shrink rho by when evaluating the stepsize",
				"datatype": float, 
				"default": 0.5
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
			"rho_min": self.check_rho_min, 
			"max iterations": self.check_iterations, 
			"omega shrinking": self.check_omega
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
		if self.factors['alpha_1'] <= 0 :
			raise ValueError('Alpha_1 needs to be positive')
	
	def check_alpha_2(self) -> bool :
		if self.factors['alpha_2'] <= 0 :
			raise ValueError('Alpha_2 needs to be positive')
	
	def check_rho_min(self) -> bool : 
		if self.factors['rho_min'] <= 0 : 
			raise ValueError('rho_min needs to be positive')


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
		
	def check_iterations(self) -> None :
		if type(self.factors['max iterations']) is not int : 
			raise ValueError(
				"the maximum number of iterations has to be an integer"
			)
		
	def check_omega(self) -> None : 
		if self.factors['omega shrinking'] < 0 and self.factors['omega shrinking'] > 1 : 
			raise ValueError(
				"Omega must be positive and less than 1"
			)

	def __init__(self, name="ASTROMoRF-GEO", fixed_factors: dict | None = None) -> None:
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


	def initialise_factors(self,problem) :
		#initialise factors: 
		self.recommended_solns = []
		self.intermediate_budgets = []

		self.kappa = 1
		self.delta_power = 2 if self.factors['crn_across_solns'] else 4

		self.expended_budget = 0

		self.S = np.array([])
		self.f = np.array([])
		self.g = np.array([])
		self.d = self.factors['subspace dimension']

		self.delta_max = self.calculate_max_radius(problem)

		self.delta_k = 10 ** (ceil(log(self.delta_max * 2, 10) - 1) / problem.dim)
		self.rho_k = self.delta_k
		self.rhos = []
		self.deltas = []
		self.n = problem.dim
		self.d = self.factors['subspace dimension']
		self.deg = self.factors['polynomial degree'] 
		self.q = int(0.5*(self.d+1)*(self.d+2)) # comb(self.d + self.deg, self.deg) +  self.n * self.d
		self.p = self.n+1
		self.epsilon_1, self.epsilon_2 = self.factors['interpolation update tol'] #epsilon is the tolerance in the interpolation set update 
		self.random_initial = self.factors['random directions']
		self.alpha_1 = self.factors['alpha_1'] #shrink the trust region radius in set improvement 
		self.alpha_2 = self.factors['alpha_2'] #shrink the stepsize reduction  
		self.rho_min = self.factors['rho_min']
		
		self.beta_k = 0.5
		self.gamma_k = 0.5
		self.mu_k = 0.5


		#set up initial Solution
		self.current_solution = self.create_new_solution(problem.factors["initial_solution"], problem)
		self.recommended_solns.append(self.current_solution)
		self.intermediate_budgets.append(self.expended_budget)

		self.basis = self.polynomial_basis_instantiation()(self.deg, problem, X=None, dim=problem.dim)
		self.reduced_basis = self.polynomial_basis_instantiation()(self.deg, problem, X=None, dim=self.d)
		index_set = IndexSet('total-order', orders=np.tile([2], self.q))
		self.index_set = index_set.get_basis()[:,range(self.d-1, -1, -1)]

		problem.simulate(self.current_solution, 1)
		self.s_old = np.array(self.current_solution.x)
		self.f_old = self.blackbox_evaluation(problem, solution=self.current_solution)[1]
		print(f'function evaluation of current solution at instantiation: {self.f_old}')
		self.U = self.initialise_subspace(self.current_solution, self.delta_k)

		self.S_red, self.f_red = self.sample_set(problem, 'new', self.current_solution, self.delta_k, self.rho_k, self.f_old, self.U)

		# Construct the sample set for the subspace 
		self.S_full = self.generate_set(problem, self.n + 1, self.s_old, self.delta_k)
		self.f_full = np.zeros((self.n + 1, 1))
		self.f_full[0, :] = self.f_old #first row gets the old function values 

	def solve(self, problem):
		#initialise factors: 
		self.initialise_factors(problem)

		k = 0


		while self.expended_budget < problem.factors['budget'] : 
			k += 1 
			print(f'iteration: {k} \t expended budget {self.expended_budget} \t current objective function value: {self.current_solution.x} with type {type(self.current_solution.x)}')

			if k == 1 :
				self.current_solution = self.create_new_solution(self.current_solution.x, problem)
				# current_solution = self.create_new_solution(current_solution.x, problem)
				
				self.current_solution, self.expended_budget = self.calculate_kappa(k, problem, self.expended_budget, self.current_solution, self.delta_k)
			
				self.recommended_solns.append(self.current_solution)
				self.intermediate_budgets.append(self.expended_budget)
			elif self.factors['crn_across_solns'] :
				# since incument was only evaluated with the sample size of previous incumbent, here we compute its adaptive sample size
				# adaptive sampling
				# current_solution, expended_budget = sampling_instance(problem, k, current_solution, delta_k, expended_budget, False)
				lambda_max = problem.factors['budget'] - self.expended_budget
				sample_size = self.current_solution.n_reps
				while True:
					print(f'number of replications of the current solution: {self.current_solution.n_reps}')
					sig2 = self.current_solution.objectives_var[0]
					stopping = self.get_stopping_time(sig2, self.delta_k, k, problem, self.expended_budget)
					if (sample_size >= min(stopping, lambda_max) or self.expended_budget >= problem.factors['budget']):
						break
					problem.simulate(self.current_solution, 1)
					self.expended_budget += 1
					sample_size += 1
				
			#build random model 
			self.current_solution, self.delta_k, self.S_red, self.f_red, self.expended_budget, self.U, self.gamma_k, self.beta_k, self.mu_k = self.construct_model(problem, self.current_solution, self.S_red, self.f_red, self.delta_k, k, self.expended_budget, self.gamma_k, self.beta_k, self.mu_k)
			print('CONSTRUCTED MODEL\n')
			#solve random model 
			candidate_solution, self.delta_k = self.solve_subproblem(problem, self.current_solution, self.delta_k, self.U)
			print('FOUND CANDIDATE SOLUTION \n')

			#sample candidate solution
			candidate_solution, fval_tilde = self.blackbox_evaluation(problem, solution=candidate_solution)
			print('ADAPTIVELY SIMULATED CANDIDATE SOLUTION \n')

			#evaluate model
			self.current_solution, self.delta_k, self.recommended_solns, self.expended_budget, self.intermediate_budgets = self.evaluate_candidate_solution(problem, self.U, self.S_red, self.f_red, fval_tilde, self.delta_k, self.current_solution, candidate_solution, self.recommended_solns, self.expended_budget, self.intermediate_budgets)	
			print('EVALUATED CANDIDATE SOLUTION \n')


			# print('new solution: ', new_solution.x)

		return self.recommended_solns, self.intermediate_budgets


	def blackbox_evaluation(self, problem, solution: None | Solution = None, value: None | np.ndarray = None, visted_pts_list: None | list[Solution] = None) -> tuple[Solution, float] : 
		if solution is None and value is None and visted_pts_list is None:
			raise ValueError('At least one of solution, value or visted points list has to be passed') 
		elif value is not None : 
			solution = self.create_new_solution(tuple(value), problem)
		
		#Case where we are reusing the design point, we can reuse the replications
		if (visted_pts_list is not None) and (solution.x in [a.x for a in visted_pts_list]) : 
			x_vals = [a.x for a in visted_pts_list] 
			index = x_vals.index(solution.x)
			fval_tilde = -1 * problem.minmax[0] * visted_pts_list[index].objectives_gradients_mean
			return visted_pts_list[index], fval_tilde
		
		
		if self.factors['crn_across_solns'] :
			problem.simulate(solution, self.current_solution.n_reps) 
			self.expended_budget += self.current_solution.n_reps 
			fval_tilde = -1 * problem.minmax[0] * solution.objectives_mean
			return solution, fval_tilde
		else :
			solution, self.expended_budget = self.adaptive_sampling(problem, self.k, solution, self.delta_k, self.expended_budget)
			fval_tilde = -1 * problem.minmax[0] * solution.objectives_mean
			return solution, fval_tilde


	def solve_subproblem(self, problem: Problem, current_solution:Solution, delta_k: float, U: np.ndarray) :
		"""
			Solves the trust-region subproblem
		"""
		omega_s = self.factors['omega shrinking']
		
		cons =  NonlinearConstraint(lambda x : norm(x), 0, self.delta_k)
		
		obj = lambda x: self.model_evaluate(x, U).item(0)
		stepsize = minimize(obj, np.zeros(self.n), method='trust-constr', constraints=cons, options={'disp': False}).x
		s_new = self.s_old + stepsize

		print(f'CANDIDATE SOLUTION: {s_new}')


		for i in range(problem.dim):
			if s_new[i] <= problem.lower_bounds[i]:
				s_new[i] = problem.lower_bounds[i] + 0.01
			elif s_new[i] >= problem.upper_bounds[i]:
				s_new[i] = problem.upper_bounds[i] - 0.01

		step_dist = norm(s_new - self.s_old, ord=np.inf)

		# Safety step implemented in BOBYQA
		if step_dist < omega_s*self.rho_k:
			self.ratio = -0.1
			self.delta_k = max(0.5*self.delta_k, self.rho_k)
			self.S_red, self.f_red, self.delta_k, self.rho_k, self.U = self.improve_interpolation_geometry(problem, self.s_old, self.f_old, self.delta_k, self.rho_k, self.U, self.S_red, self.f_red)
			


		candidate_solution = self.create_new_solution(tuple(s_new), problem)


		print(f'stepsize: {stepsize}')

		return candidate_solution, delta_k

	def evaluate_candidate_solution(self, problem, U, S_red, f_red, fval_tilde, delta_k, current_solution, candidate_solution, recommended_solns, expended_budget, intermediate_budgets) :
		#turn S_red and f_red into a list of vectors and list of floats - it is each row is a interpolation point 
		S_red_list = S_red.tolist() 
		f_red_list = [a[0] for a in f_red.tolist()]




		#pattern search
		#TODO: fval should be a list of the solution of all the interpolation solutions 
		if ((min(f_red_list) < fval_tilde) and ((f_red_list[0] - min(f_red_list))>= self.factors["ps_sufficient_reduction"] * delta_k**2)) or ((candidate_solution.objectives_var[0]/ (candidate_solution.n_reps * candidate_solution.objectives_mean[0]**2)) > 0.75):
			fval_tilde = min(f_red_list)
			candidate_solution_x = S_red_list[f_red_list.index(min(f_red_list))]  # type: ignore
			candidate_solution = self.create_new_solution(tuple(candidate_solution_x), problem)

		stepsize = np.subtract(np.array(candidate_solution.x), np.array(current_solution.x))
		model_eval_old = self.model_evaluate(np.array(current_solution.x), U).item()
		model_eval_new = self.model_evaluate(np.array(candidate_solution.x), U).item()

		del_f =  f_red_list[0] - fval_tilde #self.f_old - f_new 
		del_m = model_eval_old - model_eval_new

		if del_f < 0:
			rho = 0
		else:
			# difference = np.subtract(candidate_solution.x, current_solution.x)
			rho = (del_f/del_m)[0] if type(del_f/del_m) == list else del_f/del_m

		print(f'DIFFERENCE IN CANDIDATE EVALUATION AT MODEL AND FUNCTION: {abs(fval_tilde - model_eval_new)}')
		print(f'The model evaluation for the old value is {model_eval_old} and for the candidate value it is {model_eval_new}')
		print(f'The old function value is {f_red_list[0]} and the new function value is {fval_tilde}')

		print(f'numerator of ratio is {del_f} and the denominator is {del_m}')
		print(f'the ratio is {rho}\n')

		s_new = np.array(candidate_solution.x)
		self.S_red, self.f_red = self.sample_set(problem, 'replace', current_solution, delta_k, self.rho_k, self.f_old, self.U, self.S_red, self.f_red, s_new, fval_tilde)

		# successful: accept
		if rho >= self.factors['eta_1']:
			print('SUCCESSFUL')
			# new_x = candidate_x
			current_solution = candidate_solution
			# final_ob = candidate_solution.objectives_mean
			recommended_solns.append(candidate_solution)
			intermediate_budgets.append(expended_budget)
			delta_k = min(delta_k, self.delta_max)
			self.s_old = np.array(current_solution.x)
			self.f_old = fval_tilde
			
		# very successful: expand and accept
		elif rho >= self.factors['eta_2'] :
			print('VERY SUCCESSFUL')
			current_solution = candidate_solution
			# final_ob = candidate_solution.objectives_mean
			recommended_solns.append(candidate_solution)
			intermediate_budgets.append(expended_budget)
			delta_k = min(self.factors['gamma_1'] * delta_k, self.delta_max)
			self.s_old = np.array(current_solution.x)
			self.f_old = fval_tilde
		
		# unsuccessful: shrink and reject
		else:
			print('UNSUCCESSFUL')
			delta_k = min(self.factors['gamma_2'] * delta_k, self.delta_max)
			self.S_red, self.f_red, delta_k, self.rho_k, self.U = self.improve_interpolation_geometry(problem, self.s_old, self.f_old, delta_k, self.rho_k, self.U, self.S_red, self.f_red)
			# new_solution = current_solution
			# recommended_solns.append(current_solution)
			# final_ob = fval[0]

		return current_solution, delta_k, recommended_solns, expended_budget, intermediate_budgets
	
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
		SAMPLING METHODS  
	"""

	def calculate_pilot_run(self, k, problem, expended_budget) : 
		lambda_min = self.factors['lambda_min']
		lambda_max = problem.factors['budget'] - expended_budget
		return ceil(max(lambda_min * log(10 + k, 10) ** 1.1, min(0.5 * problem.dim, lambda_max))-1)

	def calculate_kappa(self, k, problem, expended_budget, current_solution, delta_k) :
		lambda_max = problem.factors['budget'] - expended_budget
		pilot_run = self.calculate_pilot_run(k, problem, expended_budget)

		#calculate kappa
		problem.simulate(current_solution, pilot_run)
		expended_budget += pilot_run

		# current_solution, expended_budget = self.__calculate_kappa(problem, current_solution, delta_k, expended_budget)
		sample_size = pilot_run
		
		while True:
			rhs_for_kappa = current_solution.objectives_mean
			sig2 = current_solution.objectives_var[0]

			self.kappa = rhs_for_kappa * np.sqrt(pilot_run) / (delta_k ** (self.delta_power / 2))
			stopping = self.get_stopping_time(sig2, delta_k, k, problem, expended_budget)
			if (sample_size >= min(stopping, lambda_max) or expended_budget >= problem.factors['budget']):
				# calculate kappa
				self.kappa = (rhs_for_kappa * np.sqrt(pilot_run)/ (delta_k ** (self.delta_power / 2)))
				# print("kappa "+str(kappa))
				break
			problem.simulate(current_solution, 1)
			expended_budget += 1
			sample_size += 1

		return current_solution, expended_budget

	def get_stopping_time(self, sig2: float, delta: float, k: int, problem: Problem, expended_budget: int) -> int:
		"""
		Compute the sample size based on adaptive sampling stopping rule using the optimality gap
		"""
		pilot_run = self.calculate_pilot_run(k, problem, expended_budget)
		if self.kappa == 0:
			self.kappa = 1

		# compute sample size
		raw_sample_size = pilot_run * max(1, sig2 / (self.kappa**2 * delta**self.delta_power))
		# Convert out of ndarray if it is
		if isinstance(raw_sample_size, np.ndarray):
			raw_sample_size = raw_sample_size[0]
		# round up to the nearest integer
		sample_size: int = ceil(raw_sample_size)
		return sample_size
	
	#this is sample_type = 'conditional after'
	def adaptive_sampling_1(self, problem: Problem, k: int, new_solution: Solution, delta_k: float, used_budgetL: int) :
		lambda_max = problem.factors['budget'] - used_budget
		pilot_run = self.calculate_pilot_run(k, problem, used_budget)

		problem.simulate(new_solution, pilot_run)
		used_budget += pilot_run
		sample_size = pilot_run

		# adaptive sampling
		while True:
			sig2 = new_solution.objectives_var[0]
			stopping = self.get_stopping_time(sig2, delta_k, k, problem, used_budget)
			if (sample_size >= min(stopping, lambda_max) or used_budget >= problem.factors['budget']):
				break
			problem.simulate(new_solution, 1)
			used_budget += 1
			sample_size += 1

		return new_solution, used_budget
	
	#this is sample_type = 'conditional before'	
	def adaptive_sampling_2(self, problem, k, new_solution, delta_k, used_budget) : 
		lambda_max = problem.factors['budget'] - used_budget
		sample_size = new_solution.n_reps 
		sig2 = new_solution.objectives_var[0]

		while True:
			stopping = self.get_stopping_time(sig2, delta_k, k, problem, used_budget)
			if (sample_size >= min(stopping, lambda_max) or used_budget >= problem.factors['budget']):
				break
			problem.simulate(new_solution, 1)
			used_budget += 1
			sample_size += 1
			sig2 = new_solution.objectives_var[0]
		return new_solution, used_budget


	def adaptive_sampling(self, problem, k, new_solution, delta_k, used_budget, sample_after=True) :
		if sample_after :
			return self.adaptive_sampling_1(problem, k, new_solution, delta_k, used_budget)
		
		return self.adaptive_sampling_2(problem, k, new_solution, delta_k, used_budget)
	

	def construct_model(self, problem: Problem, current_solution: Solution, X: np.ndarray, fX: np.ndarray, delta_k: float, k: int, expended_budget: int, gamma_k: float, beta_k: float, mu_k: float) : 

		#initialise subspace 
		U0 = self.initialise_subspace(current_solution, delta_k)

		#run the variable projection algorithm 
		active_subspace, coefficients, fX, expended_budget, delta_k, gamma_k, beta_k, mu_k  = self.var_proj_model(problem, current_solution, X, fX, U0, self.d, self.deg, delta_k, gamma_k, beta_k, mu_k) 

		print('COMPLETED VARIABLE PROJECTION\n')

		self.coefficients = coefficients

		return current_solution, delta_k, X, fX, expended_budget, active_subspace, gamma_k, beta_k, mu_k
	
	def model_evaluate(self, X, U) : 
		X = np.array(X)	
		Y = (U.T @ X.T).T
		V = self.reduced_basis.V(Y)
		Vc = V @ self.coefficients
		return Vc 

	def grad(self, X, coeff, U):
		if len(X.shape) == 1:
			one_d = True
			X = X.reshape(1,-1)	
		else:
			one_d = False	
		
		#currently DV is of the number of iteraitons
		DV = self.reduced_basis.DV(X) #should be shape (problem.dim, len(coeff), self.d)

		# Compute gradient on projected space
		Df = np.tensordot(DV, coeff, axes = (1,0)) #should be shape (problem.dim, self.d, 1)

		# Inflate back to whole space
		Df = np.tensordot(Df, U.T, axes=(1,0)) #result should be shape (problem.dim, 1, problem.dim) - should also be tensor product between Df and U and should be between self.d
		
		if one_d:
			return Df.reshape(X.shape[1])
		else:
			return np.squeeze(Df, axis=1) #needs to be shape (problem.dim, problem.dim)


	def initialise_subspace(self, current_solution: Solution, delta_k: float) -> np.ndarray : 
		"""build the initial subspace by sampling points within the trust region

		Args:
			current_solution (Solution): The current Solution 
			delta_k (float): _description_

		Returns:
			np.ndarray: _description_
		"""
		#inner function to allow for mapping 
		def Z_mapped(Z, current_solution, delta_k) : 
			x_k = np.array(current_solution.x)
			# Normalize each vector to have unit norm
			norms = np.linalg.norm(Z, axis=1, keepdims=True)
			unit_vectors = Z / norms
			
			# Scale to the desired radius - need to multiply each vector by 
			scaled_vectors = unit_vectors * delta_k
			
			# Shift to be centered at x_k
			final_vectors = scaled_vectors + x_k[:, np.newaxis]
			return final_vectors 	
		
		Z = np.random.randn(self.n, self.d) #sample from normal distribution to get d samples of dimension n 

		#need to translate and scale the points into the trust region 
		Z = Z_mapped(Z, current_solution, delta_k)
		#undergo QR decomposition to get initial U 
		U, _ =  np.linalg.qr(Z)

		return U 

	def var_proj_model(self, problem: Problem, current_solution: Solution, X: np.ndarray, fX: np.ndarray, U: np.ndarray, d: int, deg: int, delta_k: float, gamma_k: float, beta_k: float, mu_k: float, conv_tol: float = 1e-05) -> tuple[
		np.ndarray,
		list[float],
		np.ndarray,
		int,
		float,
		float, 
		float, 
		float,
		] : 
		"""_summary_

		Args:
			problem (Problem): 
			X (np.ndarray): _description_
			fX (np.ndarray): _description_
			U (np.ndarray): The initial active subspace matrix 
			d (int): _description_
			deg (int): _description_
			delta_k (float): _description_
			model_vals (dict[str,float]): _description_

		Returns:
			tuple[ np.ndarray, list[float], np.ndarray, int, float, list[Solution] ]: _description_
		"""
		n = 1
		# previous_U = np.full((self.n,self.d),np.inf)
		#loop until U has converged in the GN step 
		while True and n <= self.factors['max iterations'] :
			# print(f'iteration number of the variable projection model: {n}')
			#! Calculate the coefficients of the model
			coeff = self.coefficient(X,fX,U)

			#FIX: unsure if the criticality check should be here?
			if False and delta_k < mu_k*norm(coeff) : #criticality check 
				print('criticality check started')
				delta_k = max(self.factors['omega_shrinking']*delta_k, mu_k)
				f_old = -1 * problem.minmax[0] * current_solution.objectives_mean
				X, fX, fval, visited_pts, mu_k, delta_k = self.improve_interpolation_geometry(current_solution, problem, visited_pts, fval, f_old, delta_k, mu_k, U, X, fX) #!Need to remove ratio, rho_k, and f_old 
				print(f'criticality check ended with budget {self.expended_budget}')
				n += 1
				continue 

			#! undergo Gaussian Newton to update active subspace 
			#build the jacobian
			def jacobian(U_flat):
				return self.varpro_jacobian(X, fX, U_flat)

			#build the residual 
			def residual(U_flat):
				return self.varpro_residual(X, fX, U_flat)

			#* This gauss_newton algorithm undergoes steps 16 to 31 of the algorithm 
			U0_flat = U.flatten() 
			# print('undergoing gauss newton step')
			U_flat, gamma_k, beta_k = self.gauss_newton(residual, jacobian, U0_flat, gamma_k, beta_k) 
			# print(f'\n GAUSS NEWTON STEP COMPLETED AT ITERATION {n}')
			# print('finished gauss newton step')
			U_plus = U_flat.reshape(-1, self.d)

			#TODO: change this to the halting conditions of gauss_netwon in PSDR 
			if np.allclose(U_plus, U, rtol=conv_tol) : 
				# print('subspace has converged')
				U = U_plus 
				break 
			else : 
				U = U_plus 
				# previous_U = U_plus
				n += 1
			
		coeff, U = self.finish(X, fX, U) #realigns the active subspace and also recalculates the model coefficients 
		print(f'Finished variable projection model with {n} iterations')
		return U, coeff, fX, self.expended_budget, delta_k, gamma_k, beta_k, mu_k
		


	def coefficient(self, X: np.ndarray, fX: np.ndarray, U: np.ndarray) -> np.ndarray : 
		x = [np.array(a).reshape(-1,1) for a in X] #list of numpy vectors of shape (n,1)
		Y = [(U.T @ a).reshape(1,-1) for a in x] #list of vectors of shape (1,d)
		V = self.reduced_basis.V(np.vstack(Y)) 
		coeff = np.matmul(pinv(V),fX)  #fit the initial coefficients
		return coeff


	#TODO: Clean up this function
	# Remove profile from grads  
	def finish(self, X, fX, U):
		r""" Given final U, rotate and find coefficients
		"""

		Y = (U.T @ X.T).T
		# Step 1: Apply active subspaces to the profile function at samples X
		# to rotate onto the most important directions
		# Step 2: Flip signs such that average slope is positive in the coordinate directions
		coeff = self.coefficient(X, fX, U)
		grads = self.grad(Y, coeff, U)


		if U.shape[1] > 1 : 
			Ur = scipy.linalg.svd(grads.T, full_matrices = False)[0] #grads.T should be (problem.dim, problem.dim)
			U = Ur @ U #should be (problem.dim,d) => Ur should have shape (problem.dim, problem.dim)
		else : 
			print(f'mean of grads: {np.mean(grads, axis=0).shape}') #this needs to be (problem.dim, problem.dim)
			U = (U.T @ np.sign(grads)).T 
			print(f'shape of U after rotating: {U.shape}')
	
		# Step 3: final fit	
		coeff = self.coefficient(X, fX, U)

		return coeff, U



	"""
		VARIABLE PROJECTION CODE AND GAUSSIAN NEWTON STEP

	"""


	def varpro_residual(self, X, fX, U_flat):
		U = U_flat.reshape(X.shape[1],-1)
		#V = self.V(X, U)
		Y = (U.T @ X.T).T
		# self.basis = self.Basis(self.degree, Y)
		V = self.reduced_basis.V(Y)
		if self.reduced_basis.__name__ == 'ArnoldiPolynomialBasis':
			# In this case, V is orthonormal
			c = V.T @ fX
		else:
			c = scipy.linalg.lstsq(V, fX)[0].flatten()
		r = fX - V.dot(c).reshape(-1,1)
		return r

	def varpro_jacobian(self, X, fX, U_flat):
		# Get dimensions
		M, m = X.shape
		U = U_flat.reshape(X.shape[1],-1)
		m, n = U.shape

		Y = (U.T @ X.T).T
		# self.basis = self.Basis(self.degree, Y)
		V = self.reduced_basis.V(Y)
		DV = self.reduced_basis.DV(Y)

		# if isinstance(self.basis, ArnoldiPolynomialBasis):
		if self.reduced_basis.__name__ == 'ArnoldiPolynomialBasis' : 
			# In this case, V is orthonormal
			c = V.T @ fX
			Y = np.copy(V)
			s = np.ones(V.shape[1])
			ZT = np.eye(V.shape[1])
		else:
			c = scipy.linalg.lstsq(V, fX)[0].flatten()
			Y, s, ZT = scipy.linalg.svd(V, full_matrices = False)
			s = np.array([np.inf if x == 0.0 else x for x in s]) 


		r = fX.reshape(-1,) - V.dot(c).reshape(-1,)


		N = V.shape[1]
		J1 = np.zeros((M,m,n))
		J2 = np.zeros((N,m,n))

		for ell in range(n):
			for k in range(m):
				DVDU_k = X[:,k,None]*DV[:,:,ell]

				# This is the first term in the VARPRO Jacobian minus the projector out fron
				J1[:, k, ell] = DVDU_k.dot(c)
				# This is the second term in the VARPRO Jacobian before applying V^-
				J2[:, k, ell] = DVDU_k.T.dot(r) 

		# Project against the range of V
		J1 -= np.tensordot(Y, np.tensordot(Y.T, J1, (1,0)), (1,0))
		# Apply V^- by the pseudo inverse
		J2 = np.tensordot(np.diag(1./s),np.tensordot(ZT, J2, (1,0)), (1,0))
		J = -( J1 + np.tensordot(Y, J2, (1,0)))
		return J.reshape(J.shape[0], -1)
	
	def orth(self,U):
		""" Orthgonalize, but keep directions"""
		U, R = np.linalg.qr(U, mode = 'reduced')
		U = np.dot(U, np.diag(np.sign(np.diag(R)))) 
		return U

	def grassmann_trajectory(self, U_flat, Delta_flat, t):
		Delta = Delta_flat.reshape(-1, self.d)
		U = U_flat.reshape(-1, self.d)
		Y, s, ZT = scipy.linalg.svd(Delta, full_matrices = False, lapack_driver = 'gesvd')
		UZ = np.dot(U, ZT.T)
		U_new = np.dot(UZ, np.diag(np.cos(s*t))) + np.dot(Y, np.diag(np.sin(s*t)))
		U_new = self.orth(U_new).flatten()
		return U_new

	def linesearch_armijo(self, f, g, p, x0, alpha, gamma_k, beta_k):
		"""Back-Tracking Line Search to satify Armijo Condition

			f(x0 + alpha*p) < f(x0) + alpha * ftol * <g,p>

		Parameters
		----------
		f : callable
			objective function, f: R^n -> R
		g : np.array((n,))
			gradient
		p : np.array((n,))
			descent direction
		x0 : np.array((n,))
			current location (this concerns the active subspace matrix )
		maxiter : int [optional] default = 10
			maximum number of iterations of backtrack
		trajectory: function(x0, p, t) [Optional]
			Function that returns next iterate 
		Returns
		-------
		float
			alpha: backtracking coefficient (alpha = 1 implies no backtracking)
		"""

		r = f(x0)

		r_norm = norm(r)
		x = np.copy(x0)
		fx = np.inf
		
		t = 1
		n = 1
		while True and n <= self.factors['max iterations'] :
			try:
				# print(f'starting linesearch with iteration {n}')
				x = self.grassmann_trajectory(x0, p, t) #* line 26 
				r_plus = f(x) #* line 27

				#* lines 28-30
				if norm(r_plus) < r_norm + (alpha * beta_k * t):
					break
			except BadStep:
				pass
			
			n += 1
			t *= gamma_k
		return x #, t, fx



	def gauss_newton(self, f, F, x0, gamma_k, beta_k):
		r"""A Gauss-Newton solver for unconstrained nonlinear least squares problems.

		Given a vector valued function :math:`\mathbf{f}:\mathbb{R}^m \to \mathbb{R}^M`
		and its Jacobian :math:`\mathbf{F}:\mathbb{R}^m\to \mathbb{R}^{M\times m}`,
		solve the nonlinear least squares problem:

		.. math::

			\min_{\mathbf{x}\in \mathbb{R}^m} \| \mathbf{f}(\mathbf{x})\|_2^2.

		Normal Gauss-Newton computes a search direction :math:`\mathbf{p}\in \mathbb{R}^m`
		at each iterate by solving least squares problem

		.. math::
			
			\mathbf{p}_k \leftarrow \mathbf{F}(\mathbf{x}_k)^+ \mathbf{f}(\mathbf{x}_k)

		and then computes a new step by solving a line search problem for a step length :math:`\alpha`
		satisfying the Armijo conditions:

		.. math::
			
			\mathbf{x}_{k+1} \leftarrow \mathbf{x}_k + \alpha \mathbf{p}_k.
			
		This implementation offers several features that modify this basic outline.

		First, the user can specify a nonlinear *trajectory* along which candidate points 
		can move; i.e.,

		.. math::

			\mathbf{x}_{k+1} \leftarrow T(\mathbf{x}_k, \mathbf{p}_k, \alpha). 
		
		Second, the user can specify a custom solver for computing the search direction :math:`\mathbf{p}_k`.

		Parameters
		----------
		f : callable
			residual, :math:`\mathbf{f}: \mathbb{R}^m \to \mathbb{R}^M`
		F : callable
			Jacobian of residual :math:`\mathbf{f}`; :math:`\mathbf{F}: \mathbb{R}^m \to \mathbb{R}^{M \times m}`
		tol: float [optional] default = 1e-8
			gradient norm stopping criterion
		tol_normdx: float [optional] default = 1e-12
			norm of control update stopping criterion
		maxiter : int [optional] default = 100
			maximum number of iterations of Gauss-Newton
		linesearch: callable, returns new x
			f : callable, residual, f: R^n -> R^m
			g : gradient, R^n
			p : descent direction, R^n
			x0 : current iterate, R^n
		gnsolver: [optional] callable, returns search direction p 
			Parameters: 
				F: current Jacobian
				f: current residual

			Returns:
				p: search step
				s: singular values of Jacobian
		verbose: int [optional] default = 0
			if >= print convergence history diagnostics

		Returns
		-------
		numpy.array((dof,))
			returns x^* (optimizer)
		int
			info = 0: converged with norm of gradient below tol
			info = 1: norm of gradient did not converge, but ||dx|| below tolerance
			info = 2: did not converge, max iterations exceeded
		"""
		#* lines 17-18
		def gn_solver(J_flat, r):
			Y, s, ZT = scipy.linalg.svd(J_flat, full_matrices = False, lapack_driver = 'gesvd')
			# Apply the pseudoinverse
			return -ZT[:-self.d**2,:].T.dot(np.diag(1/s[:-self.d**2]).dot(Y[:,:-self.d**2].T.dot(r)))

		n = len(x0)


		# if trajectory is None:
		# 	trajectory = lambda x0, p, t: x0 + t * p

		linesearch = self.linesearch_armijo
				
		x = np.copy(x0)
		f_eval = f(x) #this
		F_eval = F(x) #this is the vectorised Jacobian
		
		grad = F_eval.T @ f_eval #* Line 16 in Algorithm

		
		# Compute search direction 
		#* Lines 17-18
		dx = gn_solver(F_eval, f_eval)
		
		# Check we got a valid search direction
		if not np.all(np.isfinite(dx)):
			raise RuntimeError("Non-finite search direction returned") 
		
		
		# If Gauss-Newton step is not a descent direction, use -gradient instead
		vec_grad = grad.reshape(grad.shape[0]*grad.shape[1],)
		vec_dx = dx.reshape(dx.shape[0]*dx.shape[1],)
		#* lines 20 to 23 in algorithm
		alpha = np.trace(np.inner(grad, dx))

		if alpha >= 0:
			dx = -grad
			alpha = np.trace(np.inner(grad, dx))


		# Back tracking line search
		#* lines 24-31
		x = linesearch(f, vec_grad, vec_dx, x, alpha, gamma_k, beta_k) 

		return x, gamma_k, beta_k


	"""
		CONSTRUCTION OF INTERPOLATION SETS ALGORITHMS 
	"""
	def generate_set(self, problem, num, s_old, delta_k):
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
		"""
			Sets the max distance for each direction in the current trust region 
		"""
		scale = delta
		for j in range(len(dirn)):
			if dirn[j] < 0.0:
				scale = min(scale, lower[j] / dirn[j])
			elif dirn[j] > 0.0:
				scale = min(scale, upper[j] / dirn[j])
		return scale

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
				scale = self.get_scale(Q[:,i], self.delta_k, lower, upper) 
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
		

	"""
		UPDATING GEOMETRY FROM OMoRF 
	"""

	def improve_interpolation_geometry(self, problem, s_old, f_old, delta_k, rho_k, U, S_red, f_red) : #, unsuccessful_iteration_counter, ratio):		
		dist = max(self.epsilon_1*delta_k, self.epsilon_2*rho_k)

		if max(norm(S_red-s_old, axis=1, ord=np.inf)) > dist:
			S_red, f_red = self.sample_set(problem, 'improve', s_old, delta_k, rho_k, f_old, U, S=S_red, f=f_red)
		
		elif delta_k == rho_k:
			delta_k = self.alpha_2* rho_k

			"""if unsuccessful_iteration_counter >= 3 and ratio < 0:
				# print('UNSUCCESSFUL 3 OR MORE TIMES')
				if rho_k >= 250*self.rho_min:
					rho_k = self.alpha_1*rho_k
				
				elif 16*self.rho_min < rho_k < 250*self.rho_min:
					rho_k = np.sqrt(rho_k*self.rho_min)
				
				else:
					rho_k = self.rho_min"""
		
		return S_red, f_red, delta_k, rho_k, U

	def sample_set(self, problem, method, current_solution, delta_k, rho_k, f_old, U, S=None, f=None, s_new=None, f_new=None):
		q = self.q
		s_old = np.array(current_solution.x)

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
			S, f = self.LU_pivoting(problem, S, f, s_old, delta_k, S_hat, f_hat, U)
		
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
			S, f = self.LU_pivoting(problem, S, f, s_old, delta_k, S_hat, f_hat, U, method='improve')
		
		elif method == 'new':
			S_hat = f_hat = np.array([])
			S = np.zeros((q, self.n))
			f = np.zeros((q, 1))
			S[0, :] = s_old
			f[0, :] = f_old
			S, f = self.LU_pivoting(problem, S, f, s_old, delta_k, S_hat, f_hat, U, method='new')

		return S, f

	def LU_pivoting(self, problem, S, f, s_old, delta_k, S_hat, f_hat, active_subspace, method=None) :
		psi_1 = 1.0e-4
		psi_2 = 0.25

		phi_function, phi_function_deriv = self.get_phi_function_and_derivative(S_hat, s_old, delta_k, active_subspace)
		q = self.q

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
					s = self.find_new_point(problem, v, phi_function, phi_function_deriv)
					if np.unique(np.vstack((S[:k, :], s)), axis=0).shape[0] != k+1:
						s = self.find_new_point_alternative(problem, v, phi_function, S[:k, :], s_old, delta_k)
				except:
					s = self.find_new_point_alternative(problem, v, phi_function, S[:k, :], s_old, delta_k)
				if f_hat.size > 0 and M[index] >= abs(np.dot(v, phi_function(s))):
					s = S_hat[index,:]
					S[k, :] = s
					f[k, :] = f_hat[index]
					S_hat = np.delete(S_hat, index, 0)
					f_hat = np.delete(f_hat, index, 0)
				else:
					S[k, :] = s
					f[k, :] = self.blackbox_evaluation(problem, value=s)[1] 
			
			#Update U factorisation in LU algorithm
			phi = phi_function(s)
			U[k,k] = np.dot(v, phi)
			for i in range(k+1,q):
				U[k,i] += phi[i]
				for j in range(k):
					U[k,i] -= (phi[j]*U[j,i]) / U[j,j]
		return S, f

	def get_phi_function_and_derivative(self, S_hat, s_old, delta_k, active_subspace):
		Del_S = delta_k

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
			return np.dot(active_subspace, phi_deriv)
		
		return phi_function, phi_function_deriv

	def find_new_point(self, problem, v, phi_function, phi_function_deriv, s_old, delta_k):
		#change bounds to be defined using the problem and delta_k
		bounds_l = np.maximum(np.array(problem.lower_bounds).reshape(s_old.shape), s_old-delta_k)
		bounds_u = np.minimum(np.array(problem.upper_bounds).reshape(s_old.shape), s_old+delta_k)

		bounds = []
		for i in range(self.n):
			bounds.append((bounds_l[i], bounds_u[i])) 

		obj1 = lambda s: np.dot(v, phi_function(s))
		jac1 = lambda s: np.dot(phi_function_deriv(s), v)
		obj2 = lambda s: -np.dot(v, phi_function(s))
		jac2 = lambda s: -np.dot(phi_function_deriv(s), v)
		
		res1 = minimize(obj1, s_old, method='TNC', jac=jac1, \
				bounds=bounds, options={'disp': False})
		res2 = minimize(obj2, s_old, method='TNC', jac=jac2, \
				bounds=bounds, options={'disp': False})
		
		s = res1.x if abs(res1.fun) > abs(res2.fun) else res2.x
		return s

	def find_new_point_alternative(self, problem, v, phi_function, S, s_old, delta_k):
		S_tmp = self.generate_set(problem, int(0.5*(self.n+1)*(self.n+2)), s_old, delta_k)
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