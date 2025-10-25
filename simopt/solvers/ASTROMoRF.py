#type: ignore
"""
Summary
-------
The ASTROMoRF (Adaptive Sampling for Trust-Region Optimisation by Moving Ridge Functions) progressively builds local models using
interpolation on a reduced subspace constructed through Active Subspace dimensionality reduction.
The use of Active Subspace reduction allows for a reduced number of interpolation points to be evaluated in the model construction.

"""
#TODO: change subspace dimension to being dynamic
#TODO: 1. WANT TO ENSURE 2d+1 interpolation points every time - problem with basis.DV in grads 
#TODO: 2. HAVE A PROBLEM WITH THE NUMERATOR BEING NEGATIVE AND THE DENOMINATOR BEING POSITIVE - CHECK THE RATIO
	#! THIS IS BECAUSE THE MODEL IS NOT ACCURATE ENOUGH - CHECK THE MODEL EVALUATION
	#! MIGHT NEED TO JUST TAKE POLYRIDGE FUNCTION AND USE THAT 
		#* USING THIS AND SEEMS TO PERFORM A LOT BETTER 
#TODO: 3. NEED TO FIX DV, V, AND DDV FOR RETURNING PROBLEM.DIM SIZE FOR 2d+1 INTERPOLATION POINTS
#?: SEE IF WE CAN REDUCE THE NUMBER OF TIMES WE HAVE TO SAMPLE - may reduce the interpolation set size? 
#?: SEE IF WE CAN REUSE MORE THAN ONE DESIGN POINT - Not possible as then we won't have a span of the region
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

from numpy.linalg import norm, pinv, qr
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, linprog
from scipy import optimize
from scipy.special import factorial
import scipy
import math

import matplotlib.pyplot as plt


# from simopt.linear_algebra_base import finite_difference_gradient


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
from simopt.solvers.active_subspaces.polyridge import PolynomialRidgeApproximation
from simopt.solvers.GeometryImprovement import GeometryImprovement
from simopt.solvers.TrustRegion.Sampling import SamplingRule
from simopt.utils import classproperty, override


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

	@classproperty
	@override 
	def class_name(cls) -> str:
		return "ASTROMoRF"

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
				"default": False
			},
			"original_sampling_rule" : {
				"description": "Flag to enable original sampling rule",
				"datatype": bool, 
				"default": False
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
			"gamma_3": {
				"description": "trust_region radius increase rate after a very successful iteration",
				"datatype": float, 
				"default": 2.5
			},
			"easy_solve": {
				"description": "solve the subproblem approximately with Cauchy point",
				"datatype": bool,
				"default": False
			}, 
			"lambda_min": {
				"description": "minimum sample size",
				"datatype": int,
				"default": 7
			},
			"polynomial basis": {
				"description": "Polynomial basis to use in model construction",
				"datatype": str, 
				"default": "MonomialPolynomialBasis"
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
			'initial subspace dimension': {
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
			'old_implementation' : {
				"description": "Flag to enable old implementation of ASTROMoRF",
				"datatype": bool,
				"default": True
			},
			'performance_benchmarking_flag' : {
				"description": "Flag to enable performance benchmarking",
				"datatype": bool,
				"default": False
			},
			# "sampling rule" : { #! Added in to run sampling experiment
			# 	"description": "An instance of the sampling rule being used",
			# 	"datatype": str,
			# 	"default": 'ASTROMoRFSampling'
			# },  
		}
	
	@property
	def check_factor_list(self) -> dict[str, Callable] : 
		return {
			"crn_across_solns": self.check_crn_across_solns,
			"original_sampling_rule": self.check_sampling_flag,
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
			'initial subspace dimension': self.check_dimension_reduction,
			"random directions": self.check_random_directions,
			"alpha_1": self.check_alpha_1,
			"alpha_2": self.check_alpha_2,
			"rho_min": self.check_rho_min, 
			"max iterations": self.check_iterations,
			"rho_min": self.check_rho_min,
			'old_implementation': self.check_old_implementation,
			'performance_benchmarking_flag': self.check_performance_benchmarking_flag,
			# "sampling rule": self.check_sampling_rule, #! Added in to run sampling experiment
			}
		
	def check_sampling_flag(self) -> bool : 
		return isinstance(self.factors['original_sampling_rule'], bool)
	
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
		return self.factors['initial subspace dimension'] >= 1
	
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

	#! Added in to run sampling experiment
	# def check_sampling_rule(self) -> None:
	# 	if self.factors['sampling rule'] is None : 
	# 		raise ValueError('sampling rule is not implemented')

	def check_performance_benchmarking_flag(self) -> bool:
		if self.factors['performance_benchmarking_flag'] not in [True, False]:
			raise ValueError('performance_benchmarking_flag needs to be True or False')
		return True

	def check_old_implementation(self) -> bool:
		if self.factors['old_implementation'] not in [True, False]:
			raise ValueError('old_implementation needs to be True or False')
		return True

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
		
	def check_rho_min(self) -> bool : 
		return self.factors['rho_min'] > 0
	

	def dynamic_subspace_dimension(self, k: int) -> int : 
		"""
		   Determine the subspace dimension at iteration k. The subspace dimension increases 
		   Logarithmically with the iteration number k.

		Args:
			iteration_no (int): the iteration number k

		Returns:
			int: the subspace dimension at iteration k
		"""
		initial_d = self.factors['initial subspace dimension']
		if k < 2 :
			return initial_d
		else :
			return min(floor(initial_d * 1+log(k,10)), self.n)


	def __init__(self, name="ASTROMoRF", fixed_factors: dict | None = None) -> None:
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
	
	# def sample_instantiation(self) :
	# 	class_name = self.factors['sampling rule'].strip()
	# 	module = importlib.import_module('simopt.solvers.TrustRegion.Sampling')
	# 	sampling_instance = getattr(module, class_name)(self)
	# 	return sampling_instance
	

	def construct_proxy_d(self, problem: Problem) -> int : 
		"""
			Constructs a proxy initial guess of the subspace dimension through the following steps:
				1. Obtain alog(n) random solutions around the problem [x_1,x_2,...,x_{alog(n)}]
				2. Estimate the gradient vectors of the problem at each random solution 
				3. Construct the uncentred covariance matrix of functional derivatives C through a Monte Carlo estimate on the alog(n) gradient vectors
				4. Factorise C through eigendecomposition and sort the eigenpairs into a list
				5. Create a second list of tuples (i,j) where i is the index of the eigenpairs in the list from step 2, and j is the distance between the eigen pairs of 
				|\lambda_{i-1}-\lambda_{i}|. 
				6. pick d to be the i in the list of tuples that contains the smallest j in their tuple out of all j's in the list of tuples

		Args:
			problem (Problem): The simulation model 

		Returns: 
			int: The subspace dimension

		NOTE: 
			This method of finding a parameter for d will require an additional alog(n^{2n}) responses of the simulation model.
		 #! Before implementing this proxy, we need to find a way of finding this covariance matrix without taking an exponential number of responses.
			- Look at some form of sensitivity analysis to find the covariance matrix


		"""
		no_solns = ceil(10 * problem.dim * log(problem.dim)) #! This is the number of random solutions to generate
		# Designate random number generator for random sampling
		find_next_soln_rng = self.rng_list[1]

		# Generate many dummy solutions without replication only to find a reasonable maximum radius
		dummy_solns: list[Solution] = []
		for _ in range(no_solns):
			random_soln_tuple = problem.get_random_solution(find_next_soln_rng)
			random_soln = self.create_new_solution(random_soln_tuple, problem)
			dummy_solns.append(random_soln)

		# Calculate the gradient of the problem at each dummy solution
		gradients = []
		for sol in dummy_solns:
			grad = self.finite_difference(sol, problem) 
			gradients.append(grad)

		C = sum([np.outer(a,a) for a in gradients])/no_solns
		#decompose C into eigenpairs and sort the eigenpairs by descendeing eigenvalue
		# Compute eigenvalues and eigenvectors
		eigenvalues, eigenvectors = np.linalg.eigh(C)
		
		# Create list of (eigenvalue, eigenvector) tuples
		eigenpairs = [(eigenvalues[i], eigenvectors[:, i]) for i in range(len(eigenvalues))]
		
		# Sort by eigenvalue in descending order
		eigenpairs.sort(key=lambda pair: pair[0], reverse=True)

		eigenvalues = [pair[0] for pair in eigenpairs]
		# gaps = [eigenvalues[i - 1] - eigenvalues[i] for i in range(1, len(eigenvalues))]
		# d = gaps.index(min(gaps)) + 1

		

		# find r by finding the smallest d that keeps the explained variance above a certain threshold
		numerator = [(i,sum(eigenvalues[:i])) for i in range(1, len(eigenvalues))]
		explained_variance = [(i+1, numerator[i][1]/sum(eigenvalues)) for i in range(len(numerator))]
		sig_lvl = 0.9
		explained_variance_filter = [i for i in explained_variance[:-1] if i[1] >= sig_lvl] #! Remove the last one as it is always 1.0
		#if no eigenvalues are above the threshold, other than the  then we need to reduce the sig_lvl
		while len(explained_variance) == 0  :
			sig_lvl -= 0.05
			explained_variance_filter = [i for i in explained_variance[:-1] if i[1] >= sig_lvl]
		
		d = explained_variance_filter[0][0] 
		self.U_init = np.column_stack([vec for _, vec in eigenpairs[:d]])

		print(f'the final significance level is {sig_lvl} and the subspace dimension is {d}')

		return d


	def q_setter(self, d: int) -> int :
		"""
			Calculate the number of basis functions in a d-dimensional polynomial of degree p through combinatorial formula

		Args:
			d (int): the dimension of the subspace
			p (int): the degree of the polynomial

		Returns:
			int: the number of basis functions in a d-dimensional polynomial of degree p
		"""
		return int(0.5*(d+1)*(d+2)) # comb(d + self.deg, self.deg) +  self.n * d


	def initialise_factors(self,problem: Problem) -> None :
		"""Initialise all the class factors being applied in the solver

		Args:
			problem (Problem): The simulation model 
		"""
		#initialise factors: 
		self.recommended_solns = []
		self.intermediate_budgets = []

		self.kappa = 1
		self.delta_power = 2 if self.factors['crn_across_solns'] else 4

		self.expended_budget = 0

		self.S = np.array([])
		self.f = np.array([])
		self.g = np.array([])
		self.d = self.factors['initial subspace dimension']

		self.collection_of_subspaces = dict()
		self.collection_of_subspaces[0] = self.d

		self.delta_max = self.calculate_max_radius(problem)

		self.delta_k = 10 ** (ceil(log(self.delta_max * 2, 10) - 1) / problem.dim)
		# self.rho_k = self.delta_k
		self.rhos = []
		self.deltas = []
		self.n = problem.dim

		# if not self.factors['old_implementation'] :
		# 	self.d = self.construct_proxy_d(problem) 
		# else : 
		# 	self.d = self.factors['initial subspace dimension']

		self.deg = self.factors['polynomial degree'] 
		self.q = self.q_setter(self.d) # comb(self.d + self.deg, self.deg) +  self.n * self.d
		self.p = self.n+1
		self.epsilon_1, self.epsilon_2 = self.factors['interpolation update tol'] #epsilon is the tolerance in the interpolation set update 
		self.random_initial = self.factors['random directions']
		self.alpha_1 = self.factors['alpha_1'] #shrink the trust region radius in set improvement 
		self.alpha_2 = self.factors['alpha_2'] #shrink the stepsize reduction  
		self.rho_min = self.factors['rho_min']


		#set up initial Solution
		self.current_solution = self.create_new_solution(problem.factors["initial_solution"], problem)
		self.recommended_solns.append(self.current_solution)
		self.intermediate_budgets.append(self.expended_budget)
		self.visited_points = []

		self.reduced_basis = self.polynomial_basis_instantiation()(self.deg, problem, X=None, dim=self.d)

		prox_sol = self.create_new_solution(problem.factors["initial_solution"], problem)
		problem.simulate(prox_sol)
		self.f_old = -1 * problem.minmax[0] * prox_sol.objectives_mean

		self.geometry_improvement = GeometryImprovement(problem, self)

		self.model = PolynomialRidgeApproximation(self.deg, self.d, problem, self.reduced_basis, self.geometry_improvement)

		if self.factors['performance_benchmarking_flag'] :
			self.performance_benchmarking = True
			self.runtimes = []
			self.iterations = []
			


		# if (self.factors['sampling rule'] != 'Default') :
		# 	self.sampling_instance = self.sample_instantiation()

	@override 
	def solve(self, problem):
		#initialise factors: 
		self.initialise_factors(problem)


		k = 0

		while self.expended_budget < problem.factors['budget'] : 
			#set the new subspace dimension and number of samples 
			self.d = self.dynamic_subspace_dimension(k)
			self.q = self.q_setter(self.d) 
			self.collection_of_subspaces[k] = self.d
			#update dimension in the basis 
			
			
			k += 1 
			print(f'Iteration {k} with budget {self.expended_budget} and subspace dimension {self.d}')
			if k == 1 :
				self.current_solution = self.create_new_solution(self.current_solution.x, problem)
				# current_solution = self.create_new_solution(current_solution.x, problem)
				if len(self.visited_points) == 0:
					self.visited_points.append(self.current_solution)
				
				self.current_solution, self.expended_budget = self.calculate_kappa(k, problem, self.expended_budget, self.current_solution, self.delta_k)
			
				self.recommended_solns.append(self.current_solution)
				self.intermediate_budgets.append(self.expended_budget)
			elif self.factors['crn_across_solns'] and not self.factors['original_sampling_rule'] :
				# since incument was only evaluated with the sample size of previous incumbent, here we compute its adaptive sample size
				# adaptive sampling
				lambda_max = problem.factors['budget'] - self.expended_budget
				sample_size = self.current_solution.n_reps
				while True:
					sig2 = self.current_solution.objectives_var[0]
					stopping = self.get_stopping_time(sig2, self.delta_k, k, problem, self.expended_budget)
					if (sample_size >= min(stopping, lambda_max) or self.expended_budget >= problem.factors['budget']):
						break
					problem.simulate(self.current_solution, 1)
					self.expended_budget += 1
					sample_size += 1
				

			#build random model 
			self.current_solution, self.delta_k, self.fval, self.expended_budget, interpolation_solns, self.U, self.visited_points = self.construct_model(problem, self.current_solution, self.delta_k, k, self.expended_budget, self.visited_points)

			#solve random model 
			candidate_solution, self.delta_k, self.visited_points = self.solve_subproblem(problem, self.current_solution, self.delta_k, self.visited_points, self.U)
			

			#sample candidate solution
			# candidate_solution, fval_tilde = self.blackbox_evaluation(problem, solution=candidate_solution)
			candidate_solution, fval_tilde = self.simulate_candidate_soln(k, problem, candidate_solution, self.current_solution)


			#evaluate model
			self.current_solution, self.delta_k, self.recommended_solns, self.expended_budget, self.intermediate_budgets = self.evaluate_candidate_solution(problem, self.U, self.fval, fval_tilde, self.delta_k, interpolation_solns, self.current_solution, candidate_solution, self.recommended_solns, self.expended_budget, self.intermediate_budgets)	
		
		#record the performance 
		# if self.factors['performance_benchmarking_flag'] :
		# 	self.record_performance_benchmarking(self.runtimes, self.iterations)

		[print(f'At iteration {i}, the subspace dimension was {self.collection_of_subspaces[i]}') for i in self.collection_of_subspaces.keys()]
		return self.recommended_solns, self.intermediate_budgets
	

	#! REFACTOR: Added in optionality for additional sampling methods
	def evaluate_interpolation_points(self, k: int, problem: Problem, visited_index: int, X:np.ndarray, visited_pts: list[Solution]) -> tuple[np.ndarray, list[Solution], list[Solution]]:
		"""
			Run adaptive sampling on the model construction design points to obtain a sample 
			average of their responses.
		Args:
			problem (Problem): The current simulation model 
			X (np.ndarray): A (M,n) numpy matrix of M n-dimensional design points used to construct the 
							interpolation models
			visited_pts (list[float]): A list of the points already simulated by the solver

		Returns:
			tuple[np.ndarray, list[Solution], list[Solution]]: Consists of function values fX of shape (M,), the interpolation solutions and the visited points
		"""	
		# if self.factors['sampling rule'] == 'ASTROMoRFSampling' :
		# 	adaptive_sampling = lambda x : self.adaptive_sampling(*x)
		# else : 
		# 	adaptive_sampling = lambda x : self.sampling_instance.sampling_rule(*x)
		fX = []	 
		interpolation_solutions = []
		for idx,x in enumerate(X) : 
			#for the current solution, we don't need to simulate
			if (idx == 0) and (k==1) :
				fX.append(-1 * problem.minmax[0] * self.current_solution.objectives_mean) 
				interpolation_solutions.append(self.current_solution)
			
			#reuse the replications for x_k
			elif idx == 0: 
				self.current_solution, self.expended_budget = self.adaptive_sampling(problem, k, self.current_solution, self.delta_k, self.expended_budget, False)
				fX.append(-1 * problem.minmax[0] * self.current_solution.objectives_mean)
				interpolation_solutions.append(self.current_solution)

			elif (idx==1) and ((norm(np.array(self.current_solution.x)-np.array(visited_pts[visited_index].x)) != 0) and visited_pts is not None) :
				reuse_solution = visited_pts[visited_index]
				reuse_solution, self.expended_budget = self.adaptive_sampling(problem, k, reuse_solution, self.delta_k, self.expended_budget, False)
				fX.append(-1 * problem.minmax[0] * reuse_solution.objectives_mean)
				interpolation_solutions.append(reuse_solution)
			#For new points, run the simulation with pilot run
			else :
				solution = self.create_new_solution(tuple(x), problem)
				solution, self.expended_budget = self.adaptive_sampling(problem, k, solution, self.delta_k, self.expended_budget)
				fX.append(-1 * problem.minmax[0] * solution.objectives_mean)
				interpolation_solutions.append(solution)

		# print(f'shape of fX is {np.array(fX).shape} and the number of interpolation solutions is {len(interpolation_solutions)}')

		return np.array(fX), interpolation_solutions, visited_pts

	#! REFACTOR: Added in optionality for additional sampling methods
	def simulate_candidate_soln(self, k: int, problem: Problem, candidate_solution: Solution, current_solution: Solution) -> tuple[Solution, float]:
		"""
			Run adaptive sampling on the candidate solution to obtain a sample average of the 
			response to the candidate solution.

		Args:
			problem (Problem): The Simulation Problem being run.
			candidate_solution (Solution): The candidate solution selected by the current iteration
			current_solution (Solution): The incumbent solution of the solver

		Returns:
			tuple[Solution, float]: Consists of the candidate solution and its evaluated solution
		"""
		# if self.factors['sampling rule'] == 'ASTROMoRFSampling' :
		# 	adaptive_sampling = lambda x : self.adaptive_sampling(*x)
		# else : 
		# 	adaptive_sampling = lambda x : self.sampling_instance.sampling_rule(*x)

		if self.factors['crn_across_solns'] and not self.factors['original_sampling_rule']:
			problem.simulate(candidate_solution, current_solution.n_reps) 
			self.expended_budget += current_solution.n_reps 
			fval_tilde = -1 * problem.minmax[0] * candidate_solution.objectives_mean
			return candidate_solution, fval_tilde
			
		else :
			candidate_solution, self.expended_budget = self.adaptive_sampling(problem, k, candidate_solution, self.delta_k, self.expended_budget)
			fval_tilde = -1 * problem.minmax[0] * candidate_solution.objectives_mean
			return candidate_solution, fval_tilde
		 


	def solve_subproblem(self, problem: Problem, current_solution:Solution, delta_k: float, visited_pts_list: list[Solution], U: np.ndarray) -> tuple[Solution, float, list[Solution]] :
		"""Solves the trust-region subproblem.

		Args:
			problem (Problem): The simulation Model being optimised over
			current_solution (Solution): The incumbent solution of the solver
			delta_k (float): The current trust-region radius
			visited_pts_list (list[Solution]): A list of previously simulated solutions by the solver
			U (np.ndarray): The (n,d) active subsapce matrix

		Returns:
			tuple[Solution, float, list[Solution]]: The candidate solution, 
													The trust-region radius, 
													The list of visited solutions.
		"""
		
		cons =  NonlinearConstraint(lambda x : norm(x), 0, self.delta_k)
		
		obj = lambda x: self.model_evaluate(x).item(0)
		stepsize = minimize(obj, np.zeros(self.n), method='trust-constr', constraints=cons, options={'disp': False}).x
		s_new = np.array(current_solution.x) + stepsize


		for i in range(problem.dim):
			if s_new[i] <= problem.lower_bounds[i]:
				s_new[i] = problem.lower_bounds[i] + 0.01
			elif s_new[i] >= problem.upper_bounds[i]:
				s_new[i] = problem.upper_bounds[i] - 0.01


		candidate_solution = self.create_new_solution(tuple(s_new), problem)

		self.visited_points.append(candidate_solution) 


		# Safety step implemented in BOBYQA
		# if norm(stepsize, ord=np.inf) < self.factors['omega_shrinking']*self.rho_k:
		# 	self.delta_k = max(0.5*self.delta_k, self.rho_k)

		return candidate_solution, delta_k, visited_pts_list

	def evaluate_candidate_solution(self, problem: Problem, U: np.ndarray, fval, fval_tilde: float, delta_k: float, interpolation_solns: list[Solution], current_solution: Solution, candidate_solution: Solution, recommended_solns: list[Solution], expended_budget: int, intermediate_budgets: list[int]) :

		gamma_1 = self.factors['gamma_1']
		gamma_2 = self.factors['gamma_2']
		gamma_3 = self.factors['gamma_3']
		#pattern search
		#TODO: fval should be a list of the solution of all the interpolation solutions 
		if ((min(fval) < fval_tilde) and ((fval[0] - min(fval))>= self.factors["ps_sufficient_reduction"] * delta_k**2)) or ((candidate_solution.objectives_var[0]/ (candidate_solution.n_reps * candidate_solution.objectives_mean[0]**2)) > 0.75):
			fval_tilde = min(fval)
			candidate_solution = interpolation_solns[fval.index(min(fval))]  # type: ignore

		stepsize = np.subtract(np.array(candidate_solution.x), np.array(current_solution.x))
		model_eval_old = self.model_evaluate(np.array(current_solution.x)).item()
		model_eval_new = self.model_evaluate(np.array(candidate_solution.x)).item()

		del_f =  self.f_old - fval_tilde #self.f_old - f_new 
		del_m = model_eval_old - model_eval_new

		# rho = del_f/del_m

		# if rho < 0 : 
		# 	if self.rho_k >= 250*self.rho_min:
		# 			self.rho_k = self.alpha_1*self.rho_k
				
		# 	elif 16*self.rho_min < self.rho_k < 250*self.rho_min:
		# 		self.rho_k = np.sqrt(self.rho_k*self.rho_min)
			
		# 	else:
		# 		self.rho_k = self.rho_min
		
		if del_f < 0:
			rho = 0
		else :
			rho = del_f/del_m
		
		step_dist = norm(np.array(candidate_solution.x)-np.array(current_solution.x))
		# successful: accept
		if rho >= self.factors['eta_1']:
			# new_x = candidate_x
			current_solution = candidate_solution
			# final_ob = candidate_solution.objectives_mean
			recommended_solns.append(candidate_solution)
			intermediate_budgets.append(expended_budget)
			# delta_k = min(delta_k, self.delta_max)
			delta_k = min(gamma_1*delta_k, self.delta_max)
			self.f_old = fval_tilde
			
			# very successful: expand and accept
			if rho >= self.factors['eta_2'] :
				delta_k = min(gamma_1 * delta_k, gamma_1 * step_dist, self.delta_max)
			
		# unsuccessful: shrink and reject
		else:
			delta_k = gamma_2 * delta_k
			# delta_k = max(min(gamma_1*self.delta_k, step_dist), self.delta_max)
			# delta_k = max(gamma_2*self.delta_k, step_dist, self.rho_k)
			
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
		pilot_run = ceil(max(lambda_min * log(10 + k, 10) ** 1.1, min(0.5 * problem.dim, lambda_max))-1)
		
		return pilot_run

	def calculate_kappa(self, k, problem, expended_budget, current_solution, delta_k) :

		# if self.factors['original_sampling_rule'] == True :
		# 	print('RUNNING ORIGINAL SAMPLING RULE') 
		# 	lambda_max = problem.factors['budget'] - expended_budget
		# 	problem.simulate(current_solution,1)
		# 	expended_budget += 1
		# 	sample_size = 1
			
		# 	fn = current_solution.objectives_mean 
		# 	sig2 = current_solution.objectives_var[0]
		# 	self.kappa = fn/(delta_k ** 2)
		# 	stopping = self.get_stopping_time(sig2, delta_k, k, problem, expended_budget)
		# 	# print(f'stopping time when calculating kappa is {stopping}')
		# 	if sample_size >= stopping or sample_size >= lambda_max or expended_budget >= problem.factors['budget'] :
		# 		self.kappa = fn / (delta_k**2)
		# 		return current_solution, expended_budget 
		# else :
		lambda_max = problem.factors['budget'] - expended_budget
		pilot_run = self.calculate_pilot_run(k, problem, expended_budget)

		#calculate kappa
		problem.simulate(current_solution, pilot_run)
		expended_budget += pilot_run
		sample_size = pilot_run
		while True:
			rhs_for_kappa = current_solution.objectives_mean
			sig2 = current_solution.objectives_var[0]

			self.kappa = rhs_for_kappa * np.sqrt(pilot_run) / (delta_k ** (self.delta_power / 2))
			stopping = self.get_stopping_time(sig2, delta_k, k, problem, expended_budget)
			# print(f'stopping time when calculating kappa is {stopping}')
			
			if (sample_size >= min(stopping, lambda_max) or expended_budget >= problem.factors['budget']):
				# calculate kappa
				self.kappa = (rhs_for_kappa * np.sqrt(pilot_run)/ (delta_k ** (self.delta_power / 2)))
				return current_solution, expended_budget
			
			problem.simulate(current_solution, 1)
			expended_budget += 1
			sample_size += 1

	
	def get_stopping_time(self, sig2: float, delta: float, k:int, problem: Problem, expended_budget: int) -> int:
		"""
		Compute the sample size based on adaptive sampling stopping rule using the optimality gap
		"""
		pilot_run = self.calculate_pilot_run(k, problem, expended_budget)
		if self.kappa == 0:
			self.kappa = 1

		# compute sample size
		#! This raw sample size is growing exponentially large due to lambda_max
		raw_sample_size = pilot_run * max(1, sig2 / (self.kappa**2 * delta**self.delta_power))
		# Convert out of ndarray if it is
		if isinstance(raw_sample_size, np.ndarray):
			raw_sample_size = raw_sample_size[0]
		# round up to the nearest integer
		sample_size: int = ceil(raw_sample_size)
		return sample_size
	
	def samplesize(self, k,sig,delta) -> int:
		alpha_k = 1
		lambda_k: int = 1 if k==1 else floor(self.factors['lambda_min']*math.log(k)**1.5)
		# S_k = math.floor(max(5,lambda_k,(lambda_k*sig)/((self.kappa^2)*delta**(2*(1+1/alpha_k)))))
		S_k = ceil((lambda_k*sig**2)/(self.kappa**2*delta**4))
		return S_k
	
	def original_sampling_rule(self, problem: Problem, k: int, new_solution: Solution, delta_k: float, used_budget: int) -> tuple[Solution, int]:
		lambda_max = problem.factors['budget'] - used_budget
		# print(f'RUNNING ORIGINAL SAMPLING RULE on {new_solution.x}')
		lambda_k: int = 1 if k==1 else floor(self.factors['lambda_min']*math.log(k)**1.5)
		sample_size = new_solution.n_reps
		if new_solution.n_reps < 2 :
				sim_no = 2-new_solution.n_reps
				problem.simulate(new_solution,sim_no)
				# used_budget += sim_no
				sample_size += sim_no
		
		#Calculate smallest n in adaptive rule 
		while True : 
			problem.simulate(new_solution,1)
			# used_budget += 1
			sample_size += 1

			sig2 = new_solution.objectives_var[0]
			stopping = self.samplesize(k, sig2, delta_k) 
			# print(f'stopping time during adaptive_sampling_1 is {stopping}')
			if (sample_size >= stopping) : 
				# print(f'sample_size is {sample_size}')
				N_k: int = max(min(sample_size, lambda_max), lambda_k)
				break 
			
		adapted_solution = self.create_new_solution(new_solution.x, problem)
		problem.simulate(adapted_solution, N_k)
		used_budget += N_k
		return adapted_solution, used_budget
	
	def two_stage_sampling(self, problem: Problem, k: int, new_solution: Solution, delta_k: float, used_budget: int) -> tuple[Solution, int]:
		pass 
	
	def adaptive_sampling_1(self, problem: Problem, k: int, new_solution: Solution, delta_k: float, used_budget: int) :
		# adaptive sampling
		lambda_max = problem.factors['budget'] - used_budget
		pilot_run = self.calculate_pilot_run(k, problem, used_budget)

		problem.simulate(new_solution, pilot_run)
		used_budget += pilot_run
		sample_size = pilot_run
		sig_init = new_solution.objectives_var[0]

		#if the variance is too large, we switch to the original sampling rule as the two-stage will produce exponentially large sample sizes
		# if sig_init > 1.0 : 
		# 	new_solution, used_budget = self.original_sampling_rule(problem, k, new_solution, delta_k, used_budget)
		# 	return new_solution, used_budget
		# else :

		while True:
			sig2 = new_solution.objectives_var[0]
			stopping = self.get_stopping_time(sig2, delta_k, k, problem, used_budget)
			if ((sample_size >= min(stopping, lambda_max)) or used_budget >= problem.factors['budget']):
				return new_solution, used_budget
			problem.simulate(new_solution, 1)
			
			used_budget += 1
			sample_size += 1

		# return new_solution, used_budget
	
	#this is sample_type = 'conditional before'	
	def adaptive_sampling_2(self, problem, k, new_solution, delta_k, used_budget) : 

		lambda_max = problem.factors['budget'] - used_budget
		if new_solution.n_reps < 2 :
			sim_no = 2-new_solution.n_reps
			problem.simulate(new_solution,sim_no)
			used_budget += sim_no
			sample_size = sim_no

		sample_size = new_solution.n_reps 
		sig2 = new_solution.objectives_var[0]

		# #if the variance is too large, we switch to the original sampling rule as the two-stage will produce exponentially large sample sizes
		# if sig2 > 1.0 : 
		# 	new_solution, used_budget = self.original_sampling_rule(problem, k, new_solution, delta_k, used_budget)
		# 	return new_solution, used_budget
		# else :
		while True:
			stopping = self.get_stopping_time(sig2, delta_k, k, problem, used_budget)
			if (sample_size >= min(stopping, lambda_max) or used_budget >= problem.factors['budget']):
				return new_solution, used_budget
			problem.simulate(new_solution, 1)
			
			used_budget += 1
			sample_size += 1
			sig2 = new_solution.objectives_var[0]



	def adaptive_sampling(self, problem, k, new_solution, delta_k, used_budget, sample_after=True) :
		if sample_after :
			return self.adaptive_sampling_1(problem, k, new_solution, delta_k, used_budget) #solutions not previously evaluated
		
		return self.adaptive_sampling_2(problem, k, new_solution, delta_k, used_budget) #solutions already evaluated
	

	def construct_model(self, problem: Problem, current_solution: Solution, delta_k: float, k: int, expended_budget: int, visited_points_list: list[Solution]) : 
		#construct initial active subspace 
		# U0 = self.initialise_subspace_rand(current_solution, delta_k)
		if self.factors['performance_benchmarking_flag'] :
			start_time = time.time()

		if self.factors['old_implementation'] ==  True :
			init_S_full = self.geometry_improvement.generate_set(self.d, np.array(current_solution.x), delta_k) #(d, n)
			U, _ = np.linalg.qr(init_S_full.T)
		else :
			if k == 1:
				init_S_full = self.geometry_improvement.generate_set(self.d, np.array(current_solution.x), delta_k) #(d, n)
				U, _ = np.linalg.qr(init_S_full.T)
			#* Using the last iteration's active subspace matrix for the initial estimate of U in the next iteration should provide a better guess. 
			else : 
				init_S_full = self.geometry_improvement.generate_set(self.d, np.array(current_solution.x), delta_k) #(d, n)
				U = self.initialise_subspace_covar(init_S_full, problem)
			# U = self.U_init

		X, f_index = self.construct_interpolation_set(current_solution, problem, U, delta_k, k, visited_points_list)

		
		fX, interpolation_solutions, visited_points_list = self.evaluate_interpolation_points(k,problem, f_index, X, visited_points_list)
		self.f_old = fX[0,0]

		#get the function value of the current solution - this is the first value in the array of X values 
		fval = fX.flatten().tolist()


		self.model.fit(X, fX, np.array(current_solution.x), self.f_old, delta_k, interpolation_solutions, self.d, visited_points_list, U)

		#set delta_k and rho_k after model fitting 
		self.delta_k = self.model.delta_k 
		interpolation_solutions = self.model.interpolation_sols
		self.visited_points = visited_points_list


		# self.coefficients = coefficients
		self.coefficients = self.model.coef

		if self.factors['performance_benchmarking_flag'] :
			end_time = time.time()
			runtime = end_time - start_time
			print(f'Model construction took {runtime} seconds on iteration {k} with {self.model.iterations} iterations of the Gauss-Newton algorithm')
			self.runtimes.append((k,runtime))
			self.iterations.append((k, self.model.iterations))

		return current_solution, delta_k, fval, expended_budget, interpolation_solutions, self.model._U, self.visited_points

	def model_evaluate(self, X) : 
		if self.factors['old_implementation'] == False :
			N = ceil(0.2/self.delta_k**2)
		else : 
			N =1 
		if len(X.shape) == 1 :
			X = X.reshape(-1,1)
		val = 0
		for _ in range(N) : 
			val += self.model.eval(X, self.d)

		val = val/N	
		return val
		

	def initialise_subspace_rand(self, current_solution: Solution, delta_k: float) -> np.ndarray : 
		"""build the initial subspace by sampling points within the trust region

		Args:
			current_solution (Solution): The current Solution 
			delta_k (float): _description_

		Returns:
			np.ndarray: _description_
		"""
		#TODO: Change from normal distribution to using generating set 
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
		#! Z = self.generate_set(problem, self.d, np.array(self.current_solution.x), self.delta_k).T 
		#undergo QR decomposition to get initial U 
		U, _ =  np.linalg.qr(Z)

		return U 
	
	def model_finite_diff(self, x: np.ndarray, problem: Problem) -> np.ndarray :
		lower_bound = problem.lower_bounds
		upper_bound = problem.upper_bounds
		# grads = np.zeros((problem.dim,r)) #Take r gradient approximations
		# problem.simulate(new_solution,1)
		# fn = -1 * problem.minmax[0] * new_solution.objectives_mean
		fn = self.model_evaluate(x).item()
		#if BdsCheck isn't implemented then we assume central finite differencing 
		alpha = 1e-8
		
		FnPlusMinus = np.zeros((problem.dim, 3))
		grad = np.zeros(problem.dim)
		for i in range(problem.dim):
			# Initialization.
			x1 = x.tolist()
			x2 = x.tolist()
			# Forward stepsize.
			steph1 = alpha
			# Backward stepsize.
			steph2 = alpha

			# Check variable bounds.
			if x1[i] + steph1 > upper_bound[i]:
				steph1 = np.abs(upper_bound[i] - x1[i])
			if x2[i] - steph2 < lower_bound[i]:
				steph2 = np.abs(x2[i] - lower_bound[i])

			# Decide stepsize.
			# Central diff.
			FnPlusMinus[i, 2] = min(steph1, steph2)
			x1[i] = x1[i] + FnPlusMinus[i, 2]
			x2[i] = x2[i] - FnPlusMinus[i, 2]

			fn1, fn2 = 0,0 
			fn1  = self.model_evaluate(np.array(x1)).item()
			# First column is f(x+h,y).
			FnPlusMinus[i, 0] = fn1
			# problem.simulate(x2_solution)
			# fn2 = -1 * problem.minmax[0] * x2_solution.objectives_mean
			fn2  = self.model_evaluate(np.array(x2)).item()
			# Second column is f(x-h,y).
			FnPlusMinus[i, 1] = fn2

			# Calculate gradient.
			grad[i] = (fn1 - fn2) / (2 * FnPlusMinus[i, 2])

		return grad 

	
	#* This is a more accurate subspace to start with 
	def initialise_subspace_covar(self, X: np.ndarray, problem: Problem) -> np.ndarray :
		#get gradient approximations of each point in X 
		grads = []
		for x in X : 
			# x = np.array(x)
			# x = x.reshape(-1,1)
			grad = self.model_finite_diff(x, problem)
			# print(f'gradient of {x} is {grad}')
			grads.append(grad)

		covar = sum([np.outer(a,a) for a in grads])/len(X)
		#perform eigenvalue decomposition   
		eigvals, eigvecs = np.linalg.eigh(covar) 
		#sort the eigenvalues in descending order
		sorted_indices = np.argsort(eigvals)[::-1]

		#return the eigenvectors as the new active subspace
		return eigvecs[:, sorted_indices[:self.d]]  # (n, d)


	def finite_difference(self, new_solution: Solution, problem: Problem, alpha: float = 1.0e-8, BdsCheck: None | np.ndarray = None) -> np.ndarray :
		"""Calculates a gradient approximation of the solution on the problem

		:math:
			`\nabla F(x_n) = \frac{F(x_n+\alpha)-F(x_n-\alpha)}{2\alpha}
			\nabla F(x_n) = \frac{F(x_n+\alpha)-F(x_n)}{2\alpha}
			\nabla F(x_n) = \frac{F(x_n)-F(x_n-\alpha)}{2\alpha}`

		where :math:`x_n` is new_solution and :math:`F` is the objective function of the problem 

		Args:
			new_solution (Solution): The value for where the gradient is 
			problem (Problem): The simulation-optimisation problem being solved
			alpha (float): The perturbation size. Defaults to 1.0e-8
			BdsCheck (None | np.ndarray, optional): _description_. Defaults to None.

		Returns:
			np.ndarray: The output of the problem's gradient evaluated at new_solution
		"""
		lower_bound = problem.lower_bounds
		upper_bound = problem.upper_bounds
		# grads = np.zeros((problem.dim,r)) #Take r gradient approximations
		problem.simulate(new_solution,1)
		fn = -1 * problem.minmax[0] * new_solution.objectives_mean

		#if BdsCheck isn't implemented then we assume central finite differencing 
		if BdsCheck is None : 
			BdsCheck =  np.zeros(problem.dim)
		
		new_x = new_solution.x
		FnPlusMinus = np.zeros((problem.dim, 3))
		grad = np.zeros(problem.dim)
		for i in range(problem.dim):
			# Initialization.
			x1 = list(new_x)
			x2 = list(new_x)
			# Forward stepsize.
			steph1 = alpha
			# Backward stepsize.
			steph2 = alpha

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
			x1_solution = self.create_new_solution(tuple(x1), problem)
			if BdsCheck[i] != -1:
				problem.simulate(x1_solution)
				fn1 = -1 * problem.minmax[0] * x1_solution.objectives_mean
				# First column is f(x+h,y).
				FnPlusMinus[i, 0] = fn1
			x2_solution = self.create_new_solution(tuple(x2), problem)
			if BdsCheck[i] != 1:
				problem.simulate(x2_solution)
				fn2 = -1 * problem.minmax[0] * x2_solution.objectives_mean
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

	""" 
		CONSTRUCTION OF INTERPOLATION SETS ALGORITHMS 
	"""

	def standard_basis(self, problem: Problem, index: int, dim: int) -> list[float]:
			"""
			Creates a standard basis vector e_i in the space of dimension equal to the problem dimension. Where i is at the index of the index parameter
			Args:
				index (int): the location of the value 1 in the standard basis vector
			
			Returns:
				np.array: a standard basis vector of the form (0,0,...,0,1,0,...,0), where the 1 is in the location of index
			"""
			arr = np.zeros(dim)
			arr[index] = 1.0
			return arr

	def interpolation_points(self, problem: Problem, x_k: np.ndarray, delta: float, U: np.ndarray) -> list[np.ndarray]:
		"""
		Constructs an interpolation set without reusing points
		
		Args:
			delta (TYPE): Description
		
		Returns:
			[np.array]: Description
		"""

		Y = [x_k]
		epsilon = 0.01
		#build the basis that spans the trust region in the projected space 
		for i in range(0, self.d):
			plus = Y[0] + delta * self.standard_basis(problem, i, self.n)
			minus = Y[0] - delta * self.standard_basis(problem, i, self.n)

			if sum(x_k) != 0: #check if x_k is not the origin
				# block constraints
				if minus[i] <= problem.lower_bounds[i]:
					minus[i] = problem.lower_bounds[i] + epsilon
				if plus[i] >= problem.upper_bounds[i]:
					plus[i] = problem.upper_bounds[i] - epsilon

			Y.append(plus)
			Y.append(minus)

		#fill the remaining points with vectors in the span of current Y
		# if len(Y) < problem.dim : 
		# 	remaining_pts = problem.dim - (2*self.d + 1) 
		# 	for idx in range(remaining_pts) : 
		# 		Y.append(Y[idx] + Y[-idx])	
		return Y #!should contain problem.dim interp olation points 


	# generate the basis (rotated coordinate) (the first vector comes from the visited design points (origin basis)
	def get_rotated_basis(self, first_basis: np.ndarray, rotate_index: np.ndarray, U: np.ndarray) -> np.ndarray:
		rotate_matrix = np.array(first_basis)
		rotation = np.matrix([[0, -1], [1, 0]])

		#! We have 7 points in the visited points list, leading to a rotate index of length 7, we want a rotate 

		# rotate the coordinate basis based on the first basis vector (first_basis)
		# choose two dimensions which we use for the rotation (0,i)
		for i in range(1,len(rotate_index)):
			v1 = np.array([[first_basis[rotate_index[0]]],  [first_basis[rotate_index[i]]]])
			v2 = np.dot(rotation, v1)
			rotated_basis = np.copy(first_basis)
			rotated_basis[rotate_index[0]] = v2[0][0]
			rotated_basis[rotate_index[i]] = v2[1][0]
			# stack the rotated vector
			rotate_matrix = np.vstack((rotate_matrix,rotated_basis))
		return rotate_matrix

	# compute the interpolation points (2d+1) using the rotated coordinate basis (reuse one design point)
	def get_rotated_basis_interpolation_points(self, problem: Problem, x_k: np.ndarray, delta: float, rotate_matrix: np.ndarray, reused_x: np.ndarray, U: np.ndarray) -> list[np.ndarray]:
		Y = [x_k]
		epsilon = 0.01
		#! We have changed the range to be the rotation matrix 
		for i in range(len(rotate_matrix)):
			if i == 0:
				plus = np.array(reused_x)
			else:
				plus = Y[0] + (U @ np.array(delta * rotate_matrix[i]).reshape(-1,1)).reshape(-1,).tolist()
			minus = Y[0] - (U @ np.array(delta * rotate_matrix[i]).reshape(-1,1)).reshape(-1,).tolist()

			if sum(x_k) != 0:
				# block constraints
				for j in range(problem.dim):
					if minus[j] <= problem.lower_bounds[j]:
						minus[j] = problem.lower_bounds[j] + epsilon
					elif minus[j] >= problem.upper_bounds[j]:
						minus[j] = problem.upper_bounds[j] - epsilon
					if plus[j] <= problem.lower_bounds[j]:
						plus[j] = problem.lower_bounds[j] + epsilon
					elif plus[j] >= problem.upper_bounds[j]:
						plus[j] = problem.upper_bounds[j] - epsilon

			Y.append(plus)
			Y.append(minus)
		# if len(Y) < problem.dim : 
		# 	remaining_pts = problem.dim % (2*self.d + 1)
		# 	#? I need to make sure that 
		# 	for idx in range(remaining_pts) : 
		# 		new_pt = Y[idx] + Y[-idx]

		# 		#check constraints 
		# 		for j in range(problem.dim):
		# 			if new_pt[j] <= problem.lower_bounds[j]:
		# 				minus[j] = problem.lower_bounds[j] + epsilon
		# 			elif new_pt[j] >= problem.upper_bounds[j]:
		# 				new_pt[j] = problem.upper_bounds[j] - epsilon
					

		return Y #!should contain problem.dim interpolation points 


	#! This is the only sample set construction method that gets called 
	def construct_interpolation_set(self, current_solution: Solution, problem: Problem, U: np.ndarray, delta_k: float, k: int, visited_pts_list: list[Solution]) -> tuple[list[np.ndarray], int] : 
		x_k = np.array(current_solution.x)
		Dist = []
		for i in range(len(visited_pts_list)):
			Dist.append(norm(np.array(visited_pts_list[i].x) - x_k)-delta_k)
			# If the design point is outside the trust region, we will not reuse it (distance = -big M)
			if Dist[i] > 0:
				Dist[i] = -delta_k*10000

		# Find the index of visited design points list for reusing points
		# The reused point will be the farthest point from the center point among the design points within the trust region
		f_index = Dist.index(max(Dist))

		# If it is the first iteration or there is no design point we can reuse within the trust region, use the coordinate basis
		if (k == 1) or (norm(x_k - np.array(visited_pts_list[f_index].x))==0) :
			# Construct the interpolation set without reuse 
			Y = self.interpolation_points(problem, x_k, delta_k, U)

		# Else if we will reuse one design point
		elif k > 1 :
			visited_pts_array = np.array(visited_pts_list[f_index].x)
			diff_array = U.T @ (visited_pts_array - x_k)
			first_basis = (diff_array) / norm(diff_array)

			# if first_basis has some non-zero components, use rotated basis for those dimensions
			rotate_list = np.nonzero(first_basis)[0]
			rotate_matrix = self.get_rotated_basis(first_basis, rotate_list, U)

			# if first_basis has some zero components, use coordinate basis for those dimensions
			for i in range(self.d):
				if first_basis[i] == 0 :
					rotate_matrix = np.vstack((rotate_matrix, self.standard_basis(problem, i, self.d)))

			# construct the interpolation set
			Y = self.get_rotated_basis_interpolation_points(problem, x_k, delta_k, rotate_matrix, visited_pts_list[f_index].x, U)
		return np.vstack(Y), f_index
	

	def record_performance_benchmarking(self, runtimes: list[tuple[int,float]], iterations: list[tuple[int, int]]) -> None:
		"""Record the performance benchmarking data to a CSV file.

		Args:
			runtimes (List[tuple[int, float]]): A list of tuples containing the iteration number and runtime.
			iterations (List[tuple[int, int]]): A list of tuples containing the iteration number and number of iterations.
		"""
		runtimes_list = [i[1] for i in runtimes]
		gn_iterations = [i[1] for i in iterations]
		iterations = [i[0] for i in iterations]


		rand_num = str(random.randint(0, 1000000))
		directory = os.path.join(os.getcwd(), "experiments", "Progress_review_exp_2_data")
		filename = 'performance_benchmarking_' + rand_num + '-' + self.name + '.csv'
		path = os.path.join(directory, filename)

		os.makedirs(os.path.dirname(path), exist_ok=True)

		with open(path, 'w', newline='') as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow(['Iteration', 'Runtime (s)', 'Gauss-Newton Iterations'])
			for i in range(len(iterations)):
				writer.writerow([iterations[i], runtimes_list[i], gn_iterations[i]])

		print(f'saved performance benchmarking data to {filename}')