#type: ignore 
"""mini-batch SGD 

TODO: add bounds in

"""
from __future__ import annotations
from typing import Callable

import warnings

#import ridge function approximation from PSDR-Master
import sys

sys.path.append('~/Desktop/Simopt')

warnings.filterwarnings("ignore")

from simopt.base import (
	ConstraintType,
	ObjectiveType,
	Solver,
	VariableType,
)
from simopt.solvers.active_subspaces.polyridge import *
import numpy as np 
from numpy.linalg import norm
from scipy.optimize import minimize, NonlinearConstraint

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
			"eta_1" : {
				"description": "",
				"datatype": float, 
				"default": 0.1
			}, 
			"eta_2": {
				"description": "",
				"datatype": float, 
				"default": 0.7

			},
			"initial radius": {
				"description": "",
				"datatype": float, 
				"default": 0.0

			}, 
			"delta": {
				"description": "size of the trust-region radius",
				"datatype": float,
				"default": 5.0
			}, 
			"delta_max": {
				"description": "",
				"datatype": float, 
				"default": 0.0

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
			"gamma_3": {
				"description": "",
				"datatype": float, 
				"default": 2.5
			},
			"gamma_shrinking": {
				"description": "",
				"datatype": float, 
				"default": 0.5
			}, 
			"omega_shrinking": {
				"description": "",
				"datatype": float,
				"default": 0.5
			}, 
			"dimension reduction": {
				"description": "dimension size of the active subspace",
				"datatype": int, 
				"default": 2
			}, 
			"adaptive dimension check": {
				"description": "flag for if AIC will be applied",
				"datatype": bool, 
				"default": False
			}, 
			"adaptive interpolation construction": {
				"description": "flag for if sampling for the interpolation sets is adaptive",
				"datatype": bool, 
				"default": False
			}
		}
	
	@property
	def check_factor_list(self) -> dict[str, Callable] : 
		return {
			"crn_across_solns": self.check_crn_across_solns,
		}
	
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
		"""if fixed_factors is None:
			fixed_factors = {}
		self.name = name
		self.objective_type = "single"
		self.constraint_type = "box"
		self.variable_type = "continuous"
		self.gradient_needed = False
		self.specifications = {
			"crn_across_solns": {
				"description": "use CRN across solutions?",
				"datatype": bool,
				"default": True
			},
			"eta_1" : {
				"description": "",
				"datatype": float, 
				"default": 0.1
			}, 
			"eta_2": {
				"description": "",
				"datatype": float, 
				"default": 0.7

			},
			"initial radius": {
				"description": "",
				"datatype": float, 
				"default": 0.0

			}, 
			"delta": {
				"description": "size of the trust-region radius",
				"datatype": float,
				"default": 5.0
			}, 
			"delta_max": {
				"description": "",
				"datatype": float, 
				"default": 0.0

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
			"gamma_3": {
				"description": "",
				"datatype": float, 
				"default": 2.5
			},
			"gamma_shrinking": {
				"description": "",
				"datatype": float, 
				"default": 0.5
			}, 
			"omega_shrinking": {
				"description": "",
				"datatype": float,
				"default": 0.5
			}, 
			"dimension reduction": {
				"description": "dimension size of the active subspace",
				"datatype": int, 
				"default": 2
			}, 
			"adaptive dimension check": {
				"description": "flag for if AIC will be applied",
				"datatype": bool, 
				"default": False
			}, 
			"adaptive interpolation construction": {
				"description": "flag for if sampling for the interpolation sets is adaptive",
				"datatype": bool, 
				"default": False
			}

		}
		self.check_factor_list = {
			"crn_across_solns": self.check_crn_across_solns,
		}"""
		super().__init__(name, fixed_factors)


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
		q, grad, Hessian = model.coefficients
		new_x = solution.x
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
			# Search engine - solve subproblem
			def subproblem(s) : 
				# return fval[0] + np.dot(curr_point, grad) + np.dot(np.multiply(curr_point, Hessian), s)
				return model.local_model_evaluate(s)
			
			con_f = lambda s: norm(s-np.array(new_x))
			nlc = NonlinearConstraint(con_f, 0, delta)

			solve_subproblem = minimize(subproblem, np.array(new_x), method='trust-constr', constraints=nlc)
			candidate_x =  solve_subproblem.x


		# handle the box constraints
		for i in range(problem.dim):
			if candidate_x[i] <= problem.lower_bounds[i]:
				candidate_x[i] = problem.lower_bounds[i] + 0.01
			elif candidate_x[i] >= problem.upper_bounds[i]:
				candidate_x[i] = problem.upper_bounds[i] - 0.01
		
		candidate_solution = self.create_new_solution(tuple(candidate_x), problem)
		if self.factors['random_model type'] == 'random_model_reuse' :
			#we only append to the visited points list if we care about reusing points
			visited_pts_list.append(candidate_solution) 

		return candidate_solution, visited_pts_list
	
	def evaluate_candidate_solution(self, model, problem, fval_tilde, delta_k, rho_k, current_solution, candidate_solution, recommended_solns, polynomial_basis, interpolation_set, active_subspace_set, subspace_matrix) :
		"""
		Evaluate the candidate solution, by looking at the ratio comparison 
		
		Args:
			model (random_model): the local model
			delta_k (float): the current trust-region radius size
			candidate_solution (base.Solution): the current iterations candidate solution
			recommended_solns ([]): Description
		"""
		fval = model.fval

		step_size = candidate_solution.x - current_solution.x

		if norm(step_size) <= self.factors['gamma_shrinking'] * rho_k : 
			delta_k = max(self.factors['omega_shrinking']*delta_k, rho_k) 
			interpolation_set, active_subspace_set, subspace_matrix, rho_k, delta_k = self.interpolation_update(current_solution, problem, polynomial_basis, interpolation_set, active_subspace_set, subspace_matrix, delta_k, rho_k, True)
			return current_solution, delta_k, recommended_solns, interpolation_set, active_subspace_set, subspace_matrix, rho_k


		if (model.local_model_evaluate(np.zeros(problem.dim)) - model.local_model_evaluate(np.array(candidate_solution.x) - np.array(current_solution.x))) <= 0:
			rho = 0
		else:
			difference = np.subtract(candidate_solution.x, current_solution.x)
			rho = (fval[0] - fval_tilde) / (model.local_model_evaluate(np.zeros(problem.dim)) - model.local_model_evaluate(difference))
		
		# Successful iteration
		if rho >= self.factors['eta_1']:
			current_solution = candidate_solution
			recommended_solns.append(candidate_solution)
		else : 
			current_solution = current_solution
		
		#Trust-Region Radius
		if rho >= self.factors['eta_2']:
			delta_k = max(self.factors['gamma_2']*delta_k, self.factors['gamma_3']*norm(step_size))
		
		elif rho < self.factors['eta_2'] and rho >= self.factors['eta_1'] : 
			delta_k = max(self.factors['gamma_1']*delta_k,norm(step_size),rho_k)

		else:
			delta_k = max(min(self.factors['gamma_1'] * delta_k, norm(step_size)), rho_k)

		#Update active subspace matrix and rho_k
		#TODO: remove a point from interpolation sets and set active subspace and rho_{k+1} to old values
		if rho >= self.factors['eta_1'] :
			 #remove a point from interpolation and active subspace sets 
			interpolation_set = self.geometry_improvement(interpolation_set, problem, polynomial_basis, delta_k, False)
			active_subspace_set =  self.geometry_improvement(interpolation_set, problem, polynomial_basis, delta_k, False)
		else :
			interpolation_set, active_subspace_set, subspace_matrix, rho_k, delta_k = self.interpolation_update(current_solution, problem, polynomial_basis, interpolation_set, active_subspace_set, subspace_matrix, delta_k, rho_k, True)

		return current_solution, delta_k, recommended_solns, interpolation_set, active_subspace_set, subspace_matrix, rho_k

	def get_fvals(self, samples, problem, budget) : 
		fvals = []
		for s in samples :
			s = [a for a in s.flatten()]
			# print('s: ', tuple(s))
			new_solution = self.create_new_solution(tuple(s), problem)
			problem.simulate(new_solution,1)
			budget += 1
			fvals.append(-1 * problem.minmax[0] * new_solution.objectives_mean)
			# print('current fvals: ',fvals, '\n')

		return np.array(fvals).reshape((len(fvals),1)), budget


	def generate_samples(self, lower_bound, upper_bound, new_x, problem) : 
		samples = [] 
		while len(samples) < problem.dim + 1 :
			sample = np.random.normal(size=(len(new_x),1)) 
			#check bounds if inside then append 
			if np.all(sample >= lower_bound) and np.all(sample <= upper_bound) : 
				samples.append(sample)

		return samples


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
		visited_pts_list = []

		new_x = problem.factors["initial_solution"]
		new_solution = self.create_new_solution(new_x, problem)
		recommended_solns.append(new_solution)
		intermediate_budgets.append(expended_budget)
		#Set algorithmic parameters 
		model_construction_parameters = {
		'subspace_dim': 2, 
		'polynomial_degree': 1,
		'step_length_reduction': 0.5,
		'beta': 10**(-6)
		}

		lower_bound = problem.lower_bounds 
		upper_bound = problem.upper_bounds

		#Build an initial set for the active subspace 
		#sample problem.dim + 1 points from normal distr. 
		samples = self.generate_samples(lower_bound, upper_bound, new_x, problem)
		
		#Construct the basis matrix for the active subspace 
		# model = data_driven_ridge(problem)
		model = PolynomialRidgeApproximation(degree=7, subspace_dimension=2, beta=1e-4, disp=True, maxiter=500, ftol=-1, gtol=-1)
		# expended_budget = subspace_construction_budget
		fvals, expended_budget = self.get_fvals(samples, problem,expended_budget)
		# subspace_matrix, model_coeff, num_poly_basis = model.solve_variable_projection_ridge_approx(samples, fvals, **model_construction_parameters)
		model.fit(samples,fvals)
		#Build an initial space for model construction with 0.5*(problem.dim + 1)*(problem.dim + 2)
		interpolation_construction = trust_region_interpolation_points(problem, new_solution)
		interpolation_set = interpolation_construction.interpolation_points(delta_k)

		k=0

		while expended_budget < problem.factors["budget"]:
			k += 1 

			#build random model 
			int_fvals, expended_budget = self.get_fvals(interpolation_set, expended_budget)
			current_solution, delta_k, construction_budget, interpolation_solns, visited_pts_list, sample_size = model.predict(interpolation_set, )
			expended_budget = construction_budget # the additions to the expended budget is done in model.construct_model

			#solve random model
			candidate_solution, visited_pts_list = self.solve_subproblem(delta_k, model, problem, current_solution, visited_pts_list)

			candidate_solution, sampling_budget = self.subspace_sampling(problem, candidate_solution, k, delta_k, expended_budget, sample_size)
			fval_tilde = -1 * problem.minmax[0] * candidate_solution.objectives_mean
			expended_budget = sampling_budget

			#evaluate model
			model, problem, fval_tilde, delta_k, interpolation_solns, candidate_solution, recommended_solns
			current_solution, delta_k, recommended_solns, interpolation_set, active_subspace_set, subspace_matrix, rho_k = self.evaluate_candidate_solution(model, problem, fval_tilde, delta_k, rho_k, current_solution,\
																				candidate_solution, recommended_solns, model.polynomial_basis,\
																					 interpolation_set, active_subspace_set, subspace_matrix)	

			intermediate_budgets.append(expended_budget)

		return recommended_solns, intermediate_budgets

	#This is the update for the interpolation sets as defined in the paper
	def interpolation_update(self, current_solution, problem, polynomial_basis, interpolation_set, active_subspace_set, subspace_matrix, delta_k, rho_k, geometry_improving_flag) : 
		tol = max(2*delta_k, 10*rho_k)
		for elem_int,elem_as in zip(interpolation_set, active_subspace_set)  : 
			if max(norm(current_solution.x - elem_int)) > tol : 
				#geometry inmproving algorithm on interpolation_set
				interpolation_set = self.geometry_improvement(interpolation_set, problem, polynomial_basis, delta_k, geometry_improving_flag)
				#set new subspace matrix, rho_k+1, delta_k+1
				return interpolation_set, active_subspace_set, subspace_matrix, rho_k, delta_k 
			elif max(norm(current_solution.x - elem_as)) > tol : 
				#geometry improving algorithm on active_subspace_set
				active_subspace_set = self.geometry_improvement(active_subspace_set, problem, polynomial_basis, delta_k, geometry_improving_flag)
				#reconstruct basis matrix of active subspace 
				subspace_matrix = self.construct_active_subspace(active_subspace_set, problem)
				#set rho_k+1 and delta_k+1
				return interpolation_set, active_subspace_set, subspace_matrix, rho_k, delta_k
			
		if delta_k==rho_k : 
			#shrink rho_k+1 and delta_k+1
			rho_k = 0.1 * rho_k 
			delta_k = 0.5 * delta_k

			return interpolation_set, active_subspace_set, subspace_matrix, rho_k, delta_k 
		
		return interpolation_set, active_subspace_set, subspace_matrix, rho_k, delta_k
		

	#the set parameter is of the form {x_k,x^2,...,x^q,...}
	def geometry_improvement(self, set, problem, polynomial_basis, delta_k, geometry_improving_flag) : 
		#build pivot polynomials 
		pivot_polynomials = [polynomial_basis[0](set[0])]
		q = len(set)
		for i in range(1, q) : 
			pivot_polynomials.append(lambda x : polynomial_basis[i](x) - (polynomial_basis[i](set[0])/polynomial_basis[0](set[0]))*polynomial_basis[0](x))

		new_set = [set.pop(0)] 
		for i in range(1, q) :
			if geometry_improving_flag :
				obj_fn_gi = lambda x : np.abs(pivot_polynomials[i](x))
				x_t = minimize(obj_fn_gi)
				new_solution = self.create_new_solution(tuple(x_t), problem)
				problem.simulate(new_solution,1)

			else : 
				obj_fn = lambda x : np.abs(pivot_polynomials[i](x_t))/(max(norm(x-x_t)**4/delta_k**4,1))
				x_t = max(list(map(obj_fn, set)))
				set.remove(x_t)

			new_set.append(x_t)

			#update pivot polynomials 
			for j in range(i, q) :
				pivot_polynomials[j] = lambda x : pivot_polynomials[j](x) - (pivot_polynomials[j](x_t)/pivot_polynomials[j](x_t))* pivot_polynomials[i](x)

		return new_set
	


"""
	Class that represents the geometry of the solution space. It is able to construct an interpolation set and handle geometric behaviours of the space
"""
class trust_region_interpolation_points :
	def __init__(self, problem, current_val):
		self.problem = problem
		self.current_val = current_val

	def assign_current_val(self, current_val) :
		"""
			Assigns the current iteration solution to the class member current_val

		Args:
			current_val (np.array): the current iteration solution 
		"""
		self.current_val = current_val

	def interpolation_points(self, delta):
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
	
