from __future__ import annotations
from typing import Callable

from numpy.linalg import norm, pinv
import numpy as np
from math import ceil
import warnings
from scipy.optimize import NonlinearConstraint
from scipy.optimize import minimize
warnings.filterwarnings("ignore")
import importlib
import copy


from simopt.base import (
    ConstraintType,
    ObjectiveType,
    Problem,
	Solution,
    Solver,
    VariableType,
)

from simopt.solvers.active_subspaces.basis import *

# from .astrodf_ext import adaptive_sampling
# from .tr_with_reuse_pts import random_model_reuse


"""
	Class for a probabilistic trust region, as based on bandeira et al (2014).
"""
#TODO: Provide data for starting delta
class trust_region(Solver) :

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
			"reuse_points": {
				"description": "reuse the previously visited points",
				"datatype": bool,
				"default": False
			},
			"sampling_rule" : {
				"description": "An instance of the sampling rule being used",
				"datatype": str,
				"default": 'simopt.solvers.trust_region_class:basic_sampling' #just returns 10 every time
			}, 
			"lambda_min": {
				"description": "minimum sample size",
				"datatype": int,
				"default": 4
			},
			"geometry instance": {
				"description": "Instance of the geometric behaviours of the space where trust region values are sampled from",
				"datatype": str,
				"default": "simopt.solvers.trust_region_class:trust_region_geometry"
			},
			"poly_basis": {
				"description": "Polynomial basis to use in model construction",
				"datatype": str, 
				"default": "NaturalPolynomialBasis"
			}, 
			"random_model type" : {
				"description": "The type of random model used",
				"datatype": str,
				"default": "simopt.solvers.trust_region_class:random_model" 
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
			"poly_basis": self.check_poly_basis, 
			"random_model type": self.check_random_model_type,
			"sampling_rule": self.check_sampling_rule,
		}
	
	def __init__(self, name="TRUSTREGION", fixed_factors: dict | None = None) -> None :
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
	
	def check_random_model_type(self) -> bool:
		return True 
	
	def check_sampling_rule(self) -> bool:
		return True 

	#nice way to allow for different types of random models
	def model_instantiation(self) :
		module_name, class_name = self.factors['random_model type'].split(':')
		module = importlib.import_module(module_name)
		return getattr(module, class_name)
	
	def polynomial_basis_instantiation(self) :
		class_name = self.factors['poly_basis']
		module = importlib.import_module('simopt.solvers.active_subspaces.basis')
		return getattr(module, class_name)

	def sample_instantiation(self) :
		module_name, class_name = self.factors['sampling_rule'].split(':')
		module = importlib.import_module(module_name)
		sampling_instance = getattr(module, class_name)(self)
		return sampling_rule(self, sampling_instance)

	def geometry_type_instantiation(self) :
		module_name, class_name = self.factors['geometry instance'].split(':')
		module = importlib.import_module(module_name)
		return getattr(module, class_name)
	
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
			def subproblem(s) : 
				Hessian_matrix = self.construct_symmetric_matrix(Hessian)
				return fval[0] + np.dot(s,grad) + np.dot(np.matmul(s,Hessian_matrix),s)
			
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
		if self.factors['random_model type'] == 'random_model_reuse' :
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
	#TODO: implement the adaptive solving rule
	#TODO: ensure that the kappa is being handled by the sampling instance correctly
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
		model_construction_parameters = {
		'w': 0.85, 
		'mu':1000,
		'beta':10, 
		'criticality_threshold': 0.1, 
		'skip_criticality': True,
		'lambda_min': self.factors['lambda_min']
		}

		#Dynamically load in different sampling rule, geometry type, and random model
		sampling_instance = self.sample_instantiation()
		geometry_instance = self.geometry_type_instantiation()(problem)
		poly_basis_instance = self.polynomial_basis_instantiation()(2, None, problem.dim)
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
			model, problem, fval_tilde, delta_k, interpolation_solns, candidate_solution, recommended_solns
			new_solution, delta_k, recommended_solns = self.evaluate_candidate_solution(model, problem, fval_tilde, delta_k, interpolation_solns, current_solution,\
																			   candidate_solution, recommended_solns)	
			

			intermediate_budgets.append(expended_budget)


			# print('new solution: ', new_solution.x)

		return recommended_solns, intermediate_budgets


"""
	Class that represents the geometry of the solution space. It is able to construct an interpolation set and handle geometric behaviours of the space
"""
class trust_region_geometry :
	def __init__(self, problem):
		self.problem = problem

	def standard_basis(self, index):
		"""
		Creates a standard basis vector e_i in the space of dimension equal to the problem dimension. Where i is at the index of the index parameter
		Args:
			index (int): the location of the value 1 in the standard basis vector
		
		Returns:
			np.array: a standard basis vector of the form (0,0,...,0,1,0,...,0), where the 1 is in the location of index
		"""
		arr = np.zeros(self.problem.dim)
		arr[index] = 1.0
		return arr

	def interpolation_points(self, current_solution, delta):
		"""
		Constructs an interpolation set of 
		
		Args:
			delta (TYPE): Description
		
		Returns:
			[np.array]: Description
		"""
		x_k = current_solution
		d = self.problem.dim

		Y = [x_k]
		epsilon = 0.01
		for i in range(0, d):
			plus = Y[0] + delta * self.standard_basis(i)
			minus = Y[0] - delta * self.standard_basis(i)

			if sum(x_k) != 0: #check if x_k is not the origin
				# block constraints
				if minus[i] <= self.problem.lower_bounds[i]:
					minus[i] = self.problem.lower_bounds[i] + epsilon
				if plus[i] >= self.problem.upper_bounds[i]:
					plus[i] = self.problem.upper_bounds[i] - epsilon

			Y.append(plus)
			Y.append(minus)
		return Y	

class random_model :
	"""
	Class for a stochastic interpolation model. This is currently the best surrogate model to use in stochastic trust-region algorithms
	
	Attributes:
		coefficients ([np.array]): a list of values containing the coefficients of the model, along with the Jacobian matrix and the Hessian matrix
		current_solution (base.Solution): The solution for which the random model is being centered around
		fval ([float]): The function evaluations at each sample point of the interpolation set
		interpolation_sets (trust_region_geometry): An instance of the trust-region space to sample from around the current solution
		problem (base.Problem): the current simulation-optimisation problem being solved
		sampling_rule (sampling_rule): instance of the sampling rule to be applied for calculating new function value
	
	Deleted Attributes:
		sample_size (int): number of times to sample
	"""

	def __init__(self, geometry_instance, tr_instance, poly_basis, problem, sampling_instance, model_construction_parameters) :
		self.coefficients = [] 
		self.geometry_instance = geometry_instance
		self.tr_instance = tr_instance
		self.problem = problem
		self.sampling_instance = sampling_instance
		self.poly_basis = poly_basis
		self.fval = None
		#in the case of the random_model the visited points list is not being added to, for reuse, it grows every iteration
		# self.visited_pts_list = visited_pts_list
		self.model_construction_parameters = model_construction_parameters
		self.M = None

	#nice way to allow for different types of random models

	#Constructs the model
	def construct_model(self, current_solution, delta, k, expended_budget, visited_pts_list) -> tuple[
        Solution,
        float,
        int,
        list[Solution],
        list[Solution],
		int
    ]:
		interpolation_solns = []
		j = 0
		# interpolation_sets = self.geometry_type_instantiation()(self.problem, current_solution.x)
		d = self.problem.dim

		while True:
			fval = []
			j = j + 1
			delta_k = delta * self.model_construction_parameters['w'] ** (j - 1)

			#calculate kappa - model construction happens once per iteration, so this will only happen once per iteration
			if hasattr(self.sampling_instance.sampling_rule, 'calculate_kappa') and k==1 :
				#only calculate if the sampling instance has the class 'calculate_kappa' defined
				lambda_max = self.problem.factors['budget'] - expended_budget
				lambda_min = self.model_construction_parameters["lambda_min"]
				pilot_run = ceil(max(lambda_min, min(.5 * self.problem.dim, lambda_max)) - 1)
				self.problem.simulate(current_solution, pilot_run)
				expended_budget += pilot_run
				sample_size = pilot_run
				expended_budget = self.sampling_instance.sampling_rule.calculate_kappa(self.problem, current_solution, delta_k, k, expended_budget, sample_size)

			# construct the interpolation set
			empty_geometry = copy.deepcopy(self.geometry_instance)
			
			Z = empty_geometry.interpolation_points(np.zeros(self.problem.dim), delta_k)
			Y = self.geometry_instance.interpolation_points(np.array(current_solution.x), delta_k)


			for i in range(len(Y)):
				# For X_0, we don't need to simulate the system
				if (k == 1) and (i==0):
					self.problem.simulate(current_solution,1)
					fval.append(-1 * self.problem.minmax[0] * current_solution.objectives_mean)
					interpolation_solns.append(current_solution)

				# Otherwise, we need to simulate the system
				else:
					interpolation_pt_solution = self.tr_instance.create_new_solution(tuple(Y[i]), self.problem)
					# check if there is existing result
					self.problem.simulate(interpolation_pt_solution, 1)
					expended_budget += 1
					init_sample_size = 1
					sig_2 = 0

					interpolation_pt_solution, sampling_budget = self.sampling_instance(self.problem, interpolation_pt_solution, k, delta_k, expended_budget, init_sample_size, sig_2)

					# current_solution = new_solution
					expended_budget = sampling_budget

					fval.append(-1 * self.problem.minmax[0] * interpolation_pt_solution.objectives_mean)
					interpolation_solns.append(interpolation_pt_solution)			
			
			
			# construct the model and get the model coefficients
			q, grad, Hessian = self.coefficient(Z, fval)

			if not self.model_construction_parameters['skip_criticality']:
				# check the condition and break
				if norm(grad) > self.model_construction_parameters['criticality_threshold']:
					break

			if delta_k <= self.model_construction_parameters['mu'] * norm(grad):
				break

		self.coefficients = [q, grad, Hessian]
		self.fval = fval
		delta_k = min(max(self.model_construction_parameters['beta'] * norm(grad), delta_k), delta)


		return current_solution, delta_k, expended_budget, interpolation_solns, visited_pts_list, 1

	#Calculate the Model coefficients
	#TODO: When dealing with tensors for M, q, grad, and Hessian, reshape to be matrices and vectors
	def coefficient(self, Y, fval):
		d = self.problem.dim
		Y = np.row_stack(Y) #reshape Y to be a matrix of (M,d)
		# print('Y (after reshape): ', Y)
		self.poly_basis.assign_interpolation_set(Y)
		M = self.poly_basis.V(Y) # now constructs M based on the polynomial basis being used
		q = np.matmul(pinv(M), fval)
		# print('q: ', q)
				
		grad = q[1:d + 1]
		grad = np.reshape(grad, d)

		if self.poly_basis.degree > 1 :
			Hessian = q[d + 1:len(fval)]
			Hessian = np.reshape(Hessian, d)
		else : 
			Hessian = []
			# self.M = M
		return q, grad, Hessian
		
	def local_model_evaluate(self, x_k):
		"""
			Calculate the solution of the local model at the point x_k
		
		Args:
			x_k ([float]): the current iteration's solution value
		"""
		q = self.coefficients[0]
		interpolation_set = x_k.reshape((1,len(x_k)))
		# interpolation_set = np.row_stack(interpolation_set)
		X = self.poly_basis.V(interpolation_set)[0]
		if len(X[0].shape) == 2 : 
			X = [i[0,0] for i in X]
		evaluation = np.dot(X,q)
		return evaluation

#function to handle basic sampling. For ASTRODF, this will be more complicated
class sampling_rule :
	def __init__(self, tr_instance, sampling_rule) :
		self.tr_instance = tr_instance
		self.sampling_rule = sampling_rule
		self.kappa = None

	#When sampling_rule is called is the behaviour of the sampling rule
	def __call__(self, problem, current_solution, n, delta_k, used_budget, init_sample_size, sig2, sample_after=True) :
		current_solution, budget = self.sampling_rule(problem, current_solution, n, delta_k, used_budget, init_sample_size, sig2, sample_after)
		return current_solution, budget

	# def __call__(self, *params) :
	# 	current_solution, budget = self.sampling_instance(*params)
	# 	return current_solution, budget
	
#This is a basic dynamic sampling rule - samples the objective fuction more 

class basic_sampling :
	def __init__(self, tr_instance) :
		self.tr_instance = tr_instance

	def __call__(self, problem, current_solution, n, delta_k, used_budget, init_sample_size, sig2, sample_after) : 
		#sample 10 times 
		sample_number = 10
		problem.simulate(current_solution,sample_number)
		used_budget += sample_number
		return current_solution, used_budget