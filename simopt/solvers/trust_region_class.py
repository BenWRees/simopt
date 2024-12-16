from __future__ import annotations
from typing import Callable
from abc import abstractmethod

from numpy.linalg import pinv
from numpy.linalg import norm
import numpy as np
from math import ceil, factorial
import warnings
from scipy.optimize import NonlinearConstraint
from scipy.optimize import minimize
from scipy.special import eval_legendre
warnings.filterwarnings("ignore")
import importlib
import copy

from itertools import combinations_with_replacement

from simopt.base import (
    ConstraintType,
    ObjectiveType,
    Problem,
	Solution,
    Solver,
    VariableType,
)
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
				"default": "simopt.solvers.trust_region_class:natural_basis"
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
		module_name, class_name = self.factors['poly_basis'].split(':')
		module = importlib.import_module(module_name)
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
	
	"""
	def dot_prod(self, vector, tensor) :
		vals = []
		for i in range(len(tensor)) :
			vals.append(vector[i][0]*tensor[i][0])
		# res1 = np.tensordot(vector, tensor, axes=([0, 1], [0, 1]))
		# res2 = np.tensordot(vector, tensor, axes=([0, 1], [2, 3]))
		sum = 0
		for i in range(len(vals),2) : 
			sum += np.dot(vals[i], vals[i+1])

		# sum = np.tensordot(res1,res2)
		return sum"""
	
	"""
	def mat_mul(self, matrix, vector) :
		vector_res = []
		for row in matrix :	#idx is row number
			row_val= 0
			for i in range(len(row)) : 
				row_val += vector[i]*row[i] 
			vector_res.append(row_val)

		return np.array(vector_res).reshape(len(vector_res),1,*vector_res[0].shape)"""


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
		#TODO: Here we are seeing that for 2d legendre, this condition is being satisfied almost all the time.
		# Problem with calculating step size?
		model_reduction = model.local_model_evaluate(np.zeros(problem.dim), delta_k) - model.local_model_evaluate(stepsize, delta_k)
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
		poly_basis_instance = self.polynomial_basis_instantiation()(problem, 2)
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

			print('new solution: ', new_solution.x)

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
	
	# generate the coordinate vector corresponding to the variable number v_no
	def get_coordinate_vector(self, v_no):
		size = self.problem.dim
		arr = np.zeros(size)
		arr[v_no] = 1.0
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

		Y = [[x_k]]
		epsilon = 0.01
		for i in range(0, d):
			plus = Y[0] + delta * self.standard_basis(i)
			minus = Y[0] - delta * self.standard_basis(i)

			if sum(x_k) != 0: #check if x_k is not the origin
				# block constraints
				if minus[0][i] <= self.problem.lower_bounds[i]:
					minus[0][i] = self.problem.lower_bounds[i] + epsilon
				if plus[0][i] >= self.problem.upper_bounds[i]:
					plus[0][i] = self.problem.upper_bounds[i] - epsilon

			Y.append(plus)
			Y.append(minus)
		return Y
	

"""class chebyshev_interpolation(trust_region_geometry) : 
	def __init__(self, problem):
		super().__init__(problem)

	def generate_chebyshev_nodes(self, current_solution, delta) : 
		x_k = current_solution
		d = self.problem.dim

		Y = [[x_k]]
		epsilon = 0.01
		for i in range(0,2*d) :
			for n in range(d) : 
				chebyshev_point = []
				int_pt = [np.cos(np.pi * ((2*i+1)/(2*(2*d + 1))))]

			if sum(x_k) != 0: #check if x_k is not the origin
				# block constraints
				if int_pt[0][i] <= self.problem.lower_bounds[i]:
					int_pt[0][i] = self.problem.lower_bounds[i] + epsilon

			Y.append(int_pt)
		return Y 

	def calculate_chebyshev_fn(self, interpolation_set) : 
		chebyshev_points = []
		for pt in interpolation_set : 
			break

	def transform_interpolation_set(self, interpolation_set, delta) :
		trans_int_set = [] 
		current_sol = interpolation_set[0][0]
		trans_int_set.append(current_sol)
		for pt in interpolation_set[1:] : 
			#move to current sol 
			pt += current_sol
			#scale to delta
			pt *= delta
			trans_int_set.append(pt)

		return trans_int_set
		
	def interpolation_points(self, current_solution, delta):
		int_set = self.generate_chebyshev_nodes(current_solution, delta)
		return self.transform_interpolation_set(int_set, delta)"""
	

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
					interpolation_pt_solution = self.tr_instance.create_new_solution(tuple(Y[i][0]), self.problem)
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
			q, grad, Hessian = self.coefficient(Z, fval, delta_k)

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
	def coefficient(self, Y, fval, delta):
		d = self.problem.dim
		M = self.poly_basis.construct_matrix(Y, delta) # now constructs M based on the polynomial basis being used
		q = np.matmul(pinv(M), fval)
				
		grad = q[1:d + 1]
		grad = np.reshape(grad, d)

		if self.poly_basis.max_degree > 1 :
			Hessian = q[d + 1:len(fval)]
			Hessian = np.reshape(Hessian, d)
		else : 
			Hessian = []
			# self.M = M
		return q, grad, Hessian
	
	"""def collapse_tensor(self, tensor) :
		#tensor is of a shape (n,n,m,1). Want to reshape it to being (n,m*n)
		n,m,x,y = tensor.shape 
		reshaped_tensor = tensor.reshape(n, x * m)
		return reshaped_tensor"""
		
	def local_model_evaluate(self, x_k, delta):
		"""
			Calculate the solution of the local model at the point x_k
		
		Args:
			x_k ([float]): the current iteration's solution value
		"""
		q = self.coefficients[0]
		X = np.array([self.poly_basis.poly_basis_fn([[a] for a in [x_k]*len(q) ], 0, j, delta) for j in range(len(q))]) #tensor
		if len(X[0].shape) == 2 : 
			X = [i[0,0] for i in X]
		evaluation = np.dot(X,q)
		return evaluation

	"""def construct_symmetric_matrix(self,column_vector) : 
		column_vector = np.array(column_vector).reshape((len(column_vector),1))
		# Get dimensions
		# m, _, n, _ = column_vector.shape
		m,n = column_vector.shape

		# Initialize the symmetric matrix
		symmetric_matrix = np.zeros((m,m))


		# Fill in the symmetric matrix
		for i in range(m):
			for j in range(m):
				symmetric_matrix[i, j] = column_vector[abs(i - j)]

		return symmetric_matrix"""

	"""def dot_prod_tensor(self, vector1, vector2) : 
		#check if the size is mismatched
		if vector1.shape[1] != vector2.shape[1] : 
			diff = abs(vector1.shape[1] - vector2.shape[1])
			#the elements in vector 1 are smaller than in vector 2
			new_vect = []
			if vector1.shape[1] < vector2.shape[1] :
				for vect in vector1 : 
					zeros = np.zeros((diff,1))
					vect = np.vstack((vect,zeros))
					new_vect.append(vect)
				vector1 = np.array(new_vect)
			else : 
				for vect in vector2 : 
					zeros = np.zeros((diff,1))
					vect = np.vstack((vect,zeros))
					new_vect.append(vect)
				vector2 = np.array(new_vect) 

		sum = 0
		for i in range(vector1.shape[0]) : 
			sum += np.matmul(vector1[i].T,vector2[i])[0,0]

		return sum"""

	"""def matmul_tensor(self, tensor, vector) : 
		tensor_shape = tensor.shape 

		res_vect = []
		for j in range(tensor_shape[0]) : 
			sum_vect = np.zeros((tensor_shape[2], tensor_shape[3]))
			for i in range(tensor_shape[1]) :
				sum_vect += vector[i][0]*tensor[j,i]
			res_vect.append(sum_vect) 

		res = np.array(res_vect).reshape((len(res_vect),1, *res_vect[0].shape))
		return res"""

	"""def pinv_of_tensor(self, tensor) :
		# Get the shape of the input tensor
		n, m, j, k = tensor.shape
		
		# Initialize an empty array to store the pseudo-inverse tensor with shape (m, n, j, k)
		pseudo_inv_tensor = np.empty((m, n, j, k))
		
		# Iterate over each (j, k) slice and compute the pseudo-inverse along (n, m) axes
		for i in range(j):
			for l in range(k):
				# Extract the (n, m) matrix at position (i, l)
				matrix = tensor[:, :, i, l]
				
				# Compute the pseudo-inverse of the (n, m) matrix
				U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
				S_inv = np.diag([1/s if s > 1e-10 else 0 for s in S])
				pseudo_inv_matrix = Vt.T @ S_inv @ U.T
				
				# Store the result in the corresponding position in the pseudo-inverse tensor
				pseudo_inv_tensor[:, :, i, l] = pseudo_inv_matrix
		
		return pseudo_inv_tensor """

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


class polynomial_basis : 
	def __init__(self, problem, max_degree) : 
		self.problem = problem 
		self.max_degree = max_degree
		self.interpolation_set = None
		
	#Construct a matrix Row by Row
	def construct_matrix(self, interpolation_set, delta) :
		self.assign_interpolation_set(interpolation_set)
		no_rows = len(interpolation_set)
		no_cols = self.solve_no_cols(interpolation_set)
		matrix = []
		for i in range(no_rows) :
			row = []
			for j in range(no_cols) : 
				val = self.poly_basis_fn(interpolation_set,i,j, delta)	
				row.append(val) 
			matrix.append(row)

		X = np.array(matrix)
		return X
	
	@abstractmethod
	def poly_basis_fn(self, interpolation_set, row_num, col_num, delta): 
		raise NotImplementedError
	
	def assign_interpolation_set(self,interpolation_set) : 
		self.interpolation_set = interpolation_set

	@abstractmethod
	def solve_no_cols(self,interpolation_set) -> int: 
		raise NotImplementedError

class monimal_basis(polynomial_basis) : 
	def __init__(self, problem, max_degree):
		super().__init__(problem, max_degree)

	def poly_basis_fn(self, interpolation_set, row_num, col_num, delta):
		val = interpolation_set[row_num][0]
		#calculate the whole row,
		row = [1] 
		for exp in range(1, self.max_degree+1) : 
			row =  np.append(row, val**exp)

		return row[col_num]
	
	def solve_no_cols(self, interpolation_set):
		val = interpolation_set[0][0]
		return 1 + (len(val)*self.max_degree)


class natural_basis(polynomial_basis) : 
	def __init__(self, problem, max_degree) : 
		super().__init__(problem, max_degree)
	#natural basis function for each element in the matrix
	#each row should be of the form [1,x_1,x_2,...,x_p,(x_1)^2/2,]
	#No of cols needed: 1 + len(current_solution.x) +  
	def poly_basis_fn(self, interpolation_set, row_num, col_num, delta) : 
		val = interpolation_set[row_num][0]
		p = len(val)
		# Step 1: Define the full list of basis terms, up to max_degree
		basis_terms = []
	
		# Add the constant term 1
		basis_terms.append(lambda v: np.float64(1))
		
		# Add all first-order terms x_1, x_2, ..., x_p
		for i in range(p):
			basis_terms.append(lambda v, i=i: v[i].astype(np.float64).item())
		 
		
		# Add higher-order terms up to degree max_degree
		for degree in range(2, self.max_degree+1):
			for comb in combinations_with_replacement(range(p), degree):
				def term_func(v, comb=comb, degree=degree):
					result = 1
					for idx in comb:
						result *= v[idx]
					inner_result = result / factorial(degree)
					return inner_result
				basis_terms.append(term_func)
		# Evaluate the basis term at the given row (vector)
		res = basis_terms[col_num](val)
		# return basis_terms[col_num](val)
		return res
	
	def solve_no_cols(self, interpolation_set):
		val = interpolation_set[0][0]
		no_higher_order_terms = len([a for degree in range(2,self.max_degree+1) for a in combinations_with_replacement(range(len(val)), degree)]) 
		return no_higher_order_terms*(self.max_degree - 1) + len(val) + 1
		# return len(interpolation_set)
	
class lagrange_basis(polynomial_basis) : 
	def __init__(self, problem, max_degree) : 
		super().__init__(problem, max_degree)

	#lagrange polynomical function for each element in the matrix
	def poly_basis_fn(self, interpolation_set, row_num, col_num, delta) :
		current_val = interpolation_set[row_num][0]
		denom_val = interpolation_set[col_num][0]

		def basis_fn(x) :
			denominator = np.linalg.norm(denom_val - x)
			numerator = np.linalg.norm(current_val - x)
			if denominator == 0 : 
				denominator = 1 
			return numerator/denominator
		# basis_fn = lambda x : (np.linalg.norm(current_val - x))/(np.linalg.norm(denom_val - x))
		lagrange_list = [basis_fn(a) for idx,a in enumerate(interpolation_set) if col_num != idx]
		lagrange = np.prod(lagrange_list)
		return lagrange 
	
	def solve_no_cols(self, interpolation_set):
		return len(interpolation_set)
	
class NFP(polynomial_basis) : 
	def __init__(self, problem, max_degree) : 
		super().__init__(problem, max_degree) 

	def poly_basis_fn(self, interpolation_set, row_num, col_num, delta) :
		if col_num == 0 : 
			return 1
		val = np.array(interpolation_set[row_num])
		res = np.prod([(val-np.array(a)) for a in interpolation_set[:col_num]])
		return float(res)
	
	def solve_no_cols(self, interpolation_set):
		return len(interpolation_set)

class legendre_basis(polynomial_basis) : 
	def __init__(self, problem, max_degree, delta=0):
		super().__init__(problem, max_degree)

	def project_set(self, interpolation_set, delta) : 
		#project into spherical coordinates
		projected_set = []
		curr_soln = interpolation_set[0][0]
		for val in interpolation_set :
			val=val[0]
			r = norm(val)
			rad = np.array([0] * len(val))
			rad[0] = delta
			rad = rad + curr_soln
			v1_u = val/norm(val)
			v2_u = rad/norm(rad)

			if np.array_equal(val, np.zeros(val.shape)) :
				v1_u = np.zeros(v1_u.shape)


			angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

			
			projected_set.append(angle)
		return projected_set

	def project_to_current_trust_region(self, res) : 
		pass 
	
	#TODO: can also do norm of val and pass through eval_legendre
	def poly_basis_fn(self, interpolation_set, row_num, col_num, delta):	
		no_of_cols = self.solve_no_cols(interpolation_set)
		interpolation_set = self.project_set(interpolation_set, delta)
		val = np.array(interpolation_set[row_num])
		row = [] 
		for i in range(no_of_cols) :
			row.extend(eval_legendre(i, np.cos(val)).flatten())
		return row[col_num]
	
	def solve_no_cols(self, interpolation_set):
		return len(interpolation_set)*self.problem.dim
	




