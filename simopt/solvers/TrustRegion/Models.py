#type: ignore
# TODO: 
#?	- Implement the acquisition function in GPModel
#?  - Fix the optimisation bottleneck in the _fit function for GPModel 

from numpy.linalg import norm, pinv
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.linalg import eigh, expm, logm
import numpy as np
import scipy 
from scipy.optimize import minimize, NonlinearConstraint
from math import ceil
import warnings
warnings.filterwarnings("ignore")
import copy
import time

from simopt.base import (
	Problem,
	Solution,
	Solver,
	VariableType,
)

from ..active_subspaces.basis import *
from .TrustRegion import * 
from .Sampling import * 
from .Geometry import *

__all__ = ['RandomModel', 'RandomModelReuse', 'GPModel']

class RandomModel :
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
			if hasattr(self.sampling_instance.__class__, "calculate_kappa") and k==1 :
				#only calculate if the sampling instance has the class 'calculate_kappa' defined
				lambda_max = self.problem.factors['budget'] - expended_budget
				lambda_min = self.model_construction_parameters["lambda_min"]
				pilot_run = ceil(max(lambda_min, min(.5 * self.problem.dim, lambda_max)) - 1)
				self.problem.simulate(current_solution, pilot_run)
				expended_budget += pilot_run
				sample_size = pilot_run
				expended_budget = self.sampling_instance.calculate_kappa(self.problem, current_solution, delta_k, k, expended_budget, sample_size)

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
	def coefficient(self, Y: list[np.ndarray], fval: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		d = self.problem.dim
		Y = np.row_stack(Y) #reshape Y to be a matrix of (M,d)
		self.poly_basis.assign_interpolation_set(Y)
		M = self.poly_basis.V(Y) # now constructs M based on the polynomial basis being used
		q = np.matmul(pinv(M), fval)
				
		grad = q[1:d + 1]
		grad = np.reshape(grad, d)

		if self.poly_basis.degree > 1 :
			Hessian = q[d + 1:len(fval)]
			Hessian = np.reshape(Hessian, d)
		else : 
			Hessian = np.zeros(d,)
		
		self.M = M
		return q, grad, Hessian
		
	def local_model_evaluate(self, x_k: list[float]) -> float:
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
	
class RandomModelReuse(RandomModel) :
	def __init__(self, geometry_instance, tr_instance, poly_basis, problem, sampling_instance, model_construction_parameters) :
		super().__init__(geometry_instance, tr_instance, poly_basis, problem, sampling_instance, model_construction_parameters)
	
	def construct_model(self, current_solution, delta, k, expended_budget, visited_pts_list) -> tuple[
		Solution,
		float,
		int,
		list[Solution],
		list[Solution],
		int
	]:
		interpolation_solns = []
		x_k = current_solution.x
		reuse_points = True
		lambda_min: int = self.model_construction_parameters["lambda_min"]
		
		j = 0
		budget: int = self.problem.factors["budget"]
		lambda_max = budget - expended_budget
		pilot_run = ceil(max(lambda_min, min(.5 * self.problem.dim, lambda_max)) - 1)
		# lambda_max = budget / (15 * sqrt(problem.dim))
		
		if len(visited_pts_list) == 0 :
			visited_pts_list.append(current_solution)

		while True:
			fval = []
			j = j + 1
			delta_k = delta * self.model_construction_parameters['w'] ** (j - 1)

			#calculate kappa - model construction happens once per iteration, so this will only happen once per iteration
			if hasattr(self.sampling_instance.__class__, 'calculate_kappa') and k==1 :
				#only calculate if the sampling instance has the class 'calculate_kappa' defined
				self.problem.simulate(current_solution, pilot_run)
				expended_budget += pilot_run
				sample_size = pilot_run
				expended_budget = self.sampling_instance.calculate_kappa(self.problem, current_solution, delta_k, k, expended_budget, sample_size)
			else : 
				self.problem.simulate(current_solution, 2) 
				expended_budget += 2

			# Calculate the distance between the center point and other design points
			Dist = []
			for i in range(len(visited_pts_list)):
				Dist.append(norm(np.array(visited_pts_list[i].x) - np.array(x_k))-delta_k)
				# If the design point is outside the trust region, we will not reuse it (distance = -big M)
				if Dist[i] > 0:
					Dist[i] = -delta_k*10000

			# Find the index of visited design points list for reusing points
			# The reused point will be the farthest point from the center point among the design points within the trust region
			f_index = Dist.index(max(Dist))

			# If it is the first iteration or there is no design point we can reuse within the trust region, use the coordinate basis
			if (k == 1) or (norm(np.array(x_k) - np.array(visited_pts_list[f_index].x))==0) or not reuse_points :
				# Construct the interpolation set


				empty_geometry = copy.deepcopy(self.geometry_instance)
			
				Z = empty_geometry.interpolation_points(np.zeros(self.problem.dim), delta_k)
				Y = self.geometry_instance.interpolation_points(np.array(current_solution.x), delta_k)

			# Else if we will reuse one design point
			elif k > 1:
				first_basis = (np.array(visited_pts_list[f_index].x)-np.array(x_k)) / norm(np.array(visited_pts_list[f_index].x)-np.array(x_k))
				# if first_basis has some non-zero components, use rotated basis for those dimensions
				rotate_list = np.nonzero(first_basis)[0]
				rotate_matrix = self.geometry_instance.get_rotated_basis(first_basis, rotate_list)

				# if first_basis has some zero components, use coordinate basis for those dimensions
				for i in range(self.problem.dim):
					if first_basis[i] == 0:
						rotate_matrix = np.vstack((rotate_matrix, self.geometry_instance.standard_basis(i)))

				# construct the interpolation set
				Y = self.geometry_instance.get_rotated_basis_interpolation_points(np.array(x_k), delta_k, rotate_matrix, visited_pts_list[f_index].x)
				
				empty_geometry = copy.deepcopy(self.geometry_instance)
				Z = empty_geometry.get_rotated_basis_interpolation_points(np.zeros(self.problem.dim), delta_k, rotate_matrix, np.array(visited_pts_list[f_index].x) - np.array(x_k))
			else:
				error_msg = "Error in constructing the interpolation set"
				raise ValueError(error_msg)
	
			# Evaluate the function estimate for the interpolation points
			for i in range(2 * self.problem.dim + 1):
				# for x_0, we don't need to simulate the new solution
				if (k == 1) and (i == 0):
					# self.problem.simulate(current_solution,1) #no need to simulate the new solution
					fval.append(-1 * self.problem.minmax[0] * current_solution.objectives_mean)
					interpolation_solns.append(current_solution)
				# reuse the replications for x_k (center point, i.e., the incumbent solution)
				elif (i == 0):
					#SAMPLING STRAT 1
					init_sample_size = current_solution.n_reps
					sig2 = current_solution.objectives_var

					current_solution, sampling_budget = self.sampling_instance(self.problem, current_solution,\
																  k, delta_k, expended_budget, init_sample_size, sig2, False)
					expended_budget = sampling_budget
					fval.append(-1 * self.problem.minmax[0] * current_solution.objectives_mean)
					interpolation_solns.append(current_solution)

				# else if reuse one design point, reuse the replications
				elif (i == 1) and (norm(np.array(x_k) - np.array(visited_pts_list[f_index].x)) != 0) and reuse_points :
					reuse_solution = visited_pts_list[f_index]
					#SAMPLING STRAT 2
					init_sample_size = reuse_solution.n_reps
					# sig2 = self.sampling_instance.sampling_rule.get_sig_2(visited_pts_list[f_index])
					sig2 = reuse_solution.objectives_var
					
					reuse_solution, sampling_budget = self.sampling_instance(self.problem, reuse_solution,\
																  k, delta_k, expended_budget, init_sample_size, sig2, False)
					expended_budget = sampling_budget
					fval.append(-1 * self.problem.minmax[0] * reuse_solution.objectives_mean)
					interpolation_solns.append(reuse_solution)

				# for new points, run the simulation with pilot run
				else:
					#SAMPLING STRAT 3
					interpolation_pt_solution = self.tr_instance.create_new_solution(tuple(Y[i]), self.problem)
					visited_pts_list.append(interpolation_pt_solution)
					self.problem.simulate(interpolation_pt_solution, pilot_run)
					expended_budget += pilot_run 
					init_sample_size = pilot_run
		
					interpolation_pt_solution, sampling_budget = self.sampling_instance(self.problem, interpolation_pt_solution, k, delta_k, expended_budget, init_sample_size,0)
					expended_budget = sampling_budget
					fval.append(-1 * self.problem.minmax[0] * interpolation_pt_solution.objectives_mean)
					interpolation_solns.append(interpolation_pt_solution)

			# get the current model coefficients
			q, grad, Hessian = self.coefficient(Z, fval)

			if not self.model_construction_parameters['skip_criticality']:
				# check the condition and break
				if norm(grad) > self.model_construction_parameters['criticality_threshold']:
					break

			if delta_k <= self.model_construction_parameters['mu'] * norm(grad):
				break

			# If a model gradient norm is zero, there is a possibility that the code stuck in this while loop
			if norm(grad) == 0:
				break
		
		#save the final coefficients and function values
		self.coefficients = [q, grad, Hessian]
		self.fval = fval
		delta_k = min(max(self.model_construction_parameters['beta'] * norm(grad), delta_k), delta)

		return current_solution, delta_k, expended_budget, interpolation_solns, visited_pts_list, pilot_run


class GPModel(RandomModel) :
	"""
		Constructs a Gaussian Process as the random model.
		The class inherits coefficient, local_model_evaluate, and construct_model from random_model that are the only public functions of this class. 
		
		Parameters: 
			L: np.ndarray(m,m)
				Distance matrix
			alpha: np.ndarray(M)
				Weights for the Gaussian process kernel
			beta: np.ndarray(N)
				Weights for the polynomial component in the LegendreTensorBasis	
			K: np.ndarray() 
				The covariance matrix 
			obj: float
				Log-likelihood objective function value
		Parent Class:
			random_model
	"""
	def __init__(self, geometry_instance: TrustRegionGeometry , tr_instance: TrustRegion, poly_basis: Basis, problem: Problem, sampling_instance: SamplingRule , model_construction_parameters: dict):

		self.structure = model_construction_parameters['structure']

		# self.mu = model_construction_parameters['mu']

		self.rank = None # currently disabled due to conditioning issues
		self.n_init = model_construction_parameters['n_init']
		self.degree = model_construction_parameters['degree']

		self._best_score = np.inf

		self.nugget = model_construction_parameters['nugget']
		if model_construction_parameters['nugget'] is None:
			self.nugget = 5*np.finfo(float).eps

		if self.structure == 'scalar_mult':
			assert model_construction_parameters['Lfixed'] is not None, "Must specify 'Lfixed' to use scalar_mult"
			self.Lfixed = model_construction_parameters['Lfixed']

		self.cov = None

		super().__init__(geometry_instance, tr_instance, poly_basis, problem, sampling_instance, model_construction_parameters)

	#!! THIS FUNCTION DEFINITION CAN'T BE MODIFIED 
	def construct_model(self, current_solution: Solution, delta: float, k: int, expended_budget: int, visited_pts_list: list[np.array]) -> tuple[
		Solution,
		float,
		int,
		list[Solution],
		list[Solution],
		int
	]:
		"""Constructs the trust region radius (delta) and the coefficients of the poly basis 

		Args:
			current_solution (Solution): The current incumbent solution of the trust region
			delta (float): the current trust region radius
			k (int): the current iteration of the solver
			expended_budget (int): The budget expended already by the solver
			visited_pts_list (list[np.array]): A list of visited points to be used when reusing interpolation points

		Returns:
			tuple[ Solution, float, int, list[Solution], list[Solution], int ]: _description_
		"""
		# while True : 
			
			#update next iteration's trust region
			# delta_k = delta * 0.5**(k-1)

		#construct the trust_region_interpolation points
		Y = self.geometry_instance.interpolation_points(np.array(current_solution.x),delta)
		d = self.problem.dim
		
		#Find the function values of each interpolation_pt 
		fvals = []
		for int_pt in Y : 
			interpolation_solution = self.tr_instance.create_new_solution(tuple(int_pt), self.problem)
			self.problem.simulate(interpolation_solution, 1)
			expended_budget += 1
			fn = -1 * self.problem.minmax[0] * interpolation_solution.objectives_mean
			fvals.append(fn)

		Y = np.row_stack(Y)
		#fit the Gaussian process 
		self.fit(Y,fvals)

		self.fval = fvals[0]
		q = self.beta.flatten()
		grad = q[1:d + 1]
		grad = np.reshape(grad, d)

		if self.poly_basis.degree > 1 :
			Hessian = q[d + 1:len(fvals)]
			Hessian = np.reshape(Hessian, d)
		else : 
			Hessian = np.zeros(d,)
		
		self.coefficients = [q,grad,Hessian]

		return current_solution, delta, expended_budget, interpolation_solution, visited_pts_list, 1
	
	def _calculate_cov(self, X: np.ndarray, Y: list[float], K: np.ndarray) -> np.ndarray :
		"""Generates the Covariance matrix of the Gaussian Process

		Args:
			X (np.ndarray): Set of M n-dim points to form a matrix of shape (M,n)
			Y (list[float]): A list of M values of the problem evaluated at each point in the set of X 
			K (np.ndarray): The Kernel function

		Returns:
			np.ndarray: The covariance matrix
		"""
		KK = np.exp(-0.5*squareform(pdist(Y, 'sqeuclidean')))
		ew, ev = eigh(KK)	
		I = (ew > 500*np.finfo(float).eps)
		z = ev[:,I].dot( np.diag(1./ew[I]).dot(ev[:,I].T.dot(K.T)))
		#z = np.dot(ev[:,I], np.dot(np.diag(1./ew[I]).dot(ev[:,I].T.dot( K.T)) ))
		cov = np.array([ 1 - np.dot(K[i,:], z[:,i]) for i in range(X.shape[0])])
		cov[cov< 0] = 0.

		return cov

	#!! Implement this method
	def prediction(self, x: np.ndarray, exploration_param: float) -> float : 
		"""run the expected improvement (EI) acquisition function to find the next best point. Called when solving the subproblem

		Args:
			x (np.ndarray): current point to check

		Returns:
			float: The expected improvement of the Gaussian Process under the current point
		"""
		# Determine model's predictive distribution (mean and
		# standard-deviation)
		mu_x, sigma_x = self.predictive_distribution(np.atleast_2d(x))

		gamma_x = (mu_x - (incumbent + self.kappa)) / sigma_x
		# Compute EI based on some temporary variables that can be reused in
		# gradient computation
		tmp_erf = erf(gamma_x / np.sqrt(2))
		tmp_ei_no_std = 0.5*gamma_x * (1 + tmp_erf) \
			+ np.exp(-gamma_x**2/2)/np.sqrt(2*np.pi)
		ei = sigma_x * tmp_ei_no_std

		return ei


	
	#!! THIS FUNCTION DEFINITION CAN'T BE MODIFIED 
	def local_model_evaluate(self, x_k: list[float]) -> float:
		"""Gets the evaluation of the model at the solution x_k

		Args:
			x_k (list[float]): The vector to evaluate the model at 

		Returns:
			float: the model evaluation at x_k
		"""
		interpolation_set = x_k.reshape((1,len(x_k)))
		return self.eval(interpolation_set)

	#?L is the distance matrix used to construct elements of the 
	def _make_L(self, ell):
		r""" Constructs the L matrix from the parameterization corresponding to the structure
		"""


		if self.structure == 'const':
			return np.exp(ell)*np.eye(self.m)
		elif self.structure == 'scalar_mult':
			return np.exp(ell)*self.Lfixed
		elif self.structure == 'diag':
			return np.diag(np.exp(ell))	
		elif self.structure == 'tril':
			# Construct the L matrix	
			L = np.zeros((self.m*self.m,), dtype = ell.dtype)
			L[self.tril_flat] = ell
			L = L.reshape(self.m,self.m)
			
			# This is a more numerically stable way to compute expm(L) - I
			#Lexp = L.dot(scipy.linalg.expm(L))
			
			# JMH 8 Aug 2019: I'm less sure about the value of parameterizing as L*expm(L)
			# Specifically, having the zero matrix easily accessible doesn't seem like a good thing
			# and he gradient is more accurately computed using this form. 
			Lexp = scipy.linalg.expm(L)

			print(f'Lexp: {Lexp}')
			
			return Lexp


	#TODO: Clean up function
	def _log_marginal_likelihood(self, ell, X = None, y = None, return_obj = True, return_grad = False, return_alpha_beta = False):
		
		if X is None: X = self.X
		if y is None: y = self.y 

		# Extract basic constants
		M = X.shape[0]	
		m = X.shape[1]	


		L = self._make_L(ell)
		# Compute the squared distance
		Y = np.dot(L, X.T).T
		dij = pdist(Y, 'sqeuclidean') 
		

		#Kernel Matrix
		K = np.exp(-0.5*squareform(dij))

		# Solve the linear system to compute the coefficients
		#alpha = np.dot(ev,(1./(ew+tikh))*np.dot(ev.T,y)) 
		A = np.vstack([np.hstack([K + self.nugget*np.eye(K.shape[0]), self.V]), 
					   np.hstack([self.V.T, np.zeros((self.V.shape[1], self.V.shape[1]))])])
		b = np.hstack([y, np.zeros(self.V.shape[1])])

		# As A can be singular, we use an eigendecomposition based inverse
		ewA, evA = eigh(A)
		I = (np.abs(ewA) > 5*np.finfo(float).eps)
		x = np.dot(evA[:,I],(1./ewA[I])*np.dot(evA[:,I].T,b))

		alpha = x[:M]
		beta = x[M:]

		if return_alpha_beta:
			return alpha, beta
		
		print(f'Gaussian Kernel: {K + self.nugget*np.eye(K.shape[0])} ')

		ew, ev = scipy.linalg.eigh(K + self.nugget*np.eye(K.shape[0]))
		#if np.min(ew) <= 0:
		#	bonus_regularization = -2*np.min(ew)+1e-14 
		#	ew += bonus_regularization
		#	K += bonus_regularization*np.eye(K.shape[0])

		if return_obj:
			#TODO: rewrite this to undergo the optimisation solving and return the result 
			# Should this be with yhat or y?
			# yhat = y - np.dot(V, beta)
			# Doesn't matter because alpha in nullspace of V.T
			# RW06: (5.8)
			with np.errstate(invalid = 'ignore'):
				obj = 0.5*np.dot(y, alpha) + 0.5*np.sum(np.log(ew))
			if not return_grad:
				return obj
		
		# Now compute the gradient
		# Build the derivative of the covariance matrix K wrt to L
		if self.structure == 'tril':
			dK = np.zeros((M,M, len(ell)))
			for idx, (k, el) in enumerate(self.tril_ij):
				eidx = np.zeros(ell.shape)
				eidx[idx] = 1.
				# Approximation of the matrix exponential derivative [MH10]
				h = 1e-10
				dL = np.imag(self._make_L(ell + 1j*h*eidx))/h
				dY = np.dot(dL, X.T).T
				for i in range(M):
					# Evaluate the dot product
					# dK[i,j,idx] -= np.dot(Y[i] - Y[j], dY[i] - dY[j])
					dK[i,:,idx] -= np.sum((Y[i] - Y)*(dY[i] - dY), axis = 1)
			for idx in range(len(self.tril_ij)):
				dK[:,:,idx] *= K
	
		elif self.structure == 'diag':
			dK = np.zeros((M,M, len(ell)))
			for idx, (k, el) in enumerate(self.tril_ij):
				for i in range(M):
					dK[i,:,idx] -= (Y[i,k] - Y[:,k])*(Y[i,el] - Y[:,el])
			for idx in range(len(self.tril_ij)):
				dK[:,:,idx] *= K	

		elif self.structure in ['const', 'scalar_mult']:
			# In the scalar case everything drops and we 
			# simply need 
			# dK[i,j,1] = (Y[i] - Y[j])*(Y[i] - Y[j])*K
			# which we have already computed
			dK = -(squareform(dij)*K).reshape(M,M,1)
				
		# Now compute the gradient
		grad = np.zeros(len(ell))
	
		for k in range(len(ell)):
			#Kinv_dK = np.dot(ev, np.dot(np.diag(1./(ew+tikh)),np.dot(ev.T,dK[:,:,k])))
			#I = (ew > 0.1*np.sqrt(np.finfo(float).eps))
			#I = (ew>5*np.finfo(float).eps)
			#print "k", k, "dK", dK.shape

			Kinv_dK = np.dot(ev, (np.dot(ev.T,dK[:,:,k]).T/ew).T)
			# Note flipped signs from RW06 eq. 5.9
			grad[k] = 0.5*np.trace(Kinv_dK)
			grad[k] -= 0.5*np.dot(alpha, np.dot(alpha, dK[:,:,k]))

		# self.grad = grad #9/01/25 added by me 
		if return_obj and return_grad:
			return obj, grad
		if not return_obj:
			return grad


	def _obj(self, ell, X = None, y = None):
		return self._log_marginal_likelihood(ell, X, y, 
			return_obj = True, return_grad = False, return_alpha_beta = False)
	
	def _grad(self, ell, X = None, y = None):
		return self._log_marginal_likelihood(ell, X, y, 
			return_obj = False, return_grad = True, return_alpha_beta = False)

	
	def _fit_init(self, X, y):
		m = self.m = X.shape[1]
		self.X = X
		self.y = y
		# Setup structure based properties
		if self.structure == 'tril':
			if self.rank is None: rank = m
			else: rank = self.rank
			
			self.tril_ij = [ (i,j) for i, j in zip(*np.tril_indices(m)) if i >= (m - rank)]
			self.tril_flat = np.array([ i*m + j for i,j in self.tril_ij])

		elif self.structure == 'diag':
			self.tril_ij = [ (i,i) for i in range(m)]


		# Cache Vandermonde matrix on sample points
		if self.degree is not None:
			#TODO: change to be any basis
			self.basis = LegendreTensorBasis(self.degree, X = X) 
			self.V = self.basis.V(X)
		else:
			self.V = np.zeros((X.shape[0],0))
		


	def fit(self, X, y, L0 = None):
		""" Fit a Gaussian process model

		Parameters
		----------
		X: array-like (M, m)
			M input coordinates of dimension m
		y: array-like (M,)
			y[i] is the output at X[i]
		"""
		X = np.array(X)
		y = np.array(y).flatten()
	
		# Initialized cached values for fit
		self._fit_init(X, y)	
		ell0 = None

		if L0 is None:
			L0 = np.eye(self.m)

		if self.structure == 'tril':
			ell0 = np.array([L0[i,j] for i, j in self.tril_ij])
		elif self.structure == 'diag':
			if len(L0.shape) == 1:
				ell0 = L0.flatten()
			else:
				ell0 = np.array([L0[i,i] for i, j in self.tril_ij])
		elif self.structure == 'scalar_mult':
			ell0 = np.array(L0.flatten()[0])
		elif self.structure == 'const':
			ell0 = np.array(L0.flatten()[0])

		# Actually do the fitting
		# TODO: Implement multiple initializations
		print('RUNNING _fit')
		self._fit(ell0)



	def _fit(self, ell0):
		# the implementation in l_bfgs_b seems flaky when we have invalid values
		# ell, obj, d = fmin_l_bfgs_b(self._obj, ell0, fprime = self._grad, disp = True)
		#!!Bottlenecking at this blackbox optimisation
		# NfEval = 1
		# time_now = time.time()
		# def debug_callback(X) :
		# 	global NfEval 
		# 	global time_now
		# 	time_diff = time.time() - time_now 
		# 	print(f'{NfEval}\t{X}\t{self._obj(X)}\t{time_diff}')
		# 	NfEval += 1
		# 	time_now = time.time()

		print('Iter\t ell0_val\t obj(ell0)\t time\t')
		res = minimize(self._obj, 
				ell0, 
				jac = self._grad,
				method = 'L-BFGS-B',
				# options = {'disp': True},
				# callback=debug_callback,
			)
		print('FINISHED OPTIMISING')
		ell = res.x
		self.L = self._make_L(ell)
		self.alpha, self.beta = self._log_marginal_likelihood(ell, 
			return_obj = False, return_grad = False, return_alpha_beta = True)
		self._ell = ell


	def eval(self, Xnew):
		Y = np.dot(self.L, self.X.T).T
		Ynew = np.dot(self.L, Xnew.T).T
		dij = cdist(Ynew, Y, 'sqeuclidean')	
		K = np.exp(-0.5*dij)
		if self.degree is not None:
			V = self.basis.V(Xnew)
		else:
			V = np.zeros((Xnew.shape[0],0))

		fXnew = np.dot(K, self.alpha) + np.dot(V, self.beta)
		
		self.cov = self._calculate_cov(Xnew, Y, K)

		return fXnew
