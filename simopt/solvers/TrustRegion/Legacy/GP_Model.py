import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
import scipy.linalg
from scipy.linalg import eigh, expm, logm
from scipy.optimize import fmin_l_bfgs_b
import scipy.optimize
from itertools import product
from numpy.linalg import norm

from ..TrustRegion import * 
from ..Sampling import * 
from ..Geometry import *

from ...active_subspaces.basis import *

from ....base import (
	Problem,
	Solver, 
	Solution,
)
		


class GP_Model(random_model) :
	"""
		Constructs a Gaussian Process as the random model.
		The class inherits coefficient, local_model_evaluate, and construct_model from random_model that are the only public functions of this class. 


	Parent Class:
		random_model
	"""
	def __init__(self, geometry_instance: trust_region_geometry , tr_instance: trust_region, poly_basis: Basis, problem: Problem, sampling_instance: sampling_rule , model_construction_parameters: dict):

		self.structure = model_construction_parameters['structure']

		self.mu = model_construction_parameters['mu']

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
		#TODO: Run this in a loop until we have decreased delta enough
		# while True : 
			
			#update next iteration's trust region
			# delta_k = delta * 0.5**(k-1)

		#construct the trust_region_interpolation points
		Y = self.geometry_instance.interpolation_points(np.array(current_solution.x),delta)
		
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



			# if delta_k <= self.mu * norm(grad) : 
			# 	break

		self.fval = fvals[0]

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
	
	def _calculate_mean(self, X: np.ndarray) -> float : 
		"""The mean function of the gaussian process

		Args:
			X (np.ndarray): Set of M n-dim points to form a matrix of shape (M,n)

		Returns:
			float: The mean of the Gaussian proces
		"""
		pass 


	def prediction(self, x: np.ndarray) -> float : 
		"""run an acquisition function to find the next best point. Called when solving the subproblem

		Args:
			x (np.ndarray): current point to check

		Returns:
			float: The expected improvement of the Gaussian Process under the current point
		"""
		pass  

	def local_model_evaluate(self, x_k: list[float]) -> float:
		"""Gets the evaluation of the model at the solution x_k

		Args:
			x_k (list[float]): The vector to evaluate the model at 

		Returns:
			float: the model evaluation at x_k
		"""
		interpolation_set = x_k.reshape((1,len(x_k)))
		return self.eval(interpolation_set)


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
			
			return Lexp

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
		
		# Covariance matrix
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

		ew, ev = scipy.linalg.eigh(K + self.nugget*np.eye(K.shape[0]))
		#if np.min(ew) <= 0:
		#	bonus_regularization = -2*np.min(ew)+1e-14 
		#	ew += bonus_regularization
		#	K += bonus_regularization*np.eye(K.shape[0])

		if return_obj:
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
		self._fit(ell0)



	def _fit(self, ell0):
		# the implementation in l_bfgs_b seems flaky when we have invalid values
		#ell, obj, d = fmin_l_bfgs_b(self._obj, ell0, fprime = self._grad, disp = True)
		res = scipy.optimize.minimize(self._obj, 
				ell0, 
				jac = self._grad,
				#method = 'L-BFGS-B',
				#options = {'disp': True}, 
			)
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
