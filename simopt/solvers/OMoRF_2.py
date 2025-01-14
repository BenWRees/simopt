# type: ignore
from __future__ import annotations

import warnings
warnings.filterwarnings('ignore')

from scipy.optimize import minimize, NonlinearConstraint, linprog
import numpy as np
from numpy.linalg import norm, qr
from scipy.special import factorial
from scipy.stats import linregress
import importlib
from math import comb

from .active_subspaces.basis import *
from .active_subspaces.polyridge import * 
from .active_subspaces.subspace import *
from .active_subspaces.index_set import IndexSet

from simopt.base import (
	ObjectiveType,
	ConstraintType,
	VariableType,
	Problem,
	Solution,
	Solver
	)
#TODO: - HANDLE GRADIENTS ENSURING THEY ARE NEVER GETTING 0 
#TODO: - HANDLE CONSTRAINTS WHEN SELECTING CANDIDATE SOLUTION
class OMoRF(Solver) :

	@property
	def objective_type(self) -> ObjectiveType: 
		return ObjectiveType.SINGLE
	
	@property
	def constraint_type(self) -> ConstraintType :
		return ConstraintType.BOX
	
	@property 
	def variable_type(self) -> VariableType :
		return VariableType.CONTINUOUS

	@property
	def gradient_needed(self) -> bool :
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
			"sampling rule" : {
				"description": "An instance of the sampling rule being used",
				"datatype": str,
				"default": 'BasicSampling' #just returns 10 every time
			}, 
			"lambda_min": {
				"description": "minimum sample size",
				"datatype": int,
				"default": 4
			},
			"geometry instance": {
				"description": "Instance of the geometric behaviours of the space where trust region values are sampled from",
				"datatype": str,
				"default": "TrustRegionGeometry"
			},
			"polynomial basis": {
				"description": "Polynomial basis to use in model construction",
				"datatype": str, 
				"default": "NaturalPolynomialBasis"
			}, 
			"polynomial degree": {
				"description": "degree of the polynomial",
				"datatype": int, 
				"default": 2
			},
			"model type" : {
				"description": "The type of random model used",
				"datatype": str,
				"default": "RandomModel" 
			},
			"model construction parameters": {
				"description": "List of initial parameters for the model construction",
				"datatype": dict, 
				"default": {'w': 0.85, 
					'mu':1000,
					'beta':10, 
					'criticality_threshold': 0.1, 
					'skip_criticality': True,
					'lambda_min': 4
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
				"default": 5
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
			"delta_max": self.check_delta_max,
			"delta": self.check_delta,
			"lambda_min": self.check_lambda_min,
			"geometry instance": self.check_geometry_instance, 
			"polynomial basis": self.check_poly_basis, 
			"model type": self.check_random_model_type,
			"model construction parameters": self.check_model_construction_parameters,
			"sampling rule": self.check_sampling_rule,
			"polynomial degree": self.check_poly_degree,
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
	
	def check_poly_degree(self) -> bool : 
		self.factors['polynomial degree'] >= 1
	
	def check_random_model_type(self) -> bool:
		return True 
	
	def check_model_construction_parameters(self) -> bool :
		return isinstance(self.factors['model construction parameters'], dict)
	
	def check_sampling_rule(self) -> bool:
		return True 

	#nice way to allow for different types of random models
	def model_instantiation(self) :
		class_name = self.factors['model type']
		module = importlib.import_module('simopt.solvers.TrustRegion.Models')
		return getattr(module, class_name)
	
	def polynomial_basis_instantiation(self) :
		class_name = self.factors['polynomial basis']
		module = importlib.import_module('simopt.solvers.active_subspaces.basis')
		return getattr(module, class_name)

	def sample_instantiation(self) :
		class_name = self.factors['sampling rule']
		module = importlib.import_module('simopt.solvers.TrustRegion.Sampling')
		sampling_instance = getattr(module, class_name)(self)
		return sampling_instance

	def geometry_type_instantiation(self) :
		class_name = self.factors['geometry instance']
		module = importlib.import_module('simopt.solvers.TrustRegion.Geometry')
		return getattr(module, class_name)

	
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

	def _set_bounds(self,problem) : 
		self.bounds = [problem.lower_bounds, problem.upper_bounds]

	def _set_iterate(self):
		ind_min = np.argmin(self.f) #get the index of the smallest function value 
		self.s_old = self.S[ind_min,:] #get the x-val that corresponds to the smallest function value 
		self.f_old = self.f[ind_min] #get the smallest function value 


	def _set_delta(self, val) : 
		self.delta_k = val

	def _set_counter(self, count) :
		self.unsuccessful_iteration_counter = count 

	def _set_ratio(self, ratio) : 
		self.ratio = ratio 

	def _set_rho_k(self, value):
		self.rho_k = value 

	

	def _blackbox_evaluation(self, s, problem):
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
			new_solution = self.create_new_solution(tuple(s), problem) 
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

	#! Add to geometry
	def _generate_set(self, num):
		"""
		Generates an initial set of samples using either coordinate directions or orthogonal, random directions
		"""
		bounds_l = np.maximum(np.array(self.bounds[0]).reshape(self.s_old.shape), self.s_old-self.delta_k)
		bounds_u = np.minimum(np.array(self.bounds[1]).reshape(self.s_old.shape), self.s_old+self.delta_k)

		if self.random_initial:
			direcs = self._random_directions(num, bounds_l-self.s_old, bounds_u-self.s_old)
		else:
			direcs = self._coordinate_directions(num, bounds_l-self.s_old, bounds_u-self.s_old)
		S = np.zeros((num, self.n))
		S[0, :] = self.s_old
		for i in range(1, num):
			S[i, :] = self.s_old + np.minimum(np.maximum(bounds_l-self.s_old, direcs[i, :]), bounds_u-self.s_old)
		return S

	#! Add to Geometry
	def _coordinate_directions(self, num_pnts, lower, upper):
		"""
		Generates coordinate directions
		"""
		at_lower_boundary = (lower > -1.e-8 * self.delta_k)
		at_upper_boundary = (upper < 1.e-8 * self.delta_k)
		direcs = np.zeros((num_pnts, self.n))
		for i in range(1, num_pnts):
			if 1 <= i < self.n + 1:
				dirn = i - 1
				step = self.delta_k if not at_upper_boundary[dirn] else -self.delta_k
				direcs[i, dirn] = step
			elif self.n + 1 <= i < 2*self.n + 1:
				dirn = i - self.n - 1
				step = -self.delta_k
				if at_lower_boundary[dirn]:
					step = min(2.0*self.delta_k, upper[dirn])
				if at_upper_boundary[dirn]:
					step = max(-2.0*self.delta_k, lower[dirn])
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
	
	#! Add to Geometry
	def _get_scale(dirn, delta, lower, upper):
		scale = delta
		for j in range(len(dirn)):
			if dirn[j] < 0.0:
				scale = min(scale, lower[j] / dirn[j])
			elif dirn[j] > 0.0:
				scale = min(scale, upper[j] / dirn[j])
		return scale

	#! Add to Geometry
	def _random_directions(self, num_pnts, lower, upper):
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
				scale = self._get_scale(-Q[:,i], self.delta_k, lower, upper)
				direcs[:, self.n+i] = -scale * Q[:,i]
		idx_active = np.where(active)[0]
		for i in range(nactive):
			idx = idx_active[i]
			direcs[idx, ninactive+i] = 1.0 if idx_l[idx] else -1.0
			direcs[:, ninactive+i] = self._get_scale(direcs[:, ninactive+i], self.delta_k, lower, upper) * direcs[:, ninactive+i]
			sign = 1.0 if idx_l[idx] else -1.0
			if upper[idx] - lower[idx] > self.delta_k:
				direcs[idx, self.n+ninactive+i] = 2.0*sign*self.delta_k
			else:
				direcs[idx, self.n+ninactive+i] = 0.5*sign*(upper[idx] - lower[idx])
			direcs[:, self.n+ninactive+i] = self._get_scale(direcs[:, self.n+ninactive+i], 1.0, lower, upper)*direcs[:, self.n+ninactive+i]
		for i in range(num_pnts - 2*self.n):
			dirn = np.random.normal(size=(self.n,))
			for j in range(nactive):
				idx = idx_active[j]
				sign = 1.0 if idx_l[idx] else -1.0
				if dirn[idx]*sign < 0.0:
					dirn[idx] *= -1.0
			dirn = dirn / norm(dirn)
			scale = self._get_scale(dirn, self.delta_k, lower, upper)
			direcs[:, 2*self.n+i] = dirn * scale
		return np.vstack((np.zeros(self.n), direcs[:, :num_pnts].T))

	#! Add to Geometry
	def _update_geometry_omorf(self, problem, S_full, f_full, S_red, f_red):
		dist = max(self.epsilon_1*self.delta_k, self.epsilon_2*self.rho_k)

		if max(norm(S_full-self.s_old, axis=1, ord=np.inf)) > dist:
			S_full, f_full = self._sample_set(problem, 'improve', S_full, f_full) #!! THIS IS CAUSING THERE TO BE TOO MANY X VALUES
			try:
				# print('UPDATING GEOMETRY')
				self._fit_subspace(S_full, problem) 
			except:
				pass
		
		elif max(norm(S_red-self.s_old, axis=1, ord=np.inf)) > dist:
			S_red, f_red = self._sample_set(problem, 'improve', S_red, f_red, full_space=False)
		
		elif self.delta_k == self.rho_k:
			# self.delta_k = self.alpha_2*self.rho_k
			self._set_delta(self.alpha_2*self.rho_k)


			# print(f'unsuccessful counter: {self.unsuccessful_iteration_counter}')
			# print(f'ratio: {self.ratio}')
			
			if self.unsuccessful_iteration_counter >= 3 and self.ratio < 0:
				# print('UNSUCCESSFUL 3 OR MORE TIMES')
				if self.rho_k >= 250*self.rho_min:
					# self.rho_k = self.alpha_1*self.rho_k
					self._set_rho_k(self.alpha_1*self.rho_k)
				
				elif 16*self.rho_min < self.rho_k < 250*self.rho_min:
					# self.rho_k = np.sqrt(self.rho_k*self.rho_min)
					self._set_rho_k(np.sqrt(self.rho_k*self.rho_min))
				
				else:
					# self.rho_k = self.rho_min
					self._set_rho_k(self.rho_min)
		
		return S_full, f_full, S_red, f_red
	
	#! Add to Geometry
	def _sample_set(self, problem, method, S=None, f=None, s_new=None, f_new=None, full_space=True):
		q = self.p if full_space else self.q

		dist = max(self.epsilon_1*self.delta_k, self.epsilon_2*self.rho_k)
		
		if method == 'replace':
			S_hat = np.vstack((S, s_new))
			f_hat = np.vstack((f, f_new))
			if S_hat.shape != np.unique(S_hat, axis=0).shape:
				S_hat, indices = np.unique(S_hat, axis=0, return_index=True)
				f_hat = f_hat[indices]
			elif f_hat.size > q and max(norm(S_hat-self.s_old, axis=1, ord=np.inf)) > dist:
				S_hat, f_hat = self._remove_furthest_point(S_hat, f_hat, self.s_old)
			S_hat, f_hat = self._remove_point_from_set(S_hat, f_hat, self.s_old)
			S = np.zeros((q, self.n))
			f = np.zeros((q, 1))
			S[0, :] = self.s_old
			f[0, :] = self.f_old
			S, f = self._LU_pivoting(problem, S, f, S_hat, f_hat, full_space)

		elif method == 'improve':
			S_hat = np.copy(S)
			f_hat = np.copy(f)
			if max(norm(S_hat-self.s_old, axis=1, ord=np.inf)) > dist:
				S_hat, f_hat = self._remove_furthest_point(S_hat, f_hat, self.s_old)
			S_hat, f_hat = self._remove_point_from_set(S_hat, f_hat, self.s_old)
			S = np.zeros((q, self.n))
			f = np.zeros((q, 1))
			S[0, :] = self.s_old
			f[0, :] = self.f_old
			S, f = self._LU_pivoting(problem, S, f, S_hat, f_hat, full_space, 'improve')
		
		elif method == 'new':
			S_hat = f_hat = np.array([])
			S = np.zeros((q, self.n))
			f = np.zeros((q, 1))
			S[0, :] = self.s_old
			f[0, :] = self.f_old
			S, f = self._LU_pivoting(problem, S, f, S_hat, f_hat, full_space, 'new')
		
		return S, f
	
	#! Add to Geometry
	def _LU_pivoting(self, problem, S, f, S_hat, f_hat, full_space, method=None):
		psi_1 = 1.0e-4
		psi_2 = 1.0 if full_space else 0.25

		phi_function, phi_function_deriv = self._get_phi_function_and_derivative(S_hat, full_space)
		q = self.p if full_space else self.q

		#Initialise U matrix of LU factorisation of M matrix (see Conn et al.)
		U = np.zeros((q,q))
		U[0,:] = phi_function(self.s_old)
  
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
					s = self._find_new_point(v, phi_function, phi_function_deriv, full_space)
					if np.unique(np.vstack((S[:k, :], s)), axis=0).shape[0] != k+1:
						s = self._find_new_point_alternative(v, phi_function, S[:k, :])
				except:
					s = self._find_new_point_alternative(v, phi_function, S[:k, :])
				if f_hat.size > 0 and M[index] >= abs(np.dot(v, phi_function(s))):
					s = S_hat[index,:]
					S[k, :] = s
					f[k, :] = f_hat[index]
					S_hat = np.delete(S_hat, index, 0)
					f_hat = np.delete(f_hat, index, 0)
				else:
					S[k, :] = s
					f[k, :] = self._blackbox_evaluation(s, problem)
			
			#Update U factorisation in LU algorithm
			phi = phi_function(s)
			U[k,k] = np.dot(v, phi)
			for i in range(k+1,q):
				U[k,i] += phi[i]
				for j in range(k):
					U[k,i] -= (phi[j]*U[j,i]) / U[j,j]
		return S, f

	#! Add to Geometry
	def _get_phi_function_and_derivative(self, S_hat, full_space):
		Del_S = self.delta_k

		if full_space:
			if S_hat.size > 0:
				Del_S = max(norm(S_hat-self.s_old, axis=1, ord=np.inf))
			
			def phi_function(s):
				s_tilde = np.divide((s - self.s_old), Del_S)
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
				Del_S = max(norm(np.dot(S_hat-self.s_old,self.U.U), axis=1))
			
			def phi_function(s):
				u = np.divide(np.dot((s - self.s_old), self.U.U), Del_S)
				# print(f'shape of u: {u.shape}')
				try:
					m,n = u.shape
				except:
					m = 1
					u = u.reshape(1,-1)
				phi = np.zeros((m, self.q))
				for k in range(self.q):
					phi[:,k] = np.prod(np.divide(np.power(u, self.index_set[k,:]), factorial(self.index_set[k,:])), axis=1) #FIX: Rewrite
				if m == 1:
					return phi.flatten()
				else:
					return phi
			
			def phi_function_deriv(s):
				u = np.divide(np.dot((s - self.s_old), self.U.U), Del_S)
				phi_deriv = np.zeros((self.d, self.q))
				for i in range(self.d):
					for k in range(1, self.q):
						if self.index_set[k, i] != 0.0: #Fix: Rewrite
							tmp = np.zeros(self.d)
							tmp[i] = 1
							phi_deriv[i,k] = self.index_set[k, i] * np.prod(np.divide(np.power(u, self.index_set[k,:]-tmp), factorial(self.index_set[k,:]))) #FIX: Rewrite
				phi_deriv = np.divide(phi_deriv.T, Del_S).T
				return np.dot(self.U.U, phi_deriv)
		
		return phi_function, phi_function_deriv
	
	def _find_new_point(self, v, phi_function, phi_function_deriv, full_space=False):
		#change bounds to be defined using the problem and delta_k
		bounds_l = np.maximum(np.array(self.bounds[0]).reshape(self.s_old.shape), self.s_old-self.delta_k)
		bounds_u = np.minimum(np.array(self.bounds[1]).reshape(self.s_old.shape), self.s_old+self.delta_k)

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
			res1 = minimize(obj1, self.s_old, method='TNC', jac=jac1, \
					bounds=bounds, options={'disp': False})
			res2 = minimize(obj2, self.s_old, method='TNC', jac=jac2, \
					bounds=bounds, options={'disp': False})
			#FIX: Don't want to rely on this (possibly)
			if abs(res1['fun']) > abs(res2['fun']):
				s = res1['x']
			else:
				s = res2['x']
		return s

	def _find_new_point_alternative(self, v, phi_function, S):
		S_tmp = self._generate_set(int(0.5*(self.n+1)*(self.n+2)))
		M = np.absolute(np.dot(phi_function(S_tmp), v).flatten())
		indices = np.argsort(M)[::-1][:len(M)]
		for index in indices:
			s = S_tmp[index,:]
			if np.unique(np.vstack((S, s)), axis=0).shape[0] == S.shape[0]+1:
				return s
		return S_tmp[indices[0], :]

	@staticmethod
	def _remove_point_from_set(S, f, s):
		ind_current = np.where(norm(S-s, axis=1, ord=np.inf) == 0.0)[0]
		S = np.delete(S, ind_current, 0)
		f = np.delete(f, ind_current, 0)
		return S, f

	@staticmethod
	def _remove_furthest_point(S, f, s):
		ind_distant = np.argmax(norm(S-s, axis=1, ord=np.inf))
		S = np.delete(S, ind_distant, 0)
		f = np.delete(f, ind_distant, 0)
		return S, f

	def _remove_points_outside_limits(self):
		ind_inside = np.where(norm(self.S-self.s_old, axis=1, ord=np.inf) <= max(self.epsilon_1*self.delta_k, \
				self.epsilon_2*self.rho_k))[0]
		S = self.S[ind_inside, :]
		f = self.f[ind_inside]
		return S, f

	def finite_difference_gradient(self, new_solution: Solution, problem: Problem) -> np.ndarray :
		"""Calculate the finite difference gradient of the problem at new_solution.

		Args:
			new_solution (Solution): The solution at which to calculate the gradient.
			problem (Problem): The problem`that contains the function to differentiate.

		Returns:
			np.ndarray: The solution value of the gradient 

			int: The expended budget 
		"""
		alpha = 1e-3		
		lower_bound = problem.lower_bounds
		upper_bound = problem.upper_bounds
		problem.simulate(new_solution,1)
		fn = -1 * problem.minmax[0] * new_solution.objectives_mean
		self.expended_budget += 1
		# new_solution = self.create_new_solution(tuple(x), problem)

		new_x = new_solution.x
		forward = np.isclose(new_x, lower_bound, atol = 10**(-7)).astype(int)
		backward = np.isclose(new_x, upper_bound, atol = 10**(-7)).astype(int)
		# BdsCheck: 1 stands for forward, -1 stands for backward, 0 means central diff.
		BdsCheck = np.subtract(forward, backward)

		# print(f'BdsCheck: {BdsCheck}')

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

			fn1 = 0 
			fn2 = 0
			x1_solution = self.create_new_solution(tuple(x1), problem)
			if BdsCheck[i] != -1:
				problem.simulate_up_to([x1_solution], 1)
				fn1 = -1 * problem.minmax[0] * x1_solution.objectives_mean
				self.expended_budget +=1
				# First column is f(x+h,y).
				FnPlusMinus[i, 0] = fn1
			
			x2_solution = self.create_new_solution(tuple(x2), problem)
			if BdsCheck[i] != 1:
				problem.simulate_up_to([x2_solution], 1)
				fn2 = -1 * problem.minmax[0] * x2_solution.objectives_mean
				self.expended_budget +=1
				# Second column is f(x-h,y).
				FnPlusMinus[i, 1] = fn2

			# Calculate gradient.
			if BdsCheck[i] == 0:
				grad[i] = (fn1 - fn2) / (2 * FnPlusMinus[i, 2])
			elif BdsCheck[i] == 1:
				grad[i] = (fn1 - fn) / FnPlusMinus[i, 2]
			elif BdsCheck[i] == -1:
				grad[i] = (fn - fn2) / FnPlusMinus[i, 2]


		# print(f'grad shape: {grad.shape}')
		return grad


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
			grads[idx, :] = self.finite_difference_gradient(x_solution, problem)

		# print(f'grads: {grads}')
		return grads 

	def _fit_subspace(self, X:np.ndarray, problem: Problem) -> None:
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
		
		self.S = np.array([])
		self.f = np.array([])
		self.g = np.array([])
		self.d = self.factors['subspace dimension']

		self.expended_budget = 0
		# self.delta_k = self.factors['delta']
		self._set_delta(self.factors['delta'])
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

		self._set_bounds(problem)
		
		""" 
		self.s_old = self._apply_scaling(s_old) #shifts the old solution into the unit ball
		"""

		self.k = 0
		self._set_counter(0)

		sampling_instance = self.sample_instantiation()

		#basis construction
		 
		# This returns 
		index_set = IndexSet('total-order', orders=np.tile([2], self.q))
		self.index_set = index_set.get_basis()[:,range(self.d-1, -1, -1)]
		
		self.poly_basis = self.polynomial_basis_instantiation()(self.factors['polynomial degree'], dim=self.factors['subspace dimension'])
		geometry_instance = self.geometry_type_instantiation()(problem)

		#instantiate ActiveSubspace and use it to construct the active subspace matrix 
		self.U = ActiveSubspace()

		self.s_old = np.array(current_x)
		self.f_old = self._blackbox_evaluation(self.s_old,problem)

		# Construct the sample set for the subspace 
		S_full = self._generate_set(self.d)
		f_full = np.zeros((self.d, 1))
		f_full[0, :] = self.f_old #first row gets the old function values 

		reset_budget = self.expended_budget

		#get the rest of the function evaluations - write as a function
		for i in range(1, self.d):
			#simulate the problem at each component of f_
			f_full[i, :] = self._blackbox_evaluation(S_full[i, :], problem)
			self.expended_budget = reset_budget #!QUICK FIX TO CHECK IF WE NEED TO FILL f_full
			#case where we use up our whole budget getting the function values 
			if not self.expended_budget < problem.factors['budget'] :
				return self.recommended_solns, self.intermediate_budget


		

		#initial subspace calculation - requires gradients of f_full 
		self._fit_subspace(S_full, problem)
		# self.s_old = self.U._U.T @ self.s_old 
		# print(f's_old shape (after mapping):{self.s_old.shape}')

		#This constructs the sample set for the model construction
		S_red, f_red = self._sample_set(problem, 'new', full_space=False)

		# print(f'AS Matrix: {self.U.U}')
		self.local_model = PolynomialRidgeApproximation(self.deg, self.d, problem, self.poly_basis)
		# self.local_model.fit(S_red, f_red, U0=self.U.U)
		
		while self.expended_budget < problem.factors['budget'] :
			# print(f'K: {self.k}')
			#if rho has been decreased too much we end the algorithm  
			if self.rho_k <= self.rho_min:
				break
			
			#BUILD MODEL
			try: 
				self.local_model.fit(S_red, f_red, U0=self.U.U) #this should be the model instance construct model
			except: #thrown if the sample set is not defined properly 
				S_red, f_red = self._sample_set(problem, 'improve', S_red, f_red, full_space=False)
				intermediate_budgets.append(expended_budget)
				continue 

			#SOLVE THE SUBPROBLEM
			candidate_solution, S_full, S_red, f_full, f_red, reset_flag = self.solve_subproblem(problem, S_full, S_red, f_full, f_red)

			candidate_fval = self._blackbox_evaluation(np.array(candidate_solution.x),problem)

			if reset_flag :
				recommended_solns.append(self.current_solution)
				intermediate_budgets.append(self.expended_budget) 
				self.k +=1
				break 
			
			#EVALUATE THE CANDIDATE SOLUTION
			S_red, S_full, f_red, f_full = self.evaluate_candidate_solution(problem, candidate_fval, candidate_solution, S_red, S_full, f_red, f_full)
		

			self.recommended_solns.append(self.current_solution) 
			self.intermediate_budgets.append(self.expended_budget)

			# print(f'EXPENDED BUDGET: {self.expended_budget}')

			self.k+=1


		return self.recommended_solns, self.intermediate_budgets



	def solve_subproblem(self, problem: Problem, S_full:np.ndarray, S_red: np.ndarray, f_full: float, f_red: float) :
		"""
			Solves the trust-region subproblem for ``trust-region`` or ``omorf`` methods
		"""
		
		omega_s = self.factors['omega_shrinking']
		reset_flag = False

		# bounds_l = np.maximum(np.array(self.bounds[0]).reshape(self.s_old.shape), self.s_old-self.delta_k)
		# bounds_u = np.minimum(np.array(self.bounds[1]).reshape(self.s_old.shape), self.s_old+self.delta_k)
		
		# #The bounds here are the trust region bounds in each dimension - can change to taking the norm of x
		# bounds = []
		# for i in range(self.n):
		# 	bounds.append((bounds_l[i], bounds_u[i]))


		cons =  NonlinearConstraint(lambda x : norm(self.s_old - x), 0, self.delta_k)
		
			
		# res = minimize(lambda x: np.asscalar(self.local_model.eval(x)), self.s_old, \
		# 		method='TNC', jac=lambda x: np.dot(self.U, my_poly.get_polyfit_grad(np.dot(x,self.U))).flatten(), \
		# 		bounds=bounds, options={'disp': False})
		obj = lambda x: self.local_model.eval(x)[0]
		res = minimize(obj, self.s_old, method='trust-constr', constraints=cons, options={'disp': False})
		s_new = res.x
		# m_new = res.fun 

		# print(f'CANDIDATE SOLUTION: {s_new}')

		for i in range(problem.dim):
			if s_new[i] <= problem.lower_bounds[i]:
				s_new[i] = problem.lower_bounds[i] + 0.01
			elif s_new[i] >= problem.upper_bounds[i]:
				s_new[i] = problem.upper_bounds[i] - 0.01

		# print(f'CANDIDATE SOLUTION (AFTER CONSTRAINTS):{s_new}')

		candidate_solution = self.create_new_solution(tuple(s_new), problem)

		step_dist = norm(s_new - self.s_old, ord=np.inf)

		# Safety step implemented in BOBYQA
		if step_dist < omega_s*self.rho_k:
			# self.ratio= -0.1
			self._set_ratio(-0.1)
			self._set_counter(3)
			# self.delta_k = max(0.5*self.delta_k, self.rho_k)
			self._set_delta(max(0.5*self.delta_k, self.rho_k))
			S_full, f_full, S_red, f_red = self._update_geometry_omorf(problem, S_full, f_full, S_red, f_red)
			
		
		#this is a breaking condition
		if self.rho_k <= self.rho_min:
			reset_flag=True

		return candidate_solution, S_full, S_red, f_full, f_red, reset_flag

	#TODO: Rewrite for Simopt
	def evaluate_candidate_solution(self, problem: Problem, fval_tilde: float, candidate_solution: Solution, S_red: np.ndarray, S_full: np.ndarray, f_red: np.ndarray, f_full: np.ndarray) :
		
		gamma_1 = self.factors['gamma_1']
		gamma_2 = self.factors['gamma_2']
		gamma_3 = self.factors['gamma_3']
		eta_1 = self.factors['eta_1']
		eta_2 = self.factors['eta_2']
		s_new = np.array(candidate_solution.x)
		
		del_f = self.f_old - fval_tilde #self.f_old - f_new 
		# del_m = np.asscalar(my_poly.get_polyfit(np.dot(self.s_old,self.U))) - m_new
		del_m = self.local_model.eval(self.s_old)[0] - self.local_model.eval(s_new)[0]

		step_dist = norm(np.array(candidate_solution.x) - self.s_old, ord=np.inf)

		#in the case that the denominator is very small 
		if abs(del_m) < 100*np.finfo(float).eps:
			# self.ratio = 1.0
			self._set_ratio(1.0)
		else:
			# self.ratio = del_f/del_m
			self._set_ratio(del_f/del_m)

		self._set_iterate

		"""ind_min = np.argmin(self.f) #get the index of the smallest function value 
		self.s_old = self.S[ind_min,:] #get the x-val that corresponds to the smallest function value 
		self.f_old = np.asscalar(self.f[ind_min]) #get the smallest function value""" 

		#add candidate value and corresponding fn val to interpolation sets
		S_red, f_red = self._sample_set(problem, 'replace', S_red, f_red, s_new, fval_tilde, full_space=False)
		S_full, f_full = self._sample_set(problem, 'replace', S_full, f_full, s_new, fval_tilde)

		# print(f'RATIO COMPARISON VALUE: {self.ratio}')

		if self.ratio >= eta_2:
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
			S_full, f_full, S_red, f_red = self._update_geometry_omorf(problem, S_full, f_full, S_red, f_red)

		return S_red, S_full, f_red, f_full