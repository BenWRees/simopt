"""
	The following functions define constructuting the vandermonde matrix and it's columnwise derivative that is used in model construction. 
"""

import sys
import os.path as o
sys.path.append(
o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), ".."))
)  #

from math import ceil, isnan, isinf, comb, log, floor
from copy import deepcopy

import numpy as np 

from numpy.linalg import norm, pinv, qr
from numpy.polynomial.chebyshev import chebvander, chebder, chebroots, cheb2poly
from scipy.optimize import minimize, NonlinearConstraint, linprog
from scipy import optimize
from scipy.special import factorial
from sklearn.metrics import r2_score
import scipy
import math
import random
from enum import Enum


import geometry_improvement_test as geo
import design_set_test as design


from simopt.base import (
	Problem,
	Solution,
)

#TODO: Generalise Polynomial Basis 

from simopt.experiment_base import instantiate_problem


class polynomial_basis(Enum) : 
	pass 

def chebyshev_V(X: np.ndarray, delta: float) -> np.ndarray :
	"""
		Generate the Vandermonde Matrix with a chebyshev basis. The design points are projected into the d-dimensional hypercube to ensure well-conditioning 
	Args:
		X (np.ndarray): The design set of shape (M,n) or (M,d) where n is the dimension of the original problem and d is the subspace dimension.

	Returns:
		np.ndarray: A vandermonde matrix of shape (M,q) where q is the length of the polynomial basis 
	"""
	degree = 2
	M, dim = X.shape
	indices = index_set(degree, dim).astype(int)
	X = scale(X,delta)
	print(f'Design Set matrix: {X}')
	# print(f'the number of rows in V is {M} and the number of columns is {len(self.indices)}')
	assert X.shape[1] == dim, "Expected %d dimensions, got %d" % (dim, X.shape[1])
	V_coordinate = [chebvander(X[:,k], degree) for k in range(dim)]

	
	V = np.ones((M, len(indices)), dtype = X.dtype)
	
	for j, alpha in enumerate(indices):
		for k in range(dim):
			V[:,j] *= V_coordinate[k][:,alpha[k]]
	return V

def scale(X: np.ndarray,delta: np.ndarray) -> np.ndarray:
	r""" Apply the scaling to the input coordinates
	"""
	#map to the unit ball 
	X_unit_ball = (X - X[0])/delta

	# Compute max absolute value along each row
	max_abs = np.max(np.abs(X_unit_ball), axis=1, keepdims=True)  # shape (M,1)
	
	# Avoid division by zero (for rows that are all zeros)
	max_abs[max_abs == 0] = 1.0
	
	# Map unit ball to cube [-1,1]^d
	X_hypercube = X_unit_ball / max_abs
	
	return X_hypercube
	

# def fit_coef(X: np.ndarray, fX: np.ndarray, U: np.ndarray) -> np.ndarray : 
# 	"""
# 		Finds the coefficients of the interpolation model by solving the system of equations:
# 				V(U^TX)coeff = fX
# 	Args:
# 		X (np.ndarray): The design set of design points of shape (M,n)
# 		fX (np.ndarray): corresponding function estimates of design points of shape (M,1)
# 		U (np.ndarray): The active subspace of shape (n,d)
# 		delta (float): radius of the current trust-region

# 	Returns:
# 		np.ndarray: A list of the coefficients of shape (q,1)
# 	"""
# 	#First find coefficients using chebyshev 

# 	#Then translate into a natural polynomial basis of m(x) c_0 + c_1x + c_2x^2 
# 	pass 


# def DV(X: np.ndarray) -> np.ndarray : 
# 	"""
# 		Column-wise derivative of the Vandermonde matrix

# 		Given design points this creates the Vandermonde-like matrix whose entries
# 		correspond to the derivatives of each of basis elements
# 	Args:
# 		X (np.ndarray): The design set of shape (M,d) where d is the subspace dimension.

# 	Returns:
# 		np.ndarray: Derivative of Vandermonde matrix  of shape (M,q,d) where DV[i,j,:] is the gradient of the
# 			partial derivative of the j-th basis function with respect to the x_k component of the d-dimensional vector 
# 			and evaluated at i-th design point 
# 	"""
# 	pass


def cheb_to_monomial_multivariate(c_cheb: np.ndarray) -> np.ndarray:
	"""
	Convert multivariate Chebyshev expansion coefficients into multivariate
	monomial (power basis) coefficients.

	Args:
		c_cheb (np.ndarray): coefficients in Chebyshev basis (q,1)

	Returns:
		np.ndarray: coefficients in the natural monomial basis (q,1),
					ordered according to index_set(degree, d)
	"""
	degree = 2 #self.factors['polynomial degree]
	d = 3 #self.factors['initial subspace dimension']
	indices = index_set(degree, d)  # shape (q, d)
	q = indices.shape[0]
	mono_coef = np.zeros((q, 1), dtype=float)

	for j, alpha in enumerate(indices):
		cj = float(c_cheb[j, 0])
		if cj == 0.0:
			continue

		# 1D Chebyshev → power-basis for each coordinate
		p_list = []
		for k in range(d):
			mk = int(alpha[k])
			cheb_series = np.zeros(mk + 1)
			cheb_series[mk] = 1.0
			pk = cheb2poly(cheb_series)
			p_list.append(pk)

		# Expand tensor product
		shapes = [len(p) for p in p_list]
		for beta in np.ndindex(*shapes):
			prod = np.prod([p_list[k][beta[k]] for k in range(d)])
			if prod == 0.0:
				continue
			beta_arr = np.array(beta, dtype=int)
			mask = np.all(indices == beta_arr, axis=1)
			if not np.any(mask):
				continue
			idx = np.where(mask)[0][0]
			mono_coef[idx, 0] += cj * prod

	return mono_coef


def fit_coef(X: np.ndarray, fX: np.ndarray, U: np.ndarray) -> np.ndarray:
	"""
	Finds the coefficients of the interpolation model by solving:
			V(U^T X) * coeff = fX

	Args:
		X (np.ndarray): The design set of design points of shape (M,n)
		fX (np.ndarray): corresponding function estimates of design points of shape (M,1)
		U (np.ndarray): The active subspace of shape (n,d)

	Returns:
		np.ndarray: coefficients expressed in the natural monomial basis
		ordered according to index_set(degree, d) (shape (q,1)).
	"""
	Z = X @ U
	delta = np.max(np.linalg.norm(X - X[0:1], axis=1))
	if delta == 0:
		delta = 1.0

	V = chebyshev_V(Z, delta)
	c_cheb = pinv(V) @ fX

	d = Z.shape[1]
	mono_coef = cheb_to_monomial_multivariate(c_cheb, d)
	return mono_coef


def DV(X: np.ndarray) -> np.ndarray : 
	"""
		Column-wise derivative of the Vandermonde matrix for the Chebyshev basis.

		Arguments:
			X (M, d) design points (already in subspace coordinates)

		Returns:
			DV (M, q, d) where DV[i,j,k] = d/d x_k of j-th Chebyshev basis function evaluated at X[i,:]
	"""
	# degree hard-coded in chebyshev_V
	degree = 2
	M, d = X.shape

	# Multi-indices and number of basis functions
	indices = index_set(degree, d)  # shape (q, d)
	q = indices.shape[0]

	# Precompute 1D Chebyshev basis values T_m(x) for each coordinate and degree m
	# and precompute their derivatives T'_m(x) by converting basis series using chebder then evaluating
	from numpy.polynomial.chebyshev import chebval, chebder

	# V_coord[k] will be (M, degree+1) with columns m = T_m(X[:,k])
	V_coord = [chebvander(X[:, k], degree) for k in range(d)]  # each is (M, degree+1)

	# Precompute derivatives values: D_coord[k][:, m] = d/dx T_m evaluated at X[:,k]
	D_coord = []
	for k in range(d):
		xk = X[:, k]
		# for each m (0..degree) compute derivative series for T_m then evaluate
		Mvals = np.zeros((M, degree + 1), dtype=float)
		for m in range(degree + 1):
			# cheb series for T_m: zeros with 1 at index m
			cheb_series = np.zeros(m + 1)
			cheb_series[m] = 1.0
			# derivative series in Chebyshev basis
			cheb_der_series = chebder(cheb_series)
			# evaluate derivative at xk
			if cheb_der_series.size == 0:  # derivative of constant T0 is zero
				Mvals[:, m] = 0.0
			else:
				Mvals[:, m] = chebval(xk, cheb_der_series)
		D_coord.append(Mvals)

	# Now assemble DV: for each basis (multi-index alpha), partial derivative wrt x_k is:
	# d/dx_k [ ∏_m T_{alpha_m}(x_m) ] = T'_{alpha_k}(x_k) * ∏_{m != k} T_{alpha_m}(x_m)
	DV_tensor = np.zeros((M, q, d), dtype=float)

	for j, alpha in enumerate(indices):
		# For efficiency, precompute the product over all coordinates of T_{alpha_m}(x_m)
		# then for each k, replace one factor by derivative.
		# But easier: for each k compute product_{m != k} V_coord[m][:, alpha[m]]
		for k in range(d):
			# start with derivative factor for coordinate k
			der_k = D_coord[k][:, int(alpha[k])]  # shape (M,)
			# product of other coordinates
			prod_others = np.ones(M, dtype=float)
			for m in range(d):
				if m == k:
					continue
				prod_others *= V_coord[m][:, int(alpha[m])]
			# final derivative for basis j wrt x_k evaluated at all M points:
			DV_tensor[:, j, k] = der_k * prod_others

	return DV_tensor
	
def build_Dmat() -> np.ndarray:
	"""
		Constructs the scalar derivative matrix
	Returns:
		np.ndarray: The scalar derivative of shape (degree+1, degree) where degree is the degree of the polynomial 
	"""
	degree = 2 #self.factors['polynomial degree']
	Dmat = np.zeros((degree+1, degree))
	I = np.eye(degree + 1)
	for j in range(degree + 1):
		Dmat[j,:] = chebder(I[:,j])

	return Dmat 

def full_index_set(degree:int, n: int) -> np.ndarray:
	"""

	Args:
		degree (int): The degree of the polynomial
		n (int): The dimension of the design points

	Returns:
		np.ndarray: _description_
	"""
	if n == 1:
		I = np.array([[degree]])
	else:
		II = full_index_set(degree, n-1)
		m = II.shape[0]
		I = np.hstack((np.zeros((m, 1)), II))
		for i in range(1, degree+1):
			II = full_index_set(degree-i, n-1)
			m = II.shape[0]
			T = np.hstack((i*np.ones((m, 1)), II))
			I = np.vstack((I, T))
	return I

def index_set(degree: int, n: int) -> np.ndarray: 
	"""

	Args:
		degree (int): The degree of the polynomial
		n (int): The dimension of the design points

	Returns:
		np.ndarray: Multi-indices ordered as columns with shape ()
	"""	
	I = np.zeros((1, n), dtype = np.int64)
	for i in range(1, degree+1):
		II = full_index_set(i, n)
		I = np.vstack((I, II))
	return I[:,::-1].astype(int)


#! SOME HELPER FUNCTIONS
def active_subspace(C: np.ndarray, d: int) -> np.ndarray:
	# Eigen-decomposition
	eigvals, eigvecs = np.linalg.eigh(C)
	# Sort in descending order
	idx = np.argsort(eigvals)[::-1]
	U = eigvecs[:, idx[:d]]
	return U

def cond_number(A: np.ndarray) -> float :
	U, S, Vt = np.linalg.svd(A)  # singular value decomposition
	cond_num = S[0] / S[-1]      # σ_max / σ_min
	return cond_num

def check_flatness(coef: list[float]) -> bool : 
	#Returns true if gradient small 
	x = np.linspace(0, 10, 500)

	# Derivative coefficients: [a1, 2*a2, 3*a3, ...]
	deriv_coeffs = [i * coef[i] for i in range(1, len(coef))]
	grad = np.polyval(list(reversed(deriv_coeffs)), x)

	max_grad = np.max(np.abs(grad))
	if max_grad < 0.01 : 
		return True 
	else :
		return False


def main() :
	# Instantiate problem, tr radius, and active subspace
	problem = instantiate_problem("ROSENBROCK-1")
	tests = 10000
	obj_fun = lambda x : np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
	dim = problem.dim
	deltas = [random.uniform(0.0000001, 5) for _ in range(tests)]
	C = np.random.rand(dim, dim)
	C = C.T @ C
	U = active_subspace(C, 3)
	

	cond_numbers = []
	matrices = []
	coefficients  = []
	# Initial point (center of local region)
	for i,delta in enumerate(deltas) :
		x0 = [random.uniform(0,100) for _ in range(dim)]
		current_solution = Solution(tuple(x0), problem)

		#Build design set 
		X, _ = design.construct_interpolation_set(current_solution, problem, U, delta, 1, [current_solution])
		fX = np.array([obj_fun(x) for x in X]).reshape(-1,1)

		V_X = chebyshev_V(X, delta) 
		print(f'Vandermonde matrix of test {i}: \n {V_X}')
		coef = pinv(V_X) @ fX 
		coefficients.append(coef)
		matrices.append(V_X)
		cond_numbers.append(cond_number(V_X))

	print("\n--- PERFORMANCE CHECKS ON VANDERMONDE MATRIX ---")
	print(f'Number of Experiments {len(deltas)}')
	print(f'Vandermonde Matrix shape: {matrices[-1].shape}')
	print(f'Vandermonde Condition Number: {max(cond_numbers)}')
	print(f'Are the models float? \n {[check_flatness(a) for a in coefficients]}')

	print("--- PERFORMANCE CHECKS FINISHED ---\n")



if __name__ == '__main__' : 
	main()