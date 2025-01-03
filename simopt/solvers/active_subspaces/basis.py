#type: ignore 
""" Descriptions of various bases"""

from abc import abstractmethod

__all__ = ['PolynomialTensorBasis', #THIS IS A BASE CLASS TO BE INHERITED BY SPECIFIC TENSOR BASIS CLASSES 
	'MonomialTensorBasis', 
	'LegendreTensorBasis',
	'ChebyshevTensorBasis',
	'LaguerreTensorBasis',
	'HermiteTensorBasis',
	'ArnoldiPolynomialBasis', #FIX THIS ONE
	'MonomialPolynomialBasis',
	'NaturalPolynomialBasis',
	'LagrangePolynomialBasis',
	'NFPPolynomialBasis'
 ]



import numpy as np
from numpy.polynomial.legendre import legvander, legder, legroots 
from numpy.polynomial.chebyshev import chebvander, chebder, chebroots
from numpy.polynomial.hermite import hermvander, hermder, hermroots
from numpy.polynomial.laguerre import lagvander, lagder, lagroots
from itertools import combinations_with_replacement
from math import factorial, comb
import sympy as sp


class Basis(object):
	pass



################################################################################
# Indexing utility functions for total degree
################################################################################


def _full_index_set(n, d):
	""" A helper function for index_set.
	
	Parameters
	----------
	n : int
		degree of polynomial
	d : int
		number of variables, dimension
	"""
	if d == 1:
		I = np.array([[n]])
	else:
		II = _full_index_set(n, d-1)
		m = II.shape[0]
		I = np.hstack((np.zeros((m, 1)), II))
		for i in range(1, n+1):
			II = _full_index_set(n-i, d-1)
			m = II.shape[0]
			T = np.hstack((i*np.ones((m, 1)), II))
			I = np.vstack((I, T))
	return I

def index_set(n, d):
	"""Enumerate multi-indices for a total degree of order `n` in `d` variables.
	
	Parameters
	----------
	n : int
		degree of polynomial
	d : int
		number of variables, dimension
	Returns
	-------
	I : ndarray
		multi-indices ordered as columns
	"""
	I = np.zeros((1, d), dtype = np.integer)
	for i in range(1, n+1):
		II = _full_index_set(i, d)
		I = np.vstack((I, II))
	return I[:,::-1].astype(int)


class PolynomialTensorBasis(Basis):
	r""" Generic tensor product basis of fixed total degree

	This class constructs a tensor product basis of dimension :math:`n`
	of fixed given degree :math:`p` given a basis for polynomials
	in one variable. Namely, this basis is composed of elements:

	This is a base class

	.. math::

		\psi_j(\mathbf x) := \prod_{i=1}^n \phi_{[\boldsymbol \alpha_j]_i}(x_i) 
			\quad \sum_{i=1}^n [\boldsymbol \alpha_j]_i \le p;
			\quad \phi_i \in \mathcal{P}_{i}(\mathbb{R})


	Parameters
	----------
	dim: int
		The input dimension of the space
	degree: int
		The total degree of polynomials
	polyvander: function
		Function providing the scalar Vandermonde matrix (i.e., numpy.polynomial.polynomial.polyvander)
	polyder: function
		Function providing the derivatives of scalar polynomials (i.e., numpy.polynomial.polynomial.polyder)	
 
	"""

	def __init__(self, degree, X = None, dim = None):
		self.degree = int(degree)
		if X is not None:
			self.X = np.atleast_2d(X)
			self.dim = self.X.shape[1]
			self.set_scale(self.X)
		elif dim is not None:
			self.dim = int(dim)
			self.X = None
	
		self.indices = index_set(self.degree, self.dim).astype(int)
		self._build_Dmat()

	def __len__(self):
		return len(self.indices)
	
	def assign_interpolation_set(self,X) : 
		self.X = X

	def _build_Dmat(self):
		""" Constructs the (scalar) derivative matrix
		"""
		self.Dmat = np.zeros( (self.degree+1, self.degree))
		I = np.eye(self.degree + 1)
		for j in range(self.degree + 1):
			self.Dmat[j,:] = self.polyder(I[:,j])

	def set_scale(self, X):
		r""" Construct an affine transformation of the domain to improve the conditioning
		"""
		self._set_scale(np.array(X))

	def _set_scale(self, X):
		r""" default scaling to [-1,1]
		"""
		self._lb = np.min(X, axis = 0)
		self._ub = np.max(X, axis = 0)

	def _scale(self, X):
		r""" Apply the scaling to the input coordinates
		"""
		try:
			return 2*(X-self._lb[None,:])/(self._ub[None,:] - self._lb[None,:]) - 1
		except AttributeError:
			return X

	def _dscale(self):
		r""" returns the scaling associated with the scaling transform
		"""
		try:
			return (2./(self._ub - self._lb))
		except AttributeError:
			raise NotImplementedError

	def V(self, X = None):
		r""" Builds the Vandermonde matrix associated with this basis

		Given points :math:`\mathbf x_i \in \mathbb{R}^n`, 
		this creates the Vandermonde matrix

		.. math::

			[\mathbf{V}]_{i,j} = \phi_j(\mathbf x_i)

		where :math:`\phi_j` is a multivariate polynomial as defined in the class definition.

		Parameters
		----------
		X: array-like (M, n)
			Points at which to evaluate the basis at where :code:`X[i]` is one such point in 
			:math:`\mathbf{R}^n`.
		
		Returns
		-------
		V: np.array
			Vandermonde matrix
		"""
		if X is None and self.X is not None:
			X = self.X
		elif X is None:
			raise NotImplementedError

		X = X.reshape(-1, self.dim)
		X = self._scale(np.array(X))
		M = X.shape[0]
		assert X.shape[1] == self.dim, "Expected %d dimensions, got %d" % (self.dim, X.shape[1])
		V_coordinate = [self.vander(X[:,k], self.degree) for k in range(self.dim)]
		
		V = np.ones((M, len(self.indices)), dtype = X.dtype)
		
		for j, alpha in enumerate(self.indices):
			for k in range(self.dim):
				V[:,j] *= V_coordinate[k][:,alpha[k]]
		return V

	def VC(self, X, c):
		r""" Evaluate the product of the Vandermonde matrix and a vector

		This evaluates the product :math:`\mathbf{V}\mathbf{c}`
		where :math:`\mathbf{V}` is the Vandermonde matrix defined in :code:`V`.
		This is done without explicitly constructing the Vandermonde matrix to save
		memory.	
		 
		Parameters
		----------
		X: array-like (M,n)
			Points at which to evaluate the basis at where :code:`X[i]` is one such point in 
			:math:`\mathbf{R}^n`.
		c: array-like 
			The vector to take the inner product with.
		
		Returns
		-------
		Vc: np.array (M,)
			Product of Vandermonde matrix and :math:`\mathbf c`

		NOTE
		----
		This is an optimisation technique not currently implemented in the simopt library
		"""
		X = X.reshape(-1, self.dim)
		X = self._scale(np.array(X))
		M = X.shape[0]
		c = np.array(c)
		assert len(self.indices) == c.shape[0]

		if len(c.shape) == 2:
			oneD = False
		else:
			c = c.reshape(-1,1)
			oneD = True

		V_coordinate = [self.vander(X[:,k], self.degree) for k in range(self.dim)]
		out = np.zeros((M, c.shape[1]))	
		for j, alpha in enumerate(self.indices):

			# If we have a non-zero coefficient
			if np.max(np.abs(c[j,:])) > 0.:
				col = np.ones(M)
				for ell in range(self.dim):
					col *= V_coordinate[ell][:,alpha[ell]]

				for k in range(c.shape[1]):
					out[:,k] += c[j,k]*col
		if oneD:
			out = out.flatten()
		return out

	def DV(self, X):
		r""" Column-wise derivative of the Vandermonde matrix

		Given points :math:`\mathbf x_i \in \mathbb{R}^n`, 
		this creates the Vandermonde-like matrix whose entries
		correspond to the derivatives of each of basis elements;
		i.e., 

		.. math::

			[\mathbf{V}]_{i,j} = \left. \frac{\partial}{\partial x_k} \psi_j(\mathbf{x}) 
				\right|_{\mathbf{x} = \mathbf{x}_i}.

		Parameters
		----------
		X: array-like (M, n)
			Points at which to evaluate the basis at where :code:`X[i]` is one such point in 
			:math:`\mathbf{R}^n`.

		Returns
		-------
		Vp: np.array (M, N, n)
			Derivative of Vandermonde matrix where :code:`Vp[i,j,:]`
			is the gradient of :code:`V[i,j]`. 
		"""
		X = X.reshape(-1, self.dim)
		X = self._scale(np.array(X))
		M = X.shape[0]
		V_coordinate = [self.vander(X[:,k], self.degree) for k in range(self.dim)]
		
		N = len(self.indices)
		DV = np.ones((M, N, self.dim), dtype = X.dtype)

		try:
			dscale = self._dscale()
		except NotImplementedError:
			dscale = np.ones(X.shape[1])	


		for k in range(self.dim):
			for j, alpha in enumerate(self.indices):
				for q in range(self.dim):
					if q == k:
						DV[:,j,k] *= np.dot(V_coordinate[q][:,0:-1], self.Dmat[alpha[q],:])
					else:
						DV[:,j,k] *= V_coordinate[q][:,alpha[q]]
			# Correct for transform
			DV[:,:,k] *= dscale[k] 		

		return DV
	
	def DDV(self, X):
		r""" Column-wise second derivative of the Vandermonde matrix

		Given points :math:`\mathbf x_i \in \mathbb{R}^n`, 
		this creates the Vandermonde-like matrix whose entries
		correspond to the derivatives of each of basis elements;
		i.e., 

		.. math::

			[\mathbf{V}]_{i,j} = \left. \frac{\partial^2}{\partial x_k\partial x_\ell} \psi_j(\mathbf{x}) 
				\right|_{\mathbf{x} = \mathbf{x}_i}.

		Parameters
		----------
		X: array-like (M, n)
			Points at which to evaluate the basis at where :code:`X[i]` is one such point in 
			:math:`\mathbf{R}^m`.

		Returns
		-------
		Vpp: np.array (M, N, n, n)
			Second derivative of Vandermonde matrix where :code:`Vpp[i,j,:,:]`
			is the Hessian of :code:`V[i,j]`. 
		"""
		X = X.reshape(-1, self.dim)
		X = self._scale(np.array(X))
		M = X.shape[0]
		V_coordinate = [self.vander(X[:,k], self.degree) for k in range(self.dim)]
		
		N = len(self.indices)
		DDV = np.ones((M, N, self.dim, self.dim), dtype = X.dtype)

		try:
			dscale = self._dscale()
		except NotImplementedError:
			dscale = np.ones(X.shape[1])	


		for k in range(self.dim):
			for ell in range(k, self.dim):
				for j, alpha in enumerate(self.indices):
					for q in range(self.dim):
						if q == k == ell:
							# We need the second derivative
							eq = np.zeros(self.degree+1)
							eq[alpha[q]] = 1.
							der2 = self.polyder(eq, 2)
							DDV[:,j,k,ell] *= V_coordinate[q][:,0:len(der2)].dot(der2)
						elif q == k or q == ell:
							DDV[:,j,k,ell] *= np.dot(V_coordinate[q][:,0:-1], self.Dmat[alpha[q],:])
						else:
							DDV[:,j,k,ell] *= V_coordinate[q][:,alpha[q]]

				# Correct for transform
				DDV[:,:,k, ell] *= dscale[k]*dscale[ell]
				DDV[:,:,ell, k] = DDV[:,:,k, ell]
		return DDV

	def roots(self, coef):
		if self.dim > 1:
			raise NotImplementedError
		r = self.polyroots(coef)
		return r*(self._ub[0] - self._lb[0])/2.0 + (self._ub[0] + self._lb[0])/2.
		 

class MonomialTensorBasis(PolynomialTensorBasis):
	"""A tensor product basis of bounded total degree built from the monomials"""
	def __init__(self, *args, **kwargs):
		self.vander = np.polynomial.polynomial.polyvander
		self.polyder = np.polynomial.polynomial.polyder
		self.polyroots = np.polynomial.polynomial.polyroots
		PolynomialTensorBasis.__init__(self, *args, **kwargs)	


	
class LegendreTensorBasis(PolynomialTensorBasis):
	"""A tensor product basis of bounded total degree built from the Legendre polynomials

	"""
	def __init__(self, *args, **kwargs):
		self.vander = legvander
		self.polyder = legder
		self.polyroots = legroots
		PolynomialTensorBasis.__init__(self, *args, **kwargs)	

class ChebyshevTensorBasis(PolynomialTensorBasis):
	"""A tensor product basis of bounded total degree built from the Chebyshev polynomials
	
	"""
	def __init__(self, *args, **kwargs):
		self.vander = chebvander
		self.polyder = chebder
		self.polyroots = chebroots
		PolynomialTensorBasis.__init__(self, *args, **kwargs)	
	
	

class LaguerreTensorBasis(PolynomialTensorBasis):
	"""A tensor product basis of bounded total degree built from the Laguerre polynomials

	"""
	def __init__(self, *args, **kwargs):
		self.vander = lagvander
		self.polyder = lagder
		self.polyroots = lagroots
		PolynomialTensorBasis.__init__(self, *args, **kwargs)	

class HermiteTensorBasis(PolynomialTensorBasis):
	"""A tensor product basis of bounded total degree built from the Hermite polynomials

	"""
	def __init__(self, *args, **kwargs):
		self.vander = hermvander
		self.polyder = hermder
		self.polyroots = hermroots
		PolynomialTensorBasis.__init__(self, *args, **kwargs)	

	def _set_scale(self, X):
		self._mean = np.mean(X, axis = 0)
		self._std = np.std(X, axis = 0)

	def _scale(self, X):
		try:
			return (X - self._mean[None,:])/self._std[None,:]/np.sqrt(2)
		except AttributeError:
			return X

	def _dscale(self):
		try:
			return 1./self._std/np.sqrt(2)
		except AttributeError:
			raise NotImplementedError

	def roots(self, coef):
		if self.dim > 1:
			raise NotImplementedError
		r = hermroots(coef)
		return r*self._std[0]*np.sqrt(2) + self._mean[0]


#TODO: Fix
class ArnoldiPolynomialBasis(Basis):
	r""" Construct a stable polynomial basis for arbitrary points using Vandermonde+Arnoldi
	"""
	def __init__(self, degree, X=None, dim=None):
		# self.X = np.copy(np.atleast_2d(X)) #X is shape M,dim where M is the number of points sampled
		# self.dim = dim #self.X.shape[1]
		if X is not None:
			self.X = np.atleast_2d(X)
			self.dim = self.X.shape[1]
		elif dim is not None:
			self.dim = int(dim)
			self.X = None
		self.degree = int(degree)
		self.indices = index_set(self.degree, self.dim)

		# self.Q, self.R = self.arnoldi()
	
	def __len__(self):
		return len(self.indices)
	
	def set_X(self,X) : 
		self.X = np.copy(np.atleast_2d(X))

	def _update_vec(self, ids):
		# Determine which column to multiply by
		diff = self.indices - ids
		# Here we pick the most recent column that is one off
		j = np.max(np.argwhere( (np.sum(np.abs(diff), axis = 1) <= 1) & (np.min(diff, axis = 1) == -1)))
		i = int(np.argwhere(diff[j] == -1))
		return i, j	

	def arnoldi(self, X):
		"""Apply the Arnoldi proceedure to build up columns of the Vandermonde matrix

		Args:
			X (np.ndarray): an (M,n) array of M interpolation points

		Returns:
			(np.ndarray, np.ndarray): Elements Q and R
				Q - An (M, n) array where the columns are an orthonormal basis of the Krylov subspace
				R - An (n, n) array where X is a basis on R, this is upper Hessenberg
				n is the length of the index set
		"""
		idx = self.indices
		M = X.shape[0]

		# Allocate memory for matrices
		Q = np.zeros((M, len(idx)))
		R = np.zeros((len(idx), len(idx)))
	
		# Generate columns of the Vandermonde matrix
		iteridx = enumerate(idx)
		# As the first column is the ones vector, we treat it as a special case
		next(iteridx)
		Q[:,0] = 1/np.sqrt(M)
		R[0,0] = np.sqrt(M)	

		# Now work on the remaining columns
		for k, ids in iteridx:
			i, j = self._update_vec(ids) 
			# Form new column
			q = X[:,i] * Q[:,j]
	
			for j in range(k):
				R[j,k] = Q[:,j].T @ q
				q -= R[j,k]*Q[:,j]
			
			# print('q: ', q)
			R[k,k] = np.linalg.norm(q)
			Q[:,k] = q/R[k,k] if R[k,k] != 0 else 1

		self.Q = Q 
		self.R = R 
		return Q, R

	def arnoldi_X(self, X):
		r""" Generate a Vandermonde matrix corresponding to a different set of points
		"""

		self.arnoldi(X)
		W = np.zeros((X.shape[0], len(self.indices)), dtype = X.dtype)

		iteridx = enumerate(self.indices)
		# As the first column is the ones vector, we treat it as a special case
		next(iteridx)
		W[:,0] = 1/self.R[0,0]

		# Now work on the remaining columns
		for k, ids in iteridx:
			i, j = self._update_vec(ids) 
			# Form new column
			w = X[:,i] * W[:,j]
	
			for j in range(k):
				w -= self.R[j,k]*W[:,j]
			
			W[:,k] = w/self.R[k,k] if self.R[k,k] != 0  else w

		return W
		

	def V(self, X = None):
		if X is None or np.array_equal(X, self.X):
			return self.Q
		else:
			return self.arnoldi_X(X)


	def DV(self, X = None):
		if X is None or np.array_equal(X, self.X):
			X = self.X
			V = self.Q
		else:
			V = self.arnoldi_X(X)

		M = X.shape[0]
		N = self.Q.shape[1]
		n = self.X.shape[1]
		DV = np.zeros((M, N, n), dtype = self.Q.dtype)

		for ell in range(n):
			index_iterator = enumerate(self.indices)
			next(index_iterator)
			for k, ids in index_iterator:
				i, j = self._update_vec(ids)
				# Q[:,k] = X[:,i] * Q[:,j] - sum_s Q[:,s] * R[s, k]
				if i == ell:
					DV[:,k,ell] = V[:,j] + X[:,i] * DV[:,j,ell] - DV[:,0:k,ell] @ self.R[0:k,k] 
				else:
					DV[:,k,ell] = X[:,i] * DV[:,j,ell] - DV[:,0:k,ell] @ self.R[0:k,k] 
				DV[:,k,ell] /= self.R[k,k]	
		
		return DV

	def DDV(self, X = None):
		raise NotImplementedError

class PolynomialBasis(Basis) : 
	#polyder - derivative of the polynomial basis series
	#vander - vandermonde matrix of the polynomial basis series
	#polyroots - the roots of the of the polynomial basis series 

	def __init__(self, degree, X=None, dim=None) : 
		self.degree = int(degree)
		if X is not None:
			self.X = np.atleast_2d(X)
			self.dim = self.X.shape[1]
			self.set_scale(self.X)
		elif dim is not None:
			self.dim = int(dim)
			self.X = None

		self.indices = index_set(self.degree, self.dim).astype(int)
		self._build_Dmat()
		
	@abstractmethod
	def poly_basis_fn(self, interpolation_set: list[np.ndarray], row_num : int, col_num: int) -> np.float64: 
		raise NotImplementedError
	
	@abstractmethod
	def poly_basis_fn_deriv(self, coeff: np.ndarray) -> np.ndarray : 
		raise NotImplementedError
	
	@abstractmethod
	def poly_basis_roots(self, coeff: np.ndarray) -> np.ndarray : 
		raise NotImplementedError
	
	def assign_interpolation_set(self,X) : 
		self.X = X

	@abstractmethod
	def solve_no_cols(self,interpolation_set) -> int: 
		raise NotImplementedError	
	
	def _build_Dmat(self):
		""" Constructs the (scalar) derivative matrix
		"""
		self.Dmat = np.zeros( (self.degree+1, self.degree))
		I = np.eye(self.degree + 1)
		for j in range(self.degree + 1):
			self.Dmat[j,:] = self.poly_basis_fn_deriv(I[:,j])

	def set_scale(self, X):
		r""" Construct an affine transformation of the domain to improve the conditioning
		"""
		self._set_scale(np.array(X))

	def _set_scale(self, X):
		r""" default scaling to [-1,1]
		"""
		self._lb = np.min(X, axis = 0)
		self._ub = np.max(X, axis = 0)

	def _scale(self, X):
		r""" Apply the scaling to the input coordinates
		"""
		try:
			return 2*(X-self._lb[None,:])/(self._ub[None,:] - self._lb[None,:]) - 1
		except AttributeError:
			return X

	def _dscale(self):
		r""" returns the scaling associated with the scaling transform
		"""
		try:
			return (2./(self._ub - self._lb))
		except AttributeError:
			raise NotImplementedError

	#Construct a matrix Row by Row
	def V(self, interpolation_set: np.ndarray) -> np.ndarray :
		interpolation_set = interpolation_set.tolist()
		no_cols = self.solve_no_cols(interpolation_set)
		X = np.zeros((len(interpolation_set), no_cols))
		for i in range(len(interpolation_set)) :
			for j in range(no_cols) : 
				X[i,j] = self.poly_basis_fn(interpolation_set,i,j)	
		return X
	
	def DV(self, X):
		X = X.reshape(-1, self.dim)
		X = self._scale(np.array(X))
		M = X.shape[0]
		V_coordinate = [self.V(X[:,k]) for k in range(self.dim)]
		
		N = len(self.indices)
		DV = np.ones((M, N, self.dim), dtype = X.dtype)

		try:
			dscale = self._dscale()
		except NotImplementedError:
			dscale = np.ones(X.shape[1])	


		for k in range(self.dim):
			for j, alpha in enumerate(self.indices):
				for q in range(self.dim):
					if q == k:
						DV[:,j,k] *= np.dot(V_coordinate[q][:,0:-1], self.Dmat[alpha[q],:])
					else:
						DV[:,j,k] *= V_coordinate[q][:,alpha[q]]
			# Correct for transform
			DV[:,:,k] *= dscale[k] 		

		return DV
	
	def DDV(self, X):
		X = X.reshape(-1, self.dim)
		X = self._scale(np.array(X))
		M = X.shape[0]
		V_coordinate = [self.V(X[:,k]) for k in range(self.dim)]
		
		N = len(self.indices)
		DDV = np.ones((M, N, self.dim, self.dim), dtype = X.dtype)

		try:
			dscale = self._dscale()
		except NotImplementedError:
			dscale = np.ones(X.shape[1])	


		for k in range(self.dim):
			for ell in range(k, self.dim):
				for j, alpha in enumerate(self.indices):
					for q in range(self.dim):
						if q == k == ell:
							# We need the second derivative
							eq = np.zeros(self.degree+1)
							eq[alpha[q]] = 1.
							der2 = self.poly_basis_fn_deriv(self.poly_basis_fn_deriv(eq))
							print(V_coordinate[q][:,0:len(der2)].dtype)
							print(der2.dtype)
							DDV[:,j,k,ell] *= V_coordinate[q][:,0:len(der2)].dot(der2)
						elif q == k or q == ell:
							DDV[:,j,k,ell] *= np.dot(V_coordinate[q][:,0:-1], self.Dmat[alpha[q],:])
						else:
							DDV[:,j,k,ell] *= V_coordinate[q][:,alpha[q]]

				# Correct for transform
				DDV[:,:,k, ell] *= dscale[k]*dscale[ell]
				DDV[:,:,ell, k] = DDV[:,:,k, ell]
		return DDV

	def roots(self, coef):
		if self.dim > 1:
			raise NotImplementedError
		r = self.poly_basis_roots(coef)
		return r*(self._ub[0] - self._lb[0])/2.0 + (self._ub[0] + self._lb[0])/2.


class NaturalPolynomialBasis(PolynomialBasis) : 
	def __init__(self, degree, X, dim) : 
		super().__init__(degree, X, dim)

	def poly_basis_fn(self, interpolation_set, row_num, col_num) : 
		val = interpolation_set[row_num]
		p = len(val) if not isinstance(val, float) or isinstance(val, int) else 1
		
		# Define the full list of basis terms, up to max_degree
		basis_terms = []
	
		# Add the constant term 1
		basis_terms.append(lambda v: 1)
		
		# Add all first-order terms x_1, x_2, ..., x_p
		for i in range(p):
			if isinstance(val, float) or isinstance(val, int) :
				basis_terms.append(lambda v : v)
			else :
				basis_terms.append(lambda v : v[i])
		 
		
		# Add higher-order terms up to degree max_degree
		for degree in range(2, self.degree+1):
			for comb in combinations_with_replacement(range(p), degree):
				def term_func(v, comb=comb, degree=degree):
					result = 1
					for idx in comb:
						result *= v[idx]
					inner_result = result / factorial(degree)
					return inner_result
				def term_func_float(v, comb=comb, degree=degree):
					result = 1
					for idx in comb:
						result *= v
					inner_result = result / factorial(degree)
					return inner_result
				if isinstance(val, float) or isinstance(val, int) :
					basis_terms.append(term_func_float)
				else :
					basis_terms.append(term_func)
		# Evaluate the basis term at the given row (vector)
		res = basis_terms[col_num](val)
		# return basis_terms[col_num](val)
		return res
	
	def poly_basis_fn_deriv(self, coeff: np.ndarray) -> np.ndarray:
		"""Takes coefficients [a_0,a_1,...] that correspond to

		Args:
			coeff (np.ndarray): _description_

		Returns:
			np.ndarray: _description_
		"""
		n = self.degree
		 # Create symbolic variables for x1, x2, ..., xp
		x = sp.symbols('x')
		
		# Define the polynomial (1 + x)^n
		polynomial = (1 + x )**n
		
		# Compute the dot product of the coefficients and the expanded polynomial
		# The coefficients should be applied to the monomials of the expanded polynomial
		expanded_poly = sp.expand(polynomial)

		ordered_terms = expanded_poly.as_ordered_terms()

		modified_terms = []
		for idx, term in enumerate(ordered_terms):
			if idx < len(coeff):
				modified_term = int(coeff[idx]/comb(n,idx)) * term  # Multiply the term by the corresponding coefficient and get rid of the original coeff
				modified_terms.append(modified_term)
		
		dot_product = sum(modified_terms)
		
		derivative = sp.diff(dot_product, x)

		coefficients = []
		for term in derivative.as_ordered_terms():
			# Extract the coefficient of each term
			if term.has(x):
				coeff = term.coeff(x)
			else:
				coeff = term
			coefficients.append(float(coeff))
			coefficients = np.array(coefficients)

		return coefficients

	
	def poly_basis_roots(self, coeff: np.ndarray) -> np.ndarray:
		return np.roots(coeff)
	
	def solve_no_cols(self, interpolation_set):
		# val = interpolation_set[0]
		if isinstance(interpolation_set[0], float) or isinstance(interpolation_set[0], int) :
			sample_dim = 1
		else : 
			sample_dim = len(interpolation_set[0])
		
		return comb(sample_dim + self.degree, self.degree)	


class MonomialPolynomialBasis(PolynomialBasis) : 
	def __init__(self, degree, X, dim):
		super().__init__(degree, X, dim)

	def poly_basis_fn(self, interpolation_set, row_num, col_num):
		val = interpolation_set[row_num]
		#calculate the whole row,
		row = [1] 
		for exp in range(1, self.degree+1) : 
			row =  np.append(row, val**exp)

		return row[col_num]
	
	def poly_basis_fn_deriv(self, coeff: np.ndarray) -> np.ndarray:
		n = self.degree
		 # Create symbolic variables for x1, x2, ..., xp
		x = sp.symbols('x')
		
		# Define the polynomial (1 + x)^n
		polynomial = sum([coeff[i]*x**i for i in range(n+1)])
		
		derivative = sp.diff(polynomial, x)

		coefficients = []
		for term in derivative.as_ordered_terms():
			# Extract the coefficient of each term
			if term.has(x):
				coeff = term.coeff(x)
			else:
				coeff = term
			coefficients.append(float(coeff))
			coefficients = np.array(coefficients)

		return coefficients 

	def poly_basis_roots(self, coeff: np.ndarray) -> np.ndarray:
		return np.roots(coeff) 
	
	def solve_no_cols(self, interpolation_set):
		val = interpolation_set[0]
		return 1 + (len(val)*self.degree)
	
class LagrangePolynomialBasis(PolynomialBasis) : 
	def __init__(self, degree, X, dim) : 
		super().__init__(degree, X, dim)

	#lagrange polynomical function for each element in the matrix
	def poly_basis_fn(self, interpolation_set, row_num, col_num) :
		current_val = interpolation_set[row_num]
		denom_val = interpolation_set[col_num]

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
	
	#TODO: Implement the derivative function
	def DV(self, X = None):
		if X is None or np.array_equal(X, self.X):
			X = self.X
			V = self.Q
		else:
			V = self.V(X)

		M = X.shape[0]
		N = self.Q.shape[1]
		n = self.X.shape[1]
		DV = np.zeros((M, N, n), dtype = self.Q.dtype)

		for ell in range(n):
			index_iterator = enumerate(self.indices)
			next(index_iterator)
			for k, ids in index_iterator:
				i, j = self._update_vec(ids)
				# Q[:,k] = X[:,i] * Q[:,j] - sum_s Q[:,s] * R[s, k]
				if i == ell:
					DV[:,k,ell] = V[:,j] + X[:,i] * DV[:,j,ell] - DV[:,0:k,ell] @ self.R[0:k,k] 
				else:
					DV[:,k,ell] = X[:,i] * DV[:,j,ell] - DV[:,0:k,ell] @ self.R[0:k,k] 
				DV[:,k,ell] /= self.R[k,k]	
		
		return DV

	def DDV(self, X = None):
		raise NotImplementedError
	
	def poly_basis_roots(self, coeff: np.ndarray) -> np.ndarray:
		return np.roots(coeff)
	
	def solve_no_cols(self, interpolation_set):
		return len(interpolation_set)
	
class NFPPolynomialBasis(PolynomialBasis) : 
	def __init__(self, degree, X, dim) : 
		super().__init__(degree, X, dim) 

	def poly_basis_fn(self, interpolation_set, row_num, col_num) :
		if col_num == 0 : 
			return 1
		val = np.array(interpolation_set[row_num])
		res = np.prod([(val-np.array(a)) for a in interpolation_set[:col_num]])
		return float(res)
	
	#TODO: Implement the derivative function
	def DV(self, X = None):
		if X is None or np.array_equal(X, self.X):
			X = self.X
			V = self.Q
		else:
			V = self.arnoldi_X(X)

		M = X.shape[0]
		N = self.Q.shape[1]
		n = self.X.shape[1]
		DV = np.zeros((M, N, n), dtype = self.Q.dtype)

		for ell in range(n):
			index_iterator = enumerate(self.indices)
			next(index_iterator)
			for k, ids in index_iterator:
				i, j = self._update_vec(ids)
				# Q[:,k] = X[:,i] * Q[:,j] - sum_s Q[:,s] * R[s, k]
				if i == ell:
					DV[:,k,ell] = V[:,j] + X[:,i] * DV[:,j,ell] - DV[:,0:k,ell] @ self.R[0:k,k] 
				else:
					DV[:,k,ell] = X[:,i] * DV[:,j,ell] - DV[:,0:k,ell] @ self.R[0:k,k] 
				DV[:,k,ell] /= self.R[k,k]	
		
		return DV

	def DDV(self, X = None):
		raise NotImplementedError

	def poly_basis_roots(self, coeff: np.ndarray) -> np.ndarray:
		return np.roots(coeff)
	
	def solve_no_cols(self, interpolation_set):
		return len(interpolation_set)
	