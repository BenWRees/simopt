#type: ignore 
""" Polynomial functions 
	This has been rewritten to work inside the simopt library 

"""
# (c) 2019 Jeffrey M. Hokanson (jeffrey@hokanson.us)

from __future__ import print_function, division
import numpy as np
import cvxpy as cp
import scipy.linalg
from copy import copy
import textwrap
import inspect
import cloudpickle

from .basis import *


__all__ = ['PolynomialFunction', 'PolynomialApproximation', 'Function', 'BaseFunction']


#from .domains import Domain

""" misc. utilities and defintions
"""

def merge(x, y):
	z = x.copy()
	z.update(y)
	return z


def check_sample_inputs(X, fX, grads):
	if X is not None and fX is not None:
		X = np.array(X)
		fX = np.array(fX).flatten()
		assert len(X) == len(fX), "Number of samples doesn't match number of evaluations"
	else:
		X = None
		fX = None

	if grads is not None:
		grads = np.array(grads)
		if X is not None:
			assert X.shape[1] == grads.shape[1], "Dimensions of gradients doesn't match dimension of samples"

	if X is None:
		X = np.zeros((0, grads.shape[1]))
		fX = np.zeros((0,))
	if grads is None:
		grads = np.zeros((0, X.shape[1]))
	return X, fX, grads	



# I've stopped using dill entirely because this issue prevents 
# me from loading modules inside functions
# https://github.com/uqfoundation/dill/issues/219	

class BaseFunction(object):
	r""" Abstract base class for functions

	"""
	def eval(self, X, **kwargs):
		return self.__call__(X, return_grad = False)

	def grad(self, X):
		return self.__call__(X, return_grad = True)[1]

	def hessian(self, X):
		raise NotImplementedError

	def __call__(self, X, return_grad = False, **kwargs):
		if return_grad:
			return self.eval(X, **kwargs), self.grad(X)
		else:
			return self.eval(X, **kwargs)

	def predict(self, X):
		r""" Alias of __call__ to match scikit learn API
		"""
		return self.__call__(X)


#TODO: This function needs changing as it is undergone in the simopt library 
class Function(BaseFunction):
	r"""Wrapper around function specifying the domain

	Provided a function :math:`f: \mathcal{D} \subset \mathbb{R}^m \to \mathbb{R}^d`,
	and a domain :math:`\mathcal{D}`, this class acts as a wrapper for both.
	The key contribution of this class is to provide access to 
	the function on the *normalized domain* :math:`\mathcal{D}_{\text{norm}}` 
	that is a subset of the :math:`[-1,1]^m` cube; i.e.,

	.. math::

		\mathcal{D}_{\text{norm}} \subset [-1,1]^m \subset \mathbb{R}^m.

	Parameters
	----------
	fun: function or list of functions
		Either a python function or a list of functions to evaluate
	domain: Domain
		The domain on which the function is posed	
	vectorized: bool, default: False
		If True, the functions are vectorized for use with numpy.
	kwargs: dict, default: empty
		Keyword arguments to pass to the functions when evaluating function
	dask_client: dask.distributed.Client
		Client to use for multiprocessing 
	"""

	def __init__(self, funs, domain, grads = None, fd_grad = None, vectorized = False, kwargs = {},
		dask_client = None, return_grad = False):

		self.dask_client = dask_client
		self.vectorized = vectorized
		self.kwargs = kwargs
		self.return_grad = return_grad
		
		if callable(funs):
			self._funs = [funs]
		else:
			self._funs = funs

		if grads is not None:
			if callable(grads):
				grads = [grads]
			assert len(grads) == len(self._funs), "Must provide the same number of functions and gradients"
			self._grads = grads
		else:
			self._grads = None

		if dask_client is not None:
			# Pickle the functions for later use when calling distributed code
			self._funs_pickle = []
			for fun in self._funs:	
				# A big problem is when functions are imported from another module
				# dill/cloudpickle will simply want to import these functions;
				# i.e., the function is stored by a referrence to the file in which it originated.
				# See discussion at https://github.com/uqfoundation/dill/issues/123
				
				# There are also otherways to handle this problem.  For example, 
				# dask.distributed allows you to ship python files to the workers
				# see: https://stackoverflow.com/a/39295372/
			
				# So we do something more sophisticated in order to pickle these functions.
			
				# (1) We bring the function into the local scope, evaluating the function definition
				# inside this loop.
				# Specifically, we run the code inside a custom scope in order to 
				# have the code load into dill/cloudpickle rather than passing around
				# as a reference.  The limited scope prevents this from overwriting any local functions

				# Get the code 
				code = inspect.getsource(fun)
				
				# Strip indentation 
				code = textwrap.dedent(code)
				
				# Execute code	
				scope = {}
				exec(code, scope, scope)
	
				# (2) We now pickle this function 
				# scope is a dictionary of functions, and the name allows us to specify which
				self._funs_pickle.append(cloudpickle.dumps(scope[fun.__name__]))

		self.domain_app = domain
		self.domain_norm = domain.normalized_domain()
		self.domain = self.domain_norm
		self.fd_grad = fd_grad

	def eval(self, X_norm, **kwargs):
		X_norm = np.atleast_1d(X_norm)
		X = self.domain_app.unnormalize(X_norm)

		kwargs = merge(self.kwargs, kwargs)

		if len(X.shape) == 1:
			x = X.flatten()
			return np.hstack([fun(x, **kwargs) for fun in self._funs]).flatten()

		elif len(X.shape) == 2:
			if self.vectorized:
				fX = [fun(X, **kwargs) for fun in self._funs]
				for fXi in fX:
					assert len(fXi) == X.shape[0], "Must provide an array with %d entires; got %d" % (X.shape[0], len(fXi) )
				
				# Reshape if necessary so concatention works
				for i, fXi in enumerate(fX):
					fXi = np.array(fXi)
					if len(fXi.shape) == 1:
						fX[i] = fXi.reshape(len(X),1)
				return np.hstack(fX)
			else:
				return np.vstack([ np.hstack([fun(x, **kwargs) for fun in self._funs]) for x in X])


	def eval_async(self, X_norm, **kwargs):
		r""" Evaluate the function asyncronously using dask.distributed
		"""
		assert self.dask_client is not None, "A dask_client must be specified on class initialization"
		
		kwargs = merge(self.kwargs, kwargs)

		X_norm = np.atleast_1d(X_norm)
		X = self.domain_app.unnormalize(X_norm)
		X = np.atleast_2d(X)

		def subcall(funs_pickle, x, **kwargs_):
			import cloudpickle
			funs = [cloudpickle.loads(fun) for fun in funs_pickle]
			return [fun(x, **kwargs_) for fun in funs]

		results = [self.dask_client.submit(subcall, self._funs_pickle, x, **kwargs) for x in X]	
		if len(X_norm.shape) == 1:
			return results[0]
		else:
			return results


	def _shape_grad(self, X, grads):
		r""" This expects a 3-dimensional array in format [x sample #, fun #, input dim #] 
		"""
		grads = np.array(grads)
		if len(X.shape) == 1 and grads.shape[1] == 1:
			return grads.reshape(len(self.domain))
		elif len(X.shape) == 1 and grads.shape[1] != 1:
			return grads.reshape(grads.shape[1], len(self.domain))
		elif grads.shape[1] == 1:
			return grads.reshape(len(X), len(self.domain))
		else:
			return grads

	def grad(self, X_norm, **kwargs):
		kwargs = merge(self.kwargs, kwargs)
		
		X_norm = np.atleast_1d(X_norm)

		# If we've asked to use a finite difference gradient
		if self.fd_grad:
			h = 1e-7
			grads = []
			for x in np.atleast_2d(X_norm):
				fx = self.eval(x)
				grad = np.zeros(x.shape)
				for i in range(len(x)):
					ei = np.zeros(x.shape)
					ei[i] = 1.
					grad[i] = (self.eval(x + h*ei, **kwargs) - fx)/h
				# This ensures the dimensions match expectation for 
				grads.append(np.atleast_2d(grad))
			
			return self._shape_grad(X_norm, grads)

		X = self.domain_app.unnormalize(X_norm)
		D = self.domain_app._unnormalize_der() 	
		
		# Return gradient if specified
		if self._grads is not None: 
			X = np.atleast_2d(X)
			
			if self.vectorized:
				# TODO: I don't think this will get dimensions quite right
				grads = np.array([ np.array(grad(X, **kwargs)) for grad in self._grads])
				grads = np.transpose(grads, (1,0,2))
			else:
				grads = np.array([ np.vstack([grad(x, **kwargs) for grad in self._grads]) for x in X])
		
			# Correct to apply to normalized domain
			grads = grads.dot(D.T)
			return self._shape_grad(X_norm, grads)

		# Try return_grad the function definition
		elif self.return_grad:
			X = np.atleast_2d(X)	
			
			if self.vectorized:
				grads = []
				for fun in self._funs:
					fXi, gradsi = fun(X, return_grad = True, **kwargs)
					grads.append(gradsi)
				grads = np.array([ np.atleast_2d(grad) for grad in grads])
			else:
				grads = []
				for x in X:
					grad = []
					for fun in self._funs:
						fxi, gradi = fun(x, return_grad = True, **kwargs)
						grad.append(gradi)
					grads.append(np.vstack(grad))
				grads = np.array(grads)	
			grads = grads.dot(D.T)
			return self._shape_grad(X_norm, grads)
		else:
			raise NotImplementedError("Gradient not defined and finite-difference approximation not enabled")


	def __call__(self, X_norm, return_grad = False, **kwargs):
		kwargs = merge(self.kwargs, kwargs)
		if not return_grad:
			return self.eval(X_norm, **kwargs)

		if self.return_grad:
			# If the function can return both the value and gradient simultaneously
			X = self.domain_app.unnormalize(X_norm)
			X = np.atleast_2d(X)
			D = self.domain_app._unnormalize_der() 	
			if self.vectorized:
				ret = [fun(X, return_grad = True, **kwargs) for fun in self._funs]
				fX = np.hstack([r[0] for r in ret])
				grads = np.concatenate([r[1].reshape(X.shape[0], -1, len(self.domain)) for r in ret], axis = 1)
			else:
				fX = []
				grads = []
				for x in X:
					fx = []
					g = []
					for fun in self._funs:
						fxi, gi = fun(x, return_grad = True, **kwargs)
						fx.append(fxi)
						g.append(gi)
					fX.append(np.hstack(fx))
					grads.append(np.vstack(g))

				fX = np.vstack(fX)
				grads = np.array(grads)
			
			grads = grads.dot(D.T)

			if len(X_norm.shape) == 1:
				fX = fX.flatten()
			grads = self._shape_grad(X_norm, grads)
			return fX, grads
		else:
			return self.eval(X_norm, **kwargs), self.grad(X_norm, **kwargs)					

	def call_async(self, X_norm, return_grad = False, **kwargs):
		r""" Calls the function in an async. manner
		
		This mainly exists to cleanly separate eval_async which *only* returns function values
		and this function, call_async, which can optionally return gradients, like __call__.
		"""
		kwargs = merge(self.kwargs, kwargs)
		return self.eval_async(X_norm, return_grad = return_grad, **kwargs)
	
#	def __get__(self, i):
#		"""Get a particular sub-function as another Function"""
#		raise NotImplemented






def linear_fit(A, b, norm = 2, bound = None):
	r""" solve the linear optimization problem subject to constraints
	"""
	assert norm in [1,2,np.inf], "Invalid norm specified"
	assert bound in [None, 'lower', 'upper'], "invalid bound specified"

	if norm == 2 and bound == None:
		return scipy.linalg.lstsq(A, b)[0]
	else:
		x = cp.Variable(A.shape[1])
		residual = x.__rmatmul__(A) - b
		if norm == 1:   	 obj = cp.norm1(residual)
		elif norm == 2: 	 obj = cp.norm(residual)
		elif norm == np.inf: obj = cp.norm_inf(residual)
		
		if bound == 'lower':
			constraint = [residual <= 0]
		elif bound == 'upper':
			constraint = [residual >= 0]
		else:
			constraint = []
		
		# Now actually solve the problem
		problem = cp.Problem(cp.Minimize(obj), constraint)
		problem.solve(feastol = 1e-10, reltol = 1e-8, abstol = 1e-8, solver = cp.ECOS)
		return x.value


class PolynomialFunction(BaseFunction):
	r""" A polynomial function in a Legendre basis 	


	Parameters
	----------
	dimension: int
		Input dimension
	degree: int
		Degree of polynomial
	coef: array-like
		Coefficients of polynomial	
	"""
	def __init__(self, basis, coef):
		self.basis = basis
		self.coef = np.array(coef)

	def roots(self):
		return self.basis.roots(self.coef)	
		

	def V(self, X):	
		return self.basis.V(X)

	def DV(self, X):
		return self.basis.DV(X)

	def DDV(self, X):
		return self.basis.DDV(X)

	def eval(self, X):
		if len(X.shape) == 1:
			return self.V(X.reshape(1,-1)).dot(self.coef).reshape(1)
		else:
			return self.V(X).dot(self.coef)

	def grad(self, X, coef=None):
		if coef is None : 
			coef = self.coef
		if len(X.shape) == 1:
			one_d = True
			X = X.reshape(1,-1)	
		else:
			one_d = False	
		
		DV = self.DV(X)
		# Compute gradient on projected space
		Df = np.tensordot(DV, coef, axes = (1,0))
		# Inflate back to whole space
		if one_d:
			return Df.reshape(X.shape[1])
		else:
			return Df

	def hessian(self, X):
		if len(X.shape) == 1:
			one_d = True
			X = X.reshape(1,-1)	
		else:
			one_d = False
	
		DDV = self.DDV(X)
		DDf = np.tensordot(DDV, self.coef, axes = (1,0))
		if one_d:
			return DDf.reshape(X.shape[1], X.shape[1])
		else:
			return DDf
	

#TODO: Change this to work with all the basis names 
class PolynomialApproximation(PolynomialFunction):
	r""" Construct a polynomial approximation

	Parameters
	----------
	degree: int
		Degree of polynomial
	basis: ['arnold', 'legendre', 'monomial', 'chebyshev', 'laguerre', 'hermite']
		Basis in which to express the polynomial
	norm: [1, 2, np.inf]
		Norm in which to find the approximation
	bound: [None, 'lower', 'upper']
		If None, construct approximation in the specified norm;
		if 'lower' or 'upper', additionally enforce the constraint that
		the approximation is below or above the measured samples	
	"""
	def __init__(self, degree, basis = 'legendre', norm = 2, bound = None):

		degree = int(degree)
		assert degree >= 0, "Degree must be positive"
		self.degree = degree

		assert basis in ['arnoldi', 'legendre', 'monomial', 'chebyshev', 'laguerre', 'hermite']
		self.basis_name = copy(basis)

		self.basis = None
		
		assert bound in [None, 'lower', 'upper']
		self.bound = bound

		assert norm in [1,2, np.inf]
		self.norm = norm
		

	def fit(self, X, fX):
		M, m = X.shape

		# Since we don't know the input dimension until we get the data, we initialize the basis here
		if self.basis_name == 'arnoldi':
			self.basis = ArnoldiPolynomialBasis(self.degree, X = X) 
		elif self.basis_name == 'legendre':
			self.basis = LegendreTensorBasis(self.degree, X = X) 
		elif self.basis_name == 'monomial':
			self.basis = MonomialTensorBasis(self.degree, X = X) 
		elif self.basis_name == 'chebyshev':
			self.basis = ChebyshevTensorBasis(self.degree, X = X) 
		elif self.basis_name == 'laguerre':
			self.basis = LaguerreTensorBasis(self.degree, X = X) 
		elif self.basis_name == 'hermite':
			self.basis = HermiteTensorBasis(self.degree, X = X) 

		# Construct Vandermonde matrix
		V = self.basis.V(X)

		self.coef = linear_fit(V, fX, norm = self.norm, bound = self.bound)

