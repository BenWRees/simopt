# Subspace based dimension reduction techniques
from __future__ import division, print_function

import os
# Necessary for running in headless enviornoments
if 'DISPLAY' not in os.environ:
	import matplotlib
	matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


import numpy as np
import scipy.linalg
import scipy.optimize
import scipy.sparse

from matplotlib.path import Path
from copy import deepcopy

import cvxpy as cp

from simopt.base import (Problem, Solution)

from ..active_subspaces.polyridge import * 

__all__ = ['SubspaceBasedDimensionReduction',
	'ActiveSubspace', 
	]


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


#provides a class for dumping tab separated files for PGF
from matplotlib.path import Path
from copy import deepcopy

class PGF:
	def __init__(self):
		self.column_names = []
		self.columns = []

	def add(self, name, column):
		if len(self.columns) > 1:
			assert len(self.columns[0]) == len(column)

		self.columns.append(deepcopy(column))
		self.column_names.append(name)

	def keys(self):
		return self.column_names

	def __getitem__(self, key):
		i = self.column_names.index(key)
		return self.columns[i]

	def write(self, filename):
		f = open(filename,'w')

		for name in self.column_names:
			f.write(name + '\t')
		f.write("\n")		

		for j in range(len(self.columns[0])):
			for col in self.columns:
				f.write("{}\t".format(float(col[j])))
			f.write("\n")

		f.close()

	def read(self, filename):
		with open(filename,'r') as f:
			for i, line in enumerate(f):
				# Remove the newline and trailing tab if present
				line = line.replace('\t\n','').replace('\n','')
				if i == 0:
					self.column_names = line.split('\t')
					self.columns = [ [] for name in self.column_names]
				else:
					cols = line.split('\t')
					for j, col in enumerate(cols):
						self.columns[j].append(float(col))


def save_contour(fname, cs, fmt = 'matlab', simplify = 1e-3, **kwargs):
	""" Save a contour plot to a file for pgfplots

	Additional arguments are passed to iter_segements
	Important, simplify = True will remove invisible points
	"""

	def write_path_matlab(fout, x_vec, y_vec, z):
		# Now dump this data back out
		# Header is level followed by number of rows
		fout.write('%15.15e\t%15d\n' % (z, len(x_vec)))
		for x, y in zip(x_vec, y_vec):
			fout.write("%15.15e\t%15.15e\n" % (x,y))

	def write_path_prepared(fout, x_vec, y_vec, z):
		fout.write("%15.15e\t%15.15e\t%15.15e\n" % (x_vec,y_vec,z))
		fout.write("\t\t\t\n")

	if fmt == 'matlab':
		write_path = write_path_matlab
	elif fmt == 'prepared':
		write_path = write_path_prepared
	else:
		raise NotImplementedError

	with open(fname, 'w') as fout:
		for col, z in zip(cs.collections, cs.levels):
			for path in col.get_paths():
				path.simplify_threshold = simplify
				x_vec = []
				y_vec = []
				for i, ((x,y), code) in enumerate(path.iter_segments(simplify = True)):
					if code == Path.MOVETO:
						if len(x_vec) !=0:
							write_path(fout, x_vec, y_vec, z)
							x_vec = []
							y_vec = []
						x_vec.append(x)
						y_vec.append(y)
					
					elif code == Path.LINETO:
						x_vec.append(x)
						y_vec.append(y)

					elif code == Path.CLOSEPOLY:
						x_vec.append(x_vec[0])
						y_vec.append(y_vec[0])
					else:
						print("received code", code)

				write_path(fout, x_vec, y_vec, z)



class SubspaceBasedDimensionReduction(object):
	r""" Abstract base class for Subspace-Based Dimension Reduction

	Given a function :math:`f : \mathcal{D} \to \mathbb{R}`, 
	subspace-based dimension reduction identifies a subspace, 
	described by a matrix :math:`\mathbf{U} \in \mathbb{R}^{m\times n}`
	with orthonormal columns for some :math:`n \le m`.

	"""
	@property
	def U(self):
		""" A matrix defining the 'important' directions

		Returns
		-------
		np.ndarray (m, n):
			Matrix with orthonormal columns defining the important directions in decreasing order
			of precidence.
		"""
		raise NotImplementedError


	def shadow_plot(self, X = None, fX = None, dim = 1, U = None, ax = 'auto', pgfname = None):
		r""" Draw a shadow plot


		Parameters
		----------
		X: array-like (N,m)
			Input coordinates for function samples
		fX: array-like (N,)
			Values of function at sample points
		dim: int, [1,2]
			Dimension of shadow plot
		U: array-like (?,m); optional
			Subspace onto which to project the data; defaults to the subspace identifed by this class
		ax: 'auto', matplotlib.pyplot.axis, or None
			Axis on which to draw the shadow plot

		Returns
		-------
		ax: matplotlib.pyplot.axis
			Axis on which the plot is drawn
		"""

		if ax == 'auto':
			if dim == 1:
				fig, ax = plt.subplots(figsize = (6,6))
			else:
				# Hack so that plot is approximately square after adding colorbar 
				fig, ax = plt.subplots(figsize = (7.5,6))
	
		if X is None:
			X = self.X
	
		# Check dimensions
		X = np.atleast_2d(X)
		assert X.shape[1] == len(self), "Samples do not match dimension of space"	

		if U is None:
			U = self.U
		else:
			if len(U.shape) == 1:
				U = U.reshape(len(self),1)
			else:
				assert U.shape[0] == len(self), "Dimensions do not match"

		
		if dim == 1:
			if ax is not None and fX is not None and isinstance(ax, Axes):
				ax.plot(X.dot(U[:,0]), fX, 'k.')
				ax.set_xlabel(r'active coordinate $\mathbf{u}^\top \mathbf{x}$')
				ax.set_ylabel(r'$f(\mathbf{x})$')

			if pgfname is not None:
				pgf = PGF()
				pgf.add('y', X.dot(U[:,0]))
				pgf.add('fX', fX)
				pgf.write(pgfname)

		elif dim == 2:
			Y = U[:,0:2].T.dot(X.T).T
			
			if ax is not None and fX is not None and isinstance(ax, Axes):
				sc = ax.scatter(Y[:,0], Y[:,1], c = fX.flatten(), s = 3)
				ax.set_xlabel(r'active coordinate 1 $\mathbf{u}_1^\top \mathbf{x}$')
				ax.set_ylabel(r'active coordinate 2 $\mathbf{u}_2^\top \mathbf{x}$')

				plt.colorbar(sc).set_label('f(x)')
			
			if pgfname is not None and fX is not None:
				pgf = PGF()
				pgf.add('y1', Y[:,0])
				pgf.add('y2', Y[:,1])
				pgf.add('fX', fX.flatten())
				pgf.write(pgfname)
 
		else:
			raise NotImplementedError		

		return ax

	def shadow_envelope(self, X, fX, ax = None, ngrid = None, pgfname = None, verbose = True, U = None, **kwargs):
		r""" Draw a 1-d shadow plot of a large number of function samples

		Returns
		-------
		y: np.ndarray
			Projected coordinates
		lb: np.ndarray
			piecewise linear lower bound values
		ub: np.ndarray
			piecewise linear upper bound values
		"""
		if U is None:
			U = self.U[:,0]		
		else:
			if len(U.shape) > 1:
				U = U[:,0]

		# Since this is for plotting purposes, we reduce accuracy to 3 digits	
		solver_kwargs = {'verbose': verbose, 'solver': 'OSQP', 'eps_abs': 1e-3, 'eps_rel': 1e-3}				

		X = np.array(X)
		fX = np.array(fX)
		assert len(X) == len(fX), "Number of inputs did not match number of outputs"
		if len(fX.shape) > 1:
			fX = fX.flatten()
			assert len(fX) == len(X), "Expected fX to be a vector"

		y = X.dot(U)
		if ngrid is None:
			# Determine the minimum number of bins
			ngrid = 25
			while True:
				yy = np.linspace(np.min(y), np.max(y), ngrid)
				h = yy[1] - yy[0]	
				if ngrid == 3:
					break 
				# Make sure we have at least two entries in every bin:
				items, counts = np.unique(np.floor( (y - yy[0])/h), return_counts = True)
				# We ignore the last count of the bins as that is the right endpoint and will only ever have one
				if (np.min(counts[:-1]) >= 5) and len(items) == ngrid:
					break
				else:
					ngrid -= 1
		else:
			yy = np.linspace(np.min(y), np.max(y), ngrid)
			h = yy[1] - yy[0]

		h = float(h)

		# Build the piecewise linear interpolation matrix
		j = np.floor( (y - yy[0])/h ).astype(np.integer)
		row = []
		col = []
		val = []

		# Points not at the right endpoint
		row += np.arange(len(y)).tolist()
		col += j.tolist()
		val += ((  (yy[0]+ (j+1)*h) - y )/h).tolist()

		# Points not at the right endpoint
		I = (j != len(yy) - 1)
		row += np.argwhere(I).flatten().tolist()
		col += (j[I]+1).tolist()
		val += ( (y[I] - (yy[0] + j[I]*h)  )/h).tolist()

		A = scipy.sparse.coo_matrix((val, (row, col)), shape = (len(y), len(yy)))
		A = cp.Constant(A)
		ub = cp.Variable(len(yy))
		#ub0 = [ max(max(fX[j == i]), max(fX[j== i+1]))  for i in np.arange(0,ngrid-1)] +[max(fX[j == ngrid - 1])]
		#ub.value = np.array(ub0).flatten()
		prob = cp.Problem(cp.Minimize(cp.sum(ub)), [A*ub >= fX.flatten()])
		prob.solve(**solver_kwargs)
		ub = ub.value
		
		lb = cp.Variable(len(yy))
		#lb0 = [ min(min(fX[j == i]), min(fX[j== i+1]))  for i in np.arange(0,ngrid-1)] +[min(fX[j == ngrid - 1])]
		#lb.value = np.array(lb0).flatten()
		prob = cp.Problem(cp.Maximize(cp.sum(lb)), [A*lb <= fX.flatten()])
		prob.solve(**solver_kwargs)
		lb = lb.value

		if ax is not None:
			ax.fill_between(yy, lb, ub, **kwargs) 

		if pgfname is not None:
			pgf = PGF()
			pgf.add('y', yy)
			pgf.add('lb', lb)
			pgf.add('ub', ub)	
			pgf.write(pgfname)
		
		return y, lb, ub


	def _init_dim(self, X = None, grads = None):
		if X is not None and len(X) > 0:
			self._dimension = len(X[0])
		elif grads is not None:
			self._dimension = len(grads[0])
		else:
			raise Exception("Could not determine dimension of ambient space")


	def __len__(self):
		return self._dimension

	@property
	def X(self):
		return np.zeros((0,len(self)))
	
	@property
	def fX(self):
		return np.zeros((0,len(self)))

	@property
	def grads(self):
		return np.zeros((0,len(self)))

	def _fix_subspace_signs(self, U, X = None, fX = None, grads = None):
		r""" Orient the subspace so that the average slope is positive

		Since subspaces have no associated direction (they are invariant to a sign flip)
		here we fix the sign such that the function is increasing on average along the direction
		u_i.  This approach uses either gradient or sample information, with a preference for
		gradient information if it is availible.
		"""
		if grads is not None and len(grads) > 0:
			return self._fix_subspace_signs_grads(U, grads)
		else:
			return self._fix_subspace_signs_samps(U, X, fX)	

	def _fix_subspace_signs_samps(self, U, X, fX):
		sgn = np.zeros(len(U[0]))
		for k in range(len(U[0])):
			for i in range(len(X)):
				for j in range(i+1, len(X)):
					denom = U[:,k] @ (X[i] - X[j])
					if np.abs(denom) > 0:
						sgn[k] += (fX[i] - fX[j])/denom

		# If the sign is zero, keep the current orientation
		sgn[sgn == 0] = 1
		return U.dot(np.diag(np.sign(sgn)))	

	def _fix_subspace_signs_grads(self, U, grads):
		return U.dot(np.diag(np.sign(np.mean(grads.dot(U), axis = 0))))

	
	def approximate_lipschitz(self, X = None, fX = None, grads = None,  dim = None):
		r""" Approximate the Lipschitz matrix on the low-dimensional subspace
		"""
		raise NotImplementedError
 
class ActiveSubspace(SubspaceBasedDimensionReduction):
	r"""Computes the active subspace gradient samples

	Given the function :math:`f:\mathcal{D} \to \mathbb{R}`,
	the active subspace is defined as the eigenvectors corresponding to the 
	largest eigenvalues of the average outer-product of gradients:

	.. math::

		\mathbf{C} := \int_{\mathbf{x}\in \mathcal{D}} \nabla f(\mathbf{x}) \nabla f(\mathbf{x})^\top \  \mathrm{d}\mathbf{x}
		\in \mathbb{R}^{m\times m}.

	By default, this class assumes that we are provided with gradients
	evaluated at random samples over the domain and estimates the matrix :math:`\mathbf{C}`
	using Monte-Carlo integration. However, if provided a weight corresponding to a quadrature rule,
	this will be used instead to approximate this matrix; i.e.,
		
	.. math::

		\mathbf{C} \approx \sum_{i=1}^N w_i \nabla f(\mathbf{x}_i) \nabla f(\mathbf{x}_i)^\top.

	"""
	def __init__(self):
		self._U = None
		self._s = None

	def fit(self, grads, weights = None):
		r""" Find the active subspace

		Parameters
		----------
		grads: array-like (N,m)
			Gradient samples of function (tacitly assumed to be uniform on the domain
			or from a quadrature rule with corresponding weight).
		weights: array-like (N,), optional
			Weights corresponding to a quadrature rule associated with the samples of the gradient.

		"""
		self._init_dim(grads = grads)

		self._grads = np.array(grads).reshape(-1,len(self))
		N = len(self._grads)
		if weights is None:
			weights = np.ones(N)/N
			
		self._weights = np.array(weights)
		self._U, self._s, VT = scipy.linalg.svd(np.sqrt(self._weights)*self._grads.T, full_matrices=False) #Added full_matrices
		# Pad s with zeros if we don't have as many gradient samples as dimension of the space
		self._s = np.hstack([self._s, np.zeros(self._dimension - len(self._s))])
		self._C = self._U.T @ np.diag(self._s**2) @ self._U #switched transposed _U around

		# Fix +/- scaling so average gradient is positive	
		self._U = self._fix_subspace_signs_grads(self._U, self._grads)	


	def fit(self, grads: np.ndarray, subspace_dimension: int) -> None:
		"""Calculate the active subspace matrix by 

		Args:
			X (np.ndarray): of shape (M,n) that represents M n-dimensional sample points
			fX (np.ndarray): The function values of the M sample points in the shape (M,1)
			subspace_dimension (int): The dimension of the subspace 

		Sets the matrix self._U which will be a (n,subspace_dimension) numpy matrix
		"""
		#construct the covariance matrix  
		cov_matrix = self._covariance_matrix(grads) 
		
		#take the eigendecomposition of the covariance matrix
		_,W = np.linalg.eigh(cov_matrix)
		#sort the eigenvalues from increasing size with the eigenvectors 
		#* Doesn't need sorting, we just take elements from the back
		W = W[::-1]
		#take the first subspace_dimension eiegenvectors from the sorted list and stack them as columns 
		self._U = W[:, :subspace_dimension]


	def _covariance_matrix(self, grads: np.ndarray) -> np.ndarray : 
		"""Construct a covariance matrix using interpolation

		Args:
			grads (np.ndarray): (N,m) matrix of N grad evaluations of the function

		Returns:
			np.ndarray: The (m,m) covariance matrix
		"""
		M = grads.shape[0]
		return (grads.T @ grads)/M

		


	#TODO: This function needs rewriting to instead sample the simopt problem: 
	#	- pass a simopt problem and a matrix of sample points as an argument instead 
	# 	- sample the problem at different 
	def fit_function(self, problem, delta, current_solution, N_gradients):
		r""" Automatically estimate active subspace using a quadrature rule

		Parameters
		----------
		problem: simopt.Problem
			The problem for which gradients can be sampled from
		delta: float 
			The trust-region radius 
		current_solution: simopt.Solution
			the incumbent solution of the solver
		N_gradients: int
			Maximum number of gradient samples to use
		
		"""
		# X, w = fun.domain.quadrature_rule(N_gradients)
		X = self.samples(delta, problem, current_solution, N_gradients)
		grads, expended_budget = self.grad(problem, X)
		self.fit(grads)

		return expended_budget
			

	def samples(self, delta: float, problem: Problem, current_solution: Solution, N_samples: int) -> list[Solution] : 
		"""Samples N_samples from the trust_region_radius

		Args:
			delta (float): The current Trust-Region Radius
			problem (simopt.Problem): The current sim-opt problem being solved
			current_solution (simopt.Solution): The current incumbent solution
			N_samples (int): The number of samples being taken 

		Returns:
			list: a list of length N_samples of different solutions in the trust_region 
		"""
		x_k = current_solution.x
		n = len(x_k)
		init_col = np.array(x_k).reshape((n,1))
		Y = [current_solution]
		for _ in range(N_samples-1) : 
			#uniformly sample a vector a of shape (n,1) that has |a-x_k|<delta
			while True : 
				sample = np.random.uniform(-1,1,(n,1))
				if np.linalg.norm(sample) <= 1 : 
					break 
			sample = delta * sample + init_col #scale and translate to TR region 
			sample = Solution(tuple(sample), problem)
			Y.append(sample)


		return Y  

	def grad(self, problem: Problem, samples: list[Solution]) -> tuple[np.ndarray, int] : 
		"""
			get grads at locations of different samples

		Args:
			problem (simopt.Problem): The problem being worked on 
			samples (list[Solution]): A list of N_sample solutions at different points in the trust-region

		Returns:
			np.ndarray: A (d,n) matrix of samples where d is the problem.dim
			int: The increase in expended_budget 
		"""
		expended_budget = 0
		grads = np.zeros((problem.dim,1))
		if problem.gradient_available : 
			for idx,sample in enumerate(samples) : 
				problem.simulate(sample, 1)
				expended_budget += 1
				grads[:, idx] = sample.objectives_gradients_mean 
		else : 
			for idx,sample in enumerate(samples) : 
				grad = self.finite_diff(sample, problem) 
				expended_budget += 2 * problem.dim
				grads[:, idx] = grad  
		return grads, expended_budget

	def finite_diff(self, solution: Solution, problem: Problem) -> np.ndarray:
		""" Solve a Finite Difference Approximation 

		Args:
			solution (np.ndarray): current solution value being approximated
			problem (Problem): The current sim-opt problem being solved 

		Returns:
			np.ndarray: A (d,1) matrix of the gradient approximation of the problem at the solution.
		"""
		alpha = 1e-2
		lower_bound = problem.lower_bounds
		upper_bound = problem.upper_bounds
		# grads = np.zeros((problem.dim,r)) #Take r gradient approximations
		problem.simulate(solution,1)
		
		new_x = solution.x
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
			FnPlusMinus[i, 2] = min(steph1, steph2)
			x1[i] = x1[i] + FnPlusMinus[i, 2]
			x2[i] = x2[i] - FnPlusMinus[i, 2]

			fn1, fn2 = 0,0 
			x1_solution = Solution(tuple(x1), problem)
			problem.simulate_up_to([x1_solution], 1)
			fn1 = -1 * problem.minmax[0] * x1_solution.objectives_mean
			# First column is f(x+h,y).
			FnPlusMinus[i, 0] = fn1

			x2_solution = Solution(tuple(x2), problem)
			problem.simulate_up_to([x2_solution], 1)
			fn2 = -1 * problem.minmax[0] * x2_solution.objectives_mean
			# Second column is f(x-h,y).
			FnPlusMinus[i, 1] = fn2

			# Calculate gradient.
			grad[i] = (fn1 - fn2) / (2 * FnPlusMinus[i, 2])
		
		return grad

	@property
	def U(self):
		# return np.copy(self._U)
		return self._U #this should still return a shallow copy 

	@property
	def C(self):
		return self._C

	@property
	def singvals(self):
		return self._s

	# TODO: Plot of eigenvalues (with optional boostrapped estimate)

	# TODO: Plot of eigenvector angles with bootstrapped replicates.
