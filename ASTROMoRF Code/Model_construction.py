"""
	The following functions define model Construction for ASTROMoRF
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
from numpy.polynomial.chebyshev import chebvander, chebder, chebroots
from scipy.optimize import minimize, NonlinearConstraint, linprog
from scipy import optimize
from scipy.special import factorial
from sklearn.metrics import r2_score
import scipy
import math


import geometry_improvement_test as geo
import design_set_test as design


from simopt.base import (
	Problem,
	Solution,
)

from simopt.experiment_base import instantiate_problem


def construct_model(
		k: int, problem: Problem, current_solution: Solution, delta: float, expended_budget: int, visited_pts: list[Solution]
		) -> tuple[
			Solution, float, callable, np.ndarray, list[float], int, list[Solution], list[Solution]
			]: 
	"""
		Builds a local approximation of the response surface within a trust region (defined as ||x-x_k||<=delta).
		The method fit recovers the local approximation given a design set of 2d+1 design points and a corresponding active subspace U of shape (n,d)
		That projects the n-dimensional design points to a d-dimensional subspace.

	Args:
		k (int): The solver's current iteration 
		problem (Problem): The SO problem with dimension n
		current_solution (Solution): The solvers incumbent solution
		delta (float): The current trust-region radius
		expended_budget (int): The current expended budget of the solver
		visited_pts (list[Solution]): The solutions already visited by the solver

	Returns:
		tuple[ Solution, float, callable, np.ndarray, list[float], int, list[Solution], list[Solution] ]:
			- The current solution of the SO solver 
			- The updated trust-region radius after fitting 
			- The local model as a function that takes a numpy vector of shape (n,1) and returns a float (this is given as model = lambda x : self.model_evaluate(x,U))
			- The final computed active subspace of the iteration of shape (n,d)
			- A list of the function estimates of the objective function at each of the final design points
			- The simulation budget expended so far by the solver
			- The list of solutions of the final design points 
			- The design points previously visited by the solver 
	"""
	d = 1 #self.factors['initial subspace dimension']
	init_S_full = geo.generate_set(problem, d, current_solution, delta)
	U, _ = np.linalg.qr(init_S_full.T)

	X, f_index = design.construct_interpolation_set(current_solution, problem, U, delta, k, visited_pts)

	# fX, current_solution, interpolation_solutions, visited_points_list, expended_budget = evaluate_interpolation_points(k, problem, current_solution, expended_budget, f_index, X, visited_points_list, delta_k)
	fX = np.vstack([np.linalg.norm(x)**2 for x in X]) 
	interpolation_solutions = [Solution(tuple(x), problem) for x in X]
	f_x = fX[0,0] # -1 * problem.minmax[0] * current_solution.objectives_mean

	fval = fX.flatten().tolist()


	U, model, X, fX, interpolation_sols, visited_pts, delta = fit(k, problem, X, fX, current_solution, f_x, delta, interpolation_solutions, visited_pts, U) 

	fval = fX.flatten().tolist()

	return current_solution, delta, model, U, fval, expended_budget, interpolation_solutions, visited_pts

def model_evaluate(x: np.ndarray, coef: np.ndarray, U: np.ndarray) -> float :
	"""
		Evaluates the local approximated model at a given design point.

	Args:
		x (np.ndarray): Design point to evaluate of shape (1,n)
		coef (np.ndarray): The coefficients of the local model of shape (q,1)
		U (np.ndarray): The active subspace matrix of shape (n,d)

	Returns:
		float: The evaluation of the model at x, given as m(U^Tx)
	"""     
	if len(x.shape) != 2 or x.shape[1] != 1 : 
		x = x.reshape(1,-1)

	if len(coef.shape) != 2 : 
		coef = coef.reshape(-1,1)

	#project design point 
	y = x @ U #(1,d) 

	#build vandermonde matrix of shape (1,q)
	V_matrix = V(y)


	#find evaluation: 
	res = V_matrix @ coef 
	return res.item()



def fit(
		k:int, problem:Problem, X:np.ndarray, fX:np.ndarray, current_solution:Solution, f_x: float, delta: float, interpolation_solutions:list[Solution], visited_pts: list[Solution], U0: np.ndarray
		) -> tuple[
			np.ndarray, callable, np.ndarray, np.ndarray, list[Solution], list[Solution], float
			] : 
	"""
		Fits the design set and evaluated points to a local model with recovered active subspace. 
		It undergoes a loop until the active subspace converges. 
		First, it improves the design set X and constructing an interpolation model until it can ensure the criticality step is satisfied. 
		Second, after fixing the model coefficients, it updates the active subspace through a variable projection scheme
	Args:
		k (int): the iteration number
		problem (Problem): The SO problem with dimension n
		X (np.ndarray): design set of shape (M,n)
		fX (np.ndarray): corresponding function estimates of design points of shape (M,1)
		current_solution (Solution): The current solution of the solver
		f_x (float): The estimated function value at the current solution
		delta (float): The current trust-region radius
		interpolation_solutions (list[Solution]): A list of the design points in the 
		visited_pts (list[Solution]): The list of solutions already visited by the solver
		U0 (np.ndarray): The initial estimate for the active subspace of shape (n,d)

	Returns:
		tuple[np.ndarray, callable, np.ndarray, np.ndarray, list[Solution], list[Solution], float ]:  
			- The final computed active subspace of the iteration of shape (n,d)
			- The local model as a function that takes a numpy vector of shape (n,1) and returns a float (this is given as model = lambda x : self.model_evaluate(x,coef, U))
			- The design set after going through fitting of shape (M,n)
			- The function estimates of the objective function at each of the final design points of shape (M,1)
			- The design points as solution objects 
			- The visited design points included those computed during solving
			- The updated trust-region radius.

	NOTES:
	TODO: Need to implement different cases where handling box constraints 
	"""
	n,d = U0.shape
	degree = 2 #self.factors['polynomial degree']

	# Orthogonalize just to make sure the starting value satisfies constraints	
	U0, R = np.linalg.qr(U0, mode = 'reduced') 
	U0 = np.dot(U0, np.diag(np.sign(np.diag(R))))

	prev_U = np.zeros(U0.shape)
	U = np.copy(U0)
	k = 1
	#* undergo geometry improvement on X, by calculating coefficients and checking they meet criticality conditions
	while np.linalg.norm(prev_U @ prev_U.T - U @ U.T, ord='fro') > 1e-10:
		print(f"At iteration {k} the distance from convergence is {np.linalg.norm(prev_U @ prev_U.T - U @ U.T, ord='fro')}")
		#* Ensure the design set has a good geometry for constructing a model 
		while True : 
			coef = fit_coef(X, fX, U)
			if delta <= np.linalg.norm(grad(X, coef, U)) : 
				break
			else :
				#! Need to update trust-region radius 
				X, fX, interpolation_solutions, visited_pts, delta = geo.improve_geometry(k, problem, current_solution, f_x, delta, U, visited_pts, X, fX)
		
		#set the old U and update
		prev_U = np.copy(U)
		# if d == 1 : 
		# 	U = fit_affine(problem, X, fX ) 
		# else : 
		U = fit_varpro(problem, X, fX, U)
		k += 1
	print(f"The fitting has completed with {k} iterations with convergence: {np.linalg.norm(prev_U @ prev_U.T - U @ U.T, ord='fro')}")

	coef = fit_coef(X,fX,U)

	#final fitting of the coefficients and rotating the final U 
	U = rotate_U(X, fX, coef, U)
	coef = fit_coef(X,fX,U)

	model = lambda x : model_evaluate(x,coef,U)

	return  U, model, X, fX, interpolation_solutions, visited_pts, delta


def fit_coef(X:np.ndarray, fX: np.ndarray, U: np.ndarray) -> np.ndarray : 
	"""
		Finds the coefficients of the interpolation model by solving the system of equations:
				V(U^TX)coeff = fX
	Args:
		X (np.ndarray): The design set of design points of shape (M,n)
		fX (np.ndarray): corresponding function estimates of design points of shape (M,1)
		U (np.ndarray): The active subspace of shape (n,d)

	Returns:
		np.ndarray: A list of the coefficients of shape (q,1)
	"""
	#project design points for vandermonde construction: 
	Y = X @ U
	V_matrix = V(Y) #shape (M,q)
	coef = pinv(V_matrix) @ fX 

	if len(coef.shape) != 2 : 
		coef = coef.reshape(-1,1)

	return coef 

def fit_affine(problem: Problem, X: np.ndarray, fX: np.ndarray) -> np.ndarray:
	"""
		Solves the affine problem for when subspace dimension is 1 
	Args:
		problem (Problem): The SO problem
		X (np.ndarray): The design set of M n-dimensional design points (shape (M,n))
		fX (np.ndarray): The corresponding function estimations of each design point (shape (M,1))

	Returns:
		np.ndarray: The active subspace matrix (shape (n,1))
	"""	 
	# Normalize the domain 
	# lb = np.min(X, axis = 0)
	lb = problem.lower_bounds
	# ub = np.max(X, axis = 0)
	ub = problem.upper_bounds
	# dom = BoxDomain(lb, ub) 
	XX = np.hstack([normalise(problem, X), np.ones((X.shape[0],1))])

	# Normalize the output
	fX = (fX - np.min(fX))/(np.max(fX) - np.min(fX))
	b = np.zeros(fX.shape)

	b = np.matmul(np.linalg.pinv(XX), fX)	

	U = b[0:-1].reshape(-1,1)
	# Correct for transform 
	U = normalise_der(problem).dot(U)
	# Force to have unit norm
	U /= np.linalg.norm(U)
	return U	


def normalise(problem, X) -> np.ndarray : 
	if len(X.shape) == 1:
		X = X.reshape(-1, problem.dim) 
		c = center(problem)
		D = normalise_der(problem)
		X_norm = D.dot( (X - c.reshape(1,-1)).T ).T
		return X_norm.flatten() 
		
	else:
		c = center(problem)
		D = normalise_der(problem)
		X_norm = D.dot( (X - c.reshape(1,-1)).T ).T
		return X_norm 

def normalise_der(problem) :
	"""Derivative of normalization function"""
	norm = lambda x : x
	norm_ub = norm(problem.upper_bounds)
	norm_lb = norm(problem.lower_bounds)

	slope = np.ones(problem.dim)
	I = (norm_ub != norm_lb) & np.isfinite(norm_lb) & np.isfinite(norm_ub)
	slope[I] = 2.0/(norm_ub[I] - norm_lb[I])
	return np.diag(slope) 

def center(problem) : 
	norm = lambda x : x
	norm_ub = norm(np.array(problem.upper_bounds))
	norm_lb = norm(np.array(problem.lower_bounds))

	c = np.zeros(problem.dim)
	I = np.isfinite(norm_lb) & np.isfinite(norm_ub)
	c[I] = (norm_lb[I] + norm_ub[I])/2.0
	return c




def grassmann_trajectory(U: np.ndarray, Delta: np.ndarray, t: float) -> np.ndarray:
	"""
		Calculates the geodesic along the Grassmann manifold 
	Args:
		U (np.ndarray): The active subspace matrix of shape (n,d)
		Delta (np.ndarray): The search direction along the Grassmann manifold with shape (n,d)
		t (float): Independent parameter in the line equation takes values between (0,infty) and is selected to ensure convergence. 

	Returns:
		np.ndarray: The new candidate for the active subspace based on the step made of shape (n,d)
	"""
	print(f'Delta in grassmann_trajectory is {Delta}')
	Y, sig, ZT = scipy.linalg.svd(Delta, full_matrices = False, lapack_driver = 'gesvd')

	UZ = np.dot(U, ZT.T)
	U_new = np.dot(UZ, np.diag(np.cos(sig*t))) + np.dot(Y, np.diag(np.sin(sig*t))) #! THIS IS THE GEODESIC EQUATION

	#Correct the new step U by ensuring it is orthonormal with consistent sign on the elements
	U_new, R = np.linalg.qr(U_new, mode = 'reduced') 
	U_new = np.dot(U_new, np.diag(np.sign(np.diag(R))))
	return U_new

def residual(X: np.ndarray, fX: np.ndarray, U: np.ndarray) -> np.ndarray:
	"""
		Construct the Residual of the model fitting, such that 
		r = fX - V(U^TX)coeff
	Args:
		X (np.ndarray): The design set of design points of shape (M,n)
		fX (np.ndarray): The corresponding function estimates of design points of shape (M,1) 
		U (np.ndarray): The active subspace of shape (n,d)

	Returns:
		np.ndarray: The residual error for each design point on the local model of shape (M,1)
	"""
	Y = X @ U
	c = fit_coef(X,fX,U) #shape(q,1)
	model_fX = np.array([model_evaluate(np.array(x).reshape(1,-1),c,U) for x in X]).reshape(-1,1) #A list of length M with float elements
	r = fX - model_fX
	return r

#! THIS NEEDS CHECKING OVER
def jacobian(X: np.ndarray, fX:np.ndarray, U:np.ndarray) -> np.ndarray : 
	"""
		Constructs the Jacobian of the residual with respect to the active subspace
	Args:
		X (np.ndarray): The design set of design points of shape (M,n)
		fX (np.ndarray): The corresponding function estimates of design points of shape (M,1) 
		U (np.ndarray): The active subspace of shape (n,d)

	Returns:
		np.ndarray: A tensor of shape (M,n,d) where each element is the partial derivative of the i-th residual component with respect to the (j,k)th entry of the active subspace 
	"""
	
	#FIRST ENSURE THAT THE ARGUMENTS HAVE DIMENSIONS THAT MATCH
	assert X.shape[1] == U.shape[0], "X should have columns equal to the number of rows in U"
	assert X.shape[0] == fX.shape[0], "The number of samples in the design set X should match the number of function estimations in fX" 
	assert fX.shape[1] == 1, "The function estimates of the design set should be a column vector"
	
	#get dimensions
	M, n = X.shape
	n, d = U.shape

	Y = X @ U

	#find the residual 
	Y = X @ U
	c = fit_coef(X,fX,U) #shape(q,1)
	model_fX = np.array([model_evaluate(np.array(x).reshape(1,-1),c,U) for x in X]).reshape(-1,1) #A list of length M with float elements



	r = fX - model_fX


	#! FROM HERE THE FUNCTION NEEDS CHECKING
	#find the vandermonde matrix and derivative of the vandermonde matrix of the projected design set 
	V_matrix = V(Y) #shape (M,q)
	DV_matrix = DV(Y) #shape (M,q,n)

	M,q = V_matrix.shape

	Y, s, ZT = scipy.linalg.svd(V_matrix, full_matrices = False)
	# s = np.array([np.inf if x == 0.0 else x for x in s]) 
	with np.errstate(divide='ignore', invalid='ignore'):
		D = np.diag(1.0 / s)
		D[np.isinf(D)] = 0  # convert inf to 0 if desired

	J1 = np.zeros((M, n, d))
	J2 = np.zeros((q, n, d))

	# populate the Jacobian
	for k in range(d):
		for j in range(n):
			
			#This is the derivative of U  
			DVDU_k =  X[:,j,None]*DV_matrix[:,:,k] #shape (M,q)

			#first term in the Jacobian 
			J1[:, j, k] = DVDU_k.dot(c).flatten() #shape (M,)
			
			#second term of the Jacobian before V(U)^-
			J2[:, j, k] = DVDU_k.T.dot(r).flatten() #shape of (M,)

	# project J1 against the range of V
	J1 -= np.tensordot(Y, np.tensordot(Y.T, J1, axes=(1,0)), axes=(1,0))  # shape: (M,)

	# apply pseudo-inverse via SVD components
	J2_projected = np.tensordot(D, np.tensordot(ZT, J2, axes=(1,0)), axes=(1,0))  # shape: (q, n, d)

	# combine terms to get full Jacobian
	Jac = -(J1 + np.tensordot(Y, J2_projected, axes=(1,0)))  # shape: (M, n, d)

	
	return Jac



def fit_varpro(problem: Problem, X: np.ndarray, fX: np.ndarray, U: np.ndarray) -> np.ndarray : 
	"""
		Runs a Gauss-Newton

	Args:
		problem (Problem): The SO problem with dimension n
		X (np.ndarray): design set of shape (M,n)
		fX (np.ndarray): corresponding function estimates of design points of shape (M,1)
		U (np.ndarray): The active subspace of shape (n,d)

	Returns:
		np.ndarray: The active subspace of shape (n,d)
	"""

	def gn_solver(Jac: np.ndarray, residual: np.ndarray) -> np.ndarray :
		"""
			An anonymous function to compute the Gauss-Newton step to find a descent direction
		Args:
			Jac (np.ndarray): The Jacobian of the residual with respect to the active subspace. It has shape (M,n,d)
			residual (np.ndarray): The residual of the current model approximation with shape (M,1)

		Returns:
			np.ndarray: A vectorised form of the descent direction with shape (nd,). The full descent direction has shape (n,d)
		"""         
		M,n,d = Jac.shape
		Jac_vec = Jac.reshape(X.shape[0],-1) #reshapes (M,n,d) to (M,nd) 
		#compute short form SVD
		Y, sig, ZT = scipy.linalg.svd(Jac_vec, full_matrices = False, lapack_driver = 'gesvd') #Y has shape (M,M), sig has shape (M,nd), and ZT has shape (nd,nd)
		
		# Find descent direction
		# Extract singular values from s (the diagonal)
		sing_vals = np.diag(sig)  # shape (M,)

		# Compute Y^T r
		YTr = (Y.T @ residual).flatten()  # shape (M,)

		# s_inv = 1.0 / sig  # shape (M,)
		tol = np.max(sig) * np.finfo(float).eps
		s_inv = np.where(sig > tol, 1.0 / sig, 0.0)

		Delta_vec = -ZT.T @ (s_inv * YTr)  # shape (n*d,)

		return Delta_vec

	jacobian_variable_U = lambda U : jacobian(X,fX,U)
	residual_variable_U = lambda U : residual(X,fX,U) 

	U = gauss_newton_solver(residual_variable_U, jacobian_variable_U, U, grassmann_trajectory, gn_solver) 

	return U 


def gauss_newton_solver(residual: callable, jacobian: callable, U: np.ndarray, trajectory: callable, gn_solver: callable) -> np.ndarray : 
	"""
		Solves the Gauss_newton problem on the Grassmann manifold:
			vec(Delta) = -vec(Jac(U))^{+}r(U)

	Args:
		residual (callable): Function that takes the active subspace U of shape (n,d) and calculates the residual of the predicted model under a fixed design set. Returns a matrix of shape (M,1)
		jacobian (callable): Function that takes the active subspace U of shape (n,d) and calculates the Jacobian of the residual with respect to U. Returns a matrix of shape (M,n,d)
		U (np.ndarray): The subspace matrix with shape (n,d)
		trajectory (callable): The Grassman trajectory that returns the new step of the active subspace matrix of shape (n,d). Returns a new active subspace of shape (n,d)
		gn_solver (callable): The Gauss-Newton step that returns the vectorised descent direction of shape (nd,)

	Returns:
		np.ndarray: A new active subspace matrix U_+ of shape (n,d)
	"""

	#initial values for res and Jac and Grad
	max_iter = 100
	res = residual(U) #shape (M,1)
	Jac = jacobian(U) #shape (M,n,d)
	Grad = np.tensordot(res.ravel(), Jac, axes=(0, 0))  # (n,d)

	#Compute tolerances 
	Grad_norm = norm(Grad)
	tol = max(1e-10*Grad_norm, 1e-14)
	tol_Delta_norm = 1e-12

	#loop over linesearch until the norm of the gauss-newton step, the norm of the Grad or the norm of Res(U) increases
	for i in range(max_iter) :
		# print(f'running iteration {i} of the Gauss-Newton Algorithm')
		residual_increased = False

		Jac_vec = Jac.reshape(Jac.shape[0], -1) #shape (M, nd)
		Delta_vec = gn_solver(Jac, res) #shape (nd,)
		Delta = Delta_vec.reshape(Jac.shape[1], Jac.shape[2]) #shape (n,d)

		print(f'Delta in gauss_newton_solver: {Delta}')


		# backtracking: find acceptable step gamma (t) along geodesic trajectory
		U_new, step = backtracking(residual, Grad, Delta, U, trajectory)

		res_candidate = residual(U_new)
		Jac_candidate = jacobian(U_new)
		Grad_candidate = np.tensordot(res_candidate.ravel(), Jac_candidate, axes=(0, 0))


		if norm(res_candidate) >= norm(res) : 
			residual_increased = True 
		else :
			#Update the residual, jacobian, Gradient, and active subspace
			res = res_candidate 
			Jac = Jac_candidate
			Grad = Grad_candidate
			U = U_new 
			

		#Termination Conditions 
		if Grad_norm < tol or norm(Delta) < tol_Delta_norm or residual_increased : 
			return U_new 
		
	return U
		 

def backtracking(residual: callable, Grad: np.ndarray, delta: np.ndarray, U:np.ndarray, trajectory: callable) -> tuple[np.ndarray, float]  :
	"""
		Backtracking line search to satisfy the Armijo Condition:
			residual(U + alpha*delta) < residual(U) + alpha*beta*gamma
			where: 
				- alpha is <Grad, delta>
				- beta is a control parameter in (0,1)
				- gamma is the backtracking coefficient

	Args:
		residual (callable): Function that takes the active subspace U of shape (n,d) and calculates the residual of the predicted model under a fixed design set 
		Grad (np.ndarray): Gradient of the active subspace matrix on the Grassmann manifold of shape (n,d)
		delta (np.ndarray): The Gauss-Newton step of shaoe (n,d)
		U (np.ndarray): The active subspace matrix with shape (n,d)
		trajectory (callable): A function that takes an active subspace matrix U of shape (n,d), the Gauss-Newton step d of shape (n,d), and some float 0<t<infty as arguments 
		and returns an active subspace matrix along the geodesic of the Grassmann manifold t steps from U  tangent to d. 

	Returns:
		tuple[np.ndarray, np.ndarray, float]: 
			- The new active subspace matrix U of shape (n,d)
			- The residual at the final active subspace matrix U of shape (M,1)
			- The backtracking coefficient gamma  (gamma=1 implies no backtracking)
	"""	 
	#initialise control parameter, step shrink factor, and max iterations
	beta = 1e-4
	rho = 0.5
	max_iter = 100

	# directional derivative
	alpha = np.inner(Grad.reshape(-1,), delta.reshape(-1,))  # vecGrad^T vec(delta) in matrix form

	# If direction is not a descent direction, flip to negative gradient
	if alpha >= 0:
		delta = -Grad
		alpha = np.inner(Grad.reshape(-1,), delta.reshape(-1,))

	# starting objective and residual
	init_res = residual(U)

	step_size = 1.0
	for i in range(max_iter):
		# print(f'running iteration {i} of the backtracking')
		U_candidate = trajectory(U, delta, step_size)
		res_candidate = residual(U_candidate)

		# Armijo condition: f(U + t delta) <= f(U) + t * beta * alpha
		if norm(res_candidate) <= norm(init_res) + step_size * beta * alpha:
			# success
			#Make sure U_new is orthonormal
			U_candidate, _ = np.linalg.qr(U_candidate)
			U_candidate = np.sign(np.diag(_)) * U_candidate  # ensure consistent orientation
			return U_candidate, step_size

		# otherwise shrink step
		step_size *= rho

	# if not found, return the best we have (the last candidate)
	U_candidate = trajectory(U, delta, step_size)
	#Make sure U_new is orthonormal
	U_candidate, _ = np.linalg.qr(U_candidate)
	U_candidate = np.sign(np.diag(_)) * U_candidate  # ensure consistent orientation
	return U_candidate, step_size


def rotate_U(X:np.ndarray, fX: np.ndarray, coef:np.ndarray, U:np.ndarray) -> np.ndarray : 
	"""
		Rotates the active subspace matrix onto the most important direction of 
	Args:
		X (np.ndarray): design set of shape (M,n)
		fX (np.ndarray): corresponding function estimates of design points of shape (M,1)
		coef (np.ndarray): The coefficients of the local model of shape (q,1)
		U (np.ndarray): The active subspace of shape (n,d)

	Returns:
		np.ndarray: The rotated active subspace matrix of shape (n,d)
	"""

	# Step 1: Apply active subspaces to the profile function at samples X
			# to rotate onto the most important directions
	if U.shape[1] > 1 :   
		grads = grad(X, coef, U)
		active_grads = grads @ U 
		# We only need the short-form SVD
		Ur = scipy.linalg.svd(active_grads.T, full_matrices = False)[0]
		U = U @ Ur

	U = U

	# Step 2: Flip signs such that average slope is positive in the coordinate directions
	coef = fit_coef(X, fX, U)
	grads = grad(X, coef, U)
	active_grads = grads @ U 
	U = U = U.dot(np.diag(np.sign(np.mean(active_grads, axis = 0))))

	return U


def grad(X: np.ndarray, coef: np.ndarray, U:np.ndarray) -> np.ndarray:
	"""
		Computes the gradients of the local model at each design point of the design set X 
	Args:
		X (np.ndarray): design set of shape (M,n)
		coef (np.ndarray): The coefficients of the local model of shape (q,1)
		U (np.ndarray): The active subspace of shape (n,d)

	Returns:
		np.ndarray: The gradients of the model at each design point X
	"""
	if len(X.shape) == 1:
		one_d = True
		X = X.reshape(1,-1)	
	else:
		one_d = False	
	
	Y = X @ U
	DV_matrix = DV(Y)
	# Compute gradient on projected space
	Df = np.tensordot(DV_matrix, coef, axes = (1,0))
	# Inflate back to whole space
	Df = np.squeeze(Df, axis=-1) 
	Df = Df @ U.T

	if one_d:
		return Df.reshape(Y.shape[1])
	else:
		return Df


def V(X: np.ndarray) -> np.ndarray :
	"""
		Generate the Vandermonde Matrix
	Args:
		X (np.ndarray): The design set of shape (M,n) or (M,d) where n is the dimension of the original problem and d is the subspace dimension.

	Returns:
		np.ndarray: A vandermonde matrix of shape (M,q) where q is the length of the polynomial basis 
	"""
	# X = X.reshape(-1, self.dim)
	degree = 2 #self.factors['polynomial degree']
	indices = index_set(degree, X.shape[1]).astype(int)
	X = scale(X)
	M = X.shape[0]
	V_coordinate = [chebvander(X[:,k], degree) for k in range(X.shape[1])]
	
	V_matrix = np.ones((M, len(indices)), dtype = X.dtype)
	
	for j, alpha in enumerate(indices):
		for k in range(X.shape[1]):
			V_matrix[:,j] *= V_coordinate[k][:,alpha[k]]
	return V_matrix


def DV(X: np.ndarray) -> np.ndarray : 
	"""
		Column-wise derivative of the Vandermonde matrix

		Given design points this creates the Vandermonde-like matrix whose entries
		correspond to the derivatives of each of basis elements
	Args:
		X (np.ndarray): The design set of shape (M,n) or (M,d) where n is the dimension of the original problem and d is the subspace dimension.

	Returns:
		np.ndarray: Derivative of Vandermonde matrix  of shape (M,q,n) where DV[i,j,:] is the gradient of V[i,j]. 
	"""
	degree = 2 #self.factors['polynomial degree]
	indices = index_set(degree, X.shape[1]).astype(int)
	X = scale(X)
	Dmat = build_Dmat()
	M = X.shape[0]
	V_coordinate = [chebvander(X[:,k], degree) for k in range(X.shape[1])]
	
	N = len(indices)
	DV_matrix = np.ones((M, N, X.shape[1]), dtype = X.dtype)

	try:
		dscale_matrix = dscale(X)
	except NotImplementedError:
		dscale_matrix = np.ones(X.shape[1])	



	for k in range(X.shape[1]):
		for j, alpha in enumerate(indices):
			for q in range(X.shape[1]):
				if q == k:
					DV_matrix[:,j,k] *= np.dot(V_coordinate[q][:,0:-1], Dmat[alpha[q],:])
				else:
					DV_matrix[:,j,k] *= V_coordinate[q][:,alpha[q]]
		# Correct for transform
		DV_matrix[:,:,k] *= dscale_matrix[k] 		

	return DV_matrix 

def scale(X:np.ndarray) -> np.ndarray : 
	"""
		Apply scaling to the input coordinates so that all 
	Args:
		X (np.ndarray): The design set of shape (M,n) or (M,d) where n is the dimension of the original problem and d is the subspace 

	Returns:
		np.ndarray: The scaled design set of shape (M,n) or (M,d) where n is the dimension of the original problem and d is the subspace 

	"""
	# lb = np.min(X, axis = 0)
	# ub = np.max(X, axis=0)
	# try:
	# 	return 2*(X-lb[None,:])/(ub[None,:] - lb[None,:]) - 1
	# except AttributeError:
	# 	return X 
	# Convert to float to avoid integer division

	X = np.asarray(X, dtype=float)
	
	# Compute min and max per column
	lb = np.min(X, axis=0)
	ub = np.max(X, axis=0)
	denom = ub - lb

	# Handle constant columns safely (avoid divide-by-zero)
	denom_safe = np.where(denom == 0, 1.0, denom)
	
	# Perform column-wise linear scaling
	X_scaled = 2.0 * (X - lb) / denom_safe - 1.0
	
	# For constant columns, center them to zero
	X_scaled[:, denom == 0] = 0.0
	
	return X_scaled





def dscale(X: np.ndarray) -> np.ndarray : 
	"""
		Returns the scaling associated with the scaling transform
	Args:
		X (np.ndarray): The design set of shape (M,n) or (M,d) where n is the dimension of the original problem and d is the subspace 

	Returns:
		np.ndarray: The scaled design set of shape (M,n) or (M,d) where n is the dimension of the original problem and d is the subspace 
	"""
	lb = np.min(X, axis = 0)
	ub = np.max(X, axis = 0)
	try : 
		res = 2/(ub[None,:]-lb[None,:])
	except AttributeError:
		res = X

	return res.flatten()
	
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


def main() :
	# Instantiate example problem (f(x) = ||x||^2)
	problem = instantiate_problem("ROSENBROCK-1")
	dim = problem.dim

	# Initial point (center of local region)
	x0 = np.zeros((1,dim))
	current_solution = Solution(tuple(x0), problem)

	# Model construction parameters
	delta_k = 0.5       # trust region radius
	k = 0
	expended_budget = 0
	visited_points_list = [current_solution]

	# Construct model and active subspace
	# k, problem, current_solution, delta, expended_budget, visited_pts
	current_solution, delta_k, model, U, fval, expended_budget, interpolation_solutions, visited_pts  = construct_model(
		k, problem, current_solution, delta_k, expended_budget, visited_points_list
	)

	# Evaluate model performance over random samples in the ball
	n_test = 200
	X_test, f_true, f_model = [], [], []

	obj_fn = lambda x : np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

	for i in range(n_test):
		direction = np.random.randn(1,dim)
		direction /= np.linalg.norm(direction)
		radius = np.random.rand() * delta_k
		x = x0 + radius * direction
		X_test.append(x)
		f_true.append(obj_fn(x))
		f_model.append(model(x))

	X_test = np.array(X_test)
	f_true = np.array(f_true)
	f_model = np.array(f_model)

	# Compute error metrics
	mse = np.mean((f_true - f_model) ** 2)
	rel_error = np.mean(np.abs(f_true - f_model) / (np.abs(f_true) + 1e-8))

	### DEBUG CHECK ###
	print("\n--- Final diagnostic checks ---")

	# Compare distributions
	print(f"True f mean/std: {np.mean(f_true):.4f}, {np.std(f_true):.4f}")
	print(f"Pred f mean/std: {np.mean(f_model):.4f}, {np.std(f_model):.4f}")

	# R² and correlation
	print(f"R² score: {r2_score(f_true, f_model):.4f}")
	corr = np.corrcoef(f_true.ravel(), f_model.ravel())[0,1]
	print(f"Pearson correlation: {corr:.4f}")

	# Check relative scale
	ratio = np.mean(f_model) / (np.mean(f_true) + 1e-12)
	print(f"Average f_model / average true: {ratio:.4f}")

	print("--- End final diagnostic checks ---\n")
	### END DEBUG CHECK ###

	print("Active subspace matrix U shape:", U.shape)
	print("Mean squared error:", mse)
	print("Mean relative error:", rel_error)
	print("Average true f(x):", np.mean(f_true))
	print("Average model prediction:", np.mean(f_model)) 



if __name__ == '__main__' : 
	main()