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

class BadStep(Exception) :
    pass 

def construct_model( problem: Problem, current_solution: Solution, delta_k: float, k: int, expended_budget: int, visited_points_list: list[Solution]) -> tuple[np.ndarray, Solution, float, list[float], int, list[Solution], np.ndarray, list[Solution]] : 
    #construct initial active subspace 
    d = 3
    init_S_full = geo.generate_set(problem, d, current_solution, delta_k)
    U, _ = np.linalg.qr(init_S_full.T)

    X, f_index = design.construct_interpolation_set(current_solution, problem, U, delta_k, k, visited_points_list)

    #np.array(fX), current_solution, interpolation_solutions, visited_pts, expended_budget
    # fX, current_solution, interpolation_solutions, visited_points_list, expended_budget = evaluate_interpolation_points(k, problem, current_solution, expended_budget, f_index, X, visited_points_list, delta_k)
    fX = np.vstack([np.linalg.norm(x)**2 for x in X]) 
    interpolation_solutions = [Solution(tuple(x), problem) for x in X]
    f_old = fX[0,0]

    #get the function value of the current solution - this is the first value in the array of X values 
    fval = fX.flatten().tolist()


    U, coef, X, fX, interpolation_sols, visited_pts_list, delta_k = fit(k, problem, X, fX, current_solution, f_old, delta_k, interpolation_solutions, visited_points_list, U)


    return coef, current_solution, delta_k, fval, expended_budget, interpolation_sols, U, visited_points_list	

def model_evaluate(problem: Problem, X: np.ndarray, U: np.ndarray, coef: np.ndarray) -> np.ndarray :
    """
        Evaluates the model at each design point x of the M design points in X 
    Args:
        X (np.ndarray): The design set of design points of shape (M,n)
        U (np.ndarray): The active subspace matrix of shape (n,d)
        coef (np.ndarray): The coefficients of the local model of shape (q,1) where q is len(index_set(degree, dim).astype(int))

    Returns:
        np.ndarray: The model evaluations of each design point of shape (M,1)
    """
    if len(X.shape) == 1 : 
        X.reshape(1,-1)

    M = X.shape[0]
    d = U.shape[1]

    # Ensure c is a column vector
    coef = np.asarray(coef).reshape(-1, 1)

    # Compute the Vandermonde matrix for the projected points
    # V_matrix = V(X, U, d)  # shape (M, q)
    # Evaluate polynomial: V * c
    # mX = V_matrix @ coef  # shape (M, 1)

    # center test/evaluation points the same way we did for fitting
    c_vec = center(U, problem)                 # you already have center(U,problem)
    X_centered = (X - c_vec.reshape(1, -1))    # ensure shape (M, n)
    V_matrix = V(X_centered, U, d)             # V will project X_centered onto U internally
    print("\n--- model_evaluate: debug ---")
    print("U shape:", U.shape)
    print("First row of U:", U[0])
    print("V shape:", V_matrix.shape)
    print("coef shape:", coef.shape)
    print("V mean/std/min/max:", V_matrix.mean(), V_matrix.std(), V_matrix.min(), V_matrix.max())
    mX = V_matrix @ coef
    print("pred mean/std/min/max:", mX.mean(), mX.std(), mX.min(), mX.max())
    print("--- end model_evaluate: debug ---\n")    

    if mX.shape[0] == 1 : 
        mX.reshape(-1,)

    ### DEBUG CHECK ###
    print("\n--- model_evaluate diagnostics ---")
    print(f"Predictions mean/std/min/max: {np.mean(mX):.4f}, {np.std(mX):.4f}, {np.min(mX):.4f}, {np.max(mX):.4f}")
    print(f"U shape: {U.shape}")
    print("--- end model_evaluate diagnostics ---\n")
    ### END DEBUG CHECK ###

    return mX 

def grad( X: np.ndarray, coef: np.ndarray, U: np.ndarray) -> np.ndarray :
    n,d = U.shape
    if len(X.shape) == 1:
        one_d = True
        X = X.reshape(1,-1)	
    else:
        one_d = False	
    
    DV_matrix = DV(X, np.eye(n), d)
    # Compute gradient on projected space
    Df = np.tensordot(DV_matrix, coef, axes = (1,0))
    # Inflate back to whole space
    Df = Df.dot(U.T)
    if one_d:
        return Df.reshape(X.shape[1])
    else:
        return Df

def fit(k: int, problem: Problem, X: np.ndarray, fX: np.ndarray, current_solution: Solution, f: float, delta_k, interpolation_sols, visited_pts_list: list[Solution], U0: np.ndarray):
    d = U0.shape[1]

    lb = problem.lower_bounds
    ub = problem.upper_bounds

    X = np.array(X)
    fX = np.array(fX).flatten()	


    assert X.shape[0] == fX.shape[0], "Dimensions of input do not match"

    # Check if we have enough data to make problem overdetermined
    m = X.shape[1]
    q = 2
    degree = 2
    n_param = scipy.special.comb(d+q, q)	# Polynomial contribution
    n_param += m*q - (q*(q+1))//2			# Number of parameters in Grassmann manifold
    #! REMOVED CHECK FOR UNDERTERMINED MODEL 
    # if len(fX) < n_param:
    # 	mess = "A polynomial ridge approximation of degree %d and subspace dimension %d of a %d-dimensional function " % (d, n, m)
    # 	mess += "requires at least %d samples to not be underdetermined" % (n_param, )
    # 	raise UnderdeterminedException(mess) 	

    # Special case where solution is convex and no iteration is required
    if d == 1 and degree == 1:
        U = fit_affine(problem, X, U0, fX)	
        coef = fit_coef(problem, X, fX, U)	
        return  U, coef, X, fX, interpolation_sols, visited_pts_list, delta_k


    # Check that U0 has the right shape
    U0 = np.array(U0)
    assert U0.shape[0] == X.shape[1], "U0 has %d rows, expected %d based on X" % (U0.shape[0], X.shape[1])
    assert U0.shape[1] == d, "U0 has %d columns; expected %d" % (U0.shape[1], d)

    # Orthogonalize just to make sure the starting value satisfies constraints	
    U0, R = np.linalg.qr(U0, mode = 'reduced') 
    U0 = np.dot(U0, np.diag(np.sign(np.diag(R))))

    prev_U = np.zeros(U0.shape)
    U = np.copy(U0)
    
    k=0
    #* undergo geometry improvement on X, by calculating coefficients and checking they meet criticality conditions
    while np.linalg.norm(prev_U - U) > 1e-10 or k <= 1000:
        j = 0
        print(f'iteration {k} of fitting the model')
        while True : 
            print(f'iteration {k}.{j} of improving design set')
            coef = fit_coef(problem, X, fX, U)
            if delta_k <= np.linalg.norm(grad(X, coef, U))*10 : 
                break
            else :
                X, fX, interpolation_solutions, visited_pts, delta_k = geo.improve_geometry(k, problem, current_solution, f, delta_k, U, visited_pts, X, fX)
        
        #set the old U and update
        prev_U = np.copy(U)
        U, _ = fit_varpro(problem, X, fX, U)

        k += 1


    U, coef = finish(problem, X, fX, U)

        #TODO: Implement other fitting when bound is not none 
        # if lb is None and ub is None:
        # 	_fit_varpro(X, fX, U0, d)
        # 	prev_U = np.copy(U)
        # 	U = _U
        # 	k += 1
        # else:	
        # 	fit_alternating(X, fX, U0, d)
        # 	prev_U = np.copy(U)
        # 	U = _U
        # 	k += 1 	
    return  U, coef, X, fX, interpolation_sols, visited_pts_list, delta_k

def fit_coef( problem, X, fX, U):
    r""" Returns the linear coefficients
    """
    lb = problem.lower_bounds
    ub = problem.upper_bounds
    dim = U.shape[1]

    # V_matrix = V(X,U,dim)
    # c = np.matmul(np.linalg.pinv(V_matrix), fX)

    # basis = Basis(degree, X = Y) 
    c_vec = center(U, problem)               # center is already defined in your file
    X_centered = (X - c_vec.reshape(1, -1))  # shape (M, n)
    V_matrix = V(X_centered, U, dim)
    c = np.matmul(np.linalg.pinv(V_matrix), fX)

    ### DEBUG CHECK ###
    # print("\n--- fit_coef diagnostics ---")
    # print(f"V_matrix shape: {V_matrix.shape}")
    # print(f"Condition number of V_matrix: {np.linalg.cond(V_matrix):.3e}")
    # print(f"fX mean/std/min/max: {np.mean(fX):.4f}, {np.std(fX):.4f}, {np.min(fX):.4f}, {np.max(fX):.4f}")
    # print(f"coef mean/std/min/max: {np.mean(c):.4f}, {np.std(c):.4f}, {np.min(c):.4f}, {np.max(c):.4f}")
    # print("--- end fit_coef diagnostics ---\n")
    ### END DEBUG CHECK ###

    #TODO: Need to implement bound fitting
    # if lb is None and ub is None:
    # 	c = two_norm_fit(V, fX)
    # elif bound == 'lower':
    # 	c = bound_fit(-V, -fX, norm = 2)
    # elif bound == 'upper':
    # 	c = bound_fit(V, fX, norm = 2)
    
    return c

def fit_affine(problem: Problem, X, U, fX): 
    r""" Solves the affine 
    """
    # Normalize the domain 
    # lb = np.min(X, axis = 0)
    lb = problem.lower_bounds
    # ub = np.max(X, axis = 0)
    ub = problem.upper_bounds
    # dom = BoxDomain(lb, ub) 
    XX = np.hstack([normalize(X, U, problem), np.ones((X.shape[0],1))])

    # Normalize the output
    fX = (fX - np.min(fX))/(np.max(fX) - np.min(fX))
    b = np.matmul(np.linalg.pinv(XX), fX)

    #TODO: Need to implement bound fitting
    # b = np.zeros(fX.shape)
    # if lb is None and ub is None:
    # 	b = two_norm_fit(XX, fX)
    # elif self.bound == 'lower':
    # 	# fX >= XX b
    # 	b = bound_fit(XX, fX, norm = 2)
    # elif self.bound == 'upper':
    # 	b = bound_fit(-XX, -fX, norm = 2)	 	

    U = b[0:-1].reshape(-1,1)
    # Correct for transform 
    U = normalize_der(U, problem).dot(U)
    # Force to have unit norm
    U /= np.linalg.norm(U)
    return U

def normalize(X, U, problem) : 
    """ Given a points in the application space, convert it to normalized units
    
    Parameters
    ----------
    X: np.ndarray((M,m))
        points in the domain to normalize
    """
    try:
        X.shape
    except AttributeError:
        X = np.array(X)
    if len(X.shape) == 1:
        X = X.reshape(1, U.shape[0])
        c = center(U, problem)
        D = normalize_der(U, problem)	
        X_norm = D.dot( (X - c.reshape(1,-1)).T ).T
        return X_norm.flatten()
    else:
        c = center(U, problem)
        D = normalize_der(U, problem)	
        X_norm = D.dot( (X - c.reshape(1,-1)).T ).T
        return X_norm

# def center(U, problem) : 
#     norm = lambda x : x
#     norm_ub = norm(problem.upper_bounds)
#     norm_lb = norm(problem.lower_bounds)

#     c = np.zeros(U.shape[0])
#     I = np.isfinite(norm_lb) & np.isfinite(norm_ub)
#     c[I] = (norm_lb[I] + norm_ub[I])/2.0
#     return c

# def normalize_der(U, problem) :
#     """Derivative of normalization function"""
#     norm = lambda x : x
#     norm_ub = norm(problem.upper_bounds)
#     norm_lb = norm(problem.lower_bounds)

#     slope = np.ones(U.shape[0])
#     I = (norm_ub != norm_lb) & np.isfinite(norm_lb) & np.isfinite(norm_ub)
#     slope[I] = 2.0/(norm_ub[I] - norm_lb[I])
#     return np.diag(slope) 

def normalize_der(U, problem):
    """Derivative of normalization function (robust to lists/tuples)"""
    # Convert to NumPy arrays
    norm_ub = np.asarray(problem.upper_bounds, dtype=float)
    norm_lb = np.asarray(problem.lower_bounds, dtype=float)

    # Ensure proper shape (one value per variable)
    if norm_lb.ndim > 1:
        norm_lb = norm_lb.flatten()
    if norm_ub.ndim > 1:
        norm_ub = norm_ub.flatten()

    slope = np.ones(U.shape[0])

    # Identify finite, non-degenerate bounds
    mask = (norm_ub != norm_lb) & np.isfinite(norm_lb) & np.isfinite(norm_ub)
    idx = np.where(mask)[0]  # Convert boolean mask → integer indices

    slope[idx] = 2.0 / (norm_ub[idx] - norm_lb[idx])
    return np.diag(slope)


# def center(U, problem):
#     """Compute center of normalized variable bounds"""
#     norm_ub = np.asarray(problem.upper_bounds, dtype=float)
#     norm_lb = np.asarray(problem.lower_bounds, dtype=float)

#     # Ensure proper shape
#     if norm_lb.ndim > 1:
#         norm_lb = norm_lb.flatten()
#     if norm_ub.ndim > 1:
#         norm_ub = norm_ub.flatten()

#     # Project bounds onto subspace
#     proj_lb = norm_lb @ U
#     proj_ub = norm_ub @ U

#     c = np.zeros(U.shape[1])
#     mask = np.isfinite(proj_lb) & np.isfinite(proj_ub)
#     idx = np.where(mask)[0]

#     c[idx] = (proj_lb[idx] + proj_ub[idx]) / 2.0
#     return c	

def center(U, problem):
    """Compute center of variable bounds in original space (length = n).
    Returns midpoint (lb+ub)/2 for finite bounds; leaves zeros for non-finite.
    """
    norm_ub = np.asarray(problem.upper_bounds, dtype=float)
    norm_lb = np.asarray(problem.lower_bounds, dtype=float)

    # Ensure proper shape
    if norm_lb.ndim > 1:
        norm_lb = norm_lb.flatten()
    if norm_ub.ndim > 1:
        norm_ub = norm_ub.flatten()

    # Ensure length matches ambient dimension
    n = U.shape[0]
    center_vec = np.zeros(n, dtype=float)

    # If bounds are scalars, broadcast
    if norm_lb.size == 1:
        norm_lb = np.full(n, norm_lb.item(), dtype=float)
    if norm_ub.size == 1:
        norm_ub = np.full(n, norm_ub.item(), dtype=float)

    # Truncate or pad with infinities if sizes mismatch
    if norm_lb.size != n:
        if norm_lb.size > n:
            norm_lb = norm_lb[:n]
        else:
            pad = np.full(n - norm_lb.size, np.inf, dtype=float)
            norm_lb = np.concatenate([norm_lb, pad])
    if norm_ub.size != n:
        if norm_ub.size > n:
            norm_ub = norm_ub[:n]
        else:
            pad = np.full(n - norm_ub.size, -np.inf, dtype=float)
            norm_ub = np.concatenate([norm_ub, pad])

    mask = np.isfinite(norm_lb) & np.isfinite(norm_ub)
    center_vec[mask] = 0.5 * (norm_lb[mask] + norm_ub[mask])
    return center_vec

def grassmann_trajectory( problem, U_flat, Delta_flat, t):
    d = U_flat.reshape(problem.dim, -1).shape[1]
    Delta = Delta_flat.reshape(-1, d)
    U = U_flat.reshape(-1, d)
    Y, s, ZT = scipy.linalg.svd(Delta, full_matrices = False, lapack_driver = 'gesvd')
    UZ = np.dot(U, ZT.T)
    U_new = np.dot(UZ, np.diag(np.cos(s*t))) + np.dot(Y, np.diag(np.sin(s*t)))

    U_new, R = np.linalg.qr(U_new, mode = 'reduced') 
    U_new = np.dot(U_new, np.diag(np.sign(np.diag(R))))
    U_new = U_new.flatten()
    return U_new

def fit_varpro( problem: Problem, X: np.ndarray, fX: np.ndarray, U0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, d = U0.shape

    def gn_solver(J_flat, r):
        # Replace NaNs/infs with large finite numbers
        J_flat = np.nan_to_num(J_flat, nan=0.0, posinf=1e8, neginf=-1e8)
        r = np.nan_to_num(r, nan=0.0, posinf=1e8, neginf=-1e8)

        U, s, VT = scipy.linalg.svd(J_flat, full_matrices=False)
        # Truncate tiny singular values
        s_inv = np.array([1/si if si > 1e-12 else 0.0 for si in s])
        Delta_flat = -VT.T @ np.diag(s_inv) @ (U.T @ r)
        return Delta_flat, s

    jacobian = lambda U_flat: varpro_jacobian(X, fX, U_flat)
    residual = lambda U_flat: varpro_residual(X, fX, U_flat)

    trajectory = lambda U_flat, Delta_flat, t: grassmann_trajectory(problem, U_flat, Delta_flat, t)

    U0_flat = U0.flatten() 
    print(f'U0_flat in fit_varpro is {U0_flat}')
    U_flat = gauss_newton(residual, jacobian, U0_flat, trajectory, gn_solver) 
    
    # iterations = iterations
    U = U_flat.reshape(n, d)

    U, coef = finish(problem, X, fX, U)	

    return U, coef

def varpro_residual( X: np.ndarray, fX: np.ndarray, U_flat: np.ndarray) -> np.ndarray:
    print(f'U_flat in varpro_residual is {U_flat}')
    U = U_flat.reshape(X.shape[1],-1)
    n, d = U.shape
    #V = V(X, U)
    # basis = Basis(degree, Y)
    V_matrix = V(X, U, d)
    # c = scipy.linalg.lstsq(V_matrix, fX)[0].flatten()
    # Regularization parameter
    lambda_reg = 1e-8  

    # Solve (V^T V + lambda I) c = V^T fX
    VtV = V_matrix.T @ V_matrix + lambda_reg * np.eye(V_matrix.shape[1])
    Vtf = V_matrix.T @ fX
    c = np.linalg.solve(VtV, Vtf)

    r = fX - V_matrix.dot(c)
    return r

def varpro_jacobian(X: np.ndarray, fX: np.ndarray, U_flat: np.ndarray, tol=1e-12) -> np.ndarray:
    M, n = X.shape
    U = U_flat.reshape(X.shape[1], -1)
    n, d = U.shape

    V_matrix = V(X, U, d)
    DV_matrix = DV(X, U, d)

    # Solve for c with Tikhonov regularization
    lambda_reg = 1e-8
    VtV = V_matrix.T @ V_matrix + lambda_reg * np.eye(V_matrix.shape[1])
    Vtf = V_matrix.T @ fX
    c = np.linalg.solve(VtV, Vtf)

    r = fX - V_matrix.dot(c)

    # Compute SVD of V_matrix safely
    Y, s, ZT = scipy.linalg.svd(V_matrix, full_matrices=False)
    # Truncate tiny singular values to avoid division by zero
    # s_inv = np.array([1/si if si > tol else 0.0 for si in s])
    s_inv = np.where(s > 1e-12, 1/s, 0.0)

    J1 = np.zeros((M, n, d))
    J2 = np.zeros((V_matrix.shape[1], n, d))

    for ell in range(d):
        for k in range(n):
            DVDU_k = X[:, k, None] * DV_matrix[:, :, ell]
            J1[:, k, ell] = DVDU_k.dot(c)
            J2[:, k, ell] = DVDU_k.T.dot(r)

    # Project against the range of V
    J1 -= np.tensordot(Y, np.tensordot(Y.T, J1, (1, 0)), (1, 0))

    # Apply truncated pseudoinverse
    s_inv = np.where(s > 1e-12, 1/s, 0.0)
    J2 = np.tensordot(np.diag(s_inv), np.tensordot(ZT, J2, (1, 0)), (1, 0))

    J = -(J1 + np.tensordot(Y, J2, (1, 0)))
    if np.any(np.isnan(J)) or np.any(np.isinf(J)):
        print("NaNs detected in varpro_jacobian")
        print("J2", J2)
        print("s_inv", s_inv)
    return J.reshape(J.shape[0], -1)

# def varpro_jacobian( X: np.ndarray, fX: np.ndarray, U_flat: np.ndarray) -> np.ndarray:
#     # print(f'U_flat in varpro_jacobian is {U_flat}')
#     M, n = X.shape
#     U = U_flat.reshape(X.shape[1],-1)
#     n, d = U.shape

#     Y = X @ U
#     # basis = Basis(degree, Y)
#     V_matrix = V(X, U, d)
#     DV_matrix = DV(X, U, d)

#     # c = scipy.linalg.lstsq(V_matrix, fX)[0].flatten()
#     # Regularization parameter
#     lambda_reg = 1e-8  

#     # Solve (V^T V + lambda I) c = V^T fX
#     VtV = V_matrix.T @ V_matrix + lambda_reg * np.eye(V_matrix.shape[1])
#     Vtf = V_matrix.T @ fX
#     c = np.linalg.solve(VtV, Vtf)
#     Y, s, ZT = scipy.linalg.svd(V_matrix, full_matrices = False)
#     s = np.array([np.inf if x == 0.0 else x for x in s]) 
#     r = fX - V_matrix.dot(c)
#     N = V_matrix.shape[1]
#     J1 = np.zeros((M,n,d))
#     J2 = np.zeros((N,n,d))
#     for ell in range(d):
#         for k in range(n):
#             DVDU_k = X[:,k,None]*DV_matrix[:,:,ell]
            
#             # This is the first term in the VARPRO Jacobian minus the projector out fron
#             J1[:, k, ell] = DVDU_k.dot(c)
#             # This is the second term in the VARPRO Jacobian before applying V^-
#             J2[:, k, ell] = DVDU_k.T.dot(r) 

#     # Project against the range of V
#     J1 -= np.tensordot(Y, np.tensordot(Y.T, J1, (1,0)), (1,0))
#     # Apply V^- by the pseudo inverse
#     J2 = np.tensordot(np.diag(1./s),np.tensordot(ZT, J2, (1,0)), (1,0))
#     J = -( J1 + np.tensordot(Y, J2, (1,0)))
#     return J.reshape(J.shape[0], -1)


def finish( problem: Problem, X: np.ndarray, fX: np.ndarray, U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r""" Given final U, rotate and find coefficients
    """

    # Step 1: Apply active subspaces to the profile function at samples X
    # to rotate onto the most important directions
    if U.shape[1] > 1:
        coef = fit_coef(problem, X, fX, U)
        grads = grad(X, coef, U)
        # We only need the short-form SVD
        grads_r = grads @ U
        Ur = scipy.linalg.svd(grads_r.T, full_matrices = False)[0]
        U = U @ Ur

    # Step 2: Flip signs such that average slope is positive in the coordinate directions
    coef = fit_coef(problem, X, fX, U)
    grads = grad(X, coef, U)
    grads_r = grads @ U    # shape (M, d)
    signs = np.sign(np.mean(grads_r, axis=0))
    U = U * signs.reshape(1, -1)
    # U = U * np.sign(np.mean(grads, axis=0))[np.newaxis, :]

    
    # Step 3: final fit	
    coef = fit_coef(problem, X, fX, U)

    ### DEBUG CHECK ###
    grads = grad(X, coef, U)
    C = (grads.T @ grads) / grads.shape[0]
    svals = np.linalg.svd(C, compute_uv=False)
    # print("\n--- Gradient covariance diagnostics ---")
    # print(f"Top 10 singular values: {svals[:10]}")
    # print("--- End gradient covariance diagnostics ---\n")
    ### END DEBUG CHECK ###

    return U, coef


#! === GAUSS-NEWTON METHOD === 
def linesearch_armijo( f, g, p, x0,  trajectory):
    bt_factor=0.5
    ftol=1e-4
    maxiter= 100
    
    dg = np.inner(g, p)
    assert dg <= 0, 'Descent direction p is not a descent direction: p^T g = %g >= 0' % (dg, )
    iterations = 0

    alpha = 1

    
    fx0 = f(x0)

    fx0_norm = np.linalg.norm(fx0)
    x = np.copy(x0)
    fx = np.inf
    success = False
    for it in range(maxiter):
        try:
            iterations += 1
            x = trajectory(x0, p, alpha)
            fx = f(x)
            fx_norm = np.linalg.norm(fx)
            if fx_norm < fx0_norm + alpha * ftol * dg:
                success = True
                break
        except BadStep:
            pass
            
        alpha *= bt_factor

    # If we haven't found a good step, stop
    if not success:
        alpha = 0
        x = x0
        fx = fx0
    return x, alpha, fx, iterations


def gauss_newton(f: callable, F: callable, x0: np.ndarray, trajectory: callable, gnsolver: callable) -> np.ndarray:
    tol=1e-10
    tol_normdx=1e-12
    maxiter=10000

    n = x0.shape[0]
    if maxiter <= 0: return x0, 4
    iterations = 0

    linesearch = linesearch_armijo

    x0 = np.nan_to_num(x0, nan=0.0, posinf=1e8, neginf=-1e8)
    x = np.copy(x0)
    f_eval = f(x)
    F_eval = F(x)
    grad = F_eval.T @ f_eval

    normgrad = np.linalg.norm(grad)
    info = None
    #rescale tol by norm of initial gradient
    tol = max(tol*normgrad, 1e-14)

    normdx = 1
    for it in range(maxiter):
        residual_increased = False
        
        print(f'F_eval arg being passed to gnsolver in gauss_newton is: \n {F_eval} ')
        # Compute search direction
        dx, s = gnsolver(F_eval, f_eval)
        
        # Check we got a valid search direction
        if not np.all(np.isfinite(dx)):
            raise RuntimeError("Non-finite search direction returned") 
        
        # If Gauss-Newton step is not a descent direction, use -gradient instead
        if np.inner(grad, dx) >= 0:
            dx = -grad
        
        # Back tracking line search
        x_new, alpha, f_eval_new, iterations_linesearch = linesearch(f, grad, dx, x, trajectory)
        iterations += iterations_linesearch

        normf = np.linalg.norm(f_eval_new)	
        if np.linalg.norm(f_eval_new) >= np.linalg.norm(f_eval):
            residual_increased = True
        else:
            #f_eval = f(x)
            f_eval = f_eval_new
            x = x_new

            normdx = np.linalg.norm(dx)
            F_eval = F(x)
            grad = F_eval.T @ f_eval_new
        
        # Termination conditions
        normgrad = np.linalg.norm(grad)
        if normgrad < tol or normdx < tol_normdx or residual_increased:
            break

    if normgrad > tol and normdx > tol_normdx and it > maxiter-1 and np.linalg.norm(f_eval_new) < np.linalg.norm(f_eval):
        raise Exception('Stopping criteria not determined!')

    return x

def full_index_set(n, d):
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
        II = full_index_set(n, d-1)
        m = II.shape[0]
        I = np.hstack((np.zeros((m, 1)), II))
        for i in range(1, n+1):
            II = full_index_set(n-i, d-1)
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
        II = full_index_set(i, d)
        I = np.vstack((I, II))
    return I[:,::-1].astype(int)

def _build_Dmat() -> np.ndarray:
    """ Constructs the (scalar) derivative matrix
    """
    degree = int(2) #self.factors['poynomial degree']
    Dmat = np.zeros( (degree+1, degree))
    I = np.eye(degree + 1)
    for j in range(degree + 1):
        Dmat[j,:] = chebder(I[:,j])

    return Dmat

def set_scale(X):
    r""" default scaling to [-1,1]
    """
    lb = np.min(X, axis = 0)
    ub = np.max(X, axis = 0)
    return lb, ub

def scale(X):
    """Scale each column of X to [-1, 1]."""
    lb, ub = np.min(X, axis=0), np.max(X, axis=0)
    # Prevent division by zero
    denom = np.where(ub - lb == 0, 1.0, ub - lb)
    X_scaled = 2*(X - lb) / denom - 1
    return X_scaled

def _dscale(X: np.ndarray) -> np.ndarray:
    r""" returns the scaling associated with the scaling transform
    """
    lb, ub = np.min(X, axis = 0),  np.max(X, axis = 0)
    try:
        return (2./(ub - lb))
    except AttributeError:
        raise NotImplementedError

def V(X, U, dim):
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
        Vandermonde matrix of shape (M, N) where M is the number of desired points and N is the number of Basis Elements
    """
    degree = 2 #int(self.factors['polynomial degree'])
    
    if len(X.shape) == 1 : 
        X = X.reshape(1,-1)

    indices = index_set(degree, dim).astype(int)
    #Project Y
    Y = X @ U
    Y = scale(np.array(Y))
    M = Y.shape[0]
    assert Y.shape[1] == dim, "Expected %d dimensions, got %d" % (dim, Y.shape[1])
    V_coordinate = [chebvander(Y[:,k], degree) for k in range(dim)]

    V = np.ones((M, len(indices)), dtype = Y.dtype)
    
    for j, alpha in enumerate(indices):
        for k in range(dim):
            V[:,j] *= V_coordinate[k][:,alpha[k]]
    return V

def VC(X, U, c, dim):
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
    """
    degree = 2 #int(self.factors['polynomial degree'])
    if dim is None or X is None:
        raise NotImplementedError
    
    Y = X @ U
    Y = scale(np.array(Y))
    M = Y.shape[0]
    c = np.array(c)
    indices = index_set(degree, dim).astype(int)
    assert len(indices) == c.shape[0]

    if len(c.shape) == 2:
        oneD = False
    else:
        c = c.reshape(-1,1)
        oneD = True

    V_coordinate = [chebvander(Y[:,k], degree) for k in range(dim)]
    out = np.zeros((M, c.shape[1]))	
    for j, alpha in enumerate(indices):

        # If we have a non-zero coefficient
        if np.max(np.abs(c[j,:])) > 0.:
            col = np.ones(M)
            for ell in range(dim):
                col *= V_coordinate[ell][:,alpha[ell]]

            for k in range(c.shape[1]):
                out[:,k] += c[j,k]*col
    if oneD:
        out = out.flatten()
    return out

def DV(X, U, dim):
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
    degree = 2 #int(self.factors['polynomial degree'])
    Dmat = _build_Dmat()
    indices = index_set(degree, dim).astype(int)
    if len(X.shape) == 1:
        X = X.reshape(1,-1)
    # X = X.reshape(-1, self.dim)
    #Project X
    Y = X @ U
    Y = scale(np.array(Y))
    M = Y.shape[0]
    V_coordinate = [chebvander(Y[:,k], degree) for k in range(dim)]
    
    N = len(indices)
    DV = np.ones((M, N, dim), dtype = Y.dtype)

    try:
        dscale = _dscale(Y)
    except NotImplementedError:
        dscale = np.ones(Y.shape[1])	


    for k in range(dim):
        for j, alpha in enumerate(indices):
            for q in range(dim):
                if q == k:
                    DV[:,j,k] *= np.dot(V_coordinate[q][:,0:-1], Dmat[alpha[q],:])
                else:
                    DV[:,j,k] *= V_coordinate[q][:,alpha[q]]
        # Correct for transform
        DV[:,:,k] *= dscale[k] 		

    return DV

def DDV(X, dim):
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
    degree = 2 #int(self.factors['polynomial degree'])
    Dmat = _build_Dmat()
    indices = index_set(degree, dim).astype(int)
    if len(X.shape) == 1:
        X = X.reshape(1,-1)
    # X = X.reshape(-1, self.dim)
    # X = self.scale(np.array(X))
    M = X.shape[0]
    V_coordinate = [chebvander(X[:,k], degree) for k in range(dim)]
    
    N = len(indices)
    DDV = np.ones((M, N, dim, dim), dtype = X.dtype)

    try:
        dscale = _dscale(X)
    except NotImplementedError:
        dscale = np.ones(X.shape[1])	


    for k in range(dim):
        for ell in range(k, dim):
            for j, alpha in enumerate(indices):
                for q in range(dim):
                    if q == k == ell:
                        # We need the second derivative
                        eq = np.zeros(degree+1)
                        eq[alpha[q]] = 1.
                        der2 = chebder(eq, 2)
                        DDV[:,j,k,ell] *= V_coordinate[q][:,0:len(der2)].dot(der2)
                    elif q == k or q == ell:
                        DDV[:,j,k,ell] *= np.dot(V_coordinate[q][:,0:-1], Dmat[alpha[q],:])
                    else:
                        DDV[:,j,k,ell] *= V_coordinate[q][:,alpha[q]]

            # Correct for transform
            DDV[:,:,k, ell] *= dscale[k]*dscale[ell]
            DDV[:,:,ell, k] = DDV[:,:,k, ell]
    return DDV

def roots(X, coef, dim):
    lb,ub = np.min(X, axis = 0), np.max(X, axis = 0)
    if dim > 1:
        raise NotImplementedError
    r = chebroots(coef)
    return r*(ub[0] - lb[0])/2.0 + (ub[0] + lb[0])/2.


def main() :
    # Instantiate example problem (f(x) = ||x||^2)
    problem = instantiate_problem("ROSENBROCK-1")
    dim = len(problem.lower_bounds)

    # Initial point (center of local region)
    x0 = np.zeros((dim,1))
    current_solution = Solution(tuple(x0), problem)

    # Model construction parameters
    delta_k = 0.5       # trust region radius
    k = 0
    expended_budget = 0
    visited_points_list = [current_solution]

    # Construct model and active subspace
    coef, current_solution, delta_k, fval, expended_budget, interpolation_sols, U, visited_pts = construct_model(
        problem, current_solution, delta_k, k, expended_budget, visited_points_list
    )

    # Evaluate model performance over random samples in the ball
    n_test = 200
    X_test, f_true, f_model = [], [], []

    obj_fn = lambda x : np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

    for _ in range(n_test):
        direction = np.random.randn(dim).reshape(-1,1)
        direction /= np.linalg.norm(direction)
        radius = np.random.rand() * delta_k
        x = x0 + radius * direction #shape (dim,1)
        X_test.append(x)
        f_true.append(obj_fn(x))
        f_model.append(model_evaluate(problem, np.array(x).reshape(-1,dim), U, coef)[0])

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