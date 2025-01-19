"""Base Module that provides Linear Algebra Functions that are commonly used:
    - Finite Differencing Calculation 
    - Fast Matrix Multiplication for vandermonde matrix and vector of coefficients 
"""

import numpy as np 

from simopt.base import Solution, Problem, Solver
from .solvers.active_subspaces.basis import Basis

__all__ = ['finite_difference_gradient', 'matrix_multiplication']

def finite_difference_gradient(solver: Solver, new_solution: Solution, problem: Problem, alpha: float = 1.0e-8, BdsCheck: None | np.ndarray = None) -> np.ndarray :
    """Calculates a gradient approximation of the solution on the problem

    :math:
        `\nabla F(x_n) = \frac{F(x_n+\alpha)-F(x_n-\alpha)}{2\alpha}
        \nabla F(x_n) = \frac{F(x_n+\alpha)-F(x_n)}{2\alpha}
        \nabla F(x_n) = \frac{F(x_n)-F(x_n-\alpha)}{2\alpha}`

    where :math:`x_n` is new_solution and :math:`F` is the objective function of the problem 

    Args:
        new_solution (Solution): The value for where the gradient is 
        problem (Problem): The simulation-optimisation problem being solved
        alpha (float): The perturbation size. Defaults to 1.0e-8
        BdsCheck (None | np.ndarray, optional): _description_. Defaults to None.

    Returns:
        np.ndarray: The output of the problem's gradient evaluated at new_solution
    """
    lower_bound = problem.lower_bounds
    upper_bound = problem.upper_bounds
    # grads = np.zeros((problem.dim,r)) #Take r gradient approximations
    problem.simulate(new_solution,1)
    fn = -1 * problem.minmax[0] * new_solution.objectives_mean

    #if BdsCheck isn't implemented then we assume central finite differencing 
    if BdsCheck is None : 
        BdsCheck =  np.zeros(problem.dim)
    
    new_x = new_solution.x
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

        fn1, fn2 = 0,0 
        x1_solution = solver.create_new_solution(tuple(x1), problem)
        if BdsCheck[i] != -1:
            problem.simulate(x1_solution)
            fn1 = -1 * problem.minmax[0] * x1_solution.objectives_mean
            # First column is f(x+h,y).
            FnPlusMinus[i, 0] = fn1
        x2_solution = solver.create_new_solution(tuple(x2), problem)
        if BdsCheck[i] != 1:
            problem.simulate(x2_solution)
            fn2 = -1 * problem.minmax[0] * x2_solution.objectives_mean
            # Second column is f(x-h,y).
            FnPlusMinus[i, 1] = fn2

        # Calculate gradient.
        if BdsCheck[i] == 0:
            grad[i] = (fn1 - fn2) / (2 * FnPlusMinus[i, 2])
        elif BdsCheck[i] == 1:
            grad[i] = (fn1 - fn) / FnPlusMinus[i, 2]
        elif BdsCheck[i] == -1:
            grad[i] = (fn - fn2) / FnPlusMinus[i, 2]


    return grad 


def matrix_multiplication(X: np.ndarray, c: np.ndarray, basis: Basis = None) -> np.ndarray:
    """Does fast matrix multiplication between a matrix and a column vector

    Args:
        X (np.ndarray): A matrix of shape (M,n). The convention is M independently drawn samples of vectors with dimension n
        c (array-like): A list of values with length n that is shaped to be a column vector of shape (n,1)

    Returns:
        np.ndarray: A vector of shape (M,1) which represents :math:`Xc`
    """
    #in the case where X is not being used to construct a Vandermonde Matrix
    if basis is None : 
        try :
            res = np.dot(X,c)
        except : #If dimension exceptions are thrown handle it 
            pass 

        return res

    X = X.reshape(-1, basis.dim)
    X = basis.scale(np.array(X))
    M = X.shape[0]
    c = np.array(c)
    assert len(basis.indices) == c.shape[0]

    if len(c.shape) == 2:
        oneD = False
    else:
        c = c.reshape(-1,1)
        oneD = True

    V_coordinate = [basis.vander(X[:,k], basis.degree) for k in range(basis.dim)]
    out = np.zeros((M, c.shape[1]))	
    for j, alpha in enumerate(basis.indices):

        # If we have a non-zero coefficient
        if np.max(np.abs(c[j,:])) > 0.:
            col = np.ones(M)
            for ell in range(basis.dim):
                col *= V_coordinate[ell][:,alpha[ell]]

            for k in range(c.shape[1]):
                out[:,k] += c[j,k]*col
    if oneD:
        out = out.flatten()
    return out