"""
    Functions to check for the criticality step of the model. 
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


def criticality_test(k, problem, current_solution, f_x, delta, U, visited_pts, X, fX, interpolation_solutions, expended_budget) : 
      if norm(self.grad(X,coef,U)) < tol : 
        if delta <= self.factors['mu'] * norm(self.grad(X,coef,U)) and self.fully_linear_test(X,fX, coef, U, delta, tol) :
            break 
        else : 
            X, fX, interpolation_solutions, visited_pts, delta, expended_budget = self.improve_geometry(k, problem, current_solution, f_x, delta, U, visited_pts, X, fX, interpolation_solutions, expended_budget)



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


if __name__ == '__main__' : 
    main()