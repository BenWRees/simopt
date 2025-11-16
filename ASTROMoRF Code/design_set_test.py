"""
    This module contains functions to construct the interpolation set for the ASTRO-MORF algorithm using active subspaces.
    It includes methods for reusing previously evaluated design points within the trust region and generating rotated basis vectors
    This is a test to model how design sets are constructed in each iteration of ASTRO-MoRF with active subspaces.
"""
import sys
import os.path as o
sys.path.append(
	o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), ".."))
)  #

from simopt.base import Problem, Solution
import numpy as np
from numpy.linalg import norm

from simopt.experiment_base import instantiate_problem


def column_vectors_U(index: int, U: np.ndarray) -> np.ndarray:
    """
    Get the index column vector of U. The column vectors are orthonormal basis vectors that span the active subspace. 

    Args:
        problem (Problem): The SO problem 
        index (int): The index of the column vector
        U (np.ndarray): The active subspace matrix

    Returns:
        np.ndarray: The n-dimensional column vector at the given index
    """
    col_vector = U[:, index].reshape(-1,1)
    return col_vector

def interpolation_points_without_reuse(problem: Problem, x_k: np.ndarray, delta: float, U: np.ndarray) -> list[np.ndarray]:
    """
    Constructs an interpolation set without reusing points
    
    Args:
        problem (Problem): The current simulation model
        x_k (np.ndarray): The current incumbent solution
        delta (float): The current trust-region radius
        U (np.ndarray): The (n,d) active subspace matrix
    
    Returns:
        [np.array]: A list of (2d+1) n-dimensional design points for interpolation
    """

    Y = [x_k]
    epsilon = 0.01
    d = U.shape[1]
    #build the basis that spans the trust region in the projected space 
    for i in range(0, d):
        plus = Y[0] + delta * column_vectors_U(i, U)
        minus = Y[0] - delta * column_vectors_U(i, U)

        if sum(x_k) != 0: #check if x_k is not the origin
            # block constraints
            if minus[i] <= problem.lower_bounds[i]:
                minus[i] = problem.lower_bounds[i] + epsilon
            if plus[i] >= problem.upper_bounds[i]:
                plus[i] = problem.upper_bounds[i] - epsilon

        Y.append(plus)
        Y.append(minus)

    return Y 


# generate the mutually orthonormal rotated basis using A_k1 as the first basis vector
def get_rotated_basis(A_k1: np.ndarray, U: np.ndarray) -> list[np.ndarray]:
    """
    Generate the other d-1 rotated coordinate basis using A_k1 as the first basis vector. 
    We use Gram-Schmidt process to generate the orthonormal basis.
    Args:
        A_k1 (np.ndarray): The first direction vector for the reused design point
        d (int): The subspace dimension and the number of vectors to have 
        U (np.ndarray): The (n,d) active subspace matrix

    Returns:
        list[np.ndarray]: A list of d d-dimensional rotated basis vectors each with shape (d,1)
    """

    # Start with A_normalized as first vector
    basis = [A_k1]
    d = U.shape[1]
    # Generate candidate vectors from the standard basis
    I = np.eye(d)
    candidates = [I[:,i].reshape(-1,1) for i in range(1,d)]
    


    #Build successive orthononormal basis using Gram-Schmidt process from A_k1
    for c in candidates:
        v = c.copy()
        #calculate gram-schmidt projection
        for b in basis:
            dot_prod = v.T @ b
            v -= dot_prod.item() * b 

        #Normalize v
        v = v / np.linalg.norm(v)

        basis.append(v.reshape(-1,1))

        if len(basis) == d:
            break

    return basis

# compute the interpolation points (2d+1) using the rotated coordinate basis (reuse one design point)
def interpolation_points_with_reuse(problem: Problem, x_k: np.ndarray, reused_x: np.ndarray, delta: float, rotation_vectors: list[np.ndarray], U: np.ndarray) -> list[np.ndarray]:
    Y = [x_k, reused_x]
    epsilon = 0.01
    d = U.shape[1]


    for i in range(1, d):
        plus = Y[0] + delta * (U @ rotation_vectors[i])

        #block constraints
        for j in range(problem.dim) :
            if plus[j] <= problem.lower_bounds[j]:
                plus[j] = problem.lower_bounds[j] + epsilon
            if plus[j] >= problem.upper_bounds[j]:
                plus[j] = problem.upper_bounds[j] - epsilon

        Y.append(plus)

    for i in range(d):
        minus = Y[0] - delta * (U @ rotation_vectors[i])
        
        # block constraints
        for j in range(problem.dim):
            if minus[j] <= problem.lower_bounds[j]:
                minus[j] = problem.lower_bounds[j] + epsilon
            if minus[j] >= problem.upper_bounds[j]:
                minus[j] = problem.upper_bounds[j] - epsilon
        Y.append(minus)
                    

    return Y 


#! This is the only sample set construction method that gets called 
def construct_interpolation_set(current_solution: Solution, problem: Problem, U: np.ndarray, delta_k: float, k: int, visited_pts_list: list[Solution]) -> tuple[list[np.ndarray], int] : 
    x_k = np.array(current_solution.x).reshape(-1,1) #current solution as n-dim vector
    Dist = []
    for i in range(len(visited_pts_list)):
        dist_of_pt = norm(U.T @ (np.array(visited_pts_list[i].x).reshape(-1,1) - x_k))
        
        # If the design point is outside the trust region, then make sure it isn't considered for reuse
        if dist_of_pt <= delta_k:
            Dist.append(dist_of_pt) 
        else : 
            Dist.append(-1)

    # Find the index of visited design points list for reusing points
    # The reused point will be the farthest point from the center point among the design points within the trust region
    f_index = Dist.index(max(Dist))

    # If it is the first iteration or there is no design point we can reuse within the trust region, use the coordinate basis
    if (k == 1) or (norm(x_k - np.array(visited_pts_list[f_index].x))==0) :
        Y = interpolation_points_without_reuse(problem, x_k, delta_k, U)

    # Else if we will reuse one design point
    elif k > 1 :
        reused_pt = np.array(visited_pts_list[f_index].x).reshape(-1,1)
        print(f'Reusing point at index {f_index} with coordinates {reused_pt.flatten()}')
        diff_array = U.T @ (reused_pt - x_k) #has shape (d,1)
        A_k1 = (diff_array) / norm(diff_array) #has shape (d,1)
        P_k = norm(diff_array)

        # rotate_list = np.nonzero(A_k1)[0] #! gets the row indices of the non-zero components of A_k1 
        d = U.shape[1]
        rotate_matrix: list[np.ndarray] = get_rotated_basis(A_k1, U)

        # construct the interpolation set
        Y = interpolation_points_with_reuse(problem, x_k, reused_pt, delta_k, rotate_matrix, U)
    return np.vstack([v.ravel() for v in Y]), f_index



def main() : 
    problem = instantiate_problem("EXAMPLE-1")
    solution =  Solution((2.0,2.0), problem)

    visited_pts_list = [solution]
    delta = 1.0

    # Go through iterations, and after each iteration, update t
    for iter in range(1,6) :
        print(f'The visited points list before iteration {iter}: ')
        print([sol.x for sol in visited_pts_list])
        print(f'The trust region radius delta_{iter}: {delta}')
        design_set, reused_index = construct_interpolation_set(current_solution=solution, problem=problem, U=np.array([[0.7071],[-0.7071]]), delta_k=delta, k=iter, visited_pts_list=visited_pts_list)

        print(f"Design Set for iteration {iter}: ")
        print(design_set)
        print(f"Reused Index for iteration {iter}: ")
        print(reused_index)

        delta = 1.5*delta
        visited_pts_list.append(solution)
        #create a new solution by choosing a point from the design set 
        new_x = tuple(design_set[iter % design_set.shape[0]].flatten())
        solution = Solution(new_x, problem)

        print(f'New Solution for iteration {iter}: ')
        print(solution.x)
        print("--------------------------------------------------")


if __name__ == "__main__":
    main()