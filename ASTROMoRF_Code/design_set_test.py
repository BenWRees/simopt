"""This module contains functions to construct the interpolation set for the ASTRO-MORF.

algorithm using active subspaces.


It includes methods for reusing previously evaluated design points within the trust
region and generating rotated basis vectors
This is a test to model how design sets are constructed in each iteration of ASTRO-MoRF
with active subspaces.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(sys.modules[__name__].__file__).parent.parent.resolve()))  #

import numpy as np
from numpy.linalg import norm

from simopt.base import Problem, Solution


def clamp_with_epsilon(
    val: float, lower_bound: float, upper_bound: float, epsilon: float = 0.01
) -> float:
    """Clamp a value within bounds while avoiding exact boundary values.

    Adds a small epsilon to the lower bound or subtracts it from the upper bound
    if `val` lies outside the specified range.

    Args:
            val (float): The value to clamp.
            lower_bound (float): Minimum acceptable value.
            upper_bound (float): Maximum acceptable value.
            epsilon (float, optional): Small margin to avoid returning exact boundary
                    values. Defaults to 0.01.

    Returns:
            float: The adjusted value, guaranteed to lie strictly within the bounds.
    """
    if val <= lower_bound:
        return lower_bound + epsilon
    if val >= upper_bound:
        return upper_bound - epsilon
    return val


def column_vectors_U(index: int, U: np.ndarray) -> np.ndarray:  # noqa: N802, N803
    """Get the index column vector of U. The column vectors are orthonormal basis.

    vectors that span the active subspace.

    Args:
            problem (Problem): The SO problem
            index (int): The index of the column vector
            U (np.ndarray): The active subspace matrix

    Returns:
            np.ndarray: The n-dimensional column vector at the given index
    """
    return U[:, index].reshape(-1, 1)


def interpolation_points_without_reuse(  # noqa: D417
    current_solution: Solution,
    problem: Problem,
    U: np.ndarray,  # noqa: N803
    delta: float,
) -> list[np.ndarray]:
    """Constructs a 2d+1 interpolation set without reusing points.

    Points placed at adaptively computed radius for optimal coverage of typical
    candidate locations.

    Args:
            problem (Problem): The current simulation model
            U (np.ndarray): The (n,d) active subspace matrix

    Returns:
            [np.array]: A list of 2d+1 n-dimensional design points for interpolation
    """
    _n, d = U.shape
    x_k = np.array(current_solution.x).reshape(-1, 1)
    Y = [x_k]  # noqa: N806
    lower_bounds = problem.lower_bounds
    upper_bounds = problem.upper_bounds

    # Adaptively compute interpolation radius based on problem characteristics
    interpolation_radius = delta

    for i in range(0, d):
        plus = Y[0] + interpolation_radius * column_vectors_U(i, U)
        minus = Y[0] - interpolation_radius * column_vectors_U(i, U)

        plus = plus.flatten().tolist()
        minus = minus.flatten().tolist()

        if sum(x_k) != 0:
            minus = [
                clamp_with_epsilon(val, lower_bounds[j], upper_bounds[j])
                for j, val in enumerate(minus)
            ]
            plus = [
                clamp_with_epsilon(val, lower_bounds[j], upper_bounds[j])
                for j, val in enumerate(plus)
            ]

            minus = np.array(minus).reshape(-1, 1)
            plus = np.array(plus).reshape(-1, 1)

        Y.append(plus)
        Y.append(minus)

    return Y


# generate the mutually orthonormal rotated basis using A_k1 as the first basis vector
def get_rotated_basis(A_k1: np.ndarray, U: np.ndarray) -> list[np.ndarray]:  # noqa: N803
    """Generate the other d-1 rotated coordinate basis using A_k1 as the first basis.

    vector.

    We use Gram-Schmidt process to generate the orthonormal basis.

    Args:
            A_k1 (np.ndarray): The first direction vector for the reused design point
            d (int): The subspace dimension and the number of vectors to have
            U (np.ndarray): The (n,d) active subspace matrix

    Returns:
            list[np.ndarray]: A list of d d-dimensional rotated basis vectors each with
            shape (d,1)
    """
    _n, d = U.shape
    # Start with A_normalized as first vector
    basis = [A_k1]
    # Generate candidate vectors from the standard basis
    I = np.eye(d)  # noqa: E741, N806
    candidates = [I[:, i].reshape(-1, 1) for i in range(1, d)]

    # Build successive orthononormal basis using Gram-Schmidt process from A_k1
    for c in candidates:
        v = c.copy()
        # calculate gram-schmidt projection
        for b in basis:
            dot_prod = v.T @ b
            v -= dot_prod.item() * b

        # Normalize v
        v = v / np.linalg.norm(v)
        basis.append(v.reshape(-1, 1))

        if len(basis) == d:
            break

    return basis


# compute the interpolation points (2d+1) using the rotated coordinate basis (reuse one design point)
def interpolation_points_with_reuse(  # noqa: D417
    current_solution: Solution,
    problem: Problem,
    reused_x: np.ndarray,
    rotation_vectors: list[np.ndarray],
    U: np.ndarray,  # noqa: N803
    delta: float,
) -> list[np.ndarray]:
    """Constructs a 2d+1 interpolation set with reusing one design point.

            Points placed at adaptively computed radius for optimal coverage of typical
            candidate locations.

    Args:
            problem (Problem): The current simulation model
            x_k (np.ndarray): The current incumbent solution
            reused_x (np.ndarray): The design point to be reused
            delta (float): The current trust-region radius
            rotation_vectors (list[np.ndarray]): The rotated coordinate basis vectors
            U (np.ndarray): The (n,d) active subspace matrix

    Returns:
            list[np.ndarray]: A list of 2d+1 n-dimensional design points for
            interpolation
    """
    _n, d = U.shape
    x_k = np.array(current_solution.x).reshape(-1, 1)
    Y = [x_k, reused_x]  # noqa: N806

    # Adaptively compute interpolation radius based on problem characteristics
    interpolation_radius = delta

    for i in range(1, d):
        plus = Y[0] + interpolation_radius * (U @ rotation_vectors[i])
        plus = plus.flatten().tolist()

        # block constraints
        if sum(x_k) != 0:
            plus = [
                clamp_with_epsilon(
                    val, problem.lower_bounds[j], problem.upper_bounds[j]
                )
                for j, val in enumerate(plus)
            ]
            plus = np.array(plus).reshape(-1, 1)

        Y.append(plus)

    for i in range(d):
        minus = Y[0] - interpolation_radius * (U @ rotation_vectors[i])
        minus = minus.flatten().tolist()

        # block constraints
        if sum(x_k) != 0:
            minus = [
                clamp_with_epsilon(
                    val, problem.lower_bounds[j], problem.upper_bounds[j]
                )
                for j, val in enumerate(minus)
            ]
            minus = np.array(minus).reshape(-1, 1)

        Y.append(minus)

    return Y


def construct_interpolation_set(  # noqa: D417
    k: int,
    current_solution: Solution,
    problem: Problem,
    U: np.ndarray,  # noqa: N803
    delta: float,
    visited_pts: list[Solution],
) -> tuple[list[np.ndarray], int]:
    """Constructs the interpolation set either by reusing one design point from the.

    visited points list or not reusing any design points.

            This is the only method that is called to build the interpolation set.

    Args:
            current_solution (Solution): The current incumbent solution
            problem (Problem): The current simulation model
            U (np.ndarray): The (n,d) active subspace matrix
            delta (float): The current trust-region radius
            k (int): The current iteration number
            visited_pts_list (list[Solution]): The list of previously visited solutions

    Returns:
            tuple[list[np.ndarray], int]: A tuple containing the list of interpolation
            points and the index of the reused point
    """
    _n, _d = U.shape
    x_k = np.array(current_solution.x).reshape(
        -1, 1
    )  # current solution as n-dim vector
    Dist = []  # noqa: N806

    try:
        for i in range(len(visited_pts)):
            dist_of_pt = norm(U.T @ (np.array(visited_pts[i].x).reshape(-1, 1) - x_k))

            # If the design point is outside the trust region, then make sure it isn't considered for reuse
            if dist_of_pt <= delta:
                Dist.append(dist_of_pt)
            else:
                Dist.append(-1)

        # Find the index of visited design points list for reusing points
        # The reused point will be the farthest point from the center point among the design points within the trust region
        f_index = Dist.index(max(Dist))
    except Exception:
        #! This error is usually thrown as self.visited_points is None or empty
        visited_pts = [current_solution]
        f_index = 0

    # If it is the first iteration or there is no design point we can reuse within the trust region, use the coordinate basis
    if (k == 1) or norm(x_k - np.array(visited_pts[f_index].x).reshape(-1, 1)) == 0:
        Y = interpolation_points_without_reuse(current_solution, problem, U, delta)  # noqa: N806

    # Else if we will reuse one design point
    elif k > 1:
        reused_pt = np.array(visited_pts[f_index].x).reshape(-1, 1)
        diff_array = U.T @ (reused_pt - x_k)  # has shape (d,1)
        A_k1 = (diff_array) / norm(diff_array)  # has shape (d,1)  # noqa: N806
        norm(diff_array)

        rotate_matrix: list[np.ndarray] = get_rotated_basis(A_k1, U)

        # construct the interpolation set
        Y = interpolation_points_with_reuse(  # noqa: N806
            current_solution, problem, reused_pt, rotate_matrix, U, delta
        )
    return np.vstack([v.ravel() for v in Y]), f_index


"""
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
		design_set, reused_index = construct_interpolation_set(current_solution=solution,
		problem=problem, U=np.array([[0.7071],[-0.7071]]), delta_k=delta, k=iter,
		visited_pts_list=visited_pts_list)

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
"""
