import os.path as o  # noqa: D100
import random
import sys

import numpy as np

from simopt.experiment_base import instantiate_problem
from simopt.solvers.active_subspaces.compute_optimal_dim import (
    find_best_polynomial_degree,
    find_best_subspace_dimension,
)

sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))  # noqa: PTH100, PTH118, PTH120


def update_model_factors_dimensions(problem_name: str, new_dim: int) -> dict:  # noqa: D417
    """Find the model associated with model_name and update the dimension of the factor.

    values to the new_dim.

            Return the updated model factors.

    Args:
            model_name (str): Name of Model
            new_dim (int): New dimension for the model

    Returns:
            dict: Updated model factors
    """
    new_factors = {}
    if problem_name == "DYNAMNEWS-1":
        new_factors = {
            "num_prod": new_dim,
            "c_utility": [6 + j for j in range(new_dim)],
            "init_level": [3] * new_dim,
            "price": [9] * new_dim,
            "cost": [5] * new_dim,
        }
    elif problem_name == "FACSIZE-1" or problem_name == "FACSIZE-2":
        A = np.random.rand(new_dim, new_dim)  # noqa: N806
        new_factors = {
            "mean_vec": [500] * new_dim,
            "cov": (np.dot(A, A.T) * 100).tolist(),
            "capacity": [random.randint(100, 900) for _ in range(new_dim)],
            "n_fac": new_dim,
        }
    elif problem_name == "FIXEDSAN-1":
        new_factors = {}
    elif problem_name == "AIRLINE-1" or problem_name == "AIRLINE-2":
        num_classes = random.randint(2, new_dim // 2)
        odf_leg_matrix = np.random.randint(0, 2, (new_dim, num_classes))
        new_factors = {
            "num_classes": num_classes,
            "ODF_leg_matrix": odf_leg_matrix.tolist(),
            "prices": tuple([random.randint(50, 300) for _ in range(new_dim)]),
            "capacity": tuple([random.randint(20, 150) for _ in range(num_classes)]),
            "booking limits": tuple([random.randint(5, 20) for _ in range(new_dim)]),
            "alpha": tuple([random.uniform(0, 5) for _ in range(new_dim)]),
            "beta": tuple([random.uniform(2, 10) for _ in range(new_dim)]),
            "gamma_shape": tuple([random.uniform(2, 10) for _ in range(new_dim)]),
            "gamma_scale": tuple([random.uniform(10, 50) for _ in range(new_dim)]),
        }
    elif problem_name == "NETWORK-1":
        process_prob_elem = 1 / new_dim
        mode_transit_time = [
            round(np.random.uniform(0.01, 5), 3) for _ in range(new_dim)
        ]
        lower_limits_transit_time = [x / 2 for x in mode_transit_time]
        upper_limits_transit_time = [2 * x for x in mode_transit_time]
        new_factors = {
            "process_prob": [process_prob_elem] * new_dim,
            "cost_process": [0.1 / (x + 1) for x in range(new_dim)],
            "cost_time": [round(np.random.uniform(0.01, 1), 3) for _ in range(new_dim)],
            "mode_transit_time": mode_transit_time,
            "lower_limits_transit_time": lower_limits_transit_time,
            "upper_limits_transit_time": upper_limits_transit_time,
            "n_networks": new_dim,
        }
    elif problem_name == "CONTAM-2":
        new_factors = {
            "stages": new_dim,
            "prev_decision": (0,) * new_dim,
        }

    return new_factors


def update_problem_factor_dimensions(  # noqa: D417
    problem_name: str, new_dim: int, budget: int
) -> dict:
    """Update the dimension of the factor values in problem_factors to the new_dim.

            Return the updated problem factors.

    Args:
            problem_factors (dict): Problem factors to be updated
            new_dim (int): New dimension for the problem factors

    Returns:
            dict: Updated problem factors
    """
    new_factors = {}
    if problem_name == "DYNAMNEWS-1":
        new_factors = {
            "initial_solution": (3,) * new_dim,
            "budget": budget,
        }
    elif problem_name == "FACSIZE-1":
        new_factors = {
            "initial_solution": (100,) * new_dim,
            "installation_costs": (1,) * new_dim,
            "epsilon": 0.05,
            "budget": budget,
        }
    elif problem_name == "FACSIZE-2":
        new_factors = {
            "initial_solution": (300,) * new_dim,
            "installation_costs": (1,) * new_dim,
            "installation_budget": 500.0,
            "budget": budget,
        }
    elif problem_name == "FIXEDSAN-1":
        new_factors = {
            "budget": budget,
        }
    elif problem_name == "AIRLINE-1" or problem_name == "AIRLINE-2":
        new_factors = {
            "initial_solution": (3,) * new_dim,
            "budget": budget,
        }
    elif problem_name == "NETWORK-1":
        init_soln_elem = 1 / new_dim
        new_factors = {
            "initial_solution": (init_soln_elem,) * new_dim,
            "budget": budget,
        }
    elif problem_name == "CONTAM-2":
        new_factors = {
            "initial_solution": (1,) * new_dim,
            "prev_cost": [1] * new_dim,
            "error_prob": [0.2] * new_dim,
            "upper_thres": [0.1] * new_dim,
            "budget": budget,
        }
    return new_factors


def main(  # noqa: D103
    problem_name: str,
    n_macroreps: int = 5,
    budget: int = 1000,
    new_dim: int | None = None,
) -> None:
    # Update problem and model factors based on new_dim
    if new_dim is not None:
        problem_factors = update_problem_factor_dimensions(
            problem_name, new_dim, budget
        )
        model_factors = update_model_factors_dimensions(problem_name, new_dim)
    else:
        problem_factors = {"budget": budget}
        model_factors = {}

    # problem = problem_directory[problem_name](fixed_factors=problem_factors, model_factors=model_factors)
    problem = instantiate_problem(
        problem_name,
        problem_fixed_factors=problem_factors,
        model_fixed_factors=model_factors,
    )

    # Find optimal subspace dimension
    print("\n" + "=" * 70)
    print("FINDING OPTIMAL SUBSPACE DIMENSION")
    print(f"Problem: {problem_name} with dimension {problem.dim}")
    print("=" * 70)

    result = find_best_subspace_dimension(
        problem=problem,
        n_macroreps=n_macroreps,
        consistency_weight=0.3,
        quality_weight=0.7,
    )

    optimal_dimension = result["optimal_dimension"]
    print(f"\n{'=' * 70}")
    print(f"OPTIMAL SUBSPACE DIMENSION: {optimal_dimension}")
    print(f"{'=' * 70}")
    print(f"Statistics: {result['statistics'][optimal_dimension]}")
    print("\nAll dimensions tested:")
    for dim, stats in result["statistics"].items():
        print(f"  Dim {dim}: {stats}")

    # Find optimal polynomial degree using the optimal subspace dimension
    print("\n\n" + "=" * 70)
    print("FINDING OPTIMAL POLYNOMIAL DEGREE")
    print(f"Using optimal subspace dimension: {optimal_dimension}")
    print("=" * 70)

    degree_result = find_best_polynomial_degree(
        problem=problem,
        subspace_dimension=optimal_dimension,
        n_macroreps=n_macroreps,
        min_degree=1,
        max_degree=6,
        success_weight=0.4,
        quality_weight=0.6,
    )

    optimal_degree = degree_result["optimal_degree"]
    print(f"\n{'=' * 70}")
    print(f"OPTIMAL POLYNOMIAL DEGREE: {optimal_degree}")
    print(f"{'=' * 70}")
    print(f"Statistics: {degree_result['statistics'][optimal_degree]}")
    print("\nAll degrees tested:")
    for deg, stats in degree_result["statistics"].items():
        print(f"  Degree {deg}: {stats}")

    # Print final summary
    print("\n\n" + "=" * 70)
    print("FINAL RECOMMENDATIONS")
    print("=" * 70)
    print(f"Problem: {problem_name}")
    print(f"Optimal Subspace Dimension: {optimal_dimension}")
    print(f"Optimal Polynomial Degree: {optimal_degree}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    # EXAMPLE usage: python optimal_subspace_cli.py DYNAMNEWS-1 --n_macroreps 10 --budget 600 --new_dim 10

    parser = argparse.ArgumentParser(
        description="Compute optimal active subspace dimension for a given problem."
    )
    parser.add_argument(
        "problem_name",
        type=str,
        help="Name of the problem to analyze (e.g., 'DYNAMNEWS-1').",
    )

    parser.add_argument(
        "--n_macroreps",
        type=int,
        default=5,
        help="Number of macroreplications to run for each dimension.",
    )

    parser.add_argument(
        "--budget", type=int, default=1000, help="Budget constraint for the problem."
    )

    parser.add_argument(
        "--new_dim",
        type=int,
        default=None,
        help="New dimension for the problem factors.",
    )

    args = parser.parse_args()
    main(
        problem_name=args.problem_name,
        n_macroreps=args.n_macroreps,
        budget=args.budget,
        new_dim=args.new_dim,
    )
