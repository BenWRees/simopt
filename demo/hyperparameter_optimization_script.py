"""Simple script demonstrating hyperparameter search for ASTROMORF."""

from typing import cast

from simopt.base import Problem
from simopt.directory import problem_directory
from simopt.experiment_base import ProblemSolver
from simopt.solvers.active_subspaces.compute_optimal_dim import (
    find_best_polynomial_degree,
    find_best_subspace_dimension,
)


def solve_hyperparameter_optimization(
    target_problem: Problem,
    max_dimension: int,
    min_dimension: int,
    min_degree: int,
    max_degree: int,
    consistency_weight: float,
    quality_weight: float,
    solver_name: str,
    n_macroreps: int,
    hyperopt_solver: str,
    hyperopt_budget: int,
    verbose: bool,
) -> dict[str, object]:
    """Compatibility wrapper that performs dimension then degree selection.

    The previous demo used a removed helper. This wrapper keeps the same call
    shape and returns a similar result dictionary.
    """
    _ = (hyperopt_solver, hyperopt_budget)

    dim_result = find_best_subspace_dimension(
        problem=target_problem,
        solver_name=solver_name,
        n_macroreps=n_macroreps,
        max_dimension=max_dimension,
        min_dimension=min_dimension,
        consistency_weight=consistency_weight,
        quality_weight=quality_weight,
        verbose=verbose,
    )
    best_dim = int(dim_result["optimal_dimension"])

    degree_result = find_best_polynomial_degree(
        problem=target_problem,
        subspace_dimension=best_dim,
        solver_name=solver_name,
        n_macroreps=n_macroreps,
        min_degree=min_degree,
        max_degree=max_degree,
        verbose=verbose,
    )
    best_degree = int(degree_result["optimal_degree"])

    all_evaluated_solutions = [
        ("subspace", d) for d in dim_result.get("all_dimensions", [])
    ] + [("degree", p) for p in degree_result.get("all_degrees", [])]

    return {
        "best_solution": (best_dim, best_degree),
        "n_evaluations": len(all_evaluated_solutions),
        "all_evaluated_solutions": all_evaluated_solutions,
        "hyperopt_experiment": None,
        "hyperopt_problem": target_problem,
        "dimension_search": dim_result,
        "degree_search": degree_result,
    }


def main():  # noqa: ANN201
    """Run hyperparameter optimization and use the result."""
    # Step 1: Create your target problem
    print("Setting up target problem...")
    problem = problem_directory["DYNAMNEWS-1"](fixed_factors={"budget": 600})

    # Step 2: Run hyperparameter optimization
    print("\nRunning Bayesian optimization for hyperparameters...")
    result = solve_hyperparameter_optimization(
        target_problem=problem,
        max_dimension=problem.dim,
        min_dimension=1,
        min_degree=1,
        max_degree=6,
        consistency_weight=0.2,
        quality_weight=0.8,
        solver_name="ASTROMORF",
        n_macroreps=3,  # Lower for faster demo
        hyperopt_solver="ASTRODF",  # Use Bayesian optimization
        hyperopt_budget=20,  # Number of configurations to try
        verbose=True,
    )

    # Step 3: Extract optimal hyperparameters
    optimal_dim, optimal_deg = cast(tuple[int, int], result["best_solution"])
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")
    print("Optimal hyperparameters found:")
    print(f"  Subspace dimension: {optimal_dim}")
    print(f"  Polynomial degree: {optimal_deg}")
    print(f"\nConfigurations evaluated: {cast(int, result['n_evaluations'])}")
    print(
        "All evaluated solutions: "
        f"{cast(list[tuple[str, int]], result['all_evaluated_solutions'])[:10]}..."
    )

    # Step 4: Use optimal hyperparameters to solve the target problem
    print(f"\n{'=' * 70}")
    print("RUNNING ASTROMORF WITH OPTIMAL HYPERPARAMETERS")
    print(f"{'=' * 70}\n")

    optimal_solver_factors = {
        "initial subspace dimension": optimal_dim,
        "polynomial degree": optimal_deg,
    }

    final_experiment = ProblemSolver(
        solver_name="ASTROMORF",
        problem=problem,
        solver_fixed_factors=optimal_solver_factors,
    )

    # Run solver with optimal hyperparameters
    final_experiment.run(n_macroreps=5)
    final_experiment.post_replicate(
        n_postreps=50, crn_across_budget=True, crn_across_macroreps=False
    )

    # Display final results
    print("\nFinal results using optimal hyperparameters:")
    for mrep in range(5):
        if len(final_experiment.all_est_objectives[mrep]) > 0:
            final_obj = final_experiment.all_est_objectives[mrep][-1]
            print(f"  Macroreplication {mrep + 1}: {final_obj:.6f}")

    print(f"\n{'=' * 70}")
    print("COMPLETE")
    print(f"{'=' * 70}")

    # Optional: Access the full hyperparameter optimization experiment
    # for more detailed analysis
    result["hyperopt_experiment"]
    print("\nHyperparameter optimization experiment available at:")
    print("  result['hyperopt_experiment']")
    print("  result['hyperopt_problem']")

    return result


if __name__ == "__main__":
    result = main()
