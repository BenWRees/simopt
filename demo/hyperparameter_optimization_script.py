"""
Simple script demonstrating hyperparameter optimization for ASTROMORF.

This script shows how to use solve_hyperparameter_optimization() to find
optimal hyperparameters using Bayesian optimization in a clean, straightforward way.
"""

from simopt.directory import problem_directory
from simopt.solvers.astromorf import solve_hyperparameter_optimization
from simopt.experiment_base import ProblemSolver


def main():
    """Run hyperparameter optimization and use the result."""
    
    # Step 1: Create your target problem
    print("Setting up target problem...")
    problem = problem_directory['DYNAMNEWS-1'](fixed_factors={'budget': 600})
    
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
        verbose=True
    )
    
    # Step 3: Extract optimal hyperparameters
    optimal_dim, optimal_deg = result['best_solution']
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Optimal hyperparameters found:")
    print(f"  Subspace dimension: {optimal_dim}")
    print(f"  Polynomial degree: {optimal_deg}")
    print(f"\nConfigurations evaluated: {result['n_evaluations']}")
    print(f"All evaluated solutions: {result['all_evaluated_solutions'][:10]}...")
    
    # Step 4: Use optimal hyperparameters to solve the target problem
    print(f"\n{'='*70}")
    print(f"RUNNING ASTROMORF WITH OPTIMAL HYPERPARAMETERS")
    print(f"{'='*70}\n")
    
    optimal_solver_factors = {
        'initial subspace dimension': optimal_dim,
        'polynomial degree': optimal_deg
    }
    
    final_experiment = ProblemSolver(
        solver_name="ASTROMORF",
        problem=problem,
        solver_fixed_factors=optimal_solver_factors
    )
    
    # Run solver with optimal hyperparameters
    final_experiment.run(n_macroreps=5)
    final_experiment.post_replicate(
        n_postreps=50, 
        crn_across_budget=True, 
        crn_across_macroreps=False
    )
    
    # Display final results
    print(f"\nFinal results using optimal hyperparameters:")
    for mrep in range(5):
        if len(final_experiment.all_est_objectives[mrep]) > 0:
            final_obj = final_experiment.all_est_objectives[mrep][-1]
            print(f"  Macroreplication {mrep+1}: {final_obj:.6f}")
    
    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")
    
    # Optional: Access the full hyperparameter optimization experiment
    # for more detailed analysis
    hyperopt_experiment = result['hyperopt_experiment']
    print(f"\nHyperparameter optimization experiment available at:")
    print(f"  result['hyperopt_experiment']")
    print(f"  result['hyperopt_problem']")
    
    return result


if __name__ == "__main__":
    result = main()
