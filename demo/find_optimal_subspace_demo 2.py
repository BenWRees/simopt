"""
Demo script for finding the optimal subspace dimension for ASTROMoRF.

This script demonstrates how to use the find_best_subspace_dimension function
to automatically determine the best subspace dimension for a given problem.
"""

import sys
import os.path as o

sys.path.append(
    o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), ".."))
)

from simopt.directory import problem_directory
from simopt.solvers.active_subspaces.compute_optimal_dim import (
    find_optimal_d,
    find_best_subspace_dimension,
    compute_optimal_polynomial_degree
)


def demo_gradient_based_method():
    """
    Demonstrate the fast gradient-based method for estimating subspace dimension.
    
    This method uses eigenanalysis of gradient covariance to quickly estimate
    the intrinsic dimensionality of the problem.
    """
    print("=" * 80)
    print("DEMO 1: Fast Gradient-Based Dimension Estimation")
    print("=" * 80)
    
    # Create a problem instance
    problem = problem_directory['DYNAMNEWS-1'](
        fixed_factors={'budget': 1000}
    )
    
    print(f"\nProblem: {problem.name}")
    print(f"Full dimension: {problem.dim}")
    
    # Find optimal dimension using gradient-based method
    print("\nComputing optimal subspace dimension using gradient covariance...")
    optimal_d = find_optimal_d(problem)
    
    print(f"\nEstimated optimal subspace dimension: {optimal_d}")
    
    # Compute optimal polynomial degree for this dimension
    optimal_degree = compute_optimal_polynomial_degree(optimal_d)
    print(f"Recommended polynomial degree: {optimal_degree}")
    
    return optimal_d


def demo_experiment_based_method(quick_test: bool = True):
    """
    Demonstrate the comprehensive experiment-based method for finding best dimension.
    
    This method runs actual solver experiments with different dimensions and
    evaluates both variance stability and trajectory quality.
    
    Args:
        quick_test: If True, use reduced settings for faster demonstration
    """
    print("\n" + "=" * 80)
    print("DEMO 2: Comprehensive Experiment-Based Dimension Selection")
    print("=" * 80)
    
    # Create a problem instance
    problem = problem_directory['DYNAMNEWS-1'](
        fixed_factors={'budget': 600}
    )
    
    print(f"\nProblem: {problem.name}")
    print(f"Full dimension: {problem.dim}")
    
    if quick_test:
        print("\n[QUICK TEST MODE: Using reduced settings for demonstration]")
        n_macroreps = 3  # Reduced from 10 for quick testing
        n_postreps = 20  # Reduced from 100 for quick testing
        candidate_dims = None  # Will auto-generate around estimated value
    else:
        print("\n[FULL MODE: Using production settings]")
        n_macroreps = 10
        n_postreps = 100
        candidate_dims = None
    
    # Run comprehensive evaluation
    print("\nRunning experiments to find best subspace dimension...")
    print("This will test multiple dimensions and evaluate:")
    print("  1. Variance in solutions across macroreplications")
    print("  2. Quality of terminal solutions")
    print("  3. Trajectory stability")
    
    result = find_best_subspace_dimension(
        problem=problem,
        solver_name='ASTROMORF',
        candidate_dimensions=candidate_dims,
        n_macroreps=n_macroreps,
        n_postreps=n_postreps,
        variance_threshold=0.1,
        min_improvement=0.05,
        solver_fixed_factors={'crn_across_solns': False}
    )
    
    # Display results
    print("\n" + "-" * 80)
    print("RESULTS")
    print("-" * 80)
    
    best_d = result['best_dimension']
    print(f"\n✓ Best subspace dimension: {best_d}")
    
    print(f"\nCandidate dimensions tested: {result['candidate_dimensions']}")
    
    print("\nVariance scores (lower is better):")
    for d in result['candidate_dimensions']:
        variance = result['variance_scores'][d]
        print(f"  d={d}: {variance:.6f}")
    
    print("\nQuality scores (terminal objective, lower is better for minimization):")
    for d in result['candidate_dimensions']:
        quality = result['quality_scores'][d]
        print(f"  d={d}: {quality:.6f}")
    
    print("\n" + "-" * 80)
    print(f"\nRECOMMENDATION: Use subspace dimension d={best_d} for ASTROMoRF on {problem.name}")
    print("-" * 80)
    
    return result


def demo_custom_dimension_testing():
    """
    Demonstrate testing specific custom dimensions.
    
    Useful when you want to compare specific dimension values based on
    domain knowledge or previous experiments.
    """
    print("\n" + "=" * 80)
    print("DEMO 3: Testing Custom Dimension Candidates")
    print("=" * 80)
    
    problem = problem_directory['ROSENBROCK-1'](
        fixed_factors={'budget': 400},
        model_fixed_factors={'variance': 5.0}
    )
    
    print(f"\nProblem: {problem.name}")
    print(f"Full dimension: {problem.dim}")
    
    # Specify custom dimensions to test
    custom_dims = [2, 4, 6, 8]
    print(f"\nTesting custom dimensions: {custom_dims}")
    
    # Quick test with reduced replications
    result = find_best_subspace_dimension(
        problem=problem,
        solver_name='ASTROMORF',
        candidate_dimensions=custom_dims,
        n_macroreps=3,  # Quick test
        n_postreps=20,  # Quick test
        variance_threshold=0.15,
        solver_fixed_factors={'crn_across_solns': False}
    )
    
    best_d = result['best_dimension']
    print(f"\n✓ Best among tested dimensions: {best_d}")
    
    return result


def compare_methods():
    """
    Compare the fast gradient-based method with experiment-based method.
    """
    print("\n" + "=" * 80)
    print("COMPARISON: Gradient-Based vs. Experiment-Based Methods")
    print("=" * 80)
    
    problem = problem_directory['NETWORK-1'](
        fixed_factors={'budget': 500}
    )
    
    print(f"\nProblem: {problem.name}")
    print(f"Full dimension: {problem.dim}")
    
    # Method 1: Fast gradient-based
    print("\n1. Fast gradient-based estimation...")
    import time
    start = time.time()
    d_gradient = find_optimal_d(problem)
    time_gradient = time.time() - start
    print(f"   Result: d={d_gradient}")
    print(f"   Time: {time_gradient:.2f} seconds")
    
    # Method 2: Experiment-based (quick version)
    print("\n2. Experiment-based selection (quick test)...")
    start = time.time()
    # Test dimensions around the gradient-based estimate
    test_dims = [max(1, d_gradient - 1), d_gradient, min(d_gradient + 1, problem.dim)]
    result_experiment = find_best_subspace_dimension(
        problem=problem,
        candidate_dimensions=test_dims,
        n_macroreps=3,
        n_postreps=20,
        solver_fixed_factors={'crn_across_solns': False}
    )
    time_experiment = time.time() - start
    d_experiment = result_experiment['best_dimension']
    print(f"   Result: d={d_experiment}")
    print(f"   Time: {time_experiment:.2f} seconds")
    
    # Summary
    print("\n" + "-" * 80)
    print("SUMMARY")
    print("-" * 80)
    print(f"\nGradient-based method:")
    print(f"  - Fast: {time_gradient:.2f}s")
    print(f"  - Good initial estimate: d={d_gradient}")
    print(f"  - Use for quick analysis or dimension estimation")
    
    print(f"\nExperiment-based method:")
    print(f"  - Slower: {time_experiment:.2f}s")
    print(f"  - More reliable: d={d_experiment}")
    print(f"  - Considers variance and trajectory quality")
    print(f"  - Use for final dimension selection")
    
    if d_gradient == d_experiment:
        print(f"\n✓ Both methods agree: d={d_gradient}")
    else:
        print(f"\n⚠ Methods differ: gradient={d_gradient}, experiment={d_experiment}")
        print("  Consider using experiment-based result for production runs")


def main():
    """Run all demonstrations."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Demonstrate methods for finding optimal subspace dimension"
    )
    parser.add_argument(
        '--demo',
        type=int,
        choices=[1, 2, 3, 4],
        help='Run specific demo (1-4), or all if not specified'
    )
    parser.add_argument(
        '--full',
        action='store_true',
        help='Use full settings (slower but more accurate)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.demo == 1 or args.demo is None:
            demo_gradient_based_method()
        
        if args.demo == 2 or args.demo is None:
            demo_experiment_based_method(quick_test=not args.full)
        
        if args.demo == 3 or args.demo is None:
            demo_custom_dimension_testing()
        
        if args.demo == 4 or args.demo is None:
            compare_methods()
        
        print("\n" + "=" * 80)
        print("ALL DEMOS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
