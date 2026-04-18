"""Module for computing optimal subspace dimensions and polynomial degrees for.

ASTROMORF.

This module provides functions to:
1. Determine the optimal polynomial degree based on available sample budget
2. Select the best subspace dimension for consistent and high-quality optimization
trajectories
3. Perform 2D hyperparameter search to find optimal (dimension, degree) pairs
4. Solve hyperparameter tuning as a simulation optimization problem using Bayesian
optimization
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from simopt.base import Problem


def _test_single_polynomial_degree(args):  # noqa: ANN001, ANN202
    """Worker function to test a single polynomial degree in parallel.

    Must be at module level for pickling with ProcessPoolExecutor.

    Args:
        args: Tuple of (degree, problem, subspace_dimension, solver_name, n_macroreps,
        verbose)

    Returns:
        Tuple of (degree, statistics) or None if failed
    """
    degree, problem, subspace_dimension, solver_name, n_macroreps, verbose = args

    import os
    import re

    from simopt.experiment_base import ProblemSolver

    print(f"\n{'=' * 60}")
    print(f"Testing polynomial degree: {degree}")
    print(f"{'=' * 60}")

    if verbose:
        logging.info(f"\n{'=' * 60}")
        logging.info(f"Testing polynomial degree: {degree}")
        logging.info(f"{'=' * 60}")

    # Create solver with this degree
    solver_factors = {"polynomial degree": degree, "Record Diagnostics": True}
    if subspace_dimension is not None:
        solver_factors["initial subspace dimension"] = subspace_dimension

    # Create ProblemSolver instance
    experiment = ProblemSolver(
        solver_name=solver_name, problem=problem, solver_fixed_factors=solver_factors
    )

    # Run macroreplications
    try:
        experiment.run(n_macroreps=n_macroreps)

        # Post-replicate to get objective estimates
        experiment.post_replicate(
            n_postreps=50, crn_across_budget=True, crn_across_macroreps=False
        )

        # Parse diagnostics files to extract success rates and model quality
        success_rates = []
        r_squared_values = []
        final_objectives = []

        for mrep in range(n_macroreps):
            # Get final objective
            if len(experiment.all_est_objectives[mrep]) > 0:
                final_obj = experiment.all_est_objectives[mrep][-1]
                if not np.isnan(final_obj) and not np.isinf(final_obj):
                    final_objectives.append(final_obj)

        # Try to read diagnostics file for this experiment
        # Diagnostics files are created during the run with iteration-level info
        diag_dir = os.path.join(os.getcwd(), "Diagnostics")  # noqa: PTH109, PTH118
        if os.path.exists(diag_dir):  # noqa: PTH110
            # Look for diagnostics files matching this problem
            diag_files = [
                f
                for f in os.listdir(diag_dir)  # noqa: PTH208
                if f.startswith(f"astromorf_diagnostics_{problem.name}")
            ]

            # Parse diagnostics files for this experiment
            for diag_file in sorted(diag_files)[-n_macroreps:]:
                try:
                    with open(os.path.join(diag_dir, diag_file)) as f:  # noqa: PTH118, PTH123
                        content = f.read()

                        # Look for: "The number of successful iterations was X and the number of unsuccessful iterations was Y"
                        success_match = re.search(
                            r"The number of successful iterations was (\d+) and the number of unsuccessful iterations was (\d+)",
                            content,
                        )
                        if success_match:
                            successful = int(success_match.group(1))
                            unsuccessful = int(success_match.group(2))
                            total = successful + unsuccessful
                            if total > 0:
                                success_rates.append(successful / total)

                        # Extract R² values - look for: "R² (goodness of fit):     0.9234"
                        r2_matches = re.findall(
                            r"R²\s*\(goodness of fit\):\s*([\d.]+)", content
                        )
                        if r2_matches:
                            r_squared_values.extend([float(r2) for r2 in r2_matches])
                except Exception as e:
                    if verbose:
                        logging.debug(
                            f"Could not parse diagnostics file {diag_file}: {e!s}"
                        )
                    continue

        # If we couldn't parse diagnostics, use a simpler metric based on final objectives
        if len(success_rates) == 0 or len(r_squared_values) == 0:
            if verbose:
                logging.warning(
                    f"Could not parse diagnostics for degree {degree}, using simplified metrics"
                )

            # Use coefficient of variation as a proxy for success
            # Lower CV suggests more stable/successful optimization
            if len(final_objectives) > 1:
                cv = np.std(final_objectives, ddof=1) / abs(np.mean(final_objectives))
                # Convert CV to a success rate (lower CV = higher success)
                success_rate = max(0.0, 1.0 - min(1.0, cv))
                success_rates = [success_rate]

                # Use a moderate R² estimate
                r_squared_values = [0.75]  # Assume moderate quality if we can't measure

        if len(final_objectives) == 0:
            if verbose:
                logging.warning(f"No valid results for degree {degree}, skipping")
            return None

        # Calculate statistics
        mean_success_rate = np.mean(success_rates) if len(success_rates) > 0 else 0.5
        mean_r_squared = np.mean(r_squared_values) if len(r_squared_values) > 0 else 0.5
        mean_obj = np.mean(final_objectives)
        std_obj = np.std(final_objectives, ddof=1)

        statistics = {
            "success_rate": mean_success_rate,
            "mean_r_squared": mean_r_squared,
            "mean_objective": mean_obj,
            "std_objective": std_obj,
            "n_successful": len(final_objectives),
        }

        if verbose:
            logging.info(f"  Success rate: {mean_success_rate:.2%}")
            logging.info(f"  Mean R²: {mean_r_squared:.4f}")
            logging.info(f"  Mean objective: {mean_obj:.4f}")
            logging.info(f"  Std objective: {std_obj:.4f}")

        # Return results as tuple (degree, statistics)
        return (degree, statistics)

    except Exception as e:
        if verbose:
            logging.error(f"Error testing degree {degree}: {e!s}")
        return None


def find_best_polynomial_degree(
    problem: Problem,
    subspace_dimension: int | None = None,
    solver_name: str = "ASTROMORF",
    n_macroreps: int = 10,
    min_degree: int = 1,
    max_degree: int = 8,
    success_weight: float = 0.65,
    quality_weight: float = 0.35,
    verbose: bool = True,
) -> dict:
    """Find the optimal polynomial degree for ASTROMORF that achieves the most.

    successful.

    iterations and best model quality across multiple macroreplications.

    This function evaluates different polynomial degrees by running the solver
    multiple times (macroreplications) and measuring:
    1. Success rate: Percentage of successful iterations vs unsuccessful
    2. Model quality: Average R² (goodness of fit) across iterations
    3. Final solution quality: Mean final objective value

    Args:
        problem (Problem): The simulation-optimization problem to solve
        subspace_dimension (int | None): Subspace dimension to use. If None, uses
        default
        solver_name (str): Name of the solver to use (default: "ASTROMORF")
        n_macroreps (int): Number of macroreplications to run per degree (default: 10)
        min_degree (int): Minimum polynomial degree to test (default: 1)
        max_degree (int): Maximum polynomial degree to test (default: 4)
        success_weight (float): Weight for success rate metric (0-1, default: 0.5)
        quality_weight (float): Weight for model quality metric (0-1, default: 0.5)
        verbose (bool): Whether to print progress information (default: True)

    Returns:
        dict: Dictionary containing:
            - 'optimal_degree' (int): The recommended polynomial degree
            - 'all_degrees' (list[int]): All tested degrees
            - 'success_scores' (list[float]): Success rate score for each degree
            - 'quality_scores' (list[float]): Model quality score for each degree
            - 'combined_scores' (list[float]): Combined weighted score for each degree
            - 'statistics' (dict): Detailed statistics for each degree including:
                - 'success_rate': Percentage of successful iterations
                - 'mean_r_squared': Average R² across all iterations
                - 'mean_objective': Mean final objective value
                - 'std_objective': Std dev of final objectives

    Example:
        >>> from simopt.directory import problem_directory
        >>> problem = problem_directory['DYNAMNEWS-1'](fixed_factors={'budget': 600})
        >>> result = find_best_polynomial_degree(problem, subspace_dimension=3,
        n_macroreps=5)
        >>> print(f"Optimal degree: {result['optimal_degree']}")
        Optimal degree: 2

    Notes:
        - Higher degrees allow more complex models but may overfit
        - Lower degrees are more robust but may underfit
        - Success rate measures iteration efficiency
        - Model quality (R²) measures how well the polynomial fits the data
        - The function balances both metrics with configurable weights
    """
    # Validate inputs
    if success_weight < 0 or quality_weight < 0:
        raise ValueError("Weights must be non-negative")
    if abs(success_weight + quality_weight - 1.0) > 1e-6:
        raise ValueError("Weights must sum to 1.0")
    if n_macroreps < 2:
        raise ValueError("Need at least 2 macroreplications to measure consistency")
    if min_degree < 1:
        min_degree = 1
    if max_degree < min_degree:
        raise ValueError("max_degree must be >= min_degree")

    degrees_to_test = list(range(min_degree, max_degree + 1))

    if verbose:
        logging.info(f"Testing polynomial degrees: {degrees_to_test}")
        if subspace_dimension:
            logging.info(f"Using subspace dimension: {subspace_dimension}")
        logging.info(f"Running {n_macroreps} macroreplications per degree")
        logging.info(
            f"Success weight: {success_weight}, Quality weight: {quality_weight}"
        )
        logging.info(f"Parallelizing experiments across {len(degrees_to_test)} degrees")

    # Store results for each degree (process-safe with proper synchronization)
    all_statistics = {}  # degree -> statistics dict

    # Run experiments in parallel using ProcessPoolExecutor
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # Determine number of workers (limit to avoid overwhelming system)
    max_workers = 1  # max(1, int(100/n_macroreps))  # Limit to 4 parallel experiments

    if verbose:
        logging.info(f"Using {max_workers} parallel workers (processes)")

    # Prepare arguments for each degree test
    test_args = [
        (degree, problem, subspace_dimension, solver_name, n_macroreps, verbose)
        for degree in degrees_to_test
    ]

    # Submit all degree tests to process pool
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_degree = {
            executor.submit(_test_single_polynomial_degree, args): args[0]
            for args in test_args
        }

        # Collect results as they complete
        for future in as_completed(future_to_degree):
            degree = future_to_degree[future]
            try:
                result = future.result()
                if result is not None:
                    # Unpack results
                    degree_val, statistics = result

                    # Store results (process-safe as we're collecting sequentially here)
                    all_statistics[degree_val] = statistics

                    if verbose:
                        print(f"✓ Completed degree {degree_val}/{max_degree}")
            except Exception as e:
                if verbose:
                    logging.error(
                        f"Exception collecting results for degree {degree}: {e!s}"
                    )

    if verbose:
        logging.info("\nAll parallel experiments completed")
        logging.info(
            f"Successfully tested {len(all_statistics)} out of {len(degrees_to_test)} degrees"
        )

    # Check if we have any valid results
    if len(all_statistics) == 0:
        raise RuntimeError("No successful experiments completed")

    # Compute scores for each degree
    tested_degrees = sorted(all_statistics.keys())

    # Extract metrics for normalization
    # Only filter NaNs (data collection failures), not infs (will handle with epsilon)
    all_success_rates = np.array(
        [all_statistics[d]["success_rate"] for d in tested_degrees]
    )
    all_r_squared = np.array(
        [all_statistics[d]["mean_r_squared"] for d in tested_degrees]
    )
    all_mean_objectives = np.array(
        [all_statistics[d]["mean_objective"] for d in tested_degrees]
    )

    # Replace NaN values with sensible defaults (infs are ok, will be clipped)
    all_success_rates = np.nan_to_num(
        all_success_rates, nan=0.0, posinf=1.0, neginf=0.0
    )
    all_r_squared = np.nan_to_num(all_r_squared, nan=0.0, posinf=1.0, neginf=0.0)
    all_mean_objectives = np.nan_to_num(
        all_mean_objectives, nan=np.inf, posinf=np.inf, neginf=-np.inf
    )

    success_min, success_max = np.min(all_success_rates), np.max(all_success_rates)
    r2_min, r2_max = np.min(all_r_squared), np.max(all_r_squared)
    obj_min, obj_max = np.min(all_mean_objectives), np.max(all_mean_objectives)

    # Use epsilon to avoid division by zero - small perturbation is better than filtering
    eps = 1e-10
    success_range = max(success_max - success_min, eps)
    r2_range = max(r2_max - r2_min, eps)
    obj_range = max(obj_max - obj_min, eps)

    success_scores = []
    quality_scores = []
    combined_scores = []

    for degree in tested_degrees:
        # Success score: higher success rate is better
        # Normalize so that 0 = best (highest success rate)
        success_rate = all_statistics[degree]["success_rate"]
        success_rate = np.nan_to_num(success_rate, nan=0.0, posinf=1.0, neginf=0.0)
        success_score = (success_max - success_rate) / success_range

        # Quality score: combines R² (model quality) and objective value (solution quality)
        # Higher R² is better - normalize so that 0 = best (highest R²)
        r_squared = all_statistics[degree]["mean_r_squared"]
        r_squared = np.nan_to_num(r_squared, nan=0.0, posinf=1.0, neginf=0.0)
        r2_score = (r2_max - r_squared) / r2_range

        # Objective value score depends on problem type
        mean_obj = all_statistics[degree]["mean_objective"]
        mean_obj = np.nan_to_num(
            mean_obj, nan=np.inf if problem.minmax[0] == -1 else -np.inf
        )

        if problem.minmax[0] == -1:
            # Minimization: lower (more negative) mean is better
            # Normalize so that 0 = best (most negative/lowest value)
            obj_score = (mean_obj - obj_min) / obj_range
        else:
            # Maximization: higher (more positive) mean is better
            # Normalize so that 0 = best (highest value)
            obj_score = (obj_max - mean_obj) / obj_range

        # Clip scores to [0, 1] range in case of numerical issues
        success_score = np.clip(success_score, 0.0, 1.0)
        r2_score = np.clip(r2_score, 0.0, 1.0)
        obj_score = np.clip(obj_score, 0.0, 1.0)

        # Quality combines R² and objective (equal weight to each)
        quality_score = 0.5 * r2_score + 0.5 * obj_score

        # Combined score: lower is better
        combined_score = success_weight * success_score + quality_weight * quality_score

        success_scores.append(success_score)
        quality_scores.append(quality_score)
        combined_scores.append(combined_score)

        if verbose:
            logging.info(f"\nDegree {degree}:")
            logging.info(
                f"  Success score: {success_score:.4f} (Rate: {success_rate:.2%})"
            )
            logging.info(f"  Quality score: {quality_score:.4f} (R²: {r_squared:.4f})")
            logging.info(f"  Combined score: {combined_score:.4f}")

    # Find the degree with the best (lowest) combined score
    best_idx = np.argmin(combined_scores)
    optimal_degree = tested_degrees[best_idx]

    if verbose:
        logging.info(f"\n{'=' * 60}")
        logging.info(f"OPTIMAL POLYNOMIAL DEGREE: {optimal_degree}")
        logging.info(f"  Combined score: {combined_scores[best_idx]:.4f}")
        logging.info(
            f"  Success rate: {all_statistics[optimal_degree]['success_rate']:.2%}"
        )
        logging.info(
            f"  Mean R²: {all_statistics[optimal_degree]['mean_r_squared']:.4f}"
        )
        logging.info(
            f"  Mean objective: {all_statistics[optimal_degree]['mean_objective']:.4f}"
        )
        logging.info(f"{'=' * 60}")

    # Return comprehensive results
    return {
        "optimal_degree": optimal_degree,
        "all_degrees": tested_degrees,
        "success_scores": success_scores,
        "quality_scores": quality_scores,
        "combined_scores": combined_scores,
        "statistics": all_statistics,
    }


def _test_single_dimension(args):  # noqa: ANN001, ANN202
    """Worker function to test a single dimension in parallel.

    Must be at module level for pickling with ProcessPoolExecutor.

    Args:
        args: Tuple of (dim, problem, solver_name, n_macroreps, max_dimension, verbose)

    Returns:
        Tuple of (dim, final_objectives, statistics) or None if failed
    """
    dim, problem, solver_name, n_macroreps, max_dimension, verbose = args

    from simopt.experiment_base import ProblemSolver

    print(f"\n{'=' * 60}")
    print(f"Testing subspace dimension: {dim}/{max_dimension}")
    print(f"{'=' * 60}")
    if verbose:
        logging.info(f"\n{'=' * 60}")
        logging.info(f"Testing subspace dimension: {dim}/{max_dimension}")
        logging.info(f"{'=' * 60}")

    # Create solver with this dimension
    solver_factors = {"initial subspace dimension": dim}

    # Create ProblemSolver instance
    experiment = ProblemSolver(
        solver_name=solver_name, problem=problem, solver_fixed_factors=solver_factors
    )

    # Run macroreplications
    try:
        experiment.run(n_macroreps=n_macroreps)

        # Post-replicate to get objective estimates
        # This is necessary to get reliable objective values
        experiment.post_replicate(
            n_postreps=50, crn_across_budget=True, crn_across_macroreps=False
        )

        # Extract final objective values from each macroreplication
        # Use all_est_objectives which contains the post-replicated objective estimates
        final_objectives = []
        for mrep in range(n_macroreps):
            # Check if we have objective estimates for this macroreplication
            if len(experiment.all_est_objectives[mrep]) > 0:
                # Get the final (best) objective value from this macroreplication
                final_obj = experiment.all_est_objectives[mrep][-1]
                # Filter out invalid values
                if not np.isnan(final_obj) and not np.isinf(final_obj):
                    final_objectives.append(final_obj)

        if len(final_objectives) == 0:
            if verbose:
                logging.warning(f"No valid results for dimension {dim}, skipping")
            return None  # Return None for failed experiments

        # Calculate statistics
        mean_obj = np.mean(final_objectives)
        # Handle std calculation: need at least 2 values for ddof=1
        if len(final_objectives) > 1:  # noqa: SIM108
            std_obj = np.std(final_objectives, ddof=1)
        else:
            std_obj = 0.0  # Only one value, no variation

        # Calculate CV, avoiding inf values
        if abs(mean_obj) > 1e-10:  # noqa: SIM108
            cv_obj = std_obj / abs(mean_obj)
        else:
            # Mean is near zero, CV is undefined
            # Use a large but finite value instead of inf
            cv_obj = 1e10

        min_obj = np.min(final_objectives)
        max_obj = np.max(final_objectives)

        statistics = {
            "mean": mean_obj,
            "std": std_obj,
            "cv": cv_obj,
            "min": min_obj,
            "max": max_obj,
            "n_successful": len(final_objectives),
        }

        if verbose:
            logging.info(f"  Mean objective: {mean_obj:.4f}")
            logging.info(f"  Std dev: {std_obj:.4f}")
            logging.info(f"  CV: {cv_obj:.4f}")
            logging.info(f"  Range: [{min_obj:.4f}, {max_obj:.4f}]")

        # Return results as tuple (dimension, final_objectives, statistics)
        return (dim, final_objectives, statistics)

    except Exception as e:
        if verbose:
            logging.error(f"Error testing dimension {dim}: {e!s}")
        return None  # Return None for failed experiments


def find_best_subspace_dimension(
    problem: Problem,
    solver_name: str = "ASTROMORF",
    n_macroreps: int = 5,
    max_dimension: int | None = None,
    min_dimension: int = 1,
    consistency_weight: float = 0.3,
    quality_weight: float = 0.7,
    verbose: bool = True,
) -> dict:
    """Find the optimal subspace dimension for ASTROMORF that achieves consistent,.

    high-quality optimization trajectories across multiple macroreplications.

    This function evaluates different subspace dimensions by running the solver
    multiple times (macroreplications) and measuring:
    1. Consistency: How similar the final results are across runs (low variance/CV)
    2. Quality: How good the average final solution is (low mean objective)

    An optimal dimension balances these two objectives, achieving reliable
    performance (consistency) while finding good solutions (quality).

    Args:
        problem (Problem): The simulation-optimization problem to solve
        solver_name (str): Name of the solver to use (default: "ASTROMORF")
        n_macroreps (int): Number of macroreplications to run per dimension (default:
        10)
                          More macroreps give better consistency estimates but take
                          longer
        max_dimension (int | None): Maximum subspace dimension to test
                                   If None, uses problem.dim
        min_dimension (int): Minimum subspace dimension to test (default: 1)
        consistency_weight (float): Weight for consistency metric (0-1, default: 0.3)
                                   Controls importance of low std deviation across runs
        quality_weight (float): Weight for quality metric (0-1, default: 0.7)
                               Controls importance of best mean objective value
                               consistency_weight + quality_weight should equal 1.0
        verbose (bool): Whether to print progress information (default: True)

    Returns:
        dict: Dictionary containing:
            - 'optimal_dimension' (int): The recommended subspace dimension
            - 'all_dimensions' (list[int]): All tested dimensions
            - 'consistency_scores' (list[float]): Consistency score for each dimension
            - 'quality_scores' (list[float]): Quality score for each dimension
            - 'combined_scores' (list[float]): Combined weighted score for each
            dimension
            - 'final_objectives' (dict): Final objective values for each dimension
            - 'statistics' (dict): Detailed statistics for each dimension

    Example:
        >>> from simopt.directory import problem_directory
        >>> problem = problem_directory['DYNAMNEWS-1'](fixed_factors={'budget': 600})
        >>> result = find_best_subspace_dimension(problem, n_macroreps=10)
        >>> print(f"Optimal dimension: {result['optimal_dimension']}")
        Optimal dimension: 3
        >>> print(f"Score:
        {result['combined_scores'][result['optimal_dimension']-1]:.4f}")
        Score: 0.8456

    Notes:
        - Higher dimensions allow more complex models but require more samples
        - Lower dimensions are more robust but may miss important problem structure
        - Consistency is measured by standard deviation across macroreplications
        - Quality is measured by mean final objective value (normalized)
        - The function minimizes a weighted combination:
          score = consistency_weight * std_score + quality_weight * mean_score
        - By default, weights are (0.3, 0.7) to prioritize best mean objective value
        - Use higher quality_weight (e.g., 0.8) to strongly prioritize best mean
        - Use higher consistency_weight (e.g., 0.6) to prioritize reliability/robustness
    """
    # Validate inputs
    if consistency_weight < 0 or quality_weight < 0:
        raise ValueError("Weights must be non-negative")
    if abs(consistency_weight + quality_weight - 1.0) > 1e-6:
        raise ValueError("Weights must sum to 1.0")
    if n_macroreps < 2:
        raise ValueError("Need at least 2 macroreplications to measure consistency")

    # Determine dimension range
    if max_dimension is None:
        max_dimension = problem.dim

    if min_dimension < 1:
        min_dimension = 1
    if max_dimension > problem.dim:
        max_dimension = problem.dim
    if min_dimension > max_dimension:
        raise ValueError("min_dimension must be <= max_dimension")

    dimensions_to_test = list(range(min_dimension, max_dimension))

    if verbose:
        logging.info(f"Testing subspace dimensions: {dimensions_to_test}")
        logging.info(f"Running {n_macroreps} macroreplications per dimension")
        logging.info(
            f"Consistency weight: {consistency_weight}, Quality weight: {quality_weight}"
        )
        logging.info(
            f"Parallelizing experiments across {len(dimensions_to_test)} dimensions"
        )

    # Store results for each dimension (process-safe with proper synchronization)
    all_final_objectives = {}  # dim -> list of final objective values
    all_statistics = {}  # dim -> statistics dict

    # Run experiments in parallel using ProcessPoolExecutor

    # Determine number of workers (limit to avoid overwhelming system)
    max_workers = 1  # max(1, int(100/n_macroreps))  # Limit to 4 parallel experiments

    if verbose:
        logging.info(f"Using {max_workers} parallel workers (processes)")

    # Prepare arguments for each dimension test
    # test_args = [
    #     (dim, problem, solver_name, n_macroreps, max_dimension, verbose)
    #     for dim in dimensions_to_test
    # ]

    # Collect results as they complete
    for dim in dimensions_to_test:
        try:
            result = _test_single_dimension(
                (dim, problem, solver_name, n_macroreps, max_dimension, verbose)
            )
            if result is not None:
                # Unpack results
                dim_val, final_objectives, statistics = result

                # Store results (process-safe as we're collecting sequentially here)
                all_final_objectives[dim_val] = final_objectives
                all_statistics[dim_val] = statistics

                if verbose:
                    print(f"✓ Completed dimension {dim_val}/{max_dimension}")
        except Exception as e:
            if verbose:
                logging.error(
                    f"Exception collecting results for dimension {dim}: {e!s}"
                )

    # Submit all dimension tests to process pool
    # with ProcessPoolExecutor(max_workers=max_workers) as executor:
    #     # Submit all tasks
    #     future_to_dim = {executor.submit(_test_single_dimension, args): args[0] for args in test_args}

    #     # Collect results as they complete
    #     for future in as_completed(future_to_dim):
    #         dim = future_to_dim[future]
    #         try:
    #             result = future.result()
    #             if result is not None:
    #                 # Unpack results
    #                 dim_val, final_objectives, statistics = result

    #                 # Store results (process-safe as we're collecting sequentially here)
    #                 all_final_objectives[dim_val] = final_objectives
    #                 all_statistics[dim_val] = statistics

    #                 if verbose:
    #                     print(f"✓ Completed dimension {dim_val}/{max_dimension}")
    #         except Exception as e:
    #             if verbose:
    #                 logging.error(f"Exception collecting results for dimension {dim}: {str(e)}")

    if verbose:
        logging.info("\nAll parallel experiments completed")
        logging.info(
            f"Successfully tested {len(all_statistics)} out of {len(dimensions_to_test)} dimensions"
        )

    # Check if we have any valid results
    if len(all_statistics) == 0:
        raise RuntimeError("No successful experiments completed")

    # Compute scores for each dimension
    tested_dims = sorted(all_statistics.keys())

    # Extract metrics for normalization
    all_means = np.array([all_statistics[d]["mean"] for d in tested_dims])
    all_cvs = np.array([all_statistics[d]["cv"] for d in tested_dims])
    all_stds = np.array([all_statistics[d]["std"] for d in tested_dims])

    # Replace NaN values with sensible defaults (infs handled via clipping later)
    all_means = np.nan_to_num(all_means, nan=np.inf, posinf=np.inf, neginf=-np.inf)
    all_cvs = np.nan_to_num(all_cvs, nan=1e10, posinf=1e10, neginf=0.0)
    all_stds = np.nan_to_num(all_stds, nan=1e10, posinf=1e10, neginf=0.0)

    # Normalize metrics to [0, 1] range for fair comparison
    # For minimization: score = (value - min) / (max - min)
    # Lower is better, so normalized score of 0 is best

    mean_min, mean_max = np.min(all_means), np.max(all_means)
    cv_min, cv_max = np.min(all_cvs), np.max(all_cvs)
    std_min, std_max = np.min(all_stds), np.max(all_stds)

    # Use epsilon to avoid division by zero
    eps = 1e-10
    mean_range = max(mean_max - mean_min, eps)
    max(cv_max - cv_min, eps)
    std_range = max(std_max - std_min, eps)

    consistency_scores = []
    quality_scores = []
    combined_scores = []

    # Compute consistency and quality scores
    for dim in tested_dims:
        # Consistency score: lower std dev is better
        # Normalize to [0, 1] where 0 is best (lowest std)
        std_dev = all_statistics[dim]["std"]
        std_dev = np.nan_to_num(std_dev, nan=1e10, posinf=1e10, neginf=0.0)
        consistency_score = (std_dev - std_min) / std_range

        # Quality score: depends on problem type
        mean_obj = all_statistics[dim]["mean"]
        mean_obj = np.nan_to_num(
            mean_obj, nan=np.inf if problem.minmax[0] == -1 else -np.inf
        )

        if problem.minmax[0] == -1:
            # Minimization: lower (more negative) mean is better
            # Normalize so that 0 = best (most negative/closest to 0)
            quality_score = (mean_obj - mean_min) / mean_range
        else:
            # Maximization: higher (more positive) mean is better
            # Normalize so that 0 = best (highest value)
            quality_score = (mean_max - mean_obj) / mean_range

        # Clip scores to [0, 1] range in case of numerical issues
        consistency_score = np.clip(consistency_score, 0.0, 1.0)
        quality_score = np.clip(quality_score, 0.0, 1.0)

        # Combined score: lower is better for both metrics
        combined_score = (
            consistency_weight * consistency_score + quality_weight * quality_score
        )

        consistency_scores.append(consistency_score)
        quality_scores.append(quality_score)
        combined_scores.append(combined_score)

        if verbose:
            logging.info(f"\nDimension {dim}:")
            logging.info(
                f"  Consistency score: {consistency_score:.4f} (Std: {std_dev:.4f})"
            )
            logging.info(f"  Quality score: {quality_score:.4f} (Mean: {mean_obj:.4f})")
            logging.info(f"  Combined score: {combined_score:.4f}")

    # Find the dimension with the best (lowest) combined score
    best_idx = np.argmin(combined_scores)
    optimal_dimension = tested_dims[best_idx]

    if verbose:
        logging.info(f"\n{'=' * 60}")
        logging.info(f"OPTIMAL SUBSPACE DIMENSION: {optimal_dimension}")
        logging.info(f"  Combined score: {combined_scores[best_idx]:.4f}")
        logging.info(
            f"  Mean objective: {all_statistics[optimal_dimension]['mean']:.4f}"
        )
        logging.info(
            f"  Standard deviation: {all_statistics[optimal_dimension]['std']:.4f}"
        )
        logging.info(
            f"  Coefficient of variation: {all_statistics[optimal_dimension]['cv']:.4f}"
        )
        logging.info(f"{'=' * 60}")

    # Return comprehensive results
    return {
        "optimal_dimension": optimal_dimension,
        "all_dimensions": tested_dims,
        "consistency_scores": consistency_scores,
        "quality_scores": quality_scores,
        "combined_scores": combined_scores,
        "final_objectives": all_final_objectives,
        "statistics": all_statistics,
    }
