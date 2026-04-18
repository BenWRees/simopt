"""A script to test the performance of different initial subspace dimensions and.

different.

polynomial basis types for the ASTROMORF solver across a set of benchmark problems.
The script uses data farming to systematically vary the chosen factor and evaluate its
impact
on solver performance.

This script supports:
1. Testing different fixed subspace dimensions on the same problem
2. Testing different polynomial basis types on the same problem
3. HPC-scalable execution via SLURM job arrays or individual job submission

Usage:
        # Test subspace dimensions
        python journal_factors_test.py --factor subspace --problem FACSIZE-1 --dim 10
        --budget 5000

        # Test polynomial basis types
        python journal_factors_test.py --factor basis --problem FACSIZE-1 --dim 10
        --budget 5000

        # For HPC array jobs (run a single design point)
        python journal_factors_test.py --factor subspace --problem FACSIZE-1 --dim 10
        --budget 5000 --task-id 0
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import sys
from collections.abc import MutableSequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast

import numpy as np

from simopt.problem import ProblemLike

# Take the current directory, find the parent, and add it to the system path
sys.path.append(str(Path.cwd().parent))

from simopt.experiment_base import (
    Problem,
    ProblemSolver,
    ProblemsSolvers,
    create_design,
    instantiate_problem,
    instantiate_solver,
)
from simopt.solvers.astromorf import PolyBasisType

problems_optimal_hyper: dict = {
    "DYNAMNEWS-1": {"subspace_dimension": 20, "polynomial_degree": 4},  # Optimal
    "SAN-1": {"subspace_dimension": 40, "polynomial_degree": 4},  # Optimal
    "ROSENBROCK-1": {"subspace_dimension": 5, "polynomial_degree": 4},  # Optimal
    "NETWORK-1": {"subspace_dimension": 20, "polynomial_degree": 4},  # optimal
    "SSCONT-1": {"subspace_dimension": 1, "polynomial_degree": 2},  # Optimal
}

SCALABLE_PROBLEMS = [
    "FIXEDSAN-1",
    "NETWORK-1",
    "ROSENBROCK-1",
    "SAN-1",
    "DYNAMNEWS-1",
    "FACSIZE-2",
    "FACSIZE-1",
    "CONTAM-2",
]

solver_renames = {
    "ASTROMORF": "ASTROMoRF",
    "OMoRF": "OMoRF",
    "ADAM": "ADAM",
    "ASTRODF": "ASTRO-DF",
    "NELDMD": "NELDER MEAD",
    "RNDSRCH": "RANDOM SEARCH",
    "STRONG": "STRONG",
}

_T = TypeVar("_T")


class _SupportsShuffleRandInt(Protocol):
    def shuffle(self, x: MutableSequence[_T]) -> None:
        """Shuffle a mutable sequence in-place."""

    def randint(self, a: int, b: int) -> int:
        """Return an integer N such that a <= N <= b."""


# ============================================================================
# CONFIGURATION DATACLASS
# ============================================================================


@dataclass
class ExperimentConfig:
    """Configuration for data farming experiments."""

    # Factor to test: "subspace" or "basis"
    factor_type: str = "subspace"

    # Problem configuration
    problem_name: str = "FACSIZE-1"
    problem_dim: int = 10
    budget: int = 5000

    # Experiment settings
    n_macroreps: int = 10
    n_postreps: int = 100
    # n_postreps_init_opt: int = 200

    # CRN settings
    crn_across_solns: bool = False
    crn_across_budget: bool = True
    crn_across_macroreps: bool = False
    crn_across_init_opt: bool = True

    # Subspace dimension settings (when factor_type="subspace")
    subspace_dims: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8])
    polynomial_degree: int = 2
    fixed_basis: PolyBasisType = PolyBasisType.HERMITE

    # Polynomial basis settings (when factor_type="basis")
    basis_types: list[PolyBasisType] = field(
        default_factory=lambda: list(PolyBasisType)
    )
    fixed_subspace_dim: int = 4

    # HPC settings
    task_id: int | None = None  # For SLURM array jobs
    output_dir: Path = field(default_factory=lambda: Path("experiments"))

    def __post_init__(self):  # noqa: ANN204
        """Validate and process configuration after initialization."""
        if self.factor_type not in ("subspace", "basis"):
            raise ValueError(
                f"factor_type must be 'subspace' or 'basis', got: {self.factor_type}"
            )

        # Ensure output directory exists
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================================
# POLYNOMIAL BASIS TYPE UTILITIES
# ============================================================================

# Mapping from PolyBasisType enum to readable names
POLY_BASIS_NAMES = {
    PolyBasisType.HERMITE: "Hermite",
    PolyBasisType.LEGENDRE: "Legendre",
    PolyBasisType.CHEBYSHEV: "Chebyshev",
    PolyBasisType.MONOMIAL: "Monomial",
    PolyBasisType.NATURAL: "Natural",
    PolyBasisType.MONOMIAL_POLY: "MonomialPoly",
    PolyBasisType.LAGRANGE: "Lagrange",
    PolyBasisType.NFP: "NFP",
    PolyBasisType.LAGUERRE: "Laguerre",
}


def get_all_poly_basis_types() -> list[PolyBasisType]:
    """Return all available polynomial basis types."""
    return list(PolyBasisType)


def poly_basis_from_string(name: str) -> PolyBasisType:
    """Convert a string name to PolyBasisType enum."""
    name_upper = name.upper().replace("-", "_").replace(" ", "_")
    try:
        return PolyBasisType[name_upper]
    except KeyError:
        # Try matching by value
        for pbt in PolyBasisType:
            if pbt.value.upper() == name_upper:
                return pbt
        raise ValueError(f"Unknown polynomial basis type: {name}")  # noqa: B904


# ============================================================================
# PROBLEM SCALING UTILITIES
# ============================================================================


def scale_dimension(problem_name: str, dimension: int, budget: int) -> Problem:  # noqa: D417
    """Instantiate a problem with a scaled dimension.

    All model and problem factors that depend on the dimension are updated
    before instantiation to ensure consistency.

    Args:
            problem_name: The abbreviated name of the problem (e.g., "FACSIZE-2")
            dimension: The desired dimension for the problem

    Returns:
            A Problem instance configured for the specified dimension
    """
    if problem_name not in SCALABLE_PROBLEMS:
        # For non-scalable problems, just instantiate with defaults
        return instantiate_problem(problem_name, {"budget": budget})

    # Build the factors for the new dimension
    model_factors = get_scaled_model_factors(problem_name, dimension)
    problem_factors = get_scaled_problem_factors(problem_name, dimension)
    problem_factors["budget"] = budget

    # Instantiate the problem with the scaled factors
    problem = instantiate_problem(
        problem_name,
        problem_fixed_factors=problem_factors,
        model_fixed_factors=model_factors,
    )

    # Set the problem dimension explicitly
    problem.dim = dimension

    # Post-initialization updates for factors that can't be validated during
    # construction
    post_init_updates(problem, problem_name, dimension)

    return problem


def get_scaled_model_factors(problem_name: str, dimension: int) -> dict:
    """Generate model factors scaled to the specified dimension.

    Args:
            problem_name: The abbreviated name of the problem
            dimension: The target dimension

    Returns:
            Dictionary of model factors appropriate for the dimension
    """
    # Deterministic RNGs based on problem name + dimension
    seed = int(
        hashlib.sha256(f"{problem_name}:{dimension}".encode()).hexdigest(), 16
    ) % (2**32)
    rng_py = random.Random(seed)
    np.random.default_rng(seed)

    if problem_name == "DYNAMNEWS-1":
        return {
            "num_prod": dimension,
            "c_utility": [float(6 + j) for j in range(dimension)],
            "init_level": [3] * dimension,
            "price": [9.0] * dimension,
            "cost": [5.0] * dimension,
        }

    if problem_name in ("FACSIZE-1", "FACSIZE-2"):
        # Use diagonal covariance to avoid expensive Cholesky and reduce rejection rate
        # With mean=500 and std=50 (variance=2500), P(X<0) ≈ 0 for each dimension
        # This makes rejection sampling nearly instant
        variance = 2500.0  # std = 50, mean = 500, so P(X<0) is negligible
        cov_matrix = np.eye(dimension) * variance
        return {
            "mean_vec": [500.0] * dimension,
            "cov": cov_matrix.tolist(),
            "capacity": [float(rng_py.randint(100, 900)) for _ in range(dimension)],
            "n_fac": dimension,
        }

    if problem_name == "SAN-1":
        # Calculate appropriate num_nodes for the number of edges (dimension)
        # For a DAG: we need num_nodes such that we can have 'dimension' edges
        # with a path from node 1 to num_nodes
        num_nodes = compute_num_nodes_for_dag(dimension)
        arcs = build_san_dag(num_nodes, dimension, rng=rng_py)
        return {
            "num_arcs": dimension,
            "num_nodes": num_nodes,
            "arcs": arcs,
            "arc_means": tuple(
                round(rng_py.uniform(1, 10), 2) for _ in range(dimension)
            ),
        }

    if problem_name == "FIXEDSAN-1":
        num_nodes = max(2, rng_py.randint(2, max(2, dimension)))
        return {
            "num_arcs": dimension,
            "num_nodes": num_nodes,
            "arc_means": tuple(float(rng_py.randint(1, 10)) for _ in range(dimension)),
        }

    if problem_name == "ROSENBROCK-1":
        return {
            "x": (2.0,) * dimension,
            "variance": 0.4,
        }
    if problem_name == "ZAKHAROV-1":
        return {
            "x": (2.0,) * dimension,
            "variance": 0.1,
        }

    if problem_name == "NETWORK-1":
        process_prob_elem = 1.0 / dimension
        mode_transit_time = [
            round(rng_py.uniform(0.01, 5), 3) for _ in range(dimension)
        ]
        return {
            "process_prob": [process_prob_elem] * dimension,
            "cost_process": [0.1 / (x + 1) for x in range(dimension)],
            "cost_time": [round(rng_py.uniform(0.01, 1), 3) for _ in range(dimension)],
            "mode_transit_time": mode_transit_time,
            "lower_limits_transit_time": [x / 2 for x in mode_transit_time],
            "upper_limits_transit_time": [2 * x for x in mode_transit_time],
            "n_networks": dimension,
        }

    if problem_name == "CONTAM-2":
        return {
            "stages": dimension,
            "prev_decision": (0.0,) * dimension,
        }

    return {}


def get_scaled_problem_factors(problem_name: str, dimension: int) -> dict:
    """Generate problem factors scaled to the specified dimension.

    Only includes factors that will pass validation during construction.
    Factors that depend on model state are updated post-initialization.

    Args:
            problem_name: The abbreviated name of the problem
            dimension: The target dimension

    Returns:
            Dictionary of problem factors appropriate for the dimension
    """
    if problem_name == "DYNAMNEWS-1":
        return {
            "initial_solution": (3.0,) * dimension,
        }

    if problem_name in ("FACSIZE-1", "FACSIZE-2"):
        # NOTE: installation_costs is validated against NUM_FACILITIES constant (=3)
        # So we can't pass it here - it will be updated post-initialization
        return {
            "initial_solution": (100.0,) * dimension,
            "installation_budget": 500.0
            * (dimension / 3),  # Scale budget with dimension
        }

    if problem_name in ("SAN-1", "FIXEDSAN-1"):
        # NOTE: arc_costs is validated against NUM_ARCS constant (=13)
        # So we can't pass it here - it will be updated post-initialization
        return {
            "initial_solution": (1.0,) * dimension,
        }

    if problem_name == "ROSENBROCK-1" or problem_name == "ZAKHAROV-1":
        return {
            "initial_solution": (2.0,) * dimension,
        }

    if problem_name == "NETWORK-1":
        return {
            "initial_solution": (1.0 / dimension,) * dimension,
        }

    if problem_name == "CONTAM-2":
        return {
            "initial_solution": (0.0,) * dimension,
        }

    return {}


def post_init_updates(problem: ProblemLike, problem_name: str, dimension: int) -> None:
    """Update problem factors after initialization for factors that couldn't be set.

    during construction.

    Some factors are validated against hardcoded constants during construction,
    so they need to be updated after the problem is instantiated.

    Args:
            problem: The problem instance to update
            problem_name: The abbreviated name of the problem
            dimension: The target dimension
    """
    if problem_name in ("FACSIZE-1", "FACSIZE-2"):
        # Update installation_costs after construction to match the new dimension
        problem.factors["installation_costs"] = (1.0,) * dimension

    elif problem_name in ("SAN-1", "FIXEDSAN-1"):
        # Update arc_costs after construction to match the new dimension
        # arc_costs is used in replicate(): np.sum(arc_costs / x)
        problem.factors["arc_costs"] = (1.0,) * dimension


def compute_num_nodes_for_dag(num_edges: int) -> int:
    """Compute an appropriate number of nodes for a DAG with the given number of edges.

    For a DAG with n nodes where we need a path from 1 to n:
    - Minimum edges needed: n-1 (a simple path)
    - Maximum edges possible: n*(n-1)/2 (complete DAG)

    We want to find the smallest n such that n*(n-1)/2 >= num_edges
    and n-1 <= num_edges (so we have enough edges for connectivity).

    Args:
            num_edges: Desired number of edges

    Returns:
            Number of nodes to use
    """
    # We need at least num_edges + 1 nodes in the worst case (simple path),
    # but we want fewer nodes with more edges between them.
    # Solve: n*(n-1)/2 >= num_edges => n^2 - n - 2*num_edges >= 0
    # n >= (1 + sqrt(1 + 8*num_edges)) / 2

    import math

    min_nodes = math.ceil((1 + math.sqrt(1 + 8 * num_edges)) / 2)

    # Ensure we have at least 2 nodes and the path is possible
    min_nodes = max(2, min_nodes)

    # Also ensure num_edges >= min_nodes - 1 (need at least a spanning path)
    # If not, we need more nodes
    while min_nodes - 1 > num_edges:
        min_nodes -= 1

    return min_nodes


def build_san_dag(  # noqa: D417
    num_nodes: int,
    num_edges: int,
    rng: _SupportsShuffleRandInt | None = None,
) -> list[tuple[int, int]]:
    """Build a directed acyclic graph (DAG) suitable for the SAN model.

    The SAN model requires:
    1. Directed edges (arcs) from lower-numbered to higher-numbered nodes
    2. A path must exist from node 1 to node num_nodes
    3. Every node must be reachable from node 1 (for backtracking to work)

    This function first creates a simple sequential path from 1 to num_nodes
    (1→2→3→...→n), then adds additional random forward edges until reaching num_edges.

    Args:
            num_nodes: Number of nodes (numbered 1 to num_nodes)
            num_edges: Desired number of directed edges

    Returns:
            List of (source, target) tuples representing directed edges

    Raises:
            ValueError: If the requested configuration is impossible
    """
    min_edges = num_nodes - 1  # Simple path from 1 to num_nodes
    max_edges = num_nodes * (num_nodes - 1) // 2  # Complete DAG

    if num_edges < min_edges:
        raise ValueError(
            f"Cannot create DAG with path 1→{num_nodes}: need at least {min_edges} edges, "  # noqa: E501
            f"but only {num_edges} requested"
        )

    if num_edges > max_edges:
        raise ValueError(
            f"Cannot create DAG with {num_edges} edges: maximum possible is {max_edges} "  # noqa: E501
            f"for {num_nodes} nodes"
        )

    edges = set()

    # Step 1: Create a guaranteed SEQUENTIAL path from node 1 to node num_nodes
    # This ensures every node has a predecessor reachable from node 1
    # Path: 1 → 2 → 3 → ... → num_nodes
    for i in range(1, num_nodes):
        edges.add((i, i + 1))

    # Step 2: Add additional random forward edges until we reach num_edges
    if len(edges) < num_edges:
        if rng is None:
            rng = cast(_SupportsShuffleRandInt, random)
        # Generate all possible forward edges not yet in the graph
        all_possible_edges = []
        for i in range(1, num_nodes):
            for j in range(i + 1, num_nodes + 1):
                edge = (i, j)
                if edge not in edges:
                    all_possible_edges.append(edge)

        # Shuffle and add as many as needed
        rng.shuffle(all_possible_edges)
        edges_needed = num_edges - len(edges)

        for edge in all_possible_edges[:edges_needed]:
            edges.add(edge)

    return list(edges)


def validate_solver_and_problem_names(solver_name: str, problem_name: str) -> None:
    """Pre-flight validation for clearer errors before heavy work."""
    try:
        _ = instantiate_solver(
            solver_name=solver_name,
            fixed_factors={},
            solver_rename=solver_renames.get(solver_name, solver_name),
        )
    except Exception as e:
        raise ValueError(
            f"Unknown or invalid solver code '{solver_name}'. Original error: {e}"
        ) from e

    try:
        _ = instantiate_problem(
            problem_name, problem_fixed_factors=None, model_fixed_factors=None
        )
    except Exception as e:
        raise ValueError(
            f"Unknown problem code '{problem_name}'. Original error: {e}"
        ) from e


def build_connected_graph(  # noqa: D417
    num_nodes: int,
    num_edges: int,
    rng: _SupportsShuffleRandInt | None = None,
) -> list[tuple[int, int]]:
    """Build a connected graph with the specified number of nodes and edges.

    Starts with a spanning tree to ensure connectivity, then adds random edges
    until the desired number is reached.

    Args:
            num_nodes: Number of nodes in the graph
            num_edges: Desired number of edges

    Returns:
            List of (node1, node2) tuples representing edges
    """
    if num_edges < num_nodes - 1:
        raise ValueError(
            f"Cannot create connected graph: need at least {num_nodes - 1} edges for {num_nodes} nodes"  # noqa: E501
        )

    edges = set()

    if rng is None:
        rng = cast(_SupportsShuffleRandInt, random)

    # Create a spanning tree first to ensure connectivity
    nodes = list(range(num_nodes))
    rng.shuffle(nodes)
    for i in range(1, num_nodes):
        a = nodes[i]
        b = nodes[rng.randint(0, i - 1)]
        edges.add((min(a, b), max(a, b)))

    # Add random edges until we reach the desired number
    max_possible_edges = num_nodes * (num_nodes - 1) // 2
    target_edges = min(num_edges, max_possible_edges)

    attempts = 0
    max_attempts = target_edges * 10  # Prevent infinite loop
    while len(edges) < target_edges and attempts < max_attempts:
        a = rng.randint(0, num_nodes - 1)
        b = rng.randint(0, num_nodes - 1)
        if a != b:
            edges.add((min(a, b), max(a, b)))
        attempts += 1

    return list(edges)


# Default problem dimensions and model factor mappings
PROBLEM_DIM_FACTORS = {
    "FACSIZE-1": ("num_prod",),
    "FACSIZE-2": ("num_prod",),
    "DYNAMNEWS-1": ("num_prod",),
    "CONTAM-2": ("n_sources",),
    "NETWORK-1": ("num_nodes",),
    "ROSENBROCK-1": ("dim",),
    "ZAKHAROV-1": ("dim",),
    "FIXEDSAN-1": ("num_arcs",),
    "SAN-1": ("num_arcs",),
    "SSCONT-1": ("num_retailers",),
    "IRONORECONT-1": ("num_prod",),
}


def get_problem_fixed_factors(
    problem_name: str, dim: int, budget: int
) -> dict[str, Any]:
    """Get problem fixed factors based on problem name and desired dimension."""
    factors = {"budget": budget}

    # Handle problem-specific dimension factors
    if problem_name in PROBLEM_DIM_FACTORS:
        for factor_name in PROBLEM_DIM_FACTORS[problem_name]:
            factors[factor_name] = dim

    return factors


def get_model_fixed_factors(problem_name: str, dim: int) -> dict[str, Any]:
    """Get model fixed factors for problems that require them."""
    model_factors = {}

    # Some problems need model-level dimension specifications
    if problem_name in ("CONTAM-2",):
        model_factors["n_sources"] = dim
    elif problem_name in ("NETWORK-1",):
        model_factors["num_nodes"] = dim

    return model_factors


# ============================================================================
# EXPERIMENT RUNNERS
# ============================================================================


def run_single_design_point(
    config: ExperimentConfig,
    solver_factors: dict[str, Any],
    design_point_id: str,
    path_file_name: str | Path,
) -> dict[str, Any]:
    """Run a single design point experiment.

    This function is designed for HPC parallelization - each design point
    can be run independently as a separate job.

    Args:
            config: Experiment configuration
            solver_factors: Solver factor settings for this design point
            design_point_id: Unique identifier for this design point
            path_file_name: Path for saving output files
    Returns:
            Dictionary containing experiment results
    """
    logger = logging.getLogger(__name__)

    # Create output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = config.output_dir / f"{design_point_id}_{timestamp}.pickle"

    logger.info(f"Running design point: {design_point_id}")
    logger.info(f"Solver factors: {solver_factors}")

    # Get problem/model factors
    # problem_factors = get_problem_fixed_factors(
    #     config.problem_name, config.problem_dim, config.budget
    # )
    # model_factors = get_model_fixed_factors(config.problem_name, config.problem_dim)

    # Instantiate solver and problem
    try:
        solver = instantiate_solver(
            solver_name="ASTROMORF",
            fixed_factors=solver_factors,
            solver_rename="ASTROMoRF",
        )

        problem = scale_dimension(
            problem_name=config.problem_name,
            dimension=config.problem_dim,
            budget=config.budget,
        )

        # problem = instantiate_problem(
        #     problem_name=config.problem_name,
        #     problem_fixed_factors=problem_factors,
        #     model_fixed_factors=model_factors if model_factors else None,
        # )
    except Exception as e:
        logger.error(f"Failed to instantiate solver/problem: {e}")
        raise

    # Run experiment
    experiment = ProblemSolver(
        problem=problem,
        solver=solver,
        file_name_path=str(output_file),
    )

    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    experiment.run(n_macroreps=config.n_macroreps, n_jobs=n_jobs)

    # Post-process
    experiment.post_replicate(
        n_postreps=config.n_postreps,
        crn_across_budget=config.crn_across_budget,
        crn_across_macroreps=config.crn_across_macroreps,
    )
    file_name_pickle = config.output_dir / f"{path_file_name}_POSTREPS.pickle"
    # Save post-rep pickle into the same output directory used for this design point
    experiment.record_experiment_results(str(file_name_pickle))
    experiment.log_experiment_results(file_path=str(output_file.with_suffix(".txt")))

    logger.info(f"Experiment completed. Results saved to: {output_file}")

    return {
        "design_point_id": design_point_id,
        "output_file": str(output_file),
        "solver_factors": solver_factors,
        "config": {
            "problem_name": config.problem_name,
            "problem_dim": config.problem_dim,
            "budget": config.budget,
            "n_macroreps": config.n_macroreps,
        },
    }


def run_subspace_dimension_experiments(
    config: ExperimentConfig,
) -> list[dict[str, Any]]:
    """Run experiments comparing different fixed subspace dimensions.

    Args:
            config: Experiment configuration with subspace dimensions to test

    Returns:
            List of result dictionaries from each design point
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Running subspace dimension experiments on {config.problem_name}")
    logger.info(f"Testing dimensions: {config.subspace_dims}")

    results = []

    # Generate design points for subspace dimensions
    design_points = []
    for dim in config.subspace_dims:
        # Ensure dimension doesn't exceed problem dimension
        if dim <= config.problem_dim:
            solver_factors = {
                "initial subspace dimension": dim,
                "polynomial degree": problems_optimal_hyper.get(
                    config.problem_name, {}
                ).get("polynomial_degree", config.polynomial_degree),
                "polynomial basis": config.fixed_basis,
                "adaptive subspace dimension": False,  # Fixed subspace
                "crn_across_solns": config.crn_across_solns,
            }
            design_point_id = f"ASTROMORF_subspace_{dim}_on_{config.problem_name}"
            design_points.append((design_point_id, solver_factors))

    # If task_id is specified, only run that specific design point (for HPC array jobs)
    if config.task_id is not None:
        if config.task_id < len(design_points):
            design_point_id, solver_factors = design_points[config.task_id]
            path_file_name = (
                f"{config.problem_name}_subspace_experiment_on_{design_point_id}"
            )
            result = run_single_design_point(
                config, solver_factors, design_point_id, path_file_name
            )
            results.append(result)
        else:
            logger.warning(
                f"Task ID {config.task_id} out of range (max: {len(design_points) - 1})"
            )
    else:
        # Run all design points sequentially
        for design_point_id, solver_factors in design_points:
            path_file_name = (
                f"{config.problem_name}_subspace_experiment_on_{design_point_id}"
            )
            result = run_single_design_point(
                config, solver_factors, design_point_id, path_file_name
            )
            results.append(result)

    return results


def run_polynomial_basis_experiments(config: ExperimentConfig) -> list[dict[str, Any]]:
    """Run experiments comparing different polynomial basis types.

    Args:
            config: Experiment configuration with basis types to test

    Returns:
            List of result dictionaries from each design point
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Running polynomial basis experiments on {config.problem_name}")
    logger.info(
        f"Testing basis types: {[POLY_BASIS_NAMES.get(b, str(b)) for b in config.basis_types]}"  # noqa: E501
    )

    results = []

    # Generate design points for polynomial basis types
    design_points = []
    for basis_type in config.basis_types:
        solver_factors = {
            "initial subspace dimension": config.fixed_subspace_dim,
            "polynomial degree": config.polynomial_degree,
            "polynomial basis": basis_type,
            "adaptive subspace dimension": False,  # Fixed subspace for fair comparison
            "crn_across_solns": config.crn_across_solns,
        }
        basis_name = POLY_BASIS_NAMES.get(basis_type, basis_type.value)
        design_point_id = f"ASTROMORF_basis_{basis_name}_on_{config.problem_name}"
        design_points.append((design_point_id, solver_factors))

    # If task_id is specified, only run that specific design point (for HPC array jobs)
    if config.task_id is not None:
        if config.task_id < len(design_points):
            design_point_id, solver_factors = design_points[config.task_id]
            path_file_name = f"{config.problem_name}_polynomial_basis_experiment_on_{design_point_id}"  # noqa: E501
            result = run_single_design_point(
                config, solver_factors, design_point_id, path_file_name
            )
            results.append(result)
        else:
            logger.warning(
                f"Task ID {config.task_id} out of range (max: {len(design_points) - 1})"
            )
    else:
        # Run all design points sequentially
        for design_point_id, solver_factors in design_points:
            path_file_name = f"{config.problem_name}_polynomial_basis_experiment_on_{design_point_id}"  # noqa: E501
            result = run_single_design_point(
                config, solver_factors, design_point_id, path_file_name
            )
            results.append(result)

    return results


def run_full_factorial_experiment(config: ExperimentConfig) -> list[dict[str, Any]]:
    """Run a full factorial experiment varying both subspace dimensions and basis types.

    This creates a cross-product of all subspace dimensions and basis types.

    Args:
            config: Experiment configuration

    Returns:
            List of result dictionaries from each design point
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Running full factorial experiment on {config.problem_name}")

    results = []

    # Generate all combinations
    design_points = []
    for dim in config.subspace_dims:
        if dim > config.problem_dim:
            continue
        for basis_type in config.basis_types:
            solver_factors = {
                "initial subspace dimension": dim,
                "polynomial degree": config.polynomial_degree,
                "polynomial basis": basis_type,
                "adaptive subspace dimension": False,
                "crn_across_solns": config.crn_across_solns,
            }
            basis_name = POLY_BASIS_NAMES.get(basis_type, basis_type.value)
            design_point_id = (
                f"ASTROMORF_d{dim}_basis_{basis_name}_on_{config.problem_name}"
            )
            design_points.append((design_point_id, solver_factors))

    logger.info(f"Total design points: {len(design_points)}")

    # If task_id is specified, only run that specific design point
    if config.task_id is not None:
        if config.task_id < len(design_points):
            design_point_id, solver_factors = design_points[config.task_id]
            path_file_name = (
                f"{config.problem_name}_full_factorial_experiment_on_{design_point_id}"
            )
            result = run_single_design_point(
                config, solver_factors, design_point_id, path_file_name
            )
            results.append(result)
        else:
            logger.warning(
                f"Task ID {config.task_id} out of range (max: {len(design_points) - 1})"
            )
    else:
        for design_point_id, solver_factors in design_points:
            path_file_name = (
                f"{config.problem_name}_full_factorial_experiment_on_{design_point_id}"
            )
            result = run_single_design_point(
                config, solver_factors, design_point_id, path_file_name
            )
            results.append(result)

    return results


# ============================================================================
# HPC UTILITIES
# ============================================================================


def generate_hpc_config(config: ExperimentConfig, output_path: Path) -> Path:
    """Generate a JSON configuration file for HPC job submission.

    Args:
            config: Experiment configuration
            output_path: Path to save the configuration file

    Returns:
            Path to the generated configuration file
    """
    # Determine number of design points
    if config.factor_type == "subspace":
        n_design_points = len(
            [d for d in config.subspace_dims if d <= config.problem_dim]
        )
    elif config.factor_type == "basis":
        n_design_points = len(config.basis_types)
    else:
        # Full factorial
        n_design_points = len(
            [d for d in config.subspace_dims if d <= config.problem_dim]
        ) * len(config.basis_types)

    hpc_config = {
        "experiment_type": config.factor_type,
        "problem_name": config.problem_name,
        "problem_dim": config.problem_dim,
        "budget": config.budget,
        "n_macroreps": config.n_macroreps,
        "n_postreps": config.n_postreps,
        "n_design_points": n_design_points,
        "subspace_dims": config.subspace_dims
        if config.factor_type in ("subspace", "full")
        else [config.fixed_subspace_dim],
        "basis_types": [b.value for b in config.basis_types]
        if config.factor_type in ("basis", "full")
        else [config.fixed_basis.value],
        "polynomial_degree": config.polynomial_degree,
        "output_dir": str(config.output_dir),
    }

    config_file = output_path / "hpc_config.json"
    with open(config_file, "w") as f:  # noqa: PTH123
        json.dump(hpc_config, f, indent=2)

    return config_file


def generate_slurm_array_script(
    config: ExperimentConfig,
    output_path: Path,
    time_limit: str = "4:00:00",
    partition: str = "batch",
    cpus_per_task: int = 1,
    mem_per_cpu: str = "4G",
) -> Path:
    """Generate a SLURM array job script for running experiments on HPC.

    Args:
            config: Experiment configuration
            output_path: Directory to save the SLURM script
            time_limit: Wall clock time limit per task
            partition: SLURM partition to use
            cpus_per_task: Number of CPUs per task
            mem_per_cpu: Memory per CPU

    Returns:
            Path to the generated SLURM script
    """
    # Calculate number of array tasks
    if config.factor_type == "subspace":
        n_tasks = len([d for d in config.subspace_dims if d <= config.problem_dim])
    elif config.factor_type == "basis":
        n_tasks = len(config.basis_types)
    else:
        n_tasks = len(
            [d for d in config.subspace_dims if d <= config.problem_dim]
        ) * len(config.basis_types)

    # Get the path to this script
    script_path = Path(__file__).resolve()

    slurm_content = f"""#!/bin/bash
#SBATCH --job-name=astromorf_{config.factor_type}_{config.problem_name}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem-per-cpu={mem_per_cpu}
#SBATCH --time={time_limit}
#SBATCH --array=0-{n_tasks - 1}
#SBATCH --output={output_path}/logs/slurm_%A_%a.out
#SBATCH --error={output_path}/logs/slurm_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=$USER@soton.ac.uk

# Create log directory if it doesn't exist
mkdir -p {output_path}/logs

# Activate conda environment
source $HOME/miniconda3/bin/activate simopt

# Run the experiment for this task ID
python {script_path} \\
	--factor {config.factor_type} \\
	--problem {config.problem_name} \\
	--dim {config.problem_dim} \\
	--budget {config.budget} \\
	--n-macroreps {config.n_macroreps} \\
	--n-postreps {config.n_postreps} \\
	--output-dir {output_path} \\
	--task-id $SLURM_ARRAY_TASK_ID

echo "Task $SLURM_ARRAY_TASK_ID completed"
"""

    slurm_file = output_path / f"run_{config.factor_type}_{config.problem_name}.slurm"
    with open(slurm_file, "w") as f:  # noqa: PTH123
        f.write(slurm_content)

    # Make executable
    os.chmod(slurm_file, 0o755)  # noqa: PTH101

    return slurm_file


def generate_csv_design_matrix(config: ExperimentConfig, output_path: Path) -> Path:
    """Generate a CSV file containing the full design matrix for the experiment.

    This can be used to manually submit individual jobs or track experiment progress.

    Args:
            config: Experiment configuration
            output_path: Directory to save the CSV file

    Returns:
            Path to the generated CSV file
    """
    rows = []
    task_id = 0

    if config.factor_type == "subspace":
        for dim in config.subspace_dims:
            if dim <= config.problem_dim:
                rows.append(
                    {
                        "task_id": task_id,
                        "design_point_id": f"ASTROMORF_subspace_{dim}_on_{config.problem_name}",  # noqa: E501
                        "subspace_dim": dim,
                        "polynomial_basis": config.fixed_basis.value,
                        "polynomial_degree": config.polynomial_degree,
                        "problem_name": config.problem_name,
                        "problem_dim": config.problem_dim,
                        "budget": config.budget,
                    }
                )
                task_id += 1

    elif config.factor_type == "basis":
        for basis_type in config.basis_types:
            basis_name = POLY_BASIS_NAMES.get(basis_type, basis_type.value)
            rows.append(
                {
                    "task_id": task_id,
                    "design_point_id": f"ASTROMORF_basis_{basis_name}_on_{config.problem_name}",  # noqa: E501
                    "subspace_dim": config.fixed_subspace_dim,
                    "polynomial_basis": basis_type.value,
                    "polynomial_degree": config.polynomial_degree,
                    "problem_name": config.problem_name,
                    "problem_dim": config.problem_dim,
                    "budget": config.budget,
                }
            )
            task_id += 1

    else:  # Full factorial
        for dim in config.subspace_dims:
            if dim > config.problem_dim:
                continue
            for basis_type in config.basis_types:
                basis_name = POLY_BASIS_NAMES.get(basis_type, basis_type.value)
                rows.append(
                    {
                        "task_id": task_id,
                        "design_point_id": f"ASTROMORF_d{dim}_basis_{basis_name}_on_{config.problem_name}",  # noqa: E501
                        "subspace_dim": dim,
                        "polynomial_basis": basis_type.value,
                        "polynomial_degree": config.polynomial_degree,
                        "problem_name": config.problem_name,
                        "problem_dim": config.problem_dim,
                        "budget": config.budget,
                    }
                )
                task_id += 1

    csv_file = (
        output_path / f"design_matrix_{config.factor_type}_{config.problem_name}.csv"
    )

    if rows:
        fieldnames = rows[0].keys()
        with open(csv_file, "w", newline="") as f:  # noqa: PTH123
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return csv_file


# ============================================================================
# USING CREATE_DESIGN FUNCTION (INTEGRATED WITH SIMOPT DATA FARMING)
# ============================================================================


def run_experiment_with_create_design(
    factor_type: str,
    problem_names: list[str],
    problem_dim: int,
    budget: int,
    n_macroreps: int = 10,
    n_postreps: int = 100,
    n_postreps_init_opt: int = 200,
    subspace_dims: list[int] | None = None,
    basis_types: list[PolyBasisType] | None = None,
) -> None:
    """Run experiment using SimOpt's create_design function for NOLHS design generation.

    This approach integrates with SimOpt's data farming infrastructure.

    Args:
            factor_type: "subspace" or "basis"
            problem_names: List of problem names to test
            problem_dim: Problem dimension
            budget: Simulation budget
            n_macroreps: Number of macroreplications
            n_postreps: Number of post-replications
            n_postreps_init_opt: Number of post-replications for initial/optimal
            subspace_dims: List of subspace dimensions to test (for "subspace" factor)
            basis_types: List of basis types to test (for "basis" factor)
    """
    logger = logging.getLogger(__name__)

    solver_abbr_name = "ASTROMORF"

    if factor_type == "subspace":
        # Use cross_design_factors for subspace dimensions (discrete levels)
        if subspace_dims is None:
            subspace_dims = list(range(1, min(problem_dim + 1, 9)))

        solver_factor_headers = []  # No continuous factors
        solver_factor_settings = []
        solver_fixed_factors = {
            "adaptive subspace dimension": False,
            "polynomial degree": 2,
        }
        # Use cross_design_factors for the discrete subspace dimensions
        solver_cross_design_factors = {
            "crn_across_solns": [False],
            "initial subspace dimension": subspace_dims,
        }

    elif factor_type == "basis":
        # Use cross_design_factors for polynomial basis types
        if basis_types is None:
            basis_types = list(PolyBasisType)

        solver_factor_headers = []
        solver_factor_settings = []
        solver_fixed_factors = {
            "adaptive subspace dimension": False,
            "initial subspace dimension": 4,
            "polynomial degree": 2,
        }
        solver_cross_design_factors = {
            "crn_across_solns": [False],
            "polynomial basis": basis_types,
        }

    else:
        raise ValueError(f"Unknown factor type: {factor_type}")

    # Create design
    solver_design_list = create_design(
        name=solver_abbr_name,
        factor_headers=solver_factor_headers,
        factor_settings=solver_factor_settings,
        n_stacks=1,
        fixed_factors=solver_fixed_factors,
        cross_design_factors=solver_cross_design_factors,
    )

    logger.info(f"Generated {len(solver_design_list)} design points")

    # Problem factors
    problem_fixed_factors = [{"budget": budget} for _ in problem_names]

    # Create solver name list
    solver_names = [solver_abbr_name] * len(solver_design_list)

    # Create ProblemsSolvers experiment
    experiment = ProblemsSolvers(
        solver_factors=solver_design_list,
        problem_factors=problem_fixed_factors,
        solver_names=solver_names,
        problem_names=problem_names,
    )

    experiment.check_compatibility()

    logger.info("Running experiments...")
    experiment.run(n_macroreps)

    # Post-process
    experiment.post_replicate(
        n_postreps=n_postreps,
        crn_across_budget=True,
        crn_across_macroreps=False,
    )

    experiment.post_normalize(
        n_postreps_init_opt=n_postreps_init_opt,
        crn_across_init_opt=True,
    )

    experiment.record_group_experiment_results()
    experiment.log_group_experiment_results()
    experiment.report_group_statistics()


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run data farming experiments for ASTROMoRF solver factors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test subspace dimensions on FACSIZE-1
  python journal_factors_test.py --factor subspace --problem FACSIZE-1 --dim 10 --budget
  5000
  
  # Test polynomial basis types
  python journal_factors_test.py --factor basis --problem FACSIZE-1 --dim 10 --budget
  5000
  
  # Generate SLURM scripts for HPC
  python journal_factors_test.py --factor subspace --problem FACSIZE-1 --dim 30 --budget
  10000 --generate-slurm
  
  # Run a specific task (for HPC array jobs)
  python journal_factors_test.py --factor subspace --problem FACSIZE-1 --dim 10 --task-
  id 3
		""",
    )

    # Required arguments
    parser.add_argument(
        "--factor",
        type=str,
        choices=["subspace", "basis", "full"],
        required=True,
        help="Factor to test: 'subspace' for subspace dimensions, 'basis' for polynomial basis types, 'full' for factorial",  # noqa: E501
    )
    parser.add_argument(
        "--problem",
        type=str,
        required=True,
        help="Problem name (e.g., FACSIZE-1, DYNAMNEWS-1)",
    )

    # Problem configuration
    parser.add_argument(
        "--dim",
        type=int,
        default=10,
        help="Problem dimension (default: 10)",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=5000,
        help="Simulation budget (default: 5000)",
    )

    # Experiment settings
    parser.add_argument(
        "--n-macroreps",
        type=int,
        default=10,
        help="Number of macroreplications (default: 10)",
    )
    parser.add_argument(
        "--n-postreps",
        type=int,
        default=100,
        help="Number of postreplications (default: 100)",
    )

    # Factor-specific settings
    parser.add_argument(
        "--subspace-dims",
        type=str,
        default=None,
        help="Comma-separated list of subspace dimensions to test (default: 1-8)",
    )
    parser.add_argument(
        "--basis-types",
        type=str,
        default=None,
        help="Comma-separated list of basis types to test (default: all)",
    )
    parser.add_argument(
        "--fixed-subspace-dim",
        type=int,
        default=4,
        help="Fixed subspace dimension when testing basis types (default: 4)",
    )
    parser.add_argument(
        "--polynomial-degree",
        type=int,
        default=2,
        help="Polynomial degree (default: 2)",
    )

    # HPC settings
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="Task ID for SLURM array jobs (runs only that design point)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments",
        help="Output directory for results (default: experiments)",
    )
    parser.add_argument(
        "--generate-slurm",
        action="store_true",
        help="Generate SLURM scripts instead of running experiments",
    )
    parser.add_argument(
        "--generate-csv",
        action="store_true",
        help="Generate CSV design matrix",
    )

    # Logging
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Parse subspace dimensions
    subspace_dims = None
    if args.subspace_dims:
        subspace_dims = [int(x.strip()) for x in args.subspace_dims.split(",")]
        print(f"Using subspace dimensions: {subspace_dims}")
    else:
        subspace_dims = list(range(1, min(args.dim + 1, 9)))

    # Parse basis types
    basis_types = None
    if args.basis_types:
        basis_types = [
            poly_basis_from_string(x.strip()) for x in args.basis_types.split(",")
        ]
    else:
        basis_types = list(PolyBasisType)

    # Create experiment configuration
    config = ExperimentConfig(
        factor_type=args.factor,
        problem_name=args.problem,
        problem_dim=args.dim,
        budget=args.budget,
        n_macroreps=args.n_macroreps,
        n_postreps=args.n_postreps,
        subspace_dims=subspace_dims,
        basis_types=basis_types,
        fixed_subspace_dim=args.fixed_subspace_dim,
        polynomial_degree=args.polynomial_degree,
        task_id=args.task_id,
        output_dir=Path(args.output_dir),
    )

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate HPC scripts if requested
    if args.generate_slurm:
        logger.info("Generating SLURM array job script...")
        slurm_file = generate_slurm_array_script(config, config.output_dir)
        config_file = generate_hpc_config(config, config.output_dir)
        csv_file = generate_csv_design_matrix(config, config.output_dir)

        logger.info(f"Generated SLURM script: {slurm_file}")
        logger.info(f"Generated config file: {config_file}")
        logger.info(f"Generated design matrix: {csv_file}")
        logger.info(f"\nTo submit: sbatch {slurm_file}")
        return

    if args.generate_csv:
        csv_file = generate_csv_design_matrix(config, config.output_dir)
        logger.info(f"Generated design matrix: {csv_file}")
        return

    # Run experiments
    logger.info(f"Starting {args.factor} factor experiments on {args.problem}")

    if args.factor == "subspace":
        results = run_subspace_dimension_experiments(config)
    elif args.factor == "basis":
        results = run_polynomial_basis_experiments(config)
    else:  # full
        results = run_full_factorial_experiment(config)

    # Summary
    logger.info(f"\nCompleted {len(results)} design points")
    for result in results:
        logger.info(f"  - {result['design_point_id']}: {result['output_file']}")


if __name__ == "__main__":
    main()
