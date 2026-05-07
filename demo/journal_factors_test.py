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
    create_design,
    instantiate_problem,
    instantiate_solver,
)
from simopt.experiment import (
    scale_dimension, 
    ProblemSolver, 
    ProblemsSolvers
)
from simopt.solvers.astromorf import PolyBasisType

problems_optimal_hyper: dict = {
    "DYNAMNEWS-1": {"subspace_dimension": 5, "polynomial_degree": 2, "subproblem_regularisation": 0.41555633606245307, "ps_sufficient_reduction": 0.15328431662449404},  # Optimal
    "SAN-1": {"subspace_dimension": 20, "polynomial_degree": 3, "subproblem_regularisation": 0.3672349090967656, "ps_sufficient_reduction": 0.0},  # Optimal
    "ROSENBROCK-1": {"subspace_dimension": 20, "polynomial_degree": 2, "subproblem_regularisation": 0.20519775542894717, "ps_sufficient_reduction": 0.4542256535063405},  # Optimal
    "NETWORK-1": {"subspace_dimension": 14, "polynomial_degree": 4, "subproblem_regularisation": 0.13066366948000724, "ps_sufficient_reduction": 0.04598983542928581},  # optimal
    "PARAMESTI-1": {"subspace_dimension": 2, "polynomial_degree": 4, "subproblem_regularisation": 0.007341421780035384, "ps_sufficient_reduction": 0.8271734245206864},  # optimal
}

cabs_factors = {
    "SAN-1": {
        "gamma": 0.8828659814241923,
        "c_p": 0.05,
        "c_g": 0.8276721063224712,
        "eps_n": 0.075446911280005,
        "eps_a": 0.019727959607387935,
        "rho_max": 0.7,
        "w_safe": 9,
        "eta_safe": 0.012820503556735807,
        "c2_est": 2.932949582570289,
        "delta_inc_cap": 6
    },
    "NETWORK-1": {
        "gamma": 0.8647739721818226,
        "c_p": 0.5403447406081348,
        "c_g": 0.6186382309199568,
        "eps_n": 0.3569033022334193,
        "eps_a": 0.28876757267132447,
        "rho_max": 0.8396378178805266,
        "w_safe": 27,
        "eta_safe": 0.18097364726133436,
        "c2_est": 1.6545417830400007,
        "delta_inc_cap": 10
    },
    "DYNAMNEWS-1": {
        "gamma": 0.8750975631420834,
        "c_p": 0.49953709364063265,
        "c_g": 1.0620182449212108,
        "eps_n": 0.10625298620014088,
        "eps_a": 0.16205859972104644,
        "rho_max": 0.9176733547075296,
        "w_safe": 13,
        "eta_safe": 0.10514082537844505,
        "c2_est": 0.8597437563781269,
        "delta_inc_cap": 10
      },
    "ROSENBROCK-1": {
        "gamma": 0.9063311271815354,
        "c_p": 0.2822516618558208,
        "c_g": 0.20336157033204208,
        "eps_n": 0.1826008507019749,
        "eps_a": 0.06585786198277199,
        "rho_max": 0.8086197817978706,
        "w_safe": 28,
        "eta_safe": 0.15908771092198087,
        "c2_est": 1.6397546858971714,
        "delta_inc_cap": 6
    },
    "PARAMESTI-1": {
        "gamma": 0.9699167233004381,
        "c_p": 0.3291961798936611,
        "c_g": 0.2689741889666983,
        "eps_n": 0.30441179989308964,
        "eps_a": 0.06863955237379822,
        "rho_max": 0.8021774211019107,
        "w_safe": 18,
        "eta_safe": 0.19913824979089106,
        "c2_est": 2.430262228386464,
        "delta_inc_cap": 10
     },
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

    # ASTROMoRF subproblem regularisation factor.
    # Swept in factor_type="full"; held at fixed_regularisation otherwise.
    regularisation_values: list[float] = field(
        default_factory=lambda: [0.0, 0.05, 0.15, 0.5]
    )
    fixed_regularisation: float = 0.15

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
                "subproblem_regularisation": problems_optimal_hyper.get(
                    config.problem_name, {}
                ).get("subproblem_regularisation", config.fixed_regularisation),
                "ps_sufficient_reduction": problems_optimal_hyper.get(
                    config.problem_name, {}
                ).get("ps_sufficient_reduction", 0.0),
                "CABS factors": cabs_factors.get(config.problem_name, {}),
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
            "initial subspace dimension": problems_optimal_hyper.get(
                config.problem_name, {}
            ).get("initial subspace dimension", config.fixed_subspace_dim),
            "polynomial degree": config.polynomial_degree,
            "polynomial basis": basis_type,
            "adaptive subspace dimension": True, 
            "subproblem_regularisation": problems_optimal_hyper.get(
                config.problem_name, {}
            ).get("subproblem_regularisation", config.fixed_regularisation),
            "ps_sufficient_reduction": problems_optimal_hyper.get(
                config.problem_name, {}
            ).get("ps_sufficient_reduction", config.fixed_sufficient_reduction),
            "CABS factors": cabs_factors.get(config.problem_name, {}),
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

    # Generate all combinations: subspace x basis x subproblem_regularisation
    design_points = []
    for dim in config.subspace_dims:
        if dim > config.problem_dim:
            continue
        for basis_type in config.basis_types:
            for reg in config.regularisation_values:
                solver_factors = {
                    "initial subspace dimension": dim,
                    "polynomial degree": config.polynomial_degree,
                    "polynomial basis": basis_type,
                    "adaptive subspace dimension": False,
                    "subproblem_regularisation": reg,
                    "crn_across_solns": config.crn_across_solns,
                    "CABS factors": cabs_factors.get(config.problem_name, {}),
                }
                basis_name = POLY_BASIS_NAMES.get(basis_type, basis_type.value)
                # reg encoded with fixed precision so design-point IDs are stable
                reg_tag = f"{reg:.4g}".replace(".", "p")
                design_point_id = (
                    f"ASTROMORF_d{dim}_basis_{basis_name}_reg{reg_tag}"
                    f"_on_{config.problem_name}"
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
# SENSITIVITY AGGREGATION
# ============================================================================
#
# Aggregates per-design-point pickles written by run_single_design_point into
# main-effects and pairwise-interaction summaries across the three factors:
#     subspace dimension, polynomial basis, subproblem_regularisation.
#
# Reads existing pickles only — does not change how they are written.


def _basis_to_str(value: Any) -> str:
    """Stringify a polynomial basis factor (enum or str) for grouping."""
    if isinstance(value, PolyBasisType):
        return POLY_BASIS_NAMES.get(value, value.value)
    return str(value)


def _final_objective_stats(
    experiment: Any,
) -> tuple[float | None, float | None, int]:
    """Extract the mean and std of the final-budget objective across macroreps.

    Uses post-replicated objectives when available (preferred — these are the
    quantities used for performance comparison), and falls back to the
    in-run estimated objectives otherwise.
    """
    macro_finals: list[float] = []

    # Preferred: post-replicated objective curves on each macroreplicate.
    objective_curves = getattr(experiment, "objective_curves", None)
    if objective_curves:
        for curve in objective_curves:
            y_vals = getattr(curve, "y_vals", None)
            if y_vals is not None and len(y_vals) > 0:
                macro_finals.append(float(y_vals[-1]))

    # Fallback: raw per-macrorep estimated objectives produced during run().
    if not macro_finals:
        all_est = getattr(experiment, "all_est_objectives", None) or []
        for mrep_vals in all_est:
            if mrep_vals is not None and len(mrep_vals) > 0:
                macro_finals.append(float(mrep_vals[-1]))

    if not macro_finals:
        return None, None, 0

    arr = np.asarray(macro_finals, dtype=float)
    mean_val = float(arr.mean())
    std_val = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return mean_val, std_val, arr.size


def _extract_design_point_record(
    pickle_path: Path,
) -> dict[str, Any] | None:
    """Load one design-point pickle and return a flat record of factors+metrics.

    Returns None if the pickle cannot be read or does not contain the
    factors needed for sensitivity grouping.
    """
    import pickle  # local import: only needed during aggregation

    try:
        with pickle_path.open("rb") as f:
            experiment = pickle.load(f)  # noqa: S301 — trusted internal artifact
    except Exception as e:  # noqa: BLE001
        logging.getLogger(__name__).warning(
            f"Skipping unreadable pickle {pickle_path.name}: {e}"
        )
        return None

    solver = getattr(experiment, "solver", None)
    problem = getattr(experiment, "problem", None)
    if solver is None or problem is None:
        return None

    factors = getattr(solver, "factors", {}) or {}
    subspace_dim = factors.get("initial subspace dimension")
    basis = factors.get("polynomial basis")
    regularisation = factors.get("subproblem_regularisation")

    if subspace_dim is None or basis is None or regularisation is None:
        # Older pickles (pre-regularisation) — flag with NaN so they are
        # excluded from grouping rather than corrupting summaries.
        return None

    mean_obj, std_obj, n_macro = _final_objective_stats(experiment)
    if mean_obj is None:
        return None

    return {
        "pickle_file": pickle_path.name,
        "problem_name": getattr(problem, "name", ""),
        "subspace_dim": int(subspace_dim),
        "basis": _basis_to_str(basis),
        "regularisation": float(regularisation),
        "polynomial_degree": factors.get("polynomial degree"),
        "n_macroreps": n_macro,
        "mean_obj": mean_obj,
        "std_obj": std_obj if std_obj is not None else float("nan"),
    }


def _group_mean_std(
    rows: list[dict[str, Any]], keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Group rows by the given factor keys and aggregate mean_obj across them.

    Each design-point row contributes its mean_obj as a single observation,
    so the std reported here is the dispersion of design-point means within
    each group (i.e. variation explained by factor levels not in `keys`).
    """
    buckets: dict[tuple, list[float]] = {}
    for row in rows:
        bucket_key = tuple(row[k] for k in keys)
        buckets.setdefault(bucket_key, []).append(row["mean_obj"])

    out: list[dict[str, Any]] = []
    for bucket_key, values in buckets.items():
        arr = np.asarray(values, dtype=float)
        record: dict[str, Any] = dict(zip(keys, bucket_key))
        record["n_design_points"] = int(arr.size)
        record["mean_obj"] = float(arr.mean())
        record["std_obj"] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        record["min_obj"] = float(arr.min())
        record["max_obj"] = float(arr.max())
        out.append(record)
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of homogeneous-keyed dict rows as a CSV file."""
    if not rows:
        # Always create the file so downstream pipelines do not see a missing
        # path on HPC; but leave it empty + header-less.
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_sensitivity_results(
    output_dir: Path, config: ExperimentConfig
) -> Path | None:
    """Aggregate per-design-point pickles into main-effects + interaction CSVs.

    Scans `output_dir` for the POSTREPS pickles emitted by
    `run_single_design_point`, extracts factor levels and final-budget
    objective statistics, and writes:

        - design_points_summary.csv : one row per design point
        - summary_main_effects.csv  : main effect of each factor
        - summary_interactions.csv  : pairwise grouped means

    Returns the output directory on success, or None if no pickles were
    found (e.g. SLURM array workers running in isolation).
    """
    logger = logging.getLogger(__name__)

    # POSTREPS pickles are the canonical post-processed artifacts.
    candidates = sorted(output_dir.glob("*_POSTREPS.pickle"))

    # Fall back to all design-point pickles if POSTREPS are absent (older runs).
    if not candidates:
        candidates = sorted(
            p
            for p in output_dir.glob("*.pickle")
            if "_POSTREPS" not in p.name and "hpc_config" not in p.name
        )

    if not candidates:
        logger.info(
            f"No design-point pickles found in {output_dir}; "
            "skipping sensitivity aggregation."
        )
        return None

    logger.info(
        f"Aggregating sensitivity summaries over {len(candidates)} design points"
    )

    rows: list[dict[str, Any]] = []
    for pkl in candidates:
        rec = _extract_design_point_record(pkl)
        if rec is not None:
            rows.append(rec)

    if not rows:
        logger.warning(
            "No usable design-point records extracted; "
            "summaries not written."
        )
        return None

    # Per-design-point flat table.
    design_points_path = output_dir / "design_points_summary.csv"
    _write_csv(design_points_path, rows)

    # Main effects: average mean_obj across all other factors.
    main_effects: list[dict[str, Any]] = []
    for factor_key in ("subspace_dim", "basis", "regularisation"):
        for grp in _group_mean_std(rows, (factor_key,)):
            main_effects.append({"factor": factor_key, "level": grp[factor_key],
                                 "n_design_points": grp["n_design_points"],
                                 "mean_obj": grp["mean_obj"],
                                 "std_obj": grp["std_obj"],
                                 "min_obj": grp["min_obj"],
                                 "max_obj": grp["max_obj"]})
    _write_csv(output_dir / "summary_main_effects.csv", main_effects)

    # Pairwise interactions: grouped means over each factor pair.
    pairwise: list[dict[str, Any]] = []
    pair_keys: tuple[tuple[str, str], ...] = (
        ("subspace_dim", "basis"),
        ("subspace_dim", "regularisation"),
        ("basis", "regularisation"),
    )
    for keys in pair_keys:
        for grp in _group_mean_std(rows, keys):
            pairwise.append({
                "factor_a": keys[0],
                "level_a": grp[keys[0]],
                "factor_b": keys[1],
                "level_b": grp[keys[1]],
                "n_design_points": grp["n_design_points"],
                "mean_obj": grp["mean_obj"],
                "std_obj": grp["std_obj"],
                "min_obj": grp["min_obj"],
                "max_obj": grp["max_obj"],
            })
    _write_csv(output_dir / "summary_interactions.csv", pairwise)

    logger.info(
        f"Sensitivity summaries written: "
        f"{design_points_path.name}, summary_main_effects.csv, "
        f"summary_interactions.csv "
        f"(problem={config.problem_name}, factor_type={config.factor_type})"
    )
    return output_dir


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
        # Full factorial: subspace x basis x subproblem_regularisation
        n_design_points = (
            len([d for d in config.subspace_dims if d <= config.problem_dim])
            * len(config.basis_types)
            * len(config.regularisation_values)
        )

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
        "regularisation_values": (
            list(config.regularisation_values)
            if config.factor_type == "full"
            else [config.fixed_regularisation]
        ),
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
        n_tasks = (
            len([d for d in config.subspace_dims if d <= config.problem_dim])
            * len(config.basis_types)
            * len(config.regularisation_values)
        )

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
                        "subproblem_regularisation": config.fixed_regularisation,
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
                    "subproblem_regularisation": config.fixed_regularisation,
                    "problem_name": config.problem_name,
                    "problem_dim": config.problem_dim,
                    "budget": config.budget,
                }
            )
            task_id += 1

    else:  # Full factorial: subspace x basis x subproblem_regularisation
        for dim in config.subspace_dims:
            if dim > config.problem_dim:
                continue
            for basis_type in config.basis_types:
                for reg in config.regularisation_values:
                    basis_name = POLY_BASIS_NAMES.get(basis_type, basis_type.value)
                    reg_tag = f"{reg:.4g}".replace(".", "p")
                    rows.append(
                        {
                            "task_id": task_id,
                            "design_point_id": f"ASTROMORF_d{dim}_basis_{basis_name}_reg{reg_tag}_on_{config.problem_name}",  # noqa: E501
                            "subspace_dim": dim,
                            "polynomial_basis": basis_type.value,
                            "polynomial_degree": config.polynomial_degree,
                            "subproblem_regularisation": reg,
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
    regularisation_values: list[float] | None = None,
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

    if regularisation_values is None:
        regularisation_values = [0.0, 0.05, 0.15, 0.5]

    if factor_type == "subspace":
        # Use cross_design_factors for subspace dimensions (discrete levels)
        # alongside subproblem_regularisation as a co-swept factor.
        if subspace_dims is None:
            subspace_dims = list(range(1, min(problem_dim + 1, 9)))

        solver_factor_headers = []  # No continuous factors
        solver_factor_settings = []
        solver_fixed_factors = {
            "adaptive subspace dimension": False,
            "polynomial degree": 2,
        }
        solver_cross_design_factors = {
            "crn_across_solns": [False],
            "initial subspace dimension": subspace_dims,
            "subproblem_regularisation": regularisation_values,
        }

    elif factor_type == "basis":
        # Use cross_design_factors for polynomial basis types
        # alongside subproblem_regularisation as a co-swept factor.
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
            "subproblem_regularisation": regularisation_values,
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
    parser.add_argument(
        "--regularisation-values",
        type=str,
        default=None,
        help="Comma-separated list of subproblem regularisation values to "
        "sweep in --factor full (default: 0.0,0.05,0.15,0.5)",
    )
    parser.add_argument(
        "--fixed-regularisation",
        type=float,
        default=0.15,
        help="Fixed subproblem regularisation when not sweeping it "
        "(default: 0.15)",
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
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip running design points; only aggregate existing pickles in "
        "--output-dir into sensitivity summary CSVs",
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

    # Parse subproblem regularisation values
    if args.regularisation_values:
        regularisation_values = [
            float(x.strip()) for x in args.regularisation_values.split(",")
        ]
    else:
        regularisation_values = [0.0, 0.05, 0.15, 0.5]

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
        regularisation_values=regularisation_values,
        fixed_regularisation=args.fixed_regularisation,
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

    if args.aggregate_only:
        aggregate_sensitivity_results(config.output_dir, config)
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

    # Sensitivity aggregation: only when the full design has been run in this
    # process (i.e. not a single SLURM array worker). On HPC, schedule a
    # follow-up job that re-invokes this script with --aggregate-only after
    # the array completes; that path also uses this same function.
    if config.task_id is None:
        aggregate_sensitivity_results(config.output_dir, config)


if __name__ == "__main__":
    main()
