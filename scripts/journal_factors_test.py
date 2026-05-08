r"""Journal-quality datafarming experiments for the ASTROMoRF solver.

Three controlled sensitivity studies are supported.  Each study configures
ASTROMoRF differently, because the experimental question dictates the
adaptivity regime that is appropriate:

    Study               Adaptive subspace?   Why
    ─────────────────────────────────────────────────────────────────────
    subspace            NO  (fixed)          Studies sensitivity to a
                                              chosen FIXED subspace dim;
                                              CABS would mask the effect.
    basis               YES (CABS on)        Evaluates basis choice under
                                              the modern adaptive regime.
    regularisation      YES (CABS on)        Evaluates regularisation
                                              under the modern adaptive
                                              regime.

A ``full`` factorial mode is also supported.  Because the full factorial
sweeps the subspace dimension, it inherits the ``subspace`` study's
NON-adaptive configuration (sweeping a factor while CABS overrides it
would be incoherent).

Each study has its own explicit solver-factor builder; there is no shared
``adaptive`` flag that callers must remember to flip.  Adaptive parameters
come from a single per-problem registry (`ADAPTIVE_CONFIGS`).

The script is structured for embarrassingly parallel HPC execution: each
design point is independent, identified by a deterministic integer task
ID, and writes its own output files.

Examples:
--------
Local execution (full grid, current process)::

    python demo/journal_factors_test.py --study basis \\
        --problem SAN-1 --dim 20 --budget 5000

SLURM array (one task per design point)::

    python demo/journal_factors_test.py --study regularisation \\
        --problem SAN-1 --dim 20 --budget 5000 \\
        --task-id $SLURM_ARRAY_TASK_ID

Range execution (one worker takes a slice of the grid)::

    python demo/journal_factors_test.py --study full \\
        --problem SAN-1 --dim 20 --budget 5000 \\
        --design-start 0 --design-end 24
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Ensure the project root is importable when running the script directly.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from simopt.experiment import ProblemSolver, ProblemsSolvers, scale_dimension
from simopt.experiment.dimension_scaling import is_scalable
from simopt.experiment_base import Problem, instantiate_solver
from simopt.solvers.astromorf import CABS_DEFAULTS, PolyBasisType

# ============================================================================
# CONSTANTS
# ============================================================================

# Per-problem "best known" hyperparameters used to fix non-swept factors at a
# sensible operating point during a single-factor sensitivity study.
PROBLEM_OPTIMAL_HYPER: dict[str, dict[str, Any]] = {
    "DYNAMNEWS-1": {
        "subspace_dimension": 5,
        "polynomial_degree": 2,
        "subproblem_regularisation": 0.41555633606245307,
        "ps_sufficient_reduction": 0.15328431662449404,
        "polynomial basis": PolyBasisType.CHEBYSHEV,
        "lambda_min": 10
        },  
    "SAN-1": {
        "subspace_dimension": 20,
        "polynomial_degree": 3,
        "subproblem_regularisation": 0.3672349090967656,
        "ps_sufficient_reduction": 0.0,
        "polynomial basis": PolyBasisType.CHEBYSHEV,
        "lambda_min": 10
    },  
    "ROSENBROCK-1": {
        "subspace_dimension": 13,
        "polynomial_degree": 2,
        "subproblem_regularisation": 0.31317227223216765,
        "ps_sufficient_reduction": 0.28682048799461674,
        "polynomial basis": PolyBasisType.CHEBYSHEV,
        "lambda_min": 10
    },  
    "NETWORK-1": {
        "subspace_dimension": 14,
        "polynomial_degree": 4,
        "subproblem_regularisation": 0.13066366948000724,
        "ps_sufficient_reduction": 0.04598983542928581,
        "polynomial basis": PolyBasisType.CHEBYSHEV,
        "lambda_min": 10
    },  
    "PARAMESTI-1": {
        "subspace_dimension": 2,
        "polynomial_degree": 2,
        "subproblem_regularisation": 0.2203918801750902,
        "ps_sufficient_reduction": 0.7588175603433991,
        "polynomial basis": PolyBasisType.CHEBYSHEV,
        "lambda_min": 24
    },  
}

# Readable labels for the polynomial basis enum (used in design-point IDs and CSVs).
POLY_BASIS_NAMES: dict[PolyBasisType, str] = {
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

# Defaults used when the problem is not in PROBLEM_OPTIMAL_HYPER.
DEFAULT_POLY_DEGREE = 2
DEFAULT_FIXED_SUBSPACE_DIM = 4
DEFAULT_FIXED_BASIS = PolyBasisType.HERMITE
DEFAULT_FIXED_REGULARISATION = 0.15
DEFAULT_PS_SUFFICIENT_REDUCTION = 0.0

# Default factor-level sweeps.
DEFAULT_REGULARISATION_VALUES: tuple[float, ...] = (0.0, 0.05, 0.15, 0.5)
DEFAULT_BASIS_TYPES: tuple[PolyBasisType, ...] = tuple(PolyBasisType)


# ── Adaptive (CABS) configuration registry ────────────────────────────────
#
# Centralised registry of CABS hyperparameters used by the adaptive studies
# (basis and regularisation).  Keys are (problem_name, problem_dim) — entries
# specific to a (problem, dimension) pair take precedence; fall back to a
# (problem_name, None) entry that applies at any dimension.
#
# Studies configured as adaptive (`StudySpec.adaptive == True`) MUST resolve
# a non-None entry through `get_adaptive_config`; otherwise the study errors
# out at design-grid construction time, before any expensive run starts.
ADAPTIVE_CONFIGS: dict[tuple[str, int | None], dict[str, Any]] = {
    ("SAN-1", None): {
        "gamma": 0.8828659814241923,
        "c_p": 0.05,
        "c_g": 0.8276721063224712,
        "eps_n": 0.075446911280005,
        "eps_a": 0.019727959607387935,
        "rho_max": 0.7,
        "w_safe": 9,
        "eta_safe": 0.012820503556735807,
        "c2_est": 2.932949582570289,
        "delta_inc_cap": 6,
    },
    ("NETWORK-1", None): {
        "gamma": 0.8647739721818226,
        "c_p": 0.5403447406081348,
        "c_g": 0.6186382309199568,
        "eps_n": 0.3569033022334193,
        "eps_a": 0.28876757267132447,
        "rho_max": 0.8396378178805266,
        "w_safe": 27,
        "eta_safe": 0.18097364726133436,
        "c2_est": 1.6545417830400007,
        "delta_inc_cap": 10,
    },
    ("DYNAMNEWS-1", None): {
        "gamma": 0.8750975631420834,
        "c_p": 0.49953709364063265,
        "c_g": 1.0620182449212108,
        "eps_n": 0.10625298620014088,
        "eps_a": 0.16205859972104644,
        "rho_max": 0.9176733547075296,
        "w_safe": 13,
        "eta_safe": 0.10514082537844505,
        "c2_est": 0.8597437563781269,
        "delta_inc_cap": 10,
    },
    ("ROSENBROCK-1", None): {
        "gamma": 0.9063311271815354,
        "c_p": 0.2822516618558208,
        "c_g": 0.20336157033204208,
        "eps_n": 0.1826008507019749,
        "eps_a": 0.06585786198277199,
        "rho_max": 0.8086197817978706,
        "w_safe": 28,
        "eta_safe": 0.15908771092198087,
        "c2_est": 1.6397546858971714,
        "delta_inc_cap": 6,
    },
    ("PARAMESTI-1", None): {
        "gamma": 0.9348946256587236,
        "c_p": 0.40895574114002264,
        "c_g": 0.43261934903127425,
        "eps_n": 0.4522822310706563,
        "eps_a": 0.2956122095548452,
        "rho_max": 0.7653562957256419,
        "w_safe": 29,
        "eta_safe": 0.022466105434918077,
        "c2_est": 3.0,
        "delta_inc_cap": 8
    },
}


def get_adaptive_config(
    problem_name: str, problem_dim: int
) -> dict[str, Any] | None:
    """Look up the CABS factor dict for a (problem, dim) combination.

    Resolution order:
        1. Exact (problem_name, problem_dim) entry.
        2. Problem-wide (problem_name, None) entry.
        3. None — caller must decide whether to error or fall back.

    Returns a fresh dict copy so the registry is never mutated by callers.
    """
    entry = ADAPTIVE_CONFIGS.get((problem_name, problem_dim))
    if entry is None:
        entry = ADAPTIVE_CONFIGS.get((problem_name, None))
    return dict(entry) if entry is not None else None


# ============================================================================
# STUDY DESCRIPTORS
# ============================================================================
#
# Each study type is described by a `StudySpec` that captures the policy
# decisions that distinguish it from the others (adaptivity, what is swept,
# how design-point IDs are formed).  All study-specific divergence is
# concentrated in `solver_factor_builder` — the rest of the pipeline is
# study-agnostic.


@dataclass(frozen=True)
class StudySpec:
    """Static description of a single sensitivity study."""

    name: str
    sweeps_subspace: bool
    sweeps_basis: bool
    sweeps_regularisation: bool
    adaptive: bool
    description: str
    solver_factor_builder: Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class DesignPoint:
    """A single design point in a study's factor grid."""

    task_id: int
    design_point_id: str
    subspace_dim: int
    polynomial_basis: PolyBasisType
    regularisation: float

    def basis_label(self) -> str:
        """Generates Polynomial Basis Label.

        Returns:
            str: Polynomial basis label for design point.
        """
        return POLY_BASIS_NAMES.get(self.polynomial_basis, self.polynomial_basis.value)


# ============================================================================
# SOLVER-FACTOR BUILDERS (one per study)
# ============================================================================
#
# These are the ONLY places that construct the ASTROMoRF factor dict.  Each
# builder explicitly sets the adaptivity regime its study requires, so no
# caller has to remember to toggle a global flag.


def _common_solver_factors(
        config: ExperimentConfig, 
        point: DesignPoint
        ) -> dict[str, Any]:
    """Factors that every study sets the same way.

    Excludes anything study-specific (adaptivity, CABS, the swept factor).
    """
    return {
        "initial subspace dimension": int(point.subspace_dim),
        "polynomial basis": point.polynomial_basis,
        "subproblem_regularisation": float(point.regularisation),
        "polynomial degree": int(config.polynomial_degree),
        "ps_sufficient_reduction": float(config.ps_sufficient_reduction),
        "crn_across_solns": bool(config.crn_across_solns),
    }


def build_subspace_study_solver(
    config: ExperimentConfig, point: DesignPoint
) -> dict[str, Any]:
    """Strictly NON-adaptive ASTROMoRF for the subspace-dimension study.

    Adaptive subspace updates are explicitly disabled so the prescribed
    `initial subspace dimension` is honoured for the entire run.  CABS is
    not configured: with adaptivity off the solver constructs a CABS
    selector from the default factor block but never queries it.
    """
    factors = _common_solver_factors(config, point)
    factors["adaptive subspace dimension"] = False
    # CABS factors are deliberately omitted here — the solver will use its
    # own defaults block, but the selector is never consulted while
    # `adaptive subspace dimension` is False.
    return factors


def build_basis_study_solver(
    config: ExperimentConfig, point: DesignPoint
) -> dict[str, Any]:
    """ADAPTIVE ASTROMoRF for the polynomial-basis study.

    The CABS-driven adaptive regime is enabled and seeded from the
    per-problem `ADAPTIVE_CONFIGS` registry.  The sweep variable here is
    the polynomial basis only.
    """
    cabs = config.adaptive_factors
    if cabs is None:  # defensive — design-grid validation should have caught this
        raise RuntimeError(
            f"basis study requires an adaptive config for {config.problem_name}"
        )
    factors = _common_solver_factors(config, point)
    factors["adaptive subspace dimension"] = True
    factors["CABS factors"] = cabs
    return factors


def build_regularisation_study_solver(
    config: ExperimentConfig, point: DesignPoint
) -> dict[str, Any]:
    """ADAPTIVE ASTROMoRF for the subproblem-regularisation study.

    Mirrors the basis study's adaptivity setup; the sweep variable is the
    subproblem regularisation factor.
    """
    cabs = config.adaptive_factors
    if cabs is None:
        raise RuntimeError(
            f"regularisation study requires an "
            f"adaptive config for {config.problem_name}"
        )
    factors = _common_solver_factors(config, point)
    factors["adaptive subspace dimension"] = True
    factors["CABS factors"] = cabs
    return factors


def build_full_study_solver(
    config: ExperimentConfig, point: DesignPoint
) -> dict[str, Any]:
    """NON-adaptive ASTROMoRF for the full factorial.

    The full factorial sweeps the subspace dimension as one of its factors;
    enabling CABS would override it.  This study therefore reuses the
    subspace-study's fixed-subspace configuration.
    """
    return build_subspace_study_solver(config, point)


# ============================================================================
# STUDY REGISTRY
# ============================================================================


STUDY_SPECS: dict[str, StudySpec] = {
    "subspace": StudySpec(
        name="subspace",
        sweeps_subspace=True,
        sweeps_basis=False,
        sweeps_regularisation=False,
        adaptive=False,
        description="Sensitivity to the (fixed) initial subspace dimension.",
        solver_factor_builder=build_subspace_study_solver,
    ),
    "basis": StudySpec(
        name="basis",
        sweeps_subspace=False,
        sweeps_basis=True,
        sweeps_regularisation=False,
        adaptive=True,
        description="Sensitivity to the polynomial surrogate basis (adaptive solver).",
        solver_factor_builder=build_basis_study_solver,
    ),
    "regularisation": StudySpec(
        name="regularisation",
        sweeps_subspace=False,
        sweeps_basis=False,
        sweeps_regularisation=True,
        adaptive=True,
        description="Sensitivity to the subproblem regularisation (adaptive solver).",
        solver_factor_builder=build_regularisation_study_solver,
    ),
    "full": StudySpec(
        name="full",
        sweeps_subspace=True,
        sweeps_basis=True,
        sweeps_regularisation=True,
        adaptive=False,
        description="Full factorial across all three factors (non-adaptive).",
        solver_factor_builder=build_full_study_solver,
    ),
}

STUDY_CHOICES = tuple(STUDY_SPECS.keys())


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class ExperimentConfig:
    """Top-level configuration for a journal-factor experiment run."""

    study: str
    problem_name: str
    problem_dim: int
    budget: int

    n_macroreps: int = 10
    n_postreps: int = 100
    n_postreps_init_opt: int = 200

    # CRN flags, plumbed through to the solver and post-replication.
    crn_across_solns: bool = False
    crn_across_budget: bool = True
    crn_across_macroreps: bool = False
    crn_across_init_opt: bool = True

    # Factor sweep levels.
    subspace_dims: tuple[int, ...] = ()
    basis_types: tuple[PolyBasisType, ...] = DEFAULT_BASIS_TYPES
    regularisation_values: tuple[float, ...] = DEFAULT_REGULARISATION_VALUES

    # Held-fixed values when a factor is NOT being swept.
    fixed_subspace_dim: int = DEFAULT_FIXED_SUBSPACE_DIM
    fixed_basis: PolyBasisType = DEFAULT_FIXED_BASIS
    fixed_regularisation: float = DEFAULT_FIXED_REGULARISATION
    polynomial_degree: int = DEFAULT_POLY_DEGREE

    # HPC / output settings.
    output_dir: Path = field(default_factory=lambda: Path("experiments"))
    task_id: int | None = None
    design_start: int | None = None
    design_end: int | None = None

    # Resolved adaptive config (populated in __post_init__ for adaptive studies).
    adaptive_factors: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Check validity of the config and resolve derived fields.

        Checks class invariants and raises ValueError if any are violated.  
        Also resolves derived fields like `adaptive_factors` and fills in 
        defaults from the `PROBLEM_OPTIMAL_HYPER` catalogue.

        Raises:
            ValueError: If the study is not valid.
            ValueError: If the problem dimension is not positive.
            ValueError: If the budget is not positive.
            ValueError: If the subspace dimensions are outside the valid range.
            ValueError: If the regularisation values are outside the valid range.
            ValueError: If the adaptive config is required but not found.
        """
        if self.study not in STUDY_SPECS:
            raise ValueError(
                f"study must be one of {STUDY_CHOICES!r}, got {self.study!r}"
            )
        if self.problem_dim <= 0:
            raise ValueError(f"problem_dim must be positive, got {self.problem_dim}")
        if self.budget <= 0:
            raise ValueError(f"budget must be positive, got {self.budget}")

        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Resolve per-problem fixed values from the optimal-hyper catalogue.
        opt = PROBLEM_OPTIMAL_HYPER.get(self.problem_name, {})
        self.polynomial_degree = int(
            opt.get(
                "polynomial_degree", 
                self.polynomial_degree
                )
            )
        self.fixed_regularisation = float(
            opt.get("regularisation", self.fixed_regularisation)
        )
        cat_dim = int(opt.get("subspace_dim", self.fixed_subspace_dim))
        self.fixed_subspace_dim = max(1, min(cat_dim, self.problem_dim))

        # Default subspace sweep: 1..min(8, problem_dim).
        if not self.subspace_dims:
            self.subspace_dims = tuple(range(1, min(self.problem_dim, 8) + 1))

        # Validate sweep levels.
        bad_dims = [d for d in self.subspace_dims if d < 1 or d > self.problem_dim]
        if bad_dims:
            raise ValueError(
                f"subspace_dims contains values outside [1, {self.problem_dim}]: "
                f"{bad_dims}"
            )
        bad_reg = [r for r in self.regularisation_values if r < 0 or r > 1]
        if bad_reg:
            raise ValueError(
                f"regularisation_values must lie in [0, 1]; offending: {bad_reg}"
            )

        # Resolve adaptive config for adaptive studies.  Fail loudly here so
        # SLURM array workers don't silently fall back to default CABS.
        if STUDY_SPECS[self.study].adaptive:
            cabs = get_adaptive_config(self.problem_name, self.problem_dim)
            if cabs is None:
                raise ValueError(
                    f"Study {self.study!r} requires an adaptive (CABS) config "
                    f"for problem {self.problem_name!r} (dim={self.problem_dim}); "
                    f"register one in ADAPTIVE_CONFIGS."
                )
            # Ensure every CABS key is present — fill from solver defaults.
            merged = dict(CABS_DEFAULTS)
            merged.update(cabs)
            self.adaptive_factors = merged
        else:
            self.adaptive_factors = None

    @property
    def spec(self) -> StudySpec:
        """Accessor for study class variable.

        Returns:
            StudySpec: The current study of the Experiment.
        """
        return STUDY_SPECS[self.study]

    @property
    def ps_sufficient_reduction(self) -> float:
        """Accessor for ps_sufficient_reduction.

        Returns:
            float: The problem-specific sufficient reduction factor for 
            the projected search, used in the solver factors.  
            Falls back to a global default if not specified for the 
            problem.
        """
        return float(
            PROBLEM_OPTIMAL_HYPER.get(self.problem_name, {}).get(
                "ps_sufficient_reduction", DEFAULT_PS_SUFFICIENT_REDUCTION
            )
        )


# ============================================================================
# DESIGN-POINT GENERATION
# ============================================================================


def _reg_tag(reg: float) -> str:
    """Stable string tag for a regularisation value (used in design-point IDs)."""
    return f"{reg:.4g}".replace(".", "p")


def build_design_grid(config: ExperimentConfig) -> list[DesignPoint]:
    """Build the deterministic, ordered list of design points for `config`.

    The ordering is deterministic so SLURM array task IDs are reproducible
    across re-runs and partial reruns.
    """
    grid: list[DesignPoint] = []
    fixed_basis = config.fixed_basis
    fixed_reg = config.fixed_regularisation
    fixed_dim = config.fixed_subspace_dim
    spec = config.spec

    if spec.name == "subspace":
        for dim in config.subspace_dims:
            grid.append(
                DesignPoint(
                    task_id=len(grid),
                    design_point_id=(
                        f"ASTROMORF_subspace_{dim}_on_{config.problem_name}"
                    ),
                    subspace_dim=dim,
                    polynomial_basis=fixed_basis,
                    regularisation=fixed_reg,
                )
            )

    elif spec.name == "basis":
        for basis in config.basis_types:
            label = POLY_BASIS_NAMES.get(basis, basis.value)
            grid.append(
                DesignPoint(
                    task_id=len(grid),
                    design_point_id=(
                        f"ASTROMORF_basis_{label}_on_{config.problem_name}"
                    ),
                    subspace_dim=fixed_dim,
                    polynomial_basis=basis,
                    regularisation=fixed_reg,
                )
            )

    elif spec.name == "regularisation":
        for reg in config.regularisation_values:
            grid.append(
                DesignPoint(
                    task_id=len(grid),
                    design_point_id=(
                        f"ASTROMORF_reg{_reg_tag(reg)}_on_{config.problem_name}"
                    ),
                    subspace_dim=fixed_dim,
                    polynomial_basis=fixed_basis,
                    regularisation=reg,
                )
            )

    elif spec.name == "full":
        for dim in config.subspace_dims:
            for basis in config.basis_types:
                label = POLY_BASIS_NAMES.get(basis, basis.value)
                for reg in config.regularisation_values:
                    grid.append(
                        DesignPoint(
                            task_id=len(grid),
                            design_point_id=(
                                f"ASTROMORF_d{dim}_basis_{label}"
                                f"_reg{_reg_tag(reg)}_on_{config.problem_name}"
                            ),
                            subspace_dim=dim,
                            polynomial_basis=basis,
                            regularisation=reg,
                        )
                    )

    else:  # pragma: no cover - guarded by ExperimentConfig
        raise ValueError(f"Unknown study: {spec.name!r}")

    # Validation: cheap structural checks before any expensive run.
    for p in grid:
        if p.subspace_dim < 1 or p.subspace_dim > config.problem_dim:
            raise ValueError(
                f"Design point {p.design_point_id!r} has subspace_dim "
                f"{p.subspace_dim} outside [1, {config.problem_dim}]"
            )
        if not 0.0 <= p.regularisation <= 1.0:
            raise ValueError(
                f"Design point {p.design_point_id!r} has regularisation "
                f"{p.regularisation} outside [0, 1]"
            )

    return grid


# ============================================================================
# SINGLE DESIGN-POINT EXECUTION
# ============================================================================


def _scaled_problem(config: ExperimentConfig) -> Problem:
    """Build the scaled problem instance for `config`.

    Falls back to the native dimension if the problem is not registered as
    scalable.
    """
    if not is_scalable(config.problem_name):
        logging.getLogger(__name__).warning(
            "Problem %r has no registered scaler; using its native dimension.",
            config.problem_name,
        )
        return scale_dimension(
            problem_name=config.problem_name, budget=config.budget, dimension=None
        )
    return scale_dimension(
        problem_name=config.problem_name,
        budget=config.budget,
        dimension=config.problem_dim,
    )


def run_design_point(
    config: ExperimentConfig, point: DesignPoint
) -> dict[str, Any]:
    """Run a single design point: instantiate, simulate, post-replicate, persist.

    Output filenames follow the legacy convention so downstream plotting
    pipelines continue to work unchanged.
    """
    logger = logging.getLogger(__name__)

    output_pickle = config.output_dir / f"{point.design_point_id}.pickle"
    postreps_pickle = config.output_dir / (
        f"{config.problem_name}_{config.study}_experiment_on_"
        f"{point.design_point_id}_POSTREPS.pickle"
    )
    log_path = output_pickle.with_suffix(".txt")

    logger.info(
        "Running %s study, design point %s -> %s",
        config.study, point.design_point_id, output_pickle.name,
    )

    solver_factors = config.spec.solver_factor_builder(config, point)
    logger.debug("Solver factors: %s", solver_factors)

    solver = instantiate_solver(
        solver_name="ASTROMORF",
        fixed_factors=solver_factors,
        solver_rename="ASTROMoRF",
    )
    problem = _scaled_problem(config)

    if problem.dim != config.problem_dim and is_scalable(config.problem_name):
        raise RuntimeError(
            f"scaled problem dim mismatch: got {problem.dim}, "
            f"expected {config.problem_dim}"
        )

    experiment = ProblemSolver(
        problem=problem,
        solver=solver,
        file_name_path=str(output_pickle),
    )

    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    experiment.run(n_macroreps=config.n_macroreps, n_jobs=n_jobs)

    experiment.post_replicate(
        n_postreps=config.n_postreps,
        crn_across_budget=config.crn_across_budget,
        crn_across_macroreps=config.crn_across_macroreps,
    )

    experiment.record_experiment_results(str(postreps_pickle))
    experiment.log_experiment_results(file_path=str(log_path))

    logger.info("Completed %s", point.design_point_id)
    return {
        "task_id": point.task_id,
        "design_point_id": point.design_point_id,
        "study": config.study,
        "adaptive": config.spec.adaptive,
        "output_file": str(output_pickle),
        "postreps_file": str(postreps_pickle),
        "subspace_dim": point.subspace_dim,
        "polynomial_basis": point.polynomial_basis.value,
        "regularisation": point.regularisation,
    }


# ============================================================================
# DRIVER: SELECT WHICH DESIGN POINTS TO RUN
# ============================================================================


def _selected_points(
    config: ExperimentConfig, grid: list[DesignPoint]
) -> Iterable[DesignPoint]:
    """Decide which design points to execute based on the CLI selection flags.

    Precedence: --task-id > (--design-start, --design-end) > all.
    """
    if config.task_id is not None:
        if not 0 <= config.task_id < len(grid):
            raise IndexError(
                f"task_id {config.task_id} out of range [0, {len(grid)})"
            )
        return [grid[config.task_id]]

    start = config.design_start if config.design_start is not None else 0
    end = config.design_end if config.design_end is not None else len(grid)
    if not 0 <= start <= end <= len(grid):
        raise IndexError(
            f"design range [{start}, {end}) out of bounds [0, {len(grid)}]"
        )
    return grid[start:end]


def run_experiments(config: ExperimentConfig) -> list[dict[str, Any]]:
    """Run the selected subset of the design grid for `config`."""
    grid = build_design_grid(config)
    logger = logging.getLogger(__name__)
    logger.info(
        "study=%s adaptive=%s problem=%s dim=%d budget=%d -> %d design point(s) total",
        config.study, config.spec.adaptive,
        config.problem_name, config.problem_dim, config.budget, len(grid),
    )

    results: list[dict[str, Any]] = []
    for point in _selected_points(config, grid):
        results.append(run_design_point(config, point))
    return results


# ============================================================================
# SENSITIVITY AGGREGATION
# ============================================================================


def _basis_to_str(value: PolyBasisType | str) -> str:
    if isinstance(value, PolyBasisType):
        return POLY_BASIS_NAMES.get(value, value.value)
    return str(value)


def _final_objective_stats(
        experiment: ProblemSolver | ProblemsSolvers
        ) -> tuple[float | None, float | None, int]:
    """Mean / std / count of the final-budget objective across macroreplications."""
    macro_finals: list[float] = []
    for curve in getattr(experiment, "objective_curves", None) or []:
        y_vals = getattr(curve, "y_vals", None)
        if y_vals is not None and len(y_vals) > 0:
            macro_finals.append(float(y_vals[-1]))
    if not macro_finals:
        for mrep_vals in getattr(experiment, "all_est_objectives", None) or []:
            if mrep_vals is not None and len(mrep_vals) > 0:
                macro_finals.append(float(mrep_vals[-1]))
    if not macro_finals:
        return None, None, 0
    arr = np.asarray(macro_finals, dtype=float)
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return float(arr.mean()), std, arr.size


def _extract_design_point_record(pickle_path: Path) -> dict[str, Any] | None:
    import pickle  # local: only needed when aggregating

    try:
        with pickle_path.open("rb") as f:
            experiment = pickle.load(f)
    except Exception as e:
        logging.getLogger(__name__).warning(
            "Skipping unreadable pickle %s: %s", pickle_path.name, e
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
        "adaptive": bool(factors.get("adaptive subspace dimension", False)),
        "n_macroreps": n_macro,
        "mean_obj": mean_obj,
        "std_obj": float("nan") if std_obj is None else std_obj,
    }


def _group_mean_std(
    rows: list[dict[str, Any]], keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    buckets: dict[tuple, list[float]] = {}
    for row in rows:
        bucket_key = tuple(row[k] for k in keys)
        buckets.setdefault(bucket_key, []).append(row["mean_obj"])

    out: list[dict[str, Any]] = []
    for bucket_key, values in buckets.items():
        arr = np.asarray(values, dtype=float)
        record: dict[str, Any] = dict(zip(keys, bucket_key, strict=False))
        record["n_design_points"] = int(arr.size)
        record["mean_obj"] = float(arr.mean())
        record["std_obj"] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        record["min_obj"] = float(arr.min())
        record["max_obj"] = float(arr.max())
        out.append(record)
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
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
    """Aggregate POSTREPS pickles into per-design / main-effect / interaction CSVs."""
    logger = logging.getLogger(__name__)

    candidates = sorted(output_dir.glob("*_POSTREPS.pickle"))
    if not candidates:
        candidates = sorted(
            p
            for p in output_dir.glob("*.pickle")
            if "_POSTREPS" not in p.name and "hpc_config" not in p.name
        )
    if not candidates:
        logger.info(
            "No design-point pickles found in %s; skipping aggregation.", output_dir
        )
        return None

    logger.info(
        "Aggregating sensitivity summaries over %d design points", 
        len(candidates)
    )
    rows = [r for r in (_extract_design_point_record(p) for p in candidates) if r]
    if not rows:
        logger.warning(
            "No usable design-point records extracted; summaries not written."
        )
        return None

    _write_csv(output_dir / "design_points_summary.csv", rows)

    main_effects: list[dict[str, Any]] = []
    for factor_key in ("subspace_dim", "basis", "regularisation"):
        for grp in _group_mean_std(rows, (factor_key,)):
            main_effects.append(
                {
                    "factor": factor_key,
                    "level": grp[factor_key],
                    "n_design_points": grp["n_design_points"],
                    "mean_obj": grp["mean_obj"],
                    "std_obj": grp["std_obj"],
                    "min_obj": grp["min_obj"],
                    "max_obj": grp["max_obj"],
                }
            )
    _write_csv(output_dir / "summary_main_effects.csv", main_effects)

    pairwise: list[dict[str, Any]] = []
    for keys in (
        ("subspace_dim", "basis"),
        ("subspace_dim", "regularisation"),
        ("basis", "regularisation"),
    ):
        for grp in _group_mean_std(rows, keys):
            pairwise.append(
                {
                    "factor_a": keys[0],
                    "level_a": grp[keys[0]],
                    "factor_b": keys[1],
                    "level_b": grp[keys[1]],
                    "n_design_points": grp["n_design_points"],
                    "mean_obj": grp["mean_obj"],
                    "std_obj": grp["std_obj"],
                    "min_obj": grp["min_obj"],
                    "max_obj": grp["max_obj"],
                }
            )
    _write_csv(output_dir / "summary_interactions.csv", pairwise)

    logger.info(
        "Wrote sensitivity CSVs (problem=%s, study=%s)",
        config.problem_name, config.study,
    )
    return output_dir


# ============================================================================
# HPC HELPERS: design-matrix CSV, manifest JSON, SLURM array script
# ============================================================================


def write_design_matrix_csv(config: ExperimentConfig, grid: list[DesignPoint]) -> Path:
    """Persist the full design grid as a CSV (one row per task_id)."""
    csv_path = (
        config.output_dir
        / f"design_matrix_{config.study}_{config.problem_name}.csv"
    )
    rows = [
        {
            "task_id": p.task_id,
            "design_point_id": p.design_point_id,
            "subspace_dim": p.subspace_dim,
            "polynomial_basis": p.polynomial_basis.value,
            "polynomial_degree": config.polynomial_degree,
            "subproblem_regularisation": p.regularisation,
            "adaptive": config.spec.adaptive,
            "problem_name": config.problem_name,
            "problem_dim": config.problem_dim,
            "budget": config.budget,
        }
        for p in grid
    ]
    _write_csv(csv_path, rows)
    return csv_path


def write_hpc_manifest(config: ExperimentConfig, grid: list[DesignPoint]) -> Path:
    """Persist a JSON manifest summarising the experiment for HPC bookkeeping."""
    manifest = {
        "study": config.study,
        "adaptive": config.spec.adaptive,
        "problem_name": config.problem_name,
        "problem_dim": config.problem_dim,
        "budget": config.budget,
        "n_macroreps": config.n_macroreps,
        "n_postreps": config.n_postreps,
        "n_design_points": len(grid),
        "polynomial_degree": config.polynomial_degree,
        "fixed_subspace_dim": config.fixed_subspace_dim,
        "fixed_basis": config.fixed_basis.value,
        "fixed_regularisation": config.fixed_regularisation,
        "subspace_dims": list(config.subspace_dims),
        "basis_types": [b.value for b in config.basis_types],
        "regularisation_values": list(config.regularisation_values),
        "adaptive_factors": config.adaptive_factors,
        "output_dir": str(config.output_dir),
    }
    path = config.output_dir / f"hpc_config_{config.study}_{config.problem_name}.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def write_slurm_array_script(
    config: ExperimentConfig,
    grid: list[DesignPoint],
    *,
    time_limit: str = "4:00:00",
    partition: str = "batch",
    cpus_per_task: int = 1,
    mem_per_cpu: str = "4G",
) -> Path:
    """Emit a SLURM array script targeting one task per design point."""
    n_tasks = len(grid)
    if n_tasks == 0:
        raise ValueError("No design points to schedule.")

    script_path = Path(__file__).resolve()
    out_dir = config.output_dir.resolve()
    slurm_file = out_dir / f"run_{config.study}_{config.problem_name}.slurm"

    contents = f"""#!/bin/bash
#SBATCH --job-name=astromorf_{config.study}_{config.problem_name}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem-per-cpu={mem_per_cpu}
#SBATCH --time={time_limit}
#SBATCH --array=0-{n_tasks - 1}
#SBATCH --output={out_dir}/logs/slurm_%A_%a.out
#SBATCH --error={out_dir}/logs/slurm_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=$USER@soton.ac.uk

mkdir -p {out_dir}/logs
source $HOME/miniconda3/bin/activate simopt

python {script_path} \\
    --study {config.study} \\
    --problem {config.problem_name} \\
    --dim {config.problem_dim} \\
    --budget {config.budget} \\
    --n-macroreps {config.n_macroreps} \\
    --n-postreps {config.n_postreps} \\
    --output-dir {out_dir} \\
    --task-id $SLURM_ARRAY_TASK_ID

echo "Task $SLURM_ARRAY_TASK_ID completed"
"""
    slurm_file.write_text(contents)
    os.chmod(slurm_file, 0o755)  # noqa: PTH101
    return slurm_file


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_basis_list(s: str) -> list[PolyBasisType]:
    out: list[PolyBasisType] = []
    for raw in s.split(","):
        token = raw.strip()
        if not token:
            continue
        key = token.upper().replace("-", "_").replace(" ", "_")
        try:
            out.append(PolyBasisType[key])
            continue
        except KeyError:
            pass
        for pbt in PolyBasisType:
            if pbt.value.upper() == key:
                out.append(pbt)
                break
        else:
            raise ValueError(f"Unknown polynomial basis type: {token!r}")
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parses the command-line arguments for the journal factors experiment driver.  
    The returned namespace contains all the configuration options needed 
    to run the experiments, generate HPC scripts, 
    or aggregate results, depending on the flags provided.

    Args:
        argv (list[str] | None, optional): The command-line arguments to parse. 
        Defaults to None.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Datafarming sensitivity studies for ASTROMoRF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--study", choices=STUDY_CHOICES, required=True,
        help="Which sensitivity study to run. 'subspace' is non-adaptive; "
             "'basis' and 'regularisation' enable adaptive ASTROMoRF.",
    )
    parser.add_argument("--problem", required=True)
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--budget", type=int, default=5000)

    parser.add_argument("--n-macroreps", type=int, default=10)
    parser.add_argument("--n-postreps", type=int, default=100)
    parser.add_argument("--n-postreps-init-opt", type=int, default=200)

    parser.add_argument("--subspace-dims", type=str, default=None,
                        help="Comma-separated subspace dimensions to sweep.")
    parser.add_argument("--basis-types", type=str, default=None,
                        help="Comma-separated polynomial basis types to sweep.")
    parser.add_argument("--regularisation-values", type=str, default=None,
                        help="Comma-separated regularisation values to sweep.")

    parser.add_argument("--fixed-subspace-dim", type=int,
                        default=DEFAULT_FIXED_SUBSPACE_DIM)
    parser.add_argument("--fixed-basis", type=str, default=None,
                        help="Held-fixed polynomial basis when not sweeping it.")
    parser.add_argument("--fixed-regularisation", type=float,
                        default=DEFAULT_FIXED_REGULARISATION)

    parser.add_argument("--task-id", type=int, default=None,
                        help="Run a single design point (SLURM array worker).")
    parser.add_argument("--design-start", type=int, default=None,
                        help="Inclusive start index of design-point range.")
    parser.add_argument("--design-end", type=int, default=None,
                        help="Exclusive end index of design-point range.")
    parser.add_argument("--output-dir", type=str, default="experiments")

    parser.add_argument("--generate-slurm", action="store_true")
    parser.add_argument("--generate-csv", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--no-aggregate", action="store_true",
                        help="Skip the post-run aggregation step (HPC workers).")

    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    subspace_dims: tuple[int, ...] = ()
    if args.subspace_dims:
        subspace_dims = tuple(_parse_int_list(args.subspace_dims))

    basis_types: tuple[PolyBasisType, ...] = DEFAULT_BASIS_TYPES
    if args.basis_types:
        basis_types = tuple(_parse_basis_list(args.basis_types))

    regularisation_values: tuple[float, ...] = DEFAULT_REGULARISATION_VALUES
    if args.regularisation_values:
        regularisation_values = tuple(_parse_float_list(args.regularisation_values))

    fixed_basis = DEFAULT_FIXED_BASIS
    if args.fixed_basis:
        parsed = _parse_basis_list(args.fixed_basis)
        if not parsed:
            raise ValueError(f"--fixed-basis {args.fixed_basis!r} is empty")
        fixed_basis = parsed[0]

    return ExperimentConfig(
        study=args.study,
        problem_name=args.problem,
        problem_dim=args.dim,
        budget=args.budget,
        n_macroreps=args.n_macroreps,
        n_postreps=args.n_postreps,
        n_postreps_init_opt=args.n_postreps_init_opt,
        subspace_dims=subspace_dims,
        basis_types=basis_types,
        regularisation_values=regularisation_values,
        fixed_subspace_dim=args.fixed_subspace_dim,
        fixed_basis=fixed_basis,
        fixed_regularisation=args.fixed_regularisation,
        task_id=args.task_id,
        design_start=args.design_start,
        design_end=args.design_end,
        output_dir=Path(args.output_dir),
    )


def main(argv: list[str] | None = None) -> None:
    """Main entry point for the journal factors experiment driver.

    Args:
        argv (list[str] | None, optional): The command-line arguments to parse.
        Defaults to None.
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger(__name__)

    config = _config_from_args(args)
    grid = build_design_grid(config)

    if args.generate_csv:
        path = write_design_matrix_csv(config, grid)
        logger.info("Wrote design matrix: %s", path)
        return

    if args.generate_slurm:
        csv_path = write_design_matrix_csv(config, grid)
        manifest_path = write_hpc_manifest(config, grid)
        slurm_path = write_slurm_array_script(config, grid)
        logger.info("Wrote design matrix: %s", csv_path)
        logger.info("Wrote manifest: %s", manifest_path)
        logger.info("Wrote SLURM script: %s", slurm_path)
        logger.info("Submit with: sbatch %s", slurm_path)
        return

    if args.aggregate_only:
        aggregate_sensitivity_results(config.output_dir, config)
        return

    results = run_experiments(config)
    logger.info(
        "Completed %d design point(s) (study=%s, adaptive=%s)",
        len(results), config.study, config.spec.adaptive,
    )
    for r in results:
        logger.info("  task=%s id=%s -> %s", r["task_id"], r["design_point_id"],
                    r["postreps_file"])

    is_partial = (
        config.task_id is not None
        or config.design_start is not None
        or config.design_end is not None
    )
    if not args.no_aggregate and not is_partial:
        aggregate_sensitivity_results(config.output_dir, config)


if __name__ == "__main__":
    main()
