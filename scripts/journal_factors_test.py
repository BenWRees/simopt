"""Journal-grade sensitivity studies for the ASTROMoRF solver.

This module is the *execution engine* used by the manifest-driven journal
workflow.  It is intentionally split into three independent layers so that the
SLURM script and the Python driver are fully decoupled:

    1.  Design / manifest generation
        ----------------------------
        Pure functions that, given a `StudyConfig`, produce the deterministic
        list of design points for each (study, problem).  These design points
        are serialised into a single `manifest.json` (one row per global task
        id, spanning every study and every problem).  This manifest is the
        single source of truth.  Nothing downstream is allowed to re-derive
        task counts, factor levels, or design-point ids from anywhere else.

    2.  Execution engine
        -----------------
        Given a task row from the manifest, instantiate the solver and the
        problem exactly as specified, run the experiment, post-replicate, and
        emit per-(macrorep, budget) records as ``long_form.parquet`` (with a
        graceful fallback to ``long_form.csv.gz`` when no parquet engine is
        available).  This layer is fully restart-safe: a ``.done`` sentinel
        causes successful tasks to be skipped on requeue; partial work checks
        in via ``progress.json`` and is checkpointed on SIGUSR1 / SIGTERM via
        ``ProblemSolver.run_resumable``.

    3.  Lightweight aggregation helpers
        --------------------------------
        Walks the per-task ``long_form`` outputs and concatenates them into a
        single tidy file for analysis.  Statistical inference (paired
        confidence intervals, effect sizes, multiple-comparison corrections)
        is performed in :mod:`journal_aggregate`, not here — keeping the
        execution engine free of analysis-time choices.

Per-task output layout (each <design_point_id>/ directory contains exactly):

    factors.json                 the manifest row used for this task
    provenance.json              git SHA, host, argv, SLURM env vars
    progress.json                last-known stage of the run
    result.resume.pickle         run_resumable chunk checkpoint
    postreps.pickle              ProblemSolver after post_replicate()
    long_form.parquet            per-(macrorep, budget) tidy table
                                 (long_form.csv.gz if no parquet engine)
    .done                        success sentinel

Concretely, the workflow on an HPC cluster is::

    # ── Once, before submission ─────────────────────────────────────────────
    python scripts/journal_generate_manifest.py \
        --output-root $HOME/results/journal \
        --problems SAN-1 ROSENBROCK-1 DYNAMNEWS-1 NETWORK-1 \
        --budget 10000 --n-macroreps 30 --n-postreps 600 \
        --dims SAN-1=20,ROSENBROCK-1=10,DYNAMNEWS-1=10,NETWORK-1=14

    # writes:
    #     $HOME/results/journal/manifest.json
    #     $HOME/results/journal/manifest.csv

    # ── Submit one SLURM array, sized from the manifest ─────────────────────
    TOTAL=$(python -c "import json,sys; \
        print(json.load(open('$HOME/results/journal/manifest.json'))['total_tasks'])")
    sbatch --array=0-$((TOTAL-1))%64 scripts/run_journal_factors.slurm

    # ── After the array, aggregate ──────────────────────────────────────────
    python scripts/journal_aggregate.py \
        --runs-root $HOME/results/journal/runs \
        --output-dir $HOME/results/journal/analysis

The execution engine is *not* responsible for choosing factor levels — that is
the manifest's job.  This guarantees that every reviewer reproducing the work
sees the same numbers.

Random number handling
----------------------
The simopt experiment framework constructs MRG32k3a substreams keyed on
``mrep`` (the macroreplication index) and on the solver's RNG slot.  Because
all design points within this study use the same ``n_macroreps`` and the same
problem model, simulation streams are **bit-identical across factor levels for
the same mrep**.  This means common random numbers (CRN) across factor levels
is automatic: paired comparisons over the macrorep axis are valid by
construction.  The aggregator exploits this directly.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

# Make the project root importable when this file is invoked as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simopt.experiment import ProblemSolver, scale_dimension  # noqa: E402
from simopt.experiment.dimension_scaling import is_scalable  # noqa: E402
from simopt.experiment_base import Problem, instantiate_solver  # noqa: E402
from simopt.solvers.astromorf import CABS_DEFAULTS, PolyBasisType  # noqa: E402

LOGGER = logging.getLogger("journal_factors")

MANIFEST_VERSION = 2
SOLVER_ABBR = "ASTROMORF"
SOLVER_DISPLAY_NAME = "ASTROMoRF"


# ============================================================================
# Per-problem operating point registry
# ----------------------------------------------------------------------------
# These are the values that hold a non-swept factor at a sensible point.  They
# are the *operating point* of the solver on each problem (typically the result
# of a separate hyperparameter tuning).  Keys MUST match the names below
# verbatim: a sanity check at the top of `resolve_operating_point` will refuse
# to silently fall back to module-level defaults if a registered problem is
# missing a key.
# ============================================================================
PROBLEM_OPERATING_POINT: dict[str, dict[str, Any]] = {
    "DYNAMNEWS-1": {
        "subspace_dimension": 5,
        "polynomial_degree": 4,
        "subproblem_regularisation": 9.3926741094426e-06,
        "ps_sufficient_reduction": 0.2558336795133228,
        "polynomial_basis": PolyBasisType.NATURAL,
        "lambda_min": 6,
    },
    "SAN-1": {
        "subspace_dimension": 5,
        "polynomial_degree": 3,
        "subproblem_regularisation": 0.16825491384130684,
        "ps_sufficient_reduction": 0.0014138895951216063,
        "polynomial_basis": PolyBasisType.MONOMIAL,
        "lambda_min": 3,
    },
    "ROSENBROCK-1": {
        "subspace_dimension": 4,
        "polynomial_degree": 1,
        "subproblem_regularisation": 0.0060325065242926345,
        "ps_sufficient_reduction": 0.7442442345346906,
        "polynomial_basis": PolyBasisType.MONOMIAL,
        "lambda_min": 3,
    },
    "NETWORK-1": {
        "subspace_dimension": 15,
        "polynomial_degree": 3,
        "subproblem_regularisation": 0.03084158653272337,
        "ps_sufficient_reduction": 0.3290867932917176,
        "polynomial_basis": PolyBasisType.HERMITE,
        "lambda_min": 5,
    },
    "PARAMESTI-1": {
        "subspace_dimension": 2,
        "polynomial_degree": 2,
        "subproblem_regularisation": 1.8557102009866698e-05,
        "ps_sufficient_reduction": 0.3489932689036348,
        "polynomial_basis": PolyBasisType.CHEBYSHEV,
        "lambda_min": 9,
    },
}

OPERATING_POINT_KEYS: tuple[str, ...] = (
    "subspace_dimension",
    "polynomial_degree",
    "subproblem_regularisation",
    "ps_sufficient_reduction",
    "polynomial_basis",
    "lambda_min",
)

ADAPTIVE_CONFIGS: dict[str, dict[str, Any]] = {
    "SAN-1": {
        "gamma": 0.8979138034117701, "c_p": 0.03301456917974605,
        "c_g": 0.16995201607841975, "eps_n": 0.03179063701587841,
        "eps_a": 0.4068921820898637, "rho_max": 0.8800444663591878,
        "w_safe": 18, "eta_safe": 0.014634217178650853,
        "c2_est": 1.7323493147599263, "delta_inc_cap": 9,
    },
    "NETWORK-1": {
        "gamma": 0.8418390078321105, "c_p": 0.07087355092318823,
        "c_g": 0.48689466577922774, "eps_n": 0.22630487327885865,
        "eps_a": 0.24519398298639256, "rho_max": 0.819061782383329,
        "w_safe": 18, "eta_safe": 0.026726934654979863,
        "c2_est": 0.5215160989415044, "delta_inc_cap": 9,
    },
    "DYNAMNEWS-1": {
        "gamma": 0.8954811132667856, "c_p": 0.13839900124832283,
        "c_g": 0.7489076031913247, "eps_n": 0.07123425583685324,
        "eps_a": 0.061097120714973176, "rho_max": 0.6718703848854164,
        "w_safe": 15, "eta_safe": 0.09026154430417822,
        "c2_est": 0.5256829148159191, "delta_inc_cap": 6,
    },
    "ROSENBROCK-1": {
        "gamma": 0.8025795189265601, "c_p": 0.6499103358279078,
        "c_g": 0.17461702543318225, "eps_n": 0.34770197082788773,
        "eps_a": 0.27399808406597553, "rho_max": 0.790042323293872,
        "w_safe": 28, "eta_safe": 0.036547673050839516,
        "c2_est": 1.639360264931268, "delta_inc_cap": 6,
    },
    "PARAMESTI-1": {
        "gamma": 0.9118442610588682, "c_p": 0.29458464194799866,
        "c_g": 0.17155413448959728, "eps_n": 0.2783845948216283,
        "eps_a": 0.07194107335227416, "rho_max": 0.9094183864616052,
        "w_safe": 18, "eta_safe": 0.11550570058098267,
        "c2_est": 0.5658615715072871, "delta_inc_cap": 3,
    },
}


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

POLY_BASIS_FROM_NAME: dict[str, PolyBasisType] = {
    v: k for k, v in POLY_BASIS_NAMES.items()
}
# Also accept the enum's value string ("hermite", ...) so manifests are
# robust to either spelling.
POLY_BASIS_FROM_NAME.update({p.value: p for p in PolyBasisType})


STUDY_NAMES: tuple[str, ...] = ("subspace", "basis", "regularisation")


# ============================================================================
# Operating-point resolution (no silent fallbacks)
# ============================================================================
def resolve_operating_point(problem_name: str, problem_dim: int) -> dict[str, Any]:
    """Return the operating-point dict for ``problem_name``.

    A problem must be registered in :data:`PROBLEM_OPERATING_POINT` and must
    define every key in :data:`OPERATING_POINT_KEYS`.  Any missing key raises
    :class:`KeyError` immediately — this guards against the silent registry-
    fallback bug from the original implementation, which produced misleading
    "operating-point" results that were actually module-level defaults.
    """
    if problem_name not in PROBLEM_OPERATING_POINT:
        raise KeyError(
            f"No operating point registered for problem {problem_name!r}. "
            f"Add an entry to PROBLEM_OPERATING_POINT (all keys: "
            f"{', '.join(OPERATING_POINT_KEYS)})."
        )
    entry = PROBLEM_OPERATING_POINT[problem_name]
    missing = [k for k in OPERATING_POINT_KEYS if k not in entry]
    if missing:
        raise KeyError(
            f"Operating point for {problem_name!r} is missing keys: {missing}."
        )
    if problem_name not in ADAPTIVE_CONFIGS:
        raise KeyError(
            f"No adaptive (CABS) config registered for {problem_name!r}; "
            f"the basis and regularisation studies require it."
        )
    op = dict(entry)  # copy so callers can mutate safely
    # Clamp subspace dim to the requested problem dimension (e.g. when running
    # a registered problem at a smaller scaled dimension than its operating
    # point assumes).
    op["subspace_dimension"] = max(1, min(int(op["subspace_dimension"]), problem_dim))
    return op


def cabs_factors_for(problem_name: str) -> dict[str, Any]:
    """Return a complete CABS factor dict for ``problem_name``.

    Always seeded from :data:`CABS_DEFAULTS` and overridden by
    :data:`ADAPTIVE_CONFIGS` so missing keys raise loudly in the solver, not
    quietly here.
    """
    merged = dict(CABS_DEFAULTS)
    merged.update(ADAPTIVE_CONFIGS[problem_name])
    return merged


# ============================================================================
# Factor-grid generation (the corrected experimental designs)
# ============================================================================
def subspace_levels(problem_dim: int) -> list[int]:
    """Levels for the subspace-dimension sweep.

    Three-band schedule, designed so the number of levels grows
    logarithmically with ``problem_dim`` (no level explosion at high d) while
    keeping the *interesting* regions densely resolved:

    1.  **Dense low band** — every integer from 1..min(8, d).  This is the
        regime where the polynomial surrogate is most likely under-determined
        (number of basis functions C(s+p, p) approaches the per-iteration
        sample budget).  Most "weird" behaviour lives here.

    2.  **Geometric mid band** — log-spaced points from 9 up to ``problem_dim``.
        The number of mid-band points scales as
        ``ceil(8 * log10(d / 8) + 2)`` and is capped at 12, so:

            d = 10  → 3 mid points   → 10 levels total
            d = 20  → 6 mid points   → 14 levels total
            d = 50  → 9 mid points   → ~17 levels total
            d = 100 → 11 mid points  → ~19 levels total
            d = 200 → 12 mid points  → ~20 levels total

        Geometric (rather than linear) spacing matches the way the surrogate
        cost scales with subspace dimension: doubling ``s`` is the natural
        unit of "more capacity", not adding a constant number of dimensions.

    3.  **Endpoints** — both ``1`` and ``problem_dim`` are guaranteed members.
        ``problem_dim`` is included so the "full-space" behaviour (no
        projection) appears in the grid; this is the limiting case any
        reviewer will ask about.

    The grid is deduplicated and sorted; identical points produced by
    different bands collapse via ``set(...)``.  Total level count stays
    O(log d), keeping the SLURM array tractable even at d=200+.
    """
    if problem_dim < 1:
        raise ValueError(f"problem_dim must be >= 1, got {problem_dim}")
    dense = list(range(1, min(8, problem_dim) + 1))
    if problem_dim <= 8:
        levels = dense
    else:
        # Adaptive log-resolution: ~3 mid points at d=10, ~11 at d=100,
        # saturating at 12 by d≈200.
        n_log = max(2, min(12, int(np.ceil(8 * np.log10(problem_dim / 8) + 2))))
        log_pts = np.unique(
            np.round(np.geomspace(9, problem_dim, num=n_log)).astype(int)
        )
        levels = sorted(set(dense) | {int(x) for x in log_pts})
    # Final sanity: 1 <= every level <= problem_dim, monotone increasing.
    levels = [int(x) for x in levels if 1 <= int(x) <= problem_dim]
    assert levels == sorted(set(levels))
    assert levels[0] == 1 and levels[-1] == problem_dim
    return levels


def regularisation_levels() -> list[float]:
    """Levels for the subproblem-regularisation sweep on [0, 1].

    Strategy (piecewise):
        - Explicit zero (no regularisation).
        - Dense log-spaced near zero — half-decade resolution from 1e-6 to
          1e-1 to capture the well-conditioned-to-ill-conditioned transition.
        - Three additional points (0.3, 0.5, 1.0) to span the heavily
          regularised regime.

    The full grid spans seven decades of dynamic range, plus an exact 0.0.
    """
    log_points = [10 ** k for k in np.arange(-6.0, -1.0 + 1e-9, 0.5)]
    upper = [0.1, 0.3, 0.5, 1.0]
    levels = sorted({0.0, *log_points, *upper})
    # Tight numeric sanity check.
    assert levels[0] == 0.0
    assert all(0.0 <= x <= 1.0 for x in levels)
    return [float(x) for x in levels]


def basis_levels() -> list[PolyBasisType]:
    """All polynomial basis types defined in the solver, in registry order.

    A reviewer can re-derive this list from ``PolyBasisType`` directly.
    """
    return list(PolyBasisType)


def basis_degrees() -> list[int]:
    """Polynomial degrees swept jointly with basis (factorial design).

    Rationale: the original OFAT design held degree fixed at the registry
    value, which was tuned for *one* basis.  This silently penalised every
    other basis (e.g. forcing Lagrange into degree 1 if the registry basis
    was Monomial degree 1).  Sweeping degree alongside basis as a small
    factorial removes that bias.
    """
    return [1, 2, 3, 4]


# ============================================================================
# Design-point construction (one per (study, problem) cell)
# ============================================================================
@dataclass(frozen=True)
class DesignPoint:
    """A single design point.  Fully self-describing — no implicit context."""

    study: str
    problem: str
    problem_dim: int
    design_point_id: str
    swept_factor: str  # "subspace_dim" | "(basis, degree)" | "regularisation"
    # Factor values *as exposed to the solver* (these are the columns
    # downstream analysis groups on).
    subspace_dim: int
    polynomial_basis: str  # human-readable name, e.g. "Hermite"
    polynomial_degree: int
    subproblem_regularisation: float
    ps_sufficient_reduction: float
    adaptive: bool

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_float_tag(x: float) -> str:
    """Filesystem-safe representation of a float for design-point ids."""
    return f"{x:.6g}".replace(".", "p").replace("-", "m").replace("+", "")


def build_design_points(
    study: str, problem: str, problem_dim: int
) -> list[DesignPoint]:
    """Return the deterministic, ordered list of design points for one cell.

    The ordering inside a cell is stable across re-runs so that ``task_id``s
    are reproducible.  The mapping between rows of the manifest and SLURM
    array task ids is the manifest's job (see :func:`build_manifest`).
    """
    op = resolve_operating_point(problem, problem_dim)
    points: list[DesignPoint] = []

    if study == "subspace":
        # Subspace dim is the sweep; CABS is OFF (adaptive subspace would
        # mask the very effect we are studying).
        fixed_basis_name = POLY_BASIS_NAMES[op["polynomial_basis"]]
        for d in subspace_levels(problem_dim):
            dpid = f"subspace_d{d:03d}_on_{problem}"
            points.append(DesignPoint(
                study=study, problem=problem, problem_dim=problem_dim,
                design_point_id=dpid, swept_factor="subspace_dim",
                subspace_dim=d,
                polynomial_basis=fixed_basis_name,
                polynomial_degree=int(op["polynomial_degree"]),
                subproblem_regularisation=float(op["subproblem_regularisation"]),
                ps_sufficient_reduction=float(op["ps_sufficient_reduction"]),
                adaptive=False,
            ))

    elif study == "basis":
        # 2-factor factorial: basis × polynomial_degree.  CABS is ON because
        # the basis study is about steady-state adaptive performance.
        for basis in basis_levels():
            for deg in basis_degrees():
                bname = POLY_BASIS_NAMES[basis]
                dpid = f"basis_{bname}_deg{deg}_on_{problem}"
                points.append(DesignPoint(
                    study=study, problem=problem, problem_dim=problem_dim,
                    design_point_id=dpid, swept_factor="basis_x_degree",
                    subspace_dim=int(op["subspace_dimension"]),
                    polynomial_basis=bname,
                    polynomial_degree=int(deg),
                    subproblem_regularisation=float(op["subproblem_regularisation"]),
                    ps_sufficient_reduction=float(op["ps_sufficient_reduction"]),
                    adaptive=True,
                ))

    elif study == "regularisation":
        # Log-spaced regularisation sweep, CABS on.  The per-problem operating
        # point is *always* included so the paired-CI baseline is well-defined
        # (the analysis layer pairs every level against the operating-point
        # design point of the same problem).
        fixed_basis_name = POLY_BASIS_NAMES[op["polynomial_basis"]]
        levels = sorted(set(regularisation_levels())
                        | {float(op["subproblem_regularisation"])})
        for r in levels:
            dpid = f"reg_{_safe_float_tag(r)}_on_{problem}"
            points.append(DesignPoint(
                study=study, problem=problem, problem_dim=problem_dim,
                design_point_id=dpid, swept_factor="regularisation",
                subspace_dim=int(op["subspace_dimension"]),
                polynomial_basis=fixed_basis_name,
                polynomial_degree=int(op["polynomial_degree"]),
                subproblem_regularisation=float(r),
                ps_sufficient_reduction=float(op["ps_sufficient_reduction"]),
                adaptive=True,
            ))

    else:
        raise ValueError(f"Unknown study {study!r}; choose from {STUDY_NAMES}.")

    # Uniqueness of design-point ids inside a cell is a non-negotiable
    # invariant: collisions would clobber outputs.
    seen: set[str] = set()
    for p in points:
        if p.design_point_id in seen:
            raise RuntimeError(f"duplicate design_point_id {p.design_point_id!r}")
        seen.add(p.design_point_id)
    return points


# ============================================================================
# Manifest construction
# ============================================================================
@dataclass
class GlobalConfig:
    """Settings that are shared across every task in the manifest."""

    budget: int
    n_macroreps: int
    n_postreps: int
    crn_across_budget: bool = True
    crn_across_macroreps: bool = False
    crn_across_solns: bool = True
    macroreps_per_chunk: int = 5  # for run_resumable


def build_manifest(
    problems: Sequence[str],
    dims: dict[str, int],
    studies: Sequence[str],
    cfg: GlobalConfig,
    output_root: Path,
) -> dict[str, Any]:
    """Build the full manifest spanning every (study, problem) cell.

    The manifest assigns a single contiguous ``task_id`` space across all
    cells.  SLURM's ``--array=0..N-1`` indexes into this space directly — no
    arithmetic is performed in the shell.
    """
    output_root = Path(output_root)
    if not problems:
        raise ValueError("at least one problem must be specified")
    for p in problems:
        if p not in dims:
            raise KeyError(f"no dimension provided for problem {p!r}")
        if p not in PROBLEM_OPERATING_POINT:
            raise KeyError(
                f"problem {p!r} has no operating point registered; "
                f"refusing to build manifest"
            )
    for s in studies:
        if s not in STUDY_NAMES:
            raise ValueError(f"unknown study {s!r}; choose from {STUDY_NAMES}")

    tasks: list[dict[str, Any]] = []
    cell_summary: dict[str, dict[str, int]] = {}
    for study in studies:
        cell_summary[study] = {}
        for problem in problems:
            cell_points = build_design_points(study, problem, dims[problem])
            cell_summary[study][problem] = len(cell_points)
            for dp in cell_points:
                task_id = len(tasks)
                tasks.append({
                    "task_id": task_id,
                    "study": dp.study,
                    "problem": dp.problem,
                    "problem_dim": dp.problem_dim,
                    "design_point_id": dp.design_point_id,
                    "swept_factor": dp.swept_factor,
                    "factors": {
                        "subspace_dim": dp.subspace_dim,
                        "polynomial_basis": dp.polynomial_basis,
                        "polynomial_degree": dp.polynomial_degree,
                        "subproblem_regularisation": dp.subproblem_regularisation,
                        "ps_sufficient_reduction": dp.ps_sufficient_reduction,
                        "adaptive": dp.adaptive,
                    },
                    "operating_point": {
                        k: (POLY_BASIS_NAMES[v] if isinstance(v, PolyBasisType) else v)
                        for k, v in PROBLEM_OPERATING_POINT[dp.problem].items()
                    },
                    "cabs_factors":
                        cabs_factors_for(dp.problem) if dp.adaptive else None,
                    "output_subdir":
                        f"runs/{dp.problem}/{dp.study}/{dp.design_point_id}",
                })

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "solver": SOLVER_DISPLAY_NAME,
        "global": asdict(cfg),
        "problems": list(problems),
        "studies": list(studies),
        "problem_dims": {p: int(dims[p]) for p in problems},
        "design_levels": {
            "subspace_dim": {p: subspace_levels(dims[p]) for p in problems},
            "regularisation": regularisation_levels(),
            "basis": [POLY_BASIS_NAMES[b] for b in basis_levels()],
            "basis_degree": basis_degrees(),
        },
        "cell_sizes": cell_summary,
        "output_root": str(output_root.resolve()),
        "tasks": tasks,
        "total_tasks": len(tasks),
    }
    # Hash of the manifest's *substantive* content (everything that affects
    # what the workers actually do).  We deliberately exclude `created_at_utc`
    # and `manifest_sha256` so re-generating a manifest with identical inputs
    # produces a bit-identical SHA — this is what reviewers will use to verify
    # that two reported "runs of the same study" really did use the same
    # design grid.
    _hash_excluded = {"manifest_sha256", "created_at_utc"}
    body_bytes = json.dumps(
        {k: v for k, v in manifest.items() if k not in _hash_excluded},
        sort_keys=True, default=str,
    ).encode()
    manifest["manifest_sha256"] = hashlib.sha256(body_bytes).hexdigest()
    return manifest


def write_manifest(manifest: dict[str, Any], output_root: Path) -> tuple[Path, Path]:
    """Write the manifest (JSON) and a flat task CSV (for human inspection)."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "manifest.json"
    csv_path = output_root / "manifest.csv"
    json_path.write_text(json.dumps(manifest, indent=2, default=str))
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "task_id", "study", "problem", "problem_dim", "design_point_id",
            "swept_factor", "subspace_dim", "polynomial_basis",
            "polynomial_degree", "subproblem_regularisation", "adaptive",
            "output_subdir",
        ])
        for t in manifest["tasks"]:
            f = t["factors"]
            writer.writerow([
                t["task_id"], t["study"], t["problem"], t["problem_dim"],
                t["design_point_id"], t["swept_factor"],
                f["subspace_dim"], f["polynomial_basis"], f["polynomial_degree"],
                f["subproblem_regularisation"], f["adaptive"], t["output_subdir"],
            ])
    return json_path, csv_path


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    data = json.loads(Path(manifest_path).read_text())
    if data.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError(
            f"manifest version mismatch: file has "
            f"{data.get('manifest_version')!r}, code expects {MANIFEST_VERSION!r}"
        )
    return data


def lookup_task(manifest: dict[str, Any], task_id: int) -> dict[str, Any]:
    if task_id < 0 or task_id >= manifest["total_tasks"]:
        raise IndexError(
            f"task_id {task_id} out of range [0, {manifest['total_tasks']})"
        )
    row = manifest["tasks"][task_id]
    if row["task_id"] != task_id:
        raise RuntimeError(
            f"manifest is corrupt: row index {task_id} has task_id "
            f"{row['task_id']!r}"
        )
    return row


# ============================================================================
# Execution engine
# ============================================================================
def _setup_logging(verbose: bool, task_id: int | None) -> None:
    fmt = "%(asctime)s | %(levelname)s | %(name)s"
    if task_id is not None:
        fmt += f" | task={task_id}"
    fmt += " | %(message)s"
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=fmt,
        stream=sys.stdout,
    )
    # Send WARNINGs and ERRORs additionally to stderr so SLURM's .err file
    # contains only the bad news.
    err_handler = logging.StreamHandler(sys.stderr)
    err_handler.setLevel(logging.WARNING)
    err_handler.setFormatter(logging.Formatter(fmt))
    logging.getLogger().addHandler(err_handler)


def _parquet_engine() -> str | None:
    """Return the name of an installed parquet engine, or None."""
    for engine in ("pyarrow", "fastparquet"):
        try:
            __import__(engine)
            return engine
        except ImportError:
            continue
    return None


def _write_long_form(rows: list[dict[str, Any]], out_dir: Path) -> Path:
    """Write per-(macrorep, budget) records.

    Prefers Parquet (compact, typed, fast); falls back to ``long_form.csv.gz``
    with a loud warning if no Parquet engine is installed.  Both formats are
    schema-compatible.
    """
    df = pd.DataFrame.from_records(rows)
    engine = _parquet_engine()
    if engine is not None:
        path = out_dir / "long_form.parquet"
        df.to_parquet(path, engine=engine, index=False)
        return path
    path = out_dir / "long_form.csv.gz"
    LOGGER.warning(
        "No Parquet engine (pyarrow / fastparquet) installed; "
        "falling back to %s — install pyarrow for journal-grade outputs.",
        path.name,
    )
    with gzip.open(path, "wt", newline="") as fh:
        df.to_csv(fh, index=False)
    return path


def _capture_provenance(out_dir: Path, task: dict[str, Any]) -> Path:
    """Write a ``provenance.json`` capturing exact reproduction information."""
    import platform
    prov = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": str(Path.cwd()),
        "argv": sys.argv,
        "env": {
            "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
            "SLURM_ARRAY_JOB_ID": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "SLURM_ARRAY_TASK_ID": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "SLURM_CPUS_PER_TASK": os.environ.get("SLURM_CPUS_PER_TASK"),
            "SLURM_NODELIST": os.environ.get("SLURM_NODELIST"),
            "SLURM_TMPDIR": os.environ.get("SLURM_TMPDIR"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        },
        "git_sha": _git_sha_or_none(),
        "task": task,
    }
    path = out_dir / "provenance.json"
    path.write_text(json.dumps(prov, indent=2, default=str))
    return path


def _git_sha_or_none() -> str | None:
    import subprocess
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        return out.decode().strip()
    except Exception:
        return None


def _build_solver_factors(task: dict[str, Any]) -> dict[str, Any]:
    """Translate a manifest task into the dict the ASTROMoRF solver consumes.

    The factor names below are the *exact* keys ASTROMoRF expects (they are
    intentionally noisy with spaces and underscores — do not "normalise" them
    without first changing the solver).
    """
    f = task["factors"]
    basis_enum = POLY_BASIS_FROM_NAME[f["polynomial_basis"]]
    factors: dict[str, Any] = {
        "initial subspace dimension": int(f["subspace_dim"]),
        "polynomial basis": basis_enum,
        "polynomial degree": int(f["polynomial_degree"]),
        "subproblem_regularisation": float(f["subproblem_regularisation"]),
        "ps_sufficient_reduction": float(f["ps_sufficient_reduction"]),
        "crn_across_solns": bool(_global_crn_across_solns(task)),
        "adaptive subspace dimension": bool(f["adaptive"]),
    }
    if f["adaptive"]:
        if task.get("cabs_factors") is None:
            raise RuntimeError(
                f"task {task['task_id']} is adaptive but has no cabs_factors"
            )
        factors["CABS factors"] = dict(task["cabs_factors"])
    return factors


def _global_crn_across_solns(task: dict[str, Any]) -> bool:
    """Pulled out so a future per-task override path is trivial."""
    return bool(task.get("crn_across_solns", True))


def _scaled_problem(task: dict[str, Any], budget: int) -> Problem:
    name = task["problem"]
    dim = int(task["problem_dim"])
    if not is_scalable(name):
        LOGGER.warning(
            "Problem %s is not registered as scalable; using native dimension.",
            name,
        )
        return scale_dimension(problem_name=name, budget=budget, dimension=None)
    return scale_dimension(problem_name=name, budget=budget, dimension=dim)


def _build_long_form_rows(
    experiment: ProblemSolver,
    task: dict[str, Any],
) -> list[dict[str, Any]]:
    """Tidy per-(macrorep, budget) records suitable for paired analysis."""
    rows: list[dict[str, Any]] = []
    base_cols = {
        "study": task["study"],
        "problem": task["problem"],
        "problem_dim": task["problem_dim"],
        "design_point_id": task["design_point_id"],
        "swept_factor": task["swept_factor"],
        "subspace_dim": task["factors"]["subspace_dim"],
        "polynomial_basis": task["factors"]["polynomial_basis"],
        "polynomial_degree": task["factors"]["polynomial_degree"],
        "subproblem_regularisation": task["factors"]["subproblem_regularisation"],
        "adaptive": task["factors"]["adaptive"],
    }
    budgets = experiment.all_intermediate_budgets or []
    est_obj = experiment.all_est_objectives or []
    n_macroreps = min(len(budgets), len(est_obj))
    for mrep in range(n_macroreps):
        budgets_mr = budgets[mrep] or []
        objs_mr = est_obj[mrep] if mrep < len(est_obj) else []
        n_b = min(len(budgets_mr), len(objs_mr))
        for b_idx in range(n_b):
            rows.append({
                **base_cols,
                "macrorep": int(mrep),
                "budget_idx": int(b_idx),
                "budget": float(budgets_mr[b_idx]),
                "obj_postrep_mean": float(objs_mr[b_idx]),
                "is_final_budget": bool(b_idx == n_b - 1),
            })
    return rows


def _install_signal_checkpoint(
    progress_path: Path, status_holder: dict[str, Any]
) -> None:
    """Install SIGUSR1 / SIGTERM handlers that mark the task for requeue.

    The actual ProblemSolver checkpoint is handled by ``run_resumable`` (which
    installs its own SIGTERM/SIGINT handlers).  Here we additionally record
    the signal in ``progress.json`` so the post-mortem is unambiguous.
    """
    def _handler(signum, _frame):  # noqa: ANN001
        status_holder["last_signal"] = int(signum)
        try:
            progress_path.write_text(json.dumps({
                **status_holder,
                "received_signal_at_utc":
                    datetime.now(timezone.utc).isoformat(),
            }, indent=2, default=str))
        except Exception:  # pragma: no cover — best-effort
            pass
    # SIGUSR1: SLURM sends this before SIGTERM when --signal=B:USR1@... is set.
    # SIGTERM: SLURM sends this on cancellation / preemption.
    for sig in (signal.SIGUSR1, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):  # pragma: no cover — non-main thread
            pass


def execute_task(
    task: dict[str, Any],
    global_cfg: dict[str, Any],
    work_dir: Path,
    *,
    n_jobs: int = 1,
    force: bool = False,
) -> dict[str, Any]:
    """Run one design point end-to-end in ``work_dir``.

    Idempotent: if ``work_dir / .done`` exists and ``force`` is false, returns
    immediately.  All artefacts (factors.json, provenance.json,
    result.pickle, postreps.pickle, long_form.{parquet,csv.gz},
    progress.json, .done) are written into ``work_dir``.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    done_path = work_dir / ".done"
    progress_path = work_dir / "progress.json"

    if done_path.exists() and not force:
        LOGGER.info(
            "Task %s (%s) already complete; skipping.",
            task["task_id"], task["design_point_id"],
        )
        return {"status": "skipped", "task_id": task["task_id"]}

    # Persist factors and provenance up-front so a debug post-mortem has data
    # even if the run crashes midway.
    (work_dir / "factors.json").write_text(
        json.dumps(task, indent=2, default=str)
    )
    _capture_provenance(work_dir, task)

    status: dict[str, Any] = {
        "task_id": task["task_id"],
        "design_point_id": task["design_point_id"],
        "stage": "starting",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_macroreps": int(global_cfg["n_macroreps"]),
        "macroreps_per_chunk": int(global_cfg.get("macroreps_per_chunk", 5)),
    }
    progress_path.write_text(json.dumps(status, indent=2, default=str))
    _install_signal_checkpoint(progress_path, status)

    t_start = time.time()

    # ── Solver & problem ──
    factors = _build_solver_factors({
        **task,
        "crn_across_solns": global_cfg["crn_across_solns"],
    })
    LOGGER.info(
        "Running task %s [%s/%s] %s",
        task["task_id"], task["study"], task["problem"], task["design_point_id"],
    )
    LOGGER.debug("Solver factors: %s", factors)
    solver = instantiate_solver(
        solver_name=SOLVER_ABBR,
        fixed_factors=factors,
        solver_rename=SOLVER_DISPLAY_NAME,
    )
    problem = _scaled_problem(task, int(global_cfg["budget"]))
    if (is_scalable(task["problem"])
            and problem.dim != int(task["problem_dim"])):
        raise RuntimeError(
            f"scaled problem dim mismatch: got {problem.dim}, "
            f"expected {task['problem_dim']}"
        )

    result_pickle = work_dir / "result.pickle"
    postreps_pickle = work_dir / "postreps.pickle"

    # create_pickle=False suppresses the framework's auto-dump to
    # EXPERIMENT_DIR/<timestamp>/result.pickle on every post_replicate call.
    # We persist explicitly: result.resume.pickle (chunk checkpoint, written
    # by run_resumable) and postreps.pickle (post-replicated final state,
    # written by record_experiment_results below).  Both land in `work_dir`,
    # nowhere else.
    experiment = ProblemSolver(
        problem=problem, solver=solver,
        file_name_path=result_pickle,
        create_pickle=False,
    )

    # ── Resumable run ──
    status["stage"] = "run_resumable"
    progress_path.write_text(json.dumps(status, indent=2, default=str))
    chunk = int(global_cfg.get("macroreps_per_chunk", 5))
    experiment.run_resumable(
        n_macroreps_total=int(global_cfg["n_macroreps"]),
        n_jobs=int(n_jobs),
        macroreps_per_chunk=max(1, chunk),
        checkpoint_path=result_pickle.with_suffix(".resume.pickle"),
    )
    if getattr(experiment, "_termination_requested", False):
        LOGGER.warning(
            "Task %s received termination signal during run; exiting 99 "
            "for SLURM requeue.", task["task_id"],
        )
        status["stage"] = "preempted_during_run"
        progress_path.write_text(json.dumps(status, indent=2, default=str))
        return {"status": "preempted", "task_id": task["task_id"], "exit_code": 99}

    # ── Post-replication ──
    status["stage"] = "post_replicate"
    progress_path.write_text(json.dumps(status, indent=2, default=str))
    experiment.post_replicate(
        n_postreps=int(global_cfg["n_postreps"]),
        crn_across_budget=bool(global_cfg["crn_across_budget"]),
        crn_across_macroreps=bool(global_cfg["crn_across_macroreps"]),
    )
    experiment.record_experiment_results(file_name=str(postreps_pickle))

    # ── Long-form tidy output (the journal-grade artefact) ──
    status["stage"] = "write_long_form"
    progress_path.write_text(json.dumps(status, indent=2, default=str))
    rows = _build_long_form_rows(experiment, task)
    long_form_path = _write_long_form(rows, work_dir)

    # ── Done marker ──
    elapsed = time.time() - t_start
    status.update({
        "stage": "done",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": elapsed,
        "n_macroreps_completed": int(getattr(experiment, "n_macroreps", 0)),
        "long_form": long_form_path.name,
    })
    progress_path.write_text(json.dumps(status, indent=2, default=str))
    done_path.write_text(json.dumps({
        "task_id": task["task_id"],
        "design_point_id": task["design_point_id"],
        "completed_at_utc": status["completed_at_utc"],
        "wall_seconds": elapsed,
    }, indent=2))
    LOGGER.info(
        "Task %s done in %.1fs (%d macroreps).",
        task["task_id"], elapsed, status["n_macroreps_completed"],
    )
    return {"status": "completed", "task_id": task["task_id"], "exit_code": 0}


# ============================================================================
# Lightweight aggregation (used by the wrap-up script)
# ============================================================================
def collect_long_form(runs_root: Path) -> pd.DataFrame:
    """Concatenate every ``long_form.{parquet,csv.gz}`` under ``runs_root``."""
    runs_root = Path(runs_root)
    engine = _parquet_engine()
    frames: list[pd.DataFrame] = []
    if engine is not None:
        for p in sorted(runs_root.rglob("long_form.parquet")):
            frames.append(pd.read_parquet(p, engine=engine))
    for p in sorted(runs_root.rglob("long_form.csv.gz")):
        with gzip.open(p, "rt") as fh:
            frames.append(pd.read_csv(fh))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ============================================================================
# CLI
# ============================================================================
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Journal-grade sensitivity studies for ASTROMoRF — "
            "manifest-driven execution engine."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # run-task — executes a single design point.
    p_run = sub.add_parser(
        "run-task",
        help="Run the design point indexed by --task-id in the manifest.",
    )
    p_run.add_argument("--manifest", type=Path, required=True)
    p_run.add_argument("--task-id", type=int, required=True)
    p_run.add_argument(
        "--work-dir", type=Path, required=True,
        help="Output directory for this task's artefacts.",
    )
    p_run.add_argument(
        "--n-jobs", type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
        help="Macrorep parallelism (default: SLURM_CPUS_PER_TASK or 1).",
    )
    p_run.add_argument(
        "--force", action="store_true",
        help="Re-run even if .done exists.",
    )
    p_run.add_argument("--verbose", action="store_true")

    # describe-task — emit a one-line summary used by the SLURM dispatcher.
    p_desc = sub.add_parser(
        "describe-task",
        help="Print 'problem study design_point_id output_subdir' for a task.",
    )
    p_desc.add_argument("--manifest", type=Path, required=True)
    p_desc.add_argument("--task-id", type=int, required=True)

    # manifest-info — print the total task count (used by submission wrapper).
    p_info = sub.add_parser(
        "manifest-info",
        help="Print 'total_tasks <N>' and 'manifest_sha256 <H>'.",
    )
    p_info.add_argument("--manifest", type=Path, required=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.cmd == "describe-task":
        manifest = load_manifest(args.manifest)
        task = lookup_task(manifest, args.task_id)
        # Single line — stable, whitespace-separated, suitable for `read`.
        print(f"{task['problem']} {task['study']} "
              f"{task['design_point_id']} {task['output_subdir']}")
        return 0

    if args.cmd == "manifest-info":
        manifest = load_manifest(args.manifest)
        print(f"total_tasks {manifest['total_tasks']}")
        print(f"manifest_sha256 {manifest['manifest_sha256']}")
        return 0

    if args.cmd == "run-task":
        _setup_logging(args.verbose, args.task_id)
        manifest = load_manifest(args.manifest)
        task = lookup_task(manifest, args.task_id)
        result = execute_task(
            task=task,
            global_cfg=manifest["global"],
            work_dir=args.work_dir,
            n_jobs=args.n_jobs,
            force=args.force,
        )
        if result.get("status") == "preempted":
            return int(result.get("exit_code", 99))
        return 0

    parser.error(f"Unknown command {args.cmd!r}")
    return 2  # unreachable


if __name__ == "__main__":
    sys.exit(main())
