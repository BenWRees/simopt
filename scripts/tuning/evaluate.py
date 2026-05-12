"""Single-trial evaluator for ASTROMoRF tuning.

A "trial" here means: instantiate the problem at the configured dimension,
instantiate ASTROMoRF with the trial's hyperparameters, run *n_macroreps*
sequential macroreplications (so we can extract per-mrep diagnostics off
the solver instance), post-replicate the recommended trajectory, and
return both the noisy objective values and the aggregated diagnostics.

This module is intentionally *independent* of Optuna: it accepts a plain
hyperparameter dict and returns a plain dataclass. The Optuna binding
lives in ``tuner.py``.
"""

from __future__ import annotations

import logging
import os
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from simopt.experiment.dimension_scaling import scale_dimension
from simopt.experiment.run_solver import _set_up_rngs
from simopt.experiment_base import (
    ProblemSolver,
    instantiate_problem,
    instantiate_solver,
)
from simopt.solvers.astromorf import CABS_DEFAULTS, PolyBasisType

from .diagnostics import (
    MrepDiagnostics,
    aggregate_diagnostics,
    extract_macrorep_diagnostics,
    install_pattern_search_counter,
)
from .spaces import is_combinatorially_feasible

log = logging.getLogger(__name__)


# Default budget — kept as a module constant rather than a magic number so
# the smoke test, the worker, and the confirmation pass all agree.
DEFAULT_BUDGET: int = 10_000


# ── result containers ──────────────────────────────────────────────────────


@dataclass
class EvalResult:
    """Outcome of a single hyperparameter evaluation across n macroreps."""

    objectives: list[float]  # raw final-objective values (one per mrep)
    aligned_scores: list[float]  # sign-flipped (so "lower is better always")
    failure_count: int
    failure_traces: list[str] = field(default_factory=list)
    diagnostics_per_mrep: list[MrepDiagnostics] = field(default_factory=list)
    diagnostics_aggregate: dict[str, Any] = field(default_factory=dict)

    # Convenience scalars (all on aligned_scores, "lower is better").
    mean_aligned: float = float("nan")
    std_aligned: float = 0.0
    mean_objective: float = float("nan")
    std_objective: float = 0.0

    def is_failure(self) -> bool:
        return not self.objectives

    def to_record(self) -> dict[str, Any]:
        return {
            "objectives": list(self.objectives),
            "aligned_scores": list(self.aligned_scores),
            "failure_count": self.failure_count,
            "mean_aligned": self.mean_aligned,
            "std_aligned": self.std_aligned,
            "mean_objective": self.mean_objective,
            "std_objective": self.std_objective,
            "diagnostics": self.diagnostics_aggregate,
        }


# ── trial-config plumbing ──────────────────────────────────────────────────


def trial_params_to_solver_factors(params: dict[str, Any]) -> dict[str, Any]:
    """Translate the tuner's flat param dict into ASTROMoRF's alias-keyed dict.

    The tuner samples categorical fields as strings; we coerce to the
    PolyBasisType enum / int here. The CABS keys (``cabs_*``) get bundled
    into the nested ``"CABS factors"`` dict, after backfilling any keys
    not in the search space from ``CABS_DEFAULTS``.
    """
    cabs: dict[str, Any] = dict(CABS_DEFAULTS)
    cabs_overrides = {
        k.removeprefix("cabs_"): v for k, v in params.items() if k.startswith("cabs_")
    }
    cabs.update(cabs_overrides)

    basis_name = str(params["polynomial_basis"])
    basis = PolyBasisType[basis_name]

    return {
        "initial subspace dimension": int(params["subspace_dim"]),
        "polynomial degree": int(params["polynomial_degree"]),
        "polynomial basis": basis,
        "lambda_min": int(params["lambda_min"]),
        "subproblem_regularisation": float(params["subproblem_regularisation"]),
        "ps_sufficient_reduction": float(params["ps_sufficient_reduction"]),
        "adaptive subspace dimension": True,
        "CABS factors": cabs,
    }


def build_problem(
    problem_name: str, *, dimension: int | None, budget: int
) -> Any:
    """Build the problem at the requested dimension (or native if None)."""
    if dimension is not None:
        return scale_dimension(problem_name, budget=budget, dimension=dimension)
    return instantiate_problem(
        problem_name=problem_name, problem_fixed_factors={"budget": budget}
    )


def build_solver(factors: dict[str, Any]) -> Any:
    """Instantiate ASTROMoRF with alias-keyed factors."""
    return instantiate_solver("ASTROMORF", fixed_factors=factors)


# ── seeding ────────────────────────────────────────────────────────────────


def derive_mrep_index(*, base_seed: int, trial_number: int, mrep: int) -> int:
    """Derive a deterministic mrep index from (base_seed, trial, mrep).

    Deterministic seeding lets the confirmation pass and any re-run
    reproduce a trial bit-for-bit. We re-use the simopt RNG-substream
    convention (``mrep + 3``) by mapping our index there.
    """
    # Combine base_seed + trial_number into a deterministic offset, keep
    # mrep contribution distinct so a given (trial, mrep) is reproducible.
    salt = (base_seed * 2654435761 + trial_number * 1140671485) & 0xFFFFFFFF
    return (salt + mrep) & 0xFFFFFFFF


# ── core evaluator ─────────────────────────────────────────────────────────


def _extract_final_objective(
    experiment: ProblemSolver, mrep: int
) -> float | None:
    if (
        getattr(experiment, "all_est_objectives", None)
        and len(experiment.all_est_objectives[mrep]) > 0
    ):
        return float(experiment.all_est_objectives[mrep][-1])
    objs = getattr(experiment, "objective_curves", None)
    if objs and mrep < len(objs):
        curve = objs[mrep]
        if hasattr(curve, "y_vals") and len(curve.y_vals) > 0:
            return float(curve.y_vals[-1])
    return None


def _run_one_mrep(
    *,
    problem_name: str,
    dimension: int | None,
    budget: int,
    factors: dict[str, Any],
    mrep_index: int,
) -> tuple[float, float | None, MrepDiagnostics | None, float | None]:
    """Run one macroreplication and return picklable results only.

    Returns ``(wall_clock_s, final_obj_or_None, diagnostics_or_None,
    minmax_sign_or_None)``. Everything returned is plain-Python /
    dataclass so it can cross process boundaries via joblib.

    The solver and problem live entirely inside this function; we do
    NOT return them. That lets us run mreps in parallel subprocesses
    without paying for pickling the full simopt object graph back to
    the parent.
    """
    problem = build_problem(problem_name, dimension=dimension, budget=budget)
    # ``build_solver`` already returns a fresh instance — no need to deepcopy.
    solver = build_solver(factors)
    install_pattern_search_counter(solver)

    _set_up_rngs(solver, problem, mrep_index)

    t0 = time.perf_counter()
    # ``Solver.run`` constructs ``self.budget`` from the problem and then
    # calls ``solve``. ``solve`` alone leaves ``self.budget`` unset and
    # crashes.
    solver.run(problem)
    elapsed = time.perf_counter() - t0

    final_obj: float | None = None
    if (
        getattr(solver, "incumbent_solution", None) is not None
        and getattr(solver.incumbent_solution, "objectives_mean", None) is not None
    ):
        try:
            final_obj = float(solver.incumbent_solution.objectives_mean.item())
        except Exception:
            final_obj = None

    sign: float | None
    try:
        sign = float(problem.minmax[0])
    except Exception:
        sign = None

    diag: MrepDiagnostics | None
    try:
        diag = extract_macrorep_diagnostics(solver, wall_clock_s=elapsed)
    except Exception as exc:
        log.warning("Diagnostics extraction failed: %s", exc)
        diag = None

    return elapsed, final_obj, diag, sign


def _resolve_n_jobs(n_jobs: int | str | None) -> int:
    """Pick an effective parallelism level for macroreps.

    ``"auto"`` reads ``$SLURM_CPUS_PER_TASK`` so a Slurm task with N cores
    automatically runs N mreps in parallel. A plain int is honoured
    as-is. ``None`` or ``<=0`` falls back to sequential (n_jobs=1).
    """
    if n_jobs is None:
        return 1
    if isinstance(n_jobs, str):
        if n_jobs.lower() == "auto":
            try:
                return max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
            except ValueError:
                return 1
        try:
            n_jobs = int(n_jobs)
        except ValueError:
            return 1
    return max(1, int(n_jobs))


def evaluate_config(
    *,
    problem_name: str,
    dimension: int | None,
    budget: int,
    params: dict[str, Any],
    n_macroreps: int,
    base_seed: int,
    trial_number: int,
    per_trial_wall_clock_cap_s: float | None = None,
    skip_combinatorial_check: bool = False,
    n_jobs: int | str | None = 1,
) -> EvalResult:
    """Evaluate a single hyperparameter configuration.

    Parallelism
    -----------
    Macroreps run in parallel when *n_jobs* > 1 (use ``"auto"`` to pick
    up ``$SLURM_CPUS_PER_TASK``). Each worker subprocess builds its own
    problem/solver instances — no shared state.

    Wall-clock cap
    --------------
    *per_trial_wall_clock_cap_s* is a hard cap on the *whole trial's*
    wall-clock. In sequential mode we check between mreps; in parallel
    mode we use joblib's per-task timeout (any single mrep exceeding the
    cap is killed, and the trial is marked partially failed). The cap is
    a safety net for pathological configurations, not a tight schedule.
    """
    # ── combinatorial guard ──
    if not skip_combinatorial_check:
        ok, q = is_combinatorially_feasible(
            int(params["subspace_dim"]), int(params["polynomial_degree"])
        )
        if not ok:
            log.info(
                "Rejecting trial: subspace=%s degree=%s -> q=%d basis terms",
                params["subspace_dim"],
                params["polynomial_degree"],
                q,
            )
            return EvalResult(
                objectives=[],
                aligned_scores=[],
                failure_count=n_macroreps,
                failure_traces=[
                    f"combinatorial guard: q={q} > cap"
                ],
            )

    # ── factor dict ──
    try:
        factors = trial_params_to_solver_factors(params)
    except Exception as exc:
        return EvalResult(
            objectives=[],
            aligned_scores=[],
            failure_count=n_macroreps,
            failure_traces=[f"factor build error: {exc!r}\n{traceback.format_exc()}"],
        )

    effective_n_jobs = min(_resolve_n_jobs(n_jobs), n_macroreps)

    mrep_indices = [
        derive_mrep_index(base_seed=base_seed, trial_number=trial_number, mrep=mrep)
        for mrep in range(n_macroreps)
    ]

    objectives: list[float] = []
    diagnostics: list[MrepDiagnostics] = []
    failure_traces: list[str] = []
    failure_count = 0
    align_sign: float | None = None

    if effective_n_jobs > 1:
        # ── parallel path ──
        from joblib import Parallel, delayed
        from joblib.externals.loky.process_executor import TerminatedWorkerError

        log.debug(
            "Trial %s: running %d mreps with n_jobs=%d (parallel)",
            trial_number, n_macroreps, effective_n_jobs,
        )
        try:
            results = Parallel(
                n_jobs=effective_n_jobs,
                backend="loky",
                timeout=per_trial_wall_clock_cap_s,
                # Tell joblib not to capture stderr/stdout from children;
                # we want the workers' log messages to surface.
                verbose=0,
            )(
                delayed(_run_one_mrep)(
                    problem_name=problem_name,
                    dimension=dimension,
                    budget=budget,
                    factors=factors,
                    mrep_index=mrep_idx,
                )
                for mrep_idx in mrep_indices
            )
        except (TimeoutError, TerminatedWorkerError) as exc:
            log.warning(
                "Trial %s parallel mrep batch failed: %s", trial_number, exc
            )
            return EvalResult(
                objectives=[],
                aligned_scores=[],
                failure_count=n_macroreps,
                failure_traces=[f"parallel mrep batch failed: {exc!r}"],
            )
        except Exception as exc:
            log.warning(
                "Trial %s parallel mrep batch raised: %s", trial_number, exc
            )
            return EvalResult(
                objectives=[],
                aligned_scores=[],
                failure_count=n_macroreps,
                failure_traces=[f"parallel mrep batch raised: {exc!r}"],
            )

        for mrep, (elapsed, final_obj, diag, sign) in enumerate(results):
            if final_obj is None or not np.isfinite(final_obj) or diag is None:
                failure_count += 1
                failure_traces.append(
                    f"mrep {mrep}: non-finite obj or missing diag (obj={final_obj!r})"
                )
                continue
            if align_sign is None and sign is not None:
                align_sign = sign
            objectives.append(final_obj)
            diagnostics.append(diag)
    else:
        # ── sequential path (unchanged semantics) ──
        cumulative_t0 = time.perf_counter()
        for mrep, mrep_idx in enumerate(mrep_indices):
            if per_trial_wall_clock_cap_s is not None:
                elapsed_so_far = time.perf_counter() - cumulative_t0
                if elapsed_so_far > per_trial_wall_clock_cap_s:
                    remaining = n_macroreps - mrep
                    failure_count += remaining
                    failure_traces.append(
                        f"wall-clock cap {per_trial_wall_clock_cap_s:.0f}s exceeded "
                        f"after {elapsed_so_far:.0f}s; {remaining} mreps skipped"
                    )
                    break

            try:
                elapsed, final_obj, diag, sign = _run_one_mrep(
                    problem_name=problem_name,
                    dimension=dimension,
                    budget=budget,
                    factors=factors,
                    mrep_index=mrep_idx,
                )
            except Exception as exc:
                failure_count += 1
                failure_traces.append(f"mrep {mrep}: {exc!r}")
                log.warning("Trial %s mrep %s crashed: %s", trial_number, mrep, exc)
                continue

            if final_obj is None or not np.isfinite(final_obj) or diag is None:
                failure_count += 1
                failure_traces.append(
                    f"mrep {mrep}: non-finite obj or missing diag (obj={final_obj!r})"
                )
                continue

            if align_sign is None and sign is not None:
                align_sign = sign
            objectives.append(final_obj)
            diagnostics.append(diag)

    aligned = (
        [-align_sign * o for o in objectives] if align_sign is not None else []
    )

    result = EvalResult(
        objectives=objectives,
        aligned_scores=aligned,
        failure_count=failure_count,
        failure_traces=failure_traces,
        diagnostics_per_mrep=diagnostics,
        diagnostics_aggregate=aggregate_diagnostics(diagnostics),
    )

    if objectives:
        arr = np.asarray(aligned, dtype=float)
        obj_arr = np.asarray(objectives, dtype=float)
        result.mean_aligned = float(np.mean(arr))
        result.std_aligned = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        result.mean_objective = float(np.mean(obj_arr))
        result.std_objective = (
            float(np.std(obj_arr, ddof=1)) if obj_arr.size > 1 else 0.0
        )
    return result


# ── scoring (the value Optuna minimises) ───────────────────────────────────


def composite_score(
    result: EvalResult,
    *,
    n_macroreps_target: int,
    std_weight: float = 0.15,
    failure_penalty: float = 1e6,
) -> float:
    """Composite objective Optuna minimises.

    score = mean_aligned + std_weight * std_aligned + failure_penalty * f
        where f is the fraction of macroreps that failed.

    The "aligned" mean is sign-flipped so lower is always better
    regardless of whether the underlying problem is maximisation or
    minimisation. A trial with zero successful macroreps returns +inf.
    """
    if result.is_failure():
        return float("inf")
    f = result.failure_count / max(1, n_macroreps_target)
    return (
        float(result.mean_aligned)
        + std_weight * float(result.std_aligned)
        + failure_penalty * f
    )
