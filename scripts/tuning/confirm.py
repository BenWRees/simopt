"""Confirmation pass: re-evaluate the top-K trials on a disjoint seed stream.

Optuna's "best trial" selection over a noisy objective can latch onto a
favourable seed; the confirmation pass guards against that by re-running
the top-K candidates with a fresh seed sequence and a higher
macroreplication count, then selecting the candidate whose
*confirmation* mean (lower-is-better aligned) is best.

The pass writes:
  - ``confirm_<problem>.json`` -- per-candidate confirmation results
  - ``best_<problem>.json``    -- final winner (full factor dict + stats)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# BLAS thread caps (must precede numpy import).
for _var, _val in (
    ("OMP_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
    ("VECLIB_MAXIMUM_THREADS", "1"),
    ("LOKY_MAX_CPU_COUNT", "1"),
):
    os.environ.setdefault(_var, _val)

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import optuna  # noqa: E402

from scripts.tuning import results_root  # noqa: E402
from scripts.tuning.evaluate import (  # noqa: E402
    DEFAULT_BUDGET,
    composite_score,
    evaluate_config,
    trial_params_to_solver_factors,
)
from scripts.tuning.spaces import get_space  # noqa: E402
from scripts.tuning.storage import make_storage  # noqa: E402
from scripts.tuning.tuner import study_name  # noqa: E402

log = logging.getLogger("astromorf.tuning.confirm")


# A different additive offset than anything used by the tuner rungs (which
# add 0, 1009, 2018) so the confirmation pass uses fully disjoint streams.
CONFIRMATION_SEED_OFFSET: int = 9_973


def _serialize_factors(factors: dict[str, Any]) -> dict[str, Any]:
    out = dict(factors)
    if "polynomial basis" in out:
        b = out["polynomial basis"]
        out["polynomial basis"] = b.value if hasattr(b, "value") else str(b)
    return out


def _ci95(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        return (float("nan"), float("nan"))
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    sem = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
    return (mean - 1.96 * sem, mean + 1.96 * sem)


def get_top_k_trials(
    storage: Any, study_nm: str, *, k: int
) -> list[optuna.trial.FrozenTrial]:
    """Return the K best COMPLETE trials from the study, lower-objective first.

    *storage* may be a URL or an Optuna storage object.
    """
    study = optuna.load_study(study_name=study_nm, storage=storage)
    completed = [
        t
        for t in study.get_trials(
            deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)
        )
        if t.value is not None and np.isfinite(t.value)
    ]
    completed.sort(key=lambda t: t.value)
    return completed[:k]


def confirm_top_k(
    problem_name: str,
    *,
    k: int = 5,
    n_macroreps: int = 30,
    n_postreps: int = 200,
    base_seed: int = 20250428,
    budget: int = DEFAULT_BUDGET,
    storage: str | None = None,
    std_weight: float = 0.15,
    failure_penalty: float = 1e6,
    # Outer-level parallelism: how many candidate evaluations to run in parallel
    n_candidate_jobs: int = 1,
    # How many worker processes each candidate should use for its macroreps.
    # May be an int, the string 'auto', or None to auto-compute from SLURM env.
    per_candidate_n_jobs: int | str | None = None,
) -> dict[str, Any]:
    """Re-evaluate the top-K trials with a disjoint seed stream."""
    storage_obj, spec = make_storage(problem_name, storage_url=storage)
    log.info("Confirm: using storage backend=%s (%s)", spec.backend, spec.display)
    space = get_space(problem_name)
    top = get_top_k_trials(storage_obj, study_name(problem_name), k=k)
    log.info("Confirming top-%d for %s (study has %d candidates)",
             k, problem_name, len(top))

    confirm_seed = base_seed + CONFIRMATION_SEED_OFFSET

    # Local helper that evaluates one trial and returns the candidate record.
    def _eval_trial(rank: int, t: Any, mrep_n_jobs: int | str | None) -> dict[str, Any]:
        log.info(
            "Confirming rank %d (trial %d, study_value=%.6g)",
            rank,
            t.number,
            t.value,
        )
        try:
            result = evaluate_config(
                problem_name=problem_name,
                dimension=space.problem_dim if space.use_scaling else None,
                budget=budget,
                params=dict(t.params),
                n_macroreps=n_macroreps,
                base_seed=confirm_seed,
                trial_number=t.number,
                n_jobs=mrep_n_jobs,
            )
        except Exception as exc:  # defensive: synthesize a failure-shaped result
            log.exception("Trial %s evaluation crashed: %s", t.number, exc)
            result = type("_", (), {})()
            result.mean_aligned = float("nan")
            result.std_aligned = float("nan")
            result.mean_objective = float("nan")
            result.std_objective = float("nan")
            result.aligned_scores = []
            result.failure_count = n_macroreps
            result.diagnostics_aggregate = {}
            result.objectives = []

        score = composite_score(
            result,
            n_macroreps_target=n_macroreps,
            std_weight=std_weight,
            failure_penalty=failure_penalty,
        )
        ci_low, ci_high = _ci95(result.aligned_scores)
        return {
            "rank_in_study": rank,
            "trial_number": t.number,
            "study_value": float(t.value),
            "study_user_attrs": dict(t.user_attrs),
            "params": dict(t.params),
            "factors": _serialize_factors(
                trial_params_to_solver_factors(dict(t.params))
            ),
            "confirm_score": score,
            "confirm_mean_aligned": result.mean_aligned,
            "confirm_std_aligned": result.std_aligned,
            "confirm_mean_objective": result.mean_objective,
            "confirm_std_objective": result.std_objective,
            "confirm_ci95_aligned": [ci_low, ci_high],
            "confirm_failure_count": result.failure_count,
            "confirm_diagnostics": result.diagnostics_aggregate,
            "confirm_objectives": result.objectives,
        }

    # Decide CPU distribution. If SLURM provides SLURM_CPUS_PER_TASK use it,
    # otherwise fall back to the logical CPU count.
    try:
        total_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    except Exception:
        total_cpus = 1

    # Interpret explicit per-candidate setting; allow 'auto' to be passed.
    explicit_mrep_jobs: int | str | None
    if per_candidate_n_jobs is None:
        explicit_mrep_jobs = None
    elif isinstance(per_candidate_n_jobs, str) and per_candidate_n_jobs.lower() == "auto":
        explicit_mrep_jobs = "auto"
    else:
        try:
            explicit_mrep_jobs = int(per_candidate_n_jobs)  # type: ignore[arg-type]
        except Exception:
            explicit_mrep_jobs = None

    candidates: list[dict[str, Any]] = []
    if n_candidate_jobs <= 1:
        # Sequential candidate evaluation: give each trial access to all CPUs
        # (pass 'auto' so evaluate_config picks up SLURM_CPUS_PER_TASK).
        mrep_n_jobs_for_each = explicit_mrep_jobs if explicit_mrep_jobs is not None else "auto"
        for rank, t in enumerate(top):
            candidates.append(_eval_trial(rank, t, mrep_n_jobs_for_each))
    else:
        # Parallel candidate evaluation: distribute total_cpus across candidate
        # workers unless the user explicitly requested a per-candidate value.
        if explicit_mrep_jobs is None or explicit_mrep_jobs == "auto":
            per_candidate = max(1, total_cpus // n_candidate_jobs)
        else:
            per_candidate = explicit_mrep_jobs

        from joblib import Parallel, delayed

        log.info(
            "Running confirmation in parallel: %d candidates @ %s mrep-jobs each",
            n_candidate_jobs,
            str(per_candidate),
        )
        results = Parallel(n_jobs=n_candidate_jobs, backend="threading")(
            delayed(_eval_trial)(rank, t, per_candidate) for rank, t in enumerate(top)
        )
        candidates.extend(results)

    # Pick the winner: lowest confirm_score (which already includes the
    # robustness penalty and failure penalty).
    candidates.sort(key=lambda c: c["confirm_score"])
    winner = candidates[0] if candidates else None

    out = {
        "problem": problem_name,
        "ts": time.time(),
        "n_macroreps_confirm": n_macroreps,
        "n_postreps_confirm": n_postreps,
        "budget": budget,
        "confirm_seed_base": confirm_seed,
        "k": k,
        "candidates_ranked_by_confirm_score": candidates,
        "winner": winner,
    }

    out_dir = results_root() / "confirmations"
    out_dir.mkdir(parents=True, exist_ok=True)
    confirm_path = out_dir / f"confirm_{problem_name}.json"
    confirm_path.write_text(json.dumps(out, indent=2, default=str))
    log.info("Wrote confirmation results to %s", confirm_path)

    if winner is not None:
        # Direction metadata so the user can read best_<problem>.json without
        # having to remember whether the problem is min or max.
        try:
            study = optuna.load_study(study_name=study_name(problem_name), storage=storage_obj)
            raw_direction = study.user_attrs.get("raw_direction")
            minmax_sign = study.user_attrs.get("raw_minmax_sign")
        except Exception:
            raw_direction, minmax_sign = None, None

        best_dir = results_root() / "best"
        best_dir.mkdir(parents=True, exist_ok=True)
        best_path = best_dir / f"best_{problem_name}.json"
        best_path.write_text(
            json.dumps(
                {
                    "problem": problem_name,
                    "direction": raw_direction,           # "maximize" / "minimize"
                    "minmax_sign": minmax_sign,           # +1 / -1
                    "raw_better": (
                        "higher" if (minmax_sign or 0) > 0 else "lower"
                    ),
                    "selection_method": "top-K confirmation pass (disjoint seeds)",
                    "trial_number": winner["trial_number"],
                    "factors": winner["factors"],
                    "params": winner["params"],
                    "confirm_mean_objective": winner["confirm_mean_objective"],
                    "confirm_std_objective": winner["confirm_std_objective"],
                    "confirm_mean_aligned": winner["confirm_mean_aligned"],
                    "confirm_std_aligned": winner["confirm_std_aligned"],
                    "confirm_ci95_aligned": winner["confirm_ci95_aligned"],
                    "confirm_diagnostics": winner["confirm_diagnostics"],
                    "study_value": winner["study_value"],
                    "n_macroreps_confirm": n_macroreps,
                    "budget": budget,
                    "_units_note": (
                        "confirm_mean_objective is the RAW objective in the "
                        "problem's natural units. confirm_mean_aligned and "
                        "study_value are the aligned score = -minmax_sign * raw, "
                        "always lower-is-better (this is what Optuna minimises)."
                    ),
                },
                indent=2,
                default=str,
            )
        )
        log.info("Wrote winning factors to %s", best_path)
    return out


def _write_candidate_file(problem_name: str, rank: int, candidate: dict[str, Any]) -> None:
    out_dir = results_root() / "confirmations"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"confirm_{problem_name}_candidate_{rank}.json"
    path.write_text(json.dumps(candidate, indent=2, default=str))
    log.info("Wrote per-candidate result to %s", path)


def _aggregate_candidate_files(
    problem_name: str,
    *,
    k: int,
    n_macroreps: int,
    n_postreps: int,
    base_seed: int,
    budget: int,
    std_weight: float,
    failure_penalty: float,
    storage: str | None = None,
) -> dict[str, Any] | None:
    """Aggregate per-candidate JSON files into the final confirm/best JSONs."""
    import glob

    confirm_dir = results_root() / "confirmations"
    pattern = str(confirm_dir / f"confirm_{problem_name}_candidate_*.json")
    paths = sorted(glob.glob(pattern))
    if not paths:
        log.error("No per-candidate files found matching %s", pattern)
        return None

    candidates: list[dict[str, Any]] = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                cand = json.load(f)
        except Exception as exc:
            log.warning("Failed to read candidate file %s: %s", p, exc)
            continue
        candidates.append(cand)

    if not candidates:
        log.error("No readable per-candidate files to aggregate for %s", problem_name)
        return None

    # Sort by confirm_score (missing -> +inf)
    candidates.sort(key=lambda c: c.get("confirm_score", float("inf")))
    winner = candidates[0] if candidates else None

    out = {
        "problem": problem_name,
        "ts": time.time(),
        "n_macroreps_confirm": n_macroreps,
        "n_postreps_confirm": n_postreps,
        "budget": budget,
        "confirm_seed_base": base_seed + CONFIRMATION_SEED_OFFSET,
        "k": k,
        "candidates_ranked_by_confirm_score": candidates,
        "winner": winner,
    }

    confirm_path = confirm_dir / f"confirm_{problem_name}.json"
    confirm_path.write_text(json.dumps(out, indent=2, default=str))
    log.info("Wrote aggregated confirmation results to %s", confirm_path)

    if winner is not None:
        try:
            storage_obj, spec = make_storage(problem_name, storage_url=storage)
            study = optuna.load_study(study_name=study_name(problem_name), storage=storage_obj)
            raw_direction = study.user_attrs.get("raw_direction")
            minmax_sign = study.user_attrs.get("raw_minmax_sign")
        except Exception:
            raw_direction, minmax_sign = None, None

        best_dir = results_root() / "best"
        best_dir.mkdir(parents=True, exist_ok=True)
        best_path = best_dir / f"best_{problem_name}.json"
        best_path.write_text(
            json.dumps(
                {
                    "problem": problem_name,
                    "direction": raw_direction,
                    "minmax_sign": minmax_sign,
                    "raw_better": ("higher" if (minmax_sign or 0) > 0 else "lower"),
                    "selection_method": "top-K confirmation pass (disjoint seeds)",
                    "trial_number": winner.get("trial_number"),
                    "factors": winner.get("factors"),
                    "params": winner.get("params"),
                    "confirm_mean_objective": winner.get("confirm_mean_objective"),
                    "confirm_std_objective": winner.get("confirm_std_objective"),
                    "confirm_mean_aligned": winner.get("confirm_mean_aligned"),
                    "confirm_std_aligned": winner.get("confirm_std_aligned"),
                    "confirm_ci95_aligned": winner.get("confirm_ci95_aligned"),
                    "confirm_diagnostics": winner.get("confirm_diagnostics"),
                    "study_value": winner.get("study_value"),
                    "n_macroreps_confirm": n_macroreps,
                    "budget": budget,
                    "_units_note": (
                        "confirm_mean_objective is the RAW objective in the "
                        "problem's natural units. confirm_mean_aligned and "
                        "study_value are the aligned score = -minmax_sign * raw, "
                        "always lower-is-better (this is what Optuna minimises)."
                    ),
                },
                indent=2,
                default=str,
            )
        )
        log.info("Wrote winning factors to %s", best_path)
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Confirmation pass over the top-K Optuna trials."
    )
    p.add_argument("--problem", required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--n-macroreps", type=int, default=30)
    p.add_argument("--n-postreps", type=int, default=200)
    p.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    p.add_argument("--seed", type=int, default=20250428)
    p.add_argument("--storage", type=str, default=None)
    p.add_argument("--std-weight", type=float, default=0.15)
    p.add_argument("--failure-penalty", type=float, default=1e6)
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--n-candidate-jobs",
        type=int,
        default=1,
        help="Number of candidate evaluations to run in parallel (outer level)",
    )
    p.add_argument(
        "--per-candidate-n-jobs",
        type=str,
        default=None,
        help=(
            "Per-candidate macrorep n_jobs (int or 'auto'). If not set and "
            "candidates are parallelised, CPUs are divided from SLURM_CPUS_PER_TASK."
        ),
    )
    p.add_argument(
        "--candidate-index",
        type=int,
        default=None,
        help="If set (or when running under srun), evaluate only this candidate index (0-based)",
    )
    p.add_argument(
        "--aggregate",
        action="store_true",
        help="Aggregate per-candidate JSON files into final confirm and best JSONs",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    # Candidate-per-task mode: if --candidate-index is provided or we are
    # running under srun (SLURM_PROCID present) evaluate a single candidate
    # and write a per-candidate JSON file. The aggregator run (--aggregate)
    # will merge these into the final confirm_{problem}.json and best_{problem}.json.
    candidate_index = args.candidate_index
    if candidate_index is None:
        # srun provides SLURM_PROCID for each task (0..ntasks-1)
        env_idx = os.environ.get("SLURM_PROCID") or os.environ.get("SLURM_LOCALID")
        if env_idx is not None:
            try:
                candidate_index = int(env_idx)
            except Exception:
                candidate_index = None

    if args.aggregate:
        _aggregate_candidate_files(
            args.problem,
            k=args.k,
            n_macroreps=args.n_macroreps,
            n_postreps=args.n_postreps,
            base_seed=args.seed,
            budget=args.budget,
            std_weight=args.std_weight,
            failure_penalty=args.failure_penalty,
            storage=args.storage,
        )
        return

    if candidate_index is not None:
        # Evaluate a single candidate (rank = candidate_index)
        storage_obj, spec = make_storage(args.problem, storage_url=args.storage)
        space = get_space(args.problem)
        top = get_top_k_trials(storage_obj, study_name(args.problem), k=args.k)
        if candidate_index < 0 or candidate_index >= len(top):
            log.error("Requested candidate index %s out of range (0..%d)", candidate_index, len(top) - 1)
            return
        t = top[candidate_index]
        confirm_seed = args.seed + CONFIRMATION_SEED_OFFSET

        # Interpret per-candidate n_jobs
        mrep_n_jobs: int | str | None
        if args.per_candidate_n_jobs is None:
            mrep_n_jobs = "auto"
        else:
            try:
                if str(args.per_candidate_n_jobs).lower() == "auto":
                    mrep_n_jobs = "auto"
                else:
                    mrep_n_jobs = int(args.per_candidate_n_jobs)
            except Exception:
                mrep_n_jobs = "auto"

        log.info("Evaluating candidate %d (trial %d) with n_jobs=%s", candidate_index, t.number, str(mrep_n_jobs))
        try:
            result = evaluate_config(
                problem_name=args.problem,
                dimension=space.problem_dim if space.use_scaling else None,
                budget=args.budget,
                params=dict(t.params),
                n_macroreps=args.n_macroreps,
                base_seed=confirm_seed,
                trial_number=t.number,
                n_jobs=mrep_n_jobs,
            )
        except Exception as exc:
            log.exception("Candidate %s evaluation crashed: %s", candidate_index, exc)
            # Synthesize failure-shaped result
            result = type("_", (), {})()
            result.mean_aligned = float("nan")
            result.std_aligned = float("nan")
            result.mean_objective = float("nan")
            result.std_objective = float("nan")
            result.aligned_scores = []
            result.failure_count = args.n_macroreps
            result.diagnostics_aggregate = {}
            result.objectives = []

        score = composite_score(
            result,
            n_macroreps_target=args.n_macroreps,
            std_weight=args.std_weight,
            failure_penalty=args.failure_penalty,
        )
        ci_low, ci_high = _ci95(result.aligned_scores)
        candidate = {
            "rank_in_study": candidate_index,
            "trial_number": t.number,
            "study_value": float(t.value),
            "study_user_attrs": dict(t.user_attrs),
            "params": dict(t.params),
            "factors": _serialize_factors(trial_params_to_solver_factors(dict(t.params))),
            "confirm_score": score,
            "confirm_mean_aligned": result.mean_aligned,
            "confirm_std_aligned": result.std_aligned,
            "confirm_mean_objective": result.mean_objective,
            "confirm_std_objective": result.std_objective,
            "confirm_ci95_aligned": [ci_low, ci_high],
            "confirm_failure_count": result.failure_count,
            "confirm_diagnostics": result.diagnostics_aggregate,
            "confirm_objectives": result.objectives,
        }
        _write_candidate_file(args.problem, candidate_index, candidate)
        return

    # Fallback: run the full confirm_top_k in-process (backwards-compatible)
    confirm_top_k(
        problem_name=args.problem,
        k=args.k,
        n_macroreps=args.n_macroreps,
        n_postreps=args.n_postreps,
        base_seed=args.seed,
        budget=args.budget,
        storage=args.storage,
        std_weight=args.std_weight,
        failure_penalty=args.failure_penalty,
        n_candidate_jobs=args.n_candidate_jobs,
        per_candidate_n_jobs=args.per_candidate_n_jobs,
    )


if __name__ == "__main__":
    main()
