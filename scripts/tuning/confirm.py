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
) -> dict[str, Any]:
    """Re-evaluate the top-K trials with a disjoint seed stream."""
    storage_obj, spec = make_storage(problem_name, storage_url=storage)
    log.info("Confirm: using storage backend=%s (%s)", spec.backend, spec.display)
    space = get_space(problem_name)
    top = get_top_k_trials(storage_obj, study_name(problem_name), k=k)
    log.info("Confirming top-%d for %s (study has %d candidates)",
             k, problem_name, len(top))

    confirm_seed = base_seed + CONFIRMATION_SEED_OFFSET
    candidates: list[dict[str, Any]] = []
    for rank, t in enumerate(top):
        log.info(
            "Confirming rank %d (trial %d, study_value=%.6g)",
            rank,
            t.number,
            t.value,
        )
        result = evaluate_config(
            problem_name=problem_name,
            dimension=space.problem_dim if space.use_scaling else None,
            budget=budget,
            params=dict(t.params),
            n_macroreps=n_macroreps,
            base_seed=confirm_seed,
            trial_number=t.number,
        )
        score = composite_score(
            result,
            n_macroreps_target=n_macroreps,
            std_weight=std_weight,
            failure_penalty=failure_penalty,
        )
        ci_low, ci_high = _ci95(result.aligned_scores)
        candidates.append(
            {
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
        )

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
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
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
    )


if __name__ == "__main__":
    main()
