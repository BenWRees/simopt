"""Slurm-array worker entry point.

One process = one async Optuna worker. The worker pulls trials from the
shared storage backend, evaluates them, writes results back, and exits
when its trial budget OR walltime budget is exhausted. Multiple workers
running concurrently share the SQLite/Postgres storage; Optuna's
serialised trial dispatch handles the locking.

Usage (local)::

    python -m scripts.tuning.worker --problem SAN-1 --n-trials 10

Usage (Slurm array)::

    python -m scripts.tuning.worker --problem SAN-1 --n-trials 200 \\
        --walltime-s 14000 --seed-offset $SLURM_ARRAY_TASK_ID

The worker sets BLAS thread caps before importing NumPy so it never
oversubscribes the CPU it was allocated.
"""

from __future__ import annotations

# ── BLAS thread caps (must precede numpy import) ──────────────────────────
import os

for _var, _val in (
    ("OMP_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
    ("VECLIB_MAXIMUM_THREADS", "1"),
    ("LOKY_MAX_CPU_COUNT", "1"),
):
    os.environ.setdefault(_var, _val)

# ── stdlib + project imports ──────────────────────────────────────────────
import argparse
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tuning.tuner import (  # noqa: E402
    DEFAULT_BUDGET,
    DEFAULT_RUNGS,
    build_study,
    run_worker,
    storage_url_for,
    study_name,
)

log = logging.getLogger("astromorf.tuning.worker")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Async Optuna worker for ASTROMoRF tuning."
    )
    p.add_argument("--problem", required=True)
    p.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Trials this worker will attempt before exiting.",
    )
    p.add_argument(
        "--walltime-s",
        type=float,
        default=None,
        help=(
            "Soft walltime budget in seconds. Worker stops after the next "
            "trial completes once exceeded. Use just below SLURM walltime."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=20250428,
        help="Base seed shared across workers.",
    )
    p.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="Per-worker offset (typically $SLURM_ARRAY_TASK_ID).",
    )
    p.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    p.add_argument(
        "--per-trial-cap-s",
        type=float,
        default=30 * 60,
        help="Per-trial wall-clock cap. Trials exceeding this are cut short.",
    )
    p.add_argument(
        "--storage",
        type=str,
        default=None,
        help=(
            "Optuna storage URL. If omitted, derived from "
            "ASTROMORF_OPTUNA_STORAGE or default SQLite path."
        ),
    )
    p.add_argument("--std-weight", type=float, default=0.15)
    p.add_argument("--failure-penalty", type=float, default=1e6)
    p.add_argument(
        "--n-jobs",
        type=str,
        default="1",
        help=(
            "Intra-trial macroreplication parallelism. Integer or 'auto' "
            "(reads $SLURM_CPUS_PER_TASK). Defaults to 1 (sequential)."
        ),
    )
    p.add_argument(
        "--init-only",
        action="store_true",
        help=(
            "Initialise the storage backend (create schema + enqueue warmstart) "
            "and exit without running any trials. Use this before sbatch'ing a "
            "Slurm array against a fresh SQLite study to avoid CREATE TABLE races."
        ),
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    storage = args.storage or storage_url_for(args.problem)
    seed = args.seed + args.seed_offset

    if args.init_only:
        # Single-process schema creation. Always uses the *base* seed (not
        # the per-worker offset) so the warmstart trial is enqueued
        # consistently regardless of which Slurm task ran the init.
        log.info(
            "Init-only: problem=%s storage=%s study=%s",
            args.problem, storage, study_name(args.problem),
        )
        study = build_study(args.problem, storage=storage, seed=args.seed)
        n_existing = len(study.get_trials(deepcopy=False))
        log.info(
            "Init-only complete: study has %d existing trial(s) (warm-start counts as 1).",
            n_existing,
        )
        return

    log.info(
        "Starting worker: problem=%s trials=%d walltime_s=%s seed=%d "
        "n_jobs=%s storage=%s study=%s",
        args.problem,
        args.n_trials,
        args.walltime_s,
        seed,
        args.n_jobs,
        storage,
        study_name(args.problem),
    )

    t0 = time.time()
    n_done = run_worker(
        problem_name=args.problem,
        n_trials=args.n_trials,
        storage=storage,
        seed=seed,
        budget=args.budget,
        rungs=DEFAULT_RUNGS,
        std_weight=args.std_weight,
        failure_penalty=args.failure_penalty,
        per_trial_wall_clock_cap_s=args.per_trial_cap_s,
        walltime_budget_s=args.walltime_s,
        n_jobs=args.n_jobs,
    )
    elapsed = time.time() - t0
    log.info(
        "Worker exiting: trials_completed_in_study=%d elapsed_s=%.1f",
        n_done,
        elapsed,
    )


if __name__ == "__main__":
    main()
