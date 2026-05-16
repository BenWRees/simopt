"""SLURM-array worker entry point for HPC-safe ASTROMoRF tuning.

One process = one async Optuna worker. The worker pulls trials from the
shared storage backend (JournalStorage by default, Postgres or SQLite by
URL), evaluates them, writes results back, and exits when its trial
budget OR walltime budget is exhausted.

Workers coordinate purely through storage; there is no per-worker state
on disk outside of:
  - ``trials_jsonl/<problem>.w<worker_id>.jsonl`` (private to this worker)
  - the storage backend itself (shared)

Usage (local)::

    python -m scripts.tuning.worker --problem SAN-1 --n-trials 10

Usage (SLURM array)::

    python -m scripts.tuning.worker --problem SAN-1 --n-trials 50 \\
        --walltime-s 14000 --worker-id "$SLURM_ARRAY_TASK_ID"

The worker caps BLAS thread counts before importing NumPy so it never
oversubscribes the CPU(s) SLURM allocated. Trial-internal macroreplication
parallelism is controlled by ``--n-jobs auto`` (reads
``$SLURM_CPUS_PER_TASK``).
"""

from __future__ import annotations

# ── BLAS thread caps (must precede numpy/sklearn/joblib imports) ──────────
import os

for _var, _val in (
    ("OMP_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
    ("VECLIB_MAXIMUM_THREADS", "1"),
):
    os.environ.setdefault(_var, _val)

# joblib's loky backend respects LOKY_MAX_CPU_COUNT for the *parent* worker
# count. Workers inherit OMP/MKL=1 from above, so each child does serial
# BLAS. We honour SLURM_CPUS_PER_TASK if present; otherwise default to 1.
if "LOKY_MAX_CPU_COUNT" not in os.environ:
    os.environ["LOKY_MAX_CPU_COUNT"] = os.environ.get("SLURM_CPUS_PER_TASK", "1")

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


def _resolve_worker_id(args: argparse.Namespace) -> int:
    """Pick a stable per-worker integer ID.

    Priority: ``--worker-id`` > legacy ``--seed-offset`` >
    ``$SLURM_ARRAY_TASK_ID`` > process PID.
    """
    if args.worker_id is not None:
        return int(args.worker_id)
    if args.seed_offset is not None:
        return int(args.seed_offset)
    env = os.environ.get("SLURM_ARRAY_TASK_ID")
    if env is not None and env.strip():
        try:
            return int(env)
        except ValueError:
            pass
    return os.getpid() & 0xFFFF


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Async Optuna worker for ASTROMoRF tuning (HPC-safe)."
    )
    p.add_argument("--problem", required=True)
    p.add_argument(
        "--n-trials", type=int, default=20,
        help="Trials this worker will attempt before exiting.",
    )
    p.add_argument(
        "--max-trials", type=int, default=None,
        help=(
            "Global trial cap for the shared study. When set, workers stop "
            "requesting new trials once the study reaches this total."
        ),
    )
    p.add_argument(
        "--walltime-s", type=float, default=None,
        help=(
            "Soft walltime budget in seconds. Worker stops after the next "
            "trial completes once exceeded. Set ~180s below SLURM walltime."
        ),
    )
    p.add_argument(
        "--seed", type=int, default=20250428,
        help="Base seed shared across workers (sampler diverges via worker-id).",
    )
    p.add_argument(
        "--worker-id", type=int, default=None,
        help=(
            "Integer worker identifier. Used to (a) diverge the TPE sampler "
            "seed and (b) name this worker's private JSONL log. Defaults to "
            "$SLURM_ARRAY_TASK_ID or the PID."
        ),
    )
    p.add_argument(
        "--seed-offset", type=int, default=None,
        help="Deprecated alias for --worker-id (kept for back-compat).",
    )
    p.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    p.add_argument(
        "--per-trial-cap-s", type=float, default=30 * 60,
        help="Per-trial wall-clock cap. Pathological configs are cut short.",
    )
    p.add_argument(
        "--storage", type=str, default=None,
        help=(
            "Optuna storage URL. If omitted, derived from "
            "ASTROMORF_OPTUNA_STORAGE (default: JournalStorage)."
        ),
    )
    p.add_argument("--std-weight", type=float, default=0.15)
    p.add_argument("--failure-penalty", type=float, default=1e6)
    p.add_argument(
        "--n-jobs", type=str, default="1",
        help=(
            "Intra-trial macroreplication parallelism. Integer or 'auto' "
            "(reads $SLURM_CPUS_PER_TASK). Defaults to 1 (sequential)."
        ),
    )
    p.add_argument(
        "--startup-jitter-s", type=float, default=15.0,
        help=(
            "Random startup delay (0..N) so a SLURM array of workers doesn't "
            "open storage simultaneously. Set 0 to disable (e.g. for smoke)."
        ),
    )
    p.add_argument(
        "--init-only", action="store_true",
        help=(
            "Initialise the storage backend (create study + enqueue warmstart) "
            "and exit without running any trials. Use before sbatch'ing a "
            "large array if you want one-shot setup."
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
    worker_id = _resolve_worker_id(args)
    storage = args.storage  # None -> resolved by storage layer

    if args.init_only:
        log.info(
            "Init-only: problem=%s study=%s storage=%s",
            args.problem, study_name(args.problem),
            storage or "(env/default)",
        )
        study, spec = build_study(
            args.problem,
            storage=storage,
            seed=args.seed,
            worker_id=0,
            max_trials=args.max_trials,
        )
        n_existing = len(study.get_trials(deepcopy=False))
        log.info(
            "Init-only complete: backend=%s storage=%s n_trials=%d "
            "(warm-start counts as 1).",
            spec.backend, spec.display, n_existing,
        )
        return

    log.info(
        "Worker %d starting: problem=%s n_trials=%d walltime_s=%s seed=%d "
        "n_jobs=%s storage=%s study=%s",
        worker_id, args.problem, args.n_trials, args.walltime_s,
        args.seed, args.n_jobs, storage or "(env/default)",
        study_name(args.problem),
    )

    t0 = time.time()
    n_done = run_worker(
        problem_name=args.problem,
        n_trials=args.n_trials,
        storage=storage,
        seed=args.seed,
        worker_id=worker_id,
        max_trials=args.max_trials,
        budget=args.budget,
        rungs=DEFAULT_RUNGS,
        std_weight=args.std_weight,
        failure_penalty=args.failure_penalty,
        per_trial_wall_clock_cap_s=args.per_trial_cap_s,
        walltime_budget_s=args.walltime_s,
        n_jobs=args.n_jobs,
        startup_jitter_s=args.startup_jitter_s,
    )
    elapsed = time.time() - t0
    log.info(
        "Worker %d exiting: trials_in_study=%d elapsed_s=%.1f",
        worker_id, n_done, elapsed,
    )


if __name__ == "__main__":
    main()
