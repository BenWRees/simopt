"""Local smoke test for the ASTROMoRF tuner.

Runs a tiny tuning study end-to-end (a handful of trials, single ASHA
rung) so the user can verify the code path before submitting any Slurm
jobs. Also useful in CI.

Usage::

    python -m scripts.tuning.smoke --problem DYNAMNEWS-1 --n-trials 2
"""

from __future__ import annotations

# BLAS thread caps before numpy import.
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

import argparse
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tuning.tuner import Rung, run_worker  # noqa: E402

log = logging.getLogger("astromorf.tuning.smoke")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke test for the ASTROMoRF tuner.")
    p.add_argument("--problem", default="DYNAMNEWS-1")
    p.add_argument("--n-trials", type=int, default=2)
    p.add_argument(
        "--budget",
        type=int,
        default=2_000,
        help="Smoke budget; production tuning uses 10 000.",
    )
    p.add_argument("--seed", type=int, default=20250428)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    # One short rung only -- enough to exercise the full pipeline cheaply.
    rungs = (Rung(step=0, n_macroreps=2, n_postreps=10),)
    log.info("Smoke run: problem=%s n_trials=%d budget=%d", args.problem,
             args.n_trials, args.budget)
    t0 = time.time()
    n = run_worker(
        problem_name=args.problem,
        n_trials=args.n_trials,
        seed=args.seed,
        budget=args.budget,
        rungs=rungs,
        per_trial_wall_clock_cap_s=10 * 60,
        walltime_budget_s=30 * 60,
    )
    log.info("Smoke complete: %d trials in study after %.1fs", n, time.time() - t0)


if __name__ == "__main__":
    main()
