"""Mark zombie ``RUNNING`` trials as ``FAIL`` after a worker crash/timeout.

Workers that are killed mid-trial (Slurm timeout, OOM, node failure) leave
their trial in state ``RUNNING`` because Optuna only commits a terminal
state when the objective returns or raises. The next worker's heartbeat
sweep handles this automatically (see ``tuner.build_study``'s heartbeat
params), but you may also want to run this manually -- e.g. when you
won't be submitting another array for a while and want a clean dashboard.

The CLI is safe to run while workers are active: it only touches trials
whose state is ``RUNNING`` AND whose last heartbeat is older than
``--max-age-s`` seconds. The default (600s) is twice the grace period
used in ``build_study`` so we never race a live worker.

Usage::

    python -m scripts.tuning.cleanup --problems SAN-1,NETWORK-1
    python -m scripts.tuning.cleanup --problems SAN-1 --max-age-s 1800
    python -m scripts.tuning.cleanup --problems SAN-1 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import optuna  # noqa: E402

from scripts.tuning.tuner import storage_url_for, study_name  # noqa: E402

log = logging.getLogger("astromorf.tuning.cleanup")


def _trial_age_s(trial: optuna.trial.FrozenTrial) -> float | None:
    """Best-effort 'how long since this trial was last touched'."""
    candidates: list[datetime] = []
    if trial.datetime_complete is not None:
        candidates.append(trial.datetime_complete)
    if trial.datetime_start is not None:
        candidates.append(trial.datetime_start)
    # Some Optuna versions stamp a heartbeat into user_attrs.
    hb = trial.user_attrs.get("__optuna_heartbeat__") if trial.user_attrs else None
    if isinstance(hb, (int, float)):
        candidates.append(datetime.fromtimestamp(hb, tz=timezone.utc))
    if not candidates:
        return None
    last = max(candidates)
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - last).total_seconds()


def cleanup_problem(
    problem_name: str,
    *,
    max_age_s: float = 600.0,
    dry_run: bool = False,
    storage: str | None = None,
) -> dict[str, int]:
    """Mark zombie RUNNING trials as FAIL for *problem_name*.

    Returns a counts dict: ``{"running": N, "marked_fail": M, "kept": K}``.
    """
    storage_url = storage or storage_url_for(problem_name)
    try:
        study = optuna.load_study(
            study_name=study_name(problem_name), storage=storage_url
        )
    except KeyError:
        log.warning("No study for %s at %s -- skipping.", problem_name, storage_url)
        return {"running": 0, "marked_fail": 0, "kept": 0}

    storage_obj = study._storage  # underlying RDBStorage

    running = [
        t
        for t in study.get_trials(
            deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)
        )
    ]
    marked = 0
    kept = 0
    for t in running:
        age = _trial_age_s(t)
        if age is None or age >= max_age_s:
            age_str = f"{age:.0f}s" if age is not None else "unknown age"
            log.info(
                "%s: marking trial %d FAIL (state=RUNNING, %s)",
                problem_name, t.number, age_str,
            )
            if not dry_run:
                storage_obj.set_trial_state_values(
                    t._trial_id, state=optuna.trial.TrialState.FAIL
                )
            marked += 1
        else:
            log.info(
                "%s: keeping trial %d RUNNING (age %.0fs < %.0fs threshold).",
                problem_name, t.number, age, max_age_s,
            )
            kept += 1
    return {"running": len(running), "marked_fail": marked, "kept": kept}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mark zombie RUNNING trials as FAIL after a worker crash."
    )
    p.add_argument(
        "--problems",
        required=True,
        help="Comma-separated problem names.",
    )
    p.add_argument(
        "--max-age-s",
        type=float,
        default=600.0,
        help=(
            "Only mark trials whose last activity is older than this many "
            "seconds. Default 600s == 2x the heartbeat grace period."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing to the DB.",
    )
    p.add_argument("--storage", default=None, help="Optuna storage URL override.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    totals = {"running": 0, "marked_fail": 0, "kept": 0}
    for name in [s.strip() for s in args.problems.split(",") if s.strip()]:
        counts = cleanup_problem(
            name,
            max_age_s=args.max_age_s,
            dry_run=args.dry_run,
            storage=args.storage,
        )
        log.info("%s: %s", name, counts)
        for k, v in counts.items():
            totals[k] += v
    log.info("Totals: %s%s", totals, " (DRY-RUN)" if args.dry_run else "")


if __name__ == "__main__":
    main()
