"""Export an Optuna JournalStorage study to a SQLite snapshot for optuna-dashboard.

Why
---
The canonical tuning backend is :class:`optuna.storages.JournalStorage`
(see :mod:`scripts.tuning.storage`). ``optuna-dashboard`` historically
ships with rock-solid SQLite/Postgres support; its Journal support has
matured but the SQLite snapshot path is what we recommend for HPC because:

- The dashboard process never touches the live journal — zero risk of
  interfering with running workers.
- SQLite snapshots are atomic (we rename a ``.tmp`` over the target),
  so the dashboard never sees a half-written file.
- The snapshot is a self-contained file you can ``scp`` to a laptop and
  open offline, without any of the journal-backend dependencies.

How
---
We use :func:`optuna.copy_study` to copy every study (one per problem)
from the journal into a fresh SQLite file. The copy is performed in a
temp file, then atomically renamed over the live snapshot. Optuna
``load_study`` against the journal is read-only-effective — it replays
the log into in-memory state and never takes the symlink lock, so the
copy is safe to run concurrently with live tuning workers.

Usage
-----
::

    # One-shot snapshot for one problem.
    python -m scripts.tuning.export_dashboard_db --problem SAN-1

    # All five problems.
    python -m scripts.tuning.export_dashboard_db --all

    # Refresh every 60 seconds (used by launch_dashboard.sh).
    python -m scripts.tuning.export_dashboard_db --all --watch 60

    # Custom output path.
    python -m scripts.tuning.export_dashboard_db --problem SAN-1 \\
        --out /tmp/dashboard_SAN-1.db
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path

# BLAS caps (the exporter does very little NumPy work, but be safe).
for _var, _val in (
    ("OMP_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
):
    os.environ.setdefault(_var, _val)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import optuna  # noqa: E402

from scripts.tuning import results_root  # noqa: E402
from scripts.tuning.storage import (  # noqa: E402
    dashboard_db_path_for,
    journal_path_for,
    make_storage,
)
from scripts.tuning.tuner import study_name  # noqa: E402


def combined_db_path() -> Path:
    """Canonical path for the combined all-problems snapshot."""
    d = results_root() / "dashboard"
    d.mkdir(parents=True, exist_ok=True)
    return d / "all_studies.db"

log = logging.getLogger("astromorf.tuning.export_dashboard_db")


DEFAULT_PROBLEMS: tuple[str, ...] = (
    "DYNAMNEWS-1", "SAN-1", "NETWORK-1", "ROSENBROCK-1", "PARAMESTI-1",
)


def _new_sqlite_storage(db_path: Path) -> optuna.storages.RDBStorage:
    """Build a fresh-file SQLite storage at *db_path*.

    The snapshot is single-writer (this process) and many-reader (the
    dashboard), so we don't need WAL, but we set ``busy_timeout`` to be
    safe in case the dashboard happens to checkpoint a long read.
    """
    return optuna.storages.RDBStorage(
        url=f"sqlite:///{db_path}",
        engine_kwargs={
            "connect_args": {"timeout": 30, "check_same_thread": False}
        },
    )


def export_one(
    problem_name: str,
    *,
    out_path: Path | None = None,
    source_storage_url: str | None = None,
) -> Path:
    """Snapshot one problem's study from Journal -> SQLite atomically.

    Returns the path of the final SQLite file.
    """
    src_storage, src_spec = make_storage(
        problem_name, storage_url=source_storage_url
    )
    src_study_name = study_name(problem_name)
    out_path = out_path or dashboard_db_path_for(problem_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Sanity: confirm the source study exists. If not, skip cleanly so
    # ``--all`` doesn't fail on problems whose study hasn't started yet.
    try:
        optuna.load_study(study_name=src_study_name, storage=src_storage)
    except (KeyError, ValueError) as exc:
        log.info("Skipping %s: source study not found yet (%s).",
                 problem_name, exc)
        return out_path

    # Atomic rename pattern: write to .tmp, then ``os.replace`` over the
    # live snapshot. ``os.replace`` is POSIX-atomic so the dashboard
    # never sees a partial file. We don't truncate the live file first.
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    dst_storage = _new_sqlite_storage(tmp_path)

    # ``copy_study`` enumerates trials in the source and re-inserts them
    # into the destination, preserving params / user_attrs / values /
    # state. It is read-only on the source.
    t0 = time.perf_counter()
    optuna.copy_study(
        from_study_name=src_study_name,
        from_storage=src_storage,
        to_storage=dst_storage,
        to_study_name=src_study_name,
    )

    # Close the SQLAlchemy engine so the tmp file isn't held open when
    # we rename it. SQLAlchemy doesn't lock the file post-commit, but
    # being explicit avoids platform surprises.
    try:
        dst_storage.engine.dispose()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    os.replace(tmp_path, out_path)
    elapsed = time.perf_counter() - t0
    log.info(
        "Snapshot %s: %s -> %s (%.2fs, %.1f KiB).",
        problem_name, src_spec.display, out_path,
        elapsed, out_path.stat().st_size / 1024,
    )
    return out_path


def export_all(
    problems: list[str],
    *,
    out_dir: Path | None = None,
    source_storage_url: str | None = None,
) -> list[Path]:
    """Snapshot each problem to its OWN SQLite file (per-problem mode)."""
    out_paths: list[Path] = []
    for prob in problems:
        out = (out_dir / f"{prob}.db") if out_dir else None
        try:
            out_paths.append(export_one(
                prob, out_path=out, source_storage_url=source_storage_url
            ))
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to export %s: %s", prob, exc)
    return out_paths


def export_combined(
    problems: list[str],
    *,
    out_path: Path | None = None,
    source_storage_url: str | None = None,
) -> Path:
    """Snapshot every problem into a SINGLE SQLite file.

    optuna-dashboard can only point at one storage URL, but a SQLite
    storage can hold many studies (distinguished by ``study_name``).
    Bundling all problems into one file makes the dashboard's sidebar
    show every problem at once.

    Implementation
    --------------
    1. Pick the destination path (default: ``dashboard/all_studies.db``).
    2. Write into ``<path>.tmp`` so the live snapshot is never touched
       until the very end.
    3. For each problem, ``optuna.copy_study`` from the journal into the
       tmp SQLite. Each study lives under its original
       ``astromorf_tune_<problem>_b<budget>`` name so the dashboard
       shows the same labels users see in CLI tools.
    4. ``os.replace`` the tmp file over the live snapshot. POSIX-atomic;
       the dashboard never sees a half-written file.

    A problem whose study hasn't been created yet (e.g. you launched
    the dashboard before the first worker booted) is silently skipped.
    """
    out_path = out_path or combined_db_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    dst_storage = _new_sqlite_storage(tmp_path)

    t0 = time.perf_counter()
    n_studies = 0
    for prob in problems:
        try:
            src_storage, src_spec = make_storage(
                prob, storage_url=source_storage_url
            )
            sname = study_name(prob)
            try:
                optuna.load_study(study_name=sname, storage=src_storage)
            except (KeyError, ValueError):
                log.info("Skipping %s: source study not found yet.", prob)
                continue
            optuna.copy_study(
                from_study_name=sname,
                from_storage=src_storage,
                to_storage=dst_storage,
                to_study_name=sname,
            )
            n_studies += 1
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to add %s to combined snapshot: %s", prob, exc)

    try:
        dst_storage.engine.dispose()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    if n_studies == 0:
        # Don't replace the existing snapshot with an empty one — that
        # would briefly make the dashboard show "no studies" while
        # tuning is still warming up.
        tmp_path.unlink(missing_ok=True)
        log.info("No studies ready yet; leaving existing snapshot untouched.")
        return out_path

    os.replace(tmp_path, out_path)
    elapsed = time.perf_counter() - t0
    log.info(
        "Combined snapshot: %d studies -> %s (%.2fs, %.1f KiB).",
        n_studies, out_path, elapsed, out_path.stat().st_size / 1024,
    )
    return out_path


def _watch_loop(
    problems: list[str],
    *,
    interval_s: float,
    out_dir: Path | None,
    out_path: Path | None,
    combined: bool,
    source_storage_url: str | None,
) -> None:
    """Refresh snapshots every *interval_s* seconds until SIGINT.

    The loop sleeps in short slices so Ctrl-C is responsive even with a
    long interval.
    """
    mode = "combined" if combined else "per-problem"
    log.info("Watch mode (%s): refreshing every %.0fs. Ctrl-C to stop.",
             mode, interval_s)
    while True:
        t0 = time.time()
        if combined:
            export_combined(
                problems, out_path=out_path,
                source_storage_url=source_storage_url,
            )
        else:
            export_all(
                problems, out_dir=out_dir,
                source_storage_url=source_storage_url,
            )
        elapsed = time.time() - t0
        sleep_left = max(0.0, interval_s - elapsed)
        # 1s sleep slices to keep KeyboardInterrupt responsive.
        while sleep_left > 0:
            chunk = min(1.0, sleep_left)
            time.sleep(chunk)
            sleep_left -= chunk


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export Optuna JournalStorage studies to SQLite for the dashboard."
    )
    # The exporter accepts one of: single problem, --all, or an explicit
    # comma-separated list via --problems. Make the mutually-exclusive
    # group optional so callers may also rely on defaults.
    g = p.add_mutually_exclusive_group(required=False)
    g.add_argument("--problem", help="Single problem name to export.")
    g.add_argument("--all", action="store_true",
                   help="Export every known problem (DYNAMNEWS-1, SAN-1, ...).")
    g.add_argument(
        "--problems",
        help="Comma-separated problem list (alternative to --all / --problem).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output SQLite file path. With --problem, this is the single "
            "per-problem snapshot. With --all in combined mode (default), "
            "this is the combined all-studies file "
            "(default: results/astromorf_tuning/dashboard/all_studies.db)."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Directory for per-problem snapshot files when --per-problem is "
            "used (default: results/astromorf_tuning/dashboard/)."
        ),
    )
    p.add_argument(
        "--per-problem",
        action="store_true",
        help=(
            "Write one SQLite file PER problem (legacy mode). Default is "
            "to bundle every problem into a single all_studies.db so the "
            "dashboard shows them all at once."
        ),
    )
    p.add_argument(
        "--journal", "--storage",
        dest="storage",
        default=None,
        help=(
            "Override source storage URL (default: ASTROMORF_OPTUNA_STORAGE "
            "env or the default journal location)."
        ),
    )
    p.add_argument(
        "--watch",
        type=float,
        default=None,
        help=(
            "If set, run the exporter in a loop, refreshing every N seconds. "
            "Used by launch_dashboard.sh to keep the snapshot fresh."
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

    if args.problem:
        problems = [args.problem]
    elif args.problems:
        problems = [s.strip() for s in args.problems.split(",") if s.strip()]
    else:
        problems = list(DEFAULT_PROBLEMS)

    # Single-problem -> one file. Multi-problem -> combined by default,
    # one-file-per-problem only if --per-problem is set.
    single = bool(args.problem) and len(problems) == 1 and not args.all
    combined = not single and not args.per_problem

    if args.watch is not None:
        _watch_loop(
            problems,
            interval_s=float(args.watch),
            out_dir=args.out_dir,
            out_path=args.out,
            combined=combined,
            source_storage_url=args.storage,
        )
        return

    if single:
        export_one(
            args.problem,
            out_path=args.out,
            source_storage_url=args.storage,
        )
    elif combined:
        export_combined(
            problems,
            out_path=args.out,
            source_storage_url=args.storage,
        )
    else:
        export_all(
            problems,
            out_dir=args.out_dir,
            source_storage_url=args.storage,
        )


if __name__ == "__main__":
    main()
