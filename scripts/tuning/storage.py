"""HPC-safe Optuna storage layer for ASTROMoRF tuning.

This module is the *single* place that decides which storage backend the
tuner uses and how to construct it safely under heavy parallelism. All
other modules (``tuner``, ``worker``, ``confirm``, ``collect``,
``cleanup``, ``export_dashboard_db``) MUST go through :func:`make_storage`.

Architecture
------------
**JournalStorage is the canonical primary backend.** It is the default,
the recommended path, the only backend assumed by the SLURM scripts, and
the only backend our HPC workflow targets. The remaining backends exist
to support optional dashboards and edge-case fallbacks; they are NOT
required to run the pipeline.

Backends
--------
``journal`` (DEFAULT, PRIMARY)
    Optuna's :class:`JournalStorage` backed by ``JournalFileBackend`` with
    ``JournalFileSymlinkLock``. This is an append-only log of operations
    coordinated by an ``os.symlink``-based filesystem mutex. Properties:

    - No DDL. No "table already exists". No ``database is locked``.
    - Safe on local FS *and* on shared NFS — ``os.symlink`` is atomic on
      NFSv3+, which is exactly the primitive Optuna's
      ``JournalFileSymlinkLock`` relies on.
    - Append-only journal entries; readers replay the log and never
      block writers (they don't take the symlink lock).
    - Stale-lock recovery: stale symlinks left over from a crashed
      writer are reclaimed after a TTL (handled by Optuna).
    - Crash consistency: the journal is sync-friendly; a partially-
      written final entry is discarded on next load.

    Lock-contention scaling: bounded by symlink-acquire latency
    (typically 1-20ms on local FS, 10-100ms on NFS). With ~10 journal
    ops/trial and ~400 trials/problem distributed over 12h across 8
    workers, that is ~5 ops/minute — orders of magnitude below the
    bottleneck threshold.

``sqlite`` (EXPORT / FALLBACK ONLY)
    :class:`RDBStorage` against a local SQLite file with WAL mode,
    ``busy_timeout=60000ms``, and a transient-error retry middleware.

    **Not used as a live distributed backend.** Two legitimate uses:

    1. Dashboard snapshots produced by
       :mod:`scripts.tuning.export_dashboard_db` (one writer, many
       readers — the dashboard).
    2. Single-node debugging / smoke tests when you don't want the
       journal directory to accumulate.

    Never set ``ASTROMORF_OPTUNA_STORAGE`` to ``sqlite://`` for a real
    SLURM array run. The pipeline will technically work, but you'll hit
    the limitations documented in the SQLite manual.

``postgres`` (OPTIONAL ADVANCED)
    :class:`RDBStorage` against a Postgres DSN. Supported but never
    required. Only useful if you happen to have a Postgres instance
    accessible from compute nodes AND want >16 concurrent workers per
    problem. For typical SLURM array tuning, Journal is faster to set up
    and equally robust.

Resolving the backend
---------------------
The storage URL is taken from (in priority order):

1. The ``--storage`` CLI flag (workers / confirm / collect / cleanup).
2. The ``ASTROMORF_OPTUNA_STORAGE`` env var.
3. Default: journal under ``$ASTROMORF_TUNING_DIR/journal/<problem>.log``.

URL syntax
~~~~~~~~~~
- ``journal:///abs/path/to/dir`` (one file per problem will live under it)
- A bare directory path (no ``://``) is treated as a journal directory.
- ``sqlite:///abs/path.db`` (snapshots / fallback only)
- ``postgresql+psycopg2://USER:PW@HOST/DB`` (optional advanced)

Atomic study creation
---------------------
:func:`create_or_load_study` is the ONLY function that creates studies.
It guards study creation with a filesystem lockfile (separate from the
JournalStorage symlink lock) so that the warm-start enqueue runs exactly
once even under a thundering herd of array workers.

NFS / shared filesystem behaviour
---------------------------------
JournalFileSymlinkLock uses ``os.symlink`` to acquire the lock, which is
atomic under POSIX and on NFSv3+. The journal file itself is opened in
``O_APPEND`` mode; appends shorter than ``PIPE_BUF`` are atomic on local
FS, and Optuna writes one journal entry per ``write()`` call. On NFS,
appends within a single client are atomic; across clients they remain
serialised by the symlink lock that surrounds every write batch.

For very wobbly NFS exports (e.g. NFSv2 or aggressive client caches),
consider pointing the journal at a per-node scratch directory and using
the ``export_dashboard_db`` snapshot pipeline to consolidate. This is
rare on modern HPC.
"""

from __future__ import annotations

import contextlib
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import optuna
from optuna.storages import BaseStorage

from . import results_root

log = logging.getLogger(__name__)


# ── tunables ──────────────────────────────────────────────────────────────

# Default SQLite busy_timeout (ms). Optuna respects ``engine_kwargs`` so
# we forward ``connect_args={"timeout": ...}`` to the sqlite3 driver.
SQLITE_BUSY_TIMEOUT_MS: int = 60_000

# Postgres engine pool settings. Workers are mostly idle between trials,
# so we recycle connections aggressively to dodge server idle drops.
POSTGRES_POOL_RECYCLE_S: int = 1_800
POSTGRES_POOL_SIZE: int = 4

# Maximum retries on transient storage errors during study creation.
CREATE_MAX_RETRIES: int = 24

# Heartbeat tuning. Trials whose worker dies are marked FAIL after
# ``grace_period`` seconds without a heartbeat. With 60s/300s a fast
# worker rotation costs us at most 5 minutes of "phantom RUNNING" state.
HEARTBEAT_INTERVAL_S: int = 60
HEARTBEAT_GRACE_PERIOD_S: int = 300


# ── backend enum ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StorageSpec:
    """Resolved backend descriptor."""

    backend: str   # "journal" | "postgres" | "sqlite"
    location: str  # journal: dir path; postgres/sqlite: full DSN
    display: str   # human-readable "journal:///path" / "postgresql+..."


def _journal_dir_default() -> Path:
    return results_root() / "journal"


def _sqlite_dir_default() -> Path:
    return results_root() / "studies"


def journal_path_for(problem_name: str) -> Path:
    """Return the canonical journal log file for *problem_name*.

    This is the single source of truth that every caller (worker,
    exporter, dashboard launcher, cleanup) uses to find a problem's
    journal. Honours :envvar:`ASTROMORF_OPTUNA_STORAGE` if it points at
    a journal directory; otherwise uses the default tuning dir.
    """
    spec = resolve_spec(problem_name)
    if spec.backend == "journal":
        return Path(spec.location) / f"{problem_name}.log"
    # If the user has chosen a non-journal backend, still expose where
    # *the journal would live* — useful for the dashboard exporter when
    # the user explicitly asks for a journal source.
    return _journal_dir_default() / f"{problem_name}.log"


def dashboard_db_path_for(problem_name: str) -> Path:
    """Canonical SQLite snapshot path used by the dashboard."""
    d = results_root() / "dashboard"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{problem_name}.db"


def resolve_spec(
    problem_name: str,
    *,
    storage_url: str | None = None,
) -> StorageSpec:
    """Resolve the (backend, location) the tuner should use for *problem_name*.

    Priority: explicit ``storage_url`` > ``ASTROMORF_OPTUNA_STORAGE`` >
    journal default.
    """
    raw = storage_url or os.environ.get("ASTROMORF_OPTUNA_STORAGE")
    if raw:
        raw = raw.strip()

    if not raw:
        d = _journal_dir_default()
        d.mkdir(parents=True, exist_ok=True)
        return StorageSpec(
            backend="journal",
            location=str(d),
            display=f"journal://{d}",
        )

    if raw.startswith("postgresql"):
        return StorageSpec(backend="postgres", location=raw, display=raw)

    if raw.startswith("sqlite://"):
        return StorageSpec(backend="sqlite", location=raw, display=raw)

    if raw.startswith("journal://"):
        d = Path(raw[len("journal://"):])
        d.mkdir(parents=True, exist_ok=True)
        return StorageSpec(
            backend="journal",
            location=str(d),
            display=f"journal://{d}",
        )

    # Bare path -> treat as journal directory (the HPC-safe default).
    if "://" not in raw:
        d = Path(raw)
        d.mkdir(parents=True, exist_ok=True)
        return StorageSpec(
            backend="journal",
            location=str(d),
            display=f"journal://{d}",
        )

    # Unknown DSN scheme: hand straight to Optuna and hope for the best.
    log.warning("Unrecognised storage URL %r; passing to Optuna as-is.", raw)
    return StorageSpec(backend="other", location=raw, display=raw)


# ── storage construction ──────────────────────────────────────────────────


def _make_journal_storage(spec: StorageSpec, problem_name: str) -> BaseStorage:
    """Construct a JournalStorage with NFS-safe symlink locking."""
    from optuna.storages import JournalStorage
    # File-backend API moved between Optuna releases; try the new path first.
    try:
        from optuna.storages.journal import (  # type: ignore[attr-defined]
            JournalFileBackend,
            JournalFileSymlinkLock,
        )
    except ImportError:  # pragma: no cover - older Optuna
        from optuna.storages import (  # type: ignore[attr-defined]
            JournalFileStorage as JournalFileBackend,
            JournalFileSymlinkLock,
        )

    journal_path = Path(spec.location) / f"{problem_name}.log"
    journal_path.parent.mkdir(parents=True, exist_ok=True)

    lock_obj = JournalFileSymlinkLock(str(journal_path))
    try:
        backend = JournalFileBackend(str(journal_path), lock_obj=lock_obj)
    except TypeError:
        # Older signature: lock is configured separately.
        backend = JournalFileBackend(str(journal_path))
        with contextlib.suppress(AttributeError):
            backend.lock_obj = lock_obj  # type: ignore[attr-defined]
    return JournalStorage(backend)


def _make_rdb_storage(spec: StorageSpec) -> BaseStorage:
    """Construct an RDBStorage (postgres or sqlite) with safe engine kwargs."""
    engine_kwargs: dict[str, Any] = {}
    if spec.backend == "postgres":
        engine_kwargs = {
            "pool_pre_ping": True,
            "pool_recycle": POSTGRES_POOL_RECYCLE_S,
            "pool_size": POSTGRES_POOL_SIZE,
            "max_overflow": POSTGRES_POOL_SIZE,
            "connect_args": {"connect_timeout": 30},
        }
    elif spec.backend == "sqlite":
        # WAL + busy_timeout. WAL is set via PRAGMA on the first connect;
        # we use SQLAlchemy's "connect" event for that below.
        engine_kwargs = {
            "connect_args": {
                "timeout": SQLITE_BUSY_TIMEOUT_MS / 1000.0,
                "check_same_thread": False,
            }
        }

    storage = optuna.storages.RDBStorage(
        url=spec.location,
        engine_kwargs=engine_kwargs,
        heartbeat_interval=HEARTBEAT_INTERVAL_S,
        grace_period=HEARTBEAT_GRACE_PERIOD_S,
        failed_trial_callback=optuna.storages.RetryFailedTrialCallback(
            max_retry=3, inherit_intermediate_values=False
        ),
    )

    if spec.backend == "sqlite":
        _apply_sqlite_wal(storage)
    return storage


def _apply_sqlite_wal(storage: optuna.storages.RDBStorage) -> None:
    """Enable WAL + busy_timeout on every SQLite connection."""
    try:
        from sqlalchemy import event
    except ImportError:  # pragma: no cover
        return
    engine = storage.engine  # type: ignore[attr-defined]

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, _):  # noqa: ANN001
        try:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.close()
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not enable SQLite WAL: %s", exc)


def make_storage(
    problem_name: str, *, storage_url: str | None = None
) -> tuple[BaseStorage, StorageSpec]:
    """Construct an Optuna storage object for *problem_name*.

    Returns ``(storage, spec)``. ``spec`` is useful for logging /
    diagnostics so the caller can record which backend was actually used.
    """
    spec = resolve_spec(problem_name, storage_url=storage_url)
    if spec.backend == "journal":
        return _make_journal_storage(spec, problem_name), spec
    if spec.backend in ("postgres", "sqlite"):
        return _make_rdb_storage(spec), spec
    # Unknown DSN: let Optuna's factory decide.
    return optuna.storages.get_storage(spec.location), spec


# ── filesystem lockfile (atomic study creation) ───────────────────────────


@contextlib.contextmanager
def _study_create_lock(problem_name: str, timeout_s: float = 120.0) -> Iterator[None]:
    """Filesystem mutex guarding the very first study creation.

    JournalStorage's create_study is already idempotent under concurrent
    callers, but we still serialise to keep the warm-start enqueue
    deterministic. Uses ``os.O_EXCL`` for an NFS-safe lockfile; backs off
    on collision; auto-released on context exit.
    """
    lockdir = results_root() / ".locks"
    lockdir.mkdir(parents=True, exist_ok=True)
    lockfile = lockdir / f"{problem_name}.create.lock"

    deadline = time.time() + timeout_s
    fd: int | None = None
    while True:
        try:
            fd = os.open(str(lockfile), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.write(fd, f"pid={os.getpid()} t={time.time():.3f}\n".encode())
            break
        except FileExistsError:
            # Stale lock detection: if older than 5 min, remove and retry.
            try:
                age = time.time() - lockfile.stat().st_mtime
            except FileNotFoundError:
                continue
            if age > 300:
                log.warning("Removing stale create-lock for %s (age=%.0fs).",
                            problem_name, age)
                with contextlib.suppress(FileNotFoundError):
                    lockfile.unlink()
                continue
            if time.time() >= deadline:
                raise TimeoutError(
                    f"Could not acquire study create lock for {problem_name} "
                    f"within {timeout_s}s"
                )
            time.sleep(0.5 + random.uniform(0, 0.5))
    try:
        yield
    finally:
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)
        with contextlib.suppress(FileNotFoundError):
            lockfile.unlink()


def create_or_load_study(
    *,
    study_name: str,
    storage: BaseStorage,
    sampler: optuna.samplers.BaseSampler,
    pruner: optuna.pruners.BasePruner,
    direction: str = "minimize",
    problem_name: str | None = None,
) -> optuna.Study:
    """Atomic, retry-safe ``create_study(load_if_exists=True)``.

    Wraps the operation in a filesystem lock and a retry loop. Returns the
    loaded study; warm-start enqueue is the *caller's* responsibility (we
    don't want the storage layer to know about that).
    """
    import sqlalchemy.exc

    backoff_s = 0.5
    last_exc: Exception | None = None
    lock_key = problem_name or study_name

    for attempt in range(CREATE_MAX_RETRIES):
        try:
            with _study_create_lock(lock_key):
                return optuna.create_study(
                    study_name=study_name,
                    storage=storage,
                    sampler=sampler,
                    pruner=pruner,
                    direction=direction,
                    load_if_exists=True,
                )
        except (sqlalchemy.exc.OperationalError,
                sqlalchemy.exc.IntegrityError) as exc:
            msg = str(exc).lower()
            transient = (
                "already exists" in msg
                or "database is locked" in msg
                or "unique constraint" in msg
                or "deadlock detected" in msg
            )
            if not transient:
                raise
            last_exc = exc
        except Exception as exc:  # JournalStorage uses its own exceptions
            last_exc = exc
            if attempt == CREATE_MAX_RETRIES - 1:
                raise
        sleep_s = min(backoff_s * (1.5 ** attempt) + random.uniform(0, 0.5), 15.0)
        log.warning(
            "create_or_load_study(%s) attempt %d/%d transient: %s; retry in %.2fs",
            study_name, attempt + 1, CREATE_MAX_RETRIES,
            type(last_exc).__name__ if last_exc else "?", sleep_s,
        )
        time.sleep(sleep_s)
    raise RuntimeError(
        f"create_or_load_study({study_name!r}) failed after "
        f"{CREATE_MAX_RETRIES} retries"
    ) from last_exc
