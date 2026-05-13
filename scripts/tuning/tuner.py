"""Optuna study factory + objective binding for the ASTROMoRF tuner.

Concurrency model
-----------------
This module assumes the storage layer (:mod:`scripts.tuning.storage`) is
already HPC-safe (default: JournalStorage). The tuner adds:

- **Per-worker sampler seed**. TPESampler is seeded with
  ``base_seed + worker_id * 7919``, so concurrent workers explore
  different regions of the search space rather than redundantly
  re-proposing the same points after the startup phase.
- **Constant liar** for parallel TPE so in-flight trials are accounted
  for when proposing the next point.
- **Cached narrowed-range snapshot**. The expensive ``study.get_trials``
  read used to drive adaptive narrowing is cached per worker and
  refreshed every ``NARROW_REFRESH_EVERY`` trials, not every trial.
- **Per-worker JSONL diagnostics**. Each worker writes to its own
  ``trials_jsonl/<problem>.w<worker_id>.jsonl`` so there is *zero*
  cross-process contention on the diagnostics log. The ``collect``
  module concatenates them at report time.

Resilience
----------
- :class:`RetryFailedTrialCallback` automatically resurrects trials
  whose worker crashed (e.g. SLURM timeout, OOM, segfault) — they are
  re-enqueued with the same params for the next available worker.
- The wall-clock callback ``_stop_walltime`` exits cleanly before SLURM
  kills the job, giving the in-flight trial a chance to finish.
- ``catch=(Exception,)`` ensures one bad trial cannot kill the worker.

Trial lifecycle
---------------
Each trial:
1. Samples hyperparameters (respecting any narrowed ranges from
   :mod:`spaces`).
2. Runs rung 0 (4 mreps), reports score, asks pruner.
3. If kept, runs rung 1 (8 fresh mreps), reports score, asks pruner.
4. If kept, runs rung 2 (16 fresh mreps), reports score; this score is
   what Optuna stores.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import optuna
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler

from . import results_root
from .evaluate import (
    DEFAULT_BUDGET,
    EvalResult,
    composite_score,
    evaluate_config,
)
from .spaces import (
    NARROW_AFTER_TRIALS,
    NARROW_EXPLORATION_PROB,
    CategoricalParam,
    FloatParam,
    IntParam,
    NarrowedRanges,
    SearchSpace,
    compute_narrowed_ranges,
    get_space,
    is_combinatorially_feasible,
    warmstart_params,
)
from .storage import (
    StorageSpec,
    create_or_load_study,
    make_storage,
    resolve_spec,
)

log = logging.getLogger(__name__)


# ── ASHA rung schedule ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class Rung:
    """One rung of the ASHA promotion ladder."""

    step: int
    n_macroreps: int
    n_postreps: int


DEFAULT_RUNGS: tuple[Rung, ...] = (
    Rung(step=0, n_macroreps=4, n_postreps=30),
    Rung(step=1, n_macroreps=8, n_postreps=50),
    Rung(step=2, n_macroreps=16, n_postreps=100),
)


# Refresh the cached narrowed-range snapshot every N completed trials.
# Higher = less storage traffic; lower = faster localisation. 10 is a
# good compromise for the 100-trial-per-worker regime.
NARROW_REFRESH_EVERY: int = 10

# Worker startup jitter (seconds). Workers sleep a random fraction of
# this window before opening storage so they don't slam the backend
# simultaneously when a SLURM array launches.
WORKER_STARTUP_JITTER_S: float = 15.0


# ── storage URL (back-compat shims) ────────────────────────────────────────


def storage_url_for(problem_name: str) -> str:
    """Return a *display* URL for *problem_name*.

    Kept for backward compatibility with downstream tools (``confirm``,
    ``collect``, ``cleanup``) that historically passed strings around.
    For *creating* storage objects, use :func:`make_storage` from
    :mod:`scripts.tuning.storage` instead.
    """
    return resolve_spec(problem_name).display


def study_name(problem_name: str) -> str:
    return f"astromorf_tune_{problem_name}_b{DEFAULT_BUDGET}"


def trials_jsonl_path(problem_name: str, worker_id: int | None = None) -> Path:
    """Per-worker JSONL diagnostics path.

    When ``worker_id is None`` we use the legacy single-file path. The
    new SLURM workers always pass a worker_id derived from
    ``$SLURM_ARRAY_TASK_ID`` (or PID if running locally).
    """
    base = results_root() / "trials_jsonl"
    base.mkdir(parents=True, exist_ok=True)
    if worker_id is None:
        return base / f"{problem_name}.jsonl"
    return base / f"{problem_name}.w{worker_id:04d}.jsonl"


def all_trials_jsonl_paths(problem_name: str) -> list[Path]:
    """Return every per-worker JSONL plus the legacy single-file path."""
    base = results_root() / "trials_jsonl"
    if not base.exists():
        return []
    out: list[Path] = []
    legacy = base / f"{problem_name}.jsonl"
    if legacy.exists():
        out.append(legacy)
    out.extend(sorted(base.glob(f"{problem_name}.w*.jsonl")))
    return out


class _JsonlWriter:
    """Persistent per-worker JSONL writer.

    Opens the worker's JSONL file once and keeps the handle alive for
    the lifetime of the worker process. Each write is a single line
    appended + flushed; we do NOT ``fsync`` because the file is private
    to this process and the SLURM cleanup hook will sync on exit.

    Compared to the legacy ``with path.open('a') as f: f.write(...)``
    pattern this saves one ``open()`` + ``close()`` per rung
    (3 syscalls saved per rung × 3 rungs × ~50 trials/worker × 8
    workers/problem = ~3.6k syscalls saved per problem-run on the
    shared HPC filesystem). The previous pattern also caused some
    NFS clients to flush their buffer on every close.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        # Line-buffered append so each ``\n`` flushes to the OS buffer
        # (but NOT to disk; that's fsync, which we deliberately skip).
        self._fh = path.open("a", encoding="utf-8", buffering=1)

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, default=_json_default, sort_keys=True)
        self._fh.write(line + "\n")

    def close(self) -> None:
        try:
            self._fh.flush()
        finally:
            self._fh.close()


def _json_default(o: Any) -> Any:
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
    except Exception:
        pass
    if hasattr(o, "value"):  # enums
        return o.value
    return repr(o)


# ── sampling helpers ──────────────────────────────────────────────────────


def _suggest_param(
    trial: optuna.Trial,
    param: FloatParam | IntParam | CategoricalParam,
    narrowed: NarrowedRanges,
    rng_state: dict[str, Any],
) -> Any:
    """Suggest a value for *param* via Optuna, applying narrowing."""
    explore = rng_state["rng"].random() < NARROW_EXPLORATION_PROB
    if isinstance(param, FloatParam):
        if not explore and param.name in narrowed.floats:
            lo, hi = narrowed.floats[param.name]
        else:
            lo, hi = param.low, param.high
        log_scale = param.scale == "log" and lo > 0
        return trial.suggest_float(param.name, lo, hi, log=log_scale)
    if isinstance(param, IntParam):
        if not explore and param.name in narrowed.ints:
            lo, hi = narrowed.ints[param.name]
        else:
            lo, hi = param.low, param.high
        log_scale = param.scale == "log" and lo > 0
        return trial.suggest_int(param.name, lo, hi, log=log_scale)
    # Categorical: always use the original choice set (Optuna forbids
    # dynamic categorical spaces).
    return trial.suggest_categorical(param.name, list(param.choices))


def _sample_trial_params(
    trial: optuna.Trial, space: SearchSpace, narrowed: NarrowedRanges
) -> dict[str, Any]:
    """Sample a full hyperparameter dict from *space* for one Optuna trial."""
    rng_state = {"rng": __import__("random").Random(trial.number * 9301 + 49297)}
    params: dict[str, Any] = {}
    for p in space.all_params():
        params[p.name] = _suggest_param(trial, p, narrowed, rng_state)

    feasible, q = is_combinatorially_feasible(
        int(params["subspace_dim"]), int(params["polynomial_degree"])
    )
    trial.set_user_attr("basis_terms", q)
    if not feasible:
        trial.set_user_attr("rejected_combinatorial", True)
        log.info(
            "Trial %d pruned: q=%d > cap (subspace=%s degree=%s)",
            trial.number, q, params["subspace_dim"], params["polynomial_degree"],
        )
        raise optuna.TrialPruned(
            f"basis terms q={q} exceeds combinatorial cap"
        )
    return params


# ── narrowing snapshot (cached) ───────────────────────────────────────────


class _NarrowingCache:
    """Per-worker cache to avoid hammering storage with full study reads.

    Refreshes every ``NARROW_REFRESH_EVERY`` trials. Each completed trial
    bumps the local counter; only when it crosses the refresh threshold
    do we re-read ``study.get_trials``. The cost of a stale window is at
    most ``NARROW_REFRESH_EVERY`` trials of suboptimal narrowing — far
    cheaper than a full storage read on every trial.
    """

    def __init__(self, space: SearchSpace) -> None:
        self.space = space
        self._snapshot: NarrowedRanges = NarrowedRanges(
            floats={}, ints={}, categoricals={}, n_trials_used=0
        )
        self._last_refresh_trial: int = -10**9
        self._last_refresh_wallclock: float = 0.0

    def get(self, trial: optuna.Trial) -> NarrowedRanges:
        n = trial.number
        if n - self._last_refresh_trial >= NARROW_REFRESH_EVERY:
            self._refresh(trial.study)
            self._last_refresh_trial = n
        return self._snapshot

    def _refresh(self, study: optuna.Study) -> None:
        completed = []
        for t in study.get_trials(
            deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)
        ):
            if t.value is None:
                continue
            completed.append({"params": dict(t.params), "value": float(t.value)})
        self._snapshot = compute_narrowed_ranges(self.space, completed)
        self._last_refresh_wallclock = time.time()


# ── objective ──────────────────────────────────────────────────────────────


def _build_objective(
    *,
    problem_name: str,
    space: SearchSpace,
    rungs: Iterable[Rung],
    budget: int,
    base_seed: int,
    std_weight: float,
    failure_penalty: float,
    per_trial_wall_clock_cap_s: float | None,
    n_jobs: int | str = 1,
    worker_id: int = 0,
):
    """Build the closure Optuna will call for each trial."""
    rungs_t = tuple(rungs)
    cache = _NarrowingCache(space)
    jsonl_path = trials_jsonl_path(problem_name, worker_id=worker_id)
    jsonl_writer = _JsonlWriter(jsonl_path)

    def objective(trial: optuna.Trial) -> float:
        narrowed = cache.get(trial)
        trial.set_user_attr("narrowed_n_trials", narrowed.n_trials_used)

        params = _sample_trial_params(trial, space, narrowed)
        trial.set_user_attr("hparams", params)
        trial.set_user_attr("worker_id", worker_id)

        composite_history: list[float] = []
        last_result: EvalResult | None = None
        for rung in rungs_t:
            t0 = time.perf_counter()
            result = evaluate_config(
                problem_name=problem_name,
                dimension=space.problem_dim if space.use_scaling else None,
                budget=budget,
                params=params,
                n_macroreps=rung.n_macroreps,
                base_seed=base_seed + rung.step * 1009,
                trial_number=trial.number,
                per_trial_wall_clock_cap_s=per_trial_wall_clock_cap_s,
                skip_combinatorial_check=True,
                n_jobs=n_jobs,
            )
            elapsed = time.perf_counter() - t0
            score = composite_score(
                result,
                n_macroreps_target=rung.n_macroreps,
                std_weight=std_weight,
                failure_penalty=failure_penalty,
            )
            composite_history.append(score)
            last_result = result

            jsonl_writer.write(
                {
                    "ts": time.time(),
                    "trial_number": trial.number,
                    "worker_id": worker_id,
                    "rung_step": rung.step,
                    "rung_n_macroreps": rung.n_macroreps,
                    "rung_wall_clock_s": elapsed,
                    "params": params,
                    "score": score,
                    "result": result.to_record(),
                }
            )

            trial.report(score, step=rung.step)
            if trial.should_prune() and rung.step < rungs_t[-1].step:
                trial.set_user_attr("pruned_at_rung", rung.step)
                trial.set_user_attr("scores_per_rung", composite_history)
                raise optuna.TrialPruned()

        if last_result is None:
            return float("inf")
        trial.set_user_attr("scores_per_rung", composite_history)
        trial.set_user_attr("final_mean_objective", last_result.mean_objective)
        trial.set_user_attr("final_std_objective", last_result.std_objective)
        trial.set_user_attr("final_mean_aligned", last_result.mean_aligned)
        trial.set_user_attr("final_std_aligned", last_result.std_aligned)
        trial.set_user_attr("failure_count", last_result.failure_count)
        trial.set_user_attr("diagnostics", last_result.diagnostics_aggregate)
        return composite_history[-1]

    # Attach the writer to the closure so the caller can close it on
    # shutdown (the open file handle would otherwise leak until process
    # exit — harmless, but ugly).
    objective._jsonl_writer = jsonl_writer  # type: ignore[attr-defined]
    return objective


# ── study factory ─────────────────────────────────────────────────────────


def build_study(
    problem_name: str,
    *,
    storage: str | None = None,
    seed: int = 20250428,
    worker_id: int = 0,
    enqueue_warmstart: bool = True,
    n_startup_trials: int = 24,
) -> tuple[optuna.Study, StorageSpec]:
    """Create-or-load the per-problem study (HPC-safe).

    Returns ``(study, spec)``. ``spec`` describes the resolved backend
    so the caller can log which storage was actually used.

    Concurrency
    -----------
    - Storage is constructed via :func:`scripts.tuning.storage.make_storage`
      (default: JournalStorage, no DDL races).
    - The create-or-load operation is wrapped in a filesystem lockfile and
      a transient-error retry loop (see
      :func:`scripts.tuning.storage.create_or_load_study`).
    - Sampler seed is per-worker so concurrent workers diverge instead of
      proposing the same points.
    """
    storage_obj, spec = make_storage(problem_name, storage_url=storage)

    sampler = TPESampler(
        seed=seed + worker_id * 7919,
        multivariate=True,
        group=True,
        n_startup_trials=n_startup_trials,
        constant_liar=True,
    )
    pruner = SuccessiveHalvingPruner(
        min_resource=1, reduction_factor=3, min_early_stopping_rate=0
    )

    study = create_or_load_study(
        study_name=study_name(problem_name),
        storage=storage_obj,
        sampler=sampler,
        pruner=pruner,
        direction="minimize",
        problem_name=problem_name,
    )

    # Direction metadata (idempotent — only set once).
    if "raw_direction" not in study.user_attrs:
        try:
            raw_sign = _probe_minmax_sign(problem_name)
            study.set_user_attr(
                "raw_direction", "maximize" if raw_sign > 0 else "minimize"
            )
            study.set_user_attr("raw_minmax_sign", int(raw_sign))
            log.info(
                "Study %s: raw_direction=%s (minmax_sign=%d). Optuna minimises "
                "the *aligned* score (= -minmax_sign * raw_obj).",
                problem_name,
                "maximize" if raw_sign > 0 else "minimize",
                raw_sign,
            )
        except Exception as exc:  # never let metadata block init
            log.warning("Could not probe minmax for %s: %s", problem_name, exc)

    # Warm-start: enqueue the previously-known-best params once. Concurrent
    # workers race here, but ``enqueue_trial(skip_if_exists=True)`` is
    # idempotent and ``study.user_attrs`` is last-write-wins so the worst
    # case is the warm-start trial is enqueued more than once — which the
    # ``skip_if_exists`` check still suppresses.
    if enqueue_warmstart and study.user_attrs.get("warmstart_enqueued") is not True:
        try:
            ws = warmstart_params(problem_name)
            study.enqueue_trial(ws, skip_if_exists=True)
            study.set_user_attr("warmstart_enqueued", True)
            study.set_user_attr("warmstart_params", ws)
            log.info("Warm-start enqueued for %s: %s", problem_name, ws)
        except KeyError:
            log.info("No warm-start defined for %s", problem_name)
        except Exception as exc:
            log.warning("Warm-start enqueue raced or failed for %s: %s",
                        problem_name, exc)
    return study, spec


def _probe_minmax_sign(problem_name: str) -> int:
    """Return +1 (maximise) or -1 (minimise) for *problem_name*."""
    from scripts.tuning.spaces import get_space
    from simopt.experiment.dimension_scaling import scale_dimension
    from simopt.experiment_base import instantiate_problem

    space = get_space(problem_name)
    if space.use_scaling:
        problem = scale_dimension(
            problem_name, budget=DEFAULT_BUDGET, dimension=space.problem_dim
        )
    else:
        problem = instantiate_problem(
            problem_name=problem_name,
            problem_fixed_factors={"budget": DEFAULT_BUDGET},
        )
    return int(problem.minmax[0])


# ── public entry: run a worker ─────────────────────────────────────────────


def run_worker(
    *,
    problem_name: str,
    n_trials: int,
    storage: str | None = None,
    seed: int = 20250428,
    worker_id: int = 0,
    budget: int = DEFAULT_BUDGET,
    rungs: Iterable[Rung] = DEFAULT_RUNGS,
    std_weight: float = 0.15,
    failure_penalty: float = 1e6,
    per_trial_wall_clock_cap_s: float | None = 30 * 60,
    walltime_budget_s: float | None = None,
    n_jobs: int | str = 1,
    startup_jitter_s: float = WORKER_STARTUP_JITTER_S,
) -> int:
    """Drive ``study.optimize`` for at most *n_trials* (or until walltime).

    Returns the number of trials this worker actually completed.
    """
    if startup_jitter_s > 0:
        import random
        jitter = random.uniform(0, startup_jitter_s)
        log.info("Worker %d sleeping %.2fs to spread storage startup load.",
                 worker_id, jitter)
        time.sleep(jitter)

    space = get_space(problem_name)
    study, spec = build_study(
        problem_name, storage=storage, seed=seed, worker_id=worker_id
    )
    log.info("Worker %d using storage backend=%s (%s) for study %s",
             worker_id, spec.backend, spec.display, study_name(problem_name))

    objective = _build_objective(
        problem_name=problem_name,
        space=space,
        rungs=rungs,
        budget=budget,
        base_seed=seed,
        std_weight=std_weight,
        failure_penalty=failure_penalty,
        per_trial_wall_clock_cap_s=per_trial_wall_clock_cap_s,
        n_jobs=n_jobs,
        worker_id=worker_id,
    )

    deadline = (time.time() + walltime_budget_s) if walltime_budget_s else None

    def _stop_walltime(study, trial):
        if deadline is not None and time.time() >= deadline:
            log.info("Worker %d: walltime budget exhausted; stopping study.",
                     worker_id)
            study.stop()

    callbacks = [_stop_walltime] if deadline else []
    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            callbacks=callbacks,
            gc_after_trial=False,
            catch=(Exception,),
        )
    finally:
        # Close the JSONL handle so its buffer is flushed before exit.
        writer = getattr(objective, "_jsonl_writer", None)
        if writer is not None:
            try:
                writer.close()
            except Exception:  # noqa: BLE001
                pass
        n_done = sum(
            1
            for t in study.get_trials(deepcopy=False)
            if t.state
            in (
                optuna.trial.TrialState.COMPLETE,
                optuna.trial.TrialState.PRUNED,
                optuna.trial.TrialState.FAIL,
            )
        )
    return n_done
