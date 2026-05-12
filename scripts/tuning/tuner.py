"""Optuna study factory + objective binding for the ASTROMoRF tuner.

Storage abstraction
-------------------
Default: SQLite at ``results/astromorf_tuning/studies/<problem>.db``.
Override globally via ``ASTROMORF_OPTUNA_STORAGE`` (any URL Optuna
understands, e.g. ``postgresql+psycopg2://user:pw@host/db``).

Sampler / pruner
----------------
- :class:`optuna.samplers.TPESampler` (multivariate=True, group=True) so
  trials are aware of param dependencies.
- :class:`optuna.pruners.SuccessiveHalvingPruner` (ASHA) with three rungs
  matching the macroreplication schedule. The pruner is fed
  ``trial.report(score, step=rung)`` after each rung.

Trial lifecycle
---------------
Each trial:
1. Samples hyperparameters (respecting any narrowed ranges from
   :mod:`spaces`).
2. Runs rung 0 (4 mreps), reports score, asks pruner.
3. If kept, runs rung 1 (8 fresh mreps), reports score, asks pruner.
4. If kept, runs rung 2 (16 fresh mreps), reports score; this score is
   what Optuna stores.

Each completed trial writes a JSON snapshot under ``trials_jsonl/`` with
the full diagnostics + raw objectives (the SQLite study only stores the
scalar objective + user_attrs, not the long lists). ``trials_jsonl/`` is
the source of truth for the report writer.
"""

from __future__ import annotations

import json
import logging
import os
import threading
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


# ── storage ────────────────────────────────────────────────────────────────


def storage_url_for(problem_name: str) -> str:
    """Return the Optuna storage URL for *problem_name*.

    Honours the ``ASTROMORF_OPTUNA_STORAGE`` env var (taken verbatim if
    it appears to be a full DSN, otherwise treated as a directory). Falls
    back to a per-problem SQLite file under ``results_root()``.
    """
    override = os.environ.get("ASTROMORF_OPTUNA_STORAGE")
    if override:
        if "://" in override:
            # Treat as a full DSN; if Postgres, the same DB hosts every
            # problem's study (Optuna namespaces by study_name).
            return override
        # Treat as a directory -> per-problem SQLite under it.
        d = Path(override)
        d.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{d / f'{problem_name}.db'}"
    studies_dir = results_root() / "studies"
    studies_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{studies_dir / f'{problem_name}.db'}"


def study_name(problem_name: str) -> str:
    return f"astromorf_tune_{problem_name}_b{DEFAULT_BUDGET}"


def trials_jsonl_path(problem_name: str) -> Path:
    p = results_root() / "trials_jsonl" / f"{problem_name}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


_JSONL_LOCK = threading.Lock()  # process-local lock; for cross-process use file flock.


def _append_trial_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append a JSON line atomically.

    SQLite is the cross-process source of truth; the JSONL file is a
    convenience export for the reporting layer. We use a per-process lock
    + append-mode write; multiple workers writing concurrently are fine
    because each line is atomic at the OS level for short records.
    """
    line = json.dumps(record, default=_json_default, sort_keys=True)
    with _JSONL_LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


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
    """Suggest a value for *param* via Optuna, applying narrowing.

    Narrowing is applied probabilistically: with probability
    ``NARROW_EXPLORATION_PROB`` we sample from the original full bounds
    instead of the narrowed window. This keeps the explorer alive even
    after the search localises.
    """
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
    # Categorical: ALWAYS use the original choice set. Optuna's
    # ``CategoricalDistribution`` rejects a different choice list on any
    # subsequent trial of the same study (``ValueError: CategoricalDistribution
    # does not support dynamic value space.``). ``narrowed.categoricals``
    # is intentionally ignored here; ``compute_narrowed_ranges`` no longer
    # populates it.
    return trial.suggest_categorical(param.name, list(param.choices))


def _sample_trial_params(
    trial: optuna.Trial, space: SearchSpace, narrowed: NarrowedRanges
) -> dict[str, Any]:
    """Sample a full hyperparameter dict from *space* for one Optuna trial."""
    rng_state = {"rng": __import__("random").Random(trial.number * 9301 + 49297)}
    params: dict[str, Any] = {}
    for p in space.all_params():
        params[p.name] = _suggest_param(trial, p, narrowed, rng_state)

    # Combinatorial guard: if the sampled (subspace, degree) is infeasible,
    # raise TrialPruned so Optuna marks it pruned (cheap rejection that
    # still informs the TPE prior).
    feasible, q = is_combinatorially_feasible(
        int(params["subspace_dim"]), int(params["polynomial_degree"])
    )
    trial.set_user_attr("basis_terms", q)
    if not feasible:
        trial.set_user_attr("rejected_combinatorial", True)
        log.info(
            "Trial %d pruned: q=%d > cap (subspace=%s degree=%s)",
            trial.number,
            q,
            params["subspace_dim"],
            params["polynomial_degree"],
        )
        raise optuna.TrialPruned(
            f"basis terms q={q} exceeds combinatorial cap"
        )
    return params


# ── narrowing snapshot ─────────────────────────────────────────────────────


def _completed_trials_for_narrowing(
    study: optuna.Study,
) -> list[dict[str, Any]]:
    """Materialise the completed-trials list expected by ``compute_narrowed_ranges``."""
    out: list[dict[str, Any]] = []
    for t in study.get_trials(
        deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)
    ):
        if t.value is None:
            continue
        out.append({"params": dict(t.params), "value": float(t.value)})
    return out


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
):
    """Build the closure Optuna will call for each trial."""

    rungs_t = tuple(rungs)

    def objective(trial: optuna.Trial) -> float:
        # Snapshot the narrowed-range view ONCE per trial so all parameter
        # samples within the trial see a consistent space.
        narrowed = compute_narrowed_ranges(
            space, _completed_trials_for_narrowing(trial.study)
        )
        trial.set_user_attr("narrowed_n_trials", narrowed.n_trials_used)

        params = _sample_trial_params(trial, space, narrowed)
        trial.set_user_attr("hparams", params)

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
                skip_combinatorial_check=True,  # already checked at sample time
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

            # Persist a JSONL record per-rung (so even pruned trials leave
            # diagnostics behind for later analysis).
            _append_trial_jsonl(
                trials_jsonl_path(problem_name),
                {
                    "ts": time.time(),
                    "trial_number": trial.number,
                    "rung_step": rung.step,
                    "rung_n_macroreps": rung.n_macroreps,
                    "rung_wall_clock_s": elapsed,
                    "params": params,
                    "score": score,
                    "result": result.to_record(),
                },
            )

            trial.report(score, step=rung.step)
            if trial.should_prune() and rung.step < rungs_t[-1].step:
                trial.set_user_attr("pruned_at_rung", rung.step)
                trial.set_user_attr("scores_per_rung", composite_history)
                raise optuna.TrialPruned()

        # Trial finished all rungs.
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

    return objective


# ── study factory ─────────────────────────────────────────────────────────


def build_study(
    problem_name: str,
    *,
    storage: str | None = None,
    seed: int = 20250428,
    enqueue_warmstart: bool = True,
    max_retries: int = 12,
) -> optuna.Study:
    """Create-or-load the per-problem study.

    Concurrency note
    ----------------
    On a fresh SQLite study, ``optuna.create_study`` always calls
    ``RDBStorage(storage)`` which runs ``metadata.create_all`` BEFORE the
    ``load_if_exists`` check. Under a Slurm array of N workers all
    starting simultaneously this races: every worker queries the schema,
    sees no tables, and tries to ``CREATE TABLE`` -- the first wins, the
    rest crash with ``OperationalError: table studies already exists``.

    Two layers of defence:
      1. The automation script pre-initialises each study in a single
         process (``worker.py --init-only``) before any sbatch, so by the
         time array workers start the schema already exists and
         ``create_all`` is a no-op.
      2. This retry loop catches the race anyway, in case the script is
         bypassed or the storage is hit from another path.
    """
    import sqlalchemy.exc

    storage_url = storage or storage_url_for(problem_name)
    sampler = TPESampler(
        seed=seed,
        multivariate=True,
        group=True,
        n_startup_trials=10,
        constant_liar=True,  # important for parallel workers
    )
    pruner = SuccessiveHalvingPruner(
        min_resource=1, reduction_factor=3, min_early_stopping_rate=0
    )

    # Build the storage explicitly so we can enable Optuna's heartbeat
    # mechanism. With heartbeats every 60s and a 5-minute grace period,
    # trials whose worker dies (Slurm timeout, OOM, node failure) are
    # auto-marked FAIL by the next worker that loads the study -- no
    # zombie RUNNING trials accumulate across resume cycles.
    storage_obj = optuna.storages.RDBStorage(
        url=storage_url,
        heartbeat_interval=60,
        grace_period=300,
    )

    last_exc: Exception | None = None
    study: optuna.Study | None = None
    for attempt in range(max_retries):
        try:
            study = optuna.create_study(
                study_name=study_name(problem_name),
                storage=storage_obj,
                sampler=sampler,
                pruner=pruner,
                direction="minimize",
                load_if_exists=True,
            )
            break
        except (sqlalchemy.exc.OperationalError, sqlalchemy.exc.IntegrityError) as exc:
            msg = str(exc).lower()
            transient = (
                "already exists" in msg
                or "database is locked" in msg
                or "unique constraint" in msg
            )
            if not transient:
                raise
            last_exc = exc
            backoff_s = min(0.25 * (2 ** attempt), 8.0)
            # Add a small per-attempt jitter so multiple racing workers
            # don't keep colliding in lockstep.
            import random
            backoff_s += random.uniform(0, 0.25)
            log.warning(
                "build_study(%s) attempt %d/%d hit transient %s: %s; "
                "retrying in %.2fs",
                problem_name, attempt + 1, max_retries,
                type(exc).__name__, msg.splitlines()[0][:120], backoff_s,
            )
            import time as _time
            _time.sleep(backoff_s)
    if study is None:
        raise RuntimeError(
            f"build_study({problem_name!r}) failed after {max_retries} retries"
        ) from last_exc

    # Direction metadata. Optuna's *internal* direction is always
    # "minimize" because we feed it the aligned score (raw * -minmax_sign).
    # Stash the *raw* direction here so downstream code (report, dashboard)
    # can convert back to natural units without re-probing the problem.
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
        except Exception as exc:  # never let this block init
            log.warning("Could not probe minmax for %s: %s", problem_name, exc)

    # Warm-start: enqueue the previously-known-best params once.
    if enqueue_warmstart and study.user_attrs.get("warmstart_enqueued") is not True:
        try:
            ws = warmstart_params(problem_name)
            study.enqueue_trial(ws, skip_if_exists=True)
            study.set_user_attr("warmstart_enqueued", True)
            study.set_user_attr("warmstart_params", ws)
            log.info("Warm-start enqueued for %s: %s", problem_name, ws)
        except KeyError:
            log.info("No warm-start defined for %s", problem_name)
    return study


def _probe_minmax_sign(problem_name: str) -> int:
    """Return +1 (maximise) or -1 (minimise) for *problem_name*.

    Done in a sub-context so the import cycle is local to this call.
    """
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
    budget: int = DEFAULT_BUDGET,
    rungs: Iterable[Rung] = DEFAULT_RUNGS,
    std_weight: float = 0.15,
    failure_penalty: float = 1e6,
    per_trial_wall_clock_cap_s: float | None = 30 * 60,
    walltime_budget_s: float | None = None,
    n_jobs: int | str = 1,
) -> int:
    """Drive ``study.optimize`` for at most *n_trials* (or until walltime).

    Returns the number of trials this worker actually completed.

    *n_jobs* controls intra-trial macroreplication parallelism. ``"auto"``
    reads ``$SLURM_CPUS_PER_TASK`` (the recommended setting on Slurm).
    """
    space = get_space(problem_name)
    study = build_study(problem_name, storage=storage, seed=seed)
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
    )

    deadline = (time.time() + walltime_budget_s) if walltime_budget_s else None
    n_done = 0

    def _stop_walltime(study, trial):
        if deadline is not None and time.time() >= deadline:
            study.stop()

    # We pass our own walltime callback so the Slurm worker shuts down
    # cleanly before the job is killed.
    callbacks = [_stop_walltime] if deadline else []
    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            callbacks=callbacks,
            gc_after_trial=True,
            catch=(Exception,),  # never let one bad trial kill the worker
        )
    finally:
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
