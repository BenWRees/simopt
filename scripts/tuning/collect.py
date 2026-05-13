"""Export per-problem study artefacts: trials CSV + top-20 JSON.

Reads the Optuna SQLite/Postgres study and the JSONL trial log and
produces:

    results/astromorf_tuning/exports/<problem>/
        trials.csv           -- one row per completed trial (params + metrics)
        top20.json           -- top-20 trials with diagnostics + 95% CI
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import optuna  # noqa: E402

from scripts.tuning import results_root  # noqa: E402
from scripts.tuning.evaluate import trial_params_to_solver_factors  # noqa: E402
from scripts.tuning.storage import make_storage  # noqa: E402
from scripts.tuning.tuner import (  # noqa: E402
    all_trials_jsonl_paths,
    study_name,
)

log = logging.getLogger("astromorf.tuning.collect")


def _ci95(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        return (float("nan"), float("nan"))
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    sem = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
    return (mean - 1.96 * sem, mean + 1.96 * sem)


def _load_jsonl_index(paths: list[Path]) -> dict[int, list[dict[str, Any]]]:
    """Map trial_number -> list of per-rung records, concatenated across files.

    Per-worker JSONL files are merged here; the worker_id field on each
    record disambiguates which worker wrote it. Records are sorted by
    rung_step so the final rung is always last.
    """
    index: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tn = rec.get("trial_number")
                if tn is None:
                    continue
                index[int(tn)].append(rec)
    for v in index.values():
        v.sort(key=lambda r: (r.get("rung_step", 0), r.get("ts", 0)))
    return index


def _serialize_factors(factors: dict[str, Any]) -> dict[str, Any]:
    out = dict(factors)
    if "polynomial basis" in out:
        b = out["polynomial basis"]
        out["polynomial basis"] = b.value if hasattr(b, "value") else str(b)
    return out


def export_problem(
    problem_name: str,
    *,
    storage: str | None = None,
    top_k: int = 20,
    out_dir: Path | None = None,
) -> dict[str, Path]:
    """Export trials.csv and top20.json for a single problem."""
    storage_obj, spec = make_storage(problem_name, storage_url=storage)
    log.info("Collect: storage backend=%s (%s)", spec.backend, spec.display)
    study = optuna.load_study(study_name=study_name(problem_name), storage=storage_obj)
    jsonl_index = _load_jsonl_index(all_trials_jsonl_paths(problem_name))

    out_dir = out_dir or (results_root() / "exports" / problem_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build a flat row per trial.
    rows: list[dict[str, Any]] = []
    for t in study.get_trials(deepcopy=False):
        row: dict[str, Any] = {
            "trial_number": t.number,
            "state": t.state.name,
            "value": (
                float(t.value)
                if t.value is not None and np.isfinite(t.value)
                else None
            ),
            "datetime_start": (
                t.datetime_start.isoformat() if t.datetime_start else None
            ),
            "duration_s": (
                t.duration.total_seconds() if t.duration is not None else None
            ),
        }
        for k, v in t.params.items():
            row[f"param.{k}"] = v
        for k, v in (t.user_attrs or {}).items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                row[f"attr.{k}"] = v
            else:
                row[f"attr.{k}"] = json.dumps(v, default=str)
        # Pull the final-rung diagnostics from JSONL if available.
        recs = jsonl_index.get(t.number, [])
        if recs:
            last = recs[-1]
            res = last.get("result", {})
            for k, v in (res.get("diagnostics") or {}).items():
                if isinstance(v, (int, float)):
                    row[f"diag.{k}"] = v
            row["final_rung_score"] = last.get("score")
            row["final_rung_step"] = last.get("rung_step")
        rows.append(row)

    # Write trials.csv
    csv_path = out_dir / "trials.csv"
    if rows:
        all_keys: list[str] = []
        seen: set[str] = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    all_keys.append(k)
        import csv

        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    else:
        csv_path.write_text("")

    # Build top-K JSON.
    completed = [
        t
        for t in study.get_trials(
            deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)
        )
        if t.value is not None and np.isfinite(t.value)
    ]
    completed.sort(key=lambda t: t.value)
    top: list[dict[str, Any]] = []
    for rank, t in enumerate(completed[:top_k]):
        recs = jsonl_index.get(t.number, [])
        diagnostics = {}
        objectives: list[float] = []
        aligned: list[float] = []
        if recs:
            last = recs[-1]
            res = last.get("result", {}) or {}
            diagnostics = res.get("diagnostics", {}) or {}
            objectives = list(res.get("objectives", []) or [])
            aligned = list(res.get("aligned_scores", []) or [])
        ci = _ci95(aligned) if aligned else (float("nan"), float("nan"))
        top.append(
            {
                "rank": rank,
                "trial_number": t.number,
                "study_value": float(t.value),
                "params": dict(t.params),
                "factors": _serialize_factors(
                    trial_params_to_solver_factors(dict(t.params))
                ),
                "user_attrs": {
                    k: v for k, v in (t.user_attrs or {}).items() if k != "diagnostics"
                },
                "diagnostics": diagnostics,
                "objectives": objectives,
                "aligned_scores": aligned,
                "mean_aligned": (
                    float(np.mean(aligned)) if aligned else float("nan")
                ),
                "std_aligned": (
                    float(np.std(aligned, ddof=1)) if len(aligned) > 1 else 0.0
                ),
                "ci95_aligned": [ci[0], ci[1]],
            }
        )

    top_path = out_dir / f"top{top_k}.json"
    top_path.write_text(json.dumps(top, indent=2, default=str))

    # Study metadata.
    pruned_combinatorial = sum(
        1
        for t in study.get_trials(deepcopy=False)
        if t.user_attrs and t.user_attrs.get("rejected_combinatorial")
    )
    meta_path = out_dir / "study_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "problem": problem_name,
                "study_name": study.study_name,
                "n_trials_total": len(study.get_trials(deepcopy=False)),
                "n_trials_complete": len(completed),
                "n_trials_pruned_combinatorial": pruned_combinatorial,
                "best_value": (
                    float(completed[0].value) if completed else None
                ),
                "user_attrs": dict(study.user_attrs or {}),
            },
            indent=2,
            default=str,
        )
    )

    log.info(
        "Exported %s: %d trials -> %s, top%d -> %s",
        problem_name,
        len(rows),
        csv_path,
        top_k,
        top_path,
    )
    return {"trials_csv": csv_path, "top_json": top_path, "meta": meta_path}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export Optuna study artefacts for one or more problems."
    )
    p.add_argument(
        "--problems",
        type=str,
        required=True,
        help="Comma-separated problem names.",
    )
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--storage", type=str, default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    for name in [s.strip() for s in args.problems.split(",") if s.strip()]:
        export_problem(name, storage=args.storage, top_k=args.top_k)


if __name__ == "__main__":
    main()
