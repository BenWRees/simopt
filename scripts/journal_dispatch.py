"""SLURM dispatcher: map ``SLURM_ARRAY_TASK_ID`` → task row.

This script is intentionally minimal — it imports nothing from ``simopt`` so
it loads in milliseconds and never fails on import errors during the SLURM
critical path.  All it does is read ``manifest.json`` and emit shell-friendly
``KEY=VALUE`` lines for the requested task id.  The SLURM script sources its
output via ``eval``.

Usage::

    eval "$(python scripts/journal_dispatch.py "$MANIFEST" "$SLURM_ARRAY_TASK_ID")"
    # Now $PROBLEM, $STUDY, $DESIGN_POINT_ID, $OUTPUT_SUBDIR, $TOTAL_TASKS,
    # $MANIFEST_SHA256, $N_MACROREPS, $N_POSTREPS, $BUDGET are defined.

If the task id is out of range the script exits 64 with a clear message; the
SLURM script should treat this as a fatal misconfiguration.
"""
from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path


def _shell_quote(value: object) -> str:
    return shlex.quote(str(value))


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(
            "usage: journal_dispatch.py MANIFEST.json TASK_ID",
            file=sys.stderr,
        )
        return 64
    manifest_path = Path(argv[1])
    try:
        task_id = int(argv[2])
    except ValueError:
        print(f"task id must be an integer; got {argv[2]!r}", file=sys.stderr)
        return 64

    try:
        manifest = json.loads(manifest_path.read_text())
    except FileNotFoundError:
        print(f"manifest not found: {manifest_path}", file=sys.stderr)
        return 64
    except json.JSONDecodeError as e:
        print(f"manifest is not valid JSON: {e}", file=sys.stderr)
        return 64

    total = int(manifest.get("total_tasks", 0))
    if not 0 <= task_id < total:
        print(
            f"task_id {task_id} out of range [0, {total}); "
            f"check your --array bound",
            file=sys.stderr,
        )
        return 64

    tasks = manifest["tasks"]
    row = tasks[task_id]
    if int(row["task_id"]) != task_id:
        print(
            f"manifest is corrupt: row index {task_id} has task_id "
            f"{row['task_id']!r}",
            file=sys.stderr,
        )
        return 70  # EX_SOFTWARE-ish

    global_cfg = manifest["global"]
    output_root = manifest["output_root"]
    out = {
        "TASK_ID": task_id,
        "PROBLEM": row["problem"],
        "STUDY": row["study"],
        "PROBLEM_DIM": row["problem_dim"],
        "DESIGN_POINT_ID": row["design_point_id"],
        "OUTPUT_SUBDIR": row["output_subdir"],
        "OUTPUT_ROOT": output_root,
        "WORK_DIR": str(Path(output_root) / row["output_subdir"]),
        "TOTAL_TASKS": total,
        "MANIFEST_SHA256": manifest.get("manifest_sha256", ""),
        "N_MACROREPS": global_cfg["n_macroreps"],
        "N_POSTREPS": global_cfg["n_postreps"],
        "BUDGET": global_cfg["budget"],
        "MACROREPS_PER_CHUNK": global_cfg.get("macroreps_per_chunk", 5),
    }
    for k, v in out.items():
        print(f"{k}={_shell_quote(v)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
