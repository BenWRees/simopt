"""Generate per-problem Slurm array scripts for the ASTROMoRF tuner.

Each problem gets a separate ``.slurm`` script because (a) wall-time
budgets differ wildly (PARAMESTI-1 at native dim=2 is trivial, NETWORK-1
at dim=100 is hours) and (b) the user can resubmit a single problem
without re-submitting the others.

A single array task = a single async Optuna worker. Many tasks =
parallel search; they coordinate through the shared SQLite/Postgres
storage. cpus-per-task=1 by design (no nested parallelism per user
preference); macroreps run sequentially inside each trial so we can
extract diagnostics.

Usage::

    python HPC_code/tuning/generate_slurm.py \\
        --partition batch --time 06:00:00 --mem-per-cpu 4G --mail-user $USER@soton.ac.uk
"""

from __future__ import annotations

import argparse
import os
import stat
from pathlib import Path
from textwrap import dedent

REPO_ROOT = Path(__file__).resolve().parents[2]

# Per-problem Slurm defaults. Tweak via CLI overrides.
#
# array_size           = number of concurrent async Optuna workers
# cpus_per_task        = cores per worker (also intra-trial mrep parallelism)
# n_trials_per_worker  = trials each worker attempts before exiting
# time                 = Slurm walltime
#
# With cpus_per_task > 1, each worker runs that many mreps in parallel
# inside every trial. With ASHA rungs of 4/8/16 mreps, a worker with
# cpus_per_task=4 finishes a trial roughly 4x faster than a single-CPU
# worker (until limited by the rung size at rung 0). Total CPU footprint
# is array_size * cpus_per_task.
PROBLEM_DEFAULTS: dict[str, dict[str, str | int]] = {
    "DYNAMNEWS-1":  {"array_size": 8, "cpus_per_task": 8, "n_trials_per_worker": 28, "time": "24:00:00"},
    "SAN-1":        {"array_size": 8, "cpus_per_task": 8, "n_trials_per_worker": 28, "time": "24:00:00"},
    "NETWORK-1":    {"array_size": 8, "cpus_per_task": 8, "n_trials_per_worker": 28, "time": "24:00:00"},
    "ROSENBROCK-1": {"array_size": 8, "cpus_per_task": 8, "n_trials_per_worker": 28, "time": "24:00:00"},
    "PARAMESTI-1":  {"array_size": 2, "cpus_per_task": 8, "n_trials_per_worker": 24, "time": "24:00:00"},
}


SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=astromorf_tune_{problem}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem-per-cpu={mem_per_cpu}
#SBATCH --time={time}
#SBATCH --array=0-{array_max}{maxconcurrent}
#SBATCH --output={log_dir}/tune_{problem}_%A_%a.out
#SBATCH --error={log_dir}/tune_{problem}_%A_%a.err
{mail_lines}

set -euo pipefail

# Cap BLAS thread counts to 1 PER PROCESS so the {cpus_per_task} joblib
# subprocesses don't over-subscribe the cores Slurm allocated. The joblib
# loky backend forks {cpus_per_task} child processes; each child inherits
# these caps and runs its mrep on a single core (which is what NumPy/BLAS
# wants for serial ASTROMoRF runs anyway).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
# Tell joblib's loky backend how many cores it may actually use.
export LOKY_MAX_CPU_COUNT="${{SLURM_CPUS_PER_TASK:-{cpus_per_task}}}"

# Resolve the repo root at *submit* time (not generation time) so this
# script is portable across machines/checkouts. Override with
# REPO_ROOT=... sbatch ... to force a specific path.
REPO_ROOT="${{REPO_ROOT:-${{SLURM_SUBMIT_DIR:-$PWD}}}}"
LOG_DIR="${{LOG_DIR:-$REPO_ROOT/results/astromorf_tuning/logs}}"
mkdir -p "$LOG_DIR"

# Conda env (override CONDA_BASE / ENV_NAME if your cluster differs).
CONDA_BASE="${{CONDA_BASE:-$HOME/miniconda3}}"
ENV_NAME="${{ENV_NAME:-simopt}}"
# shellcheck disable=SC1091
source "$CONDA_BASE/bin/activate" "$ENV_NAME"

# Optional: point storage at a shared Postgres for cross-node scale.
# export ASTROMORF_OPTUNA_STORAGE="postgresql+psycopg2://USER:PW@HOST/optuna"

# Soft walltime budget: stop the worker ~3 min before SLURM kills it so
# its current trial can finish and the next worker can pick up the rest.
WALLTIME_S={walltime_s}

cd "$REPO_ROOT"

python -m scripts.tuning.worker \\
    --problem {problem} \\
    --n-trials {n_trials_per_worker} \\
    --walltime-s "$WALLTIME_S" \\
    --seed 20250428 \\
    --seed-offset "$SLURM_ARRAY_TASK_ID" \\
    --budget 10000 \\
    --per-trial-cap-s 1800 \\
    --std-weight 0.15 \\
    --failure-penalty 1000000 \\
    --n-jobs auto

echo "Worker $SLURM_ARRAY_TASK_ID finished at $(date -Is)"
"""


def _time_to_seconds(t: str) -> int:
    """Convert HH:MM:SS or DD-HH:MM:SS to seconds."""
    days = 0
    if "-" in t:
        d, t = t.split("-", 1)
        days = int(d)
    parts = [int(x) for x in t.split(":")]
    while len(parts) < 3:
        parts.append(0)
    h, m, s = parts[:3]
    return days * 86400 + h * 3600 + m * 60 + s


def _render(problem: str, args: argparse.Namespace) -> str:
    cfg = PROBLEM_DEFAULTS[problem]
    array_max = int(cfg["array_size"]) - 1
    n_trials_per_worker = int(args.n_trials_per_worker or cfg["n_trials_per_worker"])
    cpus_per_task = int(args.cpus_per_task or cfg["cpus_per_task"])
    time_str = args.time or str(cfg["time"])
    walltime_s = max(60, _time_to_seconds(time_str) - 180)
    maxconcurrent = f"%{args.max_concurrent}" if args.max_concurrent else ""

    mail_lines = ""
    if args.mail_user:
        mail_lines = (
            "#SBATCH --mail-type=BEGIN,END,FAIL\n"
            f"#SBATCH --mail-user={args.mail_user}"
        )

    # Default to a $REPO_ROOT-relative log dir resolved by the script at
    # submit time; only override when the caller explicitly passes one.
    log_dir = args.log_dir or "$REPO_ROOT/results/astromorf_tuning/logs"

    return SLURM_TEMPLATE.format(
        problem=problem,
        partition=args.partition,
        mem_per_cpu=args.mem_per_cpu,
        time=time_str,
        array_max=array_max,
        maxconcurrent=maxconcurrent,
        mail_lines=mail_lines,
        log_dir=log_dir,
        walltime_s=walltime_s,
        n_trials_per_worker=n_trials_per_worker,
        cpus_per_task=cpus_per_task,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate per-problem Slurm scripts for ASTROMoRF tuning."
    )
    p.add_argument("--partition", default="batch")
    p.add_argument("--mem-per-cpu", default="4G")
    p.add_argument(
        "--time",
        default=None,
        help="Override wall-time. Uses per-problem default if omitted.",
    )
    p.add_argument(
        "--n-trials-per-worker",
        type=int,
        default=None,
        help="Override trials/worker; uses per-problem default if omitted.",
    )
    p.add_argument(
        "--cpus-per-task",
        type=int,
        default=None,
        help=(
            "Cores per worker (also intra-trial mrep parallelism via "
            "joblib). Uses per-problem default if omitted."
        ),
    )
    p.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="If set, becomes %N in the array spec (cap concurrent tasks).",
    )
    p.add_argument("--mail-user", default=None)
    p.add_argument(
        "--log-dir",
        default=None,
        help="Slurm stdout/err log dir. Defaults to results/astromorf_tuning/logs.",
    )
    p.add_argument(
        "--problems",
        default="DYNAMNEWS-1,SAN-1,NETWORK-1,ROSENBROCK-1,PARAMESTI-1",
    )
    p.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent),
        help="Where to write the .slurm files (default: alongside this script).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    problems = [s.strip() for s in args.problems.split(",") if s.strip()]
    written: list[Path] = []
    for prob in problems:
        if prob not in PROBLEM_DEFAULTS:
            raise SystemExit(f"Unknown problem {prob!r}; known: {sorted(PROBLEM_DEFAULTS)}")
        contents = _render(prob, args)
        path = out_dir / f"tune_{prob}.slurm"
        path.write_text(contents)
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        written.append(path)
        print(f"wrote {path}")

    # submit_all.sh -- run from the repo root so SLURM_SUBMIT_DIR == repo root.
    submit_all = out_dir / "submit_all.sh"
    lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
    lines.append('# Submit from the repo root so SLURM_SUBMIT_DIR resolves correctly.')
    lines.append('SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"')
    lines.append('REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"')
    lines.append('cd "$REPO_ROOT"')
    for p in written:
        lines.append(f'sbatch "$SCRIPT_DIR/{p.name}"')
    lines.append("echo 'Submitted all tuning arrays.'")
    submit_all.write_text("\n".join(lines) + "\n")
    submit_all.chmod(submit_all.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"wrote {submit_all}")

    # confirm_all.sh -- one-shot post-tuning confirmation pass.
    confirm_all = out_dir / "confirm_all.sh"
    confirm_lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
    confirm_lines.append('SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"')
    confirm_lines.append('REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"')
    confirm_lines.append('cd "$REPO_ROOT"')
    confirm_lines.append(
        'CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"; '
        'ENV_NAME="${ENV_NAME:-simopt}"; '
        'source "$CONDA_BASE/bin/activate" "$ENV_NAME"'
    )
    for prob in problems:
        confirm_lines.append(
            f"python -m scripts.tuning.confirm --problem {prob} --k 5"
        )
        confirm_lines.append(
            f"python -m scripts.tuning.collect --problems {prob}"
        )
    confirm_lines.append(
        f"python -m scripts.tuning.report --problems "
        f"{','.join(problems)}"
    )
    confirm_all.write_text("\n".join(confirm_lines) + "\n")
    confirm_all.chmod(confirm_all.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"wrote {confirm_all}")


if __name__ == "__main__":
    main()
