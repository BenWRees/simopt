"""Generate per-problem SLURM array scripts for the ASTROMoRF tuner.

Each problem gets its own ``.slurm`` script because (a) wall-time budgets
differ by problem (PARAMESTI-1 at native dim=2 is trivial, NETWORK-1 at
dim=100 is hours) and (b) the user can resubmit a single problem without
touching the others.

One SLURM array task = one async Optuna worker process. Workers coordinate
through the shared storage backend (JournalStorage by default — HPC-safe;
Postgres by env var; SQLite only as a last-resort single-node fallback).

Key invariants
--------------
1. **SBATCH directives cannot use shell variables.** ``#SBATCH --output=...``
   is parsed by SLURM before the shell runs, so any ``$VAR`` in the value
   is treated as a literal directory name. We therefore use *relative*
   paths in SBATCH directives; SLURM resolves them against
   ``$SLURM_SUBMIT_DIR``, which equals the repo root provided the user
   submits via ``submit_all.sh`` (which ``cd``s there first).
2. Workers cap BLAS threads to 1 per process so the joblib-loky children
   never oversubscribe the cores SLURM allocated.
3. Walltime budget passed to the worker is 180s below SLURM walltime, so
   the in-flight trial has a chance to finish gracefully before
   ``scancel`` arrives.

Usage::

    python HPC_code/tuning/generate_slurm.py \\
        --partition batch --time 12:00:00 --mem-per-cpu 4G \\
        --mail-user $USER@soton.ac.uk
"""

from __future__ import annotations

import argparse
import stat
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Per-problem SLURM defaults. Tweak via CLI overrides.
#
#   array_size           = number of concurrent async Optuna workers
#   cpus_per_task        = cores per worker (also intra-trial mrep parallelism)
#   n_trials_per_worker  = trials each worker attempts before exiting
#   time                 = SLURM walltime
#
# Sizing rationale (12h target):
#
# - rung 0 (4 mreps) on cpus_per_task=4 -> ~1 mrep round-trip per trial.
# - rung 1 (8 mreps) -> 2 rounds.
# - rung 2 (16 mreps) -> 4 rounds.
# - With ASHA pruning, most trials die at rung 0; the survivor budget at
#   rung 2 is roughly array_size * n_trials_per_worker / 9.
# - 8 workers * 50 trials = 400 trials/study, ~45 survivors at rung 2.
#   At ~120s/mrep wall-clock for ROSENBROCK-1 at dim=100, the math is:
#     400 * 4 mreps (rung 0)       =  1600 mrep-equivalents
#     ~133 * 8 mreps (rung 1)      =  1066
#     ~45 * 16 mreps (rung 2)      =   720
#   total: ~3400 mreps / 8 workers / 4 cpus = ~106 mrep "rounds" per
#   worker = ~3.5h wall-clock at 120s/mrep. Comfortable inside 12h.
PROBLEM_DEFAULTS: dict[str, dict[str, str | int]] = {
    "DYNAMNEWS-1":  {"array_size": 8, "cpus_per_task": 4, "n_trials_per_worker": 50, "time": "12:00:00"},
    "SAN-1":        {"array_size": 8, "cpus_per_task": 4, "n_trials_per_worker": 50, "time": "12:00:00"},
    "NETWORK-1":    {"array_size": 8, "cpus_per_task": 4, "n_trials_per_worker": 50, "time": "12:00:00"},
    "ROSENBROCK-1": {"array_size": 8, "cpus_per_task": 4, "n_trials_per_worker": 50, "time": "12:00:00"},
    "PARAMESTI-1":  {"array_size": 4, "cpus_per_task": 4, "n_trials_per_worker": 40, "time": "06:00:00"},
}


# NOTE: SBATCH --output / --error use RELATIVE paths because SLURM does
# not expand shell variables in directives. Relative paths are resolved
# against $SLURM_SUBMIT_DIR; submit_all.sh guarantees that this is the
# repo root by cd'ing there before sbatch.
SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=astromorf_tune_{problem}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem-per-cpu={mem_per_cpu}
#SBATCH --time={time}
#SBATCH --array=0-{array_max}{maxconcurrent}
#SBATCH --output={log_dir_rel}/tune_{problem}_%A_%a.out
#SBATCH --error={log_dir_rel}/tune_{problem}_%A_%a.err
{mail_lines}

set -euo pipefail

# BLAS thread caps: 1 per process so the {cpus_per_task} joblib children
# don't over-subscribe the cores SLURM allocated. Each child inherits
# these caps and runs its mrep on a single core, which is what NumPy/BLAS
# wants for serial ASTROMoRF runs.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
# Tell joblib's loky backend exactly how many cores it may use.
export LOKY_MAX_CPU_COUNT="${{SLURM_CPUS_PER_TASK:-{cpus_per_task}}}"

# Resolve repo root at *submit* time (not generation time) so this script
# is portable across machines / checkouts. Override with
# REPO_ROOT=... sbatch ... to force a specific path.
REPO_ROOT="${{REPO_ROOT:-${{SLURM_SUBMIT_DIR:-$PWD}}}}"
mkdir -p "$REPO_ROOT/{log_dir_rel}"
mkdir -p "$REPO_ROOT/results/astromorf_tuning/journal"
mkdir -p "$REPO_ROOT/results/astromorf_tuning/.locks"

# Conda env (override CONDA_BASE / ENV_NAME for non-default clusters).
CONDA_BASE="${{CONDA_BASE:-$HOME/miniconda3}}"
ENV_NAME="${{ENV_NAME:-simopt}}"
# shellcheck disable=SC1091
source "$CONDA_BASE/bin/activate" "$ENV_NAME"

# Storage backend: JournalStorage on the HPC shared/local filesystem.
# This is the canonical, self-contained, no-server backend. There is NO
# database server anywhere in this pipeline; workers coordinate through
# an append-only journal log guarded by an os.symlink-based mutex.
#
# To inspect runs with optuna-dashboard, use:
#     bash HPC_code/tuning/launch_dashboard.sh
# which exports a SQLite snapshot from the journal — the live journal
# itself is never touched by the dashboard.
export ASTROMORF_OPTUNA_STORAGE="${{ASTROMORF_OPTUNA_STORAGE:-journal://$REPO_ROOT/results/astromorf_tuning/journal}}"

# Soft walltime: stop the worker ~3 min before SLURM kills it so its
# current trial finishes and the next worker can pick up.
WALLTIME_S={walltime_s}

cd "$REPO_ROOT"

python -m scripts.tuning.worker \\
    --problem {problem} \\
    --n-trials {n_trials_per_worker} \\
    --max-trials {global_trial_limit} \\
    --walltime-s "$WALLTIME_S" \\
    --seed 20250428 \\
    --worker-id "$SLURM_ARRAY_TASK_ID" \\
    --budget 10000 \\
    --per-trial-cap-s 1800 \\
    --std-weight 0.15 \\
    --failure-penalty 1000000 \\
    --n-jobs auto \\
    --startup-jitter-s 15

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
    global_trial_limit = int(cfg["array_size"]) * n_trials_per_worker + 1
    walltime_s = max(60, _time_to_seconds(time_str) - 180)
    maxconcurrent = f"%{args.max_concurrent}" if args.max_concurrent else ""

    mail_lines = ""
    if args.mail_user:
        mail_lines = (
            "#SBATCH --mail-type=BEGIN,END,FAIL\n"
            f"#SBATCH --mail-user={args.mail_user}"
        )

    # SBATCH paths must be relative (resolved by SLURM against the submit
    # directory) OR absolute. We use relative, resolved against the repo
    # root supplied by submit_all.sh.
    log_dir_rel = args.log_dir or "results/astromorf_tuning/logs"

    return SLURM_TEMPLATE.format(
        problem=problem,
        partition=args.partition,
        mem_per_cpu=args.mem_per_cpu,
        time=time_str,
        array_max=array_max,
        maxconcurrent=maxconcurrent,
        mail_lines=mail_lines,
        log_dir_rel=log_dir_rel,
        walltime_s=walltime_s,
        n_trials_per_worker=n_trials_per_worker,
        cpus_per_task=cpus_per_task,
        global_trial_limit=global_trial_limit,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate per-problem SLURM scripts for ASTROMoRF tuning."
    )
    p.add_argument("--partition", default="batch")
    p.add_argument("--mem-per-cpu", default="4G")
    p.add_argument(
        "--time", default=None,
        help="Override wall-time. Uses per-problem default if omitted.",
    )
    p.add_argument(
        "--n-trials-per-worker", type=int, default=None,
        help="Override trials/worker; uses per-problem default if omitted.",
    )
    p.add_argument(
        "--cpus-per-task", type=int, default=None,
        help=(
            "Cores per worker (also intra-trial mrep parallelism). "
            "Uses per-problem default if omitted."
        ),
    )
    p.add_argument(
        "--max-concurrent", type=int, default=None,
        help="If set, becomes %N in the array spec (cap concurrent tasks).",
    )
    p.add_argument("--mail-user", default=None)
    p.add_argument(
        "--log-dir", default=None,
        help=(
            "SLURM stdout/err log dir, RELATIVE to repo root (must NOT start "
            "with /). Defaults to results/astromorf_tuning/logs."
        ),
    )
    p.add_argument(
        "--problems",
        default="DYNAMNEWS-1,SAN-1,NETWORK-1,ROSENBROCK-1,PARAMESTI-1",
    )
    p.add_argument(
        "--out-dir", default=str(Path(__file__).resolve().parent),
        help="Where to write the .slurm files (default: alongside this script).",
    )
    return p.parse_args()


def _write_submit_all(out_dir: Path, written: list[Path]) -> None:
    submit_all = out_dir / "submit_all.sh"
    lines = [
        "#!/usr/bin/env bash",
        "# Submit every per-problem SLURM array for ASTROMoRF tuning.",
        "#",
        "# CRITICAL: this script cd's to the repo root before sbatch so that",
        "# SLURM_SUBMIT_DIR == repo root. The .slurm files use RELATIVE log",
        "# paths in their #SBATCH directives (because SLURM does not expand",
        "# shell variables in directives), so the working directory at sbatch",
        "# time decides where logs are written.",
        "set -euo pipefail",
        '',
        'SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"',
        'REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"',
        'cd "$REPO_ROOT"',
        '',
        '# Ensure log + storage dirs exist so the array tasks don\'t race to',
        '# create them.',
        'mkdir -p "$REPO_ROOT/results/astromorf_tuning/logs"',
        'mkdir -p "$REPO_ROOT/results/astromorf_tuning/journal"',
        'mkdir -p "$REPO_ROOT/results/astromorf_tuning/.locks"',
        'mkdir -p "$REPO_ROOT/results/astromorf_tuning/trials_jsonl"',
        '',
        '# Pre-init each study in a single-process call. With JournalStorage',
        '# this is *not* required for correctness (the journal handles',
        '# concurrent creates), but it warms file caches and ensures the',
        '# warm-start trial is enqueued once. Safe to skip if you trust the',
        '# storage layer.',
        'if [[ "${SKIP_INIT:-0}" != "1" ]]; then',
        '  CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"',
        '  ENV_NAME="${ENV_NAME:-simopt}"',
        '  # shellcheck disable=SC1091',
        '  source "$CONDA_BASE/bin/activate" "$ENV_NAME"',
        '  for prob in DYNAMNEWS-1 SAN-1 NETWORK-1 ROSENBROCK-1 PARAMESTI-1; do',
        '    python -m scripts.tuning.worker --problem "$prob" --init-only',
        '  done',
        'fi',
        '',
    ]
    for p in written:
        lines.append(f'sbatch "$SCRIPT_DIR/{p.name}"')
    lines.append("echo 'Submitted all tuning arrays.'")
    submit_all.write_text("\n".join(lines) + "\n")
    submit_all.chmod(submit_all.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"wrote {submit_all}")


def _write_confirm_all(out_dir: Path, problems: list[str]) -> None:
    confirm_all = out_dir / "confirm_all.sh"
    lines = [
        "#!/usr/bin/env bash",
        "# Post-tuning confirmation + export + report pipeline.",
        "# Run this after every tune_<PROBLEM>.slurm array has finished.",
        "set -euo pipefail",
        '',
        'SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"',
        'REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"',
        'cd "$REPO_ROOT"',
        '',
        'CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"',
        'ENV_NAME="${ENV_NAME:-simopt}"',
        '# shellcheck disable=SC1091',
        'source "$CONDA_BASE/bin/activate" "$ENV_NAME"',
        '',
        '# Mark any zombie RUNNING trials as FAIL before confirming. The',
        '# heartbeat mechanism already does this on the next worker load,',
        '# but at this point there are no more workers running.',
        f'python -m scripts.tuning.cleanup --problems {",".join(problems)} --max-age-s 600',
        '',
    ]
    for prob in problems:
        lines.append(f"python -m scripts.tuning.confirm --problem {prob} --k 5")
        lines.append(f"python -m scripts.tuning.collect --problems {prob}")
    lines.append(
        f"python -m scripts.tuning.report --problems {','.join(problems)}"
    )
    confirm_all.write_text("\n".join(lines) + "\n")
    confirm_all.chmod(confirm_all.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"wrote {confirm_all}")


def main() -> None:
    args = _parse_args()
    if args.log_dir and args.log_dir.startswith("/"):
        raise SystemExit(
            "--log-dir must be a path RELATIVE to the repo root; absolute "
            "paths would silently break because #SBATCH does not expand "
            "shell variables in directives."
        )
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

    _write_submit_all(out_dir, written)
    _write_confirm_all(out_dir, problems)


if __name__ == "__main__":
    main()
