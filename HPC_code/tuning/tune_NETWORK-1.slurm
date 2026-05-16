#!/bin/bash
#SBATCH --job-name=astromorf_tune_NETWORK-1
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=20:00:00
#SBATCH --array=0-7
#SBATCH --output=$REPO_ROOT/results/astromorf_tuning/logs/tune_NETWORK-1_%A_%a.out
#SBATCH --error=$REPO_ROOT/results/astromorf_tuning/logs/tune_NETWORK-1_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=$USER@soton.ac.uk

set -euo pipefail

# Cap BLAS thread counts to 1 PER PROCESS so the 4 joblib
# subprocesses don't over-subscribe the cores Slurm allocated. The joblib
# loky backend forks 4 child processes; each child inherits
# these caps and runs its mrep on a single core (which is what NumPy/BLAS
# wants for serial ASTROMoRF runs anyway).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
# Tell joblib's loky backend how many cores it may actually use.
export LOKY_MAX_CPU_COUNT="${SLURM_CPUS_PER_TASK:-4}"

# Resolve the repo root at *submit* time (not generation time) so this
# script is portable across machines/checkouts. Override with
# REPO_ROOT=... sbatch ... to force a specific path.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/results/astromorf_tuning/logs}"
mkdir -p "$LOG_DIR"

# Conda env (override CONDA_BASE / ENV_NAME if your cluster differs).
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
ENV_NAME="${ENV_NAME:-simopt}"
# shellcheck disable=SC1091
source "$CONDA_BASE/bin/activate" "$ENV_NAME"

# Optional: point storage at a shared Postgres for cross-node scale.
# export ASTROMORF_OPTUNA_STORAGE="postgresql+psycopg2://USER:PW@HOST/optuna"

# Soft walltime budget: stop the worker ~3 min before SLURM kills it so
# its current trial can finish and the next worker can pick up the rest.
WALLTIME_S=71820

cd "$REPO_ROOT"

GLOBAL_TRIAL_LIMIT=401

python -m scripts.tuning.worker \
    --problem NETWORK-1 \
    --n-trials 28 \
    --max-trials "$GLOBAL_TRIAL_LIMIT" \
    --walltime-s "$WALLTIME_S" \
    --seed 20250428 \
    --seed-offset "$SLURM_ARRAY_TASK_ID" \
    --budget 10000 \
    --per-trial-cap-s 1800 \
    --std-weight 0.15 \
    --failure-penalty 1000000 \
    --n-jobs auto

echo "Worker $SLURM_ARRAY_TASK_ID finished at $(date -Is)"
