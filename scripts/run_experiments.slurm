#!/bin/bash
#SBATCH --job-name=simopt_bench
#SBATCH --array=0-29
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

# One array task = one (problem, solver) pair = one ProblemSolver instance.
# Each task uses 1 node, 1 process, cpus-per-task threads (joblib over mreps).
# After all 30 tasks finish, run:
#   python scripts/run_experiments.py --combine results/all.pkl --output-dir results
# to assemble a ProblemsSolvers object from the per-pair pickles.
#
# To regenerate this script with profiling-tuned --time:
#   python scripts/run_experiments.py \
#       --profile --generate-slurm scripts/run_experiments.slurm \
#       --n-macroreps 100 --budget 10000 \
#       --output-dir results --cpus-per-task 4

set -euo pipefail
mkdir -p slurm_logs

# CRITICAL: cap thread/process oversubscription. Without these, each joblib
# worker spawns N BLAS threads, giving N*N processes per task and OOM risk.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export LOKY_MAX_CPU_COUNT="${SLURM_CPUS_PER_TASK:-1}"

# Activate environment (edit per cluster):
# source ~/miniconda3/bin/activate simopt

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

python "${REPO_ROOT}/scripts/run_experiments.py" \
    --slurm-mode \
    --n-macroreps "${N_MACROREPS:-100}" \
    --budget "${BUDGET:-10000}" \
    --output-dir "${OUTPUT_DIR:-results}" \
    --cpus-per-task "${SLURM_CPUS_PER_TASK:-1}"
