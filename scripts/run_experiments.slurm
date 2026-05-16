#!/bin/bash
#SBATCH --job-name=simopt_bench
#SBATCH --array=0-29
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/home/br7g18/slurm_logs/%x_%A_%a.out
#SBATCH --error=/home/br7g18/slurm_logs/%x_%A_%a.err

# One array task = one (problem, solver) pair = one ProblemSolver instance.
# Each task uses 1 node, 1 process, cpus-per-task threads (joblib over mreps).
# After all 30 tasks finish, run:
#   python scripts/run_experiments.py --combine results/all.pkl --output-dir results
# to assemble a ProblemsSolvers object from the per-pair pickles.
#
# To regenerate this script with profiling-tuned --time:
#   python scripts/run_experiments.py \
#       --profile --generate-slurm scripts/run_experiments.slurm \
#       --n-macroreps 30 --budget 10000 \
#       --output-dir results --cpus-per-task 10

set -euo pipefail

# SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# # Compute repository root relative to the directory where sbatch was invoked
# # (SLURM_SUBMIT_DIR). Fall back to the script directory when unset.
# REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}/.." && pwd)"
# cd "$REPO_ROOT"

# echo "SCRIPT_DIR=${SCRIPT_DIR}"
# echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}"
# echo "REPO_ROOT=${REPO_ROOT}"

mkdir -p "$HOME/slurm_logs" || true

# CRITICAL: cap thread/process oversubscription. Without these, each joblib
# worker spawns N BLAS threads, giving N*N processes per task and OOM risk.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export LOKY_MAX_CPU_COUNT="${SLURM_CPUS_PER_TASK:-10}"

# Activate environment (edit per cluster):
source ~/miniconda3/bin/activate simopt


# Default output directory to ~/results/comparison unless overridden.
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/results/comparison}"
mkdir -p "$OUTPUT_DIR"

python "$HOME/simopt/scripts/run_experiments.py" \
    --slurm-mode \
    --n-macroreps "${N_MACROREPS:-30}" \
    --budget "${BUDGET:-10000}" \
    --output-dir "$OUTPUT_DIR" \
    --cpus-per-task "${SLURM_CPUS_PER_TASK:-10}"
