#!/bin/bash
#SBATCH --job-name=confirm_all
#SBATCH --output=logs/confirm_all_%A_%a.out
#SBATCH --error=logs/confirm_all_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
# Number of candidate tasks (top-K). Keep in sync with --k passed to confirm.py.
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=30
# Memory per CPU
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-4%5

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Compute REPO_ROOT relative to the directory where sbatch was invoked
# (SLURM_SUBMIT_DIR). If that variable is unset fall back to the
# script's location. This avoids paths under /var/spool when Slurm
# copies the script into the spool directory.
REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

# Debugging helpful for first run: record resolved paths in the log
echo "SCRIPT_DIR=${SCRIPT_DIR}"
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}"
echo "REPO_ROOT=${REPO_ROOT}"


source ~/miniconda3/bin/activate simopt

problems=(DYNAMNEWS-1 SAN-1 NETWORK-1 ROSENBROCK-1 PARAMESTI-1)
idx=${SLURM_ARRAY_TASK_ID}
problem=${problems[$idx]}

echo "[${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}] Starting per-candidate confirm run for $problem on $(hostname)"

# Run one confirm worker per candidate as separate Slurm tasks. Each task
# will detect its task index via SLURM_PROCID and evaluate that candidate's
# macroreplications in parallel using the task's CPUs (pass --per-candidate-n-jobs auto).
srun python "$REPO_ROOT/scripts/tuning/confirm.py" --problem "$problem" --k 5 --n-macroreps 30 --per-candidate-n-jobs auto

echo "[${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}] Candidate tasks finished, aggregating results"
python "$REPO_ROOT/scripts/tuning/confirm.py" --problem "$problem" --k 5 --aggregate --n-macroreps 30

python "$REPO_ROOT/scripts/tuning/collect.py" --problem "$problem"

echo "[${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}] Finished $problem"
