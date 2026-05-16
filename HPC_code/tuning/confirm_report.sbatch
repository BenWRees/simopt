#!/bin/bash
#SBATCH --job-name=confirm_report
#SBATCH --output=logs/confirm_report_%j.out
#SBATCH --error=logs/confirm_report_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Use SLURM_SUBMIT_DIR when available (sbatch runs scripts from spool),
# otherwise fall back to deriving repo root from script location.
REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

echo "SCRIPT_DIR=${SCRIPT_DIR}"
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}"
echo "REPO_ROOT=${REPO_ROOT}"

source ~/miniconda3/bin/activate simopt

echo "[${SLURM_JOB_ID}] Running final report"

python "$REPO_ROOT/scripts/tuning/report.py" --problems DYNAMNEWS-1,SAN-1,NETWORK-1,ROSENBROCK-1,PARAMESTI-1

echo "[${SLURM_JOB_ID}] Report finished"
