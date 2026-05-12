#!/bin/bash
#SBATCH --job-name=astromorf-cabs
#SBATCH --array=0-29
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs_opt/%x_%A_%a.out
#SBATCH --error=slurm_logs_opt/%x_%A_%a.err

# SLURM array template for optimize_astromorf_cabs_slurm.py.
# One array task = one (problem, config shard) pair.
# Update --array to match (num_problems * CONFIG_SHARDS): 0-(N-1).
# Example: 5 problems x 6 shards -> array 0-29.
# Current default PROBLEMS count: 5, CONFIG_SHARDS=6 -> array 0-29.

set -euo pipefail
mkdir -p slurm_logs_opt

# Cap thread oversubscription for BLAS/joblib.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export LOKY_MAX_CPU_COUNT="${SLURM_CPUS_PER_TASK:-1}"

# Activate environment (edit per cluster):
# source ~/miniconda3/bin/activate simopt

source "$HOME/miniconda3/bin/activate" simopt

# Experiment settings (override via exported env vars before sbatch).
PROBLEMS="${PROBLEMS:-NETWORK-1,DYNAMNEWS-1,SAN-1,ROSENBROCK-1,PARAMESTI-1}"
N_CONFIGS="${N_CONFIGS:-100}"
N_MACROREPS="${N_MACROREPS:-10}"
N_POSTREPS="${N_POSTREPS:-20}"
BUDGET="${BUDGET:-10000}"
MIN_SUBSPACE="${MIN_SUBSPACE:-2}"
MAX_SUBSPACE="${MAX_SUBSPACE:-99}"
DEGREES="${DEGREES:-2,3,4}"
BASIS_TYPES="${BASIS_TYPES:-hermite,chebyshev,monomial,natural,monomial_polynomial,laguerre}"
LAMBDA_MIN_MIN="${LAMBDA_MIN_MIN:-3}"
LAMBDA_MIN_MAX="${LAMBDA_MIN_MAX:-30}"
SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-results/astromorf_hyperopt}"
ARRAY_OFFSET="${ARRAY_OFFSET:-0}"
CONFIG_SHARDS="${CONFIG_SHARDS:-6}"

python simopt/scripts/optimize_astromorf_cabs_slurm.py \
    --problems "$PROBLEMS" \
    --n-configs "$N_CONFIGS" \
    --n-macroreps "$N_MACROREPS" \
    --n-postreps "$N_POSTREPS" \
    --budget "$BUDGET" \
    --min-subspace "$MIN_SUBSPACE" \
    --max-subspace "$MAX_SUBSPACE" \
    --degrees "$DEGREES" \
    --basis-types "$BASIS_TYPES" \
    --lambda-min-min "$LAMBDA_MIN_MIN" \
    --lambda-min-max "$LAMBDA_MIN_MAX" \
    --seed "$SEED" \
    --output-dir "$OUTPUT_DIR" \
    --array-offset "$ARRAY_OFFSET" \
    --config-shards "$CONFIG_SHARDS"
