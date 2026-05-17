#!/bin/bash
# =============================================================================
# Journal sensitivity analysis — aggregation stage.
#
# Submitted with `--dependency=afterany:<ARRAY_JOB_ID>` so this job runs once
# the per-design-point array has finished (regardless of individual task
# success).  Walks `$OUTPUT_ROOT/runs/<problem>/<study>/<dpid>/long_form.*`,
# stacks them into a single tidy table, and computes the paired-bootstrap CI
# tables used by the journal analysis layer.
#
# Required environment (exported via --export=ALL,...):
#     OUTPUT_ROOT    Absolute path to the journal output root
#                    (must contain runs/...; analysis/ will be created)
#     REPO_ROOT      Absolute path to the simopt git checkout  (optional)
#     CONDA_ENV      Conda env name                            (default: simopt)
# =============================================================================
#SBATCH --job-name=astromorf_journal_agg
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x/%j.out
#SBATCH --error=logs/%x/%j.err
#SBATCH --mail-type=END,FAIL

set -Eeuo pipefail
shopt -s inherit_errexit

: "${OUTPUT_ROOT:?OUTPUT_ROOT=<absolute path> must be set via --export}"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
CONDA_ENV="${CONDA_ENV:-simopt}"

if [ ! -d "$OUTPUT_ROOT/runs" ]; then
    echo "FATAL: $OUTPUT_ROOT/runs does not exist; nothing to aggregate." >&2
    exit 64
fi

CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Keep BLAS single-threaded; the aggregator is mostly pandas / bootstrap and
# benefits from process-level parallelism inside scipy, not nested threading.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p "$OUTPUT_ROOT/analysis"

cd "$REPO_ROOT"

srun --kill-on-bad-exit=1 \
    python -u "$REPO_ROOT/scripts/journal_aggregate.py" \
        --runs-root  "$OUTPUT_ROOT/runs" \
        --output-dir "$OUTPUT_ROOT/analysis"

echo "[$(date -u +%FT%TZ)] aggregation complete — outputs in $OUTPUT_ROOT/analysis"
