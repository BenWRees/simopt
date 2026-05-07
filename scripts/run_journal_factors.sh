#!/bin/bash
# =============================================================================
# Journal sensitivity studies for ASTROMoRF — all three studies in one array.
#
# Each array task runs ONE design point for ONE study:
#   * tasks [0, N_SUBSPACE)                                  -> subspace study
#   * tasks [N_SUBSPACE, N_SUBSPACE + N_BASIS)               -> basis study
#   * tasks [..., N_SUBSPACE + N_BASIS + N_REGULARISATION)   -> regularisation
#
# - subspace study      = NON-adaptive ASTROMoRF (fixed subspace dim)
# - basis study         = ADAPTIVE ASTROMoRF (CABS on, settings from registry)
# - regularisation study = ADAPTIVE ASTROMoRF (CABS on, settings from registry)
#
# Per-design-point pickles are written under
#   $OUTPUT_ROOT/<study>/
# These are everything you need to copy back locally for plotting:
#   <study>/<design_point_id>.pickle               # raw experiment object
#   <study>/<problem>_<study>_..._POSTREPS.pickle  # post-replicated curves
#   <study>/<design_point_id>.txt                  # plain-text summary
#
# After the array completes, run the aggregation locally (or on the head node)
# to produce CSV summaries:
#
#   for s in subspace basis regularisation; do
#     python demo/journal_factors_test.py \
#       --study $s --problem $PROBLEM --dim $DIM --budget $BUDGET \
#       --output-dir experiments/journal/$s --aggregate-only
#   done
# =============================================================================
#SBATCH --job-name=astromorf_journal
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=06:00:00
# Default array size assumes problem dim >= 8 and the default sweep levels:
#   N_SUBSPACE=8 (dims 1..8), N_BASIS=9 (all PolyBasisType), N_REG=4.
# Total = 8 + 9 + 4 = 21.  Adjust the upper bound below if you change sweeps
# or the problem dimension is < 8.  To re-derive sizes for your configuration:
#   python demo/journal_factors_test.py --study subspace --problem $PROBLEM \
#     --dim $DIM --budget $BUDGET --generate-csv --output-dir /tmp/check
#   tail -n +2 /tmp/check/design_matrix_subspace_*.csv | wc -l
# (repeat for --study basis and --study regularisation).
#SBATCH --array=0-20
#SBATCH --output=experiments/journal/logs/slurm_%A_%a.out
#SBATCH --error=experiments/journal/logs/slurm_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=$USER@soton.ac.uk

set -euo pipefail

# ── Experiment configuration ─────────────────────────────────────────────────
# Override at submission time, e.g.:
#   sbatch --export=ALL,PROBLEM=NETWORK-1,DIM=14,BUDGET=8000 scripts/run_journal_factors.slurm
PROBLEM=${PROBLEM:-SAN-1}
DIM=${DIM:-20}
BUDGET=${BUDGET:-5000}
N_MACROREPS=${N_MACROREPS:-10}
N_POSTREPS=${N_POSTREPS:-100}

# Per-study design-grid sizes.  These MUST match the SBATCH --array upper
# bound above (sum - 1).  See the comment block on the array directive
# for how to recompute them if you change sweeps.
N_SUBSPACE=${N_SUBSPACE:-8}
N_BASIS=${N_BASIS:-9}
N_REG=${N_REG:-4}

OUTPUT_ROOT=${OUTPUT_ROOT:-experiments/journal}

# ── Environment ──────────────────────────────────────────────────────────────
source "$HOME/miniconda3/bin/activate" simopt

# Resolve the repository root (scripts/ lives one level below it).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p "$OUTPUT_ROOT/logs"
mkdir -p "$OUTPUT_ROOT/subspace" "$OUTPUT_ROOT/basis" "$OUTPUT_ROOT/regularisation"

# ── Dispatch: SLURM_ARRAY_TASK_ID -> (study, local_task_id) ──────────────────
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

if   [ "$TASK_ID" -lt "$N_SUBSPACE" ]; then
    STUDY=subspace
    LOCAL_ID=$TASK_ID
elif [ "$TASK_ID" -lt "$((N_SUBSPACE + N_BASIS))" ]; then
    STUDY=basis
    LOCAL_ID=$((TASK_ID - N_SUBSPACE))
elif [ "$TASK_ID" -lt "$((N_SUBSPACE + N_BASIS + N_REG))" ]; then
    STUDY=regularisation
    LOCAL_ID=$((TASK_ID - N_SUBSPACE - N_BASIS))
else
    echo "Task $TASK_ID exceeds total design points $((N_SUBSPACE + N_BASIS + N_REG))" >&2
    exit 1
fi

OUT_DIR="$OUTPUT_ROOT/$STUDY"

echo "[$(date -u +%FT%TZ)] task=$TASK_ID study=$STUDY local_id=$LOCAL_ID problem=$PROBLEM dim=$DIM budget=$BUDGET"

# `--no-aggregate` keeps each worker independent — aggregation happens once
# after the whole array finishes (see the post-array commands at the top).
python demo/journal_factors_test.py \
    --study "$STUDY" \
    --problem "$PROBLEM" \
    --dim "$DIM" \
    --budget "$BUDGET" \
    --n-macroreps "$N_MACROREPS" \
    --n-postreps "$N_POSTREPS" \
    --output-dir "$OUT_DIR" \
    --task-id "$LOCAL_ID" \
    --no-aggregate

echo "[$(date -u +%FT%TZ)] task=$TASK_ID complete"
