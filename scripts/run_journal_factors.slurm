#!/bin/bash
# =============================================================================
# Journal-grade sensitivity studies for ASTROMoRF — manifest-driven SLURM array.
#
#   manifest.json   ──►   SLURM array task   ──►   $OUTPUT_ROOT/runs/.../
#
# This script is deliberately ignorant of study structure, factor levels, and
# task counts.  It reads them all from the manifest produced beforehand by:
#
#     python scripts/journal_generate_manifest.py \
#         --output-root $OUTPUT_ROOT --problems ... --dims ... \
#         --budget ... --n-macroreps ... --n-postreps ...
#
# Then submit the array sized exactly to the manifest:
#
#     TOTAL=$(python scripts/journal_dispatch.py "$MANIFEST" 0 \
#                | awk -F= '/^TOTAL_TASKS/{print $2}' | tr -d "'")
#     sbatch --export=ALL,MANIFEST=$MANIFEST,OUTPUT_ROOT=$OUTPUT_ROOT \
#            --array=0-$((TOTAL-1))%64 \
#            scripts/run_journal_factors.slurm
#
# Required environment (exported via --export=ALL,...):
#     MANIFEST       Absolute path to manifest.json
#     OUTPUT_ROOT    Absolute path to the journal output root
#     REPO_ROOT      Absolute path to the simopt git checkout  (optional;
#                    defaults to $SLURM_SUBMIT_DIR)
#     CONDA_ENV      Conda env name                            (default: simopt)
#
# Each array task is fully restart-safe:
#   • work happens in $SLURM_TMPDIR/<dpid>
#   • on success, the work dir is atomically rsync'd to OUTPUT_ROOT
#   • a `.done` sentinel makes resubmission idempotent
#   • SIGUSR1 (sent 120 s before the wall-clock limit) triggers a clean exit
#     so SLURM can requeue the task; partial state is checkpointed by
#     ProblemSolver.run_resumable
# =============================================================================
#SBATCH --job-name=astromorf_journal
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x/%A/task_%a.out
#SBATCH --error=logs/%x/%A/task_%a.err
#SBATCH --mail-type=FAIL,ARRAY_TASKS
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

set -Eeuo pipefail
shopt -s inherit_errexit

# ── Required environment ────────────────────────────────────────────────────
: "${MANIFEST:?MANIFEST=<path to manifest.json> must be set via --export}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT=<absolute path> must be set via --export}"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
CONDA_ENV="${CONDA_ENV:-simopt}"

if [ ! -f "$MANIFEST" ]; then
    echo "FATAL: manifest not found: $MANIFEST" >&2
    exit 64
fi

# ── Decode this task from the manifest (no shell arithmetic, no constants) ──
# journal_dispatch.py is dependency-light (stdlib only) so this is fast and
# robust even if simopt has an import error.
DISPATCH_OUT="$(python "$REPO_ROOT/scripts/journal_dispatch.py" \
                  "$MANIFEST" "$SLURM_ARRAY_TASK_ID")"
eval "$DISPATCH_OUT"

echo "================================================================="
echo "  task_id        : $TASK_ID / $TOTAL_TASKS"
echo "  problem        : $PROBLEM (dim $PROBLEM_DIM)"
echo "  study          : $STUDY"
echo "  design point   : $DESIGN_POINT_ID"
echo "  budget         : $BUDGET"
echo "  n_macroreps    : $N_MACROREPS"
echo "  n_postreps     : $N_POSTREPS"
echo "  manifest sha   : $MANIFEST_SHA256"
echo "  final output   : $OUTPUT_ROOT/$OUTPUT_SUBDIR"
echo "================================================================="

FINAL_DIR="$OUTPUT_ROOT/$OUTPUT_SUBDIR"
mkdir -p "$FINAL_DIR"

# ── Idempotency: skip if already completed ──────────────────────────────────
if [ -f "$FINAL_DIR/.done" ]; then
    echo "[$(date -u +%FT%TZ)] task already complete — skipping."
    exit 0
fi

# ── Conda env ───────────────────────────────────────────────────────────────
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Pin thread counts so per-task multiprocessing does not oversubscribe cores.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export LOKY_MAX_CPU_COUNT="${SLURM_CPUS_PER_TASK:-1}"

# ── Scratch work directory + atomic rsync-back ──────────────────────────────
TMP_ROOT="${SLURM_TMPDIR:-/tmp}"
WORK_DIR="$TMP_ROOT/journal_${SLURM_JOB_ID:-na}_${SLURM_ARRAY_TASK_ID:-na}"
mkdir -p "$WORK_DIR"

cleanup() {
    local exit_code=$?
    # Always try to rsync whatever exists — even on partial failure we want
    # the progress.json / provenance.json for the post-mortem.  --partial keeps
    # half-written files visible; rsync's rename-then-replace gives us atomic
    # publication for completed files.
    if [ -d "$WORK_DIR" ]; then
        rsync -a --partial --human-readable "$WORK_DIR"/ "$FINAL_DIR"/ || true
    fi
    rm -rf "$WORK_DIR" || true
    exit "$exit_code"
}
trap cleanup EXIT
# SIGUSR1: SLURM warns us 120 s before the wall-clock limit; exit 99 so the
# job is requeued.  ProblemSolver.run_resumable installs its own SIGTERM
# handler that checkpoints the partial run to result.resume.pickle.
handle_usr1() {
    echo "[$(date -u +%FT%TZ)] received SIGUSR1 — preparing requeue."
    exit 99
}
trap handle_usr1 USR1

# ── Provenance (env snapshot for post-mortem) ───────────────────────────────
env | sort > "$WORK_DIR/env.txt"
{
    echo "job_id=$SLURM_JOB_ID"
    echo "array_job_id=$SLURM_ARRAY_JOB_ID"
    echo "array_task_id=$SLURM_ARRAY_TASK_ID"
    echo "node=$(hostname)"
    echo "started_at=$(date -u +%FT%TZ)"
    echo "manifest_sha256=$MANIFEST_SHA256"
} > "$WORK_DIR/slurm.txt"

cd "$REPO_ROOT"

# ── Execute exactly one design point ────────────────────────────────────────
# `srun` gives us correct cgroup accounting on preemption/cancellation and
# ensures the inner process inherits the task's CPU/memory limits.
srun --kill-on-bad-exit=1 \
    python -u "$REPO_ROOT/scripts/journal_factors_test.py" run-task \
        --manifest "$MANIFEST" \
        --task-id  "$SLURM_ARRAY_TASK_ID" \
        --work-dir "$WORK_DIR" \
        --n-jobs   "${SLURM_CPUS_PER_TASK:-1}"

echo "[$(date -u +%FT%TZ)] task $SLURM_ARRAY_TASK_ID completed cleanly."
