#!/bin/bash
#
# Two-phase SLURM submission script for AirlineRM experiments.
#
# Phase 1: Each array task runs one ProblemSolver (run + post_replicate_policy).
# Phase 2: A single dependency job collects all pickles into one CSV per experiment.
#
# Usage:
#   sbatch scripts/run_airline_rm_experiment.slurm
#
# Override settings via exported env vars before submitting, e.g.:
#   export BUDGET=100000 N_MACROREPS=30
#   sbatch scripts/run_airline_rm_experiment.slurm

#SBATCH --job-name=airline-rm-submit
#SBATCH --output=logs_van/airline-rm-submit-%j.out
#SBATCH --error=logs_van/airline-rm-submit-%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --nodes=1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=BEGIN,END,FAIL

# send mail to this address
#SBATCH --mail-user=B.Rees@soton.ac.uk

source /home/br7g18/miniconda3/bin/activate simopt

# ── Experiment settings (override with exported env vars) ──────────────────
N_MACROREPS="${N_MACROREPS:-20}"
BUDGET="${BUDGET:-75000}"
N_LOOKAHEAD_REPS="${N_LOOKAHEAD_REPS:-10}"
N_LOADFACTOR_REPS="${N_LOADFACTOR_REPS:-200}"
N_POSTREPS="${N_POSTREPS:-400}"
N_JOBS="${N_JOBS:-20}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs_van}"
RUN_ID="${RUN_ID:-${SLURM_JOB_ID}}"
EXPERIMENTS="${EXPERIMENTS:-adp_vs_davn_sgd}"
SOLVER_LABELS="${SOLVER_LABELS:-}"

EXPERIMENT_PY="/home/br7g18/simopt/demo/AirlineRM_experiment.py"

# Shared CLI args for all subcommands
SHARED_ARGS=(
    --budget "$BUDGET"
    --n-macroreps "$N_MACROREPS"
    --n-lookahead-reps "$N_LOOKAHEAD_REPS"
    --n-loadfactor-reps "$N_LOADFACTOR_REPS"
    --n-jobs "$N_JOBS"
    --output-dir "$OUTPUT_DIR"
    --run-id "$RUN_ID"
    --experiments "$EXPERIMENTS"
)
if [[ -n "$SOLVER_LABELS" ]]; then
    SHARED_ARGS+=(--solver-labels "$SOLVER_LABELS")
fi

mkdir -p logs_van

# ── Determine number of tasks ─────────────────────────────────────────────
N_TASKS=$(python "$EXPERIMENT_PY" list-tasks "${SHARED_ARGS[@]}" --format count)
if [[ -z "$N_TASKS" || "$N_TASKS" -le 0 ]]; then
    echo "ERROR: list-tasks returned '$N_TASKS' -- expected a positive integer." >&2
    exit 1
fi
echo "Task count: $N_TASKS"

# ── Phase 1: submit array job (one task per ProblemSolver) ─────────────────
PHASE1_JOBID=$(sbatch --parsable \
    --job-name="airline-rm-phase1" \
    --output="logs_van/airline-rm-phase1-%A_%a.out" \
    --error="logs_van/airline-rm-phase1-%A_%a.err" \
    --time=2-12:00:00 \
    --cpus-per-task="$N_JOBS" \
    --mem=32G \
    --nodes=1 \
    --array="0-$((N_TASKS - 1))" \
    --mail-type=FAIL \
    --mail-user=B.Rees@soton.ac.uk \
    --signal=SIGUSR1@90 \
    --wrap="
source /home/br7g18/miniconda3/bin/activate simopt
echo \"Phase 1 task \$SLURM_ARRAY_TASK_ID / $N_TASKS starting on \$(date)\"
python $EXPERIMENT_PY run-solver ${SHARED_ARGS[*]} \
    --task-index \$SLURM_ARRAY_TASK_ID \
    --n-postreps $N_POSTREPS
echo \"Phase 1 task \$SLURM_ARRAY_TASK_ID finished on \$(date)\"
")

echo "Submitted Phase 1 array job: $PHASE1_JOBID (tasks 0..$((N_TASKS - 1)))"

# ── Phase 2: submit collection job (depends on all Phase 1 tasks) ──────────
PHASE2_JOBID=$(sbatch --parsable \
    --job-name="airline-rm-phase2" \
    --output="logs_van/airline-rm-phase2-%j.out" \
    --error="logs_van/airline-rm-phase2-%j.err" \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=16G \
    --nodes=1 \
    --dependency="afterok:$PHASE1_JOBID" \
    --mail-type=END,FAIL \
    --mail-user=B.Rees@soton.ac.uk \
    --wrap="
source /home/br7g18/miniconda3/bin/activate simopt
echo \"Phase 2 (collect-results) starting on \$(date)\"
python $EXPERIMENT_PY collect-results ${SHARED_ARGS[*]}
echo \"Phase 2 finished on \$(date)\"
")

echo "Submitted Phase 2 collection job: $PHASE2_JOBID (depends on $PHASE1_JOBID)"
echo ""
echo "Monitor with:"
echo "  squeue -j $PHASE1_JOBID,$PHASE2_JOBID"
