#!/usr/bin/env bash
# =============================================================================
# run_astromorf_hyperparameter_tuning.sh
# -----------------------------------------------------------------------------
# One-command HPC tuning of ASTROMoRF for simulation budget = 10 000 across
# DYNAMNEWS-1, SAN-1, NETWORK-1, ROSENBROCK-1 (dim=100) and PARAMESTI-1 (dim=2).
#
# What this script does, in order:
#   1. PREFLIGHT  -- verify the repo root, Slurm, conda env, optuna, writable
#                    results dir.
#   2. GENERATE   -- regenerate the per-problem .slurm files via
#                    HPC_code/tuning/generate_slurm.py (idempotent).
#   3. SUBMIT     -- sbatch every tune_<PROBLEM>.slurm, capturing the job IDs
#                    in results/astromorf_tuning/automation_state.json.
#                    Skipped if a previous submission's jobs are still queued
#                    or running (resume).
#   4. WAIT       -- poll `squeue` until none of our jobs remain. Skipped if
#                    the user passed --skip-wait or no jobs were submitted.
#   5. CONFIRM    -- per-problem disjoint-seed confirmation pass over top-K.
#   6. COLLECT    -- export trials.csv + top20.json + study_meta.json.
#   7. REPORT     -- write recommendations.md + recommendation_table.csv.
#
# RESUME
# ------
# All intermediate state lives in:
#   * the per-problem Optuna SQLite study (results/.../studies/<problem>.db)
#   * results/astromorf_tuning/automation_state.json (job IDs from last submit)
#   * the per-problem JSONL diagnostics log
#
# Re-running this script is always safe:
#   - existing studies are loaded via Optuna's `load_if_exists=True`
#   - already-submitted jobs are detected via squeue + the state file; no
#     duplicate sbatch
#   - confirm/collect/report rebuild their outputs from the studies, so they
#     can be re-run any time
#
# OPTUNA STORAGE
# --------------
# By default each problem gets its own SQLite study under
#   results/astromorf_tuning/studies/<PROBLEM>.db
# To use a single Postgres backend across nodes (recommended above ~16 workers
# per problem):
#   export ASTROMORF_OPTUNA_STORAGE="postgresql+psycopg2://USER:PW@HOST/optuna"
#
# SLURM ARRAYS
# ------------
# Each tune_<PROBLEM>.slurm is an array of N async Optuna workers (default
# 16 for the dim-100 problems, 4 for PARAMESTI-1). Workers share the same
# study and pull the next trial atomically from storage. cpus-per-task=1 by
# design (no nested parallelism); macroreps run sequentially within a trial
# so per-mrep diagnostics can be extracted from the solver instance.
#
# CLUSTER CUSTOMISATION (override via env vars)
# ---------------------------------------------
#   CONDA_BASE             default: $HOME/miniconda3
#   ENV_NAME               default: simopt
#   SLURM_PARTITION        default: batch
#   MAX_CONCURRENT         default: unset (no per-array cap)
#   MEM_PER_CPU            default: 4G
#   MAIL_USER              default: $USER@$(hostname -d)  (best-effort)
#   CPUS_PER_TASK          default: unset (uses per-problem default: 4)
#                          Each worker runs this many macroreps in
#                          parallel inside every trial via joblib.
#   ASTROMORF_OPTUNA_STORAGE  see "OPTUNA STORAGE" above
#   POLL_INTERVAL_S        default: 60   -- squeue poll cadence
#   PROBLEMS               default: DYNAMNEWS-1,SAN-1,NETWORK-1,ROSENBROCK-1,PARAMESTI-1
#
# FLAGS
# -----
#   --regenerate-slurm     force regeneration of the .slurm files
#   --skip-submit          do not sbatch (use when jobs are already running)
#   --skip-wait            do not poll squeue (use when jobs are known done)
#   --finalise-only        skip preflight/submit/wait, just confirm+collect+report
#   --dry-run              print every action without executing it
#   -h, --help             show this header
#
# Usage::
#   bash run_astromorf_hyperparameter_tuning.sh
#   bash run_astromorf_hyperparameter_tuning.sh --finalise-only
#   SLURM_PARTITION=long MAX_CONCURRENT=8 bash run_astromorf_hyperparameter_tuning.sh
# =============================================================================

set -euo pipefail
shopt -s extglob

# ── locate this script and the repo root ──────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$REPO_ROOT/$(basename "${BASH_SOURCE[0]}")"
cd "$REPO_ROOT"

# ── defaults (env-overridable) ────────────────────────────────────────────
: "${CONDA_BASE:=$HOME/miniconda3}"
: "${ENV_NAME:=simopt}"
: "${SLURM_PARTITION:=batch}"
: "${MEM_PER_CPU:=4G}"
: "${MAX_CONCURRENT:=}"           # empty -> no array cap
: "${POLL_INTERVAL_S:=60}"
: "${PROBLEMS:=DYNAMNEWS-1,SAN-1,NETWORK-1,ROSENBROCK-1,PARAMESTI-1}"
: "${MAIL_USER:=}"
: "${CPUS_PER_TASK:=}"            # empty -> per-problem default from generator
: "${ASTROMORF_OPTUNA_STORAGE:=}"
export ASTROMORF_OPTUNA_STORAGE   # downstream Python honours this

RESULTS_DIR="$REPO_ROOT/results/astromorf_tuning"
STATE_FILE="$RESULTS_DIR/automation_state.json"
HPC_DIR="$REPO_ROOT/HPC_code/tuning"

# ── arg parsing ───────────────────────────────────────────────────────────
REGEN_SLURM=0
SKIP_SUBMIT=0
SKIP_WAIT=0
FINALISE_ONLY=0
DRY_RUN=0
while (( $# > 0 )); do
    case "$1" in
        --regenerate-slurm) REGEN_SLURM=1 ;;
        --skip-submit)      SKIP_SUBMIT=1 ;;
        --skip-wait)        SKIP_WAIT=1 ;;
        --finalise-only)    FINALISE_ONLY=1; SKIP_SUBMIT=1; SKIP_WAIT=1 ;;
        --dry-run)          DRY_RUN=1 ;;
        -h|--help)
            sed -n '2,80p' "$SCRIPT_PATH"
            exit 0
            ;;
        *)
            echo "Unknown flag: $1" >&2
            exit 2
            ;;
    esac
    shift
done

# ── pretty-printing helpers ──────────────────────────────────────────────
log()  { printf '\033[1;34m[tuning]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[tuning] WARN:\033[0m %s\n' "$*" >&2; }
fail() { printf '\033[1;31m[tuning] FAIL:\033[0m %s\n' "$*" >&2; exit 1; }
run()  {
    log "+ $*"
    if (( DRY_RUN == 0 )); then
        eval "$@"
    fi
}

# ── conda activation (best-effort) ───────────────────────────────────────
activate_conda() {
    if [[ -n "${CONDA_DEFAULT_ENV:-}" && "${CONDA_DEFAULT_ENV}" == "$ENV_NAME" ]]; then
        log "Conda env '$ENV_NAME' already active."
        return
    fi
    if [[ ! -f "$CONDA_BASE/bin/activate" ]]; then
        warn "Conda not found at $CONDA_BASE/bin/activate; assuming Python is on PATH."
        return
    fi
    # shellcheck disable=SC1091
    source "$CONDA_BASE/bin/activate" "$ENV_NAME" \
        || fail "Could not activate conda env '$ENV_NAME' (CONDA_BASE=$CONDA_BASE)."
    log "Activated conda env '$ENV_NAME'."
}

# ── preflight ────────────────────────────────────────────────────────────
preflight() {
    log "==== PREFLIGHT ===="
    log "Repo root: $REPO_ROOT"
    [[ -d "$REPO_ROOT/scripts/tuning" ]] \
        || fail "scripts/tuning/ not found under $REPO_ROOT"
    [[ -d "$HPC_DIR" ]] \
        || fail "HPC_code/tuning/ not found under $REPO_ROOT"
    [[ -f "$HPC_DIR/generate_slurm.py" ]] \
        || fail "HPC_code/tuning/generate_slurm.py not found"

    activate_conda

    log "Checking optuna availability..."
    if ! python -c 'import optuna; print("optuna", optuna.__version__)' 2>/dev/null; then
        fail "optuna is not importable. Install it with: pip install -e '.[tuning]'"
    fi

    log "Checking results dir is writable: $RESULTS_DIR"
    mkdir -p "$RESULTS_DIR/studies" "$RESULTS_DIR/logs" \
        || fail "Cannot create $RESULTS_DIR"
    [[ -w "$RESULTS_DIR" ]] || fail "$RESULTS_DIR is not writable"

    if (( SKIP_SUBMIT == 0 || SKIP_WAIT == 0 )); then
        log "Checking Slurm availability..."
        command -v sbatch >/dev/null 2>&1 \
            || fail "sbatch not found on PATH. If running off-cluster, use --finalise-only."
        command -v squeue >/dev/null 2>&1 \
            || fail "squeue not found on PATH."
        log "Slurm ok (sbatch=$(command -v sbatch))."
    fi

    if [[ -n "$ASTROMORF_OPTUNA_STORAGE" ]]; then
        log "Storage backend (env): $ASTROMORF_OPTUNA_STORAGE"
    else
        log "Storage backend: SQLite (default), one DB per problem under $RESULTS_DIR/studies/"
    fi
}

# ── generate slurm ───────────────────────────────────────────────────────
need_regen_slurm() {
    (( REGEN_SLURM == 1 )) && return 0
    [[ ! -f "$HPC_DIR/submit_all.sh" ]] && return 0
    for p in $(echo "$PROBLEMS" | tr ',' ' '); do
        [[ ! -f "$HPC_DIR/tune_${p}.slurm" ]] && return 0
    done
    return 1
}

generate_slurm() {
    log "==== GENERATE SLURM SCRIPTS ===="
    if ! need_regen_slurm; then
        log "Slurm scripts already present; pass --regenerate-slurm to force."
        return
    fi
    local args=(
        --partition "$SLURM_PARTITION"
        --mem-per-cpu "$MEM_PER_CPU"
        --problems "$PROBLEMS"
        --out-dir "$HPC_DIR"
    )
    [[ -n "$MAX_CONCURRENT" ]] && args+=(--max-concurrent "$MAX_CONCURRENT")
    [[ -n "$MAIL_USER"      ]] && args+=(--mail-user "$MAIL_USER")
    [[ -n "$CPUS_PER_TASK"  ]] && args+=(--cpus-per-task "$CPUS_PER_TASK")
    run python "$HPC_DIR/generate_slurm.py" "${args[@]}"
}

# ── state file helpers ───────────────────────────────────────────────────
state_get() {
    # $1 = key (problem name); echoes job_id or empty string.
    [[ -f "$STATE_FILE" ]] || { echo ""; return; }
    python - <<PY
import json, sys
try:
    s = json.load(open(r"$STATE_FILE"))
except Exception:
    sys.exit(0)
print(s.get("jobs", {}).get(r"$1", ""))
PY
}

state_set() {
    # $1 = problem, $2 = job_id
    python - <<PY
import json, os
p = r"$STATE_FILE"
data = {}
if os.path.exists(p):
    try:
        data = json.load(open(p))
    except Exception:
        data = {}
data.setdefault("jobs", {})[r"$1"] = r"$2"
data["last_action"] = "submit"
import time
data["last_action_ts"] = time.time()
json.dump(data, open(p, "w"), indent=2)
PY
}

job_still_in_queue() {
    # $1 = job_id; returns 0 if it appears in squeue, else 1.
    [[ -z "$1" ]] && return 1
    squeue -h -j "$1" 2>/dev/null | grep -q '.' && return 0
    return 1
}

# ── init studies (avoid SQLite CREATE-TABLE race in Slurm arrays) ────────
init_studies() {
    # Each per-problem SQLite study is created here in a *single* process so
    # that, by the time the Slurm array launches N workers in parallel, the
    # schema already exists and ``optuna.create_study(..., load_if_exists=True)``
    # takes the load path instead of racing CREATE TABLE statements. Idempotent:
    # if the study is already initialised, this is a cheap load + no-op.
    log "==== INIT STUDIES (single-process schema setup) ===="
    activate_conda
    for p in $(echo "$PROBLEMS" | tr ',' ' '); do
        run python -m scripts.tuning.worker --problem "$p" --init-only
    done
}

# ── submit ───────────────────────────────────────────────────────────────
submit_jobs() {
    log "==== SUBMIT ===="
    if (( SKIP_SUBMIT == 1 )); then
        log "Submit phase skipped (--skip-submit or --finalise-only)."
        return
    fi
    local any_submitted=0
    for p in $(echo "$PROBLEMS" | tr ',' ' '); do
        local script="$HPC_DIR/tune_${p}.slurm"
        [[ -f "$script" ]] || fail "Missing slurm script: $script"
        local existing
        existing="$(state_get "$p")"
        if [[ -n "$existing" ]] && job_still_in_queue "$existing"; then
            log "Skip $p: job $existing still in queue."
            continue
        fi
        log "Submitting $p ($script)"
        if (( DRY_RUN == 1 )); then
            local job_id="DRY-$RANDOM"
        else
            local out
            out="$(REPO_ROOT="$REPO_ROOT" sbatch --parsable "$script")" \
                || fail "sbatch failed for $p"
            local job_id="${out%% *}"
        fi
        log "  -> job id $job_id"
        state_set "$p" "$job_id"
        any_submitted=1
    done
    if (( any_submitted == 0 )); then
        log "Nothing new to submit."
    fi
}

# ── wait ─────────────────────────────────────────────────────────────────
wait_jobs() {
    log "==== WAIT ===="
    if (( SKIP_WAIT == 1 )); then
        log "Wait phase skipped (--skip-wait or --finalise-only)."
        return
    fi
    [[ -f "$STATE_FILE" ]] || { log "No state file, nothing to wait on."; return; }
    local active=1
    while (( active > 0 )); do
        active=0
        local active_list=""
        for p in $(echo "$PROBLEMS" | tr ',' ' '); do
            local jid
            jid="$(state_get "$p")"
            if [[ -n "$jid" ]] && job_still_in_queue "$jid"; then
                active=$((active + 1))
                active_list+=" $p:$jid"
            fi
        done
        if (( active == 0 )); then
            log "All tuning jobs have left the queue."
            break
        fi
        log "Active:$active_list  (re-checking in ${POLL_INTERVAL_S}s)"
        sleep "$POLL_INTERVAL_S"
    done
}

# ── confirm + collect + report ───────────────────────────────────────────
finalise() {
    log "==== CONFIRM / COLLECT / REPORT ===="
    activate_conda  # idempotent
    for p in $(echo "$PROBLEMS" | tr ',' ' '); do
        local db="$RESULTS_DIR/studies/${p}.db"
        if [[ ! -f "$db" && -z "$ASTROMORF_OPTUNA_STORAGE" ]]; then
            warn "No SQLite study at $db; skipping $p (did the array fail?)"
            continue
        fi
        run python -m scripts.tuning.confirm --problem "$p" --k 5
        run python -m scripts.tuning.collect --problems "$p"
    done
    run python -m scripts.tuning.report --problems "$PROBLEMS"
}

# ── final summary ────────────────────────────────────────────────────────
print_outputs() {
    log "==== OUTPUTS ===="
    cat <<EOF
Recommendation report:
  $RESULTS_DIR/recommendations.md
  $RESULTS_DIR/recommendation_table.csv

Per-problem winners:
EOF
    for p in $(echo "$PROBLEMS" | tr ',' ' '); do
        echo "  $RESULTS_DIR/best/best_${p}.json"
    done
    cat <<EOF

Per-problem confirmation tables:
EOF
    for p in $(echo "$PROBLEMS" | tr ',' ' '); do
        echo "  $RESULTS_DIR/confirmations/confirm_${p}.json"
    done
    cat <<EOF

Per-problem study artefacts:
EOF
    for p in $(echo "$PROBLEMS" | tr ',' ' '); do
        echo "  $RESULTS_DIR/exports/${p}/trials.csv"
        echo "  $RESULTS_DIR/exports/${p}/top20.json"
        echo "  $RESULTS_DIR/studies/${p}.db   (Optuna SQLite, if not using Postgres)"
    done
    echo
    log "Done."
}

# ── main ─────────────────────────────────────────────────────────────────
preflight
generate_slurm
# Pre-init the Optuna storage *before* sbatch'ing so the array workers
# don't race CREATE TABLE on a fresh SQLite DB. Skipped in finalise-only.
if (( SKIP_SUBMIT == 0 )); then
    init_studies
fi
submit_jobs
wait_jobs
finalise
print_outputs
