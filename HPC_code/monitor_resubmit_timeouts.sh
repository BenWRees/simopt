#!/usr/bin/env bash
set -euo pipefail

# monitor_resubmit_timeouts.sh
# Submit SLURM job files, monitor final states, and resubmit only TIMEOUTed jobs.
# Usage:
#   monitor_resubmit_timeouts.sh <slurm_dir> [user] [poll_secs] [max_resubmissions]
#
# By default the script will process all '*.slurm' files in <slurm_dir>.
# To restrict which files are submitted/monitored, set the environment
# variable `MONITOR_PATTERN` to a glob (for example
# `MONITOR_PATTERN=journal_setup_OMoRF_*`). If unset, defaults to '*.slurm'.
#
# Example:
#   MONITOR_PATTERN=journal_setup_OMoRF_* ./monitor_resubmit_timeouts.sh simopt/HPC_code/generated_slurm_files br7g18 60 0

SLURM_DIR="${1:-}"
USER_TO_WATCH="${2:-$(whoami)}"
POLL_SECS="${3:-60}"
MAX_RESUBS="${4:-0}"
# Pattern to match files in the directory. Can be set with env var MONITOR_PATTERN
MONITOR_PATTERN="${MONITOR_PATTERN:-*.slurm}"

if [ -z "$SLURM_DIR" ]; then
    echo "Usage: $0 <slurm_dir> [user] [poll_secs] [max_resubmissions]"
    exit 1
fi

if [ ! -d "$SLURM_DIR" ]; then
    echo "Directory not found: $SLURM_DIR"
    exit 1
fi

declare -A JOBID_BY_FILE
declare -A RESUB_COUNT

submit_all() {
    local f out jid
    while IFS= read -r -d '' f; do
        # Determine job-name from the slurm file (fallback to basename)
        job_name=$(grep -m1 '^#SBATCH --job-name=' "$f" 2>/dev/null | sed -E 's/^#SBATCH --job-name=//')
        if [ -z "$job_name" ]; then
            # strip extension
            job_name=$(basename "$f" .slurm)
        fi

        # 1) Check squeue for an active job with this name for the watched user
        jid=$(squeue -u "$USER_TO_WATCH" -h -o "%A|%j" 2>/dev/null | awk -F'|' -v name="$job_name" '$2==name {print $1}' | tail -n1)
        if [ -n "$jid" ]; then
            JOBID_BY_FILE["$f"]="$jid"
            RESUB_COUNT["$f"]=0
            echo "Attached to existing queued job ${jid} for $(basename "$f") (job-name=${job_name})"
            continue
        fi

        # 2) If no active job, check sacct for the most recent job with this name
        sacct_line=$(sacct -u "$USER_TO_WATCH" -n -P -o JobID,State,JobName 2>/dev/null | awk -F'|' -v name="$job_name" 'index($3,name){print $1"|"$2}' | tail -n1)
        if [ -n "$sacct_line" ]; then
            prev_jid=$(echo "$sacct_line" | cut -d'|' -f1)
            prev_state=$(echo "$sacct_line" | cut -d'|' -f2)
            upper_prev_state=$(echo "$prev_state" | tr '[:lower:]' '[:upper:]')

            if [[ "$upper_prev_state" == *"TIMEOUT"* ]]; then
                # Previous run timed out; resubmit
                echo "Previous job ${prev_jid} for $(basename "$f") had state=${prev_state}; resubmitting."
                out=$(sbatch "$f" 2>&1) || { echo "Failed to resubmit $f: $out"; continue; }
                jid=$(echo "$out" | grep -oE '[0-9]+' | tail -n1)
                JOBID_BY_FILE["$f"]="$jid"
                RESUB_COUNT["$f"]=1
                echo "Resubmitted $(basename "$f") -> ${jid} (was ${prev_jid})"
                continue
            fi

            if [[ "$upper_prev_state" == *"COMPLETED"* ]] || [[ "$upper_prev_state" == *"CANCELLED"* ]]; then
                echo "Found previous job ${prev_jid} for $(basename "$f") with terminal state=${prev_state}; not submitting."
                # do not add to watch list
                continue
            fi

            # For other terminal states, leave decision to monitoring loop (do not auto-resubmit here)
            echo "Found previous job ${prev_jid} for $(basename "$f") with state=${prev_state}; submitting new job."
        fi

        # 3) No active or relevant previous job found: submit a fresh job
        out=$(sbatch "$f" 2>&1) || { echo "Failed to submit $f: $out"; continue; }
        jid=$(echo "$out" | grep -oE '[0-9]+' | tail -n1)
        JOBID_BY_FILE["$f"]="$jid"
        RESUB_COUNT["$f"]=0
        echo "Submitted $(basename "$f") -> ${jid}"
    done < <(find "$SLURM_DIR" -maxdepth 1 -type f -name "$MONITOR_PATTERN" -print0)
}

get_final_state() {
    # Takes jobid, prints state (or empty)
    local jid="$1"
    local sacct_out line state
    # Try sacct first (may require account access)
    sacct_out=$(sacct -j "$jid" -o State -n -P 2>/dev/null || true)
    if [ -n "$sacct_out" ]; then
        # Take first non-empty line
        line=$(printf "%s\n" "$sacct_out" | sed -n '1p')
        state=$(echo "$line" | cut -d'|' -f1)
        echo "$state"
        return
    fi

    # Fallback to scontrol
    sctrl_out=$(scontrol show job "$jid" 2>/dev/null || true)
    if [ -n "$sctrl_out" ]; then
        state=$(echo "$sctrl_out" | sed -n 's/.*JobState=\([^ ]*\).*/\1/p' | head -n1)
        echo "$state"
        return
    fi

    echo ""
}

is_in_queue() {
    local jid="$1"
    squeue -j "$jid" -h 2>/dev/null | grep -q .
}

resubmit_file() {
    local f="$1" out newjid
    out=$(sbatch "$f" 2>&1) || { echo "Resubmit failed for $f: $out"; return 1; }
    newjid=$(echo "$out" | grep -oE '[0-9]+' | tail -n1)
    JOBID_BY_FILE["$f"]="$newjid"
    echo "Resubmitted $(basename "$f") -> ${newjid} (count=${RESUB_COUNT[$f]})"
}

echo "Submitting all slurm files from: $SLURM_DIR"
submit_all

echo "Monitoring jobs for TIMEOUT (user: $USER_TO_WATCH). Poll every ${POLL_SECS}s. Max resubmissions: ${MAX_RESUBS}" 

while [ ${#JOBID_BY_FILE[@]} -gt 0 ]; do
    for f in "${!JOBID_BY_FILE[@]}"; do
        jid=${JOBID_BY_FILE["$f"]}

        # If still in squeue, skip
        if is_in_queue "$jid"; then
            echo "Job ${jid} ($(basename "$f")) still queued."
            continue
        fi

        # Not in queue: fetch final state (retry sacct a few times if empty)
        state=""
        retries=0
        while [ -z "$state" ] && [ $retries -lt 3 ]; do
            state=$(get_final_state "$jid")
            if [ -z "$state" ]; then
                sleep 2
            fi
            retries=$((retries+1))
        done

        if [ -z "$state" ]; then
            echo "Could not determine final state for job $jid (file $(basename "$f")). Removing from watch list." 
            unset JOBID_BY_FILE["$f"]
            unset RESUB_COUNT["$f"]
            continue
        fi

        echo "Job ${jid} ($(basename "$f")) finished with state=${state}"

        upper=$(echo "$state" | tr '[:lower:]' '[:upper:]')

        if [[ "$upper" == *"TIMEOUT"* ]]; then
            # increment and check max
            RESUB_COUNT["$f"]=$((RESUB_COUNT["$f"] + 1))
            if [ "$MAX_RESUBS" -ne 0 ] && [ "${RESUB_COUNT[$f]}" -gt "$MAX_RESUBS" ]; then
                echo "Max resubmissions reached for $(basename "$f"); not resubmitting."
                unset JOBID_BY_FILE["$f"]
                unset RESUB_COUNT["$f"]
                continue
            fi

            echo "Resubmitting $(basename "$f") due to TIMEOUT (previous job $jid)."
            if resubmit_file "$f"; then
                # resubmitted; continue watching new jobid
                continue
            else
                echo "Resubmit attempt failed for $(basename "$f"); removing from watch list." 
                unset JOBID_BY_FILE["$f"]
                unset RESUB_COUNT["$f"]
                continue
            fi
        fi

        # If completed or cancelled, do not resubmit
        if [[ "$upper" == *"COMPLETED"* ]] || [[ "$upper" == *"CANCELLED"* ]] || [[ "$upper" == *"CANCELLED+"* ]]; then
            echo "Job ${jid} ($(basename "$f")) finished with ${state}; not resubmitting."
            unset JOBID_BY_FILE["$f"]
            unset RESUB_COUNT["$f"]
            continue
        fi

        # Other terminal states: log and remove
        if [[ "$upper" == *"FAILED"* ]] || [[ "$upper" == *"NODE_FAIL"* ]] || [[ "$upper" == *"OUT_OF_MEMORY"* ]] || [[ "$upper" == *"PREEMPTED"* ]]; then
            echo "Job ${jid} ($(basename "$f")) ended with ${state}; not resubmitting."
            unset JOBID_BY_FILE["$f"]
            unset RESUB_COUNT["$f"]
            continue
        fi

        # Unknown state
        echo "Job ${jid} ($(basename "$f")) ended with unhandled state=${state}; not resubmitting."
        unset JOBID_BY_FILE["$f"]
        unset RESUB_COUNT["$f"]
    done

    if [ ${#JOBID_BY_FILE[@]} -gt 0 ]; then
        sleep "$POLL_SECS"
    fi
done

echo "Monitoring complete; no remaining jobs to watch."
