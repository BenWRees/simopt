#!/usr/bin/env bash
# =============================================================================
# journal_experiments.sh — unified submitter for the journal experiment suite.
#
# This launcher submits THREE things, in two independent groups:
#
#   GROUP A — "base" workflow (legacy; unchanged):
#     • run_experiments.slurm          (full benchmark sweep)
#     • run_crn.slurm                  (CRN comparison)
#
#   GROUP B — "journal" workflow (new, manifest-driven):
#     1. python journal_generate_manifest.py …
#         → writes $OUTPUT_ROOT/manifest.json (single source of truth)
#     2. sbatch --array=0-N-1%CONCURRENCY run_journal_factors.slurm
#         → N derived from the manifest, never hard-coded
#     3. sbatch --dependency=afterany:<array-id> aggregate_journal.slurm
#         → runs paired-CI aggregation once the array drains
#
# Default (no args): both groups are submitted.  Use --mode to restrict.
#
# Usage:
#   scripts/journal_experiments.sh [options] [--mode all|base|journal]
#
# Common options:
#   --mode {all|base|journal}        Which group to submit       [all]
#   --dry-run                        Print sbatch commands; submit nothing
#   -h, --help                       Show this help
#
# Journal-group options (ignored in --mode base):
#   --output-root DIR                Root for the strict-isolation layout
#                                    (10 subdirs auto-created underneath:
#                                     {comparison,crn,subspace,poly_bases,
#                                      regularisation}_{logs,results})
#                                    [\$HOME/journal_experiment]
#   --problems "P1 P2 ..."           Problem list
#                                    [SAN-1 ROSENBROCK-1 DYNAMNEWS-1 NETWORK-1]
#   --dims  "P=D,P=D,..."            Per-problem dims
#                                    [all problems at d=100]
#   --studies "S1 S2 ..."            Restrict to one or more studies (any of
#                                    subspace / basis / regularisation).  When
#                                    set, --job-name becomes astromorf_<study>.
#                                    [empty = all three]
#   --mail-type TYPE                 Override #SBATCH --mail-type (e.g. FAIL).
#                                    [empty = honour SLURM directive]
#   --budget N                       Per-macrorep budget         [10000]
#   --n-macroreps N                  Macroreps per design point  [30]
#   --n-postreps N                   Postreps per recommended x  [600]
#   --concurrency N                  SLURM array concurrency (%) [64]
#   --walltime HH:MM:SS              Per-task walltime           [12:00:00]
#   --array-cpus N                   --cpus-per-task             [8]
#   --array-mem  SIZE                --mem                       [24G]
#   --partition  NAME                --partition                 [batch]
#   --conda-env  NAME                Conda env name              [simopt]
#   --skip-aggregator                Do not submit the dependency aggregator
#   --no-pyarrow-check               Skip pyarrow importability check
#   --split-by-study                 Submit the journal workflow as THREE
#                                    independent arrays (subspace / basis /
#                                    regularisation), each with IRIDIS-tuned
#                                    walltime / memory / % concurrency.
#                                    OUTPUT_ROOT/<study>/ holds each study's
#                                    manifest + runs + analysis.
#                                    Mutually exclusive with --studies.
#
# Examples:
#   # Default — submit base + journal
#   scripts/journal_experiments.sh
#
#   # Only journal, dim=100 everywhere, lower concurrency
#   scripts/journal_experiments.sh --mode journal --concurrency 32
#
#   # Dry-run on a custom problem list
#   scripts/journal_experiments.sh --mode journal --dry-run \\
#       --problems "SAN-1 ROSENBROCK-1" \\
#       --dims "SAN-1=20,ROSENBROCK-1=10"
# =============================================================================

set -Eeuo pipefail
# inherit_errexit requires bash >= 4.4; not present on macOS' system bash.
# The HPC cluster ships a modern bash so we enable it when available.
shopt -s inherit_errexit 2>/dev/null || true

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/.." && pwd)"

# ── Defaults ─────────────────────────────────────────────────────────────────
MODE="all"
DRY_RUN=0
SKIP_AGGREGATOR=0
SKIP_PYARROW_CHECK=0
SPLIT_BY_STUDY=0                      # IRIDIS-tuned per-study submission

# Per-study IRIDIS-tuned profile used when --split-by-study is set.  Each can
# be overridden via environment if needed.
SUBSPACE_CONCURRENCY="${SUBSPACE_CONCURRENCY:-16}"
SUBSPACE_WALLTIME="${SUBSPACE_WALLTIME:-24:00:00}"
SUBSPACE_MEM="${SUBSPACE_MEM:-32G}"
BASIS_CONCURRENCY="${BASIS_CONCURRENCY:-32}"
BASIS_WALLTIME="${BASIS_WALLTIME:-08:00:00}"
BASIS_MEM="${BASIS_MEM:-20G}"
REG_CONCURRENCY="${REG_CONCURRENCY:-32}"
REG_WALLTIME="${REG_WALLTIME:-04:00:00}"
REG_MEM="${REG_MEM:-16G}"

# Strict-isolation layout root — every experiment's logs + results live here.
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/journal_experiment}"
PROBLEMS="${PROBLEMS:-SAN-1 ROSENBROCK-1 DYNAMNEWS-1 NETWORK-1}"
DIMS_SPEC=""                          # built from PROBLEMS if empty
STUDIES=""                            # empty = manifest generator's default (all)
MAIL_TYPE=""                          # empty = honour #SBATCH directive
BUDGET="${BUDGET:-10000}"
N_MACROREPS="${N_MACROREPS:-30}"
N_POSTREPS="${N_POSTREPS:-600}"
CONCURRENCY="${CONCURRENCY:-64}"
WALLTIME="${WALLTIME:-12:00:00}"
ARRAY_CPUS="${ARRAY_CPUS:-8}"
ARRAY_MEM="${ARRAY_MEM:-24G}"
PARTITION="${PARTITION:-batch}"
CONDA_ENV="${CONDA_ENV:-simopt}"

PYTHON="${PYTHON:-python}"

# ── Helpers ──────────────────────────────────────────────────────────────────
log()  { printf "[%s] %s\n" "$(date -u +%FT%TZ)" "$*"; }
die()  { printf "ERROR: %s\n" "$*" >&2; exit 1; }
note() { printf "  • %s\n" "$*"; }

usage() {
    sed -n '2,/^# ===/{ /^# ===$/q; s/^# \{0,1\}//; p; }' "$0" \
        | sed '/^$/q' >/dev/null  # no-op; we keep the docstring for readers
    sed -n '/^# Usage:/,/^# ===/{ /^# ===$/q; s/^# \{0,1\}//; p; }' "$0"
}

require_file() {
    [[ -f "$1" ]] || die "required file not found: $1"
}

# Build the default dims string ("P1=100,P2=100,...") from $PROBLEMS when the
# user did not pass --dims explicitly.
default_dims_for() {
    local p out=""
    for p in $1; do
        out+="${p}=100,"
    done
    printf "%s" "${out%,}"
}

# Submit-or-print: honours --dry-run and returns the sbatch jobid on stdout.
# ── Strict-isolation filesystem layout (single source of truth) ─────────────
# Every experiment is mapped to exactly one logs/ and one results/ directory
# under $OUTPUT_ROOT.  Cross-experiment writes are not possible because the
# launcher always derives both paths from the canonical study key — no
# wildcards, no shared prefixes, no fallbacks.
#
#   key            logs subdir              results subdir
#   ─────────────  ───────────────────────  ────────────────────────────
#   comparison     comparison_logs          comparison_results
#   crn            crn_logs                 crn_results
#   subspace       subspace_logs            subspace_results
#   basis          poly_bases_logs          poly_bases_results
#   regularisation regularisation_logs      regularisation_results
ISOLATION_KEYS=(comparison crn subspace basis regularisation)

# Map study/experiment key → on-disk subdir prefix.  Only this function knows
# the mapping; everything else routes through it.  No call-site is allowed to
# concatenate paths from raw study names.
isolation_subdir_for() {
    case "$1" in
        comparison)     echo "comparison" ;;
        crn)            echo "crn" ;;
        subspace)       echo "subspace" ;;
        basis)          echo "poly_bases" ;;        # journal study key ≠ dir name
        regularisation) echo "regularisation" ;;
        *) die "unknown experiment key: $1" ;;
    esac
}

logs_dir_for()    { echo "$OUTPUT_ROOT/$(isolation_subdir_for "$1")_logs"; }
results_dir_for() { echo "$OUTPUT_ROOT/$(isolation_subdir_for "$1")_results"; }

# Create the entire 10-subdir layout under $OUTPUT_ROOT idempotently.  Called
# at startup; safe to re-invoke (mkdir -p is a no-op when the dir exists, and
# we never delete anything here).
ensure_isolated_layout() {
    mkdir -p "$OUTPUT_ROOT"
    local k sub
    for k in "${ISOLATION_KEYS[@]}"; do
        sub=$(isolation_subdir_for "$k")
        mkdir -p "$OUTPUT_ROOT/${sub}_logs" "$OUTPUT_ROOT/${sub}_results"
    done
}

maybe_sbatch_parsable() {
    if [[ $DRY_RUN -eq 1 ]]; then
        printf "DRYRUN_JOBID_%s\n" "$RANDOM"
        printf "  sbatch %s\n" "$*" >&2
        return 0
    fi
    sbatch --parsable "$@"
}

# ── Argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)              MODE="$2"; shift 2 ;;
        --dry-run)           DRY_RUN=1; shift ;;
        --output-root)       OUTPUT_ROOT="$2"; shift 2 ;;
        --problems)          PROBLEMS="$2"; shift 2 ;;
        --dims)              DIMS_SPEC="$2"; shift 2 ;;
        --studies)           STUDIES="$2"; shift 2 ;;
        --mail-type)         MAIL_TYPE="$2"; shift 2 ;;
        --budget)            BUDGET="$2"; shift 2 ;;
        --n-macroreps)       N_MACROREPS="$2"; shift 2 ;;
        --n-postreps)        N_POSTREPS="$2"; shift 2 ;;
        --concurrency)       CONCURRENCY="$2"; shift 2 ;;
        --walltime)          WALLTIME="$2"; shift 2 ;;
        --array-cpus)        ARRAY_CPUS="$2"; shift 2 ;;
        --array-mem)         ARRAY_MEM="$2"; shift 2 ;;
        --partition)         PARTITION="$2"; shift 2 ;;
        --conda-env)         CONDA_ENV="$2"; shift 2 ;;
        --skip-aggregator)   SKIP_AGGREGATOR=1; shift ;;
        --split-by-study)    SPLIT_BY_STUDY=1; shift ;;
        --no-pyarrow-check)  SKIP_PYARROW_CHECK=1; shift ;;
        -h|--help)           usage; exit 0 ;;
        *)                   die "unknown argument: $1 (try --help)" ;;
    esac
done

case "$MODE" in
    all|base|journal) ;;
    *) die "--mode must be one of: all, base, journal (got $MODE)" ;;
esac

# ── Pre-flight checks ────────────────────────────────────────────────────────
if ! command -v sbatch >/dev/null 2>&1 && [[ $DRY_RUN -eq 0 ]]; then
    die "sbatch not found in PATH (use --dry-run to test locally)"
fi

# Canonicalise --output-root.  A bare relative path (e.g. "sens_anal") would
# otherwise be created under whatever the launcher's CWD is — typically $HOME
# when invoked interactively — which silently violates the "everything under
# ~/journal_experiment/" spec.  Resolution rules:
#   • Absolute path        → use as-is (allows e.g. /scratch/$USER/journal).
#   • Path starting with ~ → expanded via the shell.
#   • Anything else        → anchored under $HOME/journal_experiment/<that>,
#                             with a loud warning so the user sees what changed.
case "$OUTPUT_ROOT" in
    /*) ;;                                  # already absolute
    "~"|"~/"*) OUTPUT_ROOT="${OUTPUT_ROOT/#\~/$HOME}" ;;
    *)
        _raw="$OUTPUT_ROOT"
        OUTPUT_ROOT="$HOME/journal_experiment/$_raw"
        log "WARN: --output-root '$_raw' is relative; anchored to $OUTPUT_ROOT"
        log "      (pass an absolute path with --output-root to override)"
        unset _raw
        ;;
esac

# Activate the simopt conda env so subsequent python -c invocations (notably
# the pyarrow importability check) use the same interpreter the SLURM jobs
# will use.  Idempotent: a no-op when the env is already active.  Silent when
# conda is not in PATH (defers any failure to the pyarrow check itself, which
# will produce a clearer error message).
activate_conda_env() {
    if [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV" ]]; then
        return
    fi
    local conda_base
    conda_base="$(conda info --base 2>/dev/null || true)"
    if [[ -z "$conda_base" || ! -f "$conda_base/etc/profile.d/conda.sh" ]]; then
        return
    fi
    # shellcheck disable=SC1091
    source "$conda_base/etc/profile.d/conda.sh"
    if conda activate "$CONDA_ENV" 2>/dev/null; then
        log "activated conda env '$CONDA_ENV' (python: $(command -v python))"
        PYTHON="$(command -v python)"
    else
        log "WARN: could not activate conda env '$CONDA_ENV'; PATH-resolved python will be used"
    fi
}
activate_conda_env

# Create the strict-isolation layout under $OUTPUT_ROOT unconditionally.  This
# is idempotent (mkdir -p) and never deletes anything — re-running the
# launcher leaves prior results intact under their per-experiment subtrees.
ensure_isolated_layout
log "isolation root: $OUTPUT_ROOT"
log "  layout: {comparison,crn,subspace,poly_bases,regularisation}_{logs,results}"

# ── GROUP A: legacy base workflow (comparison + CRN) ─────────────────────────
# Each legacy job is forced into its own isolated log + result directory via
# explicit `--output`, `--error`, and `OUTPUT_DIR` overrides at submit time.
# The underlying SLURM scripts are unchanged; the launcher rewrites the paths
# that those scripts honour.
submit_base_workflow() {
    log "[base] submitting legacy SLURM jobs into isolated layout"
    # (job script,     experiment key)
    local pairs=(
        "run_experiments.slurm:comparison"
        "run_crn.slurm:crn"
    )
    local pair job key path logs res
    for pair in "${pairs[@]}"; do
        job="${pair%:*}"; key="${pair##*:}"
        path="$DIR/$job"
        require_file "$path"
        logs=$(logs_dir_for    "$key")
        res=$( results_dir_for "$key")
        log "[base/$key] logs->$logs  results->$res"
        if [[ $DRY_RUN -eq 1 ]]; then
            printf "  sbatch --output=%s/slurm_%%A_%%a.out --error=%s/slurm_%%A_%%a.err --export=ALL,OUTPUT_DIR=%s %s\n" \
                "$logs" "$logs" "$res" "$path"
            continue
        fi
        local out
        out=$(sbatch \
                --job-name="simopt_$key" \
                --output="$logs/slurm_%A_%a.out" \
                --error="$logs/slurm_%A_%a.err" \
                --export="ALL,OUTPUT_DIR=$res" \
                "$path" 2>&1) \
            || die "sbatch failed for $job ($key): $out"
        if [[ $out =~ ([0-9]+) ]]; then
            note "submitted $job ($key) -> job ${BASH_REMATCH[1]}"
        else
            note "submitted $job ($key) -> response: $out"
        fi
    done
}

# ── GROUP B: journal sensitivity workflow ────────────────────────────────────
submit_journal_workflow() {
    log "[journal] preparing manifest-driven sensitivity workflow"

    local manifest_script="$DIR/journal_generate_manifest.py"
    local dispatch_script="$DIR/journal_dispatch.py"
    local exec_script="$DIR/journal_factors_test.py"
    local slurm_array="$DIR/run_journal_factors.slurm"
    local slurm_agg="$DIR/aggregate_journal.slurm"
    require_file "$manifest_script"
    require_file "$dispatch_script"
    require_file "$exec_script"
    require_file "$slurm_array"
    require_file "$slurm_agg"

    # pyarrow check — long_form.parquet is the canonical artefact; the
    # journal aggregator transparently falls back to .csv.gz when pyarrow is
    # missing, but for journal-grade runs we want the typed Parquet output.
    if [[ $SKIP_PYARROW_CHECK -eq 0 ]]; then
        if ! "$PYTHON" -c "import pyarrow" >/dev/null 2>&1; then
            die "pyarrow not importable in this environment; install with \
'pip install pyarrow' or pass --no-pyarrow-check to proceed with CSV.gz \
fallback (lower compression, lower fidelity for downstream analysis)"
        fi
        note "pyarrow OK — long_form.parquet will be produced"
    fi

    # Resolve dims default if user didn't pass one.
    if [[ -z "$DIMS_SPEC" ]]; then
        DIMS_SPEC="$(default_dims_for "$PROBLEMS")"
    fi

    # Strict isolation: derive logs/results paths from the study key the
    # caller installed in $JOURNAL_EXP_KEY.  When unset (legacy call paths,
    # e.g. --studies naming a single study and going straight here), fall
    # back to deriving from $STUDIES when it is unambiguous.
    local exp_key="${JOURNAL_EXP_KEY:-}"
    if [[ -z "$exp_key" ]]; then
        if [[ -n "$STUDIES" && \
              "$(echo "$STUDIES" | wc -w | tr -d ' ')" == "1" ]]; then
            exp_key="$STUDIES"
        else
            die "submit_journal_workflow called without an isolated experiment \
key (single-study run required).  Use --studies <one of: subspace, basis, \
regularisation>, or run the launcher without --studies to auto-split."
        fi
    fi
    local logs_root results_root
    logs_root=$(   logs_dir_for    "$exp_key")
    results_root=$(results_dir_for "$exp_key")
    # Each study's results live entirely under its own results_dir.  The
    # SLURM array passes $OUTPUT_ROOT=$results_root to the per-task workers
    # and the manifest writes under $results_root/runs/<problem>/<study>/...
    # so cross-study writes are physically impossible.
    OUTPUT_ROOT="$results_root"
    mkdir -p "$results_root" "$results_root/runs" "$results_root/analysis" \
             "$logs_root"

    note "exp key      : $exp_key"
    note "logs dir     : $logs_root"
    note "results dir  : $results_root"
    note "problems     : $PROBLEMS"
    note "dims         : $DIMS_SPEC"
    note "budget       : $BUDGET"
    note "n_macroreps  : $N_MACROREPS"
    note "n_postreps   : $N_POSTREPS"
    note "concurrency  : %$CONCURRENCY"
    note "walltime     : $WALLTIME"
    note "cpus / mem   : $ARRAY_CPUS cpu, $ARRAY_MEM"

    # Step 1 — manifest.
    log "[journal] generating manifest"
    local gen_cmd=(
        "$PYTHON" "$manifest_script"
        --output-root "$OUTPUT_ROOT"
        --problems    $PROBLEMS
        --dims        "$DIMS_SPEC"
        --budget      "$BUDGET"
        --n-macroreps "$N_MACROREPS"
        --n-postreps  "$N_POSTREPS"
    )
    # --studies is a pure pass-through; empty = manifest generator default
    # (every study).  IRIDIS production uses one launcher call per study so
    # each study gets its own walltime / memory / concurrency profile.
    if [[ -n "$STUDIES" ]]; then
        gen_cmd+=(--studies $STUDIES)
        note "studies      : $STUDIES"
    fi
    if [[ $DRY_RUN -eq 1 ]]; then
        printf "  %q " "${gen_cmd[@]}"; echo
    else
        "${gen_cmd[@]}" || die "manifest generation failed"
    fi

    local manifest="$OUTPUT_ROOT/manifest.json"
    if [[ $DRY_RUN -eq 0 ]]; then
        [[ -f "$manifest" ]] || die "manifest was not written: $manifest"
    fi

    # Step 2 — pull total task count from the manifest (single source of truth).
    local total
    if [[ $DRY_RUN -eq 1 && ! -f "$manifest" ]]; then
        total="UNKNOWN"
    else
        total=$("$PYTHON" -c "
import json, sys
try:
    m = json.load(open('$manifest'))
    t = int(m['total_tasks'])
    assert t > 0, 'total_tasks must be > 0'
    print(t)
except Exception as e:
    sys.stderr.write(f'manifest validation failed: {e}\n')
    sys.exit(1)
") || die "could not read total_tasks from $manifest"
        log "[journal] manifest reports $total tasks"
    fi

    # Step 3 — submit the array, sized from the manifest.
    log "[journal] submitting sensitivity array"
    local array_spec
    if [[ "$total" == "UNKNOWN" ]]; then
        array_spec="0-N_MINUS_1%$CONCURRENCY"
    else
        array_spec="0-$((total - 1))%$CONCURRENCY"
    fi
    local array_export="ALL,MANIFEST=$manifest,OUTPUT_ROOT=$OUTPUT_ROOT,REPO_ROOT=$REPO_ROOT,CONDA_ENV=$CONDA_ENV"

    # --mail-type passthrough (overrides the #SBATCH directive when set).
    local mail_args=()
    if [[ -n "$MAIL_TYPE" ]]; then
        mail_args=(--mail-type "$MAIL_TYPE")
    fi
    # Derive a per-study job name so squeue lines are self-identifying when
    # the launcher is invoked once per study.
    local job_name="astromorf_journal"
    if [[ -n "$STUDIES" && "$(echo "$STUDIES" | wc -w | tr -d ' ')" == "1" ]]; then
        job_name="astromorf_${STUDIES}"
    fi

    local array_job
    array_job=$(maybe_sbatch_parsable \
        --job-name "$job_name" \
        --partition "$PARTITION" \
        --cpus-per-task "$ARRAY_CPUS" \
        --mem "$ARRAY_MEM" \
        --time "$WALLTIME" \
        --array "$array_spec" \
        --export "$array_export" \
        --output "$logs_root/slurm_%A_%a.out" \
        --error  "$logs_root/slurm_%A_%a.err" \
        ${mail_args[@]+"${mail_args[@]}"} \
        "$slurm_array")
    note "submitted run_journal_factors.slurm -> array job $array_job"

    # Step 4 — submit the aggregator with afterany dependency.
    if [[ $SKIP_AGGREGATOR -eq 1 ]]; then
        note "aggregator skipped (--skip-aggregator)"
    else
        log "[journal] submitting aggregator (depends on $array_job)"
        local agg_export="ALL,OUTPUT_ROOT=$OUTPUT_ROOT,REPO_ROOT=$REPO_ROOT,CONDA_ENV=$CONDA_ENV"
        local agg_job
        local agg_name="agg_${job_name#astromorf_}"
        agg_job=$(maybe_sbatch_parsable \
            --job-name "$agg_name" \
            --partition "$PARTITION" \
            --dependency "afterany:$array_job" \
            --kill-on-invalid-dep=yes \
            --export "$agg_export" \
            --output "$logs_root/agg_%j.out" \
            --error  "$logs_root/agg_%j.err" \
            ${mail_args[@]+"${mail_args[@]}"} \
            "$slurm_agg")
        note "submitted aggregate_journal.slurm -> job $agg_job"
    fi

    log "[journal/$exp_key] outputs will land under:"
    note "  manifest       : $manifest"
    note "  per-task runs  : $results_root/runs/<problem>/<study>/<dpid>/"
    note "  analysis CSVs  : $results_root/analysis/"
    note "  slurm logs     : $logs_root/{slurm,agg}_*.out"
}

# ── Journal dispatch: always isolated, always one experiment per array ──────
# Strict isolation is enforced here: every journal-sensitivity SLURM array is
# scoped to exactly one study (subspace / basis / regularisation), which maps
# to its own logs_dir + results_dir.  A single unified array covering >1
# study is forbidden (it would have to write to one results dir while
# carrying data from multiple experiments, breaking the isolation guarantee).
#
# Behaviour:
#   --studies <one study>   ⇒ run just that study, in its isolated paths.
#   --studies <multiple>    ⇒ run each in its own isolated paths sequentially.
#   --studies <unset>       ⇒ run all three in their isolated paths
#                              (the IRIDIS-tuned default).
# The legacy --split-by-study flag is preserved as a no-op for backward
# compatibility, since splitting is now unconditional.
submit_journal_dispatch() {
    local base_root="$OUTPUT_ROOT"
    local saved_studies="$STUDIES"
    local saved_concurrency="$CONCURRENCY"
    local saved_walltime="$WALLTIME"
    local saved_mem="$ARRAY_MEM"

    # Resolve the list of studies to run.  Empty $STUDIES means "all three";
    # any explicit list is honoured verbatim.
    local to_run="$STUDIES"
    if [[ -z "$to_run" ]]; then
        to_run="subspace basis regularisation"
    fi

    # Per-study IRIDIS-tuned profile lookup.
    _study_profile() {
        case "$1" in
            subspace)
                echo "$SUBSPACE_CONCURRENCY $SUBSPACE_WALLTIME $SUBSPACE_MEM" ;;
            basis)
                echo "$BASIS_CONCURRENCY $BASIS_WALLTIME $BASIS_MEM" ;;
            regularisation)
                echo "$REG_CONCURRENCY $REG_WALLTIME $REG_MEM" ;;
            *) die "unknown study: $1" ;;
        esac
    }

    local study profile pct wall mem
    for study in $to_run; do
        read -r pct wall mem <<<"$(_study_profile "$study")"
        # Reset per-study globals.  $JOURNAL_EXP_KEY drives the strict-
        # isolation paths inside submit_journal_workflow.
        JOURNAL_EXP_KEY="$study"
        STUDIES="$study"
        OUTPUT_ROOT="$base_root"           # workflow rewrites to results_dir
        CONCURRENCY="$pct"
        WALLTIME="$wall"
        ARRAY_MEM="$mem"
        echo
        log "[journal/$study] %=$pct  time=$wall  mem=$mem"
        submit_journal_workflow
    done

    # Restore (defensive; nothing else reads these after dispatch).
    OUTPUT_ROOT="$base_root"
    STUDIES="$saved_studies"
    CONCURRENCY="$saved_concurrency"
    WALLTIME="$saved_walltime"
    ARRAY_MEM="$saved_mem"
    unset JOURNAL_EXP_KEY
}

# ── Dispatch on mode ─────────────────────────────────────────────────────────
case "$MODE" in
    base)
        submit_base_workflow
        ;;
    journal)
        submit_journal_dispatch
        ;;
    all)
        submit_base_workflow
        echo
        submit_journal_dispatch
        ;;
esac

log "done."
