#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Simopt Auto Commit Script
# Formats, lints, type-checks, stages, commits, and emails a report.
# =============================================================================

REPO_DIR="$HOME/Desktop/simopt"
EMAIL_TO="ben.rees@me.com"
EMAIL_SUBJECT="Simopt Auto Commit Report"
TIMESTAMP="$(date '+%Y-%m-%d %H:%M:%S')"

# Temporary log files
COMMITTED_LOG="$(mktemp)"
RUFF_FAIL_LOG="$(mktemp)"
TY_FAIL_LOG="$(mktemp)"
BOTH_FAIL_LOG="$(mktemp)"

trap 'rm -f "$COMMITTED_LOG" "$RUFF_FAIL_LOG" "$TY_FAIL_LOG" "$BOTH_FAIL_LOG"' EXIT

# -----------------------------------------------------------------------------
# Pre-flight: ensure required tools exist
# -----------------------------------------------------------------------------
check_tools() {
    local missing=0
    for tool in git ruff ty; do
        if ! command -v "$tool" &>/dev/null; then
            echo "ERROR: '$tool' is not installed or not on PATH." >&2
            missing=1
        fi
    done
    if (( missing )); then
        exit 1
    fi
}

# -----------------------------------------------------------------------------
# Collect modified-but-unstaged files (porcelain v1: ' M' in first two cols)
# -----------------------------------------------------------------------------
get_modified_unstaged_files() {
    git -C "$REPO_DIR" status --porcelain=v1 \
        | awk '$1 == "M" || $1 == "MM" || ($1 == "" && $2 == "M") { print substr($0, 4) }' \
        | while IFS= read -r rel; do
              # status --porcelain uses ' M' (space then M) for unstaged mods
              printf '%s\n' "$rel"
          done
}

# -----------------------------------------------------------------------------
# Format a single file with ruff
# -----------------------------------------------------------------------------
format_file() {
    local filepath="$1"
    ruff format "$filepath" 2>/dev/null || true
}

# -----------------------------------------------------------------------------
# Lint a single file with ruff check; capture output
# Returns 0 on success, 1 on failure
# -----------------------------------------------------------------------------
lint_file() {
    local filepath="$1"
    local output
    if output="$(ruff check "$filepath" 2>&1)"; then
        return 0
    else
        echo "$output"
        return 1
    fi
}

# -----------------------------------------------------------------------------
# Type-check a single file with ty; capture output
# Returns 0 on success, 1 on failure
# -----------------------------------------------------------------------------
typecheck_file() {
    local filepath="$1"
    local output
    if output="$(ty check "$filepath" 2>&1)"; then
        return 0
    else
        echo "$output"
        return 1
    fi
}

# -----------------------------------------------------------------------------
# Log a result to the appropriate temp file
# -----------------------------------------------------------------------------
log_committed() {
    local filepath="$1"
    echo "$filepath" >> "$COMMITTED_LOG"
}

log_failure() {
    local filepath="$1"
    local category_log="$2"   # path to the relevant temp file
    local errors="$3"
    {
        echo "--- $filepath ---"
        echo "$errors"
        echo ""
    } >> "$category_log"
}

# -----------------------------------------------------------------------------
# Process a single file: format → lint → typecheck → stage or log
# -----------------------------------------------------------------------------
process_file() {
    local relpath="$1"
    local filepath="$REPO_DIR/$relpath"

    # Only process Python files
    if [[ "$filepath" != *.py ]]; then
        return
    fi

    echo "  Processing: $relpath"

    # Step 1: auto-format
    format_file "$filepath"

    # Step 2: lint
    local ruff_ok=true ruff_output=""
    if ! ruff_output="$(lint_file "$filepath")"; then
        ruff_ok=false
    fi

    # Step 3: type-check
    local ty_ok=true ty_output=""
    if ! ty_output="$(typecheck_file "$filepath")"; then
        ty_ok=false
    fi

    # Step 4: decide
    if $ruff_ok && $ty_ok; then
        git -C "$REPO_DIR" add -- "$relpath"
        log_committed "$relpath"
        echo "    ✓ staged"
    elif ! $ruff_ok && ! $ty_ok; then
        log_failure "$relpath" "$BOTH_FAIL_LOG" \
            "RUFF ERRORS:
$ruff_output

TY ERRORS:
$ty_output"
        echo "    ✗ ruff + ty failed"
    elif ! $ruff_ok; then
        log_failure "$relpath" "$RUFF_FAIL_LOG" "$ruff_output"
        echo "    ✗ ruff failed"
    else
        log_failure "$relpath" "$TY_FAIL_LOG" "$ty_output"
        echo "    ✗ ty failed"
    fi
}

# -----------------------------------------------------------------------------
# Commit staged files (if any)
# -----------------------------------------------------------------------------
do_commit() {
    if git -C "$REPO_DIR" diff --cached --quiet; then
        echo "No files staged — skipping commit."
        return 1
    else
        git -C "$REPO_DIR" commit -m "autocommit"
        echo "Commit created."
        git -C "$REPO_DIR" push -u origin master
        echo "Pushed to origin/master."
        return 0
    fi
}

# -----------------------------------------------------------------------------
# Build and send the email report
# -----------------------------------------------------------------------------
send_report() {
    local body=""

    body+="Simopt Auto Commit Report"$'\n'
    body+="Generated: $TIMESTAMP"$'\n'
    body+="=========================================="$'\n\n'

    # Committed files
    body+="SUCCESSFULLY COMMITTED FILES"$'\n'
    body+="------------------------------------------"$'\n'
    if [[ -s "$COMMITTED_LOG" ]]; then
        body+="$(cat "$COMMITTED_LOG")"$'\n'
    else
        body+="(none)"$'\n'
    fi
    body+=$'\n'

    # Ruff-only failures
    body+="FILES FAILED — RUFF ONLY"$'\n'
    body+="------------------------------------------"$'\n'
    if [[ -s "$RUFF_FAIL_LOG" ]]; then
        body+="$(cat "$RUFF_FAIL_LOG")"$'\n'
    else
        body+="(none)"$'\n'
    fi
    body+=$'\n'

    # Ty-only failures
    body+="FILES FAILED — TY ONLY"$'\n'
    body+="------------------------------------------"$'\n'
    if [[ -s "$TY_FAIL_LOG" ]]; then
        body+="$(cat "$TY_FAIL_LOG")"$'\n'
    else
        body+="(none)"$'\n'
    fi
    body+=$'\n'

    # Both failures
    body+="FILES FAILED — BOTH RUFF AND TY"$'\n'
    body+="------------------------------------------"$'\n'
    if [[ -s "$BOTH_FAIL_LOG" ]]; then
        body+="$(cat "$BOTH_FAIL_LOG")"$'\n'
    else
        body+="(none)"$'\n'
    fi

    # Send via mail (falls back to mailx / sendmail)
    echo "$body" | mail -s "$EMAIL_SUBJECT" "$EMAIL_TO" 2>/dev/null \
        || echo "WARNING: Failed to send email. Printing report to stdout instead."

    # Always print summary to terminal
    echo ""
    echo "$body"
}

# =============================================================================
# Main
# =============================================================================
main() {
    echo "=== Simopt Auto Commit — $TIMESTAMP ==="
    echo ""

    check_tools

    cd "$REPO_DIR"

    # Gather modified, unstaged files (compatible with Bash 3.x on macOS)
    files=()
    while IFS= read -r line; do
        files+=("$line")
    done < <(
        git status --porcelain=v1 \
            | grep -E '^ M|^MM' \
            | sed 's/^...//'
    )

    if [[ ${#files[@]} -eq 0 ]]; then
        echo "No modified unstaged files found. Nothing to do."
        send_report
        exit 0
    fi

    echo "Found ${#files[@]} modified unstaged file(s)."
    echo ""

    # Process each file
    for f in "${files[@]}"; do
        process_file "$f"
    done

    echo ""

    # Commit if anything was staged
    do_commit || true

    echo ""

    # Report
    send_report
}

main "$@"
