#!/usr/bin/env bash
# Launch optuna-dashboard against a SQLite snapshot of the live Journal study.
#
# Usage:
#   bash HPC_code/tuning/launch_dashboard.sh                  # all problems
#   bash HPC_code/tuning/launch_dashboard.sh SAN-1            # one problem
#   bash HPC_code/tuning/launch_dashboard.sh SAN-1 NETWORK-1  # several
#
# Env knobs:
#   DASHBOARD_PORT       (default: first free port in 8080..8099)
#   DASHBOARD_HOST       (default: 127.0.0.1)
#   DASHBOARD_REFRESH_S  (default: 60; periodic snapshot refresh interval)
#   NO_REFRESH=1         disable the background refresh loop
#   CONDA_BASE / ENV_NAME like the rest of the pipeline
#
# This script:
#   1. cds to the repo root (so relative paths resolve correctly).
#   2. Activates the conda env.
#   3. Runs ``export_dashboard_db`` once to materialise the SQLite snapshot.
#   4. Starts a *background* refresh loop (so the dashboard stays current
#      while workers are still running). Disable with NO_REFRESH=1.
#   5. Picks a free localhost port.
#   6. Launches optuna-dashboard against the snapshot.
#   7. Prints the SSH tunnel command you need to view it from your laptop.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
ENV_NAME="${ENV_NAME:-simopt}"
# shellcheck disable=SC1091
source "$CONDA_BASE/bin/activate" "$ENV_NAME"

DASHBOARD_HOST="${DASHBOARD_HOST:-127.0.0.1}"
DASHBOARD_REFRESH_S="${DASHBOARD_REFRESH_S:-60}"
DASHBOARD_DB_DIR="$REPO_ROOT/results/astromorf_tuning/dashboard"
mkdir -p "$DASHBOARD_DB_DIR"

# Problem list: positional args or all five.
if [[ $# -gt 0 ]]; then
  PROBLEMS=("$@")
else
  PROBLEMS=(DYNAMNEWS-1 SAN-1 NETWORK-1 ROSENBROCK-1 PARAMESTI-1)
fi
PROBLEMS_CSV="$(IFS=,; echo "${PROBLEMS[*]}")"

# Find a free port in [8080..8099], or honour DASHBOARD_PORT.
pick_port() {
  if [[ -n "${DASHBOARD_PORT:-}" ]]; then
    echo "$DASHBOARD_PORT"
    return
  fi
  for port in $(seq 8080 8099); do
    # `bash` >/dev/tcp/host/port works without nc/lsof.
    if ! (echo > "/dev/tcp/${DASHBOARD_HOST}/${port}") 2>/dev/null; then
      echo "$port"
      return
    fi
  done
  echo "ERROR: no free port in 8080..8099" >&2
  exit 1
}
PORT="$(pick_port)"

echo "[launch_dashboard] Problems: ${PROBLEMS_CSV}"

# We always bundle every problem into ONE SQLite file (all_studies.db).
# optuna-dashboard takes a single storage URL on the CLI, and SQLite
# storages can hold many studies — so pointing at the combined file
# lets the sidebar show every problem at once with no version-specific
# --storage-directory hack.
COMBINED_DB="$DASHBOARD_DB_DIR/all_studies.db"

echo "[launch_dashboard] Initial snapshot -> $COMBINED_DB"
if ! python -m scripts.tuning.export_dashboard_db --problems "$PROBLEMS_CSV"; then
  echo "[launch_dashboard] --problems unsupported by exporter; retrying with --all ..."
  if ! python -m scripts.tuning.export_dashboard_db --all; then
    echo "[launch_dashboard] ERROR: initial snapshot failed" >&2
    exit 1
  fi
fi

REFRESH_PID=""
cleanup() {
  if [[ -n "$REFRESH_PID" ]] && kill -0 "$REFRESH_PID" 2>/dev/null; then
    kill "$REFRESH_PID" 2>/dev/null || true
    wait "$REFRESH_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ "${NO_REFRESH:-0}" != "1" ]]; then
  echo "[launch_dashboard] Starting background refresh every ${DASHBOARD_REFRESH_S}s ..."
  python -m scripts.tuning.export_dashboard_db \
    --problems "$PROBLEMS_CSV" --watch "$DASHBOARD_REFRESH_S" \
    > "$DASHBOARD_DB_DIR/refresh.log" 2>&1 &
  REFRESH_PID=$!
  echo "[launch_dashboard] Refresh PID=$REFRESH_PID (log: $DASHBOARD_DB_DIR/refresh.log)"
fi

if [[ ! -f "$COMBINED_DB" ]]; then
  echo "ERROR: combined snapshot not produced at $COMBINED_DB." >&2
  echo "       This usually means no study has been created yet. Run a" >&2
  echo "       worker (e.g. submit_all.sh or --init-only) and try again." >&2
  exit 1
fi
STORAGE_ARGS=("sqlite:///${COMBINED_DB}")
TARGET="$COMBINED_DB"

REMOTE_HOST="$(hostname -f 2>/dev/null || hostname)"
echo
echo "========================================================================"
echo " optuna-dashboard"
echo "========================================================================"
echo " Target:        $TARGET"
echo " Listening on:  http://${DASHBOARD_HOST}:${PORT}"
echo " Refresh:       $( [[ "${NO_REFRESH:-0}" == "1" ]] && echo OFF || echo "every ${DASHBOARD_REFRESH_S}s" )"
echo
echo " From your laptop, open an SSH tunnel:"
echo
echo "     ssh -N -L ${PORT}:${DASHBOARD_HOST}:${PORT} ${USER}@${REMOTE_HOST}"
echo
echo " Then point your browser at:  http://localhost:${PORT}"
echo "========================================================================"
echo

exec optuna-dashboard --host "$DASHBOARD_HOST" --port "$PORT" "${STORAGE_ARGS[@]}"
