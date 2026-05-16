#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs

echo "Submitting confirm array job..."
jobid=$(sbatch --parsable confirm_all.sbatch)
echo "Submitted confirm array job: $jobid"

echo "Submitting report job dependent on array job success..."
sbatch --dependency=afterok:$jobid confirm_report.sbatch
echo "Submitted report job with dependency afterok:$jobid"

echo "Done. Monitor logs/ or use squeue to watch jobs."
