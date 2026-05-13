#!/usr/bin/env bash
set -euo pipefail

# Submit three SLURM job files to the scheduler and print messages
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_FILES=("run_experiments.slurm" "run_crn.slurm" "run_journal_factors.slurm")

if ! command -v sbatch >/dev/null 2>&1; then
	echo "Error: sbatch not found in PATH." >&2
	exit 1
fi

for job in "${JOB_FILES[@]}"; do
	path="$DIR/$job"
	if [[ ! -f "$path" ]]; then
		echo "Error: SLURM file not found: $path" >&2
		exit 1
	fi

	echo "Submitting $job to SLURM queue..."
	out=$(sbatch "$path" 2>&1) || { echo "sbatch failed for $job: $out" >&2; exit 1; }

	if [[ $out =~ ([0-9]+) ]]; then
		echo "Submitted $job -> job ${BASH_REMATCH[1]}"
	else
		echo "Submitted $job -> response: $out"
	fi
done

echo "All SLURM jobs submitted."

