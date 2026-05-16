#!/usr/bin/env bash
set -euo pipefail
# Submit from the repo root so SLURM_SUBMIT_DIR resolves correctly.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKER_PY="$REPO_ROOT/scripts/tuning/worker.py"
cd "$REPO_ROOT"

# Run each worker once to generate the per-pair pickles and warm up caches. 
# This is critical for accurate profiling and tuning, since the first run of 
# each pair is much slower than subsequent
source ~/miniconda3/bin/activate simopt
python "$WORKER_PY" --problem SAN-1 --init-only --max-trials 401
python "$WORKER_PY" --problem DYNAMNEWS-1 --init-only --max-trials 401
python "$WORKER_PY" --problem NETWORK-1 --init-only --max-trials 401
python "$WORKER_PY" --problem ROSENBROCK-1 --init-only --max-trials 401
python "$WORKER_PY" --problem PARAMESTI-1 --init-only --max-trials 161

sbatch "$SCRIPT_DIR/tune_DYNAMNEWS-1.slurm"
sbatch "$SCRIPT_DIR/tune_SAN-1.slurm"
sbatch "$SCRIPT_DIR/tune_NETWORK-1.slurm"
sbatch "$SCRIPT_DIR/tune_ROSENBROCK-1.slurm"
sbatch "$SCRIPT_DIR/tune_PARAMESTI-1.slurm"
echo 'Submitted all tuning arrays.'