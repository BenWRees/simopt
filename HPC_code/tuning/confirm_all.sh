#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"; ENV_NAME="${ENV_NAME:-simopt}"; source "$CONDA_BASE/bin/activate" "$ENV_NAME"
python -m scripts.tuning.confirm --problem DYNAMNEWS-1 --k 5
python -m scripts.tuning.collect --problems DYNAMNEWS-1
python -m scripts.tuning.confirm --problem SAN-1 --k 5
python -m scripts.tuning.collect --problems SAN-1
python -m scripts.tuning.confirm --problem NETWORK-1 --k 5
python -m scripts.tuning.collect --problems NETWORK-1
python -m scripts.tuning.confirm --problem ROSENBROCK-1 --k 5
python -m scripts.tuning.collect --problems ROSENBROCK-1
python -m scripts.tuning.confirm --problem PARAMESTI-1 --k 5
python -m scripts.tuning.collect --problems PARAMESTI-1
python -m scripts.tuning.report --problems DYNAMNEWS-1,SAN-1,NETWORK-1,ROSENBROCK-1,PARAMESTI-1
