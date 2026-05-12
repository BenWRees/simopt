# ASTROMoRF hyperparameter tuning (HPC workflow)

End-to-end Optuna-based tuning of ASTROMoRF for **simulation budget = 10000**
on `DYNAMNEWS-1`, `SAN-1`, `NETWORK-1`, `ROSENBROCK-1` (all scaled to dim
= 100) and `PARAMESTI-1` (native dim = 2).

## What it does

1. Each Slurm array task launches **one async Optuna worker** that pulls
   trials from a shared SQLite (or Postgres) study.
2. Each trial is run through a 3-rung **ASHA** ladder
   (4 → 8 → 16 macroreps), so wasteful configurations are pruned early.
3. After tuning, a **confirmation pass** re-runs the top-5 trials with a
   disjoint seed stream (30 macroreps each) and picks the final winner
   on the basis of the confirmed mean + std.
4. Results are exported to JSON/CSV and a markdown recommendation report.

The workflow is **resume-safe**: if any worker dies, simply
`sbatch` the array again — the SQLite study tracks which trials are
done, in-flight, or failed, and remaining workers pick up where the
killed one stopped. There is **no per-run state** outside the storage
backend and the JSONL diagnostics log.

## Files

```
HPC_code/tuning/
  generate_slurm.py          # regenerate the .slurm scripts (see below)
  tune_<PROBLEM>.slurm       # one Slurm array per problem (16 workers each;
                             #   PARAMESTI-1 uses 4)
  submit_all.sh              # sbatch every tune_*.slurm
  confirm_all.sh             # post-tuning: confirm + collect + report
  README.md                  # this file
```

The Python tuner library lives at `scripts/tuning/` (separate package).

## Quick start

```bash
# 1. Make sure deps are installed (once).
conda env update -n simopt -f environment.yml
# or, with pip:
pip install -e '.[tuning]'

# 2. Smoke-test locally (small budget, 2 trials, ~2 min).
python -m scripts.tuning.smoke --problem DYNAMNEWS-1 --n-trials 2

# 3. Submit all five problems' tuning arrays.
bash HPC_code/tuning/submit_all.sh

# 4. (After all arrays finish) confirm + collect + write the report.
bash HPC_code/tuning/confirm_all.sh

# 5. Read recommendations.md.
cat results/astromorf_tuning/recommendations.md
```

## Cluster customisation

The Slurm scripts intentionally **avoid hard-coded paths**. They derive
the repo root from `$SLURM_SUBMIT_DIR` (set by Slurm) or `$PWD`. Override
explicitly with:

```bash
REPO_ROOT=/scratch/$USER/simopt sbatch HPC_code/tuning/tune_SAN-1.slurm
```

To regenerate the Slurm scripts with cluster-specific defaults:

```bash
python HPC_code/tuning/generate_slurm.py \
  --partition standard \
  --time 12:00:00 \
  --mem-per-cpu 4G \
  --max-concurrent 8 \
  --mail-user me@example.com \
  --n-trials-per-worker 14
```

The conda env name is `simopt` by default; override with `ENV_NAME=...
sbatch ...`. The conda base path defaults to `$HOME/miniconda3`;
override with `CONDA_BASE=...`.

## Storage backends

Default: SQLite, one DB per problem under
`results/astromorf_tuning/studies/<problem>.db`.

For larger arrays (>16 concurrent workers per problem) prefer Postgres:

```bash
export ASTROMORF_OPTUNA_STORAGE="postgresql+psycopg2://user:pw@host/optuna"
```

The same env var is honoured by every script (worker, confirm, collect,
report).

## Resume / fault tolerance

* Killed workers leave their last in-flight trial as `RUNNING` for a
  short timeout, then Optuna marks it `FAIL`. The next worker picks up.
* `sbatch HPC_code/tuning/tune_<PROBLEM>.slurm` is idempotent — workers
  do `load_if_exists=True` and the warm-start trial is enqueued only on
  first study creation.
* Each per-trial diagnostic snapshot is appended to
  `results/astromorf_tuning/trials_jsonl/<problem>.jsonl`, so even
  crashed-mid-run trials leave forensic data behind.
* Per-trial wall-clock cap (`--per-trial-cap-s`, default 1800s) prevents
  pathological configurations (e.g. high-degree polynomials at large
  subspace dim) from monopolising a worker.

## What the report contains

After running `confirm_all.sh`:

```
results/astromorf_tuning/
  recommendations.md          # human-readable per-problem report
  recommendation_table.csv    # machine-readable summary
  best/best_<problem>.json    # winning factor dict per problem
  confirmations/confirm_<problem>.json   # full top-K confirmation table
  exports/<problem>/trials.csv           # every trial flattened
  exports/<problem>/top20.json           # top-20 with diagnostics + 95% CI
  exports/<problem>/study_meta.json
  trials_jsonl/<problem>.jsonl           # per-rung diagnostics log
  studies/<problem>.db                   # the Optuna SQLite study
```

The recommendation markdown lists, per problem:

* **Recommended factors** — the dict you can drop straight into
  `instantiate_solver("ASTROMORF", fixed_factors=...)`.
* **Confirmation statistics** — mean / std / 95% CI of the
  confirmation-pass aligned objective (lower is always better).
* **Diagnostics** — accept/reject ratios, average subspace dim,
  CABS dim-change counts, pattern-search override ratio, mean
  prediction relative error, mean wall-clock per macrorep.
* **Comparison vs `PROBLEM_OPTIMAL_HYPER`** — the legacy warm-start
  values from `scripts/journal_factors_test.py`, with relative
  improvement (in the correct direction for the problem's
  minimisation/maximisation orientation).
* **Top-K confirmation table** and **top-20 study table** for inspection.

## What it deliberately does NOT do

* Does **not** modify `simopt/solvers/astromorf.py` or any existing
  experiment script (`run_experiments.py`, `journal_factors_test.py`,
  `optimize_astromorf_cabs.py`).
* Does **not** use nested parallelism — each Slurm task is `cpus-per-task=1`
  and macroreps inside a trial run sequentially (so we can extract
  per-mrep diagnostics off the solver instance).
* Does **not** optimise asymptotic post-budget behaviour. Every trial
  evaluates ASTROMoRF at exactly `budget = 10000`.
