"""Run a multi-macrorep plateau window sweep on DynamNews."""

import csv
import logging
from pathlib import Path

from simopt.experiment.run_solver import run_solver
from simopt.models.dynamnews import DynamNewsMaxProfit
from simopt.solvers.astromorf import ASTROMORF

logging.basicConfig(level=logging.INFO)

WINDOWS = [2, 3]
PROBLEM_NAME = "DYNAMNEWS-1"
CSV_PATH = Path("multi_mrep_plateau_results.csv")

FIELDNAMES = [
    "problem",
    "window",
    "mean_final_fn",
    "std_final_fn",
    "elapsed_mean",
    "n_macroreps",
    "status",
    "error",
]

# ensure CSV has header
if not CSV_PATH.exists():
    with CSV_PATH.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writeheader()

results = []
for w in WINDOWS:
    logging.info(
        "Running %s with plateau_window=%d (10 macroreps)",
        PROBLEM_NAME,
        w,
    )
    solver = ASTROMORF(fixed_factors={"Record Diagnostics": False})

    def _fixed_compute(_w: int = w) -> int:
        return int(_w)

    def _patched_compute_plateau_window(_budget: int | None) -> int:
        """A patched version of compute_plateau_window.

        ignores the budget and returns a fixed window size.
        """
        if _budget is None:
            raise ValueError("Budget cannot be None for this patched function.")
        return _fixed_compute(int(_budget))

    object.__setattr__(
        solver,
        "compute_plateau_window",
        _patched_compute_plateau_window,
    )

    problem = DynamNewsMaxProfit()
    try:
        solution_df, iteration_df, elapsed = run_solver(
            solver, problem, n_macroreps=10, n_jobs=1
        )

        final_fns: list[float] = []
        if iteration_df is not None:
            try:
                grouped = iteration_df.groupby("mrep")
                for _mrep, group in grouped:
                    final_fns.append(float(group["fn_estimate"].dropna().iloc[-1]))
            except Exception:
                final_fns = []

        mean_fn = sum(final_fns) / len(final_fns) if final_fns else None
        assert mean_fn is not None
        std_fn = (
            (sum((x - mean_fn) ** 2 for x in final_fns) / len(final_fns)) ** 0.5
            if final_fns
            else None
        )

        row = {
            "problem": PROBLEM_NAME,
            "window": w,
            "mean_final_fn": mean_fn,
            "std_final_fn": std_fn,
            "elapsed_mean": (sum(elapsed) / len(elapsed) if elapsed else None),
            "n_macroreps": 10,
            "status": "completed",
            "error": "",
        }
    except Exception as e:
        row = {
            "problem": PROBLEM_NAME,
            "window": w,
            "mean_final_fn": None,
            "std_final_fn": None,
            "elapsed_mean": None,
            "n_macroreps": 10,
            "status": "failed",
            "error": repr(e),
        }

    results.append(row)
    # append to CSV incrementally
    with CSV_PATH.open("a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writerow(row)

print("Multi-mrep DYNAMNEWS sweep complete. Results:")
for r in results:
    print(r)
print("\nCSV written to", CSV_PATH)
