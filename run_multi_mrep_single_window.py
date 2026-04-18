"""Run individual macroreps for a single problem and plateau window."""

import csv
import logging
from pathlib import Path

from simopt.experiment.run_solver import run_solver
from simopt.models.rosenbrock import RosenbrockFunctionProblem
from simopt.solvers.astromorf import ASTROMORF

logging.basicConfig(level=logging.INFO)

WINDOW = 3
PROBLEM = ("ROSENBROCK-1", RosenbrockFunctionProblem)

csv_path = Path("multi_mrep_plateau_results.csv")

FIELDNAMES = [
    "problem",
    "window",
    "mrep",
    "final_fn",
    "elapsed",
    "status",
    "error",
]

pname, Pcls = PROBLEM
problem = Pcls(fixed_factors={"budget": 200})

for mrep_id in range(10):
    logging.info("Appending mrep %d for %s window=%d", mrep_id, pname, WINDOW)
    solver_m = ASTROMORF(fixed_factors={"Record Diagnostics": False})

    def _fixed_compute(_w: int) -> int:
        return int(_w)

    def _patched_compute_plateau_window(_budget: int | None) -> int:
        """A patched version of compute_plateau_window.

        ignores the budget and returns a fixed window size.
        """
        if _budget is None:
            raise ValueError("Budget cannot be None for this patched function.")
        return _fixed_compute(int(_budget))

    object.__setattr__(
        solver_m,
        "compute_plateau_window",
        _patched_compute_plateau_window,
    )

    try:
        solution_df, iteration_df, elapsed = run_solver(
            solver_m, problem, n_macroreps=1, n_jobs=1
        )
        final_val = None
        try:
            if iteration_df is not None and not iteration_df.empty:
                final_val = float(iteration_df["fn_estimate"].dropna().iloc[-1])
        except Exception:
            final_val = None
        elapsed_mrep = (
            elapsed[0]
            if isinstance(elapsed, list | tuple) and len(elapsed) > 0
            else None
        )
        row = {
            "problem": pname,
            "window": WINDOW,
            "mrep": int(mrep_id),
            "final_fn": final_val,
            "elapsed": elapsed_mrep,
            "status": "completed",
            "error": "",
        }
    except Exception as e:
        row = {
            "problem": pname,
            "window": WINDOW,
            "mrep": int(mrep_id),
            "final_fn": None,
            "elapsed": None,
            "status": "failed",
            "error": repr(e),
        }

    with csv_path.open("a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writerow(row)

print("Append complete")
