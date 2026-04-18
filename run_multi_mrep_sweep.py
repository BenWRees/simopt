"""Run a multi-macrorep plateau window sweep on Rosenbrock."""

import csv
import logging
from pathlib import Path

from simopt.experiment.run_solver import run_solver
from simopt.models.rosenbrock import RosenbrockFunctionProblem
from simopt.solvers.astromorf import ASTROMORF

logging.basicConfig(level=logging.INFO)

WINDOWS = [2, 3]
# Restrict to ROSENBROCK for a focused multi-mrep sweep
PROBLEMS = [
    ("ROSENBROCK-1", RosenbrockFunctionProblem),
]

FIELDNAMES = [
    "problem",
    "window",
    "mrep",
    "final_fn",
    "elapsed",
    "status",
    "error",
]

results: list[dict[str, object]] = []

# ensure CSV has header
csv_path = Path("multi_mrep_plateau_results.csv")
# Write per-macrorep rows: one line per (problem, window, mrep)
with csv_path.open("w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
    writer.writeheader()

for pname, Pcls in PROBLEMS:
    for w in WINDOWS:
        logging.info(
            "Running %s with plateau_window=%d (10 macroreps)",
            pname,
            w,
        )
        solver: ASTROMORF = ASTROMORF(fixed_factors={"Record Diagnostics": False})

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

        # Use a reduced budget to speed up the multi-mrep sweep
        problem = Pcls(fixed_factors={"budget": 200})
        try:
            rows_to_write: list[dict[str, object]] = []
            for mrep_id in range(10):
                logging.info(
                    "Running mrep %d for %s window=%d",
                    mrep_id,
                    pname,
                    w,
                )
                # instantiate solver per macrorep
                solver_m = ASTROMORF(fixed_factors={"Record Diagnostics": False})

                def _fixed_compute_inner(_w: int = w) -> int:
                    return int(_w)

                def _patched_compute_plateau_window(_budget: int | None) -> int:
                    """A patched version of compute_plateau_window.

                    ignores the budget and returns a fixed window size.
                    """
                    if _budget is None:
                        raise ValueError(
                            "Budget cannot be None for this patched function."
                        )
                    return _fixed_compute(int(_budget))

                object.__setattr__(
                    solver_m,
                    "compute_plateau_window",
                    _patched_compute_plateau_window,
                )

                solution_df, iteration_df, elapsed = run_solver(
                    solver_m,
                    problem,
                    n_macroreps=1,
                    n_jobs=1,
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
                rows_to_write.append(
                    {
                        "problem": pname,
                        "window": w,
                        "mrep": int(mrep_id),
                        "final_fn": final_val,
                        "elapsed": elapsed_mrep,
                        "status": "completed",
                        "error": "",
                    }
                )
        except Exception as e:
            rows_to_write = [
                {
                    "problem": pname,
                    "window": w,
                    "mrep": -1,
                    "final_fn": None,
                    "elapsed": None,
                    "status": "failed",
                    "error": repr(e),
                }
            ]

        # append per-macrorep rows to CSV incrementally
        with csv_path.open("a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
            for r in rows_to_write:
                writer.writerow(r)
                results.append(r)

print("Multi-mrep sweep complete. Results:")
for r in results:
    print(r)
print("\nCSV written to", csv_path)
