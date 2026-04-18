"""Resume a multi-macrorep plateau window sweep, skipping completed runs."""

import csv
import logging
from pathlib import Path

from simopt.experiment.run_solver import run_solver
from simopt.models.rosenbrock import RosenbrockFunctionProblem
from simopt.models.zakharov import ZakharovFunctionProblem
from simopt.solvers.astromorf import ASTROMORF

logging.basicConfig(level=logging.INFO)

WINDOWS = [2, 3]
PROBLEMS = [
    ("ROSENBROCK-1", RosenbrockFunctionProblem),
    ("ZAKHAROV-1", ZakharovFunctionProblem),
]

csv_path = Path("multi_mrep_plateau_results.csv")
completed: set[tuple[str, int]] = set()
if csv_path.exists():
    with csv_path.open(newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for r in reader:
            if r.get("status") == "completed":
                completed.add((r["problem"], int(r["window"])))

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

results = []
for pname, Pcls in PROBLEMS:
    for w in WINDOWS:
        if (pname, w) in completed:
            logging.info("Skipping %s window=%d (already done)", pname, w)
            continue
        logging.info("Running %s with plateau_window=%d (10 macroreps)", pname, w)
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
            solver, "compute_plateau_window", _patched_compute_plateau_window
        )

        problem = Pcls()
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
                "problem": pname,
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
                "problem": pname,
                "window": w,
                "mean_final_fn": None,
                "std_final_fn": None,
                "elapsed_mean": None,
                "n_macroreps": 10,
                "status": "failed",
                "error": repr(e),
            }
        with csv_path.open("a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
            writer.writerow(row)
        results.append(row)

print("Resume run complete. Appended results:")
for r in results:
    print(r)
print("\nCSV at", csv_path)
