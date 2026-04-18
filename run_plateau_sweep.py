"""Run a plateau window sweep on Rosenbrock with diagnostics."""

import csv
import logging
from pathlib import Path

from simopt.experiment.run_solver import run_solver
from simopt.models.rosenbrock import RosenbrockFunctionProblem
from simopt.solvers.astromorf import ASTROMORF

logging.basicConfig(level=logging.INFO)

WINDOWS = [2, 3, 4, 6, 8, 12, 16]
results: list[dict[str, object]] = []

for w in WINDOWS:
    logging.info("Running sweep with plateau_window=%d", w)
    problem = RosenbrockFunctionProblem()
    solver = ASTROMORF(fixed_factors={"Record Diagnostics": True})

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

    solution_df, iteration_df, elapsed = run_solver(
        solver, problem, n_macroreps=1, n_jobs=1
    )

    # pick final fn estimate if available
    final_fn = None
    if iteration_df is not None and not iteration_df.empty:
        try:
            final_fn = float(iteration_df["fn_estimate"].dropna().iloc[-1])
        except Exception:
            final_fn = None

    # record d history if available
    d_history = getattr(solver, "d_history", None)

    results.append(
        {
            "window": w,
            "final_fn": final_fn,
            "elapsed": elapsed[0] if elapsed else None,
            "d_history": d_history,
        }
    )

# write CSV
csv_path = Path("plateau_sweep_results.csv")
with csv_path.open("w", newline="") as csvfile:
    writer = csv.DictWriter(
        csvfile,
        fieldnames=["window", "final_fn", "elapsed", "d_history"],
    )
    writer.writeheader()
    for r in results:
        writer.writerow(r)

print("Sweep complete. Results:")
for r in results:
    print(r)
print("\nCSV written to plateau_sweep_results.csv")
