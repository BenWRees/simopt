import csv
import logging
from types import MethodType
from simopt.models.rosenbrock import RosenbrockFunctionProblem
from simopt.solvers.astromorf import ASTROMORF
from simopt.experiment.run_solver import run_solver

logging.basicConfig(level=logging.INFO)

WINDOWS = [2,3,4,6,8,12,16]
results = []

for w in WINDOWS:
    logging.info(f"Running sweep with plateau_window={w}")
    problem = RosenbrockFunctionProblem()
    solver = ASTROMORF(fixed_factors={"Record Diagnostics": True})

    # bind a fixed compute_plateau_window method to this instance
    def _fixed_compute(self, budget_total):
        return int(w)
    solver.compute_plateau_window = MethodType(_fixed_compute, solver)

    solution_df, iteration_df, elapsed = run_solver(solver, problem, n_macroreps=1, n_jobs=1)

    # pick final fn estimate (last entry of iteration_df.fn_estimate) if available
    final_fn = None
    if iteration_df is not None and not iteration_df.empty:
        try:
            final_fn = float(iteration_df['fn_estimate'].dropna().iloc[-1])
        except Exception:
            final_fn = None

    # record d history if available
    d_history = getattr(solver, 'd_history', None)

    results.append({'window': w, 'final_fn': final_fn, 'elapsed': elapsed[0] if elapsed else None, 'd_history': d_history})

# write CSV
with open('plateau_sweep_results.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['window','final_fn','elapsed','d_history'])
    writer.writeheader()
    for r in results:
        writer.writerow(r)

print('Sweep complete. Results:')
for r in results:
    print(r)
print('\nCSV written to plateau_sweep_results.csv')
