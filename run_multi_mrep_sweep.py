import csv
import logging
from types import MethodType
from simopt.models.rosenbrock import RosenbrockFunctionProblem
from simopt.models.zakharov import ZakharovFunctionProblem
from simopt.solvers.astromorf import ASTROMORF
from simopt.experiment.run_solver import run_solver

logging.basicConfig(level=logging.INFO)

WINDOWS = [2, 3]
# Restrict to ROSENBROCK for a focused multi-mrep sweep
PROBLEMS = [
    ("ROSENBROCK-1", RosenbrockFunctionProblem),
]

results = []

# ensure CSV has header
csv_path = 'multi_mrep_plateau_results.csv'
# Write per-macrorep rows: one line per (problem, window, mrep)
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['problem', 'window', 'mrep', 'final_fn', 'elapsed', 'status', 'error'])
    writer.writeheader()

for pname, Pcls in PROBLEMS:
    for w in WINDOWS:
        logging.info(f"Running {pname} with plateau_window={w} (10 macroreps)")
        solver = ASTROMORF(fixed_factors={"Record Diagnostics": False})

        # Bind fixed compute_plateau_window
        def _fixed_compute(self, budget_total):
            return int(w)
        solver.compute_plateau_window = MethodType(_fixed_compute, solver)

        # Use a reduced budget to speed up the multi-mrep sweep
        problem = Pcls(fixed_factors={"budget": 200})
        try:
            # Run each macrorep independently and write its result immediately to CSV.
            rows_to_write = []
            for mrep_id in range(10):
                logging.info(f"Running mrep {mrep_id} for {pname} window={w}")
                # instantiate solver per macrorep to avoid cross-mrep state
                solver_m = ASTROMORF(fixed_factors={"Record Diagnostics": False})

                def _fixed_compute(self, budget_total):
                    return int(w)
                solver_m.compute_plateau_window = MethodType(_fixed_compute, solver_m)

                solution_df, iteration_df, elapsed = run_solver(solver_m, problem, n_macroreps=1, n_jobs=1)

                final_val = None
                try:
                    if iteration_df is not None and not iteration_df.empty:
                        final_val = float(iteration_df['fn_estimate'].dropna().iloc[-1])
                except Exception:
                    final_val = None

                elapsed_mrep = elapsed[0] if isinstance(elapsed, (list, tuple)) and len(elapsed) > 0 else None
                rows_to_write.append({
                    'problem': pname,
                    'window': w,
                    'mrep': int(mrep_id),
                    'final_fn': final_val,
                    'elapsed': elapsed_mrep,
                    'status': 'completed',
                    'error': ''
                })
        except Exception as e:
            rows_to_write = [{
                'problem': pname,
                'window': w,
                'mrep': -1,
                'final_fn': None,
                'elapsed': None,
                'status': 'failed',
                'error': repr(e)
            }]

        # append per-macrorep rows to CSV incrementally
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['problem', 'window', 'mrep', 'final_fn', 'elapsed', 'status', 'error'])
            for r in rows_to_write:
                writer.writerow(r)
                results.append(r)

print('Multi-mrep sweep complete. Results:')
for r in results:
    print(r)
print('\nCSV written to', csv_path)
