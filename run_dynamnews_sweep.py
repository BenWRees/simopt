import csv
import logging
import os
from types import MethodType
from simopt.models.dynamnews import DynamNewsMaxProfit
from simopt.solvers.astromorf import ASTROMORF
from simopt.experiment.run_solver import run_solver

logging.basicConfig(level=logging.INFO)

WINDOWS = [2, 3]
PROBLEM_NAME = "DYNAMNEWS-1"
CSV_PATH = 'multi_mrep_plateau_results.csv'

# ensure CSV has header
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['problem','window','mean_final_fn','std_final_fn','elapsed_mean','n_macroreps','status','error'])
        writer.writeheader()

results = []
for w in WINDOWS:
    logging.info(f"Running {PROBLEM_NAME} with plateau_window={w} (10 macroreps)")
    solver = ASTROMORF(fixed_factors={"Record Diagnostics": False})

    # Bind fixed compute_plateau_window
    def _fixed_compute(self, budget_total):
        return int(w)
    solver.compute_plateau_window = MethodType(_fixed_compute, solver)

    problem = DynamNewsMaxProfit()
    try:
        solution_df, iteration_df, elapsed = run_solver(solver, problem, n_macroreps=10, n_jobs=1)

        final_fns = []
        if iteration_df is not None:
            try:
                grouped = iteration_df.groupby('mrep')
                for mrep, group in grouped:
                    final_fns.append(float(group['fn_estimate'].dropna().iloc[-1]))
            except Exception:
                final_fns = []

        mean_fn = sum(final_fns) / len(final_fns) if final_fns else None
        std_fn = (sum((x - mean_fn)**2 for x in final_fns) / len(final_fns))**0.5 if final_fns else None

        row = {
            'problem': PROBLEM_NAME,
            'window': w,
            'mean_final_fn': mean_fn,
            'std_final_fn': std_fn,
            'elapsed_mean': sum(elapsed)/len(elapsed) if elapsed else None,
            'n_macroreps': 10,
            'status': 'completed',
            'error': ''
        }
    except Exception as e:
        row = {
            'problem': PROBLEM_NAME,
            'window': w,
            'mean_final_fn': None,
            'std_final_fn': None,
            'elapsed_mean': None,
            'n_macroreps': 10,
            'status': 'failed',
            'error': repr(e)
        }

    results.append(row)
    # append to CSV incrementally
    with open(CSV_PATH, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['problem','window','mean_final_fn','std_final_fn','elapsed_mean','n_macroreps','status','error'])
        writer.writerow(row)

print('Multi-mrep DYNAMNEWS sweep complete. Results:')
for r in results:
    print(r)
print('\nCSV written to', CSV_PATH)
