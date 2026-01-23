import csv
import logging
import os
from types import MethodType
from simopt.models.rosenbrock import RosenbrockFunctionProblem
from simopt.models.zakharov import ZakharovFunctionProblem
from simopt.solvers.astromorf import ASTROMORF
from simopt.experiment.run_solver import run_solver

logging.basicConfig(level=logging.INFO)

WINDOWS = [2,3]
PROBLEMS = [
    ("ROSENBROCK-1", RosenbrockFunctionProblem),
    ("ZAKHAROV-1", ZakharovFunctionProblem),
]

csv_path = 'multi_mrep_plateau_results.csv'
completed = set()
if os.path.exists(csv_path):
    with open(csv_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for r in reader:
            if r.get('status') == 'completed':
                completed.add((r['problem'], int(r['window'])))

results = []
for pname, Pcls in PROBLEMS:
    for w in WINDOWS:
        if (pname, w) in completed:
            logging.info(f"Skipping {pname} window={w} (already done)")
            continue
        logging.info(f"Running {pname} with plateau_window={w} (10 macroreps)")
        solver = ASTROMORF(fixed_factors={"Record Diagnostics": False})
        def _fixed_compute(self, budget_total):
            return int(w)
        solver.compute_plateau_window = MethodType(_fixed_compute, solver)
        problem = Pcls()
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
                'problem': pname,
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
                'problem': pname,
                'window': w,
                'mean_final_fn': None,
                'std_final_fn': None,
                'elapsed_mean': None,
                'n_macroreps': 10,
                'status': 'failed',
                'error': repr(e)
            }
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['problem','window','mean_final_fn','std_final_fn','elapsed_mean','n_macroreps','status','error'])
            writer.writerow(row)
        results.append(row)

print('Resume run complete. Appended results:')
for r in results:
    print(r)
print('\nCSV at', csv_path)
