import csv
import logging
from types import MethodType
from simopt.models.rosenbrock import RosenbrockFunctionProblem
from simopt.solvers.astromorf import ASTROMORF
from simopt.experiment.run_solver import run_solver

logging.basicConfig(level=logging.INFO)

WINDOW = 3
PROBLEM = ("ROSENBROCK-1", RosenbrockFunctionProblem)

csv_path = 'multi_mrep_plateau_results.csv'

pname, Pcls = PROBLEM
problem = Pcls(fixed_factors={"budget": 200})

for mrep_id in range(10):
    logging.info(f"Appending mrep {mrep_id} for {pname} window={WINDOW}")
    solver_m = ASTROMORF(fixed_factors={"Record Diagnostics": False})

    def _fixed_compute(self, budget_total):
        return int(WINDOW)
    solver_m.compute_plateau_window = MethodType(_fixed_compute, solver_m)

    try:
        solution_df, iteration_df, elapsed = run_solver(solver_m, problem, n_macroreps=1, n_jobs=1)
        final_val = None
        try:
            if iteration_df is not None and not iteration_df.empty:
                final_val = float(iteration_df['fn_estimate'].dropna().iloc[-1])
        except Exception:
            final_val = None
        elapsed_mrep = elapsed[0] if isinstance(elapsed, (list, tuple)) and len(elapsed) > 0 else None
        row = {'problem': pname, 'window': WINDOW, 'mrep': int(mrep_id), 'final_fn': final_val, 'elapsed': elapsed_mrep, 'status': 'completed', 'error': ''}
    except Exception as e:
        row = {'problem': pname, 'window': WINDOW, 'mrep': int(mrep_id), 'final_fn': None, 'elapsed': None, 'status': 'failed', 'error': repr(e)}

    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['problem', 'window', 'mrep', 'final_fn', 'elapsed', 'status', 'error'])
        writer.writerow(row)

print('Append complete')
