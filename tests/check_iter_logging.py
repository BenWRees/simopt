"""Check iteration logging for solvers."""

import logging

from simopt.experiment.run_solver import run_solver
from simopt.experiment_base import instantiate_problem, instantiate_solver

logging.basicConfig(level=logging.INFO)

SOLVERS = ["ADAM", "RNDSRCH", "ASTROMORF", "NELDMD", "STRONG", "ASTRODF"]

# Use the ROSENBROCK problem (use the problem class_name_abbr)
# Increase budget so Nelder-Mead has sufficient budget for initial simplex
problem = instantiate_problem("ROSENBROCK-1", problem_fixed_factors={"budget": 2000})

for sname in SOLVERS:
    try:
        solver = instantiate_solver(sname, fixed_factors={})
        print(f"\n=== Solver {sname} ===")
        sol_df, it_df, elapsed = run_solver(solver, problem, n_macroreps=1, n_jobs=1)
        print("solution_df rows:", len(sol_df))
        if it_df is None:
            print("iteration_df: None")
        else:
            print("iteration_df rows:", len(it_df))
            # check columns
            cols = list(it_df.columns)
            print("columns:", cols)
            # check consistency of lists if present
            try:
                it_list = (
                    it_df.sort_values(["mrep", "iteration"])
                    if "mrep" in it_df.columns
                    else it_df
                )
                # for single mrep, check that counts for iteration, budget_history,
                # fn_estimate are equal
                it_count = len(it_list)
                print("entries:", it_count)
                if it_count > 0:
                    # quick numeric checks
                    nan_any = it_df.isnull().values.any()
                    print("contains NaN:", nan_any)
            except Exception as e:
                print("Error inspecting iteration_df:", e)
    except Exception as e:
        print(f"Failed to run solver {sname}: {e}")
