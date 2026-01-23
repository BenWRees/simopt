import logging
import sys
from simopt.models.rosenbrock import RosenbrockFunctionProblem
from simopt.solvers.astromorf import ASTROMORF
from simopt.experiment.run_solver import run_solver
from simopt.experiment_base import instantiate_problem

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    #argument Pass of problem name

    problem_name = sys.argv[1] if len(sys.argv) > 1 else "rosenbrock"

    problem = instantiate_problem(problem_name)

    solver = ASTROMORF(fixed_factors={"Record Diagnostics": True})

    solution_df, iteration_df, elapsed = run_solver(solver, problem, n_macroreps=1, n_jobs=1)

    print("Solution DataFrame:\n", solution_df)
    print("\nIteration DataFrame:\n", iteration_df)
    print("\nElapsed times:\n", elapsed)
