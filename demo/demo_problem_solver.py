"""
This script is intended to help with debugging problems and solvers.
It create a problem-solver pairing (using the directory) and runs multiple
macroreplications of the solver on the problem.
"""

import sys
import os.path as o
import random

import numpy as np

sys.path.append(
	o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), ".."))
)  # type:ignore

# Import the ProblemSolver class and other useful functions
from simopt.experiment_base import (
	ProblemSolver,
)


def main(solver_name, problem_name, dim_size, macroreplication_no, solver_factors, budget) -> None :
	print(f"Testing solver {solver_name} on problem {problem_name}.")

	# Specify file path name for storing experiment outputs in .pickle file.
	file_name_path = (
		"experiments/outputs/" + solver_name + "_on_" + problem_name + ".pickle"
	)
	print(f"Results will be stored as {file_name_path}.")
	# Initialize an instance of the experiment class.
	myexperiment = ProblemSolver(solver_name, problem_name, solver_fixed_factors=solver_factors, problem_fixed_factors={'budget': budget},
							  file_name_path=file_name_path)
	
	# [print(f"{k}: {v}") for k, v in myexperiment.problem.model.factors.items()]
	# Run a fixed number of macroreplications of the solver on the problem.
	myexperiment.run(n_macroreps=macroreplication_no)

	print(f"Finished running solver {solver_name} on problem {problem_name}.")



if __name__ == "__main__":
	# main('ASTRODF', 'FACSIZE-1', 2, {'crn_across_solns': False}, {'budget': 5000})
	main('ASTROMoRF', 'AIRLINE-1', 10, 1, {'crn_across_solns': False}, 5000)
	# main('OMoRF', 'FACSIZE-1', 2, {'crn_across_solns': False}, {'budget': 5000})