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
	instantiate_problem,
)

def main(solver_name, problem_name, macroreplication_no, solver_factors, budget) -> None :
	print(f"Testing solver {solver_name} on problem {problem_name}.")
	
	problem_to_test = problem_name
	
	target_problem = instantiate_problem(problem_name=problem_to_test, problem_fixed_factors={}, model_fixed_factors={})
	
	max_dimension = target_problem.dim 
	max_degree = 4
	
	solver_factors = {
		"sample_size": 10,
		"n_initial_random": 5,
		"kernel_lengthscale": 1.0,
		"kernel_variance": 1.0,
		"noise_variance": 1e-6,
		"xi": 0.01,
		"n_candidates": 1000,
	}

	model_fixed_factors = {
		"target_problem": target_problem,
		"n_macroreps": 3,
		"gamma_1": 2.5,
		"gamma_2": 1.2,
		"gamma_3": 0.5,
		"consistency_weight": 0.1,
		"quality_weight": 0.9
	}	
	problem_fixed_factors = {
		"max_dimension": max_dimension,
		"max_degree": max_degree,
		"max_gamma_1": 10,
		"max_gamma_2": 10,
		"max_gamma_3": 0.999,
		"budget": budget,
	}


	# Specify file path name for storing experiment outputs in .pickle file.
	file_name_path = (
		"experiments/outputs/" + solver_name + "_on_" + problem_name + ".pickle"
	)
	print(f"Results will be stored as {file_name_path}.")
	# Initialize an instance of the experiment class.
	myexperiment = ProblemSolver(solver_name, 'ASTROMORF-HYPEROPT-2', solver_fixed_factors=solver_factors, problem_fixed_factors={'budget': budget},
							  model_fixed_factors=model_fixed_factors, file_name_path=file_name_path)
	
	# [print(f"{k}: {v}") for k, v in myexperiment.problem.model.factors.items()]
	# Run a fixed number of macroreplications of the solver on the problem.
	myexperiment.run(n_macroreps=macroreplication_no)

	print(f"Finished running solver {solver_name} on problem {problem_name}.")



if __name__ == "__main__":
	# main('ASTRODF', 'FACSIZE-1', 2, {'crn_across_solns': False}, {'budget': 5000})
	main('BO', 'DYNAMNEWS-1', 1, {'crn_across_solns': False}, 1000)
	# main('OMoRF', 'FACSIZE-1', 2, {'crn_across_solns': False}, {'budget': 5000})