"""
This script is intended to help with debugging problems and solvers.
It create a problem-solver pairing (using the directory) and runs multiple
macroreplications of the solver on the problem.
"""

import sys
import os.path as o
import json
import matplotlib.pyplot as plt 

sys.path.append(
	o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), ".."))
)  # type:ignore

# Import the ProblemSolver class and other useful functions
from simopt.experiment_base import (
	ProblemSolver,
	post_normalize,
	plot_progress_curves,
	plot_solvability_cdfs,
)


def main(solver_name, problem_name, solver_factors, problem_factors) -> None :
	# !! When testing a new solver/problem, first go to directory.py.
	# There you should add the import statement and an entry in the respective
	# dictionary (or dictionaries).
	# See directory.py for more details.

	# Specify the names of the solver and problem to test.
	# solver_name = <solver_name>
	# problem_name = <problem_name>
	# These names are strings and should match those input to directory.py.

	# Example with random search solver on continuous newsvendor problem.
	# -----------------------------------------------
	# solver_name = "OMoRF"  # Random search solver
	# problem_name = "RMITD-1"  # Continuous newsvendor problem
	# -----------------------------------------------
	print(f"Testing solver {solver_name} on problem {problem_name}.")

	# Specify file path name for storing experiment outputs in .pickle file.
	file_name_path = (
		"experiments/outputs/" + solver_name + "_on_" + problem_name + ".pickle"
	)
	print(f"Results will be stored as {file_name_path}.")
	# solver_factors = {'geometry instance': 'AstroDFGeometry', 'sampling rule': 'AdaptiveSampling', 'model type': 'RandomModelReuse', 'polynomial basis': 'AstroDFBasis'}
	# solver_factors = {'subspace dimension': 1}
	# Initialize an instance of the experiment class.
	myexperiment = ProblemSolver(solver_name, problem_name, solver_fixed_factors=solver_factors, problem_fixed_factors=problem_factors)#, model_fixed_factors={'mu': 18.0})
	# myexperiment = ProblemSolver(solver_name, problem_name)
	# Run a fixed number of macroreplications of the solver on the problem.
	myexperiment.run(n_macroreps=20)

	print(f"Finished running solver {solver_name} on problem {problem_name}.")

	# If the solver runs have already been performed, uncomment the
	# following pair of lines (and uncommmen the myexperiment.run(...)
	# line above) to read in results from a .pickle file.
	# myexperiment = read_experiment_results(file_name_path)

	# print("Post-processing results.")
	# # Run a fixed number of postreplications at all recommended solutions.
	# myexperiment.post_replicate(n_postreps=200)
	# # Find an optimal solution x* for normalization.
	# post_normalize([myexperiment], n_postreps_init_opt=200)

	# # Log results.
	# myexperiment.log_experiment_results()
	# myexperiment.log_experiments_csv()

	# print("Plotting results.")
	# # # sub_dim = myexperiment.solver.factors['subspace dimension']
	# # sub_dim = 15
	# # # Produce basic plots of the solver on the problem.
	# plot_progress_curves(
	# 	experiments=[myexperiment], plot_type="all", normalize=False, plot_title=f'All {solver_name} on {problem_name}',
	# )
	# plot_progress_curves(
	# 	experiments=[myexperiment], plot_type="mean", normalize=False, plot_title=f'Mean {solver_name} on {problem_name}',
	# )
	# plot_progress_curves(
	# 	experiments=[myexperiment],
	# 	plot_type="quantile",
	# 	beta=0.90,
	# 	normalize=False,
	# 	plot_title=f'Quantile {solver_name} on {problem_name}',
	# )
	# plot_solvability_cdfs(experiments=[myexperiment], solve_tol=0.1, plot_title=f'Solvability {solver_name} on {problem_name}')


	# # Plots will be saved in the folder experiments/plots.
	# print("Finished. Plots can be found in experiments/plots folder.")

if __name__ == "__main__":
	main('ASTROMoRF', 'FACSIZE-1',{'crn_across_solns': False}, {'budget': 1000})
	main('ASTRODF', 'FACSIZE-1',{'crn_across_solns': False}, {'budget': 1000})
	main('OMoRF', 'FACSIZE-1',{'crn_across_solns': False}, {'budget': 1000})