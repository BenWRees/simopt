"""
    The current python module generates a table of results for the computation times and accuracy of the final solutions with  
    benchmarks and a fraction of the problems solved by the macroreplications of the solver.  
"""
import sys
import os.path as o
import os
import time
from collections import defaultdict

import numpy as np 
import pandas as pd

sys.path.append(
	o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), ".."))
)  # type:ignore

# Import the ProblemsSolvers class and other useful functions
from simopt.experiment_base import ProblemsSolvers, post_normalize, plot_solvability_profiles, ProblemSolver, read_experiment_results, plot_progress_curves
from simopt.directory import problem_directory
from simopt.data_analysis_base import DataAnalysis

def get_pickle_files(folder_path):
	return [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".pickle")]

def read_previous_experiments(folder_path: str) -> list[ProblemSolver] :
	"""_summary_

	Args:
		folder_path (str): _description_
		solver_names (list[str]): _description_
		problem_names (list[str]): _description_

	Returns:
		list[ProblemSolver]: A list of problem solvers
	"""
	myexperiments = []

	for file_path in get_pickle_files(folder_path) :
		experiment = read_experiment_results(file_path)
		myexperiments.append(experiment) 
	return myexperiments

def post_replications(experiments: list[ProblemSolver]) -> list[ProblemSolver] : 
		# ps = ProblemsSolvers(experiments=experiments, file_name_path='ProblemSolvers')
		# print("Post-processing results.")
		# # Run a fixed number of postreplications at all recommended solutions.
		# ps.post_replicate(n_postreps=200)
		# Find an optimal solution x* for normalization.
		# ps.post_normalize(n_postreps_init_opt=200)
		for exp in experiments : 
			exp.post_replicate(n_postreps=200)
			# Find an optimal solution x* for normalization.
			# try :
			post_normalize([exp], n_postreps_init_opt=200)
			# except ValueError as e :
			# 	pass 
			exp.log_experiment_results()
			exp.log_experiments_csv()

		return experiments


def read_table_from_filepath(filepath: str) -> pd.DataFrame:
	"""_summary_

	Args:
		filepath (str): _description_

	Returns:
		pd.DataFrame: _description_
	"""
	df = pd.read_csv(filepath, sep='\t')
	return df

def pandas_table(myexperiments: list[ProblemSolver]) -> dict[str,pd.DataFrame] :
	"""_summary_

	Args:
		myexperiments (list[ProblemSolver]): _description_

	Returns:
		list[pd.DataFrame]: _description_
	"""
	dataframes = dict()
	for experiment in myexperiments :
		filepath = experiment.log_experiment_results_table()
		df = read_table_from_filepath(filepath)
		#remove the first 4 rows from the dataframe
		# df = df.iloc[4:].reset_index(drop=True)
		name = f'{experiment.solver.name}={experiment.problem.name}'
		df['Solved'] = df['Solved'].map({'Yes': 1, 'No': 0})
		dataframes[name]= df
	return dataframes


def get_mean_statistics(dataframes: dict[str,pd.DataFrame]) -> dict[str, pd.DataFrame]:
	"""
	Calculate the row-wise mean for each solver's dataframe.

	Args:
		dataframes (dict[str,pd.DataFrame]): A dictionary of dataframes with solver names as keys and dataframes as values.

	Returns:
		dict[str, pd.DataFrame]: A dictionary of dataframes with solver names as keys and dataframes with mean statistics as values.
	"""
	mean_statistics = {}
	for solver_name, df in dataframes.items():
		# Calculate the mean for each row
		mean_statistics[solver_name] = df.mean(axis=0, numeric_only=True).to_frame().T
	return mean_statistics


def main() : 
	file_path = sys.argv[1]
	myexperiment_list = read_previous_experiments(file_path)
	for experiment in myexperiment_list :
		print(f"Solver: {experiment.solver.name},\t Problems: {experiment.problem.name}") 
	myexperiment_list = post_replications(myexperiment_list)

	for experiment in myexperiment_list :
		name = f'{experiment.solver.name}={experiment.problem.name}'
		plot_solvability_profiles(experiments=[[experiment]], plot_type="cdf_solvability",solve_tol=0.1, plot_conf_ints=False, legend_loc="upper left", plot_title=f"Solvability Profiles of {name}")

	for experiment in myexperiment_list :
		print(f"Solver: {experiment.solver.name},\t Problems: {experiment.problem.name}")
	pandas_tables = pandas_table(myexperiment_list)

	for df in pandas_tables.values() :
		print(df)

	summary_stats = get_mean_statistics(pandas_tables)

	for solver_name, df in summary_stats.items():
		solver_name, problem_name = solver_name.split('=')
		print(f"Solver: {solver_name}, Problem: {problem_name}")
		print(df, "\n")





if __name__ == "__main__":
	main()