#type: ignore 
"""
	Summary
	-------
	This script offers the chance to plot the experiments after running (as I expect I will make a mistake in the demo_problems_solver.py file)
	Can also plot individual performances
"""
import sys
import os.path as o
import os
import time
from collections import defaultdict

sys.path.append(
	o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), ".."))
)  # type:ignore

# Import the ProblemsSolvers class and other useful functions
from simopt.experiment_base import ProblemsSolvers, PlotType, post_normalize, plot_solvability_profiles, ProblemSolver, read_experiment_results, plot_progress_curves
from simopt.directory import problem_directory
from simopt.data_analysis_base import DataAnalysis


def get_pickle_files(folder_path):
	return [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".pickle")]

def read_previous_experiments(folder_path: str) -> list[ProblemSolver] :
	"""
	Read in previous experiments from a folder path containing .pickle files

	Args:
		folder_path (str): The folder path containing .pickle files
		solver_names (list[str]): solver names to filter by
		problem_names (list[str]): 	problem names to filter by

	Returns:
		list[ProblemSolver]: A list of problem solvers
	"""
	myexperiments = []

	for file_path in get_pickle_files(folder_path) :
		experiment = read_experiment_results(file_path)
		myexperiments.append(experiment) 
	return myexperiments

def sort_problem_solver(problem_solvers: list[ProblemSolver]) -> list[list[ProblemsSolvers]] : 
	grouped = defaultdict(list)
	
	for obj in problem_solvers:
		grouped[obj.solver.name].append(obj)
	 
	res = list(grouped.values())
	return res 

#TODO: add an exception handling for if postnormalise is dividing by zero
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
			# exp.log_experiments_csv()

		return experiments

def plot_group(problemSolvers: ProblemsSolvers) -> list[str] : 
	# ps = ProblemsSolvers(experiments=experiments)
	ouput_path_1 = plot_solvability_profiles(
		experiments=problemSolvers.experiments, plot_type=PlotType.CDF_SOLVABILITY,solve_tol=0.1, plot_conf_ints=False, legend_loc="upper left", plot_title="Solvability Profiles of Problem Solvers" 
	)
	ouput_path_2 = plot_solvability_profiles(
		experiments=problemSolvers.experiments, plot_type=PlotType.QUANTILE_SOLVABILITY,solve_tol=0.1, plot_conf_ints=False, legend_loc="upper left", plot_title="Quantile Solvability Profiles of Problem Solvers" 
	)
	return [ouput_path_1, ouput_path_2]


def plot_individual(experiments: list[ProblemSolver]) -> list[str] :
	# experiments_flattened = [item for sublist in experiments for item in sublist]
	output_names = []
	for exp in experiments :
		solver_name = exp.solver.name 
		problem_name = exp.problem.name 
		
		all_output_name = plot_progress_curves(
		experiments=[exp], plot_type=PlotType.ALL, normalize=False, plot_title=f'All {solver_name} on {problem_name}',
		)
		
		mean_output_name = plot_progress_curves(
			experiments=[exp], plot_type=PlotType.MEAN, normalize=True, plot_title=f'{solver_name} on {problem_name}',
		)

		
		# quantile_output_name =plot_progress_curves(
		# 	experiments=[exp],
		# 	plot_type="quantile",
		# 	beta=0.90,
		# 	normalize=False,
		# 	plot_title=f'Quantile {solver_name} on {problem_name}',
		# )
		output_names.append(mean_output_name)

	return output_names



def main() : 
	file_path = sys.argv[1]
	myexperiment_list = read_previous_experiments(file_path)
	# myexperiments = sort_problem_solver(myexperiment_list)

	[print(f'problem name is {a.problem.name} \t and solver name is {a.solver.name}') for a in myexperiment_list]

	# [print(a.all_recommended_xs) for b in myexperiments for a in b]

	problem_solvers = post_replications(myexperiment_list)

	print('FINISHED POST REPLICATIONS')

	# for ps in problem_solvers : 
	# 	print(f'The optimal solution is {ps.optimal_solution} and the optimal value is {ps.optimal_value}')

	output_names = plot_individual(problem_solvers)
	print('The output file paths of the individual experiments are the following: \n')
	for names in output_names : 
		# for name in names : 
		print(names)
		print('\n')

	# myexperiments = ProblemsSolvers(experiments=sort_problem_solver(problem_solvers), file_name_path='ProblemSolvers')

	# for exp_group in myexperiments.experiments : 
	# 	for exp in exp_group : 
	# 		print(f'The solver name is {exp.solver.name} and the problem name is {exp.problem.name}')
	# 	print(f'there are {len(exp_group)} problems for each solver')
	# 	print('\n')

	# print(myexperiments[0][0].progress_curves)
   
	# output_paths = plot_group(myexperiments)
	# print('The output file path of the group solvability profiles is: ')
	# [print(path) for path in output_paths]

if __name__ == '__main__' : 
	main()



