"""
	Summary
	-------
	Tests the performance of ASTROMoRF against OMoRF for more deterministic problems vs more stochastic problems
"""
import sys
import os.path as o
import os
import time

sys.path.append(
	o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), ".."))
)  # type:ignore

# Import the ProblemsSolvers class and other useful functions
from simopt.experiment_base import ProblemsSolvers, ProblemSolver, read_experiment_results, post_normalize
from simopt.directory import problem_directory
from simopt.data_analysis_base import DataAnalysis

import concurrent.futures



def run_problem_solver_pair(solver_name: str, problem_name: str, variance: float) -> None : 
	print(f"Testing solver {solver_name} on problem {problem_name}.")
	if (solver_name == 'ASTROMoRF' or solver_name == 'OMoRF') and problem_name == 'ROSENBROCK-1' :
		solver_fixed_factors = {'subspace dimension': 7} 

	elif (solver_name == 'ASTROMoRF' or solver_name == 'OMoRF') and problem_name == 'DYNAMNEWS-1' : 
		solver_fixed_factors = {'subspace dimension': 2}
	
	else :
		solver_fixed_factors = {}

	if problem_name == 'DYNAMNEWS-1' :
		model_fixed_factors = {'mu': variance}  
		problem_rename = 'DYNAMNEWS-1'

	else :
		model_fixed_factors = {'variance': variance}
		problem_rename = 'ZAKHAROV-1'
	
	myexperiment = ProblemSolver(solver_name, problem_name, problem_rename= problem_rename, solver_fixed_factors=solver_fixed_factors, model_fixed_factors=model_fixed_factors, problem_fixed_factors={'budget':1000})
	# myexperiment = ProblemSolver(solver_name, problem_name)
	# Run a fixed number of macroreplications of the solver on the problem.
	myexperiment.run(n_macroreps=20)

	# If the solver runs have already been performed, uncomment the
	# following pair of lines (and uncommmen the myexperiment.run(...)
	# line above) to read in results from a .pickle file.
	# myexperiment = read_experiment_results(file_name_path)

	print("Post-processing results.")
	# Run a fixed number of postreplications at all recommended solutions.
	myexperiment.post_replicate(n_postreps=200)
	# Find an optimal solution x* for normalization.
	post_normalize([myexperiment], n_postreps_init_opt=200)

	# Log results.
	myexperiment.log_experiment_results()
	myexperiment.log_experiments_csv()



def main() : 
	solver_names = [
		'ASTROMoRF',
		'OMoRF'
	]
	problem_names = [
		'DYNAMNEWS-1',
		'ROSENBROCK-1',
		'ZAKHAROV-1'
	]
	max_workers = int(sys.argv[1])
	with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor : 
		futures = []
		
		for problem_name in problem_names :
			if problem_name == 'DYNAMNEWS' : 
				variances = [1.5, 17.5]

			else : 
				variances = [0.0, 5.0]
			
			for solver_name in solver_names :
				for variance in variances :
					futures.append(executor.submit(run_problem_solver_pair, solver_name, problem_name, variance)) 

		concurrent.futures.wait(futures)



if __name__ == '__main__' : 
	main()