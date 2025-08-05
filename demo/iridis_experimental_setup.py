"""
	Summary
	-------
	This script allows for a problem-solver pair to be run and logged into a pickle file. This allows us to run multiple processes on iridis of each problem solver pair 
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
from simopt.base import Problem
from simopt.directory import problem_directory
from simopt.data_analysis_base import DataAnalysis

import concurrent.futures


def run_problem_solver_pair(solver_name: str, problem: Problem) -> None : 
	print(f"Testing solver {solver_name} on problem {problem.name}.")
	#Lookup table for subspace dims 
	subspace_dims = {
		'ASTROMoRF': {
			'DYNAMNEWS-1': 1,
			'ROSENBROCK-1': 5,
			'ZAKHAROV-1': 8
		},
		'OMoRF': {
			'DYNAMNEWS-1': 1,
			'ROSENBROCK-1': 1,
			'ZAKHAROV-1': 7
		}
	}
	# if origina_flag == True :
	# 	solver_fixed_factors = {'subspace dimension': 7, 'original sampling rule': True}
	# 	solver_rename = solver_name + ' with original sampling'
	# else : 
	# 	solver_fixed_factors = {'subspace dimension': 7, 'original sampling rule': False}
	# 	solver_rename = solver_name + ' with two-stage sampling'

	#get the subspace_dim
	if solver_name == 'ASTROMoRF' or solver_name == 'OMoRF' :
		subspace_dim = subspace_dims[solver_name][problem.name]
		solver_fixed_factors = {'subspace dimension': subspace_dim, 'crn_across_solns':False}
	
	#case of ASTRODF (and other solvers that do not use DR)	
	else :
		solver_fixed_factors = {'crn_across_solns':False}
	
	myexperiment = ProblemSolver(solver_name, problem=problem, solver_fixed_factors=solver_fixed_factors)
	# myexperiment = ProblemSolver(solver_name, problem_name)
	# Run a fixed number of macroreplications of the solver on the problem.
	myexperiment.run(n_macroreps=20)

	# If the solver runs have already been performed, uncomment the
	# following pair of lines (and uncommmen the myexperiment.run(...)
	# line above) to read in results from a .pickle file.
	# myexperiment = read_experiment_results(file_name_path)

	# print("Post-processing results.")
	# Run a fixed number of postreplications at all recommended solutions.
	# myexperiment.post_replicate(n_postreps=200)
	# Find an optimal solution x* for normalization.
	# post_normalize([myexperiment], n_postreps_init_opt=200)

	# Log results.
	myexperiment.log_experiment_results()
	myexperiment.log_experiments_csv()



def main() : 
	solver_names = [
		# 'ASTRODF',
		# 'ASTROMoRF',
		# 'OMoRF'
		'NELDMD',
		# 'SGD',
		# 'STRONG',
		# 'SPSA'
	]
	problem_names = [
		'DYNAMNEWS-1',
		'ROSENBROCK-1',
		'ZAKHAROV-1'
	]
	# max_workers = int(sys.argv[1])
	# with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor : 
	# 	futures = []
		
	# 	for problem_name in problem_names :
	# 		# problem construction
	# 		if problem_name == 'DYNAMNEWS-1' :
	# 			model_fixed_factors = {'mu': 16.0}  
	# 		# elif problem_name == 'ROSENBROCK-1' or problem_name == 'ZAKHAROV-1' : #case of zakharov and rosenbrock
	# 		else : 
	# 			model_fixed_factors = {'variance': 5.0}
	# 		# else : 
	# 		# 	model_fixed_factors = {}

	# 		problem = problem_directory[problem_name](fixed_factors={'budget': 10000}, model_fixed_factors=model_fixed_factors)
			
	# 		for solver_name in solver_names :
	# 			futures.append(executor.submit(run_problem_solver_pair,solver_name, problem)) 

	# 	concurrent.futures.wait(futures)

	for problem_name in problem_names :
		#problem construction
		if problem_name == 'DYNAMNEWS-1' :
			model_fixed_factors = {'mu': 17.5}  
		elif problem_name == 'ROSENBROCK-1' or problem_name == 'ZAKHAROV-1' :
			model_fixed_factors = {'variance': 5.0}
		else : 
			model_fixed_factors = {}

		problem = problem_directory[problem_name](fixed_factors={'budget': 10000}, model_fixed_factors=model_fixed_factors)
		
		for solver_name in solver_names :
			run_problem_solver_pair(solver_name, problem)


# def main() :
# 	solver_names = ['ASTROMoRF']
# 	problem_names = ['ROSENBROCK-1']

# 	for problem_name in problem_names :
# 			#problem construction
# 			if problem_name == 'DYNAMNEWS-1' :
# 				model_fixed_factors = {'mu': 17.5}  
# 			elif problem_name == 'ROSENBROCK-1' or problem_name == 'ZAKHAROV-1' : #case of zakharov and rosenbrock
# 				model_fixed_factors = {'variance': 8.}
# 			else : 
# 				model_fixed_factors = {}

# 			problem = problem_directory[problem_name](fixed_factors={'budget': 500}, model_fixed_factors=model_fixed_factors)
			
# 			for solver_name in solver_names :
# 				run_problem_solver_pair(solver_name, problem, True)
# 				run_problem_solver_pair(solver_name, problem, False)






if __name__ == '__main__' : 
	main()