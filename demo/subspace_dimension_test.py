"""
	This is a script that will find the optimal subspace dimension
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
from simopt.base import Problem

import concurrent.futures

def run_different_subspaces(solver_name: str, problem: Problem, subspace_dim: int) -> ProblemSolver : 
	print(f'Running {solver_name} on {problem.name}')
	# problem = problem_directory[problem_name](fixed_factors={'budget': 100})
	# subspace_dims = range(1,problem.dim+1)

	# for dim in subspace_dims :
	new_solver_name = solver_name + ' on ' + problem.name + ' with subspace ' + str(subspace_dim)
	file_name = new_solver_name.replace(' ', '_') + '.pickle'
	myexperiment = ProblemSolver(solver_name=solver_name, solver_rename=new_solver_name, file_name_path=file_name, problem=problem, solver_fixed_factors={'subspace dimension': subspace_dim})

	return myexperiment

def run_problem_solver(myexperiment: ProblemSolver) -> None :
	print(f'Running {myexperiment.solver.name} on {myexperiment.problem.name}')
	myexperiment.run(n_macroreps=10)

	print("Post-processing results.")
	# Run a fixed number of postreplications at all recommended solutions.
	myexperiment.post_replicate(n_postreps=100)
	# Find an optimal solution x* for normalization.
	post_normalize([myexperiment], n_postreps_init_opt=100)

	# Log results.
	myexperiment.log_experiment_results()
	myexperiment.log_experiments_csv()


#! This should now run all the subspace tests in parallel
def main() : 
	max_workers = int(sys.argv[1])

	solver_names = sys.argv[2]
	problem_names = sys.argv[3]

	problem_solvers_list = []
	for problem_name in [problem_names] :# ['DYNAMNEWS-1', 'ZAKHAROV-1', 'ROSENBROCK-1'] :
		if problem_name == 'DYNAMNEWS-1' :
			problem = problem_directory[problem_name](fixed_factors={'budget': 600}, model_fixed_factors={'mu': 18.0})
		elif problem_name =='ZAKHAROV-1' or problem_name =='ROSENBROCK-1' : 
			problem = problem_directory[problem_name](fixed_factors={'budget': 600}, model_fixed_factors={'variance': 5.0})	
		else : 
			problem = problem_directory[problem_name](fixed_factors={'budget': 600})
		for solver_name in [solver_names] :#['ASTROMoRF', 'OMoRF'] :
			for dim in range(1, problem.dim + 1) :
				problem_solvers_list.append(run_different_subspaces(solver_name, problem, dim)) 

	print(f'The number of problem solvers is: {len(problem_solvers_list)}')

	#run all the problem solvers in parallel
	with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor : 
		futures = []
		for problem_solver in problem_solvers_list :
			futures.append(executor.submit(run_problem_solver,problem_solver)) 

		for future in futures :
			print(future.result())
		concurrent.futures.wait(futures)


if __name__ == '__main__' : 
	main()