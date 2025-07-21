"""
This experiment tests the performance of OMoRF against a trust region algorithm with sensitivity analysis applied to the problem
"""

import sys
import os 
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(sys.modules[__name__].__file__), ".."))) # type:ignore

from SALib.sample.sobol import sample
from SALib.analyze import sobol
from mrg32k3a.mrg32k3a import MRG32k3a
import numpy as np 
import concurrent.futures

from simopt.experiment_base import ProblemsSolvers, plot_solvability_profiles, ProblemSolver, Problem, Solution, plot_terminal_progress, post_normalize
from simopt.directory import problem_directory, solver_directory


def create_new_solution(point: tuple, problem: Problem) -> Solution : 
	new_solution = Solution(tuple(point), problem)
	new_solution.attach_rngs([MRG32k3a() for _ in range(problem.model.n_rngs)])
	return new_solution

def calculate_samples(problem: Problem, no_samples: int, samples_list: np.ndarray) -> tuple[np.ndarray,int] :
	"""Take samples generated

	Args:
		problem (Problem): The simulation model for which to run the samples on 
		no_samples (int): The number of samples being produced. This is denoted as N
		samples_list (np.ndarray): A matrix of the shape ((N*D)+2,D) where D is the number of decision variables and N is the number of samples 

	Returns:
		np.ndarray, int: The output from the simulation of the samples_list and the expended budget in the run 
	"""
	expended_budget = no_samples * (problem.dim + 1) #this is the case for the sobol sampling
	outputs = []

	for sample_point in samples_list : 
		new_solution = create_new_solution(tuple(sample_point), problem)
		problem.simulate(new_solution)
		fn = -1 * problem.minmax[0] * new_solution.objectives_mean
		outputs.append(fn[0])

	return np.array(outputs).reshape(-1,), expended_budget


def fix_bounds(problem: Problem) -> dict :
	"""Fixes any bounds in the problem dictionary if they're unbounded

	Args:
		factors (dict): the initial factors for sampling

	Returns:
		dict: the fixed factors for sampling
	"""
	old_lower_bound = list(problem.lower_bounds)
	old_upper_bound = list(problem.upper_bounds)

	minimum = -1e8
	maximum = 1e8

	lower_bound, upper_bound = [], [] 
	for bound in old_lower_bound :  
		if bound == -np.inf : 
			lower_bound.append(minimum)
		else :
			lower_bound.append(bound)

	for bound in old_upper_bound :  
		if bound == np.inf : 
			upper_bound.append(maximum)
		else : 
			upper_bound.append(bound)

	return lower_bound, upper_bound
		
"""
	Look at Sequential bifurcation 
	Sanchezes 
	Jack Kleijnen 
"""
def sensitivitity_analysis(problem_name: str, subspace_dim: int) -> tuple[Problem, int, int] :
	"""Instantiates a Problem object and applies sensitivity analsysis to reduce the dimensions of the problem

	Args:
		problem_name (str): Name of the problem whos dimensions are being reduced 
		subspace_dim (int): the dimension wanted for the subspace

	Returns:
		Problem: The reduced-dimension Problem after sensitivity analysis 
		int: The expended budget used in the sensitivity analysis 
		int: The dimension of the simulation model after reduction using sensitivity analysis 
	"""
	problem = problem_directory[problem_name](fixed_factors={'budget':5000})
	
	lb, ub = fix_bounds(problem)

	#undergo sensitivity analysis on the problem 
	init_factors = {
		'num_vars': problem.dim,
		'names': ['x'+str(i) for i in range(1,problem.dim+1)],
		'bounds': [[l,u] for l,u in zip(lb, ub)] #! This isn't working when the bounds are inf or -inf? 
	}

	

	SAMPLE_NO = 2**5 #sobol sampling needs to be 2**n
	param_values = sample(init_factors, SAMPLE_NO, calc_second_order=False)
	# print(f'param_values: {param_values}')
	#turn the param_values into a series of solutions that can be replicated 
	Y, expended_budget = calculate_samples(problem, SAMPLE_NO, param_values) 

	#get sobol indicies
	Si = sobol.analyze(init_factors, Y, print_to_console=False, calc_second_order=False)
	# print(f"Sobol indices: {Si['S1']}")


	#choose the subspace_dim largest sobol indices and find what indices they are 
	indices_to_remove = find_reduced_dim_indices(Si, subspace_dim, problem.dim)

	#reduce the problem down by removing all dimensions apart from the indices selected to keep

	#analyse sobol results to choose what dimensions to remove - obtain the indices of the dimensions to remove 
	indices_to_remove = find_reduced_dim_indices(Si, subspace_dim, problem.dim)
	
	reduced_dim_problem = create_reduced_dim_problem(indices_to_remove, problem)

	return reduced_dim_problem, expended_budget


def find_reduced_dim_indices(analysis: dict, subspace_dim: int, original_dim: int) -> list[int] : 
	"""Ensures that just enough dimensions are removed from the problem, by dynamically decreasing the threshold

	Args:
		analysis (dict): The results from sobol.analyze
		min_subspace_dim (int): the smallest subspace dimension allowed

	Returns:
		tuple[list[int], int]: The list of indices to remove
		The length of the list should be equal to the original dim - subspace dimension
	"""
	assert subspace_dim <= original_dim, 'The minimum subspace cannot be greater than the original dimension of the problem'
	inactive_vals = original_dim - subspace_dim
	#dictionary where values are the original indices and the sorted sobol indices are the keys
	sorted_sobol_indices = {i: val for i, val in sorted(enumerate(analysis['S1']), key=lambda x: x[1])}
	#take the first subspace_dim in the keys
	indices_to_remove = list(dict(list(sorted_sobol_indices.items())[:inactive_vals]).keys())

	assert len(indices_to_remove) == original_dim - subspace_dim, 'The length of the indices is not removing the right number of dimension' 

	return indices_to_remove 


		 
def remove_items_with_index(index_list: list[int], value: list) -> list : 
	new_value = [item for i, item in enumerate(value) if i not in index_list]
	# print(f'value: {value}')
	# print(f'deleted_values: {new_value}')
	# for idx in index_list :
	# 	elem_to_remove = index_list[idx]
	return new_value

def create_reduced_dim_problem(components_to_remove: list[int], problem: Problem) -> Problem : 
	"""Create a new problem instance with a reduced dimension

	Args:
		dim_to_remove (list[int]): dimensions that will be removed after SA 
		problem (Problem): The problem which will undergo dimensionality reduction

	Returns:
		Problem: The new reduced-dimension problem
	"""
	original_dim = problem.dim
	subspace_dim = original_dim - len(components_to_remove)

	#Go through model factors and anywhere that depends on the dimension of the problem, reduce 
	# new_model_dictionary = dict()
	for key, value in problem.model.factors.items() :
		if (type(value) == list or isinstance(value, np.ndarray)) and len(value) % original_dim == 0 :
			# print(f'the key is {key} \nThe factor is {value} ')
			new_value = remove_items_with_index(components_to_remove, value)
			# print(f'the reduced dimension value is: {new_value}')
			problem.model.factors[key] = new_value
		else : 
			problem.model.factors[key] = value 

	# print(f'the new model dictionary is {problem.model.factors}')
	# problem.model.factors = new_model_dictionary


	#go through problem factors 
	# new_problem_dictionary = dict()
	for key, value in problem.factors.items() :
		if (type(value) == list or isinstance(value, np.ndarray)) and len(value) % original_dim == 0 :
			# print(f'the key is {key} \nThe factor is {value} ')
			new_value = remove_items_with_index(components_to_remove, value)
			# print(f'the reduced dimension value is: {new_value}')
			problem.factors[key] = new_value
		else : 
			problem.factors[key] = value 

	# print(f'the new model dictionary is {problem.factors}')
	# problem.factors = new_problem_dictionary

		 
	return problem 


def experimentalSetUp(problem_names: list[str], subspace_dim: int) -> list[ProblemSolver] :
	"""This sets up the list of problem solvers for OMoRF and ASTRODF, where the problems on OMoRF are already reduced-dimension problems

	Args:
		problem_name (str): The name of the problems to test each Solver on

	Returns:
		list[ProblemSolver]: A list of Problem Solvers
	"""
	regular_problems = [problem_directory[a](fixed_factors={'budget': 5000}) for a in problem_names]
	reduced_dim_problems = []
	SA_budgets = []
	subspace_dims = []
	for problem_name in problem_names : 
		reduced_dim_problem, SA_budget_exp = sensitivitity_analysis(problem_name, subspace_dim)  
		reduced_dim_problems.append(reduced_dim_problem)
		SA_budgets.append(SA_budget_exp)
		subspace_dims.append(subspace_dim)

	regular_problems_and_factors = [(a, {'subspace dimension': b}) for a ,b in zip(regular_problems, subspace_dims) ]
	reduced_problems_and_budgets = [(a, {'expended budget': b}) for a,b in zip(reduced_dim_problems,SA_budgets)]


	OMoRFProblemSolvers = []
	ASTROProblemSolvers = []
	for reg_problem, dim_factor in regular_problems_and_factors :
		solver_rename = "ASTROMoRF with dimension " + str(dim_factor['subspace dimension']) + " on problem " + reg_problem.name
		file_name = solver_rename.replace(' ', '_') + '.pickle' 
		ASTROMoRFSolver = solver_directory['ASTROMoRF'](fixed_factors=dim_factor)
		ps = ProblemSolver(solver=ASTROMoRFSolver, solver_rename=solver_rename, problem=reg_problem, file_name_path=file_name)
		OMoRFProblemSolvers.append(ps) 
	
	for red_problem, exp_budget in reduced_problems_and_budgets :
		solver_rename = "ASTRODF with dimension " + str(subspace_dim) + " on problem " + red_problem.name
		file_name = solver_rename.replace(' ', '_') + '.pickle'
		ASTROSolver = solver_directory['ASTRODF'](fixed_factors=exp_budget)
		ps = ProblemSolver(solver=ASTROSolver, solver_rename=solver_rename, problem=red_problem, file_name_path=file_name)
		ASTROProblemSolvers.append(ps)

	ps_total = ASTROProblemSolvers + OMoRFProblemSolvers

	# print(f'length of ps_total: {ps_total}')


	return ps_total

def run_problem_solver(ps: ProblemSolver) -> None :
	# Run a fixed number of macroreplications of each solver on each problem.
	ps.run(n_macroreps=20)

	print("Post-processing results.")
	# Run a fixed number of postreplications at all recommended solutions.
	ps.post_replicate(n_postreps=100)
	# Find an optimal solution x* for normalization.
	post_normalize([ps], n_postreps_init_opt=100)

	ps.log_experiment_results()

def main():
	#! NEED TO ADD MORE PROBLEMS 
	problem_names = [ 
		'ROSENBROCK-1',
		'ZAKHAROV-1',
		]
	
	subspace_dim = list(range(1,15)) 

	max_workers = int(sys.argv[1])
	#calcaulate all the problem solvers 
	ps_total = []
	for dim in subspace_dim :
		ps_total.extend(experimentalSetUp(problem_names, dim))

	with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor : 
		futures = []
		for ps in ps_total : 
			future = executor.submit(run_problem_solver, ps)
			futures.append(future)
		concurrent.futures.wait(futures)


if __name__ == "__main__":
	main()
