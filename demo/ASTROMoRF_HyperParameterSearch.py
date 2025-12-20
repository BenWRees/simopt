"""
	Runs the hyperparameter search problem for ASTROMoRF on a selected problem.
	It uses the HyperParameterTuning model to find the optimal subspace dimension
	and polynomial degree for ASTROMoRF on the specified problem using Bayesian optimization.
	After running the hyperparameter search it prints the final solution found (this is the best dimension and degree).
"""

import sys
import os.path as o
import random

import numpy as np

sys.path.append(
	o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), ".."))
) 

# Import the ProblemSolver class and other useful functions
from simopt.experiment_base import (
	ProblemSolver,
	instantiate_problem,
)


def update_model_factors_dimensions(problem_name: str, new_dim: int) -> dict:
	"""
		Find the model associated with model_name and update the dimension of the factor values to the new_dim.
		Return the updated model factors.

	Args:
		model_name (str): Name of Model
		new_dim (int): New dimension for the model

	Returns:
		dict: Updated model factors
	"""
	new_factors = {}
	if problem_name == 'DYNAMNEWS-1':
		new_factors= {
			'num_prod': new_dim,
			'c_utility': [6 + j for j in range(new_dim)],
			'init_level': [3] * new_dim,
			'price': [9] * new_dim,
			'cost': [5] * new_dim,
		}
	elif problem_name == 'FACSIZE-1' or problem_name == 'FACSIZE-2':
		A = np.random.rand(new_dim, new_dim)
		new_factors= {
			'mean_vec': [500] * new_dim,
			'cov': (np.dot(A, A.T) * 100).tolist(),
			'capacity': [random.randint(100,900) for _ in range(new_dim)],
			'n_fac': new_dim,
		}
	elif problem_name == 'FIXEDSAN-1' :
		new_factors= {
		}
	elif problem_name == 'AIRLINE-1' or problem_name == 'AIRLINE-2' :
		num_classes = random.randint(2,new_dim//2)
		odf_leg_matrix = np.random.randint(0,2,(new_dim, num_classes))
		new_factors= {
			'num_classes': num_classes,
			'ODF_leg_matrix': odf_leg_matrix.tolist(),
			'prices': tuple([random.randint(50,300) for _ in range(new_dim)]),
			'capacity': tuple([random.randint(20,150) for _ in range(num_classes)]),
			'booking limits': tuple([random.randint(5,20) for _ in range(new_dim)]),
			'alpha': tuple([random.uniform(0,5) for _ in range(new_dim)]),
			'beta':  tuple([random.uniform(2,10) for _ in range(new_dim)]),
			'gamma_shape': tuple([random.uniform(2,10) for _ in range(new_dim)]),
			'gamma_scale': tuple([random.uniform(10,50) for _ in range(new_dim)]),
		}
	elif problem_name == 'NETWORK-1' :
		process_prob_elem = 1/new_dim
		mode_transit_time = [round(np.random.uniform(0.01,5),3) for _ in range(new_dim)]
		lower_limits_transit_time = [x/2 for x in mode_transit_time] 
		upper_limits_transit_time = [2*x for x in mode_transit_time]
		new_factors= {
			'process_prob': [process_prob_elem] * new_dim,
			'cost_process': [0.1 / (x + 1) for x in range(new_dim)],
			'cost_time': [round(np.random.uniform(0.01,1),3) for _ in range(new_dim)],
			'mode_transit_time': mode_transit_time,
			'lower_limits_transit_time': lower_limits_transit_time,
			'upper_limits_transit_time': upper_limits_transit_time,
			'n_networks': new_dim,
		}
	elif problem_name == 'CONTAM-2' :
		new_factors= {
			'stages': new_dim,
			'prev_decisions': (0,) * new_dim,
		}
	
	return new_factors


def update_problem_factor_dimensions(problem_name: str, new_dim: int, budget: int) -> dict:
	"""
		Update the dimension of the factor values in problem_factors to the new_dim.
		Return the updated problem factors.

	Args:
		problem_factors (dict): Problem factors to be updated
		new_dim (int): New dimension for the problem factors

	Returns:
		dict: Updated problem factors
	"""
	new_factors = {}
	if problem_name == 'DYNAMNEWS-1':
		new_factors =  {
			'initial_solution': (3,) * new_dim,
			'budget': budget,
		}
	elif problem_name == 'FACSIZE-1':
		new_factors =  {
			'initial_solution': (100,) * new_dim,
			'installation_costs': (1,) * new_dim,
			'epsilon': 0.05,
			'budget': budget,
		}
	elif problem_name == 'FACSIZE-2':
		new_factors =  {
			'initial_solution': (300,) * new_dim,
			'installation_costs': (1,) * new_dim,
			'installation_budget': 500.0,
			'budget': budget,
		}
	elif problem_name == 'FIXEDSAN-1' : 
		new_factors =  {
			'budget': budget,
		}
	elif problem_name == 'AIRLINE-1' or problem_name == 'AIRLINE-2' :
		new_factors =  {
			'initial_solution': (3,) * new_dim,
			'budget': budget,
		}
	elif problem_name == 'NETWORK-1' :
		init_soln_elem = 1/new_dim
		new_factors =  {
			'initial_solution': (init_soln_elem,) * new_dim,
			'budget': budget,
		}
	elif problem_name == 'CONTAM-2' :
		new_factors = {
			'initial_solution': (1,) * new_dim,
			'prev_cost': [1] * new_dim,
			'error_prob': [0.2] * new_dim,
			'upper_thres': [0.1] * new_dim,
			'budget': budget,
		}
	return new_factors


def main(problem_to_test: str, solver: str, problem_dim: int | None = None):
	"""Run hyperparameter optimization for ASTROMoRF on a selected problem.
	
	"""
	
	print(f"Running Hyperparameter Test.")
	macroreplication_no = 1

	# Specify file path name for storing experiment outputs in .pickle file.
	file_name_path = (
		"experiments/outputs/Hyperparameter_Test.pickle"
	)
	print(f"Results will be stored as {file_name_path}.")
	
	if problem_dim is not None : 
		target_problem_model_factors = update_model_factors_dimensions(problem_to_test, problem_dim)
		target_problem_factors = update_problem_factor_dimensions(problem_to_test, problem_dim, 4000)
	else :
		target_problem_model_factors = {}
		target_problem_factors = {}

	print(f'The model factors of {problem_to_test} are set to: \n {target_problem_model_factors} \n')
	print(f'The problem factors of {problem_to_test} are set to: \n {target_problem_factors} \n')

	target_problem = instantiate_problem(problem_name=problem_to_test, problem_fixed_factors=target_problem_factors, model_fixed_factors=target_problem_model_factors)

	max_dimension = problem_dim if problem_dim is not None else target_problem.dim
	print(f"Target problem dimension set to {max_dimension}.")
	max_degree = 5 

	if solver == 'RANDS' : 
		#work out the number of possible (dimension, degree) pairs
		possible_solutions = []
		for dim in range(1, max_dimension + 1):
			for deg in range(1, max_degree + 1):
				possible_solutions.append((dim, deg))

		solver_factors = {
			"solution_list": possible_solutions,
			"sample_size": 1,
			"crn_across_solns": True,
		} 
		model_fixed_factors = { #For random search we only want to look at the degree and dimension
			"target_problem": target_problem,
			"n_macroreps": 1,
			"consistency_weight": 0.1,
			"quality_weight": 0.9,
		}
		budget = len(possible_solutions) * solver_factors["sample_size"]
		problem_fixed_factors = {
			"max_dimension": max_dimension,
			"max_degree": max_degree,
			"budget": budget,
		}

		# Initialize an instance of the experiment class.
		myexperiment = ProblemSolver(solver, "ASTROMORF-HYPEROPT-1", solver_fixed_factors=solver_factors, problem_fixed_factors=problem_fixed_factors, model_fixed_factors=model_fixed_factors)

		print(f'Starting Hyperparameter Test on problem {problem_to_test} using solver {solver}...')
		myexperiment.run(n_macroreps=macroreplication_no)

		print(f'Completed {macroreplication_no} macroreplications. Now running post-replications to finalize results...')

		myexperiment.post_replicate(n_postreps=50)

		terminal_vals = [myexperiment.all_recommended_xs[i][-1] for i in range(len(myexperiment.all_recommended_xs))]
		#out of the terminal_vals select the integer (dimension, degree) pair that is recommended most often
		final_solution = max(set(terminal_vals), key = terminal_vals.count)
		print(f"Finished running Hyperparameter Test. Final solution found: {final_solution}")

	elif solver == 'MixedIntTRSolver' : 

		budget = 7500
		model_fixed_factors = {
			"target_problem": target_problem,
			"n_macroreps": 5,
			"gamma_1": 2.0,
			"gamma_2": 1.2,
			"gamma_3": 0.5,
			"consistency_weight": 0.1,
			"quality_weight": 0.9
		}	
		problem_fixed_factors = {
			"max_dimension": max_dimension,
			"max_degree": max_degree,
			"budget": budget,
		}

		# Initialize an instance of the experiment class. - optimising over gamma values too
		myexperiment = ProblemSolver(solver, "ASTROMORF-HYPEROPT-2", problem_fixed_factors=problem_fixed_factors, model_fixed_factors=model_fixed_factors)

		print(f'Starting Hyperparameter Test on problem {problem_to_test} using solver {solver}...')
		myexperiment.run(n_macroreps=macroreplication_no)

		print(f'Completed {macroreplication_no} macroreplications. Now running post-replications to finalize results...')

		myexperiment.post_replicate(n_postreps=50)

		terminal_vals = [myexperiment.all_recommended_xs[i][-1] for i in range(len(myexperiment.all_recommended_xs))]
		#out of the terminal_vals select the integer (dimension, degree) pair that is recommended most often
		final_solution = max(set(terminal_vals), key = terminal_vals.count)
		print(f"Finished running Hyperparameter Test. Final solution found: {final_solution}")



if __name__ == '__main__' : 
	import argparse
	parser = argparse.ArgumentParser(
		description="Compute optimal active subspace dimension and polynomial degree for a given problem."
	)
	parser.add_argument(
		"--problem",
		type=str,
		help="Name of the problem to analyze (e.g., 'DYNAMNEWS-1')."
	)
	parser.add_argument(
		"--solver",
		type=str,
		help="Solver to use to solve Hyperparameter problem.",
		default= "RANDS"
	)

	parser.add_argument(
		"--problem_dim",
		type=int, 
		help='dimension of Problem',
		default=None
	)

	args = parser.parse_args()
	main(args.problem, args.solver, args.problem_dim)