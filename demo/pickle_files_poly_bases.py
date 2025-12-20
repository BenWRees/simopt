"""
This script will produce the pickle files for the numerical results present in the journal paper. 
TODO: Rewrite this so that we load in the same JSON but run on every solver, to reduce the number of 
"""

import sys
import os.path as o
import random
from multiprocessing import Process
from sys import argv

from math import ceil, log

import numpy as np
from mrg32k3a.mrg32k3a import MRG32k3a

sys.path.append(
	o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), ".."))
)  # type:ignore

# Import the ProblemSolver class and other useful functions
from simopt.experiment_base import (
	ProblemSolver,
	instantiate_solver,
	instantiate_problem
)

from simopt.base import (
    Problem,
    Solution,
)

from simopt.solvers.active_subspaces.compute_optimal_dim import find_best_subspace_dimension, find_best_polynomial_degree

from simopt.linear_algebra_base import finite_difference_gradient

#! This needs to be populated from the results found through hyperparameter search
problems_optimal_hyper: dict = {
	'DYNAMNEWS-1': {'subspace_dimension': 5, 'polynomial_degree': 3},
	'FACSIZE-1': {'subspace_dimension': 4, 'polynomial_degree': 2},
	'FACSIZE-2': {'subspace_dimension': 6, 'polynomial_degree': 2},
	'CONTAM-2': {'subspace_dimension': 3, 'polynomial_degree': 2},
	'AIRLINE-1': {'subspace_dimension': 5, 'polynomial_degree': 3},
	'ROSENBROCK-1': {'subspace_dimension': 4, 'polynomial_degree': 2},
	'NETWORK-1': {'subspace_dimension': 6, 'polynomial_degree': 3},
}

def main(solver_name: str, problem_name: str, dim_size: int, solver_factors: dict, budget: int, macroreplication_no: int ) -> None :				
	#create multiple processes each on a different solver name
	file_name_path = run_experiment(solver_name, problem_name, dim_size, macroreplication_no, solver_factors, budget)
	print(f'SAVED AT {file_name_path}')


def run_experiment(solver_name, problem_name, dim_size, macroreplication_no, solver_factors, budget) -> str :
	"""
		Run an experiment of a solver on a problem and store the results in a .pickle file.

	Args:
		solver_name (str): Name of the solver
		problem_name (str): Name of the problem
		dim_size (int): Dimension size for the problem
		macroreplication_no (int): Number of macroreplications to run
		solver_factors (dict): Fixed factors for the solver
		budget (int): Budget for the problem
	"""

	file_name_path = (
		"experiments/outputs/" + solver_name + "_on_" + problem_name + ".pickle"
	)
	#These while loops are expected to constantly create new problem, model, and solver factors until a solver factor works 
	solver = None 
	problem = None 
	while solver == None :
		try :
			solver = instantiate_solver(solver_name, solver_factors)
		except ValueError :
			pass 

	while problem == None :
		try :
			problem = instantiate_problem(problem_name, update_problem_factor_dimensions(problem_name, dim_size, budget), update_model_factors_dimensions(problem_name, dim_size))
		except ValueError: 
			pass
	# print(f"Results will be stored as {file_name_path}.")

	#Add some additional factors to the problem and solver 
	if solver_name == 'ASTROMoRF' or solver_name == 'OMoRF':
			solver.factors['initial subspace dimension'] = problems_optimal_hyper[problem_name]['subspace_dimension']
			solver.factors['polynomial degree'] = problems_optimal_hyper[problem_name]['polynomial_degree']
			
	myexperiment = ProblemSolver(problem=problem, solver=solver, file_name_path=file_name_path)
	
	myexperiment.run(n_macroreps=macroreplication_no)

	return file_name_path


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
		new_factors= {
			'process_prob': [1/new_dim] * new_dim,
			'cost_process': [0.1 / (x + 1) for x in range(new_dim)],
			'cost_time': [0.005] * new_dim,
			'mode_transit_time':[x + 1 for x in range(new_dim)],
			'lower_limits_transit_time': [0.5 + x for x in range(new_dim)],
			'upper_limits_transit_time': [1.5 + x for x in range(new_dim)],
			'n_networks': new_dim,
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
		new_factors =  {
			'initial_solution': (0.1,) * new_dim,
			'budget': budget,
		}
	return new_factors

def load_json(json_path: str) -> list[str] :
	"""
	Load configuration from a JSON file to get solver names

	Args:
		json_path (str): Path to the JSON configuration file.
	Returns:
		list[str]: A list of solver names.
	"""
	import json
	with open(json_path, 'r') as f:
		config = json.load(f)

	# Extract individual variables if needed
	solver_names = config.get("solver_names", [])

	return solver_names

if __name__ == "__main__":

	#Pass arguments
	solver_name = argv[1]
	problem_name = argv[2]
	dim_size = int(argv[3])
	solver_factors = eval(argv[4])
	budget = int(argv[5])
	macroreplication_no = int(argv[6])

	diag = {
		'solver name': solver_name, 
		'problem name': problem_name,
		'problem dimension': dim_size,
		'solver fixed factors': solver_factors,
		'simulation budget': budget,
		'number of macroreplications': macroreplication_no
	}

	print('DIAGNOSTICS: ')
	[print(f'{a} = {b}') for a,b in diag.items()]

	main(solver_name, problem_name, dim_size, solver_factors, budget, macroreplication_no)