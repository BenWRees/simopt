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

from simopt.solvers.astromorf import PolyBasisType

#! This needs to be populated from the results found through hyperparameter search
problems_optimal_hyper: dict = {
	'DYNAMNEWS-1': {'subspace_dimension': 1, 'polynomial_degree': 2}, #Optimal
	'FACSIZE-1': {'subspace_dimension': 2, 'polynomial_degree': 2}, #Optimal
	'CONTAM-2': {'subspace_dimension': 1, 'polynomial_degree': 4}, #Optimal
	'ROSENBROCK-1': {'subspace_dimension': 9 , 'polynomial_degree': 2}, #Optimal
	'NETWORK-1': {'subspace_dimension': 6, 'polynomial_degree': 4}, #optimal
	'FIXEDSAN-1': {'subspace_dimension': 1, 'polynomial_degree': 2}, #Optimal
}

solver_renames = {
	'ASTROMORF': 'ASTROMoRF',
	'OMoRF': 'OMoRF',
	'ADAM': 'ADAM',
	'ASTRODF': 'ASTRO-DF',
	'NELDMD': 'NELDER MEAD',
	'RNDSRCH': 'RANDOM SEARCH',
	'STRONG': 'STRONG',
}

poly_basis = {
	'HERMITE': PolyBasisType.HERMITE,
	'LEGENDRE': PolyBasisType.LEGENDRE,
	'CHEBYSHEV': PolyBasisType.CHEBYSHEV,
	'MONOMIAL': PolyBasisType.MONOMIAL,
	'NATURAL': PolyBasisType.NATURAL,
	'MONOMIAL_POLY': PolyBasisType.MONOMIAL_POLY,
	'LAGUERRE': PolyBasisType.LAGUERRE,
	'NFPOLY': PolyBasisType.NFP,
	'LAGRANGE': PolyBasisType.LAGRANGE,
}

def main(solver_name: str, problem_name: str, solver_factors: dict, budget: int, macroreplication_no: int, dim_size: int | None = None ) -> None :				
	#create multiple processes each on a different solver name
	file_name_path = run_experiment(solver_name, problem_name, dim_size, macroreplication_no, solver_factors, budget)
	print(f'SAVED AT {file_name_path}')


def run_experiment(solver_name, problem_name, dim_size, macroreplication_no, solver_factors, budget) -> str :
	"""
		Run an experiment of a solver on a problem and store the results in a .pickle file.

	Args:
		solver_name (str): Name of the solver
		problem_name (str): Name of the problem
		dim_size (int): Subspace Dimension for the solver (if applicable)
		macroreplication_no (int): Number of macroreplications to run
		solver_factors (dict): Fixed factors for the solver
		budget (int): Budget for the problem
	"""

	file_name_path = f'{solver_name}_on_{problem_name}_budget{budget}_crn{solver_factors["crn_across_solns"]}'

	#if polynomial basis is specified then add to the file name path
	if solver_factors['polynomial basis'] is not None :
		file_name_path += f'_basis{solver_factors["polynomial basis"]}'

	#if initial subspace dimension is specified then add to the file name path
	if dim_size is not None :
		file_name_path += f'_dim{dim_size}'

	file_name_path += '.pickle'

	problem_dim = 100

	#If both dim_size and solver_factors['initial subspace dimension'] are None, set both to optimal values from hyperparameter search
	if dim_size is None and solver_factors.get('initial subspace dimension', None) is None :
		if (solver_name == 'ASTROMoRF' or solver_name == 'OMoRF'):
			solver_factors['initial subspace dimension'] = problems_optimal_hyper[problem_name]['subspace_dimension']
			solver_factors['polynomial degree'] = problems_optimal_hyper[problem_name]['polynomial_degree']

	#If dim_size is given but solver_factors['initial subspace dimension'] is None, set initial subspace dimension to dim_size and polynomial degree to optimal from hyperparameter search
	elif solver_factors['polynomial degree'] is None :
		if (solver_name == 'ASTROMoRF' or solver_name == 'OMoRF'):
			solver_factors['initial subspace dimension'] = dim_size
			solver_factors['polynomial degree'] = problems_optimal_hyper[problem_name]['polynomial_degree']

	#if solver_factors['polynomial basis'] is given but dim_size is None, set dim_size to initial subspace dimension and polynomial degree to optimal from hyperparameter search
	elif dim_size is None :
		if (solver_name == 'ASTROMoRF' or solver_name == 'OMoRF'):
			solver_factors['initial subspace dimension'] = problems_optimal_hyper[problem_name]['subspace_dimension']
			solver_factors['polynomial basis'] = poly_basis[solver_factors['polynomial basis']]

	#If both are given, set both to given values
	else : 
		if (solver_name == 'ASTROMoRF' or solver_name == 'OMoRF'):
			solver_factors['initial subspace dimension'] = dim_size
			solver_factors['polynomial basis'] = poly_basis[solver_factors['polynomial basis']]


	solver = None 
	problem = None 
	while solver == None :
		try :
			solver = instantiate_solver(solver_name=solver_name, fixed_factors=solver_factors, solver_rename=solver_renames[solver_name])
		except ValueError :
			pass 

	while problem == None :
		try :
			problem = instantiate_problem(problem_name, update_problem_factor_dimensions(problem_name, problem_dim, budget), update_model_factors_dimensions(problem_name, problem_dim))
		except ValueError: 
			pass

			
	myexperiment = ProblemSolver(problem=problem, solver=solver, file_name_path=file_name_path)
	
	print(f'Running {solver_name} on {problem_name} with budget {budget}, dim_size {solver.factors.get("initial subspace dimension", "N\\A")}, polynomial basis {solver_factors.get("polynomial basis", "N\\A")} for {macroreplication_no} macroreplications.')
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
			'num_arcs': new_dim,
			'num_nodes': np.random.randint(1, new_dim),
			'arc_means': tuple(np.random.randint(1,new_dim) for _ in range(new_dim))
		}
	elif problem_name == 'ROSENBROCK-1' :
		new_factors = {
			'x': (2.0,) * new_dim,
			'variance': 0.4
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
			'prev_decision': (0,) * new_dim,
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
	elif problem_name == 'ROSENBROCK-1' :
		new_factors = {
			'initial_solution': (2.0,) * new_dim,
			'budget': budget,
		}
	elif problem_name == 'FIXEDSAN-1' :
		new_factors = {
			'initial_solution': (10,) * new_dim,
			'budget': budget,
			'arc_costs': tuple(np.random.randint(1,10) for _ in range(new_dim))
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
	if argv[3].lower() == "none" : 		
		dim_size = None
	else :
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

	
	if dim_size is not None :
		main(solver_name, problem_name, solver_factors, budget, macroreplication_no, dim_size=dim_size)
	else :
		main(solver_name, problem_name, solver_factors, budget, macroreplication_no)