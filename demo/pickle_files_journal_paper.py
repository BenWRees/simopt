"""
This script will produce the pickle files for the numerical results present in the journal paper. 
"""

import sys
import os.path as o
import random
from multiprocessing import Process
from sys import argv

from math import ceil, log

import numpy as np

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

def find_optimal_d(self, problem_name: str) -> int : 
	"""
		Constructs a proxy initial guess of the subspace dimension through the following steps:
			1. Obtain alog(n) random solutions around the problem [x_1,x_2,...,x_{alog(n)}]
			2. Estimate the gradient vectors of the problem at each random solution 
			3. Construct the uncentred covariance matrix of functional derivatives C through a Monte Carlo estimate on the alog(n) gradient vectors
			4. Factorise C through eigendecomposition and sort the eigenpairs into a list
			5. Create a second list of tuples (i,j) where i is the index of the eigenpairs in the list from step 2, and j is the distance between the eigen pairs of 
			|\lambda_{i-1}-\lambda_{i}|. 
			6. pick d to be the i in the list of tuples that contains the smallest j in their tuple out of all j's in the list of tuples

	Args:
		problem_name (str): Name of the problem

	Returns: 
		int: The subspace dimension

	NOTE: 
		This method of finding a parameter for d will require an additional alog(n^{2n}) responses of the simulation model.
		#! Before implementing this proxy, we need to find a way of finding this covariance matrix without taking an exponential number of responses.
		- Look at some form of sensitivity analysis to find the covariance matrix


	"""
	problem: Problem = self.get_problem_instance(problem_name)

	no_solns = ceil(10 * problem.dim * log(problem.dim))
	# Designate random number generator for random sampling
	find_next_soln_rng = self.rng_list[1]

	# Generate many dummy solutions without replication only to find a reasonable maximum radius
	dummy_solns: list[Solution] = []
	for _ in range(no_solns):
		random_soln_tuple = problem.get_random_solution(find_next_soln_rng)
		random_soln = self.create_new_solution(random_soln_tuple, problem)
		dummy_solns.append(random_soln)

	# Calculate the gradient of the problem at each dummy solution
	gradients = []
	for sol in dummy_solns:
		grad = self.finite_difference(sol, problem) 
		gradients.append(grad)

	C = sum([np.outer(a,a) for a in gradients])/no_solns
	#decompose C into eigenpairs and sort the eigenpairs by descendeing eigenvalue
	# Compute eigenvalues and eigenvectors
	eigenvalues, eigenvectors = np.linalg.eigh(C)
	
	# Create list of (eigenvalue, eigenvector) tuples
	eigenpairs = [(eigenvalues[i], eigenvectors[:, i]) for i in range(len(eigenvalues))]
	
	# Sort by eigenvalue in descending order
	eigenpairs.sort(key=lambda pair: pair[0], reverse=True)

	eigenvalues = [pair[0] for pair in eigenpairs]
	# gaps = [eigenvalues[i - 1] - eigenvalues[i] for i in range(1, len(eigenvalues))]
	# d = gaps.index(min(gaps)) + 1

	

	# find r by finding the smallest d that keeps the explained variance above a certain threshold
	numerator = [(i,sum(eigenvalues[:i])) for i in range(1, len(eigenvalues))]
	explained_variance = [(i+1, numerator[i][1]/sum(eigenvalues)) for i in range(len(numerator))]
	sig_lvl = 0.9
	explained_variance_filter = [i for i in explained_variance[:-1] if i[1] >= sig_lvl] #! Remove the last one as it is always 1.0
	#if no eigenvalues are above the threshold, other than the  then we need to reduce the sig_lvl
	while len(explained_variance) == 0  :
		sig_lvl -= 0.05
		explained_variance_filter = [i for i in explained_variance[:-1] if i[1] >= sig_lvl]
	
	d = explained_variance_filter[0][0] 
	self.U_init = np.column_stack([vec for _, vec in eigenpairs[:d]])

	# print(f'the final significance level is {sig_lvl} and the subspace dimension is {d}')

	return d 


def main(solver_name: str, problem_name: str, dim_size: int, solver_factors: dict, budget: int) -> None :
		macroreplication_no = 100  # Number of macroreplications for each experiment					

		processes = []
		if solver_name == 'ASTROMoRF' or solver_name == 'OMoRF':
			solver_factors['initial subspace dimension'] = find_optimal_d(problem_name)

		p = Process(target=run_experiment, args=(solver_name, problem_name, dim_size, macroreplication_no, solver_factors, budget))
		p.start()
		processes.append(p)	

		for process in processes:
			process.join()





def run_experiment(solver_name, problem_name, dim_size, macroreplication_no, solver_factors, budget) -> None :
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
	solver = instantiate_solver(solver_name, solver_factors)
	problem = instantiate_problem(problem_name, update_problem_factor_dimensions(problem_name, dim_size, budget), update_model_factors_dimensions(problem_name, dim_size))

	# print(f"Results will be stored as {file_name_path}.")
	myexperiment = ProblemSolver(problem=problem, solver=solver, file_name_path=file_name_path)
	
	myexperiment.run(n_macroreps=macroreplication_no)

	# print(f"Finished running solver {solver_name} on problem {problem_name}.")

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
	if problem_name == 'DYNAMNEWS-1':
		return {
			'num_prod': new_dim,
			'c_utility': [6 + j for j in range(new_dim)],
			'init_level': [3] * new_dim,
			'price': [9] * new_dim,
			'cost': [5] * new_dim,
		}
	elif problem_name == 'FACSIZE-1':
		A = np.random.rand(new_dim, new_dim)
		return {
			'mean_vec': [100] * new_dim,
			'cov': (np.dot(A, A.T) * 100).tolist(),
			'capacity': [random.randint(100,900) for _ in range(new_dim)],
			'n_fac': new_dim,
		}
	elif problem_name == 'FIXEDSAN-1' :
		return {
			'num_arcs': new_dim,
			'num_nodes': random.randint(new_dim//2, new_dim),
			'arc_means': (1,) * new_dim ,
		}
	#TODO: check over these factors 
	elif problem_name == 'AIRLINE-1' or problem_name == 'AIRLINE-2' :
		odf_leg_matrix = np.random.randint(0,2,(2, new_dim))
		new_legs, new_odfs = odf_leg_matrix.shape
		return {
			'num_classes': new_dim,
			'ODF_leg_matrix': new_dim,
			'prices': [random.randint(50,300) for _ in range(new_dim)],
			'capacity': [random.randint(20,150) for _ in range(new_dim)],
			'booking limits': [random.randint(5,20) for _ in range(new_dim)],
			'alpha': np.random.randint(0,2,(new_dim,)).tolist(),
			'beta': np.random.randint(0,2,(new_dim,)).tolist(),
			'gamma_shape': (2.0,) * new_dim,
			'gamma_scale': (50.0,) * new_dim,

		}
	elif problem_name == 'NETWORK-1' :
		return {
			'process_prob': [0.1] * new_dim,
			'cost_process': [0.1 / (x + 1) for x in range(new_dim)],
			'cost_time': [0.005] * new_dim,
			'mode_transit_time':[x + 1 for x in range(new_dim)],
			'lower_limits_transit_time': [0.5 + x for x in range(new_dim)],
			'upper_limits_transit_time': [1.5 + x for x in range(new_dim)],
			'n_networks': new_dim,
		}


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
	if problem_name == 'DYNAMNEWS-1':
		return {
			'initial_solution': (3,) * new_dim,
			'budget': budget,
		}
	elif problem_name == 'FACSIZE-1':
		return {
			'initial_solution': (100,) * new_dim,
			'installation_costs': (1,) * new_dim,
			'epsilon': 0.05,
			'budget': budget,
		}
	elif problem_name == 'FACSIZE-2':
		return {
			'initial_solution': (300,) * new_dim,
			'installation_costs': (1,) * new_dim,
			'installation_budget': 500,
			'budget': budget,
		}
	elif problem_name == 'FIXEDSAN-1' : 
		return {
			'initial_solution': (10,) * new_dim,
			'arc_costs': (1,) * new_dim,
			'budget': budget,
		}
	elif problem_name == 'AIRLINE-1' or problem_name == 'AIRLINE-2' :
		return {
			'initial_solution': (3,) * new_dim,
			'budget': budget,
		}
	elif problem_name == 'NETWORK-1' :
		return {
			'initial_solution': (0.1,) * new_dim,
			'budget': budget,
		}

if __name__ == "__main__":
	solver_name = argv[1]
	problem_name = argv[2]
	dim_size = int(argv[3])
	solver_factors = eval(argv[4])
	budget = int(argv[5])

	if solver_name == 'ASTROMoRF' : 
		solver_factors['polynomial basis']= 'ChebyshevTensorBasis'

	main(solver_name, problem_name, dim_size, solver_factors, budget)