"""This script will produce the pickle files for the numerical results present in the journal paper.
TODO: Rewrite this so that we load in the same JSON but run on every solver, to reduce the number of 
"""

import os.path as o
import os 
import random
import sys
from sys import argv
import pickle
from pathlib import Path


import numpy as np


sys.path.append(
	o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), ".."))
)  # type:ignore

# Deterministic seeding helper
import hashlib
# Import the ProblemSolver class and other useful functions
from simopt.experiment_base import (
	ProblemSolver,
	instantiate_problem,
	instantiate_solver,
	post_normalize,	
	Problem,
	EXPERIMENT_DIR
)

from simopt.base import Solver, Problem

problems_optimal_hyper: dict = {
	'DYNAMNEWS-1': {'subspace_dimension': 20, 'polynomial_degree': 4}, #Optimal
	'SAN-1': {'subspace_dimension': 40, 'polynomial_degree': 4}, #Optimal
	'ROSENBROCK-1': {'subspace_dimension': 5 , 'polynomial_degree': 4}, #Optimal
	'NETWORK-1': {'subspace_dimension': 20, 'polynomial_degree': 4}, #optimal
	'SSCONT-1': {'subspace_dimension': 1, 'polynomial_degree': 2}, #Optimal
}

SCALABLE_PROBLEMS = [
	"FIXEDSAN-1", 
	"NETWORK-1", 
	"ROSENBROCK-1", 
	"SAN-1", 
	"DYNAMNEWS-1", 
	"FACSIZE-2", 
	"FACSIZE-1", 
	"CONTAM-2"
	]

solver_renames = {
	'ASTROMORF': 'ASTROMoRF',
	'OMoRF': 'OMoRF',
	'ADAM': 'ADAM',
	'ASTRODF': 'ASTRO-DF',
	'NELDMD': 'NELDER MEAD',
	'RNDSRCH': 'RANDOM SEARCH',
	'STRONG': 'STRONG',
}

def main(solver_name: str, problem_name: str, solver_factors: dict, budget: int, macroreplication_no: int, dim_size: int | None = None, output_dir: str | None = None ) -> None :
	# create multiple processes each on a different solver name
	if dim_size is None :
		dim_size = 100 # Default dimension size for non scalable problems
	else :
		file_name_path = run_experiment(solver_name, problem_name, dim_size, macroreplication_no, solver_factors, budget, output_dir=output_dir)
	print(f'SAVED AT {file_name_path}')

def make_checkpoint_path(
	solver: Solver,
	problem: Problem,
	experiment_tag: str | None = None,
) -> Path:
	"""
	Construct a checkpoint path for a given solver/problem pair.

	experiment_tag can be used to distinguish different runs.
	"""
	solver_name = solver.name
	problem_name = problem.name
	tag = f"_{experiment_tag}" if experiment_tag else ""
	fname = f"{solver_name}_on_{problem_name}{tag}.resume.pickle"
	return EXPERIMENT_DIR / fname


def load_or_create_problemsolver(
	solver: Solver,
	problem: Problem,
	experiment_tag: str | None = None,
	create_pickle: bool = True,
) -> tuple[ProblemSolver, Path]:
	"""
	Load an existing ProblemSolver from a checkpoint if present,
	otherwise create a new one.
	"""
	checkpoint_path = make_checkpoint_path(solver, problem, experiment_tag)

	if checkpoint_path.exists():
		# Resume from existing checkpoint
		with checkpoint_path.open("rb") as f:
			ps = pickle.load(f)
		print(f"Resuming from checkpoint {checkpoint_path}")
	else:
		# Fresh ProblemSolver
		ps = ProblemSolver(
			solver=solver,
			problem=problem,
			create_pickle=create_pickle,
		)
		print(f"Starting new experiment; no checkpoint at {checkpoint_path}")

	return ps, checkpoint_path


def run_experiment_resumable(
	solver: Solver,
	problem: Problem,
	n_macroreps_total: int,
	n_jobs: int = -1,
	macroreps_per_chunk: int = 10,
	experiment_tag: str | None = None,
	create_pickle: bool = True,
) -> tuple[ProblemSolver, Path]:
	"""
	High-level driver: load or create a ProblemSolver, then run it
	with run_resumable until the desired total number of macroreps
	is reached. Safe to call multiple times; each call will resume.
	"""
	ps, checkpoint_path = load_or_create_problemsolver(
		solver=solver,
		problem=problem,
		experiment_tag=experiment_tag,
		create_pickle=create_pickle,
	)

	# ps.n_macroreps may already be > 0 if resuming
	already_done = getattr(ps, "n_macroreps", 0)
	print(f"ProblemSolver currently has {already_done} macroreps completed.")

	if already_done >= n_macroreps_total:
		print("Requested total macroreps already reached; nothing to do.")
		return ps

	ps.run_resumable(
		n_macroreps_total=n_macroreps_total,
		n_jobs=n_jobs,
		macroreps_per_chunk=macroreps_per_chunk,
		checkpoint_path=checkpoint_path,
	)

	return ps, checkpoint_path



def run_experiment(solver_name: str, problem_name: str, problem_dim: int, macroreplication_no: int, solver_factors: dict, budget: int, output_dir: str | None = None) -> str :
	"""Run an experiment of a solver on a problem and store the results in a .pickle file.

	Args:
		solver_name (str): Name of the solver
		problem_name (str): Name of the problem
		problem_dim (int): problem dimension size
		macroreplication_no (int): Number of macroreplications to run
		solver_factors (dict): Fixed factors for the solver
		budget (int): Budget for the problem
	"""
	file_name_path = f'{solver_name}_on_{problem_name}_crn_{"True" if solver_factors["crn_across_solns"] else "False"}'

	file_name_path += '.pickle'

	# If an output directory was provided, ensure it exists and prefix filenames
	if output_dir:
		Path(output_dir).mkdir(parents=True, exist_ok=True)
		file_name_path = str(Path(output_dir) / file_name_path)

	#add problems_optimal_hyper to solver factors if solver is ASTROMORF
	if solver_name == 'ASTROMORF':
		optimal_hyper = problems_optimal_hyper.get(problem_name, {})
		if 'subspace_dimension' in optimal_hyper :
			solver_factors['initial subspace dimension'] = optimal_hyper['subspace_dimension']
		if 'polynomial_degree' in optimal_hyper :
			solver_factors['polynomial degree'] = optimal_hyper['polynomial_degree']

	if solver_name == 'OMoRF' : 
		optimal_hyper = problems_optimal_hyper.get(problem_name, {})
		if 'subspace_dimension' in optimal_hyper :
			solver_factors['initial subspace dimension'] = optimal_hyper['subspace_dimension']
	


	# Instantiate solver once; surface errors clearly rather than looping forever
	try:
		solver = instantiate_solver(
			solver_name=solver_name,
			fixed_factors=solver_factors,
			solver_rename=solver_renames[solver_name],
		)
	except KeyError as e:
		raise KeyError(f"Unknown solver code '{solver_name}'. Ensure it matches a class_name_abbr and is present in solver_renames.") from e
	except ValueError as e:
		raise RuntimeError(f"Failed to instantiate solver '{solver_name}': {e}") from e

	problem = scale_dimension(problem_name=problem_name, dimension=problem_dim, budget=budget)
		
	
	print(f'Running {solver_name} on {problem_name} with budget {budget}, dim_size {solver.factors.get("initial subspace dimension", "N/A")}, polynomial basis {solver_factors.get("polynomial basis", "N/A")} for {macroreplication_no} macroreplications.')
	n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
	myexperiment, checkpoint_path = run_experiment_resumable(
		solver=solver,
		problem=problem,
		n_macroreps_total=macroreplication_no,      # total you want in the end
		n_jobs=-1,
		macroreps_per_chunk=1,
		experiment_tag="run1",      # optional, to distinguish runs
	)

	if getattr(myexperiment, "n_macroreps", 0) >= macroreplication_no:	  
		print(f'Running Post-Replications and post-normalization and saving to {checkpoint_path}')
		postrep_no = macroreplication_no * 20
		myexperiment.post_replicate(n_postreps=postrep_no)
		file_name_pickle = f'{solver_name}_ON_{problem_name}_CRN({solver_factors["crn_across_solns"]})_POSTREPS.pickle'
		if output_dir:
			file_name_pickle = str(Path(output_dir) / file_name_pickle)
		myexperiment.record_experiment_results(file_name_pickle)
		myexperiment.log_experiment_results(file_path=file_name_path.replace('.pickle', '.txt'))
	else : 
		print(f'Not enough macroreplications done to run post-replications. Only {myexperiment.n_macroreps} done out of {macroreplication_no} requested.')
		

	return file_name_path



def scale_dimension(problem_name: str, dimension: int, budget: int) -> Problem:
	"""
	Instantiate a problem with a scaled dimension.
	
	All model and problem factors that depend on the dimension are updated
	before instantiation to ensure consistency.
	
	Args:
		problem_name: The abbreviated name of the problem (e.g., "FACSIZE-2")
		dimension: The desired dimension for the problem
		
	Returns:
		A Problem instance configured for the specified dimension
	"""
	if problem_name not in SCALABLE_PROBLEMS:
		# For non-scalable problems, just instantiate with defaults
		return instantiate_problem(problem_name, {'budget': budget})
	
	# Build the factors for the new dimension
	model_factors = get_scaled_model_factors(problem_name, dimension)
	problem_factors = get_scaled_problem_factors(problem_name, dimension)
	problem_factors['budget'] = budget
	
	# Instantiate the problem with the scaled factors
	problem = instantiate_problem(
		problem_name,
		problem_fixed_factors=problem_factors,
		model_fixed_factors=model_factors
	)
	
	# Set the problem dimension explicitly
	problem.dim = dimension
	
	# Post-initialization updates for factors that can't be validated during construction
	post_init_updates(problem, problem_name, dimension)
	
	return problem


def get_scaled_model_factors(problem_name: str, dimension: int) -> dict:
	"""
	Generate model factors scaled to the specified dimension.
	
	Args:
		problem_name: The abbreviated name of the problem
		dimension: The target dimension
		
	Returns:
		Dictionary of model factors appropriate for the dimension
	"""
	# Deterministic RNGs from problem name + dimension
	seed = int(hashlib.sha256(f"{problem_name}:{dimension}".encode()).hexdigest(), 16) % (2**32)
	rng_py = random.Random(seed)
	rng_np = np.random.default_rng(seed)

	if problem_name == 'DYNAMNEWS-1':
		return {
			'num_prod': dimension,
			'c_utility': [float(6 + j) for j in range(dimension)],
			'init_level': [3] * dimension,
			'price': [9.0] * dimension,
			'cost': [5.0] * dimension,
		}
	
	elif problem_name in ('FACSIZE-1', 'FACSIZE-2'):
		# Use diagonal covariance to avoid expensive Cholesky and reduce rejection rate
		# With mean=500 and std=50 (variance=2500), P(X<0) ≈ 0 for each dimension
		# This makes rejection sampling nearly instant
		variance = 2500.0  # std = 50, mean = 500, so P(X<0) is negligible
		cov_matrix = np.eye(dimension) * variance
		return {
			'mean_vec': [500.0] * dimension,
			'cov': cov_matrix.tolist(),
			'capacity': [float(rng_py.randint(100, 900)) for _ in range(dimension)],
			'n_fac': dimension,
		}
	
	elif problem_name == 'SAN-1':
		# Calculate appropriate num_nodes for the number of edges (dimension)
		# For a DAG: we need num_nodes such that we can have 'dimension' edges
		# with a path from node 1 to num_nodes
		num_nodes = compute_num_nodes_for_dag(dimension)
		arcs = build_san_dag(num_nodes, dimension, rng=rng_py)
		return {
			'num_arcs': dimension,
			'num_nodes': num_nodes,
			'arcs': arcs,
			'arc_means': tuple(round(rng_py.uniform(1, 10), 2) for _ in range(dimension)),
		}
	
	elif problem_name == 'FIXEDSAN-1':
		num_nodes = max(2, rng_py.randint(2, max(2, dimension)))
		return {
			'num_arcs': dimension,
			'num_nodes': num_nodes,
			'arc_means': tuple(float(rng_py.randint(1, 10)) for _ in range(dimension)),
		}
	
	elif problem_name == 'ROSENBROCK-1':
		return {
			'x': (2.0,) * dimension,
			'variance': 0.4,
		}
	elif problem_name == 'ZAKHAROV-1':
		return {
			'x': (2.0,) * dimension,
			'variance': 0.1,
		}   
	
	elif problem_name == 'NETWORK-1':
		process_prob_elem = 1.0 / dimension
		mode_transit_time = [round(rng_py.uniform(0.01, 5), 3) for _ in range(dimension)]
		return {
			'process_prob': [process_prob_elem] * dimension,
			'cost_process': [0.1 / (x + 1) for x in range(dimension)],
			'cost_time': [round(rng_py.uniform(0.01, 1), 3) for _ in range(dimension)],
			'mode_transit_time': mode_transit_time,
			'lower_limits_transit_time': [x / 2 for x in mode_transit_time],
			'upper_limits_transit_time': [2 * x for x in mode_transit_time],
			'n_networks': dimension,
		}
	
	elif problem_name == 'CONTAM-2':
		return {
			'stages': dimension,
			'prev_decision': (0.0,) * dimension,
		}
	
	return {}


def get_scaled_problem_factors(problem_name: str, dimension: int) -> dict:
	"""
	Generate problem factors scaled to the specified dimension.
	
	Only includes factors that will pass validation during construction.
	Factors that depend on model state are updated post-initialization.
	
	Args:
		problem_name: The abbreviated name of the problem
		dimension: The target dimension
		
	Returns:
		Dictionary of problem factors appropriate for the dimension
	"""
	if problem_name == 'DYNAMNEWS-1':
		return {
			'initial_solution': (3.0,) * dimension,
		}
	
	elif problem_name in ('FACSIZE-1', 'FACSIZE-2'):
		# NOTE: installation_costs is validated against NUM_FACILITIES constant (=3)
		# So we can't pass it here - it will be updated post-initialization
		return {
			'initial_solution': (100.0,) * dimension,
			'installation_budget': 500.0 * (dimension / 3),  # Scale budget with dimension
		}
	
	elif problem_name in ('SAN-1', 'FIXEDSAN-1'):
		# NOTE: arc_costs is validated against NUM_ARCS constant (=13)
		# So we can't pass it here - it will be updated post-initialization
		return {
			'initial_solution': (1.0,) * dimension,
		}
	
	elif problem_name == 'ROSENBROCK-1':
		return {
			'initial_solution': (2.0,) * dimension,
		}
	
	elif problem_name == 'ZAKHAROV-1':
		return {
			'initial_solution': (2.0,) * dimension,
		}
	
	elif problem_name == 'NETWORK-1':
		return {
			'initial_solution': (1.0 / dimension,) * dimension,
		}
	
	elif problem_name == 'CONTAM-2':
		return {
			'initial_solution': (0.0,) * dimension,
		}
	
	return {}


def post_init_updates(problem: Problem, problem_name: str, dimension: int) -> None:
	"""
	Update problem factors after initialization for factors that couldn't be set during construction.
	
	Some factors are validated against hardcoded constants during construction,
	so they need to be updated after the problem is instantiated.
	
	Args:
		problem: The problem instance to update
		problem_name: The abbreviated name of the problem
		dimension: The target dimension
	"""
	if problem_name in ('FACSIZE-1', 'FACSIZE-2'):
		# Update installation_costs after construction to match the new dimension
		problem.factors['installation_costs'] = (1.0,) * dimension
	
	elif problem_name in ('SAN-1', 'FIXEDSAN-1'):
		# Update arc_costs after construction to match the new dimension
		# arc_costs is used in replicate(): np.sum(arc_costs / x)
		problem.factors['arc_costs'] = (1.0,) * dimension


def compute_num_nodes_for_dag(num_edges: int) -> int:
	"""
	Compute an appropriate number of nodes for a DAG with the given number of edges.
	
	For a DAG with n nodes where we need a path from 1 to n:
	- Minimum edges needed: n-1 (a simple path)
	- Maximum edges possible: n*(n-1)/2 (complete DAG)
	
	We want to find the smallest n such that n*(n-1)/2 >= num_edges
	and n-1 <= num_edges (so we have enough edges for connectivity).
	
	Args:
		num_edges: Desired number of edges
		
	Returns:
		Number of nodes to use
	"""
	# We need at least num_edges + 1 nodes in the worst case (simple path),
	# but we want fewer nodes with more edges between them.
	# Solve: n*(n-1)/2 >= num_edges => n^2 - n - 2*num_edges >= 0
	# n >= (1 + sqrt(1 + 8*num_edges)) / 2
	
	import math
	min_nodes = int(math.ceil((1 + math.sqrt(1 + 8 * num_edges)) / 2))
	
	# Ensure we have at least 2 nodes and the path is possible
	min_nodes = max(2, min_nodes)
	
	# Also ensure num_edges >= min_nodes - 1 (need at least a spanning path)
	# If not, we need more nodes
	while min_nodes - 1 > num_edges:
		min_nodes -= 1
	
	return min_nodes


def build_san_dag(num_nodes: int, num_edges: int, rng: random.Random | None = None) -> list[tuple[int, int]]:
	"""
	Build a directed acyclic graph (DAG) suitable for the SAN model.
	
	The SAN model requires:
	1. Directed edges (arcs) from lower-numbered to higher-numbered nodes
	2. A path must exist from node 1 to node num_nodes
	3. Every node must be reachable from node 1 (for backtracking to work)
	
	This function first creates a simple sequential path from 1 to num_nodes
	(1→2→3→...→n), then adds additional random forward edges until reaching num_edges.
	
	Args:
		num_nodes: Number of nodes (numbered 1 to num_nodes)
		num_edges: Desired number of directed edges
		
	Returns:
		List of (source, target) tuples representing directed edges
		
	Raises:
		ValueError: If the requested configuration is impossible
	"""
	min_edges = num_nodes - 1  # Simple path from 1 to num_nodes
	max_edges = num_nodes * (num_nodes - 1) // 2  # Complete DAG
	
	if num_edges < min_edges:
		raise ValueError(
			f"Cannot create DAG with path 1→{num_nodes}: need at least {min_edges} edges, "
			f"but only {num_edges} requested"
		)
	
	if num_edges > max_edges:
		raise ValueError(
			f"Cannot create DAG with {num_edges} edges: maximum possible is {max_edges} "
			f"for {num_nodes} nodes"
		)
	
	edges = set()
	
	# Step 1: Create a guaranteed SEQUENTIAL path from node 1 to node num_nodes
	# This ensures every node has a predecessor reachable from node 1
	# Path: 1 → 2 → 3 → ... → num_nodes
	for i in range(1, num_nodes):
		edges.add((i, i + 1))
	
	# Step 2: Add additional random forward edges until we reach num_edges
	if len(edges) < num_edges:
		if rng is None:
			rng = random
		# Generate all possible forward edges not yet in the graph
		all_possible_edges = []
		for i in range(1, num_nodes):
			for j in range(i + 1, num_nodes + 1):
				edge = (i, j)
				if edge not in edges:
					all_possible_edges.append(edge)
        
		# Shuffle and add as many as needed
		rng.shuffle(all_possible_edges)
		edges_needed = num_edges - len(edges)
        
		for edge in all_possible_edges[:edges_needed]:
			edges.add(edge)
	
	return list(edges)

def validate_solver_and_problem_names(solver_name: str, problem_name: str) -> None:
	"""Pre-flight validation for clearer errors before heavy work."""
	try:
		_ = instantiate_solver(solver_name=solver_name, fixed_factors={}, solver_rename=solver_renames.get(solver_name, solver_name))
	except Exception as e:
		raise ValueError(f"Unknown or invalid solver code '{solver_name}'. Original error: {e}") from e

	try:
		_ = instantiate_problem(problem_name, problem_fixed_factors=None, model_fixed_factors=None)
	except Exception as e:
		raise ValueError(f"Unknown problem code '{problem_name}'. Original error: {e}") from e


def build_connected_graph(num_nodes: int, num_edges: int, rng: random.Random | None = None) -> list[tuple[int, int]]:
	"""
	Build a connected graph with the specified number of nodes and edges.
	
	Starts with a spanning tree to ensure connectivity, then adds random edges
	until the desired number is reached.
	
	Args:
		num_nodes: Number of nodes in the graph
		num_edges: Desired number of edges
		
	Returns:
		List of (node1, node2) tuples representing edges
	"""
	if num_edges < num_nodes - 1:
		raise ValueError(f"Cannot create connected graph: need at least {num_nodes - 1} edges for {num_nodes} nodes")
	
	edges = set()
    
	if rng is None:
		rng = random

	# Create a spanning tree first to ensure connectivity
	nodes = list(range(num_nodes))
	rng.shuffle(nodes)
	for i in range(1, num_nodes):
		a = nodes[i]
		b = nodes[rng.randint(0, i - 1)]
		edges.add((min(a, b), max(a, b)))
	
	# Add random edges until we reach the desired number
	max_possible_edges = num_nodes * (num_nodes - 1) // 2
	target_edges = min(num_edges, max_possible_edges)
	
	attempts = 0
	max_attempts = target_edges * 10  # Prevent infinite loop
	while len(edges) < target_edges and attempts < max_attempts:
		a = rng.randint(0, num_nodes - 1)
		b = rng.randint(0, num_nodes - 1)
		if a != b:
			edges.add((min(a, b), max(a, b)))
		attempts += 1
	
	return list(edges)

def load_json(json_path: str) -> list[str] :
	"""Load configuration from a JSON file to get solver names

	Args:
		json_path (str): Path to the JSON configuration file.

	Returns:
		list[str]: A list of solver names.
	"""
	import json
	with open(json_path) as f:
		config = json.load(f)

	# Extract individual variables if needed
	solver_names = config.get("solver_names", [])

	return solver_names

if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Run OMoRF pickle generation for a solver/problem")
	parser.add_argument("solver_name", type=str)
	parser.add_argument("problem_name", type=str)
	parser.add_argument("dim_size", type=str, help="Dimension size or 'None'")
	parser.add_argument("solver_factors", type=str, help="Python dict literal of solver fixed factors")
	parser.add_argument("budget", type=int)
	parser.add_argument("macroreplication_no", type=int)
	parser.add_argument("--output-dir", type=str, default=None)

	args = parser.parse_args()

	solver_name = args.solver_name
	problem_name = args.problem_name
	dim_size = None if args.dim_size.lower() == "none" else int(args.dim_size)
	solver_factors = eval(args.solver_factors)
	budget = args.budget
	macroreplication_no = args.macroreplication_no
	output_dir = args.output_dir

	diag = {
		'solver name': solver_name,
		'problem name': problem_name,
		'problem dimension': dim_size,
		'solver fixed factors': solver_factors,
		'simulation budget': budget,
		'number of macroreplications': macroreplication_no,
	}

	validate_solver_and_problem_names(solver_name, problem_name)

	if dim_size is not None:
		main(solver_name, problem_name, solver_factors, budget, macroreplication_no, dim_size=dim_size, output_dir=output_dir)
	else:
		main(solver_name, problem_name, solver_factors, budget, macroreplication_no, output_dir=output_dir)