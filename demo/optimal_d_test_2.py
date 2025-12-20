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
	Solver
)

from simopt.linear_algebra_base import finite_difference_gradient

def calculate_eigenpairs(problem: Problem) -> list[tuple] : 
	#Create Problem and solution
	# problem_fixed_factors = update_problem_factor_dimensions(problem_name, dim, budget)
	# model_fixed_factors = update_model_factors_dimensions(problem_name, dim)
	# problem: Problem = instantiate_problem(problem_name, problem_fixed_factors, model_fixed_factors)

	# starting_solution = instantiate_solver(solver_name, solver_factors)

	no_solns = ceil(10 * problem.dim * log(problem.dim))
	
	# Designate random number generator for random sampling
	find_next_soln_rng = MRG32k3a()
	# starting_solution.solution_progenitor_rngs(problem.rng_list)
	# starting_solution.attach_rngs(problem.rng_list)


	dummy_solns: list[Solution] = []
	for _ in range(no_solns):
		random_rngs = [MRG32k3a() for _ in range(problem.model.n_rngs)]
		random_soln_tuple = problem.get_random_solution(find_next_soln_rng)
		random_soln = Solution(random_soln_tuple, problem)
		random_soln.attach_rngs(random_rngs)
		dummy_solns.append(random_soln)

	# Calculate the gradient of the problem at each dummy solution
	gradients = []
	for sol in dummy_solns:
		grad = finite_difference_gradient(sol, problem) 
		gradients.append(grad)

	C = sum([np.outer(a,a) for a in gradients])/no_solns
	#decompose C into eigenpairs and sort the eigenpairs by descendeing eigenvalue
	# Compute eigenvalues and eigenvectors
	eigenvalues, eigenvectors = np.linalg.eigh(C)
	
	# Create list of (eigenvalue, eigenvector) tuples
	eigenpairs = [(eigenvalues[i], eigenvectors[:, i]) for i in range(len(eigenvalues))]
	
	# Sort by eigenvalue in descending order
	eigenpairs.sort(key=lambda pair: pair[0], reverse=True)

	#sort gradients into matrix 
	
	return eigenpairs, C, np.stack(gradients)

def mp_support(q, sigma2=1.0):
	"""Return MP support [lambda_minus, lambda_plus] for aspect ratio q and variance sigma2."""
	# q is d/n for standard formulation, requires q in (0, inf). For q>1 the nonzero eigenvalues still follow MP but see note.
	sqrtq = np.sqrt(q)
	lambda_minus = sigma2 * (1.0 - sqrtq)**2
	lambda_plus  = sigma2 * (1.0 + sqrtq)**2
	return lambda_minus, lambda_plus

def mp_pdf_grid(q, sigma2=1.0, ng=10000):
	"""
	Compute MP pdf (sigma2=1 by default) on a fine grid via closed-form formula.
	Returns grid x and pdf values for x in [lambda_minus, lambda_plus].
	"""
	lam_min, lam_max = mp_support(q, sigma2)
	if lam_max <= lam_min:
		# degenerate; return tiny grid
		xs = np.array([lam_min, lam_max])
		pdf = np.zeros_like(xs)
		return xs, pdf
	xs = np.linspace(lam_min, lam_max, ng)
	# MP pdf: f(λ) = sqrt((λ_+ - λ)(λ - λ_-)) / (2π q λ σ^2)
	# guard against division by zero at λ ~ 0
	num = np.sqrt((lam_max - xs) * (xs - lam_min))
	denom = 2.0 * np.pi * q * xs * sigma2
	# set pdf to zero where denom is zero or negative
	pdf = np.zeros_like(xs)
	mask = denom > 0
	pdf[mask] = num[mask] / denom[mask]
	# numerical cleanup
	pdf = np.maximum(pdf, 0.0)
	return xs, pdf

def mp_cdf_grid(q, sigma2=1.0, ng=10000):
	"""Return MP cdf evaluated on grid and also the median value (useful for sigma2 matching)."""
	xs, pdf = mp_pdf_grid(q, sigma2, ng=ng)
	cdf = np.cumsum(pdf)
	# multiply by grid step
	dx = xs[1] - xs[0]
	cdf = cdf * dx
	# normalize (small numerical error possible)
	cdf = cdf / (cdf[-1] + 1e-20)
	# estimate median of MP (value where cdf ~ 0.5)
	# if cdf incomplete, fallback to center
	try:
		med_idx = np.searchsorted(cdf, 0.5)
		med_val = xs[min(med_idx, len(xs)-1)]
	except Exception:
		med_val = (xs[0] + xs[-1]) / 2.0
	return xs, cdf, med_val

def estimate_sigma2_by_median(eigs, n, d, ng=2000):
	"""
	Robust estimate of sigma^2 via matching median of empirical eigenvalues to MP-median (sigma2=1).
	This is robust to a few large signal eigenvalues.
	"""
	q = d / n
	# if q>1 swap to use MP for nonzero eigenvalues; but median matching still works with q_swapped
	if q > 1.0:
		# for d>n the nonzero eigenvalues count is n and aspect ratio for MP nonzero bulk is q' = n/d < 1
		q_eff = 1.0 / q
	else:
		q_eff = q
	# compute MP median for sigma2=1
	_, _, med_mp = mp_cdf_grid(q_eff, sigma2=1.0, ng=ng)
	# empirical median: use eigenvalues not dominated by a few large ones.
	# Use robust trimmed median: median of eigenvalues after dropping top 5% largest (if >10 elements)
	eigs_desc = np.sort(eigs)[::-1]
	m = len(eigs_desc)
	drop = max(1, int(0.05 * m)) if m > 20 else 0
	trimmed = eigs_desc[drop:]
	emp_med = np.median(trimmed)
	# estimate sigma2
	sigma2_hat = emp_med / (med_mp + 1e-24)
	# ensure positive
	sigma2_hat = float(max(sigma2_hat, 0.0))
	return sigma2_hat

def marchenko_pastur_cut(eigs, n_samples, tol=0.01, ng=5000, verbose=False):
	"""
	Given empirical eigenvalues `eigs` (1D array, any order),
	and number of gradient samples n_samples (n), compute MP-based cutoff.
	Returns:
	  - k_mp: number of eigenvalues considered signal (strictly > lambda_plus*(1+tol))
	  - lambda_minus, lambda_plus
	  - sigma2_hat, q
	  - mask_signal (boolean array aligned to eigs sorted descending)
	  - diagnostics dict
	Parameters:
	  - tol: relative tolerance above lambda_plus to call signal (default 1%)
	  - ng: grid points for MP pdf/cdf
	"""
	d = eigs.size
	n = int(n_samples)
	if n <= 0:
		raise ValueError("n_samples must be > 0")
	q = float(d) / float(n)
	# If d > n, we use effective q'=n/d for MP on nonzero eigenvalues and fit sigma accordingly.
	swapped = False
	if q > 1.0:
		swapped = True
		q_eff = 1.0 / q
	else:
		q_eff = q
	# estimate sigma^2 robustly
	sigma2_hat = estimate_sigma2_by_median(eigs, n, d, ng=ng)
	# compute lambda_- and lambda_+ for effective q and estimated sigma2
	lambda_minus, lambda_plus = mp_support(q_eff, sigma2=sigma2_hat)
	# If swapped, lambda bounds apply to nonzero eigenvalues (we are working with eigs_desc which are the nonzero ones)
	# Decide signal threshold: lambda_plus * (1 + tol)
	threshold = lambda_plus * (1.0 + tol)
	# count eigenvalues > threshold
	mask_signal = eigs > threshold
	k_mp = int(np.count_nonzero(mask_signal))
	# diagnostics
	total_energy = eigs.sum()
	signal_energy = eigs[mask_signal].sum() if total_energy > 0 else 0.0
	return k_mp, lambda_plus, sigma2_hat

def explained_energy(eigs, k):
	eigs = np.sort(eigs)[::-1]
	return np.sum(eigs[:k]) / np.sum(eigs)

def principal_angles(U, V):
	M = U.T @ V
	_, s, _ = np.linalg.svd(M)
	s = np.clip(s, -1, 1)
	return np.arccos(s)

def bootstrap_subspace_stability(G, k, n_boot=100):
	rng = MRG32k3a()
	n, d = G.shape
	C_full = (G.T @ G) / n
	vals, vecs = np.linalg.eigh(C_full)
	U_base = vecs[:, np.argsort(vals)[::-1][:k]]
	all_angles = []
	for _ in range(n_boot):
		idx = np.array([rng.binomialvariate(n, 1/n) for _ in range(n)])
		Gb = G[idx]
		Cb = (Gb.T @ Gb) / n
		vb, ve = np.linalg.eigh(Cb)
		Ub = ve[:, np.argsort(vb)[::-1][:k]]
		angles = principal_angles(U_base, Ub)
		all_angles.append(angles)
	all_angles = np.array(all_angles)
	return np.mean(all_angles, axis=0), np.std(all_angles, axis=0)

def find_optimal_d(problem: Problem) : 
	energy_threshold = 0.75
	angle_threshold_deg = 10
	n_boot = 50 
	tol = 0.05

	eigenpairs, C, G = calculate_eigenpairs(problem) 
	n, _ = G.shape
	eigenvalues = np.array([a[0] for a in eigenpairs])

	k_mp, lambda_plus, sigma2_hat = marchenko_pastur_cut(eigenvalues, n_samples=n, tol=0.05, verbose=False)
	energy = explained_energy(eigenvalues, k_mp)
	
	k_energy = k_mp
	if energy < energy_threshold:
		# Increase k to meet energy threshold
		cumulative = np.cumsum(eigenvalues)
		total = cumulative[-1]
		k_energy = np.searchsorted(cumulative/total, energy_threshold) + 1

	# Step 3: Bootstrap stability check
	mean_angles, std_angles = bootstrap_subspace_stability(G, k_energy, n_boot=n_boot)
	mean_angle_deg = np.degrees(mean_angles)
	# Reduce k if the last component is unstable
	k_opt = k_energy
	while k_opt > 0 and np.any(mean_angle_deg[:k_opt] > angle_threshold_deg):
		k_opt -= 1

	return k_opt
	
	

def main() : 
	problem_name = 'DYNAMNEWS-1'
	solver_name = 'ASTROMoRF'
	solver_factors = {'crn_across_solns': False}
	dim = 20
	budget = 1000
	res = find_optimal_d(problem_name, solver_name, solver_factors, dim, budget)
	print(f'the optimal subsapce is: {res}')

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
	elif problem_name == 'ROSENBROCK-1' or problem_name=='ZAKHAROV-1' : 
		new_factors = {
			'x': (2.0,)* new_dim,
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
	elif problem_name == 'ROSENBROCK-1' or problem_name=='ZAKHAROV-1' : 
		new_factors = {
			'initial_solution': (2.0,) * new_dim,
			'budget': budget
		}
	return new_factors


if __name__ == '__main__' : 
	main()