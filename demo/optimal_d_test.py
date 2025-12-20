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

from simopt.linear_algebra_base import finite_difference_gradient

def calculate_eigenpairs(problem_name: str, solver_name: str, solver_factors: dict, dim: int, budget: int) -> list[tuple] : 
	"""
		Constructs a proxy initial guess of the subspace dimension through the following steps:
			1. Obtain alog(n) random solutions around the problem [x_1,x_2,...,x_{alog(n)}]
			2. Estimate the gradient vectors of the problem at each random solution 
			3. Construct the uncentred covariance matrix of functional derivatives C through a Monte Carlo estimate on the alog(n) gradient vectors
			4. Factorise C through eigendecomposition and sort the eigenpairs into a list
			5. Create a second list of tuples (i,j) where i is the index of the eigenpairs in the list from step 2, and j is the distance between the eigen pairs of 
			|lambda_{i-1}-lambda_{i}|. 
			6. pick d to be the i in the list of tuples that contains the smallest j in their tuple out of all j's in the list of tuples

	Args:
		problem_name (str): Name of the problem
		solver_name (str): Name of the solver 
		solver_factors (dict): the fixed factors for the solver 
		dim (int): the dimension of the problem
		budget (int): the simulation budget of the SO problem.

	Returns: 
		int: The subspace dimension

	NOTE: 
		This method of finding a parameter for d will require an additional alog(n^{2n}) responses of the simulation model.
		#! Before implementing this proxy, we need to find a way of finding this covariance matrix without taking an exponential number of responses.
		- Look at some form of sensitivity analysis to find the covariance matrix


	"""
	#Create Problem and solution
	problem_fixed_factors = update_problem_factor_dimensions(problem_name, dim, budget)
	model_fixed_factors = update_model_factors_dimensions(problem_name, dim)
	problem: Problem = instantiate_problem(problem_name, problem_fixed_factors, model_fixed_factors)

	starting_solution = instantiate_solver(solver_name, solver_factors)

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
	
	return eigenpairs, C, np.stack([a.flatten() for a in gradients])


def choose_by_energy(eigenpairs, threshold=0.95):
	"""Smallest k such that cumulative energy >= threshold. k returned as int."""
	eigs = [a[0] for a in eigenpairs]
	total = np.sum(eigs)
	if total <= 0:
		raise ValueError("Sum of eigenvalues is nonpositive.")
	cum = np.cumsum(eigs) / total
	k = int(np.searchsorted(cum, threshold) + 1)  # +1 -> number of eigenvectors
	return k, cum 

def choose_by_eigengap(eigenpairs, min_k=1, max_k=None):
	"""Choose k at index of max eigengap (k between 1..m-1). Returns k and gaps."""
	eigs = [a[0] for a in eigenpairs]
	gaps = [a - b for a,b in zip(eigs[:-1],eigs[1:])]
	if max_k is None:
		max_k = len(gaps)
	# restrict search range (1-based indexing for k)
	idx = np.argmax(gaps[min_k-1:max_k-1]) + (min_k-1)
	k = idx + 1
	return k, gaps

def residual_ratio(G, W, k) : 
	Wk = W[:, :k]
	Proj = Wk @ Wk.T
	R = G - G @ Proj
	return np.linalg.norm(R, 'fro')**2 / (np.linalg.norm(G, 'fro')**2 + 1e-16)

# def find_optimal_d(problem_name: str, solver_name: str, solver_factors: dict, dim: int, budget: int) : 
	# eigenpairs, W, G = calculate_eigenpairs(problem_name, solver_name, solver_factors, dim, budget)

	# optimal_d_and_energies = []
	# # energy rule
	# k_energy, cum = choose_by_energy(eigenpairs)
	# optimal_d_and_energies.append((int(k_energy), cum))

	# k_gap, gaps = choose_by_eigengap(eigenpairs)
	# print(f'the largest gap is {k_gap}')
	# print(f'The total gaps at each index are:\n {[float(a) for a in gaps]}')
	# [print(f'{a}: {b}') for a,b in enumerate(gaps)]
	# optimal_d_and_energies.append((int(k_gap), gaps))

	# energy_k_eig = cum[k_gap-1]
	# energy_k_energy = cum[k_energy-1]

	# min_energy = 0.8
	# max_resid_ratio = 0.05 
	# residual_ratio_energy = residual_ratio(G,W, k_energy)
	# residual_ratio_gap = residual_ratio(G,W, k_gap)
	# print(f'The residual ratio energy is {residual_ratio_energy}')
	# print(f'The residual ratio gap is {residual_ratio_gap}')
	

	# optimal_d = None 

	# if energy_k_eig >= min_energy : 
	# 	optimal_d = k_gap
	# else : 
	# 	optimal_d = k_energy
	
	# # if residual_ratio_gap <= residual_ratio_energy + max_resid_ratio : 
	# # 	optimal_d =  k_gap
	
	# # else : 
	# # 	optimal_d =  k_energy

	# return optimal_d 


def largest_eigengap(eigs):
    # eigs descending
    gaps = eigs[:-1] - eigs[1:]
    k = int(np.argmax(gaps) + 1)   # +1 => number of top eigenvalues
    return k, gaps

def log_second_derivative(eigs):
    # operate on log of positive eigenvalues (clip to avoid log(0))
    eps = 1e-16
    loge = np.log(np.maximum(eigs, eps))
    # discrete second derivative: d2[i] = loge[i+1] - 2*loge[i] + loge[i-1]
    d2 = loge[2:] - 2*loge[1:-1] + loge[:-2]
    # pick index i where curvature is most negative (largest concavity)
    if d2.size == 0:
        return 1, d2
    i = int(np.argmin(d2)) + 1  # +1 to map back to original index
    k = i + 1  # number of top eigenvalues (k = index+1 because indices start at 0)
    return k, d2

def max_curvature_knee(eigs):
    # Kneedle-like: compute distance from straight line connecting endpoints on normalized log-eig curve
    eps = 1e-16
    loge = np.log(np.maximum(eigs, eps))
    # normalize both axes
    x = np.arange(len(loge)).astype(float)
    xn = (x - x.min()) / (x.max() - x.min() + eps)
    yn = (loge - loge.min()) / (loge.max() - loge.min() + eps)
    # line from (0,yn[0]) to (1,yn[-1]) is just y = yn[0] + (yn[-1]-yn[0])*xn ; compute orthogonal distance
    # For simplicity compute vertical distance to the secant line:
    sec = yn[0] + (yn[-1] - yn[0]) * xn
    dist = sec - yn  # positive when point is below the secant (we expect a downward kink)
    i = int(np.argmax(dist))
    k = i + 1
    return k, dist

def otsu_1d_log(eigs):
    # Otsu thresholding on log-eigs: maximize between-class variance
    eps = 1e-16
    vals = np.log(np.maximum(eigs, eps))
    # sort ascending for standard Otsu, but we want split between high and low, so just use descending indices
    # We compute thresholds at boundaries between sorted descending values
    N = len(vals)
    vals_desc = vals[::-1]  # ascending -> we will iterate thresholds across ascending; easier to follow Otsu formula if ascending
    # But simpler: test splits at j = 1..N-1 and compute between-class variance
    best_j = 1
    best_score = -np.inf
    arr = vals  # original order is descending eigs -> arr is descending loge
    for j in range(1, N):
        c1 = arr[:j]
        c2 = arr[j:]
        w1 = c1.size / N
        w2 = c2.size / N
        mu1 = c1.mean() if c1.size>0 else 0.0
        mu2 = c2.mean() if c2.size>0 else 0.0
        score = w1 * w2 * (mu1 - mu2)**2
        if score > best_score:
            best_score = score
            best_j = j
    return best_j, best_score

def kmeans_1d_two_clusters_log(eigs, maxiter=100):
    # simple 2-means on log(eigs) (descending order)
    eps = 1e-16
    x = np.log(np.maximum(eigs, eps)).reshape(-1,1)
    # initialize centroids: two ends
    c1, c2 = x[0,0], x[-1,0]
    for _ in range(maxiter):
        d1 = (x[:,0] - c1)**2
        d2 = (x[:,0] - c2)**2
        assign = d1 < d2
        if assign.sum() == 0 or (~assign).sum() == 0:
            break
        newc1 = x[assign,0].mean()
        newc2 = x[~assign,0].mean()
        if np.isclose(newc1, c1) and np.isclose(newc2, c2):
            break
        c1, c2 = newc1, newc2
    # top-cluster is the cluster with larger centroid (since log-eig larger -> signal)
    top_cluster = c1 if c1 > c2 else c2
    # find smallest index that belongs to top cluster when scanning sorted descending
    # recompute final assignment and pick cutoff
    d1 = (x[:,0] - c1)**2
    d2 = (x[:,0] - c2)**2
    assign = d1 < d2
    # if top centroid is c1, then assign==True are top cluster
    if top_cluster == c1:
        top_assign = assign
    else:
        top_assign = ~assign
    # we want contiguous top group at the beginning; find last True in top_assign among prefix
    # handle possible non-contiguous assignments by taking largest prefix where majority are top
    # find first index where top_assign becomes False (i.e., cutoff)
    if top_assign.all():
        k = len(eigs)
    else:
        # find first false in top_assign (scanning from index 0)
        falses = np.where(~top_assign)[0]
        if falses.size == 0:
            k = len(eigs)
        else:
            k = int(falses[0])  # top cluster occupies indices [0..falses[0]-1]
    return k, top_assign

def consensus_cut(eigs, prefer_smaller=True):
    """
    Run detectors and return per-detector ks and a consensus k.
    prefer_smaller: if even split, choose smaller k (more parsimonious).
    """
    eigs = np.asarray(eigs).copy()
    # ensure descending
    eigs = eigs[np.argsort(eigs)[::-1]]
    results = {}
    ks = []
    k1, gaps = largest_eigengap(eigs); results['eigengap_k']=k1; ks.append(k1)
    k2, d2 = log_second_derivative(eigs); results['log2_k']=k2; ks.append(k2)
    k3, dist = max_curvature_knee(eigs); results['knee_k']=k3; ks.append(k3)
    k4, score = otsu_1d_log(eigs); results['otsu_k']=k4; ks.append(k4)
    k5, assign = kmeans_1d_two_clusters_log(eigs); results['kmeans_k']=k5; ks.append(k5)
    # aggregate: take median of proposed ks then round, or take majority if majority exists
    ks_arr = np.array(ks)
    # majority vote
    unique, counts = np.unique(ks_arr, return_counts=True)
    majority_idx = np.argmax(counts)
    if counts[majority_idx] > len(ks)//2:
        k_consensus = int(unique[majority_idx])
    else:
        # take median rounded
        k_consensus = int(np.median(ks_arr))
        if prefer_smaller:
            k_consensus = max(1, k_consensus)  # ensure >=1
    results['all_ks'] = ks
    results['consensus_k'] = k_consensus
    # diagnostics: largest gap relative to max eigenvalue
    gaps_full = eigs[:-1] - eigs[1:]
    max_gap = gaps_full.max() if gaps_full.size>0 else 0.0
    results['largest_gap'] = float(max_gap)
    results['largest_gap_rel'] = float(max_gap / (eigs[0] + 1e-24))
    results['eigs'] = eigs
    return results
	
def find_optimal_d(problem_name: str, solver_name: str, solver_factors: dict, dim: int, budget: int) : 
	rel_gap = 0.0
	consensus_val = None
	while rel_gap < 0.3 :
		print(f'the relative gap is {rel_gap}')
		eigenpairs, W, G = calculate_eigenpairs(problem_name, solver_name, solver_factors, dim, budget)
		eigenvalues = [a[0] for a in eigenpairs]
		res =  consensus_cut(eigenvalues)
		rel_gap = res['largest_gap_rel']
		consensus_val = res['consensus_k']

	return consensus_val


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

def main() : 
	problem_name = 'NETWORK-1'
	solver_name = 'ASTROMoRF'
	solver_factors = {'crn_across_solns': False}
	dim = 20
	budget = 1000
	res = find_optimal_d(problem_name, solver_name, solver_factors, dim, budget)
	print(f'the optimal subsapce is: {res}')

if __name__ == '__main__' : 
	main()