import sys
import os.path as o
import random
from multiprocessing import Process

from math import ceil, log

import numpy as np
from mrg32k3a.mrg32k3a import MRG32k3a

sys.path.append(
	o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), ".."))
)  # type:ignore

from simopt.base import (
	Problem,
	Solution,
	Solver
)

from simopt.linear_algebra_base import finite_difference_gradient

def calculate_eigenpairs(problem: Problem) -> tuple[list[tuple], np.ndarray,  np.ndarray]: 
	"""
		Calculate eigenpairs of uncentered covariance matrix of gradients
	Args:
		problem (Problem): The simulation optimization problem

	Returns:
		tuple[list[tuple], np.ndarray,  np.ndarray]:
			list of (eigenvalue, eigenvector) tuples sorted by descending eigenvalue
			uncentered covariance matrix C
			matrix of gradients G (each row is gradient at a sample point)
	"""

	# Increase sample size for more robust gradient covariance estimation
	# For higher-D problems, need more samples to reliably identify active subspace
	# Rule: at least 20*d*log(d) samples (double the original)
	no_solns = ceil(20 * problem.dim * log(problem.dim))
	
	find_next_soln_rng = MRG32k3a()


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
	eigenvalues, eigenvectors = np.linalg.eigh(C)
	
	eigenpairs = [(eigenvalues[i], eigenvectors[:, i]) for i in range(len(eigenvalues))]
	
	eigenpairs.sort(key=lambda pair: pair[0], reverse=True)

	return eigenpairs, C, np.stack(gradients)

def find_optimal_d(problem: Problem) -> int: 
	"""
	Find optimal subspace dimension based on explained variance of gradient covariance matrix
	and variance captured in projected design points.
	Selects dimension k < n that maximizes explained variance with diminishing returns tolerance,
	weighted by how well the subspace captures variance in the design space.
	
	Args:
		problem (Problem): The simulation optimization problem
		
	Returns:
		int: Optimal subspace dimension (always < problem dimension)
	"""
	# Tolerance for diminishing returns: stop if next eigenvalue adds < 2% additional variance
	variance_gain_threshold = 0.02
	
	# Calculate eigenvalues from uncentered covariance matrix of gradients
	eigenpairs, _, G = calculate_eigenpairs(problem)
	n_samples, dim = G.shape
	eigenvalues = np.array([pair[0] for pair in eigenpairs], dtype=float)
	eigenvectors = np.column_stack([pair[1] for pair in eigenpairs])
	
	# Generate design points for variance captured calculation
	find_next_soln_rng = MRG32k3a()
	no_design_points = min(100, ceil(10 * problem.dim * log(problem.dim)))
	X = np.zeros((no_design_points, dim))
	for i in range(no_design_points):
		random_soln_tuple = problem.get_random_solution(find_next_soln_rng)
		random_soln = Solution(random_soln_tuple, problem)
		X[i, :] = random_soln.x
	
	# Compute total variance and explained variance for each dimension
	total_variance = float(np.sum(eigenvalues))
	if total_variance <= 0.0:
		return max(1, dim - 1)
	
	# Cumulative explained variance fraction for each dimension
	cumulative_variance = np.cumsum(eigenvalues) / total_variance
	
	# Calculate variance captured for each candidate dimension (similar to astromorf.py lines 944-946)
	full_space_variance = np.sum(np.var(X, axis=0))
	variance_captured_scores = np.zeros(dim)
	
	for k in range(1, dim):
		# Get first k eigenvectors as subspace basis
		U_k = eigenvectors[:, :k]
		
		# Project design points to k-dimensional subspace
		X_proj = X @ U_k
		
		# Compute variance captured by this subspace
		subspace_variance = np.sum(np.var(X_proj, axis=0))
		variance_captured_scores[k] = subspace_variance / full_space_variance if full_space_variance > 1e-12 else 0.0
	
	# Find optimal dimension using both eigenvalue-based variance and design space variance captured
	k_opt = 1
	best_score = 0.0
	# Require at least 60% variance captured for robustness (increased from 50%)
	# Higher threshold ensures subspace is representative of the problem structure
	min_variance_captured = 0.60
	
	# Allow higher subspace dimensions to capture variance when needed
	# BUT: Must balance with interpolation conditioning
	# 
	# With degree=2 and 2d+1 points, need ratio >= 0.60 for good predictions:
	# - d=1: 3 pts / 3 terms = 1.00 ✓
	# - d=2: 5 pts / 6 terms = 0.83 ✓
	# - d=3: 7 pts / 10 terms = 0.70 ✓
	# - d=4: 9 pts / 15 terms = 0.60 ✓ (threshold)
	# - d=5: 11 pts / 21 terms = 0.52 ✗ (POOR - causes high prediction error)
	#
	# Therefore: Cap max_d at 4 for all problems to ensure good model quality
	if dim >= 50:
		max_d = 4  # High-D: cap at 4
	elif dim >= 20:
		max_d = 4  # High-D: cap at 4
	elif dim >= 10:
		max_d = 4  # Medium-D: cap at 4
	else:
		max_d = min(dim - 1, 4)  # Low-D: up to 4 or dim-1
	
	# Track scores for all dimensions to apply parsimony principle
	dimension_scores = []
	
	for k in range(1, min(dim, max_d + 1)):  # Only consider k < n and k <= max_d
		current_explained = cumulative_variance[k - 1]
		next_explained = cumulative_variance[k] if k < dim else 1.0
		
		# Marginal gain from adding dimension k+1
		marginal_gain = next_explained - current_explained
		
		# Key insight: variance_captured is MORE important than eigenvalue-explained variance
		# A lower-d subspace that captures design variance well will optimize better than
		# a higher-d subspace with marginally better eigenvalue coverage
		var_captured = variance_captured_scores[k]
		
		if var_captured < min_variance_captured:
			# Apply strong penalty for low variance captured
			penalty = (var_captured / min_variance_captured) ** 2
			combined_score = current_explained * penalty
		else:
			# Primary score: variance_captured (weight 70%)
			# Secondary score: eigenvalue explained variance (weight 30%)
			# This prioritizes how well the subspace represents the actual design space
			combined_score = 0.7 * var_captured + 0.3 * current_explained
		
		# Parsimony penalty: prefer lower dimensions
		# With max_d=3, this mainly helps choose between d=1,2,3
		if k <= 2:
			parsimony_penalty = 1.0  # No penalty for d=1,2
		else:
			# Mild penalty for d=3, strong penalty if somehow d>3
			parsimony_penalty = 0.95 ** (k - 2)
		combined_score *= parsimony_penalty
		
		# Diminishing returns penalty: heavily penalize dimensions with low marginal gain
		# If adding this dimension gives < 2% improvement, apply additional penalty
		if k > 1 and marginal_gain < variance_gain_threshold:
			diminishing_returns_penalty = 0.85  # 15% penalty
			combined_score *= diminishing_returns_penalty
		
		dimension_scores.append((k, combined_score, var_captured, current_explained))
		
		if combined_score > best_score:
			best_score = combined_score
			k_opt = k
		
		# Early stopping: if marginal gain is tiny AND variance captured is dropping
		if k > 1 and marginal_gain < 0.01:
			prev_var_captured = variance_captured_scores[k-1]
			if var_captured < prev_var_captured:
				# Adding this dimension hurts variance captured - stop here
				break
	
	# Ensure we never return the full dimension
	k_opt = min(k_opt, dim - 1)
	k_opt = max(1, k_opt)
	
	# Print diagnostic info with detailed scoring
	print(f"\n=== Subspace Dimension Selection ===")
	print(f"Problem dimension: {dim}")
	print(f"Maximum subspace dimension considered: {max_d} (capped for computational efficiency)")
	print(f"\nDimension Analysis:")
	print(f"{'d':<3} {'Eigenval%':<10} {'Marginal%':<10} {'VarCap%':<10} {'Score':<8} {'Status':<10}")
	print("-" * 65)
	
	# Show all dimensions that were actually considered
	display_limit = min(len(dimension_scores), 10)
	for idx in range(display_limit):
		k, score, var_cap, current_explained = dimension_scores[idx]
		var_pct = current_explained * 100
		if idx < len(dimension_scores) - 1:
			marginal = (dimension_scores[idx + 1][3] - current_explained) * 100
		else:
			marginal = 0.0
		var_cap_pct = var_cap * 100
		
		status = "✓ SELECTED" if k == k_opt else ""
		if k > 1 and marginal < 2.0:
			status = status if status else "low marginal"
		if var_cap < 0.5:
			status = status if status else "low var_cap"
		
		print(f"{k:<3} {var_pct:>9.1f} {marginal:>9.1f} {var_cap_pct:>9.1f} {score:>7.3f}  {status:<10}")
	
	print("\n" + "="*65)
	print(f"SELECTED: d={k_opt}")
	print(f"  Eigenvalue variance explained: {cumulative_variance[k_opt-1]*100:.1f}%")
	print(f"  Design variance captured:      {variance_captured_scores[k_opt]*100:.1f}%")
	print(f"  Combined score:                {best_score:.3f}")
	print("="*65 + "\n")
	
	return k_opt


def compute_optimal_polynomial_degree(subspace_dim: int, num_points: int = None, min_ratio: float = 0.60) -> int:
	"""
	Compute the optimal polynomial degree for a given subspace dimension to ensure
	well-conditioned interpolation systems.
	
	For polynomial interpolation with degree p in dimension d:
	- Number of interpolation points: 2*d + 1 (standard trust-region sampling)
	- Number of polynomial terms: C(d+p, d) = (d+p)! / (d! * p!)
	- Conditioning ratio: points / terms
	
	Good interpolation requires ratio >= min_ratio (default 0.60)
	
	Args:
		subspace_dim (int): The active subspace dimension (d)
		num_points (int, optional): Number of interpolation points. If None, uses 2*d+1
		min_ratio (float, optional): Minimum acceptable points/terms ratio (default 0.60)
	
	Returns:
		int: Optimal polynomial degree (1 or 2)
		
	Examples:
		>>> compute_optimal_polynomial_degree(1)  # 3 pts / 3 terms = 1.00
		2
		>>> compute_optimal_polynomial_degree(4)  # 9 pts / 15 terms = 0.60
		2
		>>> compute_optimal_polynomial_degree(5)  # 11 pts / 21 terms = 0.52 < 0.60
		1
	"""
	from math import comb
	
	d = subspace_dim
	
	# Default number of points for trust-region interpolation
	if num_points is None:
		num_points = 2 * d + 1
	
	# Try degree 2 first (preferred for capturing curvature)
	degree_2_terms = comb(d + 2, d)
	ratio_degree_2 = num_points / degree_2_terms
	
	if ratio_degree_2 >= min_ratio:
		return 2  # Quadratic model - good conditioning
	
	# Fall back to degree 1 (linear)
	degree_1_terms = d + 1
	ratio_degree_1 = num_points / degree_1_terms
	
	if ratio_degree_1 >= min_ratio:
		return 1  # Linear model - ensures good conditioning
	
	# If even linear doesn't work, still return 1 (safest option)
	# This case should be rare if subspace dimension is chosen properly
	return 1
