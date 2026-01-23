from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import pdist

from simopt.base import (
	Problem,
	Solution,
	Solver,
)


class ASTROMoRF_Diagnostics : 

	def __init__(self, solver: Solver, problem: Problem):
		self.solver = solver
		self.problem = problem
		self.prediction_error_history = []
		self.prev_r_squared = None  # For tracking R² trend over iterations
		self.initialize_diagnostics_file()



	def poisedness_score(self, X, normalize=True):
		"""Compute the poisedness of a projected design set for interpolation.

		Parameters
		----------
		X : ndarray of shape (M, d)
			Original design set.
		
		normalize : bool, default=True
			If True, normalize the score by the diameter of the projected domain.

		Returns:
		-------
		score : float
			A scalar measure of poisedness in [0, 1]. Higher is better.
		min_dist : float
			Minimum pairwise distance in the projected space.
		"""
		# Handle small sample sizes
		if X is None:
			return 0.0, 0.0
		if getattr(X, "shape", None) is None or X.shape[0] < 2:
			return 0.0, 0.0

		# All pairwise distances
		dists = pdist(X)

		# Minimum distance = key poisedness indicator
		min_dist = np.min(dists) if dists.size > 0 else 0.0

		if not normalize:
			return min_dist, min_dist

		# Normalize by “diameter’’ (max pairwise distance)
		max_dist = np.max(dists)
		if max_dist == 0:
			score = 0.0
		else:
			score = min_dist / max_dist

		return score, min_dist

	def initialize_diagnostics_file(self):
		# Create diagnostics file path if it doesn't exist
		# Use timestamp + process ID + random component for uniqueness across parallel processes
		self._timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
		self._unique_id = f"{self._timestamp}_pid{os.getpid()}_r{random.randint(1000, 9999)}"
		
		# Create Diagnostics directory if it doesn't exist
		diagnostics_dir = Path(__file__).parent.parent / "Diagnostics"
		diagnostics_dir.mkdir(parents=True, exist_ok=True)
		
		# Create subdirectory for this solver-problem pair
		solver_problem_dir = diagnostics_dir / f"{self.solver.name}_on_{self.problem.name}"
		solver_problem_dir.mkdir(parents=True, exist_ok=True)
		
		self.diagnostics_file_path = str(solver_problem_dir / f"astromorf_diagnostics_{self.problem.name}_{self._unique_id}.txt")
		
		# Check if file is empty and write header if needed
		file_is_empty = not os.path.exists(self.diagnostics_file_path) or os.path.getsize(self.diagnostics_file_path) == 0
		
		if file_is_empty and hasattr(self, '_timestamp') :
			with open(self.diagnostics_file_path, 'w', encoding='utf-8') as f:
				header = f'==== Starting ASTROMoRF Solver on problem: {self.problem.name} ====\n'
				header += f'Timestamp: {self._timestamp}\n'
				header += f'Initial budget: {self.solver.budget.remaining}\n\n'
				f.write(header)

	def write_diagnostics_to_txt(self, content: str, mode: str = 'a'):
		"""Write diagnostic content to a text file.
			Automatically creates file path and writes header if file is empty.
			
		Args:
			content (str): The content to write
			mode (str): File mode - 'w' for write (overwrite), 'a' for append
		"""
		with open(self.diagnostics_file_path, mode, encoding='utf-8') as f:
			f.write(content)


	def check_polynomial_complexity(self) -> None:
		"""Check polynomial basis complexity and record warnings via diagnostics.
		
		Computes the number of polynomial terms q = C(d+p, p) where d is the 
		subspace dimension and p is the polynomial degree. Records warnings
		to diagnostics if this exceeds thresholds that may cause slow performance.
		"""
		from math import comb
		
		d = self.solver.d
		p = self.solver.degree
		
		# Number of terms in total degree polynomial basis
		q = comb(d + p, p)
		
		# Number of interpolation points (2d + 1)
		num_points = 2 * d + 1
		
		# Store complexity info on solver for later use
		self.solver._polynomial_terms = q
		self.solver._conditioning_ratio = num_points / q if q > 0 else float('inf')
		
		# Check various performance thresholds and record to diagnostics
		if q > 500:
			warning_msg = (
				f"PERFORMANCE WARNING: Polynomial basis has {q} terms "
				f"(subspace_dim={d}, degree={p}). This may cause very slow "
				f"Vandermonde matrix operations. Consider reducing polynomial "
				f"degree or subspace dimension.\n"
			)
			self.write_diagnostics_to_txt(warning_msg)
		elif q > 200:
			warning_msg = (
				f"PERFORMANCE WARNING: Polynomial basis has {q} terms "
				f"(subspace_dim={d}, degree={p}). This may cause slow model "
				f"construction. Consider reducing polynomial degree to improve speed.\n"
			)
			self.write_diagnostics_to_txt(warning_msg)
		
		# Check conditioning: ratio of points to terms
		if self.solver._conditioning_ratio < 0.3:
			warning_msg = (
				f"CONDITIONING WARNING: Only {num_points} interpolation points "
				f"for {q} polynomial terms (ratio={self.solver._conditioning_ratio:.2f}). "
				f"This may lead to ill-conditioned systems. Consider reducing "
				f"polynomial degree or increasing subspace dimension.\n"
			)
			self.write_diagnostics_to_txt(warning_msg)


	def write_final_diagnostics(self):
		"""Write final prediction error analysis and solver summary to diagnostics file.
		"""
		# Write prediction error trend analysis to file
		if len(self.prediction_error_history) > 0:
			output = "\n" + "="*70 + "\n"
			output += "PREDICTION ERROR ANALYSIS\n"
			output += "="*70 + "\n"
			errors = [e['relative_error'] for e in self.prediction_error_history]
			output += f"Total iterations with predictions: {len(errors)}\n"
			output += f"Mean prediction error: {np.mean(errors)*100:.1f}%\n"
			output += f"Median prediction error: {np.median(errors)*100:.1f}%\n"
			output += f"Min prediction error: {np.min(errors)*100:.1f}%\n"
			output += f"Max prediction error: {np.max(errors)*100:.1f}%\n"
			
			# Show trend over time
			if len(errors) >= 5:
				early_errors = np.mean(errors[:len(errors)//3])
				late_errors = np.mean(errors[-len(errors)//3:])
				output += f"\nTrend: Early errors: {early_errors*100:.1f}% \u2192 Late errors: {late_errors*100:.1f}%\n"
				if late_errors > 1.5 * early_errors:
					output += "  \u26a0\ufe0f  WARNING: Prediction quality degraded significantly over iterations\n"
					output += "  Possible causes:\n"
					output += "  - Trust region too large for polynomial model\n"
					output += "  - Subspace dimension too small\n"
					output += "  - Problem has complex local structure\n"
					output += "  - Polynomial degree insufficient\n"
			output += "="*70 + "\n\n"
			self.write_diagnostics_to_txt(output)
		
		# Write final summary to file
		output = "\n" + "="*70 + "\n"
		output += '---- ASTROMoRF Solver Finished ----\n'
		output += f"ASTROMoRF completed after {self.solver.iteration_count} iterations and {self.solver.budget.used} total function evaluations.\n"
		
		total_iterations = len(self.solver.successful_iterations) + len(self.solver.unsuccessful_iterations)
		if total_iterations > 0:
			output += f'The number of successful iterations was {len(self.solver.successful_iterations)} and the number of unsuccessful iterations was {len(self.solver.unsuccessful_iterations)}.\n'
			output += f'The percentage of successful iterations was {len(self.solver.successful_iterations)/total_iterations*100:.2f}%.\n'
		else:
			output += 'No iterations were completed (solver may have failed early).\n'
		
		if hasattr(self.solver, 'incumbent_solution') and self.solver.incumbent_solution is not None:
			output += f'The best solution found has an objective value of {self.solver.incumbent_solution.objectives_mean.item():.6f} with a sample size of {self.solver.incumbent_solution.n_reps}.\n'
		else:
			output += 'No solution was found.\n'
		
		output += "="*70 + "\n\n"
		self.write_diagnostics_to_txt(output)
		
		# Print simple completion message to console
		print(f"\n\u2713 ASTROMoRF Solver completed. Diagnostics written to: {self.diagnostics_file_path}")


	def diagnose_candidate_solution(self, candidate_solution: Solution, arg2, arg3=None) :
		"""Diagnose candidate solution quality by comparing to design points and model predictions.
			
		Args:
			candidate_solution (Solution): The candidate solution to diagnose
			This method supports two call signatures for backward compatibility:
			1) (candidate_solution, model, X)
			2) (candidate_solution, X)  # model will be taken from self.solver.model if available

			arg2: Either the model callable or the design matrix X
			arg3: The design matrix X if arg2 is the model, otherwise None
			
		"""
		# Resolve arguments: detect if arg2 is a callable model or the design matrix X
		if callable(arg2):
			model = arg2
			X = arg3
		else:
			# arg2 assumed to be X (design matrix)
			X = arg2
			model = arg3 if callable(arg3) else getattr(self.solver, 'model', None)

		x_candidate = np.array(candidate_solution.x).reshape(-1, 1)
		x_current = np.array(self.solver.incumbent_x).reshape(-1, 1)
		
		# Distance metrics (computed here for diagnostics, also computed in update_parameters)
		distances_to_design = [norm(x_candidate.flatten() - X[i,:]) for i in range(X.shape[0])]
		min_dist_to_design = min(distances_to_design)
		mean_dist_to_design = np.mean(distances_to_design)
		
		# Full-space step size
		full_space_step = norm(x_candidate - x_current)
		
		# Model prediction vs actual
		x_candidate_arr = np.array(candidate_solution.x).reshape(1, -1)
		if model is None:
			model_prediction = float('nan')
		else:
			try:
				raw_pred = model(x_candidate_arr)
				model_prediction = float(np.squeeze(raw_pred))
			except Exception:
				try:
					model_prediction = float(raw_pred)
				except Exception:
					model_prediction = float('nan')
		actual_value = -1 * self.problem.minmax[0] * candidate_solution.objectives_mean.item()
		prediction_error = abs(model_prediction - actual_value)
		relative_error = prediction_error / (abs(actual_value) + 1e-10)
		
		# Current objective value
		current_value = -1 * self.problem.minmax[0] * self.solver.incumbent_solution.objectives_mean.item()
		actual_improvement = current_value - actual_value
		predicted_improvement = current_value - model_prediction
		
		diagnostics = {
			'min_dist_to_design': min_dist_to_design,
			'mean_dist_to_design': mean_dist_to_design,
			'full_space_step': full_space_step,
			'trust_region_radius': self.solver.delta,
			'step_to_radius_ratio': full_space_step / self.solver.delta if self.solver.delta > 0 else 0,
			'model_prediction': model_prediction,
			'actual_value': actual_value,
			'prediction_error': prediction_error,
			'relative_error': relative_error,
			'current_value': current_value,
			'actual_improvement': actual_improvement,
			'predicted_improvement': predicted_improvement
		}
		
		self._print_candidate_diagnostics(diagnostics)

	def _print_candidate_diagnostics(self, diagnostics: dict):
		"""Write candidate solution diagnostics to file.

		Args:
			diagnostics (dict): Diagnostic statistics
			problem (Problem): The simulation problem being solved
		"""
		output = "\n" + "="*70 + "\n"
		output += "CANDIDATE SOLUTION DIAGNOSTICS\n"
		output += "="*70 + "\n"
		output += f"Distance to nearest design point:  {diagnostics['min_dist_to_design']:.4f}\n"
		output += f"Mean distance to design points:     {diagnostics['mean_dist_to_design']:.4f}\n"
		output += f"Full-space step size:               {diagnostics['full_space_step']:.4f}\n"
		output += f"Trust region radius:                {diagnostics['trust_region_radius']:.4f}\n"
		output += f"Step to radius ratio:               {diagnostics['step_to_radius_ratio']:.1%}\n"
		output += "\nPrediction Quality:\n"
		output += f"  Model prediction:                 {diagnostics['model_prediction']:.6f}\n"
		output += f"  Actual value:                     {diagnostics['actual_value']:.6f}\n"
		output += f"  Prediction error:                 {diagnostics['prediction_error']:.6f} ({diagnostics['relative_error']*100:.1f}%)\n"
		output += "\nImprovement:\n"
		output += f"  Current objective:                {diagnostics['current_value']:.6f}\n"
		output += f"  Predicted improvement:            {diagnostics['predicted_improvement']:.6f}\n"
		output += f"  Actual improvement:               {diagnostics['actual_improvement']:.6f}\n"
		
		# Analysis of why prediction might be poor
		if diagnostics['relative_error'] > 0.15:
			output += "\n⚠️  WARNING: High prediction error detected!\n"
			if diagnostics['min_dist_to_design'] > 0.5 * diagnostics['trust_region_radius']:
				output += "  → Candidate is far from design points (interpolation issue)\n"
			if diagnostics['step_to_radius_ratio'] > 0.95:
				output += "  → Step at trust region boundary (extrapolation issue)\n"
			if abs(diagnostics['predicted_improvement']) < 1e-4:
				output += "  → Model predicts very small change (model may be too flat)\n"
		
		output += "="*70 + "\n\n"
		self.write_diagnostics_to_txt(output)


	def diagnose_model_quality(self, model: callable, model_grad: callable, X: np.ndarray, fX: np.ndarray, U: np.ndarray) :
		"""Compute comprehensive model quality diagnostics to understand poor success rates.
		
		Args:
			model (callable): The polynomial model
			model_grad (callable): The gradient of the polynomial model
			X (np.ndarray): Design points (M, n)
			fX (np.ndarray): True function values at design points (M, 1)
			U (np.ndarray): Active subspace (n, d)
		
		"""
		M = X.shape[0]
		n = X.shape[1]

		# Generate a Validation set of points within the trust region
		no_pts = 2 * self.problem.dim + 1
		# `generate_set` expects (num, delta=None), so pass the count `no_pts`
		X_test = self.solver.generate_set(no_pts)

		#create solutions for each point in the validation set, simulate each point and get true function values
		fX_test = np.zeros((no_pts,))
		for i in range(no_pts):
			x = X_test[i, :].flatten()
			sol = self.solver.create_new_solution(tuple(x), self.problem)
			self.problem.simulate(sol,20)
			fX_test[i] = -1 * self.problem.minmax[0] * sol.objectives_mean.item()
		
		# 1. Model fit quality on design points
		predictions = np.full((no_pts,), np.nan, dtype=float)
		for i in range(no_pts):
			try:
				pred_raw = model(X_test[i, :].reshape(-1, 1))
				predictions[i] = float(np.squeeze(pred_raw))
			except Exception:
				predictions[i] = np.nan
		residuals = fX_test.flatten() - predictions

		# Compute R² (coefficient of determination) robust to NaNs
		valid = ~np.isnan(predictions) & ~np.isnan(fX_test.flatten())
		if np.sum(valid) >= 2:
			ss_res = np.nansum((fX_test.flatten()[valid] - predictions[valid]) ** 2)
			ss_tot = np.nansum((fX_test.flatten()[valid] - np.mean(fX_test.flatten()[valid])) ** 2)
			r_squared = 1 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
			rmse = np.sqrt(ss_res / np.sum(valid))
			max_error = np.nanmax(np.abs(fX_test.flatten()[valid] - predictions[valid]))
		else:
			r_squared = 0.0
			rmse = float('nan')
			max_error = float('nan')
		
		# 2. Active subspace quality
		# Project design points to subspace
		X_proj = X @ U  # (M, d)
		
		# Compute spread in full space vs subspace
		full_space_variance = np.sum(np.var(X, axis=0))
		subspace_variance = np.sum(np.var(X_proj, axis=0))
		variance_captured = subspace_variance / full_space_variance if full_space_variance > 1e-12 else 0.0
		
		# Check orthonormality of U
		should_be_identity = U.T @ U
		orthonormality_error = np.max(np.abs(should_be_identity - np.eye(self.solver.d)))
		
		# 3. Design set geometry
		# Compute condition number of Vandermonde matrix
		V_matrix = self.solver.V(X_proj)
		cond_V = np.linalg.cond(V_matrix)
		
		# Compute pairwise distances in the projected space (sample 100 pairs if M is large)
		sample_size = min(M, 100)
		indices = np.random.choice(M, size=sample_size, replace=False) if M > 100 else np.arange(M)
		distances = []
		for i in range(len(indices)):
			for j in range(i+1, len(indices)):
				distances.append(norm(X_proj[indices[i], :] - X_proj[indices[j], :]))
		
		min_dist = np.min(distances) if distances else 0.0
		max_dist = np.max(distances) if distances else 0.0
		mean_dist = np.mean(distances) if distances else 0.0


		# 4. Computer poisedness score in the active subspace
		poisedness_score, min_dist_subspace = self.poisedness_score(X_proj, normalize=True)
		
		# 5. Gradient information
		# Compute model gradient at incumbent (first point in X)
		x_current = np.array(self.solver.incumbent_x).reshape(1, -1)
		# model_grad may require a `full_space` kwarg; prefer full-space gradient
		try:
			grad_val = model_grad(x_current, full_space=True)
		except TypeError:
			try:
				grad_val = model_grad(x_current)
			except Exception:
				grad_val = np.zeros((1, self.solver.problem.dim))

		# Ensure a flat vector for norm
		grad_norm = norm(np.asarray(grad_val).flatten())
		
		diagnostics = {
			'r_squared': r_squared,
			'rmse': rmse,
			'max_error': max_error,
			'variance_captured': variance_captured,
			'orthonormality_error': orthonormality_error,
			'vandermonde_condition': cond_V,
			'min_point_distance': min_dist,
			'max_point_distance': max_dist,
			'mean_point_distance': mean_dist,
			'poisedness_score': poisedness_score,
			'gradient_norm': grad_norm,
			'num_design_points': M,
			'dimension': n,
			'subspace_dim': self.solver.d,
			'delta': self.solver.delta
		}
		
		self._print_model_diagnostics(diagnostics)

	def _print_model_diagnostics(self, diagnostics: dict):
		"""Write model quality diagnostics to file.

		Args:
			diagnostics (dict): Model quality statistics
			problem (Problem): The simulation problem being solved
		"""
		output = "\n" + "="*70 + "\n"
		output += "MODEL QUALITY DIAGNOSTICS\n"
		output += "="*70 + "\n"
		
		output += "\n📊 MODEL FIT QUALITY:\n"
		r2_status = '⚠️ POOR' if diagnostics['r_squared'] < 0.7 else '✓ Good' if diagnostics['r_squared'] > 0.9 else '⚠️ Moderate'
		
		# Check R² trend if we have previous value
		if self.prev_r_squared is not None:
			r2_change = diagnostics['r_squared'] - self.prev_r_squared
			trend_arrow = "📈" if r2_change > 0.05 else "📉" if r2_change < -0.05 else "→"
			output += f"  R² (goodness of fit):     {diagnostics['r_squared']:.4f}  {r2_status}  {trend_arrow} (Δ={r2_change:+.3f})\n"
			
			if r2_change < -0.1:
				output += f"    ⚠️  R² dropped significantly! Previous: {self.prev_r_squared:.4f}\n"
		else:
			output += f"  R² (goodness of fit):     {diagnostics['r_squared']:.4f}  {r2_status}\n"
		
		# Store for next iteration
		self.prev_r_squared = diagnostics['r_squared']
		
		output += f"  RMSE:                     {diagnostics['rmse']:.4e}\n"
		output += f"  Max absolute error:       {diagnostics['max_error']:.4e}\n"
		
		output += "\n🎯 ACTIVE SUBSPACE QUALITY:\n"
		output += f"  Variance captured:        {diagnostics['variance_captured']:.4f}  {'⚠️ POOR' if diagnostics['variance_captured'] < 0.5 else '✓ Good' if diagnostics['variance_captured'] > 0.8 else '⚠️ Moderate'}\n"
		output += f"  Orthonormality error:     {diagnostics['orthonormality_error']:.4e}  {'⚠️ ISSUE' if diagnostics['orthonormality_error'] > 1e-6 else '✓ Good'}\n"
		output += f"  Subspace dimension:       {diagnostics['subspace_dim']}/{diagnostics['dimension']}\n"
		
		output += "\n📐 DESIGN SET GEOMETRY:\n"
		output += f"  Condition number:         {diagnostics['vandermonde_condition']:.4e}  {'⚠️ ILL-CONDITIONED' if diagnostics['vandermonde_condition'] > 1e3 else '⚠️ Poor' if diagnostics['vandermonde_condition'] > 1e2 else '✓ Good'}\n"
		output += f"  Poisedness score:         {diagnostics['poisedness_score']:.4f}  {'⚠️ POOR' if diagnostics['poisedness_score'] < 0.2 else '✓ Good' if diagnostics['poisedness_score'] > 0.5 else '⚠️ Moderate'}\n"
		output += f"  Number of points:         {diagnostics['num_design_points']}\n"
		output += f"  Min point distance:       {diagnostics['min_point_distance']:.4e}\n"
		output += f"  Max point distance:       {diagnostics['max_point_distance']:.4e}\n"
		output += f"  Mean point distance:      {diagnostics['mean_point_distance']:.4e}\n"
		
		output += "\n🔍 OPTIMIZATION INFO:\n"
		output += f"  Gradient norm:            {diagnostics['gradient_norm']:.4e}\n"
		output += f"  Trust-region radius:      {diagnostics['delta']:.4e}\n"
		
		# Overall assessment
		output += "\n💡 ASSESSMENT:\n"
		issues = []
		if diagnostics['r_squared'] < 0.7:
			issues.append("  ⚠️  Low R² - model doesn't fit training data well")
		if diagnostics['variance_captured'] < 0.5:
			issues.append("  ⚠️  Low variance captured - active subspace may be missing important directions")
		if diagnostics['vandermonde_condition'] > 1e10:
			issues.append("  ⚠️  Ill-conditioned Vandermonde - design points may be too close or poorly distributed")
		if diagnostics['min_point_distance'] < 1e-8:
			issues.append("  ⚠️  Points too close together - numerical instability likely")
		if diagnostics['orthonormality_error'] > 1e-6:
			issues.append("  ⚠️  Active subspace not orthonormal - projection may be inaccurate")
		
		if issues:
			for issue in issues:
				output += issue + "\n"
		else:
			output += "  ✓ All metrics look reasonable\n"
		
		output += "="*70 + "\n\n"
		self.write_diagnostics_to_txt(output)