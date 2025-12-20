"""
	Python script to show how the number of samples taken at a design point is decided throughout the run of an algorithm. 
	There are three plots:
	1. The first plot shows how the standard error of a solver changes throughout the run of the algorithm on ASTRODF 
	2. The second plot shows the amount of simulation budget used per iteration for a solver using adaptive sampling as opposed to fixed sampling
	3. The third plot shows how the value of $\frac{\lambda_k\hat{\sigma}^2(\mathbf{x},t)}{\kappa^2\Delta_k^2}$ changes as the monte carlo 
	effort (t) increases at a design point $\mathbf{x}$ within an iteration k of the algorithm.
"""
import sys
import os.path as o
sys.path.append(
	o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), ".."))
)
import json
import random
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns
from mrg32k3a.mrg32k3a import MRG32k3a


from simopt.experiment_base import instantiate_solver, instantiate_problem, ProblemSolver
from simopt.base import Solution, Problem


def densify_by_total(points, n_add):
    """
    Add a given total number of intermediate tuples along a piecewise-linear trajectory.

    Parameters:
        points (list[tuple]): Original list of coordinate tuples.
        n_add (int): Total number of new points to insert between the first and last tuple.

    Returns:
        list[tuple]: New list including original + added tuples.
    """

    if n_add <= 0 or len(points) < 2:
        return points

    # Compute lengths of each segment
    seg_lengths = []
    for i in range(len(points) - 1):
        p0, p1 = points[i], points[i+1]
        length = math.dist(p0, p1)
        seg_lengths.append(length)

    total_len = sum(seg_lengths)
    if total_len == 0:
        return points  # all points identical

    # Determine how many points to place in each segment (proportional distribution)
    points_per_seg = []
    allocated = 0

    for length in seg_lengths:
        count = (length / total_len) * n_add
        rounded = int(round(count))
        points_per_seg.append(rounded)
        allocated += rounded

    # Adjust for rounding error (ensure sum = n_add)
    while allocated > n_add:
        # remove one from the longest segment (largest rounding bias)
        idx = max(range(len(points_per_seg)), key=lambda i: points_per_seg[i])
        points_per_seg[idx] -= 1
        allocated -= 1

    while allocated < n_add:
        # add one to the longest segment
        idx = max(range(len(points_per_seg)), key=lambda i: seg_lengths[i])
        points_per_seg[idx] += 1
        allocated += 1

    # Build the output with interpolated points
    out = [points[0]]

    for i in range(len(points) - 1):
        p0, p1 = points[i], points[i+1]
        k = points_per_seg[i]

        for j in range(1, k + 1):
            t = j / (k + 1)
            interp = tuple(
                p0[d] + t * (p1[d] - p0[d])
                for d in range(len(p0))
            )
            out.append(interp)

        out.append(p1)

    return out



########################################
#    PLOT 1: STANDARD ERROR OVER RUN   #
########################################
def plot_standard_error_multiple_problems(problem_names: list[str]) :
	"""
		Generates a plot showing the standard error of solutions over the run of an algorithm for multiple problems. 
	"""
	solver_name = "ADAM" 
	sample_size = 10
	
	iterations = {}
	std_errors = {}

	for problem_name in problem_names : 
		iteration, std_error = plot_standard_error(problem_name, max_iter=100)	
		iterations[problem_name] = iteration
		std_errors[problem_name] = std_error
		
	#plot the standard errors against iterations 
	#plot the standard errors against iterations 
	plt.figure(figsize=(8,6))
	color = plt.cm.get_cmap('tab10', len(problem_names))
	for idx, problem_name in enumerate(problem_names) :
		iteration = iterations[problem_name]
		std_error = std_errors[problem_name]
		plt.plot(iteration, std_error, label=f'Standard Error over Iterations for {problem_name}', color=color(idx))
	plt.xlabel('Iteration')
	plt.ylabel('Standard Error')
	plt.title(f'Standard Error of Solutions over Iterations for {solver_name} on {problem_name}')
	plt.legend()
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(f'standard_error_{solver_name}.png')
	plt.show()


	

def plot_standard_error(problem_name: str, max_iter: int) -> tuple[list[int], list[float]]: 
	"""
		Generates a plot showing the standard error of solutions over the run of an algorithm. 
	"""
	solver_name = "ADAM" 
	sample_size = 10
	
	problem = instantiate_problem(problem_name, problem_fixed_factors={'budget': 10000})
	rng_list = [MRG32k3a() for _ in range(problem.model.n_rngs)]
	solver = instantiate_solver(solver_name)
	ps = ProblemSolver(problem=problem, solver=solver)
	ps.run(n_macroreps=1)

	xs = ps.all_recommended_xs[0]

	if len(xs) > max_iter:
		xs = xs[:max_iter]

	if len(xs) < max_iter:
		xs = densify_by_total(xs, max_iter - len(xs))
	
	iterations = list(range(1, len(xs)+1))

	std_errors = [] 


	for x in xs : 
		sol = Solution(x, problem)
		sol.attach_rngs(rng_list)
		problem.simulate(sol, sample_size)
		std_errors.append(sol.objectives_stderr.item())
	
	return iterations, std_errors

########################################
#    PLOT 2: SIMULATION BUDGET         #
########################################

def plot_simulation_budget_multiple_problems(problem_names: list[str]) :
	"""
		Generates a plot showing the simulation budget used per iteration for multiple problems. 
	"""
	solver_name = "ASTRO-DF" 
	
	plt.figure(figsize=(10,8))
	iterations_astro = {}
	iterations_trust = {}
	budgets_astro = {}
	budgets_trust = {}

	for problem_name in problem_names : 
		astro_data, trust_data = plot_simulation_budget(problem_name)	
		iterations_astro[problem_name] = astro_data[0]
		iterations_trust[problem_name] = trust_data[0]
		budgets_astro[problem_name] = astro_data[1]
		budgets_trust[problem_name] = trust_data[1]

	#get average budgets of each iteration over all problems for ASTRO-DF and Trust-Region
	avg_budgets_astro = [] 
	for i in range(1, max([len(budgets_astro[p]) for p in problem_names])+1) :
		sum_budget = 0
		count = 0
		for problem_name in problem_names :
			if i <= len(budgets_astro[problem_name]) :
				sum_budget += budgets_astro[problem_name][i-1]
				count += 1
		avg_budgets_astro.append(sum_budget / count if count > 0 else 0)

	avg_iterations_astro = list(range(1, len(avg_budgets_astro)+1))

	avg_budgets_trust = []
	for i in range(1, max([len(budgets_trust[p]) for p in problem_names])+1) :
		sum_budget = 0
		count = 0
		for problem_name in problem_names :
			if i <= len(budgets_trust[problem_name]) :
				sum_budget += budgets_trust[problem_name][i-1]
				count += 1
		avg_budgets_trust.append(sum_budget / count if count > 0 else 0)

	avg_iterations_trust = list(range(1, len(avg_budgets_trust)+1))

		
	#plot the standard errors against iterations 
	plt.figure(figsize=(8,6))
	color = plt.cm.get_cmap('tab10', 2*len(problem_names))
	for idx, problem_name in enumerate(problem_names) :
		iteration = iterations_astro[problem_name]
		budget = budgets_astro[problem_name]
		plt.plot(iteration, budget, linestyle='--', label=f'ASTRO-DF on {problem_name}', color=color(idx))

	for idx, problem_name in enumerate(problem_names) :
		new_idx = idx + len(problem_names)
		iteration = iterations_trust[problem_name]
		budget = budgets_trust[problem_name]
		plt.plot(iteration, budget, linestyle=':', label=f'Trust-Region on {problem_name}', color=color(new_idx))

	trend_astro = np.polyfit(avg_iterations_astro,avg_budgets_astro, 1)
	trend_trust = np.polyfit(avg_iterations_trust, avg_budgets_trust, 1)
  
	trend_line_astro = np.poly1d(trend_astro)
	trend_line_trust = np.poly1d(trend_trust)

	plt.plot(avg_iterations_astro, trend_line_astro(avg_iterations_astro), color='black', label='ASTRO-DF Trend Line')
	plt.plot(avg_iterations_trust, trend_line_trust(avg_iterations_trust), color='grey', label='Trust-Region Trend Line')
	
	plt.xlabel('Iteration')
	plt.ylabel('Simulation Budget Used')
	plt.title(f'Simulation Budget per Iteration')
	plt.legend(loc='lower right')
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(f'simulation_budget_{solver_name}.png')
	plt.show()

def plot_simulation_budget(problem_name: str):
	astrodf_solver = instantiate_solver("ASTRODF")
	trust_region_solver = instantiate_solver("TrustRegion")

	problem = instantiate_problem(problem_name, problem_fixed_factors={'budget': 10000})
	rng_list = [MRG32k3a() for _ in range(problem.model.n_rngs)]

	ps_astro = ProblemSolver(problem=problem, solver=astrodf_solver)
	ps_tr = ProblemSolver(problem=problem, solver=trust_region_solver)
	ps_astro.run(n_macroreps=1)
	ps_tr.run(n_macroreps=1)
	astrodf_budgets = ps_astro.all_intermediate_budgets[0]
	tr_budgets = ps_tr.all_intermediate_budgets[0]
	iterations_astro = list(range(1, len(astrodf_budgets)+1))
	iterations_tr = list(range(1, len(tr_budgets)+1))

	return [iterations_astro, astrodf_budgets], [iterations_tr, tr_budgets]


########################################################
#    PLOT 3: ADAPT SAMPLE METRIC AS SAMPLE INCREASES   #
########################################################

def get_stopping_time(k:int, sig2: float, kappa: float, lambda_min: int, delta: float, problem: Problem, budget: int, used_budget: int) -> float :
	"""
		Calculate the stopping time (sample size) based on current variance estimate, delta, and kappa
	Args:
		sig2 (float): Current variance estimate
		problem (Problem): The Simulation Problem being run.

	Returns:
		float: The calculated stopping time (sample size)
	"""
	lambda_max = budget - used_budget
	pilot_run = math.ceil(max(lambda_min * math.log(10 + k, 10) ** 1.1, min(0.5 * problem.dim, lambda_max))-1)

	if kappa == 0.0 or kappa is None:
		kappa = 1.0

	raw_sample_size = pilot_run * max(1, sig2 / (kappa**2 * delta**2))
	if isinstance(raw_sample_size, np.ndarray):
		raw_sample_size = raw_sample_size.item()
	# round up to the nearest integer
	sample_size: int = math.ceil(raw_sample_size)
	return sample_size


def calculate_kappa(problem: Problem, sol: Solution, budget: int, k: int, lambda_min: int, delta: float) -> None :
	"""
		Calculate kappa and run adaptive sampling on the incumbent solution to obtain a sample average of its response.
	Args:
		problem (Problem): The Simulation Problem being run.
	"""
	used_budget = 0
	lambda_max = budget - used_budget
	pilot_run = math.ceil(max(lambda_min * math.log(10 + k, 10) ** 1.1, min(0.5 * problem.dim, lambda_max))-1)

	kappa = 0.0

	#calculate kappa
	problem.simulate(sol, pilot_run)
	used_budget += pilot_run
	sample_size = pilot_run
	while True:
		rhs_for_kappa = sol.objectives_mean
		sig2 = sol.objectives_var[0]

		kappa = rhs_for_kappa * np.sqrt(pilot_run) / (delta ** 2)
		kappa = kappa.item()
		stopping = get_stopping_time(k, sig2, kappa, lambda_min, delta, problem, budget, used_budget)

		print(f'kappa is currently: {kappa}')
		
		if (sample_size >= min(stopping, lambda_max) or used_budget >= budget):
			# calculate kappa
			kappa = (rhs_for_kappa * np.sqrt(pilot_run)/ (delta ** 2))
			kappa = kappa.item()
			break
		
		problem.simulate(sol, 1)
		sample_size += 1

	return kappa, sol


def plot_adapt_sample_metric_multiple_problems(problem_names: list[str]) :
	"""
		Generates plots showing how the value of 
		$\frac{\lambda_k\hat{\sigma}^2(\mathbf{x},t)}{\kappa^2\Delta_k^2}$ changes as the monte carlo 
		effort (t) increases at a design point $\mathbf{x}$ within an iteration k of the algorithm for multiple problems.
	"""
	metrics = {}
	samples = {}
	for problem_name in problem_names : 
		sample_sizes, metric_values = plot_adapt_sample_metric(problem_name)
		samples[problem_name] = sample_sizes
		metrics[problem_name] = metric_values


	plt.figure(figsize=(8,6))
	linestyles = ['-', '--', ':', '-.']
	color = plt.cm.get_cmap('tab10', len(problem_names))
	for idx, problem_name in enumerate(problem_names) :
		sample_sizes = samples[problem_name]
		metric_values = metrics[problem_name]
		plt.plot(sample_sizes, metric_values, linestyle=linestyles[idx % len(linestyles)], label=f'{problem_name}', color=color(idx))
	
	#plot and x=y line for reference
	# sample_sizes_log = [np.exp(a) for a in sample_sizes]
	# plt.plot(sample_sizes, sample_sizes, linestyle='--', color='gray')
	plt.xlabel('Monte Carlo Effort (t)')
	plt.ylabel(r'$\frac{\lambda_k\hat{\sigma}^2(\mathbf{x},t)}{\kappa^2\Delta_k^2}$')
	# plt.title(r'$\frac{\lambda_k\hat{\sigma}^2(\mathbf{x},t)}{\kappa^2\Delta_k^2}$ vs Monte Carlo Effort')
	plt.legend()
	plt.yscale('log')
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(f'adapt_sample_metric.png')
	plt.show()

def plot_adapt_sample_metric(problem_name: str): 
	"""
		Generates a plot showing how the value of 
		$\frac{\lambda_k\hat{\sigma}^2(\mathbf{x},t)}{\kappa^2\Delta_k^2}$ changes as the monte carlo 
		effort (t) increases at a design point $\mathbf{x}$ within an iteration k of the algorithm.
	"""
	solver_name = "ASTROMORF" 
	sample_sizes = list(range(1, 201))
	metric_values = []

	problem = instantiate_problem(problem_name, problem_fixed_factors={'budget': 10000})
	rng_list = [MRG32k3a() for _ in range(problem.model.n_rngs)]
	solver = instantiate_solver(solver_name)
	
	x = problem.factors['initial_solution']

	lambda_min = 5 
	delta_k = 2.0
	k = 5 
	budget = 10000

	sol = Solution(x, problem)
	sol.attach_rngs(rng_list)
	kappa, sol = calculate_kappa(problem, sol, budget, k, lambda_min, delta_k)


	for t in sample_sizes : 
		sol = Solution(x, problem)
		sol.attach_rngs(rng_list)
		problem.simulate(sol, t)
		sigma_hat_sq = sol.objectives_stderr.item() ** 2
		metric = (lambda_min * sigma_hat_sq) / (kappa**2 * delta_k**2)
		metric_values.append(metric)

	return sample_sizes, metric_values




def main():
	# problem_names = ["DYNAMNEWS-1", "FACSIZE-1", "SAN-1", "FIXEDSAN-1"]
	problem_names = ["DYNAMNEWS-1", "EXAMPLE-1", "SAN-1", "FIXEDSAN-1"]
	# plot_standard_error_multiple_problems(problem_names)
	# plot_simulation_budget_multiple_problems(problem_names)
	plot_adapt_sample_metric_multiple_problems(problem_names)


if __name__ == "__main__":
	main()	