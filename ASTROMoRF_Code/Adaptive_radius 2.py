"""
    This plot shows how the mean squared error of the constructed model changes as point sampled move away from the center point.
"""

import sys
import os.path as o
sys.path.append(
    o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), ".."))
)

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mrg32k3a.mrg32k3a import MRG32k3a
from scipy.optimize import minimize, NonlinearConstraint, linprog
from scipy import optimize

from simopt.base import Solution, Problem
from simopt.experiment_base import instantiate_problem

# Import functions from the other modules
from design_set_test import construct_interpolation_set
from geometry_improvement_test import improve_geometry, generate_set
from Model_construction import construct_model, fit




def adaptive_radius_plot(problem_name,initial_radius):
    """
    Plots the mean squared error of the constructed model as a function of the radius of the interpolation set.

    Parameters:
    - problem_name: str, name of the optimization problem
    - initial_radius: float, starting radius for the interpolation set
    """
    
    problem = instantiate_problem(problem_name)

    initial_solution_array = np.array(problem.factors['initial_solution'])


    center = initial_solution_array
    f_curr, init_solutions = calculate_responses(np.array([initial_solution_array]).reshape(1, -1), problem)
    initial_solution = init_solutions[0]
    k = 2
    dim = 2
    degree = 2
    budget = 0 
    visited_pts = [initial_solution]

    init_S_full = generate_set(problem, dim, initial_solution, initial_radius)
    U0, _ = np.linalg.qr(init_S_full.T)

    radii = np.linspace(1e-3, initial_radius+1, 100)
    r_squared = []


    for radius in radii:
        #construct design set
        new_solution, new_f, X, f_values, interpolation_solutions, visited_pts, radius = construct_design_set(initial_solution, degree, problem, f_curr, U0, radius, visited_pts)

        
        # Construct model
        U, model, model_grad, X, fX, interpolation_solutions, visited_pts, delta = fit(k, degree, problem, X, f_values, new_solution, new_f, radius, interpolation_solutions, visited_pts, U0)
        
        # Evaluate MSE at a set of test points within the original radius
        r_squared_val = calculate_r_squared(initial_radius, problem, new_solution, model)
        r_squared.append(r_squared_val)

    #normalize the radii for better plotting where 0.3 means the radius is 30% of the initial radius
    radii = radii / initial_radius


    # Plotting
    print('creating plot...')
    print(f'r_squared values: {r_squared} for radii: {radii}')
    plt.figure(figsize=(8, 12))
    plt.plot(radii, r_squared, label='Model MSE', color='blue')
    #plot a region between x=0.7 and x=0.8
    plt.fill_betweenx([min(r_squared)-2, max(r_squared)+2], 0.7, 0.8, color='orange', alpha=0.45, label='Typical Radius Range')
    #plot a dashed line at x=1.0 
    plt.axvline(x=1.0, color='red', linestyle='--', label='Initial Radius')
    plt.xlabel('Radius Relative to Initial Radius')
    plt.ylabel('MSE')
    plt.ylim(min(r_squared)-2, max(r_squared)+2)
    plt.grid(True)
    # plt.legend()
    plt.savefig('adaptive_radius_r_squared_plot.png')
    plt.show()


def construct_design_set(current_solution: Solution, degree: int, problem: Problem, f_curr: float, U: np.ndarray, radius: float, visited_pts: list[Solution]) :
    """
        Construct the design set for model building. Ensures we are testing the reuse design points construction
        by first constructing a design set without reuse, then moving the current solution, and constructing
        a design set with reuse.
    """

    #First iteration of design set construction
    X, _ = construct_interpolation_set(1, current_solution, problem, U, radius, visited_pts)
    f_values, interpolation_solutions = calculate_responses(X, problem)

    U, model, model_grad, X, fX, interpolation_solutions, visited_pts, delta = fit(1, degree, problem, X, f_values, current_solution, f_curr, radius, interpolation_solutions, visited_pts, U)

    new_solution = solve_subproblem(current_solution, model, model_grad, problem, U, radius)

    new_fval, new_solution = calculate_responses(np.array([new_solution.x]).reshape(1, -1), problem)

    new_solution = new_solution[0]
    new_fval = new_fval[0]
    radius = 1.2 * radius  #increase radius for second iteration

    visited_pts.append(new_solution)

    #Second iteration of design set construction with reuse
    X, _ = construct_interpolation_set(2, new_solution, problem, U, radius, visited_pts)
    f_values, interpolation_solutions = calculate_responses(X, problem)

    print(f'Constructed design set at radius {radius} with {X.shape[0]} points.')

    return new_solution, new_fval, X, f_values, interpolation_solutions, visited_pts, radius


def generate_test_points_in_ball(center: np.ndarray, radius: float, num_points: int, n: int) -> np.ndarray:
    """Generate random test points within a ball of given radius around center."""
    rng = MRG32k3a()
    test_points = []
    center = center.reshape(-1, 1)

    def random_pt() : 
        direction = np.array([rng.normalvariate() for _ in range(n)]).reshape(-1, 1)
        direction /= norm(direction)  # Normalize to unit vector
        distance = rng.uniform(0, radius)  # Random distance within the radius
        point = center + distance * direction  # Scale and shift to center
        return point 
    
    for _ in range(num_points):
        point = random_pt() 
        test_points.append(point.flatten())

    return np.array(test_points).reshape(num_points, -1)

def calculate_r_squared(radius, problem: Problem, current_solution, model, no_pts: int = 40) -> float:
    """Calculate the mean squared error of the model within the given radius around the center."""
    #randomly generate no_pts test points within the ball of radius radius and center current_solution.x
    test_points = generate_test_points_in_ball(np.array(current_solution.x), radius, no_pts, problem.dim)

    true_values, _ = calculate_responses(test_points, problem) #shape (M, 1)

    predicted_values = [] 
    for x in test_points:
        x_reshaped = np.array(x).reshape(-1, 1)
        pred = model(x_reshaped)
        predicted_values.append(pred)

    predicted_values = np.array(predicted_values).reshape(-1, 1)  #shape (M, 1)

    residuals = true_values - predicted_values

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((true_values.flatten() - np.mean(true_values.flatten()))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    # mse = ss_res / no_pts
    
    return np.mean(ss_res)

def calculate_responses(X: np.ndarray, problem: Problem):
    """Calculate the function values at the given points X."""
    rng_list = [MRG32k3a() for _ in range(problem.model.n_rngs)]
    #Create solutions 
    solutions = [Solution(x, problem) for x in X]


    for sol in solutions:
        sol.attach_rngs(rng_list)

    #simulate solutions 
    simulated_sols = []
    for sol in solutions:
        problem.simulate(sol, 10)
        simulated_sols.append(sol)

    #get responses
    responses = np.array([-1 * problem.minmax[0] * sol.objectives_mean.item() for sol in simulated_sols]).reshape(-1,1) #shape (M, )
    return responses, simulated_sols


def solve_subproblem(current_solution: Solution, model: callable, model_grad: callable, problem: Problem, U: np.ndarray, delta: float) -> Solution :
    """
        Solves the trust-region subproblem within the reduced subspace with regularization.
        Objective: min m_k(U@z) + lambda * ||U@z||^2 / ||z||^2
        Constraint: ||x_k + U@z - x_k|| <= delta (full-space trust region)
        
        The regularization term penalizes small full-space steps for given reduced-space steps,
        encouraging solutions that make substantial progress in full space.

        NEW APPROACH: Penalize based on ratio of norms ||U·s|| / ||s|| to encourage large full-space steps
        while keeping the optimization stable. The norm is L-1 for better numerical behavior.


    Args:
        model (callable): The surrogate model function
        model_grad (callable): The surrogate model gradient function
        problem (Problem): The current simulation model
        U (np.ndarray): The (n,d) active subsapce matrix

    Returns:
        Solution: The candidate solution in the full space
    """
    n,d = U.shape
    # Get current solution in full space
    x_current = np.array(current_solution.x).reshape(-1, 1)  # shape (n, 1)
    
    # Regularization weight: prevent null-space drift
    # Very small value to avoid interfering with optimization convergence
    lambda_reg = 0.0  # Reduced from 0.01 to be even less aggressive

    def obj_fn(z):
        # z is the step in reduced space (d,)
        z_col = np.array(z).reshape(-1, 1)  # shape (d, 1)
        z_full = U @ z_col  # Full-space step, shape (n, 1)
        
        # Model value at reduced-space step: m(s)
        model_val = model(z_col.reshape(-1, 1))
        
        # Regularization: penalize when full-space step is small relative to reduced-space step
        # This encourages steps that make substantial progress in full space
        full_space_step_norm = norm(z_full, ord=1)
        reduced_space_step_norm = norm(z_col, ord=1)
        
        # if reduced_space_step_norm > 1e-10:
        # Penalty is low when full_space_step ≈ reduced_space_step (good)
        # Penalty is high when full_space_step << reduced_space_step (bad - null space drift)
        ratio_penalty = float(reduced_space_step_norm / (full_space_step_norm + 1e-10)) 
            # ratio_penalty = max(0.0, ratio_penalty)  # Only penalize when ratio > 1
        # else:
            # ratio_penalty = 0.0
        
        res = float(model_val + lambda_reg * ratio_penalty)
        return res 

    def obj_grad(z):
        # get gradient of objective function
        z_col = np.array(z).reshape(-1, 1)  # shape (d, 1)
        z_full = U @ z_col  # Full-space step, shape (n, 1)
        
        # Get gradient in reduced space (d-dimensional)
        # model_grad accepts reduced-space coordinates directly
        model_grad_reduced = model_grad(z_col.reshape(1, -1)).flatten()  # shape (d,)
        
        # Gradient of penalty term: ∂/∂s[||s|| / ||U·s||]
        # = s/(||s||·||U·s||) - (||s||/||U·s||³)·U^T·(U·s)
        full_space_step_norm = norm(z_full, ord=1)  # ||U·s||
        reduced_space_step_norm = norm(z_col, ord=1)  # ||s||
        
        if reduced_space_step_norm > 1e-10 and full_space_step_norm > 1e-10:
            term1 = z_col / (reduced_space_step_norm * full_space_step_norm)  # s/(||s||·||U·s||)
            term2 = (reduced_space_step_norm / (full_space_step_norm ** 3)) * (U.T @ z_full)  # (||s||/||U·s||³)·U^T·(U·s)
            penalty_grad = (term1 - term2).flatten()  # shape (d,)
        else:
            penalty_grad = np.zeros(d)  # Avoid division by zero at origin

        # if reduced_space_step_norm > 1e-10 and full_space_step_norm > 1e-10:
        # 	ratio = reduced_space_step_norm / full_space_step_norm
        # 	if ratio > 1.0:  # Only add gradient when penalizing
        # 		# Gradient of penalty term (approximation for numerical stability)
        # 		grad_penalty = z / (reduced_space_step_norm + 1e-10)
        # 		model_grad_reduced = model_grad_reduced + lambda_reg * grad_penalty
        model_grad_reduced += lambda_reg * penalty_grad.flatten()
        
        return model_grad_reduced
    
    def cons_fn(z):
        # Constraint: ||s|| <= delta (step size in REDUCED space)
        # Penalty term encourages large ||U·s|| relative to ||s|| for algorithmic progress
        z_col = np.array(z).reshape(-1, 1)
        return norm(z_col)

    cons = NonlinearConstraint(cons_fn, 0, delta)
    # Check model gradient at current point to diagnose flat models
    grad_at_current = obj_grad(np.zeros(d)) 


    if norm(grad_at_current) < 1e-6:
        warning = f'⚠️ WARNING: Model gradient is very small - model may be too flat!\n'
    
    # Solve trust region subproblem
    # Start from origin (no step) in reduced space
    # Relax tolerances to prevent premature xtol termination
    res = minimize(obj_fn, np.zeros(d), method='trust-constr', jac=obj_grad, 
                    constraints=cons, options={'disp': False, 'verbose': 0, 
                    'xtol': 1e-8, 'gtol': 1e-6})
    
    if not res.success:
        warning = f'⚠️ WARNING: Optimizer did not fully converge: {res.message}\n'
    

    # Ensure the reduced-space step respects the trust region (guard against optimizer drift)
    step_norm_reduced = norm(res.x)
    if step_norm_reduced > delta * 1.001:  # 0.1% tolerance
        res.x = res.x * (delta / step_norm_reduced)
        step_norm_reduced = delta
    
    # Check if step hit trust region boundary (indicates model gradient is strong)
    boundary_tolerance = 0.9 * delta

    if step_norm_reduced < boundary_tolerance :	
        # If step is very small, try Cauchy point (steepest descent to boundary)
        if step_norm_reduced < 0.5 * delta:
            # Get gradient at current point in reduced space
            grad_reduced = obj_grad(np.zeros(d))
            grad_norm = norm(grad_reduced)
            
            if grad_norm > 1e-10:
                # Cauchy point: step in steepest descent direction to trust region boundary
                cauchy_step = -(delta / grad_norm) * grad_reduced
                cauchy_obj = obj_fn(cauchy_step)
                current_obj = obj_fn(np.zeros(d))
                
                # Use Cauchy point if it improves over the optimizer's solution
                if cauchy_obj < obj_fn(res.x):
                    res.x = cauchy_step
                    step_norm_reduced = delta
            

    # Compute full space point directly (already done in obj_fn, but needed here)
    s_new = (x_current + U @ res.x.reshape(-1, 1)).flatten()

    s_new = s_new.tolist()

    

    #block constraints 
    s_new = [
        clamp_with_epsilon(val, problem.lower_bounds[j], problem.upper_bounds[j])
        for j, val in enumerate(s_new)
    ]
    s_new = np.array(s_new).flatten()



    candidate_solution = Solution(tuple(s_new), problem)
    candidate_solution.attach_rngs([MRG32k3a() for _ in range(problem.model.n_rngs)])


    return candidate_solution

def clamp_with_epsilon(val: float, lower_bound: float, upper_bound: float, epsilon: float = 0.01) -> float:
	"""Clamp a value within bounds while avoiding exact boundary values.

	Adds a small epsilon to the lower bound or subtracts it from the upper bound
	if `val` lies outside the specified range.

	Args:
		val (float): The value to clamp.
		lower_bound (float): Minimum acceptable value.
		upper_bound (float): Maximum acceptable value.
		epsilon (float, optional): Small margin to avoid returning exact boundary
			values. Defaults to 0.01.

	Returns:
		float: The adjusted value, guaranteed to lie strictly within the bounds.
	"""
	if val <= lower_bound:
		return lower_bound + epsilon
	if val >= upper_bound:
		return upper_bound - epsilon
	return val

def main() : 
    problem_name = "ROSENBROCK-1"
    delta = 2.0
    adaptive_radius_plot(problem_name, delta)

if __name__ == "__main__":
    main()