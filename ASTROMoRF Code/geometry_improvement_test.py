import sys
import os.path as o
sys.path.append(
o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), ".."))
)  #

from simopt.base import Problem, Solution
import numpy as np
from numpy.linalg import norm, pinv, qr
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, linprog
from scipy import optimize
from scipy.special import factorial
import scipy
import math
from math import ceil, isnan, isinf, comb, log, floor

from simopt.experiment_base import instantiate_problem


def evaluate_interpolation_points(k: int, problem: Problem, current_solution: Solution, expended_budget: int, visited_index: int, X:np.ndarray, visited_pts: list[Solution], delta_k: float) -> tuple[np.ndarray, Solution, list[Solution], list[Solution], int] :
    """
        Run adaptive sampling on the model construction design points to obtain a sample 
        average of their responses.
    Args:
        problem (Problem): The current simulation model 
        X (np.ndarray): A (M,n) numpy matrix of M n-dimensional design points used to construct the 
                        interpolation models
        visited_pts (list[float]): A list of the points already simulated by the solver

    Returns:
        tuple[np.ndarray, list[Solution], list[Solution]]: Consists of function values fX of shape (M,), the interpolation solutions and the visited points
    """	
    fX = []	 
    interpolation_solutions = []
    for idx,x in enumerate(X) : 
        #for the current solution, we don't need to simulate
        if (idx == 0) and (k==1) :
            fX.append(-1 * problem.minmax[0] * current_solution.objectives_mean) 
            interpolation_solutions.append(current_solution)
        
        #reuse the replications for x_k
        elif idx == 0: 
            current_solution, expended_budget = adaptive_sampling_before(problem, k, current_solution, delta_k, expended_budget)
            fX.append(-1 * problem.minmax[0] * current_solution.objectives_mean)
            interpolation_solutions.append(current_solution)

        elif (idx==1) and ((norm(np.array(current_solution.x)-np.array(visited_pts[visited_index].x)) != 0) and visited_pts is not None) :
            reuse_solution = visited_pts[visited_index]
            reuse_solution, expended_budget = adaptive_sampling_before(problem, k, reuse_solution, delta_k, expended_budget)
            fX.append(-1 * problem.minmax[0] * reuse_solution.objectives_mean)
            interpolation_solutions.append(reuse_solution)
        #For new points, run the simulation with pilot run
        else :
            solution = Solution(tuple(x), problem)
            solution, expended_budget = adaptive_sampling_after(problem, k, solution, delta_k, expended_budget)
            fX.append(-1 * problem.minmax[0] * solution.objectives_mean)
            interpolation_solutions.append(solution)


    return np.array(fX), current_solution, interpolation_solutions, visited_pts, expended_budget


def simulate_candidate_soln(self, k: int, problem: Problem, candidate_solution: Solution, expended_budget: int, current_solution: Solution, delta_k: float) -> tuple[Solution, float, int]:
		"""
			Run adaptive sampling on the candidate solution to obtain a sample average of the 
			response to the candidate solution.

		Args:
			problem (Problem): The Simulation Problem being run.
			candidate_solution (Solution): The candidate solution selected by the current iteration
			current_solution (Solution): The incumbent solution of the solver

		Returns:
			tuple[Solution, float]: Consists of the candidate solution and its evaluated solution
		"""
		if self.factors['crn_across_solns'] :
			problem.simulate(candidate_solution, current_solution.n_reps) 
			expended_budget += current_solution.n_reps 
			fval_tilde = -1 * problem.minmax[0] * candidate_solution.objectives_mean
			return candidate_solution, fval_tilde, expended_budget
			
		else :
			candidate_solution, expended_budget = self.adaptive_sampling_after(problem, k, candidate_solution, delta_k, expended_budget)
			fval_tilde = -1 * problem.minmax[0] * candidate_solution.objectives_mean
			return candidate_solution, fval_tilde, expended_budget
		

def initial_iteration_calculation(k: int, problem: Problem, expended_budget: int, current_solution: Solution, delta_k: float, visited_points: list[Solution], recommended_solns: list[Solution], intermediate_budgets: list[int]) : 
    current_solution = Solution(current_solution.x, problem)
    # current_solution = self.create_new_solution(current_solution.x, problem)
    if len(visited_points) == 0:
        visited_points.append(current_solution)
    
    current_solution, expended_budget, kappa = calculate_kappa(k, problem, expended_budget, current_solution, delta_k)

    recommended_solns.append(current_solution)
    intermediate_budgets.append(expended_budget) 

    return current_solution, expended_budget, kappa, recommended_solns, intermediate_budgets



def calculate_pilot_run(k: int, problem: Problem, expended_budget: int) -> int : 
    lambda_min = 7 
    lambda_max = 10000 - expended_budget
    pilot_run = ceil(max(lambda_min * log(10 + k, 10) ** 1.1, min(0.5 * problem.dim, lambda_max))-1)
    
    return pilot_run

def calculate_kappa(k: int, delta_power: int, kappa: float,  problem: Problem, expended_budget: int, current_solution: Solution, delta_k: float) -> tuple[Solution, int, float] :
    lambda_max = problem.factors['budget'] - expended_budget
    pilot_run = calculate_pilot_run(k, problem, expended_budget)

    #calculate kappa
    problem.simulate(current_solution, pilot_run)
    expended_budget += pilot_run
    sample_size = pilot_run
    while True:
        rhs_for_kappa = current_solution.objectives_mean
        sig2 = current_solution.objectives_var[0]

        kappa = rhs_for_kappa * np.sqrt(pilot_run) / (delta_k ** (delta_power / 2))
        stopping = get_stopping_time(sig2, delta_k, k, problem, expended_budget)
        
        if (sample_size >= min(stopping, lambda_max) or expended_budget >= problem.factors['budget']):
            # calculate kappa
            kappa = (rhs_for_kappa * np.sqrt(pilot_run)/ (delta_k ** (delta_power / 2)))
            return current_solution, expended_budget, kappa 
        
        problem.simulate(current_solution, 1)
        expended_budget += 1
        sample_size += 1


def get_stopping_time(kappa: float, sig2: float, delta: float, k:int, problem: Problem, expended_budget: int) -> tuple[float, int] :
    """
    Compute the sample size based on adaptive sampling stopping rule using the optimality gap
    """
    delta_power = 4
    pilot_run = calculate_pilot_run(k, problem, expended_budget)
    if kappa == 0:
        kappa = 1

    raw_sample_size = pilot_run * max(1, sig2 / (kappa**2 * delta**delta_power))
    if isinstance(raw_sample_size, np.ndarray):
        raw_sample_size = raw_sample_size.item()
    # round up to the nearest integer
    sample_size: int = ceil(raw_sample_size)
    return kappa, sample_size

def samplesize(k,sig, kappa, delta) -> int:
    lambda_k: int = 1 if k==1 else floor(7*math.log(k)**1.5)
    S_k = ceil((lambda_k*sig**2)/(kappa**2*delta**4))
    return S_k

def adaptive_sampling_after(problem: Problem, k: int, new_solution: Solution, delta_k: float, used_budget: int) -> tuple[Solution, int]  :
    # adaptive sampling
    lambda_max = problem.factors['budget'] - used_budget
    pilot_run = calculate_pilot_run(k, problem, used_budget)

    problem.simulate(new_solution, pilot_run)
    used_budget += pilot_run
    sample_size = pilot_run
    sig_init = new_solution.objectives_var[0]

    while True:
        sig2 = new_solution.objectives_var[0]
        stopping = get_stopping_time(sig2, delta_k, k, problem, used_budget)
        if ((sample_size >= min(stopping, lambda_max)) or used_budget >= problem.factors['budget']):
            return new_solution, used_budget
        problem.simulate(new_solution, 1)
        
        used_budget += 1
        sample_size += 1


def adaptive_sampling_before(problem: Problem, k: int, new_solution: Solution, delta_k: float, used_budget: int) -> tuple[Solution, int] : 

    lambda_max = problem.factors['budget'] - used_budget
    if new_solution.n_reps < 2 :
        sim_no = 2-new_solution.n_reps
        problem.simulate(new_solution,sim_no)
        used_budget += sim_no
        sample_size = sim_no

    sample_size = new_solution.n_reps 
    sig2 = new_solution.objectives_var[0]

    while True:
        stopping = get_stopping_time(sig2, delta_k, k, problem, used_budget)
        if (sample_size >= min(stopping, lambda_max) or used_budget >= problem.factors['budget']):
            return new_solution, used_budget
        problem.simulate(new_solution, 1)
        
        used_budget += 1
        sample_size += 1
        sig2 = new_solution.objectives_var[0]

def generate_set(problem: Problem, num: int, current_solution: Solution, delta_k: float) -> np.ndarray:
    s_old = np.array(current_solution.x).reshape(-1,1)


    bounds_l = np.maximum(np.array(problem.lower_bounds).reshape(s_old.shape), s_old-delta_k)
    bounds_u = np.minimum(np.array(problem.upper_bounds).reshape(s_old.shape), s_old+delta_k)

    direcs = random_directions(problem, num, (bounds_l-s_old).flatten(), (bounds_u-s_old).flatten(), delta_k)

    S = np.zeros((num, problem.dim))
    S[0, :] = s_old.flatten()
    bounds_l_flat = bounds_l.flatten()
    bounds_u_flat = bounds_u.flatten()
    s_old_flat = s_old.flatten()
    for i in range(1, num):
        S[i, :] = s_old_flat + np.minimum(np.maximum(bounds_l_flat-s_old_flat, direcs[i, :]), bounds_u_flat-s_old_flat)

    return S #shape (num, n)

def get_scale(dirn: list[float], delta: float, lower: np.ndarray, upper: np.ndarray) -> float:
    """
        Sets the max distance for each direction in the current trust region 
    """
    scale = delta
    for j in range(len(dirn)):
        if dirn[j] < 0.0:
            scale = min(scale, lower[j] / dirn[j])
        elif dirn[j] > 0.0:
            scale = min(scale, upper[j] / dirn[j])
    return scale

def coordinate_directions(problem: Problem, num_pnts: int, lower: np.ndarray, upper: np.ndarray, delta_k: float) -> np.ndarray:
    """
    Generates coordinate directions
    """
    n = problem.dim
    at_lower_boundary = (lower > -1.e-8 * delta_k)
    at_upper_boundary = (upper < 1.e-8 * delta_k)
    direcs = np.zeros((num_pnts, n))
    for i in range(1, num_pnts):
        if 1 <= i < n + 1:
            dirn = i - 1
            step = delta_k if not at_upper_boundary[dirn] else - delta_k
            direcs[i, dirn] = step
        elif n + 1 <= i < 2*n + 1:
            dirn = i - n - 1
            step = - delta_k
            if at_lower_boundary[dirn]:
                step = min(2.0* delta_k, upper[dirn])
            if at_upper_boundary[dirn]:
                step = max(-2.0* delta_k, lower[dirn])
            direcs[i, dirn] = step
        else:
            itemp = (i - n - 1) // n
            q = i - itemp*n - n
            p = q + itemp
            if p > n:
                p, q = q, p - n
            direcs[i, p-1] = direcs[p, p-1]
            direcs[i, q-1] = direcs[q, q-1]
    return direcs #shape (num_pnts, n)

def random_directions(problem: Problem, num_pnts: int, lower: np.ndarray, upper: np.ndarray, delta_k: float) -> np.ndarray:
    """
    Generates orthogonal, random directions
    """
    n = problem.dim
    direcs = np.zeros((n, max(2*n+1, num_pnts)))
    idx_l = (lower == 0)
    idx_u = (upper == 0)
    active = np.logical_or(idx_l, idx_u)
    inactive = np.logical_not(active)
    nactive = np.sum(active)
    ninactive = n - nactive
    if ninactive > 0:
        A = np.random.normal(size=(ninactive, ninactive))
        Qred = qr(A)[0]
        Q = np.zeros((n, ninactive))
        Q[inactive, :] = Qred
        for i in range(ninactive):
            scale = get_scale(Q[:,i], delta_k, lower, upper) 
            direcs[:, i] = scale * Q[:,i]
            scale = get_scale(-Q[:,i], delta_k, lower, upper)
            direcs[:, n+i] = -scale * Q[:,i]
    idx_active = np.where(active)[0]
    for i in range(nactive):
        idx = idx_active[i]
        direcs[idx, ninactive+i] = 1.0 if idx_l[idx] else -1.0
        direcs[:, ninactive+i] = get_scale(direcs[:, ninactive+i], delta_k, lower, upper) * direcs[:, ninactive+i]
        sign = 1.0 if idx_l[idx] else -1.0
        if upper[idx] - lower[idx] > delta_k:
            direcs[idx, n+ninactive+i] = 2.0*sign*delta_k
        else:
            direcs[idx, n+ninactive+i] = 0.5*sign*(upper[idx] - lower[idx])
        direcs[:, n+ninactive+i] = get_scale(direcs[:, n+ninactive+i], 1.0, lower, upper)*direcs[:, n+ninactive+i]
    for i in range(num_pnts - 2*n):
        dirn = np.random.normal(size=(n,))
        for j in range(nactive):
            idx = idx_active[j]
            sign = 1.0 if idx_l[idx] else -1.0
            if dirn[idx]*sign < 0.0:
                dirn[idx] *= -1.0
        dirn = dirn / norm(dirn)
        scale = get_scale(dirn, delta_k, lower, upper)
        direcs[:, 2*n+i] = dirn * scale
    return np.vstack((np.zeros(n), direcs[:, :num_pnts].T)) #shape (num_pnts, n)

def improve_geometry(k: int, problem: Problem, current_solution: Solution, current_f: float, delta_k: float,
                     U: np.ndarray, visited_pts_list: list[Solution], X: np.ndarray, fX: np.ndarray, interpolation_solutions: list[int], expended_budget: int) -> tuple[np.ndarray, np.ndarray, list[Solution], list[Solution], float, int]:
    epsilon_1 = 0.5  
    epsilon_2 = 10.0
    dist = epsilon_1*delta_k
    s_old = np.array(current_solution.x).reshape(-1,1)
    
    if max(norm(X-s_old.ravel(), axis=1, ord=np.inf)) > dist:
        X, fX, delta_k, visited_pts_list, interpolation_solutions, expended_budget = sample_set(k, problem, current_solution, delta_k, current_f, U, X, fX, visited_pts_list, interpolation_solutions, expended_budget)

    #build interpolation_sols 
    interpolation_sols = [Solution(tuple(x), problem) for x in X]

    #add on interpolation solutions that have been visited onto the visited points
    existing_x = {obj.x for obj in visited_pts_list}
    for obj in interpolation_sols:
        if obj.x not in existing_x:
            visited_pts_list.append(obj)
            existing_x.add(obj.x)  # Update the set

    #add in visited points

    
    return X, fX, interpolation_sols, visited_pts_list, delta_k, expended_budget

def sample_set(k: int, problem: Problem, current_solution: Solution, delta_k: float, current_f: float, U: np.ndarray,
               X: np.ndarray, fX: np.ndarray, visited_pts: list[Solution], interpolation_solutions: list[Solution], expended_budget: int) -> tuple[np.ndarray, np.ndarray, float, list[Solution], list[Solution], int]:
    epsilon_1 = 0.5  
    d = U.shape[1]
    q = len(index_set(2, d).astype(int))
    s_old = np.array(current_solution.x).reshape(-1,1)

    dist = epsilon_1*delta_k

    S_hat = np.copy(X)
    f_hat = np.copy(fX)
    if max(norm(S_hat-s_old.ravel(), axis=1, ord=np.inf)) > dist:
        S_hat, f_hat = remove_furthest_point(S_hat, f_hat, s_old)
    S_hat, f_hat = remove_point_from_set(S_hat, f_hat, s_old)
    X = np.zeros((q, problem.dim))
    fX = np.zeros((q, 1))
    X[0, :] = s_old.flatten()
    fX[0, :] = current_f
    X, fX, delta_k, visited_pts, interpolation_solutions, expended_budget = LU_pivoting(k, problem, X, fX, current_solution, delta_k, S_hat, f_hat, U, expended_budget, visited_pts, interpolation_solutions)

    return X, fX, delta_k, visited_pts, interpolation_solutions, expended_budget

def LU_pivoting(k: int, problem: Problem, X: np.ndarray, fX: np.ndarray, current_solution: Solution, delta_k: float, S_hat: np.ndarray, f_hat: np.ndarray, U: np.ndarray,
                expended_budget: int, visited_pts: list[Solution], interpolation_solutions: list[Solution]) -> tuple[np.ndarray, np.ndarray, float, list[Solution], list[Solution], int] :
    psi_1 = 1.0e-4
    psi_2 = 0.25
    d = U.shape[1]
    s_old = np.array(current_solution.x).reshape(-1,1)

    phi_function, phi_function_deriv, delta_k = get_phi_function_and_derivative(S_hat, current_solution, delta_k, U)
    q = len(index_set(2, d).astype(int))
    p = X.shape[0]

    #Initialise R matrix of LU factorisation of M matrix (see Conn et al.)
    R = np.zeros((p,q))
    R[0,:] = phi_function(s_old)

    #Perform the LU factorisation algorithm for the rest of the points
    for k in range(1, q):
        flag = True
        v = np.zeros(q)
        for j in range(k):
            v[j] = -R[j,k] / R[j,j]
        v[k] = 1.0

        #If there are still points to choose from, find if points meet criterion. If so, use the index to choose 
        #point with given index to be next point in regression/interpolation set
        if f_hat.size > 0:
            Phi_S_hat = np.vstack([phi_function(S_hat[i, :].reshape(-1, 1)) for i in range(S_hat.shape[0])])
            M = np.absolute(Phi_S_hat @ v)
            index = np.argmax(M)
            if M[index] < psi_1:
                flag = False
            elif (k == q - 1 and M[index] < psi_2):
                flag = False
        else:
            flag = False
        
        #If index exists, choose the point with that index and delete it from possible choices
        if flag:
            s = S_hat[index,:]
            X[k, :] = s
            fX[k, :] = f_hat[index]
            S_hat = np.delete(S_hat, index, 0)
            f_hat = np.delete(f_hat, index, 0)

        #If index doesn't exist, solve an optimisation problem to find the point in the range which best satisfies criterion
        else:
            try:
                s = find_new_point(problem, current_solution, v, phi_function, phi_function_deriv, delta_k)
                if np.unique(np.vstack((X[:k, :], s)), axis=0).shape[0] != k+1:
                    s = find_new_point_alternative(problem, current_solution, v, phi_function, X[:k, :], delta_k)
            except:
                s = find_new_point_alternative(problem, current_solution, v, phi_function, X[:k, :], delta_k)
            if f_hat.size > 0 and M[index] >= abs(np.dot(v, phi_function(s))):
                s = S_hat[index,:]
                X[k, :] = s
                fX[k, :] = f_hat[index]
                S_hat = np.delete(S_hat, index, 0)
                f_hat = np.delete(f_hat, index, 0)
            else:
                X[k, :] = s
                # soln_at_s = self.create_new_solution(tuple(s), problem)
                # fs, soln_at_s, interpolation_solutions, visited_pts, expended_budget = evaluate_interpolation_points(k,problem, current_solution, expended_budget, 0, X, visited_pts, delta_k )
                # visited_pts.append(soln_at_s)
                # interpolation_solutions.append(soln_at_s)
                fs = np.linalg.norm(s) ** 2
                fX[k, :] = fs
        
        #Update R factorisation in LU algorithm
        phi = phi_function(s)
        R[k,k] = np.dot(v, phi)
        for i in range(k+1,q):
            R[k,i] += phi[i]
            for j in range(k):
                R[k,i] -= (phi[j]*R[j,i]) / R[j,j]
    return X, fX, delta_k, visited_pts, interpolation_solutions, expended_budget

def getTotalOrderBasisRecursion(highest_order: int, dimensions: int) -> np.ndarray:
    if dimensions == 1:
        I = np.zeros((1,1))
        I[0,0] = highest_order
    else:
        for j in range(0, highest_order + 1):
            U = getTotalOrderBasisRecursion(highest_order - j, dimensions - 1)
            rows, cols = U.shape
            T = np.zeros((rows, cols + 1) ) # allocate space!
            T[:,0] = j * np.ones((1, rows))
            T[:, 1: cols+1] = U
            if j == 0:
                I = T
            elif j >= 0:
                rows_I, cols_I = I.shape
                rows_T, cols_T = T.shape
                Itemp = np.zeros((rows_I + rows_T, cols_I))
                Itemp[0:rows_I,:] = I
                Itemp[rows_I : rows_I + rows_T, :] = T
                I = Itemp
            del T
    return I

def get_basis(orders: np.ndarray) -> np.ndarray: 
    dimensions = len(orders)
    highest_order = np.max(orders)
    # Check what the cardinality will be, stop if too large!
    L = int(math.factorial(highest_order+dimensions)/(math.factorial(highest_order)*math.factorial(dimensions)))
    # Check cardinality
    if L >= int(1e6):
        raise Exception('Cardinality %.1e is >= hard cardinality limit %.1e' %(L,int(1e6)))
    # Generate basis
    total_order = np.zeros((1, dimensions))
    for i in range(1, highest_order+1):
        R = getTotalOrderBasisRecursion(i, dimensions)
        total_order = np.vstack((total_order, R))
    return total_order 	

def get_phi_function_and_derivative(S_hat: np.ndarray, current_solution: Solution, delta_k: float, U: np.ndarray) -> tuple[callable, callable, float]:
    d = U.shape[1]
    q = len(index_set(2, d).astype(int))
    s_old = np.asarray(current_solution.x).ravel()   # 1-D shape (d,)

    total_order_index_set = get_basis(np.tile([2], q))[:, range(d-1, -1, -1)]

    if S_hat.size > 0:
        # S_hat rows are points; subtract s_old (ravel) and project via U
        # Ensure shapes align: (n_points, d) - (d,) -> (n_points, d)
        delta_k = max(norm(np.dot(S_hat - s_old, U), axis=1))

    def phi_function(s: np.ndarray) -> np.ndarray:
        s = s.ravel()    # shape (d,)
        # (s - s_old) is 1-D (d,), dot with U.T -> (d,)
        u = np.dot(s - s_old, U) / delta_k
        # Make u a 2-D row vector for consistent processing
        u = np.atleast_2d(u)  # shape (1, d) for single point, (m,d) for multiple
        m = u.shape[0]

        phi = np.zeros((m, q))
        for k in range(q):
            # total_order_index_set[k,:] is length d
            exponents = total_order_index_set[k, :]
            # compute product across dimensions; use np.prod over axis=1
            # To avoid integer vs float issues, convert to float
            numerator = np.power(u, exponents)
            denom = np.array([factorial(int(e)) for e in exponents])
            # divide each column by factorials then take product across columns
            phi[:, k] = np.prod(numerator / denom, axis=1)
        # If user expects 1-D array for single point, return 1-D row to match R[0,:] assignment
        if phi.shape[0] == 1:
            return phi.flatten()   # shape (q,)
        return phi               # shape (m, q)

    def phi_function_deriv(s: np.ndarray) -> np.ndarray:
        s = s.ravel()   # shape (d,)
        u = np.dot(s - s_old, U) / delta_k   # shape (d,)
        # Build phi_deriv as (d, q) before projecting by U
        phi_deriv = np.zeros((d, q))
        for i in range(d):
            for k in range(1, q):   # note: k starts from 1 as in your original code
                exponent = total_order_index_set[k, i]
                if exponent != 0:
                    # build tmp to subtract 1 from the i-th exponent
                    tmp = np.zeros(d, dtype=int)
                    tmp[i] = 1
                    exps_minus_tmp = total_order_index_set[k, :] - tmp
                    # handle u as 1-D vector
                    # compute product over dimensions
                    # numerator: u ** exps_minus_tmp
                    numerator = np.prod(np.divide(np.power(u, exps_minus_tmp), [factorial(int(e)) for e in total_order_index_set[k, :]]))
                    phi_deriv[i, k] = exponent * numerator
        # divide by delta_k then multiply by U
        phi_deriv = phi_deriv / delta_k    # still shape (d, q)
        # return U * phi_deriv  -> dot(U, phi_deriv) gives (d, q)
        return np.dot(U, phi_deriv)

    return phi_function, phi_function_deriv, delta_k

def find_new_point(problem: Problem, current_solution: Solution, v: np.ndarray, phi_function: callable, phi_function_deriv: callable, delta_k:float) -> np.ndarray:
    #change bounds to be defined using the problem and delta_k
    s_old = np.array(current_solution.x).reshape(-1,1)	
    bounds_l = np.maximum(np.array(problem.lower_bounds).reshape(s_old.shape), s_old-delta_k)
    bounds_u = np.minimum(np.array(problem.upper_bounds).reshape(s_old.shape), s_old+delta_k)

    bounds = []
    for i in range(problem.dim):
        bounds.append((bounds_l[i], bounds_u[i])) 
    
    obj1 = lambda s: np.dot(v, phi_function(s))
    jac1 = lambda s: np.dot(phi_function_deriv(s), v)
    obj2 = lambda s: -np.dot(v, phi_function(s))
    jac2 = lambda s: -np.dot(phi_function_deriv(s), v)
    res1 = minimize(obj1, s_old, method='TNC', jac=jac1, \
            bounds=bounds, options={'disp': False})
    res2 = minimize(obj2, s_old, method='TNC', jac=jac2, \
            bounds=bounds, options={'disp': False})
    if abs(res1['fun']) > abs(res2['fun']):
        s = res1['x']
    else:
        s = res2['x']
    return s

def find_new_point_alternative(problem: Problem, current_solution: Solution, v: np.ndarray, phi_function: callable, S: np.ndarray, delta_k: float) -> np.ndarray:
    s_old = np.array(current_solution.x).reshape(-1,1)
    no_pts = int(0.5*problem.dim*(problem.dim+2))
    S_tmp = generate_set(problem, no_pts, current_solution, delta_k)
    Phi_S_hat = np.vstack([phi_function(S_tmp[i, :].reshape(-1, 1)) for i in range(S_tmp.shape[0])])
    M = np.absolute(Phi_S_hat @ v)
    indices = np.argsort(M)[::-1][:len(M)]
    for index in indices:
        s = S_tmp[index,:]
        if np.unique(np.vstack((S, s)), axis=0).shape[0] == S.shape[0]+1:
            return s
    return S_tmp[indices[0], :]

def remove_point_from_set(S: np.ndarray, f: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ind_current = np.where(norm(S-s.ravel(), axis=1, ord=np.inf) == 0.0)[0]
    S = np.delete(S, ind_current, 0)
    f = np.delete(f, ind_current, 0)
    return S, f

def remove_furthest_point(S: np.ndarray, f: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ind_distant = np.argmax(norm(S-s.ravel(), axis=1, ord=np.inf))
    S = np.delete(S, ind_distant, 0)
    f = np.delete(f, ind_distant, 0)
    return S, f

def full_index_set(n, d):
    """ A helper function for index_set.
    
    Parameters
    ----------
    n : int
        degree of polynomial
    d : int
        number of variables, dimension
    """
    if d == 1:
        I = np.array([[n]])
    else:
        II = full_index_set(n, d-1)
        m = II.shape[0]
        I = np.hstack((np.zeros((m, 1)), II))
        for i in range(1, n+1):
            II = full_index_set(n-i, d-1)
            m = II.shape[0]
            T = np.hstack((i*np.ones((m, 1)), II))
            I = np.vstack((I, T))
    return I

def index_set(n, d):
    """Enumerate multi-indices for a total degree of order `n` in `d` variables.
    
    Parameters
    ----------
    n : int
        degree of polynomial
    d : int
        number of variables, dimension
    Returns
    -------
    I : ndarray
        multi-indices ordered as columns
    """
    I = np.zeros((1, d), dtype = np.integer)
    for i in range(1, n+1):
        II = full_index_set(i, d)
        I = np.vstack((I, II))
    return I[:,::-1].astype(int)


def main() : 
    problem = instantiate_problem("EXAMPLE-1")
    solution =  Solution((0.0,0.0), problem)
    U=np.array([[0.7071],[-0.7071]])
    n,d = U.shape
    visited_pts_list = [solution]
    delta = 1.0
    expended_budget = 0
    fn = lambda x : np.linalg.norm(x) ** 2

    X = generate_set(problem, 2*n+1, solution, delta)
    interpolation_sols = [Solution(tuple(x), problem) for x in X]
    fX = np.array([fn(X[i,:]) for i in range(X.shape[0])]).reshape(-1,1)
    expended_budget += X.shape[0]
    print(f"Initial Generated Design points:\n{X}")
    print(f"Initial Function values:\n{fX}")
    current_f = fX[0]
    print(f"function value of initial solution: {current_f}")
    print("==================================================")
    # Go through iterations, and after each iteration, update t
    for iter in range(1,6) :
        print(f"Iteration {iter}")

        X, fX, interpolation_sols, visited_pts_list, delta, expended_budget = improve_geometry(iter, problem, solution, fX[0], delta, U, visited_pts_list, X, fX, interpolation_sols, expended_budget)
        expended_budget += X.shape[0]
        print(f"Improved Design points:\n{X}")
        print(f"Improved Function values:\n{fX}")

        delta = 1.5*delta
        visited_pts_list.append(solution)
        #create a new solution by choosing a point from the design set X
        new_x = tuple(X[-1,:])
        solution = Solution(new_x, problem)
        current_f = np.array([fn(np.array(new_x))])

        print(f'New Solution for iteration {iter}: ')
        print(solution.x)
        print(f'Function value of new solution: {current_f}')
        print("--------------------------------------------------")

    print(f'Final expended Budget: {expended_budget}')

if __name__ == "__main__":
    main()