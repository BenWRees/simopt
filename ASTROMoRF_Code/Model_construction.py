"""The following functions define model Construction for ASTROMoRF."""

import logging
import sys
from pathlib import Path

sys.path.append(str(Path(sys.modules[__name__].__file__).parent.parent.resolve()))  #


import numpy as np
import scipy
from numpy.linalg import norm, pinv, qr
from numpy.polynomial.hermite_e import hermeder, hermevander
from sklearn.metrics import r2_score

from simopt.base import (
    Problem,
    Solution,
)
from simopt.experiment_base import instantiate_problem


def get_scale(
    dirn: list[float], delta: float, lower: np.ndarray, upper: np.ndarray
) -> float:
    """Sets the max distance for each direction in the current trust region."""
    scale = delta
    for j in range(len(dirn)):
        if dirn[j] < 0.0:
            scale = min(scale, lower[j] / dirn[j])
        elif dirn[j] > 0.0:
            scale = min(scale, upper[j] / dirn[j])
    return scale


def random_directions(
    problem: Problem,
    num_pnts: int,
    lower: np.ndarray,
    upper: np.ndarray,
    delta_k: float,
) -> np.ndarray:
    """Generates orthogonal, random directions."""
    n = problem.dim
    direcs = np.zeros((n, max(2 * n + 1, num_pnts)))
    idx_l = lower == 0
    idx_u = upper == 0
    active = np.logical_or(idx_l, idx_u)
    inactive = np.logical_not(active)
    nactive = np.sum(active)
    ninactive = n - nactive
    if ninactive > 0:
        A = np.random.normal(size=(ninactive, ninactive))  # noqa: N806
        Qred = qr(A)[0]  # noqa: N806
        Q = np.zeros((n, ninactive))  # noqa: N806
        Q[inactive, :] = Qred
        for i in range(ninactive):
            scale = get_scale(Q[:, i], delta_k, lower, upper)
            direcs[:, i] = scale * Q[:, i]
            scale = get_scale(-Q[:, i], delta_k, lower, upper)
            direcs[:, n + i] = -scale * Q[:, i]
    idx_active = np.where(active)[0]
    for i in range(nactive):
        idx = idx_active[i]
        direcs[idx, ninactive + i] = 1.0 if idx_l[idx] else -1.0
        direcs[:, ninactive + i] = (
            get_scale(direcs[:, ninactive + i], delta_k, lower, upper)
            * direcs[:, ninactive + i]
        )
        sign = 1.0 if idx_l[idx] else -1.0
        if upper[idx] - lower[idx] > delta_k:
            direcs[idx, n + ninactive + i] = 2.0 * sign * delta_k
        else:
            direcs[idx, n + ninactive + i] = 0.5 * sign * (upper[idx] - lower[idx])
        direcs[:, n + ninactive + i] = (
            get_scale(direcs[:, n + ninactive + i], 1.0, lower, upper)
            * direcs[:, n + ninactive + i]
        )
    for i in range(num_pnts - 2 * n):
        dirn = np.random.normal(size=(n,))
        for j in range(nactive):
            idx = idx_active[j]
            sign = 1.0 if idx_l[idx] else -1.0
            if dirn[idx] * sign < 0.0:
                dirn[idx] *= -1.0
        dirn = dirn / norm(dirn)
        scale = get_scale(dirn, delta_k, lower, upper)
        direcs[:, 2 * n + i] = dirn * scale
    return np.vstack((np.zeros(n), direcs[:, :num_pnts].T))  # shape (num_pnts, n)


def generate_set(  # noqa: D103
    problem: Problem, num: int, current_solution: Solution, delta_k: float
) -> np.ndarray:
    s_old = np.array(current_solution.x).reshape(-1, 1)

    bounds_l = np.maximum(
        np.array(problem.lower_bounds).reshape(s_old.shape), s_old - delta_k
    )
    bounds_u = np.minimum(
        np.array(problem.upper_bounds).reshape(s_old.shape), s_old + delta_k
    )

    direcs = random_directions(
        problem,
        num,
        (bounds_l - s_old).flatten(),
        (bounds_u - s_old).flatten(),
        delta_k,
    )

    S = np.zeros((num, problem.dim))  # noqa: N806
    S[0, :] = s_old.flatten()
    bounds_l_flat = bounds_l.flatten()
    bounds_u_flat = bounds_u.flatten()
    s_old_flat = s_old.flatten()
    for i in range(1, num):
        S[i, :] = s_old_flat + np.minimum(
            np.maximum(bounds_l_flat - s_old_flat, direcs[i, :]),
            bounds_u_flat - s_old_flat,
        )

    return S  # shape (num, n)


def clamp_with_epsilon(
    val: float, lower_bound: float, upper_bound: float, epsilon: float = 0.01
) -> float:
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


def column_vectors_U(index: int, U: np.ndarray) -> np.ndarray:  # noqa: N802, N803
    """Get the index column vector of U. The column vectors are orthonormal basis.

    vectors that span the active subspace.

    Args:
            problem (Problem): The SO problem
            index (int): The index of the column vector
            U (np.ndarray): The active subspace matrix

    Returns:
            np.ndarray: The n-dimensional column vector at the given index
    """
    return U[:, index].reshape(-1, 1)


def compute_adaptive_interpolation_radius_fraction(  # noqa: D417
    problem: Problem,
    current_solution: Solution,
    U: np.ndarray,  # noqa: N803
    delta: float,
) -> float:
    """Compute interpolation radius as a fraction of trust-region radius.

    KEY PRINCIPLE: Interpolation radius should grow MORE SLOWLY than TR radius
    to maintain model quality as TR expands.

    PROBLEM: When TR expands (e.g., 0.5 → 1.0 after success):
    - Linear scaling: interpolation radius also doubles (0.4 → 0.8)
    - Model becomes less accurate in center where candidates are generated
    - Leads to high prediction errors after TR expansion

    SOLUTION: Use sublinear scaling - cap the effective interpolation radius
    - For small TR: use ~0.8δ (good coverage)
    - For large TR: use smaller fraction to keep points tighter (better accuracy)
    - This prevents model degradation after TR expansion

    Args:
            problem (Problem): The simulation problem
            U (np.ndarray): The (n,d) active subspace matrix

    Returns:
            float: Fraction of trust-region radius to use
    """
    d = U.shape[1]

    # Compute reference scale: average distance between bounds
    lower = np.array(problem.lower_bounds).reshape(-1, 1)
    upper = np.array(problem.upper_bounds).reshape(-1, 1)
    domain_scale = np.mean(upper - lower)

    # Base fraction depends on TR size relative to domain
    tr_relative_size = delta / domain_scale

    # Use adaptive scaling based on TR size:
    # - Small TR (< 10% of domain): use 0.75-0.80 (broader coverage)
    # - Medium TR (10-30% of domain): use 0.65-0.75 (balanced)
    # - Large TR (> 30% of domain): use 0.55-0.65 (tighter for accuracy)
    if tr_relative_size < 0.1:
        base_fraction = 0.80
    elif tr_relative_size < 0.2:
        base_fraction = 0.75
    elif tr_relative_size < 0.3:
        base_fraction = 0.70
    else:
        # For very large TR, use even smaller fraction to maintain accuracy
        base_fraction = max(0.75, 0.80 - 1.0 * tr_relative_size)

    # Adjust for subspace dimension
    if d >= 8:
        base_fraction = min(base_fraction + 0.05, 0.85)  # Slightly wider for high-D

    # Case: Trust region hitting domain bounds
    x_k = np.array(current_solution.x).reshape(-1, 1)
    dist_to_lower = x_k - lower
    dist_to_upper = upper - x_k
    min_clearance = min(np.min(dist_to_lower), np.min(dist_to_upper))

    # If trust region would place points outside bounds, reduce radius
    if min_clearance < 0.9 * delta:
        safe_fraction = 0.9 * min_clearance / delta
        base_fraction = min(base_fraction, max(safe_fraction, 0.5))

    return base_fraction


def construct_interpolation_set(
    current_solution: Solution,
    problem: Problem,
    U: np.ndarray,  # noqa: N803
    delta: float,
    k: int,  # noqa: ARG001
    visited_pts: list[Solution],  # noqa: ARG001
) -> tuple[np.ndarray, list[int]]:
    """Constructs an interpolation set of 2d+1 points within the trust region defined by.  # noqa: E501.

    delta around the current solution.

            Is the no_reuse construction
    Args:
            current_solution (Solution): The current solution of the solver
            problem (Problem): The SO problem with dimension n
            U (np.ndarray): The active subspace of shape (n,d)
            delta (float): The current trust-region radius
            k (int): The current iteration of the solver
            visited_pts (list[Solution]): The solutions already visited by the solver.

    Returns:
            tuple[np.ndarray, list[int]]:
                    - The interpolation set of design points of shape (2d+1, n)
                    - The indices of the interpolation points in the visited_pts list
    """
    x_k = np.array(current_solution.x).reshape(-1, 1)
    d = U.shape[1]
    Y = [x_k]  # noqa: N806
    lower_bounds = problem.lower_bounds
    upper_bounds = problem.upper_bounds

    # Adaptively compute interpolation radius based on problem characteristics
    compute_adaptive_interpolation_radius_fraction(problem, current_solution, U, delta)
    interpolation_radius = delta

    for i in range(0, d):
        plus = Y[0] + interpolation_radius * column_vectors_U(i, U)
        minus = Y[0] - interpolation_radius * column_vectors_U(i, U)

        plus = plus.flatten().tolist()
        minus = minus.flatten().tolist()

        if sum(x_k) != 0:
            minus = [
                clamp_with_epsilon(val, lower_bounds[j], upper_bounds[j])
                for j, val in enumerate(minus)
            ]
            plus = [
                clamp_with_epsilon(val, lower_bounds[j], upper_bounds[j])
                for j, val in enumerate(plus)
            ]

            minus = np.array(minus).reshape(-1, 1)
            plus = np.array(plus).reshape(-1, 1)

        Y.append(plus)
        Y.append(minus)
    Y = np.vstack([v.ravel() for v in Y])  # noqa: N806
    return Y, 0


def construct_model(  # noqa: D417
    k: int,
    d: int,
    degree: int,
    problem: Problem,
    current_solution: Solution,
    delta: float,
    expended_budget: int,
    visited_pts: list[Solution],
) -> tuple[
    Solution,
    float,
    callable,
    callable,
    np.ndarray,
    list[float],
    int,
    list[Solution],
    list[Solution],
]:
    """Builds a local approximation of the response surface within a trust region.

    (defined as ||x-x_k||<=delta).

            The method fit recovers the local approximation given a design set of 2d+1
            design points and a corresponding active subspace U of shape (n,d)
            That projects the n-dimensional design points to a d-dimensional subspace.

    Args:
            k (int): The solver's current iteration
            problem (Problem): The SO problem with dimension n
            current_solution (Solution): The solvers incumbent solution
            delta (float): The current trust-region radius
            expended_budget (int): The current expended budget of the solver
            visited_pts (list[Solution]): The solutions already visited by the solver

    Returns:
            tuple[ Solution, float, callable, np.ndarray, list[float], int,
            list[Solution], list[Solution] ]:
                    - The current solution of the SO solver
                    - The updated trust-region radius after fitting
                    - The local model as a function that takes a numpy vector of shape
                    (n,1) and returns a float (this is given as model = lambda x :
                    self.model_evaluate(x,U))
                    - The final computed active subspace of the iteration of shape (n,d)
                    - A list of the function estimates of the objective function at each
                    of the final design points
                    - The simulation budget expended so far by the solver
                    - The list of solutions of the final design points
                    - The design points previously visited by the solver
    """
    init_S_full = generate_set(problem, d, current_solution, delta)  # noqa: N806
    U, _ = np.linalg.qr(init_S_full.T)  # noqa: N806

    X, _f_index = construct_interpolation_set(  # noqa: N806
        current_solution, problem, U, delta, k, visited_pts
    )

    # fX, current_solution, interpolation_solutions, visited_points_list, expended_budget = evaluate_interpolation_points(k, problem, current_solution, expended_budget, f_index, X, visited_points_list, delta_k)
    fX = np.vstack([np.linalg.norm(x) ** 2 for x in X])  # noqa: N806
    interpolation_solutions = [Solution(tuple(x), problem) for x in X]
    f_x = fX[0, 0]  # -1 * problem.minmax[0] * current_solution.objectives_mean

    fval = fX.flatten().tolist()

    U, model, model_grad, X, fX, _interpolation_sols, visited_pts, delta = fit(  # noqa: N806
        k,
        degree,
        problem,
        X,
        fX,
        current_solution,
        f_x,
        delta,
        interpolation_solutions,
        visited_pts,
        U,
    )

    fval = fX.flatten().tolist()

    return (
        current_solution,
        delta,
        model,
        model_grad,
        U,
        fval,
        expended_budget,
        interpolation_solutions,
        visited_pts,
    )


def model_evaluate(  # noqa: D417
    x: np.ndarray,
    coef: np.ndarray,
    U: np.ndarray,  # noqa: N803
    degree: int,
) -> float:
    """Evaluates the local approximated model at a given design point.

    Args:
            x (np.ndarray): Design point to evaluate of shape (1,n)
            coef (np.ndarray): The coefficients of the local model of shape (q,1)
            U (np.ndarray): The active subspace matrix of shape (n,d)

    Returns:
            float: The evaluation of the model at x, given as m(U^Tx)
    """
    if len(x.shape) != 2 or x.shape[1] != 1:
        x = x.reshape(1, -1)

    if x.shape[0] == U.shape[0]:
        # project x to active subspace
        x = U.T @ x  # (d,1)

    if len(coef.shape) != 2:
        coef = coef.reshape(-1, 1)

    # print(f'shape of x: {x.shape}')
    # build vandermonde matrix of shape (1,q)
    V_matrix = V(x.T, degree)  # noqa: N806

    # print(f'shape of V_matrix: {V_matrix.shape}, shape of coef: {coef.shape}')

    # find evaluation:
    res = V_matrix @ coef
    return res.item()


def fit(
    k: int,  # noqa: ARG001
    degree: int,
    problem: Problem,
    X: np.ndarray,  # noqa: N803
    fX: np.ndarray,  # noqa: N803
    current_solution: Solution,  # noqa: ARG001
    f_x: float,  # noqa: ARG001
    delta: float,
    interpolation_solutions: list[Solution],
    visited_pts: list[Solution],
    U0: np.ndarray,  # noqa: N803
) -> tuple[
    np.ndarray,
    callable,
    callable,
    np.ndarray,
    np.ndarray,
    list[Solution],
    list[Solution],
    float,
]:
    """Fits the design set and evaluated points to a local model with recovered active.

    subspace.


            It undergoes a loop until the active subspace converges.
            First, it improves the design set X and constructing an interpolation model
            until it can ensure the criticality step is satisfied.
            Second, after fixing the model coefficients, it updates the active subspace
            through a variable projection scheme
    Args:
            problem (Problem): The SO problem with dimension n
            X (np.ndarray): design set of shape (M,n)
            fX (np.ndarray): corresponding function estimates of design points of shape
            (M,1)
            interpolation_solutions (list[Solution]): A list of the design points in the
            U0 (np.ndarray): The initial estimate for the active subspace of shape
            (n,d).

    Returns:
            tuple[np.ndarray, callable, callable, np.ndarray, np.ndarray,
            list[Solution]]:
                    - The final computed active subspace of the iteration of shape (n,d)
                    - The local model as a function that takes a numpy vector of shape
                    (n,1) and returns a float (this is given as model = lambda x :
                    self.model_evaluate(x,coef, U))
                    - The local model gradient as a function that takes a numpy vector
                    of shape (n,1) and returns a float (this is given as model_grad =
                    lambda x : self.model_evaluate(x,U))
                    - The design set after going through fitting of shape (M,n)
                    - The function estimates of the objective function at each of the
                    final design points of shape (M,1)
                    - The design points as solution objects
    """
    # Algorithmic Parameters
    beta = 10
    _n, _d = U0.shape

    # Orthogonalize just to make sure the starting value satisfies constraints
    U0, R = np.linalg.qr(U0, mode="reduced")  # noqa: N806
    U0 = np.dot(U0, np.diag(np.sign(np.diag(R))))  # noqa: N806

    prev_U = np.zeros(U0.shape)  # noqa: N806
    U = np.copy(U0)  # noqa: N806
    model_delta = float(delta)

    if degree == 1 and U.shape[1] == 1:
        V_matrix = np.hstack((np.ones((X.shape[0], 1)), X))  # (M, n+1)  # noqa: N806
        fn_coef = pinv(V_matrix) @ fX  # (n+1,1)
        fn_grad = fn_coef[1:, :].reshape(-1, 1)  # (n,1)
        U = fn_grad / norm(fn_grad)  # noqa: N806

    else:
        i = 0
        while True:  # not self.converged_subspace_check(prev_U, U) :
            subspace_tol = min(1e-2, 0.5 * max(model_delta, 1e-8))
            if converged_subspace_check(prev_U, U, subspace_tol):
                break
            # * Construct model and Criticality step
            (
                coef,
                model_delta,
                X,  # noqa: N806
                fX,  # noqa: N806
                interpolation_solutions,
            ) = criticality_check(
                problem, X, fX, U, delta, interpolation_solutions, degree
            )

            # set the old U and update
            prev_U = np.copy(U)  # noqa: N806
            U = fit_varpro(problem, X, fX, U, degree)  # noqa: N806
            i += 1

    coef = fit_coef(problem, X, fX, U, degree)

    # final fitting of the coefficients and rotating the final U
    U = rotate_U(problem, X, fX, coef, U, degree)  # noqa: N806
    coef = fit_coef(problem, X, fX, U, degree)

    def model(x):  # noqa: ANN001, ANN202
        return model_evaluate(x, coef, U, degree)  # returns a float

    def model_grad(x):  # noqa: ANN001, ANN202
        return grad(x, coef, U, degree)  # returns np.ndarray of shape (d,1)

    if delta != model_delta:
        delta = min(max(delta, beta * norm(grad(X, coef, U, degree))), model_delta)
    return U, model, model_grad, X, fX, interpolation_solutions, visited_pts, delta


def criticality_check(  # noqa: D417
    problem: Problem,
    X: np.ndarray,  # noqa: N803
    fX: np.ndarray,  # noqa: N803
    U: np.ndarray,  # noqa: N803
    delta: float,
    interpolation_solutions: list[Solution],
    degree: int,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, list[Solution]]:
    """Performs the criticality step of the trust-region method.

            It fits a local model to the design set and checks whether the criticality
            condition is satisfied.

    Args:
            problem (Problem): The SO Problem
            X (np.ndarray): The design set of shape (M,n)
            fX (np.ndarray): The corresponding function estimates of shape (M,1)
            U (np.ndarray): The current active subspace matrix
            interpolation_solutions (list[Solution]): The list of solutions in the
            interpolation set

    Returns:
            tuple[np.ndarray, float, np.ndarray, np.ndarray, list[Solution]]:
                    - The model coefficients of shape (q,1)
                    - The trust-region radius after criticality check
                    - The design set after criticality check of shape (M,n)
                    - The function estimates of the objective function at each of the
                    final design points of shape (M,1)
                    - The design points as solution objects

    RECENT CHANGES:
            > Removed updating of model_delta within the loop to ensure convergence
    """
    tol: float = 1e-6
    kappa_f: float = 10.0
    kappa_g: float = 10.0
    coef: np.ndarray | None = None
    mu: float = 1000.0

    model_delta = float(delta)

    # * Construct model and Criticality step
    while True:
        coef = fit_coef(problem, X, fX, U, degree)
        grad_vec = grad(X, coef, U, degree) @ U.T  # (M,n)
        gnorm = norm(grad_vec)

        if gnorm <= tol:
            if not fully_linear_test(
                X, fX, coef, U, model_delta, kappa_f, kappa_g, degree
            ):
                # X, fX, interpolation_solutions = geo.improve_geometry(problem, model_delta, U, X, fX, interpolation_solutions)
                # model_delta = delta * w**fitting_iter
                # fitting_iter += 1
                pass

        elif model_delta > max(mu * gnorm, 1e-12):
            # model_delta = min(model_delta, max(delta_min, mu * gnorm, 1e-12))
            # X, fX, interpolation_solutions = geo.improve_geometry(problem, model_delta, U, X, fX, interpolation_solutions)
            # fitting_iter += 1
            pass
        else:
            break

    return coef, model_delta, X, fX, interpolation_solutions


def converged_subspace_check(prev_U: np.ndarray, U: np.ndarray, tol: float) -> bool:  # noqa: N803
    """Check whether the active subspace has converged by computing the subspace.

            distance between previous and current subspace estimates.

    Args:
            prev_U (np.ndarray): Active subspace matrix from previous iteration (shape
            (n,d))
            U (np.ndarray): Active subspace from current iteration (shape (n,d))
            tol (float, optional): Convergence tolerance for subspace distance.

    Returns:
            bool: Returns True is subspace change is below tolerance and False otherwise
    """
    if np.all(prev_U == 0):
        return False

    C = prev_U.T @ U  # shape (d, d)  # noqa: N806
    # Singular values of C are cos(theta_i)
    try:
        sigma = np.linalg.svd(C, compute_uv=False)
    except Exception:
        try:
            CtC = C.T @ C  # noqa: N806
            w = np.linalg.eigvalsh(CtC)
            sigma = np.sqrt(np.maximum(w, 0.0))
        except Exception:
            return False  # safest fail-closed behavior

    sigma = np.clip(sigma, -1.0, 1.0)
    # Compute principal angles and distance
    sin_theta = np.sqrt(1.0 - sigma**2)
    subspace_dist = float(np.max(sin_theta))  # operator norm of projector difference

    return subspace_dist <= tol


def fully_linear_test(  # noqa: D417
    X: np.ndarray,  # noqa: N803
    fX: np.ndarray,  # noqa: N803
    coef: np.ndarray,
    U: np.ndarray,  # noqa: N803
    delta: float,
    kappa_f: float,
    kappa_g: float,
    degree: int,
) -> bool:
    """Check whether a model is fully linear in a trust region, using function residuals.  # noqa: E501.

    and model gradient consistency.

    Args:
            X (np.ndarray): Design set of points (shape (M,n))
            fX (np.ndarray): Corresponding function estimation of design points (shape
            (M,1))
            coef (np.ndarray): Model coefficients (shape (q,1))
            U (np.ndarray): Active subspace matrix (shape (n,d))
            kappa_f (float): Tolerance of zeroth-order fully-linear bound
            kappa_g (float): Tolerance of first-order fully-linear bound.

    Returns:
            bool: True if the model is fully-linear and False otherwise.
    """
    M, _n = X.shape  # noqa: N806

    # --- 1. Value-based condition ---
    mX = np.array(  # noqa: N806
        [model_evaluate(U.T @ np.array(x).reshape(-1, 1), coef, U, degree) for x in X]
    ).reshape(-1, 1)
    residuals = np.abs(fX - mX)
    value_condition = np.max(residuals) <= kappa_f * delta**2

    # --- 2. Gradient consistency condition ---
    m_grads = grad(X, coef, U, degree)  # shape (M, d)
    consistent = True

    for i in range(M):
        for j in range(i + 1, M):
            dx = X[i, :] - X[j, :]  # shape (n,)
            dm = mX[i, 0] - mX[j, 0]
            g_j = (U @ m_grads[j, :].reshape(-1, 1)).flatten()  # shape (n,)
            linearized_diff = np.dot(g_j, dx)
            model_error = np.abs(dm - linearized_diff)
            if model_error > kappa_g * np.linalg.norm(dx) ** 2:
                consistent = False
                break
        if not consistent:
            break

    # --- Fully linear if both conditions hold ---
    return bool(value_condition and consistent)


def fit_coef(
    problem: Problem,
    X: np.ndarray,  # noqa: N803
    fX: np.ndarray,  # noqa: N803
    U: np.ndarray,  # noqa: N803
    degree: int,
) -> np.ndarray:
    """Finds the coefficients of the interpolation model by solving the system of.

    equations:

                            V(U^TX)coeff = fX
    Args:
            X (np.ndarray): The design set of design points of shape (M,n)
            fX (np.ndarray): corresponding function estimates of design points of shape
            (M,1)
            U (np.ndarray): The active subspace of shape (n,d)
            delta (float): radius of the current trust-region.

    Returns:
            np.ndarray: A list of the coefficients of shape (q,1)
    """
    #! Handle NaN in fX and X - this is a quick fix because something is happening with the incumbent solution evaluation suddenly disappearing
    lambda_min: int = 5
    if np.isnan(fX).any():
        # If there are NaN values in fX, find the corresponding row in X, create a solution, simulate it and update fX
        nan_indices = np.where(np.isnan(fX))[0]
        for idx in nan_indices:
            x_nan = X[idx, :].ravel()
            solution_nan = Solution(tuple(x_nan), problem)
            solution_nan.attach_rngs(problem.rng_list)
            problem.simulate(solution_nan, int(lambda_min))
            fx_nan = -1 * problem.minmax[0] * solution_nan.objectives_mean.item()
            fX[idx, 0] = fx_nan

    Y = X @ U  # noqa: N806
    V_matrix = V(Y, degree)  # shape (M,q)  # noqa: N806
    coef = pinv(V_matrix) @ fX  # (q,1)

    # if np.isnan(coef).any() or np.isinf(coef).any() :
    # 	coef = scipy.linalg.lstsq(V_matrix, fX)[0] #(q,1)

    if len(coef.shape) != 2:
        coef = coef.reshape(-1, 1)

    return coef


def grassmann_trajectory(U: np.ndarray, Delta: np.ndarray, t: float) -> np.ndarray:  # noqa: N803
    """Calculates the geodesic along the Grassmann manifold.

    Args:
            U (np.ndarray): The active subspace matrix of shape (n,d)
            Delta (np.ndarray): The search direction along the Grassmann manifold with
            shape (n,d)
            t (float): Independent parameter in the line equation takes values between
            (0,infty) and is selected to ensure convergence.

    Returns:
            np.ndarray: The new candidate for the active subspace based on the step made
            of shape (n,d)
    """
    try:
        Y, sig, ZT = scipy.linalg.svd(Delta, full_matrices=False, lapack_driver="gesvd")  # noqa: N806
    except Exception:
        Y, sig, ZT = scipy.linalg.svd(Delta, full_matrices=False)  # noqa: N806

    UZ = np.dot(U, ZT.T)  # noqa: N806
    U_new = np.dot(UZ, np.diag(np.cos(sig * t))) + np.dot(Y, np.diag(np.sin(sig * t)))  # noqa: N806

    # Correct the new step U by ensuring it is orthonormal with consistent sign on the elements
    U_new, R = np.linalg.qr(U_new, mode="reduced")  # noqa: N806
    return np.dot(U_new, np.diag(np.sign(np.diag(R))))


def residual(
    problem: Problem,
    X: np.ndarray,  # noqa: N803
    fX: np.ndarray,  # noqa: N803
    U: np.ndarray,  # noqa: N803
    degree: int,
) -> np.ndarray:
    """Construct the Residual of the model fitting, such that.

            r = fX - V(U^TX)coeff
    Args:
            X (np.ndarray): The design set of design points of shape (M,n)
            fX (np.ndarray): The corresponding function estimates of design points of
            shape (M,1)
            U (np.ndarray): The active subspace of shape (n,d)
            delta (float): radius of the current trust-region.

    Returns:
            np.ndarray: The residual error for each design point on the local model of
            shape (M,1)
    """
    c = fit_coef(problem, X, fX, U, degree)  # shape(q,1)
    model_fX = np.array(  # noqa: N806
        [model_evaluate(U.T @ np.array(x).reshape(-1, 1), c, U, degree) for x in X]
    ).reshape(-1, 1)  # A list of length M with float elements
    return fX - model_fX


def jacobian(  # noqa: D417
    problem: Problem,
    X: np.ndarray,  # noqa: N803
    fX: np.ndarray,  # noqa: N803
    U: np.ndarray,  # noqa: N803
    degree: int,
) -> np.ndarray:
    """Constructs the Jacobian of the residual with respect to the active subspace.

    Args:
            X (np.ndarray): The design set of design points of shape (M,n)
            fX (np.ndarray): The corresponding function estimates of design points of
            shape (M,1)
            U (np.ndarray): The active subspace of shape (n,d)
            delta (float): radius of the current trust-region.

    Returns:
            np.ndarray: A tensor of shape (M,n,d) where each element is the partial
            derivative of the i-th residual component with respect to the (j,k)th entry
            of the active subspace
    """
    # FIRST ENSURE THAT THE ARGUMENTS HAVE DIMENSIONS THAT MATCH
    assert X.shape[1] == U.shape[0], (
        "X should have columns equal to the number of rows in U"
    )
    assert X.shape[0] == fX.shape[0], (
        "The number of samples in the design set X should match the number of function estimations in fX"
    )
    assert fX.shape[1] == 1, (
        "The function estimates of the design set should be a column vector"
    )
    d = U.shape[1]
    # get dimensions
    M, n = X.shape  # noqa: N806

    # find the residual
    Y = X @ U  # noqa: N806

    c = fit_coef(problem, X, fX, U, degree)  # shape(q,1)
    r = residual(problem, X, fX, U, degree)  # (M,1)

    #! FROM HERE THE FUNCTION NEEDS CHECKING
    # find the vandermonde matrix and derivative of the vandermonde matrix of the projected design set
    V_matrix = V(Y, degree)  # shape (M,q)  # noqa: N806
    DV_matrix = DV(Y, degree)  # shape (M,q,n)  # noqa: N806

    M, q = V_matrix.shape  # noqa: N806

    try:
        Y, s, ZT = scipy.linalg.svd(V_matrix, full_matrices=False)  # noqa: N806
    except Exception as e:
        logging.warning("SVD failed in Jacobian computation with error:", e)
        #! Need fallback for SVD failure
    # s = np.array([np.inf if x == 0.0 else x for x in s])
    with np.errstate(divide="ignore", invalid="ignore"):
        D = np.diag(1.0 / s)  # noqa: N806
        D[np.isinf(D)] = 0  # convert inf to 0 if desired

    J1 = np.zeros((M, n, d))  # noqa: N806
    J2 = np.zeros((q, n, d))  # noqa: N806

    # populate the Jacobian
    for k in range(d):
        for j in range(n):
            # This is the derivative of U
            DVDU_k = X[:, j, None] * DV_matrix[:, :, k]  # shape (M,q)  # noqa: N806

            # first term in the Jacobian
            J1[:, j, k] = DVDU_k.dot(c).flatten()  # shape (M,)

            # second term of the Jacobian before V(U)^-
            J2[:, j, k] = DVDU_k.T.dot(r).flatten()  # shape of (M,)

    # project J1 against the range of V
    J1 -= np.tensordot(  # noqa: N806
        Y, np.tensordot(Y.T, J1, axes=(1, 0)), axes=(1, 0)
    )  # shape: (M,)

    # apply pseudo-inverse via SVD components
    J2_projected = np.tensordot(  # noqa: N806
        D, np.tensordot(ZT, J2, axes=(1, 0)), axes=(1, 0)
    )  # shape: (q, n, d)

    # combine terms to get full Jacobian
    return -(J1 + np.tensordot(Y, J2_projected, axes=(1, 0)))  # shape: (M, n, d)


def fit_varpro(  # noqa: D417
    problem: Problem,
    X: np.ndarray,  # noqa: N803
    fX: np.ndarray,  # noqa: N803
    U: np.ndarray,  # noqa: N803
    degree: int,
) -> np.ndarray:
    """Runs a Gauss-Newton.

    Args:
            X (np.ndarray): design set of shape (M,n)
            fX (np.ndarray): corresponding function estimates of design points of shape
            (M,1)
            U (np.ndarray): The active subspace of shape (n,d)
            delta (float): radius of the current trust-region


    Returns:
            np.ndarray: The active subspace of shape (n,d)
    """

    def gn_solver(Jac: np.ndarray, residual: np.ndarray) -> np.ndarray:  # noqa: N803
        """An anonymous function to compute the Gauss-Newton step to find a descent.

        direction.

        Args:
                Jac (np.ndarray): The Jacobian of the residual with respect to the
                active subspace. It has shape (M,n,d)
                residual (np.ndarray): The residual of the current model approximation
                with shape (M,1).

        Returns:
                np.ndarray: A vectorised form of the descent direction with shape (nd,).
                The full descent direction has shape (n,d)
        """
        # Handle edge cases where residual or Jacobian are zero
        if np.all(residual == 0) and np.all(Jac == 0):
            return np.zeros(Jac.shape[1] * Jac.shape[2])

        if np.all(Jac == 0):
            raise ValueError("Jacobian is zero, cannot compute Gauss-Newton step.")

        if np.all(residual == 0):
            return np.zeros(Jac.shape[1] * Jac.shape[2])

        _M, _n, _d = Jac.shape  # noqa: N806
        Jac_vec = Jac.reshape(  # noqa: N806
            X.shape[0], -1
        )  # reshapes (M,n,d) to (M,nd)

        # compute short form SVD
        try:
            Y, sig, ZT = scipy.linalg.svd(  # noqa: N806
                Jac_vec, full_matrices=False, lapack_driver="gesvd"
            )  # Y has shape (M,M), sig has shape (M,nd), and ZT has shape (nd,nd)
        except Exception as e:
            logging.warning("SVD failed with error:", e)
            Y, sig, ZT = scipy.linalg.svd(Jac_vec, full_matrices=False)  # noqa: N806
        # Find descent direction
        # Use more robust tolerance - either machine epsilon scaled by max singular value
        # or absolute tolerance to handle cases where all singular values are small
        tol_relative = np.max(sig) * np.finfo(float).eps
        tol_absolute = 1e-12  # Absolute tolerance for very small singular values
        tol = max(tol_relative, tol_absolute)

        # Count and report ill-conditioning
        np.sum(sig < tol)
        (np.max(sig) / np.min(sig[sig > 0]) if np.any(sig > 0) else np.inf)

        # Compute safe inverse of singular values
        s_inv = np.where(sig > tol, 1.0 / sig, 0.0)

        # Compute Y^T r
        YTr = (Y.T @ residual).flatten()  # shape (M,)  # noqa: N806

        # Compute Delta_vec using safe inverse
        return -ZT.T @ (s_inv * YTr)  # shape (n*d,)

    def jacobian_variable_U(U):  # noqa: ANN001, ANN202, N802, N803
        return jacobian(problem, X, fX, U, degree)

    def residual_variable_U(U):  # noqa: ANN001, ANN202, N802, N803
        return residual(problem, X, fX, U, degree)

    return gauss_newton_solver(residual_variable_U, jacobian_variable_U, U, gn_solver)


def gauss_newton_solver(
    residual: callable,
    jacobian: callable,
    U: np.ndarray,  # noqa: N803
    gn_solver: callable,
) -> np.ndarray:
    """Solves the Gauss_newton problem on the Grassmann manifold:.

                    vec(Delta) = -vec(Jac(U))^{+}r(U).

    Args:
            residual (callable): Function that takes the active subspace U of shape
            (n,d) and calculates the residual of the predicted model under a fixed
            design set. Returns a matrix of shape (M,1)
            jacobian (callable): Function that takes the active subspace U of shape
            (n,d) and calculates the Jacobian of the residual with respect to U. Returns
            a matrix of shape (M,n,d)
            U (np.ndarray): The subspace matrix with shape (n,d)
            gn_solver (callable): The Gauss-Newton step that returns the vectorised
            descent direction of shape (nd,)

    Returns:
            np.ndarray: A new active subspace matrix U_+ of shape (n,d)
    """
    # initial values for res and Jac and Grad
    max_iter = 100
    res = residual(U)  # shape (M,1)
    Jac = jacobian(U)  # shape (M,n,d)  # noqa: N806
    Grad = np.tensordot(res.ravel(), Jac, axes=(0, 0))  # (n,d)  # noqa: N806

    if np.all(Jac == 0) and np.all(res == 0):
        return U

    if np.all(Jac == 0):
        raise ValueError("Jacobian is zero, cannot compute Gauss-Newton step.")

    if np.all(res == 0):
        return U

    # Compute tolerances
    Grad_norm = norm(Grad)  # noqa: N806
    tol = max(1e-10 * Grad_norm, 1e-14)
    tol_Delta_norm = 1e-12  # noqa: N806

    # loop over linesearch until the norm of the gauss-newton step, the norm of the Grad or the norm of Res(U) increases
    for _ in range(max_iter):
        residual_increased = False

        Jac.reshape(Jac.shape[0], -1)  # shape (M, nd)
        Delta_vec = gn_solver(Jac, res)  # shape (nd,)  # noqa: N806
        Delta = Delta_vec.reshape(  # noqa: N806
            Jac.shape[1], Jac.shape[2]
        )  # shape (n,d)

        # backtracking: find acceptable step gamma (t) along geodesic trajectory
        U_new, _step = backtracking(residual, Grad, Delta, U)  # noqa: N806

        res_candidate = residual(U_new)
        Jac_candidate = jacobian(U_new)  # noqa: N806
        Grad_candidate = np.tensordot(res_candidate.ravel(), Jac_candidate, axes=(0, 0))  # noqa: N806

        if norm(res_candidate) >= norm(res):
            residual_increased = True
        else:
            # Update the residual, jacobian, Gradient, and active subspace
            res = res_candidate
            Jac = Jac_candidate  # noqa: N806
            Grad = Grad_candidate  # noqa: N806
            U = U_new  # noqa: N806

        # Termination Conditions
        if Grad_norm < tol or norm(Delta) < tol_Delta_norm or residual_increased:
            return U_new

    return U


def backtracking(
    residual: callable,
    Grad: np.ndarray,  # noqa: N803
    delta: np.ndarray,
    U: np.ndarray,  # noqa: N803
) -> tuple[np.ndarray, float]:
    """Backtracking line search to satisfy the Armijo Condition:.

                    residual(U + alpha*delta) < residual(U) + alpha*beta*gamma
                    where:
                            - alpha is <Grad, delta>
                            - beta is a control parameter in (0,1)
                            - gamma is the backtracking coefficient.

    Args:
            residual (callable): Function that takes the active subspace U of shape
            (n,d) and calculates the residual of the predicted model under a fixed
            design set
            Grad (np.ndarray): Gradient of the active subspace matrix on the Grassmann
            manifold of shape (n,d)
            delta (np.ndarray): The Gauss-Newton step of shaoe (n,d)
            U (np.ndarray): The active subspace matrix with shape (n,d)

    Returns:
            tuple[np.ndarray, float]:
                    - The new active subspace matrix U of shape (n,d)
                    - The backtracking coefficient gamma  (gamma=1 implies no
                    backtracking)
    """
    # initialise control parameter, step shrink factor, and max iterations
    beta = 1e-4
    rho = 0.5
    max_iter = 100

    # directional derivative
    alpha = np.inner(
        Grad.reshape(
            -1,
        ),
        delta.reshape(
            -1,
        ),
    )  # vecGrad^T vec(delta) in matrix form

    # If direction is not a descent direction, flip to negative gradient
    if alpha >= 0:
        delta = -Grad
        alpha = np.inner(
            Grad.reshape(
                -1,
            ),
            delta.reshape(
                -1,
            ),
        )

    # starting objective and residual
    init_res = residual(U)

    step_size = 1.0
    for _ in range(max_iter):
        U_candidate = grassmann_trajectory(U, delta, step_size)  # noqa: N806
        res_candidate = residual(U_candidate)

        # Armijo condition: f(U + t delta) <= f(U) + t * beta * alpha
        if norm(res_candidate) <= norm(init_res) + step_size * beta * alpha:
            # success
            # Make sure U_new is orthonormal
            U_candidate, _ = np.linalg.qr(U_candidate)  # noqa: N806
            U_candidate = (  # noqa: N806
                np.sign(np.diag(_)) * U_candidate
            )  # ensure consistent orientation
            return U_candidate, step_size

        # otherwise shrink step
        step_size *= rho

    # if not found, return the best we have (the last candidate)
    U_candidate = grassmann_trajectory(U, delta, step_size)  # noqa: N806
    # Make sure U_new is orthonormal
    U_candidate, _ = np.linalg.qr(U_candidate)  # noqa: N806
    U_candidate = (  # noqa: N806
        np.sign(np.diag(_)) * U_candidate
    )  # ensure consistent orientation

    return U_candidate, step_size


def rotate_U(  # noqa: D417, N802
    problem: Problem,
    X: np.ndarray,  # noqa: N803
    fX: np.ndarray,  # noqa: N803
    coef: np.ndarray,
    U: np.ndarray,  # noqa: N803
    degree: int,
) -> np.ndarray:
    """Rotates the active subspace matrix onto the most important direction of.

    Args:
            X (np.ndarray): design set of shape (M,n)
            fX (np.ndarray): corresponding function estimates of design points of shape
            (M,1)
            coef (np.ndarray): The coefficients of the local model of shape (q,1)
            U (np.ndarray): The active subspace of shape (n,d).

    Returns:
            np.ndarray: The rotated active subspace matrix of shape (n,d)
    """
    # Step 1: Apply active subspaces to the profile function at samples X
    # to rotate onto the most important directions
    if U.shape[1] > 1:
        grads = grad(X, coef, U, degree)
        active_grads = grads
        # We only need the short-form SVD
        try:
            Ur = scipy.linalg.svd(active_grads.T, full_matrices=False)[0]  # noqa: N806
        except Exception as e:
            logging.warning("SVD failed with error:", e)
            #! need to add fallback that isn't svd

        U = U @ Ur  # noqa: N806

    # Step 2: Flip signs such that average slope is positive in the coordinate directions
    coef = fit_coef(problem, X, fX, U, degree)
    grads = grad(X, coef, U, degree)
    active_grads = grads  # shape (M,d)
    return U.dot(np.diag(np.sign(np.mean(active_grads, axis=0))))


def grad(X: np.ndarray, coef: np.ndarray, U: np.ndarray, degree: int) -> np.ndarray:  # noqa: D417, N803
    """Computes the gradients of the local model at each design point of the design set.

    X.

    Args:
            X (np.ndarray): design set of shape (M,n) or (M,d)
            coef (np.ndarray): The coefficients of the local model of shape (q,1)
            U (np.ndarray): The active subspace of shape (n,d)
            delta (float): radius of the current trust-region.

    Returns:
            np.ndarray: The gradients of the model at each design point X shape (M,d)
    """
    if len(X.shape) == 1:
        one_d = True
        X = X.reshape(1, -1)  # noqa: N806
    else:
        one_d = False

    # Check if X is full-space (n dimensions) or reduced-space (d dimensions)
    if X.shape[1] == U.shape[0]:  # noqa: SIM108
        # Full-space input: project to reduced space
        Y = X @ U  # noqa: N806
    else:
        # Already in reduced space
        Y = X  # noqa: N806
    DV_matrix = DV(Y, degree)  # shape (M,q,d)  # noqa: N806
    # Compute gradient on projected space
    Df = np.tensordot(DV_matrix, coef, axes=(1, 0))  # shape (M,d,1)  # noqa: N806
    # Inflate back to whole space
    Df = np.squeeze(Df, axis=-1)  # shape (M,d)  # noqa: N806

    if one_d:
        return Df.reshape(Y.shape[1])  # shape (d,)
    return Df  # shape (M,d)


def V(X: np.ndarray, degree: int) -> np.ndarray:  # noqa: D417, N802, N803
    """Generate the Vandermonde Matrix.

    Args:
            X (np.ndarray): The design set of shape (M,n) or (M,d) where n is the
            dimension of the original problem and d is the subspace dimension.

    Returns:
            np.ndarray: A vandermonde matrix of shape (M,q) where q is the length of the
            polynomial basis
    """
    M, d = X.shape  # noqa: N806

    # Otherwise, use TensorBasis approach (original implementation)
    X = scale(X)  # noqa: N806
    indices = index_set(degree, d).astype(int)
    M = X.shape[0]  # noqa: N806
    assert X.shape[1] == d, f"Expected {d} dimensions, got {X.shape[1]}"
    V_coordinate = [hermevander(X[:, k], degree) for k in range(d)]  # noqa: N806

    V = np.ones((M, len(indices)), dtype=X.dtype)  # noqa: N806
    for j, alpha in enumerate(indices):
        for k in range(d):
            V[:, j] *= V_coordinate[k][:, alpha[k]]
    return V


def DV(X: np.ndarray, degree: int) -> np.ndarray:  # noqa: N802, N803
    """Column-wise derivative of the Vandermonde matrix.

            Given design points this creates the Vandermonde-like matrix whose entries
            correspond to the derivatives of each of basis elements
    Args:
            X (np.ndarray): The design set of shape (M,d) where d is the subspace
            dimension.

    Returns:
            np.ndarray: Derivative of Vandermonde matrix  of shape (M,q,d) where
            DV[i,j,:] is the gradient of the
            partial derivative of the j-th basis function with respect to the x_k
            component of the d-dimensional vector
            and evaluated at i-th design point
    """
    M, d = X.shape  # noqa: N806

    # Otherwise, use TensorBasis approach (original implementation)
    X = scale(X)  # noqa: N806

    Dmat = build_Dmat(degree)  # noqa: N806
    indices = index_set(degree, d).astype(int)
    V_coordinate = [hermevander(X[:, k], degree) for k in range(d)]  # noqa: N806

    N = len(indices)  # noqa: N806
    DV = np.ones((M, N, d), dtype=X.dtype)  # noqa: N806

    # Get derivative scaling factors (1D array of length d)
    dscale_factors = dscale(d)

    for k in range(d):
        for j, alpha in enumerate(indices):
            for q in range(d):
                if q == k:
                    DV[:, j, k] *= np.dot(V_coordinate[q][:, 0:-1], Dmat[alpha[q], :])
                else:
                    DV[:, j, k] *= V_coordinate[q][:, alpha[q]]
            DV[:, j, k] *= dscale_factors[k]
    return DV


def scale(X):  # noqa: ANN001, ANN201, D103, N803
    return X


def dscale(d):  # noqa: ANN001, ANN201
    """Return scaling factors for derivatives (1D array of length d)."""
    return np.ones(d)


def build_Dmat(degree: int) -> np.ndarray:  # noqa: N802
    """Constructs the scalar derivative matrix.

    Returns:
            np.ndarray: The scalar derivative of shape (degree+1, degree) where degree
            is the degree of the polynomial.
    """
    Dmat = np.zeros((degree + 1, degree))  # noqa: N806
    I = np.eye(degree + 1)  # noqa: E741, N806
    for j in range(degree + 1):
        Dmat[j, :] = hermeder(I[:, j])

    return Dmat


def full_index_set(n: int, d: int) -> np.ndarray:
    """Enumerate multi-indices for a total degree of exactly `n` in `d` variables.

    Args:
            n (int): The total degree
            d (int): The number of variables

    Returns:
            np.ndarray: The multi-indices for the given total degree and number of
            variables
    """
    if d == 1:
        I = np.array([[n]])  # noqa: E741, N806
    else:
        II = full_index_set(n, d - 1)  # noqa: N806
        m = II.shape[0]
        I = np.hstack((np.zeros((m, 1)), II))  # noqa: E741, N806
        for i in range(1, n + 1):
            II = full_index_set(n - i, d - 1)  # noqa: N806
            m = II.shape[0]
            T = np.hstack((i * np.ones((m, 1)), II))  # noqa: N806
            I = np.vstack((I, T))  # noqa: E741, N806
    return I


def index_set(n: int, d: int) -> np.ndarray:
    """Enumerate multi-indices for a total degree of up to `n` in `d` variables.

    Args:
            n (int): The maximum total degree
            d (int): The number of variables

    Returns:
            np.ndarray: The multi-indices for the given maximum total degree and number
            of variables
    """
    I = np.zeros((1, d), dtype=np.int64)  # noqa: E741, N806
    for i in range(1, n + 1):
        II = full_index_set(i, d)  # noqa: N806
        I = np.vstack((I, II))  # noqa: E741, N806
    return I[:, ::-1].astype(int)  # this has length


def main() -> None:  # noqa: D103
    # Instantiate example problem (f(x) = ||x||^2)
    problem = instantiate_problem("ROSENBROCK-1")
    dim = problem.dim

    # Initial point (center of local region)
    x0 = np.zeros((1, dim))
    current_solution = Solution(tuple(x0), problem)

    # Model construction parameters
    delta_k = 0.5  # trust region radius
    k = 0
    expended_budget = 0
    visited_points_list = [current_solution]

    # Construct model and active subspace
    # current_solution, delta, model, U, fval, expended_budget, interpolation_solutions, visited_pts
    (
        current_solution,
        delta_k,
        model,
        U,  # noqa: N806
        _fval,
        expended_budget,
        _interpolation_solutions,
        visited_points_list,
    ) = construct_model(
        k, problem, current_solution, delta_k, expended_budget, visited_points_list
    )

    # Evaluate model performance over random samples in the ball
    n_test = 200
    X_test, f_true, f_model = [], [], []  # noqa: N806

    def obj_fn(x):  # noqa: ANN001, ANN202
        return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

    for _i in range(n_test):
        direction = np.random.randn(1, dim)
        direction /= np.linalg.norm(direction)
        radius = np.random.rand() * delta_k
        x = x0 + radius * direction
        X_test.append(x)
        f_true.append(obj_fn(x))
        f_model.append(model(x))

    X_test = np.array(X_test)  # noqa: N806
    f_true = np.array(f_true)
    f_model = np.array(f_model)

    # Compute error metrics
    mse = np.mean((f_true - f_model) ** 2)
    rel_error = np.mean(np.abs(f_true - f_model) / (np.abs(f_true) + 1e-8))

    ### DEBUG CHECK ###
    print("\n--- Final diagnostic checks ---")

    # Compare distributions
    print(f"True f mean/std: {np.mean(f_true):.4f}, {np.std(f_true):.4f}")
    print(f"Pred f mean/std: {np.mean(f_model):.4f}, {np.std(f_model):.4f}")

    # R² and correlation
    print(f"R² score: {r2_score(f_true, f_model):.4f}")
    corr = np.corrcoef(f_true.ravel(), f_model.ravel())[0, 1]
    print(f"Pearson correlation: {corr:.4f}")

    # Check relative scale
    ratio = np.mean(f_model) / (np.mean(f_true) + 1e-12)
    print(f"Average f_model / average true: {ratio:.4f}")

    print("--- End final diagnostic checks ---\n")
    ### END DEBUG CHECK ###

    print("Active subspace matrix U shape:", U.shape)
    print("Mean squared error:", mse)
    print("Mean relative error:", rel_error)
    print("Average true f(x):", np.mean(f_true))
    print("Average model prediction:", np.mean(f_model))


if __name__ == "__main__":
    main()
