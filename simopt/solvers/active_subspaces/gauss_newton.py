# type: ignore  # noqa: D100
# 2018 (c) Jeffrey M. Hokanson and Caleb Magruder

import numpy as np
import scipy as sp

__all__ = [
    "gauss_newton",
    "linesearch_armijo",
    "trajectory_linear",
]


class BadStep(Exception):  # noqa: N818
    pass


def trajectory_linear(x0, p, t):  # noqa: ANN001, ANN201, D103
    return x0 + t * p


def linesearch_armijo(  # noqa: ANN201
    f,  # noqa: ANN001
    g,  # noqa: ANN001
    p,  # noqa: ANN001
    x0,  # noqa: ANN001
    bt_factor=0.5,  # noqa: ANN001
    ftol=1e-4,  # noqa: ANN001
    maxiter=40,  # noqa: ANN001
    trajectory=trajectory_linear,  # noqa: ANN001
    fx0=None,  # noqa: ANN001
):
    """Back-Tracking Line Search to satify Armijo Condition.

            f(x0 + alpha*p) < f(x0) + alpha * ftol * <g,p>

    Parameters
    ----------
    f : callable
            objective function, f: R^n -> R
    g : np.array((n,))
            gradient
    p : np.array((n,))
            descent direction
    x0 : np.array((n,))
            current location
    bt_factor : float [optional] default = 0.5
            backtracking factor
    ftol : float [optional] default = 1e-4
            coefficient in (0,1); see Armijo description in Nocedal & Wright
    maxiter : int [optional] default = 10
            maximum number of iterations of backtrack
    trajectory: function(x0, p, t) [Optional]
            Function that returns next iterate

    Returns:
    -------
    float
            alpha: backtracking coefficient (alpha = 1 implies no backtracking)
    """
    dg = np.inner(g, p)
    assert dg <= 0, (
        f"Descent direction p is not a descent direction: p^T g = {dg:g} >= 0"
    )
    iterations = 0

    alpha = 1

    if fx0 is None:
        fx0 = f(x0)

    fx0_norm = np.linalg.norm(fx0)
    x = np.copy(x0)
    fx = np.inf
    success = False
    for _it in range(maxiter):
        try:
            iterations += 1
            x = trajectory(x0, p, alpha)
            fx = f(x)
            fx_norm = np.linalg.norm(fx)
            if fx_norm < fx0_norm + alpha * ftol * dg:
                success = True
                break
        except BadStep:
            pass

        alpha *= bt_factor

    # If we haven't found a good step, stop
    if not success:
        alpha = 0
        x = x0
        fx = fx0
    return x, alpha, fx, iterations


def gauss_newton(  # noqa: ANN201
    f,  # noqa: ANN001
    F,  # noqa: ANN001, N803
    x0,  # noqa: ANN001
    tol=1e-10,  # noqa: ANN001
    tol_normdx=1e-12,  # noqa: ANN001
    maxiter=100,  # noqa: ANN001
    linesearch=None,  # noqa: ANN001
    verbose=0,  # noqa: ANN001
    trajectory=None,  # noqa: ANN001
    gnsolver=None,  # noqa: ANN001
):
    r"""A Gauss-Newton solver for unconstrained nonlinear least squares problems.

    Given a vector valued function :math:`\mathbf{f}:\mathbb{R}^m \to \mathbb{R}^M`
    and its Jacobian :math:`\mathbf{F}:\mathbb{R}^m\to \mathbb{R}^{M\times m}`,
    solve the nonlinear least squares problem:

    .. math::

            \min_{\mathbf{x}\in \mathbb{R}^m} \| \mathbf{f}(\mathbf{x})\|_2^2.

    Normal Gauss-Newton computes a search direction :math:`\mathbf{p}\in \mathbb{R}^m`
    at each iterate by solving least squares problem

    .. math::

            \mathbf{p}_k \leftarrow \mathbf{F}(\mathbf{x}_k)^+ \mathbf{f}(\mathbf{x}_k)

    and then computes a new step by solving a line search problem for a step length
    :math:`\alpha`
    satisfying the Armijo conditions:

    .. math::

            \mathbf{x}_{k+1} \leftarrow \mathbf{x}_k + \alpha \mathbf{p}_k.

    This implementation offers several features that modify this basic outline.

    First, the user can specify a nonlinear *trajectory* along which candidate points
    can move; i.e.,

    .. math::

            \mathbf{x}_{k+1} \leftarrow T(\mathbf{x}_k, \mathbf{p}_k, \alpha).

    Second, the user can specify a custom solver for computing the search direction
    :math:`\mathbf{p}_k`.

    Parameters
    ----------
    f : callable
            residual, :math:`\mathbf{f}: \mathbb{R}^m \to \mathbb{R}^M`
    F : callable
            Jacobian of residual :math:`\mathbf{f}`; :math:`\mathbf{F}: \mathbb{R}^m \to
            \mathbb{R}^{M \times m}`
    tol: float [optional] default = 1e-8
            gradient norm stopping criterion
    tol_normdx: float [optional] default = 1e-12
            norm of control update stopping criterion
    maxiter : int [optional] default = 100
            maximum number of iterations of Gauss-Newton
    linesearch: callable, returns new x
            f : callable, residual, f: R^n -> R^m
            g : gradient, R^n
            p : descent direction, R^n
            x0 : current iterate, R^n
    gnsolver: [optional] callable, returns search direction p
            Parameters:
                    F: current Jacobian
                    f: current residual

    Returns:
                    p: search step
                    s: singular values of Jacobian
    verbose: int [optional] default = 0
            if >= print convergence history diagnostics

    Returns:
    -------
    numpy.array((dof,))
            returns x^* (optimizer)
    int
            info = 0: converged with norm of gradient below tol
            info = 1: norm of gradient did not converge, but ||dx|| below tolerance
            info = 2: did not converge, max iterations exceeded
    """
    len(x0)
    if maxiter <= 0:
        return x0, 4
    iterations = 0
    if verbose >= 1:
        print("Gauss-Newton Solver Iteration History")
        print(
            "  iter   |   ||f(x)||   |   ||dx||   | cond(F(x)) |   alpha    |  ||grad||  "
        )
        print(
            "---------|--------------|------------|------------|------------|------------"
        )
    if trajectory is None:

        def trajectory(x0, p, t):  # noqa: ANN001, ANN202
            return x0 + t * p

    if linesearch is None:
        linesearch = linesearch_armijo

    if gnsolver is None:
        # Scipy seems to properly check for proper allocation of working space, reporting an error with gelsd
        # so we specify using gelss (an SVD based solver)
        def gnsolver(F_eval, f_eval):  # noqa: ANN001, ANN202, N803
            dx, _, _, s = sp.linalg.lstsq(F_eval, -f_eval, lapack_driver="gelsd")
            return dx, s

    x = np.copy(x0)
    f_eval = f(x)
    F_eval = F(x)  # noqa: N806
    grad = F_eval.T @ f_eval

    normgrad = np.linalg.norm(grad)
    info = None
    # rescale tol by norm of initial gradient
    tol = max(tol * normgrad, 1e-14)

    normdx = 1
    for it in range(maxiter):
        residual_increased = False

        # Compute search direction
        dx, s = gnsolver(F_eval, f_eval)

        # Check we got a valid search direction
        if not np.all(np.isfinite(dx)):
            raise RuntimeError("Non-finite search direction returned")

        # If Gauss-Newton step is not a descent direction, use -gradient instead
        if np.inner(grad, dx) >= 0:
            dx = -grad

        # Back tracking line search
        x_new, alpha, f_eval_new, iterations_linesearch = linesearch(
            f, grad, dx, x, trajectory=trajectory
        )
        iterations += iterations_linesearch

        normf = np.linalg.norm(f_eval_new)
        if np.linalg.norm(f_eval_new) >= np.linalg.norm(f_eval):
            residual_increased = True
        else:
            # f_eval = f(x)
            f_eval = f_eval_new
            x = x_new

            normdx = np.linalg.norm(dx)
            F_eval = F(x)  # noqa: N806
            grad = F_eval.T @ f_eval_new
        #########################################################################
        # Printing section
        cond = np.inf if s[-1] == 0 else s[0] / s[-1]

        if verbose >= 1:
            normgrad = np.linalg.norm(grad)
            print(
                "    %3d  |  %1.4e  |  %8.2e  |  %8.2e  |  %8.2e  |  %8.2e"  # noqa: UP031
                % (it, normf, normdx, cond, alpha, normgrad)
            )

        # Termination conditions
        if normgrad < tol:
            if verbose:
                print(f"norm gradient {normgrad:1.3e} less than tolerance {tol:1.3e}")
            break
        if normdx < tol_normdx:
            if verbose:
                print(f"norm dx {normdx:1.3e} less than tolerance {tol_normdx:1.3e}")
            break
        if residual_increased:
            if verbose:
                print(
                    f"residual increased during line search from {np.linalg.norm(f_eval):1.5e} to {np.linalg.norm(f_eval_new):1.5e}"
                )
            break

    if normgrad <= tol:
        info = 0
        if verbose >= 1:
            print("Gauss-Newton converged successfully!")
    elif normdx <= tol_normdx:
        info = 1
        if verbose >= 1:
            print("Gauss-Newton did not converge: ||dx|| < tol")
    elif it == maxiter - 1:
        info = 2
        if verbose >= 1:
            print("Gauss-Newton did not converge: max iterations reached")
    elif np.linalg.norm(f_eval_new) >= np.linalg.norm(f_eval):
        info = 3
        if verbose >= 1:
            print("No progress made during line search")
    else:
        raise Exception("Stopping criteria not determined!")

    return x, info, iterations
