"""Cand_soln_select module."""
#! Change model construction and active subspace generation for the model-fitting method in ASTROMoRF  # noqa: D100, E501, RUF100

import os.path as o
import sys

sys.path.append(
    o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), ".."))  # noqa: PTH100, PTH118, PTH120
)  #


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import NonlinearConstraint, minimize

from ASTROMoRF_Code.Model_construction import construct_model
from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import Problem, Solution
from simopt.experiment_base import instantiate_problem


#########################
#    HELPER FUNCTIONS   #
#########################
class CallbackFunctor:  # noqa: D101
    def __init__(self, obj_fun, delta, U, x0) -> None:  # noqa: ANN001, D107, N803
        self.sols = []
        self.obj_fun = obj_fun
        self.delta = delta
        self.U = U
        self.x0 = x0

    def __call__(self, x, state=None):  # noqa: ANN001, ANN204, ARG002, D102
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        if np.linalg.norm(x - self.U.T @ self.x0, ord=2) <= self.delta + 1e-9:
            self.sols.append(np.copy(x))


def compress_points(sols, eps=0.1):  # noqa: ANN001, ANN201, D103
    # Remove consecutive points closer than eps
    filtered = [sols[0]]
    for s in sols[1:]:
        if not np.any([np.linalg.norm(s - a) < eps for a in filtered]):
            filtered.append(s)
    return np.array(filtered)


##########################
#    SUBPROBLEM PROCESS  #
##########################
def objective_fn(problem: Problem, x: np.ndarray) -> float:  # noqa: D103
    # create solution object
    x = tuple(x.flatten())
    sol = Solution(x, problem)
    sol.attach_rngs(problem.rng_list)
    problem.simulate(sol, 100)
    return sol.objectives_mean.item()


def cand_soln_objective_fn(model, model_grad, x0, U, lmbda):  # noqa: ANN001, ANN201, D103, N803
    if len(x0.shape) == 1:
        x0 = x0.reshape(-1, 1)

    _n, d = U.shape  # n: full space dim, d: reduced space dim
    projected_x = (U.T @ x0).flatten()  # Projected x0 in subspace
    x0.flatten()  # Original x0 in full space

    def obj_fn(x):  # noqa: ANN001, ANN202
        # regularizer
        x_reshape = x.reshape(-1, 1)
        print(f"Evaluating at projected x: {x.flatten()}")

        projected_step_size = x_reshape - projected_x.reshape(-1, 1)
        full_step = U @ projected_step_size.reshape(-1, 1)
        print(
            f"Projected Step Size: {projected_step_size.flatten()}, \n Full Step Size: {full_step.flatten()}"
        )
        print(
            f"projected step norm: {np.linalg.norm(projected_step_size, ord=2)}, full step norm: {np.linalg.norm(full_step, ord=2)}"
        )
        print(
            f"Regularization Term: {np.linalg.norm(projected_step_size, ord=2) ** 2 / np.linalg.norm(full_step, ord=2) ** 2}"
        )

        reg = (
            np.linalg.norm(projected_step_size, ord=2) ** 2
            / np.linalg.norm(full_step, ord=2) ** 2
        )
        res = model(x_reshape) + lmbda * reg
        print(f"Objective Value at x: {x.flatten()} is {res}")
        return res

    def obj_grad(z):  # noqa: ANN001, ANN202
        # get gradient of objective function
        z_col = np.array(z).reshape(-1, 1)  # shape (d, 1)
        z_full = U @ z_col  # Full-space step, shape (n, 1)

        model_grad_reduced = model_grad(z_col.reshape(1, -1)).flatten()  # shape (d,)

        full_space_step_norm = np.linalg.norm(z_full, ord=1)  # ||U·s||
        reduced_space_step_norm = np.linalg.norm(z_col, ord=1)  # ||s||

        if reduced_space_step_norm > 1e-10 and full_space_step_norm > 1e-10:
            term1 = z_col / (
                reduced_space_step_norm * full_space_step_norm
            )  # s/(||s||·||U·s||)
            term2 = (reduced_space_step_norm / (full_space_step_norm**3)) * (
                U.T @ z_full
            )  # (||s||/||U·s||³)·U^T·(U·s)
            penalty_grad = (term1 - term2).flatten()  # shape (d,)
        else:
            penalty_grad = np.zeros(d)  # Avoid division by zero at origin

        model_grad_reduced += lmbda * penalty_grad.flatten()

        return model_grad_reduced

    return obj_fn, obj_grad


def run_optimisation(model, model_grad, x0, U, delta, lmbda):  # noqa: ANN001, ANN201, D103, N803
    obj_fn, grad_fn = cand_soln_objective_fn(model, model_grad, x0, U, lmbda)
    if len(x0.shape) == 1:
        x0 = x0.reshape(-1, 1)

    x_init = U.T @ x0  # Start at projected x0

    # Trust-region constraint: 0 <= ||U.T @ x0 - x|| <= delta
    def cons(x):  # noqa: ANN001, ANN202
        return np.linalg.norm(x_init.flatten() - x, ord=2)

    cons = NonlinearConstraint(cons, 0, delta)

    cb = CallbackFunctor(obj_fn, delta, U, x0)

    result = minimize(
        obj_fn,
        x_init.flatten(),
        method="trust-constr",
        jac=grad_fn,
        constraints=[cons],
        callback=cb,
    )

    # check if any of the callback solutions are outside the trust region
    # for idx,sol in enumerate(cb.sols):
    #     if np.linalg.norm(sol, ord=2) > delta :
    #         print(f"Warning: Solution {idx} outside trust region: {sol}")

    return cb, result


#########################
#    RATIO COMPARISON   #
#########################
def ratio_comparison(  # noqa: D103
    problem: Problem,
    x0: np.ndarray,
    x_cand: np.ndarray,
    U: np.ndarray,  # noqa: N803
    model: callable,
) -> float:
    if len(x0.shape) == 1:
        x0 = x0.reshape(-1, 1)
    if len(x_cand.shape) == 1:
        x_cand = x_cand.reshape(-1, 1)

    full_x_cand = U @ x_cand  # Projected candidate back to full space
    f_x0 = objective_fn(problem, x0)
    f_x_cand = objective_fn(problem, full_x_cand)
    m_x0 = model(x0)
    m_x_cand = model(x_cand)
    actual_reduction = f_x0 - f_x_cand
    predicted_reduction = m_x0 - m_x_cand
    if predicted_reduction == 0:
        return 0  # Avoid division by zero
    return actual_reduction / predicted_reduction


###########################
#    PLOTTING FUNCTIONS   #
###########################
def plot_trajectory_solver(problem, model, x0, init_sol, U, delta, lmbda) -> None:  # noqa: ANN001, D103, N803
    cb, result = run_optimisation(model, x0, U, delta, lmbda)

    print("Final optimization result:", result.x)
    full_space_sol = U @ result.x.reshape(-1, 1)
    print("Final solution in full space:", full_space_sol.flatten())
    print(
        "Final objective value at result:",
        objective_fn(problem, full_space_sol.flatten()),
    )

    _fig, ax = plt.subplots(figsize=(15, 10))

    plot_trust_region(ax, delta, U, x0)
    plot_trajectory(ax, problem, cb)
    plot_contours(ax, problem, x0, init_sol, U, delta)

    # move legend location to lower right
    ax.legend(loc="lower right")
    ax.set_aspect("equal")
    ax.set_xlim(-delta - 2, delta + 2)
    ax.set_ylim(-delta - 2, delta + 2)
    ax.grid(True)
    ax.set_title(
        rf"Trust-Region Subproblem Trajectory starting at $({x0[0]},{x0[1]:},{x0[2]})$ with $\lambda={lmbda}$"
    )
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    plt.tight_layout()
    plt.savefig("trust_region_trajectory.png")


def plot_lambda_vs_optimality_gap(problem, x0, U, delta, model, model_grad) -> None:  # noqa: ANN001, D103, N803
    # plot optimality gap of different runs of the optimisation with varying lambda
    lambdas = np.linspace(0.0, 10.0, 50).tolist()
    plt.figure(figsize=(10, 5))
    ratios = []
    for lmbda in lambdas:
        cb, result = run_optimisation(model, model_grad, x0, U, delta, lmbda)
        final_sol = np.array(result.x).reshape(-1, 1)
        ratio = ratio_comparison(problem, x0, final_sol, U, model)
        # find the diffrence between the final solution and the
        # print(f'Lambda: {lmbda}, ratio: {ratio}, Final Obj: {final_obj}, Final Sol: {final_sol}')
        ratios.append(ratio)

        [print(f"solution: {a}") for a in cb.sols]
    [
        print(f"Lambda: {lmbda}, Ratio: {rat}")
        for lmbda, rat in zip(lambdas, ratios, strict=False)
    ]

    # plot a line of best fit through the (lambdas, opt_gaps) points
    sns.regplot(
        x=lambdas,
        y=ratios,
        scatter=False,
        ci=None,
        order=1,
        line_kws={"color": "red", "linestyle": "--"},
    )
    plt.plot([], [], "r--", label="General Relationship")

    plt.plot(lambdas, ratios, label="Ratio Comparison Test Value", color="blue")
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Ratio of Actual to Predicted Reduction")
    title = "Ratio of Actual to Predicted Reduction vs Lambdas\n"
    title += rf"($\mathbf{{x}}_0$={x0.flatten()}, $\Delta$={delta})"
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("lambda_vs_optimality_gap.png")


def plot_trust_region(ax, delta, U, x0) -> None:  # noqa: ANN001, D103, N803
    if len(x0.shape) == 1:
        x0 = x0.reshape(-1, 1)
    θ = np.linspace(0, 2 * np.pi, 200)
    center = U.T @ x0

    circle = center.reshape(-1, 1) + delta * np.vstack((np.cos(θ), np.sin(θ)))

    ax.fill(circle[0], circle[1], color="blue", alpha=0.2, label="Trust-Region")
    ax.plot(circle[0], circle[1], color="blue")
    ax.scatter(center[0], center[1], c="red", label="Projected Center")


def plot_trajectory(ax, problem, cb) -> None:  # noqa: ANN001, D103
    sols = np.array(cb.sols)
    # sols = compress_points(sols)

    # Plot internal trajectory points (bigger size = 80)
    if len(sols) > 2:
        ax.scatter(
            sols[1:-1, 0], sols[1:-1, 1], s=55, color="orange", label="Trajectory"
        )

    # Arrows between successive iterates
    for i in range(len(sols) - 1):
        ax.annotate(
            "",
            xy=sols[i + 1],
            xytext=sols[i],
            arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "orange"},
        )

    # Start and End as large X markers (size = 120)
    ax.scatter(
        sols[0, 0],
        sols[0, 1],
        color="green",
        marker="x",
        s=60,
        linewidths=2.5,
        label="Initial Projected Solution",
    )

    ax.scatter(
        sols[-1, 0],
        sols[-1, 1],
        color="red",
        marker="x",
        s=60,
        linewidths=2.5,
        label=f"Candidate Solution ({objective_fn(problem, sols[-1]):.2f})",
    )


def plot_contours(ax, problem, x0, init_sol, U, delta) -> None:  # noqa: ANN001, D103, N803
    # High-resolution grid
    x = np.linspace(-delta - 4, delta + 4, 1000)
    y = np.linspace(-delta - 4, delta + 4, 1000)
    X, Y = np.meshgrid(x, y)  # noqa: N806
    Z = np.zeros_like(X)  # noqa: N806

    # Evaluate objective on grid
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            pt = np.array([X[i, j], Y[i, j]])
            full_pt = x0 + U @ pt
            Z[i, j] = objective_fn(problem, full_pt)

    # Choose levels automatically but control density
    levels = 40  # fewer levels -> cleaner plot

    # Filled contours first
    ax.contourf(
        X,
        Y,
        Z,
        levels=levels,
        cmap="viridis",
        alpha=0.4,  # allows trust region + trajectory to stay visible
    )

    # Add contour lines for clarity
    ax.contour(X, Y, Z, levels=levels, colors="k", linewidths=0.5, alpha=0.7)

    # plot x0 without projection on the contour plot
    print(
        f"Plotting initial solution at: ({init_sol[0]}, {init_sol[1]}, {init_sol[2]})"
    )
    ax.scatter(
        init_sol[0],
        init_sol[1],
        c="black",
        marker="*",
        s=100,
        label=f"Initial Solution ({objective_fn(problem, init_sol):.2f})",
    )

    # plot the optimal solution
    optimal_sol = problem.optimal_solution
    print(
        f"Plotting optimal solution at: ({optimal_sol[0]}, {optimal_sol[1]}, {optimal_sol[2]})"
    )
    ax.scatter(
        optimal_sol[0],
        optimal_sol[1],
        c="purple",
        marker="P",
        s=100,
        label=f"Optimal Solution ({objective_fn(problem, optimal_sol):.2f})",
    )
    # ax.clabel(c, inline=True, fontsize=7, fmt="%.1f")


def main() -> None:  # noqa: D103
    x0 = np.array(
        [-3.5, 3.5, 1.0, 0.5, -1.5, 2.0, 1.0, 0.5, -1.0, 0.0]
    )  # Initial solution
    delta = 3.5
    k = 1
    d = 7  # if you want to plot_trajectory, d must be 2
    degree = 4
    problem_name = "DYNAMNEWS-1"
    expended_budget = 0

    problem = instantiate_problem(
        problem_name, problem_fixed_factors={"initial_solution": tuple(x0)}
    )
    rng_list = [MRG32k3a() for _ in range(problem.model.n_rngs)]
    problem.attach_rngs(rng_list)

    print(f"The current objective function is : {objective_fn(problem, x0)}")

    current_solution_x = tuple(x0)
    current_solution = Solution(current_solution_x, problem)
    current_solution.attach_rngs(rng_list)
    visited_points_list = [current_solution]

    (
        current_solution,
        _delta_k,
        model,
        model_grad,
        U,  # noqa: N806
        _fval,
        expended_budget,
        _interpolation_solutions,
        visited_points_list,
    ) = construct_model(
        k,
        d,
        degree,
        problem,
        current_solution,
        delta,
        expended_budget,
        visited_points_list,
    )

    # plot_trajectory_solver(problem, model, x0, current_solution_x, U, delta, lmbda)
    # plt.show()
    plot_lambda_vs_optimality_gap(problem, x0, U, delta, model, model_grad)
    plt.show()


if __name__ == "__main__":
    main()
