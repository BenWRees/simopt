"""Demo script showing how to plot model construction for 1D and 2D examples.

Run this script from the repository root (or your environment where simopt is
importable). It creates two figures: a 1D fit and a 2D contour with a quadratic
model overlay.
"""
from __future__ import annotations

import sys
import os.path as o
from pathlib import Path

sys.path.append(
	o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), ".."))
)  # type:ignore


import numpy as np
import matplotlib.pyplot as plt
from plotting import (
    plot_1d_model_construction,
    model_callable

)

from design_set_test import construct_interpolation_set
from simopt.experiment_base import instantiate_problem, instantiate_solver
from simopt.base import Problem, Solution
from Model_construction import construct_model, fit

from mrg32k3a.mrg32k3a import MRG32k3a


def demo_1d():
    # A simple 1D objective
    problem = instantiate_problem("ZAKHAROV-1")
    x_k = np.array([[0.0]])
    incumbent_solution = Solution(tuple(x_k.flatten()), problem)
    U = np.eye(1)

    rng_stream = MRG32k3a()
    problem.attach_rngs([rng_stream])


    delta = 1.5 
    k = 0
    expended_budget = 0
    visited_points_list = [incumbent_solution]

    # construct interpolation set
    X, _ = construct_interpolation_set(
        incumbent_solution,
        problem,
        U,
        delta,
        k,
        visited_points_list
    )

    fX = []
    # simulate every interpolation point and add the objective value to fX
    for sol in X: 
        sol = Solution(tuple(sol.flatten()), problem)
        sol.attach_rngs([rng_stream])
        problem.simulate(sol, 10)
        fX.append(-1 * problem.minmax[0] * sol.objectives_mean.item())


    #construct model
    model = model_callable(X, fX, U)

    # Show a trust region around the center (here choose center at 0.0, radius 1.2)
    ax = plot_1d_model_construction(
        problem,
        model,
        X,
        fX,
        model_labels=["Local Model fit"],
        trust_center=x_k.item(),
        trust_radius=delta,
    )
    plt.tight_layout()
    plt.savefig("model_construction_1d.png", dpi=150)
    print("Saved model_construction_1d.png")


# def demo_2d():
#     # 2D objective (smooth bowl + small nonlinearity)
#     def obj_grid(X: np.ndarray) -> np.ndarray:
#         # X is (n,2)
#         x = X[:, 0]
#         y = X[:, 1]
#         return (x - 1) ** 2 + 0.5 * (y + 2) ** 2 + 0.5 * np.sin(0.6 * x * y)

#     # center and coordinate basis points (ASTRO-style: 2*d + 1 = 5 points)
#     center = np.array([0.0, 0.0])
#     delta = 1.0
#     e1 = np.array([delta, 0.0])
#     m1 = np.array([-delta, 0.0])
#     e2 = np.array([0.0, delta])
#     m2 = np.array([0.0, -delta])

#     samples = np.vstack([center, e1, m1, e2, m2])
#     fvals = obj_grid(samples)

#     # construct matrix rows: [1, x1, x2, x1^2, x2^2]
#     M = np.column_stack([
#         np.ones(samples.shape[0]),
#         samples[:, 0],
#         samples[:, 1],
#         samples[:, 0] ** 2,
#         samples[:, 1] ** 2,
#     ])

#     q = np.linalg.pinv(M) @ fvals
#     model = model_callable_from_q(q, dim=2)

#     ax = plot_2d_model_construction(obj_grid, model, samples, x_range=(-2.5, 3.0), y_range=(-3.5, 2.5), title="2D: contour + quadratic model")
#     plt.tight_layout()
#     plt.savefig("model_construction_2d.png", dpi=150)
#     print("Saved model_construction_2d.png")


if __name__ == "__main__":
    demo_1d()
    # demo_2d()
    print("Demos complete. Files: model_construction_1d.png, model_construction_2d.png")
