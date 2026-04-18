"""Demo script showing how to plot model construction for 1D and 2D examples.

Run this script from the repository root (or your environment where simopt is
importable). It creates two figures: a 1D fit and a 2D contour with a quadratic
model overlay.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repository root is on sys.path so `import simopt` works when the
# demo is executed directly (e.g. `python3 demo/plot_model_construction.py`).
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from simopt.plotting import (  # noqa: E402
    model_callable_from_q,
    plot_1d_model_construction,
    plot_2d_model_construction,
)


def demo_1d() -> None:  # noqa: D103
    # A simple 1D objective
    def obj(x):  # noqa: ANN001, ANN202
        return np.sin(x) + 0.1 * x

    # choose three interpolation points for a quadratic fit
    xs = np.array([-3.0, 0.0, 2.5])
    ys = obj(xs)

    # build design matrix [1, x, x^2] and solve for q
    M = np.column_stack([np.ones(xs.shape[0]), xs, xs**2])  # noqa: N806
    q = np.linalg.pinv(M) @ ys

    model = model_callable_from_q(q, dim=1)

    # Show a trust region around the center (here choose center at 0.0, radius 1.2)
    plot_1d_model_construction(
        obj,
        [model],
        xs,
        ys,
        model_labels=["Quadratic fit"],
        title="1D: quadratic approximation",
        trust_center=0.0,
        trust_radius=1.2,
    )
    plt.tight_layout()
    plt.savefig("model_construction_1d.png", dpi=150)
    print("Saved model_construction_1d.png")


def demo_2d() -> None:  # noqa: D103
    # 2D objective (smooth bowl + small nonlinearity)
    def obj_grid(X: np.ndarray) -> np.ndarray:  # noqa: N803
        # X is (n,2)
        x = X[:, 0]
        y = X[:, 1]
        return (x - 1) ** 2 + 0.5 * (y + 2) ** 2 + 0.5 * np.sin(0.6 * x * y)

    # center and coordinate basis points (ASTRO-style: 2*d + 1 = 5 points)
    center = np.array([0.0, 0.0])
    delta = 1.0
    e1 = np.array([delta, 0.0])
    m1 = np.array([-delta, 0.0])
    e2 = np.array([0.0, delta])
    m2 = np.array([0.0, -delta])

    samples = np.vstack([center, e1, m1, e2, m2])
    fvals = obj_grid(samples)

    # construct matrix rows: [1, x1, x2, x1^2, x2^2]
    M = np.column_stack(  # noqa: N806
        [
            np.ones(samples.shape[0]),
            samples[:, 0],
            samples[:, 1],
            samples[:, 0] ** 2,
            samples[:, 1] ** 2,
        ]
    )

    q = np.linalg.pinv(M) @ fvals
    model = model_callable_from_q(q, dim=2)

    plot_2d_model_construction(
        obj_grid,
        model,
        samples,
        x_range=(-2.5, 3.0),
        y_range=(-3.5, 2.5),
        title="2D: contour + quadratic model",
    )
    plt.tight_layout()
    plt.savefig("model_construction_2d.png", dpi=150)
    print("Saved model_construction_2d.png")


if __name__ == "__main__":
    demo_1d()
    demo_2d()
    print("Demos complete. Files: model_construction_1d.png, model_construction_2d.png")
