"""Lightweight plotting utilities for visualising model construction.

Provides simple helpers to visualise 1D and 2D objective functions together
with fitted quadratic/local models and interpolation points. These are
presentation-friendly (matplotlib) helpers and intentionally small and
dependency-light.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Iterable, List, Tuple


def model_callable_from_q(q: np.ndarray, dim: int) -> Callable[[np.ndarray], np.ndarray]:
    """Create a callable model from ASTRO-style coefficient vector `q`.

    The expected layout is: [1, x_1, x_2, ..., x_d, x_1^2, x_2^2, ..., x_d^2]

    Args:
        q: coefficient vector of length `2*dim + 1`.
        dim: number of input dimensions.

    Returns:
        Callable that accepts an array `X` of shape (n_samples, dim) and
        returns a flat array of predictions shape (n_samples,).
    """

    q = np.asarray(q).reshape(-1)

    if q.size != 2 * dim + 1:
        raise ValueError("q length does not match expected 2*dim+1")

    def _model(X: np.ndarray) -> np.ndarray:
        Xarr = np.asarray(X)
        try:
            Xmat = Xarr.reshape(-1, dim)
        except Exception:
            # Provide a helpful message when reshape fails
            raise ValueError(f"Expected input that can be shaped (-1, {dim}); got shape {Xarr.shape}")

        ones = np.ones((Xmat.shape[0], 1))
        lin = Xmat
        sq = Xmat ** 2
        design = np.hstack((ones, lin, sq))
        preds = design @ q
        return np.asarray(preds).reshape(-1)

    return _model


def plot_1d_model_construction(
    objective: Callable[[np.ndarray], np.ndarray],
    models: Iterable[Callable[[np.ndarray], np.ndarray]] | Iterable[np.ndarray],
    sample_x: np.ndarray,
    sample_y: np.ndarray,
    x_min: float = None,
    x_max: float = None,
    n_grid: int = 400,
    model_labels: List[str] | None = None,
    title: str | None = "Model construction (1D)",
    ax: plt.Axes | None = None,
    trust_center: float | None = None,
    trust_radius: float | None = None,
    trust_color: str = "#FFA07A",
    trust_alpha: float = 0.3,
    show_trust_label: bool = True,
) -> plt.Axes:
    """Plot true objective, one or more model approximations and sample points.

    `models` may be a list of callables or a list of coefficient vectors `q`.
    If coefficient vectors are provided they must be convertible with
    `model_callable_from_q` but in that case the function cannot infer `dim`
    automatically and will assume `dim==1`.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    sample_x = np.asarray(sample_x).reshape(-1)
    sample_y = np.asarray(sample_y).reshape(-1)

    if x_min is None:
        x_min = float(sample_x.min()) - 0.2 * (sample_x.ptp() if sample_x.ptp() != 0 else 1)
    if x_max is None:
        x_max = float(sample_x.max()) + 0.2 * (sample_x.ptp() if sample_x.ptp() != 0 else 1)

    grid_x = np.linspace(x_min, x_max, n_grid)
    true_y = np.asarray(objective(grid_x)).reshape(-1)
    ax.plot(grid_x, true_y, color="#222222", lw=2, label="True objective")

    # Normalize models iterable
    models_list = list(models)
    for i, mdl in enumerate(models_list):
        if not callable(mdl):
            # assume q for 1D
            mdl = model_callable_from_q(np.asarray(mdl), dim=1)
        pred_y = np.asarray(mdl(grid_x)).reshape(-1)
        label = model_labels[i] if model_labels and i < len(model_labels) else f"Model {i+1}"
        ax.plot(grid_x, pred_y, lw=1.5, label=label)

    ax.scatter(sample_x, sample_y, color="C3", zorder=5, label="Interpolation points")

    # Draw trust region if provided (shaded band and center marker)
    if trust_center is not None and trust_radius is not None:
        left = trust_center - trust_radius
        right = trust_center + trust_radius
        ax.axvspan(left, right, color=trust_color, alpha=trust_alpha, zorder=0)
        ax.axvline(trust_center, color=trust_color, linestyle="--", linewidth=1)
        if show_trust_label:
            ax.text(
                trust_center,
                ax.get_ylim()[1],
                "  Trust region",
                va="top",
                ha="left",
                color=trust_color,
                fontsize=9,
            )
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.2)
    return ax


def plot_2d_model_construction(
    objective: Callable[[np.ndarray], np.ndarray],
    model: Callable[[np.ndarray], np.ndarray] | np.ndarray,
    sample_xy: np.ndarray,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    n_grid: int = 80,
    levels: int = 20,
    title: str | None = "Model construction (2D)",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot 2D contours of the true objective and overlay model contours and points.

    Args:
        objective: callable accepting array shape (n,2) or two arrays (X,Y) depending
                   on user preference. The function below will call it with stacked
                   2-column input.
        model: callable or coefficient vector `q` (then dim must be 2).
        sample_xy: array shape (n_points, 2) of interpolation/sample points.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    # prepare model callable
    if not callable(model):
        model = model_callable_from_q(np.asarray(model), dim=2)

    xx = np.linspace(x_range[0], x_range[1], n_grid)
    yy = np.linspace(y_range[0], y_range[1], n_grid)
    Xg, Yg = np.meshgrid(xx, yy)
    grid = np.column_stack([Xg.ravel(), Yg.ravel()])

    true_vals = objective(grid).reshape(n_grid, n_grid)
    model_vals = model(grid).reshape(n_grid, n_grid)

    cf = ax.contourf(Xg, Yg, true_vals, levels=levels, cmap="viridis", alpha=0.9)
    c = ax.contour(Xg, Yg, model_vals, levels=levels//2, colors="white", linewidths=1, alpha=0.9)
    ax.scatter(sample_xy[:, 0], sample_xy[:, 1], c="C3", edgecolor="k", zorder=5)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    if title:
        ax.set_title(title)
    plt.colorbar(cf, ax=ax, label="True objective")
    ax.clabel(c, inline=True, fontsize=8)
    ax.grid(alpha=0.15)
    return ax
