"""Scenario-specific optimality gap plot.

Plots each solver's gap to the best observed ProblemSolver solution value
for a given problem variant over the budget horizon.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import simopt.curve_utils as curve_utils
from simopt.curve import Curve
from simopt.experiment import ProblemSolver
from simopt.plot_type import PlotType
from simopt.problem import ProblemLike

from .utils import save_plot

logger = logging.getLogger(__name__)


def _objective_sign(problem: ProblemLike) -> float:
    """Return objective direction sign (+1 maximize, -1 minimize)."""
    minmax = getattr(problem, "minmax", (1,))
    if isinstance(minmax, tuple | list | np.ndarray):
        if len(minmax) == 0:
            return 1.0
        sign = float(minmax[0])
    else:
        sign = float(minmax)
    return sign if sign != 0 else 1.0


def _best_solution_value(
    experiment: ProblemSolver,
    objective_sign: float,
) -> float | None:
    """Get best solution objective estimate for one ProblemSolver instance."""
    candidates: list[float] = []

    xstar_postreps = getattr(experiment, "xstar_postreps", None)
    if xstar_postreps:
        vals = np.asarray(xstar_postreps, dtype=float).reshape(-1)
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            candidates.append(float(np.mean(vals)))

    all_est_objectives = getattr(experiment, "all_est_objectives", None)
    if all_est_objectives:
        terminal_vals: list[float] = []
        for seq in all_est_objectives:
            arr = np.asarray(seq, dtype=float).reshape(-1)
            arr = arr[np.isfinite(arr)]
            if arr.size > 0:
                terminal_vals.append(float(arr[-1]))
        if terminal_vals:
            candidates.append(
                float(max(terminal_vals) if objective_sign > 0 else min(terminal_vals))
            )

    objective_curves = getattr(experiment, "objective_curves", None)
    if objective_curves:
        curve_terminal_vals: list[float] = []
        for curve in objective_curves:
            arr = np.asarray(curve.y_vals, dtype=float).reshape(-1)
            arr = arr[np.isfinite(arr)]
            if arr.size > 0:
                curve_terminal_vals.append(float(arr[-1]))
        if curve_terminal_vals:
            candidates.append(
                float(
                    max(curve_terminal_vals)
                    if objective_sign > 0
                    else min(curve_terminal_vals)
                )
            )

    if not candidates:
        return None

    return float(max(candidates) if objective_sign > 0 else min(candidates))


def _variant_best_solution_value(
    experiments: list[ProblemSolver],
    objective_sign: float,
) -> float | None:
    """Get best solution objective estimate across all solvers for a variant."""
    values: list[float] = []
    for experiment in experiments:
        best_val = _best_solution_value(experiment, objective_sign)
        if best_val is not None and np.isfinite(best_val):
            values.append(float(best_val))

    if not values:
        return None

    return float(max(values) if objective_sign > 0 else min(values))


def plot_optimality_gap_curves(
    experiments_by_variant: dict[str, list[ProblemSolver]],
    plot_type: PlotType = PlotType.MEAN,
    beta: float = 0.50,
    n_bootstraps: int = 100,
    conf_level: float = 0.95,
    plot_conf_ints: bool = True,
    plot_title: str | None = None,
    legend_loc: str | None = None,
    ext: str = ".png",
    save_as_pickle: bool = False,
    solver_set_name: str = "SOLVER_SET",
) -> list[Path]:
    """Plot optimality gap curves normalized by variant best solution value.

    For each problem variant a subplot shows how close each solver gets to
    the best solution value observed across the provided ``ProblemSolver``
    instances for that same variant.

    The y-axis is the scaled nonnegative gap to the variant baseline:
    ``max(0, s * (f_best_variant - f(x_t))) / max(abs(f_best_variant), eps)``,
    where ``s`` is ``+1`` for maximization and ``-1`` for minimization.
    Lower is better; 0 means the solver matched the best observed value.

    Args:
        experiments_by_variant: Mapping from variant name to a list of
            ``ProblemSolver`` objects (different solvers, same variant).
            Each ``ProblemSolver`` must have been post-replicated so that
            ``objective_curves`` (or ``all_est_objectives``) are available.
        plot_type: ``PlotType.MEAN`` or ``PlotType.QUANTILE``.
        beta: Quantile level when *plot_type* is ``QUANTILE``.
        n_bootstraps: Number of bootstrap samples for confidence intervals.
        conf_level: Confidence level for bootstrap CIs.
        plot_conf_ints: Whether to plot bootstrap confidence bands.
        plot_title: Override title for the whole figure.
        legend_loc: Legend placement string (e.g. ``"best"``).
        ext: File extension for saved plots.
        save_as_pickle: Also save figure as a pickle.
        solver_set_name: Label used in the saved filename.

    Returns:
        List of file paths where the plots were saved.
    """
    if not 0 < beta < 1:
        raise ValueError("Beta quantile must be in (0, 1).")
    if n_bootstraps < 1:
        raise ValueError("Number of bootstraps must be a positive integer.")
    if not 0 < conf_level < 1:
        raise ValueError("Confidence level must be in (0, 1).")

    if legend_loc is None:
        legend_loc = "best"

    variants = list(experiments_by_variant.keys())
    n_variants = len(variants)

    fig, axes = plt.subplots(
        1,
        n_variants,
        figsize=(6 * n_variants, 5),
        squeeze=False,
    )

    file_list: list[Path] = []

    for col_idx, variant in enumerate(variants):
        ax = axes[0, col_idx]
        experiments = experiments_by_variant[variant]
        if not experiments:
            logger.warning("No experiments found for variant %s; skipping.", variant)
            continue

        objective_sign = _objective_sign(experiments[0].problem)
        variant_best_value = _variant_best_solution_value(experiments, objective_sign)
        if variant_best_value is None:
            logger.warning(
                "No finite best-solution value for variant %s; skipping.",
                variant,
            )
            continue
        gap_scale = max(abs(float(variant_best_value)), 1e-12)

        solver_handles = []
        solver_labels = []

        for exp_idx, experiment in enumerate(experiments):
            budget = float(experiment.problem.factors.get("budget", 0.0))
            if not np.isfinite(budget) or budget <= 0:
                logger.warning(
                    "Invalid budget for %s on variant %s; skipping.",
                    experiment.solver.name,
                    variant,
                )
                continue

            # Build per-macrorep gap curves from objective_curves
            gap_curves: list[Curve] = []
            if hasattr(experiment, "objective_curves") and experiment.objective_curves:
                for obj_curve in experiment.objective_curves:
                    gap_y = tuple(
                        max(
                            0.0,
                            objective_sign
                            * (float(variant_best_value) - float(y))
                            / gap_scale,
                        )
                        for y in obj_curve.y_vals
                    )
                    # Use fractional budget on x-axis
                    frac_x = tuple(x / budget for x in obj_curve.x_vals)
                    gap_curves.append(Curve(x_vals=frac_x, y_vals=gap_y))
            elif (
                hasattr(experiment, "all_est_objectives")
                and experiment.all_est_objectives
            ):
                for mrep in range(experiment.n_macroreps):
                    budgets = experiment.all_intermediate_budgets[mrep]
                    est_objs = experiment.all_est_objectives[mrep]
                    frac_x = tuple(b / budget for b in budgets)
                    gap_y = tuple(
                        max(
                            0.0,
                            objective_sign
                            * (float(variant_best_value) - float(obj))
                            / gap_scale,
                        )
                        for obj in est_objs
                    )
                    gap_curves.append(Curve(x_vals=frac_x, y_vals=gap_y))
            else:
                logger.warning(
                    "No objective data for %s on variant %s; skipping.",
                    experiment.solver.name,
                    variant,
                )
                continue

            if not gap_curves:
                continue

            color_str = f"C{exp_idx}"

            if plot_type == PlotType.MEAN:
                estimator = curve_utils.mean_of_curves(gap_curves)
            elif plot_type == PlotType.QUANTILE:
                estimator = curve_utils.quantile_of_curves(gap_curves, beta)
            elif plot_type == PlotType.ALL:
                # Plot all individual curves
                for _i, curve in enumerate(gap_curves):
                    (line,) = ax.plot(
                        curve.x_vals,
                        curve.y_vals,
                        color=color_str,
                        alpha=0.3,
                        linewidth=0.8,
                    )
                solver_handles.append(line)
                solver_labels.append(experiment.solver.name)
                continue
            else:
                raise NotImplementedError(
                    f"Plot type '{plot_type.value}' not supported for optimality gap."
                )

            (line,) = ax.plot(
                estimator.x_vals,
                estimator.y_vals,
                color=color_str,
                linewidth=1.5,
            )
            solver_handles.append(line)
            solver_labels.append(experiment.solver.name)

            # Bootstrap confidence intervals
            if plot_conf_ints and plot_type != PlotType.ALL:
                # Wrap the gap_curves into a temporary ProblemSolver-like
                # structure for bootstrap_procedure — or compute manually
                _plot_manual_bootstrap_ci(
                    ax,
                    gap_curves,
                    plot_type,
                    beta,
                    n_bootstraps,
                    conf_level,
                    color_str,
                )

        ax.set_title(variant.replace("_", " ").title(), size=12)
        ax.set_xlabel("Fraction of Budget", size=11)
        ax.set_ylabel("Fraction of Optimality Gap", size=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0.0)
        ax.tick_params(axis="both", which="major", labelsize=10)

        if solver_handles:
            ax.legend(
                handles=solver_handles,
                labels=solver_labels,
                loc=legend_loc,
                fontsize=9,
            )

    if plot_title:
        fig.suptitle(plot_title, size=14)
    fig.tight_layout()

    # Save
    file_path = save_plot(
        solver_name=solver_set_name,
        problem_name="optimality_gap",
        plot_type=plot_type,
        normalize=True,
        extra=beta,
        plot_title=plot_title,
        ext=ext,
        save_as_pickle=save_as_pickle,
    )
    file_list.append(file_path)

    return file_list


def _plot_manual_bootstrap_ci(
    ax: plt.Axes,
    gap_curves: list[Curve],
    plot_type: PlotType,
    beta: float,
    n_bootstraps: int,
    conf_level: float,
    color_str: str,
) -> None:
    """Compute and plot bootstrap confidence intervals on gap curves."""
    rng = np.random.default_rng(seed=42)
    n_curves = len(gap_curves)
    if n_curves < 2:
        return

    # Compute the point estimator
    if plot_type == PlotType.MEAN:
        point_est = curve_utils.mean_of_curves(gap_curves)
    else:
        point_est = curve_utils.quantile_of_curves(gap_curves, beta)

    # Bootstrap on a fixed x-grid so every resample has the same shape.
    x_mesh = tuple(point_est.x_vals)
    boot_y_vals: list[np.ndarray] = []
    for _ in range(n_bootstraps):
        indices = rng.choice(n_curves, size=n_curves, replace=True)
        boot_sample = [gap_curves[i] for i in indices]
        if plot_type == PlotType.MEAN:
            boot_curve = curve_utils.mean_of_curves(boot_sample)
        else:
            boot_curve = curve_utils.quantile_of_curves(boot_sample, beta)
        boot_y_vals.append(
            np.asarray([boot_curve.lookup(x) for x in x_mesh], dtype=float)
        )

    if not boot_y_vals:
        return

    boot_array = np.vstack(boot_y_vals)  # (n_bootstraps, n_points)
    alpha = 1 - conf_level
    lb = np.nanquantile(boot_array, alpha / 2, axis=0)
    ub = np.nanquantile(boot_array, 1 - alpha / 2, axis=0)

    x_vals = np.asarray(x_mesh, dtype=float)
    finite_mask = np.isfinite(lb) & np.isfinite(ub)
    if not np.any(finite_mask):
        return

    ax.plot(
        x_vals[finite_mask],
        lb[finite_mask],
        linestyle="--",
        color=color_str,
        alpha=0.5,
        linewidth=0.8,
    )
    ax.plot(
        x_vals[finite_mask],
        ub[finite_mask],
        linestyle="--",
        color=color_str,
        alpha=0.5,
        linewidth=0.8,
    )
    ax.fill_between(
        x_vals[finite_mask],
        lb[finite_mask],
        ub[finite_mask],
        color=color_str,
        alpha=0.1,
    )
