"""
Plotting utilities for data farming experiments.

This module provides functions to visualize the results of data farming experiments,
particularly for comparing solver performance across different factor levels
(e.g., subspace dimensions or polynomial basis types).
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from simopt.solvers.astromorf import PolyBasisType

if TYPE_CHECKING:
    from simopt.experiment import ProblemSolver, ProblemsSolvers


# ============================================================================
# CONSTANTS
# ============================================================================

# Mapping from PolyBasisType enum to readable names
POLY_BASIS_NAMES = {
    PolyBasisType.HERMITE: "Hermite",
    PolyBasisType.LEGENDRE: "Legendre",
    PolyBasisType.CHEBYSHEV: "Chebyshev",
    PolyBasisType.MONOMIAL: "Monomial",
    PolyBasisType.NATURAL: "Natural",
    PolyBasisType.MONOMIAL_POLY: "MonomialPoly",
    PolyBasisType.LAGRANGE: "Lagrange",
    PolyBasisType.NFP: "NFP",
    PolyBasisType.LAGUERRE: "Laguerre",
}


# ============================================================================
# DATA EXTRACTION UTILITIES
# ============================================================================

def extract_terminal_objectives(
    experiment: "ProblemSolver",
) -> list[float]:
    """
    Extract terminal objective values for each macroreplication.
    
    Args:
        experiment: A ProblemSolver object containing experiment results.
        
    Returns:
        List of terminal objective values, one per macroreplication.
    """
    terminal_objectives = []
    
    for mrep in range(experiment.n_macroreps):
        if len(experiment.all_est_objectives[mrep]) > 0:
            # Get the last (terminal) estimated objective
            terminal_obj = experiment.all_est_objectives[mrep][-1]
            terminal_objectives.append(terminal_obj)
    
    return terminal_objectives


def extract_optimality_gap_fraction(
    experiment: "ProblemSolver",
) -> list[float]:
    """
    Extract the remaining optimality gap as a fraction for each macroreplication.
    
    The optimality gap is computed as: (f(x_terminal) - f(x*)) / (f(x0) - f(x*))
    where x0 is the initial solution and x* is the optimal/best known solution.
    
    This follows standard optimization convention:
    - 0.0 = optimal solution reached (no gap remaining)
    - 1.0 = no improvement from initial solution (full gap remains)
    
    Args:
        experiment: A ProblemSolver object that has been post-normalized.
        
    Returns:
        List of optimality gap fractions, one per macroreplication.
        Values close to 0.0 indicate the solution is near optimal.
    """
    if not experiment.has_postnormalized:
        raise ValueError(
            "Experiment must be post-normalized to compute optimality gap fraction. "
            "Call experiment.post_normalize() first."
        )
    
    # Get initial and optimal objective values
    # Avoid using truth-value checks on numpy arrays (they raise ambiguous truth value errors).
    x0_postreps = getattr(experiment, "x0_postreps", None)
    xstar_postreps = getattr(experiment, "xstar_postreps", None)

    x0_obj = float(np.mean(x0_postreps)) if (x0_postreps is not None and len(x0_postreps) > 0) else None
    xstar_obj = float(np.mean(xstar_postreps)) if (xstar_postreps is not None and len(xstar_postreps) > 0) else None
    
    if x0_obj is None or xstar_obj is None:
        raise ValueError("Initial and optimal objective estimates not available.")
    
    gap_fractions = []
    
    for mrep in range(experiment.n_macroreps):
        if len(experiment.all_est_objectives[mrep]) > 0:
            terminal_obj = experiment.all_est_objectives[mrep][-1]
            
            # Compute remaining optimality gap as a fraction
            # For minimization: gap = (f(x_terminal) - f(x*)) / (f(x0) - f(x*))
            # 0.0 = optimal, 1.0 = no improvement
            total_gap = x0_obj - xstar_obj
            
            if abs(total_gap) < 1e-10:
                # If x0 and x* are essentially the same, gap is 0 (already optimal)
                gap_fraction = 0.0
            else:
                remaining_gap = terminal_obj - xstar_obj
                gap_fraction = remaining_gap / total_gap
                # Clamp to [0, 1.5] range (can be negative if terminal is better than x*)
                gap_fraction = max(gap_fraction, 0.0)
            
            gap_fractions.append(gap_fraction)
    
    return gap_fractions


def extract_fraction_closed(
    experiment: "ProblemSolver",
) -> list[float]:
    """
    Extract the fraction of the optimality gap that has been closed for each macroreplication.

    Fraction closed is defined for minimization as:
        (f(x0) - f(x_t)) / max(f(x0) - f(x*), eps)
    where x0 is the initial solution, x_t is the terminal solution, and x* is the best known solution.

    Returns values in [0, 1] (clamped). 1.0 indicates the initial gap has been fully closed.

    Args:
        experiment: A ProblemSolver object that has been post-normalized.

    Returns:
        List of fraction-closed values, one per macroreplication.
    """
    if not experiment.has_postnormalized:
        raise ValueError(
            "Experiment must be post-normalized to compute fraction closed. "
            "Call experiment.post_normalize() first."
        )

    x0_postreps = getattr(experiment, "x0_postreps", None)
    xstar_postreps = getattr(experiment, "xstar_postreps", None)

    x0_obj = float(np.mean(x0_postreps)) if (x0_postreps is not None and len(x0_postreps) > 0) else None
    xstar_obj = float(np.mean(xstar_postreps)) if (xstar_postreps is not None and len(xstar_postreps) > 0) else None

    if x0_obj is None or xstar_obj is None:
        raise ValueError("Initial and optimal objective estimates not available.")

    eps = 1e-12
    denom = x0_obj - xstar_obj

    fraction_list = []
    for mrep in range(experiment.n_macroreps):
        if len(experiment.all_est_objectives[mrep]) > 0:
            terminal_obj = experiment.all_est_objectives[mrep][-1]

            if abs(denom) < eps:
                # If initial and best are essentially the same, consider gap closed
                frac = 1.0
            else:
                closed = (x0_obj - terminal_obj) / denom
                # Clamp to [0, 1]
                frac = min(max(closed, 0.0), 1.0)

            fraction_list.append(float(frac))

    return fraction_list


def extract_solver_factor_value(
    experiment: "ProblemSolver",
    factor_name: str,
) -> Any:
    """
    Extract the value of a specific solver factor from an experiment.
    
    Args:
        experiment: A ProblemSolver object.
        factor_name: Name of the solver factor to extract.
        
    Returns:
        The value of the specified factor.
    """
    return experiment.solver.factors.get(factor_name)


def extract_convergence_rate(
    experiment: "ProblemSolver",
    method: Literal["linear_fit", "exponential_fit", "area_ratio", "budget_to_threshold"] = "area_ratio",
    threshold_fraction: float = 0.9,
) -> list[float]:
    """
    Extract convergence rate for each macroreplication.
    
    The convergence rate measures how quickly the solver improves over the budget.
    Multiple methods are available:
    
    - "linear_fit": Fits a linear regression to log(gap) vs budget and returns
      the negative slope (higher = faster convergence).
    - "exponential_fit": Fits an exponential decay model and returns the decay rate.
    - "area_ratio": Computes the ratio of area under the progress curve to the
      maximum possible area (1.0 = instant convergence, 0.0 = no improvement).
    - "budget_to_threshold": Returns the fraction of budget used to reach a
      threshold fraction of the total improvement (lower = faster convergence).
    
    Args:
        experiment: A ProblemSolver object that has been post-replicated.
        method: The method to compute convergence rate.
        threshold_fraction: For "budget_to_threshold", the fraction of total
            improvement to use as threshold (default 0.9 = 90%).
            
    Returns:
        List of convergence rates, one per macroreplication.
        Interpretation depends on method:
        - linear_fit: Higher values = faster convergence
        - exponential_fit: Higher values = faster convergence
        - area_ratio: Higher values (closer to 1) = faster convergence
        - budget_to_threshold: Lower values = faster convergence (fraction of budget)
    """
    if not experiment.has_postreplicated:
        raise ValueError(
            "Experiment must be post-replicated to compute convergence rate. "
            "Call experiment.post_replicate() first."
        )
    
    convergence_rates = []
    
    for mrep in range(experiment.n_macroreps):
        est_objectives = experiment.all_est_objectives[mrep]
        budgets = experiment.all_intermediate_budgets[mrep]
        
        if len(est_objectives) < 2:
            continue
        
        est_objectives = np.array(est_objectives)
        budgets = np.array(budgets)
        
        # Get initial and final values
        initial_obj = est_objectives[0]
        final_obj = est_objectives[-1]
        total_budget = budgets[-1]
        
        # Total improvement (for minimization, this is positive if we improved)
        total_improvement = initial_obj - final_obj
        
        if abs(total_improvement) < 1e-10:
            # No improvement - assign a default rate
            if method == "budget_to_threshold":
                convergence_rates.append(1.0)  # Never reached threshold
            else:
                convergence_rates.append(0.0)  # Zero convergence rate
            continue
        
        if method == "linear_fit":
            # Fit linear regression to normalized gap vs budget fraction
            # gap(t) = (f(x_t) - f_final) / (f_initial - f_final)
            gaps = (est_objectives - final_obj) / total_improvement
            gaps = np.maximum(gaps, 1e-10)  # Avoid log(0)
            budget_fractions = budgets / total_budget
            
            # Linear fit to log(gap) vs budget_fraction
            # log(gap) = a - rate * budget_fraction
            try:
                coeffs = np.polyfit(budget_fractions, np.log(gaps), 1)
                rate = -coeffs[0]  # Negative slope = positive convergence rate
                convergence_rates.append(max(rate, 0.0))
            except (np.linalg.LinAlgError, ValueError):
                convergence_rates.append(0.0)
                
        elif method == "exponential_fit":
            # Similar to linear_fit but directly compute exponential decay rate
            # Assumes gap(t) = gap(0) * exp(-rate * t)
            gaps = (est_objectives - final_obj) / total_improvement
            gaps = np.maximum(gaps, 1e-10)
            budget_fractions = budgets / total_budget
            
            try:
                # Use log-linear regression
                log_gaps = np.log(gaps)
                coeffs = np.polyfit(budget_fractions, log_gaps, 1)
                rate = -coeffs[0]
                convergence_rates.append(max(rate, 0.0))
            except (np.linalg.LinAlgError, ValueError):
                convergence_rates.append(0.0)
                
        elif method == "area_ratio":
            # Compute normalized area under the progress curve
            # Progress = (f_initial - f(x_t)) / (f_initial - f_final)
            # Ranges from 0 (at start) to 1 (at end)
            progress = (initial_obj - est_objectives) / total_improvement
            progress = np.clip(progress, 0.0, 1.0)
            
            # Compute area under progress curve using trapezoidal rule
            budget_fractions = budgets / total_budget
            area = np.trapz(progress, budget_fractions)
            
            # Maximum possible area is 1.0 (instant convergence)
            # Minimum area for linear convergence would be 0.5
            convergence_rates.append(float(area))
            
        elif method == "budget_to_threshold":
            # Find the fraction of budget needed to reach threshold_fraction of improvement
            progress = (initial_obj - est_objectives) / total_improvement
            budget_fractions = budgets / total_budget
            
            # Find first index where progress >= threshold_fraction
            threshold_indices = np.where(progress >= threshold_fraction)[0]
            
            if len(threshold_indices) > 0:
                first_idx = threshold_indices[0]
                budget_to_reach = budget_fractions[first_idx]
            else:
                # Never reached threshold
                budget_to_reach = 1.0
            
            convergence_rates.append(float(budget_to_reach))
            
        else:
            raise ValueError(f"Unknown convergence rate method: {method}")
    
    return convergence_rates


# ============================================================================
# MAIN PLOTTING FUNCTION
# ============================================================================

def plot_datafarming_results(
    experiments: list["ProblemSolver"] | "ProblemsSolvers",
    factor_name: str,
    y_metric: Literal["terminal_objective", "optimality_gap_fraction", "fraction_closed", "convergence_rate"] = "terminal_objective",
    convergence_method: Literal["linear_fit", "exponential_fit", "area_ratio", "budget_to_threshold"] = "area_ratio",
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    y_scale: Literal["linear", "log"] = "linear",
    figsize: tuple[float, float] = (10, 6),
    scatter_alpha: float = 0.5,
    scatter_size: float = 50,
    scatter_color: str = "steelblue",
    trendline_color: str = "darkred",
    trendline_width: float = 2.0,
    show_confidence_interval: bool = True,
    confidence_level: float = 0.95,
    show_mean_markers: bool = True,
    mean_marker_size: float = 100,
    mean_marker_color: str = "red",
    save_path: str | Path | None = None,
    show_plot: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot data farming results with factor values on x-axis and performance metric on y-axis.
    
    Creates a scatter plot where each point represents a macroreplication result,
    with a trendline (based on mean post-replication values) passing through.
    
    Args:
        experiments: Either a list of ProblemSolver objects or a ProblemsSolvers object
            containing the data farming experiment results.
        factor_name: The name of the solver factor to plot on the x-axis
            (e.g., "initial subspace dimension", "polynomial basis").
        y_metric: What to plot on the y-axis:
            - "terminal_objective": The terminal objective value
            - "optimality_gap_fraction": Remaining optimality gap (0 = optimal, 1 = no improvement)
            - "convergence_rate": Rate of convergence (method depends on convergence_method)
        convergence_method: Method to compute convergence rate (only used if y_metric="convergence_rate"):
            - "area_ratio": Area under progress curve (higher = faster, 0-1 scale)
            - "linear_fit": Slope of log-linear fit (higher = faster)
            - "exponential_fit": Exponential decay rate (higher = faster)
            - "budget_to_threshold": Fraction of budget to reach 90% improvement (lower = faster)
        title: Plot title. If None, a default title is generated.
        xlabel: X-axis label. If None, the factor_name is used.
        ylabel: Y-axis label. If None, a default based on y_metric is used.
        figsize: Figure size as (width, height).
        scatter_alpha: Transparency of scatter points (0-1).
        scatter_size: Size of scatter points.
        scatter_color: Color of scatter points.
        trendline_color: Color of the trendline.
        trendline_width: Width of the trendline.
        show_confidence_interval: Whether to show confidence interval around trendline.
        confidence_level: Confidence level for the interval (e.g., 0.95 for 95%).
        show_mean_markers: Whether to show markers at mean values for each factor level.
        mean_marker_size: Size of mean markers.
        mean_marker_color: Color of mean markers.
        save_path: Path to save the figure. If None, figure is not saved.
        show_plot: Whether to display the plot.
        
    Returns:
        Tuple of (Figure, Axes) matplotlib objects.
        
    Example:
        >>> from simopt.experiment_base import read_experiment_results
        >>> experiments = [read_experiment_results(f"exp_{i}.pickle") for i in range(1, 9)]
        >>> fig, ax = plot_datafarming_results(
        ...     experiments,
        ...     factor_name="initial subspace dimension",
        ...     y_metric="optimality_gap_fraction",
        ...     title="ASTROMoRF Performance vs Subspace Dimension"
        ... )
    """
    # Handle ProblemsSolvers input
    if hasattr(experiments, 'experiments'):
        # Flatten the nested list structure of ProblemsSolvers
        experiment_list = []
        for solver_exps in experiments.experiments:
            experiment_list.extend(solver_exps)
    else:
        experiment_list = experiments
    
    # Extract data for plotting
    factor_values = []
    y_values = []
    factor_to_y = {}  # For computing means per factor level
    # For terminal_objective we want series per problem
    groups_by_problem: dict[str, dict[Any, list[float]]] = {}
    
    for exp in experiment_list:
        # Get factor value for this experiment
        factor_val = extract_solver_factor_value(exp, factor_name)
        
        if factor_val is None:
            continue
        
        # Handle PolyBasisType enum
        if isinstance(factor_val, PolyBasisType):
            factor_val_display = POLY_BASIS_NAMES.get(factor_val, str(factor_val))
        else:
            factor_val_display = factor_val
        
        # Extract y-values for each macroreplication
        if y_metric == "terminal_objective":
            mrep_values = extract_terminal_objectives(exp)
        elif y_metric == "optimality_gap_fraction":
            try:
                mrep_values = extract_optimality_gap_fraction(exp)
            except ValueError as e:
                print(f"Warning: Could not compute optimality gap fraction: {e}")
                continue
        elif y_metric == "fraction_closed":
            try:
                mrep_values = extract_fraction_closed(exp)
            except ValueError as e:
                print(f"Warning: Could not compute fraction closed: {e}")
                continue
        elif y_metric == "convergence_rate":
            try:
                mrep_values = extract_convergence_rate(exp, method=convergence_method)
            except ValueError as e:
                print(f"Warning: Could not compute convergence rate: {e}")
                continue
        else:
            raise ValueError(f"Unknown y_metric: {y_metric}")
        
        # Add data points
        problem_name = exp.problem.name
        if problem_name not in groups_by_problem:
            groups_by_problem[problem_name] = {}
        if factor_val_display not in groups_by_problem[problem_name]:
            groups_by_problem[problem_name][factor_val_display] = []

        if y_metric == "terminal_objective":
            # Aggregate terminal objective per Problem-Solver pair (one point per experiment)
            if len(mrep_values) == 0:
                continue
            exp_val = float(np.mean(mrep_values))
            groups_by_problem[problem_name][factor_val_display].append(exp_val)

            # Also track for legacy single-series behavior if needed
            factor_values.append(factor_val_display)
            y_values.append(exp_val)
            if factor_val_display not in factor_to_y:
                factor_to_y[factor_val_display] = []
            factor_to_y[factor_val_display].append(exp_val)
        else:
            # For other metrics, keep per-macrorep points and group by problem
            for val in mrep_values:
                groups_by_problem[problem_name][factor_val_display].append(val)
                factor_values.append(factor_val_display)
                y_values.append(val)

            # Track for mean computation
            if factor_val_display not in factor_to_y:
                factor_to_y[factor_val_display] = []
            factor_to_y[factor_val_display].extend(mrep_values)
    
    if not factor_values:
        raise ValueError("No valid data points found in experiments.")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine if factor values are numeric or categorical
    try:
        numeric_factors = [float(fv) for fv in factor_values]
        is_numeric = True
    except (ValueError, TypeError):
        is_numeric = False
    
    if groups_by_problem:
        # Plot one series per problem for the selected metric
        colors = plt.cm.tab10(np.linspace(0, 1, len(groups_by_problem)))

        if is_numeric:
            # Numeric factor: plot per-problem using only that problem's factor levels
            for (problem_name, grp), color in zip(groups_by_problem.items(), colors):
                # Attempt to interpret group keys as numeric factors
                keys = list(grp.keys())
                # Try converting keys to floats; fall back to original order if conversion fails
                try:
                    numeric_keys = sorted([float(k) for k in keys])
                    key_map = {float(k): k for k in keys}
                except Exception:
                    # Keys may already be numeric or non-convertible strings
                    try:
                        numeric_keys = sorted(keys, key=lambda k: float(k))
                    except Exception:
                        numeric_keys = sorted(keys, key=str)
                x_pts = []
                y_pts = []
                means = []
                stds = []
                ns = []

                for k in numeric_keys:
                    # obtain vals from grp using original key if necessary
                    vals = grp.get(k)
                    if vals is None:
                        vals = grp.get(str(k), [])
                    if vals is None:
                        vals = []
                    if vals:
                        x_val = float(k)
                        x_pts.extend([x_val] * len(vals))
                        y_pts.extend(vals)
                        means.append(np.mean(vals))
                        stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0)
                        ns.append(len(vals))
                    else:
                        # Skip missing factor levels for this problem to avoid NaN gaps
                        continue

                if not x_pts:
                    continue

                ax.scatter(
                    x_pts, y_pts,
                    alpha=scatter_alpha,
                    s=scatter_size,
                    c=[color],
                    label=f"{problem_name} (pts)",
                    zorder=2,
                )

                # Plot trendline through means for available keys only
                x_mean_vals = [float(k) for k in numeric_keys[:len(means)]]
                ax.plot(
                    x_mean_vals, means,
                    color=color,
                    linewidth=trendline_width,
                    label=f"{problem_name} (mean)",
                    zorder=3,
                )

                if show_mean_markers:
                    ax.scatter(
                        x_mean_vals, means,
                        s=mean_marker_size,
                        c=[color],
                        marker='D',
                        edgecolors='black',
                        linewidths=1,
                        zorder=4,
                    )

                if show_confidence_interval and ns:
                    ci_multiplier = stats.t.ppf((1 + confidence_level) / 2, df=np.maximum(np.array(ns) - 1, 1))
                    ci_halfwidth = ci_multiplier * np.array(stds) / np.sqrt(np.maximum(np.array(ns), 1))
                    ax.fill_between(
                        x_mean_vals,
                        np.array(means) - ci_halfwidth,
                        np.array(means) + ci_halfwidth,
                        alpha=0.15,
                        color=color,
                        zorder=1,
                    )

        else:
            # Categorical factor - common categories across problems
            all_factors = set()
            for grp in groups_by_problem.values():
                all_factors.update(grp.keys())
            unique_factors = sorted(list(all_factors), key=str)
            x_positions = list(range(len(unique_factors)))
            factor_to_pos = {fv: i for i, fv in enumerate(unique_factors)}

            for (problem_name, grp), color in zip(groups_by_problem.items(), colors):
                means = []
                stds = []
                ns = []
                for fv in unique_factors:
                    vals = grp.get(fv, [])
                    if vals:
                        means.append(np.mean(vals))
                        stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0)
                        ns.append(len(vals))
                    else:
                        means.append(np.nan)
                        stds.append(np.nan)
                        ns.append(0)

                # scatter raw points with jitter
                x_pts = []
                y_pts = []
                for fv, pos in factor_to_pos.items():
                    vals = grp.get(fv, [])
                    for v in vals:
                        jitter = np.random.uniform(-0.2, 0.2)
                        x_pts.append(pos + jitter)
                        y_pts.append(v)

                ax.scatter(
                    x_pts, y_pts,
                    alpha=scatter_alpha,
                    s=scatter_size,
                    c=[color],
                    label=f"{problem_name} (pts)",
                    zorder=2,
                )

                ax.plot(
                    x_positions, means,
                    color=color,
                    linewidth=trendline_width,
                    label=f"{problem_name} (mean)",
                    zorder=3,
                )

                if show_mean_markers:
                    ax.scatter(
                        x_positions, means,
                        s=mean_marker_size,
                        c=[color],
                        marker='D',
                        edgecolors='black',
                        linewidths=1,
                        zorder=4,
                    )

                if show_confidence_interval:
                    ci_multiplier = stats.t.ppf((1 + confidence_level) / 2, df=np.maximum(np.array(ns) - 1, 1))
                    ci_halfwidth = ci_multiplier * np.array(stds) / np.sqrt(np.maximum(np.array(ns), 1))
                    ax.fill_between(
                        x_positions,
                        np.array(means) - ci_halfwidth,
                        np.array(means) + ci_halfwidth,
                        alpha=0.15,
                        color=color,
                        zorder=1,
                    )

            ax.set_xticks(x_positions)
            ax.set_xticklabels(unique_factors, rotation=45, ha='right')

        # Labels and return (we're done for grouped-by-problem plotting)
        if title is None:
            title = f"Data Farming Results: {factor_name}"
        if xlabel is None:
            xlabel = factor_name.title()
        if ylabel is None:
            if y_metric == "terminal_objective":
                ylabel = "Terminal Objective Value"
            elif y_metric == "optimality_gap_fraction":
                ylabel = "Optimality Gap"
            elif y_metric == "fraction_closed":
                ylabel = "Fraction of Optimality Gap Closed"
            elif y_metric == "convergence_rate":
                if convergence_method == "area_ratio":
                    ylabel = "Convergence Rate (Area Under Progress Curve)"
                elif convergence_method == "budget_to_threshold":
                    ylabel = "Fraction of Budget to Reach 90% Improvement"
                else:
                    ylabel = "Convergence Rate (Higher = Faster)"

        # Apply y-scale (log or linear). If log requested but data contains non-positive
        # values, fall back to 'symlog' and warn the user.
        if y_scale == "log":
            try:
                min_y = float(np.nanmin(np.array(y_values, dtype=float)))
            except Exception:
                min_y = None
            if min_y is None:
                ax.set_yscale("log")
            elif min_y > 0:
                ax.set_yscale("log")
            else:
                print(
                    "Warning: data contains non-positive values; using 'symlog' fallback for y-axis scaling."
                )
                linthresh = max(1e-8, abs(min_y))
                ax.set_yscale("symlog", linthresh=linthresh)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        if show_plot:
            plt.show()
        return fig, ax

    if is_numeric:
        # Numeric factor - plot as scatter with regression line
        x_plot = np.array(numeric_factors)
        y_plot = np.array(y_values)
        
        # Scatter plot of individual macroreplications
        ax.scatter(
            x_plot, y_plot,
            alpha=scatter_alpha,
            s=scatter_size,
            c=scatter_color,
            label="Macroreplication results",
            zorder=2,
        )
        
        # Compute means per factor level for trendline
        unique_factors = sorted(set(numeric_factors))
        means = []
        stds = []
        ns = []
        
        for fv in unique_factors:
            vals = factor_to_y.get(fv, [])
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0)
                ns.append(len(vals))
            else:
                means.append(np.nan)
                stds.append(np.nan)
                ns.append(0)
        
        means = np.array(means)
        stds = np.array(stds)
        ns = np.array(ns)
        
        # Plot trendline through means
        ax.plot(
            unique_factors, means,
            color=trendline_color,
            linewidth=trendline_width,
            label="Mean trend",
            zorder=3,
        )
        
        # Show mean markers
        if show_mean_markers:
            ax.scatter(
                unique_factors, means,
                s=mean_marker_size,
                c=mean_marker_color,
                marker='D',
                edgecolors='black',
                linewidths=1,
                label="Mean",
                zorder=4,
            )
        
        # Show confidence interval
        if show_confidence_interval:
            # Compute confidence interval using t-distribution
            ci_multiplier = stats.t.ppf((1 + confidence_level) / 2, df=np.maximum(ns - 1, 1))
            ci_halfwidth = ci_multiplier * stds / np.sqrt(np.maximum(ns, 1))
            
            ax.fill_between(
                unique_factors,
                means - ci_halfwidth,
                means + ci_halfwidth,
                alpha=0.2,
                color=trendline_color,
                label=f"{int(confidence_level*100)}% CI",
                zorder=1,
            )
        
    else:
        # Categorical factor - use box plot or grouped scatter
        unique_factors = sorted(set(factor_values), key=str)
        x_positions = list(range(len(unique_factors)))
        factor_to_pos = {fv: i for i, fv in enumerate(unique_factors)}
        
        # Scatter plot with jitter
        x_jittered = []
        for fv in factor_values:
            pos = factor_to_pos[fv]
            jitter = np.random.uniform(-0.2, 0.2)
            x_jittered.append(pos + jitter)
        
        ax.scatter(
            x_jittered, y_values,
            alpha=scatter_alpha,
            s=scatter_size,
            c=scatter_color,
            label="Macroreplication results",
            zorder=2,
        )
        
        # Compute means for trendline
        means = []
        stds = []
        ns = []
        
        for fv in unique_factors:
            vals = factor_to_y.get(fv, [])
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0)
                ns.append(len(vals))
            else:
                means.append(np.nan)
                stds.append(np.nan)
                ns.append(0)
        
        means = np.array(means)
        stds = np.array(stds)
        ns = np.array(ns)
        
        # Plot trendline
        ax.plot(
            x_positions, means,
            color=trendline_color,
            linewidth=trendline_width,
            label="Mean trend",
            zorder=3,
        )
        
        # Mean markers
        if show_mean_markers:
            ax.scatter(
                x_positions, means,
                s=mean_marker_size,
                c=mean_marker_color,
                marker='D',
                edgecolors='black',
                linewidths=1,
                label="Mean",
                zorder=4,
            )
        
        # Confidence interval
        if show_confidence_interval:
            ci_multiplier = stats.t.ppf((1 + confidence_level) / 2, df=np.maximum(ns - 1, 1))
            ci_halfwidth = ci_multiplier * stds / np.sqrt(np.maximum(ns, 1))
            
            ax.fill_between(
                x_positions,
                means - ci_halfwidth,
                means + ci_halfwidth,
                alpha=0.2,
                color=trendline_color,
                label=f"{int(confidence_level*100)}% CI",
                zorder=1,
            )
        
        # Set categorical x-axis labels
        ax.set_xticks(x_positions)
        ax.set_xticklabels(unique_factors, rotation=45, ha='right')
    
    # Labels and title
    if title is None:
        title = f"Data Farming Results: {factor_name}"
    if xlabel is None:
        xlabel = factor_name.title()
    if ylabel is None:
        if y_metric == "terminal_objective":
            ylabel = "Terminal Objective Value"
        elif y_metric == "optimality_gap_fraction":
            ylabel = "Optimality Gap"
        elif y_metric == "fraction_closed":
            ylabel = "Fraction of Optimality Gap Closed"
        elif y_metric == "convergence_rate":
            if convergence_method == "area_ratio":
                ylabel = "Convergence Rate (Area Under Progress Curve)"
            elif convergence_method == "budget_to_threshold":
                ylabel = "Budget Fraction to 90% Improvement (Lower = Faster)"
            else:
                ylabel = "Convergence Rate (Higher = Faster)"
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    if show_plot:
        plt.show()
    
    return fig, ax


def plot_datafarming_comparison(
    experiments: list["ProblemSolver"] | "ProblemsSolvers",
    factor_name: str,
    group_by: str | None = None,
    y_metric: Literal["terminal_objective", "optimality_gap_fraction", "fraction_closed"] = "terminal_objective",
    title: str | None = None,
    figsize: tuple[float, float] = (12, 6),
    save_path: str | Path | None = None,
    show_plot: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot data farming results with multiple groups (e.g., different problems).
    
    Creates a grouped comparison plot showing how performance varies with
    factor values across different groups.
    
    Args:
        experiments: List of ProblemSolver objects or ProblemsSolvers object.
        factor_name: The solver factor to plot on x-axis.
        group_by: Factor to group by (e.g., problem name). If None, all data
            is plotted together.
        y_metric: Performance metric for y-axis.
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save the figure.
        show_plot: Whether to display the plot.
        
    Returns:
        Tuple of (Figure, Axes) matplotlib objects.
    """
    # Handle ProblemsSolvers input
    if hasattr(experiments, 'experiments'):
        experiment_list = []
        for solver_exps in experiments.experiments:
            experiment_list.extend(solver_exps)
    else:
        experiment_list = experiments
    
    # Group experiments
    if group_by == "problem":
        groups = {}
        for exp in experiment_list:
            problem_name = exp.problem.name
            if problem_name not in groups:
                groups[problem_name] = []
            groups[problem_name].append(exp)
    else:
        groups = {"All": experiment_list}
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
    
    for (group_name, group_exps), color in zip(groups.items(), colors):
        # Extract data for this group
        factor_to_y = {}
        
        for exp in group_exps:
            factor_val = extract_solver_factor_value(exp, factor_name)
            if factor_val is None:
                continue
            
            if isinstance(factor_val, PolyBasisType):
                factor_val = POLY_BASIS_NAMES.get(factor_val, str(factor_val))
            
            if y_metric == "terminal_objective":
                mrep_values = extract_terminal_objectives(exp)
            elif y_metric == "optimality_gap_fraction":
                try:
                    mrep_values = extract_optimality_gap_fraction(exp)
                except ValueError:
                    continue
            elif y_metric == "fraction_closed":
                try:
                    mrep_values = extract_fraction_closed(exp)
                except ValueError:
                    continue
            else:
                # Fallback - treat as terminal objective
                mrep_values = extract_terminal_objectives(exp)
            
            if factor_val not in factor_to_y:
                factor_to_y[factor_val] = []
            factor_to_y[factor_val].extend(mrep_values)
        
        if not factor_to_y:
            continue
        
        # Sort factor values
        try:
            sorted_factors = sorted(factor_to_y.keys(), key=float)
            x_vals = [float(f) for f in sorted_factors]
        except (ValueError, TypeError):
            sorted_factors = sorted(factor_to_y.keys(), key=str)
            x_vals = list(range(len(sorted_factors)))
        
        means = [np.mean(factor_to_y[f]) for f in sorted_factors]
        stds = [np.std(factor_to_y[f], ddof=1) if len(factor_to_y[f]) > 1 else 0 
                for f in sorted_factors]
        
        # Plot mean with error bars
        ax.errorbar(
            x_vals, means,
            yerr=stds,
            label=group_name,
            color=color,
            marker='o',
            linewidth=2,
            capsize=5,
        )
    
    # Labels
    if title is None:
        title = f"Performance Comparison: {factor_name}"
    if y_metric == "terminal_objective":
        ylabel = "Terminal Objective"
    elif y_metric == "fraction_closed":
        ylabel = "Fraction of Optimality Gap Closed"
    else:
        ylabel = "Optimality Gap Fraction"
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(factor_name.title(), fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    if show_plot:
        plt.show()
    
    return fig, ax


def create_results_dataframe(
    experiments: list["ProblemSolver"] | "ProblemsSolvers",
    factor_names: list[str],
) -> pd.DataFrame:
    """
    Create a pandas DataFrame from experiment results for analysis.
    
    Args:
        experiments: List of ProblemSolver objects or ProblemsSolvers object.
        factor_names: List of solver factor names to include as columns.
        
    Returns:
        DataFrame with columns for factors, problem, macroreplication, and metrics.
    """
    # Handle ProblemsSolvers input
    if hasattr(experiments, 'experiments'):
        experiment_list = []
        for solver_exps in experiments.experiments:
            experiment_list.extend(solver_exps)
    else:
        experiment_list = experiments
    
    rows = []
    
    for exp in experiment_list:
        # Extract factor values
        factor_vals = {}
        for factor_name in factor_names:
            val = extract_solver_factor_value(exp, factor_name)
            if isinstance(val, PolyBasisType):
                val = POLY_BASIS_NAMES.get(val, str(val))
            factor_vals[factor_name] = val
        
        # Get terminal objectives
        terminal_objs = extract_terminal_objectives(exp)
        
        # Get optimality gap fractions if available
        try:
            gap_fractions = extract_optimality_gap_fraction(exp)
        except ValueError:
            gap_fractions = [None] * len(terminal_objs)

        # Get fraction closed if available
        try:
            frac_closed = extract_fraction_closed(exp)
        except ValueError:
            frac_closed = [None] * len(terminal_objs)
        
        # Get convergence rates if available
        try:
            conv_rates_area = extract_convergence_rate(exp, method="area_ratio")
        except ValueError:
            conv_rates_area = [None] * len(terminal_objs)
        
        try:
            conv_rates_budget = extract_convergence_rate(exp, method="budget_to_threshold")
        except ValueError:
            conv_rates_budget = [None] * len(terminal_objs)
        
        # Create row for each macroreplication
        for mrep, (obj, gap, frac, conv_area, conv_budget) in enumerate(
            zip(terminal_objs, gap_fractions, frac_closed, conv_rates_area, conv_rates_budget)
        ):
            row = {
                "problem": exp.problem.name,
                "solver": exp.solver.name,
                "macroreplication": mrep + 1,
                "terminal_objective": obj,
                "optimality_gap_fraction": gap,
                "fraction_closed": frac,
                "convergence_rate_area": conv_area,
                "convergence_rate_budget_to_90pct": conv_budget,
                **factor_vals,
            }
            rows.append(row)
    
    return pd.DataFrame(rows)


# ============================================================================
# CONVENIENCE FUNCTIONS FOR LOADING RESULTS
# ============================================================================

def load_experiments_from_pickles(
    pickle_paths: list[str | Path],
) -> list["ProblemSolver"]:
    """
    Load ProblemSolver objects from pickle files.
    
    Args:
        pickle_paths: List of paths to pickle files.
        
    Returns:
        List of ProblemSolver objects.
    """
    from simopt.experiment_base import read_experiment_results
    
    experiments = []
    for path in pickle_paths:
        try:
            exp = read_experiment_results(str(path))
            experiments.append(exp)
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")
    
    return experiments


def load_experiments_from_directory(
    directory: str | Path,
    pattern: str = "*.pickle",
) -> list["ProblemSolver"]:
    """
    Load all ProblemSolver objects from pickle files in a directory.
    
    Args:
        directory: Path to directory containing pickle files.
        pattern: Glob pattern to match pickle files.
        
    Returns:
        List of ProblemSolver objects.
    """
    directory = Path(directory)
    pickle_files = list(directory.glob(pattern))
    
    if not pickle_files:
        print(f"No files matching '{pattern}' found in {directory}")
        return []
    
    print(f"Found {len(pickle_files)} pickle files")
    return load_experiments_from_pickles(pickle_files)


# ============================================================================
# MAIN / CLI
# ============================================================================

def main():
    """Command-line interface for plotting data farming results."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Plot data farming experiment results",
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Directory containing experiment pickle files",
    )
    parser.add_argument(
        "--factor",
        type=str,
        required=True,
        help="Solver factor name to plot on x-axis",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["terminal_objective", "optimality_gap_fraction", "fraction_closed"],
        default="terminal_objective",
        help="Metric to plot on y-axis",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Plot title",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the figure",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display the plot (just save)",
    )
    
    args = parser.parse_args()
    
    # Load experiments
    experiments = load_experiments_from_directory(args.directory)
    
    if not experiments:
        print("No experiments loaded. Exiting.")
        return
    
    # Create plot
    plot_datafarming_results(
        experiments=experiments,
        factor_name=args.factor,
        y_metric=args.metric,
        title=args.title,
        save_path=args.output,
        show_plot=not args.no_show,
    )


if __name__ == "__main__":
    main()
