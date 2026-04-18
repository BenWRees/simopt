"""Function estimates plot."""

import contextlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

import simopt.curve_utils as curve_utils
from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.curve import Curve
from simopt.experiment import ProblemSolver
from simopt.plot_type import PlotType

from .utils import (
    plot_bootstrap_conf_ints,
    save_plot,
    setup_plot,
)


def _print_max_halfwidth_caption(
    curve_pairs: list[list[Curve]],
    conf_level: float,
) -> None:
    """Print caption for max halfwidth positioned below x-axis label.

    Args:
        curve_pairs: List of [lower_bound_curve, upper_bound_curve] pairs.
        conf_level: Confidence level for the interval.
    """
    # Compute max halfwidth
    max_halfwidths = []
    for curve_pair in curve_pairs:
        max_halfwidths.append(
            0.5 * curve_utils.max_difference_of_curves(curve_pair[1], curve_pair[0])
        )
    max_halfwidth = max(max_halfwidths)

    # Format caption text
    boot_cis = round(conf_level * 100)
    max_hw_round = round(max_halfwidth, 2)
    txt = f"The max halfwidth of the bootstrap {boot_cis}% CIs is {max_hw_round}."

    # Position text centered below x-axis using axes coordinates
    ax = plt.gca()
    ax.text(
        0.5,
        -0.15,
        txt,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=plt.rcParams.get("axes.labelsize", 10),
    )


def _fn_estimates_to_curves(
    fn_estimates_list: list[list[float]],
    normalize: bool = False,
) -> list[Curve]:
    """Convert function estimates lists to Curve objects.

    Args:
        fn_estimates_list: List of function estimate lists from each macroreplication.
        normalize: If True, normalize iterations to 0-1 scale.

    Returns:
        List of Curve objects.
    """
    curves = []
    for fn_estimates in fn_estimates_list:
        n_iters = len(fn_estimates)
        x_vals = list(np.linspace(0, 1, n_iters)) if normalize else list(range(n_iters))
        curves.append(Curve(x_vals=x_vals, y_vals=fn_estimates))
    return curves


def _bootstrap_curves_conf_int(
    curves: list[Curve],
    n_bootstraps: int,
    conf_level: float,
    estimator: Curve,
) -> tuple[Curve, Curve]:
    """Compute bootstrap confidence intervals for mean of curves.

    Args:
        curves: List of Curve objects from each macroreplication.
        n_bootstraps: Number of bootstrap samples.
        conf_level: Confidence level for the interval (0 < conf_level < 1).
        estimator: The original mean curve estimator.

    Returns:
        Tuple of (lower_bound_curve, upper_bound_curve).
    """
    from scipy import stats

    # Create RNG for bootstrap sampling
    bootstrap_rng = MRG32k3a(s_ss_sss_index=[2, 0, 0])
    n_curves = len(curves)

    # Generate bootstrap samples and compute means
    bootstrap_means: list[Curve] = []
    for _ in range(n_bootstraps):
        # Sample with replacement
        bs_indices = bootstrap_rng.choices(range(n_curves), k=n_curves)
        bs_curves = [curves[i] for i in bs_indices]
        bootstrap_means.append(curve_utils.mean_of_curves(bs_curves))

    # Get unique x-values from the estimator
    unique_x_vals = list(estimator.x_vals)

    # Compute confidence intervals at each x-value
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []

    for x_val in unique_x_vals:
        # Get bootstrap values at this x
        bs_vals = [curve.lookup(x_val) for curve in bootstrap_means]
        bs_vals = [v for v in bs_vals if not np.isnan(v)]

        if len(bs_vals) == 0:
            lower_bounds.append(estimator.lookup(x_val))
            upper_bounds.append(estimator.lookup(x_val))
            continue

        original_val = estimator.lookup(x_val)

        # Bias correction factor
        bs_std = np.std(bs_vals)
        z0 = (np.percentile(bs_vals, 50) - original_val) / bs_std if bs_std > 0 else 0

        alpha = 1 - conf_level
        z_alpha_lower = stats.norm.ppf(alpha / 2)
        z_alpha_upper = stats.norm.ppf(1 - alpha / 2)

        p_lower = stats.norm.cdf(2 * z0 + z_alpha_lower) * 100
        p_upper = stats.norm.cdf(2 * z0 + z_alpha_upper) * 100

        # Clamp percentiles to valid range
        p_lower = max(0, min(100, p_lower))
        p_upper = max(0, min(100, p_upper))

        lower_bounds.append(float(np.percentile(bs_vals, p_lower)))
        upper_bounds.append(float(np.percentile(bs_vals, p_upper)))

    return Curve(x_vals=unique_x_vals, y_vals=lower_bounds), Curve(
        x_vals=unique_x_vals, y_vals=upper_bounds
    )


def plot_fn_estimates(
    experiments: list[ProblemSolver],
    plot_type: PlotType = PlotType.FN_ESTIMATES_ALL,
    all_in_one: bool = True,
    normalize: bool = False,
    y_normalize: bool = False,
    n_bootstraps: int = 100,
    conf_level: float = 0.95,
    plot_conf_ints: bool = True,
    print_max_hw: bool = True,
    log_y: bool = False,
    y_limits: tuple[float, float] | None = None,
    plot_title: str | None = None,
    legend_loc: str | None = None,
    ext: str = ".png",
    save_as_pickle: bool = False,
    solver_set_name: str = "SOLVER_SET",
) -> list[Path]:
    """Plots function estimates against iteration number.

    Args:
        experiments (list[ProblemSolver]): Problem-solver pairs for different solvers
            on the same problem.
        plot_type (PlotType, optional): Type of plot to produce. Options are:
            - FN_ESTIMATES_ALL: All function estimate curves from all macroreps.
            - FN_ESTIMATES_MEAN: Mean function estimate curve across macroreps.
            Defaults to FN_ESTIMATES_ALL.
        all_in_one (bool, optional): If True, plot all curves in one figure.
            Defaults to True.
        normalize (bool, optional): If True, normalize iterations to a 0-1 scale
            so that solvers with different iteration counts can be compared.
            Defaults to False.
        y_normalize (bool, optional): If True, normalize function estimates to
            fraction of initial optimality gap. Defaults to False.
        n_bootstraps (int, optional): Number of bootstrap samples. Defaults to 100.
        conf_level (float, optional): Confidence level for confidence intervals
            (must be in (0, 1)). Defaults to 0.95.
        plot_conf_ints (bool, optional): If True, plot confidence intervals around
            the mean (only applies when plot_type is FN_ESTIMATES_MEAN).
            Defaults to True.
        print_max_hw (bool, optional): If True, print max half-width in caption.
            Defaults to True.
        log_y (bool, optional): If True, use logarithmic scale for y-axis.
            Defaults to False.
        y_limits (tuple[float, float], optional): If provided, set y-axis limits
            to (ymin, ymax). Defaults to None.
        plot_title (str, optional): Custom title for the plot
            (used only if `all_in_one=True`).
        legend_loc (str, optional): Location of legend (e.g., "best", "lower right").
        ext (str, optional): File extension for saved plots (e.g., ".png").
            Defaults to ".png".
        save_as_pickle (bool, optional): If True, save plot as a pickle file.
            Defaults to False.
        solver_set_name (str, optional): Label for solver group in plot titles.
            Defaults to "SOLVER_SET".

    Returns:
        list[Path]: List of file paths where the plots were saved.

    Raises:
        ValueError: If an unsupported plot type is specified or parameters are invalid.
    """
    # Value checking
    if plot_type not in [PlotType.FN_ESTIMATES_ALL, PlotType.FN_ESTIMATES_MEAN]:
        error_msg = (
            "Plot type must be either 'FN_ESTIMATES_ALL' or 'FN_ESTIMATES_MEAN'."
        )
        raise ValueError(error_msg)

    if n_bootstraps < 1:
        raise ValueError("Number of bootstraps must be a positive integer.")

    if not 0 < conf_level < 1:
        raise ValueError("Confidence level must be in (0, 1).")

    if legend_loc is None:
        legend_loc = "best"

    # Check if problems are the same with the same x0 and x*.
    # check_common_problem_and_reference(experiments)
    file_list: list[Path] = []

    n_experiments = len(experiments)

    if all_in_one:
        ref_experiment = experiments[0]
        setup_plot(
            plot_type=plot_type,
            solver_name=solver_set_name,
            problem_name=ref_experiment.problem.name,
            budget=ref_experiment.problem.factors["budget"],
            plot_title=plot_title,
        )
        solver_handles = []
        curve_pairs: list[list[Curve]] = []

        for exp_idx in range(n_experiments):
            experiment = experiments[exp_idx]
            color_str = "C" + str(exp_idx)
            # Convert fn_estimates to Curve objects
            fn_curves = _fn_estimates_to_curves(
                experiment.all_fn_estimates, normalize=normalize
            )

            # Optionally normalize y-values to fraction of initial optimality gap
            if y_normalize:
                # Determine reference optimal objective f* (prefer postreps mean)
                try:
                    f_star = float(np.mean(experiment.xstar_postreps))
                except Exception:
                    # Fallback: use minimum observed value across curves
                    f_star = min(
                        (min(c.y_vals) for c in fn_curves if len(c.y_vals) > 0),
                        default=0.0,
                    )

                # Determine initial value f0 as mean of first observations
                first_vals = [c.y_vals[0] for c in fn_curves if len(c.y_vals) > 0]
                f0 = float(np.mean(first_vals)) if first_vals else 1.0

                denom = f0 - f_star
                # Build new Curve objects with transformed y-values (Curve is immutable)
                transformed_curves: list[Curve] = []
                for c in fn_curves:
                    if denom != 0:
                        new_y = [(y - f_star) / denom for y in c.y_vals]
                    else:
                        new_y = [y - f_star for y in c.y_vals]
                    transformed_curves.append(
                        Curve(x_vals=list(c.x_vals), y_vals=new_y)
                    )
                fn_curves = transformed_curves

            if plot_type == PlotType.FN_ESTIMATES_ALL:
                # Plot all function estimate curves from all macroreps
                handle = fn_curves[0].plot(color_str=color_str)
                for curve in fn_curves[1:]:
                    curve.plot(color_str=color_str)
                solver_handles.append(handle)

            elif plot_type == PlotType.FN_ESTIMATES_MEAN:
                # Compute and plot mean function estimates across macroreps
                estimator = curve_utils.mean_of_curves(fn_curves)
                handle = estimator.plot(color_str=color_str)
                solver_handles.append(handle)

                # Compute bootstrap confidence intervals if requested
                if (plot_conf_ints or print_max_hw) and len(fn_curves) > 1:
                    bs_conf_int_lb_curve, bs_conf_int_ub_curve = (
                        _bootstrap_curves_conf_int(
                            curves=fn_curves,
                            n_bootstraps=n_bootstraps,
                            conf_level=conf_level,
                            estimator=estimator,
                        )
                    )
                    if plot_conf_ints:
                        plot_bootstrap_conf_ints(
                            bs_conf_int_lb_curve,
                            bs_conf_int_ub_curve,
                            color_str=color_str,
                        )
                    if print_max_hw:
                        curve_pairs.append([bs_conf_int_lb_curve, bs_conf_int_ub_curve])

        leg = plt.legend(
            handles=solver_handles,
            labels=[experiment.solver.name for experiment in experiments],
            loc=legend_loc,
        )
        if leg is not None:
            try:
                leg.get_frame().set_alpha(0.4)
            except Exception:
                contextlib.suppress(Exception)
        # X-axis label
        if normalize:
            plt.xlabel("Percentage of the run")
        else:
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        # Y-axis label: use more descriptive label when normalized
        if y_normalize:
            plt.ylabel("Fraction from optimal solution")
        else:
            plt.ylabel("Function estimate")

        if log_y:
            plt.yscale("log")

        if y_limits is not None:
            plt.ylim(y_limits)

        # Apply tight layout before adding max halfwidth text
        plt.tight_layout()

        if print_max_hw and plot_type == PlotType.FN_ESTIMATES_MEAN and curve_pairs:
            _print_max_halfwidth_caption(
                curve_pairs=curve_pairs,
                conf_level=conf_level,
            )
            # Adjust bottom margin to fit the max halfwidth text snugly
            plt.gcf().subplots_adjust(bottom=0.18)

        file_list.append(
            save_plot(
                solver_name=solver_set_name,
                problem_name=ref_experiment.problem.name,
                plot_type=plot_type,
                normalize=normalize,
                plot_title=plot_title,
                ext=ext,
                save_as_pickle=save_as_pickle,
            )
        )
    else:
        # Plot separately for each experiment
        for experiment in experiments:
            setup_plot(
                plot_type=plot_type,
                solver_name=experiment.solver.name,
                problem_name=experiment.problem.name,
                budget=experiment.problem.factors["budget"],
            )
            # Convert fn_estimates to Curve objects
            fn_curves = _fn_estimates_to_curves(
                experiment.all_fn_estimates, normalize=normalize
            )

            if plot_type == PlotType.FN_ESTIMATES_ALL:
                # Plot all function estimate curves from all macroreps
                for curve in fn_curves:
                    curve.plot()

            elif plot_type == PlotType.FN_ESTIMATES_MEAN:
                # Compute and plot mean function estimates across macroreps
                estimator = curve_utils.mean_of_curves(fn_curves)
                estimator.plot()

                # Compute bootstrap confidence intervals if requested
                if (plot_conf_ints or print_max_hw) and len(fn_curves) > 1:
                    bs_conf_int_lb_curve, bs_conf_int_ub_curve = (
                        _bootstrap_curves_conf_int(
                            curves=fn_curves,
                            n_bootstraps=n_bootstraps,
                            conf_level=conf_level,
                            estimator=estimator,
                        )
                    )
                    if plot_conf_ints:
                        plot_bootstrap_conf_ints(
                            bs_conf_int_lb_curve, bs_conf_int_ub_curve
                        )
                    if print_max_hw:
                        # Apply tight layout before adding max halfwidth text
                        plt.tight_layout()
                        _print_max_halfwidth_caption(
                            curve_pairs=[[bs_conf_int_lb_curve, bs_conf_int_ub_curve]],
                            conf_level=conf_level,
                        )
                        # Adjust bottom margin to fit the max halfwidth text snugly
                        plt.gcf().subplots_adjust(bottom=0.18)

            if normalize:
                plt.xlabel("Percentage of the run")
            else:
                plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

            # Y-axis label for individual plots
            if y_normalize:
                plt.ylabel("Fraction from optimal solution")
            else:
                plt.ylabel("Function estimate")
            file_list.append(
                save_plot(
                    solver_name=experiment.solver.name,
                    problem_name=experiment.problem.name,
                    plot_type=plot_type,
                    normalize=normalize,
                    ext=ext,
                    save_as_pickle=save_as_pickle,
                )
            )

    return file_list
