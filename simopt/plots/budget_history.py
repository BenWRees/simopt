"""Budget history plot."""

#Todo: Change iterations to be normalized so that all solvers can be compared on same x-axis
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

import simopt.curve_utils as curve_utils
from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.curve import Curve
from simopt.experiment import ProblemSolver
from simopt.plot_type import PlotType

from .utils import (
    check_common_problem_and_reference,
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


def _budget_history_to_curves(
    budget_histories: list[list[float]],
    normalize: bool = False,
) -> list[Curve]:
    """Convert budget history lists to Curve objects.

    Args:
        budget_histories: List of budget history lists from each macroreplication.
        normalize: If True, normalize iterations to 0-1 scale.

    Returns:
        List of Curve objects.
    """
    curves = []
    for budget_history in budget_histories:
        n_iters = len(budget_history)
        if normalize:
            x_vals = list(np.linspace(0, 1, n_iters))
        else:
            x_vals = list(range(n_iters))
        curves.append(Curve(x_vals=x_vals, y_vals=budget_history))
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
        conf_level: Confidence level for the interval.
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
        if bs_std > 0:
            z0 = (np.percentile(bs_vals, 50) - original_val) / bs_std
        else:
            z0 = 0

        # Adjusted percentiles (BC method)
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
        if bs_std > 0:
            z0 = (np.percentile(bs_vals, 50) - original_val) / bs_std
        else:
            z0 = 0

        # Adjusted percentiles (BC method)
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


def plot_budget_history(
    experiments: list[ProblemSolver],
    plot_type: PlotType = PlotType.BUDGET_HISTORY_ALL,
    all_in_one: bool = True,
    normalize: bool = False,
    n_bootstraps: int = 100,
    conf_level: float = 0.95,
    plot_conf_ints: bool = True,
    print_max_hw: bool = True,
    plot_title: str | None = None,
    legend_loc: str | None = None,
    ext: str = ".png",
    save_as_pickle: bool = False,
    solver_set_name: str = "SOLVER_SET",
) -> list[Path]:
    """Plots budget history (cumulative budget used) against iteration number.

    Args:
        experiments (list[ProblemSolver]): Problem-solver pairs for different solvers
            on the same problem.
        plot_type (PlotType, optional): Type of plot to produce. Options are:
            - BUDGET_HISTORY_ALL: All budget history curves from all macroreps.
            - BUDGET_HISTORY_MEAN: Mean budget history curve across macroreps.
            Defaults to BUDGET_HISTORY_ALL.
        all_in_one (bool, optional): If True, plot all curves in one figure.
            Defaults to True.
        normalize (bool, optional): If True, normalize iterations to a 0-1 scale
            so that solvers with different iteration counts can be compared.
            Defaults to False.
        n_bootstraps (int, optional): Number of bootstrap samples. Defaults to 100.
        conf_level (float, optional): Confidence level for confidence intervals
            (must be in (0, 1)). Defaults to 0.95.
        plot_conf_ints (bool, optional): If True, plot confidence intervals around
            the mean (only applies when plot_type is BUDGET_HISTORY_MEAN).
            Defaults to True.
        print_max_hw (bool, optional): If True, print max half-width in caption.
            Defaults to True.
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
    if plot_type not in [PlotType.BUDGET_HISTORY_ALL, PlotType.BUDGET_HISTORY_MEAN]:
        error_msg = (
            "Plot type must be either 'BUDGET_HISTORY_ALL' or 'BUDGET_HISTORY_MEAN'."
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
            # Convert budget_histories to Curve objects
            budget_curves = _budget_history_to_curves(
                experiment.all_budget_history, normalize=normalize
            )

            if plot_type == PlotType.BUDGET_HISTORY_ALL:
                # Plot all budget history curves from all macroreps
                handle = budget_curves[0].plot(color_str=color_str)
                for curve in budget_curves[1:]:
                    curve.plot(color_str=color_str)
                solver_handles.append(handle)

            elif plot_type == PlotType.BUDGET_HISTORY_MEAN:
                # Compute and plot mean budget history across macroreps
                estimator = curve_utils.mean_of_curves(budget_curves)
                handle = estimator.plot(color_str=color_str)
                solver_handles.append(handle)

                # Compute bootstrap confidence intervals if requested
                if (plot_conf_ints or print_max_hw) and len(budget_curves) > 1:
                    bs_conf_int_lb_curve, bs_conf_int_ub_curve = (
                        _bootstrap_curves_conf_int(
                            curves=budget_curves,
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

        plt.legend(
            handles=solver_handles,
            labels=[experiment.solver.name for experiment in experiments],
            loc=legend_loc,
        )
        if normalize:
            plt.xlabel("Percentage of the run")
        else:
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.ylim(bottom=0, top=ref_experiment.problem.factors["budget"])

        # Apply tight layout before adding max halfwidth text
        plt.tight_layout()

        if print_max_hw and plot_type == PlotType.BUDGET_HISTORY_MEAN and curve_pairs:
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
            # Convert budget_histories to Curve objects
            budget_curves = _budget_history_to_curves(
                experiment.all_budget_history, normalize=normalize
            )

            if plot_type == PlotType.BUDGET_HISTORY_ALL:
                # Plot all budget history curves from all macroreps
                for curve in budget_curves:
                    curve.plot()

            elif plot_type == PlotType.BUDGET_HISTORY_MEAN:
                # Compute and plot mean budget history across macroreps
                estimator = curve_utils.mean_of_curves(budget_curves)
                estimator.plot()

                # Compute bootstrap confidence intervals if requested
                if (plot_conf_ints or print_max_hw) and len(budget_curves) > 1:
                    bs_conf_int_lb_curve, bs_conf_int_ub_curve = (
                        _bootstrap_curves_conf_int(
                            curves=budget_curves,
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
            plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
            plt.ylim(bottom=0, top=experiment.problem.factors["budget"])
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
