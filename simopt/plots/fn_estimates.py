"""Function estimates plot."""

import contextlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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


def _iteration_fraction_for_budgets(
    recommendation_budgets: list[float],
    budget_history: list[float],
    iterations: list[float],
) -> list[float]:
    """Map each recommendation budget to a fraction-of-iterations x value.

    For every recommendation budget ``B``, we look up the iteration of the
    algorithm at which the cumulative simulation budget first reached ``B``.
    The resulting iteration numbers are then rescaled to ``[0, 1]`` using

        ``frac = (iteration - min_iter) / (max_iter - min_iter)``

    so the first logged iteration maps to ``0`` and the last to ``1``. This
    is the iteration-grid analogue of ``budget / total_budget`` used by
    ``plot_progress_curves``.
    """
    from bisect import bisect_left

    if not iterations:
        return [0.0 for _ in recommendation_budgets]
    bh = list(budget_history)
    iters = list(iterations)
    n = len(bh)
    i_min = min(iters)
    i_max = max(iters)
    span = i_max - i_min

    iter_xs: list[float] = []
    for b in recommendation_budgets:
        idx = bisect_left(bh, b)
        if idx >= n:
            idx = n - 1
        iter_xs.append(float(iters[idx]))
    if span <= 0:
        return [0.0 for _ in iter_xs]
    return [(i - i_min) / span for i in iter_xs]


def _fn_estimates_to_curves(
    experiment: ProblemSolver, *, normalize: bool
) -> list[Curve]:
    """Build per-macroreplication curves for `plot_fn_estimates`.

    Y-data is taken from the same source as ``plot_progress_curves``:

    * ``experiment.objective_curves[mrep].y_vals`` when ``normalize=False``
      (post-replicated objective estimates at each recommendation, with
      ``x0``/``x*`` substitutions handled in ``post_normalize``).
    * ``experiment.progress_curves[mrep].y_vals`` when ``normalize=True``
      (the same values normalized to optimality-gap fractions, exactly the
      formula used by ``plot_progress_curves``).

    The x-axis is converted from "fraction of budget" to "fraction of the run
    based on iteration numbers" by looking up, for each recommendation, the
    iteration at which its cumulative simulation budget was reached. This
    requires the solver's iteration log (``all_iterations`` /
    ``all_budget_history``).
    """
    source_attr = "progress_curves" if normalize else "objective_curves"
    src_curves = getattr(experiment, source_attr, None)
    if not src_curves:
        raise ValueError(
            "plot_fn_estimates requires the experiment to have been "
            f"post-normalized (`{source_attr}` is empty). Call "
            "`post_normalize(...)` before plotting."
        )

    all_intermediate_budgets = getattr(
        experiment, "all_intermediate_budgets", None
    )
    all_budget_history = getattr(experiment, "all_budget_history", None)
    all_iterations = getattr(experiment, "all_iterations", None)
    if (
        all_intermediate_budgets is None
        or all_budget_history is None
        or all_iterations is None
    ):
        raise ValueError(
            "plot_fn_estimates requires iteration-level data "
            "(`all_iterations` and `all_budget_history` from `iteration_df`) "
            "in addition to recommendation data. Re-run the solver so that "
            "iteration-level logging is captured."
        )

    n_mreps = len(src_curves)
    if (
        len(all_intermediate_budgets) != n_mreps
        or len(all_budget_history) != n_mreps
        or len(all_iterations) != n_mreps
    ):
        raise ValueError(
            "Macroreplication counts disagree across "
            "`objective_curves`/`progress_curves`, `all_intermediate_budgets`, "
            "`all_budget_history`, and `all_iterations`."
        )

    out: list[Curve] = []
    for mrep in range(n_mreps):
        y_vals = list(src_curves[mrep].y_vals)
        rec_budgets = [float(b) for b in all_intermediate_budgets[mrep]]
        bh = [float(b) for b in all_budget_history[mrep]]
        iters = [float(i) for i in all_iterations[mrep]]
        x_vals = _iteration_fraction_for_budgets(rec_budgets, bh, iters)
        if len(x_vals) != len(y_vals):
            raise ValueError(
                "Iteration-fraction x-values and y-values have mismatched "
                f"lengths ({len(x_vals)} vs {len(y_vals)}) for macrorep {mrep}."
            )
        out.append(Curve(x_vals=x_vals, y_vals=y_vals))
    return out




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
    """Plot per-iteration function estimates against fraction-of-run.

    This is the iteration-based analogue of :func:`plot_progress_curves`:

    * x-axis is always the **fraction of the run based on iteration numbers**
      (each macrorep's iteration number divided by its own maximum iteration).
    * y-axis is the **function-value estimate** the solver reported at that
      iteration (``experiment.all_fn_estimates``, taken from ``iteration_df``).
    * When ``normalize=True`` the y-axis is replaced with the **optimality gap**
      normalized to ``[0, 1]`` using the post-replicated optimum and the
      initial fn_estimate.

    Args:
        experiments (list[ProblemSolver]): Problem-solver pairs for different solvers
            on the same problem.
        plot_type (PlotType, optional): Type of plot to produce. Options are:
            - FN_ESTIMATES_ALL: All function estimate curves from all macroreps.
            - FN_ESTIMATES_MEAN: Mean function estimate curve across macroreps.
            Defaults to FN_ESTIMATES_ALL.
        all_in_one (bool, optional): If True, plot all curves in one figure.
            Defaults to True.
        normalize (bool, optional): If True, plot the optimality gap normalized
            to ``[0, 1]`` on the y-axis instead of the raw function estimate.
            The x-axis is always fraction-of-iterations regardless. Defaults
            to False.
        y_normalize (bool, optional): Deprecated alias for ``normalize``; kept
            for backward compatibility. Defaults to False.
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

    # `y_normalize` is a deprecated alias for `normalize`: either turns on the
    # optimality-gap y-transformation. The x-axis is always fraction-of-run.
    normalize_y = bool(normalize or y_normalize)

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
            # Use the same y-data as `plot_progress_curves` (post-replicated,
            # with x0/x* substitution and the standard gap normalization when
            # requested), but with x re-expressed as fraction-of-iterations.
            fn_curves = _fn_estimates_to_curves(
                experiment, normalize=normalize_y
            )

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
        # X-axis is always fraction-of-run based on iteration numbers.
        plt.xlabel("Fraction of the run")

        # Y-axis label switches with the optimality-gap transformation.
        if normalize_y:
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
            # Use the same y-data as `plot_progress_curves` (post-replicated,
            # with x0/x* substitution and the standard gap normalization when
            # requested), but with x re-expressed as fraction-of-iterations.
            fn_curves = _fn_estimates_to_curves(
                experiment, normalize=normalize_y
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

            plt.xlabel("Fraction of the run")

            # Y-axis label for individual plots
            if normalize_y:
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
