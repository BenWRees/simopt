"""Progress curve plot."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import logging

import simopt.curve_utils as curve_utils
from simopt.bootstrap import bootstrap_procedure
from simopt.experiment import ProblemSolver
from simopt.plot_type import PlotType

from .utils import (
    check_common_problem_and_reference,
    plot_bootstrap_conf_ints,
    report_max_halfwidth,
    save_plot,
    setup_plot,
)


def plot_progress_curves(
    experiments: list[ProblemSolver],
    plot_type: PlotType,
    allow_xstar_mismatch: bool = False,
    beta: float = 0.50,
    normalize: bool = True,
    all_in_one: bool = True,
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
    """Plots individual or aggregate progress curves for solvers on a single problem.

    Args:
        experiments (list[ProblemSolver]): Problem-solver pairs for different solvers
            on the same problem.
        plot_type (PlotType): Type of plot to produce (ALL, MEAN, or QUANTILE).
        beta (float, optional): Quantile level to plot (0 < beta < 1). Defaults to 0.50.
        normalize (bool, optional): If True, normalize curves by optimality gaps.
            Defaults to True.
        all_in_one (bool, optional): If True, plot all curves in one figure.
            Defaults to True.
        n_bootstraps (int, optional): Number of bootstrap samples. Defaults to 100.
        conf_level (float, optional): Confidence level for CIs (0 < conf_level < 1).
            Defaults to 0.95.
        plot_conf_ints (bool, optional): If True, plot bootstrapped confidence
            intervals. Defaults to True.
        print_max_hw (bool, optional): If True, print caption with max half-width.
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
        allow_xstar_mismatch (bool, optional): If True, tolerate small numeric
            differences in experiments' `xstar` values. If mismatches exist,
            experiments' `xstar` values will be overwritten to the first
            experiment's `xstar`. Defaults to False.

    Returns:
        list[Path]: List of file paths where the plots were saved.

    Raises:
        ValueError: If beta, conf_level, or n_bootstraps have invalid values.
    """
    # Value checking
    if not 0 < beta < 1:
        raise ValueError("Beta quantile must be in (0, 1).")
    if n_bootstraps < 1:
        raise ValueError("Number of bootstraps must be a positive integer.")
    if not 0 < conf_level < 1:
        raise ValueError("Confidence level must be in (0, 1).")

    if legend_loc is None:
        legend_loc = "best"

    # Check if problems are the same with the same x0 and x*.
    # Allow a relaxed x* check if requested (useful when combining pickles
    # that have the same problem but tiny numeric differences in reference
    # optima). If `allow_xstar_mismatch` is True, attempt to tolerate
    # numeric differences and, if necessary, align all experiments to a
    # common reference `xstar` (the first experiment's `xstar`).
    try:
        check_common_problem_and_reference(experiments)
    except ValueError as exc:
        msg = str(exc)
        if "optimal solution" in msg and allow_xstar_mismatch:
            # Compare using tolerance and, if mismatched, set all to ref.
            xstars = [np.asarray(exp.xstar) for exp in experiments]
            ref = xstars[0]
            if all(np.allclose(ref, xs) for xs in xstars[1:]):
                # differences are numeric only; treat as equal
                logging.info("xstar values differ only numerically; proceeding.")
            else:
                # Overwrite others with the reference xstar to allow plotting
                for exp in experiments:
                    exp.xstar = tuple(ref)
                logging.info(
                    "Aligned experiments' xstar to the first experiment's xstar."
                )
        else:
            raise
    file_list: list[Path] = []
    # Set up plot.
    n_experiments = len(experiments)
    if all_in_one:
        ref_experiment = experiments[0]
        setup_plot(
            plot_type=plot_type,
            solver_name=solver_set_name,
            problem_name=ref_experiment.problem.name,
            normalize=normalize,
            budget=ref_experiment.problem.factors["budget"],
            beta=beta,
            plot_title=plot_title,
        )
        solver_curve_handles = []
        curve_pairs = []
        for exp_idx in range(n_experiments):
            experiment = experiments[exp_idx]
            color_str = "C" + str(exp_idx)
            estimator = None
            if plot_type == PlotType.ALL:
                # Plot all estimated progress curves.
                if normalize:
                    handle = experiment.progress_curves[0].plot(color_str=color_str)
                    for curve in experiment.progress_curves[1:]:
                        curve.plot(color_str=color_str)
                else:
                    handle = experiment.objective_curves[0].plot(color_str=color_str)
                    for curve in experiment.objective_curves[1:]:
                        curve.plot(color_str=color_str)
            elif plot_type == PlotType.MEAN:
                # Plot estimated mean progress curve.
                if normalize:
                    estimator = curve_utils.mean_of_curves(experiment.progress_curves)
                else:
                    estimator = curve_utils.mean_of_curves(experiment.objective_curves)
                handle = estimator.plot(color_str=color_str)
            elif plot_type == PlotType.QUANTILE:
                # Plot estimated beta-quantile progress curve.
                if normalize:
                    estimator = curve_utils.quantile_of_curves(
                        experiment.progress_curves, beta
                    )
                else:
                    estimator = curve_utils.quantile_of_curves(
                        experiment.objective_curves, beta
                    )
                handle = estimator.plot(color_str=color_str)
            else:
                error_msg = f"'{plot_type.value}' is not implemented."
                raise NotImplementedError(error_msg)
            solver_curve_handles.append(handle)
            if (plot_conf_ints or print_max_hw) and plot_type != PlotType.ALL:
                bs_conf_int_lb_curve, bs_conf_int_ub_curve = bootstrap_procedure(
                    experiments=[[experiment]],
                    n_bootstraps=n_bootstraps,
                    conf_level=conf_level,
                    plot_type=plot_type,
                    beta=beta,
                    estimator=estimator,
                    normalize=normalize,
                )
                if plot_conf_ints:
                    if isinstance(bs_conf_int_lb_curve, (int, float)) or isinstance(
                        bs_conf_int_ub_curve, (int, float)
                    ):
                        error_msg = (
                            "Bootstrap confidence intervals are not available "
                            "for scalar estimators."
                        )
                        raise ValueError(error_msg)
                    plot_bootstrap_conf_ints(
                        bs_conf_int_lb_curve,
                        bs_conf_int_ub_curve,
                        color_str=color_str,
                    )
                if print_max_hw:
                    curve_pairs.append([bs_conf_int_lb_curve, bs_conf_int_ub_curve])
        leg = plt.legend(
            handles=solver_curve_handles,
            labels=[experiment.solver.name for experiment in experiments],
            loc=legend_loc,
        )
        if leg is not None:
            try:
                leg.get_frame().set_alpha(0.4)
            except Exception:
                pass
        if print_max_hw and plot_type != PlotType.ALL:
            report_max_halfwidth(
                curve_pairs=curve_pairs,
                normalize=normalize,
                conf_level=conf_level,
            )
        file_list.append(
            save_plot(
                solver_name=solver_set_name,
                problem_name=ref_experiment.problem.name,
                plot_type=plot_type,
                normalize=normalize,
                extra=beta,
                plot_title=plot_title,
                ext=ext,
                save_as_pickle=save_as_pickle,
            )
        )
    else:  # Plot separately.
        for experiment in experiments:
            setup_plot(
                plot_type=plot_type,
                solver_name=experiment.solver.name,
                problem_name=experiment.problem.name,
                normalize=normalize,
                budget=experiment.problem.factors["budget"],
                beta=beta,
            )
            estimator = None
            if plot_type == PlotType.ALL:
                # Plot all estimated progress curves.
                if normalize:
                    for curve in experiment.progress_curves:
                        curve.plot()
                else:
                    for curve in experiment.objective_curves:
                        curve.plot()
            elif plot_type == PlotType.MEAN:
                # Plot estimated mean progress curve.
                if normalize:
                    estimator = curve_utils.mean_of_curves(experiment.progress_curves)
                else:
                    estimator = curve_utils.mean_of_curves(experiment.objective_curves)
                estimator.plot()
            else:  # Must be quantile.
                # Plot estimated beta-quantile progress curve.
                if normalize:
                    estimator = curve_utils.quantile_of_curves(
                        experiment.progress_curves, beta
                    )
                else:
                    estimator = curve_utils.quantile_of_curves(
                        experiment.objective_curves, beta
                    )
                estimator.plot()
            if (plot_conf_ints or print_max_hw) and plot_type != PlotType.ALL:
                # Note: "experiments" needs to be a list of list of ProblemSolvers.
                bs_conf_int_lb_curve, bs_conf_int_ub_curve = bootstrap_procedure(
                    experiments=[[experiment]],
                    n_bootstraps=n_bootstraps,
                    conf_level=conf_level,
                    plot_type=plot_type,
                    beta=beta,
                    estimator=estimator,
                    normalize=normalize,
                )
                if plot_conf_ints:
                    if isinstance(bs_conf_int_lb_curve, (int, float)) or isinstance(
                        bs_conf_int_ub_curve, (int, float)
                    ):
                        error_msg = (
                            "Bootstrap confidence intervals are not available "
                            "for scalar estimators."
                        )
                        raise ValueError(error_msg)
                    plot_bootstrap_conf_ints(bs_conf_int_lb_curve, bs_conf_int_ub_curve)
                if print_max_hw:
                    if isinstance(bs_conf_int_lb_curve, (int, float)) or isinstance(
                        bs_conf_int_ub_curve, (int, float)
                    ):
                        error_msg = (
                            "Max halfwidth is not available for scalar estimators."
                        )
                        raise ValueError(error_msg)
                    report_max_halfwidth(
                        curve_pairs=[[bs_conf_int_lb_curve, bs_conf_int_ub_curve]],
                        normalize=normalize,
                        conf_level=conf_level,
                    )
            file_list.append(
                save_plot(
                    solver_name=experiment.solver.name,
                    problem_name=experiment.problem.name,
                    plot_type=plot_type,
                    normalize=normalize,
                    extra=beta,
                    ext=ext,
                    save_as_pickle=save_as_pickle,
                )
            )
    return file_list
