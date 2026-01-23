"""Post normalization helper functions."""

import logging

import numpy as np

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import Solution
from simopt.curve import Curve
from simopt.utils import make_nonzero

from .single import ProblemSolver


def _best_with_feasibility(
    experiments: list[ProblemSolver],
    ref_experiment: ProblemSolver,
    baseline_rngs: list[MRG32k3a],
    n_postreps_init_opt: int,
) -> tuple:
    infeasible_penalty = np.inf
    best_est_objectives = np.zeros(len(experiments))

    for experiment_idx in range(len(experiments)):
        experiment = experiments[experiment_idx]
        exp_best_est_objectives = np.zeros(experiment.n_macroreps)
        for mrep in range(experiment.n_macroreps):
            if experiment.problem.n_stochastic_constraints >= 1:
                indices = np.where(np.all(experiment.all_est_lhs[mrep] <= 0, axis=1))[0]
                all_feasible_est_objectives = experiment.all_est_objectives[mrep][
                    indices
                ]
            else:
                all_feasible_est_objectives = experiment.all_est_objectives[mrep]

            # TODO: this conversion is necessary because `all_est_objectives` may be
            # loaded from a file, which makes `all_feasible_est_objectives` a list.
            all_feasible_est_objectives = np.array(all_feasible_est_objectives)

            if len(all_feasible_est_objectives) != 0:
                exp_best_est_objectives[mrep] = np.max(
                    experiment.problem.minmax[0] * all_feasible_est_objectives
                )
            else:
                exp_best_est_objectives[mrep] = (
                    experiment.problem.minmax[0] * infeasible_penalty
                )

        best_est_objectives[experiment_idx] = np.max(exp_best_est_objectives)

    best_index = np.argmax(best_est_objectives)
    best_experiment = experiments[best_index]
    best_objective = best_experiment.problem.minmax[0] * best_est_objectives[best_index]

    if abs(best_objective) == infeasible_penalty:
        raise RuntimeError(
            "No feasible solutions found for which to estimate proxy for x*."
        )

    # TODO: this is a temporary fix to attach f* to all experiments.
    for experiment in experiments:
        experiment.fstar = best_objective

    best_mrep, best_budget = None, None
    for mrep, est_objectives in enumerate(best_experiment.all_est_objectives):
        if best_objective in est_objectives:
            best_mrep = mrep
            best_budget = np.where(est_objectives == best_objective)[0][0]
            break

    xstar = best_experiment.all_recommended_xs[best_mrep][best_budget]
    opt_soln = Solution(xstar, ref_experiment.problem)
    opt_soln.attach_rngs(rng_list=baseline_rngs, copy=False)
    ref_experiment.problem.simulate(
        solution=opt_soln, num_macroreps=n_postreps_init_opt
    )
    # Assuming only one objective.
    xstar_postreps = opt_soln.objectives[:, 0]

    return xstar, xstar_postreps


def post_normalize(
    experiments: list[ProblemSolver],
    n_postreps_init_opt: int,
    crn_across_init_opt: bool = True,
    proxy_init_val: float | None = None,
    proxy_opt_val: float | None = None,
    proxy_opt_x: tuple | None = None,
    create_pair_pickles: bool = False,
) -> None:
    """Constructs objective and normalized progress curves for a set of experiments.

    Args:
        experiments (list[ProblemSolver]): Problem-solver pairs for different solvers on
            the same problem.
        n_postreps_init_opt (int): Number of postreplications at initial (x0) and
            optimal (x*) solutions.
        crn_across_init_opt (bool, optional): If True, use CRN for postreplications at
            x0 and x*. Defaults to True.
        proxy_init_val (float, optional): Known objective value of the initial solution.
        proxy_opt_val (float, optional): Proxy or bound for the optimal objective value.
        proxy_opt_x (tuple, optional): Proxy for the optimal solution.
        create_pair_pickles (bool, optional): If True, create a pickle file for each
            problem-solver pair. Defaults to False.
    """
    # Check that all experiments have the same problem and same
    # post-experimental setup.
    ref_experiment = experiments[0]
    for experiment in experiments:
        # Check if problems are the same.
        # if experiment.problem != ref_experiment.problem:
        #     error_msg = "At least two experiments have different problems."
            # raise Exception(error_msg)
        # Check if experiments have common number of macroreps.
        if experiment.n_macroreps != ref_experiment.n_macroreps:
            error_msg = (
                "At least two experiments have different numbers of macro-replications."
            )
            raise Exception(error_msg)
        # Check if experiment has been post-replicated
        if not experiment.has_run:
            error_msg = (
                f"The experiment of {experiment.solver.name} on "
                f"{experiment.problem.name} has not been run."
            )
            raise Exception(error_msg)
        if not experiment.has_postreplicated:
            error_msg = (
                f"The experiment of {experiment.solver.name} on "
                f"{experiment.problem.name} has not been post-replicated."
            )
            raise Exception(error_msg)
        # Check if experiments have common number of post-replications.
        if getattr(experiment, "n_postreps", None) != getattr(
            ref_experiment, "n_postreps", None
        ):
            error_msg = (
                "At least two experiments have different numbers of "
                "post-replications.\n"
                "Estimation of optimal solution x* may be based on different numbers "
                "of post-replications."
            )
            raise Exception(error_msg)
    logging.info(f"Postnormalizing on Problem {ref_experiment.problem.name}.")
    # Take post-replications at common x0.
    # Create, initialize, and attach RNGs for model.
    #     Stream 0: reserved for post-replications.
    # Create baseline_rngs for reference experiment (for xstar/xstar_postreps logic)
    baseline_rngs = [
        MRG32k3a(
            s_ss_sss_index=[
                0,
                ref_experiment.problem.model.n_rngs + rng_index,
                0,
            ]
        )
        for rng_index in range(ref_experiment.problem.model.n_rngs)
    ]
    # Remove global x0/x0_postreps; handle per-experiment below
    # Moved crn_across_init_opt block inside per-experiment loop below
    # Store x0 and x* info and compute progress curves for each ProblemSolver.
    # Compute per-problem global x* across all supplied ProblemSolver instances
    # when no proxy or coded optimal is available. This ensures optimality gap
    # fractions are computed relative to the best solution across solver factors
    # for the same problem name.
    global_xstar_map: dict[str, tuple | None] = {}
    if proxy_opt_val is None and proxy_opt_x is None:
        # Group experiments by problem name
        problem_groups: dict[str, list[ProblemSolver]] = {}
        for exp in experiments:
            problem_groups.setdefault(exp.problem.name, []).append(exp)

        for pname, exps_group in problem_groups.items():
            # Skip if explicit coded optimal/proxy exists on the problem
            sample_exp = exps_group[0]
            if (
                sample_exp.problem.optimal_value is not None
                or sample_exp.problem.optimal_solution is not None
            ):
                global_xstar_map[pname] = None
                continue

            if len(exps_group) > 0:
                # Build baseline RNGs appropriate for this problem's model
                baseline_rngs_grp = [
                    MRG32k3a(
                        s_ss_sss_index=[0, sample_exp.problem.model.n_rngs + rng_index, 0]
                    )
                    for rng_index in range(sample_exp.problem.model.n_rngs)
                ]
                try:
                    gxstar, gxstar_postreps = _best_with_feasibility(
                        exps_group, sample_exp, baseline_rngs_grp, n_postreps_init_opt
                    )
                    global_xstar_map[pname] = (gxstar, gxstar_postreps)
                    logging.info(f"Global proxy x* computed for problem {pname}.")
                except RuntimeError:
                    global_xstar_map[pname] = None
            else:
                global_xstar_map[pname] = None
    for experiment in experiments:
        # Compute xstar and xstar_postreps for this experiment
        fstar_log_msg = "Finding f(x*) using "
        # Create baseline_rngs for this experiment's model
        baseline_rngs = [
            MRG32k3a(
                s_ss_sss_index=[
                    0,
                    experiment.problem.model.n_rngs + rng_index,
                    0,
                ]
            )
            for rng_index in range(experiment.problem.model.n_rngs)
        ]

        if crn_across_init_opt:
            for rng in baseline_rngs:
                rng.reset_substream()

        if proxy_opt_val is not None:
            xstar = None if proxy_opt_x is None else proxy_opt_x
            logging.info(fstar_log_msg + "provided proxy f(x*).")
            xstar_postreps = [proxy_opt_val] * n_postreps_init_opt
        elif proxy_opt_x is not None:
            logging.info(fstar_log_msg + "provided proxy x*.")
            xstar = proxy_opt_x
            opt_soln = Solution(xstar, experiment.problem)
            opt_soln.attach_rngs(rng_list=baseline_rngs, copy=False)
            experiment.problem.simulate(
                solution=opt_soln, num_macroreps=n_postreps_init_opt
            )
            xstar_postreps = list(
                opt_soln.objectives[:n_postreps_init_opt][:, 0]
            )
            if experiment.problem.n_stochastic_constraints >= 1 and any(
                opt_soln.stoch_constraints_mean > 0
            ):
                xstar, xstar_postreps = _best_with_feasibility(
                    [experiment],
                    experiment,
                    baseline_rngs,
                    n_postreps_init_opt,
                )
        elif experiment.problem.optimal_value is not None:
            logging.info(fstar_log_msg + "coded f(x*).")
            xstar = None
            xstar_postreps = [experiment.problem.optimal_value] * n_postreps_init_opt
        elif experiment.problem.optimal_solution is not None:
            logging.info(fstar_log_msg + "using coded x*.")
            xstar = experiment.problem.optimal_solution
            opt_soln = Solution(xstar, experiment.problem)
            opt_soln.attach_rngs(rng_list=baseline_rngs, copy=False)
            experiment.problem.simulate(
                solution=opt_soln, num_macroreps=n_postreps_init_opt
            )
            xstar_postreps = list(
                opt_soln.objectives[:n_postreps_init_opt][:, 0]
            )
        else:
            # Use precomputed per-problem global x* if available; otherwise
            # fall back to finding the best for this individual experiment.
            grp_best = global_xstar_map.get(experiment.problem.name) if 'global_xstar_map' in globals() or 'global_xstar_map' in locals() else None
            if grp_best:
                gxstar, gxstar_postreps = grp_best
                logging.info(fstar_log_msg + "using global best postreplicated solution as proxy for x*.")
                xstar = gxstar
                xstar_postreps = gxstar_postreps
            else:
                logging.info(
                    fstar_log_msg + "using best postreplicated solution as proxy for x*."
                )
                xstar, xstar_postreps = _best_with_feasibility(
                    [experiment], experiment, baseline_rngs, n_postreps_init_opt
                )
        # Create baseline_rngs for this experiment's model
        baseline_rngs = [
            MRG32k3a(
                s_ss_sss_index=[
                    0,
                    experiment.problem.model.n_rngs + rng_index,
                    0,
                ]
            )
            for rng_index in range(experiment.problem.model.n_rngs)
        ]

        if crn_across_init_opt:
            # Reset each rng to start of its current substream.
            for rng in baseline_rngs:
                rng.reset_substream()
        # Create baseline_rngs for this experiment's model
        baseline_rngs = [
            MRG32k3a(
                s_ss_sss_index=[
                    0,
                    experiment.problem.model.n_rngs + rng_index,
                    0,
                ]
            )
            for rng_index in range(experiment.problem.model.n_rngs)
        ]

        # Use each experiment's own x0 and x0_postreps
        x0 = experiment.problem.factors["initial_solution"]
        if proxy_init_val is not None:
            x0_postreps = [proxy_init_val] * n_postreps_init_opt
        else:
            initial_soln = Solution(x0, experiment.problem)
            initial_soln.attach_rngs(rng_list=baseline_rngs, copy=False)
            experiment.problem.simulate(
                solution=initial_soln, num_macroreps=n_postreps_init_opt
            )
            x0_postreps = list(
                initial_soln.objectives[:n_postreps_init_opt][:, 0]
            )  # 0 <- assuming only one objective

        # Compute signed initial optimality gap = f(x0) - f(x*) for this experiment
        initial_obj_val = np.mean(x0_postreps)
        opt_obj_val = np.mean(xstar_postreps)
        initial_opt_gap = float(initial_obj_val - opt_obj_val)
        initial_opt_gap = make_nonzero(initial_opt_gap, "initial_opt_gap")

        experiment.n_postreps_init_opt = n_postreps_init_opt
        experiment.crn_across_init_opt = crn_across_init_opt
        experiment.x0 = x0
        experiment.x0_postreps = x0_postreps
        if xstar is not None:
            experiment.xstar = xstar
        experiment.xstar_postreps = xstar_postreps
        # Construct objective and progress curves.
        experiment.objective_curves = []
        experiment.progress_curves = []
        for mrep in range(experiment.n_macroreps):
            est_objectives = []
            budgets = experiment.all_intermediate_budgets[mrep]
            # Substitute estimates at x0 and x* (based on N postreplicates)
            # with new estimates (based on L postreplicates).
            for budget in range(len(budgets)):
                soln = experiment.all_recommended_xs[mrep][budget]
                if np.equal(soln, x0).all():
                    est_objectives.append(np.mean(x0_postreps))
                # TODO: ensure xstar is not None.
                elif np.equal(soln, xstar).all():  # type: ignore
                    est_objectives.append(np.mean(xstar_postreps))
                else:
                    est_objectives.append(experiment.all_est_objectives[mrep][budget])
            experiment.objective_curves.append(
                Curve(
                    x_vals=budgets,
                    y_vals=est_objectives,
                )
            )
            # Normalize by initial optimality gap.
            norm_est_objectives = [
                (est_objective - opt_obj_val) / initial_opt_gap
                for est_objective in est_objectives
            ]
            frac_intermediate_budgets = [
                budget / experiment.problem.factors["budget"]
                for budget in experiment.all_intermediate_budgets[mrep]
            ]
            experiment.progress_curves.append(
                Curve(x_vals=frac_intermediate_budgets, y_vals=norm_est_objectives)
            )

        experiment.has_postnormalized = True

        # Save ProblemSolver object to .pickle file if specified.
        if create_pair_pickles:
            file_name = experiment.file_name_path.name
            experiment.record_experiment_results(file_name=file_name)
