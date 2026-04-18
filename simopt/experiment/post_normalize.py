"""Post normalization helper functions."""

import logging
from typing import cast

import numpy as np

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import MultistageProblem, Solution
from simopt.curve import Curve
from simopt.utils import make_nonzero

from .single import ProblemSolver


def _best_without_feasibility(
    experiments: list[ProblemSolver],
    ref_experiment: ProblemSolver,
    baseline_rngs: list[MRG32k3a],
    n_postreps_init_opt: int,
) -> tuple:
    """Fallback proxy x* when no feasible post-replicated solution exists."""
    best_signed_objective = -np.inf
    best_experiment: ProblemSolver | None = None
    best_mrep: int | None = None
    best_budget_idx: int | None = None

    for experiment in experiments:
        minmax = float(experiment.problem.minmax[0])
        for mrep, est_objectives in enumerate(experiment.all_est_objectives):
            arr = np.asarray(est_objectives, dtype=float)
            if arr.size == 0:
                continue
            signed = minmax * arr
            idx = int(np.argmax(signed))
            candidate = float(signed[idx])
            if candidate > best_signed_objective:
                best_signed_objective = candidate
                best_experiment = experiment
                best_mrep = mrep
                best_budget_idx = idx

    if best_experiment is None or best_mrep is None or best_budget_idx is None:
        raise RuntimeError(
            "No estimated objectives available to compute a fallback proxy for x*."
        )

    best_objective = float(best_experiment.problem.minmax[0] * best_signed_objective)
    for experiment in experiments:
        experiment.fstar = best_objective

    xstar = best_experiment.all_recommended_xs[best_mrep][best_budget_idx]
    opt_soln = Solution(xstar, ref_experiment.problem)
    opt_soln.attach_rngs(rng_list=baseline_rngs, copy=False)
    ref_experiment.problem.simulate(
        solution=opt_soln, num_macroreps=n_postreps_init_opt
    )
    xstar_postreps = opt_soln.objectives[:, 0]
    return xstar, xstar_postreps


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
        logging.warning(
            "No feasible post-replicated solutions found; falling back to best "
            "estimated objective regardless of feasibility for proxy x*."
        )
        return _best_without_feasibility(
            experiments=experiments,
            ref_experiment=ref_experiment,
            baseline_rngs=baseline_rngs,
            n_postreps_init_opt=n_postreps_init_opt,
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

    assert best_mrep is not None
    assert best_budget is not None
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
                        s_ss_sss_index=[
                            0,
                            sample_exp.problem.model.n_rngs + rng_index,
                            0,
                        ]
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
            xstar_postreps = list(opt_soln.objectives[:n_postreps_init_opt][:, 0])
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
            xstar_postreps = list(opt_soln.objectives[:n_postreps_init_opt][:, 0])
        else:
            # Use precomputed per-problem global x* if available; otherwise
            # fall back to finding the best for this individual experiment.
            grp_best = (
                global_xstar_map.get(experiment.problem.name)
                if "global_xstar_map" in globals() or "global_xstar_map" in locals()
                else None
            )
            if grp_best:
                gxstar, gxstar_postreps = grp_best
                logging.info(
                    fstar_log_msg
                    + "using global best postreplicated solution as proxy for x*."
                )
                xstar = gxstar
                xstar_postreps = gxstar_postreps
            else:
                logging.info(
                    fstar_log_msg
                    + "using best postreplicated solution as proxy for x*."
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


def post_normalize_policy(
    experiments: list[ProblemSolver],
    n_postreps_init_opt: int,
    crn_across_init_opt: bool = True,
    proxy_init_val: float | None = None,
    proxy_opt_val: float | None = None,
    proxy_opt_x: tuple | None = None,
    create_pair_pickles: bool = False,
) -> None:
    """Construct normalized progress curves using policy evaluation end-to-end.

    Unlike :func:`post_normalize`, this function chooses and evaluates ``x*`` from
    ``all_policy_solutions`` and uses ``simulate_policy`` for x0/x* anchor
    postreplications.
    """

    def _make_baseline_rngs(problem: MultistageProblem) -> list[MRG32k3a]:
        return [
            MRG32k3a(
                s_ss_sss_index=[
                    0,
                    problem.model.n_rngs + rng_index,
                    0,
                ]
            )
            for rng_index in range(problem.model.n_rngs)
        ]

    def _policy_stage0_decision(policy_solution: Solution) -> tuple:
        decisions = policy_solution.x
        if isinstance(decisions, tuple) and decisions:
            first = decisions[0]
            if isinstance(first, tuple):
                return tuple(float(v) for v in first)
        return tuple(float(v) for v in decisions)

    def _decisions_equal(lhs: tuple | None, rhs: tuple | None) -> bool:
        if lhs is None or rhs is None:
            return False
        if len(lhs) != len(rhs):
            return False
        return bool(np.equal(np.asarray(lhs), np.asarray(rhs)).all())

    def _open_loop_policy_from_decision(
        problem: MultistageProblem,
        decision: tuple,
    ) -> Solution:
        n_stages = problem.model.n_stages
        decisions = tuple(decision for _ in range(n_stages))
        return problem.create_policy_solution(decisions)

    def _evaluate_policy(
        problem: MultistageProblem,
        policy_solution: Solution,
        baseline_rngs: list[MRG32k3a],
        n_postreps: int,
    ) -> list[float]:
        eval_solution = problem.create_policy_solution(
            policy_solution.x,
            policy=policy_solution.policy,
        )
        eval_solution.attach_rngs(rng_list=baseline_rngs, copy=False)
        problem.simulate_policy(solution=eval_solution, num_macroreps=n_postreps)
        return list(eval_solution.objectives[:n_postreps][:, 0])

    def _best_policy_without_feasibility(
        exps: list[ProblemSolver],
        ref_exp: ProblemSolver,
        baseline_rngs: list[MRG32k3a],
        n_postreps: int,
    ) -> tuple[Solution, tuple, list[float]]:
        best_signed_objective = -np.inf
        best_policy_solution: Solution | None = None
        best_objective = -np.inf

        for exp in exps:
            policy_solutions = exp.all_policy_solutions
            if policy_solutions is None:
                continue
            for mrep, est_objectives in enumerate(exp.all_est_objectives):
                arr = np.asarray(est_objectives, dtype=float)
                if arr.size == 0:
                    continue
                signed = float(exp.problem.minmax[0]) * arr
                idx = int(np.argmax(signed))
                candidate_signed = float(signed[idx])
                if candidate_signed > best_signed_objective:
                    best_signed_objective = candidate_signed
                    best_objective = float(arr[idx])
                    best_policy_solution = policy_solutions[mrep][idx]

        if best_policy_solution is None:
            raise RuntimeError(
                "No policy objective estimates available to compute a proxy x*."
            )

        for exp in exps:
            exp.fstar = best_objective

        xstar = _policy_stage0_decision(best_policy_solution)
        ref_problem = cast(MultistageProblem, ref_exp.problem)
        xstar_postreps = _evaluate_policy(
            ref_problem,
            best_policy_solution,
            baseline_rngs,
            n_postreps,
        )
        return best_policy_solution, xstar, xstar_postreps

    def _best_policy_with_feasibility(
        exps: list[ProblemSolver],
        ref_exp: ProblemSolver,
        baseline_rngs: list[MRG32k3a],
        n_postreps: int,
    ) -> tuple[Solution, tuple, list[float]]:
        best_signed_objective = -np.inf
        best_objective = -np.inf
        best_policy_solution: Solution | None = None

        for exp in exps:
            policy_solutions = exp.all_policy_solutions
            if policy_solutions is None:
                continue
            for mrep, est_objectives in enumerate(exp.all_est_objectives):
                arr = np.asarray(est_objectives, dtype=float)
                if arr.size == 0:
                    continue

                if exp.problem.n_stochastic_constraints >= 1:
                    feasible = np.where(np.all(exp.all_est_lhs[mrep] <= 0, axis=1))[0]
                    if feasible.size == 0:
                        continue
                    candidate_indices = feasible
                else:
                    candidate_indices = np.arange(arr.size)

                signed = float(exp.problem.minmax[0]) * arr[candidate_indices]
                local_idx = int(np.argmax(signed))
                budget_idx = int(candidate_indices[local_idx])
                candidate_signed = float(signed[local_idx])
                if candidate_signed > best_signed_objective:
                    best_signed_objective = candidate_signed
                    best_objective = float(arr[budget_idx])
                    best_policy_solution = policy_solutions[mrep][budget_idx]

        if best_policy_solution is None:
            logging.warning(
                "No feasible post-replicated policy solutions found; falling back "
                "to the best estimated policy regardless of feasibility for proxy x*."
            )
            return _best_policy_without_feasibility(
                exps,
                ref_exp,
                baseline_rngs,
                n_postreps,
            )

        for exp in exps:
            exp.fstar = best_objective

        xstar = _policy_stage0_decision(best_policy_solution)
        ref_problem = cast(MultistageProblem, ref_exp.problem)
        xstar_postreps = _evaluate_policy(
            ref_problem,
            best_policy_solution,
            baseline_rngs,
            n_postreps,
        )
        return best_policy_solution, xstar, xstar_postreps

    if not experiments:
        raise ValueError("No experiments provided for policy post-normalization.")

    ref_experiment = experiments[0]
    for experiment in experiments:
        if not experiment.has_run:
            raise Exception(
                f"The experiment of {experiment.solver.name} on "
                f"{experiment.problem.name} has not been run."
            )
        if not experiment.has_postreplicated:
            raise Exception(
                f"The experiment of {experiment.solver.name} on "
                f"{experiment.problem.name} has not been post-replicated."
            )
        if not isinstance(experiment.problem, MultistageProblem):
            raise TypeError(
                "post_normalize_policy requires MultistageProblem experiments."
            )
        if not experiment.all_policy_solutions:
            raise RuntimeError(
                f"The experiment of {experiment.solver.name} on "
                f"{experiment.problem.name} has no policy solutions."
            )
        if experiment.n_macroreps != ref_experiment.n_macroreps:
            raise Exception(
                "At least two experiments have different numbers of macro-replications."
            )
        if getattr(experiment, "n_postreps", None) != getattr(
            ref_experiment,
            "n_postreps",
            None,
        ):
            raise Exception(
                "At least two experiments have different numbers of post-replications."
            )

    logging.info(f"Policy postnormalizing on Problem {ref_experiment.problem.name}.")

    # Precompute a per-problem global policy x* where no explicit proxy/coded
    # optimum is available.
    global_xstar_map: dict[str, tuple[Solution, tuple, list[float]] | None] = {}
    if proxy_opt_val is None and proxy_opt_x is None:
        problem_groups: dict[str, list[ProblemSolver]] = {}
        for exp in experiments:
            problem_groups.setdefault(exp.problem.name, []).append(exp)

        for pname, exps_group in problem_groups.items():
            sample_exp = exps_group[0]
            sample_problem = cast(MultistageProblem, sample_exp.problem)
            if (
                sample_problem.optimal_value is not None
                or sample_problem.optimal_solution is not None
            ):
                global_xstar_map[pname] = None
                continue

            baseline_rngs = _make_baseline_rngs(sample_problem)
            try:
                global_xstar_map[pname] = _best_policy_with_feasibility(
                    exps_group,
                    sample_exp,
                    baseline_rngs,
                    n_postreps_init_opt,
                )
                logging.info(f"Global policy proxy x* computed for problem {pname}.")
            except RuntimeError:
                global_xstar_map[pname] = None

    for experiment in experiments:
        problem = cast(MultistageProblem, experiment.problem)
        policy_solutions = experiment.all_policy_solutions
        assert policy_solutions is not None

        baseline_rngs = _make_baseline_rngs(problem)
        if crn_across_init_opt:
            for rng in baseline_rngs:
                rng.reset_substream()

        fstar_log_msg = "Finding f(x*) using "
        xstar_policy: Solution | None = None

        if proxy_opt_val is not None:
            logging.info(fstar_log_msg + "provided proxy f(x*).")
            xstar = None if proxy_opt_x is None else tuple(proxy_opt_x)
            xstar_postreps = [proxy_opt_val] * n_postreps_init_opt
            if xstar is not None:
                xstar_policy = _open_loop_policy_from_decision(problem, xstar)
        elif proxy_opt_x is not None:
            logging.info(fstar_log_msg + "provided proxy x*.")
            xstar = tuple(proxy_opt_x)
            xstar_policy = _open_loop_policy_from_decision(problem, xstar)
            xstar_postreps = _evaluate_policy(
                problem,
                xstar_policy,
                baseline_rngs,
                n_postreps_init_opt,
            )
        elif problem.optimal_value is not None:
            logging.info(fstar_log_msg + "coded f(x*).")
            xstar = None
            xstar_postreps = [problem.optimal_value] * n_postreps_init_opt
        elif problem.optimal_solution is not None:
            logging.info(fstar_log_msg + "using coded x*.")
            xstar = tuple(problem.optimal_solution)
            xstar_policy = _open_loop_policy_from_decision(problem, xstar)
            xstar_postreps = _evaluate_policy(
                problem,
                xstar_policy,
                baseline_rngs,
                n_postreps_init_opt,
            )
        else:
            grp_best = global_xstar_map.get(experiment.problem.name)
            if grp_best is not None:
                logging.info(
                    fstar_log_msg
                    + "using global best postreplicated policy as proxy for x*."
                )
                xstar_policy, xstar, xstar_postreps = grp_best
            else:
                logging.info(
                    fstar_log_msg + "using best postreplicated policy as proxy for x*."
                )
                xstar_policy, xstar, xstar_postreps = _best_policy_with_feasibility(
                    [experiment],
                    experiment,
                    baseline_rngs,
                    n_postreps_init_opt,
                )

        baseline_rngs = _make_baseline_rngs(problem)
        if crn_across_init_opt:
            for rng in baseline_rngs:
                rng.reset_substream()

        x0 = tuple(problem.factors["initial_solution"])
        x0_policy = _open_loop_policy_from_decision(problem, x0)
        if proxy_init_val is not None:
            x0_postreps = [proxy_init_val] * n_postreps_init_opt
        else:
            x0_postreps = _evaluate_policy(
                problem,
                x0_policy,
                baseline_rngs,
                n_postreps_init_opt,
            )

        initial_obj_val = np.mean(x0_postreps)
        opt_obj_val = np.mean(xstar_postreps)
        initial_opt_gap = float(initial_obj_val - opt_obj_val)
        initial_opt_gap = make_nonzero(initial_opt_gap, "initial_opt_gap")

        experiment.n_postreps_init_opt = n_postreps_init_opt
        experiment.crn_across_init_opt = crn_across_init_opt
        experiment.x0 = x0
        experiment.x0_policy = x0_policy
        experiment.x0_postreps = x0_postreps
        if xstar is not None:
            experiment.xstar = xstar
        if xstar_policy is not None:
            experiment.xstar_policy = xstar_policy
        experiment.xstar_postreps = xstar_postreps

        experiment.objective_curves = []
        experiment.progress_curves = []
        for mrep in range(experiment.n_macroreps):
            est_objectives = []
            budgets = experiment.all_intermediate_budgets[mrep]
            for budget_idx in range(len(budgets)):
                policy_sol = policy_solutions[mrep][budget_idx]
                stage0_decision = _policy_stage0_decision(policy_sol)
                if _decisions_equal(stage0_decision, x0):
                    est_objectives.append(float(np.mean(x0_postreps)))
                elif _decisions_equal(stage0_decision, xstar):
                    est_objectives.append(float(np.mean(xstar_postreps)))
                else:
                    est_objectives.append(
                        experiment.all_est_objectives[mrep][budget_idx]
                    )

            experiment.objective_curves.append(
                Curve(
                    x_vals=budgets,
                    y_vals=est_objectives,
                )
            )

            norm_est_objectives = [
                (est_objective - opt_obj_val) / initial_opt_gap
                for est_objective in est_objectives
            ]
            frac_intermediate_budgets = [
                budget / problem.factors["budget"]
                for budget in experiment.all_intermediate_budgets[mrep]
            ]
            experiment.progress_curves.append(
                Curve(x_vals=frac_intermediate_budgets, y_vals=norm_est_objectives)
            )

        experiment.has_postnormalized = True

        if create_pair_pickles:
            file_name = experiment.file_name_path.name
            experiment.record_experiment_results(file_name=file_name)
