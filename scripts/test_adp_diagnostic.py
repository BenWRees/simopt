"""Diagnostic: Compare ADP_Solver, DAVN, and SGD on VanRyzin airline revenue.

Runs all three solvers through the ProblemSolver pipeline with policy
post-replication, then prints a side-by-side comparison.
"""

import time

import numpy as np

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.experiment import ProblemSolver, post_normalize_policy
from simopt.models.vanryzin_airline_revenue import VanRyzinRevenueMultistageProblem
from simopt.problem import Solution
from simopt.solvers.ADP_Solver import ADPSolver
from simopt.solvers.DAVN import DAVN
from simopt.solvers.SGD import SGD


def _mean_terminal_objective(problem_solver: ProblemSolver) -> float | None:
    all_est_objectives = getattr(problem_solver, "all_est_objectives", None)
    if not all_est_objectives:
        return None

    terminal_values = [
        float(np.asarray(seq, dtype=float)[-1])
        for seq in all_est_objectives
        if len(seq) > 0
    ]
    if not terminal_values:
        return None
    return float(np.mean(np.asarray(terminal_values, dtype=float)))


def _mean_terminal_iterations(problem_solver: ProblemSolver) -> float | None:
    all_iterations = getattr(problem_solver, "all_iterations", None)
    if not all_iterations:
        return None

    terminal_iterations = [float(seq[-1]) for seq in all_iterations if len(seq) > 0]
    if not terminal_iterations:
        return None
    return float(np.mean(np.asarray(terminal_iterations, dtype=float)))


def normalize_decision(d: object) -> tuple:
    """Normalize a decision to a tuple for consistent processing."""
    if isinstance(d, tuple):
        return d
    return (d,)


def _policy_load_factors_for_solution(
    problem: VanRyzinRevenueMultistageProblem,
    policy_solution: Solution,
    n_reps: int,
    mrep_idx: int,
) -> list[float]:
    capacity = np.asarray(problem.model.factors["capacity"], dtype=float)
    if n_reps <= 0:
        return [float("nan")] * len(capacity)

    load_factor_sum = np.zeros_like(capacity, dtype=float)

    for rep_idx in range(n_reps):
        rng_list = [
            MRG32k3a(s_ss_sss_index=[70_000 + mrep_idx * 1_000 + rep_idx, rng_idx, 0])
            for rng_idx in range(problem.model.n_rngs)
        ]

        fresh_policy_solution = problem.create_policy_solution(
            policy_solution.x,
            policy=policy_solution.policy,
        )
        fresh_policy_solution.attach_rngs(rng_list, copy=False)

        problem.model.before_replication(fresh_policy_solution.rng_list)
        if problem.before_replicate_override is not None:
            problem.before_replicate_override(
                problem.model, fresh_policy_solution.rng_list
            )

        state = problem.model.get_initial_state()
        policy = fresh_policy_solution.policy
        if policy is None:

            def policy(current_state: dict, stage: int) -> float:  # noqa: ARG001
                return fresh_policy_solution.x[stage]  # noqa: B023

        for stage in range(problem.model.n_stages):
            decision = policy(state, stage)
            decision_tuple = normalize_decision(decision)
            _, next_state = problem.model.replicate_stage(
                state,
                decision_tuple,
                stage,
                fresh_policy_solution.rng_list,
            )
            state = next_state

        remaining_capacity = np.asarray(state["remaining_capacity"], dtype=float)
        load_factor_sum += np.divide(
            capacity - remaining_capacity,
            capacity,
            out=np.zeros_like(capacity, dtype=float),
            where=capacity > 0,
        )

    load_factor_sum /= float(n_reps)
    return [float(v) for v in load_factor_sum]


def _mean_load_factor(
    problem_solver: ProblemSolver,
    n_postreps: int,
) -> float | None:
    all_policy_solutions = getattr(problem_solver, "all_policy_solutions", None)
    if not all_policy_solutions:
        return None

    per_mrep_leg_load_factors: list[list[float]] = []
    for mrep_idx, policy_solutions in enumerate(all_policy_solutions):
        if not policy_solutions:
            continue
        final_policy_solution = policy_solutions[-1]
        problem = problem_solver.problem

        if not isinstance(problem, VanRyzinRevenueMultistageProblem):
            raise TypeError(
                f"Expected VanRyzinRevenueMultistageProblem, got {type(problem)}"
            )
        per_mrep_leg_load_factors.append(
            _policy_load_factors_for_solution(
                problem=problem,
                policy_solution=final_policy_solution,
                n_reps=n_postreps,
                mrep_idx=mrep_idx,
            )
        )

    if not per_mrep_leg_load_factors:
        return None

    mean_leg_load_factors = np.mean(
        np.asarray(per_mrep_leg_load_factors, dtype=float), axis=0
    )
    return float(np.mean(mean_leg_load_factors))


def _format_metric(value: float | None, precision: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{precision}f}"


def main() -> None:
    """Run main entry point."""
    # ── Configuration ─────────────────────────────────────────────────────
    train_budget = 50000
    n_periods = 3
    n_macroreps = 5
    n_postreps = 50

    print("=" * 80)
    print("SOLVER COMPARISON: ADP vs DAVN vs SGD (policy post-replication)")
    print(
        f"  Budget: {train_budget}  |  Macroreps: {n_macroreps}  |  "
        f"Postreps: {n_postreps}  |  Periods: {n_periods}"
    )
    print("=" * 80)

    # ── Helper to create a fresh problem instance ─────────────────────────
    def make_problem() -> VanRyzinRevenueMultistageProblem:
        return VanRyzinRevenueMultistageProblem(
            fixed_factors={"budget": train_budget, "n_lookahead_reps": 3},
            model_fixed_factors={"n_periods": n_periods},
        )

    # ── Build ProblemSolver for each solver ───────────────────────────────
    experiments = {}
    run_timings: dict[str, list[float]] = {}

    # ADP Solver
    experiments["ADP"] = ProblemSolver(
        solver=ADPSolver(
            fixed_factors={
                "wrapped_solver": "SGD",
                "wrapped_solver_factors": {
                    "r": 10,
                    "alpha": 0.9,
                    "gradient_clipping_enabled": True,
                    "gradient_clipping_value": 20.0,
                    "spsa_gradient": False,
                    "not_use_adp_solver": False,  # Ensure inner SGD evaluations account for ADP budget  # noqa: E501
                },
                "n_training_pts": 200,
                "n_macroreps_forward": 3,
                "n_mc_replicates": 5,
                "bellman_n_iters": 3,
                "kappa_0": 1.0,
                "kappa_decay": 0.9,
            }
        ),
        problem=make_problem(),
        create_pickle=False,
    )

    # SGD (policy optimisation over full sample path)
    experiments["SGD"] = ProblemSolver(
        solver=SGD(
            fixed_factors={
                "r": 10,
                "alpha": 0.9,
                "gradient_clipping_enabled": True,
                "gradient_clipping_value": 20.0,
                "spsa_gradient": False,
                "not_use_adp_solver": True,
            }
        ),
        problem=make_problem(),
        create_pickle=False,
    )

    # DAVN (deterministic — uses minimal budget)
    experiments["DAVN"] = ProblemSolver(
        solver=DAVN(fixed_factors={"evaluation_reps": 5}),
        problem=make_problem(),
        create_pickle=False,
    )

    # ── Run each solver ──────────────────────────────────────────────────
    for name, ps in experiments.items():
        if name == "ADP":
            ps.solver.forward_pass_timings = []

        print(f"\n{'─' * 40}")
        print(f"Running {name} ({n_macroreps} macroreps)...")
        t0 = time.perf_counter()
        ps.run(n_macroreps=n_macroreps, n_jobs=1)
        elapsed = time.perf_counter() - t0
        print(f"  {name} solve done in {elapsed:.1f}s")

        if name == "ADP" and getattr(ps.solver, "forward_pass_timings", None):
            run_timings[name] = list(ps.solver.forward_pass_timings)
        else:
            run_timings[name] = list(ps.timings)

        # Report checkpoints per macrorep
        for mrep in range(n_macroreps):
            n_ckpt = len(ps.all_intermediate_budgets[mrep])
            has_pol = (
                ps.all_policy_solutions is not None
                and len(ps.all_policy_solutions) > mrep
                and len(ps.all_policy_solutions[mrep]) > 0
            )
            print(
                f"  Macrorep {mrep}: {n_ckpt} checkpoints, "
                f"policy_solutions={'yes' if has_pol else 'no'}"
            )

    # ── Policy post-replication ──────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"POST-REPLICATION (policy, {n_postreps} reps per checkpoint)")
    print("=" * 80)

    for name, ps in experiments.items():
        print(f"\n  {name}: post_replicate_policy...")
        t0 = time.perf_counter()
        ps.post_replicate_policy(n_postreps=n_postreps)
        elapsed = time.perf_counter() - t0
        print(f"  {name}: done in {elapsed:.1f}s")

    # ── Policy post-normalization ────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"POST-NORMALIZATION (policy, {n_postreps} reps at x0/x*)")
    print("=" * 80)

    all_ps = list(experiments.values())
    t0 = time.perf_counter()
    post_normalize_policy(
        experiments=all_ps,
        n_postreps_init_opt=n_postreps,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s")

    # ── Final results ────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("FINAL RESULTS — Policy Post-Replication")
    print("=" * 80)

    # Header
    print(
        f"\n{'Solver':<10} {'Macrorep':<10} {'Final Budget':<14} "
        f"{'Policy Obj':<14} {'Checkpoints':<12}"
    )
    print("─" * 60)

    for name, ps in experiments.items():
        for mrep in range(n_macroreps):
            budgets = ps.all_intermediate_budgets[mrep]
            objectives = ps.all_est_objectives[mrep]
            final_budget = budgets[-1]
            final_obj = objectives[-1]
            n_ckpt = len(budgets)
            print(
                f"{name:<10} {mrep:<10} {final_budget:<14} "
                f"{final_obj:<14.2f} {n_ckpt:<12}"
            )

    # Summary statistics across macroreps
    print(f"\n{'=' * 80}")
    print("SUMMARY — Mean Final Policy Objective (across macroreps)")
    print("=" * 80)

    print(f"\n{'Solver':<10} {'Mean Obj':<14} {'Std Err':<14} {'Mean Checkpoints':<18}")
    print("─" * 56)

    for name, ps in experiments.items():
        final_objs = []
        n_ckpts = []
        for mrep in range(n_macroreps):
            final_objs.append(ps.all_est_objectives[mrep][-1])
            n_ckpts.append(len(ps.all_intermediate_budgets[mrep]))
        mean_obj = np.mean(final_objs)
        se_obj = (
            np.std(final_objs, ddof=1) / np.sqrt(len(final_objs))
            if len(final_objs) > 1
            else 0.0
        )
        mean_ckpt = np.mean(n_ckpts)
        print(f"{name:<10} {mean_obj:<14.2f} {se_obj:<14.2f} {mean_ckpt:<18.1f}")

    # Normalization results
    if all(hasattr(ps, "x0_postreps") and ps.x0_postreps is not None for ps in all_ps):
        print(f"\n{'=' * 80}")
        print("NORMALIZATION — Progress at Final Budget")
        print("=" * 80)

        for name, ps in experiments.items():
            x0_mean = np.mean(ps.x0_postreps)
            xstar_mean = np.mean(ps.xstar_postreps)
            gap = x0_mean - xstar_mean
            print(f"\n  {name}:")
            print(f"    x0 mean:    {x0_mean:.2f}")
            print(f"    x* mean:    {xstar_mean:.2f}")
            print(f"    Gap:        {gap:.2f}")
            if ps.progress_curves:
                for mrep, curve in enumerate(ps.progress_curves):
                    final_progress = curve.y_vals[-1] if curve.y_vals else float("nan")
                    print(f"    Macrorep {mrep} final progress: {final_progress:.4f}")

    print(f"\n{'=' * 80}")
    print("FINAL METRICS BY SOLVER")
    print("=" * 80)

    for name, ps in experiments.items():
        terminal_objective = _mean_terminal_objective(ps)
        iterations_mean = _mean_terminal_iterations(ps)
        runtime_samples = run_timings.get(name, [])
        runtime_mean = (
            float(np.mean(np.asarray(runtime_samples, dtype=float)))
            if runtime_samples
            else None
        )
        load_factor_mean = _mean_load_factor(ps, n_postreps=n_postreps)

        print(f"\n{name}:")
        print(
            "  Terminal objective value (mean over macroreps): "
            f"{_format_metric(terminal_objective, 2)}"
        )
        print(
            f"  Iterations (mean over macroreps): {_format_metric(iterations_mean, 2)}"
        )
        if name == "ADP":
            print(
                "  Runtime in seconds (mean over macroreps; forward pass only): "
                f"{_format_metric(runtime_mean, 2)}"
            )
        else:
            print(
                "  Runtime in seconds (mean over macroreps): "
                f"{_format_metric(runtime_mean, 2)}"
            )
        print(
            "  Load factor (mean over macroreps and legs): "
            f"{_format_metric(load_factor_mean, 4)}"
        )

    print(f"\n{'=' * 80}")
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
