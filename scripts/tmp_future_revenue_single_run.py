"""Tmp future revenue single run utilities."""

import contextlib
import time

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.models.vanryzin_airline_revenue import VanRyzinRevenueMultistageProblem
from simopt.problem import ProblemLike, Solution
from simopt.solver import Budget, BudgetExhaustedError, Solver
from simopt.solvers.ADP_Solver import ADPSolver


def setup_solver_rngs(solver: Solver, problem: ProblemLike, mrep: int = 0) -> None:
    """Setup solver rngs."""
    rng_list = [MRG32k3a(s_ss_sss_index=[2, i + 1, 0]) for i in range(3)]
    solver.attach_rngs(rng_list)
    sim_rngs = [
        MRG32k3a(s_ss_sss_index=[mrep + 3, i, 0]) for i in range(problem.model.n_rngs)
    ]
    solver.solution_progenitor_rngs = sim_rngs
    solver.rng_list = [
        MRG32k3a(s_ss_sss_index=[mrep + 3, problem.model.n_rngs + i, 0])
        for i in range(len(solver.rng_list))
    ]


def evaluate_x_immediate_only(x: tuple, eval_reps: int, seed_base: int) -> float:
    """Evaluate x immediate only."""
    eval_problem = VanRyzinRevenueMultistageProblem(
        fixed_factors={"budget": 5000, "n_lookahead_reps": 5},
        model_fixed_factors={"n_periods": 3},
    )
    sol = Solution(tuple(float(v) for v in x), eval_problem)
    eval_rngs = [
        MRG32k3a(s_ss_sss_index=[seed_base, i, 0])
        for i in range(eval_problem.model.n_rngs)
    ]
    sol.attach_rngs(eval_rngs, copy=False)
    eval_problem.simulate_immediate_reward(sol, num_macroreps=eval_reps)
    return float(sol.objectives_mean[0])


def main() -> None:
    """Run main entry point."""
    train_budget = 5000
    eval_reps = 30
    mrep = 0

    problem = VanRyzinRevenueMultistageProblem(
        fixed_factors={"budget": train_budget, "n_lookahead_reps": 3},
        model_fixed_factors={"n_periods": 3},
    )

    solver = ADPSolver(
        fixed_factors={
            "wrapped_solver": "ASTROMORF",
        }
    )
    setup_solver_rngs(solver, problem, mrep=mrep)
    solver.budget = Budget(train_budget)

    x_init = tuple(float(v) for v in problem.factors["initial_solution"])
    seed_base = 60000
    init_obj_immediate = evaluate_x_immediate_only(
        x_init,
        eval_reps=eval_reps,
        seed_base=seed_base,
    )

    t0 = time.perf_counter()
    with contextlib.suppress(BudgetExhaustedError):
        solver.solve(problem)
    train_sec = time.perf_counter() - t0

    x_best = tuple(
        float(v)
        for v in (
            solver.recommended_solns[-1].x if solver.recommended_solns else x_init
        )
    )
    best_obj_immediate = evaluate_x_immediate_only(
        x_best,
        eval_reps=eval_reps,
        seed_base=seed_base,
    )

    increase_immediate = best_obj_immediate - init_obj_immediate
    pct_increase_immediate = (
        (increase_immediate / abs(init_obj_immediate) * 100.0)
        if init_obj_immediate != 0
        else float("inf")
    )

    print("RESULT_START")
    print("metric=immediate_only")
    print(f"initial_objective={init_obj_immediate:.6f}")
    print(f"post_run_objective={best_obj_immediate:.6f}")
    print(f"absolute_increase={increase_immediate:.6f}")
    print(f"percent_increase={pct_increase_immediate:.3f}")
    print(f"budget_used={solver.budget.used}")
    print(f"train_seconds={train_sec:.3f}")
    print("RESULT_END")


if __name__ == "__main__":
    main()
