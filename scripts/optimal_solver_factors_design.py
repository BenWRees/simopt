"""Experimental Design for Finding Optimal Fixed Solver Factors.

Uses a Nearly Orthogonal Latin Hypercube Sample (NOLHS) design to explore
the factor space for each solver, then runs each design point across a
variety of problems.  The best factor configuration for every solver is
selected by aggregating the relative objective performance across all
problems (lower is better — we minimise mean normalised objective).

Solvers tested:  ADAM, ASTRODF, NELDMD, RNDSRCH, STRONG, ASTROMORF
Problems tested: SAN-1, NETWORK-1, ROSENBROCK-1, PARAMESTI-1, DYNAMNEWS-1

ASTROMoRF includes 'initial subspace dimension' and 'polynomial degree'
as design factors.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import ValidationError

from simopt.data_farming.nolhs import NOLHS
from simopt.experiment.run_solver import run_solver
from simopt.experiment_base import instantiate_problem, instantiate_solver

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default experimental settings
# ---------------------------------------------------------------------------
DEFAULT_PROBLEMS = [
    "SAN-1",
    "NETWORK-1",
    "ROSENBROCK-1",
    "PARAMESTI-1",
    "DYNAMNEWS-1",
]

DEFAULT_SOLVERS = [
    "ADAM",
    "ASTRODF",
    "NELDMD",
    "RNDSRCH",
    "STRONG",
    "ASTROMORF",
]

DEFAULT_N_MACROREPS = 5
DEFAULT_BUDGET = 10_000
DEFAULT_NUM_STACKS = 1  # NOLHS stacks (more stacks → more design points)

# ---------------------------------------------------------------------------
# Factor definitions per solver  (min, max, precision, key, cast_type)
#   precision = number of decimal places kept by the NOLHS scaler
#   cast_type = callable to coerce final value (int for integer factors)
# ---------------------------------------------------------------------------


@dataclass
class Factor:
    """A single tunable solver factor."""

    key: str
    min_val: float
    max_val: float
    precision: int = 2
    cast: type = float

    @property
    def nolhs_tuple(self) -> tuple[float, float, int]:
        """Nolhs tuple."""
        return (self.min_val, self.max_val, self.precision)


def _adam_factors() -> list[Factor]:
    return [
        Factor("r", 5, 100, 0, int),
        Factor("beta_1", 0.8, 0.99, 3, float),
        Factor("beta_2", 0.9, 0.9999, 4, float),
        Factor("alpha", 0.01, 2.0, 2, float),
    ]


def _astrodf_factors() -> list[Factor]:
    return [
        Factor("eta_1", 0.01, 0.3, 3, float),
        Factor("eta_2", 0.5, 0.95, 2, float),
        Factor("gamma_1", 1.5, 4.0, 1, float),
        Factor("gamma_2", 0.1, 0.9, 2, float),
        Factor("lambda_min", 3, 30, 0, int),
        Factor("ps_sufficient_reduction", 0.0, 1.0, 2, float),
    ]


def _neldmd_factors() -> list[Factor]:
    return [
        Factor("r", 5, 100, 0, int),
        Factor("alpha", 0.5, 2.0, 2, float),
        Factor("gammap", 1.5, 3.0, 1, float),
        Factor("betap", 0.1, 0.9, 2, float),
        Factor("delta", 0.1, 0.9, 2, float),
        Factor("initial_spread", 0.01, 0.5, 2, float),
    ]


def _rndsrch_factors() -> list[Factor]:
    return [
        Factor("sample_size", 1, 50, 0, int),
    ]


def _strong_factors() -> list[Factor]:
    return [
        Factor("n0", 5, 50, 0, int),
        Factor("n_r", 5, 50, 0, int),
        Factor("eta_0", 0.001, 0.1, 3, float),
        Factor("eta_1", 0.1, 0.5, 2, float),
        Factor("gamma_1", 0.5, 0.99, 2, float),
        Factor("gamma_2", 1.01, 2.0, 2, float),
        Factor("lambda_2", 1.001, 2.0, 3, float),
    ]


def _astromorf_factors(problem_dim: int = 100) -> list[Factor]:
    """ASTROMoRF factors including initial subspace dimension & polynomial degree."""
    max_sub_dim = max(2, problem_dim - 1)  # cannot exceed dim-1
    return [
        Factor("eta_1", 0.01, 0.3, 3, float),
        Factor("eta_2", 0.1, 0.95, 2, float),  # lowered from 0.5 — known-good ≈0.26
        Factor("gamma_1", 1.5, 4.0, 1, float),
        Factor("gamma_2", 1.05, 2.0, 2, float),
        Factor("gamma_3", 0.1, 0.9, 2, float),
        Factor("lambda_min", 3, 30, 0, int),
        Factor("subproblem_regularisation", 0.0, 0.5, 2, float),
        Factor("ps_sufficient_reduction", 0.0, 1.0, 2, float),
        Factor("mu", 100, 10000, 0, float),
        Factor("initial subspace dimension", 1.0, float(max_sub_dim), 0, int),
        Factor("polynomial degree", 2.0, 6.0, 0, int),
    ]


SOLVER_FACTOR_MAP: dict[str, Callable[[int], list[Factor]]] = {
    "ADAM": lambda _dim: _adam_factors(),
    "ASTRODF": lambda _dim: _astrodf_factors(),
    "NELDMD": lambda _dim: _neldmd_factors(),
    "RNDSRCH": lambda _dim: _rndsrch_factors(),
    "STRONG": lambda _dim: _strong_factors(),
    "ASTROMORF": lambda dim: _astromorf_factors(dim),
}


# ---------------------------------------------------------------------------
# Inter-factor constraint repair
# ---------------------------------------------------------------------------
# After NOLHS generates points independently for each factor, some
# combinations may violate pydantic model validators.  We repair them
# by swapping / nudging the offending values so both remain inside
# the original design range.


def _repair_design_point(solver_name: str, dp: dict) -> dict:
    """Enforce known inter-factor constraints on a design point *in-place*.

    Repairs are minimal: the smaller value stays, and the larger one is
    nudged just past it so the ordering constraint is satisfied.
    """
    dp = dp.copy()

    if solver_name == "ASTRODF":
        # eta_2 > eta_1
        if dp.get("eta_2", 1.0) <= dp.get("eta_1", 0.0):
            dp["eta_1"], dp["eta_2"] = (
                min(dp["eta_1"], dp["eta_2"]),
                max(dp["eta_1"], dp["eta_2"]),
            )
            if dp["eta_2"] == dp["eta_1"]:
                dp["eta_2"] = dp["eta_1"] + 0.01

    elif solver_name == "STRONG":
        # delta_T > delta_threshold
        if dp.get("delta_T", 2.0) <= dp.get("delta_threshold", 1.2):
            dp["delta_threshold"], dp["delta_T"] = (
                min(dp["delta_threshold"], dp["delta_T"]),
                max(dp["delta_threshold"], dp["delta_T"]),
            )
            if dp["delta_T"] == dp["delta_threshold"]:
                dp["delta_T"] = dp["delta_threshold"] + 0.1
        # eta_1 > eta_0
        if dp.get("eta_1", 0.3) <= dp.get("eta_0", 0.01):
            dp["eta_0"], dp["eta_1"] = (
                min(dp["eta_0"], dp["eta_1"]),
                max(dp["eta_0"], dp["eta_1"]),
            )
            if dp["eta_1"] == dp["eta_0"]:
                dp["eta_1"] = dp["eta_0"] + 0.01

    elif solver_name == "ASTROMORF":
        # eta_2 > eta_1
        if dp.get("eta_2", 0.8) <= dp.get("eta_1", 0.1):
            dp["eta_1"], dp["eta_2"] = (
                min(dp["eta_1"], dp["eta_2"]),
                max(dp["eta_1"], dp["eta_2"]),
            )
            if dp["eta_2"] == dp["eta_1"]:
                dp["eta_2"] = dp["eta_1"] + 0.01
        # gamma_1 >= gamma_2
        if dp.get("gamma_1", 2.5) < dp.get("gamma_2", 1.2):
            dp["gamma_1"], dp["gamma_2"] = dp["gamma_2"], dp["gamma_1"]

    return dp


# ---------------------------------------------------------------------------
# Dimension-update helpers (reused from existing codebase patterns)
# ---------------------------------------------------------------------------


def _update_model_factors(problem_name: str, dim: int) -> dict:
    """Return model_fixed_factors to resize the problem to *dim* decision variables."""
    if problem_name == "DYNAMNEWS-1":
        return {
            "num_prod": dim,
            "c_utility": [6 + j for j in range(dim)],
            "init_level": [3] * dim,
            "price": [9] * dim,
            "cost": [5] * dim,
        }
    if problem_name == "NETWORK-1":
        rng = np.random.default_rng(42)
        process_prob = 1.0 / dim
        mode_tt = [round(rng.uniform(0.01, 5), 3) for _ in range(dim)]
        return {
            "process_prob": [process_prob] * dim,
            "cost_process": [0.1 / (x + 1) for x in range(dim)],
            "cost_time": [round(rng.uniform(0.01, 1), 3) for _ in range(dim)],
            "mode_transit_time": mode_tt,
            "lower_limits_transit_time": [x / 2 for x in mode_tt],
            "upper_limits_transit_time": [2 * x for x in mode_tt],
            "n_networks": dim,
        }
    # Other problems resize automatically or don't need model factor changes
    return {}


def _update_problem_factors(problem_name: str, dim: int, budget: int) -> dict:
    """Return problem_fixed_factors matching *dim*."""
    if problem_name == "DYNAMNEWS-1":
        return {"initial_solution": (3,) * dim, "budget": budget}
    if problem_name == "NETWORK-1":
        init = 1.0 / dim
        return {"initial_solution": (init,) * dim, "budget": budget}
    return {"budget": budget}


# ---------------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------------


def generate_design_points(
    solver_name: str,
    problem_dim: int,
    num_stacks: int = 1,
) -> tuple[list[Factor], list[dict]]:
    """Generate NOLHS design points for a given solver.

    Returns:
        factors: the list of Factor objects
        design_points: list of dicts   {factor_key: value, ...}
    """
    factors = SOLVER_FACTOR_MAP[solver_name](problem_dim)
    nolhs_specs = [f.nolhs_tuple for f in factors]
    design = NOLHS(designs=nolhs_specs, num_stacks=num_stacks)
    raw_points = design.generate_design()  # list[list[float]]
    design_points: list[dict] = []
    for row in raw_points:
        dp = {}
        for factor, val in zip(factors, row, strict=False):
            dp[factor.key] = factor.cast(val)
        design_points.append(dp)

    # Repair any inter-factor constraint violations
    design_points = [_repair_design_point(solver_name, dp) for dp in design_points]

    log.info(
        "Generated %d NOLHS design points for %s (%d factors).",
        len(design_points),
        solver_name,
        len(factors),
    )
    return factors, design_points


def evaluate_design_point(
    solver_name: str,
    fixed_factors: dict,
    problem_name: str,
    problem_dim: int,
    budget: int,
    n_macroreps: int,
) -> dict:
    """Run a single solver config on a single problem and return summary stats.

    Returns a dict with keys:
        solver, problem, factors, mean_objective, std_objective, elapsed
    """
    # Always disable CRN for fair per-design-point comparison
    factors_to_use = {"crn_across_solns": False, **fixed_factors}

    model_factors = _update_model_factors(problem_name, problem_dim)
    problem_factors = _update_problem_factors(problem_name, problem_dim, budget)

    problem = instantiate_problem(
        problem_name,
        problem_fixed_factors=problem_factors,
        model_fixed_factors=model_factors,
    )
    solver = instantiate_solver(solver_name, fixed_factors=factors_to_use)

    t0 = time.perf_counter()
    solution_df, iteration_df, _elapsed_times = run_solver(
        solver,
        problem,
        n_macroreps=n_macroreps,
        n_jobs=1,
    )
    wall = time.perf_counter() - t0

    # Extract the terminal (best) objective from each macroreplication.
    # The iteration_df has columns: iteration, budget_history, fn_estimate, mrep
    # The solution_df has columns: step, budget, solution, mrep
    # We use fn_estimate from iteration_df if available; otherwise fall back
    # to evaluating the last recommended solution.
    if iteration_df is not None and "fn_estimate" in iteration_df.columns:
        terminal_objectives = (
            iteration_df.sort_values(["mrep", "iteration"])
            .groupby("mrep")["fn_estimate"]
            .last()
            .values
        )
    else:
        # Fallback: simulate the last recommended solution to get an estimate
        terminal_objectives = []
        for _mrep_id, grp in solution_df.groupby("mrep"):
            last_x = grp.sort_values("step")["solution"].iloc[-1]
            # Re-instantiate the problem to evaluate
            p = instantiate_problem(
                problem_name,
                problem_fixed_factors=_update_problem_factors(
                    problem_name, problem_dim, budget
                ),
                model_fixed_factors=_update_model_factors(problem_name, problem_dim),
            )
            new_sol = solver.create_new_solution(tuple(last_x), p)
            p.simulate(new_sol, num_macroreps=30)
            terminal_objectives.append(float(np.mean(new_sol.objectives_mean)))
        terminal_objectives = np.array(terminal_objectives)

    return {
        "solver": solver_name,
        "problem": problem_name,
        "factors": json.dumps(fixed_factors, default=str),
        "mean_objective": float(np.mean(terminal_objectives)),
        "std_objective": float(np.std(terminal_objectives)),
        "n_macroreps": n_macroreps,
        "wall_seconds": round(wall, 2),
    }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------


def run_experiment(
    solver_names: list[str] | None = None,
    problem_names: list[str] | None = None,
    problem_dim: int = 100,
    budget: int = DEFAULT_BUDGET,
    n_macroreps: int = DEFAULT_N_MACROREPS,
    num_stacks: int = DEFAULT_NUM_STACKS,
    output_path: str | Path = "optimal_solver_factors_results.csv",
) -> pd.DataFrame:
    """Run the full experimental design sweep.

    For each solver:
      1. Generate NOLHS design points over its tunable factors.
      2. Evaluate every design point on every problem.
      3. Select the best config by aggregated normalised performance.

    Args:
        solver_names: Which solvers to include.
        problem_names: Which problems to test on.
        problem_dim: Decision-variable dimension for resizable problems.
        budget: Simulation budget per problem.
        n_macroreps: Macro-replications per (solver-config, problem) pair.
        num_stacks: NOLHS stacks (more → denser exploration).
        output_path: Where to write the raw results CSV.

    Returns:
        DataFrame of all raw evaluation records.
    """
    solver_names = solver_names or DEFAULT_SOLVERS
    problem_names = problem_names or DEFAULT_PROBLEMS
    output_path = Path(output_path)

    all_records: list[dict] = []

    for solver_name in solver_names:
        log.info("=" * 70)
        log.info("Solver: %s", solver_name)
        log.info("=" * 70)

        _factors, design_points = generate_design_points(
            solver_name,
            problem_dim,
            num_stacks=num_stacks,
        )

        for dp_idx, dp in enumerate(design_points):
            log.info(
                "  Design point %d/%d: %s",
                dp_idx + 1,
                len(design_points),
                dp,
            )
            # Validate the design point by attempting to instantiate the solver.
            # If pydantic still rejects it after repair, skip entirely.
            try:
                _test_factors = {"crn_across_solns": False, **dp}
                instantiate_solver(solver_name, fixed_factors=_test_factors)
            except (ValidationError, ValueError) as exc:
                log.warning(
                    "  Skipping design point %d/%d — invalid factors: %s",
                    dp_idx + 1,
                    len(design_points),
                    exc,
                )
                continue

            for prob_name in problem_names:
                try:
                    record = evaluate_design_point(
                        solver_name=solver_name,
                        fixed_factors=dp,
                        problem_name=prob_name,
                        problem_dim=problem_dim,
                        budget=budget,
                        n_macroreps=n_macroreps,
                    )
                    all_records.append(record)
                    log.info(
                        "    %s  mean_obj=%.6f  (%.1fs)",
                        prob_name,
                        record["mean_objective"],
                        record["wall_seconds"],
                    )
                except Exception:
                    log.exception(
                        "    FAILED on %s with factors %s",
                        prob_name,
                        dp,
                    )

        # Checkpoint after each solver
        pd.DataFrame(all_records).to_csv(output_path, index=False)
        log.info("Checkpoint saved → %s", output_path)

    results_df = pd.DataFrame(all_records)
    results_df.to_csv(output_path, index=False)
    log.info("All results saved → %s", output_path)
    return results_df


# ---------------------------------------------------------------------------
# Analysis: pick the best factor configuration per solver
# ---------------------------------------------------------------------------


def select_best_factors(
    results_df: pd.DataFrame,
    output_path: str | Path = "optimal_solver_factors_best.json",
) -> dict:
    """Select the best factor config per solver using normalised mean objective.

    For each problem the objectives are min-max normalised (lower is better).
    The best design point per solver is the one with the lowest average
    normalised objective across all problems.

    Returns:
        dict  {solver_name: {"factors": {...}, "norm_score": float, ...}}
    """
    df = results_df.copy()

    # Normalise within each (solver, problem) group
    def _norm(group: pd.DataFrame) -> pd.Series:
        obj = group["mean_objective"]
        lo, hi = obj.min(), obj.max()
        if hi - lo < 1e-12:
            return pd.Series(0.0, index=group.index)
        return (obj - lo) / (hi - lo)

    df["norm_obj"] = df.groupby(["solver", "problem"], group_keys=False).apply(_norm)

    # Average normalised objective per (solver, factors) across problems
    agg = (
        df.groupby(["solver", "factors"])
        .agg(
            mean_norm_obj=("norm_obj", "mean"), mean_raw_obj=("mean_objective", "mean")
        )
        .reset_index()
    )

    best: dict = {}
    for solver_name, grp in agg.groupby("solver"):
        best_row = grp.loc[grp["mean_norm_obj"].idxmin()]
        best[solver_name] = {
            "factors": json.loads(best_row["factors"]),
            "mean_normalised_objective": round(float(best_row["mean_norm_obj"]), 6),
            "mean_raw_objective": round(float(best_row["mean_raw_obj"]), 6),
        }

    output_path = Path(output_path)
    output_path.write_text(json.dumps(best, indent=2, default=str))
    log.info("Best factors per solver saved → %s", output_path)

    # Print summary
    print("\n" + "=" * 70)
    print("OPTIMAL SOLVER FACTOR CONFIGURATIONS")
    print("=" * 70)
    for solver_name, info in best.items():
        print(f"\n{solver_name}:")
        print(f"  Normalised score : {info['mean_normalised_objective']:.4f}")
        print(f"  Mean objective   : {info['mean_raw_objective']:.6f}")
        print("  Factors:")
        for k, v in info["factors"].items():
            print(f"    {k:>35s} = {v}")
    print("=" * 70)

    return best


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Run main entry point."""
    parser = argparse.ArgumentParser(
        description="NOLHS experimental design to find optimal fixed solver factors.",
    )
    parser.add_argument(
        "--solvers",
        nargs="+",
        default=DEFAULT_SOLVERS,
        help="Solvers to include (default: all six).",
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        default=DEFAULT_PROBLEMS,
        help="Problems to test on.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=100,
        help="Decision-variable dimension (default: 100).",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=DEFAULT_BUDGET,
        help="Simulation budget per problem (default: 10000).",
    )
    parser.add_argument(
        "--macroreps",
        type=int,
        default=DEFAULT_N_MACROREPS,
        help="Macro-replications per evaluation (default: 5).",
    )
    parser.add_argument(
        "--stacks",
        type=int,
        default=DEFAULT_NUM_STACKS,
        help="NOLHS stacks — more stacks gives denser coverage (default: 1).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="optimal_solver_factors_results.csv",
        help="Path for raw results CSV.",
    )
    parser.add_argument(
        "--best-output",
        type=str,
        default="optimal_solver_factors_best.json",
        help="Path for best-factors JSON summary.",
    )
    args = parser.parse_args()

    results_df = run_experiment(
        solver_names=args.solvers,
        problem_names=args.problems,
        problem_dim=args.dim,
        budget=args.budget,
        n_macroreps=args.macroreps,
        num_stacks=args.stacks,
        output_path=args.output,
    )

    select_best_factors(results_df, output_path=args.best_output)


if __name__ == "__main__":
    main()
