"""Sensitivity analysis: find best solver factor values per solver/problem.

This script performs one-factor-at-a-time sweeps for each solver on each
available problem. For each solver factor (bool/int/float/Enum) it tests a
small candidate set and records the value that yields the best final objective
(lower is better). Results are written to `outputs/sensitivity_solver_factors.csv`
and JSON.

Usage:
    python scripts/OFAT_test_for_solver_factors.py --config HPC_code/config.json
    --n-macroreps 5 --eval-reps 10 --n-jobs 1
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
from pathlib import Path
from typing import Any

import pandas as pd

from demo.pickle_files_journal_paper import scale_dimension
from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.experiment.run_solver import run_solver
from simopt.experiment_base import instantiate_problem, instantiate_solver
from simopt.problem import Problem, Solution

logging.basicConfig(level=logging.INFO)


def discover_problem_names() -> list[str]:
    """Discover problem names."""
    # Use instantiate_problem to discover names by scanning modules
    import importlib
    import inspect
    import pkgutil

    module_path = "simopt.models"
    base_module = importlib.import_module(module_path)
    names: list[str] = []
    for _finder, name, _ispkg in pkgutil.walk_packages(
        base_module.__path__, prefix=module_path + "."
    ):
        try:
            submodule = importlib.import_module(name)
        except ModuleNotFoundError:
            continue
        for _, cls in inspect.getmembers(submodule, inspect.isclass):
            if cls.__module__ == name and hasattr(cls, "class_name_abbr"):
                names.append(str(cls.class_name_abbr))
    return sorted(set(names))


def discover_solver_names() -> list[str]:
    """Discover solver names."""
    import importlib
    import inspect
    import pkgutil

    module_path = "simopt.solvers"
    base_module = importlib.import_module(module_path)
    names: list[str] = []
    for _finder, name, _ispkg in pkgutil.walk_packages(
        base_module.__path__, prefix=module_path + "."
    ):
        try:
            submodule = importlib.import_module(name)
        except ModuleNotFoundError:
            continue
        for _, cls in inspect.getmembers(submodule, inspect.isclass):
            if cls.__module__ == name and hasattr(cls, "class_name_abbr"):
                names.append(str(cls.class_name_abbr))
    return sorted(set(names))


REFINED_CANDIDATES = {
    # ADAM: focus around typical defaults for learning-rate-like params
    "ADAM": {
        "beta_1": [0.5, 0.54, 0.9],
        "beta_2": [0.4995, 0.999, 0.9999],
        "alpha": [0.1, 0.3, 0.7],
        "r": [12, 24, 48],
    },
    # ASTRODF / ASTRODF variants
    "ASTRODF": {
        "eta_1": [0.01, 0.05, 0.08],
        "eta_2": [0.2, 0.5, 0.8],
        "gamma_1": [1.0, 2.0, 3.5],
        "gamma_2": [0.2, 0.3, 0.7],
        "lambda_min": [4, 6, 8],
        "ps_sufficient_reduction": [0.01, 0.05, 0.1],
    },
    "NELDMD": {"r": [12, 24, 48]},
    "RNDSRCH": {"sample_size": [4, 8, 16]},
    "STRONG": {
        "n0": [4, 6, 8],
        "n_r": [4, 6, 8],
        "eta_0": [0.001, 0.005, 0.01],
        "eta_1": [0.05, 0.15, 0.3],
        "gamma_1": [0.3, 0.45, 0.6],
        "gamma_2": [1.1, 1.33, 1.5],
        "lambda_2": [1.0, 1.414, 2.0],
    },
    "ASTROMORF": {
        # Only values that pass pydantic validation
        # gamma_1 > 1, gamma_2 > 1, gamma_2 <= gamma_1, gamma_3 < 1
        # eta_2 > eta_1, lambda_min > 2, 0.5 <= variance_explained <= 1
        "gamma_1": [2.0, 2.5, 3.5],
        "gamma_2": [1.2, 1.5, 2.0],
        "gamma_3": [0.3, 0.5, 0.8],
        "lambda_min": [4, 6, 10],
        "variance explained threshold": [0.5, 0.75, 0.95],
        "ps_sufficient_reduction": [0.01, 0.02, 0.05],
        "eta_1": [0.01, 0.03, 0.05],
        "eta_2": [0.2, 0.5, 0.8],
    },
}


def make_candidates(
    default: Any,  # noqa: ANN401
    datatype: type,
    solver_name: str | None = None,
    factor: str | None = None,
) -> list[Any]:
    """Generate up to 3 candidate values for a factor given its default and type.

    If a refined list exists for (solver_name,factor) use that first.
    """
    # solver/factor-specific overrides
    if solver_name and factor:
        sc = REFINED_CANDIDATES.get(solver_name, {})
        if factor in sc:
            return sc[factor]
    # bool
    if datatype is bool:
        return [True, False]

    # Enum types: python enums provide __members__ on the class of default
    try:
        import enum

        if isinstance(default, enum.Enum):
            return list(type(default))
    except Exception:
        pass

    # int
    if isinstance(default, int) and not isinstance(default, bool):
        low = max(1, default // 2)
        high = max(default + 1, default * 2)
        if low == default:
            return [default, high]
        return [low, default, high]

    # float
    if isinstance(default, float) or datatype is float:
        if default == 0:
            return [0.0, 0.1, 1.0]
        low = default * 0.5
        high = default * 2.0
        return [low, default, high]

    # fallback: return single default
    return [default]


def evaluate_solution(problem: Problem, x: tuple, eval_reps: int = 10) -> float:
    """Estimate objective value at x by simulating eval_reps replications.

    Returns the mean of the first (primary) objective (lower is better).
    """
    sol = Solution(x, problem)
    # attach simple RNGs (one per model RNG)
    rngs = [MRG32k3a() for _ in range(problem.model.n_rngs)]
    sol.attach_rngs(rngs)
    problem.simulate(sol, num_macroreps=eval_reps)
    # objectives_mean is numpy array; take first objective
    return float(sol.objectives_mean[0])


def _make_problem(problem_name: str, dim: int | None, budget: int) -> Problem:
    """Create a problem instance, optionally scaling dimensions."""
    if dim is not None:
        return scale_dimension(problem_name, dim, budget)
    return instantiate_problem(problem_name)


def run_one_factor_sweep(
    solver_name: str,
    problem_name: str,
    n_macroreps: int,
    eval_reps: int,
    n_jobs: int,
    factor_keys: list[str] | None = None,
    dim: int | None = None,
    budget: int = 50_000,
) -> list[dict]:
    """Run one factor sweep."""
    results = []
    logging.info(
        "Solver=%s Problem=%s dim=%s budget=%s", solver_name, problem_name, dim, budget
    )
    # instantiate problem template (with optional dimension scaling)
    problem = _make_problem(problem_name, dim, budget)

    # instantiate a solver class to inspect specifications
    solver_template = instantiate_solver(solver_name)
    specs = solver_template.specifications

    # If the solver provides no specifications but a factor key list is supplied
    # fall back to using `REFINED_CANDIDATES` entries for that solver (if available).
    if (not specs or len(specs) == 0) and factor_keys:
        fallback = {}
        rc = REFINED_CANDIDATES.get(solver_name, {})
        for k in factor_keys:
            if k in rc:
                # create a lightweight meta so the loop below will test these candidates
                fallback[k] = {"default": rc[k][0], "datatype": type(rc[k][0])}
        specs = fallback

    # baseline: default solver
    base_solver = instantiate_solver(solver_name, fixed_factors={})
    try:
        sol_df, _it_df, _elapsed = run_solver(
            base_solver, problem, n_macroreps=n_macroreps, n_jobs=n_jobs
        )
    except Exception as e:
        logging.warning(
            "Baseline run failed for %s on %s: %s", solver_name, problem_name, e
        )
        base_score = math.inf
    else:
        # evaluate final solutions per mrep
        per_mrep_scores = []
        for mrep in range(n_macroreps):
            try:
                final = sol_df[sol_df["mrep"] == mrep].iloc[-1]["solution"]
            except Exception:
                continue
            score = evaluate_solution(problem, tuple(final), eval_reps)
            per_mrep_scores.append(score)
        base_score = statistics.mean(per_mrep_scores) if per_mrep_scores else math.inf

    # For each factor do one-at-a-time sweep
    for factor, meta in specs.items():
        default = meta.get("default")
        datatype = meta.get("datatype", type(default))

        # If no default and solver-specific refined candidates exist, use them directly
        if (
            (default is None or default == {})
            and solver_name in REFINED_CANDIDATES
            and factor in REFINED_CANDIDATES[solver_name]
        ):
            candidates = REFINED_CANDIDATES[solver_name][factor]
        else:
            candidates = make_candidates(
                default, datatype, solver_name=solver_name, factor=factor
            )
        tried = []
        best_val = None
        best_score = math.inf

        for val in candidates:
            fixed = {factor: val}
            # Validate candidate by attempting to construct the solver config.
            try:
                _ = instantiate_solver(solver_name, fixed_factors=fixed)
            except Exception as e:
                logging.info("Skipping invalid candidate for %s=%s: %s", factor, val, e)
                tried.append((val, math.inf))
                continue
            solver = instantiate_solver(solver_name, fixed_factors=fixed)
            try:
                sol_df, _it_df, _elapsed = run_solver(
                    solver, problem, n_macroreps=n_macroreps, n_jobs=n_jobs
                )
            except Exception as e:
                logging.warning(
                    "Run failed for %s=%s on %s/%s: %s",
                    factor,
                    val,
                    solver_name,
                    problem_name,
                    e,
                )
                score = math.inf
            else:
                per_mrep_scores = []
                for mrep in range(n_macroreps):
                    try:
                        final = sol_df[sol_df["mrep"] == mrep].iloc[-1]["solution"]
                    except Exception:
                        continue
                    score = evaluate_solution(problem, tuple(final), eval_reps)
                    per_mrep_scores.append(score)
                score = (
                    statistics.mean(per_mrep_scores) if per_mrep_scores else math.inf
                )

            tried.append((val, score))
            if score < best_score:
                best_score = score
                best_val = val

        results.append(
            {
                "solver": solver_name,
                "problem": problem_name,
                "dim": dim,
                "factor": factor,
                "default": default,
                "tested": [t[0] for t in tried],
                "tested_scores": [t[1] for t in tried],
                "best": best_val,
                "best_score": best_score,
                "baseline_score": base_score,
            }
        )

    return results


def load_hpc_config(path: str) -> dict:
    """Load hpc config."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    return json.loads(p.read_text())
    # Return the config as-is (no forced solver filtering)


def aggregate_recommendations(results: list[dict]) -> dict:
    """Aggregate best factor values across problems for one solver.

    For numeric values returns the mean; for non-numeric returns the mode.
    """
    from collections import Counter

    by_factor: dict[str, list] = {}
    for r in results:
        factor = r["factor"]
        best = r["best"]
        by_factor.setdefault(factor, []).append(best)

    agg: dict[str, Any] = {}
    for factor, vals in by_factor.items():
        vals = [v for v in vals if v is not None and v != math.inf]
        if not vals:
            agg[factor] = None
            continue
        if all(isinstance(v, int | float) for v in vals):
            mean = float(sum(vals) / len(vals))
            if all(isinstance(v, int) for v in vals):
                agg[factor] = round(mean)
            else:
                agg[factor] = mean
        else:
            cnt = Counter(vals)
            agg[factor] = cnt.most_common(1)[0][0]
    return agg


def main() -> None:
    """Run main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="HPC_code/config.json",
        help="Path to HPC config JSON",
    )
    parser.add_argument("--n-macroreps", type=int, default=3)
    parser.add_argument("--eval-reps", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument(
        "--solvers",
        type=str,
        nargs="*",
        help="Solver abbreviations to test (default: those in config)",
    )
    parser.add_argument(
        "--problems",
        type=str,
        nargs="*",
        help="Problem abbreviations to test (default: those in config)",
    )
    args = parser.parse_args()

    cfg = load_hpc_config(args.config)
    cfg_solvers = cfg.get("solver_names", [])
    cfg_problems = cfg.get("problem_names", [])
    cfg_fixed = cfg.get("fixed_factors", [])
    cfg_dim_sizes = cfg.get("dim_sizes", [])
    cfg.get("budgets", [50_000])
    budget = 10_000

    # Use None to indicate "default dimension (no scaling)"
    dim: int | None = cfg_dim_sizes[0] if cfg_dim_sizes else None

    solvers = args.solvers or cfg_solvers or discover_solver_names()
    problems = args.problems or cfg_problems or discover_problem_names()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    recommendations = {}

    # If solvers were specified via CLI, don't restrict to config factor_keys
    use_cli_solvers = args.solvers is not None and len(args.solvers) > 0

    for solver_idx, solver in enumerate(solvers):
        factor_keys = None
        if not use_cli_solvers and solver_idx < len(cfg_fixed):
            fixed_map = cfg_fixed[solver_idx]
            factor_keys = [k for k in fixed_map if k != "crn_across_solns"]
        # When using CLI solvers, use REFINED_CANDIDATES keys if available
        if use_cli_solvers and solver in REFINED_CANDIDATES:
            factor_keys = list(REFINED_CANDIDATES[solver].keys())

        solver_results = []
        for problem in problems:
            try:
                res = run_one_factor_sweep(
                    solver,
                    problem,
                    args.n_macroreps,
                    args.eval_reps,
                    args.n_jobs,
                    factor_keys=factor_keys,
                    dim=dim,
                    budget=budget,
                )
            except Exception as e:
                logging.exception(
                    "Error running sweeps for %s on %s (dim=%s): %s",
                    solver,
                    problem,
                    dim,
                    e,
                )
                continue
            # Only filter by factor_keys if not using CLI solvers
            if not use_cli_solvers and factor_keys is not None:
                res = [r for r in res if r["factor"] in factor_keys]
            solver_results.extend(res)
            all_results.extend(res)

        agg = aggregate_recommendations(solver_results)
        recommendations[solver] = agg

    df = pd.DataFrame(all_results)
    csv_path = out_dir / "sensitivity_solver_factors.csv"
    json_path = out_dir / "sensitivity_solver_factors.json"
    df.to_csv(csv_path, index=False)
    with json_path.open("w") as f:
        json.dump(all_results, f, default=str, indent=2)

    rec_path = out_dir / "sensitivity_solver_recommendations.json"
    with rec_path.open("w") as f:
        json.dump(recommendations, f, default=str, indent=2)

    logging.info("Wrote results: %s and %s", csv_path, json_path)
    logging.info("Wrote recommendations: %s", rec_path)


if __name__ == "__main__":
    main()
