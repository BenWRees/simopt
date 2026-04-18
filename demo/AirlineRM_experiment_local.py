"""experiment runner for VANRYZIN-2 solver comparisons for local machine execution.

This script runs two experiments:

1. ADP_SOLVER (wrapped ASTROMORF) vs DAVN vs SGD (solver lookahead enabled).
2. ADP_SOLVER with wrapped solvers: ASTROMORF, ASTRODF, SGD, NELDMD.

For each problem-solver run, the script records:
- runtime statistics from macroreplications,
- per-leg load factors,
 - mean final objective (averaged across macroreplications),
- iteration statistics,
- and metadata required for local plotting.

This script is designed for local execution and saves outputs (CSVs and pickles) in a
structured directory format.
"""

# Standard imports
from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import pandas as pd

from mrg32k3a.mrg32k3a import MRG32k3a

# Project-specific imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
from simopt.experiment_base import ProblemSolver, ProblemsSolvers, instantiate_problem

# ==== HARDCODED PARAMETERS FOR LOCAL RUN ====
# These values are taken from run_airline_rm_experiment.slurm
BUDGET = 50000
N_MACROREPS = 20
N_LOOKAHEAD_REPS = 10
N_LOADFACTOR_REPS = 200
N_JOBS = 1  # For local, use 1 to avoid parallel issues
OUTPUT_DIR = "outputs/airline_hpc_experiments"
EXPERIMENT_NAME = "adp_vs_davn_sgd"  # Change as needed
SOLVER_LABEL = "ADP_ASTROMoRF"  # Change as needed


# Logger initialization
LOGGER = logging.getLogger(__name__)

# Mapping from internal experiment names to user-facing output folder names.
EXP_OUTPUT_DIRS: dict[str, str] = {
    "adp_vs_davn_sgd": "WSC2026_experiment1",
    "adp_wrapped_solver_comparison": "WSC2026_experiment2",
}


# SolverSpec class
class SolverSpec:
    """Configuration for a solver run."""

    def __init__(
        self, solver_name: str, solver_label: str, solver_factors: dict[str, Any]
    ) -> None:
        """Initialize instance."""
        self.solver_name = solver_name
        self.solver_label = solver_label
        self.solver_factors = solver_factors


# Utility: parse comma-separated list
def _parse_csv_list(text: str) -> list[str]:
    values = [chunk.strip() for chunk in text.split(",") if chunk.strip()]
    if not values:
        raise ValueError("Expected a non-empty comma-separated list.")
    return values


# Utility: recommended ADP solver factors
def _recommended_adp_solver_factors(
    wrapped_solver: str,
    budget: int,
    wrapped_solver_factors: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ADP_GP_INPUT_DIM = 90  # noqa: N806
    ADP_STAGES_TO_FIT = 3  # noqa: N806
    forward_pass_budget_fraction = 0.35
    target_inner_budget_per_sample = 5
    min_training_pts = 120
    max_training_pts = 450
    dim_scaled_training_pts = 4 * ADP_GP_INPUT_DIM
    backward_budget = (1.0 - forward_pass_budget_fraction) * float(max(1, budget))
    budget_limited_training_pts = int(
        backward_budget // max(1, target_inner_budget_per_sample * ADP_STAGES_TO_FIT)
    )
    n_training_pts = max(
        min_training_pts,
        min(
            dim_scaled_training_pts,
            max_training_pts,
            max(1, budget_limited_training_pts),
        ),
    )
    return {
        "wrapped_solver": wrapped_solver,
        "wrapped_solver_factors": wrapped_solver_factors or {},
        "n_training_pts": int(n_training_pts),
        "n_macroreps": 4,
        "n_verification_macroreps": 8,
        "forward_pass_budget_fraction": forward_pass_budget_fraction,
    }


# Utility: extract iterations
def _extract_iterations(
    problem_solver: ProblemSolver,
) -> tuple[float | None, float | None]:
    all_iterations = getattr(problem_solver, "all_iterations", None)
    if all_iterations is None:
        return None, None
    terminal_iters: list[float] = []
    for seq in all_iterations:
        if seq:
            terminal_iters.append(float(seq[-1]))
    if not terminal_iters:
        return None, None
    from statistics import mean

    return float(mean(terminal_iters)), float(max(terminal_iters))


# Utility: extract mean final objective
def _extract_mean_final_objective(problem_solver: ProblemSolver) -> float | None:
    all_fn_estimates = getattr(problem_solver, "all_fn_estimates", None)
    if not all_fn_estimates:
        return None
    terminal_estimates: list[float] = []
    for seq in all_fn_estimates:
        if seq:
            terminal_estimate = float(seq[-1])
            if np.isfinite(terminal_estimate):
                terminal_estimates.append(terminal_estimate)
    if not terminal_estimates:
        return None
    if int(problem_solver.problem.minmax[0]) == 1:
        terminal_estimates = [-estimate for estimate in terminal_estimates]
    from statistics import mean

    return float(mean(terminal_estimates))


# Utility: build problem
def _build_problem(
    budget: int,
    n_lookahead_reps: int,
) -> Any:  # noqa: ANN401
    problem_fixed_factors = {
        "budget": int(budget),
        "n_lookahead_reps": int(n_lookahead_reps),
    }
    return instantiate_problem(
        "VANRYZIN-2",
        problem_fixed_factors=problem_fixed_factors,
    )


def _best_found_solution(problem_solver: ProblemSolver) -> tuple[float, ...]:
    all_xs = getattr(problem_solver, "all_recommended_xs", None)
    if not all_xs:
        raise RuntimeError("No recommended solutions were recorded for this run.")

    all_fn = getattr(problem_solver, "all_fn_estimates", None)
    minmax = int(problem_solver.problem.minmax[0])

    if all_fn:
        best_x: tuple[float, ...] | None = None
        best_val: float | None = None

        for mrep_idx, fn_seq in enumerate(all_fn):
            if not fn_seq:
                continue
            xs_seq = all_xs[mrep_idx] if mrep_idx < len(all_xs) else []
            if not xs_seq:
                continue

            candidate_val = float(fn_seq[-1])
            # FIXME: handle the case where xs_seq[-1] is a list of lists (e.g. from
            # multiple checkpoints per macrorep)
            if [
                type(xs_seq[-1][a]) in [list, np.ndarray, tuple]
                for a in range(len(xs_seq[-1]))
            ].count(True) > 0:
                # Likely that xs_seq[-1] is a tuple of tuples for each stage's decision
                # variables,
                # we want the stage 0 decision variables as the candidate solution for
                # comparison
                # since the later stages may not be fully trained
                candidate_x = tuple(float(v) for v in xs_seq[-1][0])
            else:  # normal case where xs_seq[-1] is a list of decision variable values
                candidate_x = tuple(float(v) for v in xs_seq[-1])

            if (
                best_val is None
                or (minmax == 1 and candidate_val > best_val)
                or (minmax != 1 and candidate_val < best_val)
            ):
                best_val = candidate_val
                best_x = candidate_x

        if best_x is not None:
            return best_x

    for xs_seq in all_xs:
        if xs_seq:
            return tuple(float(v) for v in xs_seq[-1])

    raise RuntimeError("Unable to recover a terminal recommended solution.")


def _estimate_leg_load_factors(
    problem: Any,  # noqa: ANN401
    decision: tuple[float, ...],
    n_reps: int,
) -> list[float]:
    model = problem.model
    capacity = np.asarray(model.factors["capacity"], dtype=float)
    if n_reps <= 0:
        return [float("nan")] * len(capacity)

    mean_leg_sold = np.zeros_like(capacity, dtype=float)

    for rep_idx in range(n_reps):
        rng_list = [
            MRG32k3a(s_ss_sss_index=[11_000 + rep_idx, rng_idx, 0])
            for rng_idx in range(model.n_rngs)
        ]

        model.before_replication(rng_list)
        state = model.get_initial_state()

        for stage in range(model.n_stages):
            _, state = model.replicate_stage(state, decision, stage, rng_list)

        remaining_capacity = np.asarray(state["remaining_capacity"], dtype=float)
        mean_leg_sold += capacity - remaining_capacity

    mean_leg_sold /= float(n_reps)
    load_factor_per_leg = np.divide(
        mean_leg_sold,
        capacity,
        out=np.full_like(mean_leg_sold, np.nan, dtype=float),
        where=capacity > 0,
    )
    return [float(v) for v in load_factor_per_leg]


def _run_problem_solver(
    solver_spec: SolverSpec,
    n_macroreps: int,
    n_jobs: int,
    budget: int,
    n_lookahead_reps: int,
    n_loadfactor_reps: int,
    output_dir: Path,
) -> tuple[ProblemSolver, dict[str, Any]]:
    problem = _build_problem(
        budget=budget,
        n_lookahead_reps=n_lookahead_reps,
    )

    pair_pickle_name = f"{solver_spec.solver_label}_on_{problem.name}.pickle"
    pair_pickle_path = output_dir / "pairs" / pair_pickle_name

    ps = ProblemSolver(
        solver_name=solver_spec.solver_name,
        problem=problem,
        solver_rename=solver_spec.solver_label,
        solver_fixed_factors=solver_spec.solver_factors,
        file_name_path=pair_pickle_path,
        create_pickle=False,
    )

    ps.run(n_macroreps=n_macroreps, n_jobs=n_jobs)

    avg_runtime = float(mean(ps.timings)) if ps.timings else None
    total_runtime = float(sum(ps.timings)) if ps.timings else None
    iter_mean, iter_max = _extract_iterations(ps)
    mean_final_objective = _extract_mean_final_objective(ps)
    best_x = _best_found_solution(ps)
    leg_load_factors = _estimate_leg_load_factors(
        problem=problem,
        decision=best_x,
        n_reps=n_loadfactor_reps,
    )

    row = {
        "problem": ps.problem.name,
        "solver": solver_spec.solver_label,
        "solver_name": solver_spec.solver_name,
        "load_factor": leg_load_factors,
        "load_factor_mean": float(
            np.nanmean(np.asarray(leg_load_factors, dtype=float))
        ),
        "demand_proxy_mean": float(
            np.mean(
                np.asarray(problem.model.factors["gamma_shape"])
                * np.asarray(problem.model.factors["gamma_scale"])
            )
        ),
        "n_legs": len(leg_load_factors),
        "best_solution_for_load_factor": best_x,
        "n_loadfactor_reps": n_loadfactor_reps,
        "n_macroreps": n_macroreps,
        "budget": budget,
        "n_jobs": n_jobs,
        "runtime_mean_seconds": avg_runtime,
        "runtime_total_seconds": total_runtime,
        "iterations_mean": iter_mean,
        "iterations_max": iter_max,
        "mean_final_objective": mean_final_objective,
        "expected_optimality_gap": mean_final_objective,
        "has_coded_optimal_value": getattr(ps.problem, "optimal_value", None)
        is not None,
    }
    return ps, row


def _run_experiment_grid(
    experiment_name: str,
    solver_specs: list[SolverSpec],
    n_macroreps: int,
    n_jobs: int,
    budget: int,
    n_lookahead_reps: int,
    n_loadfactor_reps: int,
    output_dir: Path,
) -> tuple[list[ProblemSolver], list[SolverSpec], pd.DataFrame]:
    LOGGER.info("Running %s with %d solvers.", experiment_name, len(solver_specs))

    rows: list[dict[str, Any]] = []
    problem_solvers: list[ProblemSolver] = []

    total_runs = len(solver_specs)
    start_all = time.time()

    for run_idx, solver_spec in enumerate(solver_specs, 1):
        LOGGER.info(
            "[%s] Run %d/%d: solver=%s",
            experiment_name,
            run_idx,
            total_runs,
            solver_spec.solver_label,
        )
        ps, row = _run_problem_solver(
            solver_spec=solver_spec,
            n_macroreps=n_macroreps,
            n_jobs=n_jobs,
            budget=budget,
            n_lookahead_reps=n_lookahead_reps,
            n_loadfactor_reps=n_loadfactor_reps,
            output_dir=output_dir,
        )
        problem_solvers.append(ps)
        row["experiment"] = experiment_name
        rows.append(row)

    elapsed = time.time() - start_all
    LOGGER.info("Completed %s in %.2f seconds.", experiment_name, elapsed)
    return problem_solvers, solver_specs, pd.DataFrame(rows)


def _write_pickle(obj: Any, path: Path) -> None:  # noqa: ANN401
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(obj, fh, pickle.HIGHEST_PROTOCOL)


def _build_experiment_one_specs(budget: int) -> list[SolverSpec]:
    return [
        SolverSpec(
            solver_name="ADP_SOLVER",
            solver_label="ADP_ASTROMoRF",
            solver_factors=_recommended_adp_solver_factors(
                wrapped_solver="ASTROMORF",
                budget=budget,
            ),
        ),
        SolverSpec(
            solver_name="DAVN",
            solver_label="DAVN",
            solver_factors={},
        ),
        SolverSpec(
            solver_name="SGD",
            solver_label="SGD",
            solver_factors={},
        ),
    ]


def _build_experiment_two_specs(budget: int) -> list[SolverSpec]:
    return [
        SolverSpec(
            solver_name="ADP_SOLVER",
            solver_label="ADP_ASTROMoRF",
            solver_factors=_recommended_adp_solver_factors(
                wrapped_solver="ASTROMORF",
                budget=budget,
            ),
        ),
        SolverSpec(
            solver_name="ADP_SOLVER",
            solver_label="ADP_ASTRO-DF",
            solver_factors=_recommended_adp_solver_factors(
                wrapped_solver="ASTRODF",
                budget=budget,
            ),
        ),
        SolverSpec(
            solver_name="ADP_SOLVER",
            solver_label="ADP_SGD",
            solver_factors=_recommended_adp_solver_factors(
                wrapped_solver="SGD",
                budget=budget,
                wrapped_solver_factors={
                    # "finite_diff_use_greedy_lookahead" removed
                },
            ),
        ),
        SolverSpec(
            solver_name="ADP_SOLVER",
            solver_label="ADP_NELDER-MEAD",
            solver_factors=_recommended_adp_solver_factors(
                wrapped_solver="NELDMD",
                budget=budget,
            ),
        ),
    ]


def _all_experiment_specs(budget: int) -> dict[str, list[SolverSpec]]:
    return {
        "adp_vs_davn_sgd": _build_experiment_one_specs(budget=budget),
        "adp_wrapped_solver_comparison": _build_experiment_two_specs(budget=budget),
    }


def _filter_specs_by_solver_labels(
    solver_specs: list[SolverSpec],
    allowed_labels: set[str] | None,
) -> list[SolverSpec]:
    if not allowed_labels:
        return solver_specs
    return [spec for spec in solver_specs if spec.solver_label in allowed_labels]


def parse_args() -> argparse.Namespace:
    """Parse args."""
    parser = argparse.ArgumentParser(description="Run VANRYZIN-2 HPC experiments.")
    parser.add_argument("--n-macroreps", type=int, default=20)
    parser.add_argument("--budget", type=int, default=50000)
    parser.add_argument("--n-lookahead-reps", type=int, default=10)
    parser.add_argument(
        "--n-loadfactor-reps",
        type=int,
        default=200,
        help="Replications used to estimate per-leg load factors at each solver's best-found solution.",  # noqa: E501
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of cores for macrorep parallelization inside each run.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/airline_hpc_experiments",
        help="Directory where CSV and pickle outputs are saved.",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default="all",
        help=(
            "Comma-separated experiment names to run: "
            "adp_vs_davn_sgd,adp_wrapped_solver_comparison, or 'all'."
        ),
    )
    parser.add_argument(
        "--solver-labels",
        type=str,
        default="",
        help="Optional comma-separated solver labels to run within selected experiments.",  # noqa: E501
    )
    return parser.parse_args()


def main() -> None:
    """Run main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()

    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Output directory: %s", run_dir)

    all_specs = _all_experiment_specs(budget=int(args.budget))
    if args.experiments.strip().lower() == "all":
        selected_experiment_names = list(all_specs.keys())
    else:
        selected_experiment_names = _parse_csv_list(args.experiments)

    unknown = [name for name in selected_experiment_names if name not in all_specs]
    if unknown:
        raise ValueError(f"Unknown experiment name(s): {unknown}")

    allowed_solver_labels = (
        set(_parse_csv_list(args.solver_labels)) if args.solver_labels.strip() else None
    )

    all_problem_solvers: list[ProblemSolver] = []
    all_frames: list[pd.DataFrame] = []
    per_experiment_csvs: list[Path] = []
    per_experiment_pickles: list[Path] = []

    for exp_name in selected_experiment_names:
        solver_specs = _filter_specs_by_solver_labels(
            all_specs[exp_name],
            allowed_labels=allowed_solver_labels,
        )

        if not solver_specs:
            LOGGER.warning(
                "Skipping experiment %s because no solver labels matched filter=%s.",
                exp_name,
                sorted(allowed_solver_labels) if allowed_solver_labels else None,
            )
            continue

        problem_solvers, used_specs, exp_df = _run_experiment_grid(
            experiment_name=exp_name,
            solver_specs=solver_specs,
            n_macroreps=int(args.n_macroreps),
            n_jobs=int(args.n_jobs),
            budget=int(args.budget),
            n_lookahead_reps=int(args.n_lookahead_reps),
            n_loadfactor_reps=int(args.n_loadfactor_reps),
            output_dir=run_dir,
        )

        # ── Write outputs into the experiment-specific folder ────────────
        exp_folder_name = EXP_OUTPUT_DIRS.get(exp_name, exp_name)
        exp_output_dir = run_dir / exp_folder_name
        exp_output_dir.mkdir(parents=True, exist_ok=True)

        # CSV summary
        exp_csv = exp_output_dir / f"{exp_name}.csv"
        exp_df.to_csv(exp_csv, index=False)

        # Pickle each ProblemSolver with a clear label
        for ps, spec in zip(problem_solvers, used_specs, strict=False):
            ps_path = (
                exp_output_dir / f"{spec.solver_label}_on_{ps.problem.name}.pickle"
            )
            _write_pickle(ps, ps_path)
            LOGGER.info("Saved ProblemSolver: %s", ps_path)

        # Build and pickle a ProblemsSolvers for local post-processing
        ps_grid = [[ps] for ps in problem_solvers]
        meta = ProblemsSolvers(
            experiments=ps_grid,
            file_name_path=exp_output_dir / f"{exp_name}_ProblemsSolvers.pickle",
            experiment_name=exp_name,
        )
        meta.record_group_experiment_results()
        LOGGER.info("Saved ProblemsSolvers: %s", meta.file_name_path)

        per_experiment_csvs.append(exp_csv)
        if meta.file_name_path is not None:
            per_experiment_pickles.append(meta.file_name_path)
        all_problem_solvers.extend(problem_solvers)
        all_frames.append(exp_df)

    if not all_frames:
        raise RuntimeError(
            "No experiments were run. Check --experiments/--solver-labels filters."
        )

    all_csv = run_dir / "all_experiment_rows.csv"
    all_pickle = run_dir / "all_problemsolvers.pkl"
    pd.concat(all_frames, ignore_index=True).to_csv(all_csv, index=False)
    _write_pickle(all_problem_solvers, all_pickle)

    config_dump = {
        "n_macroreps": int(args.n_macroreps),
        "budget": int(args.budget),
        "n_lookahead_reps": int(args.n_lookahead_reps),
        "n_loadfactor_reps": int(args.n_loadfactor_reps),
        "n_jobs": int(args.n_jobs),
        "experiments": selected_experiment_names,
        "solver_labels_filter": sorted(allowed_solver_labels)
        if allowed_solver_labels
        else None,
        "output_dir": str(run_dir),
    }
    pd.DataFrame([config_dump]).to_csv(run_dir / "run_config.csv", index=False)

    LOGGER.info("Saved per-experiment CSV summaries: %s", per_experiment_csvs)
    LOGGER.info(
        "Saved per-experiment ProblemSolver pickles: %s", per_experiment_pickles
    )
    LOGGER.info("Saved merged outputs: %s and %s", all_csv, all_pickle)
    LOGGER.info("HPC run complete. Post-normalize and plotting can be done locally.")


if __name__ == "__main__":
    main()
