"""HPC experiment runner for VANRYZIN-2 solver comparisons.

This script runs two experiments:

1. ADP_SOLVER (wrapped ASTROMORF) vs DAVN vs SGD (solver lookahead enabled).
2. ADP_SOLVER with wrapped solvers: ASTROMORF, ASTRODF, SGD, NELDMD.

For each problem-solver run, the script records:
- runtime statistics from macroreplications,
- per-leg load factors,
 - mean final objective (averaged across macroreplications),
- iteration statistics,
- and metadata required for local plotting.

Subcommands
-----------
run-solver       Run a single ProblemSolver (one SLURM array task).
collect-results  Load all Phase-1 pickles and write one CSV per experiment.
list-tasks       Print the number of tasks (for SLURM --array).
run-all          Original sequential behaviour (local use).
"""

from __future__ import annotations

import argparse
import json
import logging
import numbers
import os.path as o
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import pandas as pd

from mrg32k3a.mrg32k3a import MRG32k3a

# Take the current directory, find the parent, and add it to the system path
sys.path.insert(0, o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))  # type:ignore  # noqa: PTH100, PTH118, PTH120

from simopt.base import MultistageProblem
from simopt.experiment_base import ProblemSolver, ProblemsSolvers, instantiate_problem

LOGGER = logging.getLogger(__name__)

# Mapping from internal experiment names to user-facing output folder names.
EXP_OUTPUT_DIRS: dict[str, str] = {
    "adp_vs_davn_sgd": "WSC2026_experiment1",
    "adp_wrapped_solver_comparison": "WSC2026_experiment2",
}

# GP input dimension for VANRYZIN-2 ADP value model:
# decision variables + remaining-capacity features.
ADP_GP_INPUT_DIM = 90
ADP_STAGES_TO_FIT = 3

PROBLEM_VARIANTS: tuple[str, ...] = (
    "default_fixed_factors",
    "reduced_capacity",
    "increased_demand_variance",
)
CAPACITY_REDUCTION_FACTOR = 0.7
DEMAND_VARIANCE_MULTIPLIER = 3.0


class SolverSpec:
    """Configuration for a solver run."""

    solver_name: str
    solver_label: str
    solver_factors: dict[str, Any]

    def __init__(
        self, solver_name: str, solver_label: str, solver_factors: dict[str, Any]
    ) -> None:
        """Initialize instance."""
        self.solver_name = solver_name
        self.solver_label = solver_label
        self.solver_factors = solver_factors


def _recommended_adp_solver_factors(
    wrapped_solver: str,
    budget: int,
    wrapped_solver_factors: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return ADP solver factors tuned for a 90D->1D GP fit.

    The backward fit trains one GP per stage for stages 1..T. For VANRYZIN-2,
    this means 3 fitted stages. We choose training points so each sampled state
    still receives enough inner-solver budget for a stable value estimate.
    """
    forward_pass_budget_fraction = 0.35
    target_inner_budget_per_sample = 5
    min_training_pts = 200
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
        "n_verification_macroreps": 8,
        "forward_pass_budget_fraction": forward_pass_budget_fraction,
        "refinement_buffer_max": 0.8 * int(n_training_pts),
        "n_macroreps_forward": 20,
        "n_mc_replicates": 20,
    }


def _parse_csv_list(text: str) -> list[str]:
    values = [chunk.strip() for chunk in text.split(",") if chunk.strip()]
    if not values:
        raise ValueError("Expected a non-empty comma-separated list.")
    return values


def _build_problem(
    budget: int,
    problem_variant: str,
) -> Any:  # noqa: ANN401
    problem_fixed_factors: dict[str, Any] = {
        "budget": int(budget),
    }

    if problem_variant == "default_fixed_factors":
        return instantiate_problem(
            "VANRYZIN-2",
            problem_fixed_factors=problem_fixed_factors,
        )

    base_problem = instantiate_problem(
        "VANRYZIN-2",
        problem_fixed_factors=problem_fixed_factors,
    )
    base_model_factors = base_problem.model.factors

    if problem_variant == "reduced_capacity":
        capacity = np.asarray(base_model_factors["capacity"], dtype=float)
        protection_levels = np.asarray(
            base_model_factors["protection_levels"], dtype=float
        )
        scaled_protection_levels = CAPACITY_REDUCTION_FACTOR * protection_levels

        model_fixed_factors = {
            "capacity": tuple(float(v) for v in CAPACITY_REDUCTION_FACTOR * capacity),
            "protection_levels": scaled_protection_levels.tolist(),
        }
        problem_fixed_factors["initial_solution"] = tuple(
            float(a) for v in scaled_protection_levels for a in v
        )
    elif problem_variant == "increased_demand_variance":
        gamma_shape = np.asarray(base_model_factors["gamma_shape"], dtype=float)
        gamma_scale = np.asarray(base_model_factors["gamma_scale"], dtype=float)

        model_fixed_factors = {
            "gamma_shape": tuple(
                float(v) for v in gamma_shape / DEMAND_VARIANCE_MULTIPLIER
            ),
            "gamma_scale": tuple(
                float(v) for v in gamma_scale * DEMAND_VARIANCE_MULTIPLIER
            ),
        }
    else:
        raise ValueError(f"Unknown problem variant: {problem_variant}")

    return instantiate_problem(
        "VANRYZIN-2",
        problem_fixed_factors=problem_fixed_factors,
        model_fixed_factors=model_fixed_factors,
    )


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
    return float(mean(terminal_iters)), float(max(terminal_iters))


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
    return float(mean(terminal_estimates))


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
            is_flat_multistage = (
                isinstance(problem_solver.problem, MultistageProblem)
                and isinstance(xs_seq[-1], list | np.ndarray | tuple)
                and len(xs_seq[-1])
                == problem_solver.problem.model.n_stages * problem_solver.problem.dim
                and len(xs_seq[-1]) > 0
                and isinstance(xs_seq[-1][0], numbers.Number)
            )
            is_nested_multistage = (
                isinstance(xs_seq[-1], list | np.ndarray | tuple)
                and len(xs_seq[-1]) > 0
                and isinstance(xs_seq[-1][0], list | np.ndarray | tuple)
            )

            if is_nested_multistage:
                # if any(not isinstance(x, numbers.Number) for x in xs_seq[-1]):
                candidate_x = tuple(float(v) for v in xs_seq[-1][0])
            elif is_flat_multistage:
                stage_dim = problem_solver.problem.dim
                candidate_x = tuple(float(v) for v in xs_seq[-1][:stage_dim])
            else:
                # normal case where xs_seq[-1] is a list of decisions for the first
                # stage
                candidate_x = tuple(float(v) for v in xs_seq[-1])

            if best_val is None or minmax * candidate_val > minmax * best_val:
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


# ---------------------------------------------------------------------------
# Shared helpers: task manifest, pickle paths, metric extraction
# ---------------------------------------------------------------------------


def _compute_optimality_gap_pct(
    optimal_value: float | None,
    mean_final_objective: float | None,
) -> float | None:
    """Return percentage optimality gap: 100 * (opt - obj) / opt."""
    if optimal_value is None or mean_final_objective is None or optimal_value == 0:
        return None
    return 100.0 * (optimal_value - abs(mean_final_objective)) / optimal_value


def extract_metrics_row(
    problem_solver: ProblemSolver,
    solver_spec: SolverSpec,
    n_loadfactor_reps: int,
) -> dict[str, Any]:
    """Build a single CSV-row dict from a completed ProblemSolver."""
    problem = problem_solver.problem
    avg_runtime = (
        float(mean(problem_solver.timings)) if problem_solver.timings else None
    )
    total_runtime = (
        float(sum(problem_solver.timings)) if problem_solver.timings else None
    )
    iter_mean, iter_max = _extract_iterations(problem_solver)
    mean_final_objective = _extract_mean_final_objective(problem_solver)
    best_x = _best_found_solution(problem_solver)
    leg_load_factors = _estimate_leg_load_factors(
        problem=problem,
        decision=best_x,
        n_reps=n_loadfactor_reps,
    )

    return {
        "problem": problem.name,
        "solver": solver_spec.solver_label,
        "solver_name": solver_spec.solver_name,
        "load_factor": leg_load_factors,
        "load_factor_mean": float(np.mean(np.asarray(leg_load_factors, dtype=float))),
        "demand_proxy_mean": float(
            np.mean(
                np.asarray(problem.model.factors["gamma_shape"])
                * np.asarray(problem.model.factors["gamma_scale"])
            )
        ),
        "n_legs": len(leg_load_factors),
        "best_solution_for_load_factor": best_x,
        "n_loadfactor_reps": n_loadfactor_reps,
        "n_macroreps": problem_solver.n_macroreps,
        "budget": int(problem.factors["budget"]),
        "n_jobs": getattr(problem_solver, "n_jobs", None),
        "runtime_mean_seconds": avg_runtime,
        "runtime_total_seconds": total_runtime,
        "iterations_mean": iter_mean,
        "iterations_max": iter_max,
        "mean_final_objective": mean_final_objective,
        "optimal_value": getattr(problem, "optimal_value", None),
        "has_coded_optimal_value": getattr(problem, "optimal_value", None) is not None,
        "optimality_gap_pct": _compute_optimality_gap_pct(
            getattr(problem, "optimal_value", None),
            mean_final_objective,
        ),
    }


def _build_experiment_one_specs(budget: int) -> list[SolverSpec]:
    return [
        # SolverSpec(
        # 	solver_name="VANRYZIN_SGD",
        # 	solver_label="SGD with IPA Gradients",
        # 	solver_factors={
        # 		"use_direct_gradients": True,
        # 		"r": 10,
        # 		"not_use_adp_solver": True,
        # 		"alpha": 0.9,
        #         "gradient_clipping_enabled": True,
        #         "gradient_clipping_value": 20.0,
        #         "spsa_gradient": False,
        # 	},
        # ),
        # SolverSpec(
        # 	solver_name="ADP_SOLVER",
        # 	solver_label="ADP_SGD",
        # 	solver_factors=_recommended_adp_solver_factors(
        # 		wrapped_solver="SGD",
        # 		budget=budget,
        # 		wrapped_solver_factors={
        # 			"r": 10,
        # 			"alpha": 0.9,
        # 			"gradient_clipping_enabled": True,
        # 			"gradient_clipping_value": 20.0,
        # 			"spsa_gradient": False,
        # 			"not_use_adp_solver": False,
        # 		},
        # 	),
        # ),
        SolverSpec(
            solver_name="ADP_SOLVER",
            solver_label="ADP_ASTROMoRF",
            solver_factors=_recommended_adp_solver_factors(
                wrapped_solver="ASTROMORF",
                budget=budget,
                wrapped_solver_factors={
                    "initial subspace dimension": 4,
                    "polynomial degree": 4,
                },
            ),
        )
        # SolverSpec(
        # 	solver_name="DAVN",
        # 	solver_label="DAVN",
        # 	solver_factors={},
        # ),
        # SolverSpec(
        # 	solver_name="SGD",
        # 	solver_label="SGD (FD)",
        # 	solver_factors={
        # 		"r": 10,
        #         "alpha": 0.9,
        #         "gradient_clipping_enabled": True,
        #         "gradient_clipping_value": 20.0,
        #         "spsa_gradient": False,
        # 		"not_use_adp_solver": True,
        # 	},
        # ),
    ]


def _build_experiment_two_specs(budget: int) -> list[SolverSpec]:
    return [
        SolverSpec(
            solver_name="ADP_SOLVER",
            solver_label="ADP_ASTROMoRF",
            solver_factors=_recommended_adp_solver_factors(
                wrapped_solver="ASTROMORF",
                budget=budget,
                wrapped_solver_factors={
                    "initial subspace dimension": 4,
                    "polynomial degree": 4,
                },
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
                    "r": 10,
                    "alpha": 0.9,
                    "gradient_clipping_enabled": True,
                    "gradient_clipping_value": 20.0,
                    "spsa_gradient": False,
                    "not_use_adp_solver": False,  # Ensure inner SGD evaluations account for ADP budget  # noqa: E501
                },
            ),
        ),
        SolverSpec(
            solver_name="ADP_SOLVER",
            solver_label="ADP_NELDER-MEAD",
            solver_factors=_recommended_adp_solver_factors(
                wrapped_solver="NELDMD", budget=budget, wrapped_solver_factors={"r": 5}
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


def build_task_manifest(
    budget: int,
    selected_experiments: list[str],
    allowed_solver_labels: set[str] | None,
) -> list[dict[str, Any]]:
    """Return a deterministic list of tasks (one per ProblemSolver).

    Both ``run-solver`` and ``collect-results`` call this with the same
    arguments so they agree on the task-index-to-(experiment, solver, problem variant)
    mapping.
    """
    all_specs = _all_experiment_specs(budget=budget)
    manifest: list[dict[str, Any]] = []
    task_index = 0
    for exp_name in selected_experiments:
        if exp_name not in all_specs:
            raise ValueError(f"Unknown experiment: {exp_name}")
        specs = _filter_specs_by_solver_labels(
            all_specs[exp_name], allowed_solver_labels
        )
        for spec in specs:
            for problem_variant in PROBLEM_VARIANTS:
                manifest.append(
                    {
                        "task_index": task_index,
                        "experiment_name": exp_name,
                        "problem_variant": problem_variant,
                        "solver_spec": spec,
                    }
                )
                task_index += 1
    return manifest


def _sanitize_path_token(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def _experiment_pickle_suffix(experiment_name: str) -> str:
    if experiment_name == "adp_vs_davn_sgd":
        return "1"
    if experiment_name == "adp_wrapped_solver_comparison":
        return "2"
    raise ValueError(f"Unknown experiment name for pickle suffix: {experiment_name}")


def phase1_pickle_path(
    run_dir: Path,
    experiment_name: str,
    problem_variant: str,
    solver_label: str,
) -> Path:
    """Canonical location for a Phase-1 ProblemSolver pickle."""
    safe_label = _sanitize_path_token(solver_label)
    safe_variant = _sanitize_path_token(problem_variant)
    experiment_suffix = _experiment_pickle_suffix(experiment_name)
    file_name = f"{safe_label}_variant_{safe_variant}_{experiment_suffix}.pickle"
    return run_dir / "phase1" / experiment_name / safe_variant / file_name


def _load_pickle(path: Path) -> Any:  # noqa: ANN401
    with path.open("rb") as fh:
        return pickle.load(fh)


# ---------------------------------------------------------------------------
# Shared argument parsing
# ---------------------------------------------------------------------------


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared across all subcommands."""
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
        default="outputs",
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
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Optional run identifier for output subdirectory (useful for Slurm arrays).",  # noqa: E501
    )


def _resolve_common_args(
    args: argparse.Namespace,
) -> tuple[Path, list[str], set[str] | None]:
    """Return (run_dir, selected_experiment_names, allowed_solver_labels)."""
    run_dir = Path(args.output_dir)
    if args.run_id.strip():
        run_dir = run_dir / args.run_id.strip()

    all_specs = _all_experiment_specs(budget=int(args.budget))
    if args.experiments.strip().lower() == "all":
        selected = list(all_specs.keys())
    else:
        selected = _parse_csv_list(args.experiments)

    unknown = [name for name in selected if name not in all_specs]
    if unknown:
        raise ValueError(f"Unknown experiment name(s): {unknown}")

    allowed = (
        set(_parse_csv_list(args.solver_labels)) if args.solver_labels.strip() else None
    )
    return run_dir, selected, allowed


# ---------------------------------------------------------------------------
# Subcommand: list-tasks
# ---------------------------------------------------------------------------


def cmd_list_tasks(args: argparse.Namespace) -> None:
    """Print the number of tasks (or full manifest as JSON)."""
    _, selected, allowed = _resolve_common_args(args)
    manifest = build_task_manifest(
        budget=int(args.budget),
        selected_experiments=selected,
        allowed_solver_labels=allowed,
    )
    if args.format == "json":
        out = []
        for task in manifest:
            out.append(
                {
                    "task_index": task["task_index"],
                    "experiment_name": task["experiment_name"],
                    "problem_variant": task["problem_variant"],
                    "solver_label": task["solver_spec"].solver_label,
                    "solver_name": task["solver_spec"].solver_name,
                }
            )
        print(json.dumps(out, indent=2))
    else:
        print(len(manifest))


# ---------------------------------------------------------------------------
# Subcommand: run-solver  (Phase 1 -- one SLURM array task)
# ---------------------------------------------------------------------------


def cmd_run_solver(args: argparse.Namespace) -> None:
    """Create, run, and post-replicate a single ProblemSolver, then pickle it."""
    run_dir, selected, allowed = _resolve_common_args(args)
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_task_manifest(
        budget=int(args.budget),
        selected_experiments=selected,
        allowed_solver_labels=allowed,
    )

    if args.task_index < 0 or args.task_index >= len(manifest):
        raise ValueError(
            f"--task-index {args.task_index} is out of range (0..{len(manifest) - 1})."
        )

    task = manifest[args.task_index]
    exp_name = task["experiment_name"]
    problem_variant = task["problem_variant"]
    solver_spec: SolverSpec = task["solver_spec"]

    LOGGER.info(
        "Task %d: experiment=%s  problem_variant=%s  solver=%s",
        args.task_index,
        exp_name,
        problem_variant,
        solver_spec.solver_label,
    )

    # Build problem and ProblemSolver
    problem = _build_problem(
        budget=int(args.budget),
        # n_lookahead_reps=int(args.n_lookahead_reps),
        problem_variant=problem_variant,
    )

    pickle_path = phase1_pickle_path(
        run_dir=run_dir,
        experiment_name=exp_name,
        problem_variant=problem_variant,
        solver_label=solver_spec.solver_label,
    )
    pickle_path.parent.mkdir(parents=True, exist_ok=True)

    ps = ProblemSolver(
        solver_name=solver_spec.solver_name,
        problem=problem,
        solver_rename=solver_spec.solver_label,
        solver_fixed_factors=solver_spec.solver_factors,
        file_name_path=pickle_path,
        create_pickle=False,
    )

    # Phase 1a: run
    LOGGER.info(
        "Running solver (%d macroreps, %d jobs)...", args.n_macroreps, args.n_jobs
    )
    ps.run(n_macroreps=int(args.n_macroreps), n_jobs=int(args.n_jobs))

    # Phase 1b: post-replicate (policy)
    n_postreps = int(args.n_postreps)
    if n_postreps > 0:
        LOGGER.info("Running post_replicate_policy with %d postreps...", n_postreps)
        ps.post_replicate_policy(n_postreps=n_postreps)

    # Save pickle
    ps.record_experiment_results(str(pickle_path))
    LOGGER.info("Saved ProblemSolver pickle: %s", pickle_path)

    # Save text log alongside
    txt_path = pickle_path.with_suffix(".txt")
    ps.log_experiment_results(file_path=str(txt_path))
    LOGGER.info("Saved text log: %s", txt_path)


# ---------------------------------------------------------------------------
# Subcommand: collect-results  (Phase 2 -- after all array tasks complete)
# ---------------------------------------------------------------------------


def cmd_collect_results(args: argparse.Namespace) -> None:
    """Load Phase-1 pickles and write one CSV per experiment."""
    run_dir, selected, allowed = _resolve_common_args(args)

    manifest = build_task_manifest(
        budget=int(args.budget),
        selected_experiments=selected,
        allowed_solver_labels=allowed,
    )

    # Group tasks by experiment
    tasks_by_exp: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in manifest:
        tasks_by_exp[task["experiment_name"]].append(task)

    all_frames: list[pd.DataFrame] = []

    for exp_name, tasks in tasks_by_exp.items():
        LOGGER.info(
            "Collecting results for experiment: %s (%d solvers)", exp_name, len(tasks)
        )

        problem_solvers: list[ProblemSolver] = []
        solver_specs_by_label: dict[str, SolverSpec] = {}
        rows: list[dict[str, Any]] = []
        missing: list[int] = []

        for task in tasks:
            spec: SolverSpec = task["solver_spec"]
            problem_variant = task["problem_variant"]
            pkl_path = phase1_pickle_path(
                run_dir=run_dir,
                experiment_name=exp_name,
                problem_variant=problem_variant,
                solver_label=spec.solver_label,
            )

            if not pkl_path.exists():
                missing.append(task["task_index"])
                LOGGER.warning(
                    "Missing pickle for task %d (%s): %s",
                    task["task_index"],
                    spec.solver_label,
                    pkl_path,
                )
                continue

            LOGGER.info("Loading %s", pkl_path)
            ps = _load_pickle(pkl_path)
            problem_solvers.append(ps)
            solver_specs_by_label.setdefault(spec.solver_label, spec)

            row = extract_metrics_row(
                problem_solver=ps,
                solver_spec=spec,
                n_loadfactor_reps=int(args.n_loadfactor_reps),
            )
            row["experiment"] = exp_name
            row["problem_variant"] = problem_variant
            rows.append(row)

        if missing and not args.allow_partial:
            raise RuntimeError(
                f"Experiment {exp_name}: missing pickles for task indices {missing}. "
                f"Use --allow-partial to proceed with available results."
            )

        if not rows:
            LOGGER.warning("No results for experiment %s, skipping.", exp_name)
            continue

        exp_df = pd.DataFrame(rows)

        # Write CSV
        exp_folder_name = EXP_OUTPUT_DIRS.get(exp_name, exp_name)
        csv_dir = run_dir / exp_folder_name / "CSV_Files"
        csv_dir.mkdir(parents=True, exist_ok=True)
        exp_csv = csv_dir / f"{exp_name}.csv"
        exp_df.to_csv(exp_csv, index=False)
        LOGGER.info("Wrote CSV: %s (%d rows)", exp_csv, len(rows))

        # Build ProblemsSolvers group pickle
        exp_dir = run_dir / exp_folder_name
        ps_grid: list[list[ProblemSolver]] = []
        for spec in solver_specs_by_label.values():
            spec_ps = [
                ps for ps in problem_solvers if ps.solver.name == spec.solver_label
            ]
            ps_grid.append(spec_ps)

        meta = ProblemsSolvers(
            experiments=ps_grid,
            file_name_path=exp_dir / f"{exp_name}_ProblemsSolvers.pickle",
            experiment_name=exp_name,
        )
        meta.record_group_experiment_results()
        LOGGER.info("Saved ProblemsSolvers: %s", meta.file_name_path)

        all_frames.append(exp_df)

    if not all_frames:
        raise RuntimeError("No experiments had results. Check Phase-1 outputs.")

    all_csv = run_dir / "all_experiment_rows.csv"
    pd.concat(all_frames, ignore_index=True).to_csv(all_csv, index=False)
    LOGGER.info("Wrote merged CSV: %s", all_csv)


# ---------------------------------------------------------------------------
# Subcommand: run-all  (original sequential behaviour)
# ---------------------------------------------------------------------------


def _run_problem_solver(
    solver_spec: SolverSpec,
    experiment_name: str,
    problem_variant: str,
    n_macroreps: int,
    n_jobs: int,
    budget: int,
    n_lookahead_reps: int,  # noqa: ARG001
    n_loadfactor_reps: int,
    n_postreps: int,
    output_dir: Path,
) -> tuple[ProblemSolver, dict[str, Any]]:
    problem = _build_problem(
        budget=budget,
        # n_lookahead_reps=n_lookahead_reps,
        problem_variant=problem_variant,
    )

    pair_pickle_name = Path(
        f"{experiment_name}_{solver_spec.solver_label}_{problem_variant}_on_{problem.name}.pickle"
    )
    pair_pickle_path = output_dir / pair_pickle_name

    ps = ProblemSolver(
        solver_name=solver_spec.solver_name,
        problem=problem,
        solver_rename=solver_spec.solver_label,
        solver_fixed_factors=solver_spec.solver_factors,
        file_name_path=pair_pickle_path,
        create_pickle=True,
    )

    ps.run(n_macroreps=n_macroreps, n_jobs=n_jobs)

    print(
        f"Running Post-Replications and post-normalization for {problem_variant} "
        f"and saving to {pair_pickle_name}"
    )
    ps.post_replicate_policy(n_postreps=n_postreps)
    file_name_pickle = Path(
        f"{experiment_name}_{ps.solver.name}_{problem_variant}_ON_{ps.problem.name}_POSTREPS.pickle"
    )
    if output_dir:
        file_name_pickle = output_dir / file_name_pickle
    ps.record_experiment_results(str(file_name_pickle))
    file_name_text = Path(str(file_name_pickle).replace(".pickle", ".txt"))
    ps.log_experiment_results(file_path=str(file_name_text))

    row = extract_metrics_row(
        problem_solver=ps,
        solver_spec=solver_spec,
        n_loadfactor_reps=n_loadfactor_reps,
    )
    return ps, row


def cmd_run_all(args: argparse.Namespace) -> None:
    """Original sequential behaviour: run every solver, then write CSVs."""
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

    all_frames: list[pd.DataFrame] = []

    for exp_name in selected_experiment_names:
        solver_specs = _filter_specs_by_solver_labels(
            all_specs[exp_name],
            allowed_labels=allowed_solver_labels,
        )

        exp_folder_name = Path(EXP_OUTPUT_DIRS[exp_name])
        exp_dir = run_dir / exp_folder_name

        if not solver_specs:
            LOGGER.warning(
                "Skipping experiment %s because no solver labels matched filter=%s.",
                exp_name,
                sorted(allowed_solver_labels) if allowed_solver_labels else None,
            )
            continue

        rows: list[dict[str, Any]] = []
        problem_solvers: list[ProblemSolver] = []
        used_specs_by_label: dict[str, SolverSpec] = {}

        total_runs = len(solver_specs) * len(PROBLEM_VARIANTS)
        run_idx = 0

        for solver_spec in solver_specs:
            for problem_variant in PROBLEM_VARIANTS:
                run_idx += 1
                LOGGER.info(
                    "[%s] Run %d/%d: solver=%s problem_variant=%s",
                    exp_name,
                    run_idx,
                    total_runs,
                    solver_spec.solver_label,
                    problem_variant,
                )
                ps, row = _run_problem_solver(
                    solver_spec=solver_spec,
                    experiment_name=exp_name,
                    problem_variant=problem_variant,
                    n_macroreps=int(args.n_macroreps),
                    n_jobs=int(args.n_jobs),
                    budget=int(args.budget),
                    n_lookahead_reps=int(args.n_lookahead_reps),
                    n_loadfactor_reps=int(args.n_loadfactor_reps),
                    n_postreps=int(args.n_postreps),
                    output_dir=exp_dir,
                )
                problem_solvers.append(ps)
                used_specs_by_label.setdefault(solver_spec.solver_label, solver_spec)
                row["experiment"] = exp_name
                row["problem_variant"] = problem_variant
                rows.append(row)

        exp_df = pd.DataFrame(rows)

        # CSV summary
        csv_dir = exp_dir / "CSV_Files"
        csv_dir.mkdir(parents=True, exist_ok=True)
        exp_csv = csv_dir / f"{exp_name}.csv"
        exp_df.to_csv(exp_csv, index=False)

        # ProblemsSolvers group pickle
        ps_grid: list[list[ProblemSolver]] = []
        for spec in used_specs_by_label.values():
            spec_ps = [
                ps for ps in problem_solvers if ps.solver.name == spec.solver_label
            ]
            ps_grid.append(spec_ps)

        meta = ProblemsSolvers(
            experiments=ps_grid,
            file_name_path=exp_dir / f"{exp_name}_ProblemsSolvers.pickle",
            experiment_name=exp_name,
        )
        meta.record_group_experiment_results()
        LOGGER.info("Saved ProblemsSolvers: %s", meta.file_name_path)

        all_frames.append(exp_df)

    if not all_frames:
        raise RuntimeError(
            "No experiments were run. Check --experiments/--solver-labels filters."
        )

    all_csv = run_dir / "all_experiment_rows.csv"
    pd.concat(all_frames, ignore_index=True).to_csv(all_csv, index=False)
    LOGGER.info("Wrote merged CSV: %s", all_csv)


# ---------------------------------------------------------------------------
# Argument parsing and dispatch
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse args."""
    parser = argparse.ArgumentParser(
        description="Run VANRYZIN-2 HPC experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- list-tasks --
    sp_list = subparsers.add_parser("list-tasks", help="Print task count or manifest.")
    _add_shared_args(sp_list)
    sp_list.add_argument(
        "--format",
        choices=["count", "json"],
        default="count",
        help="Output format: 'count' (integer) or 'json' (full manifest).",
    )

    # -- run-solver --
    sp_run = subparsers.add_parser(
        "run-solver", help="Run a single ProblemSolver (Phase 1)."
    )
    _add_shared_args(sp_run)
    sp_run.add_argument(
        "--task-index",
        type=int,
        required=True,
        help="Index into the task manifest (SLURM_ARRAY_TASK_ID).",
    )
    sp_run.add_argument(
        "--n-postreps",
        type=int,
        default=400,
        help="Number of policy post-replications.",
    )

    # -- collect-results --
    sp_collect = subparsers.add_parser(
        "collect-results", help="Collect Phase-1 pickles into CSVs (Phase 2)."
    )
    _add_shared_args(sp_collect)
    sp_collect.add_argument(
        "--allow-partial",
        action="store_true",
        help="Proceed even if some Phase-1 pickles are missing.",
    )

    # -- run-all --
    sp_all = subparsers.add_parser(
        "run-all", help="Run all solvers sequentially (local use)."
    )
    _add_shared_args(sp_all)
    sp_all.add_argument(
        "--n-postreps",
        type=int,
        default=400,
        help="Number of policy post-replications.",
    )

    return parser.parse_args()


def main() -> None:
    """Run main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()

    dispatch = {
        "list-tasks": cmd_list_tasks,
        "run-solver": cmd_run_solver,
        "collect-results": cmd_collect_results,
        "run-all": cmd_run_all,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
