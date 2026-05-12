"""Optimize ASTROMoRF factors for selected problems (SLURM array friendly).

Searches over initial subspace dimension, polynomial degree, polynomial basis,
lambda_min, ASTROMoRF safety factors, and CABS factors for the specified
problems. This script mirrors optimize_astromorf_cabs.py but can map each
problem to a SLURM array task ID so problems run in parallel on HPC.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from simopt.experiment.dimension_scaling import scale_dimension
from simopt.experiment_base import (
    ProblemSolver,
    instantiate_problem,
    instantiate_solver,
)
from simopt.solvers.astromorf import PolyBasisType

log = logging.getLogger(__name__)

SCALED_PROBLEMS = {"DYNAMNEWS-1", "ROSENBROCK-1", "SAN-1", "NETWORK-1"}

CABS_RANGES: dict[str, tuple[float, float] | tuple[int, int]] = {
    "gamma": (0.85, 0.99),
    "c_p": (0.05, 0.8),
    "c_g": (0.1, 1.2),
    "eps_n": (0.01, 0.5),
    "eps_a": (0.01, 0.5),
    "rho_max": (0.7, 0.98),
    "w_safe": (5, 30),
    "eta_safe": (0.01, 0.2),
    "c2_est": (0.5, 3.0),
    "delta_inc_cap": (1, 10),
}

ASTROMORF_RANGES: dict[str, tuple[float, float]] = {
    "subproblem_regularisation": (0.0, 0.5),
    "ps_sufficient_reduction": (0.0, 1.0),
}

LAMBDA_MIN_RANGE: tuple[int, int] = (3, 30)
DEFAULT_BASIS_TYPES = ",".join(p.value for p in PolyBasisType)


@dataclass(frozen=True)
class AstromorfConfig:
    """Hyperparameter configuration for ASTROMoRF."""

    subspace_dim: int
    degree: int
    polynomial_basis: PolyBasisType
    lambda_min: int
    subproblem_regularisation: float
    ps_sufficient_reduction: float
    cabs_factors: dict[str, float | int]


@dataclass
class EvalResult:
    """Evaluation result for a single hyperparameter configuration."""

    config: AstromorfConfig
    mean_objective: float
    best_objective: float
    mean_score: float
    std_score: float
    objectives: list[float]


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _sample_cabs_factors(rng: np.random.Generator) -> dict[str, float | int]:
    cabs: dict[str, float | int] = {}
    for key, bounds in CABS_RANGES.items():
        lo, hi = bounds
        if isinstance(lo, int) and isinstance(hi, int):
            cabs[key] = int(rng.integers(lo, hi + 1))
        else:
            cabs[key] = float(rng.uniform(float(lo), float(hi)))
    return cabs


def _perturb_cabs_factors(
    base: dict[str, float | int], rng: np.random.Generator, pct: float = 0.2
) -> dict[str, float | int]:
    cabs: dict[str, float | int] = {}
    for key, bounds in CABS_RANGES.items():
        lo, hi = bounds
        base_val = base[key]
        if isinstance(lo, int) and isinstance(hi, int):
            jitter = int(rng.integers(-3, 4))
            cabs[key] = int(_clamp(int(base_val) + jitter, int(lo), int(hi)))
        else:
            span = float(hi) - float(lo)
            jitter = rng.uniform(-pct, pct) * span
            cabs[key] = float(_clamp(float(base_val) + jitter, float(lo), float(hi)))
    return cabs


def _sample_astromorf_factors(rng: np.random.Generator) -> dict[str, float]:
    return {
        key: float(rng.uniform(lo, hi)) for key, (lo, hi) in ASTROMORF_RANGES.items()
    }


def _perturb_astromorf_factors(
    base: dict[str, float], rng: np.random.Generator, pct: float = 0.2
) -> dict[str, float]:
    factors: dict[str, float] = {}
    for key, (lo, hi) in ASTROMORF_RANGES.items():
        span = hi - lo
        jitter = rng.uniform(-pct, pct) * span
        factors[key] = float(_clamp(float(base[key]) + jitter, lo, hi))
    return factors


def _sample_config(
    rng: np.random.Generator,
    min_subspace: int,
    max_subspace: int,
    degrees: list[int],
    basis_types: list[PolyBasisType],
    min_lambda_min: int,
    max_lambda_min: int,
    base: AstromorfConfig | None = None,
) -> AstromorfConfig:
    if base is None:
        subspace_dim = int(rng.integers(min_subspace, max_subspace + 1))
        degree = int(rng.choice(degrees))
        polynomial_basis = rng.choice(basis_types)
        lambda_min = int(rng.integers(min_lambda_min, max_lambda_min + 1))
        astromorf_factors = _sample_astromorf_factors(rng)
        cabs_factors = _sample_cabs_factors(rng)
    else:
        subspace_dim = int(
            _clamp(
                base.subspace_dim + int(rng.integers(-5, 6)),
                min_subspace,
                max_subspace,
            )
        )
        degree = base.degree if rng.random() > 0.3 else int(rng.choice(degrees))
        polynomial_basis = (
            base.polynomial_basis
            if rng.random() > 0.3
            else rng.choice(basis_types)
        )
        lambda_min = int(
            _clamp(
                base.lambda_min + int(rng.integers(-2, 3)),
                min_lambda_min,
                max_lambda_min,
            )
        )
        astromorf_factors = _perturb_astromorf_factors(
            {
                "subproblem_regularisation": base.subproblem_regularisation,
                "ps_sufficient_reduction": base.ps_sufficient_reduction,
            },
            rng,
        )
        cabs_factors = _perturb_cabs_factors(base.cabs_factors, rng)
    return AstromorfConfig(
        subspace_dim=subspace_dim,
        degree=degree,
        polynomial_basis=polynomial_basis,
        lambda_min=lambda_min,
        subproblem_regularisation=astromorf_factors["subproblem_regularisation"],
        ps_sufficient_reduction=astromorf_factors["ps_sufficient_reduction"],
        cabs_factors=cabs_factors,
    )


def _extract_final_objective(experiment: ProblemSolver, mrep: int) -> float | None:
    if getattr(experiment, "all_est_objectives", None) and len(
        experiment.all_est_objectives[mrep]
    ) > 0:
        return float(experiment.all_est_objectives[mrep][-1])
    if getattr(experiment, "objective_curves", None):
        curve = experiment.objective_curves[mrep]
        if hasattr(curve, "y_vals") and len(curve.y_vals) > 0:
            return float(curve.y_vals[-1])
    return None


def _evaluate_config(
    problem_name: str,
    problem_budget: int,
    config: AstromorfConfig,
    n_macroreps: int,
    n_postreps: int,
) -> EvalResult | None:
    if problem_name in SCALED_PROBLEMS:
        problem = scale_dimension(problem_name, budget=problem_budget, dimension=100)
    else:
        problem = instantiate_problem(
            problem_name=problem_name, problem_fixed_factors={"budget": problem_budget}
        )

    solver_factors = {
        "initial subspace dimension": config.subspace_dim,
        "polynomial degree": config.degree,
        "polynomial basis": config.polynomial_basis,
        "lambda_min": config.lambda_min,
        "adaptive subspace dimension": True,
        "subproblem_regularisation": config.subproblem_regularisation,
        "ps_sufficient_reduction": config.ps_sufficient_reduction,
        "CABS factors": config.cabs_factors,
    }

    solver = instantiate_solver("ASTROMORF", fixed_factors=solver_factors)
    experiment = ProblemSolver(
        solver=solver,
        problem=problem,
        create_pickle=False,
    )

    experiment.run(n_macroreps=n_macroreps)
    experiment.post_replicate(
        n_postreps=n_postreps,
        crn_across_budget=True,
        crn_across_macroreps=False,
    )

    objectives: list[float] = []
    for mrep in range(experiment.n_macroreps):
        final_obj = _extract_final_objective(experiment, mrep)
        if final_obj is not None and np.isfinite(final_obj):
            objectives.append(final_obj)

    if not objectives:
        return None

    aligned_scores = [problem.minmax[0] * obj for obj in objectives]
    mean_objective = float(np.mean(objectives))
    best_objective = (
        float(max(objectives))
        if problem.minmax[0] > 0
        else float(min(objectives))
    )

    mean_score = float(np.mean(aligned_scores))
    std_score = float(np.std(aligned_scores, ddof=1)) if len(aligned_scores) > 1 else 0.0

    return EvalResult(
        config=config,
        mean_objective=mean_objective,
        best_objective=best_objective,
        mean_score=mean_score,
        std_score=std_score,
        objectives=objectives,
    )


def _run_search(
    problem_name: str,
    n_configs: int,
    n_macroreps: int,
    n_postreps: int,
    budget: int,
    min_subspace: int,
    max_subspace: int,
    degrees: list[int],
    basis_types: list[PolyBasisType],
    min_lambda_min: int,
    max_lambda_min: int,
    seed: int,
    config_shards: int,
    config_shard_index: int | None,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n_global = max(1, int(n_configs * 0.6))
    n_local = max(0, n_configs - n_global)

    results: list[EvalResult] = []

    log.info("%s: global search (%s configs)", problem_name, n_global)
    for idx in range(n_global):
        config = _sample_config(
            rng,
            min_subspace,
            max_subspace,
            degrees,
            basis_types,
            min_lambda_min,
            max_lambda_min,
        )
        if config_shards > 1 and config_shard_index is not None:
            if idx % config_shards != config_shard_index:
                continue
        log.info("%s: evaluating global config %s/%s", problem_name, idx + 1, n_global)
        try:
            result = _evaluate_config(
                problem_name,
                budget,
                config,
                n_macroreps,
                n_postreps,
            )
        except Exception as exc:
            log.warning("%s: config failed (%s)", problem_name, exc)
            continue
        if result is not None:
            results.append(result)

    results.sort(key=lambda r: r.mean_score, reverse=True)
    elite = results[: min(5, len(results))]

    if not results:
        log.warning("%s: no successful configs; skipping local refinement", problem_name)
        return {
            "problem": problem_name,
            "n_configs": n_configs,
            "n_macroreps": n_macroreps,
            "n_postreps": n_postreps,
            "budget": budget,
            "config_shards": config_shards,
            "config_shard_index": config_shard_index,
            "best": None,
            "results": [],
        }

    if n_local and elite:
        log.info("%s: local refinement (%s configs)", problem_name, n_local)
    for idx in range(n_local):
        base = elite[idx % len(elite)].config
        config = _sample_config(
            rng,
            min_subspace,
            max_subspace,
            degrees,
            basis_types,
            min_lambda_min,
            max_lambda_min,
            base=base,
        )
        if config_shards > 1 and config_shard_index is not None:
            if idx % config_shards != config_shard_index:
                continue
        log.info("%s: evaluating local config %s/%s", problem_name, idx + 1, n_local)
        try:
            result = _evaluate_config(
                problem_name,
                budget,
                config,
                n_macroreps,
                n_postreps,
            )
        except Exception as exc:
            log.warning("%s: config failed (%s)", problem_name, exc)
            continue
        if result is not None:
            results.append(result)

    results.sort(key=lambda r: r.mean_score, reverse=True)
    best = results[0] if results else None

    return {
        "problem": problem_name,
        "n_configs": n_configs,
        "n_macroreps": n_macroreps,
        "n_postreps": n_postreps,
        "budget": budget,
        "config_shards": config_shards,
        "config_shard_index": config_shard_index,
        "best": _result_to_dict(best) if best else None,
        "results": [_result_to_dict(r) for r in results],
    }


def _config_to_dict(config: AstromorfConfig) -> dict[str, Any]:
    data = asdict(config)
    data["polynomial_basis"] = config.polynomial_basis.value
    return data


def _result_to_dict(result: EvalResult) -> dict[str, Any]:
    return {
        "config": _config_to_dict(result.config),
        "mean_objective": result.mean_objective,
        "best_objective": result.best_objective,
        "mean_score": result.mean_score,
        "std_score": result.std_score,
        "objectives": result.objectives,
    }


def _resolve_slurm_problem(
    problems: list[str],
    array_index: int | None,
    array_offset: int,
    config_shards: int,
    config_shard_index: int | None,
) -> tuple[list[str], int | None, int | None]:
    resolved = array_index
    if resolved is None:
        env_value = os.getenv("SLURM_ARRAY_TASK_ID")
        if env_value:
            try:
                resolved = int(env_value)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid SLURM_ARRAY_TASK_ID value: {env_value}"
                ) from exc

    if resolved is None:
        return problems, None, config_shard_index

    adjusted = resolved - array_offset
    if config_shards <= 1 or config_shard_index is not None:
        if adjusted < 0 or adjusted >= len(problems):
            raise SystemExit(
                f"Array index {resolved} (offset {array_offset}) is out of range for "
                f"{len(problems)} problems."
            )
        return [problems[adjusted]], adjusted, config_shard_index

    total_tasks = len(problems) * config_shards
    if adjusted < 0 or adjusted >= total_tasks:
        raise SystemExit(
            f"Array index {resolved} (offset {array_offset}) is out of range for "
            f"{total_tasks} problem shards."
        )

    problem_index = adjusted // config_shards
    shard_index = adjusted % config_shards
    return [problems[problem_index]], problem_index, shard_index


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize ASTROMoRF subspace/degree/basis/lambda/safety/CABS factors "
            "(SLURM array friendly)."
        )
    )
    parser.add_argument(
        "--problems",
        type=str,
        default="NETWORK-1,SAN-1",
        help="Comma-separated problem names.",
    )
    parser.add_argument(
        "--n-configs",
        type=int,
        default=30,
        help="Total configurations per problem.",
    )
    parser.add_argument(
        "--n-macroreps",
        type=int,
        default=5,
        help="Macroreplications per configuration.",
    )
    parser.add_argument(
        "--n-postreps",
        type=int,
        default=50,
        help="Post-replications per configuration.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=10000,
        help="Problem budget per configuration.",
    )
    parser.add_argument(
        "--min-subspace",
        type=int,
        default=2,
        help="Minimum initial subspace dimension.",
    )
    parser.add_argument(
        "--max-subspace",
        type=int,
        default=60,
        help="Maximum initial subspace dimension.",
    )
    parser.add_argument(
        "--degrees",
        type=str,
        default="2,4",
        help="Comma-separated polynomial degrees to try.",
    )
    parser.add_argument(
        "--basis-types",
        type=str,
        default=DEFAULT_BASIS_TYPES,
        help="Comma-separated polynomial basis types to try.",
    )
    parser.add_argument(
        "--lambda-min-min",
        type=int,
        default=LAMBDA_MIN_RANGE[0],
        help="Minimum lambda_min to sample.",
    )
    parser.add_argument(
        "--lambda-min-max",
        type=int,
        default=LAMBDA_MIN_RANGE[1],
        help="Maximum lambda_min to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/astromorf_hyperopt",
        help="Directory for JSON results.",
    )
    parser.add_argument(
        "--array-index",
        type=int,
        default=None,
        help=(
            "Override SLURM_ARRAY_TASK_ID (useful for local testing). "
            "If unset, SLURM_ARRAY_TASK_ID is used when available."
        ),
    )
    parser.add_argument(
        "--array-offset",
        type=int,
        default=0,
        help=(
            "Offset to convert the SLURM array index to zero-based problem index. "
            "Use 1 if your array is 1-based."
        ),
    )
    parser.add_argument(
        "--config-shards",
        type=int,
        default=1,
        help=(
            "Number of shards to split the config search across. "
            "Use >1 to parallelize configs across SLURM array tasks."
        ),
    )
    parser.add_argument(
        "--config-shard-index",
        type=int,
        default=None,
        help=(
            "0-based config shard index to run. If unset, the shard index is "
            "derived from the SLURM array task id when config-shards > 1."
        ),
    )
    return parser.parse_args()


def _parse_basis_list(raw: str) -> list[PolyBasisType]:
    basis_types: list[PolyBasisType] = []
    for token in raw.split(","):
        entry = token.strip()
        if not entry:
            continue
        key = entry.upper().replace("-", "_").replace(" ", "_")
        try:
            basis_types.append(PolyBasisType[key])
            continue
        except KeyError:
            pass
        for basis in PolyBasisType:
            if basis.value.upper() == key:
                basis_types.append(basis)
                break
        else:
            raise ValueError(f"Unknown polynomial basis type: {entry!r}")
    return basis_types


def main() -> None:
    """Main entry point for the optimization script."""
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    problems = [p.strip() for p in args.problems.split(",") if p.strip()]
    degrees = [int(d.strip()) for d in args.degrees.split(",") if d.strip()]
    basis_types = _parse_basis_list(args.basis_types)
    min_lambda_min = args.lambda_min_min
    max_lambda_min = args.lambda_min_max

    if min_lambda_min > max_lambda_min:
        raise ValueError("lambda-min-min cannot exceed lambda-min-max")
    if not basis_types:
        raise ValueError("No polynomial basis types provided")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.config_shards < 1:
        raise ValueError("config-shards must be >= 1")
    if args.config_shard_index is not None and not (
        0 <= args.config_shard_index < args.config_shards
    ):
        raise ValueError("config-shard-index must be in [0, config-shards)")

    selected_problems, array_index, config_shard_index = _resolve_slurm_problem(
        problems,
        args.array_index,
        args.array_offset,
        args.config_shards,
        args.config_shard_index,
    )
    if array_index is not None:
        if args.config_shards > 1 and config_shard_index is not None:
            log.info(
                "SLURM array task %s mapped to problem %s (shard %s/%s)",
                array_index + args.array_offset,
                selected_problems[0],
                config_shard_index + 1,
                args.config_shards,
            )
        else:
            log.info(
                "SLURM array task %s mapped to problem %s",
                array_index + args.array_offset,
                selected_problems[0],
            )

    for problem_name in selected_problems:
        max_subspace = args.max_subspace
        try:
            if problem_name in SCALED_PROBLEMS:
                problem = scale_dimension(problem_name, budget=args.budget, dimension=100)
            else:
                problem = instantiate_problem(
                    problem_name=problem_name,
                    problem_fixed_factors={"budget": args.budget},
                )
            max_subspace = min(max_subspace, max(2, problem.dim - 1))
        except Exception as exc:
            log.warning(
                "Failed to read problem dimension for %s (%s)",
                problem_name,
                exc,
            )

        effective_seed = args.seed + (array_index or 0)
        summary = _run_search(
            problem_name=problem_name,
            n_configs=args.n_configs,
            n_macroreps=args.n_macroreps,
            n_postreps=args.n_postreps,
            budget=args.budget,
            min_subspace=args.min_subspace,
            max_subspace=max_subspace,
            degrees=degrees,
            basis_types=basis_types,
            min_lambda_min=min_lambda_min,
            max_lambda_min=max_lambda_min,
            seed=effective_seed,
            config_shards=args.config_shards,
            config_shard_index=config_shard_index,
        )

        shard_suffix = ""
        if args.config_shards > 1 and config_shard_index is not None:
            shard_suffix = (
                f"_shard{config_shard_index + 1}of{args.config_shards}"
            )

        output_path = output_dir / f"astromorf_cabs_{problem_name}{shard_suffix}.json"
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        log.info("Saved results to %s", output_path)


if __name__ == "__main__":
    main()
