"""HPC-efficient SLURM-native benchmark runner for SimOpt.

Architecture:
    One SLURM array task = one (problem, solver) pair = one ProblemSolver.
    The task instantiates a ProblemSolver, runs all macroreplications via
    ProblemSolver.run(n_macroreps=N, n_jobs=cpus-per-task), post-replicates
    on the recommended trajectory, and writes the resulting ProblemSolver
    object as a single pickle. A separate `--combine` step loads the 30
    pickles and constructs a ProblemsSolvers object for downstream analysis.

    No checkpoint-resume logic is used; tasks are atomic at the (problem,
    solver) level. Skip-existing prevents duplicate work on resubmission.

Critical Self-Review:

1. Parallelization strategy.
   SLURM array (one task per (problem, solver) pair) is the right grain
   here: 5 problems x 6 solvers = 30 tasks, small enough that each can have
   its own time/memory profile but large enough to saturate a queue.
   Within a task, ProblemSolver.run uses joblib over macroreplications, so
   cpus-per-task scales linearly until N_macroreps = cpus. Multiprocessing
   would be the right choice for an interactive workstation; mrep-level
   SLURM jobs would be the right choice if a single (problem, solver) pair
   exceeds the queue's max wall time.

2. Profiling.
   We profile per-pair (not globally) because runtime variance across pairs
   is large (NELDMD/PARAMESTI vs ASTROMoRF/NETWORK can differ by 100x).
   The per-pair median is multiplied by n_macroreps / cpus and padded with
   a 2x safety factor. This still under-estimates when (a) cold-start
   dominates the small profile sample, (b) postreplication cost grows
   non-linearly with n_postreps, or (c) the solver's runtime depends on
   the random seed (ASTRO-DF in particular). Pad more if in doubt.

3. SLURM parameters.
   Default array size is 30 (one task per pair). cpus-per-task defaults
   to a value that matches expected n_macroreps but is configurable.
   Memory defaults to 4GB; raise for large polynomial bases (ASTROMoRF
   degree=4 on dim=100 builds large basis matrices). Time limit is set to
   the maximum profiled per-pair runtime so the slowest pair fits; faster
   pairs simply finish early.
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

# --- Safety: cap thread/process oversubscription BEFORE numpy/joblib import ---
# Without these caps, a single SLURM task (or local run) can fork all cores
# and each fork can spawn N BLAS threads, causing N x N processes and OOM.
# These are set as defaults; SLURM tasks should override LOKY_MAX_CPU_COUNT
# explicitly to match cpus-per-task.
for _var, _val in (
    ("OMP_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
    ("VECLIB_MAXIMUM_THREADS", "1"),
    ("LOKY_MAX_CPU_COUNT", "1"),
):
    os.environ.setdefault(_var, _val)

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib  # noqa: E402

from simopt.base import Problem, Solver  # noqa: E402
from simopt.experiment import (  # noqa: E402
    ProblemSolver,
    ProblemsSolvers,
    scale_dimension,
)
from simopt.experiment_base import instantiate_problem, instantiate_solver  # noqa: E402
from simopt.solvers.astromorf import PolyBasisType  # noqa: E402

BASE_SEED: int = 20250428
DIMENSION: int = 100

SOLVER_RENAMES: dict[str, str] = {
    "ASTROMORF": "ASTROMoRF",
    "ADAM": "ADAM",
    "NELDMD": "Nelder-Mead",
    "RNDSRCH": "Random Search",
    "STRONG": "STRONG",
    "ASTRODF": "ASTRO-DF",
}
SOLVERS: list[str] = list(SOLVER_RENAMES.keys())

PROBLEMS: list[str] = [
    "DYNAMNEWS-1",
    "SAN-1",
    "NETWORK-1",
    "ROSENBROCK-1",
    "PARAMESTI-1",
]

# NON_RESCALED: set[str] = {"PARAMESTI-1"}
SCALABLE_PROBLEMS: set[str] = {
    "DYNAMNEWS-1",
    "SAN-1",
    "NETWORK-1",
    "ROSENBROCK-1",
}

DIMENSION_FACTORS: dict[str, dict[str, Any]] = {
    "DYNAMNEWS-1": {
        "subspace_dimension": 5,
        "polynomial_degree": 4,
        "subproblem_regularisation": 9.3926741094426e-06,
        "ps_sufficient_reduction": 0.2558336795133228,
        "polynomial basis": PolyBasisType.NATURAL,
        "lambda_min": 6
        },  
    "SAN-1": {
        "subspace_dimension": 5,
        "polynomial_degree": 3,
        "subproblem_regularisation": 0.16825491384130684,
        "ps_sufficient_reduction": 0.0014138895951216063,
        "polynomial basis": PolyBasisType.MONOMIAL,
        "lambda_min": 3
    },  
    "ROSENBROCK-1": {
        "subspace_dimension": 4,
        "polynomial_degree": 1,
        "subproblem_regularisation": 0.0060325065242926345,
        "ps_sufficient_reduction": 0.7442442345346906,
        "polynomial basis": PolyBasisType.MONOMIAL,
        "lambda_min": 3
    },  
    "NETWORK-1": {
        "subspace_dimension": 15,
        "polynomial_degree": 3,
        "subproblem_regularisation": 0.03084158653272337,
        "ps_sufficient_reduction": 0.3290867932917176,
        "polynomial basis": PolyBasisType.HERMITE,
        "lambda_min": 5
    },  
    "PARAMESTI-1": {
        "subspace_dimension": 2,
        "polynomial_degree": 2,
        "subproblem_regularisation": 1.8557102009866698e-05,
        "ps_sufficient_reduction": 0.3489932689036348,
        "polynomial basis": PolyBasisType.CHEBYSHEV,
        "lambda_min": 9
    },  
}

cabs_factors = {
    "SAN-1": {
        "gamma": 0.8979138034117701,
        "c_p": 0.03301456917974605,
        "c_g": 0.16995201607841975,
        "eps_n": 0.03179063701587841,
        "eps_a": 0.4068921820898637,
        "rho_max": 0.8800444663591878,
        "w_safe": 18,
        "eta_safe": 0.014634217178650853,
        "c2_est": 1.7323493147599263,
        "delta_inc_cap": 9
    },
    "NETWORK-1": {
        "gamma": 0.8418390078321105,
        "c_p": 0.07087355092318823,
        "c_g": 0.48689466577922774,
        "eps_n": 0.22630487327885865,
        "eps_a": 0.24519398298639256,
        "rho_max": 0.819061782383329,
        "w_safe": 18,
        "eta_safe": 0.026726934654979863,
        "c2_est": 0.5215160989415044,
        "delta_inc_cap": 9
    },
    "DYNAMNEWS-1": {
        "gamma": 0.8954811132667856,
        "c_p": 0.13839900124832283,
        "c_g": 0.7489076031913247,
        "eps_n": 0.07123425583685324,
        "eps_a": 0.061097120714973176,
        "rho_max": 0.6718703848854164,
        "w_safe": 15,
        "eta_safe": 0.09026154430417822,
        "c2_est": 0.5256829148159191,
        "delta_inc_cap": 6
      },
    "ROSENBROCK-1": {
        "gamma": 0.8025795189265601,
        "c_p": 0.6499103358279078,
        "c_g": 0.17461702543318225,
        "eps_n": 0.34770197082788773,
        "eps_a": 0.27399808406597553,
        "rho_max": 0.790042323293872,
        "w_safe": 28,
        "eta_safe": 0.036547673050839516,
        "c2_est": 1.639360264931268,
        "delta_inc_cap": 6
    },
    "PARAMESTI-1": {
        "gamma": 0.9118442610588682,
        "c_p": 0.29458464194799866,
        "c_g": 0.17155413448959728,
        "eps_n": 0.2783845948216283,
        "eps_a": 0.07194107335227416,
        "rho_max": 0.9094183864616052,
        "w_safe": 18,
        "eta_safe": 0.11550570058098267,
        "c2_est": 0.5658615715072871,
        "delta_inc_cap": 3
     },
}

SOLVER_FIXED_FACTORS: dict[str, dict[str, Any]] = {
    "ASTROMORF": {
        "crn_across_solns": False,
        "eta_1": 0.1,
        "eta_2": 0.8,
        "gamma_1": 2.5,
        "gamma_2": 1.2,
        "gamma_3": 0.5,
        },
    "ADAM": {"crn_across_solns": False,
             "r": 50, 
             "beta1": 0.5,
             "beta2": 0.5996,
             "alpha": 0.46
            },
    "NELDMD": {"crn_across_solns": False,
               "r": 5
               },
    "RNDSRCH": {"crn_across_solns": False,
                "sample_size": 75
                },
    "STRONG": {"crn_across_solns": False,
               "n0": 40, 
               "n_r": 42,
               "eta_0": 0.1,
               "eta_1": 0.8,
               "gamma_1": 0.5, 
               "gamma_2": 1.2, 
               "lambda_2": 1.65
               },
    "ASTRODF": {"crn_across_solns": False,
                "eta_1": 0.1, 
                "eta_2": 0.8,
                "gamma_1": 1.2,
                "gamma_2": 0.5,
                "lambda_min": 10, 
                "ps_sufficient_reduction": 0.16
                },
}


@dataclass(frozen=True)
class Pair:
    """Dataclass for Problem Solver.
    
    Tuple of (problem, solver) that defines a single experiment.
    The task_id property maps each unique pair to a unique integer in [0, K-1], 
    where K is the total number of pairs. This is used for SLURM array indexing.
    """
    
    problem: str
    solver: str

    @property
    def task_id(self) -> int:
        """Get SLURM task ID.
        
        Generates the SLURM array task ID for this (problem, solver) pair based
        on its position in the Cartesian product of PROBLEMS and SOLVERS.

        Returns:
            int: Unique task ID in the range [0, K-1],
            where K is the total number of pairs.
        """ 
        return PROBLEMS.index(self.problem) * len(SOLVERS) + SOLVERS.index(self.solver)


def all_pairs() -> list[Pair]:
    """Get all pairs of ProblemSolvers.

    Returns:
        list[Pair]: Name of every experiment.
    """
    return [Pair(p, s) for p in PROBLEMS for s in SOLVERS]


def pair_pickle_path(output_dir: Path, pair: Pair) -> Path:
    """Get pickle file path.

    Works out the absolute file path for the pickle file.

    Args:
        output_dir (Path): The directory where the pickle file should be saved.
        pair (Pair): The (problem, solver) pair for which to get the pickle path.

    Returns:
        Path: The absolute file path for the pickle file.
    """
    return (
        output_dir
        / pair.problem
        / pair.solver
        / f"{SOLVER_RENAMES[pair.solver]}_on_{pair.problem}.pkl"
    )


def atomic_pickle_dump(
        exp: ProblemSolver | ProblemsSolvers,
        path: Path
        ) -> None:
    """Dumps experiment to Pickle File.

        The dump is atomic: it first writes to a temporary file and then renames it 
        to the target path. This ensures that if the process is interrupted during the
        dump, the target file will not be left in a corrupted state.

    Args:
        exp (ProblemSolver | ProblemsSolvers): Experiment to dump.
        path (Path): The path where the pickle file should be saved.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{time.time_ns()}")
    with tmp.open("wb") as f:
        pickle.dump(exp, f, protocol=pickle.HIGHEST_PROTOCOL)
    Path.replace(tmp, path)


def build_problem(name: str, budget: int) -> Problem:
    """Build Problem.
    
    Build a Problem instance for the given problem name
    and budget, applying scaling if supported.

    Args:
        name (str): The name of the problem to build.
        budget (int): The budget to use for the problem instance, 
        which may be required for scaling.

    Returns:
        Problem: A Problem instance for the given problem name and budget.
    """    
    if name in SCALABLE_PROBLEMS:
        return scale_dimension(name, budget=budget, dimension=DIMENSION)
    return instantiate_problem(name, problem_fixed_factors={"budget": budget})


def build_solver(name: str) -> Solver:
    """Build Solver.
    
    Build a solver instance for the given solver name,
    applying any fixed factors.

    Args:
        name (str): The name of the solver to build.

    Returns:
        Solver: A solver instance for the given solver name.
    """    
    factors = SOLVER_FIXED_FACTORS.get(name, {})

    return instantiate_solver(
        solver_name=name,
        fixed_factors=dict(factors),
        solver_rename=SOLVER_RENAMES[name],
    )


def execute_pair(
    pair: Pair,
    n_macroreps: int,
    n_postreps: int,
    budget: int,
    output_dir: Path,
    n_jobs: int,
    crn: bool,
    skip_existing: bool = True,
) -> ProblemSolver | None:
    """Run one (problem, solver) pair end-to-end.

    n_jobs caps BOTH ProblemSolver.run's joblib pool AND the hardcoded
    Parallel(n_jobs=-1) inside ProblemSolver.post_replicate, via the
    parallel_backend context manager and the LOKY_MAX_CPU_COUNT env var.
    Use n_jobs=1 for safe sequential execution (recommended on laptops).

    Args:
        pair (Pair): The (problem, solver) pair to execute.
        n_macroreps (int): Number of macroreplications to run.
        n_postreps (int): Number of post-replications to run on the 
        recommended trajectory.
        budget (int): The budget to use for the problem instance, which 
        may be required for scaling.
        output_dir (Path): The directory where the results pickle should be saved.
        n_jobs (int): The number of parallel jobs to use for both 
        macroreplications and post-replications.
        skip_existing (bool, optional): If True, skip execution if the output pickle 
        already exists. Defaults to True.
        crn (bool): If True, use common random numbers across solvers.

    Returns:
        ProblemSolver | None: The ProblemSolver instance after execution, 
        or None if skipped due to existing output.
    """
    target = pair_pickle_path(output_dir, pair)
    if skip_existing and target.exists():
        print(f"[skip] {target} exists.")
        return None

    problem = build_problem(pair.problem, budget)
    solver = build_solver(pair.solver)

    # Persist CRN flag via the solver's pydantic config
    solver.config = solver.config.model_copy(update={"crn_across_solns": crn})

    if pair.solver == "ASTROMoRF":
        # Apply ASTROMoRF-specific configuration using model_copy(update=...)
        update_dict = {
            "initial subspace dimension": DIMENSION_FACTORS[pair.problem][
                "subspace_dimension"
            ],
            "polynomial degree": DIMENSION_FACTORS[pair.problem][
                "polynomial_degree"
            ],
            "subproblem_regularisation": DIMENSION_FACTORS[pair.problem][
                "subproblem_regularisation"
            ],
            "ps_sufficient_reduction": DIMENSION_FACTORS[pair.problem][
                "ps_sufficient_reduction"
            ],
            "polynomial basis": DIMENSION_FACTORS[pair.problem][
                "polynomial basis"
            ],
            "lambda_min": DIMENSION_FACTORS[pair.problem]["lambda_min"],
            "CABS factors": cabs_factors.get(pair.problem, solver.config.cabs_factors),
        }
        solver.config = solver.config.model_copy(update=update_dict)
        

    ps = ProblemSolver(
        solver=solver,
        problem=problem,
        file_name_path=target,
        create_pickle=False,
    )
    # err = ps.check_compatibility()
    # if err:
    #     raise RuntimeError(f"Incompatible {pair.solver}/{pair.problem}: {err}")

    # Force a single, capped joblib backend for this entire pair.
    # The context manager overrides Parallel(n_jobs=...) defaults; the env
    # var ensures even hardcoded n_jobs=-1 inside simopt is clamped.
    prev_loky = os.environ.get("LOKY_MAX_CPU_COUNT")
    os.environ["LOKY_MAX_CPU_COUNT"] = str(max(1, n_jobs))
    try:
        with joblib.parallel_backend("loky", n_jobs=max(1, n_jobs)):
            t0 = time.perf_counter()
            ps.run(n_macroreps=n_macroreps, n_jobs=max(1, n_jobs))
            ps.post_replicate(n_postreps=n_postreps, crn_across_budget=False, crn_across_macroreps=False)
            elapsed = time.perf_counter() - t0
    finally:
        if prev_loky is None:
            os.environ.pop("LOKY_MAX_CPU_COUNT", None)
        else:
            os.environ["LOKY_MAX_CPU_COUNT"] = prev_loky

    print(
        f"[done] {pair.solver}/{pair.problem} mreps={n_macroreps} "
        f"postreps={n_postreps} runtime={elapsed:.1f}s"
    )
    atomic_pickle_dump(ps, target)
    return ps


def profile_pair_runtime(
    pair: Pair, budget: int, n_profile_macroreps: int, n_jobs: int
) -> float:
    """Finds runtime of a single ProblemSolver instance.

    Computes the total runtime for executing a single (problem, solver) 
    pair with a specified number of macroreplications and parallel jobs. 
    This is done by building the problem and solver, running the 
    ProblemSolver, and measuring the elapsed time.

    Args:
        pair (Pair): The (problem, solver) pair to profile.
        budget (int): The budget for the problem instance.
        n_profile_macroreps (int): The number of macroreplications 
        to run for profiling.
        n_jobs (int): The number of parallel jobs to use.

    Returns:
        float: The average runtime per macroreplication 
        for the given (problem, solver) pair.
    """    
    problem = build_problem(pair.problem, budget)
    solver = build_solver(pair.solver)
    ps = ProblemSolver(solver=solver, problem=problem, create_pickle=False)
    if ps.check_compatibility():
        return float("nan")
    t0 = time.perf_counter()
    ps.run(n_macroreps=n_profile_macroreps, n_jobs=n_jobs)
    return (time.perf_counter() - t0) / max(1, n_profile_macroreps)


def run_profile(
    n_macroreps: int,
    budget: int,
    max_samples: int,
) -> dict[Pair, float]:
    """Run a runtime profile over all pairs.

    Runs a runtime profile over all (problem, solver) pairs by 
    executing a small number of macroreplications for each pair and measuring 
    the average runtime. The results are returned as a dictionary mapping 
    each pair to its profiled runtime.

    Args:
        n_macroreps (int): The number of macroreplications to run for 
        each pair during profiling
        budget (int): The budget for each problem
        n_jobs (int): The number of jobs to run in parallel
        max_samples (int): The maximum number of samples to run

    Returns:
        dict[Pair, float]: A dictionary mapping each pair 
        to its profiled runtime.
    """    
    pairs = all_pairs()
    if max_samples > 0 and len(pairs) > max_samples:
        rng = random.Random(BASE_SEED)
        pairs = rng.sample(pairs, max_samples)
    n_profile = min(3, n_macroreps)
    runtimes: dict[Pair, float] = {}
    for pair in pairs:
        print(f"[profile] {pair.problem} / {pair.solver} ({n_profile} mreps)")
        try:
            runtimes[pair] = profile_pair_runtime(pair, budget, n_profile, 1)
        except Exception as e:
            print(f"  [profile] FAILED: {type(e).__name__}: {e}")
            runtimes[pair] = float("nan")
    return runtimes


def compute_slurm_params(
    per_pair_runtime_s: dict[Pair, float],
    n_macroreps: int,
    cpus_per_task: int,
    safety_factor: float = 2.0,
    min_time_s: int = 600,
) -> dict[str, Any]:
    """Compute SLURM parameters.

    Computes the SLURM parameters for the benchmark run based 
    on the profiled per-pair runtimes, the number of macroreplications, 
    the number of CPUs per task, and optional safety factors.

    Args:
        per_pair_runtime_s (dict[Pair, float]): A dictionary mapping each 
        (problem, solver) pair to its profiled runtime in seconds.
        n_macroreps (int): The number of macroreplications to run for 
        each pair, which is used to estimate the total runtime per pair 
        based on the profiled runtime.
        cpus_per_task (int): The number of CPUs allocated per SLURM task,
        which is used to estimate the effective runtime per task by 
        accounting for parallelism.
        safety_factor (float, optional): The safety factor to multiply the 
        estimated runtime by. Defaults to 2.0.
        min_time_s (int, optional): The minimum time to allocate for 
        each task in seconds. Defaults to 600.

    Returns:
        dict[str, Any]: _description_
    """    
    valid = [v for v in per_pair_runtime_s.values() if v == v]
    med = median(valid) if valid else 60.0
    max_per_mrep = max(valid) if valid else 60.0
    per_task_s = max_per_mrep * n_macroreps / max(1, cpus_per_task)
    time_limit_s = max(min_time_s, int(per_task_s * safety_factor))
    n_pairs = len(PROBLEMS) * len(SOLVERS)
    return {
        "median_runtime_per_mrep": med,
        "max_runtime_per_mrep": max_per_mrep,
        "per_task_seconds": per_task_s,
        "time_limit_seconds": time_limit_s,
        "n_pairs": n_pairs,
        "K": n_pairs,
    }


def format_slurm_time(seconds: int) -> str:
    """Format time.

    Formats the given time in seconds into a string format accepted 
    by SLURM's --time parameter, which can be in the format 
    "days-hours:minutes:seconds" or "hours:minutes:seconds".

    Args:
        seconds (int): The time in seconds to format.

    Returns:
        str: The formatted time string.
    """    
    seconds = max(60, int(seconds))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}-{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def write_slurm_script(
    path: Path,
    params: dict[str, Any],
    cli_args: dict[str, Any],
) -> None:
    """Write a SLURM batch script.

    Writes a SLURM batch script to the specified path, using the provided parameters
    and command-line arguments to configure the script's behavior.

    Args:
        path (Path): Path to write the SLURM script. 
        params (dict[str, Any]): Dictionary of SLURM parameters, 
        including 'K' (number of tasks) and 'time_limit_seconds'.
        cli_args (dict[str, Any]): Dictionary of command-line arguments.
    """    
    k: int = params["K"]
    time_str = format_slurm_time(params["time_limit_seconds"])
    python_exec = sys.executable
    script_path = Path(__file__).resolve()
    output_dir = cli_args["output_dir"]
    budget = cli_args["budget"]
    n_macroreps = cli_args["n_macroreps"]
    cpus_per_task = cli_args["cpus_per_task"]
    mem_gb = cli_args["mem_gb"]
    crn_flag = " \\\n+    --crn" if cli_args.get("crn") else ""

    home_logs = Path.home() / "slurm_logs"
    home_logs_str = str(home_logs)
    body = f"""#!/bin/bash
#SBATCH --job-name=simopt_bench
#SBATCH --array=0-{k - 1}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem_gb}G
#SBATCH --time={time_str}
#SBATCH --output={home_logs_str}/%x_%A_%a.out
#SBATCH --error={home_logs_str}/%x_%A_%a.err

set -euo pipefail
mkdir -p {home_logs_str}

# Cap oversubscription: one BLAS thread per joblib worker.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export LOKY_MAX_CPU_COUNT="${{SLURM_CPUS_PER_TASK:-{cpus_per_task}}}"

{python_exec} {script_path} \\
    --slurm-mode \\
    --n-macroreps {n_macroreps} \\
    --budget {budget} \\
    --output-dir {output_dir} \\
    --cpus-per-task {cpus_per_task}{crn_flag}
"""
    path.write_text(body)
    path.chmod(0o755)


def combine_pickles(output_dir: Path, combined_path: Path) -> ProblemsSolvers:
    """Combine individual (problem, solver) pickles.

    Combines the individual pickles for each (problem, solver) pair into a single
    ProblemsSolvers object, which is then saved to the specified combined_path.

    Args:
        output_dir (Path): Directory containing the per-pair pickles
          in subdirectories by problem and solver.
        combined_path (Path): Path to write the combined ProblemsSolvers pickle.

    Raises:
        FileNotFoundError: If any expected per-pair pickle is missing.

    Returns:
        ProblemsSolvers: The combined problems and solvers object.
    """
    grid: list[list[ProblemSolver]] = []
    missing: list[Pair] = []
    for solver in SOLVERS:
        row: list[ProblemSolver] = []
        for problem in PROBLEMS:
            p = pair_pickle_path(output_dir, Pair(problem, solver))
            if not p.exists():
                missing.append(Pair(problem, solver))
                continue
            with p.open("rb") as f:
                row.append(pickle.load(f))
        grid.append(row)
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} pickle(s): "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    ps_all = ProblemsSolvers(experiments=grid, file_name_path=combined_path)
    atomic_pickle_dump(ps_all, combined_path)
    print(f"[combine] wrote {combined_path}")
    return ps_all


def main() -> None:
    """Main entry point for the benchmark runner.

    Implements the command-line interface for running the SimOpt benchmark in either
    SLURM mode or local profiling mode, as well as combining results from individual 
    pickles.

    Raises:
        SystemExit: Error parsing arguments or no pairs match filters.
        SystemExit: Invalid SLURM_ARRAY_TASK_ID in SLURM mode.
        SystemExit: Missing pickles in combine mode.
        RuntimeError: Incompatible problem/solver pair.
    """
    parser = argparse.ArgumentParser(description="SLURM-native SimOpt benchmark runner")
    parser.add_argument("--n-macroreps", type=int, default=20)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--budget", type=int, default=10000)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--slurm-mode", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-max-samples", type=int, default=30)
    parser.add_argument(
        "--crn",
        action="store_true",
        help="Use common random numbers across solns.",
    )
    parser.add_argument(
        "--target-task-time", type=float, default=600.0,
        help="Informational only; per-pair time is set from profiling.",
    )
    parser.add_argument("--generate-slurm", type=str, default="")
    parser.add_argument(
        "--cpus-per-task", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
        help="Cap on joblib parallelism inside a task. Default 1 (SAFE). "
        "Raise on HPC nodes only; multi-process + dim=100 problems can OOM.",
    )
    parser.add_argument("--mem-gb", type=int, default=4)
    parser.add_argument(
        "--combine", type=str, default="",
        help="If set, load per-pair pickles and write a " \
        "combined ProblemsSolvers pickle here.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List the pairs that would run and exit; no work is done.",
    )
    parser.add_argument(
        "--only-problem", type=str, default="",
        help="Restrict execution to a single problem " \
        "(e.g. PARAMESTI-1). For safe smoke tests.",
    )
    parser.add_argument(
        "--only-solver", type=str, default="",
        help="Restrict execution to a single solver (e.g. RNDSRCH).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    n_postreps = 20 * args.n_macroreps

    if args.combine:
        combine_pickles(output_dir, Path(args.combine).resolve())
        return

    if args.generate_slurm:
        if args.profile:
            runtimes = run_profile(
                args.n_macroreps, args.budget, args.profile_max_samples
            )
        else:
            runtimes = dict.fromkeys(all_pairs(), 60.0)
            print("[generate-slurm] no profile; using fallback 60s/mrep.")

        params = compute_slurm_params(
            runtimes, args.n_macroreps, args.cpus_per_task
        )
        print("---- SLURM Plan ----")
        print(f"Pairs (array size):\t{params['n_pairs']}")
        print(f"Median runtime/mrep:\t{params['median_runtime_per_mrep']:.2f} s")
        print(f"Max runtime/mrep:\t{params['max_runtime_per_mrep']:.2f} s")
        print(f"Per-task wall (worst):\t{params['per_task_seconds']:.2f} s")
        print(f"SLURM --time:\t{format_slurm_time(params['time_limit_seconds'])}")
        print(f"cpus-per-task:\t{args.cpus_per_task}")
        print(f"mem:\t{args.mem_gb}G")

        write_slurm_script(
            Path(args.generate_slurm).resolve(),
            params,
            cli_args={
                "output_dir": str(output_dir),
                "budget": args.budget,
                "n_macroreps": args.n_macroreps,
                "cpus_per_task": args.cpus_per_task,
                "mem_gb": args.mem_gb,
                "crn": args.crn,
            },
        )
        print(f"Wrote SLURM script: {args.generate_slurm}")
        return

    if args.profile:
        runtimes = run_profile(
            args.n_macroreps, args.budget, args.profile_max_samples
        )
        params = compute_slurm_params(
            runtimes, args.n_macroreps, args.cpus_per_task
        )
        print("---- Profile Result ----")
        for pair, t in sorted(
            runtimes.items(), 
            key=lambda kv: -(kv[1] if kv[1] == kv[1] else 0)
            ):
            print(f"  {pair.solver:12s} on {pair.problem:14s}: {t:.3f} s/mrep")
        print(f"\nSuggested --time: {format_slurm_time(params['time_limit_seconds'])}")
        return

    pairs = all_pairs()
    if args.only_problem:
        pairs = [p for p in pairs if p.problem == args.only_problem]
    if args.only_solver:
        pairs = [p for p in pairs if p.solver == args.only_solver]
    if not pairs:
        raise SystemExit("No pairs match the filters.")

    if args.slurm_mode:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
        if not 0 <= task_id < len(pairs):
            raise SystemExit(
                f"SLURM_ARRAY_TASK_ID {task_id} out of range [0, {len(pairs)})"
            )
        my_pairs = [pairs[task_id]]
        print(f"[slurm] task {task_id}: {my_pairs[0].solver} on {my_pairs[0].problem}")
    else:
        # SAFETY: refuse to silently run all 30 pairs locally. Either filter,
        # or pass --dry-run, or set SIMOPT_ALLOW_FULL_LOCAL=1.
        if (
            len(pairs) > 3
            and not args.dry_run
            and os.environ.get("SIMOPT_ALLOW_FULL_LOCAL") != "1"
        ):
            raise SystemExit(
                f"Refusing to run {len(pairs)} pairs locally. "
                "Use --slurm-mode on HPC, or --only-problem/--only-solver to "
                "narrow, or set SIMOPT_ALLOW_FULL_LOCAL=1 to override. "
                "Recommend running --dry-run first."
            )

    if args.dry_run:
        print(f"[dry-run] {len(pairs)} pair(s) would run "
              f"(n_macroreps={args.n_macroreps}, n_postreps={n_postreps}, "
              f"budget={args.budget}, n_jobs={args.cpus_per_task}):")
        for pair in pairs:
            target = pair_pickle_path(output_dir, pair)
            status = "skip (exists)" if target.exists() else "run"
            print(f"  [{status}] {pair.solver:12s} on {pair.problem:14s} -> {target}")
        return

    n_jobs = max(1, args.cpus_per_task)
    if n_jobs > 1 and not args.slurm_mode:
        print(
            f"[warn] n_jobs={n_jobs} on a local machine is risky "
            "(deepcopies the dim=100 problem n_jobs times). "
            "Consider n_jobs=1 unless you know your machine has the RAM."
        )

    for pair in pairs:
        execute_pair(
            pair,
            n_macroreps=args.n_macroreps,
            n_postreps=n_postreps,
            budget=args.budget,
            output_dir=output_dir,
            n_jobs=n_jobs,
            crn=args.crn
        )


if __name__ == "__main__":
    main()
