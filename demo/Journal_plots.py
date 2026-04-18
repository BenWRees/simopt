"""This python script is for plotting the journal experiments.

1. We plot the comparison of the different solvers on a set of problems
2. We plot the performance of ASTROMoRF under CRN vs non-CRN settings.
3. We plot the function evaluations vs iterations for a set of problems of ASTROMoRF and
ASTRO-DF.
4. We plot the budget history vs iterations for a set of problems of ASTROMoRF and
ASTRO-DF.
5. We plot the terminal function values vs subspace dimension on a problem for ASTROMoRF
6. We plot the performance of ASTROMoRF on different Polynomial Bases.
"""

import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Final

# Take the current directory, find the parent, and add it to the system path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Fix for pickle compatibility: mrg32k3a package structure changed from
# 'mrg32k3a.python' to 'mrg32k3a.mrg32k3a'
# This allows loading pickle files created with the old package structure
from mrg32k3a import mrg32k3a as mrg32k3a_module

sys.modules["mrg32k3a.python"] = mrg32k3a_module

from simopt.experiment_base import (  # noqa: E402
    PlotType,
    ProblemSolver,
    ProblemsSolvers,
    plot_solvability_profiles,
    post_normalize,
    read_experiment_results,
)

FILEDIR: Final[str] = "WSC_experiments"


def get_problem_solvers() -> list[ProblemSolver]:
    """Get the list of problem solvers for the journal experiments.

    Iterates through all folders in FILEDIR and loads all pickle files.

    Returns:
        list[ProblemSolver]: List of problem solvers.
    """
    problem_solvers: list[ProblemSolver] = []
    base_path = Path(__file__).parent.parent.parent / FILEDIR

    if not base_path.exists():
        raise FileNotFoundError(f"Directory {base_path} does not exist.")

    # Walk through all subdirectories
    for root, _dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith((".pickle", ".pkl")):
                file_path = Path(root) / file
                try:
                    problem_solver = read_experiment_results(file_path)
                    problem_solvers.append(problem_solver)
                except Exception as e:
                    print(f"Warning: Could not load {file_path}: {e}")

    return problem_solvers


def run_post_rep(experiment: ProblemsSolvers, num_postreps: int) -> ProblemsSolvers:  # noqa: D417
    """Run post-replication processing on the list on the ProblemsSolvers.

    Args:
        experiments (ProblemsSolvers): List of problem solvers.

    Returns:
        ProblemsSolvers: Processed list of problem solvers.
    """
    experiment.post_replicate(n_postreps=num_postreps)
    assert isinstance(experiment, ProblemSolver)
    post_normalize([experiment], n_postreps_init_opt=num_postreps)
    return experiment


def get_problems_solvers(exp: list[ProblemSolver]) -> ProblemsSolvers:
    """Get the ProblemSolvers instance for the experiments being used for the.

    comparison.

    plots.

    Groups ProblemSolver instances by solver name, creating nested lists where each
    sublist contains all problem-solvers for a given solver.

    Args:
        exp (list[ProblemSolver]): List of problem solvers.

    Returns:
        ProblemsSolvers: The problem solvers for the comparison plots.
    """
    # Group experiments by solver name
    solver_groups: dict[str, list[ProblemSolver]] = defaultdict(list)
    for problem_solver in exp:
        solver_name = problem_solver.solver.name
        solver_groups[solver_name].append(problem_solver)

    # Convert to nested list format expected by ProblemsSolvers
    # Each sublist contains all problem-solvers for a given solver
    experiments: list[list[ProblemSolver]] = list(solver_groups.values())

    return ProblemsSolvers(experiments=experiments)


def main() -> None:  # noqa: D103
    experiments = get_problem_solvers()
    # [print(f'Solver: {exp.solver.name}, Problem: {exp.problem.name}, Macro-reps:
    # {exp.n_macroreps}') for exp in experiments]
    n_postreps = experiments[0].n_macroreps
    comparison_experiments = get_problems_solvers(experiments)
    for exp in comparison_experiments.experiments:
        print(f"Solver Group: {exp[0].solver.name}")
        for ps in exp:
            print(
                f"Solver: {ps.solver.name}, Problem: {ps.problem.name}, Macro-reps: {ps.n_macroreps}"  # noqa: E501
            )
    comparison_experiments = run_post_rep(
        comparison_experiments, num_postreps=n_postreps
    )

    # Further plotting code would go here.
    plot_path = plot_solvability_profiles(
        experiments=comparison_experiments.experiments,
        plot_type=PlotType.CDF_SOLVABILITY,
        solve_tol=0.1,
    )

    # show
    print(f"Plots saved at: {plot_path}")


if __name__ == "__main__":
    main()
