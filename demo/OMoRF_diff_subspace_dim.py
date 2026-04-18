"""This experiment looks at how the performance of OMoRF varies when using different.

subspace dimensions.
"""

"""
This script is intended to help with debugging problems and solvers.
It create problem-solver groups (using the directory) and runs multiple
macroreplications of each problem-solver pair.
"""

import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import the ProblemsSolvers class and other useful functions
from simopt.directory import problem_directory  # noqa: E402
from simopt.experiment_base import (  # noqa: E402
    PlotType,
    ProblemSolver,
    ProblemsSolvers,
    plot_progress_curves,
    plot_solvability_profiles,
)


def load_problem_solvers(problem_names: list[str]) -> list[list[ProblemSolver]]:
    """Write the list of ProblemSolver instances for the basis experiments.

    Args:
        bases (list[str]): the names of the different bases to test
        problem_names (list[str]): the problem names to test

    Returns:
        list[list[ProblemSolver]]: A list of lists of Problem Solvers. Each list in the
        list of lists share a common solver
    """
    # create list of Problem objects - share same instances now
    problem_instance = next(problem_directory[a]() for a in problem_names)
    max_dim = len(problem_instance.factors["initial_solution"]) + 1
    dimensions = list(range(1, max_dim))

    problem_solver_pairs = []
    for dim in dimensions:
        solver_factors = {"subspace dimension": dim}
        solver_rename = "Trust region with subspace dimension " + str(dim)
        problem_solver = ProblemSolver(
            solver_name="OMoRF",
            problem=problem_instance,
            solver_rename=solver_rename,
            solver_fixed_factors=solver_factors,
        )
        problem_solver_pairs.append([problem_solver])

    return problem_solver_pairs


def sort_problem_solvers(problemsSolvers: ProblemsSolvers) -> list[list[ProblemSolver]]:  # noqa: D417, N803
    """Sort the ProblemSolver objects in ProblemSolvers into a list of lists, where.

    each.

    list in the list of lists are ProblemSolver instances.

    with common problem.


    Args:
        problemsolvers (ProblemsSolvers): The ProblemSolvers instance to be sorted

    Returns:
        list[list[ProblemSolver]]: Sorted ProblemSolver instances where each list shares
        a common problem
    """
    problemSolvers = problemsSolvers.experiments  # noqa: N806
    return [list(item) for item in zip(*problemSolvers, strict=False)]


def plot_results(problemsolvers: ProblemsSolvers) -> None:  # noqa: D103
    solver_set_name = "OMoRF with varying subspace dimensions"
    problem_set_name = "DYNAMNEWS-1"

    plot_solvability_profiles(
        experiments=problemsolvers.experiments,
        plot_type=PlotType.CDF_SOLVABILITY,
        solver_set_name=solver_set_name,
        problem_set_name=problem_set_name,
        legend_loc="upper left",
        plot_title="CDF solvability of dimension experiment",
    )

    myexperiments = sort_problem_solvers(problemsolvers)

    for myexperiment in myexperiments:
        # basis_name = myexperiment[0].solver.factors['polynomial basis'].__class__.name
        problem_name = myexperiment[0].problem.name
        plot_progress_curves(
            experiments=myexperiment,
            plot_type=PlotType.ALL,
            normalize=True,
            plot_title=f"All Progress Curves of Basis {problem_name}",
        )
        plot_progress_curves(
            experiments=myexperiment,
            plot_type=PlotType.MEAN,
            normalize=True,
            plot_title=f"Mean Progress Curves of Basis {problem_name}",
        )
        plot_progress_curves(
            experiments=myexperiment,
            plot_type=PlotType.QUANTILE,
            beta=0.90,
            normalize=True,
            plot_title=f"Quantile Progress Curves of Basis {problem_name}",
        )


def main() -> None:  # noqa: D103
    problem_names = [
        # 'SAN-1', #!!These two take very long. Will multithread this and then run it on
        # iridis
        # 'FIXEDSAN-1',
        # 'NETWORK-1',
        "DYNAMNEWS-1"
    ]

    problem_solver_pairs = load_problem_solvers(problem_names)

    EXPERIMENT_DIR = Path.cwd() / "experiments" / time.strftime("%Y-%m-%d_%H-%M-%S")  # noqa: N806
    file_name_path = EXPERIMENT_DIR / "outputs" / "basis_test.pickle"

    # Initialize an instance of the experiment class.
    mymetaexperiment = ProblemsSolvers(
        experiments=problem_solver_pairs,
        file_name_path=Path(file_name_path),
        experiment_name="basis_test",
        create_pair_pickles=True,
    )

    # Write to log file.
    mymetaexperiment.log_group_experiment_results()

    # Run a fixed number of macroreplications of each solver on each problem.
    mymetaexperiment.run(n_macroreps=20)

    print("Post-processing results.")
    # Run a fixed number of postreplications at all recommended solutions.
    mymetaexperiment.post_replicate(n_postreps=50)
    # Find an optimal solution x* for normalization.
    mymetaexperiment.post_normalize(n_postreps_init_opt=50)

    print("Plotting results.")
    # Produce basic plots of the solvers on the problems.
    plot_results(mymetaexperiment)

    # Plots will be saved in the folder experiments/plots.
    print("Finished. Plots can be found in experiments/plots folder.")


if __name__ == "__main__":
    main()
