"""This script is intended to help with debugging problems and solvers.

It create problem-solver groups (using the directory) and runs multiple
macroreplications of each problem-solver pair.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))


# Import the ProblemsSolvers class and other useful functions
from simopt.directory import problem_directory
from simopt.experiment_base import (
    PlotType,
    ProblemSolver,
    ProblemsSolvers,
    plot_solvability_profiles,
    read_experiment_results,
)


def main() -> None:  # noqa: D103
    # !! When testing a new solver/problem, first go to directory.py.
    # There you should add the import statement and an entry in the respective
    # dictionary (or dictionaries).
    # See directory.py for more details.

    # Specify the names of the solver and problem to test.
    # These names are strings and should match those input to directory.py.
    # Ex:
    solver_names = ["TRUSTREGION", "ASTROMoRF", "OMoRF"]
    problem_names = ["DYNAMNEWS-1"]  # , "GROSS-1", "HOTEL-1", "PARAMESTI-1", "RMITD-1"]
    subspace_dims = [6]  # ,10,15,1,1]

    problem_solvers = []
    for solver in solver_names:
        problem_solvers_common_problem = []
        for idx, problem in enumerate(problem_names):
            problem_instance = problem_directory[problem](fixed_factors={"budget": 450})
            if solver == "ASTROMoRF":
                ps = ProblemSolver(
                    solver_name=solver,
                    problem=problem_instance,
                    solver_fixed_factors={"subspace dimension": subspace_dims[idx]},
                )
            if solver == "OMoRF":
                ps = ProblemSolver(
                    solver_name=solver,
                    problem=problem_instance,
                    solver_fixed_factors={"subspace dimension": subspace_dims[idx]},
                )
            elif solver == "TRUSTREGION":
                ps = ProblemSolver(
                    solver_name=solver,
                    problem=problem_instance,
                    solver_fixed_factors={
                        "geometry instance": "AstroDFGeometry",
                        "sampling rule": "AdaptiveSampling",
                        "model type": "RandomModelReuse",
                        "polynomial basis": "AstroDFBasis",
                    },
                )
            else:
                ps = ProblemSolver(solver_name=solver, problem=problem_instance)
            problem_solvers_common_problem.append(ps)

        problem_solvers.append(problem_solvers_common_problem)

    # Initialize an instance of the experiment class.
    # mymetaexperiment = ProblemsSolvers(
    # solver_names=solver_names, problem_names=problem_names, solver_factors=[{'subspace
    # dimension': 1},{'geometry instance': 'AstroDFGeometry', 'sampling rule':
    # 'AdaptiveSampling', 'model type': 'RandomModelReuse', 'polynomial basis':
    # 'AstroDFBasis'}]
    # )
    mymetaexperiment = ProblemsSolvers(experiments=problem_solvers)
    # Write to log file.
    mymetaexperiment.log_group_experiment_results()

    # Run a fixed number of macroreplications of each solver on each problem.
    mymetaexperiment.run(n_macroreps=1)

    print("Post-processing results.")
    # Run a fixed number of postreplications at all recommended solutions.
    mymetaexperiment.post_replicate(n_postreps=20)
    # Find an optimal solution x* for normalization.
    mymetaexperiment.post_normalize(n_postreps_init_opt=20)

    print("Plotting results.")
    mymetaexperiment.report_group_statistics(
        solve_tols=[0.3] * mymetaexperiment.n_problems
    )
    # Produce basic plots of the solvers on the problems.
    plot_solvability_profiles(
        experiments=mymetaexperiment.experiments,
        plot_type=PlotType.CDF_SOLVABILITY,
        solve_tol=0.3,
    )

    if mymetaexperiment.file_name_path is not None:
        file_name_path = str(mymetaexperiment.file_name_path)
        file_path = file_name_path.split("outputs")[0]
    else:
        raise ValueError("file_name_path is None")
    myexperiments = []
    for solver_name in solver_names:
        # solver_experiments = []
        for problem_name in problem_names:
            file_name_path = file_path + solver_name + "_on_" + problem_name + ".pickle"
            myexperiment = read_experiment_results(file_name_path)
            if myexperiment is not None:
                myexperiments.append(myexperiment)

    # for myexperiment in myexperiments :
    #     # myexperiment.log_experiment_results()
    #     da = DataAnalysis(myexperiment)
    #     da.show_plots(plot_type='recommended solutions map', no_pts=100)

    # Plots will be saved in the folder experiments/plots.
    print("Finished. Plots can be found in experiments/plots folder.")


if __name__ == "__main__":
    main()
