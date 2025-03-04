"""
This script is intended to help with debugging problems and solvers.
It create problem-solver groups (using the directory) and runs multiple
macroreplications of each problem-solver pair.
"""

import sys
import os.path as o

sys.path.append(
    o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), ".."))
)  # type:ignore

# Import the ProblemsSolvers class and other useful functions
from simopt.experiment_base import ProblemsSolvers, plot_solvability_profiles, ProblemSolver
from simopt.directory import problem_directory


def build_problem_solvers(solver_names: list[str], problem_names: list[str]) -> list[list[ProblemSolver]] :
    """Write out all the ProblemSolver pairs for the experiments 

    Args:
        solver_names (list[str]): names of the different solvers
        problem_names (list[str]): names of the different problems to test

    Returns:
        list[list[ProblemSolver]]: A list of lists of Problem Solvers. Each list in the list of lists share a common solver 
    """
    problem_instances = [problem_directory[a]() for a in problem_names]

    problem_solver_pairs = [] 

    for solver in solver_names :
        common_solvers = [] 
        solver_factors = None
        if solver == 'TRUSTREGION' : 
            solver_factors = {'geometry instance': 'AstroDFGeometry', 'sampling rule': 'AdaptiveSampling', 'model type': 'RandomModelReuse', 'polynomial basis': 'AstroDFBasis'}
       
        for problem in problem_instances :
            problem_solver = ProblemSolver(solver_name=solver, problem=problem, solver_fixed_factors=solver_factors)
            common_solvers.append(problem_solver)
        
        problem_solver_pairs.append(common_solvers)

    return problem_solver_pairs
        

def main() -> None:
    # !! When testing a new solver/problem, first go to directory.py.
    # There you should add the import statement and an entry in the respective
    # dictionary (or dictionaries).
    # See directory.py for more details.

    # Specify the names of the solver and problem to test.
    # These names are strings and should match those input to directory.py.
    # Ex:
    solver_names = ["TRUSTREGION", "ASTRODF"]
    problem_names = ["SIMPLEFUNC-1", "EXAMPLE-1", "DYNAMNEWS-1", "NETWORK-1"] #? TRUSTREGION seems to not like NETWORK-1


    problem_solver_pairs = build_problem_solvers(solver_names, problem_names)

    mymetaexperiment = ProblemsSolvers(experiments=problem_solver_pairs)

    # Initialize an instance of the experiment class.
    mymetaexperiment = ProblemsSolvers(experiments=problem_solver_pairs)
    # mymetaexperiment = ProblemsSolvers(solver_names=solver_names, problem_names=problem_names, solver_factors=[{'geometry instance': 'AstroDFGeometry', 'sampling rule': 'AdaptiveSampling', 'model type': 'RandomModelReuse'},{}])

    # Write to log file.
    mymetaexperiment.log_group_experiment_results()

    [myexperiment.log_experiment_results() for myexperiments in mymetaexperiment.experiments for myexperiment in myexperiments]

    # Run a fixed number of macroreplications of each solver on each problem.
    mymetaexperiment.run(n_macroreps=5)

    print("Post-processing results.")
    # Run a fixed number of postreplications at all recommended solutions.
    mymetaexperiment.post_replicate(n_postreps=200)
    # Find an optimal solution x* for normalization.
    mymetaexperiment.post_normalize(n_postreps_init_opt=200)

    print("Plotting results.")
    # Produce basic plots of the solvers on the problems.
    plot_solvability_profiles(
        experiments=mymetaexperiment.experiments, plot_type="cdf_solvability", solve_tol=0.1
    )

    # Plots will be saved in the folder experiments/plots.
    print("Finished. Plots can be found in experiments/plots folder.")


if __name__ == "__main__":
    main()
