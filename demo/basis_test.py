"""
This script is intended to help with debugging problems and solvers.
It create problem-solver groups (using the directory) and runs multiple
macroreplications of each problem-solver pair.
"""

import sys
import os 
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(sys.modules[__name__].__file__), ".."))) # type:ignore

# Import the ProblemsSolvers class and other useful functions
from simopt.experiment_base import ProblemSolver, ProblemsSolvers, plot_solvability_profiles, plot_progress_curves
from simopt.directory import problem_directory


def load_problem_solvers(bases: list[str], problem_names: list[str]) -> list[list[ProblemSolver]] : 
    """
        Write the list of ProblemSolver instances for the basis experiments

    Args:
        bases (list[str]): the names of the different bases to test
        problem_names (list[str]): the problem names to test 

    Returns:
        list[list[ProblemSolver]]: A list of lists of Problem Solvers. Each list in the list of lists share a common solver 
    """

    #create list of Problem objects - share same instances now 
    problem_instances = [problem_directory[a]() for a in problem_names]


    problem_solver_pairs = [] 
    for basis in bases : 
        common_solvers = [] 
        for problem in problem_instances : 
            solver_rename = 'Trust region with ' + basis + ' Basis'
            problem_solver = ProblemSolver(solver_name='TRUSTREGION', problem=problem, 
                                           solver_rename=solver_rename, solver_fixed_factors={'poly_basis': basis})
            common_solvers.append(problem_solver)
        problem_solver_pairs.append(common_solvers)

    return problem_solver_pairs


def sort_problem_solvers(problemsSolvers: ProblemsSolvers) -> list[list[ProblemSolver]] :
    """ Sort the ProblemSolver objects in ProblemSolvers into a list of lists, where each list in the list of lists are ProblemSolver instances
    with common problem 


    Args:
        problemsolvers (ProblemsSolvers): The ProblemSolvers instance to be sorted

    Returns:
        list[list[ProblemSolver]]: Sorted ProblemSolver instances where each list shares a common problem
    """
    problemSolvers = problemsSolvers.experiments
    sorted_problem_solvers_list = [list(item) for item in zip(*problemSolvers)]
    
    return sorted_problem_solvers_list 

def plot_results(problemsolvers: ProblemsSolvers, bases: list[str], problem_names: list[str]) -> None : 
    solver_set_name = 'Trust Region with Bases: '.join([a + ' '  for a in bases[:-1]]) + bases[-1]
    problem_set_name = 'Problems: '.join([a + ', '  for a in problem_names[:-1]]) + problem_names[-1]


    plot_solvability_profiles(experiments=problemsolvers.experiments, plot_type="cdf_solvability", 
                              solver_set_name=solver_set_name, problem_set_name=problem_set_name, legend_loc='lower right')
    
    myexperiments = sort_problem_solvers(problemsolvers)
    
    for myexperiment in myexperiments :
        plot_progress_curves(experiments=myexperiment, plot_type="all", normalize=True)
        plot_progress_curves(experiments=myexperiment, plot_type="mean", normalize=True)
        plot_progress_curves(experiments=myexperiment, plot_type="quantile", beta=0.90, normalize=True)

def main():
    bases = [#'PolynomialTensorBasis', - This is a base class
	'MonomialTensorBasis', 
	'LegendreTensorBasis',
	'ChebyshevTensorBasis',
	'LaguerreTensorBasis',
	'HermiteTensorBasis',
	'ArnoldiPolynomialBasis',
	'NaturalPolynomialBasis',
	'LagrangePolynomialBasis',
	'NFPPolynomialBasis'
    ]

    problem_names = [
        'EXAMPLE-1',
        'SIMPLEFUNC-1',
        'SAN-1',
        'FIXEDSAN-1',
        'NETWORK-1',
        'DYNAMNEWS-1'
    ]


    problem_solver_pairs = load_problem_solvers(bases, problem_names)

    EXPERIMENT_DIR = os.path.join(os.getcwd(), "experiments", time.strftime("%Y-%m-%d_%H-%M-%S"))
    file_name_path = os.path.join(EXPERIMENT_DIR, "outputs")
    file_name_path = os.path.join(file_name_path, 'basis_test.pickle')

    # Initialize an instance of the experiment class.
    mymetaexperiment = ProblemsSolvers(experiments=problem_solver_pairs, file_name_path=file_name_path, experiment_name='basis_test', create_pair_pickles=True)


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
    plot_results(mymetaexperiment, bases, problem_names)

    # Plots will be saved in the folder experiments/plots.
    print("Finished. Plots can be found in experiments/plots folder.")

if __name__ == "__main__":
    main()