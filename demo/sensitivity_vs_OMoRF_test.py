"""
This experiment tests the performance of OMoRF against a trust region algorithm with sensitivity analysis applied to the problem
"""

import sys
import os.path as o
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), ".."))) # type:ignore

# Import the ProblemSolver class and other useful functions
from simopt.experiment_base import ProblemsSolvers, plot_solvability_profiles, ProblemSolver, Problem
# from simopt.base import adaptive_sampling 
# from ..base import adaptive_sampling



def sensitivitity_analysis(problem_name: str) -> Problem :
    """Instantiates a Problem object and applies sensitivity analsysis to reduce the dimensions of the problem

    Args:
        problem_name (str): Name of the problem whos dimensions are being reduced 

    Returns:
        Problem: The reduced-dimension Problem after sensitivity analysis 
    """

    

    problem = Problem(name=problem_name)



def OMoRF_experimen(problem_name: str, solver_fixed_factors: dict[str,dict]) -> ProblemSolver :
    """This function sets up a ProblemSolver object for the solver OMoRF on the 

    Args:
        problem_name (str): The name of the problem to test the Solver on
        solver_fixed_factors (dict[str,dict]): The solver Fixed Factors to be passed to TRUSTREGION

    Returns:
        ProblemSolver: A ProblemSolver experiment where the Solver OMoRF is testing on the problem specified in problem_name
    """
    pass 

def main():
    solver_names = ["TRUSTREGION", "OMoRF"]
    problem_names = ["CNTNEWS-1", "SAN-1"]


    # Initialize an instance of the experiment class.
    mymetaexperiment = ProblemsSolvers(solver_names = solver_names, problem_names = problem_names)

    # Write to log file.
    mymetaexperiment.log_group_experiment_results()

    # Run a fixed number of macroreplications of each solver on each problem.
    mymetaexperiment.run(n_macroreps=3)

    print("Post-processing results.")
    # Run a fixed number of postreplications at all recommended solutions.
    mymetaexperiment.post_replicate(n_postreps=50)
    # Find an optimal solution x* for normalization.
    mymetaexperiment.post_normalize(n_postreps_init_opt=50)

    print("Plotting results.")
    #Plot solvability of OMORF against TRUSTREGION
    plot_solvability_profiles(experiments=mymetaexperiment.experiments, plot_type="diff_cdf_solvability")

    # Plots will be saved in the folder experiments/plots.
    print("Finished. Plots can be found in experiments/plots folder.")

if __name__ == "__main__":
    main()