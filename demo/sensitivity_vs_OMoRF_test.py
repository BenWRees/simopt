"""
This experiment tests the performance of OMoRF against a trust region algorithm with sensitivity analysis applied to the problem
"""

import sys
import os.path as o
import os
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), ".."))) # type:ignore

# Import the ProblemSolver class and other useful functions
from simopt.experiment_base import ProblemSolver, read_experiment_results, post_normalize, plot_progress_curves, plot_solvability_cdfs
# from simopt.base import adaptive_sampling 
# from ..base import adaptive_sampling
from simopt.solvers.tr_with_reuse_pts import random_model_reuse, astrodf_geometry
from simopt.solvers.astrodf_ext import adaptive_sampling

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
    # Produce basic plots of the solvers on the problems.
    plot_solvability_profiles(experiments=mymetaexperiment.experiments, plot_type="cdf_solvability")

    # Plots will be saved in the folder experiments/plots.
    print("Finished. Plots can be found in experiments/plots folder.")

if __name__ == "__main__":
    main()