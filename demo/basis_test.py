"""
This script is intended to help with debugging problems and solvers.
It create problem-solver groups (using the directory) and runs multiple
macroreplications of each problem-solver pair.
"""

import sys
import os.path as o
import os
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), ".."))) # type:ignore

# Import the ProblemsSolvers class and other useful functions
from simopt.experiment_base import ProblemsSolvers, plot_solvability_profiles, ProblemSolver

def main():
    # !! When testing a new solver/problem, first go to directory.py.
    # There you should add the import statement and an entry in the respective
    # dictionary (or dictionaries).
    # See directory.py for more details.

    # Specify the names of the solver and problem to test.
    # These names are strings and should match those input to directory.py.
    # Ex:
    solver_names = ["TRUSTREGION"]
    problem_names = ["SIMPLEFUNC-1", 'EXAMPLE-1']

    problem_solver_pairs = []
    for problem in problem_names :
        common_problems = [] 
        common_problems.append(ProblemSolver(solver_name='TRUSTREGION', problem_name=problem, solver_rename='Trust Region with Monimal Basis', solver_fixed_factors={'poly_basis':"simopt.solvers.trust_region_class:monimal_basis"}))
        common_problems.append(ProblemSolver(solver_name='TRUSTREGION', problem_name=problem, solver_rename='Trust Region with Natural Basis', solver_fixed_factors={'poly_basis':"simopt.solvers.trust_region_class:natural_basis"}))
        common_problems.append(ProblemSolver(solver_name='TRUSTREGION', problem_name=problem, solver_rename='Trust Region with Legendre Basis', solver_fixed_factors={'poly_basis':"simopt.solvers.trust_region_class:legendre_basis"}))
        common_problems.append(ProblemSolver(solver_name='TRUSTREGION', problem_name=problem, solver_rename='Trust Region with Newton Fundamental Polynomial Basis', solver_fixed_factors={'poly_basis':"simopt.solvers.trust_region_class:NFP"}))
        # common_problems.append(ProblemSolver(solver_name='TRUSTREGION', problem_name=problem, solver_rename='Trust Region with Chebyshev Interpolation', solver_fixed_factors={'poly_basis':"simopt.solvers.trust_region_class:chebyshev_interpolation"}))
        problem_solver_pairs.append(common_problems)

    problem_solver_pairs = [[a,b] for a,b in zip(problem_solver_pairs[0], problem_solver_pairs[1])]
    print(len(problem_solver_pairs[0]))

    print([a.solver.name for i in problem_solver_pairs for a in i])
    # Initialize an instance of the experiment class.
    mymetaexperiment = ProblemsSolvers(experiments=problem_solver_pairs)

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
    plot_solvability_profiles(experiments=mymetaexperiment.experiments, plot_type="cdf_solvability")

    # Plots will be saved in the folder experiments/plots.
    print("Finished. Plots can be found in experiments/plots folder.")

if __name__ == "__main__":
    main()