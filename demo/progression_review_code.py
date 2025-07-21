import sys
import os.path as o
import json
import matplotlib.pyplot as plt 

sys.path.append(
	o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), ".."))
)  # type:ignore

# Import the ProblemSolver class and other useful functions
from simopt.experiment_base import (
	ProblemSolver,
	post_normalize,
	plot_progress_curves,
	plot_solvability_cdfs,
)


def run_solver(exp_name, solver_name, problem_name, solver_factors, problem_factors) :
	print(f"Testing solver {solver_name} on problem {problem_name}.")
	problem_factors['budget'] = 1000

	# Specify file path name for storing experiment outputs in .pickle file.
	file_name_path = (
		"experiments/outputs/" + solver_name + "_on_" + problem_name + ".pickle"
	)
	print(f"Results will be stored as {file_name_path}.")
	# solver_factors = {'geometry instance': 'AstroDFGeometry', 'sampling rule': 'AdaptiveSampling', 'model type': 'RandomModelReuse', 'polynomial basis': 'AstroDFBasis'}
	# solver_factors = {'subspace dimension': 1}
	# Initialize an instance of the experiment class.
	myexperiment = ProblemSolver(solver_name, problem_name, solver_rename=exp_name, solver_fixed_factors=solver_factors, problem_fixed_factors=problem_factors,create_pickle=True)#, model_fixed_factors={'mu': 18.0})
	# myexperiment = ProblemSolver(solver_name, problem_name)
	# Run a fixed number of macroreplications of the solver on the problem.
	myexperiment.run(n_macroreps=5)

	print(f"Finished running solver {solver_name} on problem {problem_name}.")

	# print("Post-processing results.")
	# # Run a fixed number of postreplications at all recommended solutions.
	# myexperiment.post_replicate(n_postreps=50)
	# # Find an optimal solution x* for normalization.
	# post_normalize([myexperiment], n_postreps_init_opt=50)

	# Log results.
	myexperiment.log_experiment_results()
	myexperiment.log_experiments_csv()

#Get the facility sizing results for the solvers 
def experiment_1() : 
	run_solver('ASTRODF_FACSIZE','ASTRODF', 'FACSIZE-1',{'crn_across_solns': False}, {})
	run_solver('SPSA_FACSIZE','SPSA', 'FACSIZE-1',{'crn_across_solns': False}, {})
	run_solver('ADAM_FACSIZE','ADAM', 'FACSIZE-1',{'crn_across_solns': False}, {})
	run_solver('OMoRF_FACSIZE','OMoRF', 'FACSIZE-1',{'subspace dimension': 7, 'crn_across_solns': False}, {})

	run_solver('STRONG_FACSIZE','STRONG', 'FACSIZE-1',{'crn_across_solns': True}, {})
	run_solver('STRONG_DYNAMNEWS','STRONG', 'DYNAMNEWS-1',{'crn_across_solns': True}, {})
	run_solver('STRONG_ROSENBROCK','STRONG', 'ROSENBROCK-1',{'crn_across_solns': True}, {})
	run_solver('STRONG_ZAKHAROV','STRONG', 'ZAKHAROV-1',{'crn_across_solns': True}, {})

	run_solver('RNDSRCH_FACSIZE','RNDSRCH', 'FACSIZE-1',{'crn_across_solns': False}, {})
	run_solver('RNDSRCH_DYNAMNEWS','RNDSRCH', 'DYNAMNEWS-1',{'crn_across_solns': False}, {})
	run_solver('RNDSRCH_ROSENBROCK','RNDSRCH', 'ROSENBROCK-1',{'crn_across_solns': False}, {})
	run_solver('RNDSRCH_ZAKHAROV','RNDSRCH', 'ZAKHAROV-1',{'crn_across_solns': False}, {})

	#NOTE: Have to use original sampling rule, as sigma^2 grows too big
	run_solver('ASTROMoF_FACSIZE','ASTROMoRF', 'FACSIZE-1',{'subspace dimension': 7,'crn_across_solns': False, 'original_sampling_rule': True}, {})
	

#Test the new ASTROMoRF implementation on ASTROMoRF
def experiment_2() : 
	# for i in range(1,10) :
	# 	name = 'ASTROMoRF_on_subspace_dim_' + str(i)
	# 	run_solver(name,'ASTROMoRF', 'DYNAMNEWS-1',{'crn_across_solns': False, 'old_implementation': True, 'subspace dimension':i, 'performance_benchmarking_flag': False}, {})
	run_solver('ASTROMoRF_on_subspace_dim_6','ASTROMoRF', 'DYNAMNEWS-1',{'crn_across_solns': False, 'old_implementation': True, 'subspace dimension':2, 'performance_benchmarking_flag': True}, {})
	run_solver('ASTROMORF_refined','ASTROMoRF', 'DYNAMNEWS-1',{'crn_across_solns': False, 'old_implementation': False, 'performance_benchmarking_flag': True}, {})


# #Test VNSP, and both Adaptive Sampling Techniques on ASTROMoRF
# def experiment_2() :
# 	sampling_rules = ['ASTROMoRFSampling', 'OriginalAdaptiveSampling', 'BasicSampling']

# 	for sampling_rule in sampling_rules :
# 		exp_name = 'ASTROMoRF_on_sampling_rule_' + sampling_rule
# 		# run_solver('ASTROMoRF', 'FACSIZE-1',{'subspace dimension': 7,'crn_across_solns': False, 'sampling rule': sampling_rule}, {})
# 		run_solver(exp_name, 'ASTROMoRF', 'ROSENBROCK-1',{'subspace dimension': 5,'crn_across_solns': True, 'sampling rule': sampling_rule}, {})
# 		# run_solver('ASTROMoRF', 'ZAKHAROV-1',{'subspace dimension': 7,'crn_across_solns': False, 'sampling rule': sampling_rule}, {})
# 		# run_solver('ASTROMoRF', 'DYNAMNEWS-1',{'subspace dimension': 7,'crn_across_solns': False, 'sampling rule': sampling_rule}, {})


# #test a selection of polynomial bases on ASTROMoRF 
# def experiment_3() : 
# 	bases = ['MonomialTensorBasis', 
# 	'LegendreTensorBasis',
# 	'ChebyshevTensorBasis',
# 	'LaguerreTensorBasis',
# 	'HermiteTensorBasis',
# 	'MonomialPolynomialBasis',
# 	'NaturalPolynomialBasis',
# 	'LagrangePolynomialBasis',
# 	'NFPPolynomialBasis',
# 	'AstroDFBasis'
#  ]
# 	for poly in bases : 
# 		exp_name = 'ASTROMoRF_on_poly_basis_' + poly
# 		# run_solver('ASTROMoRF', 'FACSIZE-1',{'subspace dimension': 7,'crn_across_solns': True, 'polynomial basis': poly}, {})
# 		run_solver(exp_name, 'ASTROMoRF', 'ROSENBROCK-1',{'subspace dimension': 5,'crn_across_solns': True, 'polynomial basis': poly}, {})
# 		# run_solver('ASTROMoRF', 'ZAKHAROV-1',{'subspace dimension': 7,'crn_across_solns': True, 'polynomial basis': poly}, {})
# 		# run_solver('ASTROMoRF', 'DYNAMNEWS-1',{'subspace dimension': 7,'crn_across_solns': True, 'polynomial basis': poly}, {})
		 


def main() :
	# Specify the names of the solver and problem to test.
	# solver_name = <solver_name>
	# problem_name = <problem_name>
	# These names are strings and should match those input to directory.py.

	# Example with random search solver on continuous newsvendor problem.
	# -----------------------------------------------
	# solver_name = "OMoRF"  # Random search solver
	# problem_name = "RMITD-1"  # Continuous newsvendor problem
	# -----------------------------------------------
	experiment_num = int(sys.argv[1])
	if experiment_num == 1 :
		experiment_1()
	elif experiment_num == 2 :
		experiment_2()
	else :
		print("Please select a valid experiment number.")
		exit(1)


if __name__ == "__main__":
	main()
