"""mini-batch SGD 

TODO: add bounds in

"""
from __future__ import annotations
from typing import Callable

from simopt.linear_algebra_base import finite_difference_gradient

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from simopt.base import (
    ConstraintType,
    ObjectiveType,
    Solver,
    VariableType,
)


class SGD(Solver):
	"""
		The mini-batch Stochastic Gradient Descent (SGD) solver
	
	Attributes
	----------
	name : string
		name of solver
	objective_type : string
		description of objective types:
			"single" or "multi"
	constraint_type : string
		description of constraints types:
			"unconstrained", "box", "deterministic", "stochastic"
	variable_type : string
		description of variable types:
			"discrete", "continuous", "mixed"
	gradient_needed : bool
		indicates if gradient of objective function is needed
	factors : dict
		changeable factors (i.e., parameters) of the solver
	specifications : dict
		details of each factor (for GUI, data validation, and defaults)
	rng_list : list of mrg32k3a.mrg32k3a.MRG32k3a objects
		list of RNGs used for the solver's internal purposes
	check_factor_list : dict 
		functions to check each fixed factor is performing
	"""

	@property
	def objective_type(self) -> ObjectiveType: 
		return ObjectiveType.SINGLE
	
	@property
	def constraint_type(self) -> ConstraintType : 
		return ConstraintType.BOX
	
	@property
	def variable_type(self) -> VariableType :
		return VariableType.CONTINUOUS
	
	@property
	def gradient_needed(self) -> bool:
		return False 
	
	@property 
	def specifications(self) -> dict[str, dict] :
		return {
			"crn_across_solns": {
				"description": "use CRN across solutions?",
				"datatype": bool,
				"default": True
			},
			"r": {
				"description": "number of replications taken at each solution",
				"datatype": int,
				"default": 30
			},
			"alpha": {
				"description": "step size",
				"datatype": float,
				"default": 0.9  # Changing the step size matters a lot.
			},
			"gradient clipping check" : {
				"description": "checks if gradient clipping is in use",
				"datatype": bool, 
				"default": True
			},
			"gradient clipping" : {
				"description": "gives a gradient clipping value",
				"datatype": float, 
				"default": 20.0
			}, 
			"SPSA-like gradient": {
				"description": "flag for using an spsa-like gradient",
				"datatype": bool, 
				"default": False #Haven't got this working
			}
		}
	
	@property 
	def check_factor_list(self) -> dict[str, Callable] : 
		return {
			"crn_across_solns": self.check_crn_across_solns,
			"r": self.check_r,
			"alpha": self.check_alpha,
			"gradient clipping check": self.check_gradient_clipping_bool,
			"gradient clipping": self.check_gradient_clipping,
			"SPSA-like gradient": self.check_spsa_gradient
		}


	def __init__(self, name="SGD", fixed_factors: dict | None =None) -> None:
		"""
			Initialisation of ths SGD solver see base.Solver 
		
		Parameters
		----------
		name : str, optional
			user-specified name for solver
		fixed_factors : None, optional
			fixed_factors of the solver
		"""
		super().__init__(name, fixed_factors)

	def check_r(self):
		return self.factors["r"] > 0

	def check_alpha(self):
		return self.factors["alpha"] > 0

	def check_gradient_clipping(self) :
		return True

	def check_gradient_clipping_bool(self) :
		return True	

	def check_spsa_gradient(self) :
		return True

	def solve(self, problem):
		"""
		Run a single macroreplication of a solver on a problem.
		
		Arguments
		---------
		problem : Problem object
			simulation-optimization problem to solve
		
		Returns
		-------
		recommended_solns : list of Solution objects
			list of solutions recommended throughout the budget
		intermediate_budgets : list of ints
			list of intermediate budgets when recommended solutions changes
		
		Deleted Parameters
		------------------
		crn_across_solns : bool
			indicates if CRN are used when simulating different solutions
		"""
		recommended_solns = []
		intermediate_budgets = []
		expended_budget = 0

		# Default values.
		r = self.factors["r"]
		alpha = self.factors["alpha"]
		grad_clip_val = self.factors['gradient clipping']
		grad_clip_check = self.factors['gradient clipping check']
		spsa_check = self.factors['SPSA-like gradient']

		# Upper bound and lower bound.
		lower_bound = np.array(problem.lower_bounds)
		upper_bound = np.array(problem.upper_bounds)

		# Start with the initial solution.
		new_solution = self.create_new_solution(problem.factors["initial_solution"], problem)
		recommended_solns.append(new_solution)
		intermediate_budgets.append(expended_budget)

		# Initialize the timestep.
		t = 1
		while expended_budget < problem.factors["budget"]:
			# Update timestep.
			t = t + 1
			new_x = new_solution.x

			forward = np.isclose(new_x, lower_bound, atol = 10**(-7)).astype(int)
			backward = np.isclose(new_x, upper_bound, atol = 10**(-7)).astype(int)
			# BdsCheck: 1 stands for forward, -1 stands for backward, 0 means central diff.
			BdsCheck = np.subtract(forward, backward)
				

			# Use finite difference to estimate gradient if IPA gradient is not available.
			if spsa_check :
				grad = self.finite_diff_spsa(new_solution, t, BdsCheck, problem)
				expended_budget += 2* r
			else :
				grad = self.finite_diff(new_solution, BdsCheck, problem)
				expended_budget += (2 * problem.dim) * r

			#undergo gradient clipping if necessary
			if grad_clip_check== True and np.linalg.norm(grad) >= grad_clip_val  : 
				grad = grad_clip_val * (grad/np.linalg.norm(grad))

			# Convert new_x from tuple to list.
			new_x = list(new_x)
			# Loop through all the dimensions.
			for i in range(problem.dim):
				# Update new_x and adjust it for box constraints.
				new_x[i] = new_solution.x[i] - alpha*grad[i]

			# Create new solution based on new x
			new_solution = self.create_new_solution(tuple(new_x), problem)
			recommended_solns.append(new_solution)
			intermediate_budgets.append(expended_budget)
		return recommended_solns, intermediate_budgets


	#gradient approximation of 
	def finite_diff_spsa(self, new_solution, k, BdsCheck, problem) : 
		"""
			SPSA-like finite difference approximation of the simulation model
		
		Parameters
		----------
		new_solution : base.Solution
			the current iterations solution 
		k : int 
			the current iteration value
		BdsCheck : np.array([float])
			check location of current solution to boundary to decide on type of finite difference approximation
		problem : base.Problem
			The simulation optimisation problem
		
		Returns
		-------
		np.array([float])
			The averaged gradient approximation from a number of gradient approximations at the current solutions value
		"""
		r = self.factors['r']
		x_k = new_solution.x
		c_k = self.factors['alpha']/(k+1)**0.101
		lower_bound = problem.lower_bounds
		upper_bound = problem.upper_bounds
		gbar = []
		for _ in range(r) :
			delta = self.rng_list[2].choices([-1, 1], [0.5, 0.5], k=problem.dim)

			thetaplus = np.add(x_k, np.multiply(c_k, delta))
			thetaminus = np.subtract(x_k, np.multiply(c_k, delta))

			#check bounds
			if np.any(np.multiply(c_k,delta) < np.array(lower_bound)) :
				new_displacement = np.abs(np.subtract(lower_bound, x_k))
				thetaminus = np.subtract(x_k,new_displacement)

			if np.any(np.multiply(c_k,delta) > np.array(upper_bound)) : 
				new_displacement = np.abs(np.subtract(upper_bound, x_k))
				thetaplus = np.add(x_k, new_displacement)


			thetaplus_sol = self.create_new_solution(tuple(thetaplus), problem)
			thetaminus_sol = self.create_new_solution(tuple(thetaminus), problem)

			problem.simulate(thetaplus_sol, 1)
			problem.simulate(thetaminus_sol, 1)


			finite_diff = np.divide(np.subtract(thetaplus_sol.objectives_mean, thetaminus_sol.objectives_mean),(2*c_k))
			ghat = np.dot(-1, problem.minmax) * np.multiply(finite_diff,delta)
			gbar.append(ghat)
		# ghat = np.dot(-1, problem.minmax) * np.divide((thetaplus_sol.objectives_mean - thetaminus_sol.objectives_mean)/((step_weight_plus + step_weight_minus) * c), delta)
		return np.mean(gbar,axis=0)


	def finite_diff(self, new_solution, BdsCheck, problem):
		"""
			finite difference approximation of the simulation model
		
		Parameters
		----------
		new_solution : base.Solution
			the current iterations solution 
		BdsCheck : np.array([float])
			check location of current solution to boundary to decide on type of finite difference approximation
		problem : base.Problem
			The simulation optimisation problem
		
		Returns
		-------
		np.array([float])
			The averaged gradient approximation from a number of gradient approximations at the current solutions value
		"""
		r = self.factors['r']
		grads = np.zeros((problem.dim,r))
		alpha = self.factors['alpha']

		for batch in range(r) :
			grad =  finite_difference_gradient(self,new_solution,problem, alpha=alpha, BdsCheck=BdsCheck)

			grads[:,batch] = grad

		grad_mean = np.mean(grads,axis=1)

		return grad_mean