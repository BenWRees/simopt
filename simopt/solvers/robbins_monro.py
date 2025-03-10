#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 15:22:54 2024

@author: benjaminrees

TODO: add bounds in
"""

from __future__ import annotations
from typing import Callable

from simopt.base import (
    ConstraintType,
    ObjectiveType,
    Solver,
    VariableType,
)

class RobbinsMonro(Solver):
	"""
		The Robbins-Monro algorithm that finds the root of a stochastic function using a predetermined level to the function
	
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
		return ConstraintType.UNCONSTRAINED
	
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
					"default": False
				},
				"stepsize function" : {
					"description": "the gain function for each iteration",
					"datatype": Callable, 
					"default": self.stepsize_fn
				},
				"alpha" : {
					"description": "the value of the function at the root",
					"datatype": float,
					"default": 0.0
				}
		}
	
	@property
	def check_factor_list(self) -> dict[str, Callable] : 
		return {
			"crn_across_solns": self.check_crn_across_solns,
			"stepsize function" : self.check_stepsize_fn,
			"alpha": self.check_alpha
		}

	def __init__(self, name="ROBBINSMONRO", fixed_factors: dict | None =None) -> None:
		"""
			Initialisation of the Robbins-Monro solver. see base.Solver
		
		Parameters
		----------
		name : str, optional
			user-specified name for solver
		fixed_factors : None, optional
			fixed_factors of the solver
		"""
		super().__init__(name, fixed_factors)


	def check_stepsize_fn(self) : 
		return True

	def stepsize_fn(self, n) :
		return 1/(3*n)

	def check_alpha(self) :
		return isinstance(self.factors['alpha'], float)

		
	def solve(self, problem) :
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
		budget = problem.factors["budget"]
		alpha = problem.optimal_value
		stepsize_fn = self.factors['stepsize function']
		# Reset iteration and data storage arrays
		n = 1 # to avoid division by zero        
		new_solution = self.create_new_solution(problem.factors['initial_solution'], problem)
		expended_budget = 0
		recommended_solns = []
		intermediate_budgets = [] 
		intermediate_budgets.append(expended_budget)
		recommended_solns.append(new_solution)
		# best_solution = new_solution

		while expended_budget < budget: 

			new_x = list(new_solution.x)
			problem.simulate(new_solution, 1)
			observation = new_solution.objectives_mean[0]
			
			new_x = new_solution.x + stepsize_fn(n) * (alpha-observation)

			#create a new soluution based on x and append
			new_solution = self.create_new_solution(tuple(new_x), problem)
			expended_budget += 1

			intermediate_budgets.append(expended_budget)
			recommended_solns.append(new_solution)
			problem.simulate(new_solution, 1)
			n += 1
		return recommended_solns, intermediate_budgets
