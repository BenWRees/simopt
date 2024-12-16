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
import numpy as np 

class KieferWolfowitz(Solver):
	"""
		The Kiefer-Wolfowitz algorithm (FDSA) that finds the optimal value of a stochastic regression function.
	
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
			"stepsize function a" : {
				"description": "the gain function for each iteration",
				"datatype": Callable, 
				"default": self.stepsize_fn_a
			},
			
			"stepsize function c" : {
				"description": "the gain function for each gradient approximation",
				"datatype": Callable, 
				"default": self.stepsize_fn_c              
			},
			"gradient clipping check" : {
				"description": "checks if gradient clipping is in use",
				"datatype": bool, 
				"default": True
			},
			"gradient clipping" : {
				"description": "gives a gradient clipping value",
				"datatype": float, 
				"default": 5.0
			}
		}

	@property 
	def check_factor_list(self) -> dict[str, Callable] : 
		return {
			"crn_across_solns": self.check_crn_across_solns,
			"stepsize function a" : self.check_stepsize_fn_a,
			"stepsize function c": self.check_stepsize_fn_c,
			"gradient clipping check": self.check_gradient_clipping_bool,
			"gradient clipping": self.check_gradient_clipping
		}

	def __init__(self, name="KIEFERWOLFOWITZ", fixed_factors: dict | None =None) -> None:
		"""
			Initialisation of FDSA. see base.Solver
		
		Parameters
		----------
		name : str, optional
			user-specified name for solver
		fixed_factors : None, optional
			fixed_factors of the solver
		"""
		super().__init__(name, fixed_factors)


	def check_stepsize_fn_a(self) : 
		return True
	
	def check_stepsize_fn_c(self) : 
		return True

	def stepsize_fn_a(self, n) :
		return 1/(4 *n)

	def stepsize_fn_c(self, n) :
		return 2/(n**(1/3))

	def check_gradient_clipping(self) :
		return True	

	def check_gradient_clipping_bool(self) :
		return True	
		
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
		grad_clip_val = self.factors['gradient clipping']
		grad_clip_check = self.factors['gradient clipping check']
		# Reset iteration and data storage arrays
		n = 1 # to avoid division by zero        
		new_solution = self.create_new_solution(problem.factors['initial_solution'], problem)
		expended_budget = 0
		recommended_solns = []
		intermediate_budgets = [] 
		intermediate_budgets.append(expended_budget)
		recommended_solns.append(new_solution)

		# Upper bound and lower bound.
		lower_bound = np.array(problem.lower_bounds)
		upper_bound = np.array(problem.upper_bounds)

		while expended_budget < budget: 

			new_x = new_solution.x


			forward = np.isclose(new_x, lower_bound, atol = 10**(-7)).astype(int)
			backward = np.isclose(new_x, upper_bound, atol = 10**(-7)).astype(int)
			# BdsCheck: 1 stands for forward, -1 stands for backward, 0 means central diff.
			# BdsCheck = np.subtract(forward, backward)
			BdsCheck = np.zeros(problem.dim)			
			#calculate central finite differencing of solution 
			diff = self.finite_diff(new_solution, BdsCheck, problem, n)

			#undergo gradient clipping if necessary
			if grad_clip_check== True and np.linalg.norm(diff) >= grad_clip_val  : 
				diff = grad_clip_val * (diff/np.linalg.norm(diff))

			#calculate stepsize
			new_x = list(new_x)
			for i in range(problem.dim) :
				new_x[i] = new_solution.x[i] + problem.minmax[0] * self.factors['stepsize function a'](n) * diff[i]

			
			#create a new soluution based on x and append
			new_solution = self.create_new_solution(tuple(new_x), problem)
			expended_budget += 2 * problem.dim

			intermediate_budgets.append(expended_budget)
			# print('budgets: ',intermediate_budgets)
			recommended_solns.append(new_solution)
			n += 1
		return recommended_solns, intermediate_budgets

	def finite_diff(self, new_solution, BdsCheck, problem, n):
		"""
			Finite difference approximation of the simulation model
		
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
			The gradient approximation of the simulation model at the current iteration's solution value
		"""
		alpha = self.factors["stepsize function c"](n)
		lower_bound = problem.lower_bounds
		upper_bound = problem.upper_bounds
		# grads = np.zeros((problem.dim,r)) #Take r gradient approximations
		problem.simulate(new_solution,1)
		fn = -1 * problem.minmax[0] * new_solution.objectives_mean
		
		new_x = new_solution.x
		FnPlusMinus = np.zeros((problem.dim, 3))
		grad = np.zeros(problem.dim)
		for i in range(problem.dim):
			# Initialization.
			x1 = list(new_x)
			x2 = list(new_x)
			# Forward stepsize.
			steph1 = alpha
			# Backward stepsize.
			steph2 = alpha

			# Check variable bounds.
			if x1[i] + steph1 > upper_bound[i]:
				steph1 = np.abs(upper_bound[i] - x1[i])
			if x2[i] - steph2 < lower_bound[i]:
				steph2 = np.abs(x2[i] - lower_bound[i])

			# Decide stepsize.
			# Central diff.
			if BdsCheck[i] == 0:
				FnPlusMinus[i, 2] = min(steph1, steph2)
				x1[i] = x1[i] + FnPlusMinus[i, 2]
				x2[i] = x2[i] - FnPlusMinus[i, 2]
			# Forward diff.
			elif BdsCheck[i] == 1:
				FnPlusMinus[i, 2] = steph1
				x1[i] = x1[i] + FnPlusMinus[i, 2]
			# Backward diff.
			else:
				FnPlusMinus[i, 2] = steph2
				x2[i] = x2[i] - FnPlusMinus[i, 2]

			fn1, fn2 = 0,0 
			x1_solution = self.create_new_solution(tuple(x1), problem)
			if BdsCheck[i] != -1:
				problem.simulate_up_to([x1_solution], 1)
				fn1 = -1 * problem.minmax[0] * x1_solution.objectives_mean
				# First column is f(x+h,y).
				FnPlusMinus[i, 0] = fn1
			x2_solution = self.create_new_solution(tuple(x2), problem)
			if BdsCheck[i] != 1:
				problem.simulate_up_to([x2_solution], 1)
				fn2 = -1 * problem.minmax[0] * x2_solution.objectives_mean
				# Second column is f(x-h,y).
				FnPlusMinus[i, 1] = fn2

			# Calculate gradient.
			if BdsCheck[i] == 0:
				grad[i] = (fn1 - fn2) / (2 * FnPlusMinus[i, 2])
			elif BdsCheck[i] == 1:
				grad[i] = (fn1 - fn) / FnPlusMinus[i, 2]
			elif BdsCheck[i] == -1:
				grad[i] = (fn - fn2) / FnPlusMinus[i, 2]


		return grad

	