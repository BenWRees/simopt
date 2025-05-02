from __future__ import annotations

from typing import Callable

import numpy as np
from mrg32k3a.mrg32k3a import MRG32k3a

from simopt.base import ConstraintType, Model, Problem, VariableType

DIM: int = 15
class ZakharovFunction(Model):
	"""
	A RosenBrock function model with stochastic noise.

	Attributes
	----------
	name : string
		name of model
	n_rngs : int
		number of random-number generators used to run a simulation replication
	n_responses : int
		number of responses (performance measures)
	factors : dict
		changeable factors of the simulation model
	specifications : dict
		details of each factor (for GUI, data validation, and defaults)
	check_factor_list : dict
		switch case for checking factor simulatability

	Arguments
	---------
	fixed_factors : dict
		fixed_factors of the simulation model

	See also
	--------
	base.Model
	"""

	@property
	def name(self) -> str: 
		return "ZAKHAROV"
	
	@property
	def n_rngs(self) -> int:
		return 1
	
	@property
	def n_responses(self) -> int: 
		return 1
	
	@property 
	def specifications(self) -> dict[str, dict] : 
		return {
			"x": {
				"description": "point to evaluate",
				"datatype": tuple,
				"default": (2.0,) * DIM
			},
			
			"function" : {
				"description": "deterministic function part",
				"datatype": Callable,
				"default": self.function_to_eval
			}, 
			"variance": {
				'description': 'variance of the noise',
				'datatype': float,
				'default': 0.1
			}, 
		}
	
	@property
	def check_factor_list(self) -> dict[str, Callable] : 
		return {
			"x": self.check_x,
			'function': self.check_function_to_eval,
			"variance": self.check_variance
		}
	
	def check_variance(self) -> None:
		if self.factors['variance'] < 0 : 
			raise ValueError("Variance must be non-negative")

	def __init__(self, fixed_factors: dict | None = None) -> None:
		# Set factors of the simulation model.
		super().__init__(fixed_factors)

	def check_function_to_eval(self) -> bool : 
		return True
	
	def function_to_eval(self, x: np.ndarray) -> float :
		term1 = np.sum(x ** 2)
		term2 = np.sum(0.5 * np.arange(1, DIM + 1) * x)
		
		return term1 + term2**2 + term2**4
		
	def check_x(self) -> bool:
		# Assume f(x) can be evaluated at any x in R^d.
		return True

	def check_simulatable_factors(self) -> bool:
		return True

	def replicate(self, rng_list: list[MRG32k3a]) -> tuple[dict, dict]:
		"""
		Evaluate a deterministic function f(x) with stochastic noise.

		Arguments
		---------
		rng_list : list of mrg32k3a.mrg32k3a.MRG32k3a objects
			rngs for model to use when simulating a replication

		Returns
		-------
		responses : dict
			performance measures of interest
			"est_f(x)" = f(x) evaluated with stochastic noise
		"""
		# Designate random number generator for stochastic noise.
		noise_rng = rng_list[0]
		x = np.array(self.factors["x"])
		# fn_eval_at_x =  1/(1+ np.exp(-x))-0.5 + noise_rng.normalvariate()
		fn_eval_at_x = self.factors['function'](x) + noise_rng.normalvariate(sigma=self.factors['variance'])

		# Compose responses and gradients.
		responses = {"est_f(x)": fn_eval_at_x}
		gradients = {
			response_key: {
				factor_key: np.nan for factor_key in self.specifications
			}
			for response_key in responses
		}
		return responses, gradients



class ZakharovFunctionProblem (Problem) :
	"""
	Base class to implement simulation-optimization problems.

	Attributes
	----------
	name : string
		name of problem
	dim : int
		number of decision variables
	n_objectives : int
		number of objectives
	n_stochastic_constraints : int
		number of stochastic constraints
	minmax : tuple of int (+/- 1)
		indicator of maximization (+1) or minimization (-1) for each objective
	constraint_type : string
		description of constraints types:
			"unconstrained", "box", "deterministic", "stochastic"
	variable_type : string
		description of variable types:
			"discrete", "continuous", "mixed"
	lower_bounds : tuple
		lower bound for each decision variable
	upper_bounds : tuple
		upper bound for each decision variable
	gradient_available : bool
		indicates if gradient of objective function is available
	optimal_value : tuple
		optimal objective function value
	optimal_solution : tuple
		optimal solution
	model : Model object
		associated simulation model that generates replications
	model_default_factors : dict
		default values for overriding model-level default factors
	model_fixed_factors : dict
		combination of overriden model-level factors and defaults
	model_decision_factors : set of str
		set of keys for factors that are decision variables
	rng_list : list of mrg32k3a.mrg32k3a.MRG32k3a objects
		list of RNGs used to generate a random initial solution
		or a random problem instance
	factors : dict
		changeable factors of the problem
			initial_solution : tuple
				default initial solution from which solvers start
			budget : int > 0
				max number of replications (fn evals) for a solver to take
	specifications : dict
		details of each factor (for GUI, data validation, and defaults)

	Arguments
	---------
	name : str
		user-specified name for problem
	fixed_factors : dict
		dictionary of user-specified problem factors
	model_fixed factors : dict
		subset of user-specified non-decision factors to pass through to the model

	See also
	--------
	base.Problem
	"""
	@property
	def n_objectives(self) -> int:
		return 1

	@property
	def n_stochastic_constraints(self) -> int:
		return 0

	@property
	def minmax(self) -> tuple[int]:
		return (-1,)

	@property
	def constraint_type(self) -> ConstraintType:
		return ConstraintType.BOX

	@property
	def variable_type(self) -> VariableType:
		return VariableType.CONTINUOUS

	@property
	def gradient_available(self) -> bool:
		return True

	@property
	def optimal_value(self) -> float | None:
		# Change if f is changed
		# TODO: figure out what f is
		return 0.0

	@property
	def optimal_solution(self) -> tuple:
		# Change if f is changed
		# TODO: figure out what f is
		return (0,) * self.dim

	@property
	def model_default_factors(self) -> dict:
		return {}

	@property
	def model_fixed_factors(self) -> dict:
		return {}

	@model_fixed_factors.setter
	def model_fixed_factors(self, value: dict | None) -> None:
		# TODO: figure out if fixed factors should change
		pass

	@property
	def model_decision_factors(self) -> set[str]:
		return {"x"}

	@property
	def specifications(self) -> dict[str, dict]:
		return {
			"initial_solution": {
				"description": "initial solution",
				"datatype": tuple,
				"default": (2.0,) * DIM
			},
			"budget": {
				"description": "max # of replications for a solver to take",
				"datatype": int,
				"default": 1000,
				"isDatafarmable": False
			},
		}

	@property
	def check_factor_list(self) -> dict[str, Callable]:
		return {
			"initial_solution": self.check_initial_solution,
			"budget": self.check_budget,
		}

	@property
	def dim(self) -> int:
		return len(self.factors["initial_solution"])

	@property
	def lower_bounds(self) -> tuple:
		return (-5,) * self.dim

	@property
	def upper_bounds(self) -> tuple:
		return (10,) * self.dim


	def __init__(self, name: str ="ZAKHAROV-1", fixed_factors: dict | None =None, model_fixed_factors: dict | None =None) -> None:
		super().__init__(name=name, fixed_factors=fixed_factors, model_fixed_factors=model_fixed_factors, model=ZakharovFunction)


	def vector_to_factor_dict(self, vector: tuple) -> dict:
		"""
		Convert a vector of variables to a dictionary with factor keys

		Arguments
		---------
		vector : tuple
			vector of values associated with decision variables

		Returns
		-------
		factor_dict : dictionary
			dictionary with factor keys and associated values
		"""
		factor_dict = {
			"x": vector[:]
		}
		return factor_dict

	def factor_dict_to_vector(self, factor_dict: dict) -> tuple:
		"""
		Convert a dictionary with factor keys to a vector
		of variables.

		Arguments
		---------
		factor_dict : dictionary
			dictionary with factor keys and associated values

		Returns
		-------
		vector : tuple
			vector of values associated with decision variables
		"""
		vector = (factor_dict["x"],)
		return vector

	def response_dict_to_objectives(self, response_dict: dict) -> tuple:
		"""
		Convert a dictionary with response keys to a vector
		of objectives.

		Arguments
		---------
		response_dict : dictionary
			dictionary with response keys and associated values

		Returns
		-------
		objectives : tuple
			vector of objectives
		"""
		objectives = (response_dict["est_f(x)"],)
		return objectives

	def response_dict_to_stoch_constraints(self, response_dict: dict) -> tuple:
		"""
		Convert a dictionary with response keys to a vector
		of left-hand sides of stochastic constraints: E[Y] <= 0

		Arguments
		---------
		response_dict : dictionary
			dictionary with response keys and associated values

		Returns
		-------
		stoch_constraints : tuple
			vector of LHSs of stochastic constraint
		"""
		stoch_constraints = ()
		return stoch_constraints

	def deterministic_objectives_and_gradients(self, x: tuple) -> tuple:
		"""
		Compute deterministic components of objectives for a solution `x`.

		Arguments
		---------
		x : tuple
			vector of decision variables

		Returns
		-------
		det_objectives : tuple
			vector of deterministic components of objectives
		det_objectives_gradients : tuple
			vector of gradients of deterministic components of objectives
		"""
		det_objectives = (0,)
		det_objectives_gradients = ((0,) * self.dim,)
		return det_objectives, det_objectives_gradients

	def deterministic_stochastic_constraints_and_gradients(self, x: tuple) -> tuple[tuple, tuple]:
		"""
		Compute deterministic components of stochastic constraints
		for a solution `x`.

		Arguments
		---------
		x : tuple
			vector of decision variables

		Returns
		-------
		det_stoch_constraints : tuple
			vector of deterministic components of stochastic constraints
		det_stoch_constraints_gradients : tuple
			vector of gradients of deterministic components of
			stochastic constraints
		"""
		det_stoch_constraints = ()
		det_stoch_constraints_gradients = ()
		return det_stoch_constraints, det_stoch_constraints_gradients

	def check_deterministic_constraints(self, x: tuple) -> bool :
		"""
		Check if a solution `x` satisfies the problem's deterministic
		constraints.

		Arguments
		---------
		x : tuple
			vector of decision variables

		Returns
		-------
		satisfies : bool
			indicates if solution `x` satisfies the deterministic constraints.
		"""
		# Superclass method will check box constraints.
		# Can add other constraints here.
		return super().check_deterministic_constraints(x)

	def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
		"""
		Generate a random solution for starting or restarting solvers.

		Arguments
		---------
		rand_sol_rng : mrg32k3a.mrg32k3a.MRG32k3a object
			random-number generator used to sample a new random solution

		Returns
		-------
		x : tuple
			vector of decision variables
		"""
		# x = tuple([rand_sol_rng.uniform(-2, 2) for _ in range(self.dim)])
		x = tuple(rand_sol_rng.mvnormalvariate(mean_vec=np.zeros(self.dim), cov=np.eye(self.dim), factorized=False))
		return x




