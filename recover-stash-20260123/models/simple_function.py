from __future__ import annotations

from typing import Annotated, Any, ClassVar, Self

import numpy as np
from pydantic import BaseModel, Field, model_validator

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import (
	ConstraintType,
	Model,
	Objective,
	Problem,
	RepResult,
	VariableType,
)


class SimpleFunctionModelConfig(BaseModel):
	"""Configuration model for Simple Function Model."""

	x: Annotated[
		float,
		Field(
			default=2.0,
			description="point to evaluate",
		),
	]
	function: Annotated[
		Any,
		Field(
			default=None,
			description="deterministic function part",
		),
	]

	@model_validator(mode="after")
	def _validate_model(self) -> Self:
		if self.function is None:
			object.__setattr__(self, "function", lambda x: x**2)
		return self


class SimpleFunctionModel(Model):
	"""A model that is a deterministic function evaluated with noise."""

	class_name_abbr: ClassVar[str] = "SIMPLEFUNC"
	class_name: ClassVar[str] = "Deterministic Function + Noise"
	config_class: ClassVar[type[BaseModel]] = SimpleFunctionModelConfig
	n_rngs: ClassVar[int] = 1
	n_responses: ClassVar[int] = 1

	def __init__(self, fixed_factors: dict | None = None) -> None:
		super().__init__(fixed_factors)

	def function_to_eval(self, x: float) -> float:
		"""Default function to evaluate."""
		return x**2

	def before_replicate(self, rng_list: list[MRG32k3a]) -> None:  # noqa: D102
		self.noise_rng = rng_list[0]

	def replicate(self) -> tuple[dict, dict]:
		"""Evaluate a deterministic function f(x) with stochastic noise.

		Returns:
			tuple[dict, dict]: A tuple containing:
				- responses (dict): Performance measures of interest.
					"est_f(x)" = f(x) evaluated with stochastic noise
				- gradients (dict): Gradient estimates.
		"""
		noise_rng = self.noise_rng
		x = np.array(self.factors["x"])
		fn_eval_at_x = self.factors["function"](x) + noise_rng.normalvariate()

		responses = {"est_f(x)": fn_eval_at_x}
		gradients = {"est_f(x)": {"x": tuple(2 * x)}}
		return responses, gradients



class SimpleFunctionProblemConfig(BaseModel):
	"""Configuration model for Simple Function Problem."""

	initial_solution: Annotated[
		tuple[float, ...],
		Field(
			default=(2.0,),
			description="initial solution",
		),
	]
	budget: Annotated[
		int,
		Field(
			default=1000,
			description="max # of replications for a solver to take",
			gt=0,
			json_schema_extra={"isDatafarmable": False},
		),
	]


class SimpleFunctionProblem(Problem):
	"""Simple Function Problem - minimizes a deterministic function with noise."""

	class_name_abbr: ClassVar[str] = "SIMPLEFUNC-1"
	class_name: ClassVar[str] = "Simple Function Problem"
	config_class: ClassVar[type[BaseModel]] = SimpleFunctionProblemConfig
	model_class: ClassVar[type[Model]] = SimpleFunctionModel
	n_objectives: ClassVar[int] = 1
	n_stochastic_constraints: ClassVar[int] = 0
	minmax: ClassVar[tuple[int, ...]] = (-1,)
	constraint_type: ClassVar[ConstraintType] = ConstraintType.UNCONSTRAINED
	variable_type: ClassVar[VariableType] = VariableType.CONTINUOUS
	gradient_available: ClassVar[bool] = True
	optimal_value: ClassVar[float | None] = 0.0
	model_default_factors: ClassVar[dict] = {}
	model_decision_factors: ClassVar[set[str]] = {"x"}

	@property
	def optimal_solution(self) -> tuple:  # noqa: D102
		return (0,) * self.dim

	_dim: int | None = None

	@property
	def dim(self) -> int:  # noqa: D102
		if self._dim is not None:
			return self._dim
		return len(self.factors["initial_solution"])

	@dim.setter
	def dim(self, value: int) -> None:
		self._dim = value

	@property
	def lower_bounds(self) -> tuple:  # noqa: D102
		return (-np.inf,) * self.dim

	@property
	def upper_bounds(self) -> tuple:  # noqa: D102
		return (np.inf,) * self.dim

	def vector_to_factor_dict(self, vector: tuple) -> dict:  # noqa: D102
		return {"x": vector[:]}

	def factor_dict_to_vector(self, factor_dict: dict) -> tuple:  # noqa: D102
		return (factor_dict["x"],)

	def replicate(self, x: tuple) -> RepResult:  # noqa: D102
		responses, gradients = self.model.replicate()
		objectives = [Objective(stochastic=responses["est_f(x)"])]
		return RepResult(objectives=objectives)

	def check_deterministic_constraints(self, x: tuple) -> bool:  # noqa: D102
		return super().check_deterministic_constraints(x)

	def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:  # noqa: D102
		return tuple(
			rand_sol_rng.mvnormalvariate(
				mean_vec=np.zeros(self.dim),
				cov=np.eye(self.dim),
				factorized=False,
			)
		)




