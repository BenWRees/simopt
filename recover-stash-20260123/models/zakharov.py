from __future__ import annotations

from typing import Annotated, ClassVar

import numpy as np
from pydantic import BaseModel, Field

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import (
	ConstraintType,
	Model,
	Objective,
	Problem,
	RepResult,
	VariableType,
)

DIM: int = 15
class ZakharovFunctionConfig(BaseModel):
	"""Configuration model for Zakharov Function."""

	x: Annotated[
		tuple[float, ...],
		Field(
			default=(2.0,) * DIM,
			description="point to evaluate",
		),
	]
	variance: Annotated[
		float,
		Field(
			default=0.1,
			description="variance of the noise",
			ge=0,
		),
	]


class ZakharovFunction(Model):
	"""A Zakharov function model with stochastic noise."""

	class_name_abbr: ClassVar[str] = "ZAKHAROV"
	class_name: ClassVar[str] = "Zakharov Function with Stochastic Noise"
	config_class: ClassVar[type[BaseModel]] = ZakharovFunctionConfig
	n_rngs: ClassVar[int] = 1
	n_responses: ClassVar[int] = 1

	def __init__(self, fixed_factors: dict | None = None) -> None:
		super().__init__(fixed_factors)

	def zakharov_function(self, x: np.ndarray) -> float:
		"""Evaluate the Zakharov function at point x."""
		term1 = np.sum(x ** 2)
		term_2_list = [0.5 * (a + 1) * x for a, x in enumerate(x.flatten().tolist())]
		term2 = np.sum(term_2_list)
		return term1 + term2**2 + term2**4

	def before_replicate(self, rng_list: list[MRG32k3a]) -> None:  # noqa: D102
		self.noise_rng = rng_list[0]

	def replicate(self) -> tuple[dict, dict]:
		"""Evaluate Zakharov function f(x) with stochastic noise.

		Returns:
			tuple[dict, dict]: A tuple containing:
				- responses (dict): Performance measures of interest.
					"est_f(x)" = f(x) evaluated with stochastic noise
				- gradients (dict): Gradient estimates (NaN for this model).
		"""
		noise_rng = self.noise_rng
		x = np.array(self.factors["x"])
		fn_eval_at_x = self.zakharov_function(x) + noise_rng.normalvariate(
			sigma=self.factors["variance"]
		)

		responses = {"est_f(x)": fn_eval_at_x}
		gradients = {"est_f(x)": {"x": np.nan, "variance": np.nan}}
		return responses, gradients



class ZakharovFunctionProblemConfig(BaseModel):
	"""Configuration model for Zakharov Function Problem."""

	initial_solution: Annotated[
		tuple[float, ...],
		Field(
			default=(2.0,) * DIM,
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


class ZakharovFunctionProblem(Problem):
	"""Minimize the Zakharov Function with Stochastic Noise."""

	class_name_abbr: ClassVar[str] = "ZAKHAROV-1"
	class_name: ClassVar[str] = "Minimise the Zakharov Function with Stochastic Noise"
	config_class: ClassVar[type[BaseModel]] = ZakharovFunctionProblemConfig
	model_class: ClassVar[type[Model]] = ZakharovFunction
	n_objectives: ClassVar[int] = 1
	n_stochastic_constraints: ClassVar[int] = 0
	minmax: ClassVar[tuple[int, ...]] = (-1,)
	constraint_type: ClassVar[ConstraintType] = ConstraintType.BOX
	variable_type: ClassVar[VariableType] = VariableType.CONTINUOUS
	gradient_available: ClassVar[bool] = True
	optimal_value: ClassVar[float | None] = 0.0
	model_default_factors: ClassVar[dict] = {}
	model_decision_factors: ClassVar[set[str]] = {"x"}
	_dim: int | None = None

	@property
	def optimal_solution(self) -> tuple:  # noqa: D102
		return (0,) * self.dim

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
		return (-5,) * self.dim

	@property
	def upper_bounds(self) -> tuple:  # noqa: D102
		return (10,) * self.dim

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




