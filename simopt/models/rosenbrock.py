from __future__ import annotations  # noqa: D100

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


class RosenbrockFunctionConfig(BaseModel):
    """Configuration model for Rosenbrock Function."""

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


class RosenbrockFunction(Model):
    """A Rosenbrock function model with stochastic noise."""

    class_name_abbr: ClassVar[str] = "ROSENBROCK"
    class_name: ClassVar[str] = "ROSENBROCK"
    config_class: ClassVar[type[BaseModel]] = RosenbrockFunctionConfig
    n_rngs: ClassVar[int] = 1
    n_responses: ClassVar[int] = 1

    def __init__(self, fixed_factors: dict | None = None) -> None:  # noqa: D107
        super().__init__(fixed_factors)

    def rosenbrock_function(self, x: np.ndarray, noise: MRG32k3a) -> float:
        """Evaluate the Rosenbrock function at point x."""
        terms = []
        for i in range(1, len(x)):
            terms.append(100 * (x[i] - x[i - 1] ** 2) ** 2 + (1 - x[i - 1]) ** 2)
        return np.sum(terms) + noise.normalvariate(sigma=self.factors["variance"])

    def before_replicate(self, rng_list: list[MRG32k3a]) -> None:  # noqa: D102
        self.noise_rng = rng_list[0]

    def replicate(self) -> tuple[dict, dict]:
        """Evaluate Rosenbrock function f(x) with stochastic noise.

        Returns:
                tuple[dict, dict]: A tuple containing:
                        - responses (dict): Performance measures of interest.
                                "est_f(x)" = f(x) evaluated with stochastic noise
                        - gradients (dict): Gradient estimates (NaN for this model).
        """
        noise_rng = self.noise_rng
        x = np.array(self.factors["x"])
        fn_eval_at_x = self.rosenbrock_function(x, noise_rng)

        # analytic gradient w.r.t. x
        grad = np.zeros_like(x, dtype=float)
        n_grad = 1.0
        d = x.size
        if d >= 2:
            for i in range(1, d):
                term_grad = x[i] - n_grad * x[i - 1] ** 2
                # contribution from y_i = 100*term^2 + (1-x[i-1])^2 (deterministic)
                grad[i] += 200.0 * term_grad
                grad[i - 1] += -400.0 * n_grad * term_grad * x[i - 1] + 2.0 * (
                    x[i - 1] - 1.0
                )

        responses = {"est_f(x)": fn_eval_at_x}
        gradients = {"est_f(x)": {"x": tuple(grad.tolist()), "variance": np.nan}}
        return responses, gradients


class RosenbrockFunctionProblemConfig(BaseModel):
    """Configuration model for Rosenbrock Function Problem."""

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


class RosenbrockFunctionProblem(Problem):
    """Minimize the Rosenbrock Function with Stochastic Noise."""

    class_name_abbr: ClassVar[str] = "ROSENBROCK-1"
    class_name: ClassVar[str] = "Min of the Rosenbrock Function with Stochastic Noise"
    config_class: ClassVar[type[BaseModel]] = RosenbrockFunctionProblemConfig
    model_class: ClassVar[type[Model]] = RosenbrockFunction
    n_objectives: ClassVar[int] = 1
    n_stochastic_constraints: ClassVar[int] = 0
    minmax: ClassVar[tuple[int, ...]] = (-1,)
    constraint_type: ClassVar[ConstraintType] = ConstraintType.BOX
    variable_type: ClassVar[VariableType] = VariableType.CONTINUOUS
    gradient_available: ClassVar[bool] = True
    optimal_value: ClassVar[float | None] = 0.0
    optimal_solution: ClassVar[tuple] = (1.0,) * DIM
    model_default_factors: ClassVar[dict] = {}
    model_decision_factors: ClassVar[set[str]] = {"x"}
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
        return (-5,) * self.dim

    @property
    def upper_bounds(self) -> tuple:  # noqa: D102
        return (10,) * self.dim

    def vector_to_factor_dict(self, vector: tuple) -> dict:  # noqa: D102
        return {"x": vector[:]}

    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:  # noqa: D102
        return (factor_dict["x"],)

    def replicate(self, _x: tuple) -> RepResult:  # noqa: D102
        responses, _gradients = self.model.replicate()
        objectives = [Objective(stochastic=responses["est_f(x)"])]
        return RepResult(objectives=objectives)

    def check_deterministic_constraints(self, x: tuple) -> bool:  # noqa: D102
        return super().check_deterministic_constraints(x)

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:  # noqa: D102
        return tuple(
            rand_sol_rng.mvnormalvariate(
                mean_vec=np.zeros(self.dim).tolist(),
                cov=np.eye(self.dim),
                factorized=False,
            )
        )
