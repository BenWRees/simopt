"""Simulation optimization problem for ASTROMORF hyperparameter tuning.

This module defines a SimOpt Problem for optimizing ASTROMORF's hyperparameters
(subspace dimension and polynomial degree) by treating it as a proper integer-valued
simulation optimization problem that can be solved with any SimOpt solver.
"""

from __future__ import annotations

import logging
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


class ASTROMORFHyperparameterModelConfig(BaseModel):
    """Configuration model for ASTROMORF hyperparameter evaluation.

    This model runs ASTROMORF with given hyperparameters (dimension, degree)
    on a target problem and returns the quality metrics as a stochastic response.
    """

    dimension: Annotated[
        int,
        Field(
            default=6,
            description="subspace dimension",
            ge=1,
        ),
    ]
    degree: Annotated[
        int,
        Field(
            default=7,
            description="polynomial degree",
            ge=1,
        ),
    ]
    gamma_1: Annotated[
        float,
        Field(
            default=2.5,
            description="ASTROMORF gamma_1 parameter",
            gt=1,
        ),
    ]
    gamma_2: Annotated[
        float,
        Field(
            default=1.2,
            description="ASTROMORF gamma_2 parameter",
            gt=1,
        ),
    ]
    gamma_3: Annotated[
        float,
        Field(
            default=0.5,
            description="ASTROMORF gamma_3 parameter",
            gt=0,
            lt=1,
        ),
    ]
    subproblem_regularisation: Annotated[
        float,
        Field(
            default=0.5,
            description="Regularisation parameter for the subproblem",
            gt=0,
            le=1,
        ),
    ]
    target_problem: Annotated[
        Any,
        Field(
            default=None,
            description="SimOpt problem to optimize",
        ),
    ]
    consistency_weight: Annotated[
        float,
        Field(
            default=0.2,
            description="weight for consistency (low variance) metric",
            ge=0,
            le=1,
        ),
    ]
    quality_weight: Annotated[
        float,
        Field(
            default=0.8,
            description="weight for quality (low mean) metric",
            ge=0,
            le=1,
        ),
    ]
    solver: Annotated[
        str,
        Field(
            default="ASTROMORF",
            description="Solver to evaluate (ASTROMORF)",
        ),
    ]
    n_macroreps: Annotated[
        int,
        Field(
            default=3,
            description="number of macroreplications per evaluation",
            ge=1,
        ),
    ]

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        if self.target_problem is None:
            raise ValueError("target_problem must be specified")

        if self.consistency_weight + self.quality_weight != 1:
            raise ValueError("consistency_weight and quality_weight must sum to 1")

        # Ensure gamma ordering: gamma_1 >= gamma_2 >= gamma_3
        if self.gamma_3 > self.gamma_2 or self.gamma_2 > self.gamma_1:
            raise ValueError(
                "Invalid gamma ordering: require gamma_1 >= gamma_2 >= gamma_3"
            )

        return self


class ASTROMORFHyperparameterModel(Model):
    """Model for evaluating ASTROMORF hyperparameter configurations.

    This model runs ASTROMORF with given hyperparameters (dimension, degree)
    on a target problem and returns the quality metrics as a stochastic response.
    """

    class_name_abbr: ClassVar[str] = "ASTROMORF-HYPEROPT"
    class_name: ClassVar[str] = "ASTROMORF Hyperparameter Model"
    config_class: ClassVar[type[BaseModel]] = ASTROMORFHyperparameterModelConfig
    n_rngs: ClassVar[int] = 1
    n_responses: ClassVar[int] = 1

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize the ASTROMORF hyperparameter model.

        Args:
            fixed_factors (dict | None): Fixed factors of the model.
                If None, use default values.
        """
        super().__init__(fixed_factors)

    def replicate(self) -> tuple[dict, dict]:
        """Simulate one replication: run ASTROMORF with given hyperparameters.

        Returns:
            tuple[dict, dict]: A tuple containing:
                - responses (dict): Performance measures of interest, including:
                    - "weighted_score": Weighted score to minimize.
                - gradients (dict): A dictionary of gradient estimates for
                    each response (all NaN for this problem).
        """
        from simopt.experiment_base import ProblemSolver, instantiate_solver

        dimension = int(self.factors["dimension"])
        degree = int(self.factors["degree"])
        gamma_1 = self.factors["gamma_1"]
        gamma_2 = self.factors["gamma_2"]
        gamma_3 = self.factors["gamma_3"]
        subproblem_regularisation = self.factors["subproblem_regularisation"]

        # Sanitize gamma ordering: ASTROMORF expects gamma_1 >= gamma_2 >= gamma_3
        try:
            gammas = [float(gamma_1), float(gamma_2), float(gamma_3)]
            # If any gamma is None, skip reordering
            if None not in gammas:
                sorted_gammas = sorted(gammas, reverse=True)
                if sorted_gammas != gammas:
                    logging.warning(
                        f"Reordering gamma values to satisfy gamma_1 >= gamma_2 >= gamma_3: {gammas} -> {sorted_gammas}"  # noqa: E501
                    )
                    gamma_1, gamma_2, gamma_3 = sorted_gammas
        except Exception:
            # If conversion fails or gammas missing, leave as-is and let downstream checks raise  # noqa: E501
            pass

        # Create solver with these hyperparameters
        if (
            gamma_1 is None
            and gamma_2 is None
            and gamma_3 is None
            and subproblem_regularisation is None
        ):
            solver_factors = {
                "initial subspace dimension": dimension,
                "polynomial degree": degree,
            }
        else:
            solver_factors = {
                "initial subspace dimension": dimension,
                "polynomial degree": degree,
                "gamma_1": gamma_1,
                "gamma_2": gamma_2,
                "gamma_3": gamma_3,
                "subproblem_regularisation": subproblem_regularisation,
            }
        solver = instantiate_solver(
            solver_name=self.factors["solver"],
            fixed_factors=solver_factors,
        )

        # Create ProblemSolver instance
        experiment = ProblemSolver(
            solver=solver, problem=self.factors["target_problem"], create_pickle=False
        )

        print(
            f"Evaluating {experiment.solver.name} on {experiment.problem.name} with dimension={dimension}, degree={degree}, gamma_1={gamma_1}, gamma_2={gamma_2}, gamma_3={gamma_3}, subproblem_regularisation={subproblem_regularisation}."  # noqa: E501
        )

        # Run macroreplications
        experiment.run(n_macroreps=self.factors["n_macroreps"])

        # Post-replicate to get reliable objective estimates
        experiment.post_replicate(
            n_postreps=50, crn_across_budget=True, crn_across_macroreps=False
        )

        # Extract final objective values
        final_objectives = []
        for mrep in range(experiment.n_macroreps):
            if len(experiment.all_est_objectives[mrep]) > 0:
                final_obj = experiment.all_est_objectives[mrep][-1]
                if not np.isnan(final_obj) and not np.isinf(final_obj):
                    final_objectives.append(final_obj)

        if len(final_objectives) == 0:
            # Failed evaluation - return high penalty
            print(
                f"No valid objective values obtained for dimension={dimension}, degree={degree}."  # noqa: E501
            )
            responses = {"weighted_score": 1e6}
            return responses, {}

        # Calculate statistics
        mean_obj = (
            -1 * self.factors["target_problem"].minmax[0] * np.mean(final_objectives)
        )  # turns maximisation into minimization

        # Find standard deviation between macroreplicaitons
        # first group together all estimated objectives across macroreplications
        grouped_objectives = {}
        for mrep in range(experiment.n_macroreps):
            for idx, est_obj in enumerate(experiment.all_est_objectives[mrep]):
                if idx not in grouped_objectives:
                    grouped_objectives[idx] = []
                grouped_objectives[idx].append(est_obj)
        # Now find the std dev across solutions for each macroreplication
        std_devs = []
        for idx, objs in grouped_objectives.items():  # noqa: B007
            if len(objs) > 1:
                std_devs.append(np.std(objs, ddof=1))
            else:
                std_devs.append(0)  # Penalize if only one rep
        std_obj = np.mean(std_devs)

        print(
            f"for dimension {dimension}, degree {degree}, gamma_1 {gamma_1}, gamma_2 {gamma_2}, gamma_3 {gamma_3}, subproblem_regularisation {subproblem_regularisation}: mean_obj = {mean_obj}, std_obj = {std_obj}"  # noqa: E501
        )

        weighted_score = (
            self.factors["quality_weight"] * mean_obj
            + self.factors["consistency_weight"] * std_obj
        )

        responses = {"weighted_score": weighted_score}
        return responses, {}


class ASTROMORFHyperparameterProblemTwoDimConfig(BaseModel):
    """Configuration model for ASTROMORF 2D Hyperparameter Optimization Problem.

    A problem configuration that optimizes ASTROMORF hyperparameters
    (subspace dimension and polynomial degree) on a target problem.
    """

    initial_solution: Annotated[
        tuple[int, ...],
        Field(
            default=(3, 2),
            description="initial hyperparameter configuration (dimension, degree)",
        ),
    ]
    budget: Annotated[
        int,
        Field(
            default=100,
            description="max number of hyperparameter evaluations",
            gt=0,
            json_schema_extra={"isDatafarmable": False},
        ),
    ]
    max_dimension: Annotated[
        int | None,
        Field(
            default=None,
            description="maximum subspace dimension to test",
        ),
    ]
    max_degree: Annotated[
        int,
        Field(
            default=8,
            description="maximum polynomial degree to test",
            ge=1,
        ),
    ]

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        if self.max_dimension is not None and self.max_dimension < 1:
            raise ValueError("max_dimension must be at least 1")
        return self


class ASTROMORFHyperparameterProblemTwoDim(Problem):
    """Simulation optimization problem for finding optimal ASTROMORF hyperparameters.

    Decision variables:
        x[0]: subspace dimension (integer, 1 to problem.dim)
        x[1]: polynomial degree (integer, 1 to max_degree)

    Objective:
        Minimize weighted_score = quality_weight * normalized_mean + consistency_weight
        * normalized_std

    This treats hyperparameter optimization as a proper 2D integer-valued SO problem
    that can be solved with any SimOpt solver (e.g., ASTRO-DF for Bayesian
    optimization).
    """

    class_name_abbr: ClassVar[str] = "ASTROMORF-HYPEROPT-1"
    class_name: ClassVar[str] = "ASTROMORF Hyperparameter Optimization"
    config_class: ClassVar[type[BaseModel]] = ASTROMORFHyperparameterProblemTwoDimConfig
    model_class: ClassVar[type[Model]] = ASTROMORFHyperparameterModel
    n_objectives: ClassVar[int] = 1
    n_stochastic_constraints: ClassVar[int] = 0
    minmax: ClassVar[tuple[int, ...]] = (-1,)
    constraint_type: ClassVar[ConstraintType] = ConstraintType.BOX
    variable_type: ClassVar[VariableType] = VariableType.DISCRETE
    gradient_available: ClassVar[bool] = False
    optimal_value: ClassVar[float | None] = None
    optimal_solution: tuple | None = None
    model_default_factors: ClassVar[dict] = {
        "consistency_weight": 0.2,
        "quality_weight": 0.8,
        "solver": "ASTROMORF",
        "n_macroreps": 3,
    }
    model_decision_factors: ClassVar[set[str]] = {"dimension", "degree"}
    _dim: int | None = None

    @property
    def dim(self) -> int:  # noqa: D102
        if self._dim is not None:
            return self._dim
        return 2

    @dim.setter
    def dim(self, value: int) -> None:
        self._dim = value

    @property
    def lower_bounds(self) -> tuple:  # noqa: D102
        return (1, 1)

    @property
    def upper_bounds(self) -> tuple:  # noqa: D102
        return (self.factors["max_dimension"], self.factors["max_degree"])

    def __init__(
        self,
        fixed_factors: dict | None = None,
        model_fixed_factors: dict | None = None,
    ) -> None:
        """Initialize the ASTROMORF hyperparameter optimization problem.

        Args:
            fixed_factors (dict | None): Fixed factors of the problem.
                If None, use default values.
            model_fixed_factors (dict | None): Fixed factors of the model.
                If None, use default values.
        """
        if fixed_factors is None:
            fixed_factors = {}

        # Get target problem from model_fixed_factors
        if model_fixed_factors is None:
            model_fixed_factors = {}

        target_problem = model_fixed_factors.get("target_problem")
        if target_problem is None:
            raise ValueError("Must specify 'target_problem' in model_fixed_factors")

        # Set max_dimension default based on target problem
        if (
            "max_dimension" not in fixed_factors
            or fixed_factors["max_dimension"] is None
        ):
            fixed_factors["max_dimension"] = target_problem.dim

        # Store target problem for reference
        self.target_problem = target_problem

        # Initialize with base class
        super().__init__(
            fixed_factors=fixed_factors,
            model_fixed_factors=model_fixed_factors,
        )

    def vector_to_factor_dict(self, vector: tuple) -> dict:  # noqa: D102
        dimension, degree = vector
        return {"dimension": int(dimension), "degree": int(degree)}

    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:  # noqa: D102
        return (int(factor_dict["dimension"]), int(factor_dict["degree"]))

    def replicate(self, _x: tuple) -> RepResult:  # noqa: D102
        responses, _ = self.model.replicate()
        objectives = [Objective(stochastic=responses["weighted_score"])]
        return RepResult(objectives=objectives)

    def check_deterministic_constraints(self, x: tuple) -> bool:  # noqa: D102
        return super().check_deterministic_constraints(x)

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:  # noqa: D102
        dimension = rand_sol_rng.randint(1, self.factors["max_dimension"])
        degree = rand_sol_rng.randint(1, self.factors["max_degree"])
        return (int(dimension), int(degree))


class ASTROMORFHyperparameterProblemSixDimConfig(BaseModel):
    """Configuration model for ASTROMORF 6D Hyperparameter Optimization Problem.

    A problem configuration that optimizes ASTROMORF hyperparameters
    (dimension, degree, gamma_1, gamma_2, gamma_3, subproblem_regularisation).
    """

    initial_solution: Annotated[
        tuple[int | float, ...],
        Field(
            default=(3, 2, 2.5, 1.2, 0.5, 0.5),
            description="initial hyperparameter configuration (dimension, degree, gamma_1, gamma_2, gamma_3, subproblem_regularisation)",  # noqa: E501
        ),
    ]
    budget: Annotated[
        int,
        Field(
            default=100,
            description="max number of hyperparameter evaluations",
            gt=0,
            json_schema_extra={"isDatafarmable": False},
        ),
    ]
    max_dimension: Annotated[
        int | None,
        Field(
            default=None,
            description="maximum subspace dimension to test",
        ),
    ]
    max_degree: Annotated[
        int,
        Field(
            default=8,
            description="maximum polynomial degree to test",
            ge=1,
        ),
    ]

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        if self.max_dimension is not None and self.max_dimension < 1:
            raise ValueError("max_dimension must be at least 1")
        return self


class ASTROMORFHyperparameterProblemSixDim(Problem):
    """Simulation optimization problem for finding optimal ASTROMORF hyperparameters.

    Decision variables:
        x[0]: subspace dimension (integer, 1 to problem.dim)
        x[1]: polynomial degree (integer, 1 to max_degree)
        x[2]: gamma_1 (float)
        x[3]: gamma_2 (float)
        x[4]: gamma_3 (float)
        x[5]: subproblem_regularisation (float)

    Objective:
        Minimize weighted_score = quality_weight * normalized_mean + consistency_weight
        * normalized_std

    This treats hyperparameter optimization as a proper 6D mixed-integer-valued SO
    problem
    that can be solved with any SimOpt solver (e.g., ASTRO-DF for Bayesian
    optimization).
    """

    class_name_abbr: ClassVar[str] = "ASTROMORF-HYPEROPT-2"
    class_name: ClassVar[str] = (
        "ASTROMORF Hyperparameter Optimization with Gamma values and regularisation"
    )
    config_class: ClassVar[type[BaseModel]] = ASTROMORFHyperparameterProblemSixDimConfig
    model_class: ClassVar[type[Model]] = ASTROMORFHyperparameterModel
    n_objectives: ClassVar[int] = 1
    n_stochastic_constraints: ClassVar[int] = 0
    minmax: ClassVar[tuple[int, ...]] = (-1,)
    constraint_type: ClassVar[ConstraintType] = ConstraintType.DETERMINISTIC
    variable_type: ClassVar[VariableType] = VariableType.MIXED
    gradient_available: ClassVar[bool] = False
    optimal_value: ClassVar[float | None] = None
    optimal_solution: tuple | None = None
    model_default_factors: ClassVar[dict] = {
        "consistency_weight": 0.2,
        "quality_weight": 0.8,
        "solver": "ASTROMORF",
        "n_macroreps": 3,
        "gamma_1": 2.5,
        "gamma_2": 1.2,
        "gamma_3": 0.5,
        "subproblem_regularisation": 0.5,
    }
    model_decision_factors: ClassVar[set[str]] = {
        "dimension",
        "degree",
        "gamma_1",
        "gamma_2",
        "gamma_3",
        "subproblem_regularisation",
    }
    _dim: int | None = None

    @property
    def dim(self) -> int:  # noqa: D102
        if self._dim is not None:
            return self._dim
        return 6

    @dim.setter
    def dim(self, value: int) -> None:
        self._dim = value

    @property
    def lower_bounds(self) -> tuple:  # noqa: D102
        tol = 1e-8
        gammas_lower = 1 + tol
        return (1, 1, 1e-8, gammas_lower, gammas_lower, tol)

    @property
    def upper_bounds(self) -> tuple:  # noqa: D102
        max_dimension = int(self.factors["max_dimension"])
        max_degree = int(self.factors["max_degree"])
        tol = 1 - 1e-8
        return (max_dimension, max_degree, tol, np.inf, np.inf, np.inf)

    def __init__(
        self,
        fixed_factors: dict | None = None,
        model_fixed_factors: dict | None = None,
    ) -> None:
        """Initialize the ASTROMORF hyperparameter optimization problem.

        Args:
            fixed_factors (dict | None): Fixed factors of the problem.
                If None, use default values.
            model_fixed_factors (dict | None): Fixed factors of the model.
                If None, use default values.
        """
        if fixed_factors is None:
            fixed_factors = {}

        # Get target problem from model_fixed_factors
        if model_fixed_factors is None:
            model_fixed_factors = {}

        target_problem = model_fixed_factors.get("target_problem")
        if target_problem is None:
            raise ValueError("Must specify 'target_problem' in model_fixed_factors")

        # Set max_dimension default based on target problem
        if (
            "max_dimension" not in fixed_factors
            or fixed_factors["max_dimension"] is None
        ):
            fixed_factors["max_dimension"] = int(target_problem.dim)

        # Store target problem for reference
        self.target_problem = target_problem

        # Initialize with base class
        super().__init__(
            fixed_factors=fixed_factors,
            model_fixed_factors=model_fixed_factors,
        )

    def vector_to_factor_dict(self, vector: tuple) -> dict:  # noqa: D102
        dimension, degree, gamma_1, gamma_2, gamma_3, subproblem_regularisation = vector
        return {
            "dimension": int(dimension),
            "degree": int(degree),
            "gamma_1": float(gamma_1),
            "gamma_2": float(gamma_2),
            "gamma_3": float(gamma_3),
            "subproblem_regularisation": float(subproblem_regularisation),
        }

    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:  # noqa: D102
        return (
            int(factor_dict["dimension"]),
            int(factor_dict["degree"]),
            float(factor_dict["gamma_1"]),
            float(factor_dict["gamma_2"]),
            float(factor_dict["gamma_3"]),
            float(factor_dict["subproblem_regularisation"]),
        )

    def replicate(self, _x: tuple) -> RepResult:  # noqa: D102
        responses, _ = self.model.replicate()
        objectives = [Objective(stochastic=responses["weighted_score"])]
        return RepResult(objectives=objectives)

    def check_deterministic_constraints(self, x: tuple) -> bool:  # noqa: D102
        if not super().check_deterministic_constraints(x):
            return False

        # Need gamma_1 >= gamma_2 >= gamma_3
        _, _, gamma_1, gamma_2, gamma_3, _ = x
        return gamma_1 >= gamma_2 and gamma_2 >= gamma_3

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:  # noqa: D102
        lb = self.lower_bounds
        ub = self.upper_bounds
        dimension = rand_sol_rng.randint(int(lb[0]), int(ub[0]))
        degree = rand_sol_rng.randint(int(lb[1]), int(ub[1]))
        gamma_1 = rand_sol_rng.uniform(lb[2], ub[2])
        gamma_2 = rand_sol_rng.uniform(lb[3], ub[3])
        gamma_3 = rand_sol_rng.uniform(lb[4], ub[4])
        subproblem_regularisation = rand_sol_rng.uniform(lb[5], ub[5])
        return (
            int(dimension),
            int(degree),
            float(gamma_1),
            float(gamma_2),
            float(gamma_3),
            float(subproblem_regularisation),
        )
