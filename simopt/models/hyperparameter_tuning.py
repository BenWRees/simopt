"""
    Simulation optimization problem for ASTROMORF hyperparameter tuning.

    This module defines a SimOpt Problem for optimizing ASTROMORF's hyperparameters
    (subspace dimension and polynomial degree) by treating it as a proper integer-valued
    simulation optimization problem that can be solved with any SimOpt solver.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
from mrg32k3a.mrg32k3a import MRG32k3a

from simopt.base import ConstraintType, Model, Problem, VariableType
from simopt.utils import classproperty, override


class ASTROMORFHyperparameterModel(Model):
    """Model for evaluating ASTROMORF hyperparameter configurations.

    This model runs ASTROMORF with given hyperparameters (dimension, degree)
    on a target problem and returns the quality metrics as a stochastic response.
    """

    @classproperty
    @override
    def class_name_abbr(cls) -> str:
        return "ASTROMORF-HYPEROPT"

    @classproperty
    @override
    def class_name(cls) -> str:
        return "ASTROMORF Hyperparameter Model"

    @classproperty
    @override
    def n_rngs(cls) -> int:
        return 1

    @classproperty
    @override
    def n_responses(cls) -> int:
        return 1

    @classproperty
    @override
    def specifications(cls) -> dict[str, dict]:
        return {
            "dimension": {
                "description": "subspace dimension",
                "datatype": int,
                "default": 6,
            },
            "degree": {
                "description": "polynomial degree",
                "datatype": int,
                "default": 7,
            },
            "gamma_1": {
                "description": "ASTROMORF gamma_1 parameter",
                "datatype": float,
                "default": 2.5,
            },
            "gamma_2": {
                "description": "ASTROMORF gamma_2 parameter",
                "datatype": float,
                "default": 1.2,
            },
            "gamma_3": {
                "description": "ASTROMORF gamma_3 parameter",
                "datatype": float,
                "default": 0.5,
            },
            "subproblem_regularisation": {
                "description": "Regularisation parameter for the subproblem",
                "datatype": float,
                "default": 0.5,
            },
            "target_problem": {
                "description": "SimOpt problem to optimize",
                "datatype": Problem,
                "default": None,
            },
            "consistency_weight": {
                "description": "weight for consistency (low variance) metric",
                "datatype": float,
                "default": 0.2,
            },
            "quality_weight": {
                "description": "weight for quality (low mean) metric",
                "datatype": float,
                "default": 0.8,
            },
            "solver": {
                "description": "Solver to evaluate (ASTROMORF)",
                "datatype": str,
                "default": "ASTROMORF",
            },
            "n_macroreps": {
                "description": "number of macroreplications per evaluation",
                "datatype": int,
                "default": 3,
            },
        }

    @property
    @override
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "dimension": self._check_dimension,
            "degree": self._check_degree,
            "gamma_1": self._check_gamma_1,
            "gamma_2": self._check_gamma_2,
            "gamma_3": self._check_gamma_3,
            "subproblem_regularisation": self._check_subproblem_regularisation,
            "target_problem": self._check_target_problem,
            "consistency_weight": self._check_consistency_weight,
            "quality_weight": self._check_quality_weight,
            "solver": self._check_solver_name,
            "n_macroreps": self._check_n_macroreps,
        }

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize the ASTROMORF hyperparameter model.

        Args:
            fixed_factors (dict | None): Fixed factors of the model.
                If None, use default values.
        """
        super().__init__(fixed_factors)

    def _check_dimension(self) -> None:
        if self.factors["dimension"] < 1:
            raise ValueError("dimension must be at least 1")

    def _check_degree(self) -> None:
        if self.factors["degree"] < 1:
            raise ValueError("degree must be at least 1")

    def _check_gamma_1(self) -> None:
        if 1 > self.factors["gamma_1"] :
            raise ValueError("gamma_1 must be greater than 1")
    
    def _check_gamma_2(self) -> None:
        if 1 > self.factors["gamma_2"]:
            raise ValueError("gamma_2 must be greater than 1")

    def _check_gamma_3(self) -> None:
        if self.factors["gamma_3"] <= 0 or self.factors["gamma_3"] >= 1:
            raise ValueError("gamma_3 must be between 0 and 1")
        
    def _check_subproblem_regularisation(self) -> None:
        if self.factors["subproblem_regularisation"] <= 0 or self.factors["subproblem_regularisation"] > 1:
            raise ValueError("subproblem_regularisation must be between 0 and 1")

    def _check_target_problem(self) -> None:
        if self.factors["target_problem"] is None:
            raise ValueError("target_problem must be specified")

    def _check_consistency_weight(self) -> None:
        if not 0 <= self.factors["consistency_weight"] <= 1:
            raise ValueError("consistency_weight must be between 0 and 1")

    def _check_quality_weight(self) -> None:
        if not 0 <= self.factors["quality_weight"] <= 1:
            raise ValueError("quality_weight must be between 0 and 1")

    def _check_solver_name(self) -> None:
        if not isinstance(self.factors["solver"], str):
            raise ValueError("The name of the solver must be a string")

    def _check_n_macroreps(self) -> None:
        if self.factors["n_macroreps"] < 1:
            raise ValueError("n_macroreps must be at least 1")

    @override
    def check_simulatable_factors(self) -> bool:
        if self.factors['consistency_weight'] + self.factors['quality_weight'] != 1:
            raise ValueError("consistency_weight and quality_weight must sum to 1")

        # Ensure gamma ordering: gamma_1 >= gamma_2 >= gamma_3
        if self.factors['gamma_3'] > self.factors['gamma_2'] or self.factors['gamma_2'] > self.factors['gamma_1']:
            raise ValueError("Invalid gamma ordering: require gamma_1 >= gamma_2 >= gamma_3")

        return True
        

    def replicate(self, rng_list: list[MRG32k3a]) -> tuple[dict, dict]:
        """Simulate one replication: run ASTROMORF with given hyperparameters.

        Args:
            rng_list (list[MRG32k3a]): Random number generators used to simulate
                the replication.

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
                        f"Reordering gamma values to satisfy gamma_1 >= gamma_2 >= gamma_3: {gammas} -> {sorted_gammas}"
                    )
                    gamma_1, gamma_2, gamma_3 = sorted_gammas
        except Exception:
            # If conversion fails or gammas missing, leave as-is and let downstream checks raise
            pass

        # Create solver with these hyperparameters
        if gamma_1 is None and gamma_2 is None and gamma_3 is None and subproblem_regularisation is None:
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
                "subproblem_regularisation": subproblem_regularisation
            }
        solver = instantiate_solver(
            solver_name=self.factors["solver"],
            fixed_factors=solver_factors,
        )

        # Create ProblemSolver instance
        experiment = ProblemSolver(
            solver=solver,
            problem=self.factors["target_problem"], create_pickle=False
        )


        print(f"Evaluating {experiment.solver.name} on {experiment.problem.name} with dimension={dimension}, degree={degree}, gamma_1={gamma_1}, gamma_2={gamma_2}, gamma_3={gamma_3}, subproblem_regularisation={subproblem_regularisation}.")

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
            print(f"No valid objective values obtained for dimension={dimension}, degree={degree}.")
            responses = {"weighted_score": 1e6}
            gradients = {
                response_key: {
                    factor_key: np.nan for factor_key in self.specifications
                }
                for response_key in responses
            }
            return responses, gradients

        # Calculate statistics
        mean_obj = -1 * self.factors['target_problem'].minmax[0] * np.mean(final_objectives) # turns maximisation into minimization


        #Find standard deviation between macroreplicaitons
        #first group together all estimated objectives across macroreplications
        grouped_objectives = {}
        for mrep in range(experiment.n_macroreps):
            for idx, est_obj in enumerate(experiment.all_est_objectives[mrep]):
                if idx not in grouped_objectives:
                    grouped_objectives[idx] = []
                grouped_objectives[idx].append(est_obj)
        #Now find the std dev across solutions for each macroreplication
        std_devs = []
        for idx, objs in grouped_objectives.items():
            if len(objs) > 1:
                std_devs.append(np.std(objs, ddof=1))
            else:
                std_devs.append(0)  # Penalize if only one rep
        std_obj = np.mean(std_devs)


        print(f'for dimension {dimension}, degree {degree}, gamma_1 {gamma_1}, gamma_2 {gamma_2}, gamma_3 {gamma_3}, subproblem_regularisation {subproblem_regularisation}: mean_obj = {mean_obj}, std_obj = {std_obj}')

        weighted_score = self.factors["quality_weight"] * mean_obj + self.factors["consistency_weight"] * std_obj

        responses = {"weighted_score": weighted_score}
        gradients = {
            response_key: {
                factor_key: np.nan for factor_key in self.specifications
            }
            for response_key in responses
        }
        return responses, gradients


class ASTROMORFHyperparameterProblemTwoDim(Problem):
    """Simulation optimization problem for finding optimal ASTROMORF hyperparameters.

    Decision variables:
        x[0]: subspace dimension (integer, 1 to problem.dim)
        x[1]: polynomial degree (integer, 1 to max_degree)

    Objective:
        Minimize weighted_score = quality_weight * normalized_mean + consistency_weight * normalized_std

    This treats hyperparameter optimization as a proper 2D integer-valued SO problem
    that can be solved with any SimOpt solver (e.g., ASTRO-DF for Bayesian optimization).
    """

    @classproperty
    @override
    def class_name_abbr(cls) -> str:
        return "ASTROMORF-HYPEROPT-1"

    @classproperty
    @override
    def class_name(cls) -> str:
        return "ASTROMORF Hyperparameter Optimization"

    @classproperty
    @override
    def n_objectives(cls) -> int:
        return 1

    @classproperty
    @override
    def n_stochastic_constraints(cls) -> int:
        return 0

    @classproperty
    @override
    def minmax(cls) -> tuple[int]:
        return (-1,)

    @classproperty
    @override
    def constraint_type(cls) -> ConstraintType:
        return ConstraintType.BOX

    @classproperty
    @override
    def variable_type(cls) -> VariableType:
        return VariableType.DISCRETE

    @classproperty
    @override
    def gradient_available(cls) -> bool:
        return False

    @classproperty
    @override
    def optimal_value(cls) -> float | None:
        return None

    @property
    @override
    def optimal_solution(self) -> tuple:
        return None

    @classproperty
    @override
    def model_default_factors(cls) -> dict:
        return {
            "consistency_weight": 0.2,
            "quality_weight": 0.8,
            "solver": "ASTROMORF",
            "n_macroreps": 3,
        }

    @classproperty
    @override
    def model_decision_factors(cls) -> set[str]:
        return {"dimension", "degree"}

    @classproperty
    @override
    def specifications(cls) -> dict[str, dict]:
        return {
            "initial_solution": {
                "description": "initial hyperparameter configuration (dimension, degree)",
                "datatype": tuple,
                "default": (3, 2),
            },
            "budget": {
                "description": "max number of hyperparameter evaluations",
                "datatype": int,
                "default": 100,
                "isDatafarmable": False,
            },
            "max_dimension": {
                "description": "maximum subspace dimension to test",
                "datatype": int,
                "default": None,
            },
            "max_degree": {
                "description": "maximum polynomial degree to test",
                "datatype": int,
                "default": 8,
            },
        }

    @property
    @override
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget,
            "max_dimension": self._check_max_dimension,
            "max_degree": self._check_max_degree,
        }

    @property
    @override
    def dim(self) -> int:
        return 2

    @property
    @override
    def lower_bounds(self) -> tuple:
        return (1,1)

    @property
    @override
    def upper_bounds(self) -> tuple:
        return (self.factors["max_dimension"], self.factors["max_degree"])

    def __init__(
        self,
        name: str = "ASTROMORF-HYPEROPT-1",
        fixed_factors: dict | None = None,
        model_fixed_factors: dict | None = None,
    ) -> None:
        """Initialize the ASTROMORF hyperparameter optimization problem.

        Args:
            name (str): User-specified name for problem.
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
        if "max_dimension" not in fixed_factors or fixed_factors["max_dimension"] is None:
            fixed_factors["max_dimension"] = target_problem.dim

        # Store target problem for reference
        self.target_problem = target_problem

        # Initialize with base class
        super().__init__(
            name=name,
            fixed_factors=fixed_factors,
            model_fixed_factors=model_fixed_factors,
            model=ASTROMORFHyperparameterModel,
        )

    def _check_max_dimension(self) -> None:
        if self.factors["max_dimension"] < 1:
            raise ValueError("max_dimension must be at least 1")

    def _check_max_degree(self) -> None:
        if self.factors["max_degree"] < 1:
            raise ValueError("max_degree must be at least 1")

    @override
    def vector_to_factor_dict(self, vector: tuple) -> dict:
        """Convert solution vector to factor dictionary.

        Args:
            vector (tuple): (dimension, degree).

        Returns:
            dict: Factor dictionary with integer values.
        """
        dimension, degree = vector
        return {"dimension": int(dimension), "degree": int(degree)}

    @override
    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:
        """Convert factor dictionary to solution vector.

        Args:
            factor_dict (dict): Dictionary with 'dimension' and 'degree' keys.

        Returns:
            tuple: (dimension, degree).
        """
        return (int(factor_dict["dimension"]), int(factor_dict["degree"]))

    @override
    def response_dict_to_objectives(self, response_dict: dict) -> tuple:
        """Convert response dictionary to objective values.

        Args:
            response_dict (dict): Dictionary with 'weighted_score' key.

        Returns:
            tuple: (weighted_score,) - objective to minimize.
        """
        return (response_dict["weighted_score"],)

    @override
    def response_dict_to_stoch_constraints(self, response_dict: dict) -> tuple:
        """Convert response dictionary to stochastic constraint values.

        Args:
            response_dict (dict): Response dictionary.

        Returns:
            tuple: Empty tuple (no constraints).
        """
        return ()

    @override
    def deterministic_objectives_and_gradients(self, x: tuple) -> tuple:
        """Compute deterministic components of objectives for a solution `x`.

        Args:
            x (tuple): Vector of decision variables.

        Returns:
            tuple: A tuple containing:
                - det_objectives (tuple): Vector of deterministic components of objectives.
                - det_objectives_gradients (tuple): Vector of gradients of deterministic
                    components of objectives.
        """
        det_objectives = (0,)
        det_objectives_gradients = ((0, 0),)
        return det_objectives, det_objectives_gradients

    @override
    def check_deterministic_constraints(self, x: tuple) -> bool:
        """Check if solution satisfies deterministic constraints.

        Args:
            x (tuple): Solution to check.

        Returns:
            bool: True if constraints satisfied.
        """
        return super().check_deterministic_constraints(x)

    @override
    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        """Generate a random solution.

        Args:
            rand_sol_rng (MRG32k3a): Random number generator.

        Returns:
            tuple: Random (dimension, degree) configuration.
        """
        dimension = rand_sol_rng.randint(
            self.factors["min_dimension"], self.factors["max_dimension"] 
        )
        degree = rand_sol_rng.randint(
            self.factors["min_degree"], self.factors["max_degree"] 
        )
        return (int(dimension), int(degree))


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
        Minimize weighted_score = quality_weight * normalized_mean + consistency_weight * normalized_std

    This treats hyperparameter optimization as a proper 6D mixed-integer-valued SO problem
    that can be solved with any SimOpt solver (e.g., ASTRO-DF for Bayesian optimization).
    """

    @classproperty
    @override
    def class_name_abbr(cls) -> str:
        return "ASTROMORF-HYPEROPT-2"

    @classproperty
    @override
    def class_name(cls) -> str:
        return "ASTROMORF Hyperparameter Optimization with Gamma values and regularisation"

    @classproperty
    @override
    def n_objectives(cls) -> int:
        return 1

    @classproperty
    @override
    def n_stochastic_constraints(cls) -> int:
        return 0

    @classproperty
    @override
    def minmax(cls) -> tuple[int]:
        return (-1,)

    @classproperty
    @override
    def constraint_type(cls) -> ConstraintType:
        return ConstraintType.DETERMINISTIC

    @classproperty
    @override
    def variable_type(cls) -> VariableType:
        return VariableType.MIXED

    @classproperty
    @override
    def gradient_available(cls) -> bool:
        return False

    @classproperty
    @override
    def optimal_value(cls) -> float | None:
        return None

    @property
    @override
    def optimal_solution(self) -> tuple:
        return None

    @classproperty
    @override
    def model_default_factors(cls) -> dict:
        return {
            "consistency_weight": 0.2,
            "quality_weight": 0.8,
            "solver": "ASTROMORF",
            "n_macroreps": 3,
            "gamma_1": 2.5,
            "gamma_2": 1.2,
            "gamma_3": 0.5,
            "subproblem_regularisation": 0.5,
        }

    @classproperty
    @override
    def model_decision_factors(cls) -> set[str]:
        return {"dimension", "degree", "gamma_1", "gamma_2", "gamma_3", "subproblem_regularisation"}

    @classproperty
    @override
    def specifications(cls) -> dict[str, dict]:
        return {
            "initial_solution": {
                "description": "initial hyperparameter configuration (dimension, degree, gamma_1, gamma_2, gamma_3, subproblem_regularisation)",
                "datatype": tuple,
                "default": (3, 2, 2.5, 1.2, 0.5, 0.5),
            },
            "budget": {
                "description": "max number of hyperparameter evaluations",
                "datatype": int,
                "default": 100,
                "isDatafarmable": False,
            },
            "max_dimension": {
                "description": "maximum subspace dimension to test",
                "datatype": int,
                "default": None,
            },
            "max_degree": {
                "description": "maximum polynomial degree to test",
                "datatype": int,
                "default": 8,
            },
        }

    @property
    @override
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget,
            "max_dimension": self._check_max_dimension,
            "max_degree": self._check_max_degree,
        }

    @property
    @override
    def dim(self) -> int:
        return 6

    @property
    @override
    def lower_bounds(self) -> tuple:
       tol = 1e-8
       gammas_lower = 1 + tol
       return (1, 1, 1e-8, gammas_lower, gammas_lower, tol)
    @property
    @override
    def upper_bounds(self) -> tuple:
        max_dimension = int(self.factors["max_dimension"])
        max_degree = int(self.factors["max_degree"])
        tol = 1-1e-8 
        return (max_dimension, max_degree, tol, np.inf, np.inf, np.inf)

    def __init__(
        self,
        name: str = "ASTROMORF-HYPEROPT-2",
        fixed_factors: dict | None = None,
        model_fixed_factors: dict | None = None,
    ) -> None:
        """Initialize the ASTROMORF hyperparameter optimization problem.

        Args:
            name (str): User-specified name for problem.
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
        if "max_dimension" not in fixed_factors or fixed_factors["max_dimension"] is None:
            fixed_factors["max_dimension"] = int(target_problem.dim)

        # Store target problem for reference
        self.target_problem = target_problem

        # Initialize with base class
        super().__init__(
            name=name,
            fixed_factors=fixed_factors,
            model_fixed_factors=model_fixed_factors,
            model=ASTROMORFHyperparameterModel,
        )

    def _check_max_dimension(self) -> None:
        if self.factors["max_dimension"] < 1:
            raise ValueError("max_dimension must be at least 1")

    def _check_max_degree(self) -> None:
        if self.factors["max_degree"] < 1:
            raise ValueError("max_degree must be at least 1")
    
    @override
    def vector_to_factor_dict(self, vector: tuple) -> dict:
        """Convert solution vector to factor dictionary.

        Args:
            vector (tuple): (dimension, degree, gamma_1, gamma_2, gamma_3, subproblem_regularisation).

        Returns:
            dict: Factor dictionary with integer values.
        """
        dimension, degree, gamma_1, gamma_2, gamma_3, subproblem_regularisation = vector
        return {"dimension": int(dimension), "degree": int(degree), "gamma_1": float(gamma_1), "gamma_2": float(gamma_2), "gamma_3": float(gamma_3), "subproblem_regularisation": float(subproblem_regularisation)}

    @override
    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:
        """Convert factor dictionary to solution vector.

        Args:
            factor_dict (dict): Dictionary with 'dimension' and 'degree' keys.

        Returns:
            tuple: (dimension, degree).
        """
        return (int(factor_dict["dimension"]), int(factor_dict["degree"]), float(factor_dict["gamma_1"]), float(factor_dict["gamma_2"]), float(factor_dict["gamma_3"]), float(factor_dict["subproblem_regularisation"]))

    @override
    def response_dict_to_objectives(self, response_dict: dict) -> tuple:
        """Convert response dictionary to objective values.

        Args:
            response_dict (dict): Dictionary with 'weighted_score' key.

        Returns:
            tuple: (weighted_score,) - objective to minimize.
        """
        return (response_dict["weighted_score"],)

    @override
    def response_dict_to_stoch_constraints(self, response_dict: dict) -> tuple:
        """Convert response dictionary to stochastic constraint values.

        Args:
            response_dict (dict): Response dictionary.

        Returns:
            tuple: Empty tuple (no constraints).
        """
        return ()

    @override
    def deterministic_objectives_and_gradients(self, x: tuple) -> tuple:
        """Compute deterministic components of objectives for a solution `x`.

        Args:
            x (tuple): Vector of decision variables.

        Returns:
            tuple: A tuple containing:
                - det_objectives (tuple): Vector of deterministic components of objectives.
                - det_objectives_gradients (tuple): Vector of gradients of deterministic
                    components of objectives.
        """
        det_objectives = (0,)
        det_objectives_gradients = ((0, 0, 0, 0, 0, 0,),)
        return det_objectives, det_objectives_gradients

    @override
    def check_deterministic_constraints(self, x: tuple) -> bool:
        """Check if solution satisfies deterministic constraints.

        Args:
            x (tuple): Solution to check.

        Returns:
            bool: True if constraints satisfied.
        """
        #! Need to edit these deterministic constraints to reflect bounds on between gamma values

        if not super().check_deterministic_constraints(x) :
            return False
        
        # Need gamma_1 >= gamma_2 >= gamma_3
        _, _, gamma_1, gamma_2, gamma_3, _ = x
        if not (gamma_1 >= gamma_2 and gamma_2 >= gamma_3):
            return False
        
        return True

    @override
    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        """Generate a random solution.

        Args:
            rand_sol_rng (MRG32k3a): Random number generator.

        Returns:
            tuple: Random (dimension, degree) configuration.
        """
        lb = self.lower_bounds
        ub = self.upper_bounds
        dimension = rand_sol_rng.randint(
            int(lb[0]), int(ub[0])
        )
        degree = rand_sol_rng.randint(
            int(lb[1]), int(ub[1])
        )
        gamma_1 = rand_sol_rng.uniform(
            lb[2], ub[2]
        )
        gamma_2 = rand_sol_rng.uniform(
            lb[3], ub[3]
        )
        gamma_3 = rand_sol_rng.uniform(
            lb[4], ub[4]
        )
        subproblem_regularisation = rand_sol_rng.uniform(
            lb[5], ub[5]
        )
        return (int(dimension), int(degree), float(gamma_1), float(gamma_2), float(gamma_3), float(subproblem_regularisation))