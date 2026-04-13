"""ASTROMoRF Solver.

The ASTROMoRF (Adaptive Sampling for Trust-Region Optimisation by Moving Ridge
Functions) progressively builds local models using interpolation on a reduced subspace
constructed through Active Subspace dimensionality reduction. The use of Active
Subspace reduction allows for a reduced number of interpolation points to be evaluated
in the model construction. This solver is particularly well-suited for high-dimensional
stochastic optimization problems where function evaluations are expensive.

UPDATES
-------
- Added adaptive subspace dimension adjustment based on solver progress and model
quality.
- Implemented plateau detection mechanism to reset subspace dimension when solver
progress stalls.
- Enhanced tracking of model quality metrics to inform dimension adjustment decisions.
    **The adaptive dimension logic monitors the solver's progress and adjusts the
    subspace
    dimension accordingly. If the solver experiences multiple consecutive unsuccessful
    iterations or detects a plateau in objective function values, it increases the
    subspace dimension to explore a larger portion of the search space. Conversely,
    if the solver is making good progress, it may decrease the subspace dimension to
    focus on a more promising subspace. This adaptive strategy aims to balance
    exploration
    and exploitation, improving convergence rates and solution quality.**
- Vectorized polynomial basis adapters for efficient model construction.

"""

from __future__ import annotations

import contextlib
import logging
import math
import traceback
import warnings
from collections import Counter
from collections.abc import Callable
from enum import Enum
from functools import partial
from math import ceil, log
from typing import Annotated, ClassVar, Self

import numpy as np
import scipy
import scipy.linalg
from numpy.linalg import norm, pinv, qr
from numpy.polynomial.chebyshev import chebder, chebvander
from numpy.polynomial.hermite_e import hermeder, hermevander
from numpy.polynomial.laguerre import lagder, lagvander
from numpy.polynomial.legendre import legder, legvander
from numpy.polynomial.polynomial import polyder, polyvander
from pydantic import Field, model_validator
from scipy.optimize import NonlinearConstraint, minimize
from scipy.special import betainc, factorial

from simopt.base import (
    ConstraintType,
    ObjectiveType,
    Problem,
    Solution,
    Solver,
    SolverConfig,
    VariableType,
)
from simopt.diagnostics import ASTROMoRF_Diagnostics
from simopt.solver import BudgetExhaustedError
from simopt.solvers.active_subspaces.basis import (
    MonomialPolynomialBasis,
    NaturalPolynomialBasis,
    NFPTensorBasis,
)

warnings.filterwarnings("ignore")


#! === POLYNOMIAL BASIS ADAPTERS ===


class PolyBasisType(Enum):  # noqa: D101
    # TensorBasis types (existing)
    HERMITE = "hermite"
    LEGENDRE = "legendre"
    CHEBYSHEV = "chebyshev"
    MONOMIAL = "monomial"
    # PolynomialBasis types (new)
    NATURAL = "natural"
    MONOMIAL_POLY = "monomial_polynomial"
    LAGRANGE = "lagrange"
    NFP = "nfp"
    LAGUERRE = "laguerre"


class PolynomialBasisAdapter:  # noqa: D101
    def __init__(self, vander, deriv) -> None:  # noqa: ANN001, D107
        self.vander = vander
        self.deriv = deriv

    def scale(self, X):  # noqa: ANN001, ANN201, D102, N803
        return X

    def dscale(self, d):  # noqa: ANN001, ANN201
        """Return scaling factors for derivatives (1D array of length d)."""
        return np.ones(d)


class BoxScalingAdapter(PolynomialBasisAdapter):  # noqa: D101
    def __init__(self, vander, deriv, lo, hi) -> None:  # noqa: ANN001, D107
        super().__init__(vander, deriv)
        self.lo = lo
        self.hi = hi
        self.scale_factor = None

    def scale_to_box(self, X: np.ndarray, lo: float, hi: float) -> np.ndarray:  # noqa: D102, N803
        if self.scale_factor is None:
            self.scale_factor = np.full(
                X.shape[1], (hi - lo) / np.maximum(np.ptp(X, axis=0), 1e-10)
            )
            self._offset = np.mean(X, axis=0, keepdims=True)
        return (X - self._offset) * self.scale_factor + lo

    def scale(self, X):  # noqa: ANN001, ANN201, D102, N803
        return self.scale_to_box(X, self.lo, self.hi)

    def dscale(self, d):  # noqa: ANN001, ANN201, ARG002
        """Return scaling factors for derivatives (1D array of length d)."""
        if self.scale_factor is None:
            raise ValueError("scale_factor not initialized - call scale() first")
        return self.scale_factor

    def reset_scaling(self) -> None:
        """Reset scaling parameters for new data."""
        self.scale_factor = None
        self._offset = None


class HermiteScalingAdapter(PolynomialBasisAdapter):
    """Adapter for Hermite polynomials that uses mean-std scaling."""

    def __init__(self, vander, deriv) -> None:  # noqa: ANN001, D107
        super().__init__(vander, deriv)
        self._mean = None
        self._std = None
        self._initialized = False

    def scale(self, X):  # noqa: ANN001, ANN201, D102, N803
        # Initialize scaling ONLY ONCE on first call
        if not self._initialized:
            self._mean = np.mean(X, axis=0, keepdims=True)
            self._std = np.std(X, axis=0, keepdims=True)
            # Avoid division by zero
            self._std = np.where(self._std < 1e-14, 1.0, self._std)
            self._initialized = True
        return (X - self._mean) / self._std / np.sqrt(2)

    def dscale(self, d):  # noqa: ANN001, ANN201, ARG002
        """Return scaling factors for derivatives (1D array of length d)."""
        if not self._initialized:
            raise ValueError("Scaling not initialized - call scale() first")
        return (1.0 / (self._std * np.sqrt(2))).flatten()

    def reset_scaling(self) -> None:
        """Reset scaling parameters for new data."""
        self._mean = None
        self._std = None
        self._initialized = False


class PolynomialBasisClassAdapter(PolynomialBasisAdapter):
    """Adapter for PolynomialBasis classes from active_subspaces/basis.py."""

    def __init__(self, basis_class) -> None:  # noqa: ANN001, D107
        # Store the class (not an instance) to instantiate later with problem context
        self.basis_class = basis_class
        self.basis_instance = None
        self.vander = None  # Will be set when basis is instantiated
        self.deriv = None  # Will be set when basis is instantiated

    def initialize_basis(self, degree: int, dim: int) -> None:
        """Initialize the basis instance with problem-specific parameters."""
        self.basis_instance = self.basis_class(degree, dim)
        self.vander = partial(vander_wrapper, basis_instance=self.basis_instance)
        self.deriv = partial(deriv_wrapper, basis_instance=self.basis_instance)

    def scale(self, X):  # noqa: ANN001, ANN201, D102, N803
        if self.basis_instance is not None and hasattr(self.basis_instance, "scale"):
            return self.basis_instance.scale(X)
        return X

    def dscale(self, d):  # noqa: ANN001, ANN201
        """Return scaling factors for derivatives (1D array of length d)."""
        if self.basis_instance is not None and hasattr(self.basis_instance, "_dscale"):
            try:
                return self.basis_instance._dscale()
            except Exception:
                return np.ones(d)
        return np.ones(d)


POLY_BASIS_LOOKUP: dict[PolyBasisType, PolynomialBasisAdapter] = {
    # TensorBasis types (using numpy polynomial functions directly)
    PolyBasisType.HERMITE: HermiteScalingAdapter(hermevander, hermeder),
    PolyBasisType.LEGENDRE: BoxScalingAdapter(legvander, legder, -1.0, 1.0),
    PolyBasisType.CHEBYSHEV: BoxScalingAdapter(chebvander, chebder, -1.0, 1.0),
    PolyBasisType.MONOMIAL: PolynomialBasisAdapter(polyvander, polyder),
    PolyBasisType.LAGUERRE: PolynomialBasisAdapter(lagvander, lagder),
    PolyBasisType.LAGRANGE: PolynomialBasisAdapter(lagvander, lagder),
    # PolynomialBasis types (using basis classes from active_subspaces/basis.py)
    PolyBasisType.NFP: PolynomialBasisClassAdapter(NFPTensorBasis),
    PolyBasisType.NATURAL: PolynomialBasisClassAdapter(NaturalPolynomialBasis),
    PolyBasisType.MONOMIAL_POLY: PolynomialBasisClassAdapter(MonomialPolynomialBasis),
}


class ASTROMoRFConfig(SolverConfig):
    """Configuration for ASTROMoRF solver."""

    crn_across_solns: Annotated[
        bool,
        Field(default=False, description="use CRN across solutions?"),
    ]
    mu: Annotated[
        float,
        Field(default=1000.0, gt=0, description="dampening of the criticality step"),
    ]
    eta_1: Annotated[
        float,
        Field(default=0.1, gt=0, description="threshold for a successful iteration"),
    ]
    eta_2: Annotated[
        float,
        Field(
            default=0.8,
            description="threshold for a very successful iteration",
        ),
    ]
    gamma_1: Annotated[
        float,
        Field(
            default=2.5,
            gt=1,
            description="trust-region radius increase rate after very successful iteration",  # noqa: E501
        ),
    ]
    gamma_2: Annotated[
        float,
        Field(
            default=1.2,
            gt=1,
            description="trust-region radius increase rate after successful iteration",
        ),
    ]
    gamma_3: Annotated[
        float,
        Field(
            default=0.5,
            gt=0,
            lt=1,
            description="trust-region radius decrease rate after unsuccessful iteration",  # noqa: E501
        ),
    ]
    lambda_min: Annotated[
        int, Field(default=5, gt=2, description="minimum sample size")
    ]
    subproblem_regularisation: Annotated[
        float,
        Field(
            default=0.15,
            ge=0,
            le=1,
            description="regularisation parameter for the subproblem",
        ),
    ]
    ps_sufficient_reduction: Annotated[
        float,
        Field(
            default=0.1,
            ge=0,
            description=(
                "use pattern search if with sufficient reduction, "
                "0 always allows it, large value never does"
            ),
        ),
    ]
    initial_subspace_dimension: Annotated[
        int,
        Field(
            default=4,
            ge=1,
            description="dimension size of the active subspace",
            alias="initial subspace dimension",
        ),
    ]
    polynomial_degree: Annotated[
        int,
        Field(
            default=4,
            ge=1,
            description="the degree of the local model",
            alias="polynomial degree",
        ),
    ]
    polynomial_basis: Annotated[
        PolyBasisType,
        Field(
            default=PolyBasisType.HERMITE,
            description="the polynomial basis type for the local model",
            alias="polynomial basis",
        ),
    ]
    record_diagnostics: Annotated[
        bool,
        Field(
            default=False,
            description="flag to record detailed diagnostics to a CSV file",
            alias="Record Diagnostics",
        ),
    ]
    elliptical_trust_region: Annotated[
        bool,
        Field(
            default=False,
            description="use elliptical trust-region based on model Hessian",
            alias="elliptical trust region",
        ),
    ]
    adaptive_subspace_dimension: Annotated[
        bool,
        Field(
            default=True,
            description="adaptively adjust subspace dimension based on solver progress",
            alias="adaptive subspace dimension",
        ),
    ]
    variance_explained_threshold: Annotated[
        float,
        Field(
            default=0.95,
            ge=0.5,
            le=1.0,
            description="fraction of variance to capture when determining optimal subspace dimension",  # noqa: E501
            alias="variance explained threshold",
        ),
    ]
    # (Previously supported budget-aware config fields have been removed
    # in favor of hard-coded adaptive-subspace parameters embedded in the
    # solver implementation.)

    @model_validator(mode="after")
    def _validate_eta_2_greater_than_eta_1(self) -> Self:
        if self.eta_2 <= self.eta_1:
            raise ValueError("Eta 2 must be greater than Eta 1.")
        return self

    @model_validator(mode="after")
    def _validate_gamma_1_greater_than_gamma_2(self) -> Self:
        if self.gamma_1 < self.gamma_2:
            raise ValueError("Gamma 1 must be greater than or equal to Gamma 2.")
        return self


class ASTROMORF(Solver):
    """The ASTROMoRF solver."""

    # Hard-coded adaptive-subspace defaults (budget-aware behavior is embedded)
    DEFAULT_COST_PENALTY: float = 0.05
    DEFAULT_COST_POWER: float = 2.0
    DEFAULT_BUDGET_ALPHA: float = 0.5
    DEFAULT_BASELINE_BUDGET: float = 1000.0
    DEFAULT_SUFFICIENT_SCORE_TOL: float = 0.01
    DEFAULT_PREFER_SMALL_SUFFICIENT: bool = True

    name: str = "ASTROMORF"
    config_class: ClassVar[type[SolverConfig]] = ASTROMoRFConfig
    class_name_abbr: ClassVar[str] = "ASTROMORF"
    class_name: ClassVar[str] = "ASTROMoRF"
    objective_type: ClassVar[ObjectiveType] = ObjectiveType.SINGLE
    constraint_type: ClassVar[ConstraintType] = ConstraintType.BOX
    variable_type: ClassVar[VariableType] = VariableType.CONTINUOUS
    gradient_needed: ClassVar[bool] = False

    @property
    def iteration_count(self) -> int:
        """Get the current iteration count."""
        return self._iteration_count

    @iteration_count.setter
    def iteration_count(self, value: int) -> None:
        """Set the current iteration count."""
        self._iteration_count = value

    @property
    def delta(self) -> float:
        """Get the current trust-region radius."""
        return self._delta

    @delta.setter
    def delta(self, value: float) -> None:
        """Set the current trust-region radius."""
        self._delta = value

    @property
    def delta_max(self) -> float:
        """Get the maximum trust-region radius."""
        return self._delta_max

    @delta_max.setter
    def delta_max(self, value: float) -> None:
        """Set the maximum trust-region radius."""
        self._delta_max = value

    @property
    def delta_min(self) -> float:
        """Get the minimum trust-region radius."""
        return self._delta_min

    @delta_min.setter
    def delta_min(self, value: float) -> None:
        """Set the minimum trust-region radius."""
        self._delta_min = value

    @property
    def incumbent_x(self) -> tuple[float, ...]:
        """Get the incumbent solution."""
        return self._incumbent_x

    @incumbent_x.setter
    def incumbent_x(self, value: tuple[float, ...]) -> None:
        """Set the incumbent solution."""
        self._incumbent_x = value

    @property
    def incumbent_solution(self) -> Solution:
        """Get the incumbent solution object."""
        return self._incumbent_solution

    @incumbent_solution.setter
    def incumbent_solution(self, value: Solution) -> None:
        """Set the incumbent solution object."""
        self._incumbent_solution = value

    @property
    def prev_U(self) -> np.ndarray | None:  # noqa: N802
        """Get the previous active subspace matrix."""
        return self._prev_U

    @prev_U.setter
    def prev_U(self, value: np.ndarray | None) -> None:  # noqa: N802
        """Set the previous active subspace matrix."""
        self._prev_U = value

    @property
    def prev_H(self) -> np.ndarray | None:  # noqa: N802
        """Get the previous model Hessian."""
        return self._prev_H

    @prev_H.setter
    def prev_H(self, value: np.ndarray | None) -> None:  # noqa: N802
        """Set the previous model Hessian."""
        self._prev_H = value

    @property
    def degree(self) -> int:
        """Get the polynomial degree of the local model."""
        return self._degree

    @degree.setter
    def degree(self, value: int) -> None:
        """Set the polynomial degree of the local model."""
        self._degree = value

    @property
    def basis_adapter(self) -> PolynomialBasisAdapter:
        """Get the polynomial basis adapter."""
        return self._basis_adapter

    @basis_adapter.setter
    def basis_adapter(self, value: PolynomialBasisAdapter) -> None:
        """Set the polynomial basis adapter."""
        self._basis_adapter = value

    @property
    def model_grad(self) -> Callable[[np.ndarray], np.ndarray]:
        """Get the model gradient function."""
        return self._model_grad

    @model_grad.setter
    def model_grad(self, value: Callable[[np.ndarray], np.ndarray]) -> None:
        """Set the model gradient function."""
        self._model_grad = value

    @property
    def model_hess(self) -> Callable[[np.ndarray], np.ndarray]:
        """Get the model Hessian function."""
        return self._model_hess

    @model_hess.setter
    def model_hess(self, value: Callable[[np.ndarray], np.ndarray]) -> None:
        """Set the model Hessian function."""
        self._model_hess = value

    @property
    def model(self) -> Callable[[np.ndarray], float]:
        """Get the model function."""
        return self._model

    @model.setter
    def model(self, value: Callable[[np.ndarray], float]) -> None:
        """Set the model function."""
        self._model = value

    def _set_basis(self, basis: PolyBasisType, problem: Problem | None = None) -> None:  # noqa: ARG002
        """Set the polynomial basis for the local model.

        Args:
                basis: The polynomial basis type to set.
                problem: The simulation problem (needed for PolynomialBasisClassAdapter
                        initialization).
        """
        self.basis_adapter = POLY_BASIS_LOOKUP[basis]

        # If this is a PolynomialBasisClassAdapter, initialize it with problem context
        if isinstance(self.basis_adapter, PolynomialBasisClassAdapter):
            self.basis_adapter.initialize_basis(
                self.factors["polynomial degree"],
                self.factors["initial subspace dimension"],
            )

        # set the vander and deriv functions for easy access
        self.vander = self.basis_adapter.vander
        self.polyder = self.basis_adapter.deriv

    def set_basis(self, basis: PolyBasisType, problem: Problem | None = None) -> None:
        """Set the polynomial basis for the local model.

        Args:
                basis: The polynomial basis type to set.
                problem: The simulation problem (needed for PolynomialBasisClassAdapter
                        initialization).
        """
        if problem is not None:
            # Re-initialize basis with problem context
            self._set_basis(basis, problem)
        else:
            # Initial setup without problem context
            raise ValueError("Problem context is required to set basis.")

    def _initialize_solving(self) -> None:
        """Setup the solver for the first iteration."""
        # For creating all the class members needed for the run of the algorithm
        self.d: int = self.factors["initial subspace dimension"]

        # Compute optimal polynomial degree based on subspace dimension
        # This ensures well-conditioned interpolation (points/terms ratio >= 0.60)
        self.degree = self.factors["polynomial degree"]

        if self.factors["Record Diagnostics"]:
            self.diagnostics = ASTROMoRF_Diagnostics(self, self.problem)

        # Check for potential performance issues with large polynomial basis
        if self.factors["Record Diagnostics"]:
            self.diagnostics.check_polynomial_complexity()

        self.eta_1: float = self.factors["eta_1"]
        self.eta_2: float = self.factors["eta_2"]
        self.gamma_1: float = self.factors["gamma_1"]
        self.gamma_2: float = self.factors["gamma_2"]
        self.gamma_3: float = self.factors["gamma_3"]
        self.mu: float = self.factors["mu"]
        self.lambda_min: int = self.factors["lambda_min"]

        self.set_basis(self.factors["polynomial basis"], self.problem)

        self.delta_max = self.calculate_max_radius()
        # Use same initialization as ASTRO-DF for consistency
        self.delta = 10 ** (ceil(log(self.delta_max * 2, 10) - 1) / self.problem.dim)
        self.delta_initial: float = self.delta
        self.delta_min = 0.01 * self.delta_max

        self.delta_power: int = 2 if self.factors["crn_across_solns"] else 4

        rng = self.rng_list[1]

        if "initial_solution" in self.problem.factors:
            self.incumbent_x = tuple(self.problem.factors["initial_solution"])
        else:
            self.incumbent_x = tuple(self.problem.get_random_solution(rng))

        self.incumbent_solution = self.create_new_solution(
            self.incumbent_x, self.problem
        )

        self.fval: float | None = None

        # Locked-in objective value for fn_estimates (snapshot at acceptance time)
        # This prevents fn_estimates from changing when incumbent gets more samples
        self.locked_incumbent_objective: float | None = None

        # Reset iteration count and data storage
        self.iteration_count = 1
        self.record_update = 1
        self.unsuccessful_iterations: list = []
        self.successful_iterations: list = []
        self.visited_points: list[Solution] | None = []
        self.kappa: float | None = None
        self.projection_history: list[float] = []  # NEW: track projection distances

        # Track prediction quality for TR expansion dampening
        self.recent_prediction_errors: list = []  # Store last few relative errors

        # Adaptive subspace dimension tracking
        self.consecutive_unsuccessful: int = 0
        self.consecutive_successful: int = 0
        self.recent_gradient_norms: list[float] = []
        self.d_history: list[int] = [self.d]  # Track subspace dimension over time
        self.max_d: int = self.problem.dim - 1  # Max subspace dimension
        self.previous_model_information: list = []  # Store previous model info for dimension decisions  # noqa: E501

        # Enhanced tracking for adaptive dimension stopping rule
        # Maps dimension d -> list of (iteration, residual_norm, prediction_error,
        # success)
        self.dimension_performance: dict[int, list[tuple[int, float, float, bool]]] = {}
        # Store eigenvalue spectrum from gradient outer products for variance-explained
        # stopping rule
        self.gradient_eigenvalues: list[np.ndarray] = []
        # Store model quality metrics per dimension at each iteration
        self.model_quality_by_dimension: dict[int, list[float]] = {}
        # Store the subspace stability measure (angle between consecutive subspaces)
        self.subspace_angles: list[float] = []
        # Threshold for variance explained stopping rule (fraction of total variance to
        # capture)
        self.variance_explained_threshold: float = self.factors.get(
            "variance explained threshold", 0.95
        )
        # Adaptive-subspace parameters are hard-coded via class defaults
        # (do not read these from solver factors/config)
        # Instances can still override attributes manually if necessary.
        # The defaults below were chosen from synthetic tuning experiments.
        # Note: keep these lines for clarity; they do not read from config.
        self.cost_penalty = float(self.DEFAULT_COST_PENALTY)
        self.cost_power = float(self.DEFAULT_COST_POWER)
        self.budget_alpha = float(self.DEFAULT_BUDGET_ALPHA)
        self.baseline_budget = float(self.DEFAULT_BASELINE_BUDGET)
        self.sufficient_score_tol = float(self.DEFAULT_SUFFICIENT_SCORE_TOL)
        self.prefer_small_sufficient = bool(self.DEFAULT_PREFER_SMALL_SUFFICIENT)

        # Plateau detection and dimension reset mechanism
        self.initial_subspace_dimension: int = (
            self.d
        )  # Store initial d for reset target
        self.recent_objective_values: list[
            float
        ] = []  # Track recent objective values for plateau detection
        # Number of iterations to check for plateau: compute based on available budget
        # so short/long runs adapt automatically. Use a coarse heuristic:
        # plateau_window = clamp(budget // 50, 3, 20)
        try:
            budget_total = int(self.budget.total) if hasattr(self, "budget") else None
        except Exception:
            budget_total = None

        # Compute plateau window via piecewise mapping helper
        self.plateau_window: int = int(self.compute_plateau_window(budget_total))
        self.plateau_threshold: float = (
            0.01  # Relative improvement threshold (0.5% - less strict than 0.1%)
        )
        self.in_dimension_reset: bool = (
            False  # Whether we're in a dimension reset cycle
        )
        self.reset_start_iteration: int = 0  # When the reset cycle started
        self.reset_target_d: int = self.d  # Target dimension to decay back to
        self.dimension_reset_count: int = 0  # Number of times we've triggered a reset
        self.max_dimension_resets: int = (
            5  # Maximum number of resets allowed per run (increased from 3)
        )

        # Warm starting: store the active subspace from previous iteration
        self.prev_U = None

        # Store previous Hessian for ellipsoidal trust-region construction
        self.prev_H = np.eye(self.problem.dim)

        # Track last iteration when `d` changed to avoid long suppression
        self.last_d_change_iteration: int = 0

        # create initializations of the model functions
        self.model = None
        self.model_grad = None
        self.model_hess = None

    def solve(self, problem: Problem) -> None:
        """Run a single macroreplication of the solver on a problem.

        Args:
                problem: Simulation-optimization problem to solve.
        """
        self.problem = problem
        self._initialize_solving()

        try:
            while self.budget.remaining > 0:
                # TODO: Rewrite
                self.initial_evaluation()

                # Build random model
                U, fval, interpolation_solns, X, fX = self.construct_model()  # noqa: N806

                # Store the active subspace for warm starting next iteration
                self.prev_U = U.copy()

                # Diagnose model quality
                if self.factors["Record Diagnostics"]:
                    # diagnostics.diagnose_model_quality expects (model, model_grad, X,
                    # fX, U)
                    self.diagnostics.diagnose_model_quality(
                        self.model, self.model_grad, X, fX, U
                    )

                # Solve random model
                candidate_solution = self.solve_subproblem(U)

                # Sample candidate solution
                candidate_solution, fval_tilde = self.simulate_candidate_soln(
                    candidate_solution, self.delta
                )

                # Update relative error history
                self.compute_relative_error(candidate_solution, fval_tilde)

                # Diagnose candidate solution quality
                if self.factors["Record Diagnostics"]:
                    # Pass both model and design matrix X explicitly
                    self.diagnostics.diagnose_candidate_solution(
                        candidate_solution, self.model, X
                    )

                # Evaluate model (adaptive trust region shrinkage now handled in
                # update_parameters)
                self.evaluate_candidate_solution(
                    fval,
                    fval_tilde,
                    interpolation_solns,
                    candidate_solution,
                    X,
                )

                # Record objective for plateau detection (keeps sliding window)
                try:
                    if (
                        hasattr(self, "incumbent_solution")
                        and self.incumbent_solution is not None
                    ):
                        current_objective = (
                            self.incumbent_solution.objectives_mean.item()
                        )
                        self.record_objective_for_plateau_detection(current_objective)
                except Exception:
                    pass

                # adaptive dimension update logic
                if self.factors.get("adaptive subspace dimension", False):
                    self.compute_optimal_subspace_dimension()

                # Adaptive subspace dimension adjustment
                # if self.factors.get("adaptive subspace dimension", False):
                #     # Record performance for this iteration's dimension
                #     current_pred_error = (
                #         self.recent_prediction_errors[-1]
                #         if self.recent_prediction_errors else 0.0
                #     )
                #     was_successful = self.consecutive_successful > 0 or (
                #         len(self.successful_iterations) > 0
                #         and self.iteration_count in self.successful_iterations
                #     )
                #     # Compute residual norm from model quality
                #     try:
                #         residual_norm = np.sqrt(np.sum(
                # (np.array([self.model(x.reshape(-1, 1)) for x in X]) - fX.flatten())
                # ** 2
                #         ))
                #     except Exception:
                #         residual_norm = 0.0

                #     self.record_dimension_performance(
                #         self.d, residual_norm, abs(current_pred_error), was_successful
                #     )

                #     # Record objective value for plateau detection
                #     current_objective = self.incumbent_solution.objectives_mean.item()
                #     self.record_objective_for_plateau_detection(current_objective)

                #     # Compute gradient norm for adaptive decision
                # grad_norm = norm(self.model_grad(np.zeros((1, self.problem.dim)),
                # full_space=True))
                #     new_d = self.compute_optimal_subspace_dimension(grad_norm)
                #     if new_d != self.d:
                #         self.d = new_d
                #         # Re-initialize basis adapter for new dimension
                #         self.set_basis(self.factors["polynomial basis"], self.problem)

                """
                    If the trust-region is still large and the subspace dimension is
                    high,
                    reduce the polynomial degree to 2 to improve model accuracy and
                    stability.
                    This should follow the STRONG
                """
                if (
                    self.delta >= self.delta_initial / 2
                    and self.d > self.problem.dim * 0.6
                    and self.degree != 2
                ):
                    self.degree = 2
                else:
                    # Reset to original polynomial degree
                    self.degree = self.factors["polynomial degree"]

                # At the end of every iteration record iteration data
                if self.iteration_count > 1:
                    self.iterations.append(self.iteration_count)
                    self.budget_history.append(self.budget.used)
                    # Use locked objective value to prevent fn_estimate from changing
                    # when the solution object gets more samples in later iterations
                    if self.locked_incumbent_objective is not None:
                        self.fn_estimates.append(self.locked_incumbent_objective)
                    else:
                        self.fn_estimates.append(
                            self.incumbent_solution.objectives_mean.item()
                        )
                    self.record_update += 1
                self.iteration_count += 1
        except BudgetExhaustedError:
            if self.factors["Record Diagnostics"]:
                logging.info("ASTROMoRF solver finalising...")
            # Final record at budget exhaustion
            if self.record_update < self.iteration_count:
                self.fn_estimates.append(
                    self.locked_incumbent_objective
                    if self.locked_incumbent_objective is not None
                    else self.incumbent_solution.objectives_mean.item()
                )
                self.budget_history.append(self.budget.used)
                self.iterations.append(self.iteration_count)

        except Exception as e:
            logging.error(
                f"An error occurred in the ASTROMoRF solver: {e.__class__.__name__}"
            )
            logging.error(traceback.format_exc())
        finally:
            if np.isnan(self.fn_estimates).any() or np.isinf(self.fn_estimates).any():
                logging.warning(
                    "Warning: NaN or Inf detected in function value estimates."
                )

            if self.factors["Record Diagnostics"]:
                logging.info("ASTROMoRF solver finalising...")
                self.diagnostics.write_final_diagnostics()

    # === ADAPTIVE SUBSPACE DIMENSION ===

    def record_iteration_performance(
        self, trust_region: float, current_solution: Solution, success: bool
    ) -> None:
        """Record the performance of the current iteration to be used in deciding.

        optimal subspace dimensions.


        Record in the self.previous_model_information structure a dictionary of the
        following format:
        {
            'model': a tuple of the model functions self.model and self.model_grad,
            'trust_region_radius': a float of the iteration's trust-region radius,
            'solution': a Solution object of the incumbent solution at the end of the
            iteration,
            'model_success': a bool indicating whether the iteration was successful,
            'eigenvalue_spectrum': a list[float] of the eigenvalues of the gradient
            outer product matrix of the model,
            'recommended_dimension': an int of the posterior recommended subspace
            dimension,.

        }
        """
        current_iteration_info = {}
        current_iteration_info["model"] = (self.model, self.model_grad)
        current_iteration_info["trust_region_radius"] = trust_region
        current_iteration_info["solution"] = current_solution
        current_iteration_info["model_success"] = success

        # compute eigenvalue spectrum
        eigvals = self.compute_eigenvalue_spectrum_model(current_solution, trust_region)
        current_iteration_info["eigenvalue_spectrum"] = eigvals

        # compute recommended dimension based on variance explained
        total_variance = np.sum(eigvals)
        cumulative_variance = np.cumsum(eigvals)
        variance_ratios = cumulative_variance / total_variance
        recommended_dimension = (
            np.searchsorted(variance_ratios, self.variance_explained_threshold) + 1
        )
        current_iteration_info["recommended_dimension"] = recommended_dimension

        # Attach validation_by_d from the most recent fit(), if available
        if hasattr(self, "last_validation_by_d") and isinstance(
            self.last_validation_by_d, dict
        ):
            current_iteration_info["validation_by_d"] = self.last_validation_by_d.copy()
            # clear it to avoid accidental reuse
            with contextlib.suppress(Exception):
                del self.last_validation_by_d

        self.previous_model_information.append(current_iteration_info)

    def compute_eigenvalue_spectrum_model(
        self, current_solution: Solution, delta: float
    ) -> list[float]:
        """Compute the eigenvalue spectrum of the gradient outer product matrix.

        for the given model and active subspace U.

        Args:
            current_solution (Solution): The current incumbent solution.
            delta (float): The current trust-region radius.

        Returns:
            list[float]: The eigenvalues of the gradient outer product matrix.
        """
        # heuristic for number of points; ensure at least a few samples
        no_pts = max(
            self.problem.dim,
            int(6 * np.log(max(self.problem.dim, 2)) * self.problem.dim),
        )
        x_k = np.array(current_solution.x).reshape(-1, 1)
        # Ensure problem bounds are arrays shaped (n,1) to avoid unwanted broadcasting
        lower_bounds_arr = np.array(self.problem.lower_bounds).reshape(x_k.shape)
        upper_bounds_arr = np.array(self.problem.upper_bounds).reshape(x_k.shape)

        bound_l = np.maximum(lower_bounds_arr, x_k - delta)
        bound_u = np.minimum(upper_bounds_arr, x_k + delta)

        lower = (bound_l - x_k).flatten()
        upper = (bound_u - x_k).flatten()

        directions = self.random_directions(num_pnts=no_pts, lower=lower, upper=upper)
        shifts = directions[1:, :]  # use shifts (skip the first zero if desired)
        if shifts.shape[0] == 0:
            return [0.0] * self.problem.dim
        X = np.array([x_k.flatten() + s for s in shifts])  # noqa: N806

        # Safely compute gradients; fall back to zeros on failure
        grads = []
        for x in X:
            try:
                g = self.model_grad(x.reshape(1, -1), full_space=True).flatten()
                if g.shape[0] != self.problem.dim:
                    g = np.zeros(self.problem.dim)
            except Exception:
                g = np.zeros(self.problem.dim)
            grads.append(g)
        G = np.vstack(grads)  # shape (m, n)  # noqa: N806
        C = (G.T @ G) / G.shape[0]  # shape (n, n)  # noqa: N806

        eigvals, _eigvecs = np.linalg.eigh(C)
        eigvals_desc = eigvals[::-1]  # descending
        eigenvalues = np.maximum(eigvals_desc, 0)  # ensure non-negative
        return eigenvalues.flatten().tolist()

    def compute_plateau_window(self, budget_total: int | None) -> int:
        """Compute a plateau detection window based on the total budget using.

        a piecewise mapping. Smaller budgets get smaller windows so the
        solver can trigger resets earlier; larger budgets use larger
        windows to avoid spurious resets.

        Mapping (example, clamped to [2,20]):
          budget <= 50   -> 2
          51 .. 100      -> 3
          101 .. 200     -> 4
          201 .. 500     -> 6
          501 .. 1000    -> 8
          1001 .. 5000   -> 12
          > 5000         -> 16

        Returns:
            int: plateau window size (number of iterations)
        """
        if budget_total is None:
            return 4

        try:
            b = int(max(0, int(budget_total)))
        except Exception:
            return 4

        if b <= 50:
            w = 2
        elif b <= 100:
            w = 3
        elif b <= 200:
            w = 4
        elif b <= 500:
            w = 6
        elif b <= 1000:
            w = 8
        elif b <= 5000:
            w = 12
        else:
            w = 16

        return max(2, min(20, int(w)))

    def check_intersecting_trust_regions(
        self, centre_to_check: Solution, radius_to_check: float
    ) -> bool:
        """Check if the trust-region defined by centre_to_check and radius_to_check.

        intersects with the current trust-region and solution.

        Args:
            centre_to_check (Solution): A previous iteration's solution (centre of
            trust-region)
            radius_to_check (float): A previous iteration's trust-region radius.

        Returns:
            bool: True if the trust-regions intersect, False otherwise
        """
        x_current = np.array(self.incumbent_x).reshape(-1, 1)
        x_previous = np.array(centre_to_check.x).reshape(-1, 1)
        distance = norm(x_current - x_previous)
        tol = 0.2
        # Trust-regions intersect if center distance <= sum of radii (+ small tolerance)
        return distance <= (radius_to_check + self.delta + tol)

    def intersection_amount(self, model_info: dict) -> float:
        """Compute the amount of intersection between the trust-region defined by.

        model_info and the current trust-region.

        Args:
            model_info (dict): A previous iteration's model information dictionary.

        Returns:
            float: The amount of intersection between the two trust-regions
        """
        x_current = np.array(self.incumbent_x).reshape(-1, 1)
        x_previous = np.array(model_info["solution"].x).reshape(-1, 1)
        model_delta = model_info["trust_region_radius"]

        d = np.linalg.norm(x_current - x_previous)

        # No overlap
        if d >= model_delta + self.delta:
            return 0.0

        # Full containment
        if d <= abs(model_delta - self.delta):
            return (
                1.0
                if self.delta <= model_delta
                else (model_delta / self.delta) ** self.problem.dim
            )

        # Cap heights
        h_r = (model_delta + self.delta - d) * (self.delta - model_delta + d) / (2 * d)
        h_R = (model_delta + self.delta - d) * (model_delta - self.delta + d) / (2 * d)  # noqa: N806

        # Regularized beta arguments
        z_r = 1.0 - (1.0 - h_r / self.delta) ** 2
        z_R = 1.0 - (1.0 - h_R / model_delta) ** 2  # noqa: N806

        a = (self.problem.dim + 1) / 2
        b = 0.5

        # Normalized cap volumes
        cap_r = 0.5 * betainc(a, b, z_r)
        cap_R = (  # noqa: N806
            0.5 * (model_delta / self.delta) ** self.problem.dim * betainc(a, b, z_R)
        )

        return cap_r + cap_R

    def obtain_intersecting_models(
        self, successful_models: list[dict]
    ) -> tuple[list[dict], list[float]]:
        """Obtain the list of previous models whose trust-regions intersect with the.

        current trust-region.


            and order the list by how much they intersect the current trust-region by
            descending order
        Args:
            successful_models (list[dict]): A list of previous model information
            dictionaries that were successful
        Returns:
            tuple[list[dict], list[float]]: A tuple containing a list of previous model
            information dictionaries that intersect
            with the current trust-region and a list of their corresponding intersection
            amounts.
        """
        # Populate list of intersecting models
        intersecting_models = []
        for info in successful_models:
            if self.check_intersecting_trust_regions(
                info["solution"], info["trust_region_radius"]
            ):
                intersecting_models.append(info)

        # Order intersecting models by how much they intersect the current trust-region
        # by descending order
        intersecting_models.sort(key=self.intersection_amount, reverse=True)

        # get the intersection amounts
        intersection_amounts = [
            self.intersection_amount(info) for info in intersecting_models
        ]

        return intersecting_models, intersection_amounts

    #! THIS LOGIC NEEDS TO BE FILLED IN
    def infer_optimal_dimension_from_models(
        self, intersecting_models: list[dict]
    ) -> int:
        """Infer the optimal subspace dimension based on intersecting models'.

        recommended dimensions.

        Args:
            intersecting_models (list[dict]): A list of previous model information
            dictionaries that intersect with the current trust-region.

        Returns:
            int: The inferred optimal subspace dimension
        """
        recent_eigenvalues = [
            eigenvals
            for model_info in intersecting_models
            for eigenvals in [model_info["eigenvalue_spectrum"]]
        ]

        # average the eigenvalue spectra
        max_length = max(len(eigvals) for eigvals in recent_eigenvalues)
        avg_eigenvalues = np.zeros(max_length)
        counts = np.zeros(max_length)

        for eigvals in recent_eigenvalues:
            avg_eigenvalues[: len(eigvals)] += np.array(eigvals)
            counts[: len(eigvals)] += 1

        avg_eigenvalues = avg_eigenvalues / np.maximum(counts, 1)

        total_variance = np.sum(avg_eigenvalues)
        if total_variance < 1e-10:
            return 1  # All eigenvalues are zero, return minimum dimension

        cumulative_variance = np.cumsum(avg_eigenvalues) / total_variance

        # find the smallest dimension where cumulative variance exceeds threshold
        optimal_d = (
            np.searchsorted(cumulative_variance, self.variance_explained_threshold) + 1
        )
        return min(
            optimal_d, self.problem.dim - 1
        )  # ensure not exceeding problem.dim -1

    def evaluate_and_score_candidate_dimensions(self) -> dict:
        """Evaluate candidate subspace dimensions using available validation metrics,.

        eigen-spectrum (projection residual proxy) and historical success rates.

        Returns:
            dict: mapping d -> {"val_err":..., "proj_resid":..., "success_rate":...,
            "score":...}
        """
        results = {}
        # Determine candidate range
        max_test_d = min(self.max_d, self.problem.dim - 1)
        if max_test_d < 1:
            return {1: {"score": 0.0}}

        # Gather validation_by_d candidates from most recent model fits
        # Prefer the most recent explicit validation (self.last_validation_by_d)
        validation_by_d = getattr(self, "last_validation_by_d", None)
        if validation_by_d is None:
            # Try to average validation_by_d from previous_model_information entries
            vals = {}
            counts = {}
            for info in self.previous_model_information:
                vbd = info.get("validation_by_d")
                if isinstance(vbd, dict):
                    for k, v in vbd.items():
                        k = int(k)
                        vals[k] = vals.get(k, 0.0) + float(v)
                        counts[k] = counts.get(k, 0) + 1
            if counts:
                validation_by_d = {k: vals[k] / counts[k] for k in vals}

        # Build eigen-spectrum proxy for projection residuals
        eig_source = None
        if hasattr(self, "gradient_eigenvalues") and self.gradient_eigenvalues:
            try:
                eig_source = np.array(self.gradient_eigenvalues[-1], dtype=float)
            except Exception:
                eig_source = None

        if eig_source is None:
            # fallback: average eigenvalue_spectrum from previous_model_information
            spectra = [
                np.array(info.get("eigenvalue_spectrum", []), dtype=float)
                for info in self.previous_model_information
                if info.get("eigenvalue_spectrum")
            ]
            if spectra:
                # pad to same length
                maxlen = max(s.shape[0] for s in spectra)
                padded = np.vstack(
                    [
                        np.pad(s, (0, maxlen - s.shape[0]), constant_values=0.0)
                        for s in spectra
                    ]
                )
                eig_source = np.mean(padded, axis=0)

        # If still no eigenvalues, use a uniform tiny spectrum
        if eig_source is None or eig_source.size == 0:
            eig_source = np.ones(self.problem.dim) * 1e-6

        total_var = max(1e-12, float(np.sum(eig_source)))

        # Historical success rate per recommended dimension
        success_counts = {}
        total_counts = {}
        for info in self.previous_model_information:
            d_rec = int(info.get("recommended_dimension", 1))
            total_counts[d_rec] = total_counts.get(d_rec, 0) + 1
            if info.get("model_success"):
                success_counts[d_rec] = success_counts.get(d_rec, 0) + 1

        for d in range(1, max_test_d + 1):
            # validation error (lower is better) -> convert to normalized [0,1] score
            val_err = None
            if validation_by_d and int(d) in validation_by_d:
                try:
                    val_err = float(validation_by_d[int(d)])
                except Exception:
                    val_err = None
            # If absent, set neutral value later

            # projection residual: fraction of variance NOT captured by top-d
            captured = float(np.sum(eig_source[:d]))
            proj_resid = max(0.0, 1.0 - (captured / total_var))

            # success rate
            succ = success_counts.get(d, 0)
            tot = total_counts.get(d, 0)
            success_rate = float(succ / tot) if tot > 0 else 0.0

            results[d] = {
                "val_err": val_err,
                "proj_resid": proj_resid,
                "success_rate": success_rate,
            }

        # Normalize and score
        # Gather val_errs for normalization
        vals = [v["val_err"] for v in results.values() if v["val_err"] is not None]
        if vals:
            max_val = max(vals)
            min_val = min(vals)
        else:
            max_val = None
            min_val = None

        # Weights (can be tuned later)
        w_val = 0.50
        w_proj = 0.30
        w_succ = 0.20
        # cost weight applied as penalty inside score via normalized d

        for d, metrics in results.items():
            # normalized validation score in [0,1] where 1 is best
            if metrics["val_err"] is None:
                s_val = 0.5
            else:
                if max_val is None or max_val - min_val < 1e-12:
                    s_val = 0.8
                else:
                    # smaller err -> larger score
                    s_val = 1.0 - (metrics["val_err"] - min_val) / max(
                        1e-12, (max_val - min_val)
                    )

            s_proj = 1.0 - metrics["proj_resid"]  # higher is better
            s_succ = metrics["success_rate"]

            # cost penalty (prefer smaller d). Use hard-coded budget-aware formula
            # Cost shape and budget multiplier are fixed from tuning experiments.
            cost_power = float(self.cost_power)
            cp = float(self.cost_penalty)
            budget_alpha = float(self.budget_alpha)

            cost = (float(d) / max(1.0, max_test_d)) ** float(cost_power)

            baseline_budget = float(self.baseline_budget)
            budget = getattr(self.problem, "factors", {}).get("budget", baseline_budget)
            try:
                budget = float(budget)
            except Exception:
                budget = baseline_budget
            mult = baseline_budget / max(1.0, budget)
            cp_eff = float(cp) * (float(mult) ** float(budget_alpha))

            raw_score = (
                w_val * s_val + w_proj * s_proj + w_succ * s_succ - cp_eff * cost
            )

            # clamp score to [0,1]
            score = max(0.0, min(1.0, raw_score))
            metrics["score"] = float(score)
            metrics["s_val"] = float(s_val)
            metrics["s_proj"] = float(s_proj)
            metrics["s_succ"] = float(s_succ)
            metrics["cost"] = float(cost)

        # Identify 'sufficient' candidate dimensions: those whose score is
        # within a small relative tolerance of the best score. This allows
        # selecting a smaller d that is effectively as good as the best one.
        try:
            max_score = max(v.get("score", 0.0) for v in results.values())
        except Exception:
            max_score = 0.0

        tol = float(self.sufficient_score_tol)
        if max_score > 0:
            threshold = max_score * (1.0 - float(tol))
        else:
            threshold = max_score - float(tol)

        sufficient = [
            d for d, m in results.items() if float(m.get("score", 0.0)) >= threshold
        ]
        # store last-sufficient list on the solver for downstream use / inspection
        try:
            self._last_sufficient_candidates = sorted(int(d) for d in sufficient)
        except Exception:
            self._last_sufficient_candidates = []
        # annotate individual metrics
        for d in results:
            results[d]["is_sufficient"] = d in self._last_sufficient_candidates
        return results

    def compute_optimal_subspace_dimension(self) -> None:
        """Compute the optimal subspace dimension for the current iteration based on.

        previous model information.

        Returns:
            int: The optimal subspace dimension for the current iteration.
        TODO: Complete the fallback logic for cases with no successful or intersecting
        models
        TODO: Add in logic to handle plateau detection and dimension resets
        """
        # Use scoring across candidate dimensions combining validation, projection
        # residual and success history
        # Plateau and reset logic preserved
        # Conservative safety: require a minimum number of iterations between d-changes
        min_iters_between_d_changes = 3

        if self.in_dimension_reset:
            optimal_d = self.reduce_dimension_in_plateau_reset()
            if optimal_d == self.initial_subspace_dimension:
                self.in_dimension_reset = False
        elif self.detect_plateau():
            optimal_d = self.problem.dim - 1
            self.delta = max(self.delta_initial, 0.25 * self.delta_max)
            self.in_dimension_reset = True
            self.reset_start_iteration = self.iteration_count
            self.reset_target_d = self.initial_subspace_dimension
            self.dimension_reset_count += 1
            self.consecutive_unsuccessful = 0
            self.consecutive_successful = 0
            self.recent_gradient_norms.clear()
            self.recent_prediction_errors.clear()
        else:
            # If no history, keep current d
            if not self.previous_model_information and not hasattr(
                self, "last_validation_by_d"
            ):
                return self.d

            scores = self.evaluate_and_score_candidate_dimensions()
            if not scores:
                return self.d

            # choose best d by score
            best_d = max(scores.items(), key=lambda kv: kv[1].get("score", 0.0))[0]
            optimal_d = int(best_d)

            # Prefer the smallest sufficient candidate when configured
            prefer_small = bool(self.prefer_small_sufficient)
            sufficient = getattr(self, "_last_sufficient_candidates", None)
            if prefer_small and sufficient:
                try:
                    optimal_d = int(min(sufficient))
                except Exception:
                    optimal_d = int(best_d)

            # Write diagnostic summary of candidate scores to diagnostics file (if
            # enabled)
            try:
                if self.factors.get("Record Diagnostics", False) and hasattr(
                    self, "diagnostics"
                ):
                    out = f"\nITER {self.iteration_count}: candidate-d scores and metrics:\n"  # noqa: E501
                    for d, m in sorted(scores.items()):
                        out += f"  d={d}: score={m.get('score', float('nan')):.4f}, s_val={m.get('s_val', float('nan')):.4f}, s_proj={m.get('s_proj', float('nan')):.4f}, s_succ={m.get('s_succ', float('nan')):.4f}, cost={m.get('cost', float('nan')):.4f}\n"  # noqa: E501
                    out += f"  chosen_optimal_d={optimal_d} (current_d={self.d})\n"
                    try:
                        self.diagnostics.write_diagnostics_to_txt(out)
                    except Exception:
                        logging.debug("Failed to write adaptive-d diagnostics")
            except Exception:
                pass

        # Prevent extreme oscillations: require a small cooldown before applying changes
        if not self.in_dimension_reset:
            time_since_change = self.iteration_count - getattr(
                self, "last_d_change_iteration", 0
            )
            if time_since_change < min_iters_between_d_changes:
                # Avoid changing d too frequently
                return self.d

            # Conservative step cap to avoid large jumps
            max_step = 2

            # If multiple consecutive unsuccessful iterations, be slightly more willing
            # to increase d
            if getattr(self, "consecutive_unsuccessful", 0) >= 3:
                max_step = max(max_step, 3)

            # Cap the step to a reasonable fraction of problem dimension
            cap = max(1, int(0.5 * max(1, self.problem.dim)))
            max_step = min(max_step, cap)

            if abs(optimal_d - self.d) > max_step:
                optimal_d = int(self.d + np.sign(optimal_d - self.d) * max_step)

        # If optimal_d equals current dimension, optionally allow small forced
        # exploration
        if (
            optimal_d == self.d
            and (self.iteration_count - getattr(self, "last_d_change_iteration", 0)) > 5
            and self.consecutive_unsuccessful >= 2
        ):
            optimal_d = min(self.problem.dim - 1, self.d + 1)

        # Log dimension change and update history when it happens
        if optimal_d != self.d:
            print(
                f"Iteration {self.iteration_count}: Adaptive subspace dimension change: {self.d} -> {optimal_d}"  # noqa: E501
            )
            logging.info(
                f"Adaptive subspace: {'Increasing' if optimal_d > self.d else 'Decreasing'} "  # noqa: E501
                f"d from {self.d} to {optimal_d} "
            )
            self.d = optimal_d
            # Reset prev_U since dimension changed
            self.prev_U = None
            self.prev_H = np.eye(self.problem.dim)
            # Track when the last change happened and record history
            self.last_d_change_iteration = self.iteration_count
            try:
                self.d_history.append(self.d)
            except Exception:
                # ensure d_history exists
                self.d_history = [self.d]
        return None

    # TODO: EXPAND THIS LOGIC TO INCLUDE PREDICTION ERRORS OF THE MODELS
    def heuristic_for_no_successful_models(self) -> int:
        """Heuristic to determine optimal subspace dimension when no successful models.

        exist.

            This is based on recent unsuccessful performance and prediction errors.

        Returns:
            int: The optimal subspace dimension based on heuristics.
        """
        # Start conservative: default to current dimension
        target_d = int(self.d)

        # Use exponential moving averages for stability
        ema_alpha = 0.3
        if hasattr(self, "ema_pred_error"):
            self.ema_pred_error = (
                ema_alpha
                * (
                    np.abs(self.recent_prediction_errors[-1])
                    if self.recent_prediction_errors
                    else 0.0
                )
                + (1 - ema_alpha) * self.ema_pred_error
            )
        else:
            self.ema_pred_error = (
                np.mean(self.recent_prediction_errors[-3:])
                if len(self.recent_prediction_errors) >= 1
                else 0.0
            )

        if self.consecutive_unsuccessful >= 3:
            target_d = min(self.problem.dim - 1, self.d + 2)
        elif self.consecutive_unsuccessful >= 2:
            target_d = min(self.problem.dim - 1, self.d + 1)

        # Adjust based on smoothed prediction error
        if self.ema_pred_error > 0.5:
            target_d = min(self.problem.dim - 1, target_d + 1)
        elif self.ema_pred_error < 0.1:
            target_d = max(1, target_d - 1)

        # Bound and return
        return max(1, min(self.problem.dim - 1, int(target_d)))

    def heuristic_performance_with_successful_models(
        self, successful_models: list[dict]
    ) -> int:
        """Heuristic to determine optimal subspace dimension based on successful.

        models'.

        performance.


            This considers recent prediction errors and trust-region sizes.

            We consider:
            1. The smallest dimension that has been tried multiple times and led to
            successful results
            2. The dimension that led to the lowest average prediction error among
            successful models
            3. The dimension that aligns with the current trust-region size (smaller
            dimensions for smaller trust-regions)
            4. The dimension that has shown consistent improvement in recent iterations

        Args:
            successful_models (list[dict]): A list of previous model information
            dictionaries that were successful

        Returns:
            int: The optimal subspace dimension based on heuristics.
        """
        candidate_dimensions = []

        # 1. Find the smallest dimension if it has been tried multiple times and lead to
        # successful results
        # filter d_history for successful iterations
        successful_d_history = [
            self.d_history[i - 1]
            for i in self.successful_iterations
            if i - 1 < len(self.d_history)
        ]
        d_counts = Counter(successful_d_history)
        smallest_successful_d = min(
            (d for d, count in d_counts.items() if count >= 2), default=self.d
        )
        candidate_dimensions.append(smallest_successful_d)

        # 2. The dimension that led to the lowest average prediction error among
        # successful models
        # Use per-model stored recommended_dimension when available; use global recent
        # errors as fallback
        d_pred_errors = {}
        for info in successful_models:
            d = info["recommended_dimension"]
            # get the prediction error from the recent_prediction_errors list
            # corresponding to this model
            model_pred_error = (
                np.mean(self.recent_prediction_errors[-5:])
                if self.recent_prediction_errors
                else 0.0
            )
            if d not in d_pred_errors:
                d_pred_errors[d] = []
            d_pred_errors[d].append(model_pred_error)

        avg_d_pred_errors = {d: np.mean(errors) for d, errors in d_pred_errors.items()}
        # select dimension with lowest average prediction error
        if avg_d_pred_errors:
            best_d = min(avg_d_pred_errors, key=avg_d_pred_errors.get)
            candidate_dimensions.append(best_d)

        # 3. The dimension that aligns with the current trust-region size (smaller
        # dimensions for smaller trust-regions)
        if self.delta < self.delta_initial / 4:
            smaller_d = max(1, self.d - 1)
            candidate_dimensions.append(smaller_d)
        elif self.delta > self.delta_initial / 2:
            larger_d = min(self.problem.dim - 1, self.d + 1)
            candidate_dimensions.append(larger_d)

        # 4. The dimension that has shown consistent improvement in recent iterations
        if len(self.successful_iterations) >= 3:
            recent_successful_ds = [
                self.d_history[i - 1]
                for i in self.successful_iterations[-3:]
                if i - 1 < len(self.d_history)
            ]
            if all(d == recent_successful_ds[0] for d in recent_successful_ds):
                candidate_dimensions.append(recent_successful_ds[0])

        # Out of the candidate dimensions, choose the most frequently occurring one
        if not candidate_dimensions:
            return self.d

        dimension_counts = Counter(candidate_dimensions)
        optimal_d, count = dimension_counts.most_common(1)[0]
        # If all candidates are different, pick the median dimension
        if count == 1:
            optimal_d = int(np.median(candidate_dimensions))

        return max(1, min(self.problem.dim - 1, int(optimal_d)))

    def adaptive_dimension_successful_intersecting_models(
        self, intersecting_models: list[dict], intersection_amounts: list[float]
    ) -> int:
        """Determine the optimal subspace dimension based on successful intersecting.

        models.

            We consider three strategies:
            1. Infer optimal dimension from the intersecting models' average eigenvalue
            spectra
            2. Check for consensus among intersecting models' recommended dimensions
            which significantly intersect the current trust-region
            3. Look for patterns in dimension changes among intersecting models.

            We see out of these three candidates which one is most agreed upon, and if
            all are different, we use a scoring system to select the best candidate.
            The scoring system rewards:
            - Higher dimensions when trust-region is small and smaller dimensions when
            trust-region is large
            - Dimensions suggested by models with higher intersection amounts

        Args:
            intersecting_models (list[dict]): A list of previous model information
            dictionaries that were successful
            intersection_amounts (list[float]): A list of intersection amounts with the
            current trust-region corresponding to the successful models

        Returns:
            int: The optimal subspace dimension based on successful intersecting models
        """
        candidates = []
        # we infer what the subspace dimension of the current trust-region should be
        # based on the known optimal subspace dimensions of intersecting models
        candidates.append(self.infer_optimal_dimension_from_models(intersecting_models))

        # obtain list of optimal dimensions from intersecting models sorted by how much
        # they intersect the current trust-region
        intersecting_optimal_ds = [
            info.get("recommended_dimension", self.d) for info in intersecting_models
        ]

        # see if there is some consensus among intersecting models
        # Consider only models with significant intersection (e.g., intersection amount
        # >= 0.5)
        significant_ds = [
            d
            for d, amt in zip(
                intersecting_optimal_ds, intersection_amounts, strict=False
            )
            if amt >= 0.5
        ]
        if significant_ds:
            dim_counts = Counter(significant_ds)
            optimal_d_2, count = dim_counts.most_common(1)[0]
            candidates.append(optimal_d_2)

        # see if there is any pattern that can be extrapolated from the change in
        # dimensions of intersecting models from smallest intersection
        # amount to largest

        # if there is a clear growth or decay pattern in the dimensions of intersecting
        # models, follow that pattern
        if len(intersecting_optimal_ds) >= 3:
            diffs = np.diff(intersecting_optimal_ds)
            if all(d > 0 for d in diffs):  # strictly increasing
                optimal_d_3 = min(self.d + 1, self.problem.dim - 1)
            elif all(d < 0 for d in diffs):  # strictly decreasing
                optimal_d_3 = max(self.d - 1, 1)
            else:
                optimal_d_3 = self.d
            candidates.append(optimal_d_3)

        # out of the candidates, choose the one that occurs most frequently,
        # if all candidates are different design a scoring option to select the best
        # candidate
        candidate_counts = Counter(candidates)
        optimal_d, count = candidate_counts.most_common(1)[0]

        # scoring system if all candidates are different
        if count == 1:
            scores = {}
            for candidate in candidates:
                score = 0
                # Reward closeness to current dimension
                # score -= abs(candidate - self.d)
                # Reward higher dimensions when trust-region is small and smaller
                # diimensions when trust-region is large
                if self.delta < self.delta_initial / 4:
                    score += candidate * 0.5
                else:
                    score -= candidate * 0.5
                # Reward dimensions suggested by models with higher intersection amounts
                for idx, info in enumerate(intersecting_models):
                    try:
                        if info.get("recommended_dimension", None) == candidate:
                            score += intersection_amounts[idx] * 2.0
                    except Exception:
                        continue
                scores[candidate] = score
            # Select candidate with highest score
            optimal_d = max(scores, key=scores.get)
        return optimal_d

    def record_gradient_eigenvalues(self, gradients: np.ndarray) -> None:
        """Compute and store eigenvalues from gradient outer product for dimension.

        estimation.

        Args:
            gradients: Array of shape (M, n) containing M gradient samples in n
            dimensions.
        """
        try:
            if gradients is None:
                return
            G = np.asarray(gradients)  # noqa: N806
            if G.ndim != 2 or G.shape[0] < 2:
                return
            M, _n = G.shape  # noqa: N806
            C = G.T @ G / M  # noqa: N806
            eigenvalues = np.linalg.eigvalsh(C)
            eigenvalues = np.sort(eigenvalues)[::-1]
            eigenvalues = np.maximum(eigenvalues, 0.0)
            if not hasattr(self, "gradient_eigenvalues"):
                self.gradient_eigenvalues = []
            self.gradient_eigenvalues.append(eigenvalues)
        except Exception as e:
            logging.debug(f"record_gradient_eigenvalues failed: {e}")

    # === PLATEAU DETECTION AND DIMENSION RESET METHODS ===
    #!GO THROUGH THIS LOGIC CAREFULLY
    def detect_plateau(self) -> bool:
        """Detect if the solver is in a plateau based on recent performance.

        and initiate a dimension reset if necessary.
        A plateau is defined by a lack of significant improvement in the algorithm for
        an extended perio.

         A plateau is detected only when:
        1. We have enough iteration history (plateau_window iterations)
        2. We're not already in a dimension reset cycle
        3. We haven't exceeded the maximum number of resets
        4. The objective has shown virtually zero improvement over many iterations
        5. Multiple consecutive unsuccessful iterations have occurred

        Returns:
            bool: True if a genuine plateau is detected, False otherwise.
        """
        # Need enough history to detect plateau
        if len(self.recent_objective_values) < self.plateau_window:
            return False

        # Don't trigger another reset if we're already in one
        if self.in_dimension_reset:
            return False

        # Limit the number of resets to avoid excessive dimension cycling
        if self.dimension_reset_count >= self.max_dimension_resets:
            return False

        # Require a minimum number of iterations before considering plateau
        # (relaxed to detect plateaus earlier in short runs)
        min_iterations_before_plateau = max(8, self.plateau_window * 2)
        if self.iteration_count < min_iterations_before_plateau:
            return False

        # Get recent objective values
        recent = self.recent_objective_values[-self.plateau_window :]

        # Compute relative improvement from start to end of window
        # For maximization problems (minmax = (1,)), improvement means increase
        # For minimization problems (minmax = (-1,)), improvement means decrease
        is_maximization = self.problem.minmax[0] == 1

        if is_maximization:
            best_recent = max(recent)
            worst_recent = min(recent)
        else:
            best_recent = min(recent)
            worst_recent = max(recent)

        # Compute relative improvement
        baseline = abs(recent[0]) if abs(recent[0]) > 1e-10 else 1.0
        relative_improvement = abs(best_recent - worst_recent) / baseline

        # Check if we're completely stuck (no change at all)
        no_improvement = all(
            abs(recent[i] - recent[0]) < 1e-10 for i in range(1, len(recent))
        )

        # Additional check: require more consecutive unsuccessful iterations
        # to declare a plateau (strengthened policy).
        require_consecutive_failures = self.consecutive_unsuccessful >= 2

        # Treat extreme stagnation conservatively: require both tiny improvement
        # and at least a couple of failures.
        extreme_stagnation = relative_improvement < (self.plateau_threshold * 0.05)

        # Plateau detected only if:
        # 1. Improvement is below threshold AND we have multiple consecutive failures
        # 2. OR we're completely stuck with no improvement (need stronger evidence)
        # 3. OR extreme stagnation combined with at least 2 failures
        is_plateau = (
            (
                relative_improvement < self.plateau_threshold
                and require_consecutive_failures
            )
            or (no_improvement and self.consecutive_unsuccessful >= 4)
            or (extreme_stagnation and self.consecutive_unsuccessful >= 2)
        )
        if is_plateau:
            logging.info(
                f"Plateau detected at iteration {self.iteration_count}: "
                f"relative_improvement={relative_improvement:.6f} < threshold={self.plateau_threshold}, "  # noqa: E501
                f"consecutive_unsuccessful={self.consecutive_unsuccessful}, "
                f"reset #{self.dimension_reset_count + 1}"
            )

        return is_plateau

    def reduce_dimension_in_plateau_reset(self) -> int:
        """A deterministic decay function to reduce the subspace dimension during a.

        plateau reset cycle.

            The dimension is reduced gradually back to the initial dimension over a set
            number of iterations.

             The decay follows: d(t) = d_max - (d_max - d_init) * log(1 + t) / log(1 +
             T)
             where t is the number of iterations since the reset started, and T is the
             decay duration.

        Returns:
            int: The subspace dimension for the current iteration during the reset
            cycle.
        """
        if not self.in_dimension_reset:
            return self.d

        full_d = self.max_d
        target_d = self.reset_target_d

        # Number of iterations since reset started
        t = self.iteration_count - self.reset_start_iteration

        # Decay duration: scale with dimension difference
        # More iterations for larger dimension gaps
        T = max(3, (full_d - target_d) * 2)  # noqa: N806

        if t >= T:
            # Decay complete - return to normal adaptive mode
            self.in_dimension_reset = False
            logging.info(
                f"Dimension reset complete at iteration {self.iteration_count}: "
                f"returning to d={target_d} and resuming adaptive mode"
            )
            return target_d
        # Logarithmic decay: starts fast, slows down
        # log(1+t)/log(1+T) goes from 0 to 1 as t goes from 0 to T
        decay_fraction = np.log(1 + t) / np.log(1 + T)

        # Dimension decays from full_d toward target_d
        current_d = full_d - int((full_d - target_d) * decay_fraction)

        # Ensure we stay within bounds
        return max(target_d, min(self.max_d, current_d))

    def record_objective_for_plateau_detection(self, objective_value: float) -> None:
        """Append an objective value to the sliding window used by plateau detection.

        Keeps the history bounded to avoid unbounded memory growth.
        """
        try:
            if not hasattr(self, "recent_objective_values"):
                self.recent_objective_values = []
            self.recent_objective_values.append(float(objective_value))
            # Keep a modest amount of history (twice the window) to smooth detection
            max_history = max(10, self.plateau_window * 3)
            if len(self.recent_objective_values) > max_history:
                # pop oldest
                self.recent_objective_values.pop(0)
        except Exception:
            # Don't let diagnostics recording break the solver
            pass

    # === TRUST-REGION METHODS ===

    def compute_trust_region(self, U: np.ndarray) -> tuple[callable, np.ndarray]:  # noqa: N803
        """Constructs the ellipsoidal trust-region based on the Hessian of the previous.

        model.

        Uses the stored prev_H from the previous iteration to define the trust-region
        geometry.

        Args:
            U (np.ndarray): The (n,d) active subspace matrix
        Returns:
            tuple[callable, np.ndarray]:
                - The trust-region constraint function
                - The regularized Hessian matrix H used for the ellipsoid

        """
        if hasattr(self, "prev_H") and self.factors["elliptical trust region"]:
            hess_matrix = self.prev_H.copy()  #! Project to full space
        else:
            hess_matrix = np.eye(self.problem.dim)

        hess_matrix = 0.5 * (hess_matrix + hess_matrix.T)  # Ensure symmetry

        eigvals, eigvecs = np.linalg.eigh(hess_matrix)
        eig_floor = 1e-8
        eigvals = np.maximum(eigvals, eig_floor)
        H_full = eigvecs @ np.diag(eigvals) @ eigvecs.T  # noqa: N806

        def trust_region_constraint(x):  # noqa: ANN001, ANN202
            x = np.asarray(x).reshape(-1, 1)
            if x.shape[0] == U.shape[1]:
                H_reduced = U.T @ H_full @ U  # noqa: N806
                H = 0.5 * (H_reduced + H_reduced.T)  # noqa: N806
            else:
                H = H_full  # noqa: N806
            return (x.T @ H @ x).item()

        return trust_region_constraint, H_full

    def compute_relative_error(
        self, candidate_solution: Solution, fval_tilde: float
    ) -> None:
        """Compute the relative error of the model prediction at the candidate solution.

        Args:
                model: The surrogate model function.
                candidate_solution: The candidate solution to evaluate.
                fval_tilde: The true function value at the candidate solution.
        """
        prediction = self.model(np.array(candidate_solution.x).reshape(1, -1))
        relative_error = (prediction - fval_tilde) / (abs(fval_tilde) + 1e-10)
        # Store relative error for TR expansion dampening (keep last 5)
        self.recent_prediction_errors.append(relative_error)
        if len(self.recent_prediction_errors) > 5:
            self.recent_prediction_errors.pop(0)  # Keep only last 5

    def solve_subproblem(self, U: np.ndarray) -> Solution:  # noqa: N803
        """Solves the trust-region subproblem within the reduced subspace with.

        regularization.

                Objective: min m_k(U@z) + lambda * ||U@z||^2 / ||z||^2
                Constraint: ||x_k + U@z - x_k|| <= delta (full-space trust region).

                The regularization term penalizes small full-space steps for given
                reduced-space steps,
                encouraging solutions that make substantial progress in full space.

                NEW APPROACH: Penalize based on ratio of norms ||U·s|| / ||s|| to
                encourage large full-space steps
                while keeping the optimization stable. The norm is L-1 for better
                numerical behavior.


        Args:
                model (callable): The surrogate model function
                model_grad (callable): The surrogate model gradient function
                U (np.ndarray): The (n,d) active subsapce matrix

        Returns:
                Solution: The candidate solution in the full space
        """
        # Get current solution in full space
        x_current = np.array(self.incumbent_x).reshape(-1, 1)  # shape (n, 1)
        U.T @ x_current  # shape (d, 1)

        # Regularization weight: prevent null-space drift
        # Very small value to avoid interfering with optimization convergence
        lambda0 = float(self.factors.get("subproblem_regularisation", 1e-4))
        delta_safe = max(self.delta, getattr(self, "delta_min", 1e-12), 1e-12)
        # scale so penalty magnitude ~ lambda * delta^2 remains roughly constant
        # use delta_initial as reference so lambda==lambda0 at start
        lambda_reg = lambda0 * (self.delta_initial / delta_safe) ** 2
        # implicit cap using existing delta_min to avoid extreme regularisation
        delta_min_safe = max(getattr(self, "delta_min", 0.0), 1e-12)
        lambda_cap = lambda0 * (self.delta_initial / delta_min_safe) ** 2
        lambda_reg = min(lambda_reg, lambda_cap)

        # Build reduced-space Hessian for TR geometry
        _tr_cons, H_full = self.compute_trust_region(U)  # noqa: N806
        if not np.all(np.isfinite(H_full)):
            H_full = np.eye(self.problem.dim)  # noqa: N806
        H = U.T @ H_full @ U  # Project Hessian to reduced space  # noqa: N806
        H = 0.5 * (H + H.T)  # Ensure symmetry  # noqa: N806
        if not np.all(np.isfinite(H)):
            H = np.eye(U.shape[1])  # noqa: N806
        eigvals, eigvecs = np.linalg.eigh(H)
        eigvals = np.maximum(eigvals, 1e-8)
        H_reduced = eigvecs @ np.diag(eigvals) @ eigvecs.T  # noqa: N806

        def obj_fn(z):  # noqa: ANN001, ANN202
            z_col = np.array(z).reshape(1, -1)  # shape (1, d)
            model_val = float(self.model(z_col))
            if not np.isfinite(model_val):
                # Large positive penalty to steer optimizer away, but finite
                return 1e6

            # Regularization term to penalize the norm of the full-space step
            reduced_space_step_length = norm(z.reshape(-1, 1))
            penalty = (lambda_reg * reduced_space_step_length**2) / 2

            return model_val + penalty

        def obj_grad(z):  # noqa: ANN001, ANN202
            z_col = np.array(z).reshape(1, -1)
            g_red = self.model_grad(z_col, full_space=False).flatten()  # shape (d,)
            if not np.all(np.isfinite(g_red)):
                # Replace with zero or small gradient to avoid NaNs propagating
                g_red = np.zeros_like(g_red)
            # Gradient of the regularization term: lambda * U^T * U * z
            # Since U is orthonormal, U^T @ U is the identity matrix.
            grad_penalty = lambda_reg * np.array(z).flatten()

            return g_red + grad_penalty

        def ellipsoid_constraint_fn(z):  # noqa: ANN001, ANN202
            z = np.array(z).reshape(-1, 1)
            val = (z.T @ H_reduced @ z).flatten().item()
            # The step is z, as we are optimizing from the origin in the reduced space
            if not np.isfinite(val):
                # If constraint blows up, fall back to spherical TR
                return float(z.T @ z).flatten().item()
            return val

        cons = NonlinearConstraint(ellipsoid_constraint_fn, 0, self.delta**2)

        # diagnose flattness
        grad_at_zero = obj_grad(np.zeros(self.d))
        if norm(grad_at_zero) < 1e-6 and self.factors["Record Diagnostics"]:
            warning = (
                "⚠️ WARNING: Model gradient is very small - model may be too flat!\n"
            )
            self.diagnostics.write_diagnostics_to_txt(warning)

        starting_point = np.zeros(self.d).flatten()

        res = minimize(
            obj_fn,
            starting_point,
            method="trust-constr",
            jac=obj_grad,
            constraints=[cons],
            options={"disp": False, "verbose": 0, "xtol": 1e-8, "gtol": 1e-6},
        )
        if (not res.success) or (not np.all(np.isfinite(res.x))):
            # Fallback: small Cauchy-like step along negative reduced gradient
            g0 = obj_grad(np.zeros(self.d))
            if np.all(np.isfinite(g0)) and norm(g0) > 0:
                z = -(self.delta / norm(g0)) * g0.reshape(-1, 1)
            else:
                z = np.zeros((self.d, 1))
        else:
            z = res.x.reshape(-1, 1)

        # z = res.x.reshape(-1, 1)  # shape (d, 1)

        # Project the step back to the full space
        full_space_step = U @ z

        # Add the step to the current incumbent
        s_nominal = (x_current + full_space_step).flatten()

        s_new = [
            clamp_with_epsilon(
                val, self.problem.lower_bounds[j], self.problem.upper_bounds[j]
            )
            for j, val in enumerate(s_nominal.tolist())
        ]
        s_new = np.array(s_new).flatten()

        candidate_solution = self.create_new_solution(tuple(s_new), self.problem)
        self.visited_points.append(candidate_solution)
        return candidate_solution

    def pattern_search(
        self,
        candidate_solution: Solution,
        fval: list[float],
        fval_tilde: float,
        interpolation_solns: list[Solution],
    ) -> tuple[Solution, float]:
        """Perform pattern search around the candidate solution to find a better.

        solution.

        Args:
                candidate_solution (Solution): The candidate solution to be evaluated
                fval (list[float]): The list of objective function values at
                interpolation points
                fval_tilde (float): The predicted objective function value at the
                candidate solution
                interpolation_solns (list[Solution]): The list of interpolation
                solutions
        Returns:
                tuple[Solution, float]: The best solution found and its objective
                function value
        """
        min_fval = min(fval)
        sufficient_reduction = (fval[0] - min_fval) >= self.factors[
            "ps_sufficient_reduction"
        ] * self.delta**2

        condition_met = min_fval < fval_tilde and sufficient_reduction

        high_variance = False
        if not condition_met:
            # Treat variance as low if mean is zero to avoid division by
            # zero (zero mean typically indicates negligible uncertainty)
            if candidate_solution.objectives_mean[0] == 0:
                logging.debug(
                    "Candidate solution objectives_mean is zero, "
                    "skipping variance check."
                )
            else:
                high_variance = (
                    candidate_solution.objectives_var[0]
                    / (
                        candidate_solution.n_reps
                        * candidate_solution.objectives_mean[0] ** 2
                    )
                ) > 0.75
        if condition_met or high_variance:
            fval_tilde = min_fval
            min_idx = fval.index(min_fval)
            candidate_solution = interpolation_solns[min_idx]
        return candidate_solution, fval_tilde

    def evaluate_candidate_solution(
        self,
        fval: list[float],
        fval_tilde: float,
        interpolation_solns: list[Solution],
        candidate_solution: Solution,
        X: np.ndarray,  # noqa: N803
    ) -> None:
        """Evaluates the candidate solution and updates the trust-region radius.

        accordingly.

        Args:
                model (callable): The surrogate model function
                fval (list[float]): The list of objective function values at
                interpolation points
                fval_tilde (float): The predicted objective function value at the
                candidate solution
                interpolation_solns (list[Solution]): The list of interpolation
                solutions
                candidate_solution (Solution): The candidate solution to be evaluated.
                X (np.ndarray): The design matrix.
        """
        # pattern search
        candidate_solution, fval_tilde = self.pattern_search(
            candidate_solution, fval, fval_tilde, interpolation_solns
        )

        # compute ratio
        rho = self.compute_ratio(candidate_solution, fval_tilde)

        # update parameters
        # Check if rho is the sentinel value for cautious acceptance
        cautious_accept = rho == -999.0
        self.update_parameters(
            rho, candidate_solution, X, cautious_accept=cautious_accept
        )

    def update_parameters(
        self,
        rho: float,
        candidate_solution: Solution,
        X: np.ndarray,  # noqa: N803
        cautious_accept: bool = False,
    ) -> None:
        """Update the trust-region radius and current solution based on the ratio rho.

                Also performs adaptive trust region shrinkage based on interpolation
                quality.

        Args:
                rho (float): The ratio of actual reduction to predicted reduction
                candidate_solution (Solution): The candidate solution being considered
                X (np.ndarray): Design points (M, n) for computing interpolation quality
                cautious_accept (bool): If True, accept solution but keep trust-region
                radius unchanged
        Returns:
                tuple[Solution, float]: The updated current solution and trust-region
                radius
        """

        def current_iteration_performance(success):  # noqa: ANN001, ANN202
            return self.record_iteration_performance(
                self.delta, self.incumbent_solution, success
            )

        # Adaptive trust region based on interpolation quality
        # Compute distance from candidate to nearest design point
        x_candidate = np.array(candidate_solution.x).reshape(-1, 1)
        distances_to_design = [
            norm(x_candidate.flatten() - X[i, :]) for i in range(X.shape[0])
        ]
        min_dist_to_design = min(distances_to_design)

        # If candidate is consistently far from design points, shrink trust region MORE
        # aggressively
        if min_dist_to_design > 0.6 * self.delta:
            old_delta = self.delta
            self.delta = max(0.5 * self.delta, self.delta_min)

        if cautious_accept:
            # Accept the solution because it shows actual improvement, but don't change
            # trust region
            # The model is unreliable, so we don't reward with radius increase
            self.incumbent_solution: Solution = candidate_solution
            self.incumbent_x: tuple = candidate_solution.x
            self.fval = (
                -1 * self.problem.minmax[0] * candidate_solution.objectives_mean.item()
            )
            # Lock in the objective value at acceptance time for fn_estimates
            # This prevents the estimate from changing if the solution gets more samples
            # later
            self.locked_incumbent_objective = candidate_solution.objectives_mean.item()

            self.recommended_solns.append(candidate_solution)
            self.successful_iterations.append(candidate_solution)
            self.intermediate_budgets.append(self.budget.used)

            # Track for adaptive subspace dimension
            # self.update_success_tracking(is_successful=True)
            current_iteration_performance(success=True)

            # Keep delta unchanged (no increase or decrease)
            # Optionally: could apply modest shrinkage like: self.delta = max(0.9 *
            # self.delta, self.delta_min)

        elif rho >= self.eta_1:
            self.incumbent_solution: Solution = candidate_solution
            self.incumbent_x: tuple = candidate_solution.x
            self.fval = (
                -1 * self.problem.minmax[0] * candidate_solution.objectives_mean.item()
            )
            # Lock in the objective value at acceptance time for fn_estimates
            # This prevents the estimate from changing if the solution gets more samples
            # later
            self.locked_incumbent_objective = candidate_solution.objectives_mean.item()

            self.recommended_solns.append(candidate_solution)
            self.successful_iterations.append(candidate_solution)
            self.intermediate_budgets.append(self.budget.used)

            # Track for adaptive subspace dimension
            # self.update_success_tracking(is_successful=True)
            current_iteration_performance(success=True)

            old_delta = self.delta

            # Check recent prediction quality to inform TR expansion
            # If recent predictions are poor, be more conservative with expansion
            avg_recent_error = (
                np.mean(self.recent_prediction_errors)
                if len(self.recent_prediction_errors) > 0
                else 0.0
            )

            # Dampen expansion if prediction quality is degrading
            if avg_recent_error > 0.20:  # More than 20% average error
                expansion_factor = 0.8  # Dampen expansion by 20%
            elif avg_recent_error > 0.15:  # More than 15% average error
                expansion_factor = 0.9  # Dampen expansion by 10%
            else:
                expansion_factor = 1.0  # Full expansion allowed

            if rho >= self.eta_2:
                # Very successful: use gamma_1 (larger increase) but dampen if
                # predictions poor
                new_delta = self.gamma_1 * self.delta
                self.delta: float = max(
                    min(
                        old_delta + expansion_factor * (new_delta - old_delta),
                        self.delta_max,
                    ),
                    self.delta_min,
                )
            else:
                # Moderately successful: use gamma_2 (smaller increase) with dampening
                new_delta = self.gamma_2 * self.delta
                self.delta: float = max(
                    min(
                        old_delta + expansion_factor * (new_delta - old_delta),
                        self.delta_max,
                    ),
                    self.delta_min,
                )
                current_iteration_performance(success=False)

        else:
            old_delta = self.delta
            self.delta: float = max(self.gamma_3 * self.delta, self.delta_min)

            self.unsuccessful_iterations.append(self.incumbent_solution)
            # Track for adaptive subspace dimension
            # self.update_success_tracking(is_successful=False)

    #! THIS NEEDS REWRITING
    def compute_ratio(self, candidate_solution: Solution, fval_tilde: float) -> float:
        """Compute the ratio of actual reduction to predicted reduction.

                we produce two values here: rho_effective, which is the ratio used to
                update the trust-region
                radius and force_accept, which indicates whether to force acceptance of
                the candidate solution.
                force_accept is true if the candidate solution shows statistically
                significant improvement over
                the current solution
        Args:
                model (callable): The surrogate model used for prediction
                candidate_solution (Solution): The candidate solution being evaluated
                fval_tilde (float): The predicted objective function value at the
                candidate solution.

        Returns:
                float: The effective ratio used to update the trust-region radius
                                                                        whether the step
                                                                        is
                                                                        feasible
        """
        current_f = (
            -1 * self.problem.minmax[0] * self.incumbent_solution.objectives_mean.item()
        )

        current_m = self.model(np.array(self.incumbent_x).reshape(1, -1))
        candidate_m = self.model(np.array(candidate_solution.x).reshape(1, -1))
        predicted_improvement = current_m - candidate_m
        actual_improvement = current_f - fval_tilde

        # Safeguard against very small predicted improvements
        abs_tolerance = 1e-10 * max(1.0, abs(current_m), abs(candidate_m))

        # Handle different cases
        if predicted_improvement <= abs_tolerance:
            # Model predicts no improvement or worsening
            if actual_improvement > 0:  # noqa: SIM108
                # Actual improvement despite poor model prediction - accept cautiously
                rho = -999.0  # Special sentinel for cautious acceptance
            else:
                # Both model and reality show no improvement
                rho = -1e6  # Reject
        else:
            # Normal case: model predicts improvement
            rho = actual_improvement / predicted_improvement

            # Clamp extreme ratios that might arise from numerical issues
            rho = max(-100.0, min(100.0, rho))

            # Additional check: if we have actual improvement but rho < eta_1,
            # be more lenient if the actual improvement is significant
            if actual_improvement > abs_tolerance and rho < self.eta_1:
                # Check if actual improvement is substantial relative to current value
                relative_improvement = abs(actual_improvement) / max(
                    abs(current_f), 1e-10
                )
                if relative_improvement > 0.001:  # 0.1% relative improvement
                    rho = self.eta_1  # Bump up to minimum acceptance threshold

        return rho

    #! === SAMPLING METHODS ===

    def initial_evaluation(self) -> None:
        """Perform the initial evaluation of the incumbent solution with adaptive.

        sampling.
        """
        # Update pilot run size
        self.calculate_pilot_run()
        # If this is the first iteration, evaluate the incumbent solution
        if self.iteration_count == 1:
            self.incumbent_solution = self.create_new_solution(
                self.incumbent_x, self.problem
            )
            self.visited_points.append(self.incumbent_solution)
            self.incumbent_solution = self.perform_adaptive_sampling(
                self.incumbent_solution, self.pilot_run, self.delta, compute_kappa=True
            )
            self.recommended_solns.append(self.incumbent_solution)
            self.intermediate_budgets.append(self.budget.used)
            # Lock in the initial objective value
            self.locked_incumbent_objective = (
                self.incumbent_solution.objectives_mean.item()
            )
            self.fn_estimates.append(self.locked_incumbent_objective)
            self.budget_history.append(self.budget.used)
            self.iterations.append(self.iteration_count)
            self.record_update += 1
        # if CRN is used across solutions, re-evaluate incumbent at each iteration
        elif self.factors["crn_across_solns"]:
            self.incumbent_solution = self.perform_adaptive_sampling(
                self.incumbent_solution, self.pilot_run, self.delta
            )

    def evaluate_interpolation_points(
        self,
        visited_index: int,
        X: np.ndarray,  # noqa: N803
        delta: float,
    ) -> tuple[np.ndarray, list[Solution]]:
        """Run adaptive sampling on the model construction design points to obtain a.

        sample.

                average of their responses.

        Args:
                visited_index (int): The index of the current solution in the visited
                points list
                X (np.ndarray): The design points for model construction
                delta (float): The trust-region radius.

        Returns:
                tuple[np.ndarray, list[Solution]]:
                                              The array of sample average objective
                                              function values at the design points,
                                              The list of interpolation solutions,
        """
        fX = []  # noqa: N806
        interpolation_solutions = []
        pilot_run = self.calculate_pilot_run(construct_model=True)
        no_pts_sampled = 1
        for idx in range(X.shape[0]):
            # If first iteration, reuse the incumbent solution
            if idx == 0:
                adapt_soln = self.incumbent_solution
            # If the second iteration and we can reuse points, reuse the farthest
            # point from the center point
            elif idx == 1 and norm(
                np.array(self.incumbent_x)
                - np.array(self.visited_points[visited_index].x)
            ):
                adapt_soln = self.visited_points[visited_index]
            # Otherwise, create/initialize a new solution and use that
            else:
                no_pts_sampled += 1
                # get the idx point within the (M,n) matrix X
                decision_vars = tuple(map(float, X[idx]))
                new_solution = self.create_new_solution(decision_vars, self.problem)
                self.visited_points.append(new_solution)
                self.budget.request(pilot_run)
                self.problem.simulate(new_solution, pilot_run)
                adapt_soln = new_solution

            # Don't perform adaptive sampling on x_0
            if not (idx == 0 and self.iteration_count == 0):
                adapt_soln = self.perform_adaptive_sampling(
                    adapt_soln, pilot_run, delta
                )

            fX.append(-1 * self.problem.minmax[0] * adapt_soln.objectives_mean.item())
            interpolation_solutions.append(adapt_soln)

        return np.array(fX).reshape(-1, 1), interpolation_solutions

    def simulate_candidate_soln(
        self, candidate_solution: Solution, delta: float
    ) -> tuple[Solution, float]:
        """Run adaptive sampling on the candidate solution to obtain a sample average.

        of.

        the.

                response to the candidate solution.

        Args:
                candidate_solution (Solution): The candidate solution to be evaluated
                delta (float): The trust-region radius.

        Returns:
                tuple[Solution, float]: The updated candidate solution with simulation
                results,
                The sample average objective function value at the candidate solution,
        """
        if self.factors["crn_across_solns"]:
            num_sims = self.incumbent_solution.n_reps
            self.budget.request(num_sims)
            self.problem.simulate(candidate_solution, num_sims)
        else:
            pilot_run = self.calculate_pilot_run(construct_model=True)
            candidate_solution = self.perform_adaptive_sampling(
                candidate_solution, pilot_run, delta
            )

        fval_tilde = (
            -1 * self.problem.minmax[0] * candidate_solution.objectives_mean.item()
        )

        return candidate_solution, fval_tilde

    def calculate_pilot_run(self, construct_model: bool = False) -> int | None:
        """Calculate the pilot run sample size based on the current iteration number k.

        This matches ASTRO-DF's implementation: uses budget.remaining (not budget.total)
        so that pilot_run shrinks as budget depletes, allowing more iterations.
        """
        lambda_max = self.budget.remaining
        if not construct_model:
            self.pilot_run = ceil(
                max(
                    self.lambda_min * log(10 + self.iteration_count, 10) ** 1.1,
                    min(0.5 * self.d, lambda_max),
                )
                - 1
            )
            return None
        return ceil(
            max(
                self.lambda_min * log(10 + self.iteration_count, 10) ** 1.1,
                min(0.5 * self.d, lambda_max),
            )
            - 1
        )

    def get_stopping_time(
        self,
        pilot_run: int,
        sig2: float,
        delta: float,
        kappa: float,
    ) -> int:
        """Compute the sample size using adaptive stopping based on the optimality gap.

        This matches ASTRO-DF's implementation: pilot_run is passed as a parameter.

        Args:
                pilot_run (int): Number of initial samples used in the pilot run.
                sig2 (float): Estimated variance of the solution.
                delta (float): Optimality gap threshold.
                kappa (float): Constant in the stopping time denominator.
                        If 0, it defaults to 1.

        Returns:
                int: The computed sample size, rounded up to the nearest integer.
        """
        if kappa == 0:
            kappa = 1

        # compute sample size
        raw_sample_size = pilot_run * max(
            1.0, sig2 / (kappa**2 * delta**self.delta_power)
        )
        return ceil(raw_sample_size)

    def perform_adaptive_sampling(
        self,
        solution: Solution,
        pilot_run: int,
        delta: float,
        compute_kappa: bool = False,
    ) -> Solution:
        """Perform adaptive sampling on a solution until the stopping condition is met.

        This matches ASTRO-DF's implementation: pilot_run is passed as a parameter.

        Args:
                solution (Solution): The solution object being sampled.
                pilot_run (int): The number of initial pilot runs.
                delta (float): The current trust-region radius.
                compute_kappa (bool): Whether or not to compute kappa dynamically
                (needed in
                        the first iteration).
        """
        sample_size = solution.n_reps if solution.n_reps > 0 else pilot_run
        lambda_max = self.budget.remaining

        # Initial Simulation (only if needed)
        if solution.n_reps == 0:
            self.budget.request(pilot_run)
            self.problem.simulate(solution, pilot_run)
            sample_size = pilot_run

        while True:
            # Compute variance
            sig2 = solution.objectives_var.item()

            # Compute stopping condition
            kappa: float | None = None
            if compute_kappa:
                rhs_for_kappa = solution.objectives_mean.item()
                kappa = (
                    rhs_for_kappa
                    * np.sqrt(pilot_run)
                    / (delta ** (self.delta_power / 2))
                ).item()

            # Set k to the right kappa
            if kappa is not None:
                k = kappa
            elif self.kappa is not None:
                k = self.kappa
            else:
                # TODO: figure out if we need to raise an error instead
                logging.warning("kappa is not set. Using default value of 0.")
                k = 0
            # Compute stopping time
            stopping = self.get_stopping_time(pilot_run, sig2, delta, k)

            # Stop if conditions are met
            if sample_size >= min(stopping, lambda_max) or self.budget.remaining <= 0:
                if compute_kappa:
                    self.kappa = kappa  # Update kappa only if needed
                break

            # Perform additional simulation
            self.budget.request(1)
            self.problem.simulate(solution, 1)
            sample_size += 1
        return solution

    #! === PRELIMINARY FUNCTIONS ===
    # delta = 10 ** (ceil(log(self.delta_max * 2, 10) - 1) / problem.dim)
    def calculate_max_radius(self) -> float:
        """Calculate the maximum trust-region radius based on the problem's variable.

        bounds and random sampling.

        Returns:
                float: The calculated maximum trust-region radius
        """
        find_next_soln_rng = self.rng_list[1]

        dummy_solns: list[tuple[int, ...]] = []
        for _ in range(1000 * self.problem.dim):
            random_soln = self.problem.get_random_solution(find_next_soln_rng)
            dummy_solns.append(random_soln)
        delta_max_arr: list[float | int] = []
        for i in range(self.problem.dim):
            delta_max_arr += [
                min(
                    max([sol[i] for sol in dummy_solns])
                    - min([sol[i] for sol in dummy_solns]),
                    self.problem.upper_bounds[0] - self.problem.lower_bounds[0],
                )
            ]
        return max(delta_max_arr)

    #! === DESIGN SET CONSTRUCTION ===

    def column_vectors_U(self, index: int, U: np.ndarray) -> np.ndarray:  # noqa: N802, N803
        """Get the index column vector of U. The column vectors are orthonormal basis.

        vectors that span the active subspace.

        Args:
                problem (Problem): The SO problem
                index (int): The index of the column vector
                U (np.ndarray): The active subspace matrix

        Returns:
                np.ndarray: The n-dimensional column vector at the given index
        """
        return U[:, index].reshape(-1, 1)

    def compute_adaptive_interpolation_radius_fraction(self) -> list[float]:
        """Compute the semi-axes lengths of the ellipsoidal trust-region in the active.

        subspace.

                The lengths are proportional to the square roots of the eigenvalues of
                the Hessian of the
                surrogate model projected onto the active subspace.

        Returns:
                list[float]: The list of semi-axes lengths for each dimension in the
                active subspace
        """
        # Trust-region radius (or interpolation radius)

        eps = 1e-8
        diag_tol = 1e-10

        H = (  # noqa: N806
            self.prev_H
            if self.factors["elliptical trust region"]
            else np.eye(self.problem.dim)
        )

        H = 0.5 * (H + H.T)  # noqa: N806
        # --- Check if Hessian is already (numerically) diagonal ---
        off_diag_norm = np.linalg.norm(H - np.diag(np.diag(H)), ord="fro")
        diag_norm = np.linalg.norm(np.diag(np.diag(H)), ord="fro")

        is_diagonal = off_diag_norm <= diag_tol * max(1.0, diag_norm)

        if is_diagonal:
            h = np.diag(H)
        else:
            eigvals, _eigvecs = np.linalg.eigh(H)

            h = eigvals

        h = np.maximum(h, eps)

        alpha = self.delta / np.sqrt(h)

        return alpha.tolist()

    def interpolation_points_without_reuse(self, U: np.ndarray) -> list[np.ndarray]:  # noqa: N803
        """Constructs a 2d+1 interpolation set without reusing points.

        Points placed at adaptively computed radius for optimal coverage of typical
        candidate locations.

        Args:
                U (np.ndarray): The (n,d) active subspace matrix

        Returns:
                [np.array]: A list of 2d+1 n-dimensional design points for interpolation
        """
        x_k = np.array(self.incumbent_x).reshape(-1, 1)
        Y = [x_k]  # noqa: N806
        dup_tol = 1e-8 * self.delta
        lower_bounds = self.problem.lower_bounds
        upper_bounds = self.problem.upper_bounds

        # Adaptively compute interpolation radius based on problem characteristics
        interpolation_radii = self.compute_adaptive_interpolation_radius_fraction()

        for i in range(0, self.d):
            direction = self.column_vectors_U(i, U)
            plus = Y[0] + interpolation_radii[i] * direction
            minus = Y[0] - interpolation_radii[i] * direction

            plus = plus.flatten().tolist()
            minus = minus.flatten().tolist()

            if sum(x_k) != 0:
                minus = [
                    clamp_with_epsilon(val, lower_bounds[j], upper_bounds[j])
                    for j, val in enumerate(minus)
                ]
                if degeneration_check(minus, x_k, dup_tol) or duplication_check(
                    minus, Y, dup_tol
                ):
                    minus = backoff_step(
                        x_k,
                        direction,
                        interpolation_radii[i],
                        lower_bounds,
                        upper_bounds,
                    )

                plus = [
                    clamp_with_epsilon(val, lower_bounds[j], upper_bounds[j])
                    for j, val in enumerate(plus)
                ]
                if degeneration_check(plus, x_k, dup_tol) or duplication_check(
                    plus, Y, dup_tol
                ):
                    plus = backoff_step(
                        x_k,
                        direction,
                        interpolation_radii[i],
                        lower_bounds,
                        upper_bounds,
                    )

            if minus is not None:
                minus = np.array(minus).reshape(-1, 1)
                Y.append(minus)
            if plus is not None:
                plus = np.array(plus).reshape(-1, 1)
                Y.append(plus)

        return Y

    # generate the mutually orthonormal rotated basis using A_k1 as the first basis
    # vector
    def get_rotated_basis(self, A_k1: np.ndarray, U: np.ndarray) -> list[np.ndarray]:  # noqa: ARG002, N803
        """Generate the other d-1 rotated coordinate basis using A_k1 as the first.

        basis.

        vector.

        We use Gram-Schmidt process to generate the orthonormal basis.

        Args:
                A_k1 (np.ndarray): The first direction vector for the reused design
                point
                d (int): The subspace dimension and the number of vectors to have
                U (np.ndarray): The (n,d) active subspace matrix

        Returns:
                list[np.ndarray]: A list of d d-dimensional rotated basis vectors each
                with shape (d,1)
        """
        # Start with A_normalized as first vector
        basis = [A_k1]

        # Generate candidate vectors from the FULL standard basis (not just indices 1 to
        # d-1)
        # This ensures we have enough candidates even if some are nearly parallel to
        # A_k1
        I = np.eye(self.d)  # noqa: E741, N806
        candidates = [I[:, i].reshape(-1, 1) for i in range(self.d)]

        # Build successive orthonormal basis using Gram-Schmidt process from A_k1
        for c in candidates:
            if len(basis) >= self.d:
                break

            v = c.copy()
            # calculate gram-schmidt projection
            for b in basis:
                dot_prod = v.T @ b
                v -= dot_prod.item() * b

            v_norm = np.linalg.norm(v)
            if v_norm < 1e-12:
                continue  # skip degenerate direction

            # Normalize v
            v = v / v_norm
            basis.append(v.reshape(-1, 1))

        # Safety check: if we still don't have d vectors (shouldn't happen with full
        # candidates),
        # pad with random orthogonal vectors
        while len(basis) < self.d:
            logging.warning(
                f"get_rotated_basis: Could only generate {len(basis)} of {self.d} basis vectors. "  # noqa: E501
                f"Padding with random vectors."
            )
            # Generate a random vector and orthogonalize
            rand_v = np.random.randn(self.d, 1)
            for b in basis:
                dot_prod = rand_v.T @ b
                rand_v -= dot_prod.item() * b
            v_norm = np.linalg.norm(rand_v)
            if v_norm > 1e-12:
                basis.append((rand_v / v_norm).reshape(-1, 1))
            else:
                # Extremely rare edge case - use coordinate vector with small
                # perturbation
                rand_v = np.random.randn(self.d, 1) * 0.01
                rand_v[len(basis) % self.d] += 1.0
                for b in basis:
                    dot_prod = rand_v.T @ b
                    rand_v -= dot_prod.item() * b
                v_norm = np.linalg.norm(rand_v)
                if v_norm > 1e-14:
                    basis.append((rand_v / v_norm).reshape(-1, 1))

        return basis

    # compute the interpolation points (2d+1) using the rotated coordinate basis (reuse
    # one design point)
    def interpolation_points_with_reuse(
        self,
        reused_x: np.ndarray,
        rotation_vectors: list[np.ndarray],
        U: np.ndarray,  # noqa: N803
    ) -> list[np.ndarray]:
        """Constructs a 2d+1 interpolation set with reusing one design point.

                Points placed at adaptively computed radius for optimal coverage of
                typical candidate locations.

        Args:
                reused_x (np.ndarray): The design point to be reused
                rotation_vectors (list[np.ndarray]): The rotated coordinate basis
                vectors
                U (np.ndarray): The (n,d) active subspace matrix

        Returns:
                list[np.ndarray]: A list of 2d+1 n-dimensional design points for
                interpolation
        """
        x_k = np.array(self.incumbent_x).reshape(-1, 1)
        Y = [x_k, reused_x]  # noqa: N806
        dup_tol = 1e-8 * self.delta
        lower_bounds = self.problem.lower_bounds
        upper_bounds = self.problem.upper_bounds

        # Safety check: ensure we have enough rotation vectors
        if len(rotation_vectors) < self.d:
            logging.error(
                f"interpolation_points_with_reuse: Expected {self.d} rotation vectors, "
                f"got {len(rotation_vectors)}. Falling back to interpolation without reuse."  # noqa: E501
            )
            return self.interpolation_points_without_reuse(U)

        # Adaptively compute interpolation radius based on problem characteristics
        interpolation_radii = self.compute_adaptive_interpolation_radius_fraction()

        for i in range(1, self.d):
            direction = U @ rotation_vectors[i]
            plus = Y[0] + interpolation_radii[i] * direction
            plus = plus.flatten().tolist()

            # block constraints
            if sum(x_k) != 0:
                plus = [
                    clamp_with_epsilon(val, lower_bounds[j], upper_bounds[j])
                    for j, val in enumerate(plus)
                ]
                if degeneration_check(plus, x_k, dup_tol) or duplication_check(
                    plus, Y, dup_tol
                ):
                    plus = backoff_step(
                        x_k,
                        direction,
                        interpolation_radii[i],
                        lower_bounds,
                        upper_bounds,
                    )

            if plus is not None:
                plus = np.array(plus).reshape(-1, 1)
                Y.append(plus)

        for i in range(self.d):
            direction = U @ rotation_vectors[i]
            minus = Y[0] - interpolation_radii[i] * direction
            minus = minus.flatten().tolist()

            # block constraints
            if sum(x_k) != 0:
                minus = [
                    clamp_with_epsilon(val, lower_bounds[j], upper_bounds[j])
                    for j, val in enumerate(minus)
                ]
                if degeneration_check(minus, x_k, dup_tol) or duplication_check(
                    minus, Y, dup_tol
                ):
                    minus = backoff_step(
                        x_k,
                        direction,
                        interpolation_radii[i],
                        lower_bounds,
                        upper_bounds,
                    )

            if minus is not None:
                minus = np.array(minus).reshape(-1, 1)
                Y.append(minus)

        return Y

    def construct_interpolation_set(
        self,
        U: np.ndarray,  # noqa: N803
    ) -> tuple[list[np.ndarray], int]:
        """Constructs the interpolation set either by reusing one design point from the.

        visited points list or not reusing any design points.

                This is the only method that is called to build the interpolation set.

        Args:
                U (np.ndarray): The (n,d) active subspace matrix

        Returns:
                tuple[list[np.ndarray], int]: A tuple containing the list of
                interpolation points and the index of the reused point
        """
        x_k = np.array(self.incumbent_x).reshape(
            -1, 1
        )  # current solution as n-dim vector

        # Find best reuse candidate using full-space trust region check
        # but scoring by projected distance for good poisedness
        f_index, reuse_possible = self._find_best_reuse_candidate(U, x_k)

        # If it is the first iteration or there is no design point we can reuse within
        # the trust region, use the coordinate basis
        if (self.iteration_count == 1) or not reuse_possible:
            Y = self.interpolation_points_without_reuse(U)  # noqa: N806

        # Else if we will reuse one design point
        else:
            reused_pt = np.array(self.visited_points[f_index].x).reshape(-1, 1)
            diff_array = U.T @ (reused_pt - x_k)  # has shape (d,1)
            A_k1 = (diff_array) / norm(diff_array)  # has shape (d,1)  # noqa: N806

            rotate_matrix: list[np.ndarray] = self.get_rotated_basis(A_k1, U)

            # construct the interpolation set
            Y = self.interpolation_points_with_reuse(reused_pt, rotate_matrix, U)  # noqa: N806
        return np.vstack([v.ravel() for v in Y]), f_index

    def _find_best_reuse_candidate(
        self,
        U: np.ndarray,  # noqa: N803
        x_k: np.ndarray,
    ) -> tuple[int, bool]:
        """Find the best candidate for point reuse using full-space trust region check.

        This approach checks distances in FULL SPACE (allowing more point reuse across
        iterations with different subspaces U), but scores candidates by their PROJECTED
        distance to ensure good poisedness in the reduced space.

        Key insight: A point within the full-space trust region ||x_i - x_k|| <= delta
        may project to different locations in different subspaces. By checking full-
        space
        first, we allow reusing points that were evaluated in previous iterations with
        different U matrices. We then score by projected distance to ensure the reused
        point contributes to a well-poised design set.

        For poisedness: A point that projects far from the center in the current
        subspace
        provides better geometry than one that projects near the center, as it helps
        span the polynomial basis more effectively.

        Args:
            U: The (n, d) active subspace matrix for current iteration
            x_k: Current incumbent solution as (n, 1) vector

        Returns:
            tuple[int, bool]: (index of best candidate, whether reuse is possible)
        """
        if len(self.visited_points) == 0:
            return 0, False

        candidates = []  # List of (index, full_space_dist, projected_dist)

        # Use ellipsoidal or spherical trust region in full space
        H_full = (  # noqa: N806
            self.prev_H
            if self.factors["elliptical trust region"]
            else np.eye(self.problem.dim).reshape(self.problem.dim, self.problem.dim)
        )
        H_full = 0.5 * (H_full + H_full.T)  # Ensure symmetry  # noqa: N806

        # Minimum projected distance threshold to avoid near-center points
        # (which would give poor poisedness). Using 0.45*delta balances:
        # - Reuse rate (~11%): Enough points pass to save function evaluations
        # - Poisedness quality (98% with Λ ratio ≤ 1.5): Design sets remain
        # well-conditioned
        # Monte Carlo testing showed: 0.1δ → 32% pass, 0.45δ → 98% pass, 0.5δ → 100%
        # pass
        min_projected_dist = 0.45 * self.delta

        for i in range(len(self.visited_points)):
            x_i = np.array(self.visited_points[i].x).reshape(-1, 1)
            diff_full = x_i - x_k

            # FULL-SPACE trust region check (key change from original)
            # This allows reusing points regardless of previous subspace orientation
            full_space_dist_sq = (diff_full.T @ H_full @ diff_full).item()

            if full_space_dist_sq > self.delta**2:
                continue  # Point outside full-space trust region

            full_space_dist = np.sqrt(full_space_dist_sq)

            # Skip the incumbent itself (distance ~0)
            # if full_space_dist < 1e-10 * self.delta:
            #     continue
            if np.sqrt(full_space_dist) < 1e-14:
                continue

            # Compute projected distance in current subspace
            # This determines how well the point contributes to poisedness
            y_i = U.T @ diff_full  # Project to reduced space
            projected_dist = (norm(y_i)).item()

            # Only consider points that project reasonably far from center
            # (ensures good geometry in reduced space)
            if projected_dist >= min_projected_dist:
                candidates.append((i, full_space_dist, projected_dist))

        if not candidates:
            return 0, False

        # Score candidates: prefer points with larger PROJECTED distance
        # (better poisedness) while being within full-space TR
        #
        # We use projected distance as primary criterion because:
        # 1. It directly affects poisedness of the design set
        # 2. Points far in projected space provide better basis spanning
        # 3. The full-space check already ensures we're within trust region
        best_candidate = max(candidates, key=lambda x: x[2])  # x[2] = projected_dist

        return best_candidate[0], True

    #! === GEOMETRY IMPROVEMENT ===
    def generate_set(self, num: int, delta: float | None = None) -> np.ndarray:
        """Generates a set of points around the current solution within the trust.

        region.

        Args:
                num (int): The number of points to generate
                delta (float, optional): The trust-region radius. Defaults to None.

        Returns:
                np.ndarray: A set of points around the current solution within the trust
                region
        """
        if delta is None:
            delta = self.delta

        x_k = np.array(self.incumbent_x).reshape(-1, 1)

        bounds_l = np.maximum(
            np.array(self.problem.lower_bounds).reshape(x_k.shape), x_k - delta
        )
        bounds_u = np.minimum(
            np.array(self.problem.upper_bounds).reshape(x_k.shape), x_k + delta
        )
        direcs = self.coordinate_directions(num, bounds_l - x_k, bounds_u - x_k)

        S = np.zeros((num, self.problem.dim))  # noqa: N806
        S[0, :] = x_k.flatten()
        bounds_l_flat = bounds_l.flatten()
        bounds_u_flat = bounds_u.flatten()
        x_k_flat = x_k.flatten()
        for i in range(1, num):
            S[i, :] = x_k_flat + np.minimum(
                np.maximum(bounds_l_flat - x_k_flat, direcs[i, :]),
                bounds_u_flat - x_k_flat,
            )

        return S  # shape (num, n)

    def get_scale(
        self,
        dirn: list[float],
        lower: np.ndarray,
        upper: np.ndarray,
        scale: float | None = None,
    ) -> float:
        """Calculates the scaling factor for a direction vector to ensure it stays.

        within bounds.

        Args:
                dirn (list[float]): The direction vector
                lower (np.ndarray): The lower bounds
                upper (np.ndarray): The upper bounds
                scale (float, optional): An initial scaling factor. Defaults to None.

        Returns:
                float: The scaling factor
        """
        scale = self.delta if scale is None else scale
        for j in range(len(dirn)):
            if dirn[j] < 0.0:
                scale = min(scale, lower[j] / dirn[j])
            elif dirn[j] > 0.0:
                scale = min(scale, upper[j] / dirn[j])
        return scale

    def coordinate_directions(
        self, num_pnts: int, lower: np.ndarray, upper: np.ndarray
    ) -> np.ndarray:
        """Generates coordinate directions for the given problem.

        Args:
                num_pnts (int): The number of points to generate
                lower (np.ndarray): The lower bounds
                upper (np.ndarray): The upper bounds
        Returns:
                np.ndarray: Coordinate directions within the trust region.
        """
        n = self.problem.dim
        at_lower_boundary = lower > -1.0e-8 * self.delta
        at_upper_boundary = upper < 1.0e-8 * self.delta
        direcs = np.zeros((num_pnts, n))
        for i in range(1, num_pnts):
            if 1 <= i < n + 1:
                dirn = i - 1
                step = self.delta if not at_upper_boundary[dirn] else -self.delta
                direcs[i, dirn] = step
            elif n + 1 <= i < 2 * n + 1:
                dirn = i - n - 1
                step = -self.delta
                if at_lower_boundary[dirn]:
                    step = min(2.0 * self.delta, upper[dirn])
                if at_upper_boundary[dirn]:
                    step = max(-2.0 * self.delta, lower[dirn])
                direcs[i, dirn] = step
            else:
                itemp = (i - n - 1) // n
                q = i - itemp * n - n
                p = q + itemp
                if p > n:
                    p, q = q, p - n
                direcs[i, p - 1] = direcs[p, p - 1]
                direcs[i, q - 1] = direcs[q, q - 1]
        return direcs  # shape (num_pnts, n)

    def random_directions(
        self, num_pnts: int, lower: np.ndarray, upper: np.ndarray
    ) -> np.ndarray:
        """Generates random directions for the given problem.

        Args:
                num_pnts (int): The number of points to generate
                lower (np.ndarray): The lower bounds of
                upper (np.ndarray): The upper bounds
                delta (float): The current trust-region radius.

        Returns:
                np.ndarray: Random directions within the trust region
        """
        n = self.problem.dim
        direcs = np.zeros((n, max(2 * n + 1, num_pnts)))
        idx_l = lower == 0
        idx_u = upper == 0
        active = np.logical_or(idx_l, idx_u)
        inactive = np.logical_not(active)
        nactive = np.sum(active)
        ninactive = n - nactive
        if ninactive > 0:
            A = np.random.normal(size=(ninactive, ninactive))  # noqa: N806
            Qred = qr(A)[0]  # noqa: N806
            Q = np.zeros((n, ninactive))  # noqa: N806
            Q[inactive, :] = Qred
            for i in range(ninactive):
                scale = self.get_scale(Q[:, i], lower, upper)
                direcs[:, i] = scale * Q[:, i]
                scale = self.get_scale(-Q[:, i], lower, upper)
                direcs[:, n + i] = -scale * Q[:, i]
        idx_active = np.where(active)[0]
        for i in range(nactive):
            idx = idx_active[i]
            direcs[idx, ninactive + i] = 1.0 if idx_l[idx] else -1.0
            direcs[:, ninactive + i] = (
                self.get_scale(direcs[:, ninactive + i], lower, upper)
                * direcs[:, ninactive + i]
            )
            sign = 1.0 if idx_l[idx] else -1.0
            if upper[idx] - lower[idx] > self.delta:
                direcs[idx, n + ninactive + i] = 2.0 * sign * self.delta
            else:
                direcs[idx, n + ninactive + i] = 0.5 * sign * (upper[idx] - lower[idx])
            direcs[:, n + ninactive + i] = (
                self.get_scale(direcs[:, n + ninactive + i], lower, upper, 1.0)
                * direcs[:, n + ninactive + i]
            )
        for i in range(num_pnts - 2 * n):
            dirn = np.random.normal(size=(n,))
            for j in range(nactive):
                idx = idx_active[j]
                sign = 1.0 if idx_l[idx] else -1.0
                if dirn[idx] * sign < 0.0:
                    dirn[idx] *= -1.0
            dirn = dirn / norm(dirn)
            scale = self.get_scale(dirn, lower, upper)
            direcs[:, 2 * n + i] = dirn * scale
        return np.vstack((np.zeros(n), direcs[:, :num_pnts].T))  # shape (num_pnts, n)

    def improve_geometry(
        self,
        delta: float,
        U: np.ndarray,  # noqa: N803
        X: np.ndarray,  # noqa: N803
        fX: np.ndarray,  # noqa: N803
        interpolation_solutions: list[int],
    ) -> tuple[np.ndarray, np.ndarray, list[Solution]]:
        """Improves the geometry of the interpolation set by generating a sample set.

        and.

        performing LU pivoting.

                Works on the projected design set X @ U but returns the original design
                set X.

        Args:
                delta (float): The current trust-region radius
                U (np.ndarray): The current active subspace matrix (shape (n,d))
                X (np.ndarray): The current interpolation points (shape (M, n))
                fX (np.ndarray): The function values at the interpolation points (shape
                (M, 1))
                interpolation_solutions (list[Solution]): The list of solutions in the
                interpolation set

        Returns:
                tuple[np.ndarray, np.ndarray, list[Solution]]:
                                                                Updated interpolation
                                                                points of shape (M, n),
                                                                function values of shape
                                                                (M, 1),,
                                                                interpolation solutions,
        """
        epsilon_1 = 0.01
        dist = epsilon_1 * delta
        x_k = np.array(self.incumbent_x).reshape(-1, 1)

        # Project X to subspace for geometry check
        X_projected = X @ U  # shape (M, d)  # noqa: N806
        x_k_projected = x_k.T @ U  # shape (1, d)

        if max(norm(X_projected - x_k_projected, axis=1, ord=2)) > dist:
            X, fX, interpolation_solutions = self.sample_set(  # noqa: N806
                delta, U, X, fX, interpolation_solutions
            )

        return X, fX, interpolation_solutions

    def sample_set(
        self,
        delta: float,
        U: np.ndarray,  # noqa: N803
        X: np.ndarray,  # noqa: N803
        fX: np.ndarray,  # noqa: N803
        interpolation_solutions: list[Solution],
    ) -> tuple[np.ndarray, np.ndarray, list[Solution]]:
        """Improves the current design set X using LU pivoting to identify and replace.

                ill-posed points with better alternatives while keeping well-posed
                points.

        Args:
                delta (float): The current trust-region radius
                U (np.ndarray): The current active subspace matrix (shape (n,d))
                X (np.ndarray): The current interpolation points (shape (M, n))
                fX (np.ndarray): The function values at the interpolation points (shape
                (M, 1))
                interpolation_solutions (list[Solution]): The list of solutions in the
                interpolation set

        Returns:
                tuple[np.ndarray, np.ndarray, list[Solution]]:
                                                                                                                Updated
                                                                                                                interpolation
                                                                                                                points
                                                                                                                of
                                                                                                                shape
                                                                                                                (M,
                                                                                                                n),
                                                                                                                function
                                                                                                                values
                                                                                                                of
                                                                                                                shape
                                                                                                                (M,
                                                                                                                1),
                                                                                                                interpolation
                                                                                                                solutions
        """
        epsilon_1 = 0.5
        d = U.shape[1]

        x_k = np.array(self.incumbent_x).reshape(-1, 1)
        current_f = (
            -1 * self.problem.minmax[0] * self.incumbent_solution.objectives_mean.item()
        )
        dist = epsilon_1 * delta

        # Start with existing design set as candidates
        X_candidates = np.copy(X)  # noqa: N806
        fX_candidates = np.copy(fX)  # noqa: N806

        # Filter existing points: keep only those within the current trust region
        mask = norm(X_candidates - x_k.ravel(), axis=1, ord=2) <= delta
        X_candidates = X_candidates[mask]  # noqa: N806
        fX_candidates = fX_candidates[mask]  # noqa: N806

        # Remove furthest point if it's too far (ill-posed)
        X_candidates_projected = X_candidates @ U  # noqa: N806
        x_k_projected = x_k.T @ U
        if (
            X_candidates.shape[0] > 0
            and max(norm(X_candidates_projected - x_k_projected, axis=1, ord=2)) > dist
        ):
            X_candidates, fX_candidates = self.remove_furthest_point_projected(  # noqa: N806
                X_candidates, fX_candidates, x_k, U
            )

        # Remove center point from candidates (will be added as X_new[0])
        X_candidates, fX_candidates = self.remove_point_from_set(  # noqa: N806
            X_candidates, fX_candidates, x_k
        )

        # Generate additional well-conditioned candidate points to provide alternatives
        num_additional = max(10, 2 * d + 1)
        X_additional = self.generate_set_in_subspace(delta, U, num_additional)  # noqa: N806
        # Remove center from additional candidates
        X_additional, _ = self.remove_point_from_set(  # noqa: N806
            X_additional, np.zeros((X_additional.shape[0], 1)), x_k
        )

        # Filter ALL candidates to ensure they're within trust region in projected space
        X_additional_projected = X_additional @ U  # noqa: N806
        mask = norm(X_additional_projected - x_k_projected, axis=1, ord=2) <= delta
        X_additional = X_additional[mask]  # noqa: N806
        if X_additional.shape[0] == 0:  # Fallback if all filtered out
            X_additional = self.generate_set_in_subspace(delta, U, num_additional)  # noqa: N806

        # Similarly validate existing candidates are within trust region
        X_candidates_projected = X_candidates @ U  # noqa: N806
        mask = norm(X_candidates_projected - x_k_projected, axis=1, ord=2) <= delta
        X_candidates = X_candidates[mask]  # noqa: N806
        fX_candidates = fX_candidates[mask]  # noqa: N806

        # Combine existing points with additional candidates
        if X_additional.shape[0] > 0:
            X_all_candidates = np.vstack([X_candidates, X_additional])  # noqa: N806
            fX_all_candidates = np.vstack(  # noqa: N806
                [fX_candidates, np.zeros((X_additional.shape[0], 1))]
            )
        else:
            X_all_candidates = X_candidates  # noqa: N806
            fX_all_candidates = fX_candidates  # noqa: N806

        # Build improved design set of size 2d+1 using LU pivoting
        m = 2 * d + 1
        X_new = np.zeros((m, self.problem.dim))  # noqa: N806
        fX_new = np.zeros((m, 1))  # noqa: N806
        X_new[0, :] = x_k.flatten()
        fX_new[0, :] = current_f

        # LU pivoting will select best m-1 points from candidates
        X_new, fX_new, interpolation_solutions = self.LU_pivoting(  # noqa: N806
            delta,
            X_new,
            fX_new,
            X_all_candidates,
            fX_all_candidates,
            U,
            interpolation_solutions,
        )

        return X_new, fX_new, interpolation_solutions

    def LU_pivoting(  # noqa: N802
        self,
        delta: float,
        X: np.ndarray,  # noqa: N803
        fX: np.ndarray,  # noqa: N803
        X_improved: np.ndarray,  # noqa: N803
        fX_improved: np.ndarray,  # noqa: N803
        U: np.ndarray,  # noqa: N803
        interpolation_solutions: list[Solution],
    ) -> tuple[np.ndarray, np.ndarray, list[Solution]]:
        """Improves the interpolation set using LU pivoting.

        Args:
                delta (float): The current trust-region radius
                X (np.ndarray): The current interpolation points (shape (M, n))
                fX (np.ndarray): The function values at the interpolation points (shape
                (M, 1))
                X_improved (np.ndarray): The current sample set (shape (M, n))
                fX_improved (np.ndarray): The function values at the sample set points
                (shape (M, 1))
                U (np.ndarray): The current active subspace matrix (shape (n, d))
                interpolation_solutions (list[Solution]): The list of solutions in the
                interpolation set

        Returns:
                tuple[np.ndarray, np.ndarray, list[Solution]]:
                                                                Updated interpolation
                                                                points of shape (M, n),
                                                                function values of shape
                                                                (M, 1),
                                                                interpolation solutions,
        """
        # Less aggressive pivot thresholds - prefer reusing good existing points
        psi_1 = 0.01  # Accept pivots >= 0.01
        psi_2 = 0.1  # Last pivot >= 0.1
        x_k = np.array(self.incumbent_x).reshape(-1, 1)
        x_projected_k = U.T @ x_k  # shape (d, 1)

        phi_function, phi_function_deriv = self.get_phi_function_and_derivative(
            U, delta
        )
        q = len(self.index_set(self.degree, self.d).astype(int))
        p = X.shape[0]  # Target number of points (2d+1)

        # Initialise R matrix of LU factorisation of M matrix (see Conn et al.)
        R = np.zeros((p, q))  # noqa: N806
        R[0, :] = phi_function(x_k)

        # We'll only perform the LU-style pivot construction for at most q rows
        max_k = min(p, q)

        # Perform the LU factorisation algorithm for the rest of the points (only up to
        # q-1)
        for k in range(1, max_k):
            flag = True
            v = np.zeros(q)
            # Only indices j < k exist in R's columns because k < q here
            for j in range(k):
                v[j] = -R[j, k] / R[j, j]
            v[k] = 1.0

            # If there are still points to choose from, find if points meet criterion.
            # If so, use the index to choose
            # point with given index to be next point in regression/interpolation set
            if fX_improved.size > 0:
                Phi_X_improved = np.vstack(  # noqa: N806
                    [
                        phi_function(X_improved[i, :].reshape(-1, 1))
                        for i in range(X_improved.shape[0])
                    ]
                )
                M = np.absolute(Phi_X_improved @ v)  # noqa: N806
                index = np.argmax(M)
                # Pivot acceptance: use psi_1 normally; require psi_2 for the last pivot
                # (k == q-1)
                if M[index] < psi_1 or (k == q - 1 and M[index] < psi_2):
                    flag = False
            else:
                flag = False

            # If index exists, choose the point with that index and delete it from
            # possible choices
            if flag:
                x = X_improved[index, :]
                X[k, :] = x
                fX[k, :] = fX_improved[index]
                X_improved = np.delete(X_improved, index, 0)  # noqa: N806
                fX_improved = np.delete(fX_improved, index, 0)  # noqa: N806

            # If index doesn't exist, solve an optimisation problem to find the point in
            # the range which best satisfies criterion
            else:
                x = None
                try:
                    x_candidate = self.find_new_point(
                        delta, v, phi_function, phi_function_deriv
                    )

                    # Check for duplicates in projected space with tolerance
                    x_proj = U.T @ x_candidate.reshape(-1, 1)  # shape (d, 1)

                    # Points should be separated by at least 1% of delta in projected
                    # space
                    if norm(x_proj - x_projected_k, ord=2) <= delta:
                        x = x_candidate

                except Exception:  # If optimisation fails, try alternative method
                    try:
                        x_candidate = self.find_new_point_alternative(
                            delta, v, phi_function, X[:k, :], U
                        )

                        x_proj = U.T @ x_candidate.reshape(-1, 1)  # shape (d, 1)

                        if norm(x_proj - x_projected_k, ord=2) <= delta:
                            x = x_candidate
                    except Exception:  # Worst case, sample a random point within the trust region  # noqa: E501
                        if fX_improved.size > 0:
                            x = X_improved[index, :]
                        else:
                            random_dir = np.random.normal(size=(self.problem.dim, 1))
                            random_dir = random_dir / norm(random_dir) * delta * 0.95
                            x = x_k.flatten() + random_dir.flatten()

                # ensure new point is within trust-region -> if not, use existing point
                # if available
                if norm(x.reshape(-1, 1) - x_k, ord=2) > delta:  # noqa: SIM102
                    if fX_improved.size > 0:
                        x = X_improved[index, :]

                # Compare generated point with current best candidate -> if outside
                # tolerance, use existing point if available
                if fX_improved.size > 0 and M[index] >= abs(np.dot(v, phi_function(x))):
                    x = X_improved[index, :]
                    X[k, :] = x
                    fX[k, :] = fX_improved[index]
                    X_improved = np.delete(X_improved, index, 0)  # noqa: N806
                    fX_improved = np.delete(fX_improved, index, 0)  # noqa: N806

                else:
                    x_proj = U.T @ x.reshape(-1, 1)  # shape (d, 1)
                    # for new best point, check if it's in the trust-region (projected)
                    if norm(x_proj - x_projected_k, ord=2) > delta:
                        direction = x_proj - x_projected_k
                        scale = delta / norm(direction, ord=2) * 0.99
                        x_proj = x_projected_k + direction * scale
                        x = (U @ x_proj).ravel()

                    X[k, :] = x
                    soln_at_x = self.create_new_solution(tuple(x.ravel()), self.problem)
                    # Sample the newly generated interpolation point without
                    # re-evaluating the full design set
                    pilot_run = self.calculate_pilot_run(construct_model=True)
                    soln_at_x = self.perform_adaptive_sampling(
                        soln_at_x, pilot_run, delta
                    )
                    f_value = (
                        -1 * self.problem.minmax[0] * soln_at_x.objectives_mean.item()
                    )
                    fX[k, 0] = f_value
                    if all(
                        tuple(pt.x) != tuple(soln_at_x.x) for pt in self.visited_points
                    ):
                        self.visited_points.append(soln_at_x)
                    interpolation_solutions.append(soln_at_x)

            # Update R factorisation in LU algorithm
            phi = phi_function(X[k, :].reshape(-1, 1))
            R[k, k] = np.dot(v, phi)
            for i in range(k + 1, q):
                R[k, i] += phi[i]
                for j in range(k):
                    R[k, i] -= (phi[j] * R[j, i]) / R[j, j]

            # Check if pivot is too small (would cause poor conditioning)
            # Require full psi_1 for all points except last which needs psi_2
            min_pivot = psi_2 if k == q - 1 else psi_1
            if abs(R[k, k]) < min_pivot:
                # Try to find a better point if pivot is too small
                try:
                    s_backup = self.find_new_point_alternative(
                        delta, v, phi_function, X[:k, :], U
                    )
                    phi_backup = phi_function(s_backup)
                    R_backup = np.dot(v, phi_backup)  # noqa: N806
                    if abs(R_backup) > abs(R[k, k]):
                        s = s_backup
                        phi = phi_backup
                        R[k, k] = R_backup
                        # Update X[k] if we changed the point
                        X[k, :] = s
                        soln_at_s = self.create_new_solution(
                            tuple(s.ravel()), self.problem
                        )
                        pilot_run = self.calculate_pilot_run(construct_model=True)
                        soln_at_s = self.perform_adaptive_sampling(
                            soln_at_s, pilot_run, delta
                        )
                        f_value = (
                            -1
                            * self.problem.minmax[0]
                            * soln_at_s.objectives_mean.item()
                        )
                        fX[k, 0] = f_value
                        if all(
                            tuple(pt.x) != tuple(soln_at_s.x)
                            for pt in self.visited_points
                        ):
                            self.visited_points.append(soln_at_s)
                        if k < len(interpolation_solutions):
                            interpolation_solutions[k] = soln_at_s
                        else:
                            interpolation_solutions.append(soln_at_s)
                except Exception:
                    pass  # Keep original point if backup fails

            for i in range(k + 1, q):
                R[k, i] += phi[i]
                for j in range(k):
                    R[k, i] -= (phi[j] * R[j, i]) / R[j, j]

        # If p > q, fill remaining rows sensibly (reuse leftover improved candidates or
        # sample inside trust-region)
        if p > q:
            for k in range(q, p):
                if fX_improved.size > 0:
                    # take the best remaining candidate (first in list) to fill the slot
                    x = X_improved[0, :]
                    X[k, :] = x
                    fX[k, :] = fX_improved[0]
                    # remove used candidate
                    X_improved = np.delete(X_improved, 0, 0)  # noqa: N806
                    fX_improved = np.delete(fX_improved, 0, 0)  # noqa: N806
                else:
                    # generate a reasonable fallback sample within the trust region
                    # (projected close to incumbent)
                    random_dir = np.random.normal(size=(self.problem.dim, 1))
                    random_dir = random_dir / norm(random_dir) * delta * 0.5
                    x = x_k.flatten() + random_dir.flatten()
                    X[k, :] = x
                    soln_at_x = self.create_new_solution(tuple(x.ravel()), self.problem)
                    pilot_run = self.calculate_pilot_run(construct_model=True)
                    soln_at_x = self.perform_adaptive_sampling(
                        soln_at_x, pilot_run, delta
                    )
                    f_value = (
                        -1 * self.problem.minmax[0] * soln_at_x.objectives_mean.item()
                    )
                    fX[k, 0] = f_value
                    if all(
                        tuple(pt.x) != tuple(soln_at_x.x) for pt in self.visited_points
                    ):
                        self.visited_points.append(soln_at_x)
                    interpolation_solutions.append(soln_at_x)

        return X, fX, interpolation_solutions

    def getTotalOrderBasisRecursion(  # noqa: N802
        self, highest_order: int, dimensions: int
    ) -> np.ndarray:
        """Generates the total order basis recursively.

        Args:
                highest_order (int): The highest polynomial order
                dimensions (int): The number of dimensions

        Returns:
                np.ndarray: The total order basis of shape (L, dimensions) where L is
                the cardinality
        """
        if dimensions == 1:
            I = np.zeros((1, 1))  # noqa: E741, N806
            I[0, 0] = highest_order
        else:
            for j in range(0, highest_order + 1):
                U = self.getTotalOrderBasisRecursion(highest_order - j, dimensions - 1)  # noqa: N806
                rows, cols = U.shape
                T = np.zeros((rows, cols + 1))  # allocate space!  # noqa: N806
                T[:, 0] = j * np.ones((1, rows))
                T[:, 1 : cols + 1] = U
                if j == 0:
                    I = T  # noqa: E741, N806
                elif j >= 0:
                    rows_I, cols_I = I.shape  # noqa: N806
                    rows_T, _cols_T = T.shape  # noqa: N806
                    Itemp = np.zeros((rows_I + rows_T, cols_I))  # noqa: N806
                    Itemp[0:rows_I, :] = I
                    Itemp[rows_I : rows_I + rows_T, :] = T
                    I = Itemp  # noqa: E741, N806
                del T
        return I

    def get_basis(self, orders: np.ndarray) -> np.ndarray:
        """Generates the total order basis for the given orders.

        Args:
                orders (np.ndarray): The orders for each dimension

        Raises:
                Exception: If the cardinality is too large

        Returns:
                np.ndarray: The total order basis of shape (L, dimensions) where L is
                the cardinality
        """
        dimensions = len(orders)
        highest_order = np.max(orders)
        # Check what the cardinality will be, stop if too large!
        L = int(  # noqa: N806
            math.factorial(highest_order + dimensions)
            / (math.factorial(highest_order) * math.factorial(dimensions))
        )
        # Check cardinality
        if int(1e6) <= L:
            raise Exception(
                f"Cardinality {L:.1e} is >= hard cardinality limit {int(1e6):.1e}"
            )
        # Generate basis
        total_order = np.zeros((1, dimensions))
        for i in range(1, highest_order + 1):
            R = self.getTotalOrderBasisRecursion(i, dimensions)  # noqa: N806
            total_order = np.vstack((total_order, R))
        return total_order

    def get_phi_function_and_derivative(
        self,
        U: np.ndarray,  # noqa: N803
        delta: float,
    ) -> tuple[callable, callable]:
        """Generates the phi function and its derivative for the given sample set.

        Args:
                U (np.ndarray): The active subspace matrix (shape (n,d))
                delta (float): The trust-region radius

        Returns:
                tuple[callable, callable]: The phi function and its derivative
        """
        q = len(self.index_set(self.degree, self.d).astype(int))
        x_k = np.asarray(self.incumbent_x).ravel()

        total_order_index_set = self.get_basis(np.tile([2], q))[
            :, range(self.d - 1, -1, -1)
        ]

        def phi_function(s: np.ndarray) -> np.ndarray:
            s = s.ravel()  # shape (d,)
            u = np.dot(s - x_k, U) / delta
            u = np.atleast_2d(u)
            m = u.shape[0]

            phi = np.zeros((m, q))
            for k in range(q):
                exponents = total_order_index_set[k, :]
                numerator = np.power(u, exponents)
                denom = np.array([factorial(int(e)) for e in exponents])
                phi[:, k] = np.prod(numerator / denom, axis=1)
            if phi.shape[0] == 1:
                return phi.ravel()
            return phi

        def phi_function_deriv(s: np.ndarray) -> np.ndarray:
            s = s.ravel()
            u = np.dot(s - x_k, U) / delta
            phi_deriv = np.zeros((self.d, q))
            for i in range(self.d):
                for k in range(1, q):
                    exponent = total_order_index_set[k, i]
                    if exponent != 0:
                        tmp = np.zeros(self.d, dtype=np.int_)
                        tmp[i] = 1
                        exps_minus_tmp = total_order_index_set[k, :] - tmp
                        numerator = np.prod(
                            np.divide(
                                np.power(u, exps_minus_tmp),
                                [
                                    factorial(int(e))
                                    for e in total_order_index_set[k, :]
                                ],
                            )
                        )
                        phi_deriv[i, k] = exponent * numerator
            phi_deriv = phi_deriv / delta
            return np.dot(U, phi_deriv)

        return phi_function, phi_function_deriv

    def find_new_point(
        self,
        delta: float,
        v: np.ndarray,
        phi_function: callable,
        phi_function_deriv: callable,
    ) -> np.ndarray:
        """Finds a new point in the trust region that maximizes the absolute value of.

        the dot product with the phi function.

        Args:
                delta (float): The trust-region radius
                v (np.ndarray): The direction vector of shape (q,1)
                phi_function (callable): The phi function
                phi_function_deriv (callable): The derivative of the phi function

        Returns:
                np.ndarray: The new point in the trust region
        """
        x_k = np.array(self.incumbent_x).reshape(-1, 1)
        bounds_l = np.maximum(
            np.array(self.problem.lower_bounds).reshape(x_k.shape), x_k - delta
        )
        bounds_u = np.minimum(
            np.array(self.problem.upper_bounds).reshape(x_k.shape), x_k + delta
        )

        bounds = []
        for i in range(self.problem.dim):
            bounds.append((bounds_l[i], bounds_u[i]))

        def obj1(s):  # noqa: ANN001, ANN202
            return np.dot(v, phi_function(s))

        def jac1(s):  # noqa: ANN001, ANN202
            return np.dot(phi_function_deriv(s), v)

        def obj2(s):  # noqa: ANN001, ANN202
            return -np.dot(v, phi_function(s))

        def jac2(s):  # noqa: ANN001, ANN202
            return -np.dot(phi_function_deriv(s), v)

        res1 = minimize(
            obj1, x_k, method="TNC", jac=jac1, bounds=bounds, options={"disp": False}
        )
        res2 = minimize(
            obj2, x_k, method="TNC", jac=jac2, bounds=bounds, options={"disp": False}
        )
        return res1["x"] if abs(res1["fun"]) > abs(res2["fun"]) else res2["x"]

    def generate_set_in_subspace(
        self,
        delta: float,
        U: np.ndarray,  # noqa: N803
        num: int,
    ) -> np.ndarray:
        """Generates a set of points with good geometry in the projected subspace.

                Points are generated in the active subspace and then lifted to full
                space.
                Uses coordinate directions first, then random directions for better
                coverage.

        Args:
                delta (float): The trust-region radius
                U (np.ndarray): The active subspace matrix (shape (n, d))
                num (int): The number of points to generate

        Returns:
                np.ndarray: A set of points with good geometry in projected space (shape
                (num, n))
        """
        x_k = np.array(self.incumbent_x).reshape(-1, 1)
        x_k_projected = U.T @ x_k  # shape (d, 1)
        d = U.shape[1]  # Get subspace dimension from U

        # Generate directions in the d-dimensional subspace
        S = np.zeros((num, self.problem.dim))  # noqa: N806
        S[0, :] = x_k.flatten()

        idx = 1
        # First, add coordinate directions in the subspace (2d points)
        for j in range(min(d, num - 1)):
            for sign in [1, -1]:
                if idx >= num:
                    break
                y = np.zeros(d)
                y[j] = sign * delta

                # Lift to full space: x_k + U @ y
                s = U @ (x_k_projected + y.reshape(-1, 1))
                s = s.flatten()

                # Project back to feasible region
                bounds_l = np.array(self.problem.lower_bounds)
                bounds_u = np.array(self.problem.upper_bounds)
                s = np.maximum(bounds_l, np.minimum(bounds_u, s))

                S[idx, :] = s
                idx += 1

        # Fill remaining with random directions for better coverage
        for i in range(idx, num):
            # Generate random direction in subspace
            y = np.random.randn(d)
            y = y / norm(y)  # Normalize

            # Scale by delta (use varying scales for diversity)
            scale = delta * (0.5 + 0.5 * np.random.rand())
            y = y * scale

            # Lift to full space: x_k + U @ y
            s = U @ (x_k_projected + y.reshape(-1, 1))
            s = s.flatten()

            # Project back to feasible region
            bounds_l = np.array(self.problem.lower_bounds)
            bounds_u = np.array(self.problem.upper_bounds)
            s = np.maximum(bounds_l, np.minimum(bounds_u, s))

            S[i, :] = s

        return S

    def find_new_point_alternative(
        self,
        delta: float,
        v: np.ndarray,
        phi_function: callable,
        X: np.ndarray,  # noqa: N803
        U: np.ndarray,  # noqa: N803
    ) -> np.ndarray:
        """Finds a new point in the trust region by generating a sample set and.

        selecting the point that maximizes the.

                absolute value of the dot product with the phi function.
                Checks for duplicates in the projected space to ensure good geometry.

        Args:
                delta (float): The trust-region radius
                v (np.ndarray): The direction vector
                phi_function (callable): The phi function
                X (np.ndarray): The current sample set (shape (k, n))
                U (np.ndarray): The active subspace matrix (shape (n, d))

        Returns:
                np.ndarray: The new point in the trust region
        """
        x_k = np.array(self.incumbent_x).reshape(-1, 1)
        no_pts = max(
            int(0.5 * self.d * (self.d + 2)), 2 * self.d + 1, 20
        )  # Generate enough points in subspace
        X_tmp = self.generate_set_in_subspace(delta, U, no_pts)  # noqa: N806

        # Filter to keep only point within the trust region in projected space
        X_tmp_projected = X_tmp @ U  # noqa: N806
        x_k_projected = x_k.T @ U
        mask = norm(X_tmp_projected - x_k_projected, axis=1, ord=2) <= delta
        X_tmp = X_tmp[mask]  # noqa: N806
        if X_tmp.shape[0] == 0:  # Fallback if all filtered out
            X_tmp = self.generate_set_in_subspace(delta, U, no_pts)  # noqa: N806

        Phi_X_improved = np.vstack(  # noqa: N806
            [phi_function(X_tmp[i, :].reshape(-1, 1)) for i in range(X_tmp.shape[0])]
        )
        M = np.absolute(Phi_X_improved @ v)  # noqa: N806
        indices = np.argsort(M)[::-1][: len(M)]
        X_proj = X @ U  # noqa: N806
        for index in indices:
            x = X_tmp[index, :]
            x_proj = (x.reshape(1, -1) @ U).ravel()
            # Points should be separated by at least 1% of delta in projected space
            min_dist = (
                np.min(norm(X_proj - x_proj, axis=1)) if X_proj.shape[0] > 0 else np.inf
            )
            if min_dist >= 0.01 * delta:
                return x
        return X_tmp[indices[0], :]

    def remove_point_from_set(
        self,
        X: np.ndarray,  # noqa: N803
        fX: np.ndarray,  # noqa: N803
        x: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Removes the current solution from the sample set.

        Args:
                X (np.ndarray): The current sample set
                fX (np.ndarray): The function values corresponding to the sample set
                x (np.ndarray): The current solution to be removed.

        Returns:
                tuple[np.ndarray, np.ndarray]: The updated sample set and function
                values after removal
        """
        ind_current = np.where(norm(X - x.ravel(), axis=1, ord=2) == 0.0)[0]
        X = np.delete(X, ind_current, 0)  # noqa: N806
        fX = np.delete(fX, ind_current, 0)  # noqa: N806
        return X, fX

    def remove_furthest_point(
        self,
        X: np.ndarray,  # noqa: N803
        fX: np.ndarray,  # noqa: N803
        x: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Removes the furthest point from the current solution in the sample set.

        Args:
                X (np.ndarray): The current sample set
                fX (np.ndarray): The function values corresponding to the sample set
                x (np.ndarray): The current solution to be removed
        Returns:
                tuple[np.ndarray, np.ndarray]: The updated sample set and function
                values after removal
        """
        ind_distant = np.argmax(norm(X - x.ravel(), axis=1, ord=2))
        X = np.delete(X, ind_distant, 0)  # noqa: N806
        fX = np.delete(fX, ind_distant, 0)  # noqa: N806
        return X, fX

    def remove_furthest_point_projected(
        self,
        X: np.ndarray,  # noqa: N803
        fX: np.ndarray,  # noqa: N803
        x: np.ndarray,
        U: np.ndarray,  # noqa: N803
    ) -> tuple[np.ndarray, np.ndarray]:
        """Removes the furthest point from the current solution in the sample set based.

        on projected coordinates.

        Args:
                X (np.ndarray): The current sample set (shape (M, n))
                fX (np.ndarray): The function values corresponding to the sample set
                x (np.ndarray): The current solution to be removed (shape (n, 1))
                U (np.ndarray): The active subspace matrix (shape (n, d))

        Returns:
                tuple[np.ndarray, np.ndarray]: The updated sample set and function
                values after removal
        """
        X_projected = X @ U  # noqa: N806
        x_projected = x.T @ U
        ind_distant = np.argmax(norm(X_projected - x_projected, axis=1, ord=2))
        X = np.delete(X, ind_distant, 0)  # noqa: N806
        fX = np.delete(fX, ind_distant, 0)  # noqa: N806
        return X, fX

    #! === MODEL CONSTRUCTION ===

    def construct_model(
        self,
    ) -> tuple[
        callable,
        callable,
        np.ndarray,
        list[float],
        list[Solution],
        np.ndarray,
        np.ndarray,
    ]:
        """Builds a local approximation of the response surface within the current.

        trust.

        region (defined as ||x-x_k||<=delta).

                The method fit recovers the local approximation given a design set of
                2d+1 design points and a corresponding active subspace U of shape (n,d)
                That projects the n-dimensional design points to a d-dimensional
                subspace.

        Returns:
                tuple[callable, callable, np.ndarray, list[float], list[Solution]]:
                        - The local model as a function that takes a numpy vector of
                        shape (n,1) and returns a float
                        - The local model gradient as a function that takes a numpy
                        vector of shape (n,1) and returns gradient
                        - The final computed active subspace of the iteration of shape
                        (n,d)
                        - A list of the function estimates of the objective function at
                        each of the final design points  # noqa: D417
                        - The list of solutions of the final design points
        """
        # Reset scaling parameters for this new model construction
        if hasattr(self.basis_adapter, "reset_scaling"):
            self.basis_adapter.reset_scaling()

        # Use warm starting: if previous active subspace exists, use it as initial guess
        if self.prev_U is not None and self.prev_U.shape == (self.problem.dim, self.d):
            # Warm start with previous iteration's active subspace
            U = self.prev_U.copy()  # noqa: N806
        else:
            # Cold start: generate initial subspace from coordinate directions
            init_S_full = self.generate_set(self.d)  # noqa: N806
            U, _ = np.linalg.qr(init_S_full.T)  # noqa: N806

        X, f_index = self.construct_interpolation_set(U)  # noqa: N806

        fX, interpolation_solutions = self.evaluate_interpolation_points(  # noqa: N806
            f_index, X, self.delta
        )

        fval = fX.flatten().tolist()

        U, X, fX, interpolation_solutions = self.fit(X, fX, interpolation_solutions, U)  # noqa: N806

        fval = fX.flatten().tolist()

        return U, fval, interpolation_solutions, X, fX

    def model_evaluate(
        self,
        x_proj: np.ndarray,
        coef: np.ndarray,
        U: np.ndarray,  # noqa: N803
    ) -> float:
        """Evaluates the local approximated model at a given design point.

        Args:
                x_proj (np.ndarray): Projected design point of shape (d,1).
                coef (np.ndarray): The coefficients of the local model of shape (q,1).
                U (np.ndarray): The active subspace of shape (n,d).

        Returns:
                float: The evaluation of the model at x, given as m(U^Tx)
        """
        if len(x_proj.shape) != 2 or x_proj.shape[1] != 1:
            x_proj = x_proj.reshape(-1, 1)
        if x_proj.shape[0] == U.shape[0]:
            # project x to active subspace
            x_proj = U.T @ x_proj  # (d,1)
        if len(coef.shape) != 2:
            coef = coef.reshape(-1, 1)

        # build vandermonde matrix of shape (1,q)
        V_matrix = self.V(x_proj.T)  # (1,q)  # noqa: N806

        # find evaluation:
        res = V_matrix @ coef

        return res.item()

    def fit(
        self,
        X: np.ndarray,  # noqa: N803
        fX: np.ndarray,  # noqa: N803
        interpolation_solutions: list[Solution],
        U0: np.ndarray,  # noqa: N803
    ) -> tuple[np.ndarray, callable, callable, np.ndarray, np.ndarray, list[Solution]]:
        """Fits the design set and evaluated points to a local model with recovered.

        active subspace.

                It undergoes a loop until the active subspace converges.
                First, it improves the design set X and constructing an interpolation
                model until it can ensure the criticality step is satisfied.
                Second, after fixing the model coefficients, it updates the active
                subspace through a variable projection scheme
        Args:
                X (np.ndarray): design set of shape (M,n)
                fX (np.ndarray): corresponding function estimates of design points of
                shape (M,1)
                interpolation_solutions (list[Solution]): A list of the design points in
                the
                U0 (np.ndarray): The initial estimate for the active subspace of shape
                (n,d).

        Returns:
                tuple[np.ndarray, callable, callable, np.ndarray, np.ndarray,
                list[Solution]]:
                        - The final computed active subspace of the iteration of shape
                        (n,d)
                        - The local model as a function that takes a numpy vector of
                        shape (n,1) and returns a float (this is given as model = lambda
                        x : self.model_evaluate(x,coef, U))
                        - The local model gradient as a function that takes a numpy
                        vector of shape (n,1) and returns a float (this is given as
                        model_grad = lambda x : self.model_evaluate(x,U))
                        - The design set after going through fitting of shape (M,n)
                        - The function estimates of the objective function at each of
                        the final design points of shape (M,1)
                        - The design points as solution objects
        """
        # Algorithmic Parameters
        beta = 10

        # Orthogonalize just to make sure the starting value satisfies constraints
        U0, R = np.linalg.qr(U0, mode="reduced")  # noqa: N806
        U0 = np.dot(U0, np.diag(np.sign(np.diag(R))))  # noqa: N806

        prev_U = np.zeros(U0.shape)  # noqa: N806
        U = np.copy(U0)  # noqa: N806
        model_delta = float(self.delta)

        if self.degree == 1 and U.shape[1] == 1:
            V_matrix = np.hstack(  # noqa: N806
                (np.ones((X.shape[0], 1)), X)
            )  # (M, n+1)
            fn_coef = pinv(V_matrix) @ fX  # (n+1,1)
            fn_grad = fn_coef[1:, :].reshape(-1, 1)  # (n,1)
            U = fn_grad / norm(fn_grad)  # noqa: N806

        else:
            i = 0
            while True:  # not self.converged_subspace_check(prev_U, U) :
                subspace_tol = 1e-2
                if self.converged_subspace_check(prev_U, U, subspace_tol):
                    break
                # * Construct model and Criticality step
                (
                    coef,
                    model_delta,
                    X,  # noqa: N806
                    fX,  # noqa: N806
                    interpolation_solutions,
                ) = self.criticality_check(X, fX, U, interpolation_solutions)

                # set the old U and update
                prev_U = np.copy(U)  # noqa: N806
                U = self.fit_varpro(X, fX, U)  # noqa: N806
                i += 1

        coef = self.fit_coef(X, fX, U)

        # final fitting of the coefficients and rotating the final U
        U = self.rotate_U(X, fX, coef, U)  # noqa: N806
        coef = self.fit_coef(X, fX, U)

        # Define model functions
        self.model = partial(model_evaluate_fn, coef=coef, U=U, instance=self)
        self.model_grad = partial(model_grad_fn, coef=coef, U=U, instance=self)
        self.model_hess = partial(model_hess_fn, coef=coef, U=U, instance=self)

        # Store the current Hessian at incumbent for next iteration's trust-region
        # This ensures the trust-region is based on the previous model's curvature
        x_k = np.array(self.incumbent_x).reshape(1, -1)
        if self.factors["elliptical trust region"]:
            self.prev_H = self.model_hess(x_k, full_space=True).reshape(
                self.problem.dim, self.problem.dim
            )
        else:
            self.prev_H = np.eye(self.problem.dim).reshape(
                self.problem.dim, self.problem.dim
            )

        # Record gradient eigenvalues for adaptive dimension selection
        # Compute full-space gradients (M, n) by inflating reduced-space gradients
        if self.factors.get("adaptive subspace dimension", False) and X.shape[0] >= 2:
            try:
                # If model_grad supports batch evaluation, use it; otherwise compute
                # per-point
                try:
                    full_space_grads = self.model_grad(X, full_space=True)
                except Exception:
                    full_space_grads = np.vstack(
                        [
                            self.model_grad(x.reshape(1, -1), full_space=True).flatten()
                            for x in X
                        ]
                    )
                # Ensure shape
                if full_space_grads.ndim == 1:
                    full_space_grads = full_space_grads.reshape(1, -1)
                self.record_gradient_eigenvalues(full_space_grads)
            except Exception as e:
                logging.debug(f"Could not record gradient eigenvalues: {e}")

        # --- Per-dimension validation using existing design set (no extra sims) ---
        # Compute validation errors for candidate subspace dimensions using X and fX
        try:
            validation_by_d: dict[int, float] = {}
            max_test_d = min(self.max_d, U.shape[1])
            # Use modest cap to limit cost; test dims from 1..max_test_d
            for test_d in range(1, max_test_d + 1):
                U_cand = U[:, :test_d]  # noqa: N806
                Y = X @ U_cand  # noqa: N806
                Vmat = self.V(Y)  # noqa: N806
                # Fit coefficients in reduced space for this candidate d
                try:
                    coef_cand = self.fit_coef(X, fX, U_cand)
                except Exception:
                    # fallback: try pseudo-inverse directly on Vmat
                    try:
                        coef_cand = pinv(Vmat) @ fX
                    except Exception:
                        coef_cand = None
                if coef_cand is None:
                    continue
                preds = Vmat @ coef_cand
                # compute normalized RMSE (relative to mean abs of fX)
                err = np.sqrt(
                    np.mean(
                        (
                            fX.reshape(
                                -1,
                            )
                            - preds.reshape(
                                -1,
                            )
                        )
                        ** 2
                    )
                )
                denom = max(1e-10, np.mean(np.abs(fX)))
                validation_by_d[int(test_d)] = float(err / denom)
            # store for use when recording iteration performance
            self.last_validation_by_d = validation_by_d
        except Exception as e:
            logging.debug(f"Validation-by-d computation failed: {e}")

        if self.delta != model_delta:
            self.delta = min(
                max(self.delta, beta * norm(self.grad(X, coef, U))), model_delta
            )

        return U, X, fX, interpolation_solutions

    def criticality_check(
        self,
        X: np.ndarray,  # noqa: N803
        fX: np.ndarray,  # noqa: N803
        U: np.ndarray,  # noqa: N803
        interpolation_solutions: list[Solution],
    ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, list[Solution]]:
        """Performs the criticality step of the trust-region method.

                It fits a local model to the design set and checks whether the
                criticality condition is satisfied.

        Args:
                X (np.ndarray): The design set of shape (M,n)
                fX (np.ndarray): The corresponding function estimates of shape (M,1)
                U (np.ndarray): The current active subspace matrix
                interpolation_solutions (list[Solution]): The list of solutions in the
                interpolation set

        Returns:
                tuple[np.ndarray, float, np.ndarray, np.ndarray, list[Solution]]:
                        - The model coefficients of shape (q,1)
                        - The trust-region radius after criticality check
                        - The design set after criticality check of shape (M,n)
                        - The function estimates of the objective function at each of
                        the final design points of shape (M,1)
                        - The design points as solution objects
        """
        w: float = 0.85
        tol: float = 1e-6
        kappa_f: float = 10.0
        kappa_g: float = 10.0
        coef: np.ndarray | None = None

        model_delta = float(self.delta)

        # * Construct model and Criticality step
        fitting_iter = 0
        while True:
            coef = self.fit_coef(X, fX, U)
            grad = self.grad(X, coef, U) @ U.T  # (M,n)
            gnorm = norm(grad)

            if gnorm <= tol:
                if not self.fully_linear_test(X, fX, coef, U, kappa_f, kappa_g):
                    X, fX, interpolation_solutions = self.improve_geometry(  # noqa: N806
                        model_delta, U, X, fX, interpolation_solutions
                    )
                    model_delta = self.delta * w**fitting_iter
                    fitting_iter += 1

            elif model_delta > max(self.mu * gnorm, 1e-12):
                model_delta = min(
                    model_delta, max(self.delta_min, self.mu * gnorm, 1e-12)
                )
                X, fX, interpolation_solutions = self.improve_geometry(  # noqa: N806
                    model_delta, U, X, fX, interpolation_solutions
                )
                fitting_iter += 1
            else:
                break

        return coef, model_delta, X, fX, interpolation_solutions

    def converged_subspace_check(
        self,
        prev_U: np.ndarray,  # noqa: N803
        U: np.ndarray,  # noqa: N803
        tol: float,
    ) -> bool:
        """Check whether the active subspace has converged by computing the subspace.

                distance between previous and current subspace estimates.

        Args:
                prev_U (np.ndarray): Active subspace matrix from previous iteration
                (shape (n,d))
                U (np.ndarray): Active subspace from current iteration (shape (n,d))
                tol (float, optional): Convergence tolerance for subspace distance.

        Returns:
                bool: Returns True is subspace change is below tolerance and False
                otherwise
        """
        if np.all(prev_U == 0):
            return False
        C = prev_U.T @ U  # shape (d, d)  # noqa: N806
        # Singular values of C are cos(theta_i)
        try:
            sigma = np.linalg.svd(C, compute_uv=False)
        except Exception as e:
            logging.warning(
                "SVD failed in subspace convergence check with error: %s", e
            )
            try:
                CtC = C.T @ C  # noqa: N806
                w = np.linalg.eigvalsh(CtC)
                sigma = np.sqrt(np.maximum(w, 0.0))
            except Exception as e2:
                logging.warning("Fallback eigendecomposition failed: %s", e2)
                return False  # safest fail-closed behavior

        sigma = np.clip(sigma, -1.0, 1.0)
        # Compute principal angles and distance
        sin_theta = np.sqrt(1.0 - sigma**2)
        subspace_dist = (
            np.max(sin_theta)
        ).item()  # operator norm of projector difference

        return subspace_dist <= tol

    def fully_linear_test(
        self,
        X: np.ndarray,  # noqa: N803
        fX: np.ndarray,  # noqa: N803
        coef: np.ndarray,
        U: np.ndarray,  # noqa: N803
        kappa_f: float,
        kappa_g: float,
    ) -> bool:
        """Check whether a model is fully linear in a trust region, using function.

        residuals and model gradient consistency.

        Args:
                X (np.ndarray): Design set of points (shape (M,n))
                fX (np.ndarray): Corresponding function estimation of design points
                (shape (M,1))
                coef (np.ndarray): Model coefficients (shape (q,1))
                U (np.ndarray): Active subspace matrix (shape (n,d))
                kappa_f (float): Tolerance of zeroth-order fully-linear bound
                kappa_g (float): Tolerance of first-order fully-linear bound.

        Returns:
                bool: True if the model is fully-linear and False otherwise.
        """
        M, _n = X.shape  # noqa: N806

        # --- 1. Value-based condition ---
        mX = np.array(  # noqa: N806
            [self.model_evaluate(U.T @ np.array(x).reshape(-1, 1), coef, U) for x in X]
        ).reshape(-1, 1)
        residuals = np.abs(fX - mX)
        value_condition = np.max(residuals) <= kappa_f * self.delta**2

        # --- 2. Gradient consistency condition ---
        m_grads = self.grad(X, coef, U)  # shape (M, d)
        consistent = True

        for i in range(M):
            for j in range(i + 1, M):
                dx = X[i, :] - X[j, :]  # shape (n,)
                dm = mX[i, 0] - mX[j, 0]
                g_j = (U @ m_grads[j, :].reshape(-1, 1)).flatten()  # shape (n,)
                linearized_diff = np.dot(g_j, dx)
                model_error = np.abs(dm - linearized_diff)
                if model_error > kappa_g * np.linalg.norm(dx) ** 2:
                    consistent = False
                    break
            if not consistent:
                break

        # --- Fully linear if both conditions hold ---
        return bool(value_condition and consistent)

    def fit_coef(self, X: np.ndarray, fX: np.ndarray, U: np.ndarray) -> np.ndarray:  # noqa: N803
        """Finds the coefficients of the interpolation model by solving the system of.

        equations:

                                V(U^TX)coeff = fX
        Args:
                X (np.ndarray): The design set of shape (M,n)
                fX (np.ndarray): The corresponding function estimates of shape (M,1)
                U (np.ndarray): The active subspace of shape (n,d)
                delta (float): radius of the current trust-region.

        Returns:
                np.ndarray: A list of the coefficients of shape (q,1)
        """
        # ! Handle NaN in fX and X - this is a quick fix because something is happening
        # with the incumbent solution evaluation suddenly disappearing
        if np.isnan(fX).any():
            # If there are NaN values in fX, find the corresponding row in X, create a
            # solution, simulate it and update fX
            nan_indices = np.where(np.isnan(fX))[0]
            for idx in nan_indices:
                x_nan = X[idx, :].ravel()
                solution_nan = self.create_new_solution(tuple(x_nan), self.problem)
                self.problem.simulate(solution_nan, int(self.factors["lambda_min"]))
                fx_nan = (
                    -1 * self.problem.minmax[0] * solution_nan.objectives_mean.item()
                )
                fX[idx, 0] = fx_nan

        Y = X @ U  # noqa: N806
        V_matrix = self.V(Y)  # shape (M,q)  # noqa: N806
        M, q = V_matrix.shape  # noqa: N806

        # flatten fX to vector
        y = fX.reshape(
            -1,
        )

        # Column-scale V to improve conditioning (then undo after solving)
        col_norms = np.linalg.norm(V_matrix, axis=0)
        # avoid division by zero
        col_norms_safe = np.where(col_norms <= 0.0, 1.0, col_norms)
        V_scaled = V_matrix / col_norms_safe  # noqa: N806

        try:
            # SVD-based pseudo-inverse with truncation for numerical stability
            U_s, svals, Vh = scipy.linalg.svd(V_scaled, full_matrices=False)  # noqa: N806
            eps = np.finfo(float).eps
            tol = max(M, q) * eps * (svals.max() if svals.size > 0 else 0.0)

            # compute initial truncated pseudoinverse solution
            s_inv = np.zeros_like(svals)
            nonzero = svals > tol
            if nonzero.any():
                s_inv[nonzero] = 1.0 / svals[nonzero]

            Ut_y = U_s.T @ y  # noqa: N806
            coef_scaled = Vh.T @ (s_inv * Ut_y)
            coef = (coef_scaled / col_norms_safe).reshape(-1, 1)

            # Estimate condition number (robust to zeros)
            if nonzero.any():
                cond_est = float(svals.max() / svals[nonzero].min())
            else:
                cond_est = np.inf

            # If ill-conditioned, choose a Tikhonov (ridge) regularisation via GCV
            if cond_est > 1e8:
                # Use SVD factors to compute GCV-optimal lambda efficiently
                s2 = svals**2
                Ut_y_sq = Ut_y**2  # noqa: N806

                # Build a candidate lambda grid scaled to the problem
                s2_max = s2.max() if s2.size > 0 else 1.0
                lambdas = np.logspace(-16, -2, 15) * (s2_max + 1e-16)

                best_lambda = None
                best_gcv = np.inf
                for lam in lambdas:
                    # numerator: ||(I - H) y||^2 computed in SVD basis
                    filt = (lam / (s2 + lam)) if s2.size > 0 else np.array([1.0])
                    num = np.sum((filt**2) * Ut_y_sq)
                    # effective degrees of freedom: trace(H)
                    df = np.sum(s2 / (s2 + lam)) if s2.size > 0 else 0.0
                    denom = (M - df) ** 2
                    gcv = np.inf if denom <= 0 else num / denom
                    if gcv < best_gcv:
                        best_gcv = gcv
                        best_lambda = lam

                # Safety: if no good lambda found, fall back to a small regulariser
                if best_lambda is None or not np.isfinite(best_lambda):
                    best_lambda = 1e-12 * (s2_max + 1e-16)

                # compute Tikhonov solution via SVD factors: s/(s^2 + lambda) * Ut_y
                s_filt = np.zeros_like(svals)
                if svals.size > 0:
                    s_filt = svals / (s2 + best_lambda)
                coef_scaled = Vh.T @ (s_filt * Ut_y)
                coef = (coef_scaled / col_norms_safe).reshape(-1, 1)

        except Exception as e:
            logging.warning("SVD/pinv approach failed in fit_coef: %s", e)
            # fallback to robust least squares / pseudo-inverse
            try:
                coef = scipy.linalg.lstsq(V_matrix, fX, cond=1e-10)[0]
            except Exception as e2:
                logging.warning("Lstsq failed in fit_coef with error: %s", e2)
                coef = pinv(V_matrix) @ fX

        if len(coef.shape) != 2:
            coef = coef.reshape(-1, 1)

        return coef

    def grassmann_trajectory(
        self,
        U: np.ndarray,  # noqa: N803
        Delta: np.ndarray,  # noqa: N803
        t: float,
    ) -> np.ndarray:
        """Calculates the geodesic along the Grassmann manifold.

        Args:
                U (np.ndarray): The active subspace matrix of shape (n,d)
                Delta (np.ndarray): The search direction along the Grassmann manifold
                with shape (n,d)
                t (float): Independent parameter in the line equation takes values
                between (0,infty) and is selected to ensure convergence.

        Returns:
                np.ndarray: The new candidate for the active subspace based on the step
                made of shape (n,d)
        """
        try:
            Y, sig, ZT = scipy.linalg.svd(  # noqa: N806
                Delta, full_matrices=False, lapack_driver="gesvd"
            )
        except Exception:
            Y, sig, ZT = scipy.linalg.svd(Delta, full_matrices=False)  # noqa: N806

        UZ = np.dot(U, ZT.T)  # noqa: N806
        U_new = np.dot(UZ, np.diag(np.cos(sig * t))) + np.dot(  # noqa: N806
            Y, np.diag(np.sin(sig * t))
        )

        # Correct the new step U by ensuring it is orthonormal with consistent sign on
        # the elements
        U_new, R = np.linalg.qr(U_new, mode="reduced")  # noqa: N806
        return np.dot(U_new, np.diag(np.sign(np.diag(R))))

    def residual(self, X: np.ndarray, fX: np.ndarray, U: np.ndarray) -> np.ndarray:  # noqa: N803
        """Construct the Residual of the model fitting, such that.

                r = fX - V(U^TX)coeff
        Args:
                X (np.ndarray): The design set of shape (M,n)
                fX (np.ndarray): The corresponding function estimates of shape (M,1)
                U (np.ndarray): The active subspace of shape (n,d)
                delta (float): radius of the current trust-region.

        Returns:
                np.ndarray: The residual error for each design point on the local model
                of shape (M,1)
        """
        c = self.fit_coef(X, fX, U)  # shape(q,1)
        model_fX = np.array(  # noqa: N806
            [self.model_evaluate(U.T @ np.array(x).reshape(-1, 1), c, U) for x in X]
        ).reshape(-1, 1)  # A list of length M with float elements
        return fX - model_fX

    #! THIS NEEDS CHECKING OVER
    def jacobian(self, X: np.ndarray, fX: np.ndarray, U: np.ndarray) -> np.ndarray:  # noqa: N803
        """Constructs the Jacobian of the residual with respect to the active subspace.

        Args:
                X (np.ndarray): The design set of shape (M,n) or (M,d)
                fX (np.ndarray): The corresponding function estimates of design points
                of shape (M,1)
                U (np.ndarray): The active subspace of shape (n,d)
                delta (float): radius of the current trust-region.

        Returns:
                np.ndarray: A tensor of shape (M,n,d) where each element is the partial
                derivative of the i-th residual component with respect to the (j,k)th
                entry of the active subspace
        """
        # FIRST ENSURE THAT THE ARGUMENTS HAVE DIMENSIONS THAT MATCH
        assert X.shape[1] == U.shape[0], (
            "X should have columns equal to the number of rows in U"
        )
        assert X.shape[0] == fX.shape[0], (
            "The number of samples in the design set X should match the number of function estimations in fX"  # noqa: E501
        )
        assert fX.shape[1] == 1, (
            "The function estimates of the design set should be a column vector"
        )

        # get dimensions
        M, n = X.shape  # noqa: N806

        # find the residual
        Y = X @ U  # noqa: N806

        c = self.fit_coef(X, fX, U)  # shape(q,1)
        r = self.residual(X, fX, U)  # (M,1)

        #! FROM HERE THE FUNCTION NEEDS CHECKING
        # find the vandermonde matrix and derivative of the vandermonde matrix of the
        # projected design set
        V_matrix = self.V(Y)  # shape (M,q)  # noqa: N806
        DV_matrix = self.DV(Y)  # shape (M,q,n)  # noqa: N806

        M, q = V_matrix.shape  # noqa: N806

        try:
            Y, sig, ZT = scipy.linalg.svd(  # noqa: N806
                V_matrix, full_matrices=False, lapack_driver="gesvd"
            )
        except Exception as e:
            logging.warning("SVD failed in Jacobian computation with error: %s", e)
            #! Need to add fallback that isn't svd

        # s = np.array([np.inf if x == 0.0 else x for x in s])
        with np.errstate(divide="ignore", invalid="ignore"):
            D = np.diag(1.0 / sig)  # noqa: N806
            D[np.isinf(D)] = 0  # convert inf to 0 if desired

        J1 = np.zeros((M, n, self.d))  # noqa: N806
        J2 = np.zeros((q, n, self.d))  # noqa: N806

        # populate the Jacobian
        for k in range(self.d):
            for j in range(n):
                # This is the derivative of U
                DVDU_k = (  # noqa: N806
                    X[:, j, None] * DV_matrix[:, :, k]
                )  # shape (M,q)

                # first term in the Jacobian
                J1[:, j, k] = DVDU_k.dot(c).flatten()  # shape (M,)

                # second term of the Jacobian before V(U)^-
                J2[:, j, k] = DVDU_k.T.dot(r).flatten()  # shape of (M,)

        # project J1 against the range of V
        J1 -= np.tensordot(  # noqa: N806
            Y, np.tensordot(Y.T, J1, axes=(1, 0)), axes=(1, 0)
        )  # shape: (M,)

        # apply pseudo-inverse via SVD components
        J2_projected = np.tensordot(  # noqa: N806
            D, np.tensordot(ZT, J2, axes=(1, 0)), axes=(1, 0)
        )  # shape: (q, n, d)

        # combine terms to get full Jacobian
        return -(J1 + np.tensordot(Y, J2_projected, axes=(1, 0)))  # shape: (M, n, d)

    def fit_varpro(self, X: np.ndarray, fX: np.ndarray, U: np.ndarray) -> np.ndarray:  # noqa: N803
        """Runs a Gauss-Newton.

        Args:
                X (np.ndarray): design set of shape (M,n)
                fX (np.ndarray): corresponding function estimates of design points of
                shape (M,1)
                U (np.ndarray): The active subspace of shape (n,d)
                delta (float): radius of the current trust-region


        Returns:
                np.ndarray: The active subspace of shape (n,d)
        """

        def gn_solver(Jac: np.ndarray, residual: np.ndarray) -> np.ndarray:  # noqa: N803
            """An anonymous function to compute the Gauss-Newton step to find a descent.

            direction.

            Args:
                    Jac (np.ndarray): The Jacobian of the residual with respect to the
                    active subspace. It has shape (M,n,d)
                    residual (np.ndarray): The residual of the current model
                    approximation with shape (M,1).

            Returns:
                    np.ndarray: A vectorised form of the descent direction with shape
                    (nd,). The full descent direction has shape (n,d)
            """
            # Handle edge cases where residual or Jacobian are zero
            if np.all(residual == 0) and np.all(Jac == 0):
                return np.zeros(Jac.shape[1] * Jac.shape[2])

            if np.all(Jac == 0):
                raise ValueError("Jacobian is zero, cannot compute Gauss-Newton step.")

            if np.all(residual == 0):
                return np.zeros(Jac.shape[1] * Jac.shape[2])

            _M, _n, _d = Jac.shape  # noqa: N806
            Jac_vec = Jac.reshape(  # noqa: N806
                X.shape[0], -1
            )  # reshapes (M,n,d) to (M,nd)

            # compute short form SVD
            try:
                Y, sig, ZT = scipy.linalg.svd(  # noqa: N806
                    Jac_vec, full_matrices=False, lapack_driver="gesvd"
                )  # Y has shape (M,M), sig has shape (M,nd), and ZT has shape (nd,nd)
            except Exception as e:
                logging.warning("SVD failed with error: %s", e)
                Y, sig, ZT = scipy.linalg.svd(Jac_vec, full_matrices=False)  # noqa: N806
            # Find descent direction
            # Use more robust tolerance - either machine epsilon scaled by max singular
            # value
            # or absolute tolerance to handle cases where all singular values are small
            tol_relative = np.max(sig) * np.finfo(float).eps
            tol_absolute = 1e-12  # Absolute tolerance for very small singular values
            tol = max(tol_relative, tol_absolute)

            # Count and report ill-conditioning
            np.sum(sig < tol)
            (np.max(sig) / np.min(sig[sig > 0]) if np.any(sig > 0) else np.inf)

            # Compute safe inverse of singular values
            s_inv = np.where(sig > tol, 1.0 / sig, 0.0)

            # Compute Y^T r
            YTr = (Y.T @ residual).flatten()  # shape (M,)  # noqa: N806

            # Compute Delta_vec using safe inverse
            return -ZT.T @ (s_inv * YTr)  # shape (n*d,)

        def jacobian_variable_U(U):  # noqa: ANN001, ANN202, N802, N803
            return self.jacobian(X, fX, U)

        def residual_variable_U(U):  # noqa: ANN001, ANN202, N802, N803
            return self.residual(X, fX, U)

        return self.gauss_newton_solver(
            residual_variable_U, jacobian_variable_U, U, gn_solver
        )

    def gauss_newton_solver(
        self,
        residual: callable,
        jacobian: callable,
        U: np.ndarray,  # noqa: N803
        gn_solver: callable,
    ) -> np.ndarray:
        """Solves the Gauss_newton problem on the Grassmann manifold:.

                        vec(Delta) = -vec(Jac(U))^{+}r(U).

        Args:
                residual (callable): Function that takes the active subspace U of shape
                (n,d) and calculates the residual of the predicted model under a fixed
                design set. Returns a matrix of shape (M,1)
                jacobian (callable): Function that takes the active subspace U of shape
                (n,d) and calculates the Jacobian of the residual with respect to U.
                Returns a matrix of shape (M,n,d)
                U (np.ndarray): The subspace matrix with shape (n,d)
                gn_solver (callable): The Gauss-Newton step that returns the vectorised
                descent direction of shape (nd,)

        Returns:
                np.ndarray: A new active subspace matrix U_+ of shape (n,d)
        """
        # initial values for res and Jac and Grad
        max_iter = 100
        res = residual(U)  # shape (M,1)
        Jac = jacobian(U)  # shape (M,n,d)  # noqa: N806
        Grad = np.tensordot(res.ravel(), Jac, axes=(0, 0))  # (n,d)  # noqa: N806

        if np.all(Jac == 0) and np.all(res == 0):
            return U

        if np.all(Jac == 0):
            raise ValueError("Jacobian is zero, cannot compute Gauss-Newton step.")

        if np.all(res == 0):
            return U

        # Compute tolerances
        Grad_norm = norm(Grad)  # noqa: N806
        tol = max(1e-10 * Grad_norm, 1e-14)
        tol_Delta_norm = 1e-12  # noqa: N806
        # loop over linesearch until the norm of the gauss-newton step, the norm of the
        # Grad or the norm of Res(U) increases
        for _ in range(max_iter):
            residual_increased = False

            Jac.reshape(Jac.shape[0], -1)  # shape (M, nd)
            Delta_vec = gn_solver(Jac, res)  # shape (nd,)  # noqa: N806
            Delta = Delta_vec.reshape(  # noqa: N806
                Jac.shape[1], Jac.shape[2]
            )  # shape (n,d)

            # backtracking: find acceptable step gamma (t) along geodesic trajectory
            U_new, _step = self.backtracking(residual, Grad, Delta, U)  # noqa: N806

            res_candidate = residual(U_new)
            Jac_candidate = jacobian(U_new)  # noqa: N806
            Grad_candidate = np.tensordot(  # noqa: N806
                res_candidate.ravel(), Jac_candidate, axes=(0, 0)
            )

            if norm(res_candidate) >= norm(res):
                residual_increased = True
            else:
                # Update the residual, jacobian, Gradient, and active subspace
                res = res_candidate
                Jac = Jac_candidate  # noqa: N806
                Grad = Grad_candidate  # noqa: N806
                U = U_new  # noqa: N806

            # Termination Conditions
            if Grad_norm < tol or norm(Delta) < tol_Delta_norm or residual_increased:
                return U_new

        return U

    def backtracking(
        self,
        residual: callable,
        Grad: np.ndarray,  # noqa: N803
        delta: np.ndarray,
        U: np.ndarray,  # noqa: N803
    ) -> tuple[np.ndarray, float]:
        """Backtracking line search to satisfy the Armijo Condition:.

                        residual(U + alpha*delta) < residual(U) + alpha*beta*gamma
                        where:
                                - alpha is <Grad, delta>
                                - beta is a control parameter in (0,1)
                                - gamma is the backtracking coefficient.

        Args:
                residual (callable): Function that takes the active subspace U of shape
                (n,d) and calculates the residual of the predicted model under a fixed
                design set
                Grad (np.ndarray): Gradient of the active subspace matrix on the
                Grassmann manifold of shape (n,d)
                delta (np.ndarray): The Gauss-Newton step of shaoe (n,d)
                U (np.ndarray): The active subspace matrix with shape (n,d)

        Returns:
                tuple[np.ndarray, float]:
                        - The new active subspace matrix U of shape (n,d)
                        - The backtracking coefficient gamma  (gamma=1 implies no
                        backtracking)
        """
        # initialise control parameter, step shrink factor, and max iterations
        beta = 1e-4
        rho = 0.5
        max_iter = 100

        # directional derivative
        alpha = np.inner(
            Grad.reshape(
                -1,
            ),
            delta.reshape(
                -1,
            ),
        )  # vecGrad^T vec(delta) in matrix form

        # If direction is not a descent direction, flip to negative gradient
        if alpha >= 0:
            delta = -Grad
            alpha = np.inner(
                Grad.reshape(
                    -1,
                ),
                delta.reshape(
                    -1,
                ),
            )

        # starting objective and residual
        init_res = residual(U)

        step_size = 1.0
        for _ in range(max_iter):
            U_candidate = self.grassmann_trajectory(U, delta, step_size)  # noqa: N806
            res_candidate = residual(U_candidate)

            # Armijo condition: f(U + t delta) <= f(U) + t * beta * alpha
            if norm(res_candidate) <= norm(init_res) + step_size * beta * alpha:
                # success
                # Make sure U_new is orthonormal
                U_candidate, _ = np.linalg.qr(U_candidate)  # noqa: N806
                U_candidate = (  # noqa: N806
                    np.sign(np.diag(_)) * U_candidate
                )  # ensure consistent orientation
                return U_candidate, step_size

            # otherwise shrink step
            step_size *= rho

        # if not found, return the best we have (the last candidate)
        U_candidate = self.grassmann_trajectory(U, delta, step_size)  # noqa: N806
        # Make sure U_new is orthonormal
        U_candidate, _ = np.linalg.qr(U_candidate)  # noqa: N806
        U_candidate = (  # noqa: N806
            np.sign(np.diag(_)) * U_candidate
        )  # ensure consistent orientation
        return U_candidate, step_size

    def rotate_U(  # noqa: N802
        self,
        X: np.ndarray,  # noqa: N803
        fX: np.ndarray,  # noqa: N803
        coef: np.ndarray,
        U: np.ndarray,  # noqa: N803
    ) -> np.ndarray:
        """Rotates the active subspace matrix onto the most important direction of.

        Args:
                X (np.ndarray): design set of shape (M,n)
                fX (np.ndarray): corresponding function estimates of design points of
                shape (M,1)
                coef (np.ndarray): The coefficients of the local model of shape (q,1)
                U (np.ndarray): The active subspace of shape (n,d).

        Returns:
                np.ndarray: The rotated active subspace matrix of shape (n,d)
        """
        # Step 1: Apply active subspaces to the profile function at samples X
        # to rotate onto the most important directions
        if U.shape[1] > 1:
            grads = self.grad(X, coef, U)
            active_grads = grads
            # We only need the short-form SVD
            try:
                Ur = scipy.linalg.svd(active_grads.T, full_matrices=False)[0]  # noqa: N806
            except Exception as e:
                logging.warning("SVD failed with error: %s", e)
                #! need to add fallback that isn't svd

            U = U @ Ur  # noqa: N806

        # Step 2: Flip signs such that average slope is positive in the coordinate
        # directions
        coef = self.fit_coef(X, fX, U)
        grads = self.grad(X, coef, U)
        active_grads = grads  # shape (M,d)
        return U.dot(np.diag(np.sign(np.mean(active_grads, axis=0))))

    def grad(self, X: np.ndarray, coef: np.ndarray, U: np.ndarray) -> np.ndarray:  # noqa: N803
        """Computes the gradients of the local model at each design point of the design.

        set X.

        Args:
                X (np.ndarray): design set of shape (M,n) or (M,d)
                coef (np.ndarray): The coefficients of the local model of shape (q,1)
                U (np.ndarray): The active subspace of shape (n,d)
                delta (float): radius of the current trust-region.

        Returns:
                np.ndarray: The gradients of the model at each design point X shape
                (M,d) or (M,n)
        """
        if len(X.shape) == 1 or X.shape[0] == 1:
            one_d = True
            X = X.reshape(1, -1)  # noqa: N806
        else:
            one_d = False

        # Check if X is full-space (n dimensions) or reduced-space (d dimensions)
        if X.shape[1] == U.shape[0]:  # noqa: SIM108
            # Full-space input: project to reduced space
            Y = X @ U  # noqa: N806
        else:
            # Already in reduced space
            Y = X  # noqa: N806

        DV_matrix = self.DV(Y)  # shape (M,q,d)  # noqa: N806
        # Compute gradient on projected space
        Df = np.tensordot(DV_matrix, coef, axes=(1, 0))  # shape (M,d,1)  # noqa: N806
        # Inflate back to whole space
        Df = np.squeeze(Df, axis=-1)  # shape (M,d)  # noqa: N806

        if one_d:
            return Df.reshape(Y.shape[1])  # shape (d,)
        return Df  # shape (M,d)

    def hess(self, X: np.ndarray, coef: np.ndarray, U: np.ndarray) -> np.ndarray:  # noqa: N803
        """Computes the Hessian of the local model at each design point of the design.

        set X.

        Args:
                X (np.ndarray): design set of shape (M,n) or (M,d)
                coef (np.ndarray): The coefficients of the local model of shape (q,1)
                U (np.ndarray): The active subspace of shape (n,d).

        Returns:
                np.ndarray: The Hessians of the model at each design point X of shape
                (M,d,d)
        """
        if len(X.shape) == 1 or X.shape[0] == 1:
            one_d = True
            X = X.reshape(1, -1)  # noqa: N806
        else:
            one_d = False

        # Check if X is full-space (n dimensions) or reduced-space (d dimensions)
        if X.shape[1] == U.shape[0]:  # noqa: SIM108
            # Full-space input: project to reduced space
            Y = X @ U  # noqa: N806
        else:
            # Already in reduced space
            Y = X  # noqa: N806

        DDV_matrix = self.DDV(Y)  # shape (M,q,d,d)  # noqa: N806
        # Compute Hessian on projected space by contracting with coefficients
        # DDV_matrix has shape (M,q,d,d), coef has shape (q,1)
        Hf = np.tensordot(  # noqa: N806
            DDV_matrix, coef, axes=(1, 0)
        )  # shape (M,d,d,1)
        Hf = np.squeeze(Hf, axis=-1)  # shape (M,d,d)  # noqa: N806

        if one_d:
            return Hf.reshape(Y.shape[1], Y.shape[1])  # shape (d,d)
        return Hf  # shape (M,d,d)

    #! === VANDERMONDE CONSTRUCTION ===

    def scale(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        """Scale the design points using the basis adapter if provided.

        Args:
                X (np.ndarray): The design points to be scaled.

        Returns:
                np.ndarray: The scaled design points
        """
        if self.basis_adapter is None:
            return X
        return self.basis_adapter.scale(X)

    def dscale(self, d: int) -> np.ndarray:
        """Get the derivative scaling factors for the basis transformation.

        Args:
                d (int): The dimension of the space.

        Returns:
                np.ndarray: A 1D array of scaling factors for derivatives (length d).
        """
        if self.basis_adapter is None:
            return np.ones(d)
        return self.basis_adapter.dscale(d)

    def V(self, X: np.ndarray) -> np.ndarray:  # noqa: N802, N803
        """Generate the Vandermonde Matrix.

        Args:
                X (np.ndarray): The design set of shape (M,n) or (M,d) where n is the
                dimension of the original problem and d is the subspace dimension.

        Returns:
                np.ndarray: A vandermonde matrix of shape (M,q) where q is the length of
                the polynomial basis
        """
        M, d = X.shape  # noqa: N806

        # Check if using PolynomialBasis class adapter
        if isinstance(self.basis_adapter, PolynomialBasisClassAdapter):
            # Use the PolynomialBasis class's V method directly
            return self.basis_adapter.basis_instance.V(X)

        # Otherwise, use TensorBasis approach (original implementation)
        Xs = self.scale(X)  # noqa: N806
        indices = self.index_set(self.degree, d).astype(int)  # shape (q,d)
        q, d = indices.shape
        assert X.shape[1] == d, "Expected %d dimensions, got %d" % (d, X.shape[1])  # noqa: UP031

        V_coordinate = [self.vander(Xs[:, k], self.degree) for k in range(d)]  # noqa: N806

        V = np.ones((M, q), dtype=X.dtype)  # noqa: N806

        for k in range(d):
            Vk = V_coordinate[k][:, indices[:, k]]  # shape (M,q)  # noqa: N806
            V *= Vk  # element-wise multiplication broadcasting over M  # noqa: N806

        # for j, alpha in enumerate(indices):
        #     for k in range(d):
        #         V[:, j] *= V_coordinate[k][:, alpha[k]]
        return V

    def DV(self, X: np.ndarray) -> np.ndarray:  # noqa: N802, N803
        """Column-wise derivative of the Vandermonde matrix.

                Given design points this creates the Vandermonde-like matrix whose
                entries
                correspond to the derivatives of each of basis elements
        Args:
                X (np.ndarray): The design set of shape (M,d) where d is the subspace
                dimension.

        Returns:
                np.ndarray: Derivative of Vandermonde matrix  of shape (M,q,d) where
                DV[i,j,:] is the gradient of the
                partial derivative of the j-th basis function with respect to the x_k
                component of the d-dimensional vector
                and evaluated at i-th design point
        """
        M, d = X.shape  # noqa: N806

        # Check if using PolynomialBasis class adapter
        if isinstance(self.basis_adapter, PolynomialBasisClassAdapter):
            # Use the PolynomialBasis class's DV method directly
            return self.basis_adapter.basis_instance.DV(X)

        # Otherwise, use TensorBasis approach (original implementation)
        Xs = self.scale(X)  # noqa: N806
        indices = self.index_set(self.degree, d).astype(int)
        q, d = indices.shape

        V_coordinate = [self.vander(Xs[:, k], self.degree) for k in range(d)]  # noqa: N806
        Dmat = self.build_Dmat()  # noqa: N806
        self.dscale(d)

        V_deriv = [V_coordinate[k][:, 0:-1] @ Dmat.T for k in range(d)]  # noqa: N806
        base = np.ones((M, q), dtype=X.dtype)
        for k in range(d):
            Vk = V_coordinate[k][:, indices[:, k]]  # shape (M,q)  # noqa: N806
            base *= Vk  # element-wise multiplication broadcasting over M

        DV = np.ones((M, q, d), dtype=X.dtype)  # noqa: N806

        for k in range(d):
            ak = indices[:, k]  # shape (q,)
            pk = V_coordinate[k][:, ak]  # shape (M,q)
            pk_prime = V_deriv[k][:, ak]  # shape (M,q)

            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(np.abs(pk) > 1e-14, pk_prime / pk, 0.0)  # shape (M,q)

            DV[:, :, k] = (
                base * ratio
            )  # element-wise multiplication broadcasting over M

        # for k in range(d):
        #     for j, alpha in enumerate(indices):
        #         for q in range(d):
        #             if q == k:
        #                 DV[:, j, k] *= np.dot(
        #                     V_coordinate[q][:, 0:-1], Dmat[alpha[q], :]
        #                 )
        #             else:
        #                 DV[:, j, k] *= V_coordinate[q][:, alpha[q]]
        #         DV[:, j, k] *= dscale[k]

        return DV

    def build_Dmat(self) -> np.ndarray:  # noqa: N802
        """Constructs the (scalar) derivative matrix for polynomial basis up to.

        specified degree.

        Returns:
                np.ndarray: The derivative matrix.
        """
        Dmat = np.zeros((self.degree + 1, self.degree))  # noqa: N806
        I = np.eye(self.degree + 1)  # noqa: E741, N806
        for j in range(self.degree + 1):
            Dmat[j, :] = self.polyder(I[:, j])

        return Dmat

    def DDV(self, X: np.ndarray) -> np.ndarray:  # noqa: N802, N803
        """Column-wise second derivative of the Vandermonde matrix.

        Given design points this creates the Vandermonde-like matrix whose entries
        correspond to the second derivatives (Hessian) of each basis element.

        Args:
                X (np.ndarray): The design set of shape (M,d) where d is the subspace
                dimension.

        Returns:
                np.ndarray: Second derivative of Vandermonde matrix of shape (M,q,d,d)
                where
                DDV[i,j,k,l] is the mixed partial derivative of the j-th basis function
                with respect to x_k and x_l, evaluated at the i-th design point.
        """
        M, d = X.shape  # noqa: N806

        # Check if using PolynomialBasis class adapter
        if isinstance(self.basis_adapter, PolynomialBasisClassAdapter):
            # Use the PolynomialBasis class's DDV method directly if available
            if hasattr(self.basis_adapter.basis_instance, "DDV"):
                return self.basis_adapter.basis_instance.DDV(X)
            raise NotImplementedError(
                f"DDV not implemented for {type(self.basis_adapter.basis_instance)}"
            )

        # Otherwise, use TensorBasis approach (matching basis.py implementation)
        Xs = self.scale(X)  # noqa: N806

        indices = self.index_set(self.degree, d).astype(int)  # shape (q,d)
        q, d = indices.shape

        V_coordinate = [self.vander(Xs[:, k], self.degree) for k in range(d)]  # noqa: N806
        Dmat = self.build_Dmat()  # noqa: N806
        dscale = self.dscale(d)

        # 1d first-and second-derivative matrices per dimension
        V1 = [V_coordinate[k][:, 0:-1] @ Dmat.T for k in range(d)]  # noqa: N806
        V2 = []  # noqa: N806
        # Build second derivative matrices
        for k in range(d):
            V1_full = np.hstack(  # noqa: N806
                [V1[k], np.zeros((M, 1), dtype=V1[k].dtype)]
            )  # shape (M,degree+1)
            V2_k = V1_full[:, 0:-1].dot(  # noqa: N806
                Dmat
            )  # second derivative shape (M,degree)
            V2.append(V2_k)

        base = np.ones((M, q), dtype=Xs.dtype)
        for m in range(d):
            Vm = V_coordinate[m][:, indices[:, m]]  # shape (M,q)  # noqa: N806
            base *= Vm  # element-wise multiplication broadcasting over M

        DDV = np.ones((M, q, d, d), dtype=Xs.dtype)  # noqa: N806

        # fill DDV
        for k in range(d):
            ak = indices[:, k]  # shape (q,)
            pk = V_coordinate[k][:, ak]  # shape (M,q)
            p1k = V1[k][:, ak]  # first derivative shape (M,q)
            p2k = V2[k][:, ak]  # second derivative shape (M,q)

            with np.errstate(divide="ignore", invalid="ignore"):
                ratio1 = np.where(np.abs(pk) > 1e-14, p1k / pk, 0.0)  # shape (M,q)
                ratio2 = np.where(np.abs(pk) > 1e-14, p2k / pk, 0.0)  # shape (M,q)

            DDV[:, :, k, k] = base * ratio2 * (dscale[k] ** 2)  # diagonal terms

            for l in range(d):  # noqa: E741
                if l == k:
                    continue
                al = indices[:, l]  # shape (q,)
                pl = V_coordinate[l][:, al]  # shape (M,q)
                p1l = V1[l][:, al]  # first derivative shape (M,q)

                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio_l = np.where(np.abs(pl) > 1e-14, p1l / pl, 0.0)  # shape (M,q)

                DDV[:, :, k, l] = (
                    base * ratio1 * ratio_l * dscale[k] * dscale[l]
                )  # off-diagonal terms

        # for k in range(d):
        #     for ell in range(k, d):
        #         for j, alpha in enumerate(indices):
        #             for q in range(d):
        #                 if q == k == ell:
        #                     # Diagonal case: need second derivative w.r.t. x_k
        # # Use polyder with argument 2 to get second derivative directly
        #                     eq = np.zeros(self.degree + 1)
        #                     eq[alpha[q]] = 1.0
        #                     der2 = self.polyder(eq, 2)
        # DDV[:, j, k, ell] *= V_coordinate[q][:, 0:len(der2)].dot(der2)
        #                 elif q == k or q == ell:
        #                     # Off-diagonal: first derivative w.r.t. x_q
        #                     DDV[:, j, k, ell] *= np.dot(
        #                         V_coordinate[q][:, 0:-1], Dmat[alpha[q], :]
        #                     )
        #                 else:
        #                     # No derivative w.r.t. x_q
        #                     DDV[:, j, k, ell] *= V_coordinate[q][:, alpha[q]]

        #         # Apply scaling factors for both derivatives
        #         DDV[:, :, k, ell] *= dscale[k] * dscale[ell]
        #         # Exploit symmetry: Hessian is symmetric
        #         DDV[:, :, ell, k] = DDV[:, :, k, ell]

        return DDV

    # Module-level caches for index sets (class-level to persist across instances)
    _full_index_cache: ClassVar[dict[tuple[int, int], np.ndarray]] = {}
    _index_cache: ClassVar[dict[tuple[int, int], np.ndarray]] = {}

    def full_index_set(self, n: int, d: int) -> np.ndarray:
        """Enumerate multi-indices for a total degree of exactly `n` in `d` variables.

        Uses caching to avoid repeated recursive computation.

        Args:
                n (int): The total degree
                d (int): The number of variables

        Returns:
                np.ndarray: The multi-indices for the given total degree and number of
                variables
        """
        cache_key = (n, d)
        if cache_key in ASTROMORF._full_index_cache:
            return ASTROMORF._full_index_cache[cache_key]

        if d == 1:
            result = np.array([[n]])
        else:
            II = self.full_index_set(n, d - 1)  # noqa: N806
            m = II.shape[0]
            result = np.hstack((np.zeros((m, 1)), II))
            for i in range(1, n + 1):
                II = self.full_index_set(n - i, d - 1)  # noqa: N806
                m = II.shape[0]
                T = np.hstack((i * np.ones((m, 1)), II))  # noqa: N806
                result = np.vstack((result, T))

        ASTROMORF._full_index_cache[cache_key] = result
        return result

    def index_set(self, n: int, d: int) -> np.ndarray:
        """Enumerate multi-indices for a total degree of up to `n` in `d` variables.

        Uses caching to avoid repeated computation.

        Automatically switches to a diagonal Hessian model when the full polynomial
        basis would be underdetermined (more terms than design points). This enables
        using larger subspace dimensions without numerical issues.

        For degree n in d dimensions:
        - Full polynomial: C(d+n, n) terms
        - Diagonal model: 2d+1 terms (for degree 2)
        - Design points available: 2d+1

        Uses diagonal model when C(d+n, n) > 2d+1.

        Args:
                n (int): The maximum total degree
                d (int): The number of variables

        Returns:
                np.ndarray: The multi-indices for the given maximum total degree and
                number of variables
        """
        from math import comb

        # Calculate number of terms for full polynomial basis
        full_poly_terms = comb(d + n, n)
        design_points = 2 * d + 1

        # ONLY use diagonal Hessian for EXTREME cases where the system is
        # severely underdetermined (e.g., d=90 with 4186 terms vs 181 points).
        # For normal cases (d <= 10), we use the full polynomial with
        # pseudo-inverse/minimum-norm solution as the original algorithm did.
        #
        # Threshold: Only switch when polynomial terms exceed 10x design points.
        # This preserves original behavior for typical problems (d=1-9)
        # while preventing numerical issues for very large subspace dimensions.
        if n >= 2 and full_poly_terms > 10 * design_points:
            return self.index_set_diagonal(n, d)

        cache_key = (n, d)
        if cache_key in ASTROMORF._index_cache:
            return ASTROMORF._index_cache[
                cache_key
            ].copy()  # Return copy to prevent mutation

        I = np.zeros((1, d), dtype=np.int64)  # noqa: E741, N806
        for i in range(1, n + 1):
            II = self.full_index_set(i, d)  # noqa: N806
            I = np.vstack((I, II))  # noqa: E741, N806
        result = I[:, ::-1].astype(int)

        ASTROMORF._index_cache[cache_key] = result
        return result.copy()  # shape (num_terms, d)

    def index_set_diagonal(self, n: int, d: int) -> np.ndarray:
        """Generate index set for diagonal polynomial model.

        For degree n >= 2, this generates indices for:
        - Constant term: [0, 0, ..., 0]
        - Linear terms: [1, 0, ..., 0], [0, 1, 0, ..., 0], ..., [0, ..., 0, 1]
        - Diagonal quadratic terms: [2, 0, ..., 0], [0, 2, 0, ..., 0], ..., [0, ..., 0,
        2]

        This gives exactly 2d+1 terms, matching ASTRO-DF's model complexity.
        The model form is: f(x) = c_0 + Σ g_i * P_1(x_i) + Σ h_i * P_2(x_i)
        where P_k is the k-th degree polynomial from the chosen basis.

        Args:
            n (int): Maximum polynomial degree (must be >= 2 for diagonal quadratic)
            d (int): Number of variables (subspace dimension)

        Returns:
            np.ndarray: Multi-indices of shape (2d+1, d) for diagonal model
        """
        cache_key = ("diagonal", n, d)
        if cache_key in ASTROMORF._index_cache:
            return ASTROMORF._index_cache[cache_key].copy()

        # Start with constant term
        indices = [np.zeros(d, dtype=np.int64)]

        # Add linear terms (degree 1 in each variable)
        for i in range(d):
            idx = np.zeros(d, dtype=np.int64)
            idx[i] = 1
            indices.append(idx)

        # Add diagonal quadratic terms (degree 2 in each variable, no cross terms)
        if n >= 2:
            for i in range(d):
                idx = np.zeros(d, dtype=np.int64)
                idx[i] = 2
                indices.append(idx)

        result = np.array(indices, dtype=np.int64)
        ASTROMORF._index_cache[cache_key] = result
        return result.copy()


def clamp_with_epsilon(
    val: float, lower_bound: float, upper_bound: float, epsilon: float = 0.01
) -> float:
    """Clamp a value within bounds while avoiding exact boundary values.

    Adds a small epsilon to the lower bound or subtracts it from the upper bound
    if `val` lies outside the specified range.

    Args:
            val (float): The value to clamp.
            lower_bound (float): Minimum acceptable value.
            upper_bound (float): Maximum acceptable value.
            epsilon (float, optional): Small margin to avoid returning exact boundary
                    values. Defaults to 0.01.

    Returns:
            float: The adjusted value, guaranteed to lie strictly within the bounds.
    """
    if val <= lower_bound:
        return lower_bound + epsilon
    if val >= upper_bound:
        return upper_bound - epsilon
    return val


def duplication_check(x: np.ndarray, X: np.ndarray, tol: float = 1e-8) -> bool:  # noqa: N803
    """Check if a point x is a duplicate of any point in the set X within a given.

    tolerance.

    Args:
            x (np.ndarray): The point to check, shape (n,).
            X (np.ndarray): The set of existing points, shape (M,n).
            tol (float, optional): Tolerance for considering two points as duplicates.
                    Defaults to 1e-8.

    Returns:
            bool: True if x is a duplicate of any point in X, False otherwise.
    """
    return any(np.linalg.norm(x - xi) < tol for xi in X)


def degeneration_check(
    x_new: np.ndarray, incumbent_sol: np.ndarray, tol: float = 1e-8
) -> bool:
    """Check if a new point x_new is too close to the incumbent solution.

    Args:
            x_new (np.ndarray): The new point to check, shape (n,).
            incumbent_sol (np.ndarray): The incumbent solution point, shape (n,).
            tol (float, optional): Tolerance for considering two points as too close.
                    Defaults to 1e-8.

    Returns:
            bool: True if x_new is too close to incumbent_sol, False otherwise.
    """
    return np.linalg.norm(x_new - incumbent_sol) < tol


def backoff_step(
    incumbent_sol: np.ndarray,
    direction: np.ndarray,
    alpha: float,
    lb: np.ndarray,
    ub: np.ndarray,
    shrink: float = 0.5,
    min_alpha: float = 1e-8,
) -> np.ndarray | None:
    """Performs a backoff step from the incumbent solution in the given direction.

    Args:
            incumbent_sol (np.ndarray): The incumbent solution point, shape (n,).
            direction (np.ndarray): The direction to move away from the incumbent, shape
            (n,).
            alpha (float): Initial step size.
            lb (np.ndarray): Lower bounds for each dimension, shape (n,).
            ub (np.ndarray): Upper bounds for each dimension, shape (n,).
            shrink (float, optional): Factor to shrink the step size on each iteration.
                    Defaults to 0.5.
            min_alpha (float, optional): Minimum allowable step size before stopping.
                    Defaults to 1e-8.

    Returns:
            np.ndarray: The new point after backoff, shape (n,1).
    """
    while alpha > min_alpha:
        x_new_unbounded = incumbent_sol + alpha * direction  # shape (n,)

        # Project back into bounds
        x_new = np.array(
            [
                clamp_with_epsilon(x_comp, lb[i], ub[i])
                for i, x_comp in enumerate(x_new_unbounded.flatten().tolist())
            ]
        )

        if not degeneration_check(x_new, incumbent_sol, min_alpha):
            return x_new.reshape(-1, 1)
        alpha *= shrink

    # If no valid point found, return None
    return None


#! === BASIS WRAPPERS ===
def vander_wrapper(X, basis_instance):  # noqa: ANN001, ANN201, D103, N803
    if hasattr(basis_instance, "vander"):
        return basis_instance.vander(X)
    return None


def deriv_wrapper(coef, basis_instance):  # noqa: ANN001, ANN201, D103
    if hasattr(basis_instance, "polyder"):
        return basis_instance.polyder(coef)
    return None


#! === MODEL WRAPPERS ===
def model_evaluate_fn(x, coef, U, instance):  # noqa: ANN001, ANN201, D103, N803
    return instance.model_evaluate(x, coef, U)


def model_grad_fn(x, full_space, coef, U, instance):  # noqa: ANN001, ANN201, D103, N803
    if not full_space:
        return instance.grad(x, coef, U)
    return instance.grad(x, coef, U) @ U.T


def model_hess_fn(x, full_space, coef, U, instance):  # noqa: ANN001, ANN201, D103, N803
    if not full_space:
        return instance.hess(x, coef, U)
    return U @ instance.hess(x, coef, U) @ U.T
