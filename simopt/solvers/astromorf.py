"""ASTROMoRF Solver.

The ASTROMoRF (Adaptive Sampling for Trust-Region Optimisation by Moving Ridge
Functions) progressively builds local models using interpolation on a reduced subspace
constructed through Active Subspace dimensionality reduction. The use of Active
Subspace reduction allows for a reduced number of interpolation points to be evaluated
in the model construction. This solver is particularly well-suited for high-dimensional
stochastic optimization problems where function evaluations are expensive.

Rees, Benjamin, Christine SM Currie, and Phan Tu Vuong. 
"ASTROMoRF: Adaptive Sampling Trust-Region Optimization with Dimensionality Reduction." 
2025 Winter Simulation Conference (WSC). IEEE, 2025.

"""

from __future__ import annotations

import logging
import math
import traceback
import warnings
from collections.abc import Callable
from enum import Enum
from functools import partial
from math import ceil, log
from typing import Annotated, ClassVar, Self, cast

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
from scipy.sparse.linalg import LinearOperator
from scipy.special import factorial

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
from simopt.problem import ProblemLike
from simopt.solver import BudgetExhaustedError
from simopt.solvers.active_subspaces.basis import (
    MonomialPolynomialBasis,
    NaturalPolynomialBasis,
    NFPTensorBasis,
)

warnings.filterwarnings("ignore")


#! === LAZY IDENTITY OPERATOR ===
def _identity_operator(n: int) -> LinearOperator:
    """Creates Identity Matrix LinearOperator.

    Creates a (n,n) LinearOperator that behaves like the identity 
    matrix for matvec, rmatvec, and matmat operations, but does not
    allocate memory for the full matrix. This is useful for cases where 
    we need an identity operator but want to avoid the memory overhead 
    of creating a dense identity matrix.

    Args:
        n (int): The rank of the square identity matrix

    Returns:
        LinearOperator: The lazy identity operator that behaves 
        like np.eye(n) for matvec, rmatvec, and matmat, but does 
        not allocate memory for the full matrix.
    """
    def matvec(x: np.ndarray) -> np.ndarray :
        return np.asarray(x)

    def rmatvec(x: np.ndarray) -> np.ndarray:
        return np.asarray(x)

    def matmat(x: np.ndarray) -> np.ndarray:
        return np.asarray(x)
    
    op = LinearOperator( 
        shape=(n, n),
        dtype=float,
        matvec=matvec, # ty: ignore[unknown-argument]
        rmatvec=rmatvec, # ty: ignore[unknown-argument]
        matmat=matmat # ty: ignore[unknown-argument]
    ) 
    op._is_identity = True  # type: ignore
    return op


def _is_identity(m: np.ndarray | LinearOperator | None) -> bool:
    """Checks if m is the identity operator.

    Checks for the presence of the _is_identity attribute, 
    which is set to True for the lazy identity operator created 
    by _identity_operator. This allows us to identify the lazy 
    identity operator without having to check its structure or
    perform expensive operations.

    Args:
        m (np.ndarray | LinearOperator | None): The matrix to check.

    Returns:
        bool: _description_
    """
    if m is None : 
        raise ValueError("Matrix m cannot be None")
    return getattr(m, "_is_identity", False)


def _materialise_dense(m: np.ndarray | LinearOperator | None, n: int) -> np.ndarray:
    """Return m as a dense (n, n) ndarray.

    If m is already a dense ndarray, return it as-is.  If m is the lazy identity
    operator, return np.eye(n).  If m is a LinearOperator, apply it to np.eye(n) 
    to get a dense matrix.  Otherwise, try to convert m to an ndarray

    Args:
        m (np.ndarray | LinearOperator | None): The matrix to materialise.
        n (int): The dimension of the identity matrix if 
        m is the lazy identity operator.

    Returns:
        np.ndarray: The materialised dense matrix.
    """
    if m is None:
        raise ValueError("Matrix m cannot be None")
    if isinstance(m, np.ndarray):
        return m
    if _is_identity(m):
        return np.eye(n)
    if isinstance(m, LinearOperator):
        return m @ np.eye(n)
    return np.asarray(m)


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


# ── CABS (Decomposed Cost-Aware Bandit Selector) defaults ─────────────────
CABS_DEFAULTS: dict = {
    "gamma": 0.95,     # discount factor for running stats
    "c_p": 0.25,       # UCB coefficient on pull-count bonus β_p
    "c_g": 0.5,        # UCB coefficient on accept-count bonus β_g
    "eps_n": 0.1,      # prior pseudo-pulls per dimension
    "eps_a": 0.1,      # prior pseudo-accepts per dimension
    "rho_max": 0.9,    # safe-dimension VP-convergence gate
    "w_safe": 10,      # window for safe-dimension acceptance rate
    "eta_safe": 0.05,  # acceptance floor for "safe"
    "c2_est": 1.0,     # initial per-accept gain prior G[d] ≈ C2 * A[d]
    "delta_inc_cap": 2,  # max single-step expansion
}


class CABSSelector:
    """Decomposed Cost-Aware Bandit Selector (CABS).

    Implements the decomposed-UCB dimension-selection rule. 
    For each candidate dimension d, we maintain
    discounted running statistics:

        N[d]  — pulls (total attempts using d)
        A[d]  — successful pulls (accepted steps using d)
        G[d]  — cumulative reward R_k^{cabs} = max(delta F,0)/radius**2 
        at accepted steps

    Decomposition:
        rho(d) = A[d] / N[d]              (acceptance probability)
        g(d) = G[d] / max(A[d], eps_a)   (expected gain given acceptance)

    Decomposed UCB index per §4:

        I_k^{dec}(d) =
            ( rho·g + rho beta_g + g beta_p ) / (2d + 1)
          where beta_p = c_p  sqrt(log(sum N) / N[d]),
                beta_g = c_g  sqrt(log(sum N) / max(A[d], eps_a))

    Uses a cold-arm optimistic-under-uncertainty prior.
    For a candidate d whose accept count A[d] has not yet exceeded
    ``2eps_a``, the gain estimate ĝ(d) is replaced by the maximum
    observed ĝ across "warm" arms, encouraging exploration of unproven
    dimensions until they accumulate enough acceptance evidence.

    Selection: restrict to a "safe" candidate set (cheap nearby dims plus
    any d already meeting rho_k < rho_max with r_k ≥ eta_safe), then pick
    argmax of I_k^{dec}.
    """

    def __init__(
        self,
        d_min: int,
        d_cap: int,
        gamma: float = CABS_DEFAULTS["gamma"],
        c_p: float = CABS_DEFAULTS["c_p"],
        c_g: float = CABS_DEFAULTS["c_g"],
        eps_n: float = CABS_DEFAULTS["eps_n"],  
        eps_a: float = CABS_DEFAULTS["eps_a"],  
        rho_max: float = CABS_DEFAULTS["rho_max"],
        w_safe: int = CABS_DEFAULTS["w_safe"],  
        eta_safe: float = CABS_DEFAULTS["eta_safe"],
        c2_est: float = CABS_DEFAULTS["c2_est"],  
        delta_inc_cap: int = CABS_DEFAULTS["delta_inc_cap"],
    ) -> None:
        """Create a CABSSelector instance.

        Args:
            d_min (int): smallest dimension to consider 
            d_cap (int): largest dimension to consider 
            gamma (float, optional): discount factor for running stats. 
            Defaults to CABS_DEFAULTS["gamma"].
            c_p (float, optional): UCB coefficient on pull-count bonus. 
            Defaults to CABS_DEFAULTS["c_p"].
            c_g (float, optional): UCB coefficient on accept-count bonus. 
            Defaults to CABS_DEFAULTS["c_g"].
            eps_n (float, optional): prior pseudo-pulls per dimension. 
            Defaults to CABS_DEFAULTS["eps_n"].
            eps_a (float, optional): prior pseudo-accepts per dimension. 
            Defaults to CABS_DEFAULTS["eps_a"].
            rho_max (float, optional): safe-dimension VP-convergence gate. 
            Defaults to CABS_DEFAULTS["rho_max"].
            w_safe (int, optional): window for safe-dimension acceptance rate. 
            Defaults to CABS_DEFAULTS["w_safe"].
            eta_safe (float, optional): acceptance floor for "safe". 
            Defaults to CABS_DEFAULTS["eta_safe"].
            c2_est (float, optional): initial per-accept gain prior. 
            Defaults to CABS_DEFAULTS["c2_est"].
            delta_inc_cap (int, optional): max single-step expansion. 
            Defaults to CABS_DEFAULTS["delta_inc_cap"].
        """
        self.d_min = int(d_min)
        self.d_cap = int(max(d_min, d_cap))
        self.gamma = float(gamma)
        self.c_p = float(c_p)
        self.c_g = float(c_g)
        self.eps_n = float(eps_n)
        self.eps_a = float(eps_a)
        self.rho_max = float(rho_max)
        self.w_safe = int(w_safe)
        self.eta_safe = float(eta_safe)
        self.delta_inc_cap = int(delta_inc_cap)

        # Prior-initialised stats per dimension (§7.3).
        self.N: dict[int, float] = {}
        self.A: dict[int, float] = {}
        self.G: dict[int, float] = {}
        for d in range(self.d_min, self.d_cap + 1):
            self.N[d] = self.eps_n
            self.A[d] = 0.5 * self.eps_n
            self.G[d] = float(c2_est) * 0.5 * self.eps_n

        # Per-dimension rolling acceptance window (for safe-set gate).
        self.accept_window: dict[int, list[bool]] = {
            d: [] for d in range(self.d_min, self.d_cap + 1)
        }
        # Per-dimension last observed rho_k (for safe-set gate).
        self.last_rho: dict[int, float] = dict.fromkeys(
            range(self.d_min, self.d_cap + 1), 0.0
            )

        self.last_signals: dict = {"d_selected": self.d_min, "source": "init"}

    def update(
        self,
        d_k: int,
        accepted: bool,
        reward: float,
        rho_k: float,
    ) -> None:
        """Apply discounted update for the pulled dimension d_k."""
        d_k = int(d_k)
        if d_k not in self.N:
            self.N[d_k] = self.eps_n
            self.A[d_k] = 0.5 * self.eps_n
            self.G[d_k] = 0.5 * self.eps_n
            self.accept_window[d_k] = []
            self.last_rho[d_k] = 0.0

        # Discount all dims (slow forgetting); add increment only at d_k.
        for d in self.N:
            self.N[d] *= self.gamma
            self.A[d] *= self.gamma
            self.G[d] *= self.gamma
        self.N[d_k] += 1.0
        if accepted:
            self.A[d_k] += 1.0
            self.G[d_k] += max(0.0, float(reward))

        win = self.accept_window[d_k]
        win.append(bool(accepted))
        if len(win) > self.w_safe:
            win.pop(0)
        self.last_rho[d_k] = float(rho_k)

    def _safe_candidates(self, d_k: int) -> list[int]:
        """Return the safe candidate set S(k) per §5.

        Always includes d_k and its neighbours (clamped); additionally any
        dimension whose windowed acceptance rate ≥ eta_safe and last rho_k <
        rho_max. Allows a moderate expansion step (matches FDSS delta_inc_cap).
        """
        nearby = {d_k, max(self.d_min, d_k - 1), min(self.d_cap, d_k + 1)}
        nearby.add(min(self.d_cap, d_k + self.delta_inc_cap))

        for d, win in self.accept_window.items():
            if len(win) == 0:
                continue
            r_bar = sum(1.0 for a in win if a) / len(win)
            if r_bar >= self.eta_safe and self.last_rho[d] < self.rho_max:
                nearby.add(d)
        return sorted(d for d in nearby if self.d_min <= d <= self.d_cap)

    def select(self, d_k: int) -> int:
        """Return d_{k+1} = argmax_{d ∈ S(k)} I_k^{dec}(d)."""
        candidates = self._safe_candidates(d_k)
        if not candidates:
            self.last_signals = {"d_selected": int(d_k), "source": "empty-safe-set"}
            return int(d_k)

        total_n = max(1.0, float(sum(self.N.values())))  
        n_log = math.log(1.0 + total_n)  

        # cold-arm optimistic-under-uncertainty ĝ prior.
        observed_g = [
            self.G[d_] / max(self.eps_a, self.A[d_])
            for d_ in self.N
            if self.N[d_] > 2.0 * self.eps_n
        ]
        g_max_observed = max(observed_g) if observed_g else 1.0

        best_d = candidates[0]
        best_idx = -math.inf
        scores: dict[int, float] = {}
        for d in candidates:
            n_d = max(self.eps_n, self.N[d])  
            a_d = max(self.eps_a, self.A[d])  
            g_d = max(0.0, self.G[d])  
            p_hat = min(1.0, self.A[d] / n_d)

            g_hat = float(g_max_observed) if self.A[d] < 2.0 * self.eps_a else g_d / a_d

            beta_p = self.c_p * math.sqrt(n_log / n_d)
            beta_g = self.c_g * math.sqrt(n_log / a_d)

            # cost_pen = 2.0 * d + 1.0
            # idx = (p_hat * g_hat + p_hat * beta_g + g_hat * beta_p) / cost_pen
            cost_pen = 2.0 * d + 1.0
            exploit = (p_hat * g_hat) / cost_pen
            idx = exploit + p_hat * beta_g + g_hat * beta_p
            scores[d] = idx
            if idx > best_idx:
                best_idx = idx
                best_d = d

        self.last_signals = {
            "d_selected": int(best_d),
            "source": "cabs-ucb",
            "candidates": candidates,
            "scores": scores,
        }
        return int(best_d)


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
            description=("trust-region radius increase rate " \
            "after very successful iteration"),  
        ),
    ]
    gamma_2: Annotated[
        float,
        Field(
            default=1.2,
            gt=1,
            description=("trust-region radius increase rate " \
            "after successful iteration"),
        ),
    ]
    gamma_3: Annotated[
        float,
        Field(
            default=0.5,
            gt=0,
            lt=1,
            description=("trust-region radius decrease rate " \
            "after unsuccessful iteration"),  
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
            description="adaptively adjust subspace dimension via the CABS selector",
            alias="adaptive subspace dimension",
        ),
    ]
    cabs_factors: Annotated[
        dict,
        Field(
            default=CABS_DEFAULTS,
            description="configuration factors for the CABS selector (if adaptive_subspace_dimension is True)",
            alias="CABS factors",
        ),
    ]

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
    def prev_H(self) -> np.ndarray | LinearOperator | None:  # noqa: N802
        """Get the previous model Hessian.

        Either a dense ndarray (when the elliptical-trust-region option is on
        and a real Hessian has been computed) or a lazy
        :class:`LinearOperator` returned by ``_identity_operator`` (during
        initialisation, after subspace-dim resets, or when the elliptical
        option is off).  Consumers that need a dense matrix call
        ``_materialise_dense`` at the point of use.
        """
        return self._prev_H

    @prev_H.setter
    def prev_H(self, value: np.ndarray | LinearOperator | None) -> None:  # noqa: N802
        """Set the previous model Hessian (ndarray or LinearOperator)."""
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
    def model_grad(self) -> Callable[..., np.ndarray]:
        """Get the model gradient function."""
        assert self._model_grad is not None, "model_grad not yet set"
        return self._model_grad

    @model_grad.setter
    def model_grad(self, value: Callable[..., np.ndarray] | None) -> None:
        """Set the model gradient function."""
        self._model_grad = value

    @property
    def model_hess(self) -> Callable[..., np.ndarray]:
        """Get the model Hessian function."""
        assert self._model_hess is not None, "model_hess not yet set"
        return self._model_hess

    @model_hess.setter
    def model_hess(self, value: Callable[..., np.ndarray] | None) -> None:
        """Set the model Hessian function."""
        self._model_hess = value

    @property
    def model(self) -> Callable[[np.ndarray], float]:
        """Get the model function."""
        assert self._model is not None, "model not yet set"
        return self._model

    @model.setter
    def model(self, value: Callable[[np.ndarray], float] | None) -> None:
        """Set the model function."""
        self._model = value

    def _set_basis(
        self,
        basis: PolyBasisType,
        _problem: ProblemLike | None = None,
    ) -> None:
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

    def set_basis(
        self,
        basis: PolyBasisType,
        problem: ProblemLike | None = None,
    ) -> None:
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
        self.d: int = int(
            max(
                1, min(self.factors["initial subspace dimension"], self.problem.dim)
            )
        )

        # Compute optimal polynomial degree based on subspace dimension
        # This ensures well-conditioned interpolation (points/terms ratio >= 0.60)
        self.degree = self.factors["polynomial degree"]

        if self.factors["Record Diagnostics"]:
            # ASTROMoRF only supports Problem (not MultistageProblem)
            self.diagnostics = ASTROMoRF_Diagnostics(self, cast(Problem, self.problem))

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
        self.d_history: list[int] = [self.d]
        self.max_d: int = self.problem.dim - 1
        self.initial_subspace_dimension: int = self.d

        # CABS bandit observables (published by compute_ratio / fit, consumed by
        # _apply_cabs_update). Default to neutral values for the first iteration.
        self._last_bandit_reward: float = 0.0
        self._last_rho_k: float = 0.0
        # Snapshot of self.incumbent_x before each iteration's accept/reject step;
        # CABS reads it to determine whether the step was accepted.
        self._x_before_iter: tuple[float, ...] | None = None

        # CABS selector (single adaptive-dimension rule).
        self.cabs_factors = self.factors["CABS factors"]
        self.cabs = CABSSelector(d_min=1, d_cap=self.max_d, **self.cabs_factors)
        self.cabs_log: list[dict] = []

        # Warm starting: store the active subspace from previous iteration
        self.prev_U = None

        # Store previous Hessian for ellipsoidal trust-region construction.
        # Start as a lazy identity -- materialised only at consumer sites that
        # cannot avoid a dense matrix (eigh, etc.).
        self.prev_H = _identity_operator(self.problem.dim)

        # Track last iteration when `d` changed (used by d_history bookkeeping).
        self.last_d_change_iteration: int = 0

        # create initializations of the model functions
        self.model = None
        self.model_grad = None
        self.model_hess = None

    def solve(self, problem: ProblemLike) -> None:
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

                # Snapshot incumbent x before the iteration's accept/reject step
                # so CABS can determine whether a step was accepted.
                self._x_before_iter = tuple(self.incumbent_x)

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

                # adaptive dimension update logic
                if self.factors.get("adaptive subspace dimension", False):
                    self.compute_optimal_subspace_dimension()

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

    def compute_optimal_subspace_dimension(self) -> int | None:
        """CABS dimension update: observe outcome of last iteration, then select d.

        Uses the Decomposed Cost-Aware Bandit Selector (CABS) with the
        optimistic-under-uncertainty (OFU) prior on ĝ for cold arms
        always enabled.

        Workflow:
            1. Determine acceptance from incumbent change vs. snapshot at iter start.
            2. Feed (d_k, accepted, reward, rho_k) into CABS.update().
            3. Query CABS.select() for the next d.
            4. If d changes, reset the basis and refresh prev_U/prev_H.
        """
        try:
            x_before = self._x_before_iter
            if x_before is None:
                accepted = False
            else:
                accepted = tuple(self.incumbent_x) != tuple(x_before)

            reward = float(getattr(self, "_last_bandit_reward", 0.0))
            rho_k = float(getattr(self, "_last_rho_k", 0.0))

            self.cabs.update(
                d_k=int(self.d), accepted=accepted, reward=reward, rho_k=rho_k
            )
            d_next = int(self.cabs.select(int(self.d)))

            self.cabs_log.append(
                {
                    "iteration": int(self.iteration_count),
                    "d_prev": int(self.d),
                    "d_next": int(d_next),
                    "accepted": bool(accepted),
                    "reward": reward,
                    "rho_k": rho_k,
                }
            )

            if d_next != int(self.d):
                print(f"[ADAPTIVE SUBSPACE]: Iteration {self.iteration_count}: {self.d} -> {d_next}")  # noqa: E501
                self.d = int(d_next)
                self.set_basis(self.factors["polynomial basis"], self.problem)
                self.prev_U = None
                self.prev_H = _identity_operator(self.problem.dim)
                self.last_d_change_iteration = int(self.iteration_count)
                self.d_history.append(int(self.d))
                return int(self.d)
        except Exception as e:
            logging.debug(f"CABS dimension update failed: {e}")
        return None

    # === TRUST-REGION METHODS ===

    def compute_trust_region(
        self,
        U: np.ndarray,  # noqa: N803
    ) -> tuple[Callable[[np.ndarray], float], np.ndarray | LinearOperator]:
        """Constructs the ellipsoidal trust-region based on the Hessian of the previous.

        model.

        Uses the stored prev_H from the previous iteration to define the trust-region
        geometry.

        Args:
            U (np.ndarray): The (n,d) active subspace matrix
        Returns:
            tuple[callable, np.ndarray | LinearOperator]:
                - The trust-region constraint function
                - The regularized Hessian matrix H used for the ellipsoid

        """
        n = self.problem.dim
        use_real_hess = (
            hasattr(self, "prev_H")
            and self.prev_H is not None
            and self.factors["elliptical trust region"]
            and not _is_identity(self.prev_H)
        )

        if use_real_hess:
            # Dense path: regularise prev_H via symmetric eigh.
            hess_matrix = _materialise_dense(self.prev_H, n).copy()
            hess_matrix = 0.5 * (hess_matrix + hess_matrix.T)  # symmetrise
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

        # Identity short-circuit: eigh(I) = (1, I); regularised result is I.
        # H_full stays a LinearOperator so downstream consumers can detect
        # the identity case and avoid further O(n^2)/O(n^3) work.  The
        # constraint reduces to ||x||^2 (full space) or ||U @ x||^2 (reduced).
        H_full = _identity_operator(n)  # noqa: N806

        def trust_region_constraint(x):  # noqa: ANN001, ANN202
            x = np.asarray(x).reshape(-1, 1)
            if x.shape[0] == U.shape[1]:
                Ux = U @ x  # noqa: N806
                return float(np.dot(Ux.ravel(), Ux.ravel()))
            return float(np.dot(x.ravel(), x.ravel()))

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
        """Solves the trust-region subproblem.

        Solves the subproblem of minimizing the surrogate model 
        within the trust region defined by the current active 
        subspace and trust-region radius. It applies regularization 
        to prevent null-space drift. 



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
        _tr_cons, h_full = self.compute_trust_region(U) 
        if _is_identity(h_full):
            hess = U.T @ U  
        else:
            h_full_arr = np.asarray(h_full, dtype=np.float64)
            if not np.all(np.isfinite(h_full_arr)):
                h_full_arr = np.eye(self.problem.dim)  
            # if not np.all(np.isfinite(H_full)):
            #     H_full = np.eye(self.problem.dim)  
            hess = U.T @ h_full_arr @ U  
        hess = 0.5 * (hess + hess.T)  # Ensure symmetry  
        if not np.all(np.isfinite(hess)):
            hess = np.eye(U.shape[1])  
        eigvals, eigvecs = np.linalg.eigh(hess)
        eigvals = np.maximum(eigvals, 1e-8)
        h_reduced = eigvecs @ np.diag(eigvals) @ eigvecs.T 

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
            val = (z.T @ h_reduced @ z).flatten().item()
            # The step is z, as we are optimizing from the origin in the reduced space
            if not np.isfinite(val):
                # If constraint blows up, fall back to spherical TR
                return (z.T @ z).flatten().item()
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
        assert self.visited_points is not None
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

        else:
            old_delta = self.delta
            self.delta: float = max(self.gamma_3 * self.delta, self.delta_min)

            self.unsuccessful_iterations.append(self.incumbent_solution)

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

        # CABS bandit reward: max(δF, 0) / Δ²
        try:
            delta_sq = float(self.delta) ** 2
            self._last_bandit_reward = (
                max(float(actual_improvement), 0.0) / delta_sq
                if delta_sq > 0.0
                else 0.0
            )
        except Exception:
            self._last_bandit_reward = 0.0

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
            assert self.visited_points is not None
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
        assert pilot_run is not None
        assert self.visited_points is not None
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
            assert pilot_run is not None
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
    def calculate_max_radius(self) -> float:
        """Calculate the maximum trust-region radius.
        
        Calculates the maximum trust-region radius based on the 
        problem's variable bounds and random sampling. This method 
        generates a number of random solutions within the problem's 
        bounds, computes the range of values for each variable across 
        these samples, and then determines the maximum range while 
        ensuring it does not exceed the overall variable bounds.

        Returns:
                float: The calculated maximum trust-region radius
        """
        rng = self.rng_list[1]
        n_samples = 256 + 32 * int(np.log2(max(self.problem.dim, 2)))

        samples = np.empty((n_samples, self.problem.dim), dtype=float)
        for k in range(n_samples):
            samples[k, :] = self.problem.get_random_solution(rng)

        coord_range = samples.max(axis=0) - samples.min(axis=0)  
        bound_span = self.problem.upper_bounds[0] - self.problem.lower_bounds[0]
        return float(np.minimum(coord_range, bound_span).max())

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
        n = self.problem.dim

        hess_source = (
            self.prev_H
            if self.factors["elliptical trust region"] and self.prev_H is not None
            else _identity_operator(n)
        )

        if _is_identity(hess_source):
            # Identity short-circuit: eigenvalues are all 1, so the per-axis
            # semi-axis length is just self.delta -- no eigh, no symmetrise.
            return [self.delta] * n

        H = _materialise_dense(hess_source, n)  # noqa: N806
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
    ) -> tuple[np.ndarray, int]:
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
            assert self.visited_points is not None
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
        assert self.visited_points is not None
        if len(self.visited_points) == 0:
            return 0, False

        candidates = []  # List of (index, full_space_dist, projected_dist)

        # Use ellipsoidal or spherical trust region in full space.
        # Short-circuit when prev_H is the lazy identity: the per-point
        # quadratic-form ``diff.T @ I @ diff`` collapses to ``||diff||^2``,
        # so we skip the symmetrise and the matrix product entirely.
        hess_source = (
            self.prev_H
            if self.factors["elliptical trust region"] and self.prev_H is not None
            else _identity_operator(self.problem.dim)
        )
        identity_tr = _is_identity(hess_source)
        if identity_tr:
            H_full = None  # noqa: N806  (unused on the fast path)
        else:
            H_full = _materialise_dense(hess_source, self.problem.dim)  # noqa: N806
            H_full = 0.5 * (H_full + H_full.T)  # noqa: N806

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
            if identity_tr:
                full_space_dist_sq = float(np.dot(diff_full.ravel(), diff_full.ravel()))
            else:
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
        dirn: np.ndarray | list[float],
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
        interpolation_solutions: list[Solution],
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
                tuple[np.ndarray, np.ndarray, list[Solution]]: Updated 
                interpolation points of shape (M,n), function 
                values of shape (M, 1), interpolation solutions
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
                    Updated interpolation points of shape (M, n),
                    function values of shape (M, 1),
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
            index: int = 0
            M: np.ndarray = np.zeros(1)  # noqa: N806
            if fX_improved.size > 0:
                Phi_X_improved = np.vstack(  # noqa: N806
                    [
                        phi_function(X_improved[i, :].reshape(-1, 1))
                        for i in range(X_improved.shape[0])
                    ]
                )
                M = np.absolute(Phi_X_improved @ v)  # noqa: N806
                index = int(np.argmax(M))
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

                # Fallback if x is still None after all attempts
                if x is None:
                    x = x_k.flatten()

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

                    assert x is not None
                    X[k, :] = x
                    soln_at_x = self.create_new_solution(tuple(x.ravel()), self.problem)
                    # Sample the newly generated interpolation point without
                    # re-evaluating the full design set
                    pilot_run = self.calculate_pilot_run(construct_model=True)
                    assert pilot_run is not None
                    soln_at_x = self.perform_adaptive_sampling(
                        soln_at_x, pilot_run, delta
                    )
                    f_value = (
                        -1 * self.problem.minmax[0] * soln_at_x.objectives_mean.item()
                    )
                    fX[k, 0] = f_value
                    assert self.visited_points is not None
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
                        assert pilot_run is not None
                        soln_at_s = self.perform_adaptive_sampling(
                            soln_at_s, pilot_run, delta
                        )
                        f_value = (
                            -1
                            * self.problem.minmax[0]
                            * soln_at_s.objectives_mean.item()
                        )
                        fX[k, 0] = f_value
                        assert self.visited_points is not None
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
                    assert pilot_run is not None
                    soln_at_x = self.perform_adaptive_sampling(
                        soln_at_x, pilot_run, delta
                    )
                    f_value = (
                        -1 * self.problem.minmax[0] * soln_at_x.objectives_mean.item()
                    )
                    fX[k, 0] = f_value
                    assert self.visited_points is not None
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
        I = np.zeros((1, max(1, dimensions)))  # noqa: E741, N806
        if dimensions == 1:
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
    ) -> tuple[Callable[..., np.ndarray], Callable[..., np.ndarray]]:
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
        phi_function: Callable[..., np.ndarray],
        phi_function_deriv: Callable[..., np.ndarray],
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
        phi_function: Callable[..., np.ndarray],
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
    ) -> tuple[np.ndarray, list[float], list[Solution], np.ndarray, np.ndarray]:
        """Builds a local approximation of the response surface within the current.

        trust.

        region (defined as ||x-x_k||<=delta).

                The method fit recovers the local approximation given a design set of
                2d+1 design points and a corresponding active subspace U of shape (n,d)
                That projects the n-dimensional design points to a d-dimensional
                subspace.

        Returns:
                tuple[callable, callable, np.ndarray, list[float], list[Solution]]:
                        - The active subspace of shape (n,d)
                        - The flattened function estimates of the objective function at
                        each of the final design points as a list of floats
                        - The design points as solution objects
                        - The design set after going through fitting of shape (M,n)
                        - The function estimates of the objective function
                        at each of the final design points of shape (M,1)
        """
        # Reset scaling parameters for this new model construction
        reset_fn = getattr(self.basis_adapter, "reset_scaling", None)
        if reset_fn is not None:
            reset_fn()

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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Solution]]:
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
                tuple[np.ndarray, np.ndarray, np.ndarray, list[Solution]]:
                        - The final computed active subspace of the iteration of shape
                        (n,d)
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

        # Snapshot of initial U for VP-convergence gate rho_k computation
        U0_snapshot = np.copy(U0)  # noqa: N806

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
            # Lazy identity -- consumers materialise via _materialise_dense
            # only when an op actually requires a dense matrix.
            self.prev_H = _identity_operator(self.problem.dim)

        # CABS VP-convergence gate rho_k = ‖r_final‖² / ‖r_initial‖²
        try:
            r_initial = self.residual(X, fX, U0_snapshot)
            r_final = self.residual(X, fX, U)
            num = float(np.sum(r_final**2))
            den = float(np.sum(r_initial**2))
            self._last_rho_k = num / den if den > 1e-30 else 0.0
        except Exception:
            self._last_rho_k = 0.0

        if self.delta != model_delta:
            self.delta = float(
                min(max(self.delta, beta * norm(self.grad(X, coef, U))), model_delta)
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
                    model_delta = float(self.delta * w**fitting_iter)
                    fitting_iter += 1

            elif model_delta > max(self.mu * gnorm, 1e-12):
                model_delta = float(
                    min(model_delta, max(self.delta_min, self.mu * gnorm, 1e-12))
                )
                X, fX, interpolation_solutions = self.improve_geometry(  # noqa: N806
                    model_delta, U, X, fX, interpolation_solutions
                )
                fitting_iter += 1
            else:
                break

        return coef, float(model_delta), X, fX, interpolation_solutions

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
            Y, sig, ZT = scipy.linalg.svd(V_matrix, full_matrices=False)  # noqa: N806

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
        residual: Callable[..., np.ndarray],
        jacobian: Callable[..., np.ndarray],
        U: np.ndarray,  # noqa: N803
        gn_solver: Callable[..., np.ndarray],
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
        residual: Callable[..., np.ndarray],
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
                Ur = np.eye(U.shape[1])  # noqa: N806

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
            assert self.basis_adapter.basis_instance is not None
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
            assert self.basis_adapter.basis_instance is not None
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

    def build_Dmat2(self) -> np.ndarray:  # noqa: N802
        """Second-derivative coefficient matrix for the polynomial basis.

        Row ``j`` is ``polyder(polyder(e_j))`` -- the monomial-basis
        coefficients of ``v''_j`` for the j-th basis polynomial.  The result
        has shape ``(degree+1, max(degree-1, 1))``.

        Used **only** in :meth:`DDV` to compute second derivatives correctly
        for the elliptical trust-region Hessian; the prior implementation
        attempted a clever-but-wrong reuse of :meth:`build_Dmat` that both
        produced incorrect values and an off-by-one shape mismatch (V2 was
        ``(M, degree)`` instead of ``(M, degree+1)``, causing
        ``IndexError`` whenever ``alpha_k = degree`` appeared in the
        multi-index set).
        """
        cols = max(self.degree - 1, 1)
        Dmat2 = np.zeros((self.degree + 1, cols))  # noqa: N806
        I = np.eye(self.degree + 1)  # noqa: E741, N806
        for j in range(self.degree + 1):
            d2 = self.polyder(self.polyder(I[:, j]))
            length = min(len(d2), cols)
            if length > 0:
                Dmat2[j, :length] = d2[:length]
        return Dmat2

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

        # 1-D first-derivative Vandermonde per coordinate.
        # V1[k] has shape (M, degree+1); column j is v'_j evaluated at x_k.
        V1 = [V_coordinate[k][:, 0:-1] @ Dmat.T for k in range(d)]  # noqa: N806

        # 1-D second-derivative Vandermonde per coordinate.
        # V2[k] must have shape (M, degree+1); column j is v''_j(x_k).
        # The previous implementation produced (M, degree) AND computed the
        # wrong quantity; see :meth:`build_Dmat2` for the diagnosis.
        if self.degree < 2:
            # All second derivatives of a degree<2 polynomial are zero.
            V2 = [  # noqa: N806
                np.zeros((M, self.degree + 1), dtype=Xs.dtype) for _ in range(d)
            ]
        else:
            Dmat2 = self.build_Dmat2()  # shape (degree+1, degree-1)  # noqa: N806
            cols = Dmat2.shape[1]
            # V2_k[m, j] = sum_l V_coord[k][m, l] * Dmat2[j, l]
            #           = (V_coord[k][:, :cols] @ Dmat2.T)[m, j]
            # which is exactly v''_j(x_k) for j in 0..degree.
            V2 = [  # noqa: N806
                V_coordinate[k][:, :cols] @ Dmat2.T for k in range(d)
            ]

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
        return DDV

    # Module-level caches for index sets (class-level to persist across instances)
    _full_index_cache: ClassVar[dict[tuple[int, int], np.ndarray]] = {}
    _index_cache: ClassVar[
        dict[tuple[int, ...] | tuple[str, int, int], np.ndarray]
    ] = {}

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


def duplication_check(
    x: np.ndarray | list,
    x_set: np.ndarray | list,
    tol: float = 1e-8,
) -> bool:
    """Check if a point x is a duplicate of any point in the set X within a given.

    tolerance.

    Args:
            x (np.ndarray): The point to check, shape (n,).
            x_set (np.ndarray): The set of existing points, shape (M,n).
            tol (float, optional): Tolerance for considering two points as duplicates.
                    Defaults to 1e-8.

    Returns:
            bool: True if x is a duplicate of any point in x_set, False otherwise.
    """
    return any(np.linalg.norm(x - xi) < tol for xi in x_set)


def degeneration_check(
    x_new: np.ndarray | list | tuple,
    incumbent_sol: np.ndarray | list | tuple,
    tol: float = 1e-8,
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
    return bool(np.linalg.norm(np.asarray(x_new) - np.asarray(incumbent_sol)) < tol)


def backoff_step(
    incumbent_sol: np.ndarray | list | tuple,
    direction: np.ndarray,
    alpha: float,
    lb: np.ndarray | list | tuple,
    ub: np.ndarray | list | tuple,
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
