"""Per-problem search spaces for the ASTROMoRF tuner.

Two responsibilities:

1.  Define the canonical hyperparameter search space (continuous, integer,
    categorical) for each problem, including the combinatorial guard that
    keeps polynomial-basis size from exploding at high subspace dim.
2.  Provide adaptive *narrowing* of continuous ranges around the best
    quantile of completed trials, with a guaranteed exploration floor so
    refinement never collapses the search distribution.

Search-space schema is a plain dict (not an Optuna distribution object) so
that the same definitions can be reused by the smoke-test runner, by the
confirmation pass, and by the report writer (which compares the picked
config to the *defined* space rather than to whatever Optuna sampled).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb
from typing import Any, Literal

# Polynomial bases retained per the user's decision (excludes
# MONOMIAL_POLY, LAGUERRE, LEGENDRE, LAGRANGE, NFP).
POLY_BASIS_CHOICES: tuple[str, ...] = (
    "HERMITE",
    "CHEBYSHEV",
    "MONOMIAL",
    "NATURAL",
)

# Hard cap on polynomial basis terms q = C(d+p, p). Trials violating this
# are pruned before we even instantiate the solver.
MAX_BASIS_TERMS: int = 5000

# Number of trials before adaptive narrowing kicks in. Below this we run
# the full original search space.
NARROW_AFTER_TRIALS: int = 50

# Quantile of completed trials whose hyperparameter values define the
# narrowed window. 0.25 == top-25%.
NARROW_QUANTILE: float = 0.25

# Minimum width retained after narrowing, as a fraction of the original
# range. Prevents collapse to a degenerate point and protects exploration.
NARROW_MIN_WIDTH_FRAC: float = 0.20

# Exploration probability: even after narrowing, sample from the full
# original space with this probability per continuous parameter.
NARROW_EXPLORATION_PROB: float = 0.20


# ── search-space primitives ────────────────────────────────────────────────


@dataclass(frozen=True)
class FloatParam:
    """Continuous hyperparameter on either a uniform or log-uniform scale."""

    name: str
    low: float
    high: float
    scale: Literal["uniform", "log"] = "uniform"


@dataclass(frozen=True)
class IntParam:
    """Integer hyperparameter on either a uniform or log-uniform scale."""

    name: str
    low: int
    high: int
    scale: Literal["uniform", "log"] = "uniform"


@dataclass(frozen=True)
class CategoricalParam:
    """Categorical hyperparameter; ``choices`` are stored as plain strings."""

    name: str
    choices: tuple[str, ...]


Param = FloatParam | IntParam | CategoricalParam


@dataclass(frozen=True)
class SearchSpace:
    """Per-problem search space."""

    problem_name: str
    problem_dim: int  # post-scaling dim used to evaluate (100 or native)
    use_scaling: bool  # if True, scale_dimension is invoked
    subspace: IntParam
    degree: CategoricalParam  # categorical over int strings
    polynomial_basis: CategoricalParam
    lambda_min: IntParam
    subproblem_regularisation: FloatParam
    ps_sufficient_reduction: FloatParam
    cabs: tuple[Param, ...]
    notes: str = ""

    def all_params(self) -> tuple[Param, ...]:
        """Return every Param in a canonical order (used by reports/CSV)."""
        return (
            self.subspace,
            self.degree,
            self.polynomial_basis,
            self.lambda_min,
            self.subproblem_regularisation,
            self.ps_sufficient_reduction,
            *self.cabs,
        )


# ── per-problem CABS space (shared default) ───────────────────────────────


def _default_cabs_params() -> tuple[Param, ...]:
    return (
        FloatParam("cabs_gamma", 0.80, 0.995, "uniform"),
        FloatParam("cabs_c_p", 0.01, 1.0, "log"),
        FloatParam("cabs_c_g", 0.05, 1.5, "log"),
        FloatParam("cabs_eps_n", 0.005, 0.5, "log"),
        FloatParam("cabs_eps_a", 0.005, 0.5, "log"),
        FloatParam("cabs_rho_max", 0.6, 0.99, "uniform"),
        IntParam("cabs_w_safe", 5, 30, "uniform"),
        FloatParam("cabs_eta_safe", 0.005, 0.25, "log"),
        FloatParam("cabs_c2_est", 0.25, 4.0, "log"),
        IntParam("cabs_delta_inc_cap", 1, 10, "uniform"),
    )


# Reduced CABS for PARAMESTI-1 (dim=2): keep parameters present so the
# search shape is consistent, but tighten ranges where they are essentially
# inert at this dim.
def _paramesti_cabs_params() -> tuple[Param, ...]:
    return (
        FloatParam("cabs_gamma", 0.85, 0.99, "uniform"),
        FloatParam("cabs_c_p", 0.05, 0.5, "log"),
        FloatParam("cabs_c_g", 0.1, 1.0, "log"),
        FloatParam("cabs_eps_n", 0.01, 0.5, "log"),
        FloatParam("cabs_eps_a", 0.01, 0.5, "log"),
        FloatParam("cabs_rho_max", 0.7, 0.95, "uniform"),
        IntParam("cabs_w_safe", 5, 20, "uniform"),
        FloatParam("cabs_eta_safe", 0.01, 0.2, "log"),
        FloatParam("cabs_c2_est", 0.5, 3.0, "log"),
        IntParam("cabs_delta_inc_cap", 1, 6, "uniform"),
    )


# ── per-problem definitions ────────────────────────────────────────────────


def _dim100_space(name: str, *, max_subspace: int = 50) -> SearchSpace:
    return SearchSpace(
        problem_name=name,
        problem_dim=100,
        use_scaling=True,
        subspace=IntParam("subspace_dim", 2, max_subspace, "uniform"),
        degree=CategoricalParam("polynomial_degree", ("1", "2", "3", "4")),
        polynomial_basis=CategoricalParam("polynomial_basis", POLY_BASIS_CHOICES),
        lambda_min=IntParam("lambda_min", 3, 30, "log"),
        subproblem_regularisation=FloatParam(
            "subproblem_regularisation", 1e-6, 0.5, "log"
        ),
        ps_sufficient_reduction=FloatParam(
            "ps_sufficient_reduction", 0.0, 1.0, "uniform"
        ),
        cabs=_default_cabs_params(),
    )


def _paramesti_space() -> SearchSpace:
    return SearchSpace(
        problem_name="PARAMESTI-1",
        problem_dim=2,  # native
        use_scaling=False,
        subspace=IntParam("subspace_dim", 1, 2, "uniform"),
        degree=CategoricalParam("polynomial_degree", ("1", "2", "3", "4")),
        polynomial_basis=CategoricalParam("polynomial_basis", POLY_BASIS_CHOICES),
        lambda_min=IntParam("lambda_min", 3, 30, "log"),
        subproblem_regularisation=FloatParam(
            "subproblem_regularisation", 1e-6, 0.5, "log"
        ),
        ps_sufficient_reduction=FloatParam(
            "ps_sufficient_reduction", 0.0, 1.0, "uniform"
        ),
        cabs=_paramesti_cabs_params(),
        notes=(
            "Native dim=2: subspace dim is structurally limited to {1, 2}; "
            "CABS is included for consistency but its effect is marginal."
        ),
    )


SPACES: dict[str, SearchSpace] = {
    "DYNAMNEWS-1": _dim100_space("DYNAMNEWS-1"),
    "SAN-1": _dim100_space("SAN-1"),
    "NETWORK-1": _dim100_space("NETWORK-1"),
    "ROSENBROCK-1": _dim100_space("ROSENBROCK-1"),
    "PARAMESTI-1": _paramesti_space(),
}


def get_space(problem_name: str) -> SearchSpace:
    """Return the canonical search space for *problem_name*."""
    if problem_name not in SPACES:
        raise KeyError(
            f"No search space registered for {problem_name!r}. "
            f"Known problems: {sorted(SPACES)}"
        )
    return SPACES[problem_name]


# ── combinatorial guard ────────────────────────────────────────────────────


def basis_term_count(subspace_dim: int, degree: int) -> int:
    """Number of monomials in a total-degree polynomial basis: C(d+p, p)."""
    return comb(int(subspace_dim) + int(degree), int(degree))


def is_combinatorially_feasible(
    subspace_dim: int, degree: int, *, cap: int = MAX_BASIS_TERMS
) -> tuple[bool, int]:
    """Return (feasible, q) where q is the basis term count."""
    q = basis_term_count(subspace_dim, degree)
    return (q <= cap, q)


# ── adaptive narrowing ─────────────────────────────────────────────────────


@dataclass
class NarrowedRanges:
    """Per-parameter override ranges produced by adaptive narrowing.

    Each entry maps an Optuna parameter name to a (low, high) pair (or to
    a tuple of categorical choices). Missing entries fall back to the
    original SearchSpace bounds.
    """

    floats: dict[str, tuple[float, float]] = field(default_factory=dict)
    ints: dict[str, tuple[int, int]] = field(default_factory=dict)
    categoricals: dict[str, tuple[str, ...]] = field(default_factory=dict)
    n_trials_used: int = 0

    def is_empty(self) -> bool:
        return not (self.floats or self.ints or self.categoricals)


def _quantile_range(
    values: list[float],
    q: float,
    *,
    lo_bound: float,
    hi_bound: float,
    min_width_frac: float,
) -> tuple[float, float]:
    if not values:
        return lo_bound, hi_bound
    import numpy as np

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return lo_bound, hi_bound
    # Use the inter-quantile range of the top-q% subset as the new window.
    cutoff = np.quantile(arr, q) if 0 < q < 1 else arr.min()
    elite = arr[arr <= cutoff] if q < 0.5 else arr  # by convention "top" = lower.
    if elite.size < 3:
        elite = arr
    lo = float(np.min(elite))
    hi = float(np.max(elite))
    # Enforce minimum width.
    full_width = hi_bound - lo_bound
    min_width = min_width_frac * full_width
    if (hi - lo) < min_width:
        center = 0.5 * (lo + hi)
        lo = center - 0.5 * min_width
        hi = center + 0.5 * min_width
    return max(lo_bound, lo), min(hi_bound, hi)


def compute_narrowed_ranges(
    space: SearchSpace,
    completed_trials: list[dict[str, Any]],
    *,
    quantile: float = NARROW_QUANTILE,
    min_width_frac: float = NARROW_MIN_WIDTH_FRAC,
) -> NarrowedRanges:
    """Build a NarrowedRanges from completed trials.

    *completed_trials* is a list of dicts with keys ``params`` (mapping
    Optuna param name -> value) and ``value`` (the objective). Lower
    objective is better.
    """
    if len(completed_trials) < NARROW_AFTER_TRIALS:
        return NarrowedRanges(n_trials_used=len(completed_trials))

    # Sort by objective ascending and pick the top-quantile slice.
    finite = [t for t in completed_trials if t.get("value") is not None]
    finite.sort(key=lambda t: t["value"])
    k = max(5, int(len(finite) * quantile))
    elite_trials = finite[:k]

    out = NarrowedRanges(n_trials_used=len(finite))
    for param in space.all_params():
        if isinstance(param, FloatParam):
            vals = [
                float(t["params"][param.name])
                for t in elite_trials
                if param.name in t["params"]
            ]
            if not vals:
                continue
            new_lo, new_hi = _quantile_range(
                vals,
                1.0,  # already pre-filtered to elite
                lo_bound=param.low,
                hi_bound=param.high,
                min_width_frac=min_width_frac,
            )
            out.floats[param.name] = (new_lo, new_hi)
        elif isinstance(param, IntParam):
            vals = [
                int(t["params"][param.name])
                for t in elite_trials
                if param.name in t["params"]
            ]
            if not vals:
                continue
            lo = max(param.low, min(vals))
            hi = min(param.high, max(vals))
            full = param.high - param.low
            if (hi - lo) < max(1, int(min_width_frac * full)):
                center = (lo + hi) // 2
                pad = max(1, int(0.5 * min_width_frac * full))
                lo, hi = max(param.low, center - pad), min(param.high, center + pad)
            out.ints[param.name] = (lo, hi)
        else:
            # Categoricals are NOT narrowed: Optuna's
            # ``CategoricalDistribution`` rejects any change to the choice
            # set after the first trial (``ValueError: CategoricalDistribution
            # does not support dynamic value space.``). TPE's internal
            # exploitation handles concentration on good choices on its
            # own, so this loses nothing important.
            continue
    return out


# ── warm-start trials (the previous "PROBLEM_OPTIMAL_HYPER" + cabs_factors) ─


_WARMSTART_RAW: dict[str, dict[str, Any]] = {
    "DYNAMNEWS-1": {
        "subspace_dim": 5, "polynomial_degree": "2",
        "polynomial_basis": "CHEBYSHEV", "lambda_min": 10,
        "subproblem_regularisation": 0.4156,
        "ps_sufficient_reduction": 0.1533,
        "cabs_gamma": 0.8751, "cabs_c_p": 0.4995, "cabs_c_g": 1.062,
        "cabs_eps_n": 0.1063, "cabs_eps_a": 0.1621, "cabs_rho_max": 0.9177,
        "cabs_w_safe": 13, "cabs_eta_safe": 0.1051, "cabs_c2_est": 0.8597,
        "cabs_delta_inc_cap": 10,
    },
    "SAN-1": {
        "subspace_dim": 20, "polynomial_degree": "3",
        "polynomial_basis": "CHEBYSHEV", "lambda_min": 10,
        "subproblem_regularisation": 0.3672,
        "ps_sufficient_reduction": 0.0,
        "cabs_gamma": 0.8829, "cabs_c_p": 0.05, "cabs_c_g": 0.8277,
        "cabs_eps_n": 0.0754, "cabs_eps_a": 0.0197, "cabs_rho_max": 0.7,
        "cabs_w_safe": 9, "cabs_eta_safe": 0.0128, "cabs_c2_est": 2.933,
        "cabs_delta_inc_cap": 6,
    },
    "ROSENBROCK-1": {
        "subspace_dim": 13, "polynomial_degree": "2",
        "polynomial_basis": "CHEBYSHEV", "lambda_min": 10,
        "subproblem_regularisation": 0.3132,
        "ps_sufficient_reduction": 0.2868,
        "cabs_gamma": 0.9099, "cabs_c_p": 0.1974, "cabs_c_g": 0.1,
        "cabs_eps_n": 0.4222, "cabs_eps_a": 0.3296, "cabs_rho_max": 0.8666,
        "cabs_w_safe": 22, "cabs_eta_safe": 0.0334, "cabs_c2_est": 1.999,
        "cabs_delta_inc_cap": 2,
    },
    "NETWORK-1": {
        "subspace_dim": 14, "polynomial_degree": "4",
        "polynomial_basis": "CHEBYSHEV", "lambda_min": 10,
        "subproblem_regularisation": 0.1307,
        "ps_sufficient_reduction": 0.046,
        "cabs_gamma": 0.8648, "cabs_c_p": 0.5403, "cabs_c_g": 0.6186,
        "cabs_eps_n": 0.3569, "cabs_eps_a": 0.2888, "cabs_rho_max": 0.8396,
        "cabs_w_safe": 27, "cabs_eta_safe": 0.181, "cabs_c2_est": 1.6545,
        "cabs_delta_inc_cap": 10,
    },
    "PARAMESTI-1": {
        "subspace_dim": 2, "polynomial_degree": "2",
        "polynomial_basis": "CHEBYSHEV", "lambda_min": 24,
        "subproblem_regularisation": 0.2204,
        "ps_sufficient_reduction": 0.7588,
        "cabs_gamma": 0.9349, "cabs_c_p": 0.409, "cabs_c_g": 0.4326,
        "cabs_eps_n": 0.4523, "cabs_eps_a": 0.2956, "cabs_rho_max": 0.7654,
        "cabs_w_safe": 20, "cabs_eta_safe": 0.0225, "cabs_c2_est": 3.0,
        "cabs_delta_inc_cap": 6,
    },
}


def warmstart_params(problem_name: str) -> dict[str, Any]:
    """Return a warm-start parameter dict shaped for ``study.enqueue_trial``."""
    return dict(_WARMSTART_RAW[problem_name])
