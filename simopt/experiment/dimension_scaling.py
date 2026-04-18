"""Generic dimension scaling for simulation optimization problems.

Provides a single ``scale_dimension`` function that works for **any** Problem
subclass by introspecting the default factor structure and automatically
scaling dimension-dependent factors.

Dimension-dependent factors are detected by comparing each factor's value
against the problem's default dimension:

* **Scalar ints** equal to the default dimension are treated as "dimension
  keys" (e.g. ``num_prod``, ``n_fac``, ``num_arcs``).
* **1-D sequences** (list / tuple) whose length equals the default dimension
  are resized by cycling existing values (or truncating).
* **2-D structures** are scaled in whichever dimension(s) match.
* **Probability distributions** (sequences summing to |approx| 1.0) are
  re-normalised after resizing.
* **Diagonal matrices** are regenerated at the new size, preserving the
  average diagonal value.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from pydantic import ValidationError

if TYPE_CHECKING:
    from simopt.base import Problem

__all__ = ["scale_dimension"]


def _get_instantiate_problem() -> Callable:
    """Lazy import to avoid circular dependency with experiment_base."""
    from simopt.experiment_base import instantiate_problem

    return instantiate_problem


# ── public API ────────────────────────────────────────────────────────────────


def scale_dimension(
    problem_name: str,
    budget: int,
    dimension: int | None = None,
) -> Problem:
    """Instantiate *any* problem scaled to the requested *dimension*.

    The function creates a default instance of the problem, inspects its
    factor values and default dimension, then builds new model and problem
    factor dicts whose dimension-dependent entries have been resized to
    *dimension*.  The result is validated through the normal Pydantic
    ``_validate_model`` path.

    If the problem-factor validators reject the scaled values (this happens
    when a config checks list lengths against a **hardcoded module constant**
    such as ``NUM_FACILITIES``), the function falls back to constructing with
    default problem factors and patching the scaled values onto the instance
    after construction.

    Args:
        problem_name: Abbreviated problem name (e.g. ``"FACSIZE-2"``).
        budget: Maximum number of replications for a solver to take.
        dimension: Target number of decision variables.  When *None* the
            problem's default dimension is used.

    Returns:
        A fully initialised :class:`Problem` instance.
    """
    make = _get_instantiate_problem()

    if dimension is None:
        return make(problem_name, {"budget": budget})

    # 1. Discover the baseline structure from a default instance.
    default_problem = make(problem_name)
    default_dim = default_problem.dim

    if dimension == default_dim:
        return make(problem_name, {"budget": budget})

    # 2. Scale model and problem factors.
    scaled_model = _scale_factors(
        dict(default_problem.model.factors), default_dim, dimension
    )
    scaled_problem = _scale_factors(
        dict(default_problem.factors), default_dim, dimension
    )
    scaled_problem["budget"] = budget

    # 3. Attempt full construction; fall back to post-init patching when
    #    problem-factor validators use hardcoded constants.
    problem = _instantiate_with_fallback(
        make, problem_name, scaled_problem, scaled_model, budget
    )

    # 4. Set an explicit dim override when the subclass supports it.
    try:
        problem.dim = dimension
    except (NotImplementedError, AttributeError):
        contextlib.suppress(NotImplementedError, AttributeError)

    return problem


# ── factor scaling ────────────────────────────────────────────────────────────


def _scale_factors(factors: dict, old_dim: int, new_dim: int) -> dict:
    """Return a copy of *factors* with dimension-dependent values rescaled.

    Scalar ints are only promoted to dimension keys when there is at least
    one list/tuple factor of matching length -- this avoids false positives
    (e.g. ``budget`` accidentally equalling the dimension).
    """
    has_dim_lists = any(
        isinstance(v, list | tuple) and len(v) == old_dim for v in factors.values()
    )

    return {
        key: (
            value
            if key == "budget"
            else _scale_value(value, old_dim, new_dim, scale_scalars=has_dim_lists)
        )
        for key, value in factors.items()
    }


def _scale_value(
    value: bool | int | float | list | tuple,
    old_dim: int,
    new_dim: int,
    *,
    scale_scalars: bool = True,
) -> bool | int | float | list | tuple:
    """Scale a single factor value when it appears dimension-dependent."""
    # Booleans are a subclass of int -- never touch them.
    if isinstance(value, bool):
        return value

    # Scalar int that equals the old dimension -> replace with new dimension.
    if scale_scalars and isinstance(value, int | np.integer) and value == old_dim:
        return int(new_dim)

    # 1-D sequence whose length matches the old dimension.
    if isinstance(value, list | tuple) and len(value) == old_dim and old_dim > 0:
        if value and isinstance(value[0], list | tuple):
            return _scale_2d(value, old_dim, new_dim)
        scaled = _resize(value, new_dim)
        if _is_probability_distribution(value):
            total = sum(scaled)
            if total > 0:
                scaled = [v / total for v in scaled]
        return _as_type(scaled, value)

    # 2-D where only the inner dimension matches.
    if (
        isinstance(value, list | tuple)
        and value
        and isinstance(value[0], list | tuple)
        and all(len(row) == old_dim for row in value)
    ):
        return _as_type([_as_type(_resize(row, new_dim), row) for row in value], value)

    return value


def _scale_2d(value: list | tuple, old_dim: int, new_dim: int) -> list | tuple:
    """Scale a 2-D factor (list-of-lists / tuple-of-tuples)."""
    inner_match = all(len(row) == old_dim for row in value)

    if inner_match and len(value) == old_dim:
        # Square matrix -- use a diagonal matrix to guarantee validity
        # (e.g. positive definiteness for covariance matrices).
        diag = [value[i][i] for i in range(old_dim)]
        avg = sum(diag) / len(diag) if diag else 1.0
        result = [
            [avg if i == j else 0.0 for j in range(new_dim)] for i in range(new_dim)
        ]
        return _as_type([_as_type(row, value[0]) for row in result], value)

    if inner_match:
        # Rectangular -- only inner dimension matches.
        return _as_type([_as_type(_resize(row, new_dim), row) for row in value], value)

    # Only outer dimension matches.
    return _as_type(_resize(value, new_dim), value)


# ── instantiation with fallback ──────────────────────────────────────────────


def _instantiate_with_fallback(
    make: Callable,
    problem_name: str,
    scaled_problem: dict,
    scaled_model: dict,
    budget: int,
) -> Problem:
    """Try progressively looser construction strategies.

    1. Full construction with all scaled factors.
    2. Scaled model + default problem factors (patched post-init).
    3. Default model + default problem factors (patched post-init).
    """
    # --- attempt 1: everything scaled ---
    try:
        return make(
            problem_name,
            problem_fixed_factors=scaled_problem,
            model_fixed_factors=scaled_model,
        )
    except (ValueError, ValidationError):
        pass

    # --- attempt 2: scaled model, default problem factors ---
    # Problem-factor validators may use hardcoded constants; patch after.
    try:
        problem = make(
            problem_name,
            problem_fixed_factors={"budget": budget},
            model_fixed_factors=scaled_model,
        )
        for key, value in scaled_problem.items():
            problem._factors[key] = value
        return problem
    except (ValueError, ValidationError):
        pass

    # --- attempt 3: default model, patch everything ---
    # Model-factor validators rejected the scaled values (e.g. graph
    # connectivity).  Build with defaults and patch both sides.
    problem = make(
        problem_name,
        problem_fixed_factors={"budget": budget},
    )
    for key, value in scaled_model.items():
        problem.model._factors[key] = value
    for key, value in scaled_problem.items():
        problem._factors[key] = value
    return problem


# ── helpers ──────────────────────────────────────────────────────────────────


def _resize(seq: list | tuple, new_len: int) -> list:
    """Resize *seq* to *new_len* by cycling or truncating."""
    n = len(seq)
    if n == 0:
        return []
    if new_len <= n:
        return list(seq[:new_len])
    return [seq[i % n] for i in range(new_len)]


def _as_type(scaled: list, original: list | tuple) -> list | tuple:
    """Convert *scaled* back to the container type of *original*."""
    return tuple(scaled) if isinstance(original, tuple) else scaled


def _is_probability_distribution(seq: list | tuple) -> bool:
    """True when *seq* looks like a probability distribution (sums to ~1)."""
    try:
        return (
            all(isinstance(v, int | float) for v in seq) and abs(sum(seq) - 1.0) < 1e-6
        )
    except (TypeError, ValueError):
        return False
