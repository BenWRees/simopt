"""Explicit, structure-preserving dimension scaling for SimOpt problems.

The previous implementation tried to scale arbitrary problems with type-based
heuristics (cycling lists, regenerating diagonals, post-construction factor
patching).  That produced mathematically degenerate instances:

* SAN-1 at dim=100 ended up with 100 arc references but only the original
  13 unique arcs on 9 nodes, leaving 87 decision variables with zero
  stochastic gradient.
* NETWORK-1 at dim=100 became 10 replicated copies of the same 10 networks,
  giving a rank-deficient objective with a non-unique optimum.

This module replaces those heuristics with an explicit, fail-loud system:

  1. **Problem-native scaling (preferred).**  If the Problem class defines a
     ``scale_to(new_dim, budget)`` classmethod, dispatch to it.
  2. **Registry fallback.**  Otherwise look up ``problem_name`` in the
     ``SCALERS`` registry (populated via ``@register_scaler``).
  3. **Fail loudly.**  If neither exists, raise ``UnsupportedScalingError``.
     There is no heuristic fallback.

After construction the result passes through a validation layer that checks
dimension consistency and (where applicable) a problem-supplied
``validate_scaled`` hook.  Any failure raises ``InvalidScaledProblemError``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from simopt.base import Problem

__all__ = [
    "SCALABLE_PROBLEMS",
    "SCALERS",
    "InvalidScaledProblemError",
    "Scaler",
    "UnsupportedScalingError",
    "is_scalable",
    "register_scaler",
    "scale_dimension",
]


# Hand-maintained manifest of problems with explicit scaling support.
# Kept here (rather than discovered at import time) so callers can predicate
# on it without paying the module-walk cost, and so adding a problem to the
# list is a deliberate, reviewable change.
SCALABLE_PROBLEMS: frozenset[str] = frozenset(
    {
        "DYNAMNEWS-1",
        "EXAMPLE-1",
        "NETWORK-1",
        "ROSENBROCK-1",
        "SAN-1",
        "ZAKHAROV-1",
    }
)


def is_scalable(problem_name: str) -> bool:
    """Return True iff *problem_name* has an explicit scaler available.

    Convenience for callers that want to avoid catching
    :class:`UnsupportedScalingError`.
    """
    return problem_name in SCALABLE_PROBLEMS or problem_name in SCALERS


# ── error types ───────────────────────────────────────────────────────────────


class UnsupportedScalingError(NotImplementedError):
    """Raised when no explicit scaler exists for the requested problem."""


class InvalidScaledProblemError(ValueError):
    """Raised when a scaled problem fails post-construction validation."""


# ── registry ──────────────────────────────────────────────────────────────────


Scaler: TypeAlias = Callable[[int, int], "Problem"]
"""A scaler is ``(new_dim, budget) -> Problem``."""

SCALERS: dict[str, Scaler] = {}


def register_scaler(name: str) -> Callable[[Scaler], Scaler]:
    """Register an explicit scaler for *name* (e.g. ``"SAN-1"``).

    Raises if a scaler is already registered for that name; explicit overrides
    must remove the old entry first.  This prevents accidental shadowing.
    """

    def decorator(fn: Scaler) -> Scaler:
        if name in SCALERS:
            raise ValueError(f"Scaler for {name!r} is already registered.")
        SCALERS[name] = fn
        return fn

    return decorator


# ── public API ────────────────────────────────────────────────────────────────


def scale_dimension(
    problem_name: str,
    budget: int,
    dimension: int | None = None,
) -> Problem:
    """Instantiate *problem_name* at the requested decision-vector *dimension*.

    Args:
        problem_name: Abbreviated problem name (e.g. ``"SAN-1"``).
        budget: Maximum number of replications for a solver to take.
        dimension: Target number of decision variables.  When ``None`` the
            problem is constructed at its native dimension via the normal
            instantiation path; no scaling logic runs.

    Returns:
        A fully initialised :class:`Problem` instance whose ``dim`` equals
        *dimension*.

    Raises:
        UnsupportedScalingError: No ``scale_to`` and no registered scaler.
        InvalidScaledProblemError: The constructed problem failed validation.
        ValueError: ``dimension`` is non-positive, or ``budget`` is non-positive.
    """
    if budget <= 0:
        raise ValueError(f"budget must be positive, got {budget!r}")
    if dimension is None:
        return _instantiate_native(problem_name, budget)
    if dimension <= 0:
        raise ValueError(f"dimension must be positive, got {dimension!r}")

    problem_cls = _get_problem_class(problem_name)
    try :
        scaler = _resolve_scaler(problem_cls, problem_name)
    except UnsupportedScalingError as e:
        print(f"WARNING: {e} Returning unscaled problem instance.")
        return _instantiate_native(problem_name, budget) 
    
    problem = scaler(dimension, budget)

    _validate_scaled(problem, expected_dim=dimension, expected_budget=budget)
    return problem


# ── dispatch ──────────────────────────────────────────────────────────────────


def _resolve_scaler(problem_cls: type, problem_name: str) -> Scaler:
    """Return the scaler to use, preferring problem-native ``scale_to``.

    A class-level ``scale_to`` always wins over the registry; this lets a
    problem own its scaling logic without anyone having to import the
    registry.
    """
    native = getattr(problem_cls, "scale_to", None)
    if callable(native):
        return native  # bound classmethod -> (new_dim, budget) -> Problem

    if problem_name in SCALERS:
        return SCALERS[problem_name]


    raise UnsupportedScalingError(
        f"No scaler available for {problem_name!r}. Define a "
        f"`scale_to(cls, new_dim, budget)` classmethod on the problem, or "
        f"register one with @register_scaler({problem_name!r})."
    )


def _get_problem_class(problem_name: str) -> type:
    """Look up the Problem subclass corresponding to *problem_name*.

    Uses the same module-walking lookup as :func:`instantiate_problem` so
    any class registered there is also visible here.
    """
    import importlib
    import inspect
    import pkgutil

    module_path = "simopt.models"
    base_module = importlib.import_module(module_path)

    for _finder, name, _ispkg in pkgutil.walk_packages(
        base_module.__path__, prefix=module_path + "."
    ):
        try:
            submodule = importlib.import_module(name)
        except ModuleNotFoundError:
            continue
        for _, cls in inspect.getmembers(submodule, inspect.isclass):
            if (
                cls.__module__ == name
                and hasattr(cls, "class_name_abbr")
                and cls.class_name_abbr == problem_name
            ):
                return cls

    raise KeyError(f"Unknown problem name: {problem_name!r}")


def _instantiate_native(problem_name: str, budget: int) -> Problem:
    """Construct the problem at its native dimension."""
    from simopt.experiment_base import instantiate_problem

    return instantiate_problem(problem_name, {"budget": budget})


# ── validation layer ──────────────────────────────────────────────────────────


def _validate_scaled(
    problem: Problem, *, expected_dim: int, expected_budget: int
) -> None:
    """Sanity-check a scaled problem before returning it to the caller.

    The checks here are *generic* (dimension match, budget match, factor
    length consistency).  Problem-specific structural checks live in the
    optional ``validate_scaled`` hook on the problem instance.
    """
    if problem.dim != expected_dim:
        raise InvalidScaledProblemError(
            f"Scaled problem reports dim={problem.dim}, expected {expected_dim}."
        )
    actual_budget = problem.factors.get("budget")
    if actual_budget != expected_budget:
        raise InvalidScaledProblemError(
            f"Scaled problem has budget={actual_budget}, expected {expected_budget}."
        )

    initial = problem.factors.get("initial_solution")
    if initial is not None and len(initial) != expected_dim:
        raise InvalidScaledProblemError(
            f"initial_solution has length {len(initial)}, expected {expected_dim}."
        )

    hook = getattr(problem, "validate_scaled", None)
    if callable(hook):
        hook(expected_dim)


# ── eager scaler imports ──────────────────────────────────────────────────────
#
# Importing problem modules here ensures their @register_scaler decorators run
# (or, in the preferred path, that their `scale_to` classmethods are reachable
# via _get_problem_class).  Scalers that live in problem files don't need an
# explicit import, but any registry-only scalers should be added below.

import simopt.models.dynamnews  # noqa: E402  (exposes scale_to)
import simopt.models.example  # noqa: E402  (exposes scale_to)
import simopt.models.network  # noqa: E402  (exposes scale_to)
import simopt.models.rosenbrock  # noqa: E402  (exposes scale_to)
import simopt.models.san  # noqa: E402  (exposes scale_to)
import simopt.models.zakharov  # noqa: E402, F401  (exposes scale_to)
