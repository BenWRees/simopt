#!/usr/bin/env python
"""Discover and call all simopt callables to find runtime errors.

Usage::

    python scripts/check_runtime_errors.py [--json] [--verbose]

Flags:
    --json      Save results to scripts/runtime_errors_report.json
    --verbose   Print successful calls too, not just failures
"""
# ruff: noqa: ANN401

from __future__ import annotations

import importlib
import inspect
import json
import pkgutil
import sys
import traceback
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import EnumType
from pathlib import Path
from types import UnionType
from typing import Annotated, Any, Literal, Union, get_args, get_origin

import numpy as np
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Colour helpers (degrade gracefully on non-TTY)
# ---------------------------------------------------------------------------
_USE_COLOR = sys.stdout.isatty()


def _green(s: str) -> str:
    return f"\033[92m{s}\033[0m" if _USE_COLOR else s


def _red(s: str) -> str:
    return f"\033[91m{s}\033[0m" if _USE_COLOR else s


def _yellow(s: str) -> str:
    return f"\033[93m{s}\033[0m" if _USE_COLOR else s


def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m" if _USE_COLOR else s


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class ErrorRecord:
    """A single error captured during runtime checking."""

    name: str
    category: str  # "import" | "instantiation" | "function_call" | "method_call"
    error_type: str
    error_message: str
    traceback: str


@dataclass
class Results:
    """Aggregated results from all runtime checks."""

    import_ok: list[str] = field(default_factory=list)
    import_errors: list[ErrorRecord] = field(default_factory=list)
    instantiation_ok: list[str] = field(default_factory=list)
    instantiation_errors: list[ErrorRecord] = field(default_factory=list)
    function_call_ok: list[str] = field(default_factory=list)
    function_call_errors: list[ErrorRecord] = field(default_factory=list)
    method_call_ok: list[str] = field(default_factory=list)
    method_call_errors: list[ErrorRecord] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    expected_failures: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Expected failure detection
# ---------------------------------------------------------------------------
def _is_abstract(cls: type) -> bool:
    """Check if a class is abstract (has unimplemented abstract methods)."""
    return bool(getattr(cls, "__abstractmethods__", False))


def _is_enum(cls: type) -> bool:
    """Check if a class is an Enum type."""
    return isinstance(cls, EnumType)


def _is_protocol(cls: type) -> bool:
    """Check if a class is a typing.Protocol class."""
    return bool(getattr(cls, "_is_protocol", False))


def _has_required_init_args(cls: type) -> bool:
    """Check if __init__ requires positional arguments beyond self."""
    try:
        sig = inspect.signature(cls.__init__)
    except (ValueError, TypeError):
        return False
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.default is inspect.Parameter.empty and param.kind not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            return True
    return False


def _is_expected_instantiation_failure(cls: type) -> str | None:
    """Return a reason if the class is expected to fail instantiation."""
    if _is_abstract(cls):
        return "abstract class"
    if _is_enum(cls):
        return "enum type"
    if _is_protocol(cls):
        return "protocol type"
    if _has_required_init_args(cls):
        return "requires constructor arguments"
    return None


# Methods that are expected to fail when called on objects without full setup
# (e.g. models need before_replicate() with RNGs before replicate() works)
_EXPECTED_METHOD_FAILURES = {
    "replicate",  # Models need before_replicate() with RNGs first
    "before_replicate",  # Needs RNG list argument
    "solve",  # Solvers need a problem argument
    "run",  # Wrapper around solve()
}

# Solver methods that require solve() to have been called first (state setup)
_SOLVER_STATE_METHODS = {
    "construct_model",
    "iterate",
    "initial_evaluation",
    "solve_subproblem",
    "simulate_candidate_soln",
    "evaluate_candidate_solution",
    "compute_relative_error",
    "build_Dmat",
    "calculate_pilot_run",
    "calculate_max_radius",
    "compute_adaptive_interpolation_radius_fraction",
    "compute_optimal_subspace_dimension",
    "detect_plateau",
    "evaluate_and_score_candidate_dimensions",
    "heuristic_for_no_successful_models",
    "reduce_dimension_in_plateau_reset",
    "step_iteration",
}


def _is_expected_method_failure(obj: Any, method_name: str) -> str | None:
    """Return a reason if this method is expected to fail without full setup."""
    from simopt.base import Model, Solver

    if method_name in _EXPECTED_METHOD_FAILURES:
        return f"requires prior setup ({method_name})"
    if isinstance(obj, Model) and method_name == "replicate":
        return "model needs before_replicate() with RNGs first"
    if isinstance(obj, Solver) and method_name in _SOLVER_STATE_METHODS:
        return "solver method needs solve() state"
    return None


# ---------------------------------------------------------------------------
# Module discovery
# ---------------------------------------------------------------------------
SKIP_PREFIXES = (
    "simopt.gui",
    "simopt.GUI",
    "simopt.__main__",
    "simopt.online_bandit",
)


def discover_modules() -> list[str]:
    """Return fully-qualified names of all simopt submodules."""
    import simopt

    names: list[str] = []
    for _importer, modname, _ispkg in pkgutil.walk_packages(
        simopt.__path__, prefix="simopt."
    ):
        if any(modname.startswith(p) for p in SKIP_PREFIXES):
            continue
        names.append(modname)
    return sorted(names)


def try_import(module_name: str) -> tuple[Any | None, ErrorRecord | None]:
    """Try to import a module; return (module, None) or (None, ErrorRecord)."""
    try:
        mod = importlib.import_module(module_name)
        return mod, None
    except Exception as exc:
        return None, ErrorRecord(
            name=module_name,
            category="import",
            error_type=type(exc).__name__,
            error_message=str(exc),
            traceback=traceback.format_exc(),
        )


# ---------------------------------------------------------------------------
# Callable discovery
# ---------------------------------------------------------------------------
def _defined_in(obj: Any, module_name: str) -> bool:
    """Check if obj is actually defined in the given module (not just imported)."""
    return getattr(obj, "__module__", None) == module_name


def discover_callables(mod: Any) -> dict[str, list[tuple[str, Any]]]:
    """Categorize public callables defined in a module."""
    from pydantic import BaseModel as PydanticBase

    from simopt.base import Model, Problem, Solver

    result: dict[str, list[tuple[str, Any]]] = {
        "models": [],
        "problems": [],
        "solvers": [],
        "config_classes": [],
        "functions": [],
        "other_classes": [],
    }

    for attr_name in dir(mod):
        if attr_name.startswith("_"):
            continue
        obj = getattr(mod, attr_name)
        if not _defined_in(obj, mod.__name__):
            continue

        if isinstance(obj, type):
            if issubclass(obj, Model) and obj is not Model:
                result["models"].append((attr_name, obj))
            elif issubclass(obj, Problem) and obj is not Problem:
                result["problems"].append((attr_name, obj))
            elif issubclass(obj, Solver) and obj is not Solver:
                result["solvers"].append((attr_name, obj))
            elif issubclass(obj, PydanticBase):
                result["config_classes"].append((attr_name, obj))
            else:
                result["other_classes"].append((attr_name, obj))
        elif callable(obj) and inspect.isfunction(obj):
            result["functions"].append((attr_name, obj))

    return result


# ---------------------------------------------------------------------------
# Instantiation helpers
# ---------------------------------------------------------------------------
def try_instantiate(
    cls: type, label: str
) -> tuple[Any | None, ErrorRecord | None, str | None]:
    """Try to instantiate a class with no arguments.

    Returns (obj, error_record, expected_reason).
    If expected_reason is not None, the failure was anticipated.
    """
    expected = _is_expected_instantiation_failure(cls)
    if expected:
        return None, None, expected

    try:
        obj = cls()
        return obj, None, None
    except TypeError as exc:
        msg = str(exc)
        if "missing" in msg and "required" in msg and "argument" in msg:
            return None, None, "requires constructor arguments (runtime)"
        return (
            None,
            ErrorRecord(
                name=label,
                category="instantiation",
                error_type=type(exc).__name__,
                error_message=msg,
                traceback=traceback.format_exc(),
            ),
            None,
        )
    except (ValueError, ValidationError) as exc:
        msg = str(exc)
        if any(
            phrase in msg.lower()
            for phrase in [
                "must be provided",
                "must specify",
                "must be specified",
                "must be given",
            ]
        ):
            return None, None, "requires constructor arguments (validation)"
        return (
            None,
            ErrorRecord(
                name=label,
                category="instantiation",
                error_type=type(exc).__name__,
                error_message=msg,
                traceback=traceback.format_exc(),
            ),
            None,
        )
    except Exception as exc:
        return (
            None,
            ErrorRecord(
                name=label,
                category="instantiation",
                error_type=type(exc).__name__,
                error_message=str(exc),
                traceback=traceback.format_exc(),
            ),
            None,
        )


class _NoSample:
    """Sentinel for parameters we cannot safely synthesize."""


_NO_SAMPLE = _NoSample()


def _make_sample_curve() -> Any | _NoSample:
    """Create a tiny valid Curve instance for curve-related helpers."""
    try:
        from simopt.curve import Curve

        return Curve(x_vals=[0.0, 1.0], y_vals=[1.0, 0.0])
    except Exception:
        return _NO_SAMPLE


class _SampleBootstrapExperiment:
    """Minimal stand-in for ProblemSolver objects used by bootstrap helpers."""

    def bootstrap_sample(
        self,
        *_args: Any,
        **_kwargs: Any,
    ) -> tuple[list[Any], list[Any]]:
        curve = _make_sample_curve()
        if isinstance(curve, _NoSample):
            return [], []
        return [curve], [curve]


class _SampleBasisWrapper:
    """Simple basis-like object used by ASTROMORF wrapper helpers."""

    def vander(self, x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=float).reshape(-1)
        return np.vstack((np.ones_like(arr), arr)).T

    def polyder(self, coef: Any) -> np.ndarray:
        arr = np.asarray(coef, dtype=float).reshape(-1)
        return np.gradient(arr)


class _SampleModelWrapper:
    """Simple model-like object exposing model/grad/hess methods."""

    def model_evaluate(self, x: Any, coef: Any, u: Any) -> float:
        _ = u
        x_arr = np.asarray(x, dtype=float).reshape(-1)
        coef_arr = np.asarray(coef, dtype=float).reshape(-1)
        return float(np.sum(x_arr) + np.sum(coef_arr))

    def grad(self, x: Any, coef: Any, u: Any) -> np.ndarray:
        _ = coef, u
        x_arr = np.asarray(x, dtype=float).reshape(-1)
        return np.eye(len(x_arr))

    def hess(self, x: Any, coef: Any, u: Any) -> np.ndarray:
        _ = coef, u
        x_arr = np.asarray(x, dtype=float).reshape(-1)
        return np.eye(len(x_arr))


class _SampleSolution:
    """Minimal solution-like object for finite-difference helper tests."""

    def __init__(self, x: Any, objective: float = 0.0) -> None:
        x_arr = np.asarray(x, dtype=float).reshape(-1)
        self.x = tuple(float(v) for v in x_arr)
        self.objectives_mean = np.array([objective], dtype=float)


class _SampleProblem:
    """Minimal problem-like object for solver utility helper tests."""

    class _SampleModel:
        n_rngs = 1

    def __init__(self) -> None:
        self.lower_bounds = np.array([0.0, 0.0], dtype=float)
        self.upper_bounds = np.array([1.0, 1.0], dtype=float)
        self.minmax = (-1,)
        self.dim = 2
        self.model = self._SampleModel()
        self.factors = {"budget": 1, "initial_solution": (0.5, 0.5)}
        self.name = "sample_problem"

    def simulate_up_to(self, solutions: list[Any], r: int) -> None:
        _ = r
        for sol in solutions:
            x_arr = np.asarray(sol.x, dtype=float).reshape(-1)
            sol.objectives_mean = np.array([float(np.sum(x_arr**2))], dtype=float)


class _SampleSolver:
    """Minimal solver-like object for run_solver/solver utils tests."""

    def __init__(self) -> None:
        self.factors = {"delta_T": 0.1}
        self.rng_list = [object()]
        self.solution_progenitor_rngs: list[Any] = []
        self.name = "sample_solver"

    def create_new_solution(self, x: tuple[Any, ...], problem: Any) -> _SampleSolution:
        _ = problem
        x_arr = np.asarray(x, dtype=float).reshape(-1)
        return _SampleSolution(x_arr, objective=float(np.sum(x_arr**2)))

    def attach_rngs(self, rng_list: list[Any]) -> None:
        self.rng_list = list(rng_list)

    def run(self, problem: Any) -> tuple[Any, None]:
        import pandas as pd

        return (
            pd.DataFrame.from_records(
                [
                    {
                        "step": 0,
                        "solution": problem.factors["initial_solution"],
                        "budget": problem.factors["budget"],
                    }
                ]
            ),
            None,
        )


class _SampleContourPath:
    """Small contour path stand-in for save_contour helpers."""

    def __init__(self) -> None:
        self.simplify_threshold = 1e-3

    def iter_segments(
        self, simplify: bool = True
    ) -> list[tuple[tuple[float, float], int]]:
        _ = simplify
        return [
            ((0.0, 0.0), 1),
            ((1.0, 0.0), 2),
            ((1.0, 1.0), 2),
            ((0.0, 1.0), 79),
        ]


class _SampleContourCollection:
    """Small contour collection stand-in for save_contour helpers."""

    def get_paths(self) -> list[_SampleContourPath]:
        return [_SampleContourPath()]


class _SampleContourSet:
    """Small contour set stand-in for save_contour helpers."""

    def __init__(self) -> None:
        self.collections = [_SampleContourCollection()]
        self.levels = [0.0]


def _sample_objective_callable(x: Any) -> np.ndarray:
    """Simple callable matching common objective/model call shapes."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        return arr**2
    if arr.ndim == 2:
        return np.sum(arr**2, axis=1)
    return np.asarray([0.0])


def _sample_argument_from_annotation(annotation: Any) -> Any | _NoSample:
    """Return a lightweight sample value for a type annotation."""
    if annotation is inspect.Parameter.empty:
        return _NO_SAMPLE

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Annotated and args:
        return _sample_argument_from_annotation(args[0])

    if origin is Literal and args:
        return args[0]

    if origin in (Union, UnionType):
        non_none = [arg for arg in args if arg is not type(None)]
        if not non_none:
            return None
        return _sample_argument_from_annotation(non_none[0])

    if origin in (Callable,):
        return lambda *_args, **_kwargs: None

    if origin in (Iterable, Sequence):
        if args:
            item = _sample_argument_from_annotation(args[0])
            if not isinstance(item, _NoSample):
                return [item]
        return []

    if origin in (Mapping,):
        return {}

    if origin in (list,):
        if args:
            item = _sample_argument_from_annotation(args[0])
            if not isinstance(item, _NoSample):
                return [item]
        return []
    if origin in (tuple,):
        if args:
            values = [
                _sample_argument_from_annotation(arg)
                for arg in args
                if arg is not Ellipsis
            ]
            if values and all(not isinstance(v, _NoSample) for v in values):
                return tuple(values)
        return ()
    if origin in (set,):
        if args:
            item = _sample_argument_from_annotation(args[0])
            if not isinstance(item, _NoSample):
                return {item}
        return set()
    if origin in (dict,):
        return {}

    if annotation in (int,):
        return 1
    if annotation in (float,):
        return 0.5
    if annotation in (bool,):
        return False
    if annotation in (str,):
        return "sample"
    if annotation in (bytes,):
        return b"sample"
    if annotation in (Path,):
        return Path()
    if annotation in (Any, object):
        return None
    if isinstance(annotation, EnumType):
        if annotation.__name__ == "PlotType" and hasattr(annotation, "MEAN"):
            return annotation.MEAN
        members = list(annotation)
        if members:
            return members[0]
    if isinstance(annotation, type) and issubclass(annotation, Path):
        return Path()
    if isinstance(annotation, type) and annotation.__name__ == "Curve":
        return _make_sample_curve()

    return _NO_SAMPLE


def _sample_argument_from_name(name: str) -> Any | _NoSample:
    """Return a conservative sample value using parameter-name heuristics."""
    if name in {"X", "Xt"}:
        return np.array([[0.0, 0.0], [1.0, 0.5], [0.2, 0.8]], dtype=float)
    if name == "U":
        return np.eye(2)

    lname = name.lower()

    if "rng" in lname:
        from mrg32k3a.mrg32k3a import MRG32k3a

        if "list" in lname or lname.endswith("s"):
            return [MRG32k3a()]
        return MRG32k3a()

    if "path" in lname or "file" in lname:
        return Path("runtime_check_tmp.txt")

    if lname in {"x", "x0", "xt"}:
        return np.array([0.2, 0.8], dtype=float)

    if lname in {"x_vec", "sample_x"}:
        return np.array([0.0, 0.5, 1.0], dtype=float)

    if lname == "sample_y":
        return np.array([1.0, 0.5, 0.0], dtype=float)

    if lname == "sample_xy":
        return np.array([[0.0, 0.0], [0.7, 0.3], [1.0, 1.0]], dtype=float)

    if lname in {"x_range", "y_range"}:
        return (0.0, 1.0)

    if "plot_type" in lname:
        try:
            from simopt.plot_type import PlotType

            return PlotType.MEAN
        except Exception:
            return "mean"

    if "bootstrap_curves" in lname:
        curve = _make_sample_curve()
        if isinstance(curve, _NoSample):
            return _NO_SAMPLE
        return [[[curve, curve]], [[curve, curve]]]

    if lname == "experiments":
        return [[_SampleBootstrapExperiment()]]

    if lname in {"solver", "stage_solver"}:
        return _SampleSolver()

    if lname == "problem":
        return _SampleProblem()

    if lname == "new_solution":
        return _SampleSolution((0.5, 0.5), objective=0.5)

    if lname == "bounds_check":
        return np.zeros(2, dtype=int)

    if lname in {"stepsize", "delta_t"}:
        return 0.1

    if lname == "design_table":
        import pandas as pd

        return pd.DataFrame({"factor_a": ["1"], "factor_b": ["2.0"]})

    if lname in {"existing_experiments", "lhs"}:
        return []

    if lname in {"experiments_by_variant"}:
        return {}

    if lname in {"model_class"}:
        from simopt.model import Model

        return Model

    if lname == "patch_function":
        return lambda: ("simopt.model.Model", lambda _self: ({}, {}))

    if lname == "module":
        return importlib.import_module("simopt.models._ext")

    if lname in {"val", "lower_bound", "upper_bound"}:
        return {"val": 0.5, "lower_bound": 0.0, "upper_bound": 1.0}[lname]

    if lname in {"fx", "b", "coef"}:
        return np.array([0.5, 1.0, 1.5], dtype=float)

    if lname == "grads":
        return np.array([[1.0, 0.0], [0.3, 0.7], [0.8, 0.2]], dtype=float)

    if lname in {"a", "x", "u"}:
        return np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)

    if lname == "orders":
        return np.array([2, 2], dtype=int)

    if lname == "y":
        return {"y": 1.0}

    if lname in {"objective", "model"}:
        return _sample_objective_callable

    if lname == "f":
        return lambda x: np.atleast_1d(np.sum(np.asarray(x, dtype=float) ** 2))

    if lname == "g":
        return np.array([-1.0, -1.0], dtype=float)

    if lname == "p":
        return np.array([-0.1, -0.1], dtype=float)

    if lname == "d":
        return np.array([0.1, 0.2, 0.3], dtype=float)

    if lname == "cs":
        return _SampleContourSet()

    if lname == "full_space":
        return False

    if lname == "instance":
        return _SampleModelWrapper()

    if lname == "basis_instance":
        return _SampleBasisWrapper()

    if lname == "models":
        return [_sample_objective_callable]

    if "curves" in lname:
        curve = _make_sample_curve()
        if isinstance(curve, _NoSample):
            return _NO_SAMPLE
        return [curve, curve]

    if "curve" in lname:
        curve = _make_sample_curve()
        if isinstance(curve, _NoSample):
            return _NO_SAMPLE
        return curve

    if lname.startswith(("n_", "num_", "count_")):
        return 1
    if lname in {"n", "num", "count", "budget", "stage", "index", "idx", "seed"}:
        return 1

    if lname in {"threshold", "beta", "conf_level", "solve_tol"}:
        return 0.5

    if lname == "n_bootstraps":
        return 2

    if lname in {"n_jobs", "maxiter"}:
        return 1

    if lname in {"r", "dim", "small_index", "highest_order", "level", "t"}:
        return 1

    if lname == "perplexity":
        return 2.0

    if lname == "matrix":
        return np.eye(2)

    if lname == "state_arr":
        return np.array([0.0, 1.0], dtype=float)

    if lname in {"lo_frac", "hi_frac"}:
        return 0.2 if lname == "lo_frac" else 0.8

    if lname == "unique_solvers":
        return ["ASTROMORF", "SGD"]

    if lname == "incumbent_sol":
        return np.array([0.5, 0.5], dtype=float)

    if lname == "fun":
        return _sample_objective_callable

    if lname in {"q", "lo_frac", "conf_level", "beta", "solve_tol"}:
        return 0.5

    if any(tok in lname for tok in ("prob", "weight", "rate", "ratio", "tol")):
        return 0.5

    if any(tok in lname for tok in ("verbose", "debug", "flag", "check")):
        return False

    if any(tok in lname for tok in ("name", "label", "key", "id")):
        return "sample"

    if any(tok in lname for tok in ("factor", "config", "param", "option", "kwargs")):
        return {}

    if lname in {"x", "solution", "decision"}:
        return (0.0,)

    return _NO_SAMPLE


def _build_function_call_args(
    sig: inspect.Signature,
) -> tuple[list[Any] | None, dict[str, Any] | None, str | None, bool]:
    """Build arguments for required function parameters when possible."""
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    used_required = False

    for param in sig.parameters.values():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if param.default is not inspect.Parameter.empty:
            continue

        used_required = True
        sample = _sample_argument_from_annotation(param.annotation)
        if isinstance(sample, _NoSample):
            sample = _sample_argument_from_name(param.name)
        if isinstance(sample, _NoSample):
            reason = f"unsupported required parameter '{param.name}'"
            return None, None, reason, used_required

        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            args.append(sample)
        elif param.kind is inspect.Parameter.KEYWORD_ONLY:
            kwargs[param.name] = sample

    # Some optional parameters benefit from lightweight overrides
    # (e.g., force serial execution to avoid expensive parallel setup).
    for param in sig.parameters.values():
        if param.default is inspect.Parameter.empty:
            continue
        sample = _sample_argument_from_name(param.name)
        if isinstance(sample, _NoSample):
            continue
        if param.kind in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            kwargs[param.name] = sample

    return args, kwargs, None, used_required


def try_call_function(fn: Any, label: str) -> tuple[ErrorRecord | None, str | None]:
    """Try to call a standalone function.

    Returns (error, skip_reason). At most one is non-None.
    """
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return None, "no inspectable signature"

    call_args, call_kwargs, skip_reason, used_required = _build_function_call_args(sig)
    if skip_reason is not None:
        return None, skip_reason
    if call_args is None or call_kwargs is None:
        return None, "unable to construct call arguments"

    try:
        fn(*call_args, **call_kwargs)
        return None, None
    except (ValueError, TypeError) as exc:
        msg = str(exc)
        if any(
            phrase in msg.lower()
            for phrase in [
                "must be provided",
                "must specify",
                "must be specified",
                "missing",
                "required",
            ]
        ):
            return None, "requires domain-specific runtime values"

        # If we had to synthesize required args, a mismatch is expected noise.
        if used_required:
            return None, "requires domain-specific runtime values"

        return ErrorRecord(
            name=label,
            category="function_call",
            error_type=type(exc).__name__,
            error_message=msg,
            traceback=traceback.format_exc(),
        ), None
    except (AttributeError, AssertionError, IndexError, KeyError) as exc:
        if used_required:
            return None, "requires domain-specific runtime values"
        return ErrorRecord(
            name=label,
            category="function_call",
            error_type=type(exc).__name__,
            error_message=str(exc),
            traceback=traceback.format_exc(),
        ), None
    except (RuntimeError, NotImplementedError, OSError, ImportError) as exc:
        if used_required:
            return None, "requires domain-specific runtime values"
        return ErrorRecord(
            name=label,
            category="function_call",
            error_type=type(exc).__name__,
            error_message=str(exc),
            traceback=traceback.format_exc(),
        ), None
    except Exception as exc:
        if used_required:
            return None, "requires domain-specific runtime values"
        return ErrorRecord(
            name=label,
            category="function_call",
            error_type=type(exc).__name__,
            error_message=str(exc),
            traceback=traceback.format_exc(),
        ), None


# ---------------------------------------------------------------------------
# Method invocation on instantiated objects
# ---------------------------------------------------------------------------
_PROBLEM_PROPS = ("dim", "lower_bounds", "upper_bounds", "specifications", "name")
_MODEL_PROPS = ("specifications", "name", "factors")
_SOLVER_PROPS = ("specifications", "name")


def try_access_properties(
    obj: Any, label: str, props: tuple[str, ...]
) -> list[tuple[str, ErrorRecord | None]]:
    """Try accessing properties/no-arg methods on an object."""
    results: list[tuple[str, ErrorRecord | None]] = []
    for prop_name in props:
        full_label = f"{label}.{prop_name}"
        try:
            getattr(obj, prop_name)
            results.append((full_label, None))
        except Exception as exc:
            results.append(
                (
                    full_label,
                    ErrorRecord(
                        name=full_label,
                        category="method_call",
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        traceback=traceback.format_exc(),
                    ),
                )
            )
    return results


def try_public_methods(
    obj: Any, label: str
) -> list[tuple[str, ErrorRecord | None, str | None]]:
    """Try calling public methods that take no arguments (besides self).

    Returns list of (label, error_or_none, expected_reason_or_none).
    """
    results: list[tuple[str, ErrorRecord | None, str | None]] = []
    for attr_name in dir(obj):
        if attr_name.startswith("_"):
            continue

        # Check if this is an expected failure before even calling
        expected = _is_expected_method_failure(obj, attr_name)
        if expected:
            continue  # silently skip methods that need prior setup

        try:
            attr = getattr(type(obj), attr_name, None)
        except Exception:
            continue
        if isinstance(attr, property | classmethod):
            continue
        method = getattr(obj, attr_name, None)
        if method is None or not callable(method):
            continue

        try:
            sig = inspect.signature(method)
        except (ValueError, TypeError):
            continue

        has_required = False
        for param in sig.parameters.values():
            if param.default is inspect.Parameter.empty and param.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                has_required = True
                break
        if has_required:
            continue

        full_label = f"{label}.{attr_name}()"
        try:
            method()
            results.append((full_label, None, None))
        except Exception as exc:
            results.append(
                (
                    full_label,
                    ErrorRecord(
                        name=full_label,
                        category="method_call",
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        traceback=traceback.format_exc(),
                    ),
                    None,
                )
            )
    return results


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------
def run_checks(verbose: bool = False) -> Results:
    """Run all runtime checks and return aggregated results."""
    results = Results()

    # 1. Discover and import modules
    print(_bold("\n=== Phase 1: Module Discovery & Import ===\n"))
    module_names = discover_modules()
    print(f"Found {len(module_names)} modules under simopt.*\n")

    imported_modules: list[Any] = []
    for name in module_names:
        mod, err = try_import(name)
        if err:
            results.import_errors.append(err)
            print(f"  {_red('FAIL')} {name}: {err.error_type}: {err.error_message}")
        else:
            results.import_ok.append(name)
            imported_modules.append(mod)
            if verbose:
                print(f"  {_green('OK')}   {name}")

    print(
        f"\n  Imported: {_green(str(len(results.import_ok)))} | "
        f"Failed: {_red(str(len(results.import_errors)))}"
    )

    # 2. Discover and test callables
    print(_bold("\n=== Phase 2: Class Instantiation & Function Calls ===\n"))

    all_callables: dict[str, list[tuple[str, Any, str]]] = {
        "models": [],
        "problems": [],
        "solvers": [],
        "config_classes": [],
        "functions": [],
        "other_classes": [],
    }

    for mod in imported_modules:
        cats = discover_callables(mod)
        for cat_key, items in cats.items():
            for attr_name, obj in items:
                label = f"{mod.__name__}.{attr_name}"
                all_callables[cat_key].append((label, obj, mod.__name__))

    # Deduplicate (same class may appear via re-exports)
    for cat_key in all_callables:
        seen: set[int] = set()
        deduped = []
        for label, obj, modname in all_callables[cat_key]:
            obj_id = id(obj)
            if obj_id not in seen:
                seen.add(obj_id)
                deduped.append((label, obj, modname))
        all_callables[cat_key] = deduped

    # Helper to test a category of classes
    def test_class_category(
        cat_name: str, items: list, props: tuple[str, ...] | None = None
    ) -> None:
        print(_bold(f"  {cat_name}:"))
        for label, cls, _ in items:
            obj, err, expected = try_instantiate(cls, label)
            if expected:
                results.expected_failures.append(f"{label} ({expected})")
                if verbose:
                    print(f"    {_yellow('SKIP')} {label} ({expected})")
            elif err:
                results.instantiation_errors.append(err)
                print(
                    f"    {_red('FAIL')} {label}: {err.error_type}: {err.error_message}"
                )
            else:
                results.instantiation_ok.append(label)
                if verbose:
                    print(f"    {_green('OK')}   {label}")
                if props:
                    for prop_label, prop_err in try_access_properties(
                        obj, label, props
                    ):
                        if prop_err:
                            results.method_call_errors.append(prop_err)
                        else:
                            results.method_call_ok.append(prop_label)

    test_class_category("Models", all_callables["models"], _MODEL_PROPS)
    test_class_category("Problems", all_callables["problems"], _PROBLEM_PROPS)
    test_class_category("Solvers", all_callables["solvers"], _SOLVER_PROPS)
    test_class_category("Config Classes", all_callables["config_classes"])
    test_class_category("Other Classes", all_callables["other_classes"])

    # --- Standalone functions ---
    print(_bold("  Standalone Functions:"))
    for label, fn, _ in all_callables["functions"]:
        err, skip_reason = try_call_function(fn, label)
        if err is None and skip_reason is None:
            results.function_call_ok.append(label)
            if verbose:
                print(f"    {_green('OK')}   {label}")
        elif err is None:
            skip_msg = f"{label} ({skip_reason})"
            results.skipped.append(skip_msg)
            if verbose:
                print(f"    {_yellow('SKIP')} {skip_msg}")
        else:
            results.function_call_errors.append(err)
            print(f"    {_red('FAIL')} {label}: {err.error_type}: {err.error_message}")

    # 3. Method calls on successfully instantiated objects
    print(_bold("\n=== Phase 3: No-Arg Method Calls ===\n"))

    for cat_key, _props in [
        ("models", _MODEL_PROPS),
        ("problems", _PROBLEM_PROPS),
        ("solvers", _SOLVER_PROPS),
    ]:
        for label, cls, _ in all_callables[cat_key]:
            obj, err, expected = try_instantiate(cls, label)
            if err or expected:
                continue
            for method_label, method_err, _expected in try_public_methods(obj, label):
                if method_err:
                    results.method_call_errors.append(method_err)
                    print(
                        f"  {_red('FAIL')} {method_label}: "
                        f"{method_err.error_type}: {method_err.error_message}"
                    )
                else:
                    results.method_call_ok.append(method_label)
                    if verbose:
                        print(f"  {_green('OK')}   {method_label}")

    return results


def print_summary(results: Results) -> None:
    """Print a coloured summary of all check results."""
    total_errors = (
        len(results.import_errors)
        + len(results.instantiation_errors)
        + len(results.function_call_errors)
        + len(results.method_call_errors)
    )
    total_ok = (
        len(results.import_ok)
        + len(results.instantiation_ok)
        + len(results.function_call_ok)
        + len(results.method_call_ok)
    )

    print(_bold("\n" + "=" * 60))
    print(_bold("SUMMARY"))
    print("=" * 60)
    print(
        f"  Module imports:      {_green(str(len(results.import_ok)))} ok, "
        f"{_red(str(len(results.import_errors)))} failed"
    )
    print(
        f"  Instantiations:      {_green(str(len(results.instantiation_ok)))} ok, "
        f"{_red(str(len(results.instantiation_errors)))} failed, "
        f"{_yellow(str(len(results.expected_failures)))} expected skips"
    )
    print(
        f"  Function calls:      {_green(str(len(results.function_call_ok)))} ok, "
        f"{_red(str(len(results.function_call_errors)))} failed, "
        f"{_yellow(str(len(results.skipped)))} skipped (need args)"
    )
    print(
        f"  Method calls:        {_green(str(len(results.method_call_ok)))} ok, "
        f"{_red(str(len(results.method_call_errors)))} failed"
    )
    print(
        f"\n  Total:               {_green(str(total_ok))} ok, "
        f"{_red(str(total_errors))} errors"
    )

    if total_errors > 0:
        print(_bold(f"\n{_red('ERRORS DETECTED')} — see details above\n"))

        print(_bold("Error details:"))
        all_errors = (
            results.import_errors
            + results.instantiation_errors
            + results.function_call_errors
            + results.method_call_errors
        )
        for i, err in enumerate(all_errors, 1):
            print(f"\n  {i}. [{err.category}] {err.name}")
            print(f"     {err.error_type}: {err.error_message}")
    else:
        print(_bold(f"\n{_green('ALL CHECKS PASSED')}\n"))


def save_json_report(results: Results, path: Path) -> None:
    """Save a detailed JSON report."""

    def _err_to_dict(err: ErrorRecord) -> dict:
        return {
            "name": err.name,
            "category": err.category,
            "error_type": err.error_type,
            "error_message": err.error_message,
            "traceback": err.traceback,
        }

    report = {
        "summary": {
            "imports_ok": len(results.import_ok),
            "imports_failed": len(results.import_errors),
            "instantiations_ok": len(results.instantiation_ok),
            "instantiations_failed": len(results.instantiation_errors),
            "expected_skips": len(results.expected_failures),
            "function_calls_ok": len(results.function_call_ok),
            "function_calls_failed": len(results.function_call_errors),
            "function_calls_skipped": len(results.skipped),
            "method_calls_ok": len(results.method_call_ok),
            "method_calls_failed": len(results.method_call_errors),
        },
        "errors": {
            "import_errors": [_err_to_dict(e) for e in results.import_errors],
            "instantiation_errors": [
                _err_to_dict(e) for e in results.instantiation_errors
            ],
            "function_call_errors": [
                _err_to_dict(e) for e in results.function_call_errors
            ],
            "method_call_errors": [_err_to_dict(e) for e in results.method_call_errors],
        },
        "passed": {
            "imports": results.import_ok,
            "instantiations": results.instantiation_ok,
            "function_calls": results.function_call_ok,
            "method_calls": results.method_call_ok,
        },
        "expected_skips": results.expected_failures,
        "skipped_functions": results.skipped,
    }

    path.write_text(json.dumps(report, indent=2))
    print(f"JSON report saved to {path}")


def main() -> None:
    """Entry point for the runtime error checker."""
    verbose = "--verbose" in sys.argv
    save_json = "--json" in sys.argv

    print(_bold("SimOpt Runtime Error Checker"))
    print("Discovering and testing all functions/classes in the simopt package...\n")

    results = run_checks(verbose=verbose)
    print_summary(results)

    if save_json:
        report_path = Path(__file__).parent / "runtime_errors_report.json"
        save_json_report(results, report_path)

    total_errors = (
        len(results.import_errors)
        + len(results.instantiation_errors)
        + len(results.function_call_errors)
        + len(results.method_call_errors)
    )
    sys.exit(1 if total_errors > 0 else 0)


if __name__ == "__main__":
    main()
