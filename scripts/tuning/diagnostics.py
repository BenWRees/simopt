"""Per-macroreplication diagnostics extraction from a finished ASTROMoRF solve.

The ASTROMoRF solver exposes its iteration/CABS state directly on the
instance (``successful_iterations``, ``unsuccessful_iterations``,
``d_history``, ``cabs_log``, ``recent_prediction_errors``). The evaluator
runs each macroreplication in-process and hands the post-run solver
instance to :func:`extract_macrorep_diagnostics`.

We also wrap ``solver.pattern_search`` once before the run so we can count
how often pattern_search overrode the trust-region candidate (the truly
informative "pattern-search fallback" event, not the bare call count).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass
class MrepDiagnostics:
    """Diagnostics extracted from a single completed macroreplication."""

    iteration_count: int
    accepted_steps: int
    rejected_steps: int
    accept_ratio: float
    cabs_increases: int
    cabs_decreases: int
    avg_subspace_dim: float
    final_subspace_dim: int
    initial_subspace_dim: int
    avg_interp_set_size_approx: float
    pattern_search_overrides: int
    pattern_search_override_ratio: float
    mean_prediction_rel_error: float
    budget_used: int
    wall_clock_s: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def install_pattern_search_counter(solver: Any) -> None:
    """Wrap ``solver.pattern_search`` to count override events in-place.

    The wrapper compares the returned solution to the input candidate; if
    they differ, an interpolation point won the comparison — that is the
    "pattern-search fallback" event. Counts are accumulated on the solver
    instance under ``_ps_calls`` and ``_ps_overrides``.
    """
    original = solver.pattern_search
    solver._ps_calls = 0
    solver._ps_overrides = 0

    def counting_pattern_search(
        candidate_solution, fval, fval_tilde, interpolation_solns
    ):
        solver._ps_calls += 1
        new_cand, new_fval = original(
            candidate_solution, fval, fval_tilde, interpolation_solns
        )
        if new_cand is not candidate_solution:
            solver._ps_overrides += 1
        return new_cand, new_fval

    solver.pattern_search = counting_pattern_search  # type: ignore[assignment]


def _safe_int(x: Any) -> int:
    try:
        return int(x)
    except Exception:
        return 0


def _count_dim_changes(d_history: list[int]) -> tuple[int, int]:
    incs, decs = 0, 0
    for prev, curr in zip(d_history, d_history[1:]):
        if curr > prev:
            incs += 1
        elif curr < prev:
            decs += 1
    return incs, decs


def extract_macrorep_diagnostics(
    solver: Any, *, wall_clock_s: float
) -> MrepDiagnostics:
    """Read diagnostics off a post-run ASTROMoRF solver instance."""
    succ = len(getattr(solver, "successful_iterations", []) or [])
    unsucc = len(getattr(solver, "unsuccessful_iterations", []) or [])
    total = succ + unsucc
    accept_ratio = (succ / total) if total else 0.0

    d_history = list(getattr(solver, "d_history", []) or [])
    if not d_history:
        d_history = [_safe_int(getattr(solver, "d", 0))]
    incs, decs = _count_dim_changes(d_history)

    avg_d = float(np.mean(d_history)) if d_history else 0.0
    final_d = _safe_int(d_history[-1])
    initial_d = _safe_int(d_history[0])
    avg_interp = 2.0 * avg_d + 1.0  # standard ASTROMoRF interpolation set size

    pred_errs = list(getattr(solver, "recent_prediction_errors", []) or [])
    mean_pred = (
        float(np.mean([abs(e) for e in pred_errs if np.isfinite(e)]))
        if pred_errs
        else float("nan")
    )

    ps_overrides = _safe_int(getattr(solver, "_ps_overrides", 0))
    ps_calls = _safe_int(getattr(solver, "_ps_calls", 0))
    ps_override_ratio = (ps_overrides / ps_calls) if ps_calls else 0.0

    budget_used = _safe_int(getattr(getattr(solver, "budget", None), "used", 0))
    iteration_count = _safe_int(getattr(solver, "iteration_count", 0))

    return MrepDiagnostics(
        iteration_count=iteration_count,
        accepted_steps=succ,
        rejected_steps=unsucc,
        accept_ratio=accept_ratio,
        cabs_increases=incs,
        cabs_decreases=decs,
        avg_subspace_dim=avg_d,
        final_subspace_dim=final_d,
        initial_subspace_dim=initial_d,
        avg_interp_set_size_approx=avg_interp,
        pattern_search_overrides=ps_overrides,
        pattern_search_override_ratio=ps_override_ratio,
        mean_prediction_rel_error=mean_pred,
        budget_used=budget_used,
        wall_clock_s=wall_clock_s,
    )


def aggregate_diagnostics(per_mrep: list[MrepDiagnostics]) -> dict[str, Any]:
    """Aggregate per-mrep diagnostics into mean/std summaries for one trial."""
    if not per_mrep:
        return {}
    fields = [
        "iteration_count",
        "accepted_steps",
        "rejected_steps",
        "accept_ratio",
        "cabs_increases",
        "cabs_decreases",
        "avg_subspace_dim",
        "final_subspace_dim",
        "initial_subspace_dim",
        "avg_interp_set_size_approx",
        "pattern_search_overrides",
        "pattern_search_override_ratio",
        "mean_prediction_rel_error",
        "budget_used",
        "wall_clock_s",
    ]
    out: dict[str, Any] = {"n_macroreps": len(per_mrep)}
    for name in fields:
        vals = np.asarray(
            [getattr(d, name) for d in per_mrep], dtype=float
        )
        finite = vals[np.isfinite(vals)]
        out[f"{name}_mean"] = float(np.mean(finite)) if finite.size else float("nan")
        out[f"{name}_std"] = (
            float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
        )
    return out
