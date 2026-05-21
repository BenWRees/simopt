"""Journal-grade sensitivity-analysis plotting package for ASTROMoRF.

The package consumes the artefacts written by
``scripts/journal_aggregate.py`` and produces publication-ready figures.

Public entry points::

    from simopt.journal.plotting import load_aggregated, generate_all_figures

The CLI wrapper lives in ``scripts/generate_astromorf_sensitivity_figures.py``.
"""
from __future__ import annotations

from .data import (
    AggregatedResults,
    SchemaError,
    load_aggregated,
    validate_long_form,
)
from .figures import (
    convergence_curves,
    final_objective_distribution,
    final_objective_summary_plot,
    generate_all_figures,
    paired_effect_plot,
    ranking_plot,
)
from .stats import (
    bootstrap_mean_ci,
    final_objective_summary,
    paired_ci_against_baseline,
    rank_levels_by_effect,
)
from .style import apply_journal_style, save_journal_plot, setup_journal_plot

STUDIES: tuple[str, ...] = ("subspace", "basis", "regularisation")

__all__ = [
    "STUDIES",
    "AggregatedResults",
    "SchemaError",
    "apply_journal_style",
    "bootstrap_mean_ci",
    "convergence_curves",
    "final_objective_distribution",
    "final_objective_summary",
    "final_objective_summary_plot",
    "generate_all_figures",
    "load_aggregated",
    "paired_ci_against_baseline",
    "paired_effect_plot",
    "ranking_plot",
    "rank_levels_by_effect",
    "save_journal_plot",
    "setup_journal_plot",
    "validate_long_form",
]
