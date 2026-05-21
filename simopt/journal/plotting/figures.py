"""Figure factories for the ASTROMoRF sensitivity studies.

Each public factory produces *one standalone figure per problem*, matching
the canonical SimOpt convention used by ``plot_progress_curves`` and
``plot_solvability_profiles`` (which themselves emit one figure per
problem when ``all_in_one=False``).  No subplot grids are used.

All aesthetic decisions — fonts, line widths, legend alpha, savefig
options — are inherited from :mod:`simopt.journal.plotting.style`, which
mirrors the inline overrides applied by ``simopt.plots.utils``.  This
package therefore deliberately avoids growing a parallel styling system.

Conventions
-----------
* x-axis is the swept factor for OFAT studies (subspace / regularisation).
* For the basis study, ``mean_diff`` / ``mean_final_obj`` matrices are
  rendered as a single per-problem heatmap (one figure per problem).
* Regularisation values use a symlog x-axis so the explicit zero is
  visible alongside the seven log decades.
* Each factory returns ``list[Path]`` — one entry per (problem x format).
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from .data import LEVEL_COLUMNS, STUDIES, AggregatedResults
from .stats import (
    final_objective_summary,
    paired_ci_against_baseline,
    rank_levels_by_effect,
)
from .style import (
    CI_ALPHA,
    LABEL_SIZE,
    TICK_SIZE,
    apply_journal_style,
    colour_for_index,
    colour_for_level,
    save_journal_plot,
    setup_journal_plot,
    style_legend,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _check_study(study: str) -> None:
    if study not in STUDIES:
        raise ValueError(f"unknown study {study!r}; valid: {STUDIES!r}")


def _format_reg_xaxis() -> None:
    """Use symlog so 0.0 and 1e-6 share a sane axis (current axes)."""
    plt.xscale("symlog", linthresh=1e-6, linscale=0.5)


def _level_xlabel(study: str) -> str:
    if study == "subspace":
        return "Subspace Dimension"
    if study == "regularisation":
        return "Subproblem Regularisation"
    return "Level"


def _apply_level_xaxis(study: str) -> None:
    if study == "regularisation":
        _format_reg_xaxis()


def _sorted_levels(pdf: pd.DataFrame, level_cols: list[str]) -> list[tuple]:
    """Deterministic ordering of factor levels for reproducible figures."""
    rows = (
        pdf[level_cols]
        .drop_duplicates()
        .sort_values(level_cols, kind="mergesort")
        .itertuples(index=False, name=None)
    )
    return list(rows)


def _level_label(study: str, level: tuple) -> str:
    """Reviewer-facing label for a single (study, level) entry."""
    if study == "basis":
        return f"{level[0]}, deg={level[1]}"
    if study == "subspace":
        return f"d={level[0]}"
    return f"r={level[0]:g}"


def _resolve_level_colour(study: str, level: tuple, idx: int) -> str:
    """Pick a canonical ``"C{i}"`` colour for a level."""
    if study == "basis":
        return colour_for_level(study, level)
    return colour_for_index(idx)


def _stem(study: str, plot_name: str, problem: str) -> str:
    """Sanitised filename stem in the canonical SimOpt style.

    Mirrors ``simopt.plots.utils.save_plot`` filenaming: ``<solver>_<problem>
    _<plot_name>``.  Here ``study`` plays the solver-group role.
    """
    return f"{study}_{problem}_{plot_name}"


# ---------------------------------------------------------------------------
# 1. Convergence curves with CI bands (one figure per problem)
# ---------------------------------------------------------------------------
def convergence_curves(
    long_form: pd.DataFrame,
    *,
    study: str,
    alpha_band: float = CI_ALPHA,
    xscale: str = "linear",
    yscale: str = "linear",
    output_dir: Path | None = None,
    formats: Sequence[str] = ("png",),
    legend_loc: str | None = None,
) -> list[Path]:
    """Mean convergence trajectories per level with macrorep spread band.

    Args:
        long_form: Tidy long-form table (see :mod:`.data`).
        study: One of :data:`STUDIES`.
        alpha_band: Fill alpha for the 10-90 percentile band. Defaults to the
            canonical SimOpt CI band alpha.
        xscale / yscale: Axis scale strings passed to ``plt.xscale`` /
            ``plt.yscale``.  ``"linear"`` leaves the axis untouched.
        output_dir: Directory to write figures into. ``None`` skips saving.
        formats: One or more file extensions (``"png"``, ``"pdf"``, ...).
        legend_loc: Override for ``plt.legend(loc=...)``.

    Returns:
        list[Path]: Files written (one per problem x format).
    """
    _check_study(study)
    apply_journal_style()
    sub = long_form[long_form["study"] == study]
    problems = sorted(sub["problem"].unique())
    level_cols = list(LEVEL_COLUMNS[study])
    all_levels = _sorted_levels(sub, level_cols)
    written: list[Path] = []
    if not problems:
        return written
    if legend_loc is None:
        legend_loc = "best"

    for problem in problems:
        pdf = sub[sub["problem"] == problem]
        setup_journal_plot(
            title=f"{study.upper()} on {problem}\nConvergence Curves",
            xlabel="Budget",
            ylabel="Post-Replicated Mean Objective",
        )
        handles = []
        labels = []
        for idx, lvl in enumerate(all_levels):
            mask = np.ones(len(pdf), dtype=bool)
            for col, v in zip(level_cols, lvl, strict=False):
                mask &= (pdf[col].values == v)
            ldf = pdf[mask]
            if ldf.empty:
                continue
            grouped = ldf.groupby("budget", observed=False)["obj_postrep_mean"]
            mean = grouped.mean()
            lo = grouped.quantile(0.10)
            hi = grouped.quantile(0.90)
            colour = _resolve_level_colour(study, lvl, idx)
            label = _level_label(study, lvl)
            (line,) = plt.plot(mean.index, mean.values, color=colour)
            plt.fill_between(
                mean.index, lo.values, hi.values,
                color=colour, alpha=alpha_band, linewidth=0,
            )
            handles.append(line)
            labels.append(label)
        if xscale != "linear":
            plt.xscale(xscale)
        if yscale != "linear":
            plt.yscale(yscale)
        leg = plt.legend(handles=handles, labels=labels, loc=legend_loc)
        style_legend(leg)
        if output_dir is not None:
            written += save_journal_plot(
                _stem(study, "convergence_curves", problem),
                output_dir=output_dir, formats=formats,
            )
        else:
            plt.close(plt.gcf())
    return written


# ---------------------------------------------------------------------------
# 2. Final-objective summary plot (one figure per problem)
# ---------------------------------------------------------------------------
def final_objective_summary_plot(
    finals: pd.DataFrame,
    *,
    study: str,
    robust_quantiles: tuple[float, float] | None = None,
    output_dir: Path | None = None,
    formats: Sequence[str] = ("png",),
) -> list[Path]:
    """Per-problem summary of the final post-replicated objective.

    OFAT studies (subspace / regularisation) render a line + 95% CI error-bar
    plot of mean final objective vs. level.  The basis study renders a
    per-problem heatmap on (basis x degree).  ``robust_quantiles`` clips the
    basis heatmap's colour scale so a single catastrophic cell cannot
    saturate the colormap.
    """
    _check_study(study)
    apply_journal_style()
    sub = finals[finals["study"] == study]
    summary = final_objective_summary(sub, study=study)
    problems = sorted(summary["problem"].unique()) if not summary.empty else []
    written: list[Path] = []

    if study == "basis":
        bases = sorted(summary["polynomial_basis"].unique())
        degrees = sorted(summary["polynomial_degree"].unique())
        for problem in problems:
            pdf = summary[summary["problem"] == problem]
            mat = np.full((len(bases), len(degrees)), np.nan)
            for _, row in pdf.iterrows():
                i = bases.index(row["polynomial_basis"])
                j = degrees.index(int(row["polynomial_degree"]))
                mat[i, j] = row["mean_final_obj"]
            setup_journal_plot(
                title=f"{study.upper()} on {problem}\nFinal Objective",
                xlabel="Polynomial Degree",
                ylabel="Polynomial Basis",
            )
            imshow_kwargs: dict[str, Any] = {
                "aspect": "auto", "cmap": "viridis", "origin": "lower",
            }
            if robust_quantiles is not None:
                finite = mat[np.isfinite(mat)]
                if finite.size:
                    lo_q, hi_q = robust_quantiles
                    vmin = float(np.quantile(finite, lo_q))
                    vmax = float(np.quantile(finite, hi_q))
                    if vmax > vmin:
                        imshow_kwargs["vmin"] = vmin
                        imshow_kwargs["vmax"] = vmax
            ax = plt.gca()
            im = ax.imshow(mat, **imshow_kwargs)
            ax.set_xticks(range(len(degrees)))
            ax.set_xticklabels([str(d) for d in degrees])
            ax.set_yticks(range(len(bases)))
            ax.set_yticklabels(bases)
            cbar = plt.colorbar(
                im, ax=ax, shrink=0.85,
                label="Mean Final Obj" + (" (clipped)" if robust_quantiles else ""),
            )
            cbar.ax.tick_params(labelsize=TICK_SIZE)
            cbar.set_label(
                cbar.ax.get_ylabel(), size=LABEL_SIZE,
            )
            if output_dir is not None:
                written += save_journal_plot(
                    _stem(study, "final_objective_summary", problem),
                    output_dir=output_dir, formats=formats,
                )
            else:
                plt.close(plt.gcf())
        return written

    level_col = LEVEL_COLUMNS[study][0]
    for problem in problems:
        pdf = summary[summary["problem"] == problem].sort_values(level_col)
        setup_journal_plot(
            title=f"{study.upper()} on {problem}\nFinal Objective",
            xlabel=_level_xlabel(study),
            ylabel="Mean Final Objective",
        )
        plt.errorbar(
            pdf[level_col].astype(float).values,
            pdf["mean_final_obj"].values,
            yerr=[
                (pdf["mean_final_obj"] - pdf["ci95_lo"]).values,
                (pdf["ci95_hi"] - pdf["mean_final_obj"]).values,
            ],
            fmt="o-", color="C0", capsize=2,
        )
        _apply_level_xaxis(study)
        if output_dir is not None:
            written += save_journal_plot(
                _stem(study, "final_objective_summary", problem),
                output_dir=output_dir, formats=formats,
            )
        else:
            plt.close(plt.gcf())
    return written


# ---------------------------------------------------------------------------
# 3. Paired-effect plot (one figure per problem)
# ---------------------------------------------------------------------------
def paired_effect_plot(
    finals: pd.DataFrame,
    *,
    study: str,
    baselines: Mapping[str, Any],
    robust_quantiles: tuple[float, float] | None = None,
    output_dir: Path | None = None,
    formats: Sequence[str] = ("png",),
) -> list[Path]:
    """Paired (level - baseline) effect with 95% bootstrap CI per problem.

    OFAT studies render a line + shaded CI band with a zero baseline.  The
    basis study renders a diverging-colormap heatmap of ``mean_diff`` with
    hatching wherever the CI excludes zero (significant after CRN pairing).
    """
    _check_study(study)
    apply_journal_style()
    sub = finals[finals["study"] == study]
    paired = paired_ci_against_baseline(sub, study=study, baselines=baselines)
    if paired.empty:
        return []
    problems = sorted(paired["problem"].unique())
    written: list[Path] = []

    if study == "basis":
        bases = sorted(paired["polynomial_basis"].unique())
        degrees = sorted(paired["polynomial_degree"].unique())
        abs_diff = paired["mean_diff"].abs().to_numpy()
        if robust_quantiles is not None and abs_diff.size:
            _, hi_q = robust_quantiles
            vmax = float(np.quantile(abs_diff, hi_q)) or float(abs_diff.max() or 1.0)
        else:
            vmax = float(abs_diff.max() or 1.0)
        for problem in problems:
            pdf = paired[paired["problem"] == problem]
            mat = np.full((len(bases), len(degrees)), np.nan)
            sig = np.zeros_like(mat, dtype=bool)
            for _, row in pdf.iterrows():
                i = bases.index(row["polynomial_basis"])
                j = degrees.index(int(row["polynomial_degree"]))
                mat[i, j] = row["mean_diff"]
                sig[i, j] = (row["ci95_lo"] > 0) or (row["ci95_hi"] < 0)
            setup_journal_plot(
                title=f"{study.upper()} on {problem}\nPaired Effect (CRN)",
                xlabel="Polynomial Degree",
                ylabel="Polynomial Basis",
            )
            ax = plt.gca()
            im = ax.imshow(
                mat, aspect="auto", origin="lower", cmap="RdBu_r",
                norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
            )
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    if sig[i, j]:
                        ax.add_patch(plt.Rectangle(
                            (j - 0.5, i - 0.5), 1, 1,
                            fill=False, hatch="///",
                            edgecolor="black", linewidth=0,
                        ))
            ax.set_xticks(range(len(degrees)))
            ax.set_xticklabels([str(d) for d in degrees])
            ax.set_yticks(range(len(bases)))
            ax.set_yticklabels(bases)
            cbar = plt.colorbar(
                im, ax=ax, shrink=0.85, label="Mean Diff vs. Baseline",
            )
            cbar.ax.tick_params(labelsize=TICK_SIZE)
            cbar.set_label(cbar.ax.get_ylabel(), size=LABEL_SIZE)
            if output_dir is not None:
                written += save_journal_plot(
                    _stem(study, "paired_effect", problem),
                    output_dir=output_dir, formats=formats,
                )
            else:
                plt.close(plt.gcf())
        return written

    level_col = LEVEL_COLUMNS[study][0]
    for problem in problems:
        pdf = paired[paired["problem"] == problem].sort_values(level_col)
        setup_journal_plot(
            title=f"{study.upper()} on {problem}\nPaired Effect (CRN)",
            xlabel=_level_xlabel(study),
            ylabel="Paired Mean Diff (Level - Baseline)",
        )
        x = pdf[level_col].astype(float).values
        mean = pdf["mean_diff"].values
        lo = pdf["ci95_lo"].values
        hi = pdf["ci95_hi"].values
        plt.fill_between(x, lo, hi, alpha=CI_ALPHA, color="C0")
        plt.plot(x, mean, "o-", color="C0")
        plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
        y_lo, y_hi = plt.ylim()
        plt.ylim(min(y_lo, -1e-6), max(y_hi, 1e-6))
        _apply_level_xaxis(study)
        if output_dir is not None:
            written += save_journal_plot(
                _stem(study, "paired_effect", problem),
                output_dir=output_dir, formats=formats,
            )
        else:
            plt.close(plt.gcf())
    return written


# ---------------------------------------------------------------------------
# 4. Distribution / robustness (violin) — one figure per problem
# ---------------------------------------------------------------------------
def final_objective_distribution(
    finals: pd.DataFrame,
    *,
    study: str,
    output_dir: Path | None = None,
    formats: Sequence[str] = ("png",),
) -> list[Path]:
    """Violin plot of macrorep final-objective spread per level, per problem.

    A wide violin = high sensitivity to MRG32k3a seed within that level,
    not to the swept factor itself.  Cross-level comparison of widths is
    therefore a robustness diagnostic.
    """
    _check_study(study)
    apply_journal_style()
    sub = finals[finals["study"] == study]
    problems = sorted(sub["problem"].unique())
    level_cols = list(LEVEL_COLUMNS[study])
    written: list[Path] = []

    for problem in problems:
        pdf = sub[sub["problem"] == problem]
        groups = pdf.groupby(level_cols, observed=False)
        data: list[np.ndarray] = []
        labels: list[str] = []
        for keys, g in groups:
            data.append(g["obj_postrep_mean"].to_numpy())
            if study == "basis":
                labels.append(f"{keys[0]}\nd={keys[1]}")
            elif study == "regularisation":
                labels.append(f"{keys[0]:g}")
            else:
                labels.append(str(keys[0]))
        if not data:
            continue
        setup_journal_plot(
            title=f"{study.upper()} on {problem}\nFinal-Objective Distribution",
            xlabel=_level_xlabel(study) if study != "basis" else "Level",
            ylabel="Final Objective",
        )
        positions = np.arange(1, len(data) + 1)
        plt.violinplot(data, positions=positions, showmeans=True, showextrema=False)
        plt.xticks(positions, labels, rotation=45, ha="right")
        if output_dir is not None:
            written += save_journal_plot(
                _stem(study, "final_objective_distribution", problem),
                output_dir=output_dir, formats=formats,
            )
        else:
            plt.close(plt.gcf())
    return written


# ---------------------------------------------------------------------------
# 5. Ranking plot — one figure per problem
# ---------------------------------------------------------------------------
def ranking_plot(
    finals: pd.DataFrame,
    *,
    study: str,
    baselines: Mapping[str, Any],
    top_k: int | None = None,
    xscale: str = "linear",
    output_dir: Path | None = None,
    formats: Sequence[str] = ("png",),
) -> list[Path]:
    """Horizontal-bar ranking of levels by absolute paired effect, per problem.

    ``top_k`` clips each panel to the N most-sensitive levels (recommended
    when basis-study levels exceed ~15).  ``xscale="log"`` is essential
    when a single catastrophic level dwarfs the rest by 3+ decades.
    """
    _check_study(study)
    apply_journal_style()
    sub = finals[finals["study"] == study]
    paired = paired_ci_against_baseline(sub, study=study, baselines=baselines)
    if paired.empty:
        return []
    problems = sorted(paired["problem"].unique())
    level_cols = list(LEVEL_COLUMNS[study])
    ranked = rank_levels_by_effect(paired, level_cols=level_cols)
    written: list[Path] = []

    for problem in problems:
        rdf = ranked[ranked["problem"] == problem].copy()
        if top_k is not None:
            rdf = rdf.iloc[:top_k]
        if rdf.empty:
            continue
        labels = [
            "(" + ",".join(str(rdf.iloc[i][c]) for c in level_cols) + ")"
            for i in range(len(rdf))
        ]
        values = rdf["mean_diff"].abs().to_numpy()
        # Canonical C-cycle colours: red-ish (C3) for worse than baseline,
        # blue (C0) for better.
        colours = ["C3" if d > 0 else "C0" for d in rdf["mean_diff"]]
        setup_journal_plot(
            title=f"{study.upper()} on {problem}\nSensitivity Ranking",
            xlabel="|Mean Diff vs. Baseline|",
            ylabel="Level",
        )
        y_positions = np.arange(len(rdf))[::-1]
        plt.barh(y_positions, values, color=colours)
        plt.yticks(y_positions, labels)
        if xscale != "linear":
            plt.xscale(xscale)
        if output_dir is not None:
            written += save_journal_plot(
                _stem(study, "ranking", problem),
                output_dir=output_dir, formats=formats,
            )
        else:
            plt.close(plt.gcf())
    return written


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def generate_all_figures(
    agg: AggregatedResults,
    *,
    output_dir: Path,
    formats: Sequence[str] = ("png",),
    baselines: Mapping[str, Mapping[str, Any]] | None = None,
    studies: Iterable[str] = STUDIES,
) -> list[Path]:
    """Generate every figure for every requested study.

    ``baselines`` is a nested mapping ``study -> problem -> baseline_value``.
    For the basis study the baseline value is a ``(basis, degree)`` tuple;
    for the OFAT studies it is a scalar (int or float).

    Returns the list of written file paths (one per figure x problem x format).
    Each study's outputs are placed under ``output_dir / <study> /``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    studies = tuple(studies)
    for study in studies:
        _check_study(study)
        long_form = agg.long_form[agg.long_form["study"] == study]
        finals = agg.finals[agg.finals["study"] == study]
        if long_form.empty:
            continue
        base_for = (baselines or {}).get(study, {})
        study_dir = output_dir / study
        study_dir.mkdir(parents=True, exist_ok=True)

        written += convergence_curves(
            long_form, study=study,
            output_dir=study_dir, formats=formats,
        )
        written += final_objective_summary_plot(
            finals, study=study,
            output_dir=study_dir, formats=formats,
        )
        if base_for:
            written += paired_effect_plot(
                finals, study=study, baselines=base_for,
                output_dir=study_dir, formats=formats,
            )
            written += ranking_plot(
                finals, study=study, baselines=base_for,
                output_dir=study_dir, formats=formats,
            )
        written += final_objective_distribution(
            finals, study=study,
            output_dir=study_dir, formats=formats,
        )
    return written


__all__ = [
    "convergence_curves",
    "final_objective_distribution",
    "final_objective_summary_plot",
    "generate_all_figures",
    "paired_effect_plot",
    "ranking_plot",
]
