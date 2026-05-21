"""Figure factories for the ASTROMoRF sensitivity studies.

Each public function returns the :class:`matplotlib.figure.Figure` it built
(so callers can further tweak it) and optionally saves it to disk in one or
more formats.  Figures are deliberately kept narrow in scope — one factory
per plot type — so that the CLI orchestrator can compose them without
duplicating layout code.

Conventions
-----------
* x-axis is the swept factor for OFAT studies (subspace / regularisation).
* facet axes are one-per-problem in a grid laid out by
  :func:`_problem_grid`, which always returns a 2D layout regardless of the
  problem count (single-problem inputs are still 1x1 grids so callers can
  iterate uniformly).
* Regularisation values are plotted on a symlog x-axis so the explicit zero
  is visible alongside the seven log decades.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Colormap, TwoSlopeNorm
from matplotlib.figure import Figure

from .data import LEVEL_COLUMNS, STUDIES, AggregatedResults
from .stats import (
    final_objective_summary,
    paired_ci_against_baseline,
    rank_levels_by_effect,
)
from .style import (
    BASIS_COLOUR,
    OKABE_ITO,
    apply_journal_style,
)


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------
def _problem_grid(problems: Sequence[str]) -> tuple[int, int]:
    """Pick a (n_rows, n_cols) layout for the per-problem facet grid."""
    n = max(1, len(problems))
    n_cols = 2 if n > 1 else 1
    n_rows = math.ceil(n / n_cols)
    return n_rows, n_cols


def _facet_figure(
    problems: Sequence[str], *,
    base_size: tuple[float, float] = (3.4, 2.6),
) -> tuple[Figure, dict[str, plt.Axes]]:
    n_rows, n_cols = _problem_grid(problems)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(base_size[0] * n_cols, base_size[1] * n_rows),
        squeeze=False,
    )
    flat = [ax for row in axes for ax in row]
    mapping = {p: flat[i] for i, p in enumerate(problems)}
    for ax in flat[len(problems):]:
        ax.set_visible(False)
    return fig, mapping


def _savefig(fig: Figure, savepath: Path | None,
             formats: Sequence[str] = ("pdf",)) -> list[Path]:
    if savepath is None:
        return []
    paths: list[Path] = []
    base = Path(savepath)
    base.parent.mkdir(parents=True, exist_ok=True)
    if base.suffix:
        # caller gave an explicit extension — honour it verbatim
        fig.savefig(base)
        paths.append(base)
        return paths
    for fmt in formats:
        out = base.with_suffix(f".{fmt}")
        fig.savefig(out)
        paths.append(out)
    return paths


def _format_reg_xaxis(ax: plt.Axes) -> None:
    """Use symlog so 0.0 and 1e-6 share a sane axis."""
    ax.set_xscale("symlog", linthresh=1e-6, linscale=0.5)
    ax.set_xlabel("subproblem regularisation")


def _level_xaxis(ax: plt.Axes, study: str) -> None:
    if study == "subspace":
        ax.set_xlabel("subspace dimension")
    elif study == "regularisation":
        _format_reg_xaxis(ax)


# ---------------------------------------------------------------------------
# 1. Convergence curves with CI bands
# ---------------------------------------------------------------------------
def _sorted_levels(pdf: pd.DataFrame, level_cols: list[str]
                   ) -> list[tuple]:
    """Deterministic, study-agnostic ordering of factor levels.

    For numeric columns this is ordinary ascending numeric order; for the
    categorical basis name we sort lexicographically (a fixed reviewer-
    facing order that is stable across pandas / numpy versions).
    """
    out = (
        pdf[level_cols]
        .drop_duplicates()
        .sort_values(level_cols, kind="mergesort")
        .itertuples(index=False, name=None)
    )
    return list(out)


def _convergence_label(study: str, level: tuple) -> str:
    """Reviewer-facing label for a single (study, level) entry."""
    if study == "basis":
        return f"{level[0]}, deg={level[1]}"
    if study == "subspace":
        return f"d={level[0]}"
    return f"r={level[0]:g}"


def _attach_basis_legend(
    fig: Figure, level_handles: list, level_labels: list[str],
) -> None:
    """Attach a shared figure-level legend below the facet grid (categorical).

    The legend host is added via ``fig.add_axes`` and tagged with a
    ``"_legend"`` label so other layout helpers can ignore it.  We use a
    multi-column layout sized to keep individual entries readable for up to
    ~36 (basis x degree) cells.
    """
    n = len(level_labels)
    if n == 0:
        return
    ncol = min(6, max(2, math.ceil(n / 6)))
    fig.legend(
        level_handles, level_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=ncol,
        fontsize=7,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.0,
        borderaxespad=0.2,
    )


def _attach_numeric_colourbar(
    fig: Figure,
    axes_map: dict,
    cmap: Colormap,
    vmin: float,
    vmax: float,
    *,
    study: str,
    label: str,
    n_levels: int,
) -> None:
    """Attach a shared colourbar to the right of the facet grid (numeric)."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    norm = Normalize(vmin=vmin, vmax=vmax)
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    # Anchor on the rightmost data axis so the bar sits beside the grid.
    data_axes = list(axes_map.values())
    cb_ax = fig.add_axes([0.92, 0.15, 0.018, 0.7])
    cb_ax.set_label(f"{study}_cbar")
    fig.colorbar(mappable, cax=cb_ax, label=label,
                 ticks=_colourbar_ticks(vmin, vmax, n_levels))
    # Disable the rightmost-data-axis tick repeat for a tighter look.
    for ax in data_axes:
        ax.tick_params(axis="y", which="both", right=False)


def _colourbar_ticks(vmin: float, vmax: float, n_levels: int
                     ) -> list[float]:
    """Pick at most 6 ticks covering the swept range, integer when possible."""
    n = min(6, max(2, n_levels))
    ticks = np.linspace(vmin, vmax, n)
    if float(vmin).is_integer() and float(vmax).is_integer():
        ticks = np.unique(np.round(ticks).astype(int))
    return list(map(float, ticks))


def convergence_curves(
    long_form: pd.DataFrame,
    *,
    study: str,
    alpha_band: float = 0.25,
    xscale: str = "linear",
    yscale: str = "linear",
    savepath: Path | None = None,
    formats: Sequence[str] = ("pdf",),
) -> Figure:
    """Mean convergence trajectories per level, with macrorep spread as a band.

    The band is a percentile interval (10%-90%) across macroreps; it
    intentionally does not exclude outliers because outlier-heavy levels are
    the most diagnostically interesting in a sensitivity study.

    Legend / colourbar handling
    ---------------------------
    * Categorical studies (``basis``) emit a single, figure-level legend below
      the facet grid, with one entry per ``(polynomial_basis,
      polynomial_degree)`` cell.  Entries are sorted deterministically (basis
      name ascending, degree ascending) so figure regeneration is bit-stable.
    * Numeric OFAT studies (``subspace``, ``regularisation``) emit a shared
      colourbar to the right of the facet grid.  A colourbar is preferred to
      a discrete legend here because realistic level counts (~10-20 for
      subspace at d=100, ~13 for regularisation) make a categorical legend
      illegible.

    Neither artifact is drawn inside the data axes, so no curves are occluded.
    """
    if study not in STUDIES:
        raise ValueError(f"unknown study {study!r}")
    apply_journal_style()
    sub = long_form[long_form["study"] == study]
    problems = sorted(sub["problem"].unique())
    fig, axes = _facet_figure(problems)

    level_cols = list(LEVEL_COLUMNS[study])
    # Union of all levels seen across problems — guarantees a single legend
    # entry per unique level even if a problem omits some.
    all_levels = _sorted_levels(sub, level_cols)
    cmap = plt.get_cmap("viridis", max(2, len(all_levels)))
    # Per-level colour resolution is computed *once* so the same level uses
    # the same colour across every problem facet (essential when the legend
    # / colourbar is shared at the figure level).
    level_colour: dict[tuple, str] = {}
    for i, lvl in enumerate(all_levels):
        if study == "basis":
            level_colour[lvl] = BASIS_COLOUR.get(
                lvl[0], OKABE_ITO[i % len(OKABE_ITO)],
            )
        else:
            level_colour[lvl] = cmap(i)

    # Collect first-seen handles so the figure legend has exactly one entry
    # per level regardless of how many facets it appears in.
    legend_handles: dict[tuple, Any] = {}

    for problem in problems:
        ax = axes[problem]
        pdf = sub[sub["problem"] == problem]
        for lvl in all_levels:
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
            colour = level_colour[lvl]
            label = _convergence_label(study, lvl)
            (line,) = ax.plot(
                mean.index, mean.values, color=colour, label=label,
            )
            ax.fill_between(
                mean.index, lo.values, hi.values,
                color=colour, alpha=alpha_band, linewidth=0,
            )
            legend_handles.setdefault(lvl, line)
        ax.set_title(problem)
        ax.set_xlabel("budget")
        ax.set_ylabel("post-replicated mean objective")
        if xscale != "linear":
            ax.set_xscale(xscale)
        if yscale != "linear":
            ax.set_yscale(yscale)
    fig.suptitle(
        f"ASTROMoRF convergence - {study} sensitivity", fontsize=11,
    )

    if study == "basis":
        handles = [legend_handles[lvl] for lvl in all_levels
                   if lvl in legend_handles]
        labels = [_convergence_label(study, lvl) for lvl in all_levels
                  if lvl in legend_handles]
        # Reserve bottom room for the legend so it never overlaps the axes
        # and is not clipped when saved to PDF.
        rows = math.ceil(len(handles) / 6)
        bottom_frac = 0.06 + 0.025 * rows
        fig.tight_layout(rect=(0, bottom_frac, 1, 0.95))
        _attach_basis_legend(fig, handles, labels)
    else:
        # Numeric: shared colourbar to the right.
        flat_levels = [float(lvl[0]) for lvl in all_levels]
        fig.tight_layout(rect=(0, 0, 0.9, 0.95))
        _attach_numeric_colourbar(
            fig, axes, cmap=cmap,
            vmin=min(flat_levels), vmax=max(flat_levels),
            study=study,
            label=("subspace dimension" if study == "subspace"
                   else "subproblem regularisation"),
            n_levels=len(flat_levels),
        )
    _savefig(fig, savepath, formats)
    return fig


# ---------------------------------------------------------------------------
# 2. Final-objective summary plot
# ---------------------------------------------------------------------------
def final_objective_summary_plot(
    finals: pd.DataFrame,
    *,
    study: str,
    robust_quantiles: tuple[float, float] | None = None,
    savepath: Path | None = None,
    formats: Sequence[str] = ("pdf",),
) -> Figure:
    """Per-problem summary of the final post-replicated objective.

    * subspace / regularisation: line + 95% CI error bars vs. level.
    * basis: per-problem heatmap on (basis x degree).

    When ``robust_quantiles=(lo, hi)`` is supplied (basis only), the per-
    problem colour scale is clipped to those quantiles of the cell values.
    This prevents a single catastrophic cell (e.g. a degree-4 NFP cell that
    diverges on NETWORK-1) from saturating the entire colormap, while
    cells outside the clip still render at the extreme colours.
    """
    if study not in STUDIES:
        raise ValueError(f"unknown study {study!r}")
    apply_journal_style()
    sub = finals[finals["study"] == study]
    summary = final_objective_summary(sub, study=study)
    problems = sorted(summary["problem"].unique())
    fig, axes = _facet_figure(problems)

    if study == "basis":
        # Heatmap: rows=basis, cols=degree, colour=mean_final_obj.
        bases = sorted(summary["polynomial_basis"].unique())
        degrees = sorted(summary["polynomial_degree"].unique())
        for problem in problems:
            ax = axes[problem]
            pdf = summary[summary["problem"] == problem]
            mat = np.full((len(bases), len(degrees)), np.nan)
            for _, row in pdf.iterrows():
                i = bases.index(row["polynomial_basis"])
                j = degrees.index(int(row["polynomial_degree"]))
                mat[i, j] = row["mean_final_obj"]
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
            im = ax.imshow(mat, **imshow_kwargs)
            ax.set_xticks(range(len(degrees)))
            ax.set_xticklabels([str(d) for d in degrees])
            ax.set_yticks(range(len(bases)))
            ax.set_yticklabels(bases)
            ax.set_xlabel("polynomial degree")
            ax.set_ylabel("polynomial basis")
            ax.set_title(problem)
            fig.colorbar(im, ax=ax, shrink=0.85,
                         label="mean final obj"
                         + (" (clipped)" if robust_quantiles else ""))
        fig.suptitle("ASTROMoRF final objective — basis x degree", fontsize=11)
    else:
        level_col = LEVEL_COLUMNS[study][0]
        for problem in problems:
            ax = axes[problem]
            pdf = (
                summary[summary["problem"] == problem]
                .sort_values(level_col)
            )
            ax.errorbar(
                pdf[level_col].astype(float).values,
                pdf["mean_final_obj"].values,
                yerr=[
                    (pdf["mean_final_obj"] - pdf["ci95_lo"]).values,
                    (pdf["ci95_hi"] - pdf["mean_final_obj"]).values,
                ],
                fmt="o-", color=OKABE_ITO[5], capsize=2,
            )
            _level_xaxis(ax, study)
            ax.set_ylabel("mean final obj")
            ax.set_title(problem)
        fig.suptitle(
            f"ASTROMoRF final objective — {study} sweep", fontsize=11,
        )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _savefig(fig, savepath, formats)
    return fig


# ---------------------------------------------------------------------------
# 3. Paired-effect plot
# ---------------------------------------------------------------------------
def paired_effect_plot(
    finals: pd.DataFrame,
    *,
    study: str,
    baselines: Mapping[str, Any],
    robust_quantiles: tuple[float, float] | None = None,
    savepath: Path | None = None,
    formats: Sequence[str] = ("pdf",),
) -> Figure:
    """Paired (level - baseline) effect with 95% bootstrap CI per problem.

    For subspace / regularisation the result is a line plot with a shaded CI
    band and a horizontal zero baseline marker.  For basis it is a diverging-
    colormap heatmap of ``mean_diff`` with hatching wherever the CI excludes
    zero (i.e. statistical significance after CRN pairing).

    When ``robust_quantiles=(lo, hi)`` is supplied (basis only), the
    diverging colour scale's symmetric ``vmax`` is clipped to that quantile
    of ``|mean_diff|`` so a single catastrophic cell cannot swallow the
    visible dynamic range.  Cells outside the clip still render at the
    extreme colours.
    """
    if study not in STUDIES:
        raise ValueError(f"unknown study {study!r}")
    apply_journal_style()
    sub = finals[finals["study"] == study]
    paired = paired_ci_against_baseline(
        sub, study=study, baselines=baselines,
    )
    problems = sorted(paired["problem"].unique()) if not paired.empty else []
    fig, axes = _facet_figure(problems if problems else ["(no data)"])

    if not problems:
        return fig

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
            ax = axes[problem]
            pdf = paired[paired["problem"] == problem]
            mat = np.full((len(bases), len(degrees)), np.nan)
            sig = np.zeros_like(mat, dtype=bool)
            for _, row in pdf.iterrows():
                i = bases.index(row["polynomial_basis"])
                j = degrees.index(int(row["polynomial_degree"]))
                mat[i, j] = row["mean_diff"]
                sig[i, j] = (row["ci95_lo"] > 0) or (row["ci95_hi"] < 0)
            im = ax.imshow(
                mat, aspect="auto", origin="lower", cmap="RdBu_r",
                norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
            )
            # hatch significant cells
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
            ax.set_xlabel("polynomial degree")
            ax.set_ylabel("polynomial basis")
            ax.set_title(problem)
            fig.colorbar(im, ax=ax, shrink=0.85, label="mean diff vs. baseline")
        fig.suptitle(
            "ASTROMoRF paired effect — basis x degree (CRN)", fontsize=11,
        )
    else:
        level_col = LEVEL_COLUMNS[study][0]
        for problem in problems:
            ax = axes[problem]
            pdf = (
                paired[paired["problem"] == problem]
                .sort_values(level_col)
            )
            x = pdf[level_col].astype(float).values
            mean = pdf["mean_diff"].values
            lo = pdf["ci95_lo"].values
            hi = pdf["ci95_hi"].values
            ax.fill_between(x, lo, hi, alpha=0.25, color=OKABE_ITO[5])
            ax.plot(x, mean, "o-", color=OKABE_ITO[5])
            ax.axhline(0.0, color="black", lw=0.5)
            # Make sure 0 stays in view.
            y_lo, y_hi = ax.get_ylim()
            ax.set_ylim(min(y_lo, -1e-6), max(y_hi, 1e-6))
            _level_xaxis(ax, study)
            ax.set_ylabel("paired mean diff (level - baseline)")
            ax.set_title(problem)
        fig.suptitle(
            f"ASTROMoRF paired effect — {study} sweep (CRN)", fontsize=11,
        )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _savefig(fig, savepath, formats)
    return fig


# ---------------------------------------------------------------------------
# 4. Distribution / robustness (violin)
# ---------------------------------------------------------------------------
def final_objective_distribution(
    finals: pd.DataFrame,
    *,
    study: str,
    savepath: Path | None = None,
    formats: Sequence[str] = ("pdf",),
) -> Figure:
    """Violin plot of macrorep final-objective spread per level.

    A wide violin = high sensitivity to MRG32k3a seed within that level,
    *not* to the swept factor itself.  Cross-level comparison of violin
    widths is therefore a robustness diagnostic.
    """
    if study not in STUDIES:
        raise ValueError(f"unknown study {study!r}")
    apply_journal_style()
    sub = finals[finals["study"] == study]
    problems = sorted(sub["problem"].unique())
    fig, axes = _facet_figure(problems)

    level_cols = list(LEVEL_COLUMNS[study])
    for problem in problems:
        ax = axes[problem]
        pdf = sub[sub["problem"] == problem]
        groups = pdf.groupby(level_cols, observed=False)
        data = []
        labels = []
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
        positions = np.arange(1, len(data) + 1)
        ax.violinplot(data, positions=positions, showmeans=True,
                      showextrema=False)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("final objective")
        ax.set_title(problem)
    fig.suptitle(
        f"ASTROMoRF final-objective distribution — {study}", fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _savefig(fig, savepath, formats)
    return fig


# ---------------------------------------------------------------------------
# 5. Ranking plot
# ---------------------------------------------------------------------------
def ranking_plot(
    finals: pd.DataFrame,
    *,
    study: str,
    baselines: Mapping[str, Any],
    top_k: int | None = None,
    xscale: str = "linear",
    savepath: Path | None = None,
    formats: Sequence[str] = ("pdf",),
) -> Figure:
    """Horizontal-bar ranking of levels by absolute paired effect.

    For each problem, level labels are placed on the y-axis and bar lengths
    encode ``|mean_diff|``.  Bars are coloured red where ``mean_diff > 0``
    (level is *worse* than the baseline) and blue otherwise.

    Real-data escape hatches:

    * ``top_k`` clips each panel to the top-N most-sensitive levels.  At
      large level counts (e.g. basis = 36) the default behaviour produces
      illegible y-tick stacks; ``top_k=10`` is the journal-recommended
      value.
    * ``xscale="log"`` switches to a logarithmic |mean_diff| axis, which
      is essential when a single catastrophic level dwarfs the rest of the
      ranking by 3+ decades.
    """
    if study not in STUDIES:
        raise ValueError(f"unknown study {study!r}")
    apply_journal_style()
    sub = finals[finals["study"] == study]
    paired = paired_ci_against_baseline(
        sub, study=study, baselines=baselines,
    )
    problems = sorted(paired["problem"].unique()) if not paired.empty else []
    fig, axes = _facet_figure(problems if problems else ["(no data)"])
    if not problems:
        return fig
    level_cols = list(LEVEL_COLUMNS[study])
    ranked = rank_levels_by_effect(paired, level_cols=level_cols)
    for problem in problems:
        ax = axes[problem]
        rdf = ranked[ranked["problem"] == problem].copy()
        if top_k is not None:
            rdf = rdf.iloc[:top_k]
        labels = [
            "(" + ",".join(str(rdf.iloc[i][c]) for c in level_cols) + ")"
            for i in range(len(rdf))
        ]
        values = rdf["mean_diff"].abs().to_numpy()
        colours = [
            OKABE_ITO[6] if d > 0 else OKABE_ITO[2]
            for d in rdf["mean_diff"]
        ]
        y_positions = np.arange(len(rdf))[::-1]
        ax.barh(y_positions, values, color=colours)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("|mean diff vs. baseline|")
        ax.set_title(problem)
        if xscale != "linear":
            ax.set_xscale(xscale)
    fig.suptitle(
        f"ASTROMoRF sensitivity ranking — {study}", fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _savefig(fig, savepath, formats)
    return fig


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def generate_all_figures(
    agg: AggregatedResults,
    *,
    output_dir: Path,
    formats: Sequence[str] = ("pdf",),
    baselines: Mapping[str, Mapping[str, Any]] | None = None,
    studies: Iterable[str] = STUDIES,
) -> list[Path]:
    """Generate every figure for every requested study.

    ``baselines`` is a nested mapping ``study → problem → baseline_value``.
    For the basis study the baseline value is a ``(basis, degree)`` tuple;
    for the OFAT studies it is a scalar (int or float).

    Returns the list of written file paths (one per figure x format).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    studies = tuple(studies)
    for study in studies:
        if study not in STUDIES:
            raise ValueError(f"unknown study {study!r}")
        long_form = agg.long_form[agg.long_form["study"] == study]
        finals = agg.finals[agg.finals["study"] == study]
        if long_form.empty:
            continue
        base_for = (baselines or {}).get(study, {})
        study_dir = output_dir / study
        study_dir.mkdir(parents=True, exist_ok=True)

        # (1) convergence
        fig = convergence_curves(
            long_form, study=study,
            savepath=study_dir / "convergence_curves",
            formats=formats,
        )
        written += _glob_outputs(study_dir / "convergence_curves", formats)
        plt.close(fig)

        # (2) final-objective summary
        fig = final_objective_summary_plot(
            finals, study=study,
            savepath=study_dir / "final_objective_summary",
            formats=formats,
        )
        written += _glob_outputs(study_dir / "final_objective_summary", formats)
        plt.close(fig)

        if base_for:
            # (3) paired effect
            fig = paired_effect_plot(
                finals, study=study, baselines=base_for,
                savepath=study_dir / "paired_effect",
                formats=formats,
            )
            written += _glob_outputs(study_dir / "paired_effect", formats)
            plt.close(fig)

            # (5) ranking
            fig = ranking_plot(
                finals, study=study, baselines=base_for,
                savepath=study_dir / "ranking",
                formats=formats,
            )
            written += _glob_outputs(study_dir / "ranking", formats)
            plt.close(fig)

        # (4) distribution
        fig = final_objective_distribution(
            finals, study=study,
            savepath=study_dir / "final_objective_distribution",
            formats=formats,
        )
        written += _glob_outputs(
            study_dir / "final_objective_distribution", formats,
        )
        plt.close(fig)
    return written


def _glob_outputs(stem: Path, formats: Sequence[str]) -> list[Path]:
    return [stem.with_suffix(f".{f}") for f in formats]


__all__ = [
    "convergence_curves",
    "final_objective_distribution",
    "final_objective_summary_plot",
    "generate_all_figures",
    "paired_effect_plot",
    "ranking_plot",
]
