"""Journal plotting style — matches the canonical SimOpt aesthetic.

This module deliberately avoids introducing a parallel styling system.
The canonical SimOpt plotting code (``simopt.plots.utils.setup_plot`` /
``save_plot`` and the ``Curve.plot`` helpers used by
``plot_progress_curves`` and ``plot_solvability_profiles``) relies on
matplotlib's library defaults plus a small number of overrides applied
inline.  We mirror those overrides here so journal sensitivity figures
look indistinguishable from the rest of the SimOpt visualisation suite.

Conventions reproduced from ``simopt.plots.utils``:

* ``plt.figure()`` with the default figsize.
* ``plt.title(..., size=14)``.
* ``plt.xlabel(..., size=14)`` / ``plt.ylabel(..., size=14)``.
* ``plt.tick_params(axis="both", which="major", labelsize=12)``.
* default ``tab10`` colour cycle (``"C0"``, ``"C1"``, ...).
* data line width 2 / confidence-bound width 1 (matches ``CurveType``).
* fill_between alpha 0.2 for bootstrap CI bands.
* legend with default frame, ``leg.get_frame().set_alpha(0.4)``.
* ``plt.savefig(path, bbox_inches="tight")`` for export.
"""
from __future__ import annotations

import contextlib
from collections.abc import Sequence

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.legend import Legend

# Stable colour-cycle index per polynomial basis.  Using the default
# ``tab10`` cycle (resolved as "C{i}") keeps basis figures visually
# consistent with the rest of the SimOpt plotting code, which also draws
# from this cycle for solver/problem distinguishing colours.
BASIS_COLOUR_INDEX: dict[str, int] = {
    "Hermite": 0,
    "Legendre": 1,
    "Chebyshev": 2,
    "Monomial": 3,
    "Natural": 4,
    "MonomialPoly": 5,
    "Lagrange": 6,
    "NFP": 7,
    "Laguerre": 8,
}

PROBLEM_MARKER: dict[str, str] = {
    "SAN-1": "o",
    "ROSENBROCK-1": "s",
    "DYNAMNEWS-1": "D",
    "NETWORK-1": "^",
    "PARAMESTI-1": "v",
}

# Title / label / tick sizes used by every canonical SimOpt plot.
TITLE_SIZE = 14
LABEL_SIZE = 14
TICK_SIZE = 12
DATA_LINEWIDTH = 2
CI_LINEWIDTH = 1
CI_ALPHA = 0.2
LEGEND_FRAME_ALPHA = 0.4


def apply_journal_style() -> None:
    """Reset matplotlib to defaults so journal plots inherit the canonical style.

    The canonical SimOpt plotting helpers do not override rcParams; they
    apply sizing inline.  Calling this function resets any prior styling
    state (e.g. a previous ``mpl.rcParams.update`` call) so figures built
    in the same session render identically to ``plot_progress_curves``.
    Idempotent and safe to call repeatedly.
    """
    plt.rcdefaults()


def colour_for_level(study: str, level_value: object) -> str:
    """Return a deterministic ``"C{i}"`` colour for a level value.

    For the basis study, basis names map through ``BASIS_COLOUR_INDEX``
    so the same basis is always drawn in the same colour.  For numeric
    studies (subspace / regularisation) we fall back to a hash of the
    level so the same numeric level reuses the same cycle slot across
    figures.
    """
    if study == "basis":
        if isinstance(level_value, tuple):
            name = str(level_value[0])
        else:
            name = str(level_value)
        idx = BASIS_COLOUR_INDEX.get(name, 0)
        return f"C{idx % 10}"
    return f"C{hash(repr(level_value)) % 10}"


def colour_for_index(idx: int) -> str:
    """Return ``"C{i}"`` for the default tab10 cycle, wrapping at 10."""
    return f"C{idx % 10}"


def style_legend(leg: Legend | None) -> None:
    """Apply the canonical legend frame transparency.

    Mirrors the snippet repeated throughout ``simopt.plots`` after every
    ``plt.legend(...)`` call.
    """
    if leg is None:
        return
    with contextlib.suppress(Exception):
        leg.get_frame().set_alpha(LEGEND_FRAME_ALPHA)


def setup_journal_plot(
    *,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    """Create a new figure and apply canonical axis / title formatting.

    This is the journal-side analogue of
    ``simopt.plots.utils.setup_plot`` — same call order, same font sizes,
    same default figsize.  Plot content (lines, fills, scatter, ...) is
    added by the caller after this returns.
    """
    plt.figure()
    plt.title(title, size=TITLE_SIZE)
    plt.xlabel(xlabel, size=LABEL_SIZE)
    plt.ylabel(ylabel, size=LABEL_SIZE)
    plt.tick_params(axis="both", which="major", labelsize=TICK_SIZE)


def save_journal_plot(
    stem: str,
    *,
    output_dir,  # type: ignore[no-untyped-def]
    formats: Sequence[str] = ("png",),
    fig: Figure | None = None,
    close: bool = True,
) -> list:
    """Save the current (or supplied) figure to one path per format.

    Filenames are sanitised the same way ``simopt.plots.utils.save_plot``
    does (drop ``$``, ``\\``, collapse spaces to ``_``) so journal figure
    paths are interchangeable with canonical ones.  Returns the list of
    written :class:`pathlib.Path` objects.
    """
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clean = stem.replace("\\", "").replace("$", "").replace(" ", "_")
    target = fig if fig is not None else plt.gcf()
    written: list[Path] = []
    for fmt in formats:
        out = output_dir / f"{clean}.{fmt.lstrip('.')}"
        target.savefig(out, bbox_inches="tight")
        written.append(out)
    if close:
        plt.close(target)
    return written


__all__ = [
    "BASIS_COLOUR_INDEX",
    "CI_ALPHA",
    "CI_LINEWIDTH",
    "DATA_LINEWIDTH",
    "LABEL_SIZE",
    "LEGEND_FRAME_ALPHA",
    "PROBLEM_MARKER",
    "TICK_SIZE",
    "TITLE_SIZE",
    "apply_journal_style",
    "colour_for_index",
    "colour_for_level",
    "save_journal_plot",
    "setup_journal_plot",
    "style_legend",
]
