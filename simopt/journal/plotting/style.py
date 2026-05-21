"""Journal style — rcParams, palette, basis colour map.

One :func:`apply_journal_style` call configures matplotlib globally; figure
factories then only need to worry about content.  Calling the function is
idempotent and safe to repeat between figures (e.g. when looping over
studies in the CLI).
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Colour-blind friendly Okabe-Ito palette (8 distinct hues).  We deliberately
# avoid Matplotlib's default ``tab10`` so figures remain readable when printed
# in greyscale.
OKABE_ITO: tuple[str, ...] = (
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#CC79A7",  # reddish purple
)

# Fixed colours for the nine polynomial bases so that, across every basis
# figure in the paper, "Hermite" is always blue, "Lagrange" is always purple,
# and so on.  Order matches scripts/journal_factors_test.py:POLY_BASIS_NAMES.
BASIS_COLOUR: dict[str, str] = {
    "Hermite":      "#0072B2",
    "Legendre":     "#009E73",
    "Chebyshev":    "#56B4E9",
    "Monomial":     "#E69F00",
    "Natural":      "#D55E00",
    "MonomialPoly": "#CC79A7",
    "Lagrange":     "#7E2D8C",
    "NFP":          "#000000",
    "Laguerre":     "#999999",
}

PROBLEM_MARKER: dict[str, str] = {
    "SAN-1":        "o",
    "ROSENBROCK-1": "s",
    "DYNAMNEWS-1":  "D",
    "NETWORK-1":    "^",
    "PARAMESTI-1":  "v",
}


def apply_journal_style() -> None:
    """Set rcParams suitable for two-column journal figures.

    Deliberately conservative: we do not force LaTeX rendering (which would
    fail on systems without a TeX install) but we *do* pick a serif font
    family and reasonable sizes.  Any caller can subsequently override these
    via :func:`matplotlib.rcParams.update`.
    """
    plt.rcdefaults()
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.prop_cycle": mpl.cycler(color=OKABE_ITO),
    })


def colour_for_level(study: str, level_value: object,
                     fallback_palette: tuple[str, ...] = OKABE_ITO) -> str:
    """Return a deterministic colour for a level value in a given study.

    For the basis study we map basis name → colour via :data:`BASIS_COLOUR`,
    so the same basis is the same colour across every figure.  For the
    subspace/regularisation studies we leave colour selection to the caller's
    colormap (it makes more sense to encode a numeric ordering); this helper
    only returns a categorical fallback.
    """
    if study == "basis":
        # level_value is a (basis, degree) tuple
        if isinstance(level_value, tuple):
            return BASIS_COLOUR.get(level_value[0], fallback_palette[0])
        return BASIS_COLOUR.get(str(level_value), fallback_palette[0])
    # Categorical fallback for everything else.
    return fallback_palette[hash(repr(level_value)) % len(fallback_palette)]


__all__ = [
    "BASIS_COLOUR",
    "OKABE_ITO",
    "PROBLEM_MARKER",
    "apply_journal_style",
    "colour_for_level",
]
