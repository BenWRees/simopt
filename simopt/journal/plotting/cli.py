"""CLI entry-point for the journal sensitivity-analysis figure generator.

Usage::

    python scripts/generate_astromorf_sensitivity_figures.py \
        --analysis-dir $HOME/results/journal/analysis \
        --output-dir figures/ \
        --studies subspace basis regularisation \
        --formats pdf png

If ``--baselines-json`` is omitted, the baselines registry is read from
``scripts/journal_factors_test.py:PROBLEM_OPERATING_POINT`` directly — this
keeps the CLI in sync with whatever the experiment pipeline considers the
operating point for each problem.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

from . import STUDIES
from .data import load_aggregated
from .figures import generate_all_figures


def _baselines_from_operating_points() -> dict[str, dict[str, Any]]:
    """Pull the per-problem operating-point baselines from the experiment code.

    Falls back to an empty mapping when the experiment package is not
    importable from this Python environment (CI sandboxes, etc.); paired
    plots are then skipped cleanly rather than failing.
    """
    try:
        sys.path.insert(
            0, str(Path(__file__).resolve().parents[3])
        )
        from scripts.journal_factors_test import (
            POLY_BASIS_NAMES,
            PROBLEM_OPERATING_POINT,
        )
    except Exception:
        return {}

    subspace: dict[str, int] = {}
    regularisation: dict[str, float] = {}
    basis: dict[str, tuple[str, int]] = {}
    for problem, op in PROBLEM_OPERATING_POINT.items():
        subspace[problem] = int(op["subspace_dimension"])
        regularisation[problem] = float(op["subproblem_regularisation"])
        basis_name = POLY_BASIS_NAMES[op["polynomial_basis"]]
        basis[problem] = (basis_name, int(op["polynomial_degree"]))
    return {
        "subspace": subspace,
        "regularisation": regularisation,
        "basis": basis,
    }


def _baselines_from_json(path: Path) -> dict[str, dict[str, Any]]:
    """Parse a user-supplied baselines JSON file.

    Expected shape::

        {
          "subspace":       { "SAN-1": 3, ... },
          "regularisation": { "SAN-1": 0.01, ... },
          "basis":          { "SAN-1": ["Hermite", 2], ... }
        }
    """
    raw = json.loads(Path(path).read_text())
    out: dict[str, dict[str, Any]] = {}
    for study, mapping in raw.items():
        if study not in STUDIES:
            raise SystemExit(f"baselines JSON: unknown study {study!r}")
        out[study] = {}
        for problem, value in mapping.items():
            if study == "basis":
                if not (isinstance(value, list) and len(value) == 2):
                    raise SystemExit(
                        f"baselines JSON: basis baseline for {problem!r} "
                        f"must be [basis_name, degree]; got {value!r}"
                    )
                out[study][problem] = (str(value[0]), int(value[1]))
            elif study == "subspace":
                out[study][problem] = int(value)
            else:
                out[study][problem] = float(value)
    return out


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the figure-generation CLI."""
    p = argparse.ArgumentParser(
        description=(
            "Generate journal-quality ASTROMoRF sensitivity-analysis "
            "figures from the artefacts written by scripts/journal_aggregate.py."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--analysis-dir", type=Path, required=True,
        help="Directory containing journal_long_form.* and the summary CSVs.",
    )
    p.add_argument(
        "--output-dir", type=Path, required=True,
        help="Where to write figures (one subdir per study).",
    )
    p.add_argument(
        "--studies", nargs="+", default=list(STUDIES), choices=list(STUDIES),
        help="Which studies to generate.",
    )
    p.add_argument(
        "--formats", nargs="+", default=["pdf", "png"],
        help="Vector + raster formats to write (e.g. pdf svg png).",
    )
    p.add_argument(
        "--baselines-json", type=Path, default=None,
        help=(
            "Optional JSON mapping {study: {problem: baseline_value}}. "
            "Defaults to PROBLEM_OPERATING_POINT in journal_factors_test.py."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Generate journal-quality sensitivity figures from CLI arguments."""
    args = build_parser().parse_args(argv)
    if args.baselines_json is not None:
        baselines = _baselines_from_json(args.baselines_json)
    else:
        baselines = _baselines_from_operating_points()
    agg = load_aggregated(args.analysis_dir)
    written = generate_all_figures(
        agg,
        output_dir=args.output_dir,
        formats=tuple(args.formats),
        baselines=baselines,
        studies=tuple(args.studies),
    )
    print(f"wrote {len(written)} figure file(s) under {args.output_dir}")
    for p in written:
        print(f"  - {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
