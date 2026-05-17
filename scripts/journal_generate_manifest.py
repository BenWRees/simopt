"""Generate the journal-sensitivity manifest — the single source of truth.

Run this *once* before submitting the SLURM array.  The manifest pinpoints
every (study, problem, design point) combination, the exact factor values, the
shared global settings (budget, n_macroreps, n_postreps, CRN flags), and the
output layout under ``<output_root>/runs/<problem>/<study>/<dpid>/``.

Examples
--------
Submit the four-problem canonical study::

    python scripts/journal_generate_manifest.py \
        --output-root $HOME/results/journal \
        --problems SAN-1 ROSENBROCK-1 DYNAMNEWS-1 NETWORK-1 \
        --dims SAN-1=20,ROSENBROCK-1=10,DYNAMNEWS-1=10,NETWORK-1=14 \
        --budget 10000 --n-macroreps 30 --n-postreps 600

This writes ``$HOME/results/journal/manifest.json`` and ``manifest.csv``.

The SLURM script reads these two files only; nothing about study layout,
factor levels, or task counts is duplicated in shell.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow direct invocation from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.journal_factors_test import (  # noqa: E402
    GlobalConfig,
    STUDY_NAMES,
    build_manifest,
    write_manifest,
)


def _parse_dims(spec: str) -> dict[str, int]:
    """Parse ``PROBLEM=DIM,PROBLEM=DIM,...`` into a dict."""
    out: dict[str, int] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise argparse.ArgumentTypeError(
                f"--dims entry {part!r} must be PROBLEM=DIM"
            )
        name, dim_s = part.split("=", 1)
        try:
            dim = int(dim_s)
        except ValueError as e:
            raise argparse.ArgumentTypeError(
                f"--dims entry {part!r} has non-integer dim"
            ) from e
        if dim < 1:
            raise argparse.ArgumentTypeError(
                f"--dims entry {part!r} has non-positive dim"
            )
        out[name.strip()] = dim
    if not out:
        raise argparse.ArgumentTypeError("--dims may not be empty")
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the journal-sensitivity manifest "
            "(single source of truth for the SLURM array)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-root", type=Path, required=True,
        help="Directory under which manifest.json, manifest.csv, and runs/ live.",
    )
    parser.add_argument(
        "--problems", nargs="+", required=True,
        help="Problem names to include (e.g. SAN-1 ROSENBROCK-1).",
    )
    parser.add_argument(
        "--dims", type=_parse_dims, required=True,
        help="Per-problem scaled dimensions, as PROBLEM=DIM,PROBLEM=DIM,...",
    )
    parser.add_argument(
        "--studies", nargs="+", default=list(STUDY_NAMES),
        choices=list(STUDY_NAMES),
        help="Studies to include (default: all three).",
    )
    parser.add_argument("--budget", type=int, default=10000)
    parser.add_argument("--n-macroreps", type=int, default=30)
    parser.add_argument("--n-postreps", type=int, default=600)
    parser.add_argument("--crn-across-budget", action="store_true", default=True)
    parser.add_argument("--no-crn-across-budget",
                        dest="crn_across_budget", action="store_false")
    parser.add_argument("--crn-across-macroreps",
                        action="store_true", default=False)
    parser.add_argument("--crn-across-solns",
                        action="store_true", default=True)
    parser.add_argument("--no-crn-across-solns",
                        dest="crn_across_solns", action="store_false")
    parser.add_argument(
        "--macroreps-per-chunk", type=int, default=5,
        help="Chunk size for resumable runs (see ProblemSolver.run_resumable).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cfg = GlobalConfig(
        budget=args.budget,
        n_macroreps=args.n_macroreps,
        n_postreps=args.n_postreps,
        crn_across_budget=args.crn_across_budget,
        crn_across_macroreps=args.crn_across_macroreps,
        crn_across_solns=args.crn_across_solns,
        macroreps_per_chunk=args.macroreps_per_chunk,
    )
    manifest = build_manifest(
        problems=args.problems,
        dims=args.dims,
        studies=args.studies,
        cfg=cfg,
        output_root=args.output_root,
    )
    json_path, csv_path = write_manifest(manifest, args.output_root)
    print(f"manifest_json {json_path}")
    print(f"manifest_csv  {csv_path}")
    print(f"total_tasks   {manifest['total_tasks']}")
    print(f"manifest_sha  {manifest['manifest_sha256']}")
    for study, by_problem in manifest["cell_sizes"].items():
        total_study = sum(by_problem.values())
        per = ", ".join(f"{p}={n}" for p, n in by_problem.items())
        print(f"  {study:14s} {total_study:>4d} tasks  ({per})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
