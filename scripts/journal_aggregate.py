"""Aggregate per-task long-form outputs into journal-ready summaries.

This script is the analysis layer.  It is intentionally separate from the
execution engine so analytical choices (which CI method, which alpha, which
multiple-comparison correction) can be revised without re-running any
expensive solves.

Inputs
------
A ``runs/`` tree produced by ``scripts/run_journal_factors.slurm``:

    runs/<problem>/<study>/<design_point_id>/long_form.{parquet,csv.gz}

Outputs (under ``--output-dir``)
--------------------------------
    journal_long_form.parquet         All per-(macrorep, budget) rows.
    final_objectives.parquet          One row per (task, macrorep) at the
                                      final budget.
    summary_<study>.csv               Marginal mean + std + 95% CI per level.
    paired_ci_<study>.csv             Paired bootstrap CI vs. the operating-
                                      point baseline, per level.

The paired CI exploits the simopt framework's CRN-by-macrorep property: for
the same ``mrep`` index, all design points share the same simulation streams,
so the per-macrorep difference between two design points has dramatically
reduced variance compared with the unpaired difference of means.
"""
from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.journal_factors_test import (  # noqa: E402
    PROBLEM_OPERATING_POINT,
    POLY_BASIS_NAMES,
    _parquet_engine,
    collect_long_form,
)


def _operating_point_label(problem: str, study: str) -> tuple[str, object]:
    """Return (column_name, baseline_value) for the per-study baseline.

    The baseline is the operating-point setting of the swept factor; paired
    CIs are reported as ``level − baseline``.
    """
    op = PROBLEM_OPERATING_POINT[problem]
    if study == "subspace":
        return "subspace_dim", int(op["subspace_dimension"])
    if study == "regularisation":
        return "subproblem_regularisation", float(op["subproblem_regularisation"])
    if study == "basis":
        # For the basis × degree factorial the baseline is the (basis, degree)
        # at the operating point.
        return ("polynomial_basis", "polynomial_degree"), (
            POLY_BASIS_NAMES[op["polynomial_basis"]],
            int(op["polynomial_degree"]),
        )
    raise ValueError(f"unknown study {study!r}")


def _bootstrap_ci(
    sample: np.ndarray,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI of the mean of ``sample``.

    Returns (mean, ci_lo, ci_hi).
    """
    if sample.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = rng or np.random.default_rng(0xC0FFEE)
    n = sample.size
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = sample[idx].mean(axis=1)
    lo, hi = np.quantile(boot_means, [alpha / 2, 1 - alpha / 2])
    return float(sample.mean()), float(lo), float(hi)


def compute_final_objectives(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (task, macrorep) at the final budget."""
    if df.empty:
        return df
    return df[df["is_final_budget"]].reset_index(drop=True)


def summary_table(finals: pd.DataFrame, study: str) -> pd.DataFrame:
    """Per-level marginal mean / std / CI of the final objective.

    NOTE: marginal statistics are unpaired and do not exploit CRN.  Use the
    paired_ci table for inferential claims.
    """
    if finals.empty:
        return finals
    g = finals.groupby(["problem", _level_columns(study)], dropna=False, observed=False)
    out = []
    for keys, sub in g:
        problem = keys[0]
        level = keys[1] if len(keys) == 2 else keys[1:]
        mean, lo, hi = _bootstrap_ci(sub["obj_postrep_mean"].to_numpy())
        out.append({
            "problem": problem,
            "level": level,
            "n_macroreps": int(sub.shape[0]),
            "mean_final_obj": mean,
            "ci95_lo": lo,
            "ci95_hi": hi,
            "std_final_obj":
                float(sub["obj_postrep_mean"].std(ddof=1))
                if sub.shape[0] > 1 else float("nan"),
        })
    return pd.DataFrame(out)


def _level_columns(study: str) -> str | list[str]:
    if study == "subspace":
        return "subspace_dim"
    if study == "regularisation":
        return "subproblem_regularisation"
    if study == "basis":
        return ["polynomial_basis", "polynomial_degree"]
    raise ValueError(study)


def paired_ci_table(
    finals: pd.DataFrame,
    study: str,
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Paired bootstrap CI of (level − baseline) using the macrorep axis.

    For each problem, the baseline is the operating-point level of the swept
    factor.  Because the simopt framework constructs MRG32k3a substreams
    keyed on ``mrep``, the per-macrorep difference is a true paired sample.
    """
    if finals.empty:
        return finals
    out = []
    for problem, prob_df in finals.groupby("problem", observed=False):
        baseline_col, baseline_val = _operating_point_label(problem, study)
        if isinstance(baseline_col, str):
            base_mask = prob_df[baseline_col] == baseline_val
        else:
            base_mask = (
                (prob_df[baseline_col[0]] == baseline_val[0])
                & (prob_df[baseline_col[1]] == baseline_val[1])
            )
        if not base_mask.any():
            # Baseline design point is not in the data — skip cleanly.
            continue
        base = prob_df[base_mask].set_index("macrorep")["obj_postrep_mean"]
        level_cols = _level_columns(study)
        if isinstance(level_cols, str):
            level_iter = prob_df[[level_cols]].drop_duplicates()
        else:
            level_iter = prob_df[level_cols].drop_duplicates()
        for _, level_row in level_iter.iterrows():
            mask = np.ones(len(prob_df), dtype=bool)
            level_descr: dict[str, object] = {}
            for col, val in level_row.items():
                mask &= (prob_df[col].values == val)
                level_descr[col] = val
            sub = prob_df[mask].set_index("macrorep")["obj_postrep_mean"]
            diff = (sub - base).dropna()
            if diff.empty:
                continue
            mean, lo, hi = _bootstrap_ci(diff.to_numpy(), n_boot, alpha)
            # Hedges-g style effect size on the paired differences.
            sd = float(diff.std(ddof=1)) if diff.size > 1 else float("nan")
            effect = mean / sd if sd and not np.isnan(sd) and sd > 0 else float("nan")
            out.append({
                "problem": problem,
                **level_descr,
                "n_macroreps_paired": int(diff.size),
                "mean_diff": mean,
                "ci95_lo": lo,
                "ci95_hi": hi,
                "paired_sd": sd,
                "effect_size_d": effect,
            })
    return pd.DataFrame(out)


def write_table(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        path.write_text("")
        return
    if path.suffix == ".parquet":
        engine = _parquet_engine()
        if engine is None:
            alt = path.with_suffix(".csv.gz")
            print(
                f"WARN: no parquet engine; writing {alt.name} instead of "
                f"{path.name}",
                file=sys.stderr,
            )
            with gzip.open(alt, "wt", newline="") as fh:
                df.to_csv(fh, index=False)
            return
        df.to_parquet(path, engine=engine, index=False)
    else:
        df.to_csv(path, index=False)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Aggregate per-task long-form outputs into journal-grade summary "
            "tables (paired bootstrap CIs)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--runs-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--n-boot", type=int, default=10_000)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = collect_long_form(args.runs_root)
    if df.empty:
        print("No long-form data found under", args.runs_root, file=sys.stderr)
        return 1

    write_table(df, args.output_dir / "journal_long_form.parquet")

    finals = compute_final_objectives(df)
    write_table(finals, args.output_dir / "final_objectives.parquet")

    for study, sub in finals.groupby("study", observed=False):
        write_table(
            summary_table(sub, study),
            args.output_dir / f"summary_{study}.csv",
        )
        write_table(
            paired_ci_table(sub, study, n_boot=args.n_boot, alpha=args.alpha),
            args.output_dir / f"paired_ci_{study}.csv",
        )

    print(f"Aggregation complete; outputs in {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
