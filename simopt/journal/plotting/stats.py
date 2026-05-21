"""Statistical-summarisation utilities for the journal sensitivity studies.

All confidence intervals here are percentile bootstrap CIs, computed with a
fixed RNG seed so figure regeneration is deterministic.  This matches the
methodology already used by ``scripts/journal_aggregate.py``; the helpers
below are deliberately re-implemented so figures can be generated either
from the long-form table or from the canned summary CSVs.

Statistical choices
-------------------
* **Percentile bootstrap (rather than parametric t)** — final-objective
  distributions are right-skewed and heavy-tailed on three of the four
  registered problems.  The bootstrap is non-parametric and easy to defend
  in a journal context.
* **Paired comparison against the operating-point baseline** — the simopt
  framework keys MRG32k3a substreams on the macrorep index, so for a fixed
  ``mrep`` two design points are evaluated on bit-identical simulation
  noise.  The per-macrorep difference therefore has dramatically lower
  variance than the unpaired difference of means.  All inferential claims
  in our figures are paired.
* **Hedges-g style effect size** — mean(diff) / sd(diff).  Reported alongside
  the CI so figures can colour effect magnitude without losing significance.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .data import LEVEL_COLUMNS, STUDIES


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def bootstrap_mean_ci(
    sample: np.ndarray,
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0xC0FFEE,
) -> tuple[float, float, float]:
    """Percentile-bootstrap CI of the mean.

    Returns ``(mean, ci_lo, ci_hi)``.  Empty samples yield NaNs.  Constant
    samples yield zero-width CIs at the constant value, which is the
    statistically correct behaviour (no resampling can introduce variance
    that the data do not contain).
    """
    sample = np.asarray(sample, dtype=float)
    if sample.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = sample.size
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = sample[idx].mean(axis=1)
    lo, hi = np.quantile(boot_means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(sample.mean()), float(lo), float(hi)


# ---------------------------------------------------------------------------
# Marginal summary table (study-aware)
# ---------------------------------------------------------------------------
def _level_columns(study: str) -> list[str]:
    if study not in STUDIES:
        raise ValueError(f"unknown study {study!r}; valid: {STUDIES!r}")
    return list(LEVEL_COLUMNS[study])


def final_objective_summary(
    finals: pd.DataFrame,
    *,
    study: str,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0xC0FFEE,
) -> pd.DataFrame:
    """Per-(problem, level) marginal mean + bootstrap 95% CI + std.

    Notes:
    -----
    Marginal CIs here are *unpaired* and do not exploit CRN — they describe
    the absolute final-objective distribution at each level, not its
    significance against a baseline.  Use :func:`paired_ci_against_baseline`
    for inferential claims.
    """
    if finals.empty:
        return finals
    if study not in STUDIES:
        raise ValueError(f"unknown study {study!r}; valid: {STUDIES!r}")
    level_cols = _level_columns(study)
    out_rows: list[dict[str, Any]] = []
    for keys, g in finals.groupby(["problem", *level_cols], observed=False):
        problem = keys[0]
        level_vals = keys[1:]
        mean, lo, hi = bootstrap_mean_ci(
            g["obj_postrep_mean"].to_numpy(),
            n_boot=n_boot, alpha=alpha, seed=seed,
        )
        std = float(g["obj_postrep_mean"].std(ddof=1)) \
            if len(g) > 1 else float("nan")
        row: dict[str, Any] = {"problem": problem}
        for col, val in zip(level_cols, level_vals, strict=False):
            row[col] = val
        row.update({
            "n_macroreps": len(g),
            "mean_final_obj": mean,
            "ci95_lo": lo,
            "ci95_hi": hi,
            "std_final_obj": std,
        })
        out_rows.append(row)
    return pd.DataFrame(out_rows)


# ---------------------------------------------------------------------------
# Paired bootstrap against the operating-point baseline
# ---------------------------------------------------------------------------
def _baseline_mask(df: pd.DataFrame, study: str,
                   baseline: Any) -> pd.Series:  # noqa: ANN401 - heterogeneous baseline (scalar or tuple)
    """Boolean mask selecting rows that *are* the baseline level."""
    level_cols = _level_columns(study)
    if len(level_cols) == 1:
        return df[level_cols[0]] == baseline
    # 2-tuple baseline (basis study).
    if not (isinstance(baseline, tuple) and len(baseline) == 2):
        raise ValueError(
            f"baseline for basis study must be a 2-tuple "
            f"(polynomial_basis, polynomial_degree); got {baseline!r}"
        )
    return (
        (df[level_cols[0]] == baseline[0])
        & (df[level_cols[1]] == baseline[1])
    )


def paired_ci_against_baseline(
    finals: pd.DataFrame,
    *,
    study: str,
    baselines: Mapping[str, Any],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0xC0FFEE,
) -> pd.DataFrame:
    """Paired bootstrap CI of ``level - baseline`` per (problem, level).

    Per-macrorep pairing relies on the simopt CRN-by-macrorep property; rows
    are joined on ``macrorep`` only.

    A problem whose ``baselines[problem]`` value is absent from the data is
    silently dropped (with no row in the output); callers that need a hard
    failure can compare ``out["problem"].unique()`` to the input set.
    """
    if finals.empty:
        return finals
    level_cols = _level_columns(study)
    out_rows: list[dict[str, Any]] = []
    for problem, pdf in finals.groupby("problem", observed=False):
        if problem not in baselines:
            continue
        base = baselines[problem]
        base_mask = _baseline_mask(pdf, study, base)
        if not base_mask.any():
            continue
        base_ser = (
            pdf[base_mask]
            .drop_duplicates("macrorep")
            .set_index("macrorep")["obj_postrep_mean"]
        )
        levels = pdf[level_cols].drop_duplicates()
        for _, lvl in levels.iterrows():
            mask = np.ones(len(pdf), dtype=bool)
            level_descr: dict[str, Any] = {}
            for col in level_cols:
                v = lvl[col]
                mask &= (pdf[col].values == v)
                level_descr[col] = v
            sub = pdf[mask].drop_duplicates("macrorep")
            cand = sub.set_index("macrorep")["obj_postrep_mean"]
            diff = (cand - base_ser).dropna()
            if diff.empty:
                continue
            mean, lo, hi = bootstrap_mean_ci(
                diff.to_numpy(),
                n_boot=n_boot, alpha=alpha, seed=seed,
            )
            sd = float(diff.std(ddof=1)) if diff.size > 1 else float("nan")
            effect = (
                mean / sd if sd and not np.isnan(sd) and sd > 0
                else float("nan")
            )
            out_rows.append({
                "problem": problem,
                **level_descr,
                "n_macroreps_paired": int(diff.size),
                "mean_diff": mean,
                "ci95_lo": lo,
                "ci95_hi": hi,
                "paired_sd": sd,
                "effect_size_d": effect,
            })
    return pd.DataFrame(out_rows)


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------
def rank_levels_by_effect(
    paired: pd.DataFrame,
    *,
    level_cols: Sequence[str],
    per_problem: bool = True,
) -> pd.DataFrame:
    """Order levels by ``|mean_diff|`` descending and assign a 1-based rank.

    When ``per_problem`` is True (default) the ranking restarts within each
    problem.  Otherwise the ranking is global across all problems and may be
    used in cross-problem aggregate plots.
    """
    if paired.empty:
        return paired.assign(rank=pd.Series([], dtype=int))
    df = paired.copy()
    df["__abs_diff"] = df["mean_diff"].abs()
    if per_problem and "problem" in df.columns:
        df = (
            df.sort_values(
                by=["problem", "__abs_diff"],
                ascending=[True, False],
            )
            .reset_index(drop=True)
        )
        df["rank"] = (
            df.groupby("problem", observed=False).cumcount() + 1
        )
    else:
        df = df.sort_values("__abs_diff", ascending=False).reset_index(
            drop=True
        )
        df["rank"] = np.arange(1, len(df) + 1, dtype=int)
    keep = ["problem", *level_cols, "mean_diff", "rank"]
    keep = [c for c in keep if c in df.columns]
    return df[keep]


__all__ = [
    "bootstrap_mean_ci",
    "final_objective_summary",
    "paired_ci_against_baseline",
    "rank_levels_by_effect",
]
