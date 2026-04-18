"""Bootstrap & permutation tests for multi-macrorep experiment results.

Usage examples:
  python analysis/bootstrap_tests.py --csv multi_mrep_plateau_results.csv --problem
  ROSENBROCK-1 --window-a 2 --window-b 3 --paired --n-boot 5000 --n-perm 5000

The script expects a CSV where each row is a single macroreplication result (one value
per run).
Required columns: 'problem', a grouping column (default 'window'), and a value column
(default 'final_fn').
If the CSV contains only aggregated rows (one row per group), the script will abort with
a clear message.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd


def bootstrap_mean_diff_ci(  # noqa: D103
    x: np.ndarray,
    y: np.ndarray,
    paired: bool = False,
    n_boot: int = 10000,
    ci: float = 0.95,
    random_state: int | None = None,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(random_state)
    n = len(x)
    m = len(y)
    if paired and n != m:
        raise ValueError("Paired bootstrap requires equal-length samples")

    boot_diffs = np.empty(n_boot)
    if paired:
        x = np.asarray(x)
        y = np.asarray(y)
        for i in range(n_boot):
            idx = rng.integers(0, n, n)
            boot_diffs[i] = np.mean(x[idx] - y[idx])
    else:
        x = np.asarray(x)
        y = np.asarray(y)
        for i in range(n_boot):
            sx = rng.choice(x, size=n, replace=True)
            sy = rng.choice(y, size=m, replace=True)
            boot_diffs[i] = np.mean(sx) - np.mean(sy)

    lower = np.quantile(boot_diffs, (1.0 - ci) / 2)
    upper = np.quantile(boot_diffs, 1.0 - (1.0 - ci) / 2)
    est = np.mean(x) - np.mean(y) if not paired else np.mean(x - y)
    return float(est), float(lower), float(upper)


def permutation_test_mean_diff(  # noqa: D103
    x: np.ndarray,
    y: np.ndarray,
    paired: bool = False,
    n_perm: int = 10000,
    alternative: str = "two-sided",
    random_state: int | None = None,
) -> float:
    rng = np.random.default_rng(random_state)
    x = np.asarray(x)
    y = np.asarray(y)
    if paired and len(x) != len(y):
        raise ValueError("Paired permutation requires equal-length samples")

    if paired:
        diffs = x - y
        obs = np.mean(diffs)
        count = 0
        for _ in range(n_perm):
            signs = rng.choice([1, -1], size=len(diffs))
            stat = np.mean(signs * diffs)
            if alternative == "two-sided":
                if abs(stat) >= abs(obs):
                    count += 1
            elif alternative == "greater":
                if stat >= obs:
                    count += 1
            else:  # less
                if stat <= obs:
                    count += 1
        pval = (count + 1) / (n_perm + 1)
        return float(pval)
    obs = np.mean(x) - np.mean(y)
    pooled = np.concatenate([x, y])
    n = len(x)
    count = 0
    for _ in range(n_perm):
        idx = rng.permutation(len(pooled))
        a = pooled[idx[:n]]
        b = pooled[idx[n:]]
        stat = np.mean(a) - np.mean(b)
        if alternative == "two-sided":
            if abs(stat) >= abs(obs):
                count += 1
        elif alternative == "greater":
            if stat >= obs:
                count += 1
        else:
            if stat <= obs:
                count += 1
    pval = (count + 1) / (n_perm + 1)
    return float(pval)


def load_group_values(  # noqa: D103
    df: pd.DataFrame,
    problem: str,
    group_col: str,
    group_value,  # noqa: ANN001
    value_col: str,
) -> np.ndarray:
    sub = df[df["problem"] == problem]
    sub = sub[sub[group_col] == group_value]
    if sub.empty:
        raise ValueError(
            f"No rows found for problem={problem} and {group_col}={group_value}"
        )
    # Expect each row to be one macrorep; if only a single row present, abort (likely aggregated)  # noqa: E501
    if len(sub) == 1:
        raise ValueError(
            f"Only one row for problem={problem}, {group_col}={group_value}. The CSV appears aggregated; need per-macrorep rows."  # noqa: E501
        )
    return sub[value_col].to_numpy(dtype=float)


def main(argv=None) -> None:  # noqa: ANN001, D103
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="CSV file with per-macrorep results")
    p.add_argument("--problem", required=True)
    p.add_argument("--window-a", required=True)
    p.add_argument("--window-b", required=True)
    p.add_argument(
        "--group-col",
        default="window",
        help="Column name for group values (default: window)",
    )
    p.add_argument(
        "--value-col",
        default="final_fn",
        help="Column name for the scalar outcome (default: final_fn)",
    )
    p.add_argument(
        "--paired", action="store_true", help="Treat samples as paired (CRN)"
    )
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--n-perm", type=int, default=5000)
    p.add_argument("--ci", type=float, default=0.95)
    p.add_argument("--random-state", type=int, default=None)
    args = p.parse_args(argv)

    df = pd.read_csv(args.csv)
    if "problem" not in df.columns:
        raise ValueError("CSV must contain a 'problem' column")
    if args.group_col not in df.columns:
        raise ValueError(f"CSV missing group column: {args.group_col}")
    if args.value_col not in df.columns:
        raise ValueError(f"CSV missing value column: {args.value_col}")

    try:
        a_vals = load_group_values(
            df,
            args.problem,
            args.group_col,
            _coerce_type(df[args.group_col], args.window_a),
            args.value_col,
        )
        b_vals = load_group_values(
            df,
            args.problem,
            args.group_col,
            _coerce_type(df[args.group_col], args.window_b),
            args.value_col,
        )
    except ValueError as e:
        print("Error:", e)
        sys.exit(2)

    est, lo, hi = bootstrap_mean_diff_ci(
        a_vals,
        b_vals,
        paired=args.paired,
        n_boot=args.n_boot,
        ci=args.ci,
        random_state=args.random_state,
    )
    pval = permutation_test_mean_diff(
        a_vals,
        b_vals,
        paired=args.paired,
        n_perm=args.n_perm,
        random_state=args.random_state,
    )

    print(f"Problem: {args.problem}")
    print(
        f"Group A ({args.group_col}={args.window_a}): n={len(a_vals)} mean={np.mean(a_vals):.6g}"  # noqa: E501
    )
    print(
        f"Group B ({args.group_col}={args.window_b}): n={len(b_vals)} mean={np.mean(b_vals):.6g}"  # noqa: E501
    )
    print(
        f"Mean difference (A - B): est={est:.6g}, {int(100 * args.ci)}% CI = [{lo:.6g}, {hi:.6g}]"  # noqa: E501
    )
    print(f"Permutation p-value (two-sided): p={pval:.6g}")


def _coerce_type(series: pd.Series, value: str):  # noqa: ANN202
    # Coerce the input window value to the same dtype used in the CSV column
    dtype = series.dtype
    if pd.api.types.is_integer_dtype(dtype):
        return int(value)
    if pd.api.types.is_float_dtype(dtype):
        return float(value)
    return value


if __name__ == "__main__":
    main()
