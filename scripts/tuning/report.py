"""Generate the human-readable recommendation markdown.

Reads:
  - ``best/best_<problem>.json``           (winner per problem)
  - ``confirmations/confirm_<problem>.json`` (top-K confirmation table)
  - ``exports/<problem>/top20.json``       (top-20 with diagnostics)
  - ``exports/<problem>/study_meta.json``  (study counts)
  - the legacy ``PROBLEM_OPTIMAL_HYPER`` / ``cabs_factors`` blocks from
    ``scripts/journal_factors_test.py`` (read at runtime via importlib)

Writes:
  - ``recommendations.md``                 (the deliverable)
  - ``recommendation_table.csv``           (machine-readable summary)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tuning import results_root  # noqa: E402

log = logging.getLogger("astromorf.tuning.report")


# ── legacy reference loader ────────────────────────────────────────────────


def _load_legacy_reference() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Static-parse PROBLEM_OPTIMAL_HYPER + ADAPTIVE_CONFIGS without importing.

    journal_factors_test pulls in heavy modules at import time and is
    fragile to environment differences; we just want two top-level
    dicts. ``ast.literal_eval`` would refuse the ``PolyBasisType.X``
    references, so we walk the AST and substitute those references with
    plain strings.
    """
    import ast

    path = REPO_ROOT / "scripts" / "journal_factors_test.py"
    if not path.exists():
        return {}, {}
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except SyntaxError as exc:
        log.warning("Could not parse journal_factors_test: %s", exc)
        return {}, {}

    def _eval_node(node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -_eval_node(node.operand)
        if isinstance(node, ast.Tuple):
            return tuple(_eval_node(e) for e in node.elts)
        if isinstance(node, ast.List):
            return [_eval_node(e) for e in node.elts]
        if isinstance(node, ast.Dict):
            return {
                _eval_node(k): _eval_node(v)
                for k, v in zip(node.keys, node.values)
            }
        if isinstance(node, ast.Attribute):
            # PolyBasisType.CHEBYSHEV -> "CHEBYSHEV"
            return node.attr
        if isinstance(node, ast.Name):
            return node.id
        raise ValueError(f"unsupported node {type(node).__name__}")

    optimal: dict[str, dict[str, Any]] = {}
    adaptive: dict[str, dict[str, Any]] = {}
    for stmt in tree.body:
        # The legacy file declares these as AnnAssign (e.g. ``X: dict[...] = {...}``)
        # so we accept both forms.
        target_name: str | None = None
        value_node: ast.AST | None = None
        if isinstance(stmt, ast.Assign) and stmt.targets and isinstance(
            stmt.targets[0], ast.Name
        ):
            target_name = stmt.targets[0].id
            value_node = stmt.value
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            target_name = stmt.target.id
            value_node = stmt.value
        if target_name is None or value_node is None:
            continue
        target = target_name
        try:
            value = _eval_node(value_node)
        except Exception:
            continue
        if target == "PROBLEM_OPTIMAL_HYPER" and isinstance(value, dict):
            optimal = {str(k): dict(v) for k, v in value.items() if isinstance(v, dict)}
        elif target == "ADAPTIVE_CONFIGS" and isinstance(value, dict):
            for k, v in value.items():
                if not isinstance(v, dict):
                    continue
                # k is a tuple (problem, dim_or_None); we collapse on problem.
                if isinstance(k, tuple) and k:
                    name = str(k[0])
                    adaptive.setdefault(name, dict(v))
    return optimal, adaptive


# ── per-problem report assembly ────────────────────────────────────────────


def _safe_load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        log.warning("Could not parse %s: %s", path, exc)
        return None


def _format_factor(name: str, val: Any) -> str:
    if isinstance(val, float):
        return f"{val:.4g}"
    return str(val)


def _aligned_improvement(new_aligned: float, old_aligned: float | None) -> str:
    """Improvement of *new* over *old* in aligned units (lower is better).

    Because the aligned score is the same sign-convention regardless of
    raw direction (maximise vs minimise), the formula is identical for
    both: ``(old - new) / |old|`` with positive = better.
    """
    if (
        old_aligned is None
        or not np.isfinite(old_aligned)
        or abs(old_aligned) < 1e-12
    ):
        return "n/a"
    improvement = (old_aligned - new_aligned) / abs(old_aligned)
    return f"{improvement * 100:+.2f}%"


def _resolve_direction(problem_name: str) -> tuple[str, int]:
    """Return (label, sign) where label is 'maximise' or 'minimise'.

    Tries the Optuna study user_attrs first (single source of truth set
    in ``tuner.build_study``), then falls back to probing the problem.
    """
    # 1. study user_attrs
    try:
        import optuna  # local import to avoid a hard dep at report time
        from scripts.tuning.tuner import storage_url_for, study_name

        study = optuna.load_study(
            study_name=study_name(problem_name),
            storage=storage_url_for(problem_name),
        )
        direction = study.user_attrs.get("raw_direction")
        sign = study.user_attrs.get("raw_minmax_sign")
        if direction and sign is not None:
            return (
                "maximise" if direction == "maximize" else "minimise",
                int(sign),
            )
    except Exception:
        pass

    # 2. fall back to probing the problem
    try:
        from simopt.experiment.dimension_scaling import scale_dimension
        from simopt.experiment_base import instantiate_problem

        if problem_name in {"DYNAMNEWS-1", "SAN-1", "NETWORK-1", "ROSENBROCK-1"}:
            problem = scale_dimension(problem_name, budget=10000, dimension=100)
        else:
            problem = instantiate_problem(
                problem_name=problem_name, problem_fixed_factors={"budget": 10000}
            )
        sign = int(problem.minmax[0])
        return ("maximise" if sign > 0 else "minimise", sign)
    except Exception as exc:
        log.warning("Could not resolve minmax for %s: %s", problem_name, exc)
        return ("minimise", -1)


def _warmstart_trial_aligned(problem_name: str) -> float | None:
    """Aligned value of the warm-start trial, or None if not available.

    The warm-start trial is the one Optuna evaluated against the legacy
    ``PROBLEM_OPTIMAL_HYPER`` params (enqueued first when the study was
    created). We find it by matching params; fall back to trial 0 if no
    explicit warmstart_params attr is set.
    """
    try:
        import optuna
        from scripts.tuning.tuner import storage_url_for, study_name

        study = optuna.load_study(
            study_name=study_name(problem_name),
            storage=storage_url_for(problem_name),
        )
        ws_params = study.user_attrs.get("warmstart_params")
        trials = [
            t
            for t in study.get_trials(
                deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)
            )
            if t.value is not None and np.isfinite(t.value)
        ]
        if not trials:
            return None
        if ws_params:
            for t in trials:
                if all(
                    str(t.params.get(k)) == str(v) for k, v in ws_params.items()
                ):
                    return float(t.value)
        # Fallback: assume trial 0 is the warm-start (it's enqueued first).
        zero = [t for t in trials if t.number == 0]
        return float(zero[0].value) if zero else None
    except Exception as exc:
        log.warning("Could not locate warm-start trial for %s: %s", problem_name, exc)
        return None


def _per_problem_section(
    problem_name: str,
    legacy_optimal: dict[str, Any],
    legacy_cabs: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Build the markdown section + CSV row for one problem."""
    root = results_root()
    best = _safe_load_json(root / "best" / f"best_{problem_name}.json")
    confirm = _safe_load_json(root / "confirmations" / f"confirm_{problem_name}.json")
    top20 = _safe_load_json(root / "exports" / problem_name / "top20.json") or []
    meta = _safe_load_json(root / "exports" / problem_name / "study_meta.json") or {}

    direction, minmax_sign = _resolve_direction(problem_name)
    raw_better = "higher" if minmax_sign > 0 else "lower"

    md: list[str] = []
    md.append(f"## {problem_name}\n")
    md.append(
        f"- Direction: **{direction}** (`minmax[0] = {minmax_sign:+d}`; "
        f"better raw objective = {raw_better})"
    )
    md.append(
        f"- Optuna internally minimises the **aligned** score "
        f"`= -minmax[0] × raw_obj`; for this problem the aligned score is "
        f"`{'-raw_obj' if minmax_sign > 0 else '+raw_obj'}`."
    )
    md.append(f"- Trials total: {meta.get('n_trials_total', 'n/a')}")
    md.append(f"- Trials complete: {meta.get('n_trials_complete', 'n/a')}")
    md.append(
        f"- Trials rejected (combinatorial guard): "
        f"{meta.get('n_trials_pruned_combinatorial', 'n/a')}"
    )
    md.append("")

    if best is None:
        md.append(
            "_No `best_<problem>.json` found — confirmation pass not run yet._\n"
        )
        return "\n".join(md), {"problem": problem_name, "status": "no_best"}

    factors = best.get("factors", {})
    md.append("### Recommended factors\n")
    md.append("```json")
    md.append(json.dumps(factors, indent=2, default=str))
    md.append("```\n")

    ci = best.get("confirm_ci95_aligned", [None, None])
    md.append("### Confirmation statistics (disjoint-seed)\n")
    md.append(
        f"- Confirm n_macroreps: {best.get('n_macroreps_confirm', 'n/a')}\n"
        f"- **Confirm mean_objective (raw, {raw_better}-better):** "
        f"{best.get('confirm_mean_objective', float('nan')):.6g}\n"
        f"- Confirm std_objective (raw): {best.get('confirm_std_objective', float('nan')):.4g}\n"
        f"- Confirm mean (aligned, lower-better): "
        f"{best.get('confirm_mean_aligned', float('nan')):.6g}\n"
        f"- Confirm std (aligned): {best.get('confirm_std_aligned', float('nan')):.4g}\n"
        f"- Confirm 95% CI (aligned): [{ci[0]}, {ci[1]}]\n"
    )

    diag = best.get("confirm_diagnostics", {}) or {}
    if diag:
        md.append("### Diagnostic statistics (means across confirmation macroreps)\n")
        for key in [
            "iteration_count_mean",
            "accepted_steps_mean",
            "rejected_steps_mean",
            "accept_ratio_mean",
            "cabs_increases_mean",
            "cabs_decreases_mean",
            "avg_subspace_dim_mean",
            "final_subspace_dim_mean",
            "avg_interp_set_size_approx_mean",
            "pattern_search_overrides_mean",
            "pattern_search_override_ratio_mean",
            "mean_prediction_rel_error_mean",
            "wall_clock_s_mean",
        ]:
            v = diag.get(key)
            if v is not None:
                md.append(f"- {key}: {v:.4g}")
        md.append("")

    # Comparison vs legacy warm-start.
    # Both sides are in **aligned** units (Optuna's direction-neutral
    # "lower-better" frame) so the comparison is direction-agnostic.
    warmstart_aligned = _warmstart_trial_aligned(problem_name)
    new_aligned = best.get("study_value")  # winning trial's aligned value
    md.append("### Comparison vs `PROBLEM_OPTIMAL_HYPER` (legacy warm-start)\n")
    if legacy_optimal:
        md.append("```")
        md.append("legacy non-CABS factors:")
        md.append(json.dumps(legacy_optimal, indent=2, default=str))
        md.append("legacy CABS factors:")
        md.append(json.dumps(legacy_cabs, indent=2, default=str))
        md.append("```\n")
    if warmstart_aligned is None:
        md.append(
            "- Warm-start trial not found in study — cannot compute relative "
            "improvement. (Did the warm-start trial fail to complete?)\n"
        )
    else:
        # Convert aligned back to raw so the user sees natural units too.
        warmstart_raw = -minmax_sign * warmstart_aligned
        new_raw_proxy = -minmax_sign * (new_aligned if new_aligned is not None else float("nan"))
        md.append(
            f"- Warm-start trial aligned value: {warmstart_aligned:.6g} "
            f"(raw ≈ {warmstart_raw:.6g}, {raw_better}-better)\n"
            f"- Recommended trial aligned value: "
            f"{(new_aligned if new_aligned is not None else float('nan')):.6g} "
            f"(raw ≈ {new_raw_proxy:.6g})\n"
            f"- **Relative improvement (aligned, direction-agnostic): "
            f"{_aligned_improvement(new_aligned, warmstart_aligned)}**\n"
        )

    if confirm and confirm.get("candidates_ranked_by_confirm_score"):
        md.append("### Top-K confirmation table (lower = better)\n")
        md.append(
            "| rank | trial | study value | confirm mean (aligned) | "
            "confirm std (aligned) | failures |"
        )
        md.append(
            "|---:|---:|---:|---:|---:|---:|"
        )
        for c in confirm["candidates_ranked_by_confirm_score"]:
            md.append(
                f"| {c.get('rank_in_study')} | {c.get('trial_number')} | "
                f"{c.get('study_value', float('nan')):.6g} | "
                f"{c.get('confirm_mean_aligned', float('nan')):.6g} | "
                f"{c.get('confirm_std_aligned', float('nan')):.4g} | "
                f"{c.get('confirm_failure_count', 'n/a')} |"
            )
        md.append("")

    if top20:
        md.append(f"### Top-{len(top20)} from study (with 95% CI on aligned mean)\n")
        md.append("| rank | trial | study value | mean aligned | std aligned | CI low | CI high |")
        md.append("|---:|---:|---:|---:|---:|---:|---:|")
        for r in top20:
            ci = r.get("ci95_aligned", [float("nan"), float("nan")])
            md.append(
                f"| {r['rank']} | {r['trial_number']} | "
                f"{r['study_value']:.6g} | "
                f"{r.get('mean_aligned', float('nan')):.6g} | "
                f"{r.get('std_aligned', float('nan')):.4g} | "
                f"{ci[0]} | {ci[1]} |"
            )
        md.append("")

    md.append("### Tuning observations\n")
    md.append(
        f"- Search dimension: {len(top20[0]['params']) if top20 else 'n/a'}\n"
        f"- Combinatorial-guard rejections: "
        f"{meta.get('n_trials_pruned_combinatorial', 'n/a')}\n"
        f"- Average wall-clock per macrorep at final rung: "
        f"{diag.get('wall_clock_s_mean', float('nan'))}\n"
    )

    csv_row = {
        "problem": problem_name,
        "trial_number": best.get("trial_number"),
        "confirm_mean_objective": best.get("confirm_mean_objective"),
        "confirm_std_objective": best.get("confirm_std_objective"),
        "confirm_mean_aligned": best.get("confirm_mean_aligned"),
        "confirm_std_aligned": best.get("confirm_std_aligned"),
        "study_value": best.get("study_value"),
        "n_trials_total": meta.get("n_trials_total"),
        "n_trials_complete": meta.get("n_trials_complete"),
    }
    for k, v in (factors or {}).items():
        csv_row[f"factor.{k}"] = v
    return "\n".join(md), csv_row


def write_report(problems: list[str]) -> tuple[Path, Path]:
    """Write recommendations.md and recommendation_table.csv."""
    legacy_opt_all, legacy_cabs_all = _load_legacy_reference()

    md_parts: list[str] = ["# ASTROMoRF tuning recommendations\n"]
    md_parts.append(
        "Search budget: every config evaluated at simulation budget = 10 000.\n"
        "Selection: TPE + ASHA pruning (4→8→16 mreps), then top-5 "
        "confirmation pass on a disjoint seed stream (30 mreps each).\n"
    )

    csv_rows: list[dict[str, Any]] = []
    for name in problems:
        section_md, csv_row = _per_problem_section(
            name,
            legacy_optimal=legacy_opt_all.get(name, {}) or {},
            legacy_cabs=legacy_cabs_all.get(name, {}) or {},
        )
        md_parts.append(section_md)
        md_parts.append("\n---\n")
        csv_rows.append(csv_row)

    out_dir = results_root()
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "recommendations.md"
    md_path.write_text("\n".join(md_parts))

    csv_path = out_dir / "recommendation_table.csv"
    if csv_rows:
        keys: list[str] = []
        seen: set[str] = set()
        for r in csv_rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in csv_rows:
                w.writerow(r)

    log.info("Wrote %s and %s", md_path, csv_path)
    return md_path, csv_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate the recommendation report.")
    p.add_argument(
        "--problems",
        type=str,
        default="DYNAMNEWS-1,SAN-1,NETWORK-1,ROSENBROCK-1,PARAMESTI-1",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    problems = [s.strip() for s in args.problems.split(",") if s.strip()]
    write_report(problems)


if __name__ == "__main__":
    main()
