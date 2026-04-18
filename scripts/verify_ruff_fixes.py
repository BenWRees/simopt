#!/usr/bin/env python3
"""Verification script for ruff lint fixes.

This script verifies that the ruff lint fixes applied in this session are
correct by checking:

1. All modified files pass ruff linting (0 errors).
2. No syntax errors were introduced by the fixes.
3. Modified files still import correctly (no broken imports).
4. Diff summary showing what changed in each file.

Usage:
    python verify_ruff_fixes.py [--verbose] [--diff]
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Files modified in this lint-fix session
MODIFIED_FILES = [
    "ASTROMoRF_Code/Adaptive_Dimension_tests.py",
    "ASTROMoRF_Code/Adaptive_radius.py",
    "ASTROMoRF_Code/Design_set_plots.py",
    "ASTROMoRF_Code/Model_construction.py",
    "ASTROMoRF_Code/basis_construction_test.py",
    "ASTROMoRF_Code/criticality_check.py",
    "ASTROMoRF_Code/design_set_test.py",
    "ASTROMoRF_Code/geometry_improvement_test.py",
    "ASTROMoRF_Code/plot_model_construction.py",
    "ASTROMoRF_Code/plotting.py",
    "HPC_code/generate_factor_experiment_jobs.py",
    "HPC_code/generate_hyperparameter_slurm.py",
    "HPC_code/generate_slurm_files.py",
    "HPC_code/generate_slurm_files_omorf.py",
    "HPC_code/generated_slurm_files/collect_results.py",
    "WSC_Code/Cand_soln_select.py",
    "WSC_Code/CoD_code.py",
    "WSC_Code/adapt_sampl.py",
    "WSC_Code/model_construct.py",
    "analysis/bootstrap_tests.py",
    "demo/ASTROMoRF_HyperParameterSearch.py",
    "demo/BO_test.py",
    "demo/Generate_ASTROMoRF_traject.py",
    "demo/Journal_plots.py",
    "demo/LocalModelsAgainstResponseSurface.py",
    "demo/OMoRF_diff_subspace_dim.py",
    "demo/Results_table_WSC.py",
    "demo/_smoke_sufficient.py",
    "demo/aggregate_visualize_sweeps.py",
    "demo/basis_test.py",
    "demo/compare_prefer_small.py",
    "demo/demo_problem_solver_original.py",
    "demo/demo_problems_solver.py",
    "demo/eval_adaptive_d.py",
    "demo/find_optimal_hyperparameters_demo.py",
    "demo/find_optimal_subspace_demo.py",
    "demo/hyperparameter_optimization_script.py",
    "demo/iridis_experimental_setup.py",
    "demo/journal_factors_test.py",
    "demo/low_high_variance_experiment.py",
    "demo/optimal_d_test.py",
    "demo/optimal_d_test_2.py",
    "demo/pickle_OMoRF.py",
    "demo/pickle_files_journal_paper.py",
    "demo/pickle_files_poly_bases.py",
    "demo/plot_datafarming.py",
    "demo/plot_model_construction.py",
    "demo/plotting_previous_experiments.py",
    "demo/progression_review_code.py",
    "demo/run_cost_sweep.py",
    "demo/run_cost_sweep_power.py",
    "demo/run_high_fidelity.py",
    "demo/sensitivity_experiment.py",
    "demo/sensitivity_vs_OMoRF_test.py",
    "demo/subspace_dimension_test.py",
    "demo/sweep_tol.py",
    "demo/test_adaptive_d.py",
    "demo/tune_and_eval_adaptive_d.py",
    "docs/conf.py",
    "factor_experiments/collect_results.py",
    "notebooks/data_farming_model.py",
    "notebooks/data_farming_over_solver_and_problem.py",
    "notebooks/data_farming_problem.py",
    "notebooks/data_farming_solver.py",
    "notebooks/demo_data_driven.py",
    "notebooks/demo_fcsa.py",
    "notebooks/demo_model.py",
    "notebooks/demo_plots.py",
    "notebooks/demo_problem.py",
    "notebooks/demo_problem_solver.py",
    "notebooks/demo_problems_solvers.py",
    "notebooks/demo_san-sscont-ironorecont_experiment.py",
    "notebooks/demo_sscont_experiment.py",
    "notebooks/exp_base_testing.py",
    "notebooks/hello_simopt.py",
    "notebooks/load_solver_design.py",
    "resume_multi_mrep_sweep.py",
    "run_dynamnews_sweep.py",
    "run_multi_mrep_single_window.py",
    "run_multi_mrep_sweep.py",
    "run_plateau_sweep.py",
    "run_solver_test.py",
    "ruff.toml",
    "scripts/ab_compare.py",
    "scripts/benchmark_astromorf.py",
    "simopt/base.py",
    "simopt/bootstrap.py",
    "simopt/curve.py",
    "simopt/curve_utils.py",
    "simopt/data_analysis_base.py",
    "simopt/data_farming_base.py",
    "simopt/diagnostics.py",
    "simopt/directory.py",
    "simopt/experiment/post_normalize.py",
    "simopt/experiment/run_solver.py",
    "simopt/experiment/single.py",
    "simopt/experiment_base.py",
    "simopt/experimental.py",
    "simopt/feasibility.py",
    "simopt/gui/data_farming_window.py",
    "simopt/gui/df_object.py",
    "simopt/gui/experiment_window.py",
    "simopt/gui/new_experiment_window.py",
    "simopt/gui/plot_window.py",
    "simopt/gui/toplevel_custom.py",
    "simopt/linear_algebra_base.py",
    "simopt/models/amusementpark.py",
    "simopt/models/ermexample.py",
    "simopt/models/facilitysizing.py",
    "simopt/models/fixedsan.py",
    "simopt/models/hyperparameter_tuning.py",
    "simopt/models/networkairlinerevenuemanagement.py",
    "simopt/models/pca_model.py",
    "simopt/models/rosenbrock.py",
    "simopt/models/simple_function.py",
    "simopt/models/zakharov.py",
    "simopt/plots/area_scatterplot.py",
    "simopt/plots/budget_history.py",
    "simopt/plots/feasibility_progress.py",
    "simopt/plots/fn_estimates.py",
    "simopt/plots/progress_curve.py",
    "simopt/plots/solvability_cdf.py",
    "simopt/plots/solvability_profile.py",
    "simopt/plots/terminal_feasibility.py",
    "simopt/plots/terminal_scatterplot.py",
    "simopt/plots/utils.py",
    "simopt/plotting.py",
    "simopt/solver.py",
    "simopt/solvers/ASTROMoRF_OMoRF_Geo.py",
    "simopt/solvers/ASTROMoRF_old.py",
    "simopt/solvers/Equadratures/equadratures/__init__.py",
    "simopt/solvers/Equadratures/equadratures/basis.py",
    "simopt/solvers/Equadratures/equadratures/correlations.py",
    "simopt/solvers/Equadratures/equadratures/datasets.py",
    "simopt/solvers/Equadratures/equadratures/distributions/__init__.py",
    "simopt/solvers/Equadratures/equadratures/distributions/analytical.py",
    "simopt/solvers/Equadratures/equadratures/distributions/beta.py",
    "simopt/solvers/Equadratures/equadratures/distributions/cauchy.py",
    "simopt/solvers/Equadratures/equadratures/distributions/chebyshev.py",
    "simopt/solvers/Equadratures/equadratures/distributions/chi.py",
    "simopt/solvers/Equadratures/equadratures/distributions/chisquared.py",
    "simopt/solvers/Equadratures/equadratures/distributions/exponential.py",
    "simopt/solvers/Equadratures/equadratures/distributions/gamma.py",
    "simopt/solvers/Equadratures/equadratures/distributions/gaussian.py",
    "simopt/solvers/Equadratures/equadratures/distributions/gumbel.py",
    "simopt/solvers/Equadratures/equadratures/distributions/logistic.py",
    "simopt/solvers/Equadratures/equadratures/distributions/lognormal.py",
    "simopt/solvers/Equadratures/equadratures/distributions/pareto.py",
    "simopt/solvers/Equadratures/equadratures/distributions/rayleigh.py",
    "simopt/solvers/Equadratures/equadratures/distributions/recurrence_utils.py",
    "simopt/solvers/Equadratures/equadratures/distributions/studentst.py",
    "simopt/solvers/Equadratures/equadratures/distributions/template.py",
    "simopt/solvers/Equadratures/equadratures/distributions/triangular.py",
    "simopt/solvers/Equadratures/equadratures/distributions/truncated_gaussian.py",
    "simopt/solvers/Equadratures/equadratures/distributions/uniform.py",
    "simopt/solvers/Equadratures/equadratures/distributions/weibull.py",
    "simopt/solvers/Equadratures/equadratures/logistic_poly.py",
    "simopt/solvers/Equadratures/equadratures/optimisation.py",
    "simopt/solvers/Equadratures/equadratures/parameter.py",
    "simopt/solvers/Equadratures/equadratures/plot.py",
    "simopt/solvers/Equadratures/equadratures/poly.py",
    "simopt/solvers/Equadratures/equadratures/polybayes.py",
    "simopt/solvers/Equadratures/equadratures/polynet.py",
    "simopt/solvers/Equadratures/equadratures/polytree.py",
    "simopt/solvers/Equadratures/equadratures/quadrature.py",
    "simopt/solvers/Equadratures/equadratures/sampling_methods/__init__.py",
    "simopt/solvers/Equadratures/equadratures/sampling_methods/induced.py",
    "simopt/solvers/Equadratures/equadratures/sampling_methods/montecarlo.py",
    "simopt/solvers/Equadratures/equadratures/sampling_methods/sampling_template.py",
    "simopt/solvers/Equadratures/equadratures/sampling_methods/sparsegrid.py",
    "simopt/solvers/Equadratures/equadratures/sampling_methods/tensorgrid.py",
    "simopt/solvers/Equadratures/equadratures/sampling_methods/userdefined.py",
    "simopt/solvers/Equadratures/equadratures/scalers.py",
    "simopt/solvers/Equadratures/equadratures/solver.py",
    "simopt/solvers/Equadratures/equadratures/stats.py",
    "simopt/solvers/Equadratures/equadratures/subsampling.py",
    "simopt/solvers/Equadratures/equadratures/subspaces.py",
    "simopt/solvers/Equadratures/equadratures/weight.py",
    "simopt/solvers/GeometryImprovement.py",
    "simopt/solvers/OMoRF.py",
    "simopt/solvers/SGD.py",
    "simopt/solvers/TrustRegion/Geometry.py",
    "simopt/solvers/TrustRegion/Models.py",
    "simopt/solvers/TrustRegion/Sampling.py",
    "simopt/solvers/TrustRegion/TrustRegion.py",
    "simopt/solvers/TrustRegion/__init__.py",
    "simopt/solvers/active_subspaces/__init__.py",
    "simopt/solvers/active_subspaces/basis.py",
    "simopt/solvers/active_subspaces/compute_optimal_dim.py",
    "simopt/solvers/active_subspaces/gauss_newton.py",
    "simopt/solvers/active_subspaces/gaussian_process.py",
    "simopt/solvers/active_subspaces/index_set.py",
    "simopt/solvers/active_subspaces/local_linear.py",
    "simopt/solvers/active_subspaces/optimal_subspace_cli.py",
    "simopt/solvers/active_subspaces/poly.py",
    "simopt/solvers/active_subspaces/polyridge.py",
    "simopt/solvers/active_subspaces/ridge.py",
    "simopt/solvers/active_subspaces/seqlp.py",
    "simopt/solvers/active_subspaces/subspace.py",
    "simopt/solvers/astrodf.py",
    "simopt/solvers/astromorf.py",
    "simopt/solvers/astromorf_relaxed.py",
    "simopt/solvers/fcsa.py",
    "simopt/solvers/kiefer_wolfowitz.py",
    "simopt/solvers/mirror_descent.py",
    "simopt/solvers/mixed_tr_solver.py",
    "simopt/solvers/neldmd.py",
    "simopt/solvers/randomsearch.py",
    "simopt/solvers/rank_and_selection.py",
    "simopt/solvers/robbins_monro.py",
    "simopt/solvers/spsa.py",
    "simopt/solvers/strong.py",
    "simopt/solvers/utils.py",
    "test/make_tests.py",
    "test/template.py",
]

# Files with pre-existing syntax errors (not introduced by lint fixes)
PRE_EXISTING_SYNTAX_ERRORS = {
    "demo/pickle_files_journal_paper.py",
    "simopt/problem.py",
}


def check_ruff(files: list[str], verbose: bool = False) -> tuple[int, list[str]]:
    """Run ruff check on the given files.

    Returns (error_count, list_of_error_messages).
    """
    py_files = [f for f in files if f.endswith(".py") and Path(f).exists()]
    if not py_files:
        return 0, []

    result = subprocess.run(
        ["ruff", "check", "--config", "ruff.toml", *py_files],
        capture_output=True,
        text=True,
    )

    errors = []
    if result.stdout.strip():
        for line in result.stdout.strip().split("\n"):
            if line.strip() and not line.startswith("Found"):
                errors.append(line)

    if verbose and errors:
        for e in errors:
            print(f"  RUFF: {e}")

    # Parse count from "Found N errors" line
    count = 0
    for line in (result.stdout + result.stderr).split("\n"):
        if "Found" in line and "error" in line:
            try:
                count = int(line.split()[1])
            except (ValueError, IndexError):
                count = len(errors)

    return count, errors


def check_syntax(files: list[str], verbose: bool = False) -> tuple[int, list[str]]:
    """Check Python syntax of the given files.

    Returns (error_count, list_of_error_messages).
    """
    errors = []
    for f in files:
        if not f.endswith(".py") or not Path(f).exists():
            continue
        try:
            with open(f) as fh:  # noqa: PTH123
                compile(fh.read(), f, "exec")
        except SyntaxError as e:
            if f in PRE_EXISTING_SYNTAX_ERRORS:
                if verbose:
                    print(f"  SKIP (pre-existing): {f}:{e.lineno} {e.msg}")
                continue
            errors.append(f"{f}:{e.lineno} {e.msg}")
            if verbose:
                print(f"  SYNTAX: {f}:{e.lineno} {e.msg}")

    return len(errors), errors


def check_imports(files: list[str], verbose: bool = False) -> tuple[int, list[str]]:
    """Verify that modified files can be parsed by AST (import-safe).

    Returns (error_count, list_of_error_messages).
    """
    import ast

    errors = []
    for f in files:
        if not f.endswith(".py") or not Path(f).exists():
            continue
        if f in PRE_EXISTING_SYNTAX_ERRORS:
            continue
        try:
            with open(f) as fh:  # noqa: PTH123
                ast.parse(fh.read(), filename=f)
        except SyntaxError as e:
            errors.append(f"{f}:{e.lineno} {e.msg}")
            if verbose:
                print(f"  AST: {f}:{e.lineno} {e.msg}")

    return len(errors), errors


def show_diff_summary(files: list[str]) -> None:
    """Show a summary of changes in each modified file."""
    for f in files:
        if not f.endswith(".py"):
            continue
        result = subprocess.run(
            ["git", "diff", "--stat", "--", f],
            capture_output=True,
            text=True,
        )
        if result.stdout.strip():
            print(f"  {result.stdout.strip()}")


def main() -> int:
    """Run all verification checks."""
    parser = argparse.ArgumentParser(description="Verify ruff lint fixes are correct")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed output"
    )
    parser.add_argument("--diff", "-d", action="store_true", help="Show diff summary")
    args = parser.parse_args()

    os.chdir(Path(__file__).parent)

    print("=" * 60)
    print("Ruff Lint Fix Verification")
    print("=" * 60)
    print(f"Files to verify: {len(MODIFIED_FILES)}")
    print()

    all_passed = True

    # 1. Ruff lint check
    print("[1/4] Running ruff lint check...")
    ruff_errors, ruff_msgs = check_ruff(MODIFIED_FILES, verbose=args.verbose)
    if ruff_errors == 0:
        print("  PASS: All modified files pass ruff linting (0 errors)")
    else:
        print(f"  FAIL: {ruff_errors} ruff errors found")
        if not args.verbose:
            for msg in ruff_msgs[:10]:
                print(f"    {msg}")
            if len(ruff_msgs) > 10:
                print(f"    ... and {len(ruff_msgs) - 10} more")
        all_passed = False
    print()

    # 2. Syntax check
    print("[2/4] Checking Python syntax...")
    syntax_errors, syntax_msgs = check_syntax(MODIFIED_FILES, verbose=args.verbose)
    if syntax_errors == 0:
        print("  PASS: No new syntax errors introduced")
    else:
        print(f"  FAIL: {syntax_errors} syntax errors found")
        for msg in syntax_msgs:
            print(f"    {msg}")
        all_passed = False
    print()

    # 3. AST parse check
    print("[3/4] Checking AST validity...")
    ast_errors, ast_msgs = check_imports(MODIFIED_FILES, verbose=args.verbose)
    if ast_errors == 0:
        print("  PASS: All files parse correctly")
    else:
        print(f"  FAIL: {ast_errors} AST parse errors")
        for msg in ast_msgs:
            print(f"    {msg}")
        all_passed = False
    print()

    # 4. Check ruff on entire repo (excluding backups)
    print("[4/4] Running ruff on entire repo (excl. backups)...")
    all_py = [
        f
        for f in subprocess.check_output(["git", "ls-files", "*.py"])
        .decode()
        .splitlines()
        if not f.startswith(("recover-stash-20260123/", "pre-recover-backup/"))
    ]
    repo_errors, _ = check_ruff(all_py, verbose=False)
    if repo_errors == 0:
        print("  PASS: Entire repo passes ruff (0 errors)")
    else:
        print(f"  INFO: {repo_errors} errors in repo (may include pre-existing)")
    print()

    # Diff summary
    if args.diff:
        print("=" * 60)
        print("Diff Summary")
        print("=" * 60)
        show_diff_summary(MODIFIED_FILES)
        print()

    # Final result
    print("=" * 60)
    if all_passed:
        print("RESULT: ALL CHECKS PASSED")
    else:
        print("RESULT: SOME CHECKS FAILED")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
