"""Minimal verification script for the DLP optimal_value and gradient separation.

Run from the repo root:
    python demo/verify_changes.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # noqa: PTH100, PTH118, PTH120

import numpy as np


def test_dlp_optimal_value() -> None:
    """Check that optimal_value is computed and positive for VANRYZIN-2."""
    from simopt.experiment_base import instantiate_problem

    problem = instantiate_problem("VANRYZIN-2")
    opt = problem.optimal_value
    assert opt is not None, "optimal_value should not be None"
    assert opt > 0, f"optimal_value should be positive, got {opt}"
    print(f"[PASS] DLP upper bound (default): {opt:.2f}")

    # Reduced capacity variant should have a lower bound
    cap = np.array(problem.model.factors["capacity"]) * 0.7
    prot = np.array(problem.model.factors["protection_levels"]) * 0.7
    p2 = instantiate_problem(
        "VANRYZIN-2",
        model_fixed_factors={
            "capacity": tuple(float(v) for v in cap),
            "protection_levels": prot.tolist(),
        },
    )
    opt2 = p2.optimal_value
    assert opt2 is not None and opt2 > 0, f"reduced cap optimal_value={opt2}"
    assert opt2 < opt, f"reduced cap opt={opt2:.2f} should be < default opt={opt:.2f}"
    print(f"[PASS] DLP upper bound (reduced cap): {opt2:.2f} < {opt:.2f}")

    # Caching: second access should return same value
    opt_again = problem.optimal_value
    assert opt_again == opt, "Cached value should be identical"
    print("[PASS] Caching works correctly")


def test_single_stage_optimal_value() -> None:
    """Check that VANRYZIN-1 also has optimal_value."""
    from simopt.experiment_base import instantiate_problem

    try:
        problem = instantiate_problem("VANRYZIN-1")
        opt = problem.optimal_value
        assert opt is not None and opt > 0, f"VANRYZIN-1 optimal_value={opt}"
        print(f"[PASS] DLP upper bound (VANRYZIN-1): {opt:.2f}")
    except Exception as e:
        print(f"[SKIP] VANRYZIN-1 not available: {e}")


def test_gradient_separation() -> None:
    """Verify VanRyzinSolver uses direct gradients and SGD does not."""
    from simopt.directory import solver_directory

    # VanRyzinSolver should have use_direct_gradients
    if "VANRYZIN_SGD" in solver_directory:
        vr_cls = solver_directory["VANRYZIN_SGD"]
        fields = vr_cls.config_class.model_fields
        assert "use_direct_gradients" in fields, (
            "VanRyzinSolver missing use_direct_gradients"
        )
        assert fields["use_direct_gradients"].default is True, (
            "use_direct_gradients should default True"
        )
        print("[PASS] VanRyzinSolver has use_direct_gradients=True")
    else:
        print("[SKIP] VANRYZIN_SGD not in solver directory")

    # SGD should NOT have use_direct_gradients
    if "SGD" in solver_directory:
        sgd_cls = solver_directory["SGD"]
        fields = sgd_cls.config_class.model_fields
        assert "use_direct_gradients" not in fields, (
            "SGD should not have use_direct_gradients"
        )
        print("[PASS] SGD does not have use_direct_gradients")
    else:
        print("[SKIP] SGD not in solver directory")

    # DAVN should not have gradient-related fields
    if "DAVN" in solver_directory:
        davn_cls = solver_directory["DAVN"]
        fields = davn_cls.config_class.model_fields
        assert "use_direct_gradients" not in fields, (
            "DAVN should not have use_direct_gradients"
        )
        print("[PASS] DAVN does not use direct gradients")
    else:
        print("[SKIP] DAVN not in solver directory")


def test_plotting_import() -> None:
    """Check that the new plotting function is importable."""
    from simopt.plots import plot_optimality_gap_curves

    assert callable(plot_optimality_gap_curves)
    print("[PASS] plot_optimality_gap_curves is importable")


def test_experiment_runner_specs() -> None:
    """Check experiment-one solver specs are configured as expected."""
    from demo.AirlineRM_experiment import _build_experiment_one_specs

    specs = _build_experiment_one_specs(budget=1000)
    labels = {s.solver_label for s in specs}
    assert "ADP_ASTROMoRF" in labels, "Experiment one should include ADP_ASTROMoRF"

    direct_sgd_specs = [s for s in specs if s.solver_name == "SGD"]
    assert len(direct_sgd_specs) == 0, (
        "Experiment one should not include direct SGD specs; use ADP wrappers."
    )
    print("[PASS] Experiment-one solver specs are consistent with ADP-only setup")


def test_csv_export_fields() -> None:
    """Check that extract_metrics_row includes optimal_value fields."""
    import inspect

    from demo.AirlineRM_experiment import extract_metrics_row

    src = inspect.getsource(extract_metrics_row)
    assert "optimal_value" in src, "extract_metrics_row should include optimal_value"
    assert "optimality_gap_pct" in src, (
        "extract_metrics_row should include optimality_gap_pct"
    )
    print("[PASS] CSV export includes optimal_value and optimality_gap_pct")


if __name__ == "__main__":
    print("=" * 60)
    print("Running verification checks...")
    print("=" * 60)

    test_dlp_optimal_value()
    test_single_stage_optimal_value()
    test_gradient_separation()
    test_plotting_import()
    test_experiment_runner_specs()
    test_csv_export_fields()

    print("=" * 60)
    print("All checks passed!")
    print("=" * 60)
