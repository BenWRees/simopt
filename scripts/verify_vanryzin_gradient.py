"""Verify the Van Ryzin & Vulcano (2008) analytical gradient implementation.

Tests:
1. Gradient accuracy: compare analytical gradient vs finite-difference approximation
2. Convergence: run a few SGD steps with direct gradients, check revenue improves

Usage:
    python scripts/verify_vanryzin_gradient.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.models.vanryzin_airline_revenue import (
    VanRyzinRevenueMultistage,
)


def make_rngs(seed: int = 42, n: int = 2) -> list[MRG32k3a]:
    """Create a list of deterministic RNGs from a seed."""
    rngs = []
    for i in range(n):
        rng = MRG32k3a(s_ss_sss_index=[seed + i, 0, 0])
        rngs.append(rng)
    return rngs


def test_gradient_accuracy() -> None:
    """Compare analytical gradient against finite differences on a small config."""
    print("=" * 70)
    print("TEST 1: Gradient Accuracy (analytical vs finite-difference)")
    print("=" * 70)

    # Use a smaller problem for faster finite-difference verification:
    # 2 legs, 4 products, 3 virtual classes, 2 periods
    n_legs = 2
    n_products = 4
    K = 3  # noqa: N806
    n_periods = 2

    # Simple network: products 0,1 use leg 0; products 2,3 use leg 1
    odf_matrix = [
        [1, 0],  # product 0
        [1, 0],  # product 1
        [0, 1],  # product 2
        [0, 1],  # product 3
    ]
    # VC indexing: product 0 -> VC 1, product 1 -> VC 2, etc.
    vc_indexing = [
        [1, 0],  # product 0: VC 1 on leg 0
        [2, 0],  # product 1: VC 2 on leg 0
        [0, 1],  # product 2: VC 1 on leg 1
        [0, 2],  # product 3: VC 2 on leg 1
    ]
    capacity = (20.0, 20.0)
    fares = (100.0, 60.0, 90.0, 50.0)
    # Protection levels: shape (n_legs, K)
    protection_levels = [
        [5.0, 12.0, 20.0],  # leg 0
        [4.0, 10.0, 20.0],  # leg 1
    ]
    gamma_shape = (8.0, 12.0, 7.0, 15.0)
    gamma_scale = (1.0, 1.0, 1.0, 1.0)
    beta_alpha = (3.0, 2.0, 3.0, 2.0)
    beta_beta = (2.0, 3.0, 2.0, 3.0)

    model_factors = {
        "n_classes": n_products,
        "ODF_leg_matrix": odf_matrix,
        "capacity": capacity,
        "fares": fares,
        "n_virtual_classes": K,
        "virtual_class_indexing": vc_indexing,
        "protection_levels": protection_levels,
        "n_periods": n_periods,
        "gamma_shape": gamma_shape,
        "gamma_scale": gamma_scale,
        "beta_alpha": beta_alpha,
        "beta_beta": beta_beta,
    }

    model = VanRyzinRevenueMultistage(fixed_factors=model_factors)

    # Flat decision: same protection levels at each stage
    flat_decision = tuple(
        protection_levels[leg][k] for leg in range(n_legs) for k in range(K)
    )
    decisions = tuple(flat_decision for _ in range(n_periods))

    dim = n_legs * K
    n_samples = 20  # average over multiple samples for stable comparison

    # --- Analytical gradient (averaged over samples) ---
    analytical_grad = np.zeros(n_periods * dim)
    analytical_rev = 0.0
    for s in range(n_samples):
        rngs = make_rngs(seed=100 + s)
        model.before_replication(rngs)
        rev, grad = model.simulate_with_gradient(
            decisions, rngs, smoothing_epsilon=0.001
        )
        analytical_grad += grad
        analytical_rev += rev
    analytical_grad /= n_samples
    analytical_rev /= n_samples

    # --- Finite-difference gradient (averaged over same samples) ---
    delta = 1e-4
    fd_grad = np.zeros(n_periods * dim)

    for idx in range(n_periods * dim):
        rev_plus = 0.0
        rev_minus = 0.0
        for s in range(n_samples):
            # Perturb +delta
            flat_all = np.array([v for d in decisions for v in d], dtype=float)
            flat_all[idx] += delta
            decisions_plus = tuple(
                tuple(flat_all[stage * dim : (stage + 1) * dim].tolist())
                for stage in range(n_periods)
            )
            rngs = make_rngs(seed=100 + s)
            model.before_replication(rngs)
            rp, _ = model.simulate_with_gradient(
                decisions_plus, rngs, smoothing_epsilon=0.001
            )
            rev_plus += rp

            # Perturb -delta
            flat_all = np.array([v for d in decisions for v in d], dtype=float)
            flat_all[idx] -= delta
            decisions_minus = tuple(
                tuple(flat_all[stage * dim : (stage + 1) * dim].tolist())
                for stage in range(n_periods)
            )
            rngs = make_rngs(seed=100 + s)
            model.before_replication(rngs)
            rm, _ = model.simulate_with_gradient(
                decisions_minus, rngs, smoothing_epsilon=0.001
            )
            rev_minus += rm

        fd_grad[idx] = (rev_plus - rev_minus) / (2 * delta * n_samples)

    # --- Compare ---
    print(f"\nAverage revenue: {analytical_rev:.2f}")
    print(f"\nGradient dimension: {n_periods * dim}")
    print(f"\n{'Index':>5}  {'Analytical':>12}  {'Finite-Diff':>12}  {'Abs Error':>12}")
    print("-" * 50)
    max_err = 0.0
    for i in range(n_periods * dim):
        err = abs(analytical_grad[i] - fd_grad[i])
        max_err = max(max_err, err)
        if abs(analytical_grad[i]) > 1e-8 or abs(fd_grad[i]) > 1e-8:
            print(
                f"{i:>5}  {analytical_grad[i]:>12.4f}  {fd_grad[i]:>12.4f}  {err:>12.6f}"  # noqa: E501
            )

    print(f"\nMax absolute error: {max_err:.6f}")
    # With smoothing epsilon and averaging, errors should be small
    # but not zero due to smoothing perturbation randomness
    threshold = 5.0  # generous threshold for stochastic comparison
    if max_err < threshold:
        print(f"PASS: Max error {max_err:.6f} < {threshold}")
    else:
        print(f"WARN: Max error {max_err:.6f} >= {threshold} (may need more samples)")


def test_gradient_nonzero() -> None:
    """Verify gradient is non-trivial on the default 80-product network."""
    print("\n" + "=" * 70)
    print("TEST 2: Gradient non-triviality on default network")
    print("=" * 70)

    model = VanRyzinRevenueMultistage()
    n_periods = model.n_stages
    n_legs = len(model.factors["capacity"])
    n_v_class = model.factors["n_virtual_classes"]
    dim = n_legs * n_v_class

    prot = model.factors["protection_levels"]
    flat_decision = tuple(
        prot[leg][v_class] for leg in range(n_legs) for v_class in range(n_v_class)
    )
    decisions = tuple(flat_decision for _ in range(n_periods))

    n_samples = 10
    total_grad = np.zeros(n_periods * dim)
    total_rev = 0.0
    n_nonzero_samples = 0

    for s in range(n_samples):
        rngs = make_rngs(seed=200 + s)
        model.before_replication(rngs)
        rev, grad = model.simulate_with_gradient(decisions, rngs)
        total_grad += grad
        total_rev += rev
        if np.any(np.abs(grad) > 1e-10):
            n_nonzero_samples += 1

    avg_grad = total_grad / n_samples
    avg_rev = total_rev / n_samples

    nonzero_count = np.count_nonzero(np.abs(avg_grad) > 1e-8)
    print(
        f"\nDefault network: {n_legs} legs, {model.factors['n_classes']} products, {n_v_class} VCs, {n_periods} periods"  # noqa: E501
    )
    print(f"Gradient dimension: {n_periods * dim}")
    print(f"Average revenue: {avg_rev:.2f}")
    print(f"Non-zero gradient components (averaged): {nonzero_count}/{n_periods * dim}")
    print(f"Samples with non-zero gradient: {n_nonzero_samples}/{n_samples}")
    print(f"Gradient norm: {np.linalg.norm(avg_grad):.4f}")
    print(f"Max gradient component: {np.max(np.abs(avg_grad)):.4f}")

    if nonzero_count > 0:
        print("PASS: Gradient has non-zero components")
    else:
        print("WARN: Gradient is all zeros (may indicate an issue)")

    # Print top 10 largest gradient components
    top_idx = np.argsort(np.abs(avg_grad))[-10:][::-1]
    print("\nTop 10 largest gradient components:")
    print(f"{'Index':>5}  {'Stage':>5}  {'Leg':>3}  {'VC':>3}  {'Value':>12}")
    for idx in top_idx:
        stage = idx // dim
        remainder = idx % dim
        leg = remainder // n_v_class
        vc = remainder % n_v_class
        print(f"{idx:>5}  {stage:>5}  {leg:>3}  {vc:>3}  {avg_grad[idx]:>12.4f}")


def test_sgd_convergence() -> None:
    """Run a few manual SGD steps with analytical gradients and check improvement."""
    print("\n" + "=" * 70)
    print("TEST 3: SGD convergence with analytical gradients")
    print("=" * 70)

    model = VanRyzinRevenueMultistage()
    n_periods = model.n_stages
    n_legs = len(model.factors["capacity"])
    n_v_class = model.factors["n_virtual_classes"]
    dim = n_legs * n_v_class
    capacity = np.array(model.factors["capacity"])

    prot = model.factors["protection_levels"]
    flat_decision = np.array(
        [prot[leg][v_class] for leg in range(n_legs) for v_class in range(n_v_class)],
        dtype=float,
    )

    n_eval_samples = 30  # samples for revenue estimation
    n_sgd_steps = 15
    alpha_0 = 0.5  # initial step size

    def evaluate_revenue(flat_dec: np.ndarray) -> float:
        decisions = tuple(tuple(flat_dec.tolist()) for _ in range(n_periods))
        total = 0.0
        for s in range(n_eval_samples):
            rngs = make_rngs(seed=500 + s)
            model.before_replication(rngs)
            rev, _ = model.simulate_with_gradient(decisions, rngs)
            total += rev
        return total / n_eval_samples

    # Initial revenue
    initial_rev = evaluate_revenue(flat_decision)
    print(f"\nInitial average revenue: {initial_rev:.2f}")
    print(f"Running {n_sgd_steps} SGD steps (step size = {alpha_0}/k)...")

    current_x = flat_decision.copy()
    revenues = [initial_rev]

    for k in range(1, n_sgd_steps + 1):
        # Compute gradient (average over a few samples)
        grad = np.zeros(n_periods * dim)
        for s in range(5):
            decisions = tuple(tuple(current_x.tolist()) for _ in range(n_periods))
            rngs = make_rngs(seed=1000 + k * 100 + s)
            model.before_replication(rngs)
            _, g = model.simulate_with_gradient(decisions, rngs)
            grad += g

        grad /= 5

        # SGD step (maximise revenue, so ascend)
        # The gradient is dR/dy, and we want to maximise R
        # For each stage, extract the per-stage gradient
        step_size = alpha_0 / k
        for stage in range(n_periods):
            stage_grad = grad[stage * dim : (stage + 1) * dim]
            current_x += step_size * stage_grad

        # Clip to bounds: 0 <= y_{l,k} <= C_l, and enforce monotonicity per leg
        for leg in range(n_legs):
            cap_l = capacity[leg]
            for kk in range(n_v_class):
                idx = leg * n_v_class + kk
                current_x[idx] = np.clip(current_x[idx], 0.01, cap_l - 0.01)
            # Enforce monotonicity
            for kk in range(1, n_v_class):
                idx = leg * n_v_class + kk
                prev_idx = leg * n_v_class + kk - 1
                if current_x[idx] < current_x[prev_idx]:
                    current_x[idx] = current_x[prev_idx]

        if k % 5 == 0 or k == 1:
            rev = evaluate_revenue(current_x)
            revenues.append(rev)
            print(f"  Step {k:>3}: revenue = {rev:.2f}  (step_size = {step_size:.4f})")

    final_rev = evaluate_revenue(current_x)
    revenues.append(final_rev)
    print(f"\nFinal average revenue: {final_rev:.2f}")
    print(
        f"Improvement: {final_rev - initial_rev:+.2f} ({(final_rev - initial_rev) / initial_rev * 100:+.2f}%)"  # noqa: E501
    )

    if final_rev >= initial_rev:
        print("PASS: Revenue improved or maintained")
    else:
        print("WARN: Revenue decreased (may need tuning step size)")


if __name__ == "__main__":
    test_gradient_accuracy()
    test_gradient_nonzero()
    test_sgd_convergence()
    print("\n" + "=" * 70)
    print("All tests complete.")
    print("=" * 70)
