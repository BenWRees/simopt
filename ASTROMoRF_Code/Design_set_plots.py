"""This script creates three plots showing the evolution of PROJECTED design sets in.

ASTRO-MoRF:


Plots show the projected design set X @ U being well-poised on the projected trust
region
with center U.T @ x_k in the active subspace.

1. Initial design set with coordinate basis (projected view)
2. Design set with reused point (rotated basis, projected view)
3. Design set after geometry improvement (projected view)
"""

import sys
from pathlib import Path

sys.path.append(str(Path(sys.modules[__name__].__file__).parent.parent.resolve()))

import matplotlib.pyplot as plt
import numpy as np

# Import functions from the other modules
from design_set_test import construct_interpolation_set
from geometry_improvement_test import improve_geometry
from matplotlib.patches import Circle

from simopt.base import Solution
from simopt.experiment_base import instantiate_problem


def compute_poisedness(X, center, delta, U):  # noqa: ANN001, ANN201, N803
    """Compute a poisedness metric for a design set in the PROJECTED space.

    Uses the condition number of the polynomial basis matrix.
    Lower condition number = better conditioned = more well-poised.

    Args:
        X: Design points in FULL space (n_points x full_dim)
        center: Center point in FULL space
        delta: Trust region radius
        U: Active subspace matrix (full_dim x active_dim)

    Returns:
        condition_number: Condition number of the basis matrix (lower is better)
        log_abs_det: Log absolute determinant (higher is better for square matrices)
    """
    d = U.shape[1]  # Active subspace dimension
    s_old = np.array(center).reshape(-1, 1)

    # Build polynomial basis matrix for quadratic model
    n_points = X.shape[0]

    # Project points into active subspace and scale by delta
    # Y[i] = U.T @ (X[i] - center) / delta
    Y = np.array(  # noqa: N806
        [U.T @ (X[i, :].reshape(-1, 1) - s_old) / delta for i in range(n_points)]
    )
    Y = Y.reshape(n_points, d)  # noqa: N806

    # Build polynomial basis matrix (quadratic)
    if d == 2:
        Phi = np.zeros((n_points, 6))  # noqa: N806
        for i in range(n_points):
            y1, y2 = Y[i, 0], Y[i, 1]
            Phi[i, :] = [1, y1, y2, y1**2, y1 * y2, y2**2]
    else:
        # General case for arbitrary d
        Phi = np.ones((n_points, 1))  # noqa: N806
        for i in range(n_points):
            y = Y[i, :]
            Phi = np.hstack([Phi, y.reshape(1, -1)])  # noqa: N806

    # Compute condition number
    cond_number = np.linalg.cond(Phi)

    # For square matrices, compute log absolute determinant
    if Phi.shape[0] == Phi.shape[1]:
        det = np.linalg.det(Phi)
        log_abs_det = np.log10(np.abs(det)) if det != 0 else -np.inf
    else:
        log_abs_det = None

    return cond_number, log_abs_det


def plot_design_set(
    ax,  # noqa: ANN001
    design_points,  # noqa: ANN001
    center,  # noqa: ANN001
    radius,  # noqa: ANN001
    U,  # noqa: ANN001, N803
    title,  # noqa: ANN001
    reused_point=None,  # noqa: ANN001, N803, RUF100
) -> None:
    """Plot a PROJECTED design set with a circle showing the projected trust region.

    This function projects the full-space design points X into the active subspace
    via X @ U, and plots them relative to the projected center U.T @ center.
    The trust region is shown in the projected space.

    Args:
        ax: Matplotlib axis object
        design_points: Array of design points in FULL space (shape: n_points x full_dim)
        center: Center point in FULL space (shape: full_dim,)
        radius: Radius of the trust region
        U: Active subspace matrix (full_dim x active_dim)
        title: Title for the plot
        reused_point: Optional reused point to highlight (in FULL space)
    """
    # Project design points into active subspace: X @ U
    # Each row: (X[i] - center) @ U gives projected displacement
    center = np.array(center).ravel()
    projected_points = (design_points - center) @ U  # Shape: (n_points, active_dim)

    # Projected center is at origin (since we've subtracted center before projecting)
    projected_center = np.zeros(U.shape[1])

    # Handle 2D active subspace
    if U.shape[1] == 2:
        # Plot the trust region circle in projected space
        circle = Circle(
            projected_center,
            radius,
            fill=False,
            color="blue",
            linestyle="-",
            linewidth=2,
            label="Trust Region",
        )
        ax.add_patch(circle)

        # Plot the projected center point
        ax.plot(
            projected_center[0],
            projected_center[1],
            "ko",
            markersize=10,
            zorder=5,
            label="Center",
        )

        # Plot projected design points
        ax.plot(
            projected_points[:, 0],
            projected_points[:, 1],
            "go",
            markersize=8,
            zorder=3,
            label="Design Points",
        )

        # Highlight reused point if provided
        if reused_point is not None:
            reused_point = np.array(reused_point).ravel()
            projected_reused = (reused_point - center) @ U
            ax.plot(
                projected_reused[0],
                projected_reused[1],
                "ms",
                markersize=12,
                zorder=4,
                label="Reused Point",
            )

        # Set axis properties
        # ax.set_xlabel('u₁ (1st active direction)', fontsize=12)
        # ax.set_ylabel('u₂ (2nd active direction)', fontsize=12)
        # ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

        # Set axis limits to show the trust region nicely
        margin = radius * 0.5
        ax.set_xlim(
            projected_center[0] - radius - margin, projected_center[0] + radius + margin
        )
        ax.set_ylim(
            projected_center[1] - radius - margin, projected_center[1] + radius + margin
        )

        # Add legend
        # ax.legend(loc='upper right', fontsize=9)

    elif U.shape[1] == 1:
        # Handle 1D active subspace (plot on a line)
        ax.axvline(
            x=projected_center[0],
            color="blue",
            linestyle="-",
            linewidth=2,
            label="Center",
        )
        ax.axvline(
            x=projected_center[0] - radius,
            color="blue",
            linestyle="--",
            linewidth=1,
            alpha=0.5,
        )
        ax.axvline(
            x=projected_center[0] + radius,
            color="blue",
            linestyle="--",
            linewidth=1,
            alpha=0.5,
        )

        # Plot projected design points
        y_coords = np.zeros_like(projected_points[:, 0])
        ax.plot(
            projected_points[:, 0],
            y_coords,
            "go",
            markersize=8,
            zorder=3,
            label="Design Points",
        )

        if reused_point is not None:
            reused_point = np.array(reused_point).ravel()
            projected_reused = (reused_point - center) @ U
            ax.plot(
                projected_reused[0],
                0,
                "ms",
                markersize=12,
                zorder=4,
                label="Reused Point",
            )

        ax.set_xlabel("u₁ (active direction)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylim(-1, 1)
        ax.legend(loc="upper right", fontsize=9)
    else:
        raise ValueError(
            f"Cannot plot {U.shape[1]}D active subspace. Only 1D or 2D supported."
        )


def main() -> None:  # noqa: D103
    # Set random seed for reproducibility
    np.random.seed(42)

    # Create a simple problem
    problem = instantiate_problem("EXAMPLE-1")

    # Generate random solution in higher dimensional space
    # We'll use a problem with at least 3 dimensions and project to 2D active subspace
    full_dim = problem.dim  # Should be >= 2
    random_x = np.random.uniform(0.5, 2.5, size=full_dim)
    initial_solution = Solution(tuple(random_x), problem)
    radius = np.random.uniform(0.5, 1.5)

    print(f"Full problem dimension: {full_dim}")
    print(f"Random initial solution: {initial_solution.x}")
    print(f"Random radius: {radius:.3f}")

    # Define a 2D active subspace
    # For demonstration, use identity if problem is 2D, or first 2 columns if higher
    if full_dim == 2:
        U = np.eye(2)  # noqa: N806
        print("Using identity matrix for U (full-dimensional case)")
    else:
        # Use first 2 principal directions (or random orthonormal basis)
        U = np.eye(full_dim)[:, :2]  # noqa: N806
        print("Using first 2 columns of identity as active subspace")

    active_dim = U.shape[1]
    print(f"Active subspace dimension: {active_dim}")
    print(f"U matrix shape: {U.shape}")

    # Initialize visited points list
    visited_pts_list = [initial_solution]

    # Create figure with 3 subplots
    _fig, axes = plt.subplots(figsize=(18, 5))
    # fig.suptitle('Evolution of Projected Design Sets in ASTROMoRF\n' +
    #              f'(Design set X projected via X @ U into {active_dim}D active subspace)',
    #              fontsize=16, fontweight='bold')

    # ========== PLOT 1: Initial design set (no reuse) ==========
    print("\n--- Creating Plot 1: Initial Design Set (Projected View) ---")
    design_set_1, _ = construct_interpolation_set(
        current_solution=initial_solution,
        problem=problem,
        U=U,
        delta_k=radius,
        k=1,  # First iteration, no reuse
        visited_pts_list=visited_pts_list,
    )

    # Project to show what we're plotting
    center_1 = np.array(initial_solution.x)
    projected_set_1 = (design_set_1 - center_1) @ U
    print(f"Design set 1 shape (full space): {design_set_1.shape}")
    print(f"Projected design set 1 shape: {projected_set_1.shape}")
    print(f"Projected center (should be ~0): {U.T @ (center_1 - center_1)}")

    plot_design_set(
        ax=axes,
        design_points=design_set_1,
        center=center_1,
        radius=radius,
        U=U,
        title="1. Initial Design Set\n(Coordinate Basis in Active Subspace)",
        reused_point=None,
    )
    print("Projected points plotted in active subspace")

    # ========== PLOT 2: Design set with reused point ==========
    print("\n--- Creating Plot 2: Design Set with Reuse (Projected View) ---")
    # Move the solution to a new location in full space
    # Move along directions in the active subspace primarily
    shift_in_active_space = np.array([radius * 0.6, radius * 0.4])
    shift_in_full_space = U @ shift_in_active_space
    new_x = np.array(initial_solution.x) + shift_in_full_space
    new_solution = Solution(tuple(new_x), problem)

    print(f"Shift in active space: {shift_in_active_space}")
    print(f"Shift in full space: {shift_in_full_space}")
    print(f"New solution: {new_solution.x}")

    # Add some points to visited list to enable reuse
    visited_pts_list.append(Solution(tuple(design_set_1[1]), problem))
    visited_pts_list.append(Solution(tuple(design_set_1[2]), problem))

    # Construct design set with reuse
    design_set_2, reused_idx = construct_interpolation_set(
        current_solution=new_solution,
        problem=problem,
        U=U,
        delta_k=radius,
        k=2,  # Second iteration, enable reuse
        visited_pts_list=visited_pts_list,
    )

    reused_point = np.array(visited_pts_list[reused_idx].x) if reused_idx >= 0 else None
    print(f"Reused point index: {reused_idx}")
    if reused_point is not None:
        print(f"Reused point coordinates (full space): {reused_point}")
        projected_reused = (reused_point - np.array(new_solution.x)) @ U
        print(f"Reused point (projected): {projected_reused}")

    # Project to show what we're plotting
    center_2 = np.array(new_solution.x)
    projected_set_2 = (design_set_2 - center_2) @ U
    print(f"Design set 2 shape (full space): {design_set_2.shape}")
    print(f"Projected design set 2 shape: {projected_set_2.shape}")

    plot_design_set(
        ax=axes[0, 1],
        design_points=design_set_2,
        center=center_2,
        radius=radius,
        U=U,
        title="2. Design Set with Reuse\n(Rotated Basis in Active Subspace)",
        reused_point=reused_point,
    )
    print("Projected points plotted in active subspace")

    # ========== PLOT 3: Design set after geometry improvement ==========
    print("\n--- Creating Plot 3: Geometry Improved Design Set (Projected View) ---")
    # Reuse the design set from Plot 2 (as if criticality step failed)
    X_before_improvement = design_set_2.copy()  # noqa: N806

    # Debug: Check the distances IN PROJECTED SPACE
    center_projected = np.zeros(active_dim)  # Center in projected space is at origin
    projected_before = (X_before_improvement - center_2) @ U
    distances = np.linalg.norm(projected_before - center_projected, axis=1, ord=np.inf)
    max_dist = np.max(distances)
    threshold = 0.01 * radius  # epsilon_1 = 0.01 in improve_geometry
    print(f"Max distance from center in PROJECTED space (inf norm): {max_dist:.6f}")
    print(f"Threshold (0.01*radius): {threshold:.6f}")
    print(f"Will trigger improvement: {max_dist > threshold}")
    print(f"All projected distances: {distances}")

    # Create function values (using simple quadratic for demonstration)
    def fn(x):  # noqa: ANN001, ANN202
        return np.linalg.norm(x - np.array(new_solution.x)) ** 2

    fX_before = np.array(  # noqa: N806
        [fn(X_before_improvement[i, :]) for i in range(X_before_improvement.shape[0])]
    ).reshape(-1, 1)

    # Create interpolation solutions from the reused design set
    interpolation_sols = [Solution(tuple(x), problem) for x in X_before_improvement]

    # Compute poisedness BEFORE improvement (in projected space)
    cond_before, det_before = compute_poisedness(
        X_before_improvement, new_solution.x, radius, U
    )
    print("\nPoisedness BEFORE improvement (in projected space):")
    print(f"  Condition number: {cond_before:.4e} (lower is better)")
    if det_before is not None:
        print(f"  Log10|det|: {det_before:.4f} (higher is better)")

    print(f"\nDesign set BEFORE improvement (full space):\n{X_before_improvement}")
    print(f"Projected design set BEFORE:\n{projected_before}")

    # Shrink trust-region radius to 85% for geometry improvement
    reduced_radius = radius
    print(f"\nReducing trust-region radius: {radius:.4f} → {reduced_radius:.4f} (85%)")

    # Improve geometry (as if criticality check failed and we need better geometry)
    (
        X_improved,  # noqa: N806
        _fX_improved,  # noqa: N806
        interpolation_sols,
        visited_pts_list,
        _updated_radius,
        _,
    ) = improve_geometry(
        k=2,
        problem=problem,
        current_solution=new_solution,
        current_f=fX_before[0],
        delta_k=reduced_radius,  # Use reduced radius
        U=U,
        visited_pts_list=visited_pts_list,
        X=X_before_improvement,
        fX=fX_before,
        interpolation_solutions=interpolation_sols,
        expended_budget=0,
    )

    projected_after = (X_improved - center_2) @ U
    print(f"\nDesign set AFTER improvement (full space):\n{X_improved}")
    print(f"Projected design set AFTER:\n{projected_after}")
    print(
        f"Design sets are different: {not np.allclose(X_improved, X_before_improvement)}"
    )

    # Compute poisedness AFTER improvement (using reduced radius, in projected space)
    cond_after, det_after = compute_poisedness(
        X_improved, new_solution.x, reduced_radius, U
    )
    print("\nPoisedness AFTER improvement (in projected space):")
    print(f"  Condition number: {cond_after:.4e} (lower is better)")
    if det_after is not None:
        print(f"  Log10|det|: {det_after:.4f} (higher is better)")

    # Compare poisedness
    print(f"\n{'=' * 60}")
    print("POISEDNESS COMPARISON (in projected active subspace):")
    print(f"{'=' * 60}")
    print(f"Condition number improvement: {cond_before:.4e} → {cond_after:.4e}")
    if cond_after < cond_before:
        improvement_pct = (cond_before - cond_after) / cond_before * 100
        print(f"✓ Condition number IMPROVED by {improvement_pct:.2f}%")
    else:
        print("✗ Condition number got worse")

    if det_before is not None and det_after is not None:
        print(f"Log10|det| change: {det_before:.4f} → {det_after:.4f}")
        if det_after > det_before:
            print("✓ Determinant IMPROVED (larger absolute value)")
        else:
            print("✗ Determinant got smaller")
    print(f"{'=' * 60}\n")

    plot_design_set(
        ax=axes[1, 0],
        design_points=X_improved,
        center=center_2,
        radius=reduced_radius,  # Plot with reduced radius
        U=U,
        title="3. After Geometry Improvement\n(Optimized Placement in Active Subspace, 85% Radius)",
        reused_point=None,
    )
    print(f"Design set 3 shape (full space): {X_improved.shape}")
    print(f"Projected design set 3 shape: {projected_after.shape}")

    # ========== PLOT 4: Geometry improvement WITHOUT reused points ==========
    print("\n--- Creating Plot 4: Geometry Improvement (No Reuse, Projected View) ---")
    # Construct a fresh design set at `new_solution` without reusing visited points
    visited_pts_list_no_reuse = [initial_solution]  # empty => no reuse
    design_set_no_reuse, _ = construct_interpolation_set(
        current_solution=new_solution,
        problem=problem,
        U=U,
        delta_k=radius,
        k=2,
        visited_pts_list=visited_pts_list_no_reuse,
    )

    # Compute function values and interpolation solutions for the new design set
    center_no_reuse = np.array(new_solution.x)

    def fn_no_reuse(x):  # noqa: ANN001, ANN202
        return np.linalg.norm(x - center_no_reuse) ** 2

    fX_no_reuse = np.array(  # noqa: N806
        [
            fn_no_reuse(design_set_no_reuse[i, :])
            for i in range(design_set_no_reuse.shape[0])
        ]
    ).reshape(-1, 1)
    interpolation_sols_no_reuse = [
        Solution(tuple(x), problem) for x in design_set_no_reuse
    ]
    reduced_radius_no_reuse = radius

    # Improve geometry on the fresh design set (visited list empty => no reuse)
    (
        X_improved_no_reuse,  # noqa: N806
        _fX_improved_no_reuse,  # noqa: N806
        interpolation_sols_no_reuse,
        visited_pts_list_no_reuse,
        _updated_radius_no_reuse,
        _,
    ) = improve_geometry(
        k=2,
        problem=problem,
        current_solution=new_solution,
        current_f=fX_no_reuse[0],
        delta_k=reduced_radius_no_reuse,
        U=U,
        visited_pts_list=visited_pts_list_no_reuse,
        X=design_set_no_reuse,
        fX=fX_no_reuse,
        interpolation_solutions=interpolation_sols_no_reuse,
        expended_budget=0,
    )

    projected_improved_no_reuse = (X_improved_no_reuse - center_no_reuse) @ U
    print(
        f"\nDesign set (no reuse) BEFORE improvement (full space):\n{design_set_no_reuse}"
    )
    print(
        f"Projected design set (no reuse) BEFORE:\n{(design_set_no_reuse - center_no_reuse) @ U}"
    )
    print(
        f"\nDesign set AFTER improvement (no reuse, full space):\n{X_improved_no_reuse}"
    )
    print(f"Projected design set AFTER (no reuse):\n{projected_improved_no_reuse}")

    # Compute poisedness (projected) before/after
    cond_no_reuse_before, det_no_reuse_before = compute_poisedness(
        design_set_no_reuse, new_solution.x, radius, U
    )
    cond_no_reuse_after, det_no_reuse_after = compute_poisedness(
        X_improved_no_reuse, new_solution.x, reduced_radius_no_reuse, U
    )
    print("\nPoisedness BEFORE improvement (no reuse, projected):")
    print(f"  Condition number: {cond_no_reuse_before:.4e} (lower is better)")
    if det_no_reuse_before is not None:
        print(f"  Log10|det|: {det_no_reuse_before:.4f} (higher is better)")
    print("Poisedness AFTER improvement (no reuse, projected):")
    print(f"  Condition number: {cond_no_reuse_after:.4e} (lower is better)")
    if det_no_reuse_after is not None:
        print(f"  Log10|det|: {det_no_reuse_after:.4f} (higher is better)")

    # Fourth plot: geometry improvement without reuse (center at new_solution)
    plot_design_set(
        ax=axes[1, 1],
        design_points=X_improved_no_reuse,
        center=center_no_reuse,
        radius=reduced_radius_no_reuse,
        U=U,
        title="4. Geometry Improvement\n(No Reuse, Active Subspace)",
        reused_point=None,
    )
    plt.tight_layout()
    plt.savefig(
        "design_set_evolution_projected_4plots.png", dpi=300, bbox_inches="tight"
    )
    print("\n✓ Plots saved as 'design_set_evolution_projected_4plots.png'")
    print("✓ Fourth plot shows geometry improvement with NO reused points.")
    plt.show()


if __name__ == "__main__":
    main()
