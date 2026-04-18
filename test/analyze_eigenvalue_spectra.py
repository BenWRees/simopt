"""Analyze eigenvalue spectra of different problems to understand their structure."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from demo.pickle_files_journal_paper import SCALABLE_PROBLEMS, scale_dimension
from simopt.experiment.run_solver import _set_up_rngs
from simopt.solvers.astromorf import ASTROMORF


def analyze_problem_eigenvalues(
    problem_name: str,
    dimension: int = 100,
    n_samples: int = 50,  # noqa: ARG001
) -> list[np.ndarray]:
    """Run a few iterations of ASTROMORF and extract the eigenvalue spectrum."""
    from simopt.directory import problem_directory

    # Create problem
    budget = 2000  # Just enough for a few iterations
    if problem_name in SCALABLE_PROBLEMS:
        problem = scale_dimension(problem_name, dimension, budget)
    else:
        problem = problem_directory[problem_name](fixed_factors={"budget": budget})

    # Create solver
    solver = ASTROMORF(
        fixed_factors={
            "Record Diagnostics": False,
            "polynomial degree": 2,
            "adaptive subspace dimension": True,
        }
    )

    # Set up RNGs
    _set_up_rngs(solver, problem, mrep=42)

    # Run solver
    solver.run(problem)

    # Extract eigenvalue spectra from history
    eigenvalue_spectra = []
    for info in solver.previous_model_information:
        spec = info.get("eigenvalue_spectrum")
        if spec is not None:
            eigenvalue_spectra.append(np.array(spec, dtype=float))

    if hasattr(solver, "gradient_eigenvalues") and solver.gradient_eigenvalues:
        for spec in solver.gradient_eigenvalues:
            eigenvalue_spectra.append(np.array(spec, dtype=float))

    return eigenvalue_spectra


def compute_variance_metrics(eigenvalues: np.ndarray) -> dict:
    """Compute various metrics about variance distribution."""
    # Normalize
    total = np.sum(eigenvalues)
    if total < 1e-12:
        return {}

    normalized = eigenvalues / total
    cumulative = np.cumsum(normalized)

    # Find d for various variance thresholds
    d_50 = int(np.searchsorted(cumulative, 0.50) + 1)
    d_80 = int(np.searchsorted(cumulative, 0.80) + 1)
    d_90 = int(np.searchsorted(cumulative, 0.90) + 1)
    d_95 = int(np.searchsorted(cumulative, 0.95) + 1)
    d_99 = int(np.searchsorted(cumulative, 0.99) + 1)

    # Effective dimensionality (participation ratio)
    # Higher = more spread out, Lower = more concentrated
    effective_dim = 1.0 / np.sum(normalized**2) if np.sum(normalized**2) > 0 else 0

    # Top eigenvalue dominance
    top1_fraction = normalized[0] if len(normalized) > 0 else 0
    top3_fraction = (
        np.sum(normalized[:3]) if len(normalized) >= 3 else np.sum(normalized)
    )
    top10_fraction = (
        np.sum(normalized[:10]) if len(normalized) >= 10 else np.sum(normalized)
    )

    return {
        "d_50": d_50,
        "d_80": d_80,
        "d_90": d_90,
        "d_95": d_95,
        "d_99": d_99,
        "effective_dim": effective_dim,
        "top1_fraction": top1_fraction,
        "top3_fraction": top3_fraction,
        "top10_fraction": top10_fraction,
    }


def main() -> None:
    """Run main entry point."""
    # use arg.parse to get problem names and dimension
    parser = argparse.ArgumentParser(
        description="Analyze eigenvalue spectra of problems."
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        default=["SAN-1", "NETWORK-1", "ROSENBROCK-1", "DYNAMNEWS-1"],
        help="List of problem names to analyze",
    )
    parser.add_argument(
        "--dimension", type=int, default=100, help="Dimension to scale problems to"
    )
    args = parser.parse_args()

    problems = args.problems
    dimension = args.dimension

    print("=" * 80)
    print("EIGENVALUE SPECTRUM ANALYSIS")
    print("=" * 80)
    print(f"Problems: {problems}")
    print(f"Dimension: {dimension}")
    print("=" * 80)

    all_spectra = {}
    all_metrics = {}

    for problem_name in problems:
        print(f"\nAnalyzing {problem_name}...")

        try:
            spectra = analyze_problem_eigenvalues(problem_name, dimension)

            if spectra:
                # Use the last (most refined) spectrum
                final_spectrum = spectra[-1]
                all_spectra[problem_name] = final_spectrum

                metrics = compute_variance_metrics(final_spectrum)
                all_metrics[problem_name] = metrics

                print(
                    f"  Collected {len(spectra)} spectra, using final one with {len(final_spectrum)} eigenvalues"  # noqa: E501
                )
            else:
                print("  WARNING: No eigenvalue spectra collected!")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback

            traceback.print_exc()

    # Print comparison table
    print("\n" + "=" * 100)
    print("EIGENVALUE SPECTRUM COMPARISON")
    print("=" * 100)

    headers = [
        "Problem",
        "d_50%",
        "d_80%",
        "d_90%",
        "d_95%",
        "d_99%",
        "Eff.Dim",
        "Top1%",
        "Top3%",
        "Top10%",
    ]
    widths = [15, 8, 8, 8, 8, 8, 10, 8, 8, 8]

    header_str = " | ".join(h.ljust(w) for h, w in zip(headers, widths, strict=False))
    print(header_str)
    print("-" * len(header_str))

    for problem_name in problems:
        if problem_name not in all_metrics or all_metrics[problem_name] is None:
            print(f"{problem_name:<15} | NO DATA")
            continue

        m = all_metrics[problem_name]
        row = [
            problem_name[:15],
            str(m["d_50"]),
            str(m["d_80"]),
            str(m["d_90"]),
            str(m["d_95"]),
            str(m["d_99"]),
            f"{m['effective_dim']:.2f}",
            f"{m['top1_fraction'] * 100:.1f}%",
            f"{m['top3_fraction'] * 100:.1f}%",
            f"{m['top10_fraction'] * 100:.1f}%",
        ]
        row_str = " | ".join(str(v).ljust(w) for v, w in zip(row, widths, strict=False))
        print(row_str)

    print("=" * 100)

    # Interpretation
    print("\nINTERPRETATION:")
    print("-" * 60)

    for problem_name in problems:
        if problem_name not in all_metrics or all_metrics[problem_name] is None:
            continue

        m = all_metrics[problem_name]

        # Classify structure
        if m["top3_fraction"] > 0.95:
            structure = "HIGHLY SEPARABLE (top 3 dims capture >95%)"
        elif m["top10_fraction"] > 0.95:
            structure = "SEPARABLE (top 10 dims capture >95%)"
        elif m["d_95"] <= 20:
            structure = "MODERATELY COUPLED (d=20 captures 95%)"
        elif m["d_95"] <= 50:
            structure = "COUPLED (need d=50 for 95%)"
        else:
            structure = "HIGHLY COUPLED (need d>50 for 95%)"

        print(f"{problem_name}: {structure}")
        print(f"  - Effective dimensionality: {m['effective_dim']:.1f}")
        print(f"  - d=10 captures: ~{m['top10_fraction'] * 100:.0f}% variance")
        print(f"  - Need d={m['d_95']} for 95% variance")

    print("-" * 60)

    # Plot spectra
    try:
        _fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for idx, problem_name in enumerate(problems):
            if problem_name not in all_spectra:
                continue

            ax = axes[idx]
            spectrum = all_spectra[problem_name]

            # Normalize
            spectrum_norm = spectrum / np.sum(spectrum)
            cumulative = np.cumsum(spectrum_norm)

            # Plot
            ax.bar(
                range(1, min(31, len(spectrum_norm) + 1)),
                spectrum_norm[:30],
                alpha=0.7,
                label="Individual",
            )
            ax.plot(
                range(1, min(31, len(cumulative) + 1)),
                cumulative[:30],
                "r-",
                linewidth=2,
                label="Cumulative",
            )

            ax.axhline(
                y=0.95, color="g", linestyle="--", alpha=0.7, label="95% threshold"
            )
            ax.axvline(
                x=10, color="orange", linestyle="--", alpha=0.7, label="d=10 floor"
            )

            ax.set_xlabel("Dimension")
            ax.set_ylabel("Variance Fraction")
            ax.set_title(f"{problem_name}")
            ax.legend(fontsize=8)
            ax.set_xlim(0, 31)
            ax.set_ylim(0, 1.05)

        plt.tight_layout()
        plt.savefig("eigenvalue_spectra_comparison.png", dpi=150)
        print("\nPlot saved to: eigenvalue_spectra_comparison.png")

    except Exception as e:
        print(f"\nCould not create plot: {e}")

    print("\nAnalysis complete.")
    print(f"All metrics: {all_metrics}")
    print(f"All spectra: {all_spectra}")


if __name__ == "__main__":
    main()
