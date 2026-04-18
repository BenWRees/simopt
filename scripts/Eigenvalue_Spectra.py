#!/usr/bin/env python3
"""Compute and plot eigenvalue spectrum of the gradient covariance matrix.

for a simopt `Problem` instance. The script estimates gradients via
finite differences (central or forward) and visualizes the spectrum and
the cumulative variance explained so you can inspect the effective
dimension of the problem.

Usage example:
  python scripts/Eiegnvalue_Spectra.py --problem ROSENBROCK-1 --dim 50 \
      --n-grad-samples 200 --n-reps 5 --fd-eps 1e-5 --out rosen.png
"""

from __future__ import annotations

import argparse
import os
import sys

# Allow importing local modules (project root)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # noqa: PTH100, PTH118, PTH120

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from demo.pickle_files_journal_paper import scale_dimension
from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.problem import Problem, Solution

# Matplotlib / seaborn defaults
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


class GradientSpectrumAnalyzer:
    """Estimate gradient covariance and plot eigenvalue spectrum."""

    def __init__(
        self, problem: Problem, fd_epsilon: float = 1e-5, central: bool = True
    ) -> None:
        """Initialize instance."""
        self.problem = problem
        self.dim = problem.dim
        self.fd_epsilon = float(fd_epsilon)
        self.central = bool(central)

    def _eval_at(self, x: tuple, n_reps: int, rng_list: list[MRG32k3a]) -> float:
        sol = Solution(x, self.problem)
        sol.attach_rngs(rng_list, copy=True)
        self.problem.simulate(sol, num_macroreps=n_reps)
        val = np.atleast_1d(sol.objectives_mean)
        return float(val[0])

    def estimate_gradient(self, x: tuple, n_reps: int = 1) -> np.ndarray:
        """Estimate gradient at `x` using finite differences.

        Uses central differences by default; forward differences if
        `self.central` is False.
        """
        x_arr = np.asarray(x, dtype=float)
        grad = np.zeros(self.dim, dtype=float)

        # Pre-generate RNGs for each evaluation to avoid sharing state
        base_rngs = [MRG32k3a() for _ in range(self.problem.model.n_rngs)]

        for i in range(self.dim):
            eps = self.fd_epsilon
            x_p = x_arr.copy()
            x_p[i] += eps

            if self.central:
                x_m = x_arr.copy()
                x_m[i] -= eps

                f_p = self._eval_at(tuple(x_p), n_reps, base_rngs)
                f_m = self._eval_at(tuple(x_m), n_reps, base_rngs)
                grad[i] = (f_p - f_m) / (2 * eps)
            else:
                f0 = self._eval_at(tuple(x_arr), n_reps, base_rngs)
                f_p = self._eval_at(tuple(x_p), n_reps, base_rngs)
                grad[i] = (f_p - f0) / eps

        return grad

    def collect_gradient_samples(
        self,
        n_samples: int = 100,
        n_reps: int = 1,
    ) -> np.ndarray:
        """Collect gradients at multiple design points.

        Samples `n_samples` design points and returns an array of shape
        (n_samples, dim) where each row is the finite-difference gradient at
        the sampled point.

        Args:
            problem: the problem instance.
            n_samples: number of design points to sample.
            n_reps: macroreplications per function evaluation.
        """
        # center = np.asarray(x, dtype=float)
        # lb = np.asarray(self.problem.lower_bounds, dtype=float)
        # ub = np.asarray(self.problem.upper_bounds, dtype=float)

        # def _sample_in_ball(center: np.ndarray, radius: float, rng:
        # np.random.Generator) -> np.ndarray:
        #     d = center.size
        #     if radius <= 0:
        #         return center.copy()
        #     v = rng.normal(size=d)
        #     norm = np.linalg.norm(v)
        #     if norm == 0:
        #         v = np.ones(d)
        #         norm = np.linalg.norm(v)
        #     u = v / norm
        #     r = radius * (rng.random() ** (1.0 / d))
        #     return center + u * r

        np.random.default_rng()
        grads = np.zeros((n_samples, self.dim), dtype=float)
        # grads[0, :] = self.estimate_gradient(x, n_reps=n_reps)

        for k in range(n_samples):
            # if radius > 0.0:
            #     xk = _sample_in_ball(center, radius, rng_np)
            #     xk = np.minimum(np.maximum(xk, lb), ub)
            #     xk_t = tuple(xk.tolist())
            # else:
            xk_t = tuple(self.problem.get_random_solution(MRG32k3a()))

            grads[k, :] = self.estimate_gradient(xk_t, n_reps=n_reps)

        return grads

    @staticmethod
    def covariance_from_samples(grads: np.ndarray) -> np.ndarray:
        """Covariance from samples."""
        num_samples = grads.shape[0]
        sum_grads = []
        for grad in grads:  # grad has shape (dim,)
            if np.any(np.isnan(grad)):
                raise ValueError(
                    "NaN detected in gradient samples; cannot compute covariance."
                )
            grad = grad.reshape(-1, 1)  # shape (dim, 1)
            item_in_sum = grad @ grad.T  # shape (dim, dim)
            sum_grads.append(item_in_sum)
        return np.sum(sum_grads, axis=0) / num_samples  # shape (dim, dim)

    @staticmethod
    def spectrum_from_cov(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Spectrum from cov."""
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        return eigvals[idx], eigvecs[:, idx]


def parse_args() -> argparse.Namespace:
    """Parse args."""
    p = argparse.ArgumentParser(
        description="Plot eigenvalue spectrum of gradient covariance for a simopt problem"  # noqa: E501
    )
    p.add_argument(
        "--problem",
        required=False,
        help="Problem name (e.g., ROSENBROCK-1). Use --problems for multiple, comma-separated",  # noqa: E501
    )
    p.add_argument(
        "--problems",
        required=False,
        help="Comma-separated list of problems to plot on the same figure",
    )
    p.add_argument("--dim", type=int, required=True, help="Problem decision dimension")
    p.add_argument(
        "--budget", type=int, default=1000, help="Budget passed to scale_dimension"
    )
    p.add_argument(
        "--n-grad-samples",
        type=int,
        default=200,
        help="Number of gradient samples to collect",
    )
    p.add_argument(
        "--n-reps",
        type=int,
        default=5,
        help="Macroreplications per function evaluation",
    )
    p.add_argument(
        "--fd-eps", type=float, default=1e-5, help="Finite-difference epsilon"
    )
    p.add_argument(
        "--central",
        action="store_true",
        help="Use central differences (default: False)",
    )
    p.add_argument(
        "--n-points",
        type=int,
        default=1,
        help="Number of random feasible points to sample",
    )
    # p.add_argument("--delta", type=float, default=0.0, help="If >0, sample points in a
    # ball of this radius around center")
    p.add_argument("--out", default=None, help="Output filename for the figure (PNG)")
    return p.parse_args()


def main() -> None:
    """Run main entry point."""
    args = parse_args()
    # Determine list of problems to run (support --problems comma-separated)
    if getattr(args, "problems", None):
        problem_names = [p.strip() for p in args.problems.split(",") if p.strip()]
    elif getattr(args, "problem", None):
        problem_names = [args.problem]
    else:
        raise SystemExit("Provide --problem or --problems")

    results = []

    # We'll compute aggregate eigenvalues for each problem and store plotting info
    for pname in problem_names:
        print(f"Instantiating problem {pname} dim={args.dim} budget={args.budget}")
        problem = scale_dimension(pname, args.dim, args.budget)
        rng_list = [MRG32k3a() for _ in range(problem.model.n_rngs)]
        problem.attach_rngs(rng_list)

        analyzer = GradientSpectrumAnalyzer(
            problem, fd_epsilon=args.fd_eps, central=args.central
        )

        # choose a test point center
        # if getattr(problem, 'optimal_solution', None) is not None:
        #     x0 = tuple(problem.optimal_solution)
        # else:
        #     x0 = tuple(problem.get_random_solution(rng_list[0]))

        n_points = max(1, int(getattr(args, "n_points", 1)))
        print(f"Sampling gradients at {n_points} random feasible points for {pname}...")
        # If delta>0, sample points uniformly in a ball of radius delta around center x0
        # delta = float(getattr(args, 'delta', 0.0))
        # lb = np.asarray(problem.lower_bounds, dtype=float)
        # ub = np.asarray(problem.upper_bounds, dtype=float)

        # # determine center for ball sampling; ensure it matches problem.dim
        # center_base = np.asarray(x0, dtype=float)
        # if delta > 0.0 and center_base.size != problem.dim:
        # center_base = np.asarray(problem.get_random_solution(rng_list[0]),
        # dtype=float)

        # Sample gradients at multiple design points within the trust region
        # center = center_base
        print(f"{pname}: sampling {n_points} design points in the feasible region")
        grads = analyzer.collect_gradient_samples(
            n_samples=n_points, n_reps=args.n_reps
        )

        # grads is (n_points, dim); compute covariance across rows (samples)
        cov_agg = analyzer.covariance_from_samples(grads)
        eig_agg, _ = analyzer.spectrum_from_cov(cov_agg)

        # compute cumulative variance and statistics
        cumvar = np.cumsum(eig_agg) / np.sum(eig_agg)
        comp90 = int(np.argmax(cumvar >= 0.90) + 1)
        comp95 = int(np.argmax(cumvar >= 0.95) + 1)

        # The effective rank is given by the dimension (given by the index + 1) at which
        # the spectral gap occurs in the eigenvalue spectrum.
        # Compute the spectral gap as the ratio of consecutive eigenvalues
        eig_agg_log = np.log(eig_agg + 1e-12)  # add small value to avoid log(0)
        # spectral_gap = eig_agg_log[:-1] / (eig_agg_log[1:] + 1e-12) # add small value
        # to avoid division by zero
        spectral_gap = np.diff(
            eig_agg_log
        )  # alternative: difference in log-eigenvalues
        eff_rank = int(np.argmax(spectral_gap))

        results.append(
            {
                "name": pname,
                "eig": eig_agg,
                "cumvar": cumvar,
                "comp90": comp90,
                "comp95": comp95,
                "eff_rank": eff_rank,
            }
        )

    # Plot single combined semilogy plot for all problems
    _fig, ax = plt.subplots(figsize=(12, 6))
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, res in enumerate(results):
        color = prop_cycle[i % len(prop_cycle)]
        eigvals = np.asarray(res["eig"], dtype=float)
        n = eigvals.size
        x = np.arange(1, n + 1)

        # floor non-positive eigenvalues to a tiny positive value so semilogy can plot
        # them
        tiny = np.nextafter(0.0, 1.0)
        eig_plot = np.where(eigvals > 0, eigvals, tiny)

        eig_plot_log = np.log(eig_plot)

        ax.plot(
            x, eig_plot_log, marker="o", markersize=4, label=res["name"], color=color
        )
        # vertical lines: 90% (dashed), 95% (dotted), effective rank (dash-dot)
        # ax.axvline(res['comp90'], color=color, linestyle='--', alpha=0.8)
        # ax.axvline(res['comp95'], color=color, linestyle=':', alpha=0.8)
        ax.axvline(
            res["eff_rank"],
            color=color,
            linestyle="--",
            alpha=0.9,
            label=f"{res['name']} Effective Dimension",
        )

    ax.set_xlabel("Decision Vector Component")
    ax.set_ylabel("Eigenvalue (log scale)")
    ax.set_title("")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    out = args.out or f"spectrum_multiproblems_d{args.dim}.png"
    out = os.path.abspath(out)  # noqa: PTH100
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved combined spectrum figure to: {out}")
    plt.show()


if __name__ == "__main__":
    main()
