#!/usr/bin/env python3
"""Clean standalone test for adaptive subspace scoring (fixed copy)."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np


class DummyProblem:  # noqa: D101
    def __init__(self, dim: int) -> None:  # noqa: D107
        self.dim = int(dim)
        self.lower_bounds = tuple([-5.0] * self.dim)
        self.upper_bounds = tuple([5.0] * self.dim)
        self.factors = {"budget": 1000}


def make_prev_info(recs):  # noqa: ANN001, ANN201, D103
    info = []
    for eigvals, drec, succ in recs:
        info.append(
            {
                "eigenvalue_spectrum": list(np.asarray(eigvals, dtype=float)),
                "recommended_dimension": int(drec),
                "model_success": bool(succ),
                "validation_by_d": {1: 0.5, 2: 0.2, 3: 0.15, 4: 0.12}
                if succ
                else {1: 0.6, 2: 0.4},
            }
        )
    return info


class LocalSolver:  # noqa: D101
    def __init__(self, problem: DummyProblem) -> None:  # noqa: D107
        self.problem = problem
        self.max_d = max(1, problem.dim - 1)
        self.previous_model_information = []
        self.gradient_eigenvalues = []
        self.last_validation_by_d = None

    def evaluate_and_score_candidate_dimensions(self):  # noqa: ANN201, D102
        results = {}
        max_test_d = min(self.max_d, self.problem.dim - 1)
        if max_test_d < 1:
            return {1: {"score": 0.0}}

        validation_by_d = getattr(self, "last_validation_by_d", None)
        if validation_by_d is None:
            vals = {}
            counts = {}
            for info in self.previous_model_information:
                vbd = info.get("validation_by_d")
                if isinstance(vbd, dict):
                    for k, v in vbd.items():
                        k = int(k)
                        vals[k] = vals.get(k, 0.0) + float(v)
                        counts[k] = counts.get(k, 0) + 1
            if counts:
                validation_by_d = {k: vals[k] / counts[k] for k in vals}

        eig_source = None
        if hasattr(self, "gradient_eigenvalues") and self.gradient_eigenvalues:
            try:
                eig_source = np.array(self.gradient_eigenvalues[-1], dtype=float)
            except Exception:
                eig_source = None

        if eig_source is None:
            spectra = [
                np.array(info.get("eigenvalue_spectrum", []), dtype=float)
                for info in self.previous_model_information
                if info.get("eigenvalue_spectrum")
            ]
            if spectra:
                maxlen = max(s.shape[0] for s in spectra)
                padded = np.vstack(
                    [
                        np.pad(s, (0, maxlen - s.shape[0]), constant_values=0.0)
                        for s in spectra
                    ]
                )
                eig_source = np.mean(padded, axis=0)

        if eig_source is None or eig_source.size == 0:
            eig_source = np.ones(self.problem.dim) * 1e-6

        total_var = max(1e-12, float(np.sum(eig_source)))

        success_counts = {}
        total_counts = {}
        for info in self.previous_model_information:
            d_rec = int(info.get("recommended_dimension", 1))
            total_counts[d_rec] = total_counts.get(d_rec, 0) + 1
            if info.get("model_success"):
                success_counts[d_rec] = success_counts.get(d_rec, 0) + 1

        for d in range(1, max_test_d + 1):
            val_err = None
            if validation_by_d and int(d) in validation_by_d:
                try:
                    val_err = float(validation_by_d[int(d)])
                except Exception:
                    val_err = None

            captured = float(np.sum(eig_source[:d]))
            proj_resid = max(0.0, 1.0 - (captured / total_var))

            succ = success_counts.get(d, 0)
            tot = total_counts.get(d, 0)
            success_rate = float(succ / tot) if tot > 0 else 0.0

            results[d] = {
                "val_err": val_err,
                "proj_resid": proj_resid,
                "success_rate": success_rate,
            }

        vals = [v["val_err"] for v in results.values() if v["val_err"] is not None]
        if vals:
            max_val = max(vals)
            min_val = min(vals)
        else:
            max_val = None
            min_val = None

        w_val = 0.50
        w_proj = 0.30
        w_succ = 0.20

        for d, metrics in results.items():
            if metrics["val_err"] is None:
                s_val = 0.5
            else:
                if max_val is None or max_val - min_val < 1e-12:
                    s_val = 0.8
                else:
                    s_val = 1.0 - (metrics["val_err"] - min_val) / max(
                        1e-12, (max_val - min_val)
                    )

            s_proj = 1.0 - metrics["proj_resid"]
            s_succ = metrics["success_rate"]

            cost = float(d) / max(1.0, max_test_d)

            raw_score = w_val * s_val + w_proj * s_proj + w_succ * s_succ - 0.1 * cost
            score = max(0.0, min(1.0, raw_score))
            metrics["score"] = float(score)
            metrics["s_val"] = float(s_val)
            metrics["s_proj"] = float(s_proj)
            metrics["s_succ"] = float(s_succ)
            metrics["cost"] = float(cost)

        return results


def run_tests_with_solver(solver, problem) -> None:  # noqa: ANN001, D103
    solver.problem = problem
    solver.max_d = problem.dim - 1
    solver.previous_model_information = []

    eig = np.array([5.0, 4.0] + [0.1] * (problem.dim - 2))
    solver.gradient_eigenvalues = [eig]
    solver.last_validation_by_d = {1: 0.4, 2: 0.08, 3: 0.07, 4: 0.06}
    solver.previous_model_information = make_prev_info(
        [
            (eig, 2, True),
            (eig * 0.9, 2, True),
        ]
    )

    scores = solver.evaluate_and_score_candidate_dimensions()
    print("Scores (case1) (d -> score):")
    for d in sorted(scores.keys()):
        print(d, scores[d]["score"], scores[d])
    best = max(scores.items(), key=lambda kv: kv[1]["score"])
    print("Best d (case1):", best)

    eig2 = np.ones(problem.dim)
    solver.gradient_eigenvalues = [eig2]
    solver.last_validation_by_d = {1: 0.12, 2: 0.11, 3: 0.10, 4: 0.09, 5: 0.09}
    solver.previous_model_information = make_prev_info(
        [
            (eig2, 3, False),
            (eig2, 4, False),
        ]
    )
    scores2 = solver.evaluate_and_score_candidate_dimensions()
    print("\nScores (case2) (d -> score):")
    for d in sorted(scores2.keys()):
        print(d, scores2[d]["score"], scores2[d])
    best2 = max(scores2.items(), key=lambda kv: kv[1]["score"])
    print("Best d (case2):", best2)


def main() -> None:  # noqa: D103
    problem = DummyProblem(dim=10)
    solver = LocalSolver(problem)
    run_tests_with_solver(solver, problem)


if __name__ == "__main__":
    main()
