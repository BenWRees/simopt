#!/usr/bin/env python3
"""Evaluate adaptive subspace-d rule against oracle on synthetic trials.

Creates synthetic eigen-spectra, simulates validation error as a function
of projected residual (plus noise), and compares the adaptive rule's
chosen `d` to the oracle (d with minimum validation error).
"""

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from collections import Counter

import numpy as np


class DummyProblem:  # noqa: D101
    def __init__(self, dim: int, budget: float = 1000.0) -> None:  # noqa: D107
        self.dim = int(dim)
        # budget in problem.factors used to scale cost penalty
        self.factors = {"budget": float(budget)}


def make_prev_info(recs):  # noqa: ANN001, ANN201, D103
    info = []
    for eigvals, drec, succ in recs:
        info.append(
            {
                "eigenvalue_spectrum": list(np.asarray(eigvals, dtype=float)),
                "recommended_dimension": int(drec),
                "model_success": bool(succ),
                "validation_by_d": {1: 0.5, 2: 0.2} if succ else {1: 0.6, 2: 0.4},
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
        # copy of function used earlier in demo/test_adaptive_d.py
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

            # cost_penalty will be injected from the outer scope when running
            # experiments
            cp = getattr(self, "cost_penalty", 0.1)
            # budget-aware multiplier: lower budget -> larger effective penalty
            baseline_budget = 1000.0
            budget = getattr(self.problem, "factors", {}).get("budget", baseline_budget)
            try:
                budget = float(budget)
            except Exception:
                budget = baseline_budget
            mult = baseline_budget / max(1.0, budget)
            cp_eff = float(cp) * float(mult)
            raw_score = (
                w_val * s_val + w_proj * s_proj + w_succ * s_succ - cp_eff * cost
            )
            score = max(0.0, min(1.0, raw_score))
            metrics["score"] = float(score)
            metrics["s_val"] = float(s_val)
            metrics["s_proj"] = float(s_proj)
            metrics["s_succ"] = float(s_succ)
            metrics["cost"] = float(cost)

        return results


def make_spectrum(dim, kind="dominant", k=2, decay=0.6):  # noqa: ANN001, ANN201, D103
    if kind == "dominant":
        vals = np.concatenate([np.linspace(5.0, 1.0, k), np.ones(dim - k) * 0.1])
    elif kind == "exp":
        vals = np.exp(-np.arange(dim) * decay)
    else:
        vals = np.ones(dim)
    return np.asarray(vals, dtype=float)


def make_validation_from_spectrum(eig, noise=0.01, base=0.05, alpha=1.0):  # noqa: ANN001, ANN201, D103
    # Build validation error per d proportional to projection residual plus noise
    total = max(1e-12, float(np.sum(eig)))
    vals = {}
    dim = eig.shape[0]
    for d in range(1, dim):
        captured = float(np.sum(eig[:d]))
        proj_resid = max(0.0, 1.0 - (captured / total))
        val = base + alpha * proj_resid + random.gauss(0.0, noise)
        vals[d] = max(0.0, float(val))
    return vals


def run_experiments(n_trials=200, dim=10) -> None:  # noqa: ANN001, D103
    random.seed(1)
    np.random.seed(1)
    solver = LocalSolver(DummyProblem(dim=dim))

    exact = 0
    within1 = 0
    regrets = []
    chosen_hist = Counter()
    oracle_hist = Counter()

    for _t in range(n_trials):
        kind = random.choice(["dominant", "exp", "flat"])
        if kind == "dominant":
            k = random.randint(1, min(4, dim - 1))
            eig = make_spectrum(dim, kind="dominant", k=k)
        elif kind == "exp":
            decay = random.uniform(0.1, 1.0)
            eig = make_spectrum(dim, kind="exp", decay=decay)
        else:
            eig = make_spectrum(dim, kind="flat")

        val_by_d = make_validation_from_spectrum(eig, noise=0.02, base=0.02, alpha=1.0)

        # oracle
        oracle_d = min(val_by_d.items(), key=lambda kv: kv[1])[0]

        # configure solver
        solver.gradient_eigenvalues = [eig]
        solver.last_validation_by_d = val_by_d
        solver.previous_model_information = []

        scores = solver.evaluate_and_score_candidate_dimensions()
        chosen_d = max(scores.items(), key=lambda kv: kv[1]["score"])[0]

        chosen_hist[chosen_d] += 1
        oracle_hist[oracle_d] += 1

        if chosen_d == oracle_d:
            exact += 1
        if abs(chosen_d - oracle_d) <= 1:
            within1 += 1

        regret = val_by_d[chosen_d] - val_by_d[oracle_d]
        regrets.append(regret)

    n = n_trials
    print(f"Trials: {n}")
    print(f"Exact match rate: {exact}/{n} = {exact / n:.3f}")
    print(f"Within-1 rate: {within1}/{n} = {within1 / n:.3f}")
    print(f"Mean regret (val_err chosen - val_err oracle): {np.mean(regrets):.5f}")
    print(f"Median regret: {np.median(regrets):.5f}")
    print("Top chosen d counts:")
    for d, c in chosen_hist.most_common(10):
        print(f"  d={d}: {c}")


if __name__ == "__main__":
    run_experiments(n_trials=200, dim=10)
