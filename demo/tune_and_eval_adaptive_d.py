#!/usr/bin/env python3
"""Tune budget-aware cost parameters (cp, power, alpha) and evaluate adaptive-d rule.

This script reuses the synthetic trial generation from `eval_adaptive_d.py` but
implements a local scoring function that supports nonlinear cost shapes and an
alpha exponent on the budget multiplier.
"""

import math
import random
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np
from collections import Counter

from eval_adaptive_d import DummyProblem, make_spectrum, make_validation_from_spectrum


def score_candidates(solver, cp=0.1, power=1.0, alpha=1.0):
    # Reimplemented minimal scoring logic to support `power` and `alpha` parameters.
    results = {}
    max_test_d = min(solver.max_d, solver.problem.dim - 1)
    if max_test_d < 1:
        return {1: {"score": 0.0}}

    validation_by_d = getattr(solver, "last_validation_by_d", None)

    eig_source = None
    if hasattr(solver, "gradient_eigenvalues") and solver.gradient_eigenvalues:
        try:
            eig_source = np.array(solver.gradient_eigenvalues[-1], dtype=float)
        except Exception:
            eig_source = None

    if eig_source is None:
        spectra = [
            np.array(info.get("eigenvalue_spectrum", []), dtype=float)
            for info in solver.previous_model_information
            if info.get("eigenvalue_spectrum")
        ]
        if spectra:
            maxlen = max(s.shape[0] for s in spectra)
            padded = np.vstack([
                np.pad(s, (0, maxlen - s.shape[0]), constant_values=0.0) for s in spectra
            ])
            eig_source = np.mean(padded, axis=0)

    if eig_source is None or eig_source.size == 0:
        eig_source = np.ones(solver.problem.dim) * 1e-6

    total_var = max(1e-12, float(np.sum(eig_source)))

    success_counts = {}
    total_counts = {}
    for info in solver.previous_model_information:
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

    baseline_budget = 1000.0
    budget = getattr(solver.problem, "factors", {}).get("budget", baseline_budget)
    try:
        budget = float(budget)
    except Exception:
        budget = baseline_budget
    mult = baseline_budget / max(1.0, budget)
    mult = float(mult) ** float(alpha)

    for d, metrics in results.items():
        if metrics["val_err"] is None:
            s_val = 0.5
        else:
            if max_val is None or max_val - min_val < 1e-12:
                s_val = 0.8
            else:
                s_val = 1.0 - (metrics["val_err"] - min_val) / max(1e-12, (max_val - min_val))

        s_proj = 1.0 - metrics["proj_resid"]
        s_succ = metrics["success_rate"]

        cost = (float(d) / max(1.0, max_test_d)) ** float(power)

        cp_eff = float(cp) * float(mult)
        raw_score = w_val * s_val + w_proj * s_proj + w_succ * s_succ - cp_eff * cost
        score = max(0.0, min(1.0, raw_score))
        metrics["score"] = float(score)
        metrics["s_val"] = float(s_val)
        metrics["s_proj"] = float(s_proj)
        metrics["s_succ"] = float(s_succ)
        metrics["cost"] = float(cost)

    return results


class LocalSolver:
    def __init__(self, problem: DummyProblem):
        self.problem = problem
        self.max_d = max(1, problem.dim - 1)
        self.previous_model_information = []
        self.gradient_eigenvalues = []
        self.last_validation_by_d = None


def run_tuning(n_trials=200, dim=100, budgets=(100, 500, 1000, 5000),
               cps=(0.05, 0.1, 0.2), powers=(1.0, 2.0), alphas=(0.5, 1.0, 1.5, 2.0)):
    random.seed(1)
    np.random.seed(1)

    results = []

    for budget in budgets:
        for cp in cps:
            for power in powers:
                for alpha in alphas:
                    exact = 0
                    within1 = 0
                    regrets = []
                    chosen_hist = Counter()

                    solver = LocalSolver(DummyProblem(dim=dim, budget=budget))

                    for t in range(n_trials):
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

                        oracle_d = min(val_by_d.items(), key=lambda kv: kv[1])[0]

                        solver.gradient_eigenvalues = [eig]
                        solver.last_validation_by_d = val_by_d
                        solver.previous_model_information = []

                        scores = score_candidates(solver, cp=cp, power=power, alpha=alpha)
                        chosen_d = max(scores.items(), key=lambda kv: kv[1]["score"])[0]

                        chosen_hist[chosen_d] += 1

                        if chosen_d == oracle_d:
                            exact += 1
                        if abs(chosen_d - oracle_d) <= 1:
                            within1 += 1

                        regret = val_by_d[chosen_d] - val_by_d[oracle_d]
                        regrets.append(regret)

                    res = {
                        "budget": budget,
                        "cp": cp,
                        "power": power,
                        "alpha": alpha,
                        "exact": exact,
                        "within1": within1,
                        "mean_regret": float(np.mean(regrets)),
                        "median_regret": float(np.median(regrets)),
                        "top_chosen": chosen_hist.most_common(5),
                    }
                    results.append(res)
                    print(f"budget={budget} cp={cp} power={power} alpha={alpha} "
                          f"exact={exact}/{n_trials} within1={within1}/{n_trials} mean_regret={res['mean_regret']:.5f}")

    return results


if __name__ == '__main__':
    # quick default run
    run_tuning(n_trials=200, dim=100)
