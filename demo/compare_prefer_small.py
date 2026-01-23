#!/usr/bin/env python3
"""Compare ASTROMoRF scoring policy with and without preferring small sufficient d.

Runs synthetic trials (dim=100) and prints exact/within-1/mean-regret for both policies.
"""

import random
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from collections import Counter

import simopt.solvers.astromorf as ast


def make_spectrum(dim, kind="dominant", k=2, decay=0.6):
    if kind == "dominant":
        vals = np.concatenate([np.linspace(5.0, 1.0, k), np.ones(dim - k) * 0.1])
    elif kind == "exp":
        vals = np.exp(-np.arange(dim) * decay)
    else:
        vals = np.ones(dim)
    return np.asarray(vals, dtype=float)


def make_validation_from_spectrum(eig, noise=0.01, base=0.05, alpha=1.0):
    total = max(1e-12, float(np.sum(eig)))
    vals = {}
    dim = eig.shape[0]
    for d in range(1, dim):
        captured = float(np.sum(eig[:d]))
        proj_resid = max(0.0, 1.0 - (captured / total))
        val = base + alpha * proj_resid + random.gauss(0.0, noise)
        vals[d] = max(0.0, float(val))
    return vals


def run_compare(n_trials=200, dim=100, budget=1000.0, cp=0.05, power=2.0, alpha=0.5):
    random.seed(1)
    np.random.seed(1)

    func = ast.ASTROMORF.evaluate_and_score_candidate_dimensions

    results = {}
    for prefer_small in (False, True):
        exact = 0
        within1 = 0
        regrets = []
        chosen_hist = Counter()

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

            # build a lightweight solver-like object
            class S:
                pass

            s = S()
            s.problem = type("P", (), {"dim": dim, "factors": {"budget": float(budget)}})()
            s.max_d = max(1, dim - 1)
            s.previous_model_information = []
            s.gradient_eigenvalues = [eig]
            s.last_validation_by_d = val_by_d

            # tuned params
            s.cost_penalty = cp
            s.cost_power = power
            s.budget_alpha = alpha
            s.baseline_budget = 1000.0
            s.prefer_small_sufficient = prefer_small
            s.sufficient_score_tol = 0.02

            scores = func(s)
            chosen_d = max(scores.items(), key=lambda kv: kv[1]["score"])[0]

            chosen_hist[chosen_d] += 1
            if chosen_d == oracle_d:
                exact += 1
            if abs(chosen_d - oracle_d) <= 1:
                within1 += 1
            regrets.append(val_by_d[chosen_d] - val_by_d[oracle_d])

        results[prefer_small] = {
            "exact": exact,
            "within1": within1,
            "mean_regret": float(np.mean(regrets)),
            "chosen_hist": chosen_hist,
        }

        print(f"prefer_small={prefer_small}: exact={exact}/{n_trials} within1={within1}/{n_trials} mean_regret={results[prefer_small]['mean_regret']:.5f}")

    return results


if __name__ == '__main__':
    run_compare(n_trials=200, dim=100, budget=1000.0)
