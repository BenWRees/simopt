#!/usr/bin/env python3
"""Run a sweep of cost-penalty values for adaptive-d evaluation."""

import importlib.util
import os
from collections import Counter

import numpy as np

# load eval_adaptive_d module by path
spec = importlib.util.spec_from_file_location(
    "eval_adaptive_d",
    os.path.join(os.path.dirname(__file__), "eval_adaptive_d.py"),  # noqa: PTH118, PTH120
)
assert spec is not None
assert spec.loader is not None

e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)


def run_once(n_trials, dim, cost_penalty):  # noqa: ANN001, ANN201, D103
    import random

    random.seed(1)
    np.random.seed(1)
    solver = e.LocalSolver(e.DummyProblem(dim=dim))
    solver.cost_penalty = cost_penalty
    exact = 0
    within1 = 0
    regrets = []
    chosen_hist = Counter()
    for _t in range(n_trials):
        kind = random.choice(["dominant", "exp", "flat"])
        if kind == "dominant":
            k = random.randint(1, min(4, dim - 1))
            eig = e.make_spectrum(dim, kind="dominant", k=k)
        elif kind == "exp":
            decay = random.uniform(0.1, 1.0)
            eig = e.make_spectrum(dim, kind="exp", decay=decay)
        else:
            eig = e.make_spectrum(dim, kind="flat")

        val_by_d = e.make_validation_from_spectrum(
            eig, noise=0.02, base=0.02, alpha=1.0
        )
        oracle_d = min(val_by_d.items(), key=lambda kv: kv[1])[0]

        solver.gradient_eigenvalues = [eig]
        solver.last_validation_by_d = val_by_d
        solver.previous_model_information = []

        scores = solver.evaluate_and_score_candidate_dimensions()
        chosen_d = max(scores.items(), key=lambda kv: kv[1]["score"])[0]

        chosen_hist[chosen_d] += 1
        if chosen_d == oracle_d:
            exact += 1
        if abs(chosen_d - oracle_d) <= 1:
            within1 += 1
        regrets.append(val_by_d[chosen_d] - val_by_d[oracle_d])

    return exact / n_trials, within1 / n_trials, float(np.mean(regrets)), chosen_hist


def main() -> None:  # noqa: D103
    penalties = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
    for p in penalties:
        exact, w1, mr, hist = run_once(n_trials=200, dim=10, cost_penalty=p)
        top = sorted(hist.items(), key=lambda kv: -kv[1])[:5]
        print(
            f"penalty={p}: exact={exact:.3f}, within1={w1:.3f}, mean_regret={mr:.5f}, top_choices={top}"  # noqa: E501
        )


if __name__ == "__main__":
    main()
