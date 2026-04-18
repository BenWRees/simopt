"""run_high_fidelity module."""

#!/usr/bin/env python3
import json
import os
from collections import defaultdict

from demo.tune_and_eval_adaptive_d import run_tuning

coarse_files = [
    "demo/sweep_results_coarse_dim20.json",
    "demo/sweep_results_coarse_dim50.json",
    "demo/sweep_results_coarse_dim100.json",
]
all_entries = defaultdict(list)
for f in coarse_files:
    if not os.path.exists(f):  # noqa: PTH110
        continue
    with open(f) as fh:  # noqa: PTH123
        data = json.load(fh)
    dim = int(os.path.splitext(os.path.basename(f))[0].split("dim")[-1])  # noqa: PTH119, PTH122
    for entry in data:
        entry["dim"] = dim
        all_entries[dim].append(entry)

selected = defaultdict(list)
for dim, entries in all_entries.items():
    entries_sorted = sorted(entries, key=lambda e: e.get("mean_regret", 1e9))
    top = entries_sorted[:6]
    # keep unique combos
    seen = set()
    for e in top:
        key = (int(e["budget"]), float(e["cp"]), float(e["power"]), float(e["alpha"]))
        if key in seen:
            continue
        seen.add(key)
        selected[dim].append(key)

out_dir = "demo/high_fidelity_results"
os.makedirs(out_dir, exist_ok=True)  # noqa: PTH103
summary = {}
for dim, combos in selected.items():
    hf_results = []
    for budget, cp, power, alpha in combos:
        print(f"HF run: dim={dim} budget={budget} cp={cp} power={power} alpha={alpha}")
        res = run_tuning(
            n_trials=200,
            dim=dim,
            budgets=[budget],
            cps=[cp],
            powers=[power],
            alphas=[alpha],
        )
        hf_results.extend(res)
    out = os.path.join(out_dir, f"hf_results_dim{dim}.json")  # noqa: PTH118
    with open(out, "w") as fh:  # noqa: PTH123
        json.dump(hf_results, fh)
    summary[dim] = {"file": out, "runs": len(hf_results)}

summary_file = os.path.join(out_dir, "hf_summary.json")  # noqa: PTH118
with open(summary_file, "w") as fh:  # noqa: PTH123
    json.dump(summary, fh)
print("HF sweeps complete. Summary:", summary)
