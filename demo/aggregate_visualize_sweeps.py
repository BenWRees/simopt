#!/usr/bin/env python3
"""Aggregate and visualize coarse + high-fidelity sweep results.

Saves CSV summaries and basic plots under `demo/sweep_results/<timestamp>/`.
"""
import json
import os
from pathlib import Path
import datetime
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
COARSE_FILES = [
    ROOT / 'sweep_results_coarse_dim20.json',
    ROOT / 'sweep_results_coarse_dim50.json',
    ROOT / 'sweep_results_coarse_dim100.json',
]
HF_DIR = ROOT / 'high_fidelity_results'
OUT_BASE = ROOT / 'sweep_results'

def load_json_list(path):
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def aggregate_coarse():
    rows = []
    for p in COARSE_FILES:
        data = load_json_list(p)
        if not data:
            continue
        dim = int(p.stem.split('dim')[-1])
        for e in data:
            row = {
                'dim': dim,
                'budget': int(e.get('budget', 0)),
                'cp': float(e.get('cp', 0.0)),
                'power': float(e.get('power', 0.0)),
                'alpha': float(e.get('alpha', 0.0)),
                'exact': int(e.get('exact', 0)),
                'within1': int(e.get('within1', 0)),
                'mean_regret': float(e.get('mean_regret', 0.0)),
                'median_regret': float(e.get('median_regret', 0.0)),
            }
            rows.append(row)
    return rows


def aggregate_hf():
    rows = []
    if not HF_DIR.exists():
        return rows
    for p in HF_DIR.glob('hf_results_dim*.json'):
        data = load_json_list(p)
        dim = int(p.stem.split('dim')[-1])
        for e in data:
            row = {
                'dim': dim,
                'budget': int(e.get('budget', 0)),
                'cp': float(e.get('cp', 0.0)),
                'power': float(e.get('power', 0.0)),
                'alpha': float(e.get('alpha', 0.0)),
                'exact': int(e.get('exact', 0)),
                'within1': int(e.get('within1', 0)),
                'mean_regret': float(e.get('mean_regret', 0.0)),
                'median_regret': float(e.get('median_regret', 0.0)),
            }
            rows.append(row)
    return rows


def write_csv(rows, out_path):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def plot_by_dim(rows, out_dir, prefix):
    # For each dim, plot mean_regret vs cp (colored by budget)
    from collections import defaultdict
    by_dim = defaultdict(list)
    for r in rows:
        by_dim[r['dim']].append(r)

    for dim, rs in by_dim.items():
        fig, ax = plt.subplots(figsize=(6,4))
        budgets = [r['budget'] for r in rs]
        cps = [r['cp'] for r in rs]
        mean_regrets = [r['mean_regret'] for r in rs]
        sizes = [max(8, r['within1']) for r in rs]
        sc = ax.scatter(cps, mean_regrets, c=budgets, s=[(s/2)**1.5 for s in sizes], cmap='viridis')
        ax.set_xlabel('cp')
        ax.set_ylabel('mean_regret')
        ax.set_title(f'dim={dim} {prefix}')
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('budget')
        out = out_dir / f'{prefix}_dim{dim}_cp_vs_meanregret.png'
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)


def run():
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = OUT_BASE / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    coarse_rows = aggregate_coarse()
    hf_rows = aggregate_hf()

    write_csv(coarse_rows, out_dir / 'coarse_summary.csv')
    write_csv(hf_rows, out_dir / 'hf_summary.csv')

    plot_by_dim(coarse_rows, out_dir, 'coarse')
    plot_by_dim(hf_rows, out_dir, 'hf')

    print('Wrote outputs to', out_dir)

if __name__ == '__main__':
    run()
