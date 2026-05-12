"""HPC-ready Optuna tuner for ASTROMoRF hyperparameters.

Modules
-------
spaces     -- per-problem search spaces, warm-start trials, adaptive narrowing
diagnostics -- post-mrep extraction of solver-internal counters
evaluate   -- single-trial evaluation (n_macroreps mreps -> EvalResult)
tuner      -- Optuna study factory, objective function, top-K persistence
worker     -- Slurm-array worker entry point (one process = one async worker)
confirm    -- top-K confirmation pass with disjoint seed stream
collect    -- export trials/best to JSON/CSV
report     -- generate the human-readable recommendation markdown
smoke      -- local smoke-test entry (tiny budget, sequential)

All paths are resolved from the project root via :func:`project_root`. There
are no hard-coded absolute paths; storage URL is overridable through the
``ASTROMORF_OPTUNA_STORAGE`` environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Resolve the simopt project root from this file's location."""
    return Path(__file__).resolve().parents[2]


def results_root() -> Path:
    """Default location for tuning artefacts; respects ASTROMORF_TUNING_DIR."""
    override = os.environ.get("ASTROMORF_TUNING_DIR")
    return Path(override) if override else project_root() / "results" / "astromorf_tuning"


__all__ = ["project_root", "results_root"]
