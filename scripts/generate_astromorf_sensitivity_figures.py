#!/usr/bin/env python
"""Thin CLI wrapper around :mod:`simopt.journal.plotting.cli`.

Run::

    python scripts/generate_astromorf_sensitivity_figures.py \
        --analysis-dir $HOME/results/journal/analysis \
        --output-dir figures/

See ``python scripts/generate_astromorf_sensitivity_figures.py --help``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simopt.journal.plotting.cli import main

if __name__ == "__main__":
    sys.exit(main())
