"""Shared setup for the X-ELRA simulation studies."""

from __future__ import annotations

import json
import os
import sys

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS)
sys.path.insert(0, ROOT)

ROUTINES = os.path.join(ROOT, "routines", "x_elra.yaml")
RESULTS_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(ROOT, "figures")
SAMPLE_DIR = os.path.join(ROOT, "sample_traces")

for d in (RESULTS_DIR, FIG_DIR, SAMPLE_DIR):
    os.makedirs(d, exist_ok=True)

# Global noise grid and sample sizes used across studies (kept modest so the
# whole suite reproduces in well under a minute on a laptop).
SIGMA_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
N_LEARNERS = 200
N_STEPS = 80
SEED = 20260601


def save_json(name: str, obj) -> str:
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)
    return path
