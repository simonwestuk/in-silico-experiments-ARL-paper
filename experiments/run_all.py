"""Reproduce the entire X-ELRA evaluation: tests, model check, studies, samples.

Run: python3 experiments/run_all.py
Outputs land in results/ (JSON), sample_traces/ (JSON), and figures/ (after
running make_figures.py).
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS)
sys.path.insert(0, ROOT)
sys.path.insert(0, THIS)


def sh(cmd):
    print(f"\n$ {' '.join(cmd) if isinstance(cmd, (list, tuple)) else cmd}")
    subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    # Verification first.
    sh([sys.executable, "tests/test_determinism.py"])
    sh([sys.executable, "tests/test_boundedness.py"])
    sh([sys.executable, "tests/verify_properties.py"])
    # Studies.
    for mod in (
        "study1_oscillation",
        "study2_audit",
        "study3_sensitivity",
        "study4_robustness",
        "study5_latency_fairness",
        "study6_fairness",
        "gen_samples",
    ):
        print(f"\n===== {mod} =====")
        m = importlib.import_module(mod)
        m.main()
    print("\nAll studies complete. Run experiments/make_figures.py to (re)generate figures.")


if __name__ == "__main__":
    main()
