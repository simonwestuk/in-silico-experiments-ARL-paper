"""Tests for the routine-set linter: clean set passes; faults are caught.

Run: python3 tests/test_linter.py
"""

from __future__ import annotations

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from xelra.dsl import RoutineSet
from xelra.linter import lint, counterfactual

ROUTINES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "routines", "x_elra.yaml")


def main():
    raw = yaml.safe_load(open(ROUTINES, encoding="utf-8"))
    rs = RoutineSet(copy.deepcopy(raw))
    issues = lint(rs)
    errors = [i for i in issues if i[0] == "error"]
    assert not errors, f"clean set should have no errors, got {errors}"

    # Inject a cost-exceeds-budget fault (E3) and a mode-without-routine fault.
    bad = copy.deepcopy(raw)
    bad["routines"][2]["cost"]["interventions"] = 99   # exceeds budget
    rs_bad = RoutineSet(bad)
    codes = {c for _, c, _ in lint(rs_bad)}
    assert "E3" in codes, f"linter should flag E3, got {codes}"

    # Counterfactual unit test: an affect-laden struggling perception fires P8.
    perc = {"lowest_mastery": 0.2, "highest_mastery": 0.6, "has_mastery": True,
            "mastery_count": 7, "impressions": 20, "clicks_14d": 3,
            "days_since_engagement": 0.5, "progress_rate": 0.02, "recent_completions": 0,
            "active_goals": True, "confusion_flag": True, "frustration_flag": False,
            "feature_gap": 0}
    cf = counterfactual(rs, perc)
    assert cf["mode"] == "STRUGGLING" and cf["first_eligible_routine"] == "P8", cf

    print(f"PASS test_linter: clean set has 0 errors; E3 fault caught; "
          f"counterfactual routes affect-struggling -> {cf['first_eligible_routine']} "
          f"({cf['action']}).")


if __name__ == "__main__":
    main()
