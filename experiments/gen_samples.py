"""Generate genuine sample decision traces (JSON + PROV-O) for the repo/appendix.

Runs the controller on crafted scenarios and persists the resulting traces:
  - sample_trace_struggling.json : affective-overload intervention (P8 executes)
  - sample_trace_noaction.json   : deliberate non-intervention (all routines blocked)
  - sample_trace.prov.json       : PROV-O projection of the struggling trace
"""

from __future__ import annotations

import json
import os

from common import ROUTINES, SAMPLE_DIR
from xelra.dsl import load_routine_set
from xelra.controller import ARLController
from xelra.state import ControllerState


def main():
    rs = load_routine_set(ROUTINES)
    ctrl = ARLController(rs)

    cands = [{"action_id": f"py101_res_{k:02d}", "score": round(0.9 - 0.05 * k, 3),
              "source": "hybrid_recommender", "objective": "pedagogy"} for k in range(10)]

    # Scenario 1: STRUGGLING with active affect -> P8 affective-overload intervention.
    state = ControllerState(mode="STRUGGLING", mode_entered_at=0.0,
                            interventions_remaining=4, suggestions_remaining=8,
                            cooldowns={"P6": 5.0}, last_budget_reset=0.0)
    perc = {"lowest_mastery": 0.1, "highest_mastery": 0.85, "mean_mastery": 0.45,
            "has_mastery": True, "mastery_count": 7, "impressions": 40, "clicks_14d": 4,
            "days_since_engagement": 0.5, "progress_rate": 0.02, "recent_completions": 0,
            "active_goals": True, "confusion_flag": True, "frustration_flag": True,
            "feature_gap": 0}
    dec, trace, _ = ctrl.decide(state, perc, cands, 30.0, learner_id="std_demo_0001")
    with open(os.path.join(SAMPLE_DIR, "sample_trace_struggling.json"), "w") as fh:
        json.dump(trace.to_dict(), fh, indent=2)
    with open(os.path.join(SAMPLE_DIR, "sample_trace.prov.json"), "w") as fh:
        json.dump(trace.to_prov(), fh, indent=2)

    # Scenario 2: STRUGGLING but P3/P8 exhausted/cooled-down -> deliberate non-action
    # resolved by the totality safeguard (default recommendation, not a bounded
    # intervention); the trace still records the full evaluation path.
    state2 = ControllerState(mode="STRUGGLING", mode_entered_at=0.0,
                             interventions_remaining=0, suggestions_remaining=0,
                             cooldowns={"P3": 29.0, "P8": 29.0}, last_budget_reset=20.0)
    perc2 = dict(perc)
    dec2, trace2, _ = ctrl.decide(state2, perc2, cands, 30.0, learner_id="std_demo_0002")
    with open(os.path.join(SAMPLE_DIR, "sample_trace_noaction.json"), "w") as fh:
        json.dump(trace2.to_dict(), fh, indent=2)

    print("Sample traces written to", SAMPLE_DIR)
    print("  scenario 1 decision:", dec["action"], "via", dec["source_routine"],
          "| trace_id", trace.trace_id[:16])
    print("  scenario 2 decision:", dec2["action"], "via", dec2["source_routine"],
          "(bounded_intervention=%s)" % dec2["is_bounded_intervention"],
          "| trace_id", trace2.trace_id[:16])
    # Show the routine path of scenario 2 (records blocked routines + reasons).
    print("  scenario 2 routine path:",
          [(o["routine_id"], o["outcome"], o["reason"]) for o in trace2.to_dict()["routine_path"]])


if __name__ == "__main__":
    main()
