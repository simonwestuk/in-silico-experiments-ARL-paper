"""Property-based determinism tests for the ARL controller.

Run: python3 tests/test_determinism.py   (exits non-zero on failure)

Checks the determinism contract from the paper: for a fixed routine-set version,
identical (state, inputs) yield an identical routine path, decision, next state,
and content hash -- including after a JSON round-trip of the persisted state
(replay determinism).
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from xelra.dsl import load_routine_set
from xelra.controller import ARLController
from xelra.state import ControllerState, MODES

ROUTINES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "routines", "x_elra.yaml")


def random_perception(rng):
    return {
        "lowest_mastery": float(rng.random()),
        "highest_mastery": float(rng.random()),
        "mean_mastery": float(rng.random()),
        "has_mastery": bool(rng.random() > 0.2),
        "mastery_count": int(rng.integers(0, 10)),
        "impressions": int(rng.integers(0, 30)),
        "clicks_14d": int(rng.integers(0, 12)),
        "days_since_engagement": float(rng.random() * 30),
        "progress_rate": float(rng.random() * 0.12),
        "recent_completions": int(rng.integers(0, 3)),
        "active_goals": bool(rng.random() > 0.3),
        "confusion_flag": bool(rng.random() > 0.7),
        "frustration_flag": bool(rng.random() > 0.7),
        "feature_gap": int(rng.integers(0, 5)),
    }


def random_state(rng):
    return ControllerState(
        mode=MODES[int(rng.integers(0, len(MODES)))],
        mode_entered_at=float(rng.random() * 50),
        interventions_remaining=int(rng.integers(0, 6)),
        suggestions_remaining=int(rng.integers(0, 11)),
        cooldowns={rid: float(rng.random() * 50) for rid in ("P3", "P8")
                   if rng.random() > 0.5},
        last_transition_times=sorted(float(rng.random() * 50)
                                     for _ in range(int(rng.integers(0, 5)))),
        last_budget_reset=float(rng.random() * 50),
    )


def main():
    rs = load_routine_set(ROUTINES)
    ctrl = ARLController(rs)
    rng = np.random.default_rng(12345)
    n = 3000
    for _ in range(n):
        s = random_state(rng)
        perc = random_perception(rng)
        cands = [{"action_id": f"a{k}", "score": float(rng.random()),
                  "source": "s", "objective": "pedagogy"}
                 for k in range(int(rng.integers(0, 12)))]
        t = float(rng.random() * 60 + 50)

        d1, t1, ns1 = ctrl.decide(s, perc, cands, t)
        d2, t2, ns2 = ctrl.decide(s, perc, cands, t)
        assert t1.trace_id == t2.trace_id, "non-deterministic trace hash"
        assert d1 == d2, "non-deterministic decision"
        assert ns1.to_replay() == ns2.to_replay(), "non-deterministic next state"

        # Replay from a JSON round-trip of the persisted state.
        persisted = json.loads(json.dumps(s.to_replay()))
        s_rt = ControllerState.from_replay(persisted)
        d3, t3, _ = ctrl.decide(s_rt, perc, cands, t)
        assert t3.trace_id == t1.trace_id, "replay hash mismatch"

        # Candidate order-invariance.
        shuffled = list(cands)
        rng.shuffle(shuffled)
        _, t4, _ = ctrl.decide(s, perc, shuffled, t)
        assert t4.trace_id == t1.trace_id, "decision not invariant to candidate order"

    print(f"PASS test_determinism: {n} random cases, determinism + replay + "
          f"order-invariance all hold.")


if __name__ == "__main__":
    main()
