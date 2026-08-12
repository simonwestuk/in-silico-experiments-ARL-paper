"""Property tests for boundedness (safety) and cooldown/oscillation invariants.

Run: python3 tests/test_boundedness.py

Verifies, over many simulated streams across noise levels and archetypes:
  - Safety: bounded interventions in any budget-reset window never exceed the
    configured budget.
  - Cooldown: a routine never executes twice within its cooldown period.
  - Anti-oscillation: stance transitions within the oscillation window stay at or
    below k+1 (the +1 being the forced entry into COOLDOWN).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xelra.dsl import load_routine_set
from xelra.baselines import ARLPolicy
from xelra.perception import make_population
from xelra import metrics as M

ROUTINES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "routines", "x_elra.yaml")


def transitions_in_window(steps, window):
    times = [steps[i].t for i in range(1, len(steps)) if steps[i].stance != steps[i - 1].stance]
    worst = 0
    for t0 in times:
        worst = max(worst, sum(1 for t in times if t0 <= t < t0 + window))
    return worst


def main():
    rs = load_routine_set(ROUTINES)
    pol = ARLPolicy(rs)
    worst_window = 0
    worst_transitions = 0
    checked = 0
    for arch in ("boundary_mastery", "boundary_affect", "progressing", "accelerating"):
        for sigma in (0.0, 0.1, 0.2, 0.3):
            pop = make_population(seed=999, n_learners=60, sigma=sigma, n_steps=100,
                                  archetype_weights={arch: 1.0})
            for stream in pop:
                steps = pol.run(stream)
                checked += 1
                # Safety: interventions per reset window <= budget.
                mw = M.max_interventions_per_window(steps, rs.budget_reset_minutes)
                worst_window = max(worst_window, mw)
                assert mw <= rs.budget_interventions, (
                    f"safety violated: {mw} > budget {rs.budget_interventions}")
                # Cooldown invariant: same bounded routine not re-fired within cooldown.
                last_fire = {}
                for s in steps:
                    if s.intervened and s.trace is not None:
                        src = s.trace["decision"]["source_routine"]
                        cd = next((r.cooldown_minutes for r in rs.routines if r.id == src), 0)
                        if src in last_fire:
                            assert s.t - last_fire[src] >= cd - 1e-9, (
                                f"cooldown violated for {src}")
                        last_fire[src] = s.t
                # Anti-oscillation bound.
                w = transitions_in_window(steps, rs.oscillation_window_minutes)
                worst_transitions = max(worst_transitions, w)
                assert w <= rs.oscillation_k + 1, (
                    f"oscillation bound exceeded: {w} > k+1={rs.oscillation_k + 1}")

    print(f"PASS test_boundedness: {checked} streams. "
          f"max interventions/window = {worst_window} (budget {rs.budget_interventions}); "
          f"max transitions/window = {worst_transitions} (k+1 = {rs.oscillation_k + 1}).")


if __name__ == "__main__":
    main()
