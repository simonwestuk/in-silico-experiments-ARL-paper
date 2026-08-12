"""Study 3 -- Concurrency, partial signals, and determinism in practice.

Addresses the reviewer's concern that determinism may be brittle under
asynchronous, missing, or late-arriving signals and multi-agent suggestions.
Four checks:

  (a) Replay determinism: persist each decision's state and inputs, then have a
      fresh controller re-execute from the persisted data and recompute the
      content hash. A late/out-of-order *implementation* cannot change a recorded
      decision because the controller is a pure function of the snapshot.

  (b) Order-invariance (multi-agent suggestions): candidates from several upstream
      agents may arrive in any order; shuffling the candidate list must not change
      the decision or its hash (the controller canonicalises candidate order).

  (c) Missing-modality guard: when a data source drops out (feature_gap > 2) the
      controller enters DIAGNOSTIC and runs the P2 guard instead of acting on
      incomplete perception. We quantify how often the guard prevents an action
      that a guard-free controller would have taken on the same incomplete data.

  (d) Late-arriving signal (snapshot semantics): moving one affect signal from
      decision t to t+1 changes only the decisions at the boundary, and both runs
      remain fully replayable -- the late signal is incorporated at the next
      decision point rather than retroactively mutating an in-flight decision.
"""

from __future__ import annotations

import copy
import json

import numpy as np

from common import ROUTINES, SEED, save_json
from xelra.dsl import load_routine_set, RoutineSet
from xelra.controller import ARLController
from xelra.state import ControllerState
from xelra.modes import infer_mode
from xelra.perception import make_population, make_stream


def _init_state(rs):
    return ControllerState(interventions_remaining=rs.budget_interventions,
                           suggestions_remaining=rs.budget_suggestions)


def replay_determinism(rs, pop):
    ctrl = ARLController(rs)
    auditor = ARLController(rs)  # independent instance ("the auditor")
    total = 0
    matches = 0
    for stream in pop:
        state = _init_state(rs)
        records = []
        for (t, perc, cands) in stream:
            persisted = json.loads(json.dumps(state.to_replay()))  # round-trip
            decision, trace, state = ctrl.decide(state, perc, cands, t)
            records.append((persisted, perc, cands, t, trace.trace_id))
        for (persisted, perc, cands, t, tid) in records:
            st = ControllerState.from_replay(persisted)
            _, trace2, _ = auditor.decide(st, perc, cands, t)
            total += 1
            matches += int(trace2.trace_id == tid)
    return matches, total


def order_invariance(rs, pop, n_shuffles=8, seed=1):
    ctrl = ARLController(rs)
    rng = np.random.default_rng(seed)
    total = 0
    invariant = 0
    for stream in pop:
        state = _init_state(rs)
        for (t, perc, cands) in stream:
            # Reference decision (canonical order handled inside the controller).
            _, trace_ref, next_state = ctrl.decide(state, perc, cands, t)
            ok = True
            for _ in range(n_shuffles):
                shuffled = list(cands)
                rng.shuffle(shuffled)
                _, trace_s, _ = ctrl.decide(state, perc, shuffled, t)
                if trace_s.trace_id != trace_ref.trace_id:
                    ok = False
                    break
            total += 1
            invariant += int(ok)
            state = next_state
    return invariant, total


def _routine_set_without_guard(rs: RoutineSet) -> RoutineSet:
    """Build a guard-free variant: no DIAGNOSTIC mode, no P2 data-integrity guard.

    Missing data is then ignored and the controller infers a pedagogical mode
    from incomplete perception."""
    raw = {
        "version": rs.version + "-noguard",
        "budgets": {"interventions": rs.budget_interventions,
                    "suggestions": rs.budget_suggestions,
                    "reset_minutes": rs.budget_reset_minutes},
        "stability": {"dwell_minutes": rs.dwell_minutes,
                      "oscillation_window_minutes": rs.oscillation_window_minutes,
                      "oscillation_k": rs.oscillation_k},
        "modes": [], "routines": [],
    }
    for m in rs.modes.values():
        if m.id == "DIAGNOSTIC":
            continue
        raw["modes"].append({
            "id": m.id, "label": m.label, "precedence": m.precedence,
            "entry": m.entry, "permitted_routines": [r for r in m.permitted_routines if r != "P2"],
            "perception_inferred": m.perception_inferred,
        })
    for r in rs.routines:
        if r.id == "P2":
            continue
        raw["routines"].append({
            "id": r.id, "name": r.name, "version": r.version, "priority": r.priority,
            "objective": r.objective, "permitted_modes": [pm for pm in r.permitted_modes if pm != "DIAGNOSTIC"],
            "triggers": r.triggers, "cost": {"interventions": r.interventions_cost,
                                             "suggestions": r.suggestions_cost},
            "cooldown_minutes": r.cooldown_minutes, "action": r.action, "terminates": r.terminates,
        })
    return RoutineSet(raw)


# Pedagogical/affective actions that should NOT be taken on incomplete data.
_PEDAGOGICAL_ACTIONS = {"offer_guided_steps", "suggest_wellbeing_break",
                        "suggest_stretch_goal", "suggest_consolidation"}


def _outage_stream(seed, n_steps=40, outage=(15, 31), dt=1.5):
    """A stable NOMINAL learner who suffers a contiguous modality outage.

    During the outage feature_gap>2 (a data source has dropped out) and the
    remaining signals, if taken at face value, read as STRUGGLING (low mastery,
    no recent clicks). A controller that ignores the gap would act on incomplete
    perception; the guarded controller should instead enter DIAGNOSTIC.
    """
    rng = np.random.default_rng(seed)
    stream = []
    for i in range(n_steps):
        t = i * dt
        in_outage = outage[0] <= i < outage[1]
        if in_outage:
            perc = {"lowest_mastery": 0.2, "highest_mastery": 0.5, "mean_mastery": 0.35,
                    "has_mastery": True, "mastery_count": 7, "impressions": 12 + i,
                    "clicks_14d": 0, "days_since_engagement": 1.0, "progress_rate": 0.02,
                    "recent_completions": 0, "active_goals": True,
                    "confusion_flag": False, "frustration_flag": False, "feature_gap": 3,
                    "subgroup": "A"}
        else:
            perc = {"lowest_mastery": 0.6, "highest_mastery": 0.72, "mean_mastery": 0.66,
                    "has_mastery": True, "mastery_count": 7, "impressions": 12 + i,
                    "clicks_14d": 5, "days_since_engagement": 1.0, "progress_rate": 0.02,
                    "recent_completions": 0, "active_goals": True,
                    "confusion_flag": False, "frustration_flag": False, "feature_gap": 0,
                    "subgroup": "A"}
        cands = [{"action_id": f"r{k:03d}", "score": float(rng.random()),
                  "source": "rec", "objective": "pedagogy"} for k in range(10)]
        stream.append((t, perc, cands))
    return stream


def missing_modality_guard(rs, n_learners=120):
    ctrl = ARLController(rs)
    rs_ng = _routine_set_without_guard(rs)
    ctrl_ng = ARLController(rs_ng)

    missing_steps = 0
    guarded_to_diagnostic = 0
    guarded_inappropriate = 0
    guardfree_inappropriate = 0
    episodes = 0
    ep_guardfree_inappropriate = 0
    ep_guarded_inappropriate = 0
    det_ok = True
    for j in range(n_learners):
        stream = _outage_stream(seed=SEED + 3 + j)
        s_g = _init_state(rs)
        s_ng = _init_state(rs_ng)
        ep_g = ep_ng = False
        ids_a, ids_b = [], []
        for (t, perc, cands) in stream:
            dec_g, tr_g, s_g = ctrl.decide(s_g, perc, cands, t)
            dec_ng, _, s_ng = ctrl_ng.decide(s_ng, perc, cands, t)
            ids_a.append(tr_g.trace_id)
            if perc["feature_gap"] > 2:
                missing_steps += 1
                if dec_g["mode"] == "DIAGNOSTIC":
                    guarded_to_diagnostic += 1
                if dec_g["action"] in _PEDAGOGICAL_ACTIONS:
                    guarded_inappropriate += 1; ep_g = True
                if dec_ng["action"] in _PEDAGOGICAL_ACTIONS:
                    guardfree_inappropriate += 1; ep_ng = True
        # determinism on outage streams
        s2 = _init_state(rs)
        for (t, perc, cands) in stream:
            _, tr, s2 = ctrl.decide(s2, perc, cands, t)
            ids_b.append(tr.trace_id)
        det_ok = det_ok and (ids_a == ids_b)
        episodes += 1
        ep_guardfree_inappropriate += int(ep_ng)
        ep_guarded_inappropriate += int(ep_g)
    return {
        "missing_modality_decisions": missing_steps,
        "routed_to_diagnostic_rate": round(guarded_to_diagnostic / max(missing_steps, 1), 4),
        "guarded_inappropriate_action_rate": round(guarded_inappropriate / max(missing_steps, 1), 4),
        "guardfree_inappropriate_action_rate": round(guardfree_inappropriate / max(missing_steps, 1), 4),
        "episodes": episodes,
        "episodes_guardfree_acted_on_incomplete_data": ep_guardfree_inappropriate,
        "episodes_guarded_acted_on_incomplete_data": ep_guarded_inappropriate,
        "determinism_on_outage_streams": det_ok,
    }


def late_signal_snapshot(rs, seed=11):
    """Move one affect signal from decision i to i+1 and compare runs."""
    ctrl = ARLController(rs)
    stream = make_stream(seed=seed, archetype="boundary_affect", sigma=0.12, n_steps=80)
    # Find a step with an affect flag on, followed by a step with it off.
    idx = None
    for i in range(len(stream) - 1):
        if (stream[i][1]["confusion_flag"] and not stream[i + 1][1]["confusion_flag"]):
            idx = i
            break
    if idx is None:
        return {"applicable": False}
    late = copy.deepcopy(stream)
    late[idx][1]["confusion_flag"] = False        # signal misses decision i
    late[idx + 1][1]["confusion_flag"] = True     # and arrives at decision i+1

    def run(stream_):
        st = _init_state(rs)
        ids = []
        for (t, perc, cands) in stream_:
            _, trace, st = ctrl.decide(st, perc, cands, t)
            ids.append(trace.trace_id)
        return ids

    base_ids = run(stream)
    late_ids = run(late)
    differing = [i for i, (a, b) in enumerate(zip(base_ids, late_ids)) if a != b]
    # Determinism of each run (re-run must reproduce ids).
    det_base = run(stream) == base_ids
    det_late = run(late) == late_ids
    return {
        "applicable": True, "shift_index": idx,
        "n_differing_decisions": len(differing),
        "differing_localised_to_boundary": bool(differing and max(differing) - min(differing) <= 2),
        "both_runs_deterministic": bool(det_base and det_late),
    }


def main():
    rs = load_routine_set(ROUTINES)
    pop = make_population(seed=SEED, n_learners=150, sigma=0.15, n_steps=80,
                          archetype_weights={"boundary_mastery": 0.5, "boundary_affect": 0.5})
    rd_match, rd_total = replay_determinism(rs, pop)
    oi_inv, oi_total = order_invariance(rs, pop[:60])
    guard = missing_modality_guard(rs)
    late = late_signal_snapshot(rs)

    out = {
        "study": "study3_robustness",
        "replay_determinism": {"matches": rd_match, "total": rd_total,
                               "rate": round(rd_match / rd_total, 6)},
        "order_invariance": {"invariant": oi_inv, "total": oi_total,
                             "rate": round(oi_inv / oi_total, 6)},
        "missing_modality_guard": guard,
        "late_signal_snapshot": late,
    }
    save_json("study3_robustness.json", out)

    print("=== Study 3: concurrency / partial signals / determinism ===")
    print(f"(a) Replay determinism: {rd_match}/{rd_total} "
          f"({100*rd_match/rd_total:.2f}%) decisions reproduced their hash from persisted state.")
    print(f"(b) Order-invariance: {oi_inv}/{oi_total} "
          f"({100*oi_inv/oi_total:.2f}%) decisions invariant to candidate arrival order.")
    print(f"(c) Missing-modality guard: of {guard['missing_modality_decisions']} incomplete-data "
          f"decisions, {100*guard['routed_to_diagnostic_rate']:.0f}% routed to DIAGNOSTIC. "
          f"Inappropriate pedagogical actions on incomplete data: guarded "
          f"{100*guard['guarded_inappropriate_action_rate']:.0f}% vs guard-free "
          f"{100*guard['guardfree_inappropriate_action_rate']:.0f}%. "
          f"Episodes acting on incomplete data: guard-free "
          f"{guard['episodes_guardfree_acted_on_incomplete_data']}/{guard['episodes']} vs guarded "
          f"{guard['episodes_guarded_acted_on_incomplete_data']}/{guard['episodes']}.")
    print(f"(d) Late signal: shifting one affect signal by one decision changed "
          f"{late.get('n_differing_decisions')} decisions, localised to the boundary="
          f"{late.get('differing_localised_to_boundary')}, both runs deterministic="
          f"{late.get('both_runs_deterministic')}.")
    return out


if __name__ == "__main__":
    main()
