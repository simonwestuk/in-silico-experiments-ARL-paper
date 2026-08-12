"""Study 4 -- Runtime cost/latency and fairness (intervention-density parity).

Latency (reviewer Q10): per-decision controller evaluation + trace emission time
as a function of candidate-set size and routine-set size, plus emitted trace size.

Fairness (reviewer Q9): with two learner subgroups of differing need, we measure
bounded-intervention density per subgroup, the density parity ratio (min/max), and
the worst-case interventions per budget window. Boundedness caps density equally
across subgroups and renders any disparity measurable directly from traces; the
unbounded direct-ML policy shows a larger, unconstrained disparity.
"""

from __future__ import annotations

import json
import time

import numpy as np

from common import ROUTINES, SEED, save_json
from xelra.dsl import load_routine_set, RoutineSet
from xelra.controller import ARLController
from xelra.state import ControllerState
from xelra.baselines import ARLPolicy, B1DirectML
from xelra.perception import make_population, make_stream
from xelra import metrics as M


def _pad_routine_set(rs: RoutineSet, n_routines: int) -> RoutineSet:
    """Return a routine set inflated to n_routines by adding inert routines.

    Padding routines are permitted only in COLD_START (so they are SKIPPED in the
    active mode), which exercises the per-routine evaluation loop and trace-path
    construction cost without changing decisions."""
    raw = {
        "version": f"{rs.version}-pad{n_routines}",
        "budgets": {"interventions": rs.budget_interventions,
                    "suggestions": rs.budget_suggestions,
                    "reset_minutes": rs.budget_reset_minutes},
        "stability": {"dwell_minutes": rs.dwell_minutes,
                      "oscillation_window_minutes": rs.oscillation_window_minutes,
                      "oscillation_k": rs.oscillation_k},
        "modes": [{"id": m.id, "label": m.label, "precedence": m.precedence,
                   "entry": m.entry, "permitted_routines": m.permitted_routines,
                   "perception_inferred": m.perception_inferred} for m in rs.modes.values()],
        "routines": [],
    }
    for r in rs.routines:
        raw["routines"].append({
            "id": r.id, "name": r.name, "version": r.version, "priority": r.priority,
            "objective": r.objective, "permitted_modes": r.permitted_modes,
            "triggers": r.triggers, "cost": {"interventions": r.interventions_cost,
                                             "suggestions": r.suggestions_cost},
            "cooldown_minutes": r.cooldown_minutes, "action": r.action, "terminates": r.terminates})
    extra = max(0, n_routines - len(rs.routines))
    for i in range(extra):
        raw["routines"].append({
            "id": f"X{i:03d}", "name": f"pad_{i}", "version": "v1",
            "priority": 5, "objective": "default", "permitted_modes": ["COLD_START"],
            "triggers": "false", "cost": {"interventions": 0, "suggestions": 0},
            "cooldown_minutes": 0, "action": None, "terminates": False})
    return RoutineSet(raw)


def latency_table(rs):
    ctrl_base = ARLController(rs)
    state = ControllerState(interventions_remaining=rs.budget_interventions,
                            suggestions_remaining=rs.budget_suggestions)
    stream = make_stream(seed=1, archetype="boundary_affect", sigma=0.12, n_steps=10)
    _, perc, base_cands = stream[5]

    def time_decision(ctrl, perc, cands, reps=2000):
        st = ControllerState(interventions_remaining=rs.budget_interventions,
                             suggestions_remaining=rs.budget_suggestions)
        # warmup
        for _ in range(100):
            ctrl.decide(st, perc, cands, 10.0)
        t0 = time.perf_counter()
        for _ in range(reps):
            ctrl.decide(st, perc, cands, 10.0)
        return (time.perf_counter() - t0) / reps * 1e6  # microseconds

    by_candidates = []
    for c in [1, 5, 10, 25, 50, 100, 250, 500]:
        cands = [{"action_id": f"r{k:04d}", "score": float((k * 2654435761 % 1000) / 1000.0),
                  "source": "agent", "objective": "pedagogy"} for k in range(c)]
        us = time_decision(ctrl_base, perc, cands)
        _, trace, _ = ctrl_base.decide(state, perc, cands, 10.0)
        nbytes = len(json.dumps(trace.to_dict()))
        by_candidates.append({"n_candidates": c, "us_per_decision": round(us, 2),
                              "trace_bytes": nbytes})

    by_routines = []
    for nr in [8, 16, 32, 64, 128]:
        rs_p = _pad_routine_set(rs, nr)
        ctrl = ARLController(rs_p)
        us = time_decision(ctrl, perc, base_cands)
        _, trace, _ = ctrl.decide(
            ControllerState(interventions_remaining=rs.budget_interventions,
                            suggestions_remaining=rs.budget_suggestions),
            perc, base_cands, 10.0)
        nbytes = len(json.dumps(trace.to_dict()))
        by_routines.append({"n_routines": nr, "us_per_decision": round(us, 2),
                            "trace_bytes": nbytes})
    return {"by_candidates": by_candidates, "by_routines": by_routines}


def fairness(rs, n_per_group=150):
    # Subgroup A: mostly progressing; Subgroup B: higher need (boundary cases).
    popA = make_population(seed=SEED + 1, n_learners=n_per_group, sigma=0.12, n_steps=80,
                           archetype_weights={"progressing": 0.7, "boundary_mastery": 0.3},
                           subgroup="A")
    popB = make_population(seed=SEED + 2, n_learners=n_per_group, sigma=0.12, n_steps=80,
                           archetype_weights={"boundary_mastery": 0.5, "boundary_affect": 0.5},
                           subgroup="B")

    def densities(policy_cls):
        out = {}
        for label, pop in (("A", popA), ("B", popB)):
            dens, maxwin = [], []
            for stream in pop:
                res = policy_cls(rs).run(stream)
                dens.append(M.intervention_density_per_hour(res))
                maxwin.append(M.max_interventions_per_window(res, rs.budget_reset_minutes))
            m, ci = M.mean_ci(dens)
            out[label] = {"density_mean": round(m, 3), "density_ci": round(ci, 3),
                          "max_per_window": int(max(maxwin))}
        a, b = out["A"]["density_mean"], out["B"]["density_mean"]
        out["parity_ratio"] = round(min(a, b) / max(a, b), 4) if max(a, b) > 0 else 1.0
        out["density_gap"] = round(abs(a - b), 3)
        return out

    return {"ARL": densities(ARLPolicy), "B1": densities(B1DirectML),
            "budget_per_window": rs.budget_interventions}


def main():
    rs = load_routine_set(ROUTINES)
    lat = latency_table(rs)
    fair = fairness(rs)
    out = {"study": "study4_latency_fairness", "latency": lat, "fairness": fair}
    save_json("study4_latency_fairness.json", out)

    print("=== Study 4a: latency ===")
    print("candidates -> us/decision (trace bytes):",
          ", ".join(f"{r['n_candidates']}:{r['us_per_decision']}us" for r in lat["by_candidates"]))
    print("routines   -> us/decision:",
          ", ".join(f"{r['n_routines']}:{r['us_per_decision']}us" for r in lat["by_routines"]))
    print("=== Study 4b: fairness (intervention density per hour) ===")
    for pol in ("ARL", "B1"):
        f = fair[pol]
        print(f"  {pol}: A={f['A']['density_mean']:.2f} (max/win {f['A']['max_per_window']}), "
              f"B={f['B']['density_mean']:.2f} (max/win {f['B']['max_per_window']}), "
              f"parity={f['parity_ratio']:.2f}, gap={f['density_gap']:.2f}")
    print(f"  budget cap per window = {fair['budget_per_window']}")
    return out


if __name__ == "__main__":
    main()
