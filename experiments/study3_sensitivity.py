"""Study 3 -- Boundedness vs adaptivity: responsiveness, support, and stability.

Addresses the concern that deterministic boundedness may degrade responsiveness
or under-support learners in dynamic contexts, and the request for sensitivity of
results to controller hyperparameters (budgets, cooldowns, dwell time).

On streams with a single ground-truth 'genuine need' episode we measure:
  - time-to-support: delay from need onset to the first bounded intervention;
  - under-support rate: fraction of episodes that received no support in-window;
  - stability: mode oscillation rate (MOR) and bounded-intervention density.

Part A compares policies at the default configuration; Part B sweeps ARL's
budget, dwell time, and cooldown scale to trace the boundedness-adaptivity
trade-off.
"""

from __future__ import annotations

import copy

import numpy as np
import yaml

from common import ROUTINES, SEED, save_json
from xelra.dsl import RoutineSet, load_routine_set
from xelra.baselines import (ARLPolicy, B1DirectML, B2RuleITS, B5SmoothedML,
                             routine_set_without_anti_oscillation)
from xelra.perception import make_multi_need_stream
from xelra import metrics as M

N_EP = 150
SIGMA = 0.12


def window_support_delay(steps, onset, end, grace=9.0):
    for s in steps:
        if s.t < onset:
            continue
        if s.t > end + grace:
            break
        if s.intervened:
            return s.t - onset
    return None


def eval_policy(policy, episodes):
    delays, n_windows, under, mors, dens = [], 0, 0, [], []
    for (stream, windows) in episodes:
        steps = policy.run(stream)
        for (onset, end) in windows:
            n_windows += 1
            d = window_support_delay(steps, onset, end)
            if d is None:
                under += 1
            else:
                delays.append(d)
        mors.append(M.mode_oscillation_rate(steps))
        dens.append(M.intervention_density_per_hour(steps))
    md, mci = M.mean_ci(delays) if delays else (float("nan"), 0.0)
    return {
        "mean_delay_min": round(md, 2), "delay_ci": round(mci, 2),
        "under_support_rate": round(under / max(n_windows, 1), 4),
        "mor": round(float(np.mean(mors)), 2),
        "density_per_h": round(float(np.mean(dens)), 2),
    }


def variant(budget=None, dwell=None, cooldown_scale=1.0):
    raw = yaml.safe_load(open(ROUTINES, encoding="utf-8"))
    if budget is not None:
        raw["budgets"]["interventions"] = budget
    if dwell is not None:
        raw["stability"]["dwell_minutes"] = dwell
    if cooldown_scale != 1.0:
        for r in raw["routines"]:
            r["cooldown_minutes"] = float(r.get("cooldown_minutes", 0)) * cooldown_scale
    return RoutineSet(raw)


def main():
    rs = load_routine_set(ROUTINES)
    episodes = [make_multi_need_stream(seed=SEED + i, sigma=SIGMA) for i in range(N_EP)]

    # --- Part A: policy comparison at default configuration ---
    polA = {
        "ARL": ARLPolicy(rs),
        "B1": B1DirectML(rs),
        "B2": B2RuleITS(rs),
        "B5": B5SmoothedML(rs),
        "ARL_noAO": ARLPolicy(routine_set_without_anti_oscillation(rs)),
    }
    partA = {name: eval_policy(p, episodes) for name, p in polA.items()}

    # --- Part B: ARL hyperparameter sensitivity ---
    sweep = {"budget": [], "dwell": [], "cooldown_scale": []}
    for b in [1, 2, 3, 5, 8]:
        r = eval_policy(ARLPolicy(variant(budget=b)), episodes)
        r["value"] = b
        sweep["budget"].append(r)
    for d in [0, 2, 5, 10, 15]:
        r = eval_policy(ARLPolicy(variant(dwell=d)), episodes)
        r["value"] = d
        sweep["dwell"].append(r)
    for c in [0.25, 0.5, 1.0, 2.0, 4.0]:
        r = eval_policy(ARLPolicy(variant(cooldown_scale=c)), episodes)
        r["value"] = c
        sweep["cooldown_scale"].append(r)

    out = {
        "study": "study3_sensitivity",
        "design": {"n_episodes": N_EP, "sigma": SIGMA,
                   "defaults": {"budget": rs.budget_interventions,
                                "dwell_minutes": rs.dwell_minutes}},
        "policy_comparison": partA,
        "arl_sweep": sweep,
    }
    save_json("study3_sensitivity.json", out)

    print("=== Study 3a: responsiveness vs stability (default config) ===")
    print(f"{'policy':>9} | {'delay(min)':>10} {'under-supp':>10} {'MOR/h':>7} {'dens/h':>7}")
    for name, r in partA.items():
        print(f"{name:>9} | {r['mean_delay_min']:>10.2f} {r['under_support_rate']:>10.2f} "
              f"{r['mor']:>7.2f} {r['density_per_h']:>7.2f}")
    print("\n=== Study 3b: ARL budget sweep (stability<->support trade-off) ===")
    for r in sweep["budget"]:
        print(f"  budget={r['value']}: delay {r['mean_delay_min']:.2f} min, "
              f"under-support {r['under_support_rate']:.2f}, MOR {r['mor']:.2f}, dens {r['density_per_h']:.2f}")
    print("--- dwell sweep ---")
    for r in sweep["dwell"]:
        print(f"  dwell={r['value']}min: delay {r['mean_delay_min']:.2f}, "
              f"under-support {r['under_support_rate']:.2f}, MOR {r['mor']:.2f}")
    return out


if __name__ == "__main__":
    main()
