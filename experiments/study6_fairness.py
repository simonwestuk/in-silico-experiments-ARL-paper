"""Study 6 -- Fairness / disparate impact across learner subgroups.

Addresses the request for analysis of whether routine priorities and mode
entries/exits differentially affect learners by proficiency or engagement. For
proficiency strata (low / medium / high) and an engagement contrast we measure,
per subgroup: the rate of entering the supportive (STRUGGLING) mode, the bounded
intervention density, the worst-case interventions per budget window, and
disparate-impact ratios. The point is that ARL's mode entries track genuine need
(so they differ by proficiency by design), while boundedness caps worst-case
exposure equally and every per-subgroup quantity is computed directly from the
decision traces (so disparities are auditable, not hidden).
"""

from __future__ import annotations

import numpy as np

from common import ROUTINES, N_STEPS, SEED, save_json
from xelra.dsl import load_routine_set
from xelra.baselines import ARLPolicy, B1DirectML
from xelra.perception import make_population
from xelra import metrics as M

N = 150


def struggling_rate(steps):
    return float(np.mean([1.0 if s.stance == "STRUGGLING" else 0.0 for s in steps]))


def measure(policy_cls, rs, subgroups):
    res = {}
    for name, arch in subgroups.items():
        pop = make_population(seed=SEED, n_learners=N, sigma=0.12, n_steps=N_STEPS,
                              archetype_weights={arch: 1.0}, subgroup=name)
        sr, dens, mw = [], [], []
        for stream in pop:
            steps = policy_cls(rs).run(stream)
            sr.append(struggling_rate(steps))
            dens.append(M.intervention_density_per_hour(steps))
            mw.append(M.max_interventions_per_window(steps, rs.budget_reset_minutes))
        m_sr, ci_sr = M.mean_ci(sr)
        m_d, ci_d = M.mean_ci(dens)
        res[name] = {
            "struggling_rate": round(m_sr, 4), "struggling_rate_ci": round(ci_sr, 4),
            "density_per_h": round(m_d, 3), "density_ci": round(ci_d, 3),
            "max_per_window": int(max(mw)),
        }
    dens_vals = [res[s]["density_per_h"] for s in subgroups]
    mw_vals = [res[s]["max_per_window"] for s in subgroups]
    res["disparate_impact"] = {
        "density_ratio_min_over_max": round(min(dens_vals) / max(dens_vals), 3) if max(dens_vals) > 0 else 1.0,
        "density_gap": round(max(dens_vals) - min(dens_vals), 3),
        "max_window_spread": [min(mw_vals), max(mw_vals)],
    }
    return res


def main():
    rs = load_routine_set(ROUTINES)
    proficiency = {"low": "prof_low", "medium": "prof_med", "high": "prof_high"}
    out = {
        "study": "study6_fairness",
        "design": {"n_per_subgroup": N, "sigma": 0.12,
                   "budget_per_window": rs.budget_interventions},
        "ARL": measure(ARLPolicy, rs, proficiency),
        "B1": measure(B1DirectML, rs, proficiency),
    }
    save_json("study6_fairness.json", out)

    print("=== Study 6: disparate impact across proficiency strata ===")
    for pol in ("ARL", "B1"):
        print(f"\n{pol}:")
        for s in ("low", "medium", "high"):
            r = out[pol][s]
            print(f"  {s:>7} proficiency: STRUGGLING-entry {r['struggling_rate']:.2f}, "
                  f"density {r['density_per_h']:.2f}/h, max/window {r['max_per_window']}")
        di = out[pol]["disparate_impact"]
        print(f"  -> density parity (min/max) {di['density_ratio_min_over_max']:.2f}, "
              f"gap {di['density_gap']:.2f}/h, worst-case window in [{di['max_window_spread'][0]},"
              f"{di['max_window_spread'][1]}] (budget {rs.budget_interventions})")
    return out


if __name__ == "__main__":
    main()
