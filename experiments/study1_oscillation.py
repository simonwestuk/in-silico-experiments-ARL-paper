"""Study 1 -- Anti-oscillation and predictability under matched perception noise.

Hypotheses addressed: H2 (behavioural predictability / reduced oscillation) and
the boundedness safety property (max intervention density per window).

Design: an open-loop perturbation experiment. For each noise level sigma, a
population of learners sits near a regulatory-mode boundary; every policy (ARL,
B1 direct-ML, B2 rule-based ITS) consumes the *same* noisy perception stream.
We report, with 95% CIs across learners:
  - Mode Oscillation Rate (MOR), stance transitions per hour;
  - immediate stance reversals (A->B->A);
  - Intervention Predictability Index (IPI), 1 - H(next stance | stance);
  - bounded intervention density per hour;
  - the maximum bounded interventions in any budget-reset window (safety check).
"""

from __future__ import annotations

import numpy as np

from common import ROUTINES, SIGMA_GRID, N_LEARNERS, N_STEPS, SEED, save_json
from xelra.dsl import load_routine_set
from xelra.baselines import ARLPolicy, B1DirectML, B2RuleITS
from xelra.perception import make_population
from xelra import metrics as M


def run(archetype="boundary_mastery", weights=None):
    rs = load_routine_set(ROUTINES)
    from xelra.baselines import B5SmoothedML, routine_set_without_anti_oscillation
    policies = {
        "ARL": ARLPolicy(rs),
        "B1": B1DirectML(rs),                                  # direct ML, no regulation
        "B2": B2RuleITS(rs),                                   # transparent rule-based tutor
        "B5": B5SmoothedML(rs),                                # smoothed / hysteresis policy
        "ARL_noAO": ARLPolicy(routine_set_without_anti_oscillation(rs)),  # ablation
    }
    if weights is None:
        weights = {archetype: 1.0}

    rows = []
    for sigma in SIGMA_GRID:
        pop = make_population(seed=SEED, n_learners=N_LEARNERS, sigma=sigma,
                              n_steps=N_STEPS, archetype_weights=weights)
        per = {p: {"mor": [], "rev": [], "ipi": [], "ipi_next": [], "dens": [],
                   "maxwin": []} for p in policies}
        for stream in pop:
            for pname, pol in policies.items():
                res = pol.run(stream)
                per[pname]["mor"].append(M.mode_oscillation_rate(res))
                per[pname]["rev"].append(M.immediate_reversals(res))
                per[pname]["ipi"].append(M.ipi_stance(res))
                per[pname]["ipi_next"].append(M.ipi_next_intervention(res))
                per[pname]["dens"].append(M.intervention_density_per_hour(res))
                per[pname]["maxwin"].append(
                    M.max_interventions_per_window(res, rs.budget_reset_minutes))
        row = {"sigma": sigma}
        for pname in policies:
            for metric in ("mor", "rev", "ipi", "ipi_next", "dens"):
                m, ci = M.mean_ci(per[pname][metric])
                row[f"{pname}_{metric}"] = round(m, 4)
                row[f"{pname}_{metric}_ci"] = round(ci, 4)
            row[f"{pname}_maxwin"] = int(max(per[pname]["maxwin"]))
        rows.append(row)

    out = {
        "study": "study1_oscillation",
        "design": {
            "archetype_weights": weights, "n_learners": N_LEARNERS,
            "n_steps": N_STEPS, "sigma_grid": SIGMA_GRID,
            "budget_interventions": rs.budget_interventions,
            "budget_reset_minutes": rs.budget_reset_minutes,
            "dwell_minutes": rs.dwell_minutes, "oscillation_k": rs.oscillation_k,
        },
        "rows": rows,
    }
    return out


def main():
    out = run()
    save_json("study1_oscillation.json", out)
    # Affect-driven variant (secondary) for robustness.
    out_aff = run(archetype="boundary_affect")
    save_json("study1_oscillation_affect.json", out_aff)

    pols = ["ARL", "B1", "B2", "B5", "ARL_noAO"]
    print("=== Study 1: anti-oscillation (boundary_mastery), MOR per hour ===")
    print(f"{'sigma':>5} | " + " ".join(f"{p:>9}" for p in pols))
    for r in out["rows"]:
        print(f"{r['sigma']:>5.2f} | " + " ".join(f"{r[p+'_mor']:>9.2f}" for p in pols))
    print("\n--- IPI (predictability) and density at sigma=0.15 ---")
    r = next(r for r in out["rows"] if abs(r["sigma"] - 0.15) < 1e-9)
    for p in pols:
        print(f"  {p:>9}: MOR {r[p+'_mor']:6.2f} | IPI {r[p+'_ipi']:.3f} | "
              f"density {r[p+'_dens']:5.2f}/h | max/win {r[p+'_maxwin']}")
    return out


if __name__ == "__main__":
    main()
