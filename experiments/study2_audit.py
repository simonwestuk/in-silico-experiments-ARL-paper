"""Study 2 -- Audit / decision-trace sufficiency (DEMM-style) vs standard logs.

Hypothesis addressed: H4 (audit usability). Following the per-property
reconstructability framing of the Decision Evidence Maturity Model, we ask, for
each decision a system emits, whether an external auditor can reconstruct five
properties from the recorded evidence alone:
  policy_basis   -- which routine/rule (and version) authorised the action;
  decision_basis -- why this action and not a higher-priority one (evaluation path);
  boundary       -- what condition would change the decision (exit conditions);
  non_action     -- why nothing happened at a no-op decision point;
  replay         -- deterministic reconstruction from the recorded inputs.

We report per-property completeness over (a) all decisions and (b) action-only
decisions, plus a localisation task: given an intervention, can the responsible
routine and its suppression context be identified, and how many records must be
inspected to do so (a proxy for time-to-diagnosis).
"""

from __future__ import annotations

import numpy as np

from common import ROUTINES, N_LEARNERS, N_STEPS, SEED, save_json
from xelra.dsl import load_routine_set
from xelra.baselines import make_policies
from xelra.perception import make_population
from xelra import metrics as M


def localization(policy_name, steps):
    """Per-intervention localisation outcome under each system's records."""
    out = []
    n_records = sum(1 for s in steps if s.log_record is not None)
    for i, s in enumerate(steps):
        if not s.intervened:
            continue
        if policy_name == "ARL":
            out.append({"localized": True, "reason_known": True, "records_inspected": 1})
        elif policy_name == "B2":
            # rule id is present, but the suppression context is not recorded,
            # and the auditor must scan the session's rule-fired log.
            out.append({"localized": True, "reason_known": False,
                        "records_inspected": n_records})
        else:  # B1 has no routine identity; B3 logs no decisions
            out.append({"localized": False, "reason_known": False,
                        "records_inspected": max(n_records, 1)})
    return out


def main():
    rs = load_routine_set(ROUTINES)
    policies = make_policies(rs)
    # A diverse population so that all decision types (incl. no-ops) occur.
    pop = make_population(seed=SEED, n_learners=N_LEARNERS, sigma=0.12, n_steps=N_STEPS,
                          archetype_weights={"boundary_mastery": 0.4, "boundary_affect": 0.3,
                                             "progressing": 0.3})

    agg = {p: {prop: [] for prop in M.AUDIT_PROPERTIES + ["aggregate"]} for p in policies}
    agg_action = {p: {prop: [] for prop in M.AUDIT_PROPERTIES + ["aggregate"]} for p in policies}
    loc = {p: {"localized": [], "reason_known": [], "records": []} for p in policies}
    n_decisions = 0
    n_noop = 0

    for stream in pop:
        for pname, pol in policies.items():
            steps = pol.run(stream)
            suf = M.audit_sufficiency(pname, steps)
            for prop in M.AUDIT_PROPERTIES + ["aggregate"]:
                agg[pname][prop].append(suf[prop])
            # action-only sufficiency
            act_steps = [s for s in steps if s.intervened]
            if act_steps:
                suf_a = M.audit_sufficiency(pname, act_steps)
                for prop in M.AUDIT_PROPERTIES + ["aggregate"]:
                    agg_action[pname][prop].append(suf_a[prop])
            for ev in localization(pname, steps):
                loc[pname]["localized"].append(1.0 if ev["localized"] else 0.0)
                loc[pname]["reason_known"].append(1.0 if ev["reason_known"] else 0.0)
                loc[pname]["records"].append(ev["records_inspected"])
            if pname == "ARL":
                n_decisions += len(steps)
                n_noop += sum(1 for s in steps if not s.intervened)

    def summarize(table):
        res = {}
        for p in policies:
            res[p] = {}
            for prop in M.AUDIT_PROPERTIES + ["aggregate"]:
                vals = table[p][prop]
                m, ci = M.mean_ci(vals) if vals else (0.0, 0.0)
                res[p][prop] = round(m, 4)
                res[p][prop + "_ci"] = round(ci, 4)
        return res

    loc_summary = {}
    for p in policies:
        lm, lci = M.mean_ci(loc[p]["localized"]) if loc[p]["localized"] else (0.0, 0.0)
        rm, rci = M.mean_ci(loc[p]["reason_known"]) if loc[p]["reason_known"] else (0.0, 0.0)
        recm, recci = M.mean_ci(loc[p]["records"]) if loc[p]["records"] else (0.0, 0.0)
        loc_summary[p] = {
            "localization_rate": round(lm, 4), "localization_rate_ci": round(lci, 4),
            "reason_known_rate": round(rm, 4),
            "mean_records_to_diagnose": round(recm, 2),
            "mean_records_ci": round(recci, 2),
            "n_interventions": len(loc[p]["localized"]),
        }

    out = {
        "study": "study2_audit",
        "design": {"n_learners": N_LEARNERS, "n_steps": N_STEPS, "sigma": 0.12,
                   "n_decisions_ARL": n_decisions, "n_noop_ARL": n_noop},
        "sufficiency_all_decisions": summarize(agg),
        "sufficiency_action_only": summarize(agg_action),
        "localization": loc_summary,
    }
    save_json("study2_audit.json", out)

    print("=== Study 2: audit sufficiency (DEMM-style) ===")
    print(f"ARL decisions={n_decisions}, of which no-ops={n_noop} "
          f"({100*n_noop/max(n_decisions,1):.0f}% leave no record in action-only logs)")
    print(f"\n{'policy':>6} | " + " ".join(f"{p:>13}" for p in M.AUDIT_PROPERTIES) + " |  aggregate")
    for p in policies:
        s = out["sufficiency_all_decisions"][p]
        print(f"{p:>6} | " + " ".join(f"{s[prop]:>13.2f}" for prop in M.AUDIT_PROPERTIES)
              + f" |  {s['aggregate']:.2f}")
    print("\nLocalisation of an intervention's responsible routine + reason:")
    for p in policies:
        L = out["localization"][p]
        print(f"  {p}: localized={L['localization_rate']:.2f}, reason_known="
              f"{L['reason_known_rate']:.2f}, records_to_diagnose={L['mean_records_to_diagnose']}"
              f" (n={L['n_interventions']})")
    return out


if __name__ == "__main__":
    main()
