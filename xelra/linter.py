"""Static validation (linting) for X-ELRA routine sets.

Authoring and maintaining a routine set is a knowledge-engineering task; this
linter reduces that burden by catching common specification errors before
deployment, complementing the property-based tests and the bounded model
checker. It runs offline on a routine set and reports errors (block deployment)
and warnings (review recommended).

Checks:
  E1  duplicate routine ids
  E2  routine references an unknown mode
  E3  a bounded routine's cost exceeds the budget (it could never execute)
  E4  negative cooldown, priority, or cost
  E5  a perception-inferred specialised mode has no permitted bounded routine
      and no default fallback (it could never support a learner)
  W1  routine priorities are inconsistent with the lexicographic objective
      order (a higher-precedence objective placed below a lower one)
  W2  no totality default routine reachable from a mode (controller may emit
      empty action spaces)
  W3  two routines share a priority within the same mode (tie-break relies on id)

It also exposes ``counterfactual``: given a routine set and a perception, report
which routine would fire, for unit / counterfactual tests of routine edits.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .dsl import RoutineSet, load_routine_set, eval_predicate
from .modes import infer_mode

SPECIALISED = {"STRUGGLING", "LAPSED", "ACCELERATING", "CONSOLIDATING", "DIAGNOSTIC"}


def lint(rs: RoutineSet) -> List[Tuple[str, str, str]]:
    """Return a list of (severity, code, message)."""
    issues: List[Tuple[str, str, str]] = []
    ids = [r.id for r in rs.routines]
    for rid in set(ids):
        if ids.count(rid) > 1:
            issues.append(("error", "E1", f"duplicate routine id {rid}"))
    known_modes = set(rs.modes)
    for r in rs.routines:
        for m in r.permitted_modes:
            if m not in known_modes:
                issues.append(("error", "E2", f"routine {r.id} references unknown mode {m}"))
        if r.interventions_cost > rs.budget_interventions:
            issues.append(("error", "E3",
                           f"routine {r.id} intervention cost {r.interventions_cost} "
                           f"exceeds budget {rs.budget_interventions}; it can never execute"))
        if r.suggestions_cost > rs.budget_suggestions:
            issues.append(("error", "E3",
                           f"routine {r.id} suggestion cost exceeds the suggestion budget"))
        if min(r.priority, r.interventions_cost, r.suggestions_cost, r.cooldown_minutes) < 0:
            issues.append(("error", "E4", f"routine {r.id} has a negative priority/cost/cooldown"))

    # E5: specialised perception-inferred modes must reach a bounded routine.
    for m in rs.modes.values():
        if m.id in SPECIALISED and m.perception_inferred:
            bounded = [r for r in rs.routines
                       if m.id in r.permitted_modes
                       and (r.interventions_cost > 0 or r.suggestions_cost > 0)]
            default = [r for r in rs.routines if m.id in r.permitted_modes and r.objective == "default"]
            if not bounded and not default:
                issues.append(("error", "E5",
                               f"mode {m.id} has no permitted bounded or default routine"))

    # W1: priorities must respect the lexicographic objective order.
    for a in rs.routines:
        for b in rs.routines:
            if a.id >= b.id:
                continue
            ra, rb = rs.objective_rank(a.objective), rs.objective_rank(b.objective)
            if ra < rb and a.priority < b.priority:
                issues.append(("warning", "W1",
                               f"{a.id} ({a.objective}) outranks {b.id} ({b.objective}) by "
                               f"objective but has lower priority"))
    # W2: totality default reachable.
    if not any(r.objective == "default" for r in rs.routines):
        issues.append(("warning", "W2", "no default (totality) routine in the set"))
    # W3: priority ties within a shared mode.
    for m in rs.modes:
        prios = {}
        for r in rs.routines:
            if m in r.permitted_modes:
                prios.setdefault(r.priority, []).append(r.id)
        for p, rr in prios.items():
            if len(rr) > 1:
                issues.append(("warning", "W3",
                               f"mode {m}: routines {rr} share priority {p} (tie-break on id)"))
    return issues


def counterfactual(rs: RoutineSet, perception: Dict[str, Any]) -> Dict[str, Any]:
    """Which routine would fire for this perception (a counterfactual unit test)."""
    ns = dict(perception)
    ns.setdefault("n_candidates", 1)
    mode = infer_mode(rs, ns)
    for r in rs.routines:
        if mode not in r.permitted_modes:
            continue
        if eval_predicate(r.triggers, ns):
            return {"mode": mode, "first_eligible_routine": r.id,
                    "action": r.action, "objective": r.objective}
    return {"mode": mode, "first_eligible_routine": None, "action": None}


def main():
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "routines", "x_elra.yaml")
    rs = load_routine_set(path)
    issues = lint(rs)
    errs = [i for i in issues if i[0] == "error"]
    warns = [i for i in issues if i[0] == "warning"]
    print(f"Linted routine set version {rs.version}: {len(errs)} errors, {len(warns)} warnings")
    for sev, code, msg in issues:
        print(f"  [{sev:>7} {code}] {msg}")
    if not issues:
        print("  no issues found.")
    raise SystemExit(1 if errs else 0)


if __name__ == "__main__":
    main()
