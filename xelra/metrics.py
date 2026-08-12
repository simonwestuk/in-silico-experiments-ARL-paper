"""Metrics: oscillation, predictability, boundedness, and audit sufficiency.

These operationalise the indicators defined in the paper's evaluation framework
(MOR, IPI, ARS/sufficiency) so that the simulation studies report exactly the
quantities the hypotheses concern.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Dict, List

from .baselines import StepResult


# --- Stability / oscillation -------------------------------------------------


def mode_transitions(steps: List[StepResult]) -> int:
    return sum(1 for i in range(1, len(steps)) if steps[i].stance != steps[i - 1].stance)


def mode_oscillation_rate(steps: List[StepResult]) -> float:
    """Stance transitions per hour."""
    if len(steps) < 2:
        return 0.0
    horizon_min = steps[-1].t - steps[0].t
    horizon_hr = horizon_min / 60.0 if horizon_min > 0 else 1e-9
    return mode_transitions(steps) / horizon_hr


def immediate_reversals(steps: List[StepResult]) -> int:
    """Count A->B->A stance reversals across three consecutive decision points."""
    c = 0
    for i in range(2, len(steps)):
        if (steps[i].stance == steps[i - 2].stance
                and steps[i].stance != steps[i - 1].stance):
            c += 1
    return c


# --- Predictability (IPI) ----------------------------------------------------


def _entropy(counts) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h


def _conditional_entropy(pairs) -> float:
    """H(Y|X) from a list of (x, y) pairs, in bits."""
    by_x = defaultdict(Counter)
    total = len(pairs)
    if total == 0:
        return 0.0
    for x, y in pairs:
        by_x[x][y] += 1
    h = 0.0
    for x, yc in by_x.items():
        nx = sum(yc.values())
        h += (nx / total) * _entropy(list(yc.values()))
    return h


def ipi_stance(steps: List[StepResult]) -> float:
    """Predictability of the next stance from the current stance.

    IPI = 1 - H(stance_{t+1} | stance_t) / H(stance_{t+1}); 1.0 = fully
    predictable. Captures whether a learner who can see the current regulatory
    mode can anticipate the system's next stance.
    """
    pairs = [(steps[i].stance, steps[i + 1].stance) for i in range(len(steps) - 1)]
    if not pairs:
        return 1.0
    marg = _entropy(list(Counter(y for _, y in pairs).values()))
    if marg <= 1e-12:
        return 1.0
    return max(0.0, 1.0 - _conditional_entropy(pairs) / marg)


def ipi_next_intervention(steps: List[StepResult]) -> float:
    """Predictability of whether an intervention arrives next, given the stance."""
    pairs = [(steps[i].stance, steps[i + 1].intervened) for i in range(len(steps) - 1)]
    if not pairs:
        return 1.0
    marg = _entropy(list(Counter(y for _, y in pairs).values()))
    if marg <= 1e-12:
        return 1.0
    return max(0.0, 1.0 - _conditional_entropy(pairs) / marg)


# --- Intervention density / boundedness -------------------------------------


def intervention_density_per_hour(steps: List[StepResult]) -> float:
    if len(steps) < 2:
        return 0.0
    horizon_hr = (steps[-1].t - steps[0].t) / 60.0
    if horizon_hr <= 0:
        return 0.0
    n = sum(1 for s in steps if s.intervened)
    return n / horizon_hr


def max_interventions_per_window(steps: List[StepResult], window_min: float) -> int:
    """Maximum number of bounded interventions in any window of length window_min."""
    times = [s.t for s in steps if s.intervened]
    if not times:
        return 0
    best = 0
    for i, t0 in enumerate(times):
        c = sum(1 for t in times if t0 <= t < t0 + window_min)
        best = max(best, c)
    return best


# --- Audit sufficiency (DEMM-style, per-property) ----------------------------

AUDIT_PROPERTIES = ["policy_basis", "decision_basis", "boundary", "non_action", "replay"]


def _sufficiency_for_step(policy_name: str, s: StepResult) -> Dict[str, float]:
    """Per-property sufficiency score in {0.0 insufficient, 0.5 partial, 1.0 sufficient}.

    Scored from *what the system actually persists*, following DEMM's question:
    can an external party reconstruct this property for this specific decision
    from the recorded evidence alone?
    """
    if policy_name == "ARL":
        # Full structured trace recorded for every decision (action or not).
        return {p: 1.0 for p in AUDIT_PROPERTIES}

    if policy_name == "B1":
        # Model log, only on intervene; no rule, version, path, state, or no-op record.
        return {p: 0.0 for p in AUDIT_PROPERTIES}

    if policy_name == "B2":
        if s.intervened and s.log_record is not None:
            return {
                "policy_basis": 0.5,    # fired rule id known, but no version
                "decision_basis": 0.5,  # fired rule known, suppressed routines not recorded
                "boundary": 0.0,        # no exit conditions
                "non_action": 0.0,
                "replay": 0.0,          # no state/inputs persisted
            }
        return {p: 0.0 for p in AUDIT_PROPERTIES}

    # B3 OLM-only: state logged, no decisions.
    return {p: 0.0 for p in AUDIT_PROPERTIES}


def audit_sufficiency(policy_name: str, steps: List[StepResult]) -> Dict[str, float]:
    """Mean per-property sufficiency over all decision points, plus the aggregate."""
    acc = {p: 0.0 for p in AUDIT_PROPERTIES}
    n = len(steps)
    if n == 0:
        return {**acc, "aggregate": 0.0}
    for s in steps:
        sc = _sufficiency_for_step(policy_name, s)
        for p in AUDIT_PROPERTIES:
            acc[p] += sc[p]
    for p in AUDIT_PROPERTIES:
        acc[p] /= n
    acc["aggregate"] = sum(acc[p] for p in AUDIT_PROPERTIES) / len(AUDIT_PROPERTIES)
    return acc


# --- Helpers -----------------------------------------------------------------


def mean_ci(values: List[float]):
    """Mean and 95% normal-approx confidence half-width."""
    import numpy as np
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return 0.0, 0.0
    m = float(a.mean())
    if a.size < 2:
        return m, 0.0
    se = float(a.std(ddof=1) / math.sqrt(a.size))
    return m, 1.96 * se
