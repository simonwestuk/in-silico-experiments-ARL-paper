"""Deterministic, precedence-ordered regulatory-mode inference.

Given a perception snapshot (already materialised and serialised), the inferred
mode is the highest-precedence mode whose entry predicate is satisfied. The
function is a pure mapping from the perception namespace to a mode id, so two
identical snapshots always infer the same mode -- a precondition for the
controller's determinism contract.
"""

from __future__ import annotations

from typing import Dict, Any

from .dsl import RoutineSet, eval_predicate


def infer_mode(routine_set: RoutineSet, namespace: Dict[str, Any]) -> str:
    """Return the perception-inferred mode for a snapshot.

    Modes are tested in descending precedence order; the first satisfied entry
    predicate wins. NOMINAL has entry ``true`` and lowest precedence, so the
    function is total (it always returns a mode).
    """
    for mode in routine_set.perception_modes():
        if eval_predicate(mode.entry, namespace):
            return mode.id
    return "NOMINAL"
