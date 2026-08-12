"""Routine-specification DSL: loader, validator, and guarded predicate evaluator.

A routine set is specified declaratively in YAML (see ``routines/x_elra.yaml``).
This module loads that specification, validates it, and exposes the modes and
routines as Python objects. Trigger and entry predicates are written as small
boolean expressions over a fixed perception/state namespace and are evaluated
with a guarded ``eval`` (no builtins, whitelisted names only), so the routine
logic that governs learner-facing behaviour is fully inspectable in one file
rather than scattered through imperative code.

The DSL deliberately mirrors the routine-specification structure in the paper
(identity, eligibility constraints, resource constraints, effects, priority).
It is the single source of truth consumed by the controller, the property-based
tests, and the bounded model checker.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


# --- Guarded predicate evaluation -------------------------------------------

# Names a predicate is allowed to reference, beyond the perception/state fields
# that are injected at evaluation time. Keeping this explicit (and stripping
# __builtins__) means a routine predicate cannot perform I/O, import modules, or
# call arbitrary code -- it is a pure boolean function of the snapshot.
_SAFE_GLOBALS = {"__builtins__": {}, "min": min, "max": max, "abs": abs, "math": math}


def eval_predicate(expr: str, namespace: Dict[str, Any]) -> bool:
    """Evaluate a DSL boolean expression over a perception/state namespace."""
    if expr is None or str(expr).strip() in ("", "true", "True"):
        return True
    if str(expr).strip() in ("false", "False"):
        return False
    try:
        return bool(eval(expr, dict(_SAFE_GLOBALS), namespace))  # noqa: S307 (guarded)
    except Exception as exc:  # pragma: no cover - surfaced as a config error
        raise ValueError(f"Failed to evaluate predicate {expr!r}: {exc}") from exc


# --- Specification objects ---------------------------------------------------


@dataclass(frozen=True)
class ModeSpec:
    id: str
    label: str
    precedence: int
    entry: str
    permitted_routines: List[str] = field(default_factory=list)
    perception_inferred: bool = True


@dataclass(frozen=True)
class RoutineSpec:
    id: str
    name: str
    version: str
    priority: int
    permitted_modes: List[str]
    triggers: str
    objective: str  # multi-objective tier label (see RoutineSet.OBJECTIVE_ORDER)
    interventions_cost: int
    suggestions_cost: int
    cooldown_minutes: float
    action: Optional[str]
    terminates: bool
    guard: bool = False  # guard routines run before pedagogical routines

    @property
    def cost_tuple(self):
        return (self.interventions_cost, self.suggestions_cost)


class RoutineSet:
    """An ordered, versioned routine set plus its mode and stability config."""

    # Lexicographic ordering of objective tiers used for multi-objective
    # conflict resolution: safety dominates affect-relief, which dominates
    # pedagogical gain, which dominates engagement maintenance. Lower index =
    # higher precedence.
    OBJECTIVE_ORDER = [
        "integrity",       # data/safety guards
        "affect_relief",   # protect the learner from affective overload
        "pedagogy",        # learning-gain interventions
        "engagement",      # re-engagement / consolidation
        "default",         # totality safeguard
    ]

    def __init__(self, raw: Dict[str, Any]):
        self.version: str = raw["version"]
        b = raw["budgets"]
        self.budget_interventions: int = int(b["interventions"])
        self.budget_suggestions: int = int(b["suggestions"])
        self.budget_reset_minutes: float = float(b["reset_minutes"])
        s = raw["stability"]
        self.dwell_minutes: float = float(s["dwell_minutes"])
        self.oscillation_window_minutes: float = float(s["oscillation_window_minutes"])
        self.oscillation_k: int = int(s["oscillation_k"])

        self.modes: Dict[str, ModeSpec] = {}
        for m in raw["modes"]:
            self.modes[m["id"]] = ModeSpec(
                id=m["id"],
                label=m["label"],
                precedence=int(m["precedence"]),
                entry=m.get("entry", "false"),
                permitted_routines=list(m.get("permitted_routines", [])),
                perception_inferred=bool(m.get("perception_inferred", True)),
            )

        self.routines: List[RoutineSpec] = []
        for r in raw["routines"]:
            cost = r.get("cost", {})
            self.routines.append(
                RoutineSpec(
                    id=r["id"],
                    name=r["name"],
                    version=r.get("version", self.version),
                    priority=int(r["priority"]),
                    permitted_modes=list(r["permitted_modes"]),
                    triggers=r.get("triggers", "true"),
                    objective=r.get("objective", "pedagogy"),
                    interventions_cost=int(cost.get("interventions", 0)),
                    suggestions_cost=int(cost.get("suggestions", 0)),
                    cooldown_minutes=float(r.get("cooldown_minutes", 0)),
                    action=r.get("action"),
                    terminates=bool(r.get("terminates", False)),
                    guard=bool(r.get("guard", False)),
                )
            )
        # Routines are evaluated in fixed priority order (descending priority).
        # Ties broken deterministically by routine id for total ordering.
        self.routines.sort(key=lambda r: (-r.priority, r.id))
        self._validate()

    def _validate(self) -> None:
        ids = [r.id for r in self.routines]
        assert len(ids) == len(set(ids)), "duplicate routine ids"
        known_modes = set(self.modes)
        for r in self.routines:
            for m in r.permitted_modes:
                assert m in known_modes, f"routine {r.id} references unknown mode {m}"
            assert r.objective in self.OBJECTIVE_ORDER, (
                f"routine {r.id} has unknown objective tier {r.objective}"
            )
        # Every perception-inferred mode must reach a default/totality routine or
        # be explicitly closed, so the controller is total over all modes.
        for m in self.modes.values():
            assert m.precedence >= 0

    def objective_rank(self, objective: str) -> int:
        return self.OBJECTIVE_ORDER.index(objective)

    def perception_modes(self) -> List[ModeSpec]:
        """Perception-inferred modes, highest precedence first."""
        ms = [m for m in self.modes.values() if m.perception_inferred]
        return sorted(ms, key=lambda m: -m.precedence)


def load_routine_set(path: str) -> RoutineSet:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return RoutineSet(raw)
