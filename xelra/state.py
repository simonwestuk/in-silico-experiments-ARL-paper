"""Controller state and regulatory-mode constants for X-ELRA."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List


MODES = [
    "COLD_START",
    "ORIENTATION",
    "NOMINAL",
    "STRUGGLING",
    "LAPSED",
    "ACCELERATING",
    "CONSOLIDATING",
    "DIAGNOSTIC",
    "COOLDOWN",
]

# Learner-facing OLM labels for each regulatory mode.
MODE_LABELS = {
    "COLD_START": "Discovering",
    "ORIENTATION": "Exploring",
    "NOMINAL": "Progressing",
    "STRUGGLING": "Supporting",
    "LAPSED": "Reconnecting",
    "ACCELERATING": "Advancing",
    "CONSOLIDATING": "Reinforcing",
    "DIAGNOSTIC": "Assessing",
    "COOLDOWN": "Resting",
}


@dataclass(frozen=True)
class ControllerState:
    """Immutable controller state S_t.

    Frozen so that a state snapshot persisted in a decision trace cannot be
    mutated after the fact; state transitions return a fresh instance. This is
    what makes post-hoc replay a faithful re-execution of the recorded inputs.
    """

    mode: str = "COLD_START"
    mode_entered_at: float = 0.0  # minutes since timeline origin
    interventions_remaining: int = 5
    suggestions_remaining: int = 10
    # routine_id -> minute of last execution (for cooldown checks)
    cooldowns: Dict[str, float] = field(default_factory=dict)
    # minutes at which the most recent mode transitions occurred
    last_transition_times: List[float] = field(default_factory=list)
    # bookkeeping for budget-reset windows
    last_budget_reset: float = 0.0

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "mode_entered_at": round(self.mode_entered_at, 4),
            "budgets": {
                "interventions_remaining": self.interventions_remaining,
                "suggestions_remaining": self.suggestions_remaining,
            },
            "cooldowns": {k: round(v, 4) for k, v in sorted(self.cooldowns.items())},
            "stability_counters": {
                "mode_transitions": len(self.last_transition_times),
                "last_transition_times": [round(t, 4) for t in self.last_transition_times],
            },
        }

    def to_replay(self) -> dict:
        """Loss-less serialisation sufficient to re-execute the next decision."""
        return {
            "mode": self.mode,
            "mode_entered_at": self.mode_entered_at,
            "interventions_remaining": self.interventions_remaining,
            "suggestions_remaining": self.suggestions_remaining,
            "cooldowns": dict(self.cooldowns),
            "last_transition_times": list(self.last_transition_times),
            "last_budget_reset": self.last_budget_reset,
        }

    @classmethod
    def from_replay(cls, d: dict) -> "ControllerState":
        return cls(
            mode=d["mode"],
            mode_entered_at=float(d["mode_entered_at"]),
            interventions_remaining=int(d["interventions_remaining"]),
            suggestions_remaining=int(d["suggestions_remaining"]),
            cooldowns={k: float(v) for k, v in d.get("cooldowns", {}).items()},
            last_transition_times=[float(t) for t in d.get("last_transition_times", [])],
            last_budget_reset=float(d.get("last_budget_reset", 0.0)),
        )

    def copy_with(self, **changes) -> "ControllerState":
        # dataclasses.replace performs a shallow copy; deep-copy mutable fields
        if "cooldowns" not in changes:
            changes["cooldowns"] = dict(self.cooldowns)
        if "last_transition_times" not in changes:
            changes["last_transition_times"] = list(self.last_transition_times)
        return replace(self, **changes)
