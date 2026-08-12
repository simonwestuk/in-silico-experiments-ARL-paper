"""Decision-trace construction, deterministic hashing, and PROV-O export.

A decision trace T_t is the first-class explanation object: a structured,
versioned record of the routine evaluation path, the selected action (or
deliberate non-action), the state before/after, and the conditions monitored
next. The trace carries a deterministic content hash computed over its canonical
serialisation; re-executing the controller on the persisted (state, inputs) must
reproduce the same hash, which is how replay/audit is verified.

A PROV-O projection (W3C PROV Data Model) is provided so that traces interoperate
with provenance tooling: the decision is an Activity that ``used`` the perception
Entity and the routine-set Entity and ``wasAssociatedWith`` the controller Agent,
generating the action Entity and the trace Entity.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .state import ControllerState, MODE_LABELS


# Routine outcome vocabulary (paper Section 4.3).
SKIPPED = "SKIPPED"
BLOCKED = "BLOCKED"
EXECUTED_ACTION = "EXECUTED_ACTION"
EXECUTED_NO_ACTION = "EXECUTED_NO_ACTION"


@dataclass
class RoutineOutcome:
    routine_id: str
    outcome: str
    reason: str
    action_selected: Optional[str] = None
    mode_transition: Optional[str] = None
    terminated_evaluation: bool = False

    def to_dict(self) -> dict:
        return {
            "routine_id": self.routine_id,
            "outcome": self.outcome,
            "reason": self.reason,
            "action_selected": self.action_selected,
            "mode_transition": self.mode_transition,
            "terminated_evaluation": self.terminated_evaluation,
        }


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def trace_hash(payload: Dict[str, Any]) -> str:
    """Deterministic content hash over the replay-relevant payload."""
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


@dataclass
class DecisionTrace:
    learner_id: str
    timestamp: float
    routines_version: str
    state_before: Dict[str, Any]
    inputs_summary: Dict[str, Any]
    routine_path: List[RoutineOutcome]
    decision_action: Optional[str]
    decision_source_routine: Optional[str]
    is_bounded_intervention: bool
    state_after: Dict[str, Any]
    exit_conditions: List[Dict[str, str]]
    olm_projection: Dict[str, str]
    trace_id: str = ""

    def replay_payload(self) -> Dict[str, Any]:
        """The subset of the trace that a deterministic replay must reproduce."""
        return {
            "routines_version": self.routines_version,
            "state_before": self.state_before,
            "inputs": self.inputs_summary,
            "routine_path": [o.to_dict() for o in self.routine_path],
            "decision": {
                "action": self.decision_action,
                "source_routine": self.decision_source_routine,
            },
            "state_after": self.state_after,
        }

    def finalize(self) -> "DecisionTrace":
        self.trace_id = trace_hash(self.replay_payload())
        return self

    def to_dict(self) -> dict:
        return {
            "learner_id": self.learner_id,
            "trace_id": self.trace_id,
            "timestamp": round(self.timestamp, 4),
            "routines_version": self.routines_version,
            "state_before": self.state_before,
            "inputs_summary": self.inputs_summary,
            "routine_path": [o.to_dict() for o in self.routine_path],
            "decision": {
                "action": self.decision_action,
                "source_routine": self.decision_source_routine,
                "is_bounded_intervention": self.is_bounded_intervention,
            },
            "state_after": self.state_after,
            "exit_conditions": self.exit_conditions,
            "olm_projection": self.olm_projection,
        }

    # --- PROV-O projection ---------------------------------------------------

    def to_prov(self) -> dict:
        """Project the trace into a W3C PROV-O / PROV-DM JSON document.

        Entities: perception snapshot, routine set (plan), action, trace.
        Activity: the controller decision.
        Agent:    the ARL controller (acting on behalf of the course).
        """
        dec = f"decision:{self.trace_id[:12]}"
        perc = f"perception:{self.learner_id}:{round(self.timestamp,2)}"
        rset = f"routineset:{self.routines_version}"
        act = f"action:{self.decision_action}" if self.decision_action else "action:none"
        trc = f"trace:{self.trace_id[:12]}"
        return {
            "prefix": {
                "prov": "http://www.w3.org/ns/prov#",
                "arl": "https://example.org/arl#",
            },
            "entity": {
                perc: {"prov:type": "arl:PerceptionSnapshot"},
                rset: {"prov:type": "arl:RoutineSet", "arl:version": self.routines_version},
                act: {"prov:type": "arl:LearnerFacingAction"},
                trc: {"prov:type": "arl:DecisionTrace", "arl:trace_id": self.trace_id},
            },
            "activity": {
                dec: {
                    "prov:type": "arl:ControllerDecision",
                    "arl:mode": self.state_after.get("mode"),
                }
            },
            "agent": {"arl:controller": {"prov:type": "prov:SoftwareAgent"}},
            "used": {
                "_u1": {"prov:activity": dec, "prov:entity": perc},
                "_u2": {"prov:activity": dec, "prov:entity": rset, "prov:role": "arl:plan"},
            },
            "wasGeneratedBy": {
                "_g1": {"prov:entity": act, "prov:activity": dec},
                "_g2": {"prov:entity": trc, "prov:activity": dec},
            },
            "wasAssociatedWith": {
                "_a1": {
                    "prov:activity": dec,
                    "prov:agent": "arl:controller",
                    "prov:plan": rset,
                }
            },
        }


def build_olm_projection(mode: str, why: str, behaviour: str,
                         expected_action: str, exit_text: str) -> Dict[str, str]:
    return {
        "mode_label": MODE_LABELS.get(mode, mode),
        "why": why,
        "system_behaviour": behaviour,
        "expected_action": expected_action,
        "exit_conditions": exit_text,
    }
