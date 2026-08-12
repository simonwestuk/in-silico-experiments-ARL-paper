"""The ARL regulatory controller: a bounded, deterministic control loop.

Implements Algorithm 1 from the paper as a pure function of the persisted
controller state, the materialised perception snapshot, the ordered candidate
set, and the versioned routine set. The four phases are:

  1. Perception  -- serialise inputs; infer target mode; apply mode-transition
                    stability gating (dwell time + oscillation -> COOLDOWN).
  2. Reasoning   -- evaluate routines in fixed priority order with mode gating,
                    trigger predicates, and boundedness (budgets + cooldowns),
                    recording SKIPPED / BLOCKED / EXECUTED_ACTION /
                    EXECUTED_NO_ACTION outcomes.
  3. Action      -- default selection only if no routine executed (totality
                    safeguard).
  4. Evaluation  -- update state; compute monitored exit conditions; emit the
                    structured decision trace with a deterministic content hash.

Determinism: for a fixed routine-set version, identical (state, inputs) yield an
identical routine path, decision, next state, and trace hash. Concurrency and
partial-signal handling are addressed at the boundary (see ``decide`` docstring
and ``guard`` routines): the controller operates on a single materialised
snapshot, so late-arriving signals are incorporated at the next decision point
rather than mutating an in-flight decision.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .dsl import RoutineSet, eval_predicate
from .modes import infer_mode
from .state import ControllerState, MODE_LABELS
from .traces import (
    DecisionTrace,
    RoutineOutcome,
    SKIPPED,
    BLOCKED,
    EXECUTED_ACTION,
    EXECUTED_NO_ACTION,
    build_olm_projection,
)


class ARLController:
    def __init__(self, routine_set: RoutineSet):
        self.rs = routine_set

    # --- public API ----------------------------------------------------------

    def decide(
        self,
        state: ControllerState,
        perception: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        timestamp: float,
        learner_id: str = "learner",
        emit_trace: bool = True,
    ) -> Tuple[Dict[str, Any], DecisionTrace, ControllerState]:
        """Run one perception--reasoning--action--evaluation cycle.

        ``perception`` is a fully materialised snapshot (a dict of the fields the
        DSL predicates reference). ``candidates`` is an ordered list of
        ``{action_id, score, source, objective}`` dicts; the controller sorts
        them canonically so that the decision is invariant to candidate arrival
        order (relevant when several upstream agents contribute suggestions).
        """
        rs = self.rs

        # Canonical, order-invariant candidate ordering: score desc, then id asc.
        cands = sorted(
            candidates, key=lambda c: (-float(c.get("score", 0.0)), str(c.get("action_id")))
        )
        namespace = dict(perception)
        namespace["n_candidates"] = len(cands)

        # ---- Phase 1: budget reset + mode-transition stability gating --------
        s = self._maybe_reset_budgets(state, timestamp)
        target_mode = infer_mode(rs, namespace)
        s, mode_transition = self._apply_mode_gating(s, target_mode, timestamp)
        current_mode = s.mode

        # ---- Phase 2: evaluate ordered routines ------------------------------
        path: List[RoutineOutcome] = []
        chosen_action: Optional[str] = None
        chosen_routine: Optional[str] = None
        is_bounded = False
        acted = False

        for r in rs.routines:
            if current_mode not in r.permitted_modes:
                path.append(RoutineOutcome(r.id, SKIPPED, "mode_mismatch"))
                continue
            if not eval_predicate(r.triggers, namespace):
                path.append(RoutineOutcome(r.id, SKIPPED, "triggers_not_met"))
                continue
            ok, reason = self._check_boundedness(r, s, timestamp)
            if not ok:
                path.append(RoutineOutcome(r.id, BLOCKED, reason))
                continue
            # Execute.
            action = r.action
            if action is None:  # deliberate non-intervention routine
                path.append(
                    RoutineOutcome(r.id, EXECUTED_NO_ACTION, "executed_no_action",
                                   terminated_evaluation=r.terminates)
                )
            else:
                chosen_action = action
                chosen_routine = r.id
                is_bounded = (r.interventions_cost > 0 or r.suggestions_cost > 0)
                s = self._consume(s, r, timestamp)
                path.append(
                    RoutineOutcome(r.id, EXECUTED_ACTION, "triggers_met_and_bounded",
                                   action_selected=action,
                                   terminated_evaluation=r.terminates)
                )
            acted = True
            if r.terminates:
                break

        # ---- Phase 3: default selection (totality safeguard) -----------------
        if not acted:
            if cands:
                chosen_action = "recommend_next"
                chosen_routine = "DEFAULT"
                is_bounded = False
            else:
                chosen_action = None  # deliberate non-intervention (empty action space)
                chosen_routine = None

        # ---- Phase 4: state update, exit conditions, trace -------------------
        new_state = s
        decision = {
            "action": chosen_action,
            "source_routine": chosen_routine,
            "is_bounded_intervention": is_bounded,
            "mode": current_mode,
            "mode_transition": mode_transition,
        }
        if not emit_trace:
            # Fast path for state-space exploration / model checking: skip the
            # (hashed) trace construction. State transition is identical.
            return decision, None, new_state

        exit_conditions = self._exit_conditions(current_mode)
        olm = self._project_olm(current_mode, namespace)

        trace = DecisionTrace(
            learner_id=learner_id,
            timestamp=timestamp,
            routines_version=rs.version,
            state_before=state.to_dict(),
            inputs_summary=self._inputs_summary(namespace, cands),
            routine_path=path,
            decision_action=chosen_action,
            decision_source_routine=chosen_routine,
            is_bounded_intervention=is_bounded,
            state_after=new_state.to_dict(),
            exit_conditions=exit_conditions,
            olm_projection=olm,
        ).finalize()

        return decision, trace, new_state

    # --- Phase 1 helpers -----------------------------------------------------

    def _maybe_reset_budgets(self, state: ControllerState, t: float) -> ControllerState:
        if t - state.last_budget_reset >= self.rs.budget_reset_minutes:
            return state.copy_with(
                interventions_remaining=self.rs.budget_interventions,
                suggestions_remaining=self.rs.budget_suggestions,
                last_budget_reset=t,
            )
        return state

    def _apply_mode_gating(
        self, state: ControllerState, target: str, t: float
    ) -> Tuple[ControllerState, Optional[str]]:
        """Apply dwell-time and oscillation gating to a desired transition.

        Returns the (possibly unchanged) state and a human-readable description
        of the transition that occurred (or None).
        """
        current = state.mode
        window = self.rs.oscillation_window_minutes
        recent = [tt for tt in state.last_transition_times if tt >= t - window]

        if target == current:
            return state, None

        # If currently resting in COOLDOWN, stay until stability is restored:
        # the minimum dwell has elapsed AND recent transitions have aged out of
        # the rolling window (fewer than k). This lets COOLDOWN absorb a burst of
        # noise-driven flips rather than ping-ponging.
        if current == "COOLDOWN":
            if (t - state.mode_entered_at) < self.rs.dwell_minutes:
                return state, None
            if len(recent) >= self.rs.oscillation_k:
                return state, None
            new = state.copy_with(
                mode=target, mode_entered_at=t, last_transition_times=recent + [t]
            )
            return new, f"COOLDOWN->{target}"

        # Oscillation guard: too many recent transitions -> forced COOLDOWN.
        if len(recent) >= self.rs.oscillation_k:
            new = state.copy_with(
                mode="COOLDOWN", mode_entered_at=t, last_transition_times=recent + [t]
            )
            return new, f"{current}->COOLDOWN (oscillation)"

        # Dwell-time guard: minimum dwell before any ordinary transition.
        if (t - state.mode_entered_at) < self.rs.dwell_minutes:
            return state, None

        # Permitted transition.
        new = state.copy_with(
            mode=target, mode_entered_at=t, last_transition_times=recent + [t]
        )
        return new, f"{current}->{target}"

    # --- Phase 2 helpers -----------------------------------------------------

    def _check_boundedness(
        self, routine, state: ControllerState, t: float
    ) -> Tuple[bool, str]:
        if state.interventions_remaining < routine.interventions_cost:
            return False, "budget_interventions"
        if state.suggestions_remaining < routine.suggestions_cost:
            return False, "budget_suggestions"
        last = state.cooldowns.get(routine.id)
        if last is not None and (t - last) < routine.cooldown_minutes:
            return False, "cooldown"
        return True, ""

    def _consume(self, state: ControllerState, routine, t: float) -> ControllerState:
        cds = dict(state.cooldowns)
        cds[routine.id] = t
        return state.copy_with(
            interventions_remaining=state.interventions_remaining - routine.interventions_cost,
            suggestions_remaining=state.suggestions_remaining - routine.suggestions_cost,
            cooldowns=cds,
        )

    # --- Phase 4 helpers -----------------------------------------------------

    def _exit_conditions(self, mode: str) -> List[Dict[str, str]]:
        table = {
            "STRUGGLING": [
                {"condition": "mastery_min > 0.5", "monitored_field": "mastery"},
                {"condition": "clicks_14d >= 3", "monitored_field": "engagement"},
            ],
            "LAPSED": [{"condition": "clicks_14d >= 1", "monitored_field": "engagement"}],
            "ACCELERATING": [{"condition": "progress_rate < 0.05", "monitored_field": "progress"}],
            "CONSOLIDATING": [{"condition": "highest_mastery < 0.85", "monitored_field": "mastery"}],
            "COOLDOWN": [{"condition": "mode_transitions_in_window <= k", "monitored_field": "stability"}],
            "DIAGNOSTIC": [{"condition": "feature_gap <= 2", "monitored_field": "data_integrity"}],
        }
        return table.get(mode, [{"condition": "mode_entry_conditions_change",
                                 "monitored_field": "perception"}])

    def _project_olm(self, mode: str, ns: Dict[str, Any]) -> Dict[str, str]:
        why = {
            "STRUGGLING": "You may benefit from additional support where progress has been slower.",
            "LAPSED": "It has been a while since your last session; let's reconnect.",
            "ACCELERATING": "You are making strong progress and can take on more.",
            "CONSOLIDATING": "You are close to mastering this; let's reinforce it.",
            "DIAGNOSTIC": "Some signals are missing, so the system is checking before acting.",
            "COOLDOWN": "The system is holding steady to avoid switching support too often.",
            "NOMINAL": "You are progressing as expected.",
            "ORIENTATION": "You are getting started; the system is helping you explore.",
            "COLD_START": "Welcome -- the system is learning what you need.",
        }.get(mode, "You are progressing as expected.")
        behaviour = {
            "STRUGGLING": "It will offer targeted, bounded support and avoid overloading you.",
            "ACCELERATING": "It will suggest stretch goals while you keep momentum.",
            "CONSOLIDATING": "It will suggest review and consolidation activities.",
        }.get(mode, "It will recommend a suitable next resource.")
        expected = {
            "STRUGGLING": "Review the suggested support and try the next problem.",
            "ACCELERATING": "Take on the suggested stretch goal when ready.",
        }.get(mode, "Continue with the recommended resource.")
        exit_text = {
            "STRUGGLING": "Once you show progress, support will adjust.",
            "ACCELERATING": "If your pace slows, the system will rebalance.",
            "COOLDOWN": "Once behaviour stabilises, normal support resumes.",
        }.get(mode, "The system will adapt as your activity changes.")
        return build_olm_projection(mode, why, behaviour, expected, exit_text)

    def _inputs_summary(self, ns: Dict[str, Any], cands: List[Dict[str, Any]]) -> Dict[str, Any]:
        eng = "active"
        if ns.get("days_since_engagement", 0) > 15:
            eng = "lapsed"
        elif ns.get("clicks_14d", 0) == 0:
            eng = "declining"
        return {
            "mastery_range": [round(float(ns.get("lowest_mastery", 0.0)), 3),
                              round(float(ns.get("highest_mastery", 0.0)), 3)],
            "engagement_level": eng,
            "confusion_flag": bool(ns.get("confusion_flag", False)),
            "frustration_flag": bool(ns.get("frustration_flag", False)),
            "feature_gap": int(ns.get("feature_gap", 0)),
            "candidates_count": len(cands),
        }
