"""Policy wrappers: ARL and the three baselines, over a shared interface.

Each policy consumes the same perception stream and produces a list of
``StepResult`` records. The crucial design point is that ARL and the baselines
differ *only* in the regulatory layer:

  ARL  -- full controller: mode hysteresis (dwell + oscillation), budgets,
          cooldowns, mode gating, and a structured decision trace per step.
  B1   -- direct-ML: the (noisy) perception is mapped to a stance every step and
          an intervention is emitted whenever the stance is specialised. No
          hysteresis, no budgets, no cooldowns, no trace (model log only).
  B2   -- rule-based ITS: same per-step stance as B1 (no hysteresis) but with
          per-rule cooldowns, and a rule-fired log (no evaluation path, no
          non-action record, no versioning).
  B3   -- OLM-only: displays learner state, never intervenes (state log only).

Holding perception and recommender fixed isolates the contribution of the
regulatory layer, as the evaluation design requires.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .controller import ARLController
from .dsl import RoutineSet
from .modes import infer_mode
from .perception import SPECIALIZED_MODES
from .state import ControllerState


@dataclass
class StepResult:
    t: float
    stance: str                       # the regulatory stance/mode at this step
    intervened: bool                  # a bounded intervention was emitted
    is_bounded: bool
    action: Optional[str]
    log_record: Dict[str, Any]        # what this system persists for audit
    trace: Optional[Dict[str, Any]] = None   # full decision trace (ARL only)
    mode_transition: Optional[str] = None


def _mode_action(rs: RoutineSet, mode: str) -> Optional[str]:
    """Highest-priority bounded action permitted in a mode (for B1/B2)."""
    best = None
    for r in rs.routines:
        if mode in r.permitted_modes and (r.interventions_cost > 0 or r.suggestions_cost > 0):
            if best is None or r.priority > best.priority:
                best = r
    return best.action if best else None


class ARLPolicy:
    name = "ARL"

    def __init__(self, rs: RoutineSet):
        self.rs = rs
        self.controller = ARLController(rs)

    def run(self, stream, learner_id="L") -> List[StepResult]:
        # Warm-start the initial mode from the first materialised snapshot so the
        # controller does not pay an artificial COLD_START -> mode transition.
        t0, perc0, cands0 = stream[0]
        ns0 = dict(perc0); ns0["n_candidates"] = len(cands0)
        state = ControllerState(
            mode=infer_mode(self.rs, ns0),
            mode_entered_at=t0,
            interventions_remaining=self.rs.budget_interventions,
            suggestions_remaining=self.rs.budget_suggestions,
            last_budget_reset=t0,
        )
        out = []
        for (t, perc, cands) in stream:
            decision, trace, state = self.controller.decide(state, perc, cands, t, learner_id)
            out.append(StepResult(
                t=t,
                stance=decision["mode"],
                intervened=bool(decision["is_bounded_intervention"]),
                is_bounded=bool(decision["is_bounded_intervention"]),
                action=decision["action"],
                log_record=trace.to_dict(),     # ARL persists the full trace
                trace=trace.to_dict(),
                mode_transition=decision["mode_transition"],
            ))
        return out


class B1DirectML:
    name = "B1"

    def __init__(self, rs: RoutineSet):
        self.rs = rs

    def run(self, stream, learner_id="L") -> List[StepResult]:
        out = []
        prev = None
        for (t, perc, cands) in stream:
            ns = dict(perc); ns["n_candidates"] = len(cands)
            stance = infer_mode(self.rs, ns)
            intervened = stance in SPECIALIZED_MODES
            action = _mode_action(self.rs, stance) if intervened else None
            top = max((c["score"] for c in cands), default=0.0)
            rec = {"t": t, "learner": learner_id, "action": action,
                   "top_score": round(top, 4)} if intervened else None
            out.append(StepResult(t, stance, intervened, False, action,
                                  log_record=rec))
            prev = stance
        return out


class B2RuleITS:
    name = "B2"

    def __init__(self, rs: RoutineSet):
        self.rs = rs
        # cooldown per mode-rule, taken from the corresponding bounded routine
        self.cooldowns = {}
        for m in SPECIALIZED_MODES:
            best = None
            for r in rs.routines:
                if m in r.permitted_modes and (r.interventions_cost > 0 or r.suggestions_cost > 0):
                    if best is None or r.priority > best.priority:
                        best = r
            self.cooldowns[m] = best.cooldown_minutes if best else 0.0

    def run(self, stream, learner_id="L") -> List[StepResult]:
        out = []
        last_fire = {}
        for (t, perc, cands) in stream:
            ns = dict(perc); ns["n_candidates"] = len(cands)
            stance = infer_mode(self.rs, ns)
            intervened = False
            action = None
            if stance in SPECIALIZED_MODES:
                cd = self.cooldowns.get(stance, 0.0)
                last = last_fire.get(stance)
                if last is None or (t - last) >= cd:
                    intervened = True
                    action = _mode_action(self.rs, stance)
                    last_fire[stance] = t
            rec = {"t": t, "learner": learner_id, "rule_id": stance,
                   "action": action} if intervened else None
            out.append(StepResult(t, stance, intervened, intervened, action,
                                  log_record=rec))
        return out


class B3OLMOnly:
    name = "B3"

    def __init__(self, rs: RoutineSet):
        self.rs = rs

    def run(self, stream, learner_id="L") -> List[StepResult]:
        out = []
        for (t, perc, cands) in stream:
            # OLM-only displays learner state; no regulatory stance, no action.
            rec = {"t": t, "learner": learner_id,
                   "mastery_min": round(perc["lowest_mastery"], 4),
                   "mastery_max": round(perc["highest_mastery"], 4)}
            out.append(StepResult(t, "STATE_ONLY", False, False, None, log_record=rec))
        return out


class B5SmoothedML:
    """A stabilised direct-ML policy: EMA-smoothed perceptions, a debounced stance
    switch, and a refractory period between interventions.

    This stands in for an interpretable controller with smoothing / a stabilised
    policy (the family the reviewer asks for: smoothed or hysteresis-equipped
    controllers). It reduces oscillation relative to B1 through signal smoothing
    and stance debouncing, but, unlike ARL, it has no explicit mode automaton,
    budgets, or structured decision trace -- so it cannot bound intervention
    density by window or support per-decision audit.
    """

    name = "B5"

    def __init__(self, rs: RoutineSet, alpha: float = 0.4, debounce: int = 3,
                 refractory_min: float = 8.0):
        self.rs = rs
        self.alpha = alpha
        self.debounce = debounce
        self.refractory = refractory_min

    def run(self, stream, learner_id="L") -> List[StepResult]:
        out = []
        ema_low = None
        ema_conf = 0.0
        ema_frus = 0.0
        cur_stance = None
        cand_stance = None
        cand_count = 0
        last_fire = -1e9
        for (t, perc, cands) in stream:
            lm = perc["lowest_mastery"]
            ema_low = lm if ema_low is None else self.alpha * lm + (1 - self.alpha) * ema_low
            cf = 1.0 if perc["confusion_flag"] else 0.0
            ema_conf = self.alpha * cf + (1 - self.alpha) * ema_conf
            fr = 1.0 if perc["frustration_flag"] else 0.0
            ema_frus = self.alpha * fr + (1 - self.alpha) * ema_frus
            ns = dict(perc)
            ns["lowest_mastery"] = ema_low
            ns["confusion_flag"] = ema_conf > 0.5
            ns["frustration_flag"] = ema_frus > 0.5
            ns["n_candidates"] = len(cands)
            target = infer_mode(self.rs, ns)
            # Debounce: only adopt a new stance after it persists `debounce` steps.
            if cur_stance is None:
                cur_stance = target
            elif target == cur_stance:
                cand_stance, cand_count = None, 0
            else:
                if target == cand_stance:
                    cand_count += 1
                else:
                    cand_stance, cand_count = target, 1
                if cand_count >= self.debounce:
                    cur_stance, cand_stance, cand_count = target, None, 0
            stance = cur_stance
            intervened = False
            action = None
            if stance in SPECIALIZED_MODES and (t - last_fire) >= self.refractory:
                intervened = True
                action = _mode_action(self.rs, stance)
                last_fire = t
            rec = {"t": t, "learner": learner_id, "action": action} if intervened else None
            out.append(StepResult(t, stance, intervened, intervened, action, log_record=rec))
        return out


def routine_set_without_anti_oscillation(rs: RoutineSet) -> RoutineSet:
    """Return a copy of a routine set with the anti-oscillation mechanism removed.

    Sets the minimum dwell time to zero and the oscillation threshold to
    effectively infinite, so the controller re-infers its mode every step with no
    hysteresis. Used for the ARL ablation that isolates the contribution of the
    dwell-time and oscillation-detection mechanisms (boundedness, trace, and mode
    gating are retained)."""
    import copy
    rs2 = copy.copy(rs)
    rs2.dwell_minutes = 0.0
    rs2.oscillation_k = 10 ** 9
    return rs2


def make_policies(rs: RoutineSet) -> Dict[str, Any]:
    return {
        "ARL": ARLPolicy(rs),
        "B1": B1DirectML(rs),
        "B2": B2RuleITS(rs),
        "B3": B3OLMOnly(rs),
        "B5": B5SmoothedML(rs),
        "ARL_noAO": ARLPolicy(routine_set_without_anti_oscillation(rs)),
    }
