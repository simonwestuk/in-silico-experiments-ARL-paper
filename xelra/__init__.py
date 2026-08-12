"""X-ELRA: a reference implementation of Agentic Regulated Learning (ARL).

This package provides a bounded, deterministic regulatory controller that
translates probabilistic perceptions and candidate actions into learner-facing
interventions (or deliberate non-interventions), emitting a structured,
replayable decision trace at every decision point.

Modules
-------
state       : ControllerState and mode constants.
dsl         : YAML routine-specification loader + guarded predicate evaluator.
modes        : deterministic, precedence-ordered mode inference.
controller  : the ARL perception--reasoning--action--evaluation control loop.
traces      : decision-trace construction, deterministic hashing, PROV-O export.
baselines   : B1 (direct-ML), B2 (rule-based ITS), B3 (OLM-only) comparators.
perception  : synthetic learner simulator with controllable perception noise.
metrics     : oscillation, predictability, audit-sufficiency, fairness metrics.
"""

from .state import ControllerState, MODES, MODE_LABELS
from .controller import ARLController
from .dsl import RoutineSet, load_routine_set

__all__ = [
    "ControllerState",
    "MODES",
    "MODE_LABELS",
    "ARLController",
    "RoutineSet",
    "load_routine_set",
]

__version__ = "1.0.0"
