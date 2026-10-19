# ARL standalone implementation and in-silico evaluation suite (X-ELRA routine set)

This repository is the standalone implementation of Agentic Regulated Learning
(ARL) and the reproducible evaluation suite for the paper *Agentic Regulated
Learning: A Bounded, Deterministic Governance Architecture for Process-Level
Explainability in Adaptive Learning Systems* (West, Yang and Liu). It provides a
bounded, deterministic regulatory controller that translates probabilistic
perceptions and candidate actions into learner-facing interventions (or
deliberate non-interventions), emitting a structured, replayable decision trace
at every decision point. Every in-silico, property-based and model-checking
result in the paper is produced by this code.

The controller is **driven entirely by a declarative routine DSL**
(`routines/x_elra.yaml`), which carries the same nine regulatory modes, eight
control routines, priorities, mode gating, budgets, cooldowns and stability
parameters as the X-ELRA platform deployed in the paper's feasibility study
(hence the file name), with simplified trigger predicates, abstract action
labels and this implementation's own mode-precedence order. The DSL is consumed
by the controller, the property tests, and the bounded model checker.

## Relationship to the deployed X-ELRA platform

This is a separate codebase from the X-ELRA platform whose live deployment is
reported in Section 5.6 of the paper. That platform, and the thesis evaluation
that exercises its deployed governance code under the same simulator and
comparator policies as this suite, are archived separately at
<https://github.com/simonwestuk/x-elra-release> (Zenodo
10.5281/zenodo.21916376). The two suites share the simulator, perception
streams, seeds, comparator baselines and metrics; they differ only in the
controller under test. This implementation follows the ARL specification as
written (oscillation override fires at κ = k, windowed budget reset); the
deployed platform's implementation choices are documented in the thesis.

## Scope

This codebase is a research implementation built for controlled, reproducible
evaluation: everything here runs on synthetic data, and the studies
characterise the behaviour of the ARL design itself. It is dependency-free by
design (no service layer, persistence, or telemetry), so it is not a
production system. Where implementation policy is under-determined by the ARL
specification (for example the budget-reset policy, mode-inference precedence,
and dwell anchoring), this codebase makes explicit, documented choices, and its
results should be read as claims about the ARL design under those choices.
