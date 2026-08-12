"""Synthetic learner simulator producing perception streams with controllable noise.

The simulator generates an *open-loop* perception stream: a latent learner
trajectory plus an observation model whose noise level sigma is a free parameter.
The same stream is replayed to every policy (ARL and baselines), so any
difference in behaviour is attributable to the regulatory layer rather than to
different perceptions -- this is the matched-noise design the perturbation study
requires. (Open-loop means the policy's actions do not feed back into the latent
state; this isolates oscillation caused by perception noise from oscillation
caused by control, which is exactly what we want to measure.)

A perception snapshot is a plain dict of the fields the routine DSL references,
plus an ordered candidate list. Streams are fully determined by their seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


# Modes in which the model-driven baselines emit an intervention.
SPECIALIZED_MODES = {"STRUGGLING", "LAPSED", "ACCELERATING", "CONSOLIDATING", "DIAGNOSTIC"}


@dataclass
class Archetype:
    name: str
    base_lowest: float       # latent lowest-skill mastery at session start
    base_highest: float
    drift: float             # per-minute latent mastery drift
    affect_proneness: float  # base probability of a true affect episode
    clicks: int              # clicks in last 14 days (engagement)
    active_goals: bool
    progress_rate: float     # mastery gain per day
    days_since_engagement: float


ARCHETYPES = {
    # Latent lowest-mastery sits just ABOVE the STRUGGLING threshold (0.40) with
    # ~no drift, so STRUGGLING<->NOMINAL flips are driven purely by perception
    # noise crossing the boundary (mastery-driven oscillation).
    "boundary_mastery": Archetype("boundary_mastery", 0.45, 0.60, 0.0, 0.0, 0, True, 0.02, 1.0),
    # Mastery clearly in NOMINAL range; STRUGGLING is entered only via spurious
    # affect flags induced by perception noise (affect-driven oscillation).
    "boundary_affect": Archetype("boundary_affect", 0.55, 0.68, 0.0, 0.15, 6, True, 0.03, 0.5),
    # Stable progressing learner
    "progressing": Archetype("progressing", 0.58, 0.70, 0.0010, 0.04, 8, True, 0.03, 0.5),
    # Accelerating learner
    "accelerating": Archetype("accelerating", 0.62, 0.78, 0.0015, 0.03, 10, True, 0.08, 0.3),
    # Consolidating (near mastery)
    "consolidating": Archetype("consolidating", 0.80, 0.88, 0.0006, 0.03, 9, True, 0.02, 0.3),
    # Proficiency strata (low engagement so STRUGGLING is driven by the mastery
    # threshold), used for the fairness / disparate-impact analysis.
    "prof_low": Archetype("prof_low", 0.30, 0.45, 0.0, 0.06, 0, True, 0.02, 0.5),
    "prof_med": Archetype("prof_med", 0.45, 0.60, 0.0, 0.05, 0, True, 0.02, 0.5),
    "prof_high": Archetype("prof_high", 0.62, 0.78, 0.0, 0.03, 0, True, 0.02, 0.5),
}


def _affect_observation(rng: np.random.Generator, base: float, sigma: float) -> bool:
    """True affect episode observed through a noisy channel.

    Archetypes with zero affect-proneness have no affect dynamics and emit no
    flags (so a mastery-boundary archetype oscillates only through the mastery
    estimate). For affect-prone archetypes the true episode (Bernoulli(base)) is
    observed through a symmetric noisy channel whose flip probability grows with
    sigma, injecting spurious/missed flags -- a realistic driver of oscillation.
    """
    if base <= 0.0:
        return False
    true_aff = rng.random() < base
    flip = min(0.5, 1.6 * sigma)
    if rng.random() < flip:
        return not true_aff
    return true_aff


def make_stream(
    seed: int,
    archetype: str = "boundary_mastery",
    sigma: float = 0.10,
    n_steps: int = 80,
    dt_minutes: float = 1.5,
    n_candidates: int = 10,
    feature_gap_rate: float = 0.0,
    subgroup: str = "A",
) -> List[Tuple[float, Dict[str, Any], List[Dict[str, Any]]]]:
    """Generate a single learner's perception stream.

    Returns a list of (timestamp_minutes, perception, candidates).
    """
    rng = np.random.default_rng(seed)
    arch = ARCHETYPES[archetype]
    stream = []
    impressions = 0
    for i in range(n_steps):
        t = i * dt_minutes
        impressions += 1
        true_low = min(0.99, arch.base_lowest + arch.drift * t)
        true_high = min(0.99, arch.base_highest + arch.drift * t)
        obs_low = float(np.clip(true_low + rng.normal(0, sigma), 0.0, 1.0))
        obs_high = float(np.clip(true_high + rng.normal(0, sigma), 0.0, 1.0))
        confusion = _affect_observation(rng, arch.affect_proneness, sigma)
        frustration = _affect_observation(rng, arch.affect_proneness * 0.7, sigma)
        feature_gap = 3 if (feature_gap_rate > 0 and rng.random() < feature_gap_rate) else 0

        perception = {
            "lowest_mastery": obs_low,
            "highest_mastery": obs_high,
            "mean_mastery": float((obs_low + obs_high) / 2.0),
            "has_mastery": True,
            "mastery_count": 7,
            "impressions": impressions,
            "clicks_14d": int(arch.clicks),
            "days_since_engagement": float(arch.days_since_engagement),
            "progress_rate": float(arch.progress_rate),
            "recent_completions": 1 if archetype == "consolidating" else 0,
            "active_goals": bool(arch.active_goals),
            "confusion_flag": bool(confusion),
            "frustration_flag": bool(frustration),
            "feature_gap": int(feature_gap),
            "subgroup": subgroup,
        }
        cands = []
        for k in range(n_candidates):
            cands.append({
                "action_id": f"res_{k:03d}",
                "score": float(rng.random()),
                "source": "hybrid_recommender",
                "objective": "pedagogy",
            })
        stream.append((t, perception, cands))
    return stream


def make_need_episode_stream(
    seed: int,
    sigma: float = 0.12,
    n_steps: int = 80,
    dt_minutes: float = 1.5,
    episode_steps: int = 16,
    n_candidates: int = 10,
):
    """A learner with a single ground-truth 'genuine need' episode.

    The learner is NOMINAL, then truly struggles (low mastery, no recent clicks,
    active affect) for a contiguous window, then recovers. Returns the stream and
    the (onset, end) timestamps of the genuine-need window, so responsiveness
    (delay from onset to first support) and under-support can be measured against
    ground truth rather than against the noisy perception.
    """
    rng = np.random.default_rng(seed)
    onset = int(rng.integers(12, max(13, n_steps - episode_steps - 6)))
    end = onset + episode_steps
    stream = []
    for i in range(n_steps):
        t = i * dt_minutes
        in_need = onset <= i < end
        if in_need:
            true_low, true_high, clicks, aff = 0.20, 0.45, 0, 0.45
        else:
            true_low, true_high, clicks, aff = 0.60, 0.72, 5, 0.0
        obs_low = float(np.clip(true_low + rng.normal(0, sigma), 0.0, 1.0))
        obs_high = float(np.clip(true_high + rng.normal(0, sigma), 0.0, 1.0))
        perception = {
            "lowest_mastery": obs_low, "highest_mastery": obs_high,
            "mean_mastery": float((obs_low + obs_high) / 2.0), "has_mastery": True,
            "mastery_count": 7, "impressions": 12 + i, "clicks_14d": int(clicks),
            "days_since_engagement": 0.5, "progress_rate": 0.02, "recent_completions": 0,
            "active_goals": True,
            "confusion_flag": bool(_affect_observation(rng, aff, sigma)),
            "frustration_flag": bool(_affect_observation(rng, aff * 0.7, sigma)),
            "feature_gap": 0, "subgroup": "A",
        }
        cands = [{"action_id": f"res_{k:03d}", "score": float(rng.random()),
                  "source": "rec", "objective": "pedagogy"} for k in range(n_candidates)]
        stream.append((t, perception, cands))
    return stream, onset * dt_minutes, end * dt_minutes


def make_multi_need_stream(
    seed: int,
    sigma: float = 0.12,
    n_episodes: int = 4,
    episode_steps: int = 14,
    gap_steps: int = 22,
    dt_minutes: float = 1.5,
    n_candidates: int = 10,
):
    """A learner who struggles in several separated genuine-need episodes.

    Returns (stream, windows) where windows is a list of (onset_t, end_t). With
    several episodes over a multi-hour horizon, tight budgets or long cooldowns
    can leave later episodes under-supported, exposing the boundedness-adaptivity
    trade-off.
    """
    rng = np.random.default_rng(seed)
    onsets = [10 + e * (episode_steps + gap_steps) for e in range(n_episodes)]
    n_steps = onsets[-1] + episode_steps + 10
    windows = [(o * dt_minutes, (o + episode_steps) * dt_minutes) for o in onsets]
    need_steps = set()
    for o in onsets:
        need_steps.update(range(o, o + episode_steps))
    stream = []
    for i in range(n_steps):
        t = i * dt_minutes
        if i in need_steps:
            true_low, true_high, clicks, aff = 0.20, 0.45, 0, 0.45
        else:
            true_low, true_high, clicks, aff = 0.60, 0.72, 5, 0.0
        obs_low = float(np.clip(true_low + rng.normal(0, sigma), 0.0, 1.0))
        obs_high = float(np.clip(true_high + rng.normal(0, sigma), 0.0, 1.0))
        perception = {
            "lowest_mastery": obs_low, "highest_mastery": obs_high,
            "mean_mastery": float((obs_low + obs_high) / 2.0), "has_mastery": True,
            "mastery_count": 7, "impressions": 12 + i, "clicks_14d": int(clicks),
            "days_since_engagement": 0.5, "progress_rate": 0.02, "recent_completions": 0,
            "active_goals": True,
            "confusion_flag": bool(_affect_observation(rng, aff, sigma)),
            "frustration_flag": bool(_affect_observation(rng, aff * 0.7, sigma)),
            "feature_gap": 0, "subgroup": "A",
        }
        cands = [{"action_id": f"res_{k:03d}", "score": float(rng.random()),
                  "source": "rec", "objective": "pedagogy"} for k in range(n_candidates)]
        stream.append((t, perception, cands))
    return stream, windows


def make_population(
    seed: int,
    n_learners: int,
    sigma: float,
    archetype_weights: Dict[str, float] | None = None,
    n_steps: int = 80,
    feature_gap_rate: float = 0.0,
    subgroup: str = "A",
) -> List[List[Tuple[float, Dict[str, Any], List[Dict[str, Any]]]]]:
    """Generate a population of learner streams at a fixed noise level."""
    if archetype_weights is None:
        archetype_weights = {"boundary_mastery": 0.5, "boundary_affect": 0.5}
    rng = np.random.default_rng(seed)
    names = list(archetype_weights.keys())
    probs = np.array([archetype_weights[n] for n in names], dtype=float)
    probs = probs / probs.sum()
    pop = []
    for j in range(n_learners):
        arch = names[int(rng.choice(len(names), p=probs))]
        # Each learner gets an independent sub-seed for full reproducibility.
        sub_seed = int(rng.integers(0, 2**31 - 1))
        pop.append(make_stream(sub_seed, archetype=arch, sigma=sigma, n_steps=n_steps,
                               feature_gap_rate=feature_gap_rate, subgroup=subgroup))
    return pop
