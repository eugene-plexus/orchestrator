"""Neurotransmitter state — evolution and bicameral-loop modulation.

The NT system tracks six neurotransmitters as `{level, baseline, decay}`
triples. Between chat turns the level decays exponentially toward
baseline. Each turn's `Observations` apply impulses to specific NTs,
and the new levels modulate the next turn's bicameral parameters:

  - **max_passes** = f(cortisol, norepinephrine) — anxious / alert
    Eugene allows more deliberation; calm Eugene commits sooner.
  - **temperature** = f(dopamine, GABA) — dopamine pushes toward
    exploration / creativity; GABA pulls toward determinism.

Observation → NT impulse map (locked v0.2):

  - high agreement on the final pass → dopamine up
  - low agreement / multiple passes → cortisol up
  - quick single-pass convergence → GABA up
  - long average pass latency → norepinephrine up
  - acetylcholine (novelty / attention) and serotonin (mood / steady
    state) accumulate from baseline decay only in v0.2 — the topic-
    shift detector that would drive ACh lands in v0.3.

State persistence: in-memory only. A restart resets to neutral state,
matching the "Eugene cools down after a reboot" anatomy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime

from .._generated.models import NTLevel, NTState

# Configurable per-NT defaults. Each NT has its own decay rate because
# they don't all settle on the same timescale in vivo: dopamine and
# norepinephrine are fast; serotonin is the slow baseline regulator;
# cortisol persists for many minutes; GABA / ACh are intermediate.
# Decays expressed as "natural log decay constant per second" — the
# half-life is ln(2) / decay.
_DEFAULT_DECAYS: dict[str, float] = {
    "dopamine": 0.010,  # ~70s half-life
    "serotonin": 0.001,  # ~11min half-life
    "norepinephrine": 0.020,  # ~35s half-life
    "gaba": 0.005,  # ~2.3min half-life
    "cortisol": 0.002,  # ~5.7min half-life
    "acetylcholine": 0.010,  # ~70s half-life
}


def _neutral(name: str) -> NTLevel:
    return NTLevel(level=0.5, baseline=0.5, decay=_DEFAULT_DECAYS[name])


def neutral_state() -> NTState:
    """A baseline NT state: every NT at 0.5, baseline 0.5, with per-NT decay."""
    return NTState(
        lastUpdated=datetime.now(UTC),
        dopamine=_neutral("dopamine"),
        serotonin=_neutral("serotonin"),
        norepinephrine=_neutral("norepinephrine"),
        gaba=_neutral("gaba"),
        cortisol=_neutral("cortisol"),
        acetylcholine=_neutral("acetylcholine"),
    )


@dataclass(frozen=True)
class Observations:
    """Observables collected from a finished bicameral turn.

    The chat handler builds one of these from `BicameralOutcome` and
    feeds it to `tick()`; the NT system converts it into per-NT
    impulses. v0.3 will add topic-shift signals + connector-side
    metadata to this struct.
    """

    final_agreement: float
    """Corpus-callosum agreement score on the FINAL pass, in [0, 1]."""

    pass_count: int
    """How many passes ran. 1 = quick convergence; >1 = divergence."""

    agreement_threshold: float
    """Configured agreement bar. Used to decide if `final_agreement`
    represents a true convergence or a cap-reached termination."""

    avg_pass_latency_ms: float
    """Mean of (max over hemispheres of latencyMs) across passes."""


@dataclass(frozen=True)
class _Impulse:
    dopamine: float = 0.0
    serotonin: float = 0.0
    norepinephrine: float = 0.0
    gaba: float = 0.0
    cortisol: float = 0.0
    acetylcholine: float = 0.0


# Latency normalization. Past 30s the NE impulse saturates; below this
# range latency contributes proportionally to the boost.
_LATENCY_SATURATION_MS = 30_000.0


def _impulses_from_observations(obs: Observations) -> _Impulse:
    """Map a turn's `Observations` to per-NT impulses in [-0.5, +0.5].

    Each impulse is a one-shot nudge layered on top of the level after
    decay. Magnitudes are kept modest so single turns don't slam the
    state to extremes — the dynamics emerge from repeated observations
    over many turns.
    """
    # Agreement → dopamine. Centered at the agreement threshold so
    # "barely converged" produces no impulse and "perfect agreement"
    # is strongly rewarding.
    above_threshold = obs.final_agreement - obs.agreement_threshold
    dopamine = max(-0.3, min(0.3, above_threshold * 0.6))

    # Pass count → cortisol / GABA.
    # Quick convergence (1 pass with agreement above threshold) = GABA up.
    # Multi-pass with divergence = cortisol up.
    if obs.pass_count == 1 and obs.final_agreement >= obs.agreement_threshold:
        gaba = 0.15
        cortisol = -0.05  # mild relaxation
    elif obs.pass_count >= 2:
        # Each additional pass adds cortisol stress, saturating at
        # +0.3 by ~5 passes.
        cortisol = min(0.3, (obs.pass_count - 1) * 0.08)
        gaba = -0.05
    else:
        gaba = 0.0
        cortisol = 0.0

    # Latency → norepinephrine. Long latency = pay-attention nudge.
    norm_latency = min(1.0, obs.avg_pass_latency_ms / _LATENCY_SATURATION_MS)
    norepinephrine = (norm_latency - 0.3) * 0.4
    norepinephrine = max(-0.15, min(0.3, norepinephrine))

    return _Impulse(
        dopamine=dopamine,
        gaba=gaba,
        cortisol=cortisol,
        norepinephrine=norepinephrine,
    )


def _decay_level(level: NTLevel, elapsed_seconds: float) -> float:
    """Exponential decay of `level.level` toward `level.baseline`.

    `new = baseline + (level - baseline) * exp(-decay * elapsed)`

    decay==0 means no decay (the level holds). Negative elapsed is
    treated as zero — protects against clock-skew or test sequencing.
    """
    if elapsed_seconds <= 0 or level.decay <= 0:
        return level.level
    return level.baseline + (level.level - level.baseline) * math.exp(
        -level.decay * elapsed_seconds
    )


def _apply(level: NTLevel, impulse: float, elapsed_seconds: float) -> NTLevel:
    decayed = _decay_level(level, elapsed_seconds)
    new = max(0.0, min(1.0, decayed + impulse))
    return NTLevel(level=new, baseline=level.baseline, decay=level.decay)


def tick(
    state: NTState,
    *,
    observations: Observations | None,
    now: datetime | None = None,
) -> NTState:
    """Advance NT state: decay every level toward baseline, then apply
    impulses from `observations` (if supplied).

    Pure function — does not mutate `state`. Callers store the
    returned value back into `app.state.nt_state`.

    `now` is overridable for tests; defaults to `datetime.now(UTC)`.
    """
    now = now or datetime.now(UTC)
    elapsed_seconds = (now - state.lastUpdated).total_seconds()
    impulse = _impulses_from_observations(observations) if observations is not None else _Impulse()

    return NTState(
        lastUpdated=now,
        dopamine=_apply(state.dopamine, impulse.dopamine, elapsed_seconds),
        serotonin=_apply(state.serotonin, impulse.serotonin, elapsed_seconds),
        norepinephrine=_apply(state.norepinephrine, impulse.norepinephrine, elapsed_seconds),
        gaba=_apply(state.gaba, impulse.gaba, elapsed_seconds),
        cortisol=_apply(state.cortisol, impulse.cortisol, elapsed_seconds),
        acetylcholine=_apply(state.acetylcholine, impulse.acetylcholine, elapsed_seconds),
    )


# -------- Bicameral parameter modulation --------
#
# Both modulators return a plain Python value the bicameral loop /
# orchestrator config can consume. They're pure functions of the NT
# state + the configured baseline, so the chat handler can pre-compute
# values once per turn and pass them to the loop unchanged.


# Hard ceiling on max_passes regardless of NT bias — matches the
# `defaultMaxPasses` config field's max=10. Keeps a runaway-anxiety
# state from blocking forever.
_HARD_MAX_PASSES = 10


def modulated_max_passes(state: NTState, base_max_passes: int) -> int:
    """`max_passes = base + cortisol_boost + ne_boost`.

    At neutral NT (every level 0.5), returns `base_max_passes`. As
    cortisol or NE rise above baseline, more deliberation is allowed.
    Capped at `_HARD_MAX_PASSES`.

    Below-baseline cortisol does NOT subtract passes — Eugene at peace
    still gets the operator's configured ceiling. The asymmetry is
    deliberate: anxious Eugene deliberates longer than calm Eugene,
    but calm Eugene doesn't undershoot the operator's intent.
    """
    cortisol = state.cortisol.level
    ne = state.norepinephrine.level
    cortisol_boost = max(0, round((cortisol - 0.5) * 4))  # 0..2 over [.5, 1]
    ne_boost = max(0, round((ne - 0.5) * 2))  # 0..1 over [.5, 1]
    boosted = base_max_passes + cortisol_boost + ne_boost
    return max(1, min(_HARD_MAX_PASSES, boosted))


def modulated_temperature(state: NTState, base_temperature: float | None) -> float | None:
    """`temperature = base * (1 + 0.5*(dopamine-0.5) - 0.5*(gaba-0.5))`.

    At neutral NT, returns `base_temperature`. Dopamine pushes
    temperature up (exploration); GABA pulls it down (determinism).
    Clamped to [0.0, 2.0] to match the config field's allowed range.

    Returns None if `base_temperature` is None (operator explicitly
    didn't configure a default; NT shouldn't fabricate one).
    """
    if base_temperature is None:
        return None
    dopamine = state.dopamine.level
    gaba = state.gaba.level
    factor = 1.0 + 0.5 * (dopamine - 0.5) - 0.5 * (gaba - 0.5)
    return max(0.0, min(2.0, base_temperature * factor))
