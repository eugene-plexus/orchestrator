"""Neurotransmitter state.

v0.1 returned a single-float-per-NT shape. v0.2 promotes that to a
`{level, baseline, decay}` triple per NT and adds a `lastUpdated`
timestamp so the orchestrator can compute elapsed-time decay between
observations. This module still emits a static neutral state — actual
modulation (the load-bearing v0.2 work) lands when the bicameral loop
starts reading NT for max_passes / temperature / blend weights.
"""

from __future__ import annotations

from datetime import UTC, datetime

from .._generated.models import NTLevel, NTState

_NEUTRAL_LEVEL = NTLevel(level=0.5, baseline=0.5, decay=0.0)


def neutral_state() -> NTState:
    """A baseline NT state: every NT at 0.5, baseline 0.5, no decay."""
    return NTState(
        lastUpdated=datetime.now(UTC),
        dopamine=_NEUTRAL_LEVEL,
        serotonin=_NEUTRAL_LEVEL,
        norepinephrine=_NEUTRAL_LEVEL,
        gaba=_NEUTRAL_LEVEL,
        cortisol=_NEUTRAL_LEVEL,
        acetylcholine=_NEUTRAL_LEVEL,
    )
