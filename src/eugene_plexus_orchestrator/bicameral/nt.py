"""Neurotransmitter state.

v0.1 returns a static neutral state. The endpoint and schema exist so
downstream consumers (UI, hemisphere-driver) can begin reading NT now,
without code changes when modulation lands in v0.2+.
"""

from __future__ import annotations

from .._generated.models import NTState


def neutral_state() -> NTState:
    """A baseline NT state. Every value is 0.5 (neutral / undriven)."""
    return NTState(
        serotonin=0.5,
        dopamine=0.5,
        norepinephrine=0.5,
        acetylcholine=0.5,
        gaba=0.5,
        glutamate=0.5,
    )
