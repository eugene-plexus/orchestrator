"""Corpus callosum: trivial v0.1 agreement signal between hemisphere outputs.

The agreement metric here is intentionally simple — Jaccard similarity on
lowercased word sets. It's the dumbest thing that does any useful job.
v0.2 swaps it for an embedding-based similarity (or even an LLM-judged
disagreement signal). Until we have orchestrator-driven runs producing
real disagreements, smarter metrics are speculative.
"""

from __future__ import annotations


def jaccard_word_agreement(left: str, right: str) -> float:
    """Return a 0..1 agreement score from the word-set overlap of two outputs.

    Both empty -> 1.0 (vacuously identical). One empty -> 0.0.
    """
    left_stripped = left.strip()
    right_stripped = right.strip()
    if not left_stripped and not right_stripped:
        return 1.0
    if not left_stripped or not right_stripped:
        return 0.0
    left_words = {w for w in left_stripped.lower().split() if w}
    right_words = {w for w in right_stripped.lower().split() if w}
    if not left_words or not right_words:
        return 0.0
    intersect = left_words & right_words
    union = left_words | right_words
    return len(intersect) / len(union)


def blend(left: str, right: str) -> str:
    """Pick a final assistant message from the two hemispheres' outputs.

    v0.1 strategy: longer of the two (with left as the tie-breaker). This
    is an honest placeholder; the bicameral framework is the load-bearing
    piece, and a smart blend is meaningful only once we have data on what
    actual disagreements look like in practice.
    """
    if len(right) > len(left):
        return right
    return left
