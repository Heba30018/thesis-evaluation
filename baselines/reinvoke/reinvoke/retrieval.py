"""Ranks every tool against a user's query and returns the best matches.

A query can carry more than one intent (the Intent Extractor splits those
apart first). Each intent gets compared against every tool on its own,
and tools are ranked separately within each intent before the per-intent
rankings are merged -- that way a tool only needs to be the best answer to
one of the intents to make it into the final results, instead of needing
to score well against all of them at once.
"""

from __future__ import annotations

import numpy as np

# similarity_scores() used to convert every single tool vector to its own
# np.array() on every call (1164 conversions, every time, for every
# intent) -- that Python-level loop was real, measurable overhead on top
# of the one-time model-loading cost solved in query.py. Here the same
# tool_vectors dict (query.py loads and caches it once) gets stacked into
# ONE matrix the first time it's seen, keyed by id(), and every later
# call just reuses that matrix and does a single vectorized multiply
# instead of 1164 separate ones.
_tool_matrix_cache: dict[int, tuple[list[str], np.ndarray]] = {}


def _get_tool_matrix(tool_vectors: dict[str, list[float]]) -> tuple[list[str], np.ndarray]:
    key = id(tool_vectors)
    cached = _tool_matrix_cache.get(key)
    if cached is None:
        endpoint_ids = list(tool_vectors.keys())
        matrix = np.array([tool_vectors[eid] for eid in endpoint_ids], dtype=float)
        cached = (endpoint_ids, matrix)
        _tool_matrix_cache[key] = cached
    return cached


def similarity_scores(intent_vector: list[float], tool_vectors: dict[str, list[float]]) -> dict[str, float]:
    """Dot product between one intent's embedding and every tool's
    averaged embedding -- one vectorized matrix-vector multiply across
    every tool at once, instead of a per-tool Python loop."""
    endpoint_ids, matrix = _get_tool_matrix(tool_vectors)
    intent = np.array(intent_vector)
    scores = matrix @ intent
    return dict(zip(endpoint_ids, scores.tolist()))

def rank_within_intent(scores: dict[str, float]) -> dict[str, int]:
    """Turns raw similarity scores into ranks local to one intent: the
    lowest-scoring tool gets rank 1, the highest gets rank N. Ranking
    instead of comparing raw scores keeps one intent's naturally higher or
    lower scores from drowning out another intent's."""
    ordered = sorted(scores, key=lambda endpoint_id: scores[endpoint_id])
    return {endpoint_id: rank for rank, endpoint_id in enumerate(ordered, start=1)}


def rank_tools(intent_vectors: list[list[float]], tool_vectors: dict[str, list[float]]) -> list[tuple[str, int, float]]:
    """Scores and ranks every tool against every intent, then keeps each
    tool's best (rank, score) across all of them. Returns tools sorted
    best-first as (endpoint_id, rank, score) triples."""
    best: dict[str, tuple[int, float]] = {}

    for intent_vector in intent_vectors:
        scores = similarity_scores(intent_vector, tool_vectors)
        ranks = rank_within_intent(scores)

        for endpoint_id in tool_vectors:
            candidate = (ranks[endpoint_id], scores[endpoint_id])
            if endpoint_id not in best or candidate > best[endpoint_id]:
                best[endpoint_id] = candidate

    ranked = sorted(best.items(), key=lambda item: item[1], reverse=True)
    return [(endpoint_id, rank, score) for endpoint_id, (rank, score) in ranked]


def top_k(ranked_tools: list[tuple[str, int, float]], k: int = 5) -> list[tuple[str, int, float]]:
    """The k best tools from an already-ranked list."""
    return ranked_tools[:k]
