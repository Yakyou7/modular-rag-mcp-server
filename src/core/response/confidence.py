"""Confidence assessment for retrieval results.

This module derives a query-level confidence signal from the top reranked
result, used by the MCP tool layer to flag (or refuse) low-confidence
responses instead of silently returning weakly-related chunks.

Score scale conventions:
- After a successful LLM rerank, ``RetrievalResult.score`` carries the
  reranker's relevance score on a 0-3 scale (see config/prompts/rerank.txt:
  3=Highly Relevant, 2=Partially Relevant, 1=Marginally Relevant, 0=Not
  Relevant).  Normalised confidence = score / 3.
- Without reranking (fallback or rerank disabled), the score is an RRF
  fusion score (≈ 1/(k+rank), rank-derived).  It has NO absolute relevance
  meaning, so confidence is reported as *not assessable* rather than
  guessed — flagging low confidence off a rank-derived score would produce
  arbitrary refusals.

Design Principles:
- Honest by default: "cannot assess" is an explicit state, never faked.
- Config-driven: threshold comes from settings.yaml (response section).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.core.types import RetrievalResult

# Maximum score emitted by the LLM rerank prompt (0-3 relevance levels).
RERANK_SCORE_MAX = 3.0

# Method identifiers used in tool metadata / traces.
METHOD_RERANK = "rerank_score"
METHOD_UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class ConfidenceAssessment:
    """Result of assessing retrieval confidence for one query.

    Attributes:
        confidence: Normalised confidence in [0, 1], or None when the
            available scores do not carry absolute relevance meaning.
        method: How confidence was derived ("rerank_score" or "unavailable").
        threshold: Configured low-confidence threshold (same 0-1 scale).
        low: Whether confidence is below threshold. None when not assessable.
    """

    confidence: Optional[float]
    method: str
    threshold: float
    low: Optional[bool]

    @property
    def assessable(self) -> bool:
        """Whether a meaningful confidence value is available."""
        return self.confidence is not None


def _unavailable(threshold: float) -> ConfidenceAssessment:
    """Build an explicit 'not assessable' assessment."""
    return ConfidenceAssessment(
        confidence=None,
        method=METHOD_UNAVAILABLE,
        threshold=threshold,
        low=None,
    )


def assess_confidence(
    results: List[RetrievalResult],
    threshold: float,
    enabled: bool = True,
) -> ConfidenceAssessment:
    """Assess query confidence from the top retrieval result.

    Confidence is only derived from reranked results: the rerank metadata
    (``reranked=True`` set by CoreReranker) marks that ``score`` now holds
    the 0-3 LLM relevance score instead of the RRF fusion score.

    Args:
        results: Final (post-rerank) retrieval results, best first.
        threshold: Low-confidence threshold on the normalised 0-1 scale.
        enabled: Master switch from configuration.

    Returns:
        ConfidenceAssessment. When reranking was not applied (fallback,
        disabled, single result) or results are empty, returns an
        assessment with ``confidence=None`` and ``low=None``.
    """
    if not enabled or not results:
        return _unavailable(threshold)

    top = results[0]
    metadata = top.metadata or {}
    if not metadata.get("reranked"):
        # Fallback / disabled rerank: score is RRF (rank-derived).
        return _unavailable(threshold)

    try:
        raw = float(top.score)
    except (TypeError, ValueError):
        return _unavailable(threshold)

    raw = max(0.0, min(raw, RERANK_SCORE_MAX))
    confidence = raw / RERANK_SCORE_MAX

    return ConfidenceAssessment(
        confidence=confidence,
        method=METHOD_RERANK,
        threshold=threshold,
        low=confidence < threshold,
    )
