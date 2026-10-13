"""Unit tests for retrieval confidence assessment.

Covers:
- Confidence derivation from reranked top-1 results (0-3 scale → 0-1)
- Clamping of out-of-range rerank scores
- Explicit "not assessable" states (no rerank / fallback / disabled / empty)
- Threshold comparison on the normalised scale
"""

from __future__ import annotations

import pytest

from src.core.response.confidence import (
    METHOD_RERANK,
    METHOD_UNAVAILABLE,
    assess_confidence,
)
from src.core.types import RetrievalResult


def _result(score: float, reranked: bool = True) -> RetrievalResult:
    """Build a minimal retrieval result with rerank metadata."""
    metadata = {}
    if reranked:
        metadata = {"reranked": True, "rerank_score": score, "original_score": 0.03}
    return RetrievalResult(
        chunk_id="chunk_0001",
        score=score,
        text="示例文本",
        metadata=metadata,
    )


class TestRerankedConfidence:
    """Confidence derivation from reranked scores (0-3 scale)."""

    def test_score_3_is_full_confidence(self) -> None:
        assessment = assess_confidence([_result(3.0)], threshold=0.5)
        assert assessment.assessable
        assert assessment.method == METHOD_RERANK
        assert assessment.confidence == 1.0
        assert assessment.low is False

    def test_score_2_above_default_threshold(self) -> None:
        assessment = assess_confidence([_result(2.0)], threshold=0.5)
        assert assessment.assessable
        assert assessment.confidence == pytest.approx(2.0 / 3.0)
        assert assessment.low is False

    def test_score_1_below_default_threshold(self) -> None:
        assessment = assess_confidence([_result(1.0)], threshold=0.5)
        assert assessment.assessable
        assert assessment.confidence == pytest.approx(1.0 / 3.0)
        assert assessment.low is True

    def test_score_0_is_zero_confidence(self) -> None:
        assessment = assess_confidence([_result(0.0)], threshold=0.5)
        assert assessment.assessable
        assert assessment.confidence == 0.0
        assert assessment.low is True

    def test_fractional_score(self) -> None:
        assessment = assess_confidence([_result(2.5)], threshold=0.5)
        assert assessment.confidence == pytest.approx(2.5 / 3.0)

    def test_out_of_range_score_is_clamped(self) -> None:
        assessment = assess_confidence([_result(5.0)], threshold=0.5)
        assert assessment.confidence == 1.0

        assessment_neg = assess_confidence([_result(-1.0)], threshold=0.5)
        assert assessment_neg.confidence == 0.0

    def test_uses_top_result_only(self) -> None:
        """Confidence comes from the best result, not the average."""
        results = [_result(3.0), _result(0.0)]
        assessment = assess_confidence(results, threshold=0.5)
        assert assessment.confidence == 1.0


class TestNotAssessableStates:
    """'Cannot assess' must be explicit, never faked."""

    def test_non_reranked_results_are_not_assessable(self) -> None:
        # Plain RRF/dense scores have no absolute relevance meaning.
        results = [_result(0.0323, reranked=False)]
        assessment = assess_confidence(results, threshold=0.5)
        assert not assessment.assessable
        assert assessment.confidence is None
        assert assessment.low is None
        assert assessment.method == METHOD_UNAVAILABLE

    def test_fallback_flag_disables_assessment(self) -> None:
        # CoreReranker marks fallback results with reranked=False.
        result = RetrievalResult(
            chunk_id="chunk_0001",
            score=0.0164,
            text="示例文本",
            metadata={"reranked": False, "rerank_fallback": True},
        )
        assessment = assess_confidence([result], threshold=0.5)
        assert not assessment.assessable

    def test_empty_results_are_not_assessable(self) -> None:
        assessment = assess_confidence([], threshold=0.5)
        assert not assessment.assessable
        assert assessment.low is None

    def test_disabled_switch_short_circuits(self) -> None:
        assessment = assess_confidence([_result(3.0)], threshold=0.5, enabled=False)
        assert not assessment.assessable
        assert assessment.method == METHOD_UNAVAILABLE

    def test_non_numeric_score_is_not_assessable(self) -> None:
        result = RetrievalResult(
            chunk_id="chunk_0001",
            score=2.0,
            text="示例文本",
            metadata={"reranked": True, "rerank_score": "high"},
        )
        # Force a non-numeric score through (validation happens in __post_init__
        # only for type; strings are rejected there, so patch the value after).
        object.__setattr__(result, "score", "high")
        assessment = assess_confidence([result], threshold=0.5)
        assert not assessment.assessable
