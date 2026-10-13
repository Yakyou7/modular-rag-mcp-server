"""Unit tests for low-confidence responses and scale-aware relevance display.

Covers:
- ResponseBuilder banner + metadata wiring for ConfidenceAssessment
- 0-3 rerank scale displayed as normalised percentage (never "300.00%")
- RRF fusion scores displayed as fusion scores, not relevance percentages
- Citation JSON carrying rerank scale annotation + normalised relevance
"""

from __future__ import annotations

import pytest

from src.core.response.confidence import ConfidenceAssessment, METHOD_UNAVAILABLE
from src.core.response.citation_generator import CitationGenerator
from src.core.response.response_builder import ResponseBuilder
from src.core.types import RetrievalResult


def _reranked(score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id="chunk_0001",
        score=score,
        text="重排序是 RAG 的精排阶段。",
        metadata={"reranked": True, "rerank_score": score, "source_path": "docs/guide.pdf"},
    )


def _rrf_result(score: float = 0.0323) -> RetrievalResult:
    return RetrievalResult(
        chunk_id="chunk_0002",
        score=score,
        text="RRF 融合分数来自排名。",
        metadata={"source_path": "docs/rrf.pdf"},
    )


def _assessment(confidence, threshold=0.5, method="rerank_score"):
    low = None if confidence is None else confidence < threshold
    return ConfidenceAssessment(
        confidence=confidence, method=method, threshold=threshold, low=low
    )


class TestLowConfidenceBanner:
    """Banner + metadata wiring for low-confidence responses."""

    def test_low_confidence_prepends_banner(self) -> None:
        builder = ResponseBuilder()
        response = builder.build(
            results=[_reranked(1.0)],
            query="如何量子隧穿",
            confidence=_assessment(1.0 / 3.0),
        )
        assert response.content.startswith("## ⚠️ 低置信度结果（仅供参考）")
        assert "33%" in response.content
        assert "阈值 50%" in response.content
        # Materials are still returned — honest refusal, not silent drop.
        assert response.is_empty is False
        assert "重排序" in response.content

    def test_low_confidence_metadata_flags(self) -> None:
        builder = ResponseBuilder()
        response = builder.build(
            results=[_reranked(1.0)],
            query="q",
            confidence=_assessment(1.0 / 3.0, threshold=0.5),
        )
        assert response.metadata["low_confidence"] is True
        assert response.metadata["confidence"] == pytest.approx(0.333, abs=1e-3)
        assert response.metadata["confidence_threshold"] == 0.5
        assert response.metadata["confidence_method"] == "rerank_score"

    def test_confident_response_has_no_banner(self) -> None:
        builder = ResponseBuilder()
        response = builder.build(
            results=[_reranked(3.0)],
            query="q",
            confidence=_assessment(1.0),
        )
        assert "低置信度" not in response.content
        assert response.metadata["low_confidence"] is False

    def test_unavailable_confidence_adds_method_only(self) -> None:
        builder = ResponseBuilder()
        response = builder.build(
            results=[_rrf_result()],
            query="q",
            confidence=_assessment(None, method=METHOD_UNAVAILABLE),
        )
        assert "低置信度" not in response.content
        assert response.metadata["confidence_method"] == METHOD_UNAVAILABLE
        assert "low_confidence" not in response.metadata
        assert "confidence" not in response.metadata

    def test_no_confidence_argument_unchanged(self) -> None:
        builder = ResponseBuilder()
        response = builder.build(results=[_rrf_result()], query="q")
        assert "低置信度" not in response.content
        assert "confidence" not in response.metadata


class TestScaleAwareRelevanceDisplay:
    """Relevance must respect the score scale (rerank 0-3 vs RRF rank score)."""

    def test_rerank_score_3_shows_100_percent(self) -> None:
        builder = ResponseBuilder()
        response = builder.build(results=[_reranked(3.0)], query="q")
        assert "精排分 3/3" in response.content
        assert "100%" in response.content
        assert "300.00%" not in response.content

    def test_rerank_score_2_shows_67_percent(self) -> None:
        builder = ResponseBuilder()
        response = builder.build(results=[_reranked(2.0)], query="q")
        assert "精排分 2/3" in response.content
        assert "67%" in response.content

    def test_rrf_score_shows_fusion_label(self) -> None:
        builder = ResponseBuilder()
        response = builder.build(results=[_rrf_result(0.0323)], query="q")
        assert "融合分: 0.0323" in response.content
        assert "RRF 排名分" in response.content
        assert "相关度" not in response.content


class TestCitationScaleAnnotation:
    """Structured citations expose the score scale + normalised relevance."""

    def test_reranked_citation_carries_normalized_score(self) -> None:
        generator = CitationGenerator()
        citations = generator.generate([_reranked(2.0)])
        data = citations[0].to_dict()
        assert data["metadata"]["rerank_scale"] == "0-3"
        assert data["metadata"]["relevance_normalized"] == pytest.approx(0.667, abs=1e-3)

    def test_plain_citation_has_no_scale_annotation(self) -> None:
        generator = CitationGenerator()
        citations = generator.generate([_rrf_result()])
        data = citations[0].to_dict()
        meta = data.get("metadata", {})
        assert "relevance_normalized" not in meta
        assert "rerank_scale" not in meta
