"""Regression tests for RagasEvaluator version compatibility + base_url.

The project pins ``ragas<0.4``; the evaluator must:
1. Dispatch to the ragas 0.3.x ``evaluate()`` API when the 0.4 collections
   API is unavailable (and keep the 0.4 path for future unpinning).
2. Forward ``base_url`` from settings to every client — otherwise
   OpenAI-compatible providers (智谱 GLM) silently hit api.openai.com.

All network/LLM backends are mocked; ragas itself is only imported, and
real ragas 0.3.9 is installed in the test env.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.observability.evaluation.ragas_evaluator import RagasEvaluator


BASE_URL = "https://open.bigmodel.cn/api/paas/v4"


def _settings() -> SimpleNamespace:
    """Minimal settings double with an OpenAI-compatible provider."""
    return SimpleNamespace(
        llm=SimpleNamespace(
            provider="openai",
            model="glm-4-flash",
            api_key="test-key",
            base_url=BASE_URL,
            azure_endpoint=None,
            api_version=None,
        ),
        embedding=SimpleNamespace(
            provider="openai",
            model="embedding-3",
            api_key="test-key",
            base_url=BASE_URL,
            azure_endpoint=None,
        ),
        evaluation=SimpleNamespace(metrics=["faithfulness"]),
    )


class TestLegacyPathDispatch:
    """ragas 0.3.x (pinned): collections API missing → evaluate() pipeline."""

    def test_collections_import_failure_routes_to_legacy(self) -> None:
        evaluator = RagasEvaluator(settings=_settings(), metrics=["faithfulness"])

        with patch.dict("sys.modules", {"ragas.metrics.collections": None}):
            with patch("ragas.evaluate") as mock_evaluate, \
                 patch("ragas.metrics.faithfulness", MagicMock(), create=True), \
                 patch("ragas.llms.LangchainLLMWrapper", MagicMock(), create=True), \
                 patch("ragas.embeddings.LangchainEmbeddingsWrapper", MagicMock(), create=True), \
                 patch("langchain_openai.ChatOpenAI") as mock_chat, \
                 patch("langchain_openai.OpenAIEmbeddings") as mock_embed:
                mock_evaluate.return_value.to_pandas.return_value = pd.DataFrame(
                    {"faithfulness": [0.91]}
                )
                scores = evaluator._run_ragas("q", ["ctx"], "answer")

        assert scores == {"faithfulness": 0.91}
        mock_evaluate.assert_called_once()
        # base_url must reach the chat client — the exact bug class where
        # a compatible provider silently hits api.openai.com.
        assert mock_chat.call_count == 1
        assert mock_chat.call_args.kwargs.get("base_url") == BASE_URL
        assert mock_chat.call_args.kwargs.get("model") == "glm-4-flash"
        assert mock_embed.call_args.kwargs.get("base_url") == BASE_URL

    def test_legacy_evaluate_receives_llm_and_embeddings(self) -> None:
        evaluator = RagasEvaluator(
            settings=_settings(), metrics=["faithfulness", "answer_relevancy"]
        )
        with patch.dict("sys.modules", {"ragas.metrics.collections": None}):
            with patch("ragas.evaluate") as mock_evaluate, \
                 patch("ragas.metrics.faithfulness", MagicMock(), create=True), \
                 patch("ragas.metrics.answer_relevancy", MagicMock(), create=True), \
                 patch("ragas.llms.LangchainLLMWrapper", MagicMock(), create=True), \
                 patch("ragas.embeddings.LangchainEmbeddingsWrapper", MagicMock(), create=True), \
                 patch("langchain_openai.ChatOpenAI"), \
                 patch("langchain_openai.OpenAIEmbeddings"):
                mock_evaluate.return_value.to_pandas.return_value = pd.DataFrame(
                    {"faithfulness": [1.0], "answer_relevancy": [0.8]}
                )
                scores = evaluator._run_ragas("q", ["ctx"], "answer")

        assert scores == {"faithfulness": 1.0, "answer_relevancy": 0.8}
        kwargs = mock_evaluate.call_args.kwargs
        assert "llm" in kwargs and "embeddings" in kwargs
        assert len(kwargs["metrics"]) == 2

    def test_nan_scores_map_to_zero(self) -> None:
        """ragas yields NaN when a metric cannot be judged — must not crash."""
        evaluator = RagasEvaluator(settings=_settings(), metrics=["faithfulness"])
        with patch.dict("sys.modules", {"ragas.metrics.collections": None}):
            with patch("ragas.evaluate") as mock_evaluate, \
                 patch("ragas.metrics.faithfulness", MagicMock(), create=True), \
                 patch("ragas.llms.LangchainLLMWrapper", MagicMock(), create=True), \
                 patch("ragas.embeddings.LangchainEmbeddingsWrapper", MagicMock(), create=True), \
                 patch("langchain_openai.ChatOpenAI"), \
                 patch("langchain_openai.OpenAIEmbeddings"):
                mock_evaluate.return_value.to_pandas.return_value = pd.DataFrame(
                    {"faithfulness": [float("nan")]}
                )
                scores = evaluator._run_ragas("q", ["ctx"], "answer")

        assert scores == {"faithfulness": 0.0}


class TestModernPathBaseUrl:
    """ragas 0.4+ collections path must also forward base_url."""

    def test_collections_path_forwards_base_url(self) -> None:
        evaluator = RagasEvaluator(settings=_settings(), metrics=["faithfulness"])

        fake_collections = SimpleNamespace(
            Faithfulness=MagicMock(),
            AnswerRelevancy=MagicMock(),
            ContextPrecisionWithoutReference=MagicMock(),
        )
        # Fake metric instance: m.score(...) -> result with .value
        metric_instance = fake_collections.Faithfulness.return_value
        metric_instance.score.return_value = SimpleNamespace(value=0.87)

        with patch.dict("sys.modules", {"ragas.metrics.collections": fake_collections}):
            with patch("ragas.llms.llm_factory", MagicMock(), create=True), \
                 patch("ragas.embeddings.OpenAIEmbeddings", MagicMock(), create=True), \
                 patch("openai.AsyncOpenAI") as mock_async_openai:
                scores = evaluator._run_ragas("q", ["ctx"], "answer")

        assert scores == {"faithfulness": 0.87}
        # one client each for LLM and embeddings — both must carry base_url
        assert mock_async_openai.call_count == 2
        for call in mock_async_openai.call_args_list:
            assert call.kwargs.get("base_url") == BASE_URL


class TestBaseUrlRegressionGuard:
    """Direct guard for the bug class: OpenAI-compatible base_url dropped."""

    def test_langchain_wrapper_builder_passes_base_url(self) -> None:
        evaluator = RagasEvaluator(settings=_settings(), metrics=["faithfulness"])
        with patch("ragas.llms.LangchainLLMWrapper", MagicMock(), create=True), \
             patch("ragas.embeddings.LangchainEmbeddingsWrapper", MagicMock(), create=True), \
             patch("langchain_openai.ChatOpenAI") as mock_chat, \
             patch("langchain_openai.OpenAIEmbeddings") as mock_embed:
            evaluator._build_wrappers_langchain()

        assert mock_chat.call_args.kwargs["base_url"] == BASE_URL
        assert mock_chat.call_args.kwargs["temperature"] == 0.0
        assert mock_embed.call_args.kwargs["base_url"] == BASE_URL
        # tiktoken has no vocab for non-OpenAI embedding models — the
        # local token-length check must be off for compatible endpoints.
        assert mock_embed.call_args.kwargs["check_embedding_ctx_length"] is False
