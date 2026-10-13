"""Unit tests for the RAG answer generator used in evaluation runs.

All tests use a mock LLM — no network calls.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.observability.evaluation.answer_generator import (
    RagAnswerGenerator,
    extract_chunk_texts,
)


class _MockLLM:
    """Minimal LLM double matching the BaseLLM.chat interface."""

    def __init__(self, content: str = "根据资料，答案是……", fail: bool = False) -> None:
        self.content = content
        self.fail = fail
        self.calls: list = []

    def chat(self, messages, trace=None, **kwargs):
        if self.fail:
            raise RuntimeError("LLM down")
        self.calls.append(messages)
        return SimpleNamespace(content=self.content)


class TestExtractChunkTexts:
    """Chunk text extraction across representations."""

    def test_extracts_from_dicts(self) -> None:
        chunks = [{"text": "a"}, {"content": "b"}, {"page_content": "c"}, {}]
        assert extract_chunk_texts(chunks) == ["a", "b", "c"]

    def test_extracts_from_strings_and_objects(self) -> None:
        class Obj:
            text = "obj-text"

        assert extract_chunk_texts(["s", Obj()]) == ["s", "obj-text"]

    def test_skips_empty_texts(self) -> None:
        assert extract_chunk_texts(["", "  ", {"text": None}, "real"]) == ["real"]

    def test_none_chunks_list(self) -> None:
        assert extract_chunk_texts(None) == []


class TestRagAnswerGenerator:
    """Answer generation against the EvalRunner contract."""

    def test_generates_answer_with_grounding_prompt(self) -> None:
        llm = _MockLLM(content="RAG 将检索与生成结合。")
        generator = RagAnswerGenerator(llm=llm, max_context_chars=4000)
        answer = generator(
            "什么是 RAG？",
            [{"text": "RAG 将信息检索与文本生成相结合。"}],
        )
        assert answer == "RAG 将检索与生成结合。"
        prompt = llm.calls[0][0].content
        assert "什么是 RAG？" in prompt
        assert "RAG 将信息检索与文本生成相结合。" in prompt
        # grounding instruction present
        assert "仅根据" in prompt
        assert "无法回答" in prompt

    def test_context_numbering_and_multiple_chunks(self) -> None:
        llm = _MockLLM()
        generator = RagAnswerGenerator(llm=llm)
        generator("q", [{"text": "甲"}, {"text": "乙"}])
        prompt = llm.calls[0][0].content
        assert "[1] 甲" in prompt
        assert "[2] 乙" in prompt

    def test_context_truncated_to_budget(self) -> None:
        llm = _MockLLM()
        generator = RagAnswerGenerator(llm=llm, max_context_chars=50)
        long_text = "字" * 500
        generator("q", [{"text": long_text}])
        prompt = llm.calls[0][0].content
        assert long_text not in prompt
        assert len(prompt) < 500 + 200

    def test_empty_chunks_uses_placeholder_context(self) -> None:
        llm = _MockLLM()
        generator = RagAnswerGenerator(llm=llm)
        generator("q", [])
        prompt = llm.calls[0][0].content
        assert "（无资料）" in prompt

    def test_raises_on_empty_llm_answer(self) -> None:
        generator = RagAnswerGenerator(llm=_MockLLM(content="   "))
        with pytest.raises(ValueError, match="empty answer"):
            generator("q", [{"text": "ctx"}])

    def test_llm_failure_propagates(self) -> None:
        # EvalRunner catches generator exceptions and falls back to
        # chunk concatenation — the generator itself must raise.
        generator = RagAnswerGenerator(llm=_MockLLM(fail=True))
        with pytest.raises(RuntimeError, match="LLM down"):
            generator("q", [{"text": "ctx"}])
