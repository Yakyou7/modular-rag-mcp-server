"""RAG answer generation for evaluation runs.

Provides a callable that generates an answer from retrieved chunks via the
configured LLM, matching the ``EvalRunner.answer_generator`` contract:
``callable(query, chunks) -> str``.

Generation prompt enforces grounded answering ("answer only from the
materials; say so when they are insufficient") so that faithfulness
metrics measure the retrieval+generation stack, not the LLM's parametric
memory.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from src.libs.llm.base_llm import Message

logger = logging.getLogger(__name__)

ANSWER_PROMPT_TEMPLATE = """你是一个严谨的知识库问答助手。请仅根据下面的资料回答问题。
如果资料不足以回答问题，请明确回答"根据现有资料无法回答该问题"，不要编造。

资料:
{context}

问题: {query}

回答:"""


def extract_chunk_texts(chunks: List[Any]) -> List[str]:
    """Extract plain texts from various chunk representations.

    Args:
        chunks: List of chunk dicts (with 'text'/'content'/'page_content'),
            objects with a ``text`` attribute, or plain strings.

    Returns:
        List of non-empty text strings.
    """
    texts: List[str] = []
    for chunk in chunks or []:
        if isinstance(chunk, str):
            text = chunk
        elif isinstance(chunk, dict):
            text = chunk.get("text") or chunk.get("content") or chunk.get("page_content", "")
        elif hasattr(chunk, "text"):
            text = str(getattr(chunk, "text"))
        else:
            text = str(chunk)
        if text and text.strip():
            texts.append(text.strip())
    return texts


class RagAnswerGenerator:
    """LLM-backed answer generator following the EvalRunner contract.

    Example::

        generator = RagAnswerGenerator(settings)
        runner = EvalRunner(settings, hybrid_search=hs,
                            evaluator=evaluator, answer_generator=generator)
    """

    def __init__(
        self,
        settings: Any = None,
        llm: Optional[Any] = None,
        max_context_chars: int = 4000,
    ) -> None:
        """Initialize the generator.

        Args:
            settings: Application settings (used to build the LLM when none
                is injected).
            llm: Optional pre-built LLM (must expose ``chat([Message])``);
                mainly for tests.
            max_context_chars: Truncation budget for the concatenated
                context, keeping prompt size bounded across top_k values.
        """
        if llm is None:
            from src.libs.llm.llm_factory import LLMFactory

            llm = LLMFactory.create(settings)
        self.llm = llm
        self.max_context_chars = max_context_chars

    def __call__(self, query: str, chunks: List[Any]) -> str:
        """Generate a grounded answer for *query* from *chunks*.

        Raises on LLM failure — EvalRunner catches generator exceptions and
        falls back to plain chunk concatenation, keeping the eval run alive.
        """
        texts = extract_chunk_texts(chunks)
        context = "\n\n".join(f"[{i + 1}] {t}" for i, t in enumerate(texts))
        if len(context) > self.max_context_chars:
            context = context[: self.max_context_chars]

        prompt = ANSWER_PROMPT_TEMPLATE.format(context=context or "（无资料）", query=query)
        response = self.llm.chat([Message(role="user", content=prompt)])
        answer = (response.content or "").strip()
        if not answer:
            raise ValueError("LLM returned an empty answer")
        return answer
