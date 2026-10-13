#!/usr/bin/env python
"""完整 RAG 问答演示：检索 → 重排 → 引用 → 接地提示词 → LLM 生成答案。

这是整条链路的"最后一公里"——前面所有模块在这里汇合成一个完整回答。

用法:
    python scripts/demo_answer.py "你的问题" [集合名，默认 learning_docs]
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from src.core.settings import load_settings
from src.core.query_engine.hybrid_search import create_hybrid_search
from src.core.query_engine.query_processor import QueryProcessor
from src.core.query_engine.reranker import create_core_reranker
from src.core.response.citation_generator import CitationGenerator
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.llm.base_llm import Message
from src.libs.llm.llm_factory import LLMFactory
from src.libs.vector_store.vector_store_factory import VectorStoreFactory


# 接地（Grounding）规则：逼模型"只根据资料说话"
SYSTEM_PROMPT = (
    "你是一个严谨的知识库问答助手。规则：\n"
    "1. 只根据下面提供的资料回答问题，禁止使用资料之外的任何知识；\n"
    "2. 每个论断后面标注来源编号，如 [1] 或 [2][3]；\n"
    "3. 如果资料不足以回答，直接说\"资料中没有提到\"，禁止编造；\n"
    "4. 用简洁的中文回答。"
)


def build_context(results, citations, snippet_chars: int = 500) -> str:
    """把检索结果拼成带 [n] 编号的资料块（喂给模型的上下文）。"""
    parts = []
    for c, r in zip(citations, results):
        text = r.text[:snippet_chars].replace("\n", " ")
        parts.append(f"[{c.index}] {text}")
    return "\n\n".join(parts)


def main() -> int:
    query = sys.argv[1] if len(sys.argv) > 1 else "重排序有哪些方案？各自的特点是什么？"
    collection = sys.argv[2] if len(sys.argv) > 2 else "learning_docs"

    settings = load_settings(str(_REPO_ROOT / "config" / "settings.yaml"))

    # ── 组装检索组件（与 scripts/query.py 完全相同的组装方式） ──
    vector_store = VectorStoreFactory.create(settings, collection_name=collection)
    embedding_client = EmbeddingFactory.create(settings)
    from src.core.query_engine.dense_retriever import create_dense_retriever
    from src.core.query_engine.sparse_retriever import create_sparse_retriever
    dense_retriever = create_dense_retriever(
        settings=settings, embedding_client=embedding_client, vector_store=vector_store,
    )
    bm25_indexer = BM25Indexer(index_dir=f"data/db/bm25/{collection}")
    sparse_retriever = create_sparse_retriever(
        settings=settings, bm25_indexer=bm25_indexer, vector_store=vector_store,
    )
    sparse_retriever.default_collection = collection
    hybrid_search = create_hybrid_search(
        settings=settings, query_processor=QueryProcessor(),
        dense_retriever=dense_retriever, sparse_retriever=sparse_retriever,
    )
    reranker = create_core_reranker(settings=settings)

    # ── 1) 混合检索 + 2) 重排 ────────────────────────────────────
    results = hybrid_search.search(query=query, top_k=10)  # 默认返回纯列表
    print(f"[1] 混合检索：{len(results)} 个候选")
    if reranker.is_enabled:
        results = reranker.rerank(query=query, results=results, top_k=5).results
        print(f"[2] LLM 重排：精选 {len(results)} 个")

    if not results:
        print("没有检索到相关内容。")
        return 0

    # ── 3) 生成引用 [1][2]… ─────────────────────────────────────
    citations = CitationGenerator().generate(results)
    print(f"[3] 引用编号：{[c.index for c in citations]}")

    # ── 4) 组装接地提示词 ────────────────────────────────────────
    context = build_context(results, citations)
    print(f"[4] 上下文长度：{len(context)} 字符（每段截断 500）")

    llm = LLMFactory.create(settings)
    response = llm.chat([
        Message(role="system", content=SYSTEM_PROMPT),
        Message(role="user", content=f"资料：\n\n{context}\n\n\n问题：{query}"),
    ])

    # ── 5) 展示最终回答与可溯源的引用 ───────────────────────────
    print()
    print("=" * 64)
    print("【最终回答】")
    print("=" * 64)
    print(response.content)
    print()
    print("-" * 64)
    print("【引用来源】")
    for c in citations:
        title = c.metadata.get("title", "")
        page = f" 第{c.page}页" if c.page else ""
        snippet = c.text_snippet[:60].replace("\n", " ")
        print(f"  [{c.index}] {Path(c.source).name}{page}  （{title}）{snippet}…")
    return 0


if __name__ == "__main__":
    sys.exit(main())
