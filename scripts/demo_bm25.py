#!/usr/bin/env python
"""BM25 演示：看懂"关键词检索"的三个核心行为（无需任何 API）。

BM25 公式（本项目 k1=1.5, b=0.75）:
    score = IDF × tf×(k1+1) / (tf + k1×(1 - b + b×doc_len/avg_len))
    IDF = 词在多少文档出现过（越稀有越值钱）

实验：
  1. 真实分词管道检索 —— 精确命中 "gpt-4"（复合词修复后的效果）
  2. 词频饱和（k1）—— 词出现 20 倍，分数远不到 20 倍
  3. 长度归一化（b）—— 同词频，短文档得分更高

用法:
    python scripts/demo_bm25.py
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from src.core.types import Chunk
from src.ingestion.embedding.sparse_encoder import SparseEncoder
from src.ingestion.storage.bm25_indexer import BM25Indexer


def make_stat(chunk_id: str, tf_map: dict[str, int]) -> dict:
    """手工构造词频统计（跳过分词，精确控制变量用）。"""
    return {
        "chunk_id": chunk_id,
        "term_frequencies": dict(tf_map),
        "doc_length": sum(tf_map.values()),
    }


def show(indexer: BM25Indexer, terms: list[str], top_k: int = 3) -> None:
    results = indexer.query(query_terms=terms, top_k=top_k)
    if not results:
        print("    （无任何命中）")
    for r in results:
        print(f"    {r['chunk_id']:<10s} score={r['score']:.4f}")


def main() -> int:
    encoder = SparseEncoder()
    indexer = BM25Indexer(index_dir=str(_REPO_ROOT / "data" / "db" / "bm25"))
    demo_collection = "demo_step8"

    # ── 实验一：真实分词管道 + 精确匹配 ─────────────────────────
    print("=" * 64)
    print("实验一：真实管道检索（SparseEncoder 分词 → 建索引 → 查询）")
    print("=" * 64)
    docs = [
        Chunk(id="doc1", text="GPT-4 是 OpenAI 发布的多模态大语言模型。", metadata={"source_path": "demo"}),
        Chunk(id="doc2", text="退货需要在购买后 30 天内申请，商品需保持完好。", metadata={"source_path": "demo"}),
        Chunk(id="doc3", text="HNSW 是一种用于向量检索的图索引算法。", metadata={"source_path": "demo"}),
    ]
    term_stats = encoder.encode(docs)
    print(f"  doc1 的词频表（注意 gpt-4 是整词！）: {term_stats[0]['term_frequencies']}")
    indexer.build(term_stats, collection=demo_collection)

    for query in (["gpt-4"], ["gpt-4", "模型"], ["退货"]):
        print(f"  查询 {query}:")
        show(indexer, query)
    print("  >>> BM25 的强项：型号/术语精确匹配，命中就给高分，不沾边就是 0")

    # ── 实验二：词频饱和（k1 效应） ──────────────────────────────
    print()
    print("=" * 64)
    print("实验二：'向量'一词出现 1 / 5 / 20 次，分数怎么涨？（k1=1.5）")
    print("=" * 64)
    # 加入 5 篇不含'向量'的填充文档：若目标词出现在全部文档（df=N），
    # 经典 IDF 公式 log((N-df+0.5)/(df+0.5)) 会变成负数——词太常见就一文不值
    filler = ["检索", "系统", "数据", "算法"]
    stats2 = [
        make_stat(f"tf{tf:02d}", {"向量": tf, **{w: 5 for w in filler}})
        for tf in (1, 5, 20)
    ] + [make_stat(f"填充{i}", {f"词{i}_{j}": 2 for j in range(5)}) for i in range(5)]
    indexer.build(stats2, collection=demo_collection)
    scores = {r["chunk_id"]: r["score"] for r in indexer.query(query_terms=["向量"], top_k=3)}
    s1, s5, s20 = scores["tf01"], scores["tf05"], scores["tf20"]
    print(f"  出现 1 次  → score={s1:.4f}")
    print(f"  出现 5 次  → score={s5:.4f}  （次数×5，分数只×{s5/s1:.2f}）")
    print(f"  出现 20 次 → score={s20:.4f}  （次数×20，分数只×{s20/s1:.2f}）")
    print("  >>> 词频增长带来的分数增长是'边际递减'的：前几次出现最值钱，")
    print("      堆砌关键词刷分基本无效——这是 BM25 对 TF-IDF 的关键修正")

    # ── 实验三：长度归一化（b 效应） ─────────────────────────────
    print()
    print("=" * 64)
    print("实验三：'退货'都出现 3 次，短文档 vs 长文档（b=0.75）")
    print("=" * 64)
    stats3 = [
        make_stat("短文档", {"退货": 3, **{f"词{i}": 1 for i in range(27)}}),   # 30 词
        make_stat("长文档", {"退货": 3, **{f"词{i}": 1 for i in range(117)}}),  # 120 词
    ] + [make_stat(f"填充{i}", {f"字{i}_{j}": 3 for j in range(10)}) for i in range(6)]
    indexer.build(stats3, collection=demo_collection)
    for r in indexer.query(query_terms=["退货"], top_k=2):
        print(f"    {r['chunk_id']:<6s} score={r['score']:.4f}")
    print("  >>> 同样的词频，长文档被'打折'：它 containing 该词的概率本来就高，")
    print("      信息含量更低——这就是公式里 b × doc_len/avg_len 的作用")

    # 清理演示索引
    index_file = _REPO_ROOT / "data" / "db" / "bm25" / f"{demo_collection}_bm25.json"
    if index_file.exists():
        index_file.unlink()
    print()
    print("=" * 64)
    print("演示完成，演示索引已清理。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
