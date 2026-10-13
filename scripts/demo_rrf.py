#!/usr/bin/env python
"""RRF（倒数排名融合）演示：手算一遍两路检索结果怎么合并成一路。

核心问题：向量检索给 0.89 这样的分数，BM25 给 12.5 这样的分数——
量纲完全不同，直接相加等于拿体重加身高。RRF 的解法：不看分数，只看排名。

实验：
  1. 手工构造两路排名，逐项展示 1/(k+rank) 的计算过程
  2. 验证 RRF 的核心性质："两路都认可"的文档会胜过"单路第一"
  3. k 参数敏感性：k=1 vs k=60 vs k=1000，看排名差距被如何放大/抹平

用法:
    python scripts/demo_rrf.py
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from src.core.types import RetrievalResult
from src.core.query_engine.fusion import RRFFusion


def make_list(items: list[tuple[str, float]]) -> list[RetrievalResult]:
    """把 (chunk_id, 原始分数) 变成一路排名结果（按分数降序）。"""
    return [
        RetrievalResult(chunk_id=cid, score=score, text=f"《{cid}》的内容", metadata={})
        for cid, score in items
    ]


def show(fused: list[RetrievalResult], k: int) -> None:
    for rank, r in enumerate(fused, 1):
        print(f"    融合第{rank}名  {r.chunk_id:<8s} RRF={r.score:.5f}")


def main() -> int:
    # 场景：查询"GPT-4 价格"
    # 向量检索（懂语义）：价格解释排第 2；BM25（懂字面）：含 GPT-4 字样的排前
    dense = make_list([("语义介绍", 0.91), ("价格解释", 0.88), ("模型评测", 0.75)])
    sparse = make_list([("价格解释", 12.5), ("模型评测", 9.8), ("语义介绍", 3.1)])

    print("=" * 64)
    print("实验一：手算 RRF（k=60）")
    print("=" * 64)
    print("  向量路排名: 1.语义介绍(0.91)  2.价格解释(0.88)  3.模型评测(0.75)")
    print("  关键词路排名: 1.价格解释(12.5) 2.模型评测(9.8)   3.语义介绍(3.1)")
    print("  （注意两路分数 0.91 和 12.5 不可直接相加——量纲不同）")
    print()
    print("  RRF 贡献 = 1/(60+排名)，逐路累加：")
    print("    价格解释 = 1/(60+2) + 1/(60+1) = 0.01613 + 0.01639 = 0.03252  ← 两路都排前")
    print("    语义介绍 = 1/(60+1) + 1/(60+3) = 0.01639 + 0.01587 = 0.03226")
    print("    模型评测 = 1/(60+3) + 1/(60+2) = 0.01587 + 0.01613 = 0.03200")
    fusion = RRFFusion(k=60)
    fused = fusion.fuse([dense, sparse], top_k=5)
    show(fused, 60)
    print("  >>> '价格解释'两路都认可（第2名+第1名），胜过任何一路的第1名！")

    print()
    print("=" * 64)
    print("实验二：单路第 1 vs 双路中游 —— 谁赢？")
    print("=" * 64)
    a = make_list([("单路冠军", 0.99), ("双路中游", 0.70)])
    b = make_list([("无关内容", 5.0), ("双路中游", 4.0)])
    fused2 = RRFFusion(k=60).fuse([a, b], top_k=3)
    for rank, r in enumerate(fused2, 1):
        print(f"    融合第{rank}名  {r.chunk_id:<8s} RRF={r.score:.5f}")
    print("  >>> 单路冠军 = 1/61 = 0.01639；双路中游 = 1/62+1/62 = 0.03226 → 双路碾压")
    print("  >>> 这正是融合的意义：多路证据交叉验证，比单路的高分更可信")

    print()
    print("=" * 64)
    print("实验三：k 参数敏感性 —— 排名差距被放大还是抹平")
    print("=" * 64)
    print(f"{'k':>6s} | 第1名 RRF     | 第1名与第3名的分差")
    print("-" * 46)
    for k in (1, 10, 60, 1000):
        f = RRFFusion(k=k).fuse([dense, sparse], top_k=5)
        gap = f[0].score - f[2].score
        print(f"{k:>6d} | {f[0].score:.5f} ({f[0].chunk_id}) | {gap:.5f}")
    print("  >>> k 越小，头部排名的差距越被放大（赢者通吃）；")
    print("      k 越大，各名次的分差被抹平（更民主）。60 是论文推荐的经验值。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
