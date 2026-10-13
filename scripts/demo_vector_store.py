#!/usr/bin/env python
"""向量数据库演示：用 2 维向量（人眼可验证）看清 Chroma 的四个核心行为。

实验：
  1. 写入 + 最近邻检索 —— 向量库只管"找最近的"，不关心向量从哪来
  2. upsert 幂等性 —— 同 ID 再写入是"更新"不是"新增"
  3. 元数据过滤 —— "先按条件筛，再找最近"
  4. 距离 → 相似度分数的换算

用法:
    python scripts/demo_vector_store.py
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
from src.libs.vector_store.chroma_store import ChromaStore


def rec(id_: str, vector: list[float], **meta) -> dict:
    return {"id": id_, "vector": vector, "metadata": meta or {}}


def show(store: ChromaStore, title: str, query_vec: list[float], filters=None, top_k: int = 3) -> None:
    print(f"\n  查询向量 {query_vec}" + (f"，过滤条件 {filters}" if filters else ""))
    results = store.query(vector=query_vec, top_k=top_k, filters=filters)
    for r in results:
        print(f"    命中 {r['id']:<28s} score={r['score']:.4f}  metadata={r['metadata']}")


def main() -> int:
    settings = load_settings(str(_REPO_ROOT / "config" / "settings.yaml"))
    demo_collection = "demo_step7"

    # 用独立的演示集合，结束即删除，不污染真实知识库
    store = ChromaStore(settings, collection_name=demo_collection)
    store.clear()
    print("=" * 64)
    print(f"演示集合 {demo_collection!r} 已就绪（当前 {store.get_collection_stats()["count"]} 条）")

    # ── 实验一：写入 + 最近邻检索 ────────────────────────────────
    print()
    print("-" * 64)
    print("实验一：写入 3 条 2 维向量，查'谁离 [1, 0] 最近'")
    store.upsert([
        rec("vec_right", [1.0, 0.0], label="正右方"),
        rec("vec_up",    [0.0, 1.0], label="正上方"),
        rec("vec_ne",    [0.9, 0.1], label="偏右上的邻居"),
    ])
    print(f"  写入后集合共 {store.get_collection_stats()["count"]} 条")
    show(store, "最近邻", [1.0, 0.0])
    print("  >>> 结果按相似度降序：vec_right 完全同向(score=1.0)，vec_ne 次之，vec_up 正交")

    # ── 实验二：upsert 幂等性 ────────────────────────────────────
    print()
    print("-" * 64)
    print("实验二：把 vec_right 的向量'改指向'正上方 —— 同 ID 再写入")
    store.upsert([rec("vec_right", [0.0, 1.0], label="改成正上方了")])
    print(f"  upsert 后集合共 {store.get_collection_stats()["count"]} 条（没有变多！同 ID 是更新不是新增）")
    show(store, "更新后再查", [1.0, 0.0])
    print("  >>> vec_right 从第一名掉到与 vec_up 并列 —— 幂等写入是'重跑管道不产生脏数据'的基石")

    # ── 实验三：元数据过滤 ───────────────────────────────────────
    print()
    print("-" * 64)
    print("实验三：先按元数据筛选，再找最近（where 过滤）")
    store.clear()
    store.upsert([
        rec("doc_a_1", [1.0, 0.0], category="alpha"),
        rec("doc_a_2", [0.8, 0.2], category="alpha"),
        rec("doc_b_1", [0.95, 0.05], category="beta"),
    ])
    show(store, "不过滤", [1.0, 0.0])
    show(store, "只看 beta", [1.0, 0.0], filters={"category": "beta"})
    print("  >>> 典型用途：只在某份文档/某个类目里检索（多租户、权限隔离同理）")

    # 清理演示数据
    store.clear()
    print()
    print("=" * 64)
    print(f"演示完成，演示集合已清空（当前 {store.get_collection_stats()["count"]} 条）。真实知识库 learning_docs 未受影响。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
