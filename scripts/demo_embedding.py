#!/usr/bin/env python
"""嵌入（向量化）演示：亲眼看文字如何变成向量、语义如何变成"夹角"。

实验：
  1. 看一个真实向量的长相（2048 个数字，取前 8 个展示）
  2. 余弦相似度实验：语义近的句子夹角小，无关的句子夹角大
  3. 杀手锏：字面不同意思近 VS 字面相同意思不同 —— 语义检索的本质

用法:
    python scripts/demo_embedding.py
"""

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from src.core.settings import load_settings
from src.libs.embedding.embedding_factory import EmbeddingFactory


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """余弦相似度 = 两个向量的夹角余弦。1=方向相同(语义近) 0=正交(无关) -1=相反。"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> int:
    settings = load_settings(str(_REPO_ROOT / "config" / "settings.yaml"))
    embedder = EmbeddingFactory.create(settings)

    sentences = [
        "如何申请退货？",                       # 0 锚点
        "退货的流程是什么？",                   # 1 字面不同、意思近
        "今天天气真不错，适合出去走走。",       # 2 完全无关
        "苹果手机很好用。",                     # 3
        "iPhone 用起来挺不错的。",              # 4 字面几乎无重叠、意思近
        "苹果很甜很好吃。",                     # 5 共享"苹果"二字、意思完全不同
    ]

    print("=" * 64)
    print(f"用模型 {settings.embedding.model} 把 {len(sentences)} 句话变成向量")
    print("=" * 64)

    vectors = np.array(embedder.embed(sentences))
    print(f"  每句话 → {vectors.shape[1]} 个数字（维度 {settings.embedding.dimensions}）")
    print(f"  第 0 句向量的前 8 个数字: {np.round(vectors[0][:8], 4).tolist()}")
    print(f"  第 0 句向量的长度(模): {np.linalg.norm(vectors[0]):.4f}")
    print("  （向量本身没有可读含义，含义藏在'句与句的夹角'里）")

    print()
    print("-" * 64)
    print("余弦相似度实验（越接近 1 = 语义越近）")
    print("-" * 64)
    pairs = [
        (0, 1, "意思近（字面不同）：退货问法 A vs 退货问法 B"),
        (0, 2, "完全无关：退货 vs 天气"),
        (3, 4, "意思近（字面几乎不重叠）：苹果手机 vs iPhone"),
        (3, 5, "字面相同(苹果)但意思不同：手机 vs 水果"),
    ]
    for i, j, label in pairs:
        sim = cosine_similarity(vectors[i], vectors[j])
        print(f"  {sim:+.4f}  {label}")

    print()
    print(">>> 结论：向量抓的是'意思'，不是'字面'。")
    print(">>> 这就是语义检索能搜到'换一种说法的问题'的原因，")
    print(">>> 也是它需要 BM25 关键词检索来补充精确匹配的原因（第 8 步）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
