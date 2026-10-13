#!/usr/bin/env python
"""分块效果演示：动手改"块大小 / 重叠"两个数字，亲眼看检索原料怎么变。

实验内容：
  实验一：真实 PDF 按 默认配置(1000/200) 分块 —— 看块数与块长分布
  实验二：300/0 vs 300/60 —— 对比相邻块边界，理解"重叠"为什么能救上下文
  实验三：极小块尺寸下的降级链 —— 段落切不动切句子，句子切不动切字符

用法:
    python scripts/demo_chunker.py
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
from src.libs.loader.pdf_loader import PdfLoader
from src.libs.splitter.recursive_splitter import RecursiveSplitter


def overlap_length(prev: str, nxt: str) -> int:
    """计算下一块的开头与上一块的结尾能对上的最大字符数（即实际重叠量）。"""
    for size in range(min(len(prev), len(nxt)), 0, -1):
        if prev.endswith(nxt[:size]):
            return size
    return 0


def stats(chunks: list[str]) -> str:
    lengths = [len(c) for c in chunks]
    return (f"{len(chunks)} 块 | 平均 {sum(lengths)//len(lengths)} 字符 | "
            f"最短 {min(lengths)} | 最长 {max(lengths)}")


def boundary_view(chunks: list[str], tail: int = 60) -> None:
    ov = overlap_length(chunks[0], chunks[1])
    print(f"    块0 结尾: …{chunks[0][-tail:]!r}")
    print(f"    块1 开头: {chunks[1][:tail]!r}…")
    print(f"    >>> 两块之间实际重叠 {ov} 个字符")


def main() -> int:
    settings = load_settings(str(_REPO_ROOT / "config" / "settings.yaml"))

    # ── 实验一：真实文档 + 默认配置 ─────────────────────────────
    pdf = _REPO_ROOT / "tests" / "fixtures" / "sample_documents" / "chinese_technical_doc.pdf"
    markdown_text = PdfLoader(extract_images=False).load(pdf).text
    print("=" * 64)
    print(f"实验一：真实文档（{len(markdown_text)} 字符）按默认配置分块")
    print("=" * 64)
    default_splitter = RecursiveSplitter(settings)  # chunk_size=1000, overlap=200
    chunks = default_splitter.split_text(markdown_text)
    print(f"  配置 chunk_size=1000, overlap=200 → {stats(chunks)}")
    print(f"  块0 开头预览: {chunks[0][:100]!r}…")

    # ── 实验二：重叠的作用 ──────────────────────────────────────
    print()
    print("=" * 64)
    print("实验二：chunk_size=300，重叠 0 vs 60 —— 找到'被迫从中间切开'的边界")
    print("=" * 64)
    no_overlap = RecursiveSplitter(settings, chunk_size=300, chunk_overlap=0).split_text(markdown_text)
    with_overlap = RecursiveSplitter(settings, chunk_size=300, chunk_overlap=60).split_text(markdown_text)
    print(f"  无重叠 → {stats(no_overlap)}")
    print(f"  有重叠 → {stats(with_overlap)}")

    # 扫描所有相邻块对，找第一对"实际发生重叠"的边界
    pair_idx = None
    for i in range(len(with_overlap) - 1):
        if overlap_length(with_overlap[i], with_overlap[i + 1]) > 0:
            pair_idx = i
            break
    if pair_idx is None:
        print("  （本文档的所有切分都落在自然边界上，未触发重叠）")
    else:
        prev, nxt = with_overlap[pair_idx], with_overlap[pair_idx + 1]
        ov = overlap_length(prev, nxt)
        print(f"  块{pair_idx} 与 块{pair_idx+1} 的边界被切在句子中间：")
        print(f"    块{pair_idx} 结尾: …{prev[-(ov + 25):]!r}")
        print(f"    块{pair_idx+1} 开头: {nxt[:ov + 25]!r}…")
        print(f"    >>> 重复出现的 {ov} 个字符就是'重叠'：被切断的句子至少完整地活在一个块里")

    # ── 实验三：降级链（分隔符优先级） ──────────────────────────
    print()
    print("=" * 64)
    print("实验三：把块尺寸压到 40 字符，观察'切不动就降级'")
    print("=" * 64)
    structured = (
        "检索增强生成是一种先从外部知识库中检索相关文档片段，再把检索到的内容"
        "与用户问题一起填入提示词，最后由大语言模型基于这些资料生成有据可依的回答的技术架构。"
        "分块的大小与重叠策略直接决定检索环节的召回质量与噪声水平，是整个系统里最需要反复实验的参数。"
    )
    print(f"  原文（{len(structured)} 字符，一整段、无换行、无空格）：")

    tiny_default = RecursiveSplitter(settings, chunk_size=40, chunk_overlap=0)
    tiny_chunks = tiny_default.split_text(structured)
    print(f"  默认分隔符（含中文'。，'）→ {stats(tiny_chunks)}：")
    for i, c in enumerate(tiny_chunks):
        print(f"    块{i}: {c!r}")

    degraded = RecursiveSplitter(settings, chunk_size=40, chunk_overlap=0,
                                 separators=["\n\n", "\n", ". ", " ", ""]).split_text(structured)
    print(f"  去掉中文标点后的分隔符 → {stats(degraded)}（只能按字符硬切，词语被拦腰斩断）：")
    for i, c in enumerate(degraded[:3]):
        print(f"    块{i}: {c!r}")
    print("    >>> 分隔符优先级列表决定'最坏情况'下从哪里下刀；中文文档必须有中文标点")

    return 0


if __name__ == "__main__":
    sys.exit(main())
