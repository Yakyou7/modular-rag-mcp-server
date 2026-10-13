#!/usr/bin/env python
"""模型服务连通性冒烟测试（先测零件，再测整机）。

逐个验证 settings.yaml 中配置的三类模型接口是否可用：
  1. 对话模型（llm）          —— 负责组织答案、精炼分块
  2. 向量化模型（embedding）   —— 负责把文字变成向量
  3. 视觉模型（vision_llm）    —— 负责给 PDF 里的图片写描述

用法:
    python scripts/smoke_test_providers.py

退出码: 0 全部通过; 1 有失败项（看输出定位是哪一把"钥匙"出了问题）
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
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.llm.base_llm import Message
from src.libs.llm.base_vision_llm import ImageInput
from src.libs.llm.llm_factory import LLMFactory


def main() -> int:
    settings = load_settings(str(_REPO_ROOT / "config" / "settings.yaml"))
    results: dict[str, str] = {}

    # ── 1) 对话模型 ──────────────────────────────────────────────
    try:
        llm = LLMFactory.create(settings)
        resp = llm.chat([Message(role="user", content="请只回复四个字：连通正常")])
        results["对话模型  "] = f"OK  回复：{resp.content[:50]!r}"
    except Exception as exc:  # noqa: BLE001 — 冒烟测试要展示任何异常
        results["对话模型  "] = f"失败  {type(exc).__name__}: {exc}"

    # ── 2) 向量化模型 ────────────────────────────────────────────
    try:
        embedding = EmbeddingFactory.create(settings)
        vectors = embedding.embed(["这是一段用于测试的文本"])
        dims = len(vectors[0])
        note = "与配置一致" if dims == settings.embedding.dimensions else f"与配置声明的 {settings.embedding.dimensions} 不一致!"
        results["向量化模型"] = f"OK  返回 {len(vectors)} 个向量，维度 {dims}（{note}）"
    except Exception as exc:  # noqa: BLE001
        results["向量化模型"] = f"失败  {type(exc).__name__}: {exc}"

    # ── 3) 视觉模型 ──────────────────────────────────────────────
    try:
        vision_llm = LLMFactory.create_vision_llm(settings)
        image_path = _REPO_ROOT / "tests" / "fixtures" / "sample_documents" / "test_vision_llm.jpg"
        if not image_path.exists():
            results["视觉模型  "] = "跳过  找不到测试图片 test_vision_llm.jpg"
        else:
            resp = vision_llm.chat_with_image(
                "请用一句话描述这张图片的内容",
                ImageInput(path=str(image_path)),
            )
            results["视觉模型  "] = f"OK  回复：{resp.content[:60]!r}"
    except Exception as exc:  # noqa: BLE001
        results["视觉模型  "] = f"失败  {type(exc).__name__}: {exc}"

    print()
    print("=" * 60)
    print("模型服务连通性测试（smoke test）")
    print("=" * 60)
    for name, status in results.items():
        print(f"  [{name}] {status}")
    print("=" * 60)

    any_failed = any(s.startswith("失败") for s in results.values())
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
