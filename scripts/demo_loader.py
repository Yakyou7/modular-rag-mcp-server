#!/usr/bin/env python
"""文档加载器演示：亲眼看一个 PDF 如何变成 Markdown + 图片 + 元数据。

用法:
    python scripts/demo_loader.py [可选:PDF路径，默认用自带的带图测试文档]
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from src.libs.loader.pdf_loader import PdfLoader


def show(loader: PdfLoader, pdf_path: Path, label: str) -> None:
    doc = loader.load(pdf_path)

    print()
    print("=" * 62)
    print(f"【{label}】{pdf_path.name}")
    print("=" * 62)
    print(f"文档 ID   : {doc.id}")
    print(f"标题      : {doc.metadata.get('title')}")
    print(f"内容指纹  : {doc.metadata.get('doc_hash', '')[:16]}…（SHA256 前 16 位）")
    print(f"总字符数  : {len(doc.text)}")

    images = doc.metadata.get("images", [])
    print(f"提取图片数: {len(images)}")
    for img in images:
        print(f"  - {img['id']}  第{img['page']}页  文件={img['path']}  占位符位置={img['text_offset']}")

    print("-" * 62)
    print("Markdown 文本开头 300 字：")
    print(doc.text[:300].replace("\n", " ⏎ ") + " …")

    if images:
        first = images[0]
        start = max(first["text_offset"] - 40, 0)
        print("-" * 62)
        print(f"第 1 张图片占位符前后的文本（位置 {start} 起）：")
        print("…" + doc.text[start:first["text_offset"] + 40].replace("\n", " ⏎ ") + "…")
        stored = _REPO_ROOT / "data" / "images" / doc.metadata["doc_hash"]
        if stored.exists():
            files = sorted(p.name for p in stored.iterdir())
            print(f"磁盘上实际保存的图片文件（{stored.relative_to(_REPO_ROOT)}）: {files}")


def main() -> int:
    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        _REPO_ROOT / "tests" / "fixtures" / "sample_documents" / "with_images.pdf"
    )

    show(PdfLoader(extract_images=True), pdf_path, "开启图片提取")
    show(PdfLoader(extract_images=False), pdf_path, "关闭图片提取（对比：无 images 元数据、无占位符）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
