#!/usr/bin/env python
"""多模态检索演示：文字查询命中带图块，原图信息随结果返回（搜文字出图）。

前置：已用 scripts/ingest.py 把 with_images.pdf 摄取进 multimodal_demo 集合。

用法:
    python scripts/demo_multimodal.py
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
from src.core.query_engine.dense_retriever import create_dense_retriever
from src.core.query_engine.sparse_retriever import create_sparse_retriever
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.vector_store.vector_store_factory import VectorStoreFactory


def main() -> int:
    settings = load_settings(str(_REPO_ROOT / "config" / "settings.yaml"))
    collection = "multimodal_demo"

    vector_store = VectorStoreFactory.create(settings, collection_name=collection)
    embedding_client = EmbeddingFactory.create(settings)
    dense_retriever = create_dense_retriever(
        settings=settings, embedding_client=embedding_client, vector_store=vector_store,
    )
    sparse_retriever = create_sparse_retriever(
        settings=settings, bm25_indexer=BM25Indexer(index_dir=f"data/db/bm25/{collection}"),
        vector_store=vector_store,
    )
    sparse_retriever.default_collection = collection
    hybrid = create_hybrid_search(
        settings=settings, query_processor=QueryProcessor(),
        dense_retriever=dense_retriever, sparse_retriever=sparse_retriever,
    )

    query = "文档中的图片是什么内容？"
    print(f"查询：{query}\n")
    results = hybrid.search(query=query, top_k=3)

    def as_list(value):
        """还原被 Chroma 序列化的元数据。实际存储形态五花八门：
        可能是列表、可能是单个 dict 的字符串 repr（清洗器只保留了第一项）——逐一兜底。"""
        import ast
        import json
        if isinstance(value, (list, tuple)):
            return list(value)
        if isinstance(value, dict):
            return [value]
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                parsed = ast.literal_eval(value)
            if isinstance(parsed, dict):
                return [parsed]
            return list(parsed)
        return []

    found_image = False
    for rank, r in enumerate(results, 1):
        caps = as_list(r.metadata.get("image_captions"))
        imgs = as_list(r.metadata.get("images"))
        if not caps and not imgs:
            continue
        found_image = True
        print(f"第 {rank} 名命中带图块（chunk_id={r.chunk_id}）：")
        print(f"  块文本片段: …{r.text[-150:]!r}")
        for cap in caps:
            print(f"  图片 {cap['id']} 的描述（视觉模型生成，已被向量检索命中）:")
            print(f"    > {cap['caption']}")
        for img in imgs:
            p = Path(img["path"])
            print(f"  原图文件: {p}")
            print(f"  文件存在: {p.exists()}  所在页: 第{img.get('page')}页")

    if not found_image:
        print("（前几名结果中没有带图块）")

    print()
    print(">>> 关键机制：图片在第 4 步存盘、第 5 步描述文字缝进块文本，")
    print(">>> 于是'用文字搜图片内容'复用纯文本检索管道，零额外组件。")
    print(">>> 响应层（multimodal_assembler）可凭 metadata 里的路径把原图回传给用户。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
