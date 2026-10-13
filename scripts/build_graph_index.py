#!/usr/bin/env python
"""Build the lightweight graph index for a collection.

Reads all chunks from the collection's vector store, extracts entities and
relations via the configured LLM, and writes a per-collection SQLite graph
index (data/db/graph/{collection}_graph.sqlite3). The index powers the
optional third recall path fused into hybrid search (see settings `graph:`).

Costs: one LLM extraction call per chunk (~free with glm-4-flash).
Re-running is safe: the index is rebuilt from scratch (--no-rebuild to
append instead).

Usage:
    .venv/Scripts/python.exe scripts/build_graph_index.py --collection learning_docs
    .venv/Scripts/python.exe scripts/build_graph_index.py --collection learning_docs --limit 3

Exit codes: 0 success / 2 configuration error / 1 build failure
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the GraphRAG index for a collection.")
    parser.add_argument("--collection", default="learning_docs",
                        help="Collection to index (default: learning_docs)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only index the first N chunks (smoke runs)")
    parser.add_argument("--no-rebuild", action="store_true",
                        help="Append to the existing index instead of clearing it")
    parser.add_argument("--sample", type=int, default=10,
                        help="How many relations to print as a sample (default: 10)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from src.core.settings import load_settings
        settings = load_settings()
    except Exception as exc:
        print(f"❌ Configuration error: {exc}", file=sys.stderr)
        return 2

    # Read chunks straight from the vector store — no re-ingestion needed,
    # works for collections that were ingested before GraphRAG existed.
    try:
        import chromadb

        from src.core.settings import resolve_path
        persist_dir = str(resolve_path(settings.vector_store.persist_directory))
        client = chromadb.PersistentClient(path=persist_dir)
        collection = client.get_collection(args.collection)
        got = collection.get(include=["documents", "metadatas"])
        chunks = [
            {"id": cid, "text": (doc or "")}
            for cid, doc in zip(got["ids"], got["documents"])
        ]
        if not chunks:
            print(f"❌ Collection '{args.collection}' has no chunks.", file=sys.stderr)
            return 2
        if args.limit:
            chunks = chunks[: args.limit]
        print(f"✅ Loaded {len(chunks)} chunks from '{args.collection}'")
    except Exception as exc:
        print(f"❌ Failed to read collection: {exc}", file=sys.stderr)
        return 2

    try:
        from src.ingestion.storage.graph_indexer import GraphIndexer
        from src.ingestion.storage.graph_store import GraphStore

        from src.core.settings import resolve_path
        graph_cfg = getattr(settings, "graph", None)
        storage_dir = getattr(graph_cfg, "storage_dir", "./data/db/graph") if graph_cfg else "./data/db/graph"
        db_path = resolve_path(storage_dir) / f"{args.collection}_graph.sqlite3"
        store = GraphStore(db_path)
        indexer = GraphIndexer(store=store, settings=settings)
        print(f"🔧 Indexer ready: {db_path}")
    except Exception as exc:
        print(f"❌ Failed to prepare indexer: {exc}", file=sys.stderr)
        return 2

    print(f"⏳ Extracting entities/relations via LLM for {len(chunks)} chunks...")
    try:
        stats = indexer.build(chunks, rebuild=not args.no_rebuild)
    except Exception as exc:
        print(f"❌ Build failed: {exc}", file=sys.stderr)
        return 1

    print()
    print("=" * 60)
    print("  GRAPH INDEX BUILD REPORT")
    print("=" * 60)
    for key, value in stats.to_dict().items():
        if key == "failures" and value:
            print(f"  {key:<16s}: {len(value)} (first shown below)")
            for f in value[:5]:
                print(f"    - {f}")
        else:
            print(f"  {key:<16s}: {value}")

    sample = store.sample(limit=args.sample)
    if sample:
        print()
        print("─" * 60)
        print(f"  SAMPLE RELATIONS (top {len(sample)} by confidence)")
        print("─" * 60)
        for rel in sample:
            print(f"  {rel['source']}  --[{rel['relation']}]-->  {rel['target']}"
                  f"  (conf {rel['confidence']}, chunk {rel['chunk_id'][:16]}...)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
