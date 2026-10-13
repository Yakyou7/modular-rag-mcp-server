#!/usr/bin/env python
"""End-to-end RAGAS evaluation script.

Runs the full RAG pipeline (hybrid retrieval + optional rerank + LLM answer
generation) over a golden test set, then scores every case with Ragas
LLM-as-Judge metrics (faithfulness / answer_relevancy / context_precision)
and prints an aggregated report.

Difference vs scripts/evaluate.py:
- evaluate.py: retrieval-only metrics (hit_rate / MRR) via CustomEvaluator,
  answers are placeholders — cheap, no generation.
- this script: adds LLM answer generation + Ragas judge metrics, i.e. it
  evaluates the GENERATION side of the stack. Costs LLM tokens
  (≈ 1 generation + 1-2 judge calls per test case).

Usage:
    .venv/Scripts/python.exe scripts/run_ragas_eval.py
    .venv/Scripts/python.exe scripts/run_ragas_eval.py --collection learning_docs --limit 5
    .venv/Scripts/python.exe scripts/run_ragas_eval.py --metrics faithfulness

Exit codes:
    0 - Success
    1 - Evaluation failure
    2 - Configuration error
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUTPUT = PROJECT_ROOT / "logs" / "ragas_report.json"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run end-to-end RAGAS evaluation (retrieval + generation + LLM judge)."
    )
    parser.add_argument(
        "--test-set",
        default="config/learning_golden_test_set.json",
        help="Path to golden test set JSON (default: config/learning_golden_test_set.json)",
    )
    parser.add_argument(
        "--collection",
        default="learning_docs",
        help="Collection name to search within (default: learning_docs)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve per query (default: 5).",
    )
    parser.add_argument(
        "--metrics",
        default="faithfulness,answer_relevancy",
        help="Comma-separated ragas metrics: faithfulness, answer_relevancy, "
             "context_precision (default: faithfulness,answer_relevancy)",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable the reranker for this run (cheaper, faster).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate the first N test cases (smoke runs).",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Path to save the JSON report (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    try:
        from src.core.settings import load_settings
        settings = load_settings()
    except Exception as exc:
        print(f"❌ Configuration error: {exc}", file=sys.stderr)
        return 2

    # RagasEvaluator imports ragas at construction — fail early with a
    # clear message instead of mid-run.
    try:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator
        metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
        evaluator = RagasEvaluator(settings=settings, metrics=metrics)
    except ImportError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"❌ Failed to create RagasEvaluator: {exc}", file=sys.stderr)
        return 2

    # Build retrieval stack for the target collection
    try:
        from src.core.query_engine.query_processor import QueryProcessor
        from src.core.query_engine.hybrid_search import create_hybrid_search
        from src.core.query_engine.dense_retriever import create_dense_retriever
        from src.core.query_engine.sparse_retriever import create_sparse_retriever
        from src.core.query_engine.reranker import create_core_reranker
        from src.ingestion.storage.bm25_indexer import BM25Indexer
        from src.libs.embedding.embedding_factory import EmbeddingFactory
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory

        collection = args.collection
        vector_store = VectorStoreFactory.create(settings, collection_name=collection)
        embedding_client = EmbeddingFactory.create(settings)
        dense_retriever = create_dense_retriever(
            settings=settings,
            embedding_client=embedding_client,
            vector_store=vector_store,
        )
        bm25_indexer = BM25Indexer(index_dir=str(PROJECT_ROOT / "data" / "db" / "bm25" / collection))
        sparse_retriever = create_sparse_retriever(
            settings=settings,
            bm25_indexer=bm25_indexer,
            vector_store=vector_store,
        )
        sparse_retriever.default_collection = collection

        # Optional third recall path (GraphRAG)
        try:
            from src.core.query_engine.graph_retriever import create_graph_retriever
            graph_retriever = create_graph_retriever(
                settings, collection, vector_store=vector_store,
            )
        except Exception as exc:
            print(f"⚠️  Graph retriever unavailable: {exc}")
            graph_retriever = None

        hybrid_search = create_hybrid_search(
            settings=settings,
            query_processor=QueryProcessor(),
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            graph_retriever=graph_retriever,
        )
        print(f"✅ HybridSearch initialized for collection: {collection}"
              + ("" if graph_retriever is None else " (+graph path)"))
    except Exception as exc:
        print(f"❌ Failed to initialize retrieval: {exc}", file=sys.stderr)
        return 2

    # Optional reranker (LLM rerank costs one extra LLM call per query)
    reranker = None
    if not args.no_rerank:
        try:
            reranker = create_core_reranker(settings=settings)
            print(f"✅ Reranker enabled: {getattr(reranker, 'reranker_type', 'unknown')}")
        except Exception as exc:
            print(f"⚠️  Reranker unavailable, continuing without: {exc}")

    # LLM answer generation (grounded prompt)
    try:
        from src.observability.evaluation.answer_generator import RagAnswerGenerator
        answer_generator = RagAnswerGenerator(settings=settings)
        print("✅ Answer generator ready (LLM-backed, grounded prompt)")
    except Exception as exc:
        print(f"❌ Failed to create answer generator: {exc}", file=sys.stderr)
        return 2

    from src.observability.evaluation.eval_runner import EvalRunner

    # Apply the smoke-run limit BEFORE running (slice the test set into a
    # trimmed copy so we don't pay LLM tokens for cases we won't report).
    test_set_path = args.test_set
    if args.limit:
        import tempfile

        with open(args.test_set, encoding="utf-8") as f:
            data = json.load(f)
        data["test_cases"] = data["test_cases"][: args.limit]
        trimmed = Path(tempfile.gettempdir()) / "ragas_eval_trimmed_test_set.json"
        trimmed.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        test_set_path = str(trimmed)
        print(f"✂️  Limited to first {args.limit} test cases ({trimmed})")

    runner = EvalRunner(
        settings=settings,
        hybrid_search=hybrid_search,
        evaluator=evaluator,
        answer_generator=answer_generator,
        reranker=reranker,
    )

    print(f"\n🔍 RAGAS evaluation: metrics={metrics}")
    print(f"📄 Test set: {args.test_set}")
    print(f"🔢 Top-K: {args.top_k} | collection: {args.collection} | rerank: {not args.no_rerank}")
    print("⏳ This costs LLM tokens (generation + judge calls per case)...\n")

    t0 = time.monotonic()
    try:
        report = runner.run(
            test_set_path=test_set_path,
            top_k=args.top_k,
            collection=args.collection,
        )
    except Exception as exc:
        print(f"❌ Evaluation failed: {exc}", file=sys.stderr)
        return 1

    # Persist report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_dict = report.to_dict()
    report_dict["run_config"] = {
        "metrics": metrics,
        "top_k": args.top_k,
        "collection": args.collection,
        "rerank": not args.no_rerank,
        "limit": args.limit,
    }
    output_path.write_text(
        json.dumps(report_dict, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _print_report(report, wall_s=time.monotonic() - t0)
    print(f"💾 Report saved to: {output_path}")
    return 0


def _print_report(report, wall_s: float) -> None:
    """Print a formatted evaluation report."""
    print("=" * 60)
    print("  RAGAS EVALUATION REPORT")
    print("=" * 60)
    print(f"  Evaluator: {report.evaluator_name}")
    print(f"  Test Set:  {report.test_set_path}")
    print(f"  Queries:   {len(report.query_results)}")
    print(f"  Pipeline:  {report.total_elapsed_ms:.0f} ms | wall {wall_s:.0f} s")
    print()

    print("─" * 60)
    print("  AGGREGATE METRICS (0-1, higher is better)")
    print("─" * 60)
    if report.aggregate_metrics:
        for metric, value in sorted(report.aggregate_metrics.items()):
            bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
            print(f"  {metric:<25s} {bar} {value:.4f}")
    else:
        print("  (no metrics computed)")
    print()

    print("─" * 60)
    print("  PER-QUERY RESULTS")
    print("─" * 60)
    for i, qr in enumerate(report.query_results, 1):
        scores = " ".join(f"{k}={v:.3f}" for k, v in sorted(qr.metrics.items()))
        flag = "" if qr.metrics else "  (metrics unavailable)"
        print(f"  [{i:2d}] {scores or '(no metrics)'}{flag}  «{qr.query[:36]}»")

    print()
    print("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
