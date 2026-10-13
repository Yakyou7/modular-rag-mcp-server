"""Unit tests for the GraphRetriever and three-way hybrid search fusion.

Covers:
- Entity matching → graph hits → vector-store hydration → RetrievalResult
- Normalised graph scores + matched-entity metadata
- HybridSearch N-way RRF fusion (dense + sparse + graph)
- Fallback semantics: graph failure is non-fatal; graph rescues when
  dense+sparse both fail; behaviour unchanged when graph absent
- create_graph_retriever factory (disabled / empty index / populated)
"""

from __future__ import annotations

import pytest

from src.core.query_engine.hybrid_search import HybridSearch, HybridSearchConfig
from src.core.query_engine.fusion import RRFFusion
from src.core.types import RetrievalResult
from src.ingestion.storage.graph_store import GraphStore
from src.core.query_engine.graph_retriever import GraphRetriever, create_graph_retriever


# ---------------------------------------------------------------------------
# Fixtures / doubles
# ---------------------------------------------------------------------------

class _StubVectorStore:
    """Minimal vector store double for hydration."""

    def __init__(self, docs: dict) -> None:
        self._docs = docs
        self.fail = False

    def get_by_ids(self, ids, trace=None, **kwargs):
        if self.fail:
            raise RuntimeError("vector store down")
        return [
            {"id": i, "text": self._docs.get(i, {}).get("text", ""),
             "metadata": self._docs.get(i, {}).get("metadata", {})}
            for i in ids if i in self._docs
        ]


def _seeded_store(tmp_path) -> GraphStore:
    store = GraphStore(tmp_path / "g.sqlite3")
    store.upsert_chunk(
        chunk_id="chunk_a",
        entities=[{"name": "RRF"}, {"name": "BM25"}],
        relations=[
            {"source": "混合检索", "target": "RRF", "relation": "使用", "confidence": 0.9},
            {"source": "混合检索", "target": "BM25", "relation": "包含", "confidence": 0.8},
        ],
    )
    store.upsert_chunk(
        chunk_id="chunk_b",
        entities=[{"name": "MCP协议"}],
        relations=[{"source": "MCP协议", "target": "Agent", "relation": "服务", "confidence": 0.7}],
    )
    return store


class _StubRetriever:
    """Dense/sparse double returning a fixed ranked list."""

    def __init__(self, results, error=None) -> None:
        self._results = results
        self._error = error

    def retrieve(self, **kwargs):
        if self._error:
            raise RuntimeError(self._error)
        return list(self._results)


class _StubGraph:
    """Graph retriever double matching the GraphRetriever interface."""

    def __init__(self, results=None, error=None) -> None:
        self._results = results or []
        self._error = error
        self.calls = 0

    def retrieve(self, query, keywords, top_k=10, trace=None, **kwargs):
        self.calls += 1
        if self._error:
            raise RuntimeError(self._error)
        return list(self._results)[:top_k]


def _r(chunk_id: str, score: float = 0.5) -> RetrievalResult:
    return RetrievalResult(chunk_id=chunk_id, score=score, text=f"text-{chunk_id}")


# ---------------------------------------------------------------------------
# GraphRetriever
# ---------------------------------------------------------------------------

class TestGraphRetriever:
    def test_retrieve_hydrates_and_annotates(self, tmp_path) -> None:
        store = _seeded_store(tmp_path)
        vector = _StubVectorStore({
            "chunk_a": {"text": "混合检索用 RRF 融合", "metadata": {"source_path": "doc.pdf"}},
        })
        retriever = GraphRetriever(store=store, vector_store=vector, max_hops=1)

        results = retriever.retrieve(query="混合检索", keywords=["混合检索"], top_k=5)
        assert [r.chunk_id for r in results] == ["chunk_a"]
        assert results[0].score == 1.0  # normalised: top hit = 1.0
        assert results[0].text == "混合检索用 RRF 融合"
        assert results[0].metadata["graph_matched_entities"] == ["混合检索"]
        assert results[0].metadata["graph_rank"] == 1

    def test_retrieve_no_match_returns_empty(self, tmp_path) -> None:
        retriever = GraphRetriever(store=_seeded_store(tmp_path))
        assert retriever.retrieve(query="量子", keywords=["量子力学"]) == []

    def test_retrieve_top_k_limits(self, tmp_path) -> None:
        store = _seeded_store(tmp_path)
        retriever = GraphRetriever(store=store, max_hops=1)
        # Both chunks matched via two entities
        results = retriever.retrieve(
            query="混合检索 MCP协议", keywords=["混合检索", "MCP协议"], top_k=1,
        )
        assert len(results) == 1

    def test_hydration_failure_returns_bare_hits(self, tmp_path) -> None:
        store = _seeded_store(tmp_path)
        vector = _StubVectorStore({})
        vector.fail = True
        retriever = GraphRetriever(store=store, vector_store=vector)
        results = retriever.retrieve(query="混合检索", keywords=["混合检索"])
        assert len(results) == 1
        assert results[0].text == ""  # bare hit, still usable by RRF

    def test_trace_records_graph_stage(self, tmp_path) -> None:
        from src.core.trace.trace_context import TraceContext

        store = _seeded_store(tmp_path)
        retriever = GraphRetriever(store=store)
        trace = TraceContext(trace_type="query")
        retriever.retrieve(query="混合检索", keywords=["混合检索"], trace=trace)
        stage_names = [s["stage"] for s in trace.stages]
        assert "graph_retrieval" in stage_names
        graph_stage = next(s for s in trace.stages if s["stage"] == "graph_retrieval")
        assert graph_stage["data"]["matched_entities"] == ["混合检索"]
        assert graph_stage["data"]["result_count"] == 1


# ---------------------------------------------------------------------------
# HybridSearch three-way fusion
# ---------------------------------------------------------------------------

def _hybrid(dense, sparse, graph=None, config_overrides=None) -> HybridSearch:
    config = HybridSearchConfig(
        dense_top_k=5, sparse_top_k=5, fusion_top_k=10, parallel_retrieval=False,
    )
    if config_overrides:
        for k, v in config_overrides.items():
            setattr(config, k, v)
    return HybridSearch(
        query_processor=None,
        dense_retriever=dense,
        sparse_retriever=sparse,
        fusion=RRFFusion(k=60),
        config=config,
        graph_retriever=graph,
    )


class TestThreeWayFusion:
    def test_n_way_rrf_ordering(self, tmp_path) -> None:
        # dense: a(1) b(2) c(3) | sparse: c(1) | graph: d(1) a(2)
        hybrid = _hybrid(
            _StubRetriever([_r("a"), _r("b"), _r("c")]),
            _StubRetriever([_r("c")]),
            _StubGraph([_r("d"), _r("a")]),
        )
        results = hybrid.search("a b")
        # Weighted RRF (graph weight 0.5):
        #   c: 1/61(dense#3) + 1/61(sparse#1) = 0.03279... wait dense a(1) b(2) c(3):
        #   a: 1/61 + 0.5×1/62 = 0.02445; c: 1/63 + 1/61 = 0.03226;
        #   b: 1/62 = 0.01613; d: 0.5×1/61 = 0.00820
        assert [r.chunk_id for r in results] == ["c", "a", "b", "d"]

    def test_graph_failure_is_non_fatal(self, tmp_path) -> None:
        hybrid = _hybrid(
            _StubRetriever([_r("a")]),
            _StubRetriever([_r("b")]),
            _StubGraph(error="graph boom"),
        )
        detail = hybrid.search("a b", return_details=True)
        assert [r.chunk_id for r in detail.results] == ["a", "b"]
        assert detail.graph_error is not None
        assert detail.used_fallback is False  # two healthy paths still fuse

    def test_graph_rescues_when_both_primary_fail(self, tmp_path) -> None:
        hybrid = _hybrid(
            _StubRetriever([], error="dense down"),
            _StubRetriever([], error="sparse down"),
            _StubGraph([_r("g1"), _r("g2")]),
        )
        detail = hybrid.search("a b", return_details=True)
        assert [r.chunk_id for r in detail.results] == ["g1", "g2"]
        assert detail.used_fallback is True

    def test_all_paths_down_raises(self, tmp_path) -> None:
        hybrid = _hybrid(
            _StubRetriever([], error="dense down"),
            _StubRetriever([], error="sparse down"),
            _StubGraph(error="graph down"),
        )
        with pytest.raises(RuntimeError, match="Both retrieval paths failed"):
            hybrid.search("a b")

    def test_no_graph_behaves_as_before(self, tmp_path) -> None:
        hybrid = _hybrid(
            _StubRetriever([_r("a"), _r("b")]),
            _StubRetriever([_r("b")]),
            graph=None,
        )
        detail = hybrid.search("a b", return_details=True)
        assert [r.chunk_id for r in detail.results] == ["b", "a"]
        assert detail.graph_results is None
        assert detail.graph_error is None

    def test_enable_graph_false_skips_graph(self, tmp_path) -> None:
        graph = _StubGraph([_r("g1")])
        hybrid = _hybrid(
            _StubRetriever([_r("a")]),
            _StubRetriever([_r("b")]),
            graph=graph,
            config_overrides={"enable_graph": False},
        )
        detail = hybrid.search("a b", return_details=True)
        assert graph.calls == 0
        assert detail.graph_results is None

    def test_graph_contributes_to_fusion_input_lists(self, tmp_path) -> None:
        from src.core.trace.trace_context import TraceContext

        hybrid = _hybrid(
            _StubRetriever([_r("a")]),
            _StubRetriever([_r("b")]),
            _StubGraph([_r("c")]),
        )
        trace = TraceContext(trace_type="query")
        hybrid.search("a b", trace=trace)
        fusion_stage = next(s for s in trace.stages if s["stage"] == "fusion")
        assert fusion_stage["data"]["input_lists"] == 3


# ---------------------------------------------------------------------------
# create_graph_retriever factory
# ---------------------------------------------------------------------------

class TestCreateGraphRetriever:
    def _settings(self, enabled: bool, storage_dir: str):
        from types import SimpleNamespace
        return SimpleNamespace(graph=SimpleNamespace(
            enabled=enabled, storage_dir=storage_dir, top_k=10, max_hops=1,
        ))

    def test_disabled_returns_none(self, tmp_path) -> None:
        settings = self._settings(False, str(tmp_path))
        assert create_graph_retriever(settings, "col") is None

    def test_no_graph_section_returns_none(self) -> None:
        from types import SimpleNamespace
        assert create_graph_retriever(SimpleNamespace(), "col") is None

    def test_empty_index_returns_none(self, tmp_path) -> None:
        settings = self._settings(True, str(tmp_path))
        assert create_graph_retriever(settings, "col") is None

    def test_populated_index_returns_retriever(self, tmp_path) -> None:
        settings = self._settings(True, str(tmp_path))
        store = GraphStore(tmp_path / "col_graph.sqlite3")
        store.upsert_chunk(
            chunk_id="c1", entities=[{"name": "RRF"}],
            relations=[{"source": "RRF", "target": "BM25", "relation": "r"}],
        )
        retriever = create_graph_retriever(settings, "col")
        assert isinstance(retriever, GraphRetriever)
        results = retriever.retrieve(query="RRF", keywords=["RRF"], top_k=5)
        assert [r.chunk_id for r in results] == ["c1"]


class TestSettingsGraphSection:
    def test_real_config_parses_graph_section(self) -> None:
        from src.core.settings import load_settings

        settings = load_settings()
        assert settings.graph is not None
        # 实验特性：小语料 A/B 实测净收益为负，默认关闭（见 settings.yaml 注释）
        assert settings.graph.enabled is False
        assert settings.graph.top_k > 0
        assert settings.graph.max_hops in (1, 2)

    def test_missing_graph_section_is_none(self) -> None:
        from src.core.settings import Settings

        settings = Settings.from_dict({
            "llm": {"provider": "openai", "model": "m", "temperature": 0.0, "max_tokens": 10},
            "embedding": {"provider": "openai", "model": "e", "dimensions": 3},
            "vector_store": {"provider": "chroma", "persist_directory": "./x", "collection_name": "c"},
            "retrieval": {"dense_top_k": 1, "sparse_top_k": 1, "fusion_top_k": 1, "rrf_k": 60},
            "rerank": {"enabled": False, "provider": "none", "model": "m", "top_k": 1},
            "evaluation": {"enabled": False, "provider": "custom", "metrics": []},
            "observability": {"log_level": "INFO", "trace_enabled": False,
                              "trace_file": "./t.jsonl", "structured_logging": False},
        })
        assert settings.graph is None
