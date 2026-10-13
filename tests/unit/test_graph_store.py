"""Unit tests for the lightweight graph store, extractor and indexer.

All tests are offline: extraction uses a mock LLM, storage uses tmp SQLite
files.
"""

from __future__ import annotations

import pytest

from src.ingestion.storage.graph_indexer import (
    EntityRelationExtractor,
    GraphExtractionError,
    GraphIndexer,
)
from src.ingestion.storage.graph_store import (
    WEIGHT_HOP2,
    WEIGHT_SUBSTRING,
    GraphStore,
)


@pytest.fixture
def store(tmp_path) -> GraphStore:
    return GraphStore(tmp_path / "test_graph.sqlite3")


def _seed(store: GraphStore) -> None:
    """Two chunks: RRF—融合—BM25 (chunk a), 重排序—精排—Cross-Encoder (chunk b)."""
    store.upsert_chunk(
        chunk_id="chunk_a",
        entities=[{"name": "RRF", "type": "concept"}, {"name": "BM25", "type": "concept"},
                  {"name": "混合检索", "type": "concept"}],
        relations=[
            {"source": "混合检索", "target": "RRF", "relation": "使用", "confidence": 0.9},
            {"source": "混合检索", "target": "BM25", "relation": "包含", "confidence": 0.8},
        ],
    )
    store.upsert_chunk(
        chunk_id="chunk_b",
        entities=[{"name": "重排序", "type": "concept"}, {"name": "Cross-Encoder", "type": "model"}],
        relations=[
            {"source": "重排序", "target": "Cross-Encoder", "relation": "使用", "confidence": 0.7},
        ],
    )


class TestGraphStore:
    def test_upsert_and_stats(self, store: GraphStore) -> None:
        _seed(store)
        entities, relations = store.stats()
        assert entities == 5
        assert relations == 3

    def test_clear_empties_store(self, store: GraphStore) -> None:
        _seed(store)
        store.clear()
        assert store.stats() == (0, 0)

    def test_find_entities_exact_match(self, store: GraphStore) -> None:
        _seed(store)
        matches = store.find_entities(["rrf"])
        assert matches and matches[0].name == "RRF"
        assert matches[0].match_type == "exact"
        assert matches[0].weight == 1.0

    def test_find_entities_substring_match(self, store: GraphStore) -> None:
        _seed(store)
        # keyword "重排" must still hit entity "重排序" (Chinese substring)
        matches = store.find_entities(["重排"])
        assert matches and matches[0].name == "重排序"
        assert matches[0].match_type == "substring"
        assert matches[0].weight == WEIGHT_SUBSTRING

    def test_find_entities_no_match(self, store: GraphStore) -> None:
        _seed(store)
        assert store.find_entities(["量子力学"]) == []
        assert store.find_entities([]) == []

    def test_related_chunks_one_hop(self, store: GraphStore) -> None:
        _seed(store)
        hits = store.related_chunks(["混合检索"], max_hops=1)
        assert [h.chunk_id for h in hits] == ["chunk_a"]
        assert hits[0].score == pytest.approx(0.9 + 0.8)
        assert set(hits[0].matched_entities) == {"混合检索"}

    def test_related_chunks_two_hop_with_decay(self, store: GraphStore) -> None:
        # chunk_b is 2 hops away from 混合检索 via 重排序? No — separate graph.
        # Build a chain instead: A—B—C where query matches A only.
        store.upsert_chunk(
            chunk_id="c1",
            entities=[{"name": "A"}],
            relations=[{"source": "A", "target": "B", "relation": "相邻", "confidence": 1.0}],
        )
        store.upsert_chunk(
            chunk_id="c2",
            entities=[{"name": "B"}],
            relations=[{"source": "B", "target": "C", "relation": "相邻", "confidence": 1.0}],
        )
        one_hop = {h.chunk_id: h.score for h in store.related_chunks(["a"], max_hops=1)}
        assert one_hop == {"c1": 1.0}

        two_hop = {h.chunk_id: h.score for h in store.related_chunks(["a"], max_hops=2)}
        # c1 direct (df=1 → full weight), c2 via adjacent entity B:
        # hop2 weight × B's own hub damping (df(B)=2 → 1/sqrt(2))
        assert two_hop["c1"] == pytest.approx(1.0)
        assert two_hop["c2"] == pytest.approx(1.0 * WEIGHT_HOP2 * (1 / 2 ** 0.5))

    def test_upsert_chunk_is_replaceable(self, store: GraphStore) -> None:
        _seed(store)
        # Re-index chunk_a with different relations — old ones must go
        store.upsert_chunk(
            chunk_id="chunk_a",
            entities=[{"name": "RRF"}],
            relations=[{"source": "RRF", "target": "BM25", "relation": "对比", "confidence": 0.5}],
        )
        hits = {h.chunk_id for h in store.related_chunks(["混合检索"])}
        assert "chunk_a" not in hits  # old relations removed with the chunk
        _, relations = store.stats()
        assert relations == 2  # 1 new + 1 from chunk_b


class TestEntityRelationExtractor:
    def test_parse_plain_json(self, tmp_path) -> None:
        prompt = tmp_path / "p.txt"
        prompt.write_text("PROMPT", encoding="utf-8")
        answer = '{"entities": [{"name": "RRF", "type": "concept"}], "relations": []}'
        extractor = EntityRelationExtractor(llm=_LLMShim(answer), prompt_path=str(prompt))
        parsed = extractor.extract("RRF 是一种融合算法")
        assert parsed["entities"][0]["name"] == "RRF"
        assert parsed["relations"] == []

    def test_parse_fenced_json(self, tmp_path) -> None:
        prompt = tmp_path / "p.txt"
        prompt.write_text("PROMPT", encoding="utf-8")
        answer = '```json\n{"entities": [{"name": "BM25"}], "relations": [{"source": "BM25", "target": "RRF", "relation": "对比", "confidence": 0.7}]}\n```'
        extractor = EntityRelationExtractor(llm=_LLMShim(answer), prompt_path=str(prompt))
        parsed = extractor.extract("text")
        assert len(parsed["relations"]) == 1

    def test_json_embedded_in_prose(self, tmp_path) -> None:
        prompt = tmp_path / "p.txt"
        prompt.write_text("PROMPT", encoding="utf-8")
        answer = '好的，以下是结果：\n{"entities": [{"name": "MCP"}], "relations": []}\n希望有帮助'
        extractor = EntityRelationExtractor(llm=_LLMShim(answer), prompt_path=str(prompt))
        assert extractor.extract("t")["entities"][0]["name"] == "MCP"

    def test_malformed_json_raises(self, tmp_path) -> None:
        prompt = tmp_path / "p.txt"
        prompt.write_text("PROMPT", encoding="utf-8")
        extractor = EntityRelationExtractor(llm=_LLMShim("不是 JSON"), prompt_path=str(prompt))
        with pytest.raises(GraphExtractionError, match="valid JSON"):
            extractor.extract("t")

    def test_empty_text_raises(self, tmp_path) -> None:
        prompt = tmp_path / "p.txt"
        prompt.write_text("PROMPT", encoding="utf-8")
        extractor = EntityRelationExtractor(llm=_LLMShim("{}"), prompt_path=str(prompt))
        with pytest.raises(GraphExtractionError, match="empty"):
            extractor.extract("  ")

    def test_llm_failure_raises(self, tmp_path) -> None:
        prompt = tmp_path / "p.txt"
        prompt.write_text("PROMPT", encoding="utf-8")
        extractor = EntityRelationExtractor(llm=_LLMShim("", fail=True), prompt_path=str(prompt))
        with pytest.raises(GraphExtractionError, match="LLM call failed"):
            extractor.extract("t")


class _LLMShim:
    """Minimal LLM double returning a fixed answer."""

    def __init__(self, content: str, fail: bool = False) -> None:
        self._content = content
        self._fail = fail

    def chat(self, messages, **kwargs):
        if self._fail:
            raise RuntimeError("LLM down")

        class _Resp:
            content = self._content

        return _Resp()


class TestGraphIndexer:
    def test_build_indexes_all_chunks(self, store: GraphStore, tmp_path) -> None:
        prompt = tmp_path / "p.txt"
        prompt.write_text("P", encoding="utf-8")

        answers = [
            '{"entities": [{"name": "RRF"}], "relations": [{"source": "RRF", "target": "BM25", "relation": "r", "confidence": 0.9}]}',
            '{"entities": [{"name": "MCP"}], "relations": []}',
        ]
        extractor = EntityRelationExtractor(llm=_SeqLLM(answers), prompt_path=str(prompt))
        indexer = GraphIndexer(store=store, extractor=extractor)
        stats = indexer.build([
            {"id": "c1", "text": "chunk one"},
            {"id": "c2", "text": "chunk two"},
        ])
        assert stats.chunks_indexed == 2
        assert stats.chunks_failed == 0
        # RRF (listed) + BM25 (auto-registered from relation endpoint) + MCP
        assert stats.entities == 3
        assert stats.relations == 1

    def test_build_skips_failed_chunks(self, store: GraphStore, tmp_path) -> None:
        prompt = tmp_path / "p.txt"
        prompt.write_text("P", encoding="utf-8")
        answers = [
            "垃圾输出",  # unparseable → failed
            '{"entities": [{"name": "MCP"}], "relations": []}',
        ]
        extractor = EntityRelationExtractor(llm=_SeqLLM(answers), prompt_path=str(prompt))
        indexer = GraphIndexer(store=store, extractor=extractor)
        stats = indexer.build([{"id": "c1", "text": "a"}, {"id": "c2", "text": "b"}])
        assert stats.chunks_indexed == 1
        assert stats.chunks_failed == 1
        assert stats.failures and "c1" in stats.failures[0]

    def test_rebuild_clears_previous_index(self, store: GraphStore) -> None:
        _seed(store)

        class _EmptyExtractor:
            def extract(self, text):
                return {"entities": [], "relations": []}

        indexer = GraphIndexer(store=store, extractor=_EmptyExtractor())
        stats = indexer.build([{"id": "c_new", "text": "new"}], rebuild=True)
        assert stats.chunks_indexed == 1
        # Old seed data gone
        assert store.find_entities(["rrf"]) == []
        assert store.related_chunks(["混合检索"]) == []


class _SeqLLM:
    """LLM double returning queued answers."""

    def __init__(self, answers) -> None:
        self._answers = list(answers)

    def chat(self, messages, **kwargs):
        class _Resp:
            content = self._answers.pop(0)

        return _Resp()
