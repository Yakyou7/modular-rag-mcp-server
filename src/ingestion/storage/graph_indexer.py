"""Entity-relation extraction and graph index building.

``EntityRelationExtractor`` asks the configured LLM to extract entities and
relations from one chunk, parsing a strict JSON answer (with tolerance for
markdown fences). ``GraphIndexer`` runs extraction over a set of chunks and
writes the results into a :class:`GraphStore`.

Failure policy: extraction is LLM-dependent, so a chunk whose extraction
fails is skipped and counted — the index remains buildable, mirroring the
project's graceful-degradation convention (same spirit as the BM25 index
rebuild). A future ingestion hook can call ``GraphIndexer.build`` with the
chunk set produced by the pipeline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.core.settings import resolve_path
from src.ingestion.storage.graph_store import GraphStore

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_PATH = "config/prompts/graph_extraction.txt"


class GraphExtractionError(RuntimeError):
    """Raised when the LLM answer cannot be parsed into entities/relations."""


@dataclass
class GraphIndexStats:
    """Build statistics for one indexing run."""

    chunks_total: int = 0
    chunks_indexed: int = 0
    chunks_failed: int = 0
    entities: int = 0
    relations: int = 0
    failures: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunks_total": self.chunks_total,
            "chunks_indexed": self.chunks_indexed,
            "chunks_failed": self.chunks_failed,
            "entities": self.entities,
            "relations": self.relations,
            "failures": self.failures[:20],
        }


class EntityRelationExtractor:
    """LLM-based entity/relation extraction with strict JSON parsing."""

    def __init__(
        self,
        settings: Any = None,
        llm: Optional[Any] = None,
        prompt_path: Optional[str] = None,
    ) -> None:
        """Initialise the extractor.

        Args:
            settings: Application settings (builds the LLM when none injected).
            llm: Optional pre-built LLM exposing ``chat([Message])`` — tests.
            prompt_path: Optional prompt override (default: the graph
                extraction prompt under config/prompts/).
        """
        if llm is None:
            from src.libs.llm.llm_factory import LLMFactory

            llm = LLMFactory.create(settings)
        self.llm = llm
        path = Path(prompt_path) if prompt_path else resolve_path(DEFAULT_PROMPT_PATH)
        self.prompt_template = path.read_text(encoding="utf-8")

    def extract(self, text: str) -> Dict[str, List[Dict]]:
        """Extract entities and relations from one chunk's text.

        Args:
            text: Chunk content.

        Returns:
            Dict with ``entities`` and ``relations`` lists (raw dicts as
            produced by the LLM, validated minimally).

        Raises:
            GraphExtractionError: On LLM failure or malformed JSON.
        """
        if not text or not text.strip():
            raise GraphExtractionError("Cannot extract from empty text")

        prompt = f"{self.prompt_template}\n\nChunk:\n{text.strip()}"
        try:
            from src.libs.llm.base_llm import Message

            response = self.llm.chat([Message(role="user", content=prompt)])
            raw = response.content or ""
        except Exception as exc:
            raise GraphExtractionError(f"LLM call failed during extraction: {exc}") from exc

        return self._parse(raw)

    def _parse(self, raw: str) -> Dict[str, List[Dict]]:
        """Parse the LLM answer, tolerating markdown fences."""
        text = (raw or "").strip()
        if text.startswith("```"):
            text = text.split("```")[1] if "```" in text[3:] else text[3:]
            if text.lower().startswith("json"):
                text = text[4:]
        text = text.strip()
        # Tolerate trailing prose: keep from first '{' to last '}'
        if "{" in text and "}" in text:
            text = text[text.index("{"): text.rindex("}") + 1]

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise GraphExtractionError(f"Extraction answer is not valid JSON: {exc}") from exc

        if not isinstance(parsed, dict):
            raise GraphExtractionError("Extraction answer must be a JSON object")

        entities = parsed.get("entities") or []
        relations = parsed.get("relations") or []
        if not isinstance(entities, list) or not isinstance(relations, list):
            raise GraphExtractionError("'entities' and 'relations' must be arrays")

        clean_entities = [
            e for e in entities if isinstance(e, dict) and (e.get("name") or "").strip()
        ]
        clean_relations = [
            r for r in relations
            if isinstance(r, dict) and (r.get("source") or "").strip() and (r.get("target") or "").strip()
        ]
        return {"entities": clean_entities, "relations": clean_relations}


class GraphIndexer:
    """Builds a GraphStore from chunk texts via LLM extraction."""

    def __init__(
        self,
        store: GraphStore,
        extractor: Optional[EntityRelationExtractor] = None,
        settings: Any = None,
    ) -> None:
        """Initialise the indexer.

        Args:
            store: Target GraphStore.
            extractor: Optional pre-built extractor (tests).
            settings: Application settings (creates the default extractor).
        """
        self.store = store
        self.extractor = extractor or EntityRelationExtractor(settings=settings)

    def build(
        self,
        chunks: Iterable[Dict[str, Any]],
        rebuild: bool = True,
    ) -> GraphIndexStats:
        """Extract and index entities/relations for the given chunks.

        Args:
            chunks: Iterable of dicts with ``id`` (or ``chunk_id``) and
                ``text`` — e.g. Chroma ``collection.get()`` rows.
            rebuild: When True (default), clears the store first so the
                index reflects exactly the given chunk set.

        Returns:
            GraphIndexStats with per-chunk success/failure accounting.
        """
        chunk_list = [
            {
                "id": c.get("id") or c.get("chunk_id"),
                "text": c.get("text") or "",
            }
            for c in chunks
            if isinstance(c, dict)
        ]
        chunk_list = [c for c in chunk_list if c["id"]]

        stats = GraphIndexStats(chunks_total=len(chunk_list))
        if rebuild:
            self.store.clear()

        for chunk in chunk_list:
            try:
                parsed = self.extractor.extract(chunk["text"])
                self.store.upsert_chunk(
                    chunk_id=chunk["id"],
                    entities=parsed["entities"],
                    relations=parsed["relations"],
                )
                stats.chunks_indexed += 1
            except Exception as exc:
                stats.chunks_failed += 1
                stats.failures.append(f"{chunk['id']}: {exc}")
                logger.warning("Graph extraction failed for %s: %s", chunk["id"], exc)

        stats.entities, stats.relations = self.store.stats()
        logger.info(
            "Graph index built: %d/%d chunks, %d entities, %d relations (%d failed)",
            stats.chunks_indexed, stats.chunks_total, stats.entities,
            stats.relations, stats.chunks_failed,
        )
        return stats
