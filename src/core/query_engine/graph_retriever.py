"""Graph retriever: the third recall path for hybrid search.

Given query keywords, matches them against the indexed entity dictionary,
expands relations (1-hop by default, optional 2-hop with weight decay), and
scores the chunks linked to those relations. Hits are hydrated from the
vector store into standard ``RetrievalResult`` objects so the RRF fusion
treats the graph path exactly like dense/sparse — the graph's own scores
only order the path's candidates; the fused ranking is rank-based.

Graceful degradation: a missing or empty graph index yields an empty result
list (never an exception), so hybrid search silently falls back to
dense+sparse — consistent with the project's single-path failure policy.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, List, Optional

from src.core.types import RetrievalResult
from src.ingestion.storage.graph_store import WEIGHT_EXACT, GraphStore

if TYPE_CHECKING:
    from src.libs.vector_store.base_vector_store import BaseVectorStore

logger = logging.getLogger(__name__)


class GraphRetriever:
    """Entity-relation retrieval over the lightweight graph index."""

    def __init__(
        self,
        store: GraphStore,
        vector_store: Optional[BaseVectorStore] = None,
        max_hops: int = 1,
        collection: Optional[str] = None,
    ) -> None:
        """Initialise the retriever.

        Args:
            store: GraphStore for the target collection.
            vector_store: Used to hydrate chunk text/metadata for hits
                (same pattern as SparseRetriever). If None, hits are
                returned with empty text (still usable by RRF).
            max_hops: Entity expansion depth (1 or 2).
            collection: Collection name for trace annotations.
        """
        self.store = store
        self.vector_store = vector_store
        self.max_hops = max(1, int(max_hops))
        self.collection = collection

    def retrieve(
        self,
        query: str,
        keywords: List[str],
        top_k: int = 10,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[RetrievalResult]:
        """Retrieve chunks via entity matching + relation expansion.

        Args:
            query: Original query string (used for substring matching
                fallback when keywords miss).
            keywords: Processed query keywords from QueryProcessor.
            top_k: Maximum results.
            trace: Optional TraceContext.

        Returns:
            Ranked RetrievalResults (empty when nothing matches — never
            raises for absent index/entities).
        """
        _t0 = time.perf_counter()
        matches = self.store.find_entities(keywords or [], limit=12, query=query)
        if not matches:
            self._record_trace(trace, [], [], _t0)
            return []

        hits = self.store.related_chunks(
            entity_names=[m.name for m in matches],
            max_hops=self.max_hops,
        )
        hits = hits[:top_k]
        if not hits:
            self._record_trace(trace, matches, [], _t0)
            return []

        results = self._hydrate(hits, matches)
        self._record_trace(trace, matches, results, _t0)
        return results

    def _hydrate(self, hits: List, matches: List) -> List[RetrievalResult]:
        """Convert GraphHits to RetrievalResults, hydrating from vector store."""
        match_names = {m.name for m in matches}
        top_score = hits[0].score if hits else 0.0

        # Hydrate text/metadata from the vector store when available
        by_id: dict = {}
        if self.vector_store is not None and hits:
            try:
                records = self.vector_store.get_by_ids([h.chunk_id for h in hits])
                for rec in records or []:
                    cid = rec.get("id") if isinstance(rec, dict) else getattr(rec, "id", None)
                    if cid:
                        by_id[cid] = rec
            except Exception as exc:
                logger.warning("Graph hydration failed, returning bare hits: %s", exc)

        results = []
        for rank, hit in enumerate(hits, start=1):
            rec = by_id.get(hit.chunk_id) or {}
            text = rec.get("text") or rec.get("document") or ""
            metadata = dict(rec.get("metadata") or {})
            # Normalise graph score to 0-1 (best hit = 1.0); RRF ignores the
            # absolute value, but normalised scores keep traces readable.
            score = hit.score / top_score if top_score > 0 else 0.0
            metadata["graph_score"] = round(hit.score, 4)
            metadata["graph_matched_entities"] = [
                name for name in hit.matched_entities if name in match_names
            ][:5]
            metadata["graph_rank"] = rank
            results.append(
                RetrievalResult(
                    chunk_id=hit.chunk_id,
                    score=score,
                    text=text if isinstance(text, str) else str(text),
                    metadata=metadata,
                )
            )
        return results

    def _record_trace(self, trace: Optional[Any], matches: List, results: List, t0: float) -> None:
        """Record a graph_retrieval trace stage (mirrors dense/sparse)."""
        if trace is None:
            return
        trace.record_stage(
            "graph_retrieval",
            {
                "method": "graph",
                "collection": self.collection,
                "matched_entities": [m.name for m in matches][:8],
                "exact_matches": sum(1 for m in matches if m.match_type == "exact"),
                "top_k": getattr(self, "max_hops", 1),
                "result_count": len(results),
                "chunks": [
                    {
                        "chunk_id": r.chunk_id,
                        "score": round(r.score, 4),
                        "text": (r.text or "")[:200],
                        "source": r.metadata.get("source_path", ""),
                    }
                    for r in results
                ],
            },
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        )


def create_graph_retriever(
    settings: Any,
    collection: str,
    vector_store: Optional[BaseVectorStore] = None,
) -> Optional[GraphRetriever]:
    """Build a GraphRetriever for a collection, or None when disabled.

    Following the BM25 pattern: the caller knows the collection, the store
    file may or may not exist — an absent file degrades to an empty index
    (retriever returns []) rather than an error.

    Args:
        settings: Application settings (reads the optional ``graph`` section).
        collection: Collection name.
        vector_store: For hydration of hit text/metadata.

    Returns:
        GraphRetriever when settings.graph is enabled, else None.
    """
    graph_cfg = getattr(settings, "graph", None)
    if graph_cfg is None or not getattr(graph_cfg, "enabled", False):
        return None

    from src.core.settings import resolve_path
    from src.ingestion.storage.graph_store import GraphStore

    storage_dir = getattr(graph_cfg, "storage_dir", "./data/db/graph")
    db_path = resolve_path(storage_dir) / f"{collection}_graph.sqlite3"
    store = GraphStore(db_path)

    # An empty index would only add a no-op path; skip silently when the
    # graph has not been built yet (build via scripts/build_graph_index.py).
    entities, relations = store.stats()
    if entities == 0:
        logger.info(
            "Graph index for '%s' is empty (%s) — graph path disabled for this collection",
            collection, db_path,
        )
        return None

    return GraphRetriever(
        store=store,
        vector_store=vector_store,
        max_hops=int(getattr(graph_cfg, "max_hops", 1)),
        collection=collection,
    )
