"""Lightweight graph store for GraphRAG-style entity-relation indexing.

Persists entities and relations extracted from chunks into a per-collection
SQLite file (``data/db/graph/{collection}_graph.sqlite3``), mirroring the
BM25 index storage pattern. Designed to be:

- Dependency-free: plain sqlite3, no graph database required.
- Thread-safe: a fresh connection per operation (the retriever may run
  inside HybridSearch's ThreadPoolExecutor).
- Idempotent: ``rebuild()`` clears everything; ``upsert_chunk()`` replaces
  one chunk's relations without touching other chunks'.

Query-side scoring: chunks are scored by the sum of linked relation
confidences, weighted by how the entity matched (exact > substring) and by
hop distance (1 hop full weight, 2 hops ×0.5). The absolute scale does not
matter for RRF fusion (rank-based), but normalised scores keep traces and
debug output readable.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Score weights: exact entity match beats substring match; a 2-hop
# expansion contributes half of a direct link. Hub damping: entities
# linked from many chunks (df > HUB_DF_LIMIT) do not get the high-precision
# "entity appears verbatim in query" boost — for hub entities that match is
# noise, not signal.
WEIGHT_EXACT = 1.0
WEIGHT_SUBSTRING = 0.6
WEIGHT_HOP2 = 0.5
HUB_DF_LIMIT = 3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    name_norm TEXT UNIQUE NOT NULL,
    entity_type TEXT DEFAULT 'other',
    mentions INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS relations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_entity TEXT NOT NULL,
    target_entity TEXT NOT NULL,
    relation TEXT DEFAULT '',
    chunk_id TEXT NOT NULL,
    confidence REAL DEFAULT 1.0
);
CREATE INDEX IF NOT EXISTS idx_rel_source ON relations(source_entity);
CREATE INDEX IF NOT EXISTS idx_rel_target ON relations(target_entity);
CREATE INDEX IF NOT EXISTS idx_rel_chunk ON relations(chunk_id);
"""


@dataclass
class EntityMatch:
    """One entity matched by a query, with its match weight."""

    name: str
    weight: float
    match_type: str  # "exact" | "substring"
    entity_type: str = "other"


@dataclass
class GraphHit:
    """One chunk scored via the graph path."""

    chunk_id: str
    score: float
    matched_entities: List[str] = field(default_factory=list)


def normalize_name(name: str) -> str:
    """Normalise an entity name for matching (lowercase latin, trimmed)."""
    return " ".join(name.split()).lower()


# Backwards-compatible alias
_normalize = normalize_name


class GraphStore:
    """SQLite-backed entity-relation store for one collection."""

    def __init__(self, db_path: str | Path) -> None:
        """Open (creating if needed) the graph database file.

        Args:
            db_path: Path to ``{collection}_graph.sqlite3``.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    # ---- connection ---------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Open a short-lived connection (thread safety by isolation)."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    # ---- write side ----------------------------------------------------

    def clear(self) -> None:
        """Remove all entities and relations (full rebuild entry point)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM relations")
            conn.execute("DELETE FROM entities")

    def upsert_chunk(
        self,
        chunk_id: str,
        entities: Iterable[Dict],
        relations: Iterable[Dict],
    ) -> None:
        """Replace one chunk's graph contributions atomically.

        Args:
            chunk_id: Owning chunk identifier.
            entities: Dicts with ``name`` and optional ``type``.
            relations: Dicts with ``source``, ``target``, ``relation``,
                optional ``confidence`` (0-1, clamped).
        """
        entity_rows = []
        for e in entities or []:
            name = (e.get("name") or "").strip()
            if not name:
                continue
            entity_rows.append((_normalize(name), name, e.get("type") or "other"))

        relation_rows = []
        for r in relations or []:
            source = normalize_name(r.get("source") or "")
            target = normalize_name(r.get("target") or "")
            if not source or not target:
                continue
            try:
                confidence = float(r.get("confidence", 1.0))
            except (TypeError, ValueError):
                confidence = 1.0
            confidence = max(0.0, min(confidence, 1.0))
            relation_rows.append((source, target, r.get("relation") or "", chunk_id, confidence))

        # LLMs sometimes reference entities in relations without listing
        # them in "entities" — auto-register relation endpoints so the
        # entity dictionary stays query-complete.
        listed = {row[0] for row in entity_rows}
        known_types = {row[0]: row[2] for row in entity_rows}
        for source, target, _, _, _ in relation_rows:
            for endpoint in (source, target):
                if endpoint not in listed:
                    entity_rows.append((endpoint, endpoint, known_types.get(endpoint, "other")))
                    listed.add(endpoint)

        with self._connect() as conn:
            conn.execute("DELETE FROM relations WHERE chunk_id = ?", (chunk_id,))
            for norm, display, etype in entity_rows:
                conn.execute(
                    "INSERT INTO entities(name, name_norm, entity_type, mentions) "
                    "VALUES(?, ?, ?, 0) ON CONFLICT(name_norm) DO NOTHING",
                    (display, norm, etype),
                )
            if relation_rows:
                conn.executemany(
                    "INSERT INTO relations(source_entity, target_entity, relation, chunk_id, confidence) "
                    "VALUES(?, ?, ?, ?, ?)",
                    relation_rows,
                )
            # Recount mentions from relations touching each entity
            conn.execute(
                "UPDATE entities SET mentions = ("
                "  SELECT COUNT(*) FROM relations r"
                "  WHERE r.source_entity = entities.name OR r.target_entity = entities.name)"
            )

    def stats(self) -> Tuple[int, int]:
        """Return (entity_count, relation_count)."""
        with self._connect() as conn:
            entities = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
            relations = conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
        return entities, relations

    # ---- read side ------------------------------------------------------

    def find_entities(
        self,
        keywords: List[str],
        limit: int = 12,
        query: str = "",
    ) -> List[EntityMatch]:
        """Match a query against indexed entities.

        Matching tiers (highest precision first):
        1. ``in_query`` (weight 1.0): the full normalised entity name
           appears verbatim in the normalised query. Robust against
           tokenisers splitting multi-word terms (e.g. "跨模态检索" being
           segmented into 跨/模态/检索) — the whole-name check bypasses the
           segmentation entirely.
        2. ``exact`` (weight 1.0): a keyword equals the entity name.
        3. ``substring`` (weight 0.6, loosest): a keyword is contained in
           the entity or vice versa (both ≥2 chars, entity ≥3 chars).
           Generic short keywords like 检索 still match here, which is why
           tier 3 results are additionally damped by entity document
           frequency in :meth:`related_chunks`.

        Args:
            keywords: Query keywords (already processed by QueryProcessor).
            limit: Maximum matches returned.
            query: Original raw query string (enables tier-1 matching).

        Returns:
            EntityMatch list, best tier first.
        """
        norm_keywords = [_normalize(k) for k in (keywords or []) if k and k.strip()]
        query_norm = _normalize(query or "")
        if not norm_keywords and not query_norm:
            return []

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT e.name, e.name_norm, e.entity_type, "
                "  (SELECT COUNT(DISTINCT r.chunk_id) FROM relations r "
                "   WHERE r.source_entity = e.name_norm OR r.target_entity = e.name_norm) AS df "
                "FROM entities e"
            ).fetchall()

        def tier_of(norm: str, df: int) -> Optional[Tuple[float, str]]:
            if len(norm) >= 2 and query_norm and norm in query_norm and df <= HUB_DF_LIMIT:
                # Full-name-in-query is high precision ONLY for specific
                # (low-df) entities; hubs like "RAG" appear in every query
                # mentioning the domain and must not match at full weight.
                return (1.0, "in_query")
            for kw in norm_keywords:
                if kw == norm:
                    return (1.0, "exact")
            for kw in norm_keywords:
                if len(kw) >= 2 and len(norm) >= 3 and (kw in norm or norm in kw):
                    return (WEIGHT_SUBSTRING, "substring")
            return None

        exact: List[EntityMatch] = []
        partial: List[EntityMatch] = []
        for display, norm, etype, df in rows:
            tier = tier_of(norm, df)
            if tier is None:
                continue
            match = EntityMatch(
                name=display,
                weight=tier[0],
                match_type=tier[1],
                entity_type=etype or "other",
            )
            (exact if tier[1] in ("exact", "in_query") else partial).append(match)

        return (exact + partial)[:limit]

    def related_chunks(
        self,
        entity_names: List[str],
        max_hops: int = 1,
    ) -> List[GraphHit]:
        """Score chunks reachable from the given entities, with hub damping.

        Hub damping: an entity linked from many chunks (large document
        frequency, e.g. "RAG" extracted from nearly every chunk) carries
        little localisation signal, so its contribution is scaled by
        ``min(1, 1/sqrt(df))`` — df=1 keeps full weight, df=9 cuts to a
        third. Without this, chapter-overview chunks reachable via generic
        hub entities would outrank the specific chunk that actually
        defines the queried concept.

        Args:
            entity_names: Entity names (display or normalised) to expand from.
            max_hops: 1 = relations directly touching the entities;
                2 = also relations of adjacent entities (weight ×0.5,
                damped by the adjacent entity's own df).

        Returns:
            GraphHit list sorted by score descending. Chunks reachable via
            multiple paths accumulate scores.
        """
        seeds = {normalize_name(n) for n in entity_names}
        if not seeds:
            return []

        with self._connect() as conn:
            display_map = {}
            q = ",".join("?" * len(seeds))
            for display, norm in conn.execute(
                f"SELECT name, name_norm FROM entities WHERE name_norm IN ({q})",
                tuple(seeds),
            ):
                display_map[norm] = display

            def fetch(entity: str) -> List[Tuple[str, str, str, float]]:
                return conn.execute(
                    "SELECT source_entity, target_entity, chunk_id, confidence "
                    "FROM relations WHERE source_entity = ? OR target_entity = ?",
                    (entity, entity),
                ).fetchall()

            scores: Dict[str, float] = {}
            matched: Dict[str, set] = {}

            def accumulate(rows: List[Tuple[str, str, str, float]],
                           seed: str, damping: float, weight: float = 1.0) -> None:
                display = display_map.get(seed, seed)
                for source, target, chunk_id, confidence in rows:
                    scores[chunk_id] = scores.get(chunk_id, 0.0) + float(confidence) * damping * weight
                    matched.setdefault(chunk_id, set()).add(display)

            for seed in seeds:
                direct = fetch(seed)
                if not direct:
                    continue
                df = len({row[2] for row in direct})
                damping = min(1.0, 1.0 / (df ** 0.5))
                accumulate(direct, seed, damping)

                if max_hops >= 2:
                    adjacent = set()
                    for source, target, _, _ in direct:
                        if source not in seeds:
                            adjacent.add(source)
                        if target not in seeds:
                            adjacent.add(target)
                    for adj in adjacent:
                        adj_rows = fetch(adj)
                        if not adj_rows:
                            continue
                        adj_df = len({row[2] for row in adj_rows})
                        adj_damping = min(1.0, 1.0 / (adj_df ** 0.5))
                        # only count chunks not already reached directly
                        accumulate(
                            [r for r in adj_rows if r[2] not in scores],
                            seed, adj_damping * WEIGHT_HOP2,
                        )

        hits = [
            GraphHit(chunk_id=cid, score=score, matched_entities=sorted(matched.get(cid, set())))
            for cid, score in scores.items()
        ]
        hits.sort(key=lambda h: (-h.score, h.chunk_id))
        return hits

    def sample(self, limit: int = 10) -> List[Dict]:
        """Sample relations for debugging / demo output."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT source_entity, target_entity, relation, chunk_id, confidence "
                "FROM relations ORDER BY confidence DESC, id LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "source": r[0],
                "target": r[1],
                "relation": r[2],
                "chunk_id": r[3],
                "confidence": round(r[4], 2),
            }
            for r in rows
        ]
