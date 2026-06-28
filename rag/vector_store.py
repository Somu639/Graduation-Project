"""Vector store abstraction over ChromaDB (local) and Pinecone (production).

Provides :class:`VectorStore` with collection management, upsert of
pre-embedded reviews, similarity and hybrid (dense + lexical) search, rich
metadata filtering (rating, source, sentiment, theme tags, date range), and
helpers to build multiple purpose-built indexes (full corpus, discovery,
complaints, feature requests).
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Callable, Literal

from .embeddings import EmbeddedReview, ReviewEmbedder, _to_epoch

logger = logging.getLogger(__name__)

Backend = Literal["chroma", "pinecone", "memory"]

# Process-wide store for the dependency-free in-memory backend.
# Maps collection_name -> {id -> {"content": str, "metadata": dict}}.
_MEMORY_STORE: dict[str, dict[str, dict]] = {}
_TOKEN_RE = re.compile(r"\w+")

# Named index subsets and the predicate that decides membership.
INDEX_DEFINITIONS: dict[str, Callable[[dict], bool]] = {
    "full": lambda r: True,
    "discovery": lambda r: bool(
        r.get("discovery_features")
        or r.get("discovery_categories")
        or r.get("themes")
    ),
    "complaints": lambda r: (
        r.get("sentiment") == "negative"
        or bool(r.get("frustration_indicators"))
        or "algorithm_complaints" in (r.get("discovery_categories") or [])
        or "repetitive_content" in (r.get("discovery_categories") or [])
    ),
    "feature_requests": lambda r: (
        "feature_requests" in (r.get("discovery_categories") or [])
        or bool((r.get("theme") or {}).get("improvement_suggestion"))
    ),
}


class VectorStore:
    """Unified vector store interface for adding and searching reviews."""

    def __init__(
        self,
        backend: Backend | None = None,
        collection_name: str = "spotify-discovery",
        embedder: ReviewEmbedder | None = None,
        persist_directory: str | None = None,
    ) -> None:
        self.backend: Backend = backend or os.getenv("VECTOR_STORE", "chroma")  # type: ignore[assignment]
        self.collection_name = collection_name
        self.embedder = embedder or ReviewEmbedder()
        self.persist_directory = persist_directory or os.getenv(
            "CHROMA_PERSIST_DIR", "./.chroma"
        )
        self._collection = None  # chroma collection
        self._index = None  # pinecone index

    # ------------------------------------------------------------------ #
    # Collection / index management
    # ------------------------------------------------------------------ #
    def initialize_collection(self, name: str | None = None):
        """(Re)initialize the underlying collection/index and return it."""
        if name:
            self.collection_name = name
        self._collection = None
        self._index = None
        if self.backend == "chroma":
            return self._get_chroma()
        return self._get_pinecone()

    def _get_chroma(self):
        if self._collection is not None:
            return self._collection
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "chromadb is required for the chroma backend. See requirements.txt"
            ) from exc
        host = os.getenv("CHROMA_HOST")
        if host:
            # Connect to a standalone ChromaDB server (e.g. the compose service).
            client = chromadb.HttpClient(host=host, port=int(os.getenv("CHROMA_PORT", "8000")))
            logger.info("Chroma HTTP client -> %s:%s", host, os.getenv("CHROMA_PORT", "8000"))
        else:
            client = chromadb.PersistentClient(path=self.persist_directory)
        self._collection = client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )
        logger.info("Chroma collection ready: %s", self.collection_name)
        return self._collection

    def _get_pinecone(self):
        if self._index is not None:
            return self._index
        try:
            from pinecone import Pinecone, ServerlessSpec
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "pinecone is required for the pinecone backend. See requirements.txt"
            ) from exc
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise RuntimeError("PINECONE_API_KEY must be set for the pinecone backend.")

        pc = Pinecone(api_key=api_key)
        index_name = self.collection_name.replace("_", "-").lower()
        existing = {idx["name"] for idx in pc.list_indexes()}
        if index_name not in existing:
            pc.create_index(
                name=index_name,
                dimension=self.embedder.dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=os.getenv("PINECONE_CLOUD", "aws"),
                    region=os.getenv("PINECONE_REGION", "us-east-1"),
                ),
            )
        self._index = pc.Index(index_name)
        logger.info("Pinecone index ready: %s", index_name)
        return self._index

    # ------------------------------------------------------------------ #
    # Upsert
    # ------------------------------------------------------------------ #
    def _mem_collection(self) -> dict[str, dict]:
        return _MEMORY_STORE.setdefault(self.collection_name, {})

    def upsert_reviews(self, embedded_reviews: list[EmbeddedReview]) -> int:
        """Upsert pre-embedded reviews into the collection."""
        if not embedded_reviews:
            return 0

        if self.backend == "memory":
            col = self._mem_collection()
            for e in embedded_reviews:
                col[e.id] = {"content": e.text, "metadata": e.metadata or {}}
            logger.info("Upserted %d reviews into memory:%s", len(embedded_reviews), self.collection_name)
            return len(embedded_reviews)

        if self.backend == "chroma":
            collection = self._get_chroma()
            collection.upsert(
                ids=[e.id for e in embedded_reviews],
                embeddings=[e.embedding for e in embedded_reviews],
                documents=[e.text for e in embedded_reviews],
                metadatas=[e.metadata or {"_": ""} for e in embedded_reviews],
            )
        else:
            index = self._get_pinecone()
            vectors = [
                {"id": e.id, "values": e.embedding, "metadata": {**e.metadata, "text": e.text}}
                for e in embedded_reviews
            ]
            # Pinecone recommends batches of <=100 vectors.
            for start in range(0, len(vectors), 100):
                index.upsert(vectors=vectors[start : start + 100])

        logger.info("Upserted %d reviews into %s", len(embedded_reviews), self.collection_name)
        return len(embedded_reviews)

    def add_documents(self, records: list[dict], text_field: str = "clean_text") -> int:
        """Embed records and upsert them (convenience for the ingest pipeline)."""
        if self.backend == "memory":
            # No embeddings needed for lexical in-memory search.
            import hashlib

            embedded = []
            for r in records:
                text = r.get(text_field)
                if not text:
                    continue
                meta = ReviewEmbedder.extract_metadata(r)
                rid = str(
                    r.get("review_id")
                    or meta.get("review_id")
                    or hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]
                )
                embedded.append(EmbeddedReview(id=rid, text=text, embedding=[], metadata=meta))
            return self.upsert_reviews(embedded)

        embedded = self.embedder.build_embedded_reviews(records, text_field=text_field)
        return self.upsert_reviews(embedded)

    # ------------------------------------------------------------------ #
    # Filtering
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_filters(filters: dict | None) -> tuple[list[tuple], list[str]]:
        """Split a friendly filter dict into backend conditions + theme tags.

        Supported keys: rating (int | {min,max} | list), source (str | list),
        sentiment (str | list), date_from / date_to (ISO strings), and
        theme / themes (applied client-side).
        """
        if not filters:
            return [], []

        conditions: list[tuple] = []  # (field, op, value)
        themes: list[str] = []

        def add_scalar(field: str, value):
            if isinstance(value, (list, tuple, set)):
                conditions.append((field, "$in", list(value)))
            elif isinstance(value, dict):
                if "min" in value:
                    conditions.append((field, "$gte", value["min"]))
                if "max" in value:
                    conditions.append((field, "$lte", value["max"]))
            else:
                conditions.append((field, "$eq", value))

        for key, value in filters.items():
            if value is None:
                continue
            if key == "rating":
                add_scalar("rating", value)
            elif key == "source":
                add_scalar("source", value)
            elif key == "sentiment":
                add_scalar("sentiment", value)
            elif key in ("theme", "themes"):
                themes.extend(value if isinstance(value, (list, tuple)) else [value])
            elif key == "date_from":
                ts = _to_epoch(value)
                if ts is not None:
                    conditions.append(("date_ts", "$gte", ts))
            elif key == "date_to":
                ts = _to_epoch(value)
                if ts is not None:
                    conditions.append(("date_ts", "$lte", ts))
            else:
                add_scalar(key, value)

        return conditions, themes

    def _chroma_where(self, conditions: list[tuple]) -> dict | None:
        if not conditions:
            return None
        clauses = [{field: {op: value}} for field, op, value in conditions]
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def _pinecone_filter(self, conditions: list[tuple]) -> dict | None:
        if not conditions:
            return None
        flt: dict[str, dict] = {}
        for field, op, value in conditions:
            flt.setdefault(field, {})[op] = value
        return flt

    @staticmethod
    def _matches_themes(metadata: dict, themes: list[str]) -> bool:
        if not themes:
            return True
        stored = (metadata.get("themes", "") or "").lower()
        return any(t.lower() in stored for t in themes)

    @staticmethod
    def _mem_matches(metadata: dict, conditions: list[tuple]) -> bool:
        """Evaluate parsed filter conditions against a metadata dict (memory backend)."""
        for field, op, value in conditions:
            actual = metadata.get(field)
            if op == "$eq" and actual != value:
                return False
            if op == "$in" and actual not in value:
                return False
            if op == "$gte" and (actual is None or actual < value):
                return False
            if op == "$lte" and (actual is None or actual > value):
                return False
        return True

    @staticmethod
    def _lexical_score(query_tokens: set[str], content: str) -> float:
        doc_tokens = set(_TOKEN_RE.findall(content.lower()))
        if not query_tokens or not doc_tokens:
            return 0.0
        overlap = len(query_tokens & doc_tokens)
        return overlap / len(query_tokens)

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #
    def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
        *,
        k: int | None = None,
        filter: dict | None = None,
    ) -> list[dict]:
        """Semantic search with optional metadata filtering.

        Accepts both ``top_k``/``filters`` and the legacy ``k``/``filter`` names.

        Returns a list of ``{id, content, metadata, score}`` dicts.
        """
        top_k = k or top_k
        filters = filter or filters
        conditions, themes = self._parse_filters(filters)
        # Over-fetch when we must post-filter themes client-side.
        fetch_k = top_k * 4 if themes else top_k

        if self.backend == "memory":
            q_tokens = set(_TOKEN_RE.findall(query.lower()))
            scored = []
            for rid, rec in self._mem_collection().items():
                meta = rec["metadata"]
                if not self._mem_matches(meta, conditions):
                    continue
                if not self._matches_themes(meta, themes):
                    continue
                score = self._lexical_score(q_tokens, rec["content"])
                scored.append({"id": rid, "content": rec["content"], "metadata": meta, "score": score})
            scored.sort(key=lambda x: x["score"], reverse=True)
            return scored[:top_k]

        query_vec = self.embedder.embed_query(query)

        if self.backend == "chroma":
            results = self._search_chroma(query_vec, fetch_k, conditions)
        else:
            results = self._search_pinecone(query_vec, fetch_k, conditions)

        if themes:
            results = [r for r in results if self._matches_themes(r["metadata"], themes)]
        return results[:top_k]

    def _search_chroma(self, query_vec, fetch_k, conditions) -> list[dict]:
        collection = self._get_chroma()
        res = collection.query(
            query_embeddings=[query_vec],
            n_results=fetch_k,
            where=self._chroma_where(conditions),
            include=["documents", "metadatas", "distances"],
        )
        out: list[dict] = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for i in range(len(ids)):
            out.append(
                {
                    "id": ids[i],
                    "content": docs[i],
                    "metadata": metas[i] or {},
                    "score": 1.0 - dists[i],  # cosine distance -> similarity
                }
            )
        return out

    def _search_pinecone(self, query_vec, fetch_k, conditions) -> list[dict]:
        index = self._get_pinecone()
        res = index.query(
            vector=query_vec,
            top_k=fetch_k,
            filter=self._pinecone_filter(conditions),
            include_metadata=True,
        )
        out: list[dict] = []
        for match in res.get("matches", []):
            meta = dict(match.get("metadata", {}))
            content = meta.pop("text", "")
            out.append(
                {
                    "id": match.get("id"),
                    "content": content,
                    "metadata": meta,
                    "score": match.get("score", 0.0),
                }
            )
        return out

    def get_records(self, filters: dict | None = None, limit: int = 1000) -> list[dict]:
        """Fetch stored records (without ranking) for stats/aggregation.

        Returns a list of ``{id, content, metadata}`` dicts honoring the same
        metadata filters as :meth:`similarity_search`.
        """
        conditions, themes = self._parse_filters(filters)
        out: list[dict] = []

        if self.backend == "memory":
            for rid, rec in self._mem_collection().items():
                meta = rec["metadata"]
                if not self._mem_matches(meta, conditions):
                    continue
                if not self._matches_themes(meta, themes):
                    continue
                out.append({"id": rid, "content": rec["content"], "metadata": meta})
                if len(out) >= limit:
                    break
            return out

        if self.backend == "chroma":
            collection = self._get_chroma()
            res = collection.get(
                where=self._chroma_where(conditions),
                limit=limit,
                include=["documents", "metadatas"],
            )
            ids = res.get("ids", [])
            docs = res.get("documents", [])
            metas = res.get("metadatas", [])
            for i in range(len(ids)):
                meta = metas[i] or {}
                if not self._matches_themes(meta, themes):
                    continue
                out.append({"id": ids[i], "content": docs[i], "metadata": meta})
        else:
            # Pinecone has no native scan; approximate with a zero-vector query.
            index = self._get_pinecone()
            res = index.query(
                vector=[0.0] * self.embedder.dimension,
                top_k=limit,
                filter=self._pinecone_filter(conditions),
                include_metadata=True,
            )
            for match in res.get("matches", []):
                meta = dict(match.get("metadata", {}))
                content = meta.pop("text", "")
                if not self._matches_themes(meta, themes):
                    continue
                out.append({"id": match.get("id"), "content": content, "metadata": meta})

        return out

    def hybrid_search(
        self, query: str, metadata_filters: dict | None = None, top_k: int = 5
    ) -> list[dict]:
        """Dense + lexical hybrid search constrained by metadata filters.

        Fetches a wider candidate set via vector search (with metadata filters),
        then re-ranks by blending the normalized semantic score with lexical
        keyword overlap against the query.
        """
        candidates = self.similarity_search(
            query, top_k=top_k * 4, filters=metadata_filters
        )
        if not candidates:
            return []

        scores = [c["score"] for c in candidates]
        lo, hi = min(scores), max(scores)
        span = (hi - lo) or 1.0
        tokens = {t for t in query.lower().split() if len(t) > 2}

        for c in candidates:
            semantic = (c["score"] - lo) / span
            content_tokens = set(c["content"].lower().split())
            lexical = (
                len(tokens & content_tokens) / len(tokens) if tokens else 0.0
            )
            c["hybrid_score"] = 0.7 * semantic + 0.3 * lexical

        candidates.sort(key=lambda c: c["hybrid_score"], reverse=True)
        return candidates[:top_k]

    # ------------------------------------------------------------------ #
    # Multiple indexes
    # ------------------------------------------------------------------ #
    def get_index(self, index_name: str) -> "VectorStore":
        """Return a VectorStore bound to a named subset collection."""
        return VectorStore(
            backend=self.backend,
            collection_name=f"{self.collection_name}_{index_name}",
            embedder=self.embedder,
            persist_directory=self.persist_directory,
        )

    def build_indexes(
        self,
        records: list[dict],
        text_field: str = "clean_text",
        indexes: tuple[str, ...] = tuple(INDEX_DEFINITIONS),
    ) -> dict[str, int]:
        """Embed once, then upsert records into each requested subset index.

        Returns a mapping of index name -> number of reviews indexed.
        """
        # Build embedded records once and reuse across subsets. The memory
        # backend skips real embeddings (lexical search needs none).
        if self.backend == "memory":
            import hashlib

            embedded = []
            for r in records:
                text = r.get(text_field)
                if not text:
                    continue
                meta = ReviewEmbedder.extract_metadata(r)
                rid = str(
                    r.get("review_id")
                    or meta.get("review_id")
                    or hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]
                )
                embedded.append(EmbeddedReview(id=rid, text=text, embedding=[], metadata=meta))
        else:
            embedded = self.embedder.build_embedded_reviews(records, text_field=text_field)
        by_id = {e.id: e for e in embedded}

        counts: dict[str, int] = {}
        for name in indexes:
            predicate = INDEX_DEFINITIONS.get(name)
            if predicate is None:
                logger.warning("Unknown index '%s' skipped.", name)
                continue
            subset: list[EmbeddedReview] = []
            for record in records:
                if not predicate(record):
                    continue
                rid = str(record.get("review_id") or "")
                if rid in by_id:
                    subset.append(by_id[rid])
            # "full" lives in the base collection so default queries see everything.
            store = self if name == "full" else self.get_index(name)
            counts[name] = store.upsert_reviews(subset)
        logger.info("Built indexes: %s", counts)
        return counts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    store = VectorStore(backend="chroma", embedder=ReviewEmbedder(provider="huggingface"))
    store.add_documents(
        [
            {
                "review_id": "1",
                "clean_text": "Discover Weekly introduced me to my new favorite band",
                "source": "reddit",
                "sentiment": "positive",
                "rating": 5,
                "date": "2025-01-15T00:00:00",
                "discovery_categories": ["positive_discovery"],
            },
            {
                "review_id": "2",
                "clean_text": "The recommendations feel too repetitive lately",
                "source": "app_store",
                "sentiment": "negative",
                "rating": 2,
                "date": "2025-02-20T00:00:00",
                "discovery_categories": ["repetitive_content"],
            },
        ]
    )
    print(store.similarity_search("are recommendations repetitive?", top_k=2))
    print(store.hybrid_search("repetitive recommendations", {"source": "app_store"}, top_k=2))
