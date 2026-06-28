"""Embedding utilities for the RAG pipeline.

Provides :class:`ReviewEmbedder`, which wraps an embedding backend (OpenAI
``text-embedding-ada-002`` by default, or Voyage ``voyage-02``) and turns
cleaned/enriched review records into embedding vectors plus normalized,
filter-ready metadata (rating, date, source, sentiment, themes).
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

logger = logging.getLogger(__name__)

Provider = Literal["openai", "voyage", "huggingface"]

# Known embedding dimensions, used when provisioning a Pinecone index.
_MODEL_DIMENSIONS: dict[str, int] = {
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "voyage-02": 1024,
    "voyage-3": 1024,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}

_DEFAULT_MODELS: dict[str, str] = {
    "openai": "text-embedding-ada-002",
    "voyage": "voyage-02",
    "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
}


@dataclass
class EmbeddedReview:
    """An embedded review ready to upsert into a vector store."""

    id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


def _to_epoch(date_str: str | None) -> float | None:
    """Parse an ISO date string into an epoch timestamp for range filters."""
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00")).timestamp()
    except (ValueError, AttributeError):
        return None


class ReviewEmbedder:
    """Embed review text and build filter-ready metadata."""

    def __init__(self, provider: Provider | None = None, model: str | None = None) -> None:
        self.provider: Provider = provider or os.getenv("EMBEDDING_PROVIDER", "openai")  # type: ignore[assignment]
        self.model = model or os.getenv(
            "EMBEDDING_MODEL", _DEFAULT_MODELS.get(self.provider, "text-embedding-ada-002")
        )
        self._embeddings = None

    @property
    def dimension(self) -> int:
        """Vector dimension for the configured model (best-effort)."""
        return _MODEL_DIMENSIONS.get(self.model, 1536)

    # --- backend construction ------------------------------------------- #
    def get_embeddings(self):
        """Return a LangChain Embeddings instance, constructing it lazily."""
        if self._embeddings is not None:
            return self._embeddings

        if self.provider == "openai":
            self._embeddings = self._build_openai()
        elif self.provider == "voyage":
            self._embeddings = self._build_voyage()
        elif self.provider == "huggingface":
            self._embeddings = self._build_huggingface()
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unknown embedding provider: {self.provider}")
        return self._embeddings

    def _build_openai(self):
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "langchain-openai is required for OpenAI embeddings. See requirements.txt"
            ) from exc
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY must be set for OpenAI embeddings.")
        logger.info("Using OpenAI embeddings: %s", self.model)
        return OpenAIEmbeddings(model=self.model)

    def _build_voyage(self):
        try:
            from langchain_voyageai import VoyageAIEmbeddings
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "langchain-voyageai is required for Voyage embeddings. See requirements.txt"
            ) from exc
        if not os.getenv("VOYAGE_API_KEY"):
            raise RuntimeError("VOYAGE_API_KEY must be set for Voyage embeddings.")
        logger.info("Using Voyage embeddings: %s", self.model)
        return VoyageAIEmbeddings(model=self.model)

    def _build_huggingface(self):
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "langchain-huggingface is required for local embeddings. See requirements.txt"
            ) from exc
        logger.info("Using HuggingFace embeddings: %s", self.model)
        return HuggingFaceEmbeddings(model_name=self.model)

    # --- metadata -------------------------------------------------------- #
    @staticmethod
    def extract_metadata(record: dict) -> dict:
        """Build flat, scalar metadata for filtering from a record.

        Includes rating, date (+ epoch for range filters), source, sentiment,
        themes, and discovery features. Lists are joined to comma strings so
        they're storable in either ChromaDB or Pinecone.
        """
        themes = record.get("themes") or record.get("discovery_categories") or []
        features = record.get("discovery_features") or []
        date_str = record.get("date")

        meta: dict[str, Any] = {
            "review_id": record.get("review_id", ""),
            "source": record.get("source", "unknown"),
        }
        if record.get("rating") is not None:
            meta["rating"] = int(record["rating"])
        if record.get("sentiment"):
            meta["sentiment"] = record["sentiment"]
        if record.get("discovery_satisfaction") is not None:
            meta["discovery_satisfaction"] = int(record["discovery_satisfaction"])
        if date_str:
            meta["date"] = date_str
            ts = _to_epoch(date_str)
            if ts is not None:
                meta["date_ts"] = ts
        if record.get("helpful_count") is not None:
            meta["helpful_count"] = int(record["helpful_count"])
        if themes:
            meta["themes"] = ",".join(map(str, themes))
        if features:
            meta["features"] = ",".join(map(str, features))
        if record.get("frustration_indicators"):
            meta["frustration_indicators"] = ",".join(map(str, record["frustration_indicators"]))
        if record.get("enthusiasm_indicators"):
            meta["enthusiasm_indicators"] = ",".join(map(str, record["enthusiasm_indicators"]))
        if record.get("language"):
            meta["language"] = record["language"]
        return meta

    # --- embedding ------------------------------------------------------- #
    def embed_review(self, review_text: str, metadata: dict | None = None) -> list[float]:
        """Embed a single review's text into a vector."""
        # metadata is accepted for API symmetry; embedding is text-only.
        _ = metadata
        return self.get_embeddings().embed_query(review_text)

    def batch_embed(self, reviews: list[str]) -> list[list[float]]:
        """Embed a batch of raw review strings into vectors."""
        if not reviews:
            return []
        return self.get_embeddings().embed_documents(reviews)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self.get_embeddings().embed_query(text)

    def build_embedded_reviews(
        self, records: list[dict], text_field: str = "clean_text"
    ) -> list[EmbeddedReview]:
        """Embed records and attach metadata, ready for upsert.

        Records missing the text field are skipped.
        """
        usable = [r for r in records if r.get(text_field)]
        if not usable:
            return []

        texts = [r[text_field] for r in usable]
        vectors = self.batch_embed(texts)

        embedded: list[EmbeddedReview] = []
        for record, text, vector in zip(usable, texts, vectors):
            meta = self.extract_metadata(record)
            review_id = (
                record.get("review_id")
                or meta.get("review_id")
                or hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]
            )
            embedded.append(
                EmbeddedReview(
                    id=str(review_id), text=text, embedding=vector, metadata=meta
                )
            )
        logger.info("Embedded %d reviews", len(embedded))
        return embedded


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    embedder = ReviewEmbedder(provider="huggingface")
    vecs = embedder.batch_embed(
        ["discover weekly is great", "recommendations are stale"]
    )
    print(f"Generated {len(vecs)} vectors of dim {len(vecs[0])}")
