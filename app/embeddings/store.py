"""Vector store utilities for table schema retrieval."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from langchain_chroma import Chroma

from app.config import Settings, build_embeddings, get_settings
from app.services.cache import CacheService

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SchemaDocument:
    """A schema document ready to be embedded and persisted."""

    table_name: str
    content: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class SchemaSearchResult:
    """A search result returned from the schema vector store."""

    table_name: str
    schema_text: str
    metadata: dict[str, Any]
    columns: list[dict[str, Any]]


class SchemaVectorStore:
    """Lazy Chroma-backed vector store for schema embeddings."""

    def __init__(
        self,
        settings: Settings | None = None,
        collection_name: str = "schema_index",
        cache_service: CacheService | None = None,
    ) -> None:
        """Initialize the vector store wrapper."""

        self.settings = settings or get_settings()
        self.collection_name = collection_name
        self.cache_service = cache_service
        self._store: Chroma | None = None
        self._embedding_function: Any | None = None

    def _get_embedding_function(self) -> Any:
        """Return an embedding client with lightweight caching when possible."""

        if self._embedding_function is None:
            base_embeddings = build_embeddings(self.settings)
            self._embedding_function = CachedEmbeddings(
                base_embeddings,
                cache_service=self.cache_service,
                ttl_seconds=self.settings.EMBEDDING_CACHE_TTL_SECONDS,
            )
        return self._embedding_function

    def _get_store(self) -> Chroma:
        """Return the lazily initialized Chroma collection."""

        if self._store is None:
            Path(self.settings.VECTOR_DB_PATH).mkdir(parents=True, exist_ok=True)
            self._store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self._get_embedding_function(),
                persist_directory=self.settings.VECTOR_DB_PATH,
            )
        return self._store

    def clear(self) -> None:
        """Delete and recreate the backing collection."""

        store = self._get_store()
        store.delete_collection()
        self._store = None
        logger.info(
            "schema_collection_reset",
            extra={"event_data": {"collection_name": self.collection_name}},
        )

    def upsert_documents(self, documents: Sequence[SchemaDocument]) -> None:
        """Insert or replace schema documents in the vector store."""

        if not documents:
            return

        store = self._get_store()
        ids = [document.table_name for document in documents]
        store.delete(ids=ids)
        store.add_texts(
            texts=[document.content for document in documents],
            metadatas=[document.metadata for document in documents],
            ids=ids,
        )
        logger.info(
            "schema_documents_upserted",
            extra={"event_data": {"document_count": len(documents)}},
        )

    def similarity_search(self, query: str, k: int) -> list[SchemaSearchResult]:
        """Find the most relevant schema documents for a user query."""

        documents = self._get_store().similarity_search(query, k=k)
        results: list[SchemaSearchResult] = []
        for document in documents:
            metadata = dict(document.metadata or {})
            columns_json = metadata.get("columns_json", "[]")
            try:
                columns = json.loads(columns_json) if isinstance(columns_json, str) else list(columns_json)
            except (TypeError, json.JSONDecodeError):
                columns = []

            results.append(
                SchemaSearchResult(
                    table_name=str(metadata.get("table_name", "unknown")),
                    schema_text=document.page_content,
                    metadata=metadata,
                    columns=columns,
                ),
            )

        logger.info(
            "schema_similarity_search_completed",
            extra={
                "event_data": {
                    "query": query,
                    "top_k": k,
                    "matched_tables": [result.table_name for result in results],
                },
            },
        )
        return results


class CachedEmbeddings:
    """Minimal sync cache wrapper around the embedding client."""

    def __init__(self, base_embeddings: Any, cache_service: CacheService | None, ttl_seconds: int) -> None:
        """Initialize the embedding cache wrapper."""

        self.base_embeddings = base_embeddings
        self.cache_service = cache_service
        self.ttl_seconds = ttl_seconds

    def _cached_embedding(self, text: str) -> list[float] | None:
        """Return a locally cached embedding when available."""

        if self.cache_service is None:
            return None
        cached = self.cache_service.get_local_json("embedding", text)
        return list(cached) if cached is not None else None

    def _store_embedding(self, text: str, embedding: list[float]) -> None:
        """Persist an embedding in the local cache."""

        if self.cache_service is None:
            return
        self.cache_service.set_local_json("embedding", text, embedding, ttl_seconds=self.ttl_seconds)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed many texts with local caching."""

        embeddings: list[list[float] | None] = [self._cached_embedding(text) for text in texts]
        missing_indices = [index for index, embedding in enumerate(embeddings) if embedding is None]
        if missing_indices:
            missing_texts = [texts[index] for index in missing_indices]
            fresh_embeddings = self.base_embeddings.embed_documents(missing_texts)
            for index, embedding in zip(missing_indices, fresh_embeddings):
                embeddings[index] = embedding
                self._store_embedding(texts[index], embedding)
        return [list(embedding or []) for embedding in embeddings]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query with local caching."""

        cached = self._cached_embedding(text)
        if cached is not None:
            return cached
        embedding = self.base_embeddings.embed_query(text)
        self._store_embedding(text, embedding)
        return embedding
