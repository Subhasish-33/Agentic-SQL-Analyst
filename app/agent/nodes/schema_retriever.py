"""Schema retrieval node backed by semantic search."""

from __future__ import annotations

import json
import logging
from time import perf_counter

from app.agent.state import AgentState
from app.config import Settings, get_settings
from app.embeddings.store import SchemaSearchResult, SchemaVectorStore
from app.services.cache import CacheService
from app.utils.metrics import add_cache_hit, add_stage_latency

logger = logging.getLogger(__name__)


class SchemaRetrieverNode:
    """Retrieve only the most relevant tables for the current question."""

    def __init__(
        self,
        settings: Settings | None = None,
        vector_store: SchemaVectorStore | None = None,
        cache_service: CacheService | None = None,
    ) -> None:
        """Initialize the schema retriever."""

        self.settings = settings or get_settings()
        self.vector_store = vector_store or SchemaVectorStore(self.settings)
        self.cache_service = cache_service

    def _format_result(self, result: SchemaSearchResult) -> str:
        """Format a schema match into prompt-ready text."""

        column_descriptions: list[str] = []
        for column in result.columns:
            column_name = str(column.get("name", "unknown"))
            column_type = str(column.get("type", "UNKNOWN")).upper()
            tokens = [column_name, column_type]
            if column.get("is_primary_key"):
                tokens.append("PRIMARY KEY")
            foreign_key_target = column.get("foreign_key_to")
            if foreign_key_target:
                tokens.append(f"FOREIGN KEY -> {foreign_key_target}")
            column_descriptions.append(" ".join(tokens))

        relationships_json = result.metadata.get("relationships_json", "[]")
        try:
            relationships = json.loads(relationships_json) if isinstance(relationships_json, str) else list(relationships_json)
        except (TypeError, json.JSONDecodeError):
            relationships = []

        relationship_text = ", ".join(f"{item['from']} -> {item['to']}" for item in relationships) if relationships else "none"
        return (
            f"Table: {result.table_name}\n"
            f"Columns: {', '.join(column_descriptions)}\n"
            f"Relationships: {relationship_text}"
        )

    async def retrieve(self, state: AgentState) -> AgentState:
        """Retrieve and format the top-k relevant tables."""

        user_query = state.get("sanitized_user_query") or state.get("user_query", "")
        logger.info(
            "schema_retrieval_started",
            extra={"event_data": {"user_query": user_query, "top_k": self.settings.TOP_K_TABLES}},
        )

        started_at = perf_counter()
        cache_key = f"{user_query}|{self.settings.TOP_K_TABLES}"
        if self.cache_service is not None:
            cached_payload = await self.cache_service.get_json("schema", cache_key)
            if cached_payload is not None:
                stage_latency = int((perf_counter() - started_at) * 1000)
                return {
                    "relevant_tables": list(cached_payload["relevant_tables"]),
                    "schema_context": str(cached_payload["schema_context"]),
                    "validation_error": None,
                    "cache_hits": add_cache_hit(state.get("cache_hits"), "schema", True),
                    "latency_breakdown": add_stage_latency(state.get("latency_breakdown"), "schema", stage_latency),
                }

        results = self.vector_store.similarity_search(user_query, k=self.settings.TOP_K_TABLES)
        if not results:
            logger.warning(
                "schema_retrieval_empty",
                extra={"event_data": {"user_query": user_query}},
            )
            return {
                "relevant_tables": [],
                "schema_context": "",
                "validation_error": "No indexed schema context was found. Run the schema indexing script first.",
                "final_answer": "I could not find indexed schema context for this database. Run schema indexing and try again.",
            }

        relevant_tables = [result.table_name for result in results]
        schema_context = "\n\n".join(self._format_result(result) for result in results)
        if self.cache_service is not None:
            await self.cache_service.set_json(
                "schema",
                cache_key,
                {"relevant_tables": relevant_tables, "schema_context": schema_context},
                ttl_seconds=self.settings.SCHEMA_CACHE_TTL_SECONDS,
            )
        stage_latency = int((perf_counter() - started_at) * 1000)
        logger.info(
            "schema_retrieval_completed",
            extra={"event_data": {"matched_tables": relevant_tables}},
        )
        return {
            "relevant_tables": relevant_tables,
            "schema_context": schema_context,
            "validation_error": None,
            "cache_hits": add_cache_hit(state.get("cache_hits"), "schema", False),
            "latency_breakdown": add_stage_latency(state.get("latency_breakdown"), "schema", stage_latency),
        }

    async def __call__(self, state: AgentState) -> AgentState:
        """Reuse prefetched schema when available, otherwise retrieve on demand."""

        if state.get("schema_context") and state.get("relevant_tables") and state.get("latency_breakdown", {}).get("schema"):
            return {
                "relevant_tables": state.get("relevant_tables", []),
                "schema_context": state.get("schema_context", ""),
                "cache_hits": state.get("cache_hits", {}),
            }
        return await self.retrieve(state)
