"""Runtime wiring for the upgraded Agentic SQL Analyst."""

from __future__ import annotations

from dataclasses import dataclass

from app.agent.graph import SqlAnalystAgent
from app.config import Settings, get_settings
from app.db.connection import DatabaseClient
from app.embeddings.store import SchemaVectorStore
from app.safety.validator import QuerySafetyValidator
from app.services.cache import CacheService
from app.services.llm import GeminiLLMService
from app.services.schema_catalog import SchemaCatalogService


@dataclass(slots=True)
class AgentRuntime:
    """Shared application runtime dependencies."""

    settings: Settings
    cache_service: CacheService
    db_client: DatabaseClient
    schema_catalog: SchemaCatalogService
    vector_store: SchemaVectorStore
    llm_service: GeminiLLMService
    validator: QuerySafetyValidator
    agent: SqlAnalystAgent


def build_runtime(settings: Settings | None = None) -> AgentRuntime:
    """Create the production runtime container."""

    resolved_settings = settings or get_settings()
    cache_service = CacheService(resolved_settings)
    db_client = DatabaseClient(resolved_settings)
    schema_catalog = SchemaCatalogService(resolved_settings)
    vector_store = SchemaVectorStore(resolved_settings, cache_service=cache_service)
    llm_service = GeminiLLMService(resolved_settings)
    validator = QuerySafetyValidator(
        settings=resolved_settings,
        large_table_names=resolved_settings.LARGE_TABLES,
        schema_catalog=schema_catalog,
    )
    agent = SqlAnalystAgent(
        settings=resolved_settings,
        db_client=db_client,
        vector_store=vector_store,
        validator=validator,
        llm_service=llm_service,
        cache_service=cache_service,
        schema_catalog=schema_catalog,
    )
    return AgentRuntime(
        settings=resolved_settings,
        cache_service=cache_service,
        db_client=db_client,
        schema_catalog=schema_catalog,
        vector_store=vector_store,
        llm_service=llm_service,
        validator=validator,
        agent=agent,
    )
