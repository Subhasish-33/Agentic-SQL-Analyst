"""Application configuration, logging, and model factories."""

from __future__ import annotations

import logging
import os
import sys
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pythonjsonlogger import jsonlogger


class Settings(BaseSettings):
    """Environment-backed application settings."""

    DATABASE_URL: str = "sqlite+aiosqlite:///./agentic_sql_analyst.db"
    GEMINI_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    REDIS_URL: str = ""
    VECTOR_DB_PATH: str = "./chroma_db"
    MAX_RETRIES: int = 3
    MAX_ROWS: int = 50
    TOP_K_TABLES: int = 3
    LLM_MODEL: str = "gemini-2.5-pro"
    EMBED_MODEL: str = "all-MiniLM-L6-v2"
    GEMINI_FLASH_MODEL: str = "gemini-2.5-flash"
    GEMINI_PRO_MODEL: str = "gemini-2.5-pro"
    BLOCKED_KEYWORDS: list[str] = Field(
        default_factory=lambda: [
            "DROP",
            "DELETE",
            "TRUNCATE",
            "UPDATE",
            "INSERT",
            "ALTER",
            "CREATE",
            "GRANT",
            "REVOKE",
        ],
    )
    LARGE_TABLES: list[str] = Field(
        default_factory=lambda: ["orders", "order_items"],
    )
    LOG_LEVEL: str = "INFO"
    LLM_TIMEOUT_SECONDS: int = 30
    QUERY_TIMEOUT_SECONDS: int = 3
    DEFAULT_PAGE_SIZE: int = 50
    MAX_PAGE_SIZE: int = 50
    LARGE_RESULT_ROW_THRESHOLD: int = 10
    CONFIDENCE_THRESHOLD: float = 0.6
    SCHEMA_CACHE_TTL_SECONDS: int = 1800
    EMBEDDING_CACHE_TTL_SECONDS: int = 3600
    QUERY_RESULT_CACHE_TTL_SECONDS: int = 300

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def sync_database_url(self) -> str:
        """Return a synchronous SQLAlchemy URL for schema indexing."""

        if self.DATABASE_URL.startswith("postgresql+asyncpg"):
            return self.DATABASE_URL.replace("postgresql+asyncpg", "postgresql+psycopg", 1)
        if self.DATABASE_URL.startswith("sqlite+aiosqlite"):
            return self.DATABASE_URL.replace("sqlite+aiosqlite", "sqlite", 1)
        return self.DATABASE_URL

    @property
    def sql_dialect(self) -> str:
        """Infer the SQL dialect from the configured database URL."""

        if self.DATABASE_URL.startswith("postgresql"):
            return "postgres"
        if self.DATABASE_URL.startswith("sqlite"):
            return "sqlite"
        return "ansi"

    @property
    def intent_model(self) -> str:
        """Select a low-cost model for intent classification."""

        if self.LLM_MODEL.startswith("claude"):
            return "claude-3-5-haiku-latest"
        if self.LLM_MODEL.startswith("gpt"):
            return "gpt-4o-mini"
        return self.GEMINI_FLASH_MODEL

    @property
    def sql_generation_model(self) -> str:
        """Return the default model for complex SQL generation."""

        if self.LLM_MODEL.startswith(("claude", "gpt")):
            return self.LLM_MODEL
        return self.GEMINI_PRO_MODEL


class JsonLogFormatter(jsonlogger.JsonFormatter):
    """JSON formatter with stable production-oriented log keys."""

    def add_fields(
        self,
        log_record: dict[str, Any],
        record: logging.LogRecord,
        message_dict: dict[str, Any],
    ) -> None:
        """Inject standard fields into every log line."""

        super().add_fields(log_record, record, message_dict)
        log_record.setdefault("timestamp", datetime.now(UTC).isoformat(timespec="milliseconds"))
        log_record["level"] = record.levelname
        log_record["logger"] = record.name
        log_record.setdefault("event_data", {})


def configure_logging(log_level: str = "INFO") -> None:
    """Configure root logging once for JSON structured output."""

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(log_level.upper())

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonLogFormatter("%(timestamp)s %(level)s %(name)s %(message)s"))
    root_logger.addHandler(handler)
    logging.captureWarnings(True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings."""

    return Settings()


def build_chat_model(settings: Settings, model_name: str | None = None) -> Any:
    """Construct a chat model based on the selected provider."""

    resolved_model = model_name or settings.LLM_MODEL
    if resolved_model.startswith("gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY must be set when using a Gemini model.")
        return ChatGoogleGenerativeAI(
            model=resolved_model,
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0,
            timeout=settings.LLM_TIMEOUT_SECONDS,
            max_retries=0,
        )

    if resolved_model.startswith("claude"):
        from langchain_anthropic import ChatAnthropic

        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY must be set when using an Anthropic model.")
        return ChatAnthropic(
            model=resolved_model,
            temperature=0,
            timeout=settings.LLM_TIMEOUT_SECONDS,
            max_retries=0,
            anthropic_api_key=anthropic_api_key,
        )

    from langchain_openai import ChatOpenAI

    if not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY must be set when using an OpenAI model.")
    return ChatOpenAI(
        model=resolved_model,
        temperature=0,
        timeout=settings.LLM_TIMEOUT_SECONDS,
        max_retries=0,
        api_key=settings.OPENAI_API_KEY,
    )


def build_embeddings(settings: Settings) -> Any:
    """Construct an embeddings client for the configured provider."""

    embed_model = settings.EMBED_MODEL
    if embed_model.startswith("models/") or embed_model.startswith("gemini"):
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY must be set when using Gemini embeddings.")
        return GoogleGenerativeAIEmbeddings(model=embed_model, google_api_key=settings.GEMINI_API_KEY)
    if embed_model.startswith("text-embedding") or embed_model.startswith("openai"):
        from langchain_openai import OpenAIEmbeddings

        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY must be set when using OpenAI embeddings.")
        return OpenAIEmbeddings(model=embed_model, api_key=settings.OPENAI_API_KEY)

    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=embed_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def coerce_llm_text(response: Any) -> str:
    """Normalize LangChain chat model responses into plain text."""

    if isinstance(response, str):
        return response.strip()

    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            elif hasattr(item, "text"):
                parts.append(str(item.text))
            else:
                parts.append(str(item))
        return "".join(parts).strip()
    return str(content).strip()
