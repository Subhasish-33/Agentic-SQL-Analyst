"""Database connection and query execution helpers."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from time import perf_counter

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlglot import exp, parse_one

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class QueryExecutionResult:
    """Structured output for a completed SQL execution."""

    rows: list[dict]
    row_count: int
    execution_time_ms: int
    executed_sql: str
    result_truncated: bool


def create_async_engine_from_settings(settings: Settings) -> AsyncEngine:
    """Create an async SQLAlchemy engine with production-minded pooling."""

    common_kwargs: dict[str, object] = {
        "future": True,
        "pool_pre_ping": True,
    }
    if settings.DATABASE_URL.startswith("postgresql+asyncpg"):
        common_kwargs.update(
            {
                "pool_size": 5,
                "max_overflow": 10,
                "connect_args": {
                    "server_settings": {
                        "statement_timeout": str(settings.QUERY_TIMEOUT_SECONDS * 1000),
                    }
                },
            }
        )
    return create_async_engine(settings.DATABASE_URL, **common_kwargs)


def create_sync_engine_from_settings(settings: Settings) -> Engine:
    """Create a synchronous engine used for schema introspection and setup."""

    common_kwargs: dict[str, object] = {
        "future": True,
        "pool_pre_ping": True,
    }
    if settings.sync_database_url.startswith("postgresql+psycopg"):
        common_kwargs.update({"pool_size": 5, "max_overflow": 10})
    return create_engine(settings.sync_database_url, **common_kwargs)


class DatabaseClient:
    """Async database client for validated read-only SQL execution."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the database client and engine."""

        self.settings = settings or get_settings()
        self.engine = create_async_engine_from_settings(self.settings)

    @property
    def dialect(self) -> str:
        """Return the SQL dialect used for query generation and normalization."""

        return self.settings.sql_dialect

    def enforce_row_limit(self, sql: str) -> str:
        """Ensure the executed query is bounded by the configured row limit."""

        if re.search(r"\blimit\b", sql, flags=re.IGNORECASE):
            try:
                parsed = parse_one(sql, read=self.dialect)
                limit_expression = parsed.args.get("limit")
                if limit_expression is not None and isinstance(limit_expression.expression, exp.Literal):
                    current_limit = int(limit_expression.expression.this)
                    if current_limit > self.settings.MAX_ROWS:
                        parsed = parsed.limit(self.settings.MAX_ROWS)
                return parsed.sql(dialect=self.dialect)
            except Exception:
                return sql.rstrip().rstrip(";")

        try:
            parsed = parse_one(sql, read=self.dialect)
            if isinstance(parsed, exp.Select) and parsed.args.get("limit") is None:
                parsed = parsed.limit(self.settings.MAX_ROWS)
            return parsed.sql(dialect=self.dialect)
        except Exception:
            return f"{sql.rstrip().rstrip(';')} LIMIT {self.settings.MAX_ROWS}"

    def apply_pagination(self, sql: str, page: int, page_size: int) -> str:
        """Apply page and page-size constraints to a SELECT query."""

        bounded_sql = self.enforce_row_limit(sql)
        try:
            parsed = parse_one(bounded_sql, read=self.dialect)
            if isinstance(parsed, exp.Select):
                limit_expression = parsed.args.get("limit")
                if limit_expression is None:
                    parsed = parsed.limit(page_size)
                elif isinstance(limit_expression.expression, exp.Literal):
                    current_limit = int(limit_expression.expression.this)
                    parsed = parsed.limit(min(current_limit, page_size))
                if page > 1:
                    parsed = parsed.offset((page - 1) * page_size)
                return parsed.sql(dialect=self.dialect)
        except Exception:
            offset_clause = f" OFFSET {(page - 1) * page_size}" if page > 1 else ""
            if re.search(r"\blimit\b", bounded_sql, flags=re.IGNORECASE):
                return f"{bounded_sql.rstrip().rstrip(';')}{offset_clause}"
            return f"{bounded_sql.rstrip().rstrip(';')} LIMIT {page_size}{offset_clause}"
        return bounded_sql

    async def execute_query(self, sql: str, page: int = 1, page_size: int | None = None) -> QueryExecutionResult:
        """Execute SQL and return rows, count, execution time, and bounded SQL."""

        safe_page = max(page, 1)
        safe_page_size = min(max(page_size or self.settings.DEFAULT_PAGE_SIZE, 1), self.settings.MAX_PAGE_SIZE)
        bounded_sql = self.apply_pagination(sql, page=safe_page, page_size=safe_page_size)
        started_at = perf_counter()

        try:
            async with self.engine.connect() as connection:
                result = await asyncio.wait_for(
                    connection.execute(text(bounded_sql)),
                    timeout=self.settings.QUERY_TIMEOUT_SECONDS,
                )
                rows = [dict(row) for row in result.mappings().all()]
        except asyncio.TimeoutError as exc:
            logger.warning(
                "sql_execution_timed_out",
                extra={"event_data": {"sql": bounded_sql, "timeout_seconds": self.settings.QUERY_TIMEOUT_SECONDS}},
            )
            raise OperationalError("Statement timed out", params=None, orig=exc) from exc

        execution_time_ms = int((perf_counter() - started_at) * 1000)
        result_truncated = len(rows) >= safe_page_size
        logger.info(
            "sql_execution_completed",
            extra={
                "event_data": {
                    "sql": bounded_sql,
                    "row_count": len(rows),
                    "execution_time_ms": execution_time_ms,
                    "page": safe_page,
                    "page_size": safe_page_size,
                },
            },
        )

        return QueryExecutionResult(
            rows=rows,
            row_count=len(rows),
            execution_time_ms=execution_time_ms,
            executed_sql=bounded_sql,
            result_truncated=result_truncated,
        )

    async def healthcheck(self) -> bool:
        """Run a lightweight query to confirm database connectivity."""

        try:
            async with self.engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
            return True
        except DBAPIError:
            return False

    async def dispose(self) -> None:
        """Dispose of the underlying engine."""

        await self.engine.dispose()
