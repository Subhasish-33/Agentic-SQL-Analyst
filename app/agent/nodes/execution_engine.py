"""Execution node for validated SQL."""

from __future__ import annotations

import logging

from sqlalchemy.exc import DataError, OperationalError, ProgrammingError, SQLAlchemyError

from app.agent.state import AgentState
from app.db.connection import DatabaseClient
from app.utils.metrics import add_stage_latency

logger = logging.getLogger(__name__)


class ExecutionEngineNode:
    """Execute validated SQL against the configured database."""

    def __init__(self, db_client: DatabaseClient) -> None:
        """Initialize the execution engine node."""

        self.db_client = db_client

    async def __call__(self, state: AgentState) -> AgentState:
        """Run SQL and capture either rows or the exact execution error."""

        sql = state.get("generated_sql", "")
        logger.info(
            "sql_execution_started",
            extra={"event_data": {"sql": sql, "retry_count": state.get("retry_count", 0)}},
        )

        try:
            execution_result = await self.db_client.execute_query(
                sql,
                page=state.get("page", 1),
                page_size=state.get("page_size", self.db_client.settings.DEFAULT_PAGE_SIZE),
            )
        except (OperationalError, ProgrammingError, DataError) as exc:
            logger.warning(
                "sql_execution_failed",
                extra={"event_data": {"sql": sql, "error": str(exc)}},
            )
            return {
                "execution_result": None,
                "execution_error": str(exc),
                "row_count": 0,
                "execution_time_ms": 0,
                "latency_breakdown": add_stage_latency(state.get("latency_breakdown"), "execution", self.db_client.settings.QUERY_TIMEOUT_SECONDS * 1000),
            }
        except SQLAlchemyError as exc:
            logger.warning(
                "sql_execution_failed_sqlalchemy",
                extra={"event_data": {"sql": sql, "error": str(exc)}},
            )
            return {
                "execution_result": None,
                "execution_error": str(exc),
                "row_count": 0,
                "execution_time_ms": 0,
                "latency_breakdown": add_stage_latency(state.get("latency_breakdown"), "execution", 0),
            }

        logger.info(
            "sql_execution_succeeded",
            extra={
                "event_data": {
                    "sql": execution_result.executed_sql,
                    "row_count": execution_result.row_count,
                    "execution_time_ms": execution_result.execution_time_ms,
                },
            },
        )
        return {
            "generated_sql": execution_result.executed_sql,
            "execution_result": execution_result.rows,
            "execution_error": None,
            "row_count": execution_result.row_count,
            "execution_time_ms": execution_result.execution_time_ms,
            "result_truncated": execution_result.result_truncated,
            "latency_breakdown": add_stage_latency(
                state.get("latency_breakdown"),
                "execution",
                execution_result.execution_time_ms,
            ),
        }
