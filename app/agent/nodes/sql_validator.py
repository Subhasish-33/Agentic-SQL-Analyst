"""SQL validation node that applies the safety layer."""

from __future__ import annotations

import logging
from time import perf_counter

from app.agent.state import AgentState
from app.safety.validator import QuerySafetyValidator
from app.utils.metrics import add_stage_latency

logger = logging.getLogger(__name__)


class SQLValidatorNode:
    """Validate generated SQL using regex and AST-based checks."""

    def __init__(self, validator: QuerySafetyValidator) -> None:
        """Initialize the SQL validator node."""

        self.validator = validator

    async def __call__(self, state: AgentState) -> AgentState:
        """Validate SQL and stop the workflow if it is unsafe."""

        generated_sql = state.get("generated_sql", "")
        started_at = perf_counter()
        logger.info(
            "sql_validation_node_started",
            extra={"event_data": {"sql": generated_sql}},
        )
        validation = self.validator.validate(generated_sql, state.get("schema_context", ""))
        stage_latency = int((perf_counter() - started_at) * 1000)
        if not validation.valid:
            answer = f"I cannot run that query safely. {validation.error}"
            logger.warning(
                "sql_validation_node_failed",
                extra={"event_data": {"sql": generated_sql, "error": validation.error}},
            )
            return {
                "validated": False,
                "validation_error": validation.error,
                "final_answer": answer,
                "latency_breakdown": add_stage_latency(state.get("latency_breakdown"), "validation", stage_latency),
            }

        logger.info(
            "sql_validation_node_completed",
            extra={"event_data": {"sql": validation.normalized_sql}},
        )
        return {
            "generated_sql": validation.normalized_sql,
            "validated": True,
            "validation_error": None,
            "latency_breakdown": add_stage_latency(state.get("latency_breakdown"), "validation", stage_latency),
        }
