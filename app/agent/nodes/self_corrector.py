"""Self-correction node for error-aware SQL regeneration."""

from __future__ import annotations

import logging

from app.agent.nodes.sql_generator import SQLGeneratorNode
from app.agent.state import AgentState
from app.config import Settings, get_settings
from app.utils.metrics import add_cost, add_model_usage, add_stage_latency, add_token_usage

logger = logging.getLogger(__name__)


class SelfCorrectorNode:
    """Regenerate SQL using the previous execution error as feedback."""

    def __init__(self, sql_generator: SQLGeneratorNode, settings: Settings | None = None) -> None:
        """Initialize the self-corrector."""

        self.sql_generator = sql_generator
        self.settings = settings or get_settings()

    async def __call__(self, state: AgentState) -> AgentState:
        """Increment retry count and attempt an error-aware SQL correction."""

        next_retry = state.get("retry_count", 0) + 1
        logger.info(
            "sql_self_correction_started",
            extra={
                "event_data": {
                    "retry_count": next_retry,
                    "previous_sql": state.get("generated_sql"),
                    "execution_error": state.get("execution_error"),
                },
            },
        )

        if next_retry > self.settings.MAX_RETRIES:
            final_answer = (
                "I was unable to generate a valid query for this request. "
                f"Error: {state.get('execution_error', 'Unknown execution error.')}"
            )
            return {"retry_count": next_retry, "final_answer": final_answer}

        try:
            corrected_sql, response = await self.sql_generator.generate(state, correction_mode=True)
        except Exception as exc:
            logger.exception(
                "sql_self_correction_failed",
                extra={"event_data": {"retry_count": next_retry, "error": str(exc)}},
            )
            return {
                "retry_count": next_retry,
                "generated_sql": "",
                "validation_error": f"Self-correction failed: {exc}",
                "final_answer": "I was unable to repair the SQL query after an execution failure.",
            }

        logger.info(
            "sql_self_correction_completed",
            extra={"event_data": {"retry_count": next_retry, "corrected_sql": corrected_sql}},
        )
        return {
            "retry_count": next_retry,
            "generated_sql": corrected_sql,
            "validated": False,
            "validation_error": None,
            "latency_breakdown": add_stage_latency(state.get("latency_breakdown"), "llm_sql", response.latency_ms),
            "token_usage": add_token_usage(state.get("token_usage"), "sql_correction", response),
            "models_used": add_model_usage(state.get("models_used"), "sql_correction", response.model),
            "cost_breakdown": add_cost(state.get("cost_breakdown"), "sql_correction", response.cost_usd),
        }
