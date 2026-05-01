"""Result explanation node for turning rows into concise natural language."""

from __future__ import annotations

import logging

from app.agent.state import AgentState
from app.config import Settings, get_settings
from app.utils.guardrails import filter_output_text

logger = logging.getLogger(__name__)


class ResultExplainerNode:
    """Summarize SQL results into a concise answer for the user."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the result explainer."""

        self.settings = settings or get_settings()

    def _preview_row(self, row: dict) -> str:
        """Create a compact first-row preview for the final answer."""

        preview_pairs: list[str] = []
        for key, value in row.items():
            preview_pairs.append(f"{key}={value}")
            if len(preview_pairs) == 3:
                break
        return ", ".join(preview_pairs) if preview_pairs else "no preview fields were available"

    async def __call__(self, state: AgentState) -> AgentState:
        """Build the final natural-language response from query results."""

        rows = state.get("execution_result") or []
        row_count = state.get("row_count", len(rows))
        sql = state.get("generated_sql", "")
        relevant_tables = state.get("relevant_tables", [])

        if not rows:
            final_answer = f"No records matched your query. SQL used: {sql}"
            sql_explanation = (
                f"The query read from {', '.join(relevant_tables) if relevant_tables else 'the retrieved tables'} "
                "but returned no rows."
            )
        else:
            preview = self._preview_row(rows[0])
            if row_count > self.settings.LARGE_RESULT_ROW_THRESHOLD or state.get("result_truncated"):
                final_answer = (
                    f"Found {row_count} records. I’m summarizing the result instead of dumping the full dataset. "
                    f"A representative row looks like {preview}. SQL used: {sql}"
                )
            else:
                final_answer = (
                    f"Found {row_count} records that match your request. "
                    f"The first row looks like {preview}. SQL used: {sql}"
                )
            sql_explanation = (
                f"The query read from {', '.join(relevant_tables) if relevant_tables else 'the retrieved tables'} "
                f"and returned {row_count} rows."
            )

        logger.info(
            "result_explanation_completed",
            extra={"event_data": {"row_count": row_count, "sql": sql}},
        )
        return {"final_answer": filter_output_text(final_answer), "sql_explanation": filter_output_text(sql_explanation)}
