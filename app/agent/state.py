"""State definitions for the Agentic SQL Analyst workflow."""

from __future__ import annotations

from typing import Any, Literal, Optional

from typing_extensions import NotRequired, TypedDict


class AgentState(TypedDict, total=False):
    """Mutable state shared across LangGraph nodes."""

    user_query: str
    intent: Literal["sql", "chitchat", "ambiguous"]
    relevant_tables: list[str]
    schema_context: str
    generated_sql: str
    validated: bool
    validation_error: Optional[str]
    execution_result: Optional[list[dict]]
    execution_error: Optional[str]
    retry_count: int
    final_answer: str
    sql_explanation: str
    session_id: NotRequired[Optional[str]]
    row_count: NotRequired[int]
    execution_time_ms: NotRequired[int]
    page: NotRequired[int]
    page_size: NotRequired[int]
    sanitized_user_query: NotRequired[str]
    query_complexity: NotRequired[Literal["LOW", "MEDIUM", "HIGH"]]
    confidence_score: NotRequired[float]
    confidence_reason: NotRequired[str]
    latency_breakdown: NotRequired[dict[str, int]]
    total_latency_ms: NotRequired[int]
    token_usage: NotRequired[dict[str, Any]]
    cost_breakdown: NotRequired[dict[str, float]]
    total_cost_usd: NotRequired[float]
    models_used: NotRequired[dict[str, str]]
    cache_hits: NotRequired[dict[str, bool]]
    guardrail_violations: NotRequired[list[str]]
    guardrail_blocked: NotRequired[bool]
    query_cache_key: NotRequired[str]
    result_truncated: NotRequired[bool]
