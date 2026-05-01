"""HTTP routes for the Agentic SQL Analyst API."""

from __future__ import annotations

import logging
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.agent.graph import SqlAnalystAgent
from app.db.connection import DatabaseClient

logger = logging.getLogger(__name__)
router = APIRouter()


class QueryRequest(BaseModel):
    """Request payload for the SQL analyst endpoint."""

    query: str = Field(..., min_length=1, description="Natural-language analytics question.")
    session_id: str | None = Field(default=None, description="Optional correlation identifier for logs.")
    page: int = Field(default=1, ge=1, description="1-indexed page number for paginated SQL results.")
    page_size: int = Field(default=50, ge=1, le=50, description="Maximum number of rows to return.")


class QueryResponse(BaseModel):
    """Response payload returned by the SQL analyst endpoint."""

    answer: str
    sql: str
    rows: list[dict[str, Any]]
    row_count: int
    execution_time_ms: int
    retries: int
    intent: Literal["sql", "chitchat", "ambiguous"]
    total_latency_ms: int
    latency_breakdown: dict[str, int]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    total_cost_usd: float
    confidence_score: float
    models_used: dict[str, str]
    cache_hits: dict[str, bool]


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    database: str


def get_agent(request: Request) -> SqlAnalystAgent:
    """Resolve the shared SQL analyst agent from application state."""

    return request.app.state.agent


def get_db_client(request: Request) -> DatabaseClient:
    """Resolve the shared database client from application state."""

    return request.app.state.db_client


@router.post("/query", response_model=QueryResponse)
async def query_sql_analyst(
    payload: QueryRequest,
    agent: SqlAnalystAgent = Depends(get_agent),
) -> QueryResponse:
    """Handle natural-language queries and return SQL plus result metadata."""

    logger.info(
        "query_request_received",
        extra={
            "event_data": {
                "query": payload.query,
                "session_id": payload.session_id,
                "page": payload.page,
                "page_size": payload.page_size,
            }
        },
    )
    state = await agent.ainvoke(
        payload.query,
        session_id=payload.session_id,
        page=payload.page,
        page_size=payload.page_size,
    )
    rows = state.get("execution_result") or []
    token_usage = state.get("token_usage", {})
    response = QueryResponse(
        answer=state.get("final_answer", ""),
        sql=state.get("generated_sql", ""),
        rows=rows,
        row_count=state.get("row_count", len(rows)),
        execution_time_ms=state.get("execution_time_ms", 0),
        retries=state.get("retry_count", 0),
        intent=state.get("intent", "ambiguous"),
        total_latency_ms=state.get("total_latency_ms", 0),
        latency_breakdown=state.get("latency_breakdown", {}),
        prompt_tokens=int(token_usage.get("prompt_tokens", 0)),
        completion_tokens=int(token_usage.get("completion_tokens", 0)),
        total_tokens=int(token_usage.get("total_tokens", 0)),
        total_cost_usd=float(state.get("total_cost_usd", 0.0)),
        confidence_score=float(state.get("confidence_score", 0.0)),
        models_used=state.get("models_used", {}),
        cache_hits=state.get("cache_hits", {}),
    )
    logger.info(
        "query_request_completed",
        extra={
            "event_data": {
                "session_id": payload.session_id,
                "intent": response.intent,
                "row_count": response.row_count,
                "retries": response.retries,
            },
        },
    )
    return response


@router.get("/health", response_model=HealthResponse)
async def healthcheck(db_client: DatabaseClient = Depends(get_db_client)) -> HealthResponse:
    """Return service and database health."""

    database_healthy = await db_client.healthcheck()
    if not database_healthy:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connectivity check failed.",
        )
    return HealthResponse(status="ok", database="ok")
