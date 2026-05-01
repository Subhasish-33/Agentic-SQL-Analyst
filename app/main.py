"""FastAPI application entrypoint for the Agentic SQL Analyst."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, Request

from app.agents.runtime import build_runtime
from app.api.routes import router as api_router
from app.config import configure_logging, get_settings

settings = get_settings()
configure_logging(settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and tear down shared application dependencies."""

    runtime = build_runtime(settings)
    app.state.settings = settings
    app.state.cache_service = runtime.cache_service
    app.state.db_client = runtime.db_client
    app.state.vector_store = runtime.vector_store
    app.state.agent = runtime.agent

    logger.info(
        "application_startup_completed",
        extra={
            "event_data": {
                "database_url": settings.DATABASE_URL,
                "vector_db_path": settings.VECTOR_DB_PATH,
                "redis_enabled": bool(settings.REDIS_URL),
                "flash_model": settings.GEMINI_FLASH_MODEL,
                "pro_model": settings.GEMINI_PRO_MODEL,
            }
        },
    )
    try:
        yield
    finally:
        await runtime.db_client.dispose()
        await runtime.cache_service.close()
        logger.info("application_shutdown_completed", extra={"event_data": {}})


app = FastAPI(
    title="Agentic SQL Analyst",
    version="1.0.0",
    description="Production-grade agentic SQL analysis API powered by FastAPI and LangGraph.",
    lifespan=lifespan,
)
app.include_router(api_router)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log request and response metadata for observability."""

    started_at = perf_counter()
    response = await call_next(request)
    duration_ms = int((perf_counter() - started_at) * 1000)
    logger.info(
        "http_request_completed",
        extra={
            "event_data": {
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
                "client": request.client.host if request.client else None,
            },
        },
    )
    return response
