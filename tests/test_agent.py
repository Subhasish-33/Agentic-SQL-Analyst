"""Integration-style tests for the LangGraph SQL analyst agent."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from app.agent.graph import SqlAnalystAgent
from app.config import Settings
from app.db.connection import DatabaseClient
from app.embeddings.store import SchemaSearchResult
from app.safety.validator import QuerySafetyValidator

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "sample_schema.sql"


class FakeLLMResponse:
    """Minimal response object that mimics LangChain chat output."""

    def __init__(self, content: str) -> None:
        """Store response text for later extraction."""

        self.content = content


class FakeLLM:
    """Deterministic fake LLM used to drive graph behavior in tests."""

    def __init__(self, responses: list[str]) -> None:
        """Initialize the fake with a fixed response sequence."""

        self.responses = responses
        self.prompts: list[str] = []

    async def ainvoke(self, prompt: str) -> FakeLLMResponse:
        """Return the next queued response."""

        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("FakeLLM ran out of responses.")
        return FakeLLMResponse(self.responses.pop(0))


class FakeVectorStore:
    """Static schema retriever used for deterministic tests."""

    def similarity_search(self, query: str, k: int) -> list[SchemaSearchResult]:
        """Return a pre-baked schema result for the users table."""

        _ = (query, k)
        return [
            SchemaSearchResult(
                table_name="users",
                schema_text="Table users stores user accounts.",
                metadata={"table_name": "users", "relationships_json": "[]"},
                columns=[
                    {"name": "id", "type": "INTEGER", "is_primary_key": True, "foreign_key_to": None},
                    {"name": "name", "type": "TEXT", "is_primary_key": False, "foreign_key_to": None},
                    {"name": "email", "type": "TEXT", "is_primary_key": False, "foreign_key_to": None},
                    {"name": "signup_source", "type": "TEXT", "is_primary_key": False, "foreign_key_to": None},
                    {"name": "created_at", "type": "TIMESTAMP", "is_primary_key": False, "foreign_key_to": None},
                ],
            ),
        ]


def bootstrap_sqlite_database(tmp_path: Path) -> str:
    """Load the sample schema fixture into a temporary SQLite database."""

    database_path = tmp_path / "agent.db"
    connection = sqlite3.connect(database_path)
    try:
        connection.executescript(FIXTURE_PATH.read_text(encoding="utf-8"))
        connection.commit()
    finally:
        connection.close()
    return f"sqlite+aiosqlite:///{database_path}"


def build_agent(
    tmp_path: Path,
    intent_responses: list[str],
    sql_responses: list[str],
) -> SqlAnalystAgent:
    """Construct a test agent with fake LLMs and an on-disk SQLite database."""

    database_url = bootstrap_sqlite_database(tmp_path)
    settings = Settings(
        DATABASE_URL=database_url,
        OPENAI_API_KEY="test-key",
        LARGE_TABLES=["orders", "order_items"],
        VECTOR_DB_PATH=str(tmp_path / "chroma_db"),
    )
    validator = QuerySafetyValidator(settings=settings, large_table_names=["orders", "order_items"])
    db_client = DatabaseClient(settings)
    return SqlAnalystAgent(
        settings=settings,
        db_client=db_client,
        vector_store=FakeVectorStore(),
        intent_llm=FakeLLM(intent_responses),
        sql_llm=FakeLLM(sql_responses),
        validator=validator,
    )


@pytest.mark.asyncio
async def test_happy_path_query_to_result(tmp_path: Path) -> None:
    """A valid query should produce SQL and rows without retries."""

    agent = build_agent(
        tmp_path,
        intent_responses=["sql"],
        sql_responses=["SELECT id, name, email FROM users WHERE created_at >= '2024-01-01' LIMIT 5"],
    )

    state = await agent.ainvoke("Show me the first five users created in 2024.")
    await agent.db_client.dispose()

    assert state["intent"] == "sql"
    assert state["execution_error"] is None
    assert state["retry_count"] == 0
    assert state["row_count"] == 5
    assert len(state["execution_result"] or []) == 5
    assert "SQL used:" in state["final_answer"]


@pytest.mark.asyncio
async def test_self_correction_recovers_after_execution_error(tmp_path: Path) -> None:
    """Execution errors should trigger a corrective SQL regeneration."""

    agent = build_agent(
        tmp_path,
        intent_responses=["sql"],
        sql_responses=[
            "SELECT DATE_TRUNC('day', created_at) AS signup_day FROM users WHERE created_at >= '2024-01-01'",
            "SELECT id, name, created_at FROM users WHERE created_at >= '2024-01-01' LIMIT 3",
        ],
    )

    state = await agent.ainvoke("Show me recent users by signup day.")
    await agent.db_client.dispose()

    assert state["execution_error"] is None
    assert state["retry_count"] == 1
    assert state["row_count"] == 3
    assert "DATE_TRUNC" not in state["generated_sql"]


@pytest.mark.asyncio
async def test_retry_exhaustion_returns_error_message(tmp_path: Path) -> None:
    """The agent should stop after the configured number of correction attempts."""

    agent = build_agent(
        tmp_path,
        intent_responses=["sql"],
        sql_responses=[
            "SELECT DATE_TRUNC('day', created_at) AS signup_day FROM users WHERE created_at >= '2024-01-01'",
            "SELECT DATE_TRUNC('day', created_at) AS signup_day FROM users WHERE created_at >= '2024-01-01'",
            "SELECT DATE_TRUNC('day', created_at) AS signup_day FROM users WHERE created_at >= '2024-01-01'",
            "SELECT DATE_TRUNC('day', created_at) AS signup_day FROM users WHERE created_at >= '2024-01-01'",
        ],
    )

    state = await agent.ainvoke("Show me recent users by signup day.")
    await agent.db_client.dispose()

    assert state["retry_count"] == 3
    assert "unable to generate a valid query" in state["final_answer"].lower()
    assert state["execution_error"] is not None


@pytest.mark.asyncio
async def test_chitchat_query_exits_early(tmp_path: Path) -> None:
    """Non-SQL messages should end the workflow before retrieval or execution."""

    agent = build_agent(tmp_path, intent_responses=["chitchat"], sql_responses=[])
    state = await agent.ainvoke("What's the weather today?")
    await agent.db_client.dispose()

    assert state["intent"] == "chitchat"
    assert state["generated_sql"] == ""
    assert state["row_count"] == 0
    assert "SQL analysis" in state["final_answer"]
