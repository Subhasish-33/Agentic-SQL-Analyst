"""Evaluation harness for the upgraded Agentic SQL Analyst."""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.agent.graph import SqlAnalystAgent
from app.config import Settings
from app.db.connection import DatabaseClient
from app.embeddings.store import SchemaSearchResult
from app.safety.validator import QuerySafetyValidator
from app.services.cache import CacheService
from app.services.llm import AbstractLLMService, LLMResponse
from app.services.schema_catalog import SchemaCatalogService

DATASET_PATH = Path(__file__).resolve().parents[2] / "evaluation" / "queries.json"
FIXTURE_PATH = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "sample_schema.sql"


@dataclass(slots=True)
class EvaluationMetrics:
    """Aggregated evaluation metrics."""

    total_queries: int
    success_rate: float
    retry_rate: float
    average_latency_ms: float
    average_cost_usd: float
    failure_cases: list[dict[str, Any]]


def load_dataset(limit: int | None = None) -> list[dict[str, Any]]:
    """Load the natural-language evaluation dataset."""

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    return dataset[:limit] if limit else dataset


def bootstrap_sqlite_database() -> str:
    """Create a temporary SQLite database from the sample schema fixture."""

    temp_dir = tempfile.mkdtemp(prefix="agentic-sql-eval-")
    database_path = Path(temp_dir) / "eval.db"
    connection = sqlite3.connect(database_path)
    try:
        connection.executescript(FIXTURE_PATH.read_text(encoding="utf-8"))
        connection.commit()
    finally:
        connection.close()
    return f"sqlite+aiosqlite:///{database_path}"


class DatasetVectorStore:
    """Deterministic schema retriever backed by the evaluation dataset."""

    def __init__(self, dataset: list[dict[str, Any]], schema_catalog: SchemaCatalogService) -> None:
        """Initialize the deterministic schema store."""

        self.dataset = dataset
        self.schema_catalog = schema_catalog
        self.lookup: dict[str, dict[str, Any]] = {}
        for item in dataset:
            self.lookup[item["query"]] = item
            if item.get("sanitized_query"):
                self.lookup[item["sanitized_query"]] = item

    def similarity_search(self, query: str, k: int) -> list[SchemaSearchResult]:
        """Return the dataset-provided relevant tables for a query."""

        item = self.lookup.get(query)
        if item is None:
            return []
        results: list[SchemaSearchResult] = []
        for table_name in item.get("relevant_tables", [])[:k]:
            table = self.schema_catalog.get_catalog().get(table_name)
            if table is None:
                continue
            results.append(
                SchemaSearchResult(
                    table_name=table_name,
                    schema_text=self.schema_catalog.format_table_context(table_name),
                    metadata={
                        "table_name": table_name,
                        "relationships_json": table["relationships_json"],
                    },
                    columns=[
                        {
                            "name": column_name,
                            "type": column_type,
                            "is_primary_key": column_name in table["primary_keys"],
                            "foreign_key_to": next(
                                (relationship["to"] for relationship in table["relationships"] if relationship["from"] == column_name),
                                None,
                            ),
                        }
                        for column_name, column_type in table["columns"].items()
                    ],
                )
            )
        return results


class DeterministicEvaluationLLMService(AbstractLLMService):
    """Dataset-driven LLM simulator for offline evaluation and demos."""

    def __init__(self, settings: Settings, dataset: list[dict[str, Any]]) -> None:
        """Initialize the deterministic LLM service."""

        self.settings = settings
        self.dataset_lookup: dict[str, dict[str, Any]] = {}
        self.stage_counts: dict[tuple[str, str], int] = {}
        for item in dataset:
            self.dataset_lookup[item["query"]] = item
            if item.get("sanitized_query"):
                self.dataset_lookup[item["sanitized_query"]] = item

    def _extract_query(self, user_prompt: str) -> str:
        """Extract the raw query from the prompt body."""

        if "User question:" in user_prompt:
            return user_prompt.split("User question:", 1)[1].split("\n", 1)[0].strip()
        if "User:" in user_prompt:
            matches = re.findall(r"User:\s*(.*?)\nLabel:", user_prompt, flags=re.DOTALL)
            if matches:
                return matches[-1].strip()
        return user_prompt.strip()

    def _model_for_stage(self, stage: str, query_complexity: str, retry_count: int) -> str:
        """Mirror the production routing behavior for evaluation."""

        if stage == "intent":
            return self.settings.GEMINI_FLASH_MODEL
        if stage in {"sql_generation", "sql_correction"} and (retry_count > 0 or query_complexity != "LOW"):
            return self.settings.GEMINI_PRO_MODEL
        return self.settings.GEMINI_FLASH_MODEL

    async def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        stage: str,
        query_complexity: str = "LOW",
        retry_count: int = 0,
        preferred_model: str | None = None,
    ) -> LLMResponse:
        """Return deterministic responses from the evaluation dataset."""

        query = self._extract_query(user_prompt)
        item = self.dataset_lookup.get(query)
        if item is None:
            raise KeyError(f"No evaluation fixture found for query: {query}")

        counter_key = (item["id"], stage)
        self.stage_counts[counter_key] = self.stage_counts.get(counter_key, 0) + 1

        if stage == "intent":
            text = item["expected_intent"]
        elif stage == "sql_generation":
            text = item.get("initial_sql") or item.get("sql", "")
        elif stage == "sql_correction":
            text = item.get("retry_sql") or item.get("sql", "")
        else:
            text = item.get("sql", "")

        model_name = preferred_model or self._model_for_stage(stage, query_complexity, retry_count)
        prompt_tokens = max(1, len(f"{system_prompt}\n{user_prompt}") // 4)
        completion_tokens = max(1, len(text) // 4)
        total_tokens = prompt_tokens + completion_tokens
        cost_multiplier = 0.000001
        if "pro" in model_name:
            cost_multiplier = 0.000003
        return LLMResponse(
            text=text,
            model=model_name,
            stage=stage,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=round(total_tokens * cost_multiplier, 8),
            latency_ms=5 if stage == "intent" else 12,
        )


async def run_evaluation(limit: int | None = None) -> EvaluationMetrics:
    """Execute the evaluation dataset against the upgraded agent."""

    dataset = load_dataset(limit=limit)
    database_url = bootstrap_sqlite_database()
    settings = Settings(
        DATABASE_URL=database_url,
        GEMINI_API_KEY="demo-key",
        VECTOR_DB_PATH="./evaluation_chroma",
        LARGE_TABLES=["orders", "order_items"],
        TOP_K_TABLES=5,
    )
    cache_service = CacheService(settings)
    schema_catalog = SchemaCatalogService(settings)
    llm_service = DeterministicEvaluationLLMService(settings, dataset)
    vector_store = DatasetVectorStore(dataset, schema_catalog)
    validator = QuerySafetyValidator(
        settings=settings,
        large_table_names=["orders", "order_items"],
        schema_catalog=schema_catalog,
    )
    db_client = DatabaseClient(settings)
    agent = SqlAnalystAgent(
        settings=settings,
        db_client=db_client,
        vector_store=vector_store,
        llm_service=llm_service,
        cache_service=cache_service,
        schema_catalog=schema_catalog,
        validator=validator,
    )

    successes = 0
    retries = 0
    total_latency = 0
    total_cost = 0.0
    failure_cases: list[dict[str, Any]] = []

    for item in dataset:
        state = await agent.ainvoke(item["query"], session_id=f"eval-{item['id']}")
        total_latency += state.get("total_latency_ms", 0)
        total_cost += float(state.get("total_cost_usd", 0.0))
        if state.get("retry_count", 0) > 0:
            retries += 1

        expected_intent = item["expected_intent"]
        if expected_intent != "sql":
            success = state.get("intent") == expected_intent
        else:
            success = state.get("intent") == "sql" and state.get("execution_error") is None and state.get("confidence_score", 0.0) >= settings.CONFIDENCE_THRESHOLD

        if success:
            successes += 1
        else:
            failure_cases.append(
                {
                    "id": item["id"],
                    "query": item["query"],
                    "intent": state.get("intent"),
                    "error": state.get("execution_error") or state.get("validation_error") or state.get("final_answer"),
                    "confidence_score": state.get("confidence_score", 0.0),
                }
            )

    await db_client.dispose()
    await cache_service.close()
    return EvaluationMetrics(
        total_queries=len(dataset),
        success_rate=round(successes / len(dataset) * 100, 2),
        retry_rate=round(retries / len(dataset) * 100, 2),
        average_latency_ms=round(total_latency / len(dataset), 2),
        average_cost_usd=round(total_cost / len(dataset), 8),
        failure_cases=failure_cases,
    )


def format_metrics(metrics: EvaluationMetrics) -> dict[str, Any]:
    """Convert metrics into a CLI-friendly dictionary."""

    return {
        "total_queries": metrics.total_queries,
        "success_rate_percent": metrics.success_rate,
        "retry_rate_percent": metrics.retry_rate,
        "average_latency_ms": metrics.average_latency_ms,
        "average_cost_usd": metrics.average_cost_usd,
        "failure_cases": metrics.failure_cases,
    }


async def main(limit: int | None = None) -> dict[str, Any]:
    """Run the evaluation and return a structured result."""

    metrics = await run_evaluation(limit=limit)
    return format_metrics(metrics)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.ERROR)
    result = asyncio.run(main())
    print(json.dumps(result, indent=2))
