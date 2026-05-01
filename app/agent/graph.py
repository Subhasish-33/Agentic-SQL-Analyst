"""LangGraph orchestration for the Agentic SQL Analyst."""

from __future__ import annotations

import logging
from typing import Any, cast

from langgraph.graph import END, START, StateGraph

from app.agent.nodes.execution_engine import ExecutionEngineNode
from app.agent.nodes.intent_classifier import IntentClassifierNode
from app.agent.nodes.prefetch_context import PrefetchContextNode
from app.agent.nodes.result_explainer import ResultExplainerNode
from app.agent.nodes.schema_retriever import SchemaRetrieverNode
from app.agent.nodes.self_corrector import SelfCorrectorNode
from app.agent.nodes.sql_generator import SQLGeneratorNode
from app.agent.nodes.sql_validator import SQLValidatorNode
from app.agent.state import AgentState
from app.config import Settings, get_settings
from app.db.connection import DatabaseClient
from app.embeddings.store import SchemaVectorStore
from app.safety.validator import QuerySafetyValidator
from app.services.cache import CacheService
from app.services.llm import AbstractLLMService, GeminiLLMService, LegacyLLMServiceAdapter
from app.services.schema_catalog import SchemaCatalogService
from app.utils.complexity import assess_query_complexity
from app.utils.confidence import compute_confidence
from app.utils.guardrails import filter_output_text, sanitize_user_query
from app.utils.metrics import add_cache_hit, total_cost, total_latency_ms

logger = logging.getLogger(__name__)


class SqlAnalystAgent:
    """Production-oriented LangGraph agent for natural language SQL analysis."""

    def __init__(
        self,
        settings: Settings | None = None,
        db_client: DatabaseClient | None = None,
        vector_store: SchemaVectorStore | None = None,
        llm_service: AbstractLLMService | None = None,
        cache_service: CacheService | None = None,
        schema_catalog: SchemaCatalogService | None = None,
        intent_llm: Any | None = None,
        sql_llm: Any | None = None,
        validator: QuerySafetyValidator | None = None,
    ) -> None:
        """Initialize the agent, nodes, and compiled graph."""

        self.settings = settings or get_settings()
        self.db_client = db_client or DatabaseClient(self.settings)
        self.cache_service = cache_service or CacheService(self.settings)
        self.schema_catalog = schema_catalog or SchemaCatalogService(self.settings)
        self.vector_store = vector_store or SchemaVectorStore(self.settings, cache_service=self.cache_service)
        self.llm_service = llm_service or (
            LegacyLLMServiceAdapter(
                settings=self.settings,
                intent_llm=intent_llm,
                sql_llm=sql_llm,
            )
            if intent_llm is not None or sql_llm is not None
            else GeminiLLMService(self.settings)
        )
        self.validator = validator or QuerySafetyValidator(
            self.settings,
            schema_catalog=self.schema_catalog,
        )

        self.intent_classifier = IntentClassifierNode(self.settings, llm_service=self.llm_service, llm=intent_llm)
        self.schema_retriever = SchemaRetrieverNode(
            self.settings,
            vector_store=self.vector_store,
            cache_service=self.cache_service,
        )
        self.prefetch_context = PrefetchContextNode(self.intent_classifier, self.schema_retriever)
        self.sql_generator = SQLGeneratorNode(
            self.db_client,
            self.settings,
            llm_service=self.llm_service,
            llm=sql_llm,
        )
        self.sql_validator = SQLValidatorNode(self.validator)
        self.execution_engine = ExecutionEngineNode(self.db_client)
        self.self_corrector = SelfCorrectorNode(self.sql_generator, self.settings)
        self.result_explainer = ResultExplainerNode(self.settings)
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        """Construct and compile the LangGraph workflow."""

        workflow = StateGraph(AgentState)
        workflow.add_node("prefetch_context", self.prefetch_context)
        workflow.add_node("classify_intent", self.intent_classifier)
        workflow.add_node("retrieve_schema", self.schema_retriever)
        workflow.add_node("generate_sql", self.sql_generator)
        workflow.add_node("validate_sql", self.sql_validator)
        workflow.add_node("execute_sql", self.execution_engine)
        workflow.add_node("check_error", self.check_error)
        workflow.add_node("self_correct", self.self_corrector)
        workflow.add_node("explain_result", self.result_explainer)

        workflow.add_edge(START, "prefetch_context")
        workflow.add_edge("prefetch_context", "classify_intent")
        workflow.add_conditional_edges(
            "classify_intent",
            self.route_after_intent,
            {"retrieve_schema": "retrieve_schema", END: END},
        )
        workflow.add_conditional_edges(
            "retrieve_schema",
            self.route_after_schema_retrieval,
            {"generate_sql": "generate_sql", END: END},
        )
        workflow.add_edge("generate_sql", "validate_sql")
        workflow.add_conditional_edges(
            "validate_sql",
            self.route_after_validation,
            {"execute_sql": "execute_sql", END: END},
        )
        workflow.add_edge("execute_sql", "check_error")
        workflow.add_conditional_edges(
            "check_error",
            self.route_after_error_check,
            {"self_correct": "self_correct", "explain_result": "explain_result", END: END},
        )
        workflow.add_edge("self_correct", "validate_sql")
        workflow.add_edge("explain_result", END)
        return workflow.compile()

    async def check_error(self, state: AgentState) -> AgentState:
        """Finalize the workflow when retries are exhausted."""

        execution_error = state.get("execution_error")
        retry_count = state.get("retry_count", 0)
        if execution_error and retry_count >= self.settings.MAX_RETRIES:
            final_answer = (
                "I was unable to generate a valid query for this request. "
                f"Error: {execution_error}"
            )
            logger.warning(
                "sql_retry_exhausted",
                extra={"event_data": {"retry_count": retry_count, "execution_error": execution_error}},
            )
            return {"retry_count": retry_count, "final_answer": final_answer}
        return {"retry_count": retry_count}

    def route_after_intent(self, state: AgentState) -> str:
        """Route SQL requests into the graph and stop other intents early."""

        return "retrieve_schema" if state.get("intent") == "sql" else END

    def route_after_schema_retrieval(self, state: AgentState) -> str:
        """Stop the workflow when schema retrieval fails."""

        if state.get("validation_error"):
            return END
        return "generate_sql"

    def route_after_validation(self, state: AgentState) -> str:
        """Proceed only when validation succeeds."""

        return END if state.get("validation_error") else "execute_sql"

    def route_after_error_check(self, state: AgentState) -> str:
        """Send execution failures to self-correction or end on exhaustion."""

        if state.get("execution_error"):
            if state.get("retry_count", 0) < self.settings.MAX_RETRIES:
                return "self_correct"
            return END
        return "explain_result"

    def build_initial_state(self, user_query: str, session_id: str | None = None) -> AgentState:
        """Create the initial graph state for a request."""

        guardrail = sanitize_user_query(user_query)
        complexity = assess_query_complexity(guardrail.cleaned_text)
        return {
            "user_query": user_query,
            "intent": "ambiguous",
            "relevant_tables": [],
            "schema_context": "",
            "generated_sql": "",
            "validated": False,
            "validation_error": None,
            "execution_result": None,
            "execution_error": None,
            "retry_count": 0,
            "final_answer": "",
            "sql_explanation": "",
            "session_id": session_id,
            "row_count": 0,
            "execution_time_ms": 0,
            "page": 1,
            "page_size": self.settings.DEFAULT_PAGE_SIZE,
            "sanitized_user_query": guardrail.cleaned_text,
            "query_complexity": complexity.level,
            "latency_breakdown": {},
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "by_stage": {}},
            "cost_breakdown": {},
            "models_used": {},
            "cache_hits": {},
            "guardrail_violations": guardrail.violations,
            "guardrail_blocked": guardrail.blocked,
            "query_cache_key": guardrail.cleaned_text,
            "result_truncated": False,
        }

    async def _load_cached_result(self, cache_key: str) -> AgentState | None:
        """Attempt to resolve a previous successful query result from cache."""

        cached = await self.cache_service.get_json("query_result", cache_key)
        if cached is None:
            return None
        cached_state = cast(AgentState, cached)
        cached_state["cache_hits"] = add_cache_hit(cached_state.get("cache_hits"), "query_result", True)
        cached_state["latency_breakdown"] = {"query_cache": 1}
        cached_state["total_latency_ms"] = 1
        cached_state["total_cost_usd"] = total_cost(cached_state.get("cost_breakdown"))
        return cached_state

    async def _store_cached_result(self, cache_key: str, state: AgentState) -> None:
        """Persist successful SQL results for repeat requests."""

        if state.get("intent") != "sql" or state.get("execution_error") or not state.get("generated_sql"):
            return
        await self.cache_service.set_json(
            "query_result",
            cache_key,
            state,
            ttl_seconds=self.settings.QUERY_RESULT_CACHE_TTL_SECONDS,
        )

    def _apply_confidence_guard(self, state: AgentState) -> AgentState:
        """Compute confidence and gate low-confidence responses."""

        confidence_score, confidence_reason = compute_confidence(state)
        state["confidence_score"] = confidence_score
        state["confidence_reason"] = confidence_reason
        state["total_latency_ms"] = total_latency_ms(state.get("latency_breakdown"))
        state["total_cost_usd"] = total_cost(state.get("cost_breakdown"))
        state["final_answer"] = filter_output_text(state.get("final_answer", ""))
        if (
            state.get("intent") == "sql"
            and confidence_score < self.settings.CONFIDENCE_THRESHOLD
            and not state.get("execution_error")
            and not state.get("validation_error")
        ):
            state["final_answer"] = "I'm not fully confident in this answer. Please refine your query."
        return state

    async def ainvoke(
        self,
        user_query: str,
        session_id: str | None = None,
        *,
        page: int = 1,
        page_size: int | None = None,
    ) -> AgentState:
        """Execute the graph end to end for a user query."""

        initial_state = self.build_initial_state(user_query, session_id=session_id)
        initial_state["page"] = max(page, 1)
        initial_state["page_size"] = min(
            max(page_size or self.settings.DEFAULT_PAGE_SIZE, 1),
            self.settings.MAX_PAGE_SIZE,
        )
        if initial_state.get("guardrail_blocked"):
            initial_state["final_answer"] = "I could not safely interpret that request. Please restate it as a database question."
            initial_state["total_latency_ms"] = 0
            initial_state["total_cost_usd"] = 0.0
            return initial_state

        cache_key = f"{initial_state.get('query_cache_key')}|page={initial_state['page']}|page_size={initial_state['page_size']}"
        cached_state = await self._load_cached_result(cache_key)
        if cached_state is not None:
            logger.info(
                "sql_analyst_query_cache_hit",
                extra={"event_data": {"cache_key": cache_key, "session_id": session_id}},
            )
            return self._apply_confidence_guard(cached_state)

        initial_state["cache_hits"] = add_cache_hit(initial_state.get("cache_hits"), "query_result", False)
        logger.info(
            "sql_analyst_run_started",
            extra={
                "event_data": {
                    "user_query": user_query,
                    "sanitized_user_query": initial_state.get("sanitized_user_query"),
                    "session_id": session_id,
                    "query_complexity": initial_state.get("query_complexity"),
                }
            },
        )
        final_state = cast(AgentState, await self.graph.ainvoke(initial_state))
        final_state = self._apply_confidence_guard(final_state)
        await self._store_cached_result(cache_key, final_state)
        logger.info(
            "sql_analyst_run_completed",
            extra={
                "event_data": {
                    "user_query": user_query,
                    "intent": final_state.get("intent"),
                    "retry_count": final_state.get("retry_count", 0),
                    "row_count": final_state.get("row_count", 0),
                    "total_latency_ms": final_state.get("total_latency_ms"),
                    "latency_breakdown": final_state.get("latency_breakdown"),
                    "token_usage": final_state.get("token_usage"),
                    "total_cost_usd": final_state.get("total_cost_usd"),
                    "models_used": final_state.get("models_used"),
                    "confidence_score": final_state.get("confidence_score"),
                },
            },
        )
        return final_state
