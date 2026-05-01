"""Gemini-oriented LLM service with routing, usage, and cost tracking."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import Settings, coerce_llm_text, get_settings


@dataclass(slots=True)
class LLMResponse:
    """Normalized response returned from the LLM service."""

    text: str
    model: str
    stage: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: int


class AbstractLLMService(Protocol):
    """Protocol shared by live and test-time LLM services."""

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
        """Generate text for a given stage."""


class GeminiLLMService:
    """Gemini-backed LLM service with stage-aware model routing."""

    PRICING_PER_MILLION: dict[str, tuple[float, float]] = {
        "gemini-2.5-flash": (0.30, 2.50),
        "gemini-2.5-pro": (1.25, 10.00),
        "gemini-2.0-flash": (0.10, 0.40),
    }

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the Gemini service."""

        self.settings = settings or get_settings()
        self._models: dict[str, Any] = {}

    def _select_model(
        self,
        *,
        stage: str,
        query_complexity: str,
        retry_count: int,
        preferred_model: str | None,
    ) -> str:
        """Pick the most cost-effective model for the stage."""

        if preferred_model:
            return preferred_model
        if stage == "intent":
            return self.settings.GEMINI_FLASH_MODEL
        if stage in {"sql_correction", "sql_generation"}:
            if retry_count > 0 or query_complexity != "LOW":
                return self.settings.GEMINI_PRO_MODEL
            return self.settings.GEMINI_FLASH_MODEL
        return self.settings.GEMINI_FLASH_MODEL

    def _get_model(self, model_name: str) -> Any:
        """Return a lazily initialized Gemini chat model."""

        if model_name not in self._models:
            from langchain_google_genai import ChatGoogleGenerativeAI

            if not self.settings.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY must be set for Gemini-backed requests.")
            self._models[model_name] = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=self.settings.GEMINI_API_KEY,
                temperature=0,
                timeout=self.settings.LLM_TIMEOUT_SECONDS,
                max_retries=0,
            )
        return self._models[model_name]

    def _extract_usage(self, response: Any, prompt_text: str, completion_text: str) -> tuple[int, int, int]:
        """Extract provider token metadata or fall back to a rough estimate."""

        usage = getattr(response, "usage_metadata", None) or getattr(response, "response_metadata", {}).get("usage_metadata", {})
        prompt_tokens = int(
            usage.get("input_tokens")
            or usage.get("prompt_token_count")
            or usage.get("prompt_tokens")
            or max(1, len(prompt_text) // 4)
        )
        completion_tokens = int(
            usage.get("output_tokens")
            or usage.get("candidates_token_count")
            or usage.get("completion_tokens")
            or max(1, len(completion_text) // 4)
        )
        total_tokens = int(usage.get("total_tokens") or prompt_tokens + completion_tokens)
        return prompt_tokens, completion_tokens, total_tokens

    def _estimate_cost(self, model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate request cost in USD using the configured pricing map."""

        input_price, output_price = self.PRICING_PER_MILLION.get(
            model_name,
            self.PRICING_PER_MILLION[self.settings.GEMINI_FLASH_MODEL],
        )
        return round((prompt_tokens / 1_000_000 * input_price) + (completion_tokens / 1_000_000 * output_price), 8)

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
        """Generate stage-specific text with routing and usage tracking."""

        model_name = self._select_model(
            stage=stage,
            query_complexity=query_complexity,
            retry_count=retry_count,
            preferred_model=preferred_model,
        )
        started_at = perf_counter()
        response = await asyncio.wait_for(
            self._get_model(model_name).ainvoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            ),
            timeout=self.settings.LLM_TIMEOUT_SECONDS,
        )
        text = coerce_llm_text(response)
        prompt_tokens, completion_tokens, total_tokens = self._extract_usage(
            response,
            f"{system_prompt}\n{user_prompt}",
            text,
        )
        latency_ms = int((perf_counter() - started_at) * 1000)
        return LLMResponse(
            text=text,
            model=model_name,
            stage=stage,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=self._estimate_cost(model_name, prompt_tokens, completion_tokens),
            latency_ms=latency_ms,
        )


class LegacyLLMServiceAdapter:
    """Adapter that preserves the earlier fake-LLM testing style."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        intent_llm: Any | None = None,
        sql_llm: Any | None = None,
    ) -> None:
        """Initialize the adapter with stage-specific fake models."""

        self.settings = settings or get_settings()
        self.intent_llm = intent_llm
        self.sql_llm = sql_llm or intent_llm

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
        """Simulate the live LLM service against fake LangChain-compatible objects."""

        _ = (query_complexity, retry_count)
        llm = self.intent_llm if stage == "intent" else self.sql_llm
        if llm is None:
            raise ValueError("A fake LLM must be provided for the requested stage.")
        model_name = preferred_model or (
            self.settings.GEMINI_FLASH_MODEL if stage == "intent" else self.settings.GEMINI_PRO_MODEL
        )
        prompt = f"{system_prompt}\n\n{user_prompt}".strip()
        started_at = perf_counter()
        response = await llm.ainvoke(prompt)
        text = coerce_llm_text(response)
        prompt_tokens = max(1, len(prompt) // 4)
        completion_tokens = max(1, len(text) // 4)
        total_tokens = prompt_tokens + completion_tokens
        latency_ms = int((perf_counter() - started_at) * 1000)
        return LLMResponse(
            text=text,
            model=model_name,
            stage=stage,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=round(total_tokens / 1_000_000, 8),
            latency_ms=latency_ms,
        )
