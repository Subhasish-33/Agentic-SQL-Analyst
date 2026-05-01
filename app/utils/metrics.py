"""Helpers for updating latency, cost, and cache metrics in state."""

from __future__ import annotations

from typing import Any

from app.services.llm import LLMResponse


def add_stage_latency(existing: dict[str, int] | None, stage: str, latency_ms: int) -> dict[str, int]:
    """Return an updated latency breakdown."""

    updated = dict(existing or {})
    updated[stage] = int(updated.get(stage, 0) + latency_ms)
    return updated


def total_latency_ms(breakdown: dict[str, int] | None) -> int:
    """Return the sum of the recorded stage latencies."""

    return sum((breakdown or {}).values())


def add_cache_hit(existing: dict[str, bool] | None, cache_name: str, hit: bool) -> dict[str, bool]:
    """Return an updated cache hit map."""

    updated = dict(existing or {})
    updated[cache_name] = hit
    return updated


def add_model_usage(existing: dict[str, str] | None, stage: str, model: str) -> dict[str, str]:
    """Return an updated map of stage-to-model selections."""

    updated = dict(existing or {})
    updated[stage] = model
    return updated


def add_cost(existing: dict[str, float] | None, stage: str, amount: float) -> dict[str, float]:
    """Return an updated map of per-stage costs."""

    updated = dict(existing or {})
    updated[stage] = round(updated.get(stage, 0.0) + amount, 8)
    return updated


def total_cost(cost_breakdown: dict[str, float] | None) -> float:
    """Return the aggregated request cost."""

    return round(sum((cost_breakdown or {}).values()), 8)


def add_token_usage(existing: dict[str, Any] | None, stage: str, response: LLMResponse) -> dict[str, Any]:
    """Accumulate request-wide token usage from an LLM response."""

    usage = dict(existing or {})
    by_stage = dict(usage.get("by_stage", {}))
    by_stage[stage] = {
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "total_tokens": response.total_tokens,
        "model": response.model,
    }
    usage["by_stage"] = by_stage
    usage["prompt_tokens"] = int(usage.get("prompt_tokens", 0) + response.prompt_tokens)
    usage["completion_tokens"] = int(usage.get("completion_tokens", 0) + response.completion_tokens)
    usage["total_tokens"] = int(usage.get("total_tokens", 0) + response.total_tokens)
    return usage
