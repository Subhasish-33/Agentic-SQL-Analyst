"""Confidence scoring helpers for final response gating."""

from __future__ import annotations

from app.agent.state import AgentState


def compute_confidence(state: AgentState) -> tuple[float, str]:
    """Compute a lightweight confidence score from execution signals."""

    score = 0.4
    reasons: list[str] = []

    if state.get("validated"):
        score += 0.15
        reasons.append("validator_passed")
    if not state.get("execution_error"):
        score += 0.15
        reasons.append("execution_succeeded")
    if state.get("retry_count", 0) == 0:
        score += 0.1
        reasons.append("no_retry")
    else:
        score -= 0.08 * state.get("retry_count", 0)
        reasons.append("retry_penalty")

    complexity = state.get("query_complexity", "LOW")
    if complexity == "LOW":
        score += 0.1
        reasons.append("low_complexity")
    elif complexity == "MEDIUM":
        score += 0.05
        reasons.append("medium_complexity")

    sql_model = (state.get("models_used") or {}).get("sql_generation") or (state.get("models_used") or {}).get("sql_correction", "")
    if "pro" in sql_model:
        score += 0.08
        reasons.append("high_reasoning_model")
    elif "flash" in sql_model:
        score += 0.04
        reasons.append("fast_model")

    if state.get("guardrail_violations"):
        score -= 0.1
        reasons.append("guardrail_penalty")
    if state.get("validation_error"):
        score -= 0.25
        reasons.append("validation_error")
    if state.get("execution_error"):
        score -= 0.25
        reasons.append("execution_error")

    score = max(0.0, min(round(score, 3), 0.99))
    return score, ", ".join(reasons) if reasons else "insufficient_signal"
