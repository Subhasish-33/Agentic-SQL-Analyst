"""Heuristics for classifying natural-language query complexity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ComplexityLevel = Literal["LOW", "MEDIUM", "HIGH"]


@dataclass(slots=True)
class ComplexityAssessment:
    """Structured complexity classification result."""

    level: ComplexityLevel
    score: int
    reasons: list[str]


def assess_query_complexity(query: str) -> ComplexityAssessment:
    """Estimate SQL-generation complexity from the user query."""

    normalized = query.lower()
    score = 0
    reasons: list[str] = []

    if len(normalized.split()) > 18:
        score += 1
        reasons.append("long_query")
    if any(token in normalized for token in ("join", "across", "compare", "trend", "cohort", "funnel")):
        score += 2
        reasons.append("multi_entity_analysis")
    if any(token in normalized for token in ("top", "rank", "highest", "lowest", "most", "least")):
        score += 1
        reasons.append("ranking")
    if any(token in normalized for token in ("sum", "count", "average", "avg", "group by", "per ")) and " by " in normalized:
        score += 1
        reasons.append("aggregation")
    if any(token in normalized for token in ("month", "quarter", "year", "week", "last", "between", "before", "after")):
        score += 1
        reasons.append("time_filter")

    if score >= 4:
        return ComplexityAssessment(level="HIGH", score=score, reasons=reasons)
    if score >= 2:
        return ComplexityAssessment(level="MEDIUM", score=score, reasons=reasons)
    return ComplexityAssessment(level="LOW", score=score, reasons=reasons or ["simple_lookup"])
