"""Prompt and output guardrails for user-facing safety."""

from __future__ import annotations

import re
from dataclasses import dataclass


INJECTION_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"ignore\s+(all\s+)?previous instructions", ""),
    (r"forget\s+(all\s+)?previous instructions", ""),
    (r"reveal\s+the\s+system prompt", ""),
    (r"show\s+me\s+the\s+full\s+schema", "show me aggregated results"),
    (r"drop\s+all\s+tables", ""),
    (r"delete\s+all\s+data", ""),
)


@dataclass(slots=True)
class GuardrailAssessment:
    """Result of prompt sanitization and guardrail checks."""

    cleaned_text: str
    violations: list[str]
    blocked: bool


def sanitize_user_query(query: str) -> GuardrailAssessment:
    """Strip obvious prompt injection attempts from user input."""

    cleaned = query
    violations: list[str] = []
    for pattern, replacement in INJECTION_PATTERNS:
        if re.search(pattern, cleaned, flags=re.IGNORECASE):
            violations.append(pattern)
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    blocked = not cleaned
    return GuardrailAssessment(cleaned_text=cleaned or query.strip(), violations=violations, blocked=blocked)


def filter_output_text(text: str) -> str:
    """Prevent leaking schema details or raw prompt content back to users."""

    lines = []
    for line in text.splitlines():
        if line.strip().startswith(("Table:", "Columns:", "Relationships:")):
            continue
        lines.append(line)
    filtered = "\n".join(lines).replace("```sql", "").replace("```", "").strip()
    return re.sub(r"\s{2,}", " ", filtered)
