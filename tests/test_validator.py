"""Unit tests for the SQL safety validator."""

from __future__ import annotations

from app.config import Settings
from app.safety.validator import QuerySafetyValidator

SCHEMA_CONTEXT = """Table: users
Columns: id INTEGER PRIMARY KEY, name TEXT, email TEXT, created_at TIMESTAMP
Relationships: none

Table: orders
Columns: id INTEGER PRIMARY KEY, user_id INTEGER FOREIGN KEY -> users.id, total NUMERIC, created_at TIMESTAMP
Relationships: user_id -> users.id
"""


def build_validator() -> QuerySafetyValidator:
    """Create a validator instance for tests."""

    settings = Settings(DATABASE_URL="sqlite+aiosqlite:///./test.db", OPENAI_API_KEY="test-key", LARGE_TABLES=["orders"])
    return QuerySafetyValidator(settings=settings, large_table_names=["orders"])


def test_drop_table_is_blocked() -> None:
    """Dangerous DDL statements should fail the regex safety layer."""

    validator = build_validator()
    result = validator.validate("DROP TABLE users;", SCHEMA_CONTEXT)
    assert not result.valid
    assert result.error == "Blocked non-read-only SQL operation detected."


def test_valid_select_passes() -> None:
    """A safe read-only SELECT should pass validation."""

    validator = build_validator()
    result = validator.validate("SELECT id, name FROM users WHERE id = 1;", SCHEMA_CONTEXT)
    assert result.valid
    assert result.error is None
    assert result.normalized_sql.lower().startswith("select")


def test_unknown_table_fails_ast_validation() -> None:
    """Queries referencing tables outside the retrieved schema should be rejected."""

    validator = build_validator()
    result = validator.validate("SELECT * FROM payments WHERE id = 1;", SCHEMA_CONTEXT)
    assert not result.valid
    assert "payments" in (result.error or "")
