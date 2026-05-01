"""Read-only SQL validation and safety rules."""

from __future__ import annotations

import logging
import re
from typing import Iterable, Sequence

from pydantic import BaseModel
from sqlglot import exp, parse_one
from sqlglot.errors import ParseError

from app.config import Settings, get_settings
from app.services.schema_catalog import SchemaCatalogService

logger = logging.getLogger(__name__)


class ValidationResult(BaseModel):
    """Structured SQL validation outcome."""

    valid: bool
    error: str | None = None
    normalized_sql: str = ""


class QuerySafetyValidator:
    """Apply regex and AST checks before any SQL is executed."""

    def __init__(
        self,
        settings: Settings | None = None,
        large_table_names: Sequence[str] | None = None,
        schema_catalog: SchemaCatalogService | None = None,
    ) -> None:
        """Initialize the validator with configurable rules."""

        self.settings = settings or get_settings()
        self.large_table_names = set(large_table_names or self.settings.LARGE_TABLES)
        self.schema_catalog = schema_catalog
        blocked_pattern = "|".join(re.escape(keyword) for keyword in self.settings.BLOCKED_KEYWORDS)
        self.blocked_keywords_regex = re.compile(rf"\b(?:{blocked_pattern})\b", flags=re.IGNORECASE)
        self.union_injection_regex = re.compile(r"\bUNION\b\s+(?:ALL\s+)?SELECT\b", flags=re.IGNORECASE)

    def validate(self, sql: str, schema_context: str) -> ValidationResult:
        """Validate SQL against safety rules and known schema context."""

        cleaned_sql = sql.strip().strip(";")
        if not cleaned_sql:
            return ValidationResult(valid=False, error="The generated SQL query was empty.")

        regex_error = self._check_regex_safety(cleaned_sql)
        if regex_error:
            logger.warning(
                "sql_validation_failed_regex",
                extra={"event_data": {"sql": cleaned_sql, "error": regex_error}},
            )
            return ValidationResult(valid=False, error=regex_error, normalized_sql=cleaned_sql)

        schema_catalog = self._parse_schema_context(schema_context)
        ast_result = self._validate_ast(cleaned_sql, schema_catalog)
        if not ast_result.valid:
            logger.warning(
                "sql_validation_failed_ast",
                extra={"event_data": {"sql": cleaned_sql, "error": ast_result.error}},
            )
            return ast_result

        logger.info(
            "sql_validation_completed",
            extra={"event_data": {"sql": cleaned_sql, "tables": sorted(schema_catalog)}},
        )
        return ast_result

    def _check_regex_safety(self, sql: str) -> str | None:
        """Apply fast regex checks for dangerous or suspicious patterns."""

        if self.blocked_keywords_regex.search(sql):
            return "Blocked non-read-only SQL operation detected."
        if self.union_injection_regex.search(sql):
            return "Blocked UNION-based injection pattern detected."
        return None

    def _parse_schema_context(self, schema_context: str) -> dict[str, set[str]]:
        """Convert formatted schema context text into a table-to-columns map."""

        schema_catalog: dict[str, set[str]] = {}
        blocks = [block.strip() for block in schema_context.split("\n\n") if block.strip()]
        for block in blocks:
            table_name = ""
            columns: set[str] = set()
            for line in block.splitlines():
                if line.startswith("Table:"):
                    table_name = line.split(":", 1)[1].strip()
                elif line.startswith("Columns:"):
                    raw_columns = line.split(":", 1)[1].strip()
                    for item in raw_columns.split(","):
                        column_name = item.strip().split(" ", 1)[0]
                        if column_name:
                            columns.add(column_name)
            if table_name:
                schema_catalog[table_name] = columns
        if not schema_catalog and self.schema_catalog is not None:
            for table_name, metadata in self.schema_catalog.get_catalog().items():
                schema_catalog[table_name] = set(metadata.get("columns", {}))
        return schema_catalog

    def _validate_ast(self, sql: str, schema_catalog: dict[str, set[str]]) -> ValidationResult:
        """Parse SQL to an AST and validate tables, columns, and query shape."""

        try:
            parsed = parse_one(sql, read=self.settings.sql_dialect)
        except ParseError as exc:
            return ValidationResult(valid=False, error=f"SQL parsing failed: {exc}", normalized_sql=sql)

        if not isinstance(parsed, exp.Select):
            return ValidationResult(valid=False, error="Only SELECT statements are allowed.", normalized_sql=sql)

        alias_map: dict[str, str] = {}
        referenced_tables: set[str] = set()
        for table in parsed.find_all(exp.Table):
            real_name = table.name
            if real_name not in schema_catalog:
                return ValidationResult(
                    valid=False,
                    error=f"Query references table '{real_name}' which is not in the retrieved schema context.",
                    normalized_sql=sql,
                )
            referenced_tables.add(real_name)
            alias_map[real_name] = real_name
            alias_map[table.alias_or_name] = real_name
            if self.schema_catalog is not None and not self.schema_catalog.has_table(real_name):
                return ValidationResult(
                    valid=False,
                    error=f"Query references unknown physical table '{real_name}'.",
                    normalized_sql=sql,
                )

        if referenced_tables & self.large_table_names and parsed.args.get("where") is None:
            blocked_tables = sorted(referenced_tables & self.large_table_names)
            return ValidationResult(
                valid=False,
                error=f"Queries against large tables require a WHERE clause: {', '.join(blocked_tables)}.",
                normalized_sql=sql,
            )

        select_aliases = {
            projection.alias
            for projection in parsed.expressions
            if getattr(projection, "alias", None)
        }
        for column in self._iter_columns(parsed):
            if column.name == "*":
                continue

            if not column.table and column.name in select_aliases:
                continue

            if column.table:
                resolved_table = alias_map.get(column.table)
                if not resolved_table:
                    return ValidationResult(
                        valid=False,
                        error=f"Query references unknown table alias '{column.table}'.",
                        normalized_sql=sql,
                    )
                if column.name not in schema_catalog.get(resolved_table, set()):
                    return ValidationResult(
                        valid=False,
                        error=f"Column '{resolved_table}.{column.name}' is not present in the retrieved schema context.",
                        normalized_sql=sql,
                    )
                if self.schema_catalog is not None and not self.schema_catalog.has_column(resolved_table, column.name):
                    return ValidationResult(
                        valid=False,
                        error=f"Column '{resolved_table}.{column.name}' does not exist in the actual database schema.",
                        normalized_sql=sql,
                    )
                continue

            if not any(column.name in schema_catalog.get(table_name, set()) for table_name in referenced_tables):
                return ValidationResult(
                    valid=False,
                    error=f"Column '{column.name}' is not present in the retrieved schema context.",
                    normalized_sql=sql,
                )
            if self.schema_catalog is not None and referenced_tables:
                if not any(self.schema_catalog.has_column(table_name, column.name) for table_name in referenced_tables):
                    return ValidationResult(
                        valid=False,
                        error=f"Column '{column.name}' does not exist in the actual database schema.",
                        normalized_sql=sql,
                    )

        normalized_sql = parsed.sql(dialect=self.settings.sql_dialect)
        return ValidationResult(valid=True, normalized_sql=normalized_sql)

    def _iter_columns(self, parsed: exp.Expression) -> Iterable[exp.Column]:
        """Yield all column expressions from the AST."""

        return parsed.find_all(exp.Column)
