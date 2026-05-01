"""Schema catalog service for real-schema validation and formatting."""

from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy import inspect
from sqlalchemy.engine import Engine

from app.config import Settings, get_settings
from app.db.connection import create_sync_engine_from_settings

logger = logging.getLogger(__name__)


class SchemaCatalogService:
    """Load and cache the real database catalog for validation."""

    def __init__(self, settings: Settings | None = None, engine: Engine | None = None) -> None:
        """Initialize the schema catalog service."""

        self.settings = settings or get_settings()
        self.engine = engine or create_sync_engine_from_settings(self.settings)
        self._catalog: dict[str, dict[str, Any]] | None = None

    def get_catalog(self) -> dict[str, dict[str, Any]]:
        """Return the cached table catalog."""

        if self._catalog is not None:
            return self._catalog

        inspector = inspect(self.engine)
        catalog: dict[str, dict[str, Any]] = {}
        for table_name in inspector.get_table_names():
            columns = {str(column["name"]): str(column["type"]) for column in inspector.get_columns(table_name)}
            pk_columns = set((inspector.get_pk_constraint(table_name) or {}).get("constrained_columns") or [])
            relationships: list[dict[str, str]] = []
            for foreign_key in inspector.get_foreign_keys(table_name) or []:
                referred_table = foreign_key.get("referred_table")
                for local_column, remote_column in zip(
                    foreign_key.get("constrained_columns") or [],
                    foreign_key.get("referred_columns") or [],
                ):
                    relationships.append({"from": local_column, "to": f"{referred_table}.{remote_column}"})

            catalog[table_name] = {
                "columns": columns,
                "primary_keys": sorted(pk_columns),
                "relationships": relationships,
                "relationships_json": json.dumps(relationships),
            }

        self._catalog = catalog
        logger.info(
            "schema_catalog_loaded",
            extra={"event_data": {"table_count": len(catalog)}},
        )
        return catalog

    def has_table(self, table_name: str) -> bool:
        """Return whether a table exists in the catalog."""

        return table_name in self.get_catalog()

    def has_column(self, table_name: str, column_name: str) -> bool:
        """Return whether a column exists for a table."""

        return column_name in self.get_catalog().get(table_name, {}).get("columns", {})

    def format_table_context(self, table_name: str) -> str:
        """Render a table into the prompt-ready schema context format."""

        table = self.get_catalog()[table_name]
        column_text: list[str] = []
        for column_name, column_type in table["columns"].items():
            tokens = [column_name, column_type.upper()]
            if column_name in table["primary_keys"]:
                tokens.append("PRIMARY KEY")
            for relationship in table["relationships"]:
                if relationship["from"] == column_name:
                    tokens.append(f"FOREIGN KEY -> {relationship['to']}")
            column_text.append(" ".join(tokens))

        relationship_text = ", ".join(
            f"{relationship['from']} -> {relationship['to']}" for relationship in table["relationships"]
        ) or "none"
        return f"Table: {table_name}\nColumns: {', '.join(column_text)}\nRelationships: {relationship_text}"
