"""Schema introspection and embedding pipeline."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from sqlalchemy import inspect
from sqlalchemy.engine import Engine

from app.embeddings.store import SchemaDocument, SchemaVectorStore

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TableMetadata:
    """Structured metadata captured for a single table."""

    table_name: str
    columns: list[dict[str, str | bool | None]]
    relationships: list[dict[str, str]]


class SchemaIndexer:
    """Introspect database schemas and index them into the vector store."""

    def __init__(self, engine: Engine, vector_store: SchemaVectorStore) -> None:
        """Initialize the schema indexer."""

        self.engine = engine
        self.vector_store = vector_store

    def build_documents(self) -> list[SchemaDocument]:
        """Extract schema metadata and format it for embedding."""

        inspector = inspect(self.engine)
        documents: list[SchemaDocument] = []

        for table_name in sorted(inspector.get_table_names()):
            pk_columns = set((inspector.get_pk_constraint(table_name) or {}).get("constrained_columns") or [])
            foreign_keys = inspector.get_foreign_keys(table_name) or []
            fk_lookup: dict[str, str] = {}
            relationships: list[dict[str, str]] = []

            for foreign_key in foreign_keys:
                referred_table = foreign_key.get("referred_table")
                constrained_columns = foreign_key.get("constrained_columns") or []
                referred_columns = foreign_key.get("referred_columns") or []
                for local_column, remote_column in zip(constrained_columns, referred_columns):
                    target = f"{referred_table}.{remote_column}"
                    fk_lookup[local_column] = target
                    relationships.append({"from": local_column, "to": target})

            column_metadata: list[dict[str, str | bool | None]] = []
            column_descriptions: list[str] = []
            for column in inspector.get_columns(table_name):
                column_name = str(column["name"])
                column_type = str(column["type"]).replace(",", "")
                foreign_key_target = fk_lookup.get(column_name)
                is_primary_key = column_name in pk_columns

                column_metadata.append(
                    {
                        "name": column_name,
                        "type": column_type,
                        "is_primary_key": is_primary_key,
                        "foreign_key_to": foreign_key_target,
                    },
                )

                descriptors = [column_type.lower()]
                if is_primary_key:
                    descriptors.append("primary key")
                if foreign_key_target:
                    descriptors.append(f"foreign key to {foreign_key_target}")
                column_descriptions.append(f"{column_name} ({', '.join(descriptors)})")

            description = (
                f"Table {table_name} stores {table_name.replace('_', ' ')} data. "
                f"Columns: {', '.join(column_descriptions)}."
            )
            if relationships:
                relationship_text = ", ".join(f"{item['from']} -> {item['to']}" for item in relationships)
                description = f"{description} Relationships: {relationship_text}."

            metadata = {
                "table_name": table_name,
                "columns_json": json.dumps(column_metadata),
                "relationships_json": json.dumps(relationships),
            }
            documents.append(SchemaDocument(table_name=table_name, content=description, metadata=metadata))

        logger.info(
            "schema_documents_built",
            extra={"event_data": {"table_count": len(documents)}},
        )
        return documents

    def index_schema(self) -> int:
        """Index the current database schema into the vector store."""

        documents = self.build_documents()
        self.vector_store.clear()
        self.vector_store.upsert_documents(documents)
        logger.info(
            "schema_indexing_completed",
            extra={"event_data": {"table_count": len(documents)}},
        )
        return len(documents)
