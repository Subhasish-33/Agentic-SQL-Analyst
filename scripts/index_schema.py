"""Index database schema metadata into the configured vector store."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from app.config import configure_logging, get_settings
from app.db.connection import create_sync_engine_from_settings
from app.db.schema_indexer import SchemaIndexer
from app.embeddings.store import SchemaVectorStore


def main() -> None:
    """Run schema introspection and persist schema embeddings."""

    settings = get_settings()
    configure_logging(settings.LOG_LEVEL)
    engine = create_sync_engine_from_settings(settings)
    vector_store = SchemaVectorStore(settings)
    indexer = SchemaIndexer(engine, vector_store)
    indexed_tables = indexer.index_schema()
    engine.dispose()
    print(f"Indexed {indexed_tables} tables into {settings.VECTOR_DB_PATH}.")


if __name__ == "__main__":
    main()
