"""Caching backends for schema, embeddings, and query results."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
from dataclasses import dataclass
from time import time
from typing import Any

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CacheEntry:
    """An in-memory cache entry with expiration tracking."""

    value: Any
    expires_at: float | None


class InMemoryCache:
    """Simple TTL cache used as the default fallback backend."""

    def __init__(self) -> None:
        """Initialize the backing dictionary and async lock."""

        self._store: dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Return whether a cache entry has expired."""

        return entry.expires_at is not None and entry.expires_at <= time()

    async def get(self, key: str) -> Any | None:
        """Resolve a cached value asynchronously."""

        async with self._lock:
            entry = self._store.get(key)
            if not entry:
                return None
            if self._is_expired(entry):
                self._store.pop(key, None)
                return None
            return copy.deepcopy(entry.value)

    async def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        """Store a value asynchronously."""

        expires_at = time() + ttl_seconds if ttl_seconds else None
        async with self._lock:
            self._store[key] = CacheEntry(value=copy.deepcopy(value), expires_at=expires_at)

    def get_sync(self, key: str) -> Any | None:
        """Resolve a cached value synchronously."""

        entry = self._store.get(key)
        if not entry:
            return None
        if self._is_expired(entry):
            self._store.pop(key, None)
            return None
        return copy.deepcopy(entry.value)

    def set_sync(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        """Store a value synchronously."""

        expires_at = time() + ttl_seconds if ttl_seconds else None
        self._store[key] = CacheEntry(value=copy.deepcopy(value), expires_at=expires_at)


class CacheService:
    """Unified cache facade with optional Redis and in-memory fallback."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the cache backend."""

        self.settings = settings or get_settings()
        self.memory = InMemoryCache()
        self.redis = None

        if self.settings.REDIS_URL:
            try:
                from redis.asyncio import Redis

                self.redis = Redis.from_url(self.settings.REDIS_URL, encoding="utf-8", decode_responses=True)
            except Exception as exc:
                logger.warning(
                    "cache_redis_unavailable",
                    extra={"event_data": {"error": str(exc), "backend": "memory"}},
                )

    def build_key(self, namespace: str, raw_key: str) -> str:
        """Create a deterministic cache key for potentially long inputs."""

        digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
        return f"{namespace}:{digest}"

    async def get_json(self, namespace: str, raw_key: str) -> Any | None:
        """Retrieve a JSON-serializable value from cache."""

        cache_key = self.build_key(namespace, raw_key)
        local_value = await self.memory.get(cache_key)
        if local_value is not None:
            return local_value

        if self.redis is not None:
            raw_value = await self.redis.get(cache_key)
            if raw_value is not None:
                parsed = json.loads(raw_value)
                await self.memory.set(cache_key, parsed, ttl_seconds=60)
                return parsed
        return None

    async def set_json(self, namespace: str, raw_key: str, value: Any, ttl_seconds: int) -> None:
        """Store a JSON-serializable value in cache."""

        cache_key = self.build_key(namespace, raw_key)
        await self.memory.set(cache_key, value, ttl_seconds=ttl_seconds)
        if self.redis is not None:
            await self.redis.set(cache_key, json.dumps(value), ex=ttl_seconds)

    def get_local_json(self, namespace: str, raw_key: str) -> Any | None:
        """Retrieve a value synchronously from the in-memory cache."""

        cache_key = self.build_key(namespace, raw_key)
        return self.memory.get_sync(cache_key)

    def set_local_json(self, namespace: str, raw_key: str, value: Any, ttl_seconds: int) -> None:
        """Store a value synchronously in the in-memory cache."""

        cache_key = self.build_key(namespace, raw_key)
        self.memory.set_sync(cache_key, value, ttl_seconds=ttl_seconds)

    async def close(self) -> None:
        """Dispose of any external cache connections."""

        if self.redis is not None:
            await self.redis.aclose()
