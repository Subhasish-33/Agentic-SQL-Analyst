"""Prefetch node that parallelizes intent classification and schema retrieval."""

from __future__ import annotations

import asyncio
from typing import Any

from app.agent.nodes.intent_classifier import IntentClassifierNode
from app.agent.nodes.schema_retriever import SchemaRetrieverNode
from app.agent.state import AgentState


class PrefetchContextNode:
    """Run early cheap stages concurrently before the main graph branches."""

    def __init__(
        self,
        intent_classifier: IntentClassifierNode,
        schema_retriever: SchemaRetrieverNode,
    ) -> None:
        """Initialize the prefetch node."""

        self.intent_classifier = intent_classifier
        self.schema_retriever = schema_retriever

    def _merge_dicts(self, left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
        """Merge two shallow dictionaries."""

        merged = dict(left or {})
        merged.update(right or {})
        return merged

    async def __call__(self, state: AgentState) -> AgentState:
        """Kick off intent and schema work concurrently."""

        intent_update, schema_update = await asyncio.gather(
            self.intent_classifier.classify(state),
            self.schema_retriever.retrieve(state),
        )
        merged: AgentState = {}
        merged.update(intent_update)
        merged.update(schema_update)
        merged["latency_breakdown"] = self._merge_dicts(
            intent_update.get("latency_breakdown"),
            schema_update.get("latency_breakdown"),
        )
        merged["token_usage"] = self._merge_dicts(state.get("token_usage"), intent_update.get("token_usage"))
        merged["models_used"] = self._merge_dicts(state.get("models_used"), intent_update.get("models_used"))
        merged["cost_breakdown"] = self._merge_dicts(state.get("cost_breakdown"), intent_update.get("cost_breakdown"))
        merged["cache_hits"] = self._merge_dicts(state.get("cache_hits"), schema_update.get("cache_hits"))
        return merged
