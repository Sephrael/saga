# chapter_generation/context_providers.py
"""Context provider classes for assembling chapter context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog
from core.llm_interface import count_tokens

from models.agent_models import SceneDetail

logger = structlog.get_logger(__name__)


@dataclass
class ContextRequest:
    """Parameters describing the desired context."""

    chapter_number: int
    plot_focus: str | None
    plot_outline: dict[str, Any]
    chapter_plan: list[SceneDetail] | None = None
    agent_hints: dict[str, Any] | None = None


@dataclass
class ContextChunk:
    """A chunk of context returned by a provider."""

    text: str
    tokens: int
    provenance: dict[str, Any]
    source: str


class ContextProvider:
    """Base interface for all context providers."""

    source: str = "base"

    async def get_context(self, request: ContextRequest) -> ContextChunk:
        """Return context for the given request."""
        raise NotImplementedError


class SemanticHistoryProvider(ContextProvider):
    """Fetch context from previous chapters via vector search."""

    source = "semantic_history"

    def __init__(
        self,
        chapter_queries_module: Any | None = None,
        llm_service_instance: Any | None = None,
    ) -> None:
        from core.llm_interface import llm_service as default_llm_service
        from data_access import chapter_queries as default_chapter_queries

        self.chapter_queries = chapter_queries_module or default_chapter_queries
        self.llm_service = llm_service_instance or default_llm_service

    async def get_context(self, request: ContextRequest) -> ContextChunk:
        from chapter_generation.context_service import ContextService

        service = ContextService(self.chapter_queries, self.llm_service)
        text = await service.get_semantic_context(
            request.plot_outline, request.chapter_number
        )
        tokens = count_tokens(text, "dummy")
        return ContextChunk(text=text, tokens=tokens, provenance={}, source=self.source)


class KGFactProvider(ContextProvider):
    """Retrieve key facts from the knowledge graph."""

    source = "kg_facts"

    async def get_context(self, request: ContextRequest) -> ContextChunk:
        from prompt_data_getters import get_reliable_kg_facts_for_drafting_prompt

        text = await get_reliable_kg_facts_for_drafting_prompt(
            request.plot_outline, request.chapter_number, request.chapter_plan
        )
        tokens = count_tokens(text, "dummy")
        return ContextChunk(text=text, tokens=tokens, provenance={}, source=self.source)


class KGReasoningProvider(ContextProvider):
    """Provide reasoning constraints derived from the KG."""

    source = "kg_reasoning"

    async def get_context(self, request: ContextRequest) -> ContextChunk:
        from prompt_data_getters import get_kg_reasoning_guidance_for_prompt

        text = await get_kg_reasoning_guidance_for_prompt(
            request.plot_outline, request.chapter_number, request.chapter_plan
        )
        tokens = count_tokens(text, "dummy")
        return ContextChunk(text=text, tokens=tokens, provenance={}, source=self.source)


class PlanProvider(ContextProvider):
    """Provide the chapter scene plan."""

    source = "scene_plan"

    async def get_context(self, request: ContextRequest) -> ContextChunk:
        plan = request.chapter_plan or []
        lines = []
        for scene in plan:
            summary = scene.get("summary")
            if summary:
                lines.append(f"- {summary}")
        text = "\n".join(lines)
        tokens = count_tokens(text, "dummy")
        return ContextChunk(text=text, tokens=tokens, provenance={}, source=self.source)


class UserNoteProvider(ContextProvider):
    """Include user provided notes if any."""

    source = "user_notes"

    async def get_context(self, request: ContextRequest) -> ContextChunk:
        notes = ""
        if request.agent_hints:
            notes = request.agent_hints.get("user_notes", "")
        tokens = count_tokens(notes, "dummy")
        return ContextChunk(
            text=notes, tokens=tokens, provenance={}, source=self.source
        )
