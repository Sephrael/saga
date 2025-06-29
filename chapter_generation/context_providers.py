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
        plot_points = request.plot_outline.get("plot_points", [])
        plot_focus = None
        if isinstance(plot_points, list) and 0 < request.chapter_number <= len(
            plot_points
        ):
            plot_focus = plot_points[request.chapter_number - 1]
        query_text = (
            str(plot_focus)
            if plot_focus is not None
            else f"Narrative context relevant to events leading up to chapter {request.chapter_number}."
        )
        embedding = await self.llm_service.async_get_embedding(query_text)
        similar = await self.chapter_queries.find_similar_chapters_in_db(
            embedding,
            5,
            request.chapter_number,
        )

        lines: list[str] = []
        start = max(1, request.chapter_number - 2)
        for i in range(start, request.chapter_number):
            chap = await self.chapter_queries.get_chapter_data_from_db(i)
            if chap:
                content = chap.get("summary") or chap.get("text", "")
                lines.append(f"[Immediate Context from Chapter {i}]:\n{content}\n---\n")

        if similar:
            for item in sorted(
                similar,
                key=lambda x: x.get("score", 0.0)
                * (0.95 ** (request.chapter_number - int(x.get("chapter_number", 0)))),
                reverse=True,
            ):
                num = item.get("chapter_number")
                content = item.get("summary") or item.get("text", "")
                if num and content:
                    score = item.get("score", 0)
                    lines.append(
                        f"[Semantic Context from Chapter {num} (Similarity: {score})]:\n{content}\n---\n"
                    )

        text = "".join(lines).strip()
        tokens = count_tokens(text, "dummy")
        return ContextChunk(text=text, tokens=tokens, provenance={}, source=self.source)


class KGFactProvider(ContextProvider):
    """Retrieve key facts from the knowledge graph."""

    source = "kg_facts"

    async def get_context(self, request: ContextRequest) -> ContextChunk:
        from .context_kg_utils import get_reliable_kg_facts_for_drafting_prompt

        text = await get_reliable_kg_facts_for_drafting_prompt(
            request.plot_outline, request.chapter_number, request.chapter_plan
        )
        tokens = count_tokens(text, "dummy")
        return ContextChunk(text=text, tokens=tokens, provenance={}, source=self.source)


class KGReasoningProvider(ContextProvider):
    """Provide reasoning constraints derived from the KG."""

    source = "kg_reasoning"

    async def get_context(self, request: ContextRequest) -> ContextChunk:
        from .context_kg_utils import get_kg_reasoning_guidance_for_prompt

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


class CharacterStateProvider(ContextProvider):
    """Provide character state summaries."""

    source = "character_state"

    def __init__(self, queries_module: Any | None = None) -> None:
        from data_access import character_queries as default_character_queries

        self.character_queries = queries_module or default_character_queries

    async def get_context(self, request: ContextRequest) -> ContextChunk:
        profiles = await self.character_queries.get_character_profiles_from_db()
        lines = [f"- {name}" for name in sorted(profiles.keys())]
        text = "\n".join(lines)
        tokens = count_tokens(text, "dummy")
        return ContextChunk(text=text, tokens=tokens, provenance={}, source=self.source)


class WorldStateProvider(ContextProvider):
    """Provide world state summaries."""

    source = "world_state"

    def __init__(self, queries_module: Any | None = None) -> None:
        from data_access import world_queries as default_world_queries

        self.world_queries = queries_module or default_world_queries

    async def get_context(self, request: ContextRequest) -> ContextChunk:
        world = await self.world_queries.get_world_building_from_db()
        lines: list[str] = []
        for category, items in world.items():
            if category.startswith("_"):
                continue
            for name in items:
                lines.append(f"- {name}")
        text = "\n".join(lines)
        tokens = count_tokens(text, "dummy")
        return ContextChunk(text=text, tokens=tokens, provenance={}, source=self.source)
