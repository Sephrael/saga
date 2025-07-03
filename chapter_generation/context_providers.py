# chapter_generation/context_providers.py
"""Context provider classes for assembling chapter context."""

from __future__ import annotations

import json
from typing import Any

import structlog
from config import settings
from core.llm_interface import count_tokens, llm_service, truncate_text_by_tokens

from models.agent_models import ChapterEndState

from .context_models import ContextChunk, ContextRequest, ProviderSettings

logger = structlog.get_logger(__name__)


class ContextProvider:
    """Base interface for all context providers."""

    source: str = "base"

    async def get_context(
        self, request: ContextRequest, provider_settings: ProviderSettings | None = None
    ) -> ContextChunk:
        """Return context for the given request."""
        raise NotImplementedError


class SemanticHistoryProvider(ContextProvider):
    """Fetch context from previous chapters via vector search."""

    source = "semantic_history"

    def __init__(
        self,
        chapter_repo: Any | None = None,
        llm_service_instance: Any | None = None,
    ) -> None:
        from core.llm_interface import llm_service as default_llm_service
        from data_access import chapter_repository as default_repo

        self.chapter_queries = chapter_repo or default_repo
        self.llm_service = llm_service_instance or default_llm_service

    async def get_context(
        self, request: ContextRequest, provider_settings: ProviderSettings | None = None
    ) -> ContextChunk:
        try:
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
            recent = await self.chapter_queries.get_chapters_data_from_db(
                start, request.chapter_number
            )
            for chap in recent:
                num = chap.get("number")
                if num is None:
                    continue
                content = chap.get("summary") or chap.get("text", "")
                lines.append(
                    f"[Immediate Context from Chapter {num}]:\n{content}\n---\n"
                )

            if similar:
                for item in sorted(
                    similar,
                    key=lambda x: x.get("score", 0.0)
                    * (
                        0.95
                        ** (request.chapter_number - int(x.get("chapter_number", 0)))
                    ),
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
            return ContextChunk(
                text=text, tokens=tokens, provenance={}, source=self.source
            )
        except Exception as exc:  # pragma: no cover - log and return empty
            logger.error("SemanticHistoryProvider failed", error=exc, exc_info=True)
            return ContextChunk(text="", tokens=0, provenance={}, source=self.source)


class KGFactProvider(ContextProvider):
    """Retrieve key facts from the knowledge graph."""

    source = "kg_facts"

    async def get_context(
        self, request: ContextRequest, provider_settings: ProviderSettings | None = None
    ) -> ContextChunk:
        from .context_kg_utils import get_reliable_kg_facts_for_drafting_prompt

        text = await get_reliable_kg_facts_for_drafting_prompt(
            request.plot_outline, request.chapter_number, request.chapter_plan
        )
        tokens = count_tokens(text, "dummy")
        return ContextChunk(text=text, tokens=tokens, provenance={}, source=self.source)


class KGReasoningProvider(ContextProvider):
    """Provide reasoning constraints derived from the KG."""

    source = "kg_reasoning"

    async def get_context(
        self, request: ContextRequest, provider_settings: ProviderSettings | None = None
    ) -> ContextChunk:
        from .context_kg_utils import get_kg_reasoning_guidance_for_prompt

        text = await get_kg_reasoning_guidance_for_prompt(
            request.plot_outline, request.chapter_number, request.chapter_plan
        )
        tokens = count_tokens(text, "dummy")
        return ContextChunk(text=text, tokens=tokens, provenance={}, source=self.source)


class CanonProvider(ContextProvider):
    """Provide canonical truths to avoid contradictions."""

    source = "canon"

    async def get_context(
        self, request: ContextRequest, provider_settings: ProviderSettings | None = None
    ) -> ContextChunk:
        from .context_kg_utils import get_canonical_truths_from_kg

        lines = ["**CANONICAL TRUTHS (DO NOT CONTRADICT):**"]
        llm_used = False
        try:
            # Pass chapter_number as chapter_limit
            records = await get_canonical_truths_from_kg(
                chapter_limit=request.chapter_number
            )
        except Exception as exc:  # pragma: no cover - log and fall back
            logger.error(
                "Failed to get canonical truths for chapter %s: %s",
                request.chapter_number,
                exc,
                exc_info=True,
            )
            records = []

        for line in records:
            if line:
                lines.append(line)

        if len(lines) == 1:
            prompt = (
                "List three short canonical truths about the story based on this "
                f"plot outline:\n{json.dumps(request.plot_outline)}"
            )
            fallback, _ = await llm_service.async_call_llm(
                model_name=settings.SMALL_MODEL,
                prompt=prompt,
                temperature=settings.TEMPERATURE_SUMMARY,
                max_tokens=settings.MAX_SUMMARY_TOKENS,
                allow_fallback=True,
            )
            llm_used = True
            if fallback.strip():
                lines.append(fallback.strip())

        text = "\n".join(lines)
        tokens = count_tokens(text, "dummy")
        return ContextChunk(
            text=text,
            tokens=tokens,
            provenance={},
            source=self.source,
            from_llm_fill=llm_used,
        )


class PlanProvider(ContextProvider):
    """Provide the chapter scene plan."""

    source = "scene_plan"

    async def get_context(
        self, request: ContextRequest, provider_settings: ProviderSettings | None = None
    ) -> ContextChunk:
        """Return scene plan lines, calling the LLM for unresolved entities."""
        plan = request.chapter_plan or []
        lines: list[str] = []
        provenance: dict[str, Any] = {}
        llm_used = False
        for scene in plan:
            summary = scene.get("summary")
            if summary:
                lines.append(f"- {summary}")

        if not lines:
            prompt = (
                "Provide a short three bullet scene outline for the next chapter."
                f"\nPlot focus: {request.plot_focus}\nPlot outline: {json.dumps(request.plot_outline)}"
            )
            fallback, _ = await llm_service.async_call_llm(
                model_name=settings.SMALL_MODEL,
                prompt=prompt,
                temperature=settings.TEMPERATURE_PLANNING,
                max_tokens=settings.MAX_SUMMARY_TOKENS,
                allow_fallback=True,
            )
            llm_used = True
            for line in fallback.splitlines():
                cleaned = line.strip(" -*")
                if cleaned:
                    lines.append(f"- {cleaned}")

        unresolved: list[str] = []
        if request.agent_hints:
            unresolved = request.agent_hints.get("unresolved_entities", []) or []
        if unresolved:
            descs: dict[str, str] = {}
            for entity in unresolved:
                prompt = (
                    "Provide a one sentence description for the entity "
                    f"'{entity}' in this story."
                )
                description, _ = await llm_service.async_call_llm(
                    model_name=settings.SMALL_MODEL,
                    prompt=prompt,
                    temperature=settings.TEMPERATURE_SUMMARY,
                    max_tokens=settings.MAX_SUMMARY_TOKENS,
                    allow_fallback=True,
                )
                llm_used = True
                cleaned = description.strip().replace("\n", " ")
                if cleaned:
                    lines.append(f"- {entity}: {cleaned}")
                    descs[entity] = cleaned
            if descs:
                provenance["llm_fill_ins"] = descs

        text = "\n".join(lines)
        tokens = count_tokens(text, "dummy")
        return ContextChunk(
            text=text,
            tokens=tokens,
            provenance=provenance,
            source=self.source,
            from_llm_fill=llm_used,
        )


class UserNoteProvider(ContextProvider):
    """Include user provided notes if any."""

    source = "user_notes"

    async def get_context(
        self, request: ContextRequest, provider_settings: ProviderSettings | None = None
    ) -> ContextChunk:
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

    async def get_context(
        self, request: ContextRequest, provider_settings: ProviderSettings | None = None
    ) -> ContextChunk:
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

    async def get_context(
        self, request: ContextRequest, provider_settings: ProviderSettings | None = None
    ) -> ContextChunk:
        world = await self.world_queries.get_world_building_from_db(
            chapter_limit=request.chapter_number
        )
        lines: list[str] = []
        for category, items in world.items():
            if category.startswith("_"):
                continue
            for name in items:
                lines.append(f"- {name}")
        text = "\n".join(lines)
        if provider_settings and provider_settings.max_tokens:
            text = truncate_text_by_tokens(text, "dummy", provider_settings.max_tokens)
        tokens = count_tokens(text, "dummy")
        return ContextChunk(text=text, tokens=tokens, provenance={}, source=self.source)


class StateContextProvider(ContextProvider):
    """Provide the exact state at the end of the previous chapter."""

    source = "chapter_end_state"

    def __init__(self, queries_module: Any | None = None) -> None:
        from data_access import chapter_repository as default_repo

        self.chapter_queries = queries_module or default_repo

    async def get_context(
        self, request: ContextRequest, provider_settings: ProviderSettings | None = None
    ) -> ContextChunk:
        prev_chapter = request.chapter_number - 1
        state: ChapterEndState | None = None
        if prev_chapter <= 0:
            if request.agent_hints and request.agent_hints.get(
                "chapter_zero_end_state"
            ):
                hint = request.agent_hints["chapter_zero_end_state"]
                if isinstance(hint, ChapterEndState):
                    state = hint
                else:
                    try:
                        state = ChapterEndState.model_validate_json(hint)
                    except Exception:
                        logger.error(
                            "Failed to parse chapter 0 state from hints", exc_info=True
                        )
            if state is None:
                return ContextChunk(
                    text="", tokens=0, provenance={}, source=self.source
                )
            prev_chapter = 0
        else:
            try:
                data = await self.chapter_queries.get_chapter_data_from_db(prev_chapter)
            except Exception as exc:  # pragma: no cover - log and return empty
                logger.error(
                    "StateContextProvider failed to load chapter",
                    chapter=prev_chapter,
                    error=exc,
                    exc_info=True,
                )
                return ContextChunk(
                    text="", tokens=0, provenance={}, source=self.source
                )
            if not data or not data.get("end_state_json"):
                return ContextChunk(
                    text="", tokens=0, provenance={}, source=self.source
                )
            try:
                state = ChapterEndState.model_validate_json(data["end_state_json"])
            except Exception:
                logger.error(
                    "Failed to parse end state JSON for chapter %s",
                    prev_chapter,
                    exc_info=True,
                )
                return ContextChunk(
                    text="", tokens=0, provenance={}, source=self.source
                )

        lines = [
            "**CRITICAL STATE CONTINUITY - DO NOT CONTRADICT:**",
            f"At the end of Chapter {prev_chapter}:",
        ]
        for char in state.character_states:
            detail = f"- {char.name} was in {char.location} and {char.status}."
            if char.immediate_goal:
                detail += f" Immediate goal: {char.immediate_goal}."
            lines.append(detail)
        if state.key_world_changes:
            lines.append("Key world changes:")
            for loc, change in state.key_world_changes.items():
                lines.append(f"- {loc}: {change}")
        if state.unresolved_cliffhanger:
            lines.append(
                f"The unresolved cliffhanger is: {state.unresolved_cliffhanger}"
            )

        text = "\n".join(lines)
        tokens = count_tokens(text, "dummy")
        return ContextChunk(text=text, tokens=tokens, provenance={}, source=self.source)


class PlotFocusProvider(ContextProvider):
    """Provide the current plot focus text."""

    source = "plot_focus"

    async def get_context(
        self, request: ContextRequest, provider_settings: ProviderSettings | None = None
    ) -> ContextChunk:
        text = request.plot_focus or ""
        if provider_settings and provider_settings.max_tokens:
            text = truncate_text_by_tokens(text, "dummy", provider_settings.max_tokens)
        tokens = count_tokens(text, "dummy")
        return ContextChunk(text=text, tokens=tokens, provenance={}, source=self.source)


class UpcomingPlotPointsProvider(ContextProvider):
    """Provide a list of upcoming plot points."""

    source = "upcoming_plot_points"

    async def get_context(
        self, request: ContextRequest, provider_settings: ProviderSettings | None = None
    ) -> ContextChunk:
        points = request.plot_outline.get("plot_points", [])
        upcoming: list[str] = []
        if isinstance(points, list) and request.chapter_number < len(points):
            upcoming = points[request.chapter_number : request.chapter_number + 3]
        lines = [f"- {p}" for p in upcoming]
        text = "\n".join(lines)
        if provider_settings and provider_settings.max_tokens:
            text = truncate_text_by_tokens(text, "dummy", provider_settings.max_tokens)
        tokens = count_tokens(text, "dummy")
        return ContextChunk(text=text, tokens=tokens, provenance={}, source=self.source)
