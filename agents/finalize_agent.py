# agents/finalize_agent.py
"""Finalize chapter text and update the knowledge graph."""

import asyncio
from typing import Any, TypedDict

import numpy as np
import structlog
from core.llm_interface import llm_service
from data_access import chapter_repository

from agents.kg_maintainer_agent import KGMaintainerAgent
from models import CharacterProfile, WorldItem

logger = structlog.get_logger(__name__)


class FinalizationResult(TypedDict, total=False):
    summary: str | None
    embedding: np.ndarray | None
    summary_usage: dict[str, int] | None
    kg_usage: dict[str, int] | None


class FinalizeAgent:
    """Handle chapter finalization and KG updates."""

    def __init__(self, kg_agent: KGMaintainerAgent | None = None) -> None:
        self.kg_agent = kg_agent or KGMaintainerAgent()
        logger.info("FinalizeAgent initialized")

    async def _extract_merge_and_persist(
        self,
        plot_outline: dict[str, Any],
        character_profiles: dict[str, CharacterProfile],
        world_building: dict[str, dict[str, WorldItem]],
        chapter_number: int,
        chapter_text: str,
        from_flawed_draft: bool,
        fill_in_context: str | None,
    ) -> dict[str, int] | None:
        return await self.kg_agent.extract_and_merge_knowledge(
            plot_outline,
            character_profiles,
            world_building,
            chapter_number,
            chapter_text,
            is_from_flawed_draft=from_flawed_draft,
            fill_in_context=fill_in_context,
        )

    async def finalize_chapter(
        self,
        plot_outline: dict[str, Any],
        character_profiles: dict[str, CharacterProfile],
        world_building: dict[str, dict[str, WorldItem]],
        chapter_number: int,
        final_text: str,
        raw_llm_output: str | None = None,
        from_flawed_draft: bool = False,
        fill_in_context: str | None = None,
    ) -> FinalizationResult:
        """Finalize a chapter and persist all related updates.

        Args:
            plot_outline: The current plot outline for the novel.
            character_profiles: Known character profiles before this chapter.
            world_building: Known world elements before this chapter.
            chapter_number: The chapter number being finalized.
            final_text: The approved chapter text.
            raw_llm_output: Optional raw draft from the LLM.
            from_flawed_draft: Whether the text came from a flawed draft.
            fill_in_context: Additional context filled in by LLMs for missing
                references.

        Returns:
            A dictionary containing the summary, embedding, and token usage data.
        """
        summary_task = self.kg_agent.summarize_chapter(final_text, chapter_number)
        embedding_task = llm_service.async_get_embedding(final_text)
        kg_task = self._extract_merge_and_persist(
            plot_outline,
            character_profiles,
            world_building,
            chapter_number,
            final_text,
            from_flawed_draft,
            fill_in_context,
        )
        end_state_task = self.kg_agent.generate_chapter_end_state(
            final_text,
            chapter_number,
            fill_in_context=fill_in_context,
        )

        (summary_data, embedding, kg_usage, end_state) = await asyncio.gather(
            summary_task, embedding_task, kg_task, end_state_task
        )
        summary, summary_usage = summary_data

        await chapter_repository.save_chapter_data(
            chapter_number,
            final_text,
            raw_llm_output or "N/A",
            summary,
            embedding,
            from_flawed_draft,
            end_state.model_dump(),
        )

        return {
            "summary": summary,
            "embedding": embedding,
            "summary_usage": summary_usage,
            "kg_usage": kg_usage,
        }

    async def ingest_and_finalize_chunk(
        self,
        plot_outline: dict[str, Any],
        character_profiles: dict[str, CharacterProfile],
        world_building: dict[str, dict[str, WorldItem]],
        chunk_number: int,
        chunk_text: str,
    ) -> FinalizationResult:
        """Finalize an ingested text chunk using the regular pipeline."""
        return await self.finalize_chapter(
            plot_outline,
            character_profiles,
            world_building,
            chunk_number,
            chunk_text,
            raw_llm_output=None,
            from_flawed_draft=False,
        )
