# chapter_generation/finalization_service.py
"""Service for finalizing chapters and persisting results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agents.finalize_agent import FinalizeAgent
from agents.kg_maintainer_agent import KGMaintainerAgent
from kg_maintainer.models import CharacterProfile, WorldItem


@dataclass
class FinalizationServiceResult:
    """Outcome of finalization."""

    text: str | None


class FinalizationService:
    """Finalize chapters using :class:`FinalizeAgent`."""

    def __init__(self, agent: FinalizeAgent | None = None) -> None:
        kg_agent = KGMaintainerAgent()
        self.agent = agent or FinalizeAgent(kg_agent)

    async def finalize(
        self,
        plot_outline: dict[str, Any],
        character_profiles: dict[str, CharacterProfile],
        world_building: dict[str, dict[str, WorldItem]],
        chapter_number: int,
        final_text: str,
        raw_llm_output: str | None,
        from_flawed_draft: bool,
        fill_in_context: str | None,
    ) -> FinalizationServiceResult:
        """Return persisted chapter details."""

        await self.agent.finalize_chapter(
            plot_outline,
            character_profiles,
            world_building,
            chapter_number,
            final_text,
            raw_llm_output,
            from_flawed_draft,
            fill_in_context,
        )
        return FinalizationServiceResult(text=final_text)
