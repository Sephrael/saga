# chapter_generation/revision_service.py
"""Service for revising drafted chapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from processing.revision_manager import RevisionManager

from models import CharacterProfile, EvaluationResult, SceneDetail, WorldItem


@dataclass
class RevisionResult:
    """Result of the revision loop."""

    text: str | None
    raw_llm_output: str | None
    is_flawed_source: bool
    patched_spans: list[tuple[int, int]]


class RevisionService:
    """Apply revision cycles using :class:`RevisionManager`."""

    def __init__(self, manager: RevisionManager | None = None) -> None:
        self.manager = manager or RevisionManager()

    async def revise(
        self,
        plot_outline: dict[str, Any],
        character_profiles: dict[str, CharacterProfile],
        world_building: dict[str, dict[str, WorldItem]],
        original_text: str,
        chapter_number: int,
        evaluation_result: EvaluationResult,
        hybrid_context_for_revision: str,
        chapter_plan: list[SceneDetail] | None,
    ) -> RevisionResult:
        """Return a revised draft based on evaluation feedback."""

        revised, _ = await self.manager.revise_chapter(
            plot_outline,
            character_profiles,
            world_building,
            original_text,
            chapter_number,
            evaluation_result,
            hybrid_context_for_revision,
            chapter_plan,
        )
        if revised is None:
            return RevisionResult(None, None, False, [])
        text, raw, spans = revised
        return RevisionResult(text, raw, False, spans)
