"""Gather required planning and context before drafting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from models import SceneDetail

if TYPE_CHECKING:  # pragma: no cover - type hint import
    from orchestration.nana_orchestrator import NANA_Orchestrator


@dataclass
class PrerequisiteData:
    """Data required before drafting a chapter."""

    plot_point_focus: str | None
    plot_point_index: int
    chapter_plan: list[SceneDetail] | None
    hybrid_context_for_draft: str | None


class PrerequisitesService:
    """Service for preparing chapter prerequisites."""

    def __init__(self, orchestrator: NANA_Orchestrator) -> None:
        self.orchestrator = orchestrator

    async def prepare(self, chapter_number: int) -> PrerequisiteData:
        return await self.orchestrator._prepare_chapter_prerequisites(chapter_number)
