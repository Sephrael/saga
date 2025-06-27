"""Gather required planning and context before drafting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from models import SceneDetail

if TYPE_CHECKING:  # pragma: no cover - type hint import
    from orchestration.nana_orchestrator import NANA_Orchestrator
    from storage.file_manager import FileManager


@dataclass
class PrerequisiteData:
    """Data required before drafting a chapter."""

    plot_point_focus: str | None
    plot_point_index: int
    chapter_plan: list[SceneDetail] | None
    hybrid_context_for_draft: str | None


class PrerequisitesService:
    """Service for preparing chapter prerequisites."""

    def __init__(
        self, orchestrator: NANA_Orchestrator, file_manager: FileManager
    ) -> None:
        self.orchestrator = orchestrator
        self.file_manager = file_manager

    async def prepare(self, chapter_number: int) -> PrerequisiteData:
        return await self.orchestrator._prepare_chapter_prerequisites(chapter_number)
