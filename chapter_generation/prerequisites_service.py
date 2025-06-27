"""Gather required planning and context before drafting."""

from __future__ import annotations

from typing import TYPE_CHECKING

from models import SceneDetail

if TYPE_CHECKING:  # pragma: no cover - type hint import
    from orchestration.nana_orchestrator import NANA_Orchestrator


class PrerequisitesService:
    """Service for preparing chapter prerequisites."""

    def __init__(self, orchestrator: NANA_Orchestrator) -> None:
        self.orchestrator = orchestrator

    async def prepare(
        self, chapter_number: int
    ) -> tuple[str | None, int, list[SceneDetail] | None, str | None]:
        return await self.orchestrator._prepare_chapter_prerequisites(chapter_number)
