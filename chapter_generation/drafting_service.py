"""Service for drafting initial chapter text."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type hint import
    from orchestration.nana_orchestrator import NANA_Orchestrator

    from models import SceneDetail


@dataclass
class DraftResult:
    """Initial draft and raw output from the DraftingAgent."""

    text: str | None
    raw_llm_output: str | None


class DraftingService:
    """Handle chapter drafting through the DraftingAgent."""

    def __init__(self, orchestrator: NANA_Orchestrator) -> None:
        self.orchestrator = orchestrator

    async def draft_initial_text(
        self,
        chapter_number: int,
        plot_point_focus: str,
        hybrid_context_for_draft: str,
        chapter_plan: list[SceneDetail] | None,
    ) -> DraftResult:
        result_text, raw_llm = await self.orchestrator._draft_initial_chapter_text(
            chapter_number,
            plot_point_focus,
            hybrid_context_for_draft,
            chapter_plan,
        )
        return DraftResult(text=result_text, raw_llm_output=raw_llm)
