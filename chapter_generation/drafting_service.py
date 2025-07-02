# chapter_generation/drafting_service.py
"""Service for drafting initial chapter text."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from agents.drafting_agent import DraftingAgent

from models import SceneDetail


@dataclass
class DraftResult:
    """Initial draft and raw output from the DraftingAgent."""

    text: str | None
    raw_llm_output: str | None

    def __iter__(self) -> Iterable[str | None]:
        """Allow tuple-style unpacking of the draft result."""
        yield self.text
        yield self.raw_llm_output


class DraftingService:
    """Generate chapter drafts using :class:`DraftingAgent`."""

    def __init__(self, agent: DraftingAgent | None = None) -> None:
        self.agent = agent or DraftingAgent()

    async def draft(
        self,
        plot_outline: dict[str, Any],
        chapter_number: int,
        plot_point_focus: str,
        hybrid_context_for_draft: str,
        chapter_plan: list[SceneDetail] | None,
    ) -> DraftResult:
        """Return a draft for the specified chapter."""

        text, raw, _ = await self.agent.draft_chapter(
            plot_outline,
            chapter_number,
            plot_point_focus,
            hybrid_context_for_draft,
            chapter_plan,
        )
        return DraftResult(text=text, raw_llm_output=raw)
