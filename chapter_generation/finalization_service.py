"""Service for finalizing chapters and persisting results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog
from agents.finalize_agent import FinalizationResult
from data_access import character_queries, world_queries

if TYPE_CHECKING:  # pragma: no cover - type hint import
    from orchestration.nana_orchestrator import NANA_Orchestrator

logger = structlog.get_logger(__name__)


@dataclass
class FinalizationServiceResult:
    """Outcome of finalization."""

    text: str | None


class FinalizationService:
    """Finalize chapters via the FinalizeAgent."""

    def __init__(self, orchestrator: NANA_Orchestrator) -> None:
        self.orchestrator = orchestrator

    async def finalize_and_save_chapter(
        self,
        chapter_number: int,
        final_text_to_process: str,
        final_raw_llm_output: str | None,
        is_from_flawed_source_for_kg: bool,
    ) -> FinalizationServiceResult:
        self.orchestrator._update_rich_display(
            step=f"Ch {chapter_number} - Finalization"
        )

        result: FinalizationResult = (
            await self.orchestrator.finalize_agent.finalize_chapter(
                self.orchestrator.plot_outline,
                await character_queries.get_character_profiles_from_db(),
                await world_queries.get_world_building_from_db(),
                chapter_number,
                final_text_to_process,
                final_raw_llm_output,
                is_from_flawed_source_for_kg,
            )
        )

        self.orchestrator._accumulate_tokens(
            f"Ch{chapter_number}-Summarization", result.get("summary_usage")
        )
        self.orchestrator._accumulate_tokens(
            f"Ch{chapter_number}-KGExtractionMerge", result.get("kg_usage")
        )
        await self.orchestrator._save_debug_output(
            chapter_number, "final_summary", result.get("summary")
        )

        if result.get("embedding") is None:
            logger.error(
                "NANA CRITICAL: Failed to generate embedding for final text of Chapter %s. Text saved to file system only.",
                chapter_number,
            )
            await self.orchestrator._save_chapter_text_and_log(
                chapter_number,
                final_text_to_process,
                final_raw_llm_output,
            )
            self.orchestrator._update_rich_display(
                step=f"Ch {chapter_number} Failed - No Embedding"
            )
            return FinalizationServiceResult(text=None)

        await self.orchestrator._save_chapter_text_and_log(
            chapter_number, final_text_to_process, final_raw_llm_output
        )

        self.orchestrator.chapter_count = max(
            self.orchestrator.chapter_count, chapter_number
        )

        return FinalizationServiceResult(text=final_text_to_process)
