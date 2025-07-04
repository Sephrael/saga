# orchestration/output_service.py
"""Service for persisting chapter outputs and debug logs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
from agents.finalize_agent import FinalizationResult
from data_access import character_queries, world_queries
from storage.file_manager import FileManager

from models import ChapterEndState
from orchestration.token_accountant import Stage

if TYPE_CHECKING:  # pragma: no cover - type hints
    from .nana_orchestrator import NANA_Orchestrator

logger = structlog.get_logger(__name__)


class OutputService:
    """Handle finalization, file output and debug logging."""

    def __init__(self, orchestrator: NANA_Orchestrator) -> None:
        self.orchestrator = orchestrator
        self.file_manager: FileManager = orchestrator.file_manager

    async def save_chapter_text_and_log(
        self, chapter_number: int, final_text: str, raw_llm_log: str | None
    ) -> None:
        """Persist final chapter text and raw LLM output."""
        try:
            await self.file_manager.save_chapter_and_log(
                chapter_number, final_text, raw_llm_log or "N/A"
            )
            logger.info(
                "Saved chapter text and raw LLM log files for ch %s.", chapter_number
            )
        except OSError as exc:
            logger.error(
                "Failed writing chapter text/log files for ch %s: %s",
                chapter_number,
                exc,
                exc_info=True,
            )

    async def save_debug_output(
        self, chapter_number: int, stage_description: str, content: Any
    ) -> None:
        """Write debug data to disk if content is provided."""
        if content is None:
            return
        content_str = str(content) if not isinstance(content, str) else content
        if not content_str.strip():
            return
        try:
            await self.file_manager.save_debug_output(
                chapter_number, stage_description, content_str
            )
            logger.debug(
                "Saved debug output for Ch %s, Stage '%s'",
                chapter_number,
                stage_description,
            )
        except Exception as exc:  # pragma: no cover - log and continue
            logger.error(
                "Failed to save debug output (Ch %s, Stage '%s'): %s",
                chapter_number,
                stage_description,
                exc,
                exc_info=True,
            )

    async def finalize_and_save_chapter(
        self,
        novel_chapter_number: int,
        final_text: str,
        final_raw_llm_output: str | None,
        is_from_flawed_source_for_kg: bool,
        fill_in_context: str | None,
    ) -> tuple[str | None, ChapterEndState | None]:
        """Finalize a chapter, persist all files and update counters."""
        o = self.orchestrator
        o._update_rich_display(step=f"Ch {novel_chapter_number} - Finalization")

        result: FinalizationResult = await o.finalize_agent.finalize_chapter(
            o.plot_outline,
            await character_queries.get_character_profiles_from_db(),
            await world_queries.get_world_building_from_db(),
            novel_chapter_number,
            final_text,
            final_raw_llm_output,
            is_from_flawed_source_for_kg,
            fill_in_context=fill_in_context,
        )

        o._accumulate_tokens(
            f"Ch{novel_chapter_number}-{Stage.SUMMARIZATION.value}",
            result.get("summary_usage"),
        )
        o._accumulate_tokens(
            f"Ch{novel_chapter_number}-{Stage.KG_EXTRACTION_MERGE.value}",
            result.get("kg_usage"),
        )
        await self.save_debug_output(
            novel_chapter_number, "final_summary", result.get("summary")
        )

        if result.get("embedding") is None:
            logger.error(
                "NANA CRITICAL: Failed to generate embedding for final text of Chapter %s. Text saved to file system only.",
                novel_chapter_number,
            )
            await self.save_chapter_text_and_log(
                novel_chapter_number, final_text, final_raw_llm_output
            )
            o._update_rich_display(
                step=f"Ch {novel_chapter_number} Failed - No Embedding"
            )
            return None, result.get("chapter_end_state")

        await self.save_chapter_text_and_log(
            novel_chapter_number, final_text, final_raw_llm_output
        )

        o.repetition_tracker.update_from_text(final_text)
        o.chapter_count = max(o.chapter_count, novel_chapter_number)

        return final_text, result.get("chapter_end_state")
