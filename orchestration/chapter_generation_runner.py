from __future__ import annotations

# pragma: no cover
from dataclasses import dataclass
from enum import Enum, auto

import structlog
import utils
from config import settings
from core.db_manager import neo4j_manager

if False:  # pragma: no cover - type hints
    from .nana_orchestrator import NANA_Orchestrator

logger = structlog.get_logger(__name__)


class RunnerState(Enum):
    """States for the chapter generation runner."""

    INIT = auto()
    GENERATE_CHAPTER = auto()
    KG_MAINTENANCE = auto()
    HANDLE_ERROR = auto()
    FINISH = auto()


@dataclass
class ChapterGenerationRunner:
    """Manage dynamic chapter generation using a state machine."""

    orchestrator: NANA_Orchestrator
    attempts_this_run: int = 0
    chapters_written: int = 0
    state: RunnerState = RunnerState.INIT
    current_chapter_number: int = 0
    error: Exception | None = None

    async def run(self) -> None:
        """Execute the chapter generation loop."""
        while self.state != RunnerState.FINISH:
            if self.state == RunnerState.INIT:
                await self._init()
            elif self.state == RunnerState.GENERATE_CHAPTER:
                await self._generate_chapter()
            elif self.state == RunnerState.KG_MAINTENANCE:
                await self._kg_maintenance()
            elif self.state == RunnerState.HANDLE_ERROR:
                await self._handle_error()

    async def _init(self) -> None:
        plot_points_raw = self.orchestrator.plot_outline.get("plot_points", [])
        if isinstance(plot_points_raw, dict):
            plot_points_list = list(plot_points_raw.values())
        elif isinstance(plot_points_raw, list):
            plot_points_list = plot_points_raw
        elif plot_points_raw:
            plot_points_list = [plot_points_raw]
        else:
            plot_points_list = []

        total_concrete = len(
            [pp for pp in plot_points_list if not utils._is_fill_in(pp)]
        )
        remaining = total_concrete - self.orchestrator.chapter_count
        if remaining <= 0:
            await self.orchestrator._generate_plot_points_from_kg(
                settings.CHAPTERS_PER_RUN
            )
            await self.orchestrator.refresh_plot_outline()
        self.state = RunnerState.GENERATE_CHAPTER

    async def _generate_chapter(self) -> None:
        if self.attempts_this_run >= settings.CHAPTERS_PER_RUN:
            self.state = RunnerState.FINISH
            return

        plot_points_raw = self.orchestrator.plot_outline.get("plot_points", [])
        if isinstance(plot_points_raw, dict):
            plot_points_list = list(plot_points_raw.values())
        elif isinstance(plot_points_raw, list):
            plot_points_list = plot_points_raw
        elif plot_points_raw:
            plot_points_list = [plot_points_raw]
        else:
            plot_points_list = []

        total_concrete = len(
            [
                pp
                for pp in plot_points_list
                if not utils._is_fill_in(pp) and isinstance(pp, str) and pp.strip()
            ]
        )
        remaining = total_concrete - self.orchestrator.chapter_count
        if remaining <= 0:
            await self.orchestrator._generate_plot_points_from_kg(
                settings.CHAPTERS_PER_RUN - self.attempts_this_run
            )
            await self.orchestrator.refresh_plot_outline()
            plot_points_raw = self.orchestrator.plot_outline.get("plot_points", [])
            if isinstance(plot_points_raw, dict):
                plot_points_list = list(plot_points_raw.values())
            elif isinstance(plot_points_raw, list):
                plot_points_list = plot_points_raw
            elif plot_points_raw:
                plot_points_list = [plot_points_raw]
            else:
                plot_points_list = []

            total_concrete = len(
                [
                    pp
                    for pp in plot_points_list
                    if not utils._is_fill_in(pp) and isinstance(pp, str) and pp.strip()
                ]
            )
            remaining = total_concrete - self.orchestrator.chapter_count
            if remaining <= 0:
                logger.info(
                    "NANA: No plot points available after generation. Ending run early."
                )
                self.state = RunnerState.FINISH
                return

        self.current_chapter_number = self.orchestrator.chapter_count + 1
        logger.info(
            f"\n--- NANA: Attempting Novel Chapter {self.current_chapter_number} (attempt {self.attempts_this_run + 1}/{settings.CHAPTERS_PER_RUN}) ---"
        )
        self.orchestrator._update_rich_display(
            chapter_num=self.current_chapter_number,
            step="Starting Chapter Loop",
        )

        try:
            chapter_text_result = (
                await self.orchestrator.run_chapter_generation_process(
                    self.current_chapter_number
                )
            )
            if chapter_text_result:
                self.chapters_written += 1
                logger.info(
                    f"NANA: Novel Chapter {self.current_chapter_number}: Processed. Final text length: {len(chapter_text_result)} chars."
                )
                logger.info(
                    f"   Snippet: {chapter_text_result[:200].replace(chr(10), ' ')}..."
                )
                self.attempts_this_run += 1
                if (
                    self.current_chapter_number > 0
                    and self.current_chapter_number % settings.KG_HEALING_INTERVAL == 0
                ):
                    self.state = RunnerState.KG_MAINTENANCE
                else:
                    self.state = RunnerState.GENERATE_CHAPTER
                return
            logger.error(
                f"NANA: Novel Chapter {self.current_chapter_number}: Failed to generate or save. Halting run."
            )
            self.orchestrator._update_rich_display(
                step=f"Ch {self.current_chapter_number} Failed - Halting Run"
            )
        except Exception as e:
            self.error = e
            logger.critical(
                f"NANA: Critical unhandled error during Novel Chapter {self.current_chapter_number} writing process: {e}",
                exc_info=True,
            )
            self.orchestrator._update_rich_display(
                step=f"Critical Error Ch {self.current_chapter_number} - Halting Run"
            )
            self.state = RunnerState.HANDLE_ERROR
            return

        self.state = RunnerState.FINISH

    async def _kg_maintenance(self) -> None:
        logger.info(
            f"\n--- NANA: Triggering KG Healing/Enrichment after Chapter {self.current_chapter_number} ---"
        )
        self.orchestrator._update_rich_display(
            step=f"Ch {self.current_chapter_number} - KG Maintenance"
        )
        await self.orchestrator.kg_maintainer_agent.heal_and_enrich_kg()
        await self.orchestrator.refresh_plot_outline()
        if neo4j_manager.driver is not None:
            await self.orchestrator.refresh_knowledge_cache()
        else:
            logger.warning(
                "Neo4j driver not initialized. Skipping knowledge cache refresh."
            )
        logger.info("--- NANA: KG Healing/Enrichment cycle complete. ---")
        self.state = RunnerState.GENERATE_CHAPTER

    async def _handle_error(self) -> None:
        logger.critical("NANA: Error encountered in run: %s", self.error, exc_info=True)
        self.state = RunnerState.FINISH
