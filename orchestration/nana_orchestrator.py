# nana_orchestrator.py
import asyncio
import logging
import logging.handlers
import os
import time  # For Rich display updates
from dataclasses import dataclass
from typing import Any

import structlog
import utils
from agents.comprehensive_evaluator_agent import ComprehensiveEvaluatorAgent
from agents.drafting_agent import DraftingAgent
from agents.finalize_agent import FinalizeAgent
from agents.kg_maintainer_agent import KGMaintainerAgent
from agents.planner_agent import PlannerAgent
from agents.world_continuity_agent import WorldContinuityAgent
from chapter_generation import (
    DraftingService,
    DraftResult,
    EvaluationService,
    FinalizationService,
    PrerequisiteData,
    PrerequisitesService,
    RevisionService,
)
from config import (
    CHAPTER_LOGS_DIR,
    CHAPTERS_DIR,
    DEBUG_OUTPUTS_DIR,
    settings,
)
from core.db_manager import neo4j_manager
from data_access import (
    chapter_queries,
    character_queries,
    plot_queries,
    world_queries,
)
from initialization.data_loader import convert_model_to_objects
from initialization.genesis import run_genesis_phase
from initialization.models import PlotOutline
from kg_maintainer.models import (
    CharacterProfile,
    EvaluationResult,
    ProblemDetail,
    SceneDetail,
    WorldItem,
)
from processing.context_generator import generate_hybrid_chapter_context_logic
from processing.revision_manager import RevisionManager
from processing.text_deduplicator import TextDeduplicator
from ui.rich_display import RichDisplayManager
from utils.ingestion_utils import split_text_into_chapters

from models.user_input_models import UserStoryInputModel
from orchestration.chapter_flow import run_chapter_pipeline
from orchestration.token_tracker import TokenTracker

try:
    from rich.logging import RichHandler

    RICH_AVAILABLE = True
except Exception:  # pragma: no cover - fallback when Rich is missing
    RICH_AVAILABLE = False

    class RichHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover
            logging.getLogger(__name__).handle(record)


logger = logging.getLogger(__name__)


@dataclass
class RevisionOutcome:
    """Final text after processing and whether it is marked flawed."""

    text: str | None
    raw_llm_output: str | None
    is_flawed: bool


class NANA_Orchestrator:
    def __init__(self):
        logger.info("Initializing NANA Orchestrator...")
        self.planner_agent = PlannerAgent()
        self.drafting_agent = DraftingAgent()
        self.evaluator_agent = ComprehensiveEvaluatorAgent()
        self.world_continuity_agent = WorldContinuityAgent()
        self.kg_maintainer_agent = KGMaintainerAgent()
        self.finalize_agent = FinalizeAgent(self.kg_maintainer_agent)
        self.revision_manager = RevisionManager()

        # Chapter generation services
        self.prerequisites_service = PrerequisitesService(self)
        self.drafting_service = DraftingService(self)
        self.evaluation_service = EvaluationService(self)
        self.revision_service = RevisionService(self)
        self.finalization_service = FinalizationService(self)

        self.plot_outline: PlotOutline = PlotOutline()
        self.chapter_count: int = 0
        self.novel_props_cache: dict[str, Any] = {}
        self.token_tracker = TokenTracker()
        self.total_tokens_generated_this_run: int = 0

        self.display = RichDisplayManager()
        self.run_start_time: float = 0.0
        utils.load_spacy_model_if_needed()
        logger.info("NANA Orchestrator initialized.")

    def _update_rich_display(
        self, chapter_num: int | None = None, step: str | None = None
    ) -> None:
        self.display.update(
            plot_outline=self.plot_outline,
            chapter_num=chapter_num,
            step=step,
            total_tokens=self.total_tokens_generated_this_run,
            run_start_time=self.run_start_time,
        )

    def _accumulate_tokens(
        self, operation_name: str, usage_data: dict[str, int] | None
    ):
        self.token_tracker.add(operation_name, usage_data)
        self.total_tokens_generated_this_run = self.token_tracker.total
        self._update_rich_display()

    async def _generate_plot_points_from_kg(self, count: int) -> None:
        """Generate and persist additional plot points using the planner agent."""
        if count <= 0:
            return

        summaries: list[str] = []
        start = max(1, self.chapter_count - settings.CONTEXT_CHAPTER_COUNT + 1)
        for i in range(start, self.chapter_count + 1):
            chap = await chapter_queries.get_chapter_data_from_db(i)
            if chap and (chap.get("summary") or chap.get("text")):
                summaries.append((chap.get("summary") or chap.get("text", "")).strip())

        combined_summary = "\n".join(summaries)
        if not combined_summary.strip():
            logger.warning("No summaries available for continuation planning.")
            return

        new_points, usage = await self.planner_agent.plan_continuation(
            combined_summary, count
        )
        self._accumulate_tokens("PlanContinuation", usage)
        if not new_points:
            logger.error("Failed to generate continuation plot points.")
            return

        for desc in new_points:
            if await plot_queries.plot_point_exists(desc):
                logger.info("Plot point already exists, skipping: %s", desc)
                continue
            prev_id = await plot_queries.get_last_plot_point_id()
            await self.kg_maintainer_agent.add_plot_point(desc, prev_id or "")
            self.plot_outline.setdefault("plot_points", []).append(desc)
        self._update_novel_props_cache()

    def load_state_from_user_model(self, model: UserStoryInputModel) -> None:
        """Populate orchestrator state from a user-provided model."""
        plot_outline, _, _ = convert_model_to_objects(model)
        self.plot_outline = plot_outline

    def _update_novel_props_cache(self):
        self.novel_props_cache = {
            "title": self.plot_outline.get(
                "title", settings.DEFAULT_PLOT_OUTLINE_TITLE
            ),
            "genre": self.plot_outline.get("genre", settings.CONFIGURED_GENRE),
            "theme": self.plot_outline.get("theme", settings.CONFIGURED_THEME),
            "protagonist_name": self.plot_outline.get(
                "protagonist_name", settings.DEFAULT_PROTAGONIST_NAME
            ),
            "character_arc": self.plot_outline.get("character_arc", "N/A"),
            "logline": self.plot_outline.get("logline", "N/A"),
            "setting": self.plot_outline.get(
                "setting", settings.CONFIGURED_SETTING_DESCRIPTION
            ),
            "narrative_style": self.plot_outline.get("narrative_style", "N/A"),
            "tone": self.plot_outline.get("tone", "N/A"),
            "pacing": self.plot_outline.get("pacing", "N/A"),
            "plot_points": self.plot_outline.get("plot_points", []),
            "plot_outline_full": self.plot_outline,
        }
        self._update_rich_display()

    async def refresh_plot_outline(self) -> None:
        """Reload plot outline from the database."""
        result = await plot_queries.get_plot_outline_from_db()
        if isinstance(result, dict):
            self.plot_outline = PlotOutline(**result)
            self._update_novel_props_cache()
        else:
            logger.error("Failed to refresh plot outline from DB: %s", result)

    async def async_init_orchestrator(self):
        logger.info("NANA Orchestrator async_init_orchestrator started...")
        self._update_rich_display(step="Initializing Orchestrator")
        self.chapter_count = await chapter_queries.load_chapter_count_from_db()
        logger.info(f"Loaded chapter count from Neo4j: {self.chapter_count}")
        await plot_queries.ensure_novel_info()
        result = await plot_queries.get_plot_outline_from_db()
        if isinstance(result, Exception):
            logger.error(
                "Error loading plot outline during orchestrator init: %s",
                result,
                exc_info=result,
            )
            self.plot_outline = PlotOutline()
        else:
            self.plot_outline = (
                PlotOutline(**result) if isinstance(result, dict) else PlotOutline()
            )

        if not self.plot_outline.get("plot_points"):
            logger.warning(
                "Orchestrator init: Plot outline loaded from DB has no plot points. Initial setup might be needed or DB is empty/corrupt."
            )
        else:
            logger.info(
                f"Orchestrator init: Loaded {len(self.plot_outline.get('plot_points', []))} plot points from DB."
            )

        self._update_novel_props_cache()
        logger.info("NANA Orchestrator async_init_orchestrator complete.")
        self._update_rich_display(step="Orchestrator Initialized")

    async def perform_initial_setup(self):
        self._update_rich_display(step="Performing Initial Setup")
        logger.info("NANA performing initial setup...")
        (
            self.plot_outline,
            character_profiles,
            world_building,
            usage,
        ) = await run_genesis_phase()
        self._accumulate_tokens("Genesis-Phase", usage)

        plot_source = self.plot_outline.get("source", "unknown")
        logger.info(
            f"   Plot Outline and Characters initialized/loaded (source: {plot_source}). "
            f"Title: '{self.plot_outline.get('title', 'N/A')}'. "
            f"Plot Points: {len(self.plot_outline.get('plot_points', []))}"
        )
        world_source = world_building.get("source", "unknown")
        logger.info(f"   World Building initialized/loaded (source: {world_source}).")
        self._update_rich_display(step="Genesis State Bootstrapped")

        self._update_novel_props_cache()
        logger.info("   Initial plot, character, and world data saved to Neo4j.")
        self._update_rich_display(step="Initial State Saved")

        return True

    def _get_plot_point_info_for_chapter(
        self, novel_chapter_number: int
    ) -> tuple[str | None, int]:
        plot_points_list = self.plot_outline.get("plot_points", [])
        if not isinstance(plot_points_list, list) or not plot_points_list:
            logger.error(
                f"No plot points available in orchestrator state for chapter {novel_chapter_number}."
            )
            return None, -1

        plot_point_index = novel_chapter_number - 1

        if 0 <= plot_point_index < len(plot_points_list):
            plot_point_item = plot_points_list[plot_point_index]
            plot_point_text = (
                plot_point_item.get("description")
                if isinstance(plot_point_item, dict)
                else str(plot_point_item)
            )
            if isinstance(plot_point_text, str) and plot_point_text.strip():
                return plot_point_text, plot_point_index
            logger.warning(
                f"Plot point at index {plot_point_index} for chapter {novel_chapter_number} is empty or invalid. Using placeholder."
            )
            return settings.FILL_IN, plot_point_index
        else:
            logger.error(
                f"Plot point index {plot_point_index} is out of bounds for plot_points list (len: {len(plot_points_list)}) for chapter {novel_chapter_number}."
            )
            return None, -1

    async def _save_chapter_text_and_log(
        self, chapter_number: int, final_text: str, raw_llm_log: str | None
    ):
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None,
                self._save_chapter_files_sync_io,
                chapter_number,
                final_text,
                raw_llm_log or "N/A",
            )
            logger.info(
                f"Saved chapter text and raw LLM log files for ch {chapter_number}."
            )
        except OSError as e:
            logger.error(
                f"Failed writing chapter text/log files for ch {chapter_number}: {e}",
                exc_info=True,
            )

    def _save_chapter_files_sync_io(
        self, chapter_number: int, final_text: str, raw_llm_log: str
    ):
        chapter_file_path = os.path.join(
            CHAPTERS_DIR, f"chapter_{chapter_number:04d}.txt"
        )
        log_file_path = os.path.join(
            CHAPTER_LOGS_DIR,
            f"chapter_{chapter_number:04d}_raw_llm_log.txt",
        )
        os.makedirs(os.path.dirname(chapter_file_path), exist_ok=True)
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        with open(chapter_file_path, "w", encoding="utf-8") as f:
            f.write(final_text)
        with open(log_file_path, "w", encoding="utf-8") as f:
            f.write(raw_llm_log)

    async def _save_debug_output(
        self, chapter_number: int, stage_description: str, content: Any
    ):
        if content is None:
            return
        content_str = str(content) if not isinstance(content, str) else content
        if not content_str.strip():
            return
        loop = asyncio.get_running_loop()
        try:
            safe_stage_desc = "".join(
                c if c.isalnum() or c in ["_", "-"] else "_" for c in stage_description
            )
            file_name = f"chapter_{chapter_number:04d}_{safe_stage_desc}.txt"
            file_path = os.path.join(DEBUG_OUTPUTS_DIR, file_name)
            await loop.run_in_executor(
                None, self._save_debug_output_sync_io, file_path, content_str
            )
            logger.debug(
                f"Saved debug output for Ch {chapter_number}, Stage '{stage_description}' to {file_path}"
            )
        except Exception as e:
            logger.error(
                f"Failed to save debug output (Ch {chapter_number}, Stage '{stage_description}'): {e}",
                exc_info=True,
            )

    def _save_debug_output_sync_io(self, file_path: str, content_str: str):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content_str)

    async def perform_deduplication(
        self, text_to_dedup: str, chapter_number: int
    ) -> tuple[str, int]:
        logger.info(f"NANA: Performing de-duplication for Chapter {chapter_number}...")
        if not text_to_dedup or not text_to_dedup.strip():
            logger.info(
                f"De-duplication for Chapter {chapter_number}: Input text is empty. No action taken."
            )
            return text_to_dedup, 0
        try:
            deduper = TextDeduplicator(
                similarity_threshold=settings.DEDUPLICATION_SEMANTIC_THRESHOLD,
                use_semantic_comparison=settings.DEDUPLICATION_USE_SEMANTIC,
                min_segment_length_chars=settings.DEDUPLICATION_MIN_SEGMENT_LENGTH,
            )
            deduplicated_text, chars_removed = await deduper.deduplicate(
                text_to_dedup, segment_level="sentence"
            )
            if chars_removed > 0:
                method = (
                    "semantic"
                    if settings.DEDUPLICATION_USE_SEMANTIC
                    else "normalized string"
                )
                logger.info(
                    f"De-duplication for Chapter {chapter_number} removed {chars_removed} characters using {method} matching."
                )
            else:
                logger.info(
                    f"De-duplication for Chapter {chapter_number}: No significant duplicates found."
                )
            return deduplicated_text, chars_removed
        except Exception as e:
            logger.error(
                f"Error during de-duplication for Chapter {chapter_number}: {e}",
                exc_info=True,
            )
            return text_to_dedup, 0

    async def _run_evaluation_cycle(
        self,
        novel_chapter_number: int,
        attempt: int,
        current_text: str,
        plot_point_focus: str,
        plot_point_index: int,
        hybrid_context_for_draft: str,
        patched_spans: list[tuple[int, int]],
    ) -> tuple[
        EvaluationResult,
        list[ProblemDetail],
        dict[str, int] | None,
        dict[str, int] | None,
    ]:
        result = await self.evaluation_service.run_cycle(
            chapter_number=novel_chapter_number,
            attempt=attempt,
            current_text=current_text,
            plot_point_focus=plot_point_focus,
            plot_point_index=plot_point_index,
            hybrid_context_for_draft=hybrid_context_for_draft,
            patched_spans=patched_spans,
        )
        return (
            result.evaluation,
            result.continuity_problems,
            result.eval_usage,
            result.continuity_usage,
        )

    async def _handle_no_evaluation_fast_path(
        self,
        novel_chapter_number: int,
        initial_text: str,
        initial_raw_llm_text: str | None,
    ) -> tuple[str, str | None, bool] | None:
        """Return early with deduplicated text when evaluation is disabled."""
        if (
            not settings.ENABLE_COMPREHENSIVE_EVALUATION
            and not settings.ENABLE_WORLD_CONTINUITY_CHECK
        ):
            logger.info(
                f"NANA: Ch {novel_chapter_number} - All evaluation agents disabled. Applying de-duplication and finalizing draft."
            )
            self._update_rich_display(
                step=f"Ch {novel_chapter_number} - Skipping Revisions (disabled)"
            )
            deduplicated_text, removed_char_count = await self.perform_deduplication(
                initial_text,
                novel_chapter_number,
            )
            is_flawed = removed_char_count > 0
            if is_flawed:
                logger.info(
                    f"NANA: Ch {novel_chapter_number} - Text marked as flawed for KG due to de-duplication removing {removed_char_count} characters."
                )
                await self._save_debug_output(
                    novel_chapter_number,
                    "deduplicated_text_no_eval_path",
                    deduplicated_text,
                )
            return deduplicated_text, initial_raw_llm_text, is_flawed
        return None

    async def _deduplicate_post_draft(
        self,
        novel_chapter_number: int,
        text: str | None,
    ) -> tuple[str | None, bool]:
        """Deduplicate text after drafting and log results."""
        self._update_rich_display(
            step=f"Ch {novel_chapter_number} - Post-Draft De-duplication"
        )
        logger.info(
            f"NANA: Ch {novel_chapter_number} - Applying post-draft de-duplication."
        )
        if text is None:
            return None, False
        deduped, removed = await self.perform_deduplication(text, novel_chapter_number)
        if removed > 0:
            await self._save_debug_output(
                novel_chapter_number,
                "deduplicated_text_after_draft",
                deduped,
            )
            logger.info(
                f"NANA: Ch {novel_chapter_number} - De-duplication removed {removed} characters. Text marked as potentially flawed for KG."
            )
            return deduped, True
        logger.info(
            f"NANA: Ch {novel_chapter_number} - Post-draft de-duplication found no significant changes."
        )
        return text, False

    async def _run_revision_loop(
        self,
        novel_chapter_number: int,
        current_text: str | None,
        current_raw_llm_output: str | None,
        plot_point_focus: str,
        plot_point_index: int,
        hybrid_context_for_draft: str,
        chapter_plan: list[SceneDetail] | None,
        patched_spans: list[tuple[int, int]],
        is_from_flawed_source_for_kg: bool,
    ) -> tuple[str | None, str | None, bool, list[tuple[int, int]]]:
        result = await self.revision_service.run_revision_loop(
            chapter_number=novel_chapter_number,
            current_text=current_text,
            current_raw_llm_output=current_raw_llm_output,
            plot_point_focus=plot_point_focus,
            plot_point_index=plot_point_index,
            hybrid_context_for_draft=hybrid_context_for_draft,
            chapter_plan=chapter_plan,
            patched_spans=patched_spans,
            is_from_flawed_source_for_kg=is_from_flawed_source_for_kg,
        )
        return (
            result.text,
            result.raw_llm_output,
            result.is_flawed_source,
            result.patched_spans,
        )

    async def _deduplicate_post_revision(
        self,
        novel_chapter_number: int,
        text: str,
        is_flawed: bool,
    ) -> tuple[str, bool]:
        return await self.revision_service.deduplicate_post_revision(
            chapter_number=novel_chapter_number,
            text=text,
            is_flawed=is_flawed,
        )

    async def _prepare_chapter_prerequisites(
        self, novel_chapter_number: int
    ) -> PrerequisiteData:
        """Gather planning and context needed before drafting a chapter."""
        self._update_rich_display(
            step=f"Ch {novel_chapter_number} - Preparing Prerequisites"
        )

        plot_point_focus, plot_point_index = self._get_plot_point_info_for_chapter(
            novel_chapter_number
        )
        if plot_point_focus is None:
            logger.error(
                f"NANA: Ch {novel_chapter_number} prerequisite check failed: no concrete plot point focus (index {plot_point_index})."
            )
            return PrerequisiteData(None, -1, None, None)

        self._update_novel_props_cache()

        chapter_plan_result, plan_usage = await self.planner_agent.plan_chapter_scenes(
            self.plot_outline,
            await character_queries.get_character_profiles_from_db(),
            await world_queries.get_world_building_from_db(),
            novel_chapter_number,
            plot_point_focus,
            plot_point_index,
        )
        self._accumulate_tokens(f"Ch{novel_chapter_number}-Planning", plan_usage)

        chapter_plan: list[SceneDetail] | None = chapter_plan_result

        if (
            settings.ENABLE_SCENE_PLAN_VALIDATION
            and chapter_plan is not None
            and settings.ENABLE_WORLD_CONTINUITY_CHECK
        ):
            (
                plan_problems,
                usage,
            ) = await self.world_continuity_agent.check_scene_plan_consistency(
                self.plot_outline,
                chapter_plan,
                novel_chapter_number,
            )
            self._accumulate_tokens(
                f"Ch{novel_chapter_number}-PlanConsistency",
                usage,
            )
            await self._save_debug_output(
                novel_chapter_number,
                "scene_plan_consistency_problems",
                plan_problems,
            )
            if plan_problems:
                logger.warning(
                    f"NANA: Ch {novel_chapter_number} scene plan has {len(plan_problems)} consistency issues."
                )

        hybrid_context_for_draft = await generate_hybrid_chapter_context_logic(
            self, novel_chapter_number, chapter_plan
        )

        if settings.ENABLE_AGENTIC_PLANNING and chapter_plan is None:
            logger.warning(
                f"NANA: Ch {novel_chapter_number}: Planning Agent failed or plan invalid. Proceeding with plot point focus only."
            )
        await self._save_debug_output(
            novel_chapter_number,
            "scene_plan",
            chapter_plan if chapter_plan else "No plan generated.",
        )
        await self._save_debug_output(
            novel_chapter_number,
            "hybrid_context_for_draft",
            hybrid_context_for_draft,
        )

        return PrerequisiteData(
            plot_point_focus=plot_point_focus,
            plot_point_index=plot_point_index,
            chapter_plan=chapter_plan,
            hybrid_context_for_draft=hybrid_context_for_draft,
        )

    async def _draft_initial_chapter_text(
        self,
        novel_chapter_number: int,
        plot_point_focus: str,
        hybrid_context_for_draft: str,
        chapter_plan: list[SceneDetail] | None,
    ) -> DraftResult:
        self._update_rich_display(
            step=f"Ch {novel_chapter_number} - Drafting Initial Text"
        )
        (
            initial_draft_text,
            initial_raw_llm_text,
            draft_usage,
        ) = await self.drafting_agent.draft_chapter(
            self.plot_outline,
            novel_chapter_number,
            plot_point_focus,
            hybrid_context_for_draft,
            chapter_plan,
        )
        self._accumulate_tokens(f"Ch{novel_chapter_number}-Drafting", draft_usage)

        if not initial_draft_text:
            logger.error(
                f"NANA: Drafting Agent failed for Ch {novel_chapter_number}. No initial draft produced."
            )
            await self._save_debug_output(
                novel_chapter_number,
                "initial_draft_fail_raw_llm",
                initial_raw_llm_text or "Drafting Agent returned None for raw output.",
            )
            return DraftResult(text=None, raw_llm_output=None)

        await self._save_debug_output(
            novel_chapter_number, "initial_draft", initial_draft_text
        )
        return DraftResult(text=initial_draft_text, raw_llm_output=initial_raw_llm_text)

    async def _process_and_revise_draft(
        self,
        novel_chapter_number: int,
        initial_draft_text: str,
        initial_raw_llm_text: str | None,
        plot_point_focus: str,
        plot_point_index: int,
        hybrid_context_for_draft: str,
        chapter_plan: list[SceneDetail] | None,
    ) -> RevisionOutcome:
        fast_path_result = await self._handle_no_evaluation_fast_path(
            novel_chapter_number,
            initial_draft_text,
            initial_raw_llm_text,
        )
        if fast_path_result is not None:
            text, raw, flawed = fast_path_result
            return RevisionOutcome(text=text, raw_llm_output=raw, is_flawed=flawed)

        current_text_to_process: str | None = initial_draft_text
        current_raw_llm_output: str | None = initial_raw_llm_text
        is_from_flawed_source_for_kg = False
        patched_spans: list[tuple[int, int]] = []

        (
            current_text_to_process,
            flawed_after_dedup,
        ) = await self._deduplicate_post_draft(
            novel_chapter_number,
            current_text_to_process,
        )
        if flawed_after_dedup:
            is_from_flawed_source_for_kg = True

        (
            current_text_to_process,
            current_raw_llm_output,
            is_from_flawed_source_for_kg,
            patched_spans,
        ) = await self._run_revision_loop(
            novel_chapter_number,
            current_text_to_process,
            current_raw_llm_output,
            plot_point_focus,
            plot_point_index,
            hybrid_context_for_draft,
            chapter_plan,
            patched_spans,
            is_from_flawed_source_for_kg,
        )
        if current_text_to_process is None:
            logger.critical(
                f"NANA: Ch {novel_chapter_number} - current_text_to_process is None after revision loop. Aborting chapter."
            )
            return RevisionOutcome(text=None, raw_llm_output=None, is_flawed=True)

        (
            current_text_to_process,
            is_from_flawed_source_for_kg,
        ) = await self._deduplicate_post_revision(
            novel_chapter_number,
            current_text_to_process,
            is_from_flawed_source_for_kg,
        )

        return RevisionOutcome(
            text=current_text_to_process,
            raw_llm_output=current_raw_llm_output,
            is_flawed=is_from_flawed_source_for_kg,
        )

    async def _finalize_and_save_chapter(
        self,
        novel_chapter_number: int,
        final_text_to_process: str,
        final_raw_llm_output: str | None,
        is_from_flawed_source_for_kg: bool,
    ) -> str | None:
        result = await self.finalization_service.finalize_and_save_chapter(
            chapter_number=novel_chapter_number,
            final_text_to_process=final_text_to_process,
            final_raw_llm_output=final_raw_llm_output,
            is_from_flawed_source_for_kg=is_from_flawed_source_for_kg,
        )
        return result.text

    async def _validate_plot_outline(self, novel_chapter_number: int) -> bool:
        if (
            not self.plot_outline
            or not self.plot_outline.get("plot_points")
            or not self.plot_outline.get("protagonist_name")
        ):
            logger.error(
                f"NANA: Cannot write Ch {novel_chapter_number}: Plot outline or critical plot data missing."
            )
            self._update_rich_display(
                step=f"Ch {novel_chapter_number} Failed - Missing Plot Outline"
            )
            return False
        return True

    async def _process_prereq_result(
        self,
        novel_chapter_number: int,
        prereq_result: PrerequisiteData,
    ) -> PrerequisiteData | None:
        plot_point_focus = prereq_result.plot_point_focus
        plot_point_index = prereq_result.plot_point_index
        chapter_plan = prereq_result.chapter_plan
        hybrid_context_for_draft = prereq_result.hybrid_context_for_draft

        if plot_point_focus is None or hybrid_context_for_draft is None:
            self._update_rich_display(
                step=f"Ch {novel_chapter_number} Failed - Prerequisites Incomplete"
            )
            return None
        return PrerequisiteData(
            plot_point_focus=plot_point_focus,
            plot_point_index=plot_point_index,
            chapter_plan=chapter_plan,
            hybrid_context_for_draft=hybrid_context_for_draft,
        )

    async def _process_initial_draft(
        self,
        novel_chapter_number: int,
        draft_result: DraftResult,
    ) -> DraftResult | None:
        initial_draft_text = draft_result.text
        initial_raw_llm_text = draft_result.raw_llm_output
        if initial_draft_text is None:
            self._update_rich_display(
                step=f"Ch {novel_chapter_number} Failed - No Initial Draft"
            )
            return None
        return DraftResult(text=initial_draft_text, raw_llm_output=initial_raw_llm_text)

    async def _process_revision_result(
        self,
        novel_chapter_number: int,
        revision_result: RevisionOutcome,
    ) -> RevisionOutcome | None:
        processed_text = revision_result.text
        processed_raw_llm = revision_result.raw_llm_output
        is_flawed = revision_result.is_flawed
        if processed_text is None:
            self._update_rich_display(
                step=f"Ch {novel_chapter_number} Failed - Revision/Processing Error"
            )
            return None
        return RevisionOutcome(
            text=processed_text,
            raw_llm_output=processed_raw_llm,
            is_flawed=is_flawed,
        )

    async def _finalize_and_log(
        self,
        novel_chapter_number: int,
        processed_text: str,
        processed_raw_llm: str | None,
        is_flawed: bool,
    ) -> str | None:
        final_text_result = await self._finalize_and_save_chapter(
            novel_chapter_number, processed_text, processed_raw_llm, is_flawed
        )

        if final_text_result:
            status_message = (
                "Successfully Generated"
                if not is_flawed
                else "Generated (Marked with Flaws)"
            )
            logger.info(
                f"=== NANA: Finished Novel Chapter {novel_chapter_number} - {status_message} ==="
            )
            self._update_rich_display(
                step=f"Ch {novel_chapter_number} - {status_message}"
            )
        else:
            logger.error(
                f"=== NANA: Failed Novel Chapter {novel_chapter_number} - Finalization/Save Error ==="
            )
            self._update_rich_display(
                step=f"Ch {novel_chapter_number} Failed - Finalization Error"
            )
        return final_text_result

    async def run_chapter_generation_process(
        self, novel_chapter_number: int
    ) -> str | None:
        return await run_chapter_pipeline(self, novel_chapter_number)

    def _validate_critical_configs(self) -> bool:
        critical_str_configs = {
            "OLLAMA_EMBED_URL": settings.OLLAMA_EMBED_URL,
            "OPENAI_API_BASE": settings.OPENAI_API_BASE,
            "EMBEDDING_MODEL": settings.EMBEDDING_MODEL,
            "NEO4J_URI": settings.NEO4J_URI,
            "LARGE_MODEL": settings.LARGE_MODEL,
            "MEDIUM_MODEL": settings.MEDIUM_MODEL,
            "SMALL_MODEL": settings.SMALL_MODEL,
            "NARRATOR_MODEL": settings.NARRATOR_MODEL,
        }
        missing_or_empty_configs = []
        for name, value in critical_str_configs.items():
            if not value or not isinstance(value, str) or not value.strip():
                missing_or_empty_configs.append(name)

        if missing_or_empty_configs:
            logger.critical(
                f"NANA CRITICAL CONFIGURATION ERROR: The following critical configuration(s) are missing or empty: {', '.join(missing_or_empty_configs)}. Please set them (e.g., in .env file or environment variables) and restart."
            )
            return False

        if settings.EXPECTED_EMBEDDING_DIM <= 0:
            logger.critical(
                f"NANA CRITICAL CONFIGURATION ERROR: EXPECTED_EMBEDDING_DIM must be a positive integer, but is {settings.EXPECTED_EMBEDDING_DIM}."
            )
            return False

        logger.info("Critical configurations validated successfully.")
        return True

    async def run_novel_generation_loop(self):
        logger.info("--- NANA: Starting Novel Generation Run ---")

        if not self._validate_critical_configs():
            self._update_rich_display(step="Critical Config Error - Halting")
            await self.display.stop()
            return

        self.total_tokens_generated_this_run = 0
        self.run_start_time = time.time()
        self.display.start()
        try:
            async with neo4j_manager:
                await neo4j_manager.create_db_schema()
                logger.info("NANA: Neo4j connection and schema verified.")

                await self.kg_maintainer_agent.load_schema_from_db()
                logger.info("NANA: KG schema loaded into maintainer agent.")

                await self.async_init_orchestrator()

            plot_points_exist = (
                self.plot_outline
                and self.plot_outline.get("plot_points")
                and len(
                    [
                        pp
                        for pp in self.plot_outline.get("plot_points", [])
                        if not utils._is_fill_in(pp)
                    ]
                )
                > 0
            )

            if (
                not plot_points_exist
                or not self.plot_outline.get("title")
                or utils._is_fill_in(self.plot_outline.get("title"))
            ):
                logger.info(
                    "NANA: Core plot data missing or insufficient (e.g., no title, no concrete plot points). Performing initial setup..."
                )
                if not await self.perform_initial_setup():
                    logger.critical("NANA: Initial setup failed. Halting generation.")
                    self._update_rich_display(step="Initial Setup Failed - Halting")
                    return
                self._update_novel_props_cache()

            # KG pre-population handled within run_genesis_phase

            logger.info("\n--- NANA: Starting Novel Writing Process ---")

            plot_points_raw = self.plot_outline.get("plot_points", [])
            if isinstance(plot_points_raw, list):
                plot_points_list = plot_points_raw
            elif isinstance(plot_points_raw, dict):
                plot_points_list = list(plot_points_raw.values())
            elif plot_points_raw:
                plot_points_list = [plot_points_raw]
            else:
                plot_points_list = []

            total_concrete_plot_points = len(
                [
                    pp
                    for pp in plot_points_list
                    if not utils._is_fill_in(pp) and isinstance(pp, str) and pp.strip()
                ]
            )

            remaining_plot_points_to_address_in_novel = (
                total_concrete_plot_points - self.chapter_count
            )

            logger.info(
                f"NANA: Current Novel Chapter Count (State): {self.chapter_count}"
            )
            logger.info(
                f"NANA: Total Concrete Plot Points in Outline: {total_concrete_plot_points}"
            )
            logger.info(
                f"NANA: Remaining Concrete Plot Points to Cover in Novel: {remaining_plot_points_to_address_in_novel}"
            )

            if remaining_plot_points_to_address_in_novel <= 0:
                await self._generate_plot_points_from_kg(settings.CHAPTERS_PER_RUN)
                await self.refresh_plot_outline()

            logger.info(
                f"NANA: Starting dynamic chapter loop (max {settings.CHAPTERS_PER_RUN} chapter(s) this run)."
            )

            chapters_successfully_written_this_run = 0
            attempts_this_run = 0
            while attempts_this_run < settings.CHAPTERS_PER_RUN:
                plot_points_raw = self.plot_outline.get("plot_points", [])
                if isinstance(plot_points_raw, list):
                    plot_points_list = plot_points_raw
                elif isinstance(plot_points_raw, dict):
                    plot_points_list = list(plot_points_raw.values())
                elif plot_points_raw:
                    plot_points_list = [plot_points_raw]
                else:
                    plot_points_list = []

                total_concrete_plot_points = len(
                    [
                        pp
                        for pp in plot_points_list
                        if not utils._is_fill_in(pp)
                        and isinstance(pp, str)
                        and pp.strip()
                    ]
                )
                remaining_plot_points_to_address_in_novel = (
                    total_concrete_plot_points - self.chapter_count
                )

                if remaining_plot_points_to_address_in_novel <= 0:
                    await self._generate_plot_points_from_kg(
                        settings.CHAPTERS_PER_RUN - attempts_this_run
                    )
                    await self.refresh_plot_outline()
                    plot_points_raw = self.plot_outline.get("plot_points", [])
                    if isinstance(plot_points_raw, list):
                        plot_points_list = plot_points_raw
                    elif isinstance(plot_points_raw, dict):
                        plot_points_list = list(plot_points_raw.values())
                    elif plot_points_raw:
                        plot_points_list = [plot_points_raw]
                    else:
                        plot_points_list = []

                    total_concrete_plot_points = len(
                        [
                            pp
                            for pp in plot_points_list
                            if not utils._is_fill_in(pp)
                            and isinstance(pp, str)
                            and pp.strip()
                        ]
                    )
                    remaining_plot_points_to_address_in_novel = (
                        total_concrete_plot_points - self.chapter_count
                    )
                    if remaining_plot_points_to_address_in_novel <= 0:
                        logger.info(
                            "NANA: No plot points available after generation. Ending run early."
                        )
                        break

                current_novel_chapter_number = self.chapter_count + 1

                logger.info(
                    f"\n--- NANA: Attempting Novel Chapter {current_novel_chapter_number} (attempt {attempts_this_run + 1}/{settings.CHAPTERS_PER_RUN}) ---"
                )
                self._update_rich_display(
                    chapter_num=current_novel_chapter_number,
                    step="Starting Chapter Loop",
                )

                try:
                    chapter_text_result = await self.run_chapter_generation_process(
                        current_novel_chapter_number
                    )
                    if chapter_text_result:
                        chapters_successfully_written_this_run += 1
                        logger.info(
                            f"NANA: Novel Chapter {current_novel_chapter_number}: Processed. Final text length: {len(chapter_text_result)} chars."
                        )
                        logger.info(
                            f"   Snippet: {chapter_text_result[:200].replace(chr(10), ' ')}..."
                        )

                        if (
                            current_novel_chapter_number > 0
                            and current_novel_chapter_number
                            % settings.KG_HEALING_INTERVAL
                            == 0
                        ):
                            logger.info(
                                f"\n--- NANA: Triggering KG Healing/Enrichment after Chapter {current_novel_chapter_number} ---"
                            )
                            self._update_rich_display(
                                step=f"Ch {current_novel_chapter_number} - KG Maintenance"
                            )
                            await self.kg_maintainer_agent.heal_and_enrich_kg()
                            await self.refresh_plot_outline()
                            logger.info(
                                "--- NANA: KG Healing/Enrichment cycle complete. ---"
                            )
                    else:
                        logger.error(
                            f"NANA: Novel Chapter {current_novel_chapter_number}: Failed to generate or save. Halting run."
                        )
                        self._update_rich_display(
                            step=f"Ch {current_novel_chapter_number} Failed - Halting Run"
                        )
                        break
                except Exception as e:
                    logger.critical(
                        f"NANA: Critical unhandled error during Novel Chapter {current_novel_chapter_number} writing process: {e}",
                        exc_info=True,
                    )
                    self._update_rich_display(
                        step=f"Critical Error Ch {current_novel_chapter_number} - Halting Run"
                    )
                    break

                attempts_this_run += 1

            final_chapter_count_from_db = (
                await chapter_queries.load_chapter_count_from_db()
            )
            logger.info("\n--- NANA: Novel writing process finished for this run ---")
            logger.info(
                f"NANA: Successfully processed {chapters_successfully_written_this_run} chapter(s) in this run."
            )
            logger.info(
                f"NANA: Current total chapters in database after this run: {final_chapter_count_from_db}"
            )

            logger.info(
                f"NANA: Total LLM tokens generated this run: {self.total_tokens_generated_this_run}"
            )
            self._update_rich_display(
                chapter_num=self.chapter_count, step="Run Finished"
            )

        except Exception as e:
            logger.critical(
                f"NANA: Unhandled exception in orchestrator main loop: {e}",
                exc_info=True,
            )
            self._update_rich_display(step="Critical Error in Main Loop")
        finally:
            await self.display.stop()

    async def run_ingestion_process(self, text_file: str) -> None:
        """Ingest existing text and populate the knowledge graph."""
        logger.info("--- NANA: Starting Ingestion Process ---")

        if not self._validate_critical_configs():
            await self.display.stop()
            return

        self.display.start()
        self.run_start_time = time.time()
        async with neo4j_manager:
            await neo4j_manager.create_db_schema()
            if neo4j_manager.driver is not None:
                await plot_queries.ensure_novel_info()
            else:
                logger.warning(
                    "Neo4j driver not initialized. Skipping NovelInfo setup."
                )
            await self.kg_maintainer_agent.load_schema_from_db()

        with open(text_file, encoding="utf-8") as f:
            raw_text = f.read()

        chunks = split_text_into_chapters(raw_text)
        plot_outline = {"title": "Ingested Narrative", "plot_points": []}
        character_profiles: dict[str, CharacterProfile] = {}
        world_building: dict[str, dict[str, WorldItem]] = {}
        summaries: list[str] = []

        for idx, chunk in enumerate(chunks, 1):
            self._update_rich_display(chapter_num=idx, step="Ingesting Text")
            result = await self.finalize_agent.ingest_and_finalize_chunk(
                plot_outline,
                character_profiles,
                world_building,
                idx,
                chunk,
            )
            if result.get("summary"):
                summaries.append(str(result["summary"]))
                plot_outline["plot_points"].append(result["summary"])

            if idx % settings.KG_HEALING_INTERVAL == 0:
                logger.info(
                    f"--- NANA: Triggering KG Healing/Enrichment after Ingestion Chunk {idx} ---"
                )
                self._update_rich_display(step=f"Ch {idx} - KG Maintenance")
                await self.kg_maintainer_agent.heal_and_enrich_kg()
                await self.refresh_plot_outline()

        await self.kg_maintainer_agent.heal_and_enrich_kg()
        combined_summary = "\n".join(summaries)
        continuation, _ = await self.planner_agent.plan_continuation(combined_summary)
        if continuation:
            plot_outline["plot_points"].extend(continuation)
        self.plot_outline = plot_outline
        self.chapter_count = len(chunks)
        await plot_queries.save_plot_outline_to_db(plot_outline)
        await self.display.stop()
        logger.info("NANA: Ingestion process completed.")


def setup_logging_nana():
    # Step 1: Configure structlog to prepare data and pass it to the standard logger.
    structlog.configure(
        processors=[
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            # This processor formats log messages with positional arguments (e.g., %s)
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            # This processor passes the structured log data to the standard logger,
            # which RichHandler will then receive.
            structlog.stdlib.render_to_log_kwargs,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Step 2: Set up the root logger and clear previous configurations.
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(settings.LOG_LEVEL_STR)

    # Step 3: Configure handlers. RichHandler will now do the console formatting.
    if settings.LOG_FILE:
        try:
            file_path = (
                settings.LOG_FILE
                if os.path.isabs(settings.LOG_FILE)
                else os.path.join(settings.BASE_OUTPUT_DIR, settings.LOG_FILE)
            )
            log_dir = os.path.dirname(file_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                file_path,
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
                mode="a",
                encoding="utf-8",
            )
            # Use a standard formatter for the file log.
            file_formatter = logging.Formatter(
                settings.LOG_FORMAT, datefmt=settings.LOG_DATE_FORMAT
            )
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
        except Exception as e:
            print(f"Error setting up file logger: {e}")

    # Configure Console Handler
    if RICH_AVAILABLE and settings.ENABLE_RICH_PROGRESS:
        # Let RichHandler control its own formatting.
        # We turn its decorations back ON and do NOT set a formatter on it.
        console_handler = RichHandler(
            level=settings.LOG_LEVEL_STR,
            rich_tracebacks=True,
            show_path=False,
            markup=True,
            show_time=True,  # Turn back ON
            show_level=True,  # Turn back ON
        )
        root_logger.addHandler(console_handler)
    else:
        # Fallback to a standard stream handler with a standard formatter
        stream_handler = logging.StreamHandler()
        stream_formatter = logging.Formatter(
            settings.LOG_FORMAT, datefmt=settings.LOG_DATE_FORMAT
        )
        stream_handler.setFormatter(stream_formatter)
        root_logger.addHandler(stream_handler)

    # Set levels for noisy loggers
    logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)
    logging.getLogger("neo4j").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    log = structlog.get_logger()
    log.info(
        "NANA Logging setup complete.",
        log_level=logging.getLevelName(settings.LOG_LEVEL_STR),
    )
