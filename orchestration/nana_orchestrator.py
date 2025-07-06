# orchestration/nana_orchestrator.py
"""Primary orchestrator coordinating all SAGA agent interactions."""

import time  # For Rich display updates
from typing import Any

import structlog
from agents.comprehensive_evaluator_agent import ComprehensiveEvaluatorAgent
from agents.drafting_agent import DraftingAgent
from agents.finalize_agent import FinalizeAgent
from agents.kg_maintainer_agent import KGMaintainerAgent
from agents.planner_agent import PlannerAgent
from chapter_generation import (
    ContextProfileName,
    DraftResult,
    PrerequisiteData,
)
from chapter_generation import (
    create_from_settings as create_context_service,
)
from config import settings
from core.db_manager import neo4j_manager
from core.llm_interface import llm_service
from core.usage import TokenUsage
from data_access import chapter_repository
from ingestion.ingestion_manager import IngestionManager
from initialization.genesis import run_genesis_phase
from initialization.models import PlotOutline
from kg_maintainer.models import (
    SceneDetail,
)
from processing.repetition_analyzer import RepetitionAnalyzer
from processing.repetition_tracker import RepetitionTracker
from processing.revision_manager import RevisionManager
from storage.file_manager import FileManager
from ui.rich_display import RichDisplayManager

import utils
from models.agent_models import ChapterEndState
from models.user_input_models import UserStoryInputModel
from orchestration.chapter_flow import run_chapter_pipeline
from orchestration.chapter_generation_runner import ChapterGenerationRunner
from orchestration.knowledge_service import KnowledgeService

# KnowledgeCache is now primarily managed by StateManagementService
from orchestration.models import KnowledgeCache, RevisionOutcome
from orchestration.output_service import OutputService
from orchestration.prerequisite_service import PrerequisiteService
from orchestration.service_layer import ChapterServiceLayer
from orchestration.services.deduplication_service import DeduplicationService
from orchestration.services.evaluation_revision_service import EvaluationRevisionService
from orchestration.services.initialization_service import InitializationService
from orchestration.services.state_management_service import StateManagementService
from orchestration.services.token_management_service import TokenManagementService
from orchestration.token_accountant import (
    Stage,
)  # TokenAccountant is now in TokenManagementService

logger = structlog.get_logger(__name__)


class NANA_Orchestrator:
    def __init__(self, file_manager: FileManager | None = None):
        logger.info("Initializing NANA Orchestrator...")
        self.file_manager = file_manager or FileManager()
        self.planner_agent = PlannerAgent()
        self.drafting_agent = DraftingAgent()
        self.evaluator_agent = ComprehensiveEvaluatorAgent()
        self.kg_maintainer_agent = KGMaintainerAgent()
        self.finalize_agent = FinalizeAgent(self.kg_maintainer_agent)
        self.revision_manager = RevisionManager()
        self.repetition_tracker = RepetitionTracker()
        self.repetition_analyzer = RepetitionAnalyzer(tracker=self.repetition_tracker)
        self.service_layer = ChapterServiceLayer(
            drafting_agent=self.drafting_agent,
            evaluator_agent=self.evaluator_agent,
            revision_manager=self.revision_manager,
            finalize_agent=self.finalize_agent,
        )

        self.context_service = (
            create_context_service()
        )  # Independent of other orchestrator services for its __init__
        self.deduplication_service = (
            DeduplicationService()
        )  # Independent for its __init__

        # Initialize display and token_manager early as they are needed by many services
        self.display = RichDisplayManager()
        self.run_start_time: float = 0.0  # TokenManagementService will read this
        self.token_manager = TokenManagementService(
            self.display, self
        )  # Pass display manager and self

        # Initialize services that manage state or specific functionalities
        # Note: Potential circular dependency between KnowledgeService and StateManagementService
        # if their __init__ methods immediately access each other via the orchestrator instance.
        self.knowledge_service = KnowledgeService(
            self
        )  # Depends on orchestrator instance (for other services like token_manager)
        self.state_manager = StateManagementService(
            self
        )  # Manages plot_outline, chapter_count, etc. (depends on KS, TM)
        # knowledge_cache is now accessed via self.state_manager.knowledge_cache

        self.output_service = OutputService(self)  # Depends on TM, SM
        self.prerequisite_service = PrerequisiteService(self)  # Depends on TM, SM
        self.evaluation_revision_service = EvaluationRevisionService(
            self
        )  # Depends on TM, SM, OS

        # InitializationService needs references to orchestrator and other services.
        self.initialization_service = InitializationService(
            self
        )  # Depends on SM, TM, display

        utils.load_spacy_model_if_needed()
        logger.info("NANA Orchestrator initialized.")

    # ------------------------------------------------------------------
    # Properties delegating to StateManagementService for backward
    # compatibility with the previous orchestrator API.
    # ------------------------------------------------------------------

    @property
    def plot_outline(self) -> PlotOutline:
        """Return the current plot outline."""
        return self.state_manager.get_plot_outline()

    @plot_outline.setter
    def plot_outline(self, value: PlotOutline) -> None:
        self.state_manager.set_plot_outline(value)

    @property
    def chapter_count(self) -> int:
        return self.state_manager.get_chapter_count()

    @chapter_count.setter
    def chapter_count(self, value: int) -> None:
        self.state_manager.set_chapter_count(value)

    @property
    def next_chapter_context(self) -> str | None:
        return self.state_manager.get_next_chapter_context()

    @next_chapter_context.setter
    def next_chapter_context(self, value: str | None) -> None:
        self.state_manager.set_next_chapter_context(value)

    @property
    def pending_fill_ins(self) -> list[str]:
        return self.state_manager.get_pending_fill_ins()

    @pending_fill_ins.setter
    def pending_fill_ins(self, value: list[str]) -> None:
        self.state_manager.pending_fill_ins = value

    @property
    def missing_references(self) -> dict[str, set[str]]:
        return self.state_manager.get_missing_references()

    @property
    def last_chapter_end_state(self) -> ChapterEndState | None:
        return self.state_manager.get_last_chapter_end_state()

    @property
    def chapter_zero_end_state(self) -> ChapterEndState | None:
        return self.state_manager.get_chapter_zero_end_state()

    @property
    def knowledge_cache(self) -> KnowledgeCache:
        return self.state_manager.get_knowledge_cache()

    @knowledge_cache.setter
    def knowledge_cache(self, value: KnowledgeCache) -> None:
        self.state_manager.knowledge_cache = value

    @property
    def completed_plot_points(self) -> set[str]:
        return self.state_manager.get_completed_plot_points()

    @completed_plot_points.setter
    def completed_plot_points(self, value: set[str]) -> None:
        self.state_manager.completed_plot_points = value

    def _update_rich_display(
        self, chapter_num: int | None = None, step: str | None = None
    ) -> None:
        # This will be called by token_manager, or directly if other updates are needed
        self.token_manager._update_rich_display(chapter_num=chapter_num, step=step)

    def _accumulate_tokens(
        self,
        stage: str | Stage,
        usage_data: dict[str, int] | TokenUsage | None,
        chapter_num: int | None = None,
        current_step_for_display: str | None = None,
    ) -> None:
        """Delegates token accumulation and display updates to TokenManagementService."""
        self.token_manager.accumulate_tokens(
            stage, usage_data, chapter_num, current_step_for_display
        )
        # total_tokens_generated_this_run is now managed by token_manager

    async def _generate_plot_points_from_kg(self, count: int) -> None:
        """Delegate to :class:`KnowledgeService` for plot point generation."""
        await self.knowledge_service.generate_plot_points_from_kg(count)

    def load_state_from_user_model(self, model: UserStoryInputModel) -> None:
        """Populate orchestrator state from a user-provided model via StateManagementService."""
        self.state_manager.load_state_from_user_model(model)

    def _update_novel_props_cache(self) -> None:
        """Delegate to StateManagementService to refresh property cache (which in turn uses KnowledgeService)."""
        self.state_manager._update_novel_props_cache()

    async def refresh_plot_outline(self) -> None:
        """Reload plot outline via StateManagementService."""
        await self.state_manager.refresh_plot_outline()

    async def refresh_knowledge_cache(self) -> None:
        """Reload character profiles and world building caches via StateManagementService."""
        await self.state_manager.refresh_knowledge_cache()

    async def async_init_orchestrator(self):
        """Initialize orchestrator state using StateManagementService."""
        logger.info("NANA Orchestrator async_init_orchestrator started...")
        self.token_manager._update_rich_display(step="Initializing Orchestrator")
        await self.state_manager.async_init_state()
        # Properties like chapter_count, plot_outline, completed_plot_points, knowledge_cache
        # are now managed by self.state_manager
        logger.info("NANA Orchestrator async_init_orchestrator complete.")
        self.token_manager._update_rich_display(step="Orchestrator Initialized")

    async def perform_initial_setup(self) -> bool:
        self.token_manager._update_rich_display(step="Performing Initial Setup")
        logger.info("NANA performing initial setup...")
        (
            self.plot_outline,  # This self.plot_outline is from run_genesis_phase, will be set in state_manager
            character_profiles,
            world_building,
            usage,
        ) = (
            await run_genesis_phase()
        )  # This function likely needs to update the state_manager's plot_outline
        self._accumulate_tokens(
            Stage.GENESIS_PHASE,
            usage,
            current_step_for_display="Genesis State Bootstrapped",
        )

        # After run_genesis_phase, plot_outline is set. We need to update the state_manager's copy.
        # Assuming run_genesis_phase returns the plot_outline object
        self.state_manager.set_plot_outline(self.plot_outline)

        plot_source = self.state_manager.get_plot_outline().get("source", "unknown")
        logger.info(
            f"   Plot Outline and Characters initialized/loaded (source: {plot_source}). "
            f"Title: '{self.state_manager.get_plot_outline().get('title', 'N/A')}'. "
            f"Plot Points: {len(self.state_manager.get_plot_outline().get('plot_points', []))}"
        )
        world_source = world_building.get("source", "unknown")
        logger.info(f"   World Building initialized/loaded (source: {world_source}).")
        # This display update is covered by the _accumulate_tokens call for GENESIS_PHASE
        # self._update_rich_display(step="Genesis State Bootstrapped")

        # Update knowledge cache through state_manager
        current_kc = self.state_manager.get_knowledge_cache()
        current_kc.characters = character_profiles
        current_kc.world = world_building
        # refresh_knowledge_cache will internally update state_manager's cache from knowledge_service
        await self.state_manager.refresh_knowledge_cache()
        self.state_manager._update_novel_props_cache()  # Uses knowledge_service via state_manager
        logger.info("   Initial plot, character, and world data saved to Neo4j.")
        self.token_manager._update_rich_display(
            step="Initial State Saved"
        )  # Keep this specific update

        await self.state_manager.refresh_plot_outline()
        if neo4j_manager.driver is not None:
            await self.state_manager.refresh_knowledge_cache()
        else:
            logger.warning(
                "Neo4j driver not initialized. Skipping knowledge cache refresh."
            )

        # Load chapter zero end state into state_manager
        chapter_zero_end_state = await self.state_manager.load_previous_end_state(0)
        self.state_manager.set_chapter_zero_end_state(chapter_zero_end_state)

        next_chapter_context = await self.context_service.build_hybrid_context(
            self,
            1,
            None,
            (
                {
                    "chapter_zero_end_state": self.state_manager.get_chapter_zero_end_state()
                }
                if self.state_manager.get_chapter_zero_end_state()
                else None
            ),
            profile_name=ContextProfileName.DEFAULT,
        )
        self.state_manager.set_next_chapter_context(next_chapter_context)

        return True

    async def _save_chapter_text_and_log(
        self, chapter_number: int, final_text: str, raw_llm_log: str | None
    ) -> None:
        """Delegate to :class:`OutputService` for persistence."""
        await self.output_service.save_chapter_text_and_log(
            chapter_number, final_text, raw_llm_log
        )

    async def _save_debug_output(
        self, chapter_number: int, stage_description: str, content: Any
    ) -> None:
        """Delegate to :class:`OutputService` for debug logs."""
        await self.output_service.save_debug_output(
            chapter_number, stage_description, content
        )

    async def perform_deduplication(
        self,
        text_to_dedup: str,
        chapter_number: int,
        context_description: str = "general",
    ) -> tuple[str, int]:
        """Delegates deduplication to the DeduplicationService."""
        return await self.deduplication_service.perform_deduplication(
            text_to_dedup, chapter_number, context_description
        )

    async def _load_previous_end_state(
        self, chapter_number: int
    ) -> ChapterEndState | None:
        """Return the ChapterEndState for ``chapter_number`` via StateManagementService."""
        return await self.state_manager.load_previous_end_state(chapter_number)

    # _run_evaluation_cycle, _ensure_evaluation_result_object,
    # _execute_and_process_evaluation, _execute_and_process_revision,
    # _check_revision_similarity, _handle_max_revisions_reached
    # Methods moved to EvaluationRevisionService:
    # _run_evaluation_cycle, _ensure_evaluation_result_object,
    # _execute_and_process_evaluation, _execute_and_process_revision,
    # _check_revision_similarity, _handle_max_revisions_reached

    async def _run_revision_loop(  # Signature changed to match the new call from _process_and_revise_draft
        self,
        novel_chapter_number: int,
        current_text_to_process: str,
        current_raw_llm_output: str | None,
        plot_point_focus: str,
        plot_point_index: int,
        hybrid_context_for_draft: str,
        chapter_plan: list[SceneDetail] | None,
        initial_patched_spans: list[tuple[int, int]],
        is_from_flawed_source_for_kg: bool,
    ) -> tuple[str | None, str | None, bool, list[tuple[int, int]]]:
        """Delegates the revision loop to the EvaluationRevisionService."""

        # Text preparation (deduplication) before the loop happens in _process_and_revise_draft
        # before calling this _run_revision_loop.
        # current_text_to_process is already prepared (deduplicated).

        plot_outline = self.state_manager.get_plot_outline()

        return await self.evaluation_revision_service.run_revision_loop(
            novel_chapter_number=novel_chapter_number,
            initial_text_to_process=current_text_to_process,  # Pass the prepared text
            initial_raw_llm_output=current_raw_llm_output,
            plot_point_focus=plot_point_focus,
            plot_point_index=plot_point_index,
            hybrid_context_for_draft=hybrid_context_for_draft,
            chapter_plan=chapter_plan,
            initial_patched_spans=initial_patched_spans,
            initial_is_flawed=is_from_flawed_source_for_kg,  # Pass current flaw status
            plot_outline=plot_outline,
        )

    # _handle_no_evaluation_fast_path is kept in orchestrator as it's a bypass
    async def _handle_no_evaluation_fast_path(
        self,
        novel_chapter_number: int,
        initial_text: str,  # This is the original, non-deduped text
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
            self.token_manager._update_rich_display(  # Specific step update
                chapter_num=novel_chapter_number,
                step=f"Ch {novel_chapter_number} - Skipping Revisions (disabled)",
            )
            # Deduplication happens here for the fast path
            deduplicated_text, removed_char_count = await self.perform_deduplication(
                initial_text,  # Use the original initial_text for this path
                novel_chapter_number,
                context_description="no_eval_fast_path",
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
        self.token_manager._update_rich_display(  # Specific step update
            chapter_num=novel_chapter_number,
            step=f"Ch {novel_chapter_number} - Post-Draft De-duplication",
        )
        logger.info(
            f"NANA: Ch {novel_chapter_number} - Applying post-draft de-duplication."
        )
        if text is None:
            return None, False
        # Context "post_draft" or "initial_draft_dedup"
        deduped, removed = await self.perform_deduplication(
            text, novel_chapter_number, "post_draft_initial"
        )
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

    # _prepare_text_for_evaluation was removed as its logic is now inside EvaluationRevisionService's loop.

    # The following methods:
    # _run_revision_loop (the old one with full logic)
    # _handle_max_revisions_reached
    # _execute_and_process_evaluation
    # _execute_and_process_revision
    # _check_revision_similarity
    # _run_evaluation_cycle (was embedded in _execute_and_process_evaluation)
    # _ensure_evaluation_result_object
    # have been moved to or their logic incorporated into EvaluationRevisionService.
    # The NANA_Orchestrator's _run_revision_loop is now a simple delegation.

    # Methods moved to InitializationService:
    # _validate_critical_configs, _initialize_run, _setup_db_and_kg_schema,
    # _ensure_initial_setup, perform_initial_setup.
    # The main call will be to self.initialization_service.setup_and_prepare_run()

    async def _deduplicate_post_revision(
        self,
        novel_chapter_number: int,
        text: str,
        is_flawed: bool,
    ) -> tuple[str, bool]:
        (
            dedup_text_after_rev,
            removed_after_rev,
        ) = await self.perform_deduplication(
            text, novel_chapter_number, "post_revision"
        )
        if removed_after_rev > 0:
            logger.info(
                "NANA: Ch %s - De-duplication after revisions removed %s characters.",
                novel_chapter_number,
                removed_after_rev,
            )
            text = dedup_text_after_rev
            is_flawed = True
            await self._save_debug_output(
                novel_chapter_number,
                "deduplicated_text_after_revision",
                text,
            )
        if len(text) < settings.MIN_ACCEPTABLE_DRAFT_LENGTH:
            logger.warning(
                "NANA: Final chosen text for Ch %s is short (%s chars). Marked as flawed for KG.",
                novel_chapter_number,
                len(text),
            )
            is_flawed = True
        return text, is_flawed

    async def _prepare_chapter_prerequisites(
        self, novel_chapter_number: int
    ) -> PrerequisiteData:
        """Gather planning and context needed before drafting."""
        return await self.prerequisite_service.gather(novel_chapter_number)

    async def _draft_initial_chapter_text(
        self,
        novel_chapter_number: int,
        plot_point_focus: str,
        hybrid_context_for_draft: str,
        chapter_plan: list[SceneDetail] | None,
    ) -> DraftResult:
        # Display update is handled by _accumulate_tokens after this
        # self._update_rich_display(
        #     step=f"Ch {novel_chapter_number} - Drafting Initial Text"
        # )
        result = await self.service_layer.draft_chapter(
            self.state_manager.get_plot_outline(),
            novel_chapter_number,
            plot_point_focus,
            hybrid_context_for_draft,
            chapter_plan,
        )
        initial_draft_text = result.text
        initial_raw_llm_text = result.raw_llm_output
        draft_usage = result.usage  # Assuming DraftResult includes usage
        self._accumulate_tokens(
            Stage.DRAFTING,  # Simpler stage name
            draft_usage,
            chapter_num=novel_chapter_number,
            current_step_for_display=f"Ch {novel_chapter_number} - Drafted",
        )

        if not initial_draft_text:
            logger.error(
                f"NANA: Drafting Agent failed for Ch {novel_chapter_number}. No initial draft produced."
            )
            await self._save_debug_output(
                novel_chapter_number,
                "initial_draft_fail_raw_llm",
                initial_raw_llm_text or "Drafting Agent returned None for raw output.",
            )
            return DraftResult(text=None, raw_llm_output=None, usage=draft_usage)

        await self._save_debug_output(
            novel_chapter_number, "initial_draft", initial_draft_text
        )
        return DraftResult(
            text=initial_draft_text,
            raw_llm_output=initial_raw_llm_text,
            usage=draft_usage,
        )

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
        (
            current_text_to_process,
            current_raw_llm_output,
            is_from_flawed_source_for_kg,
            fast_path_taken,
        ) = await self._handle_initial_draft_processing(
            novel_chapter_number, initial_draft_text, initial_raw_llm_text
        )

        if fast_path_taken:
            return RevisionOutcome(
                text=current_text_to_process,
                raw_llm_output=current_raw_llm_output,
                is_flawed=is_from_flawed_source_for_kg,
            )

        # If fast path not taken, current_text_to_process and is_from_flawed_source_for_kg are already set
        # current_raw_llm_output is also set (it's initial_raw_llm_text from the helper)
        patched_spans: list[tuple[int, int]] = []

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

    async def _handle_initial_draft_processing(
        self,
        novel_chapter_number: int,
        initial_draft_text: str,
        initial_raw_llm_text: str | None,
    ) -> tuple[str | None, str | None, bool, bool]:
        """Handles fast path for no evaluation and initial deduplication."""
        fast_path_result = await self._handle_no_evaluation_fast_path(
            novel_chapter_number,
            initial_draft_text,
            initial_raw_llm_text,
        )
        if fast_path_result is not None:
            text, raw, flawed = fast_path_result
            return text, raw, flawed, True  # True indicates fast path taken

        current_text_to_process: str | None = initial_draft_text
        is_from_flawed_source_for_kg = False

        (
            current_text_to_process,
            flawed_after_dedup,
        ) = await self._deduplicate_post_draft(
            novel_chapter_number,
            current_text_to_process,
        )
        if flawed_after_dedup:
            is_from_flawed_source_for_kg = True

        return (
            current_text_to_process,
            initial_raw_llm_text,
            is_from_flawed_source_for_kg,
            False,
        )

    async def _finalize_and_save_chapter(
        self,
        novel_chapter_number: int,
        final_text_to_process: str,
        final_raw_llm_output: str | None,
        is_from_flawed_source_for_kg: bool,
        fill_in_context: str | None,
    ) -> tuple[str | None, ChapterEndState | None]:
        """Delegate to :class:`OutputService` for finalization."""
        return await self.output_service.finalize_and_save_chapter(
            novel_chapter_number,
            final_text_to_process,
            final_raw_llm_output,
            is_from_flawed_source_for_kg,
            fill_in_context,
        )

    async def _validate_plot_outline(self, novel_chapter_number: int) -> bool:
        plot_outline = self.state_manager.get_plot_outline()
        if (
            not plot_outline
            or not plot_outline.get("plot_points")
            or not plot_outline.get("protagonist_name")
        ):
            logger.error(
                f"NANA: Cannot write Ch {novel_chapter_number}: Plot outline or critical plot data missing."
            )
            self.token_manager._update_rich_display(  # Specific step update
                chapter_num=novel_chapter_number,
                step=f"Ch {novel_chapter_number} Failed - Missing Plot Outline",
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
        fill_in_context = prereq_result.fill_in_context

        if plot_point_focus is None or hybrid_context_for_draft is None:
            self.token_manager._update_rich_display(  # Specific step update
                chapter_num=novel_chapter_number,
                step=f"Ch {novel_chapter_number} Failed - Prerequisites Incomplete",
            )
            return None
        return PrerequisiteData(
            plot_point_focus=plot_point_focus,
            plot_point_index=plot_point_index,
            chapter_plan=chapter_plan,
            hybrid_context_for_draft=hybrid_context_for_draft,
            fill_in_context=fill_in_context,
        )

    async def _process_initial_draft(
        self,
        novel_chapter_number: int,
        draft_result: DraftResult,
    ) -> DraftResult | None:
        initial_draft_text = draft_result.text
        initial_raw_llm_text = draft_result.raw_llm_output
        if initial_draft_text is None:
            self.token_manager._update_rich_display(  # Specific step update
                chapter_num=novel_chapter_number,
                step=f"Ch {novel_chapter_number} Failed - No Initial Draft",
            )
            return None
        return DraftResult(
            text=initial_draft_text,
            raw_llm_output=initial_raw_llm_text,
            usage=draft_result.usage,
        )

    async def _process_revision_result(
        self,
        novel_chapter_number: int,
        revision_result: RevisionOutcome,
    ) -> RevisionOutcome | None:
        processed_text = revision_result.text
        processed_raw_llm = revision_result.raw_llm_output
        is_flawed = revision_result.is_flawed
        if processed_text is None:
            self.token_manager._update_rich_display(  # Specific step update
                chapter_num=novel_chapter_number,
                step=f"Ch {novel_chapter_number} Failed - Revision/Processing Error",
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
        fill_in_context: str | None,
    ) -> str | None:
        """Finalize the chapter then refresh caches and context."""

        combined_fill_ins: list[str] = []
        pending_fill_ins_from_state = self.state_manager.get_pending_fill_ins()
        if pending_fill_ins_from_state:
            combined_fill_ins.extend(pending_fill_ins_from_state)
            self.state_manager.clear_pending_fill_ins()
        if fill_in_context:
            combined_fill_ins.append(
                fill_in_context
            )  # This could also be managed by state_manager if needed

        final_text_result, end_state = await self._finalize_and_save_chapter(
            novel_chapter_number,
            processed_text,
            processed_raw_llm,
            is_flawed,
            "\n".join(combined_fill_ins) or None,
        )

        # self.last_chapter_end_state is managed by state_manager
        # The following logic is now encapsulated in _update_state_after_chapter_finalization
        # (which calls state_manager.update_state_after_chapter_finalization)
        # and _log_chapter_finalization_status
        await self._update_state_after_chapter_finalization(
            novel_chapter_number, end_state
        )
        await self._log_chapter_finalization_status(
            novel_chapter_number, final_text_result, is_flawed
        )
        return final_text_result

    async def _update_state_after_chapter_finalization(
        self, novel_chapter_number: int, end_state: ChapterEndState | None
    ) -> None:
        """Delegates to StateManagementService to update caches and context."""
        await self.state_manager.update_state_after_chapter_finalization(
            novel_chapter_number, end_state
        )
        # _store_pending_fill_ins is called within state_manager's method

    # _store_pending_fill_ins is now part of StateManagementService

    async def _log_chapter_finalization_status(
        self, novel_chapter_number: int, final_text_result: str | None, is_flawed: bool
    ) -> None:
        """Logs the status of chapter finalization."""
        if final_text_result:
            status_message = (
                "Successfully Generated"
                if not is_flawed
                else "Generated (Marked with Flaws)"
            )
            logger.info(
                f"=== NANA: Finished Novel Chapter {novel_chapter_number} - {status_message} ==="
            )
            self.token_manager._update_rich_display(  # Specific step update
                chapter_num=novel_chapter_number,  # Pass chapter_num here
                step=f"Ch {novel_chapter_number} - {status_message}",
            )
        else:
            logger.error(
                f"=== NANA: Failed Novel Chapter {novel_chapter_number} - Finalization/Save Error ==="
            )
            self.token_manager._update_rich_display(  # Specific step update
                chapter_num=novel_chapter_number,  # Pass chapter_num here
                step=f"Ch {novel_chapter_number} Failed - Finalization Error",
            )
        return

    async def run_chapter_generation_process(
        self, novel_chapter_number: int
    ) -> str | None:
        return await run_chapter_pipeline(self, novel_chapter_number)

    # _validate_critical_configs, _initialize_run, _setup_db_and_kg_schema,
    # _ensure_initial_setup, and perform_initial_setup (which was called by _ensure_initial_setup)
    # are now handled by InitializationService.
    # The orchestrator will call `self.initialization_service.setup_and_prepare_run()`

    async def _run_generation(self) -> None:
        """Run the chapter generation runner and log summary stats."""
        logger.info("\n--- NANA: Starting Novel Writing Process ---")

        runner = ChapterGenerationRunner(self)
        await runner.run()

        final_chapter_count_from_db = await chapter_repository.load_chapter_count()
        logger.info("\n--- NANA: Novel writing process finished for this run ---")
        logger.info(
            "NANA: Successfully processed %s chapter(s) in this run.",
            runner.chapters_written,
        )
        logger.info(
            "NANA: Current total chapters in database after this run: %s",
            final_chapter_count_from_db,
        )
        logger.info(
            "NANA: Total LLM tokens generated this run: %s",
            self.token_manager.get_total_tokens_generated_this_run(),
        )
        self.token_manager._update_rich_display(
            chapter_num=self.state_manager.get_chapter_count(), step="Run Finished"
        )

    # _setup_and_prepare_run is now effectively self.initialization_service.setup_and_prepare_run()
    # async def _setup_and_prepare_run(self) -> bool:
    # ... content removed ...

    async def run_novel_generation_loop(self):
        logger.info("--- NANA: Starting Novel Generation Run ---")

        # Call the InitializationService to handle all setup
        if not await self.initialization_service.setup_and_prepare_run():
            # InitializationService logs errors and updates display for critical failures.
            # It will also handle stopping the display if needed for some critical config errors.
            # Ensure display is stopped if not already by a critical error handler in init service.
            if (
                self.display.live and self.display.live.is_started
            ):  # Check if Rich Live instance is active
                await self.display.stop()
            await self.shutdown()  # Ensure services are shut down
            return

        # InitializationService sets up the environment but does not start the
        # Rich progress display. Start it now that initialization succeeded.
        self.display.start()
        try:
            await self._run_generation()
        except Exception as exc:
            logger.critical(
                "NANA: Unhandled exception in orchestrator main loop: %s",
                exc,
                exc_info=True,
            )
            self.token_manager._update_rich_display(
                step="Critical Error in Main Loop"
            )  # Use token_manager
        finally:
            await self.display.stop()
            await self.shutdown()

    async def run_ingestion_process(self, text_file: str) -> None:
        """Ingest existing text and populate the knowledge graph."""
        logger.info("--- NANA: Starting Ingestion Process ---")

        # Use InitializationService for critical config validation and basic setup
        # We don't need the full `setup_and_prepare_run` as genesis phase isn't run for ingestion.
        if not self.initialization_service._validate_critical_configs():
            # display.update_basic might be better if display isn't fully live
            self.display.update_basic(step="Critical Config Error - Halting Ingestion")
            await self.display.stop()
            await self.shutdown()  # Ensure services are shut down
            return

        # Initialize run time and start display for ingestion specifically
        self.token_manager.set_run_start_time(
            time.time()
        )  # Set run time via token manager
        self.display.start()
        self.token_manager._update_rich_display(step="Ingestion Process Started")

        manager = IngestionManager(
            finalize_agent=self.finalize_agent,
            planner_agent=self.planner_agent,
            kg_maintainer=self.kg_maintainer_agent,
            file_manager=self.file_manager,
        )

        await manager.ingest(text_file)
        await self.state_manager.refresh_plot_outline()
        if neo4j_manager.driver is not None:
            await self.state_manager.refresh_knowledge_cache()
        else:
            logger.warning(
                "Neo4j driver not initialized. Skipping knowledge cache refresh."
            )

        # chapter_count is managed by state_manager, refresh it
        self.state_manager.set_chapter_count(
            await chapter_repository.load_chapter_count()
        )
        await self.display.stop()
        await self.shutdown()
        logger.info("NANA: Ingestion process completed.")

    async def shutdown(self) -> None:
        """Close the Neo4j driver and the LLM service."""
        if neo4j_manager.driver is not None:
            await neo4j_manager.close()
        await llm_service.aclose()
