# orchestration/nana_orchestrator.py
"""Primary orchestrator coordinating all SAGA agent interactions."""

import asyncio
import time  # For Rich display updates
from typing import Any, Awaitable # Using typing.Awaitable

import structlog
import utils
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
from data_access import (
    chapter_repository,
    character_queries,
    plot_queries,
    world_queries,
)
from ingestion.ingestion_manager import IngestionManager
from initialization.data_loader import convert_model_to_objects
from initialization.genesis import run_genesis_phase
from initialization.models import PlotOutline
from kg_maintainer.models import (
    EvaluationResult,
    ProblemDetail,
    SceneDetail,
)
from processing.repetition_analyzer import RepetitionAnalyzer
from processing.repetition_tracker import RepetitionTracker
from processing.revision_manager import RevisionManager
from processing.text_deduplicator import TextDeduplicator
from storage.file_manager import FileManager
from ui.rich_display import RichDisplayManager
from utils.plot import get_plot_point_info

from models.agent_models import ChapterEndState
from models.user_input_models import UserStoryInputModel
from orchestration.chapter_flow import run_chapter_pipeline
from orchestration.chapter_generation_runner import ChapterGenerationRunner
from orchestration.knowledge_service import KnowledgeService
from orchestration.models import KnowledgeCache, RevisionOutcome
from orchestration.output_service import OutputService
from orchestration.prerequisite_service import PrerequisiteService
from orchestration.service_layer import ChapterServiceLayer
from orchestration.token_accountant import Stage, TokenAccountant

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

        self.context_service = create_context_service()

        self.plot_outline: PlotOutline = PlotOutline()
        self.chapter_count: int = 0
        self.novel_props_cache: dict[str, Any] = {}
        self.knowledge_cache = KnowledgeCache()
        self.knowledge_service = KnowledgeService(self)
        self.output_service = OutputService(self)
        self.prerequisite_service = PrerequisiteService(self)
        self.token_accountant = TokenAccountant()
        self.total_tokens_generated_this_run: int = 0
        self.completed_plot_points: set[str] = set()

        self.next_chapter_context: str | None = None
        self.last_chapter_end_state: ChapterEndState | None = None
        self.pending_fill_ins: list[str] = []
        self.chapter_zero_end_state: ChapterEndState | None = None
        self.missing_references: dict[str, set[str]] = {
            "characters": set(),
            "locations": set(),
        }

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
        self, stage: str, usage_data: dict[str, int] | TokenUsage | None
    ) -> None:
        self.token_accountant.record_usage(stage, usage_data)
        self.total_tokens_generated_this_run = self.token_accountant.total
        self._update_rich_display()

    async def _generate_plot_points_from_kg(self, count: int) -> None:
        """Delegate to :class:`KnowledgeService` for plot point generation."""
        await self.knowledge_service.generate_plot_points_from_kg(count)

    def load_state_from_user_model(self, model: UserStoryInputModel) -> None:
        """Populate orchestrator state from a user-provided model."""
        plot_outline, _, _ = convert_model_to_objects(model)
        self.plot_outline = plot_outline

    def _update_novel_props_cache(self) -> None:
        """Delegate to :class:`KnowledgeService` to refresh property cache."""
        self.knowledge_service.update_novel_props_cache()

    async def refresh_plot_outline(self) -> None:
        """Reload plot outline from the database via :class:`KnowledgeService`."""
        await self.knowledge_service.refresh_plot_outline()

    async def refresh_knowledge_cache(self) -> None:
        """Reload character profiles and world building caches."""
        await self.knowledge_service.refresh_knowledge_cache()

    async def async_init_orchestrator(self):
        logger.info("NANA Orchestrator async_init_orchestrator started...")
        self._update_rich_display(step="Initializing Orchestrator")
        self.chapter_count = await chapter_repository.load_chapter_count()
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
        self.completed_plot_points = set(await plot_queries.get_completed_plot_points())
        await self.refresh_knowledge_cache()
        logger.info("NANA Orchestrator async_init_orchestrator complete.")
        self._update_rich_display(step="Orchestrator Initialized")

    async def perform_initial_setup(self) -> bool:
        self._update_rich_display(step="Performing Initial Setup")
        logger.info("NANA performing initial setup...")
        (
            self.plot_outline,
            character_profiles,
            world_building,
            usage,
        ) = await run_genesis_phase()
        self._accumulate_tokens(Stage.GENESIS_PHASE.value, usage)

        plot_source = self.plot_outline.get("source", "unknown")
        logger.info(
            f"   Plot Outline and Characters initialized/loaded (source: {plot_source}). "
            f"Title: '{self.plot_outline.get('title', 'N/A')}'. "
            f"Plot Points: {len(self.plot_outline.get('plot_points', []))}"
        )
        world_source = world_building.get("source", "unknown")
        logger.info(f"   World Building initialized/loaded (source: {world_source}).")
        self._update_rich_display(step="Genesis State Bootstrapped")

        self.knowledge_cache.characters = character_profiles
        self.knowledge_cache.world = world_building
        await self.refresh_knowledge_cache()
        self._update_novel_props_cache()
        logger.info("   Initial plot, character, and world data saved to Neo4j.")
        self._update_rich_display(step="Initial State Saved")

        await self.refresh_plot_outline()
        if neo4j_manager.driver is not None:
            await self.refresh_knowledge_cache()
        else:
            logger.warning(
                "Neo4j driver not initialized. Skipping knowledge cache refresh."
            )
        try:
            data = await chapter_repository.get_chapter_data(0)
            if data and data.get("end_state_json"):
                self.chapter_zero_end_state = ChapterEndState.model_validate_json(
                    data["end_state_json"]
                )
        except Exception as exc:
            logger.error("Failed to load chapter 0 end state: %s", exc, exc_info=True)
        self.next_chapter_context = await self.context_service.build_hybrid_context(
            self,
            1,
            None,
            (
                {"chapter_zero_end_state": self.chapter_zero_end_state}
                if self.chapter_zero_end_state
                else None
            ),
            profile_name=ContextProfileName.DEFAULT,
        )

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

    async def _load_previous_end_state(
        self, chapter_number: int
    ) -> ChapterEndState | None:
        """Return the ChapterEndState for ``chapter_number`` if available."""
        if chapter_number <= 0:
            return self.chapter_zero_end_state
        try:
            data = await chapter_repository.get_chapter_data(chapter_number)
        except Exception as exc:  # pragma: no cover - log and skip
            logger.error(
                "Failed to load chapter data for end state",
                chapter=chapter_number,
                error=exc,
                exc_info=True,
            )
            return None
        if data and data.get("end_state_json"):
            try:
                return ChapterEndState.model_validate_json(data["end_state_json"])
            except Exception as exc:  # pragma: no cover - log and skip
                logger.error(
                    "Failed to parse end state JSON",
                    chapter=chapter_number,
                    error=exc,
                    exc_info=True,
                )
        return None

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
        list[ProblemDetail],
    ]:
        self._update_rich_display(
            step=f"Ch {novel_chapter_number} - Evaluation Cycle {attempt} (Parallel)"
        )

        tasks_to_run: list[Awaitable[Any]] = [] # Use Awaitable directly
        task_names: list[str] = []

        ignore_spans = patched_spans

        if settings.ENABLE_COMPREHENSIVE_EVALUATION:
            scoped_outline = utils.get_scoped_plot_outline(
                self.plot_outline, novel_chapter_number
            )
            tasks_to_run.append(
                self.evaluator_agent.evaluate_chapter_draft(
                    scoped_outline,
                    current_text,
                    novel_chapter_number,
                    plot_point_focus,
                    plot_point_index,
                    hybrid_context_for_draft,
                    ignore_spans=ignore_spans,
                )
            )
            task_names.append("evaluation")

        results = await asyncio.gather(*tasks_to_run)

        eval_result_obj: EvaluationResult | None = None
        eval_usage = None
        continuity_problems: list[ProblemDetail] = []
        continuity_usage = None

        result_idx = 0
        if "evaluation" in task_names:
            eval_result_obj, eval_usage = results[result_idx]

        evaluation_result = self._ensure_evaluation_result_object(eval_result_obj)

        repetition_probs = await self.repetition_analyzer.analyze(current_text)
        evaluation_result.problems_found.extend(repetition_probs)
        if repetition_probs:
            evaluation_result.needs_revision = True
            evaluation_result.reasons.append("Repetition issues detected")

        return (
            evaluation_result,
            continuity_problems,
            eval_usage,
            continuity_usage,
            repetition_probs,
        )

    def _ensure_evaluation_result_object(
        self, eval_result_data: Any
    ) -> EvaluationResult:
        """Ensures that the evaluation result is an EvaluationResult object."""
        if isinstance(eval_result_data, EvaluationResult):
            return eval_result_data

        data = eval_result_data or {}
        return EvaluationResult(
            needs_revision=data.get("needs_revision", False),
            reasons=data.get("reasons", []),
            problems_found=data.get("problems_found", []),
            coherence_score=data.get("coherence_score"),
            consistency_issues=data.get("consistency_issues"),
            plot_deviation_reason=data.get("plot_deviation_reason"),
            thematic_issues=data.get("thematic_issues"),
            narrative_depth_issues=data.get("narrative_depth_issues"),
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

    async def _prepare_text_for_evaluation(
        self, novel_chapter_number: int, text: str, is_flawed: bool
    ) -> tuple[str, bool]:
        """Deduplicate text before evaluation and update flaw flag."""
        deduped_text, removed_chars = await self.perform_deduplication(
            text, novel_chapter_number
        )
        if removed_chars > 0:
            logger.info(
                "Ch %s Rev-Loop: De-duplication removed %s chars before evaluation.",
                novel_chapter_number,
                removed_chars,
            )
            is_flawed = True
        return deduped_text, is_flawed

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
        """Iteratively revise a chapter draft until it passes evaluation."""
        revisions_made = 0
        needs_revision = True
        last_eval_result: EvaluationResult | None = None
        while (
            needs_revision and revisions_made < settings.MAX_REVISION_CYCLES_PER_CHAPTER
        ):
            attempt = revisions_made + 1
            if current_text is None:
                logger.error(
                    "NANA: Ch %s - Text became None before processing cycle %s. Aborting chapter.",
                    novel_chapter_number,
                    attempt,
                )
                return None, None, True, patched_spans

            (
                current_text,
                is_from_flawed_source_for_kg,
            ) = await self._prepare_text_for_evaluation(
                novel_chapter_number,
                current_text,
                is_from_flawed_source_for_kg,
            )

            (
                evaluation_result,
                continuity_problems,
                needs_revision_after_eval,
            ) = await self._execute_and_process_evaluation(
                novel_chapter_number,
                attempt,
                current_text,
                plot_point_focus,
                plot_point_index,
                hybrid_context_for_draft,
                patched_spans,
            )
            last_eval_result = evaluation_result
            needs_revision = needs_revision_after_eval

            if not needs_revision:
                logger.info(
                    "NANA: Ch %s draft passed evaluation (Attempt %s). Text is considered good.",
                    novel_chapter_number,
                    attempt,
                )
                self._update_rich_display(
                    step=f"Ch {novel_chapter_number} - Passed Evaluation"
                )
                break  # Exit revision loop

            # If revision is needed, mark as flawed and proceed
            is_from_flawed_source_for_kg = True
            logger.warning(
                "NANA: Ch %s draft (Attempt %s) needs revision. Reasons: %s",
                novel_chapter_number,
                attempt,
                "; ".join(evaluation_result.reasons),
            )
            self._update_rich_display(
                step=f"Ch {novel_chapter_number} - Revision Attempt {attempt}"
            )

            # Perform revision
            (
                current_text,
                current_raw_llm_output,
                patched_spans,
                revision_successful,
                should_break_loop,
            ) = await self._execute_and_process_revision(
                novel_chapter_number=novel_chapter_number,
                attempt=attempt,
                current_text=current_text,
                current_raw_llm_output=current_raw_llm_output,
                evaluation_result=evaluation_result,
                hybrid_context_for_draft=hybrid_context_for_draft,
                chapter_plan=chapter_plan,
                is_from_flawed_source_for_kg=is_from_flawed_source_for_kg,
                patched_spans=patched_spans,
                continuity_problems=continuity_problems,
            )

            if should_break_loop:  # e.g. similarity break
                break

            if revision_successful:
                revisions_made += 1
            else:  # Revision failed or didn't produce usable text
                revisions_made += 1  # Still counts as an attempt
                needs_revision = True  # Ensure loop continues or hits max attempts
                continue  # Try next revision cycle

        # After loop: Handle root cause analysis if still needs revision
        if (
            needs_revision
        ):  # This implies max revisions were hit or loop broken for other reasons
            await self._handle_max_revisions_reached(
                novel_chapter_number, last_eval_result
            )
        elif not needs_revision:  # Passed evaluation
            is_from_flawed_source_for_kg = False  # Reset if it passed

        return (
            current_text,
            current_raw_llm_output,
            is_from_flawed_source_for_kg,  # This reflects the final state
            patched_spans,
        )

    async def _handle_max_revisions_reached(
        self,
        novel_chapter_number: int,
        last_eval_result: EvaluationResult | None,
    ) -> None:
        """Handles the scenario where max revisions are reached."""
        if last_eval_result is not None:
            root_cause = self.revision_manager.identify_root_cause(
                [p.model_dump() for p in last_eval_result.problems_found],
                self.plot_outline,
                await character_queries.get_character_profiles_from_db(),
                await world_queries.get_world_building_from_db(),
            )
            if root_cause:
                logger.warning(
                    "NANA: Ch %s - Root cause analysis after max revisions: %s",
                    novel_chapter_number,
                    root_cause,
                )
                lower_cause = root_cause.lower()
                if "character profile" in lower_cause or "world element" in lower_cause:
                    await self.kg_maintainer_agent.heal_and_enrich_kg()

    async def _execute_and_process_evaluation(
        self,
        novel_chapter_number: int,
        attempt: int,
        current_text: str,
        plot_point_focus: str,
        plot_point_index: int,
        hybrid_context_for_draft: str,
        patched_spans: list[tuple[int, int]],
    ) -> tuple[EvaluationResult, list[ProblemDetail], bool]:
        """Runs evaluation and processes the results, returning eval data and if revision is needed."""
        (
            eval_result_obj,
            continuity_problems,
            eval_usage,
            continuity_usage,
            repetition_problems,
        ) = await self._run_evaluation_cycle(
            novel_chapter_number,
            attempt,
            current_text,
            plot_point_focus,
            plot_point_index,
            hybrid_context_for_draft,
            patched_spans,
        )

        self._accumulate_tokens(
            f"Ch{novel_chapter_number}-{Stage.EVALUATION.value}-Attempt{attempt}",
            eval_usage,
        )
        self._accumulate_tokens(
            f"Ch{novel_chapter_number}-{Stage.CONTINUITY_CHECK.value}-Attempt{attempt}",
            continuity_usage,
        )

        evaluation_result = self._ensure_evaluation_result_object(eval_result_obj)

        await self._save_debug_output(
            novel_chapter_number,
            f"evaluation_result_attempt_{attempt}",
            evaluation_result,
        )
        await self._save_debug_output(
            novel_chapter_number,
            f"continuity_problems_attempt_{attempt}",
            continuity_problems,
        )
        await self._save_debug_output(
            novel_chapter_number,
            f"repetition_problems_attempt_{attempt}",
            repetition_problems,
        )

        if continuity_problems:
            logger.warning(
                "NANA: Ch %s (Attempt %s) - Consistency checker found %s issues.",
                novel_chapter_number,
                attempt,
                len(continuity_problems),
            )
            evaluation_result.problems_found.extend(continuity_problems)
            if not evaluation_result.needs_revision:
                evaluation_result.needs_revision = True
            unique_reasons = set(evaluation_result.reasons)
            unique_reasons.add("Continuity issues detected")
            evaluation_result.reasons = sorted(list(unique_reasons))

        if repetition_problems:
            evaluation_result.problems_found.extend(repetition_problems)
            if not evaluation_result.needs_revision:
                evaluation_result.needs_revision = True
            if "Repetition issues detected" not in evaluation_result.reasons:
                evaluation_result.reasons.append("Repetition issues detected")

        return evaluation_result, continuity_problems, evaluation_result.needs_revision

    async def _execute_and_process_revision(
        self,
        novel_chapter_number: int,
        attempt: int,
        current_text: str,
        current_raw_llm_output: str | None,
        evaluation_result: EvaluationResult,
        hybrid_context_for_draft: str,
        chapter_plan: list[SceneDetail] | None,
        is_from_flawed_source_for_kg: bool,  # Though not directly used, passed for consistency
        patched_spans: list[tuple[int, int]],
        continuity_problems: list[ProblemDetail],
    ) -> tuple[str, str | None, list[tuple[int, int]], bool, bool]:
        """
        Executes a revision attempt and processes its outcome.

        Returns:
            A tuple containing:
            - new_text: The revised text.
            - new_raw_llm_output: The raw LLM output from revision.
            - new_patched_spans: Updated list of patched spans.
            - revision_successful: Boolean indicating if revision produced a change.
            - should_break_loop: Boolean indicating if the revision loop should be exited (e.g., due to similarity).
        """
        revision_outcome, revision_usage = await self.revision_manager.revise_chapter(
            self.plot_outline,
            await character_queries.get_character_profiles_from_db(),
            await world_queries.get_world_building_from_db(),
            current_text,
            novel_chapter_number,
            evaluation_result,
            hybrid_context_for_draft,
            chapter_plan,
            revision_cycle=attempt - 1,
            is_from_flawed_source=is_from_flawed_source_for_kg,
            already_patched_spans=patched_spans,
            continuity_problems=continuity_problems,
        )
        self._accumulate_tokens(
            f"Ch{novel_chapter_number}-{Stage.REVISION.value}-Attempt{attempt}",
            revision_usage,
        )

        if (
            revision_outcome
            and revision_outcome[0]
            and len(revision_outcome[0]) > 50
            and len(revision_outcome[0]) >= len(current_text) * 0.5
        ):
            new_text, rev_raw_output, new_patched_spans = revision_outcome
            if new_text and new_text != current_text:
                is_too_similar = await self._check_revision_similarity(
                    novel_chapter_number, attempt, current_text, new_text
                )
                if is_too_similar:
                    # Accept the similar text but signal to break the loop
                    return (
                        new_text,
                        rev_raw_output or current_raw_llm_output,
                        new_patched_spans,
                        True,
                        True,
                    )

                logger.info(
                    "NANA: Ch %s - Revision attempt %s successful. New text length: %s. Re-evaluating.",
                    novel_chapter_number,
                    attempt,
                    len(new_text),
                )
                await self._save_debug_output(
                    novel_chapter_number,
                    f"revised_text_attempt_{attempt}",
                    new_text,
                )
                return (
                    new_text,
                    rev_raw_output or current_raw_llm_output,
                    new_patched_spans,
                    True,
                    False,
                )
            else:  # Revision produced same text or empty text
                logger.error(
                    "NANA: Ch %s - Revision attempt %s did not change text or produced empty. Proceeding with previous draft, marked as flawed.",
                    novel_chapter_number,
                    attempt,
                )
                # No change, but counts as an attempt. Return original text. Loop continues.
                return current_text, current_raw_llm_output, patched_spans, False, False
        else:  # Revision failed to produce usable text
            logger.error(
                "NANA: Ch %s - Revision attempt %s failed to produce usable text. Proceeding with previous draft.",
                novel_chapter_number,
                attempt,
            )
            # Revision failed. Return original text. Loop continues.
            return current_text, current_raw_llm_output, patched_spans, False, False

    async def _check_revision_similarity(
        self,
        novel_chapter_number: int,
        attempt: int,
        current_text: str,
        new_text: str,
    ) -> bool:
        """Checks if the revised text is too similar to the current text.
        Returns True if too similar (and loop should break), False otherwise.
        """
        new_embedding, prev_embedding = await asyncio.gather(
            llm_service.async_get_embedding(new_text),
            llm_service.async_get_embedding(current_text),
        )
        if new_embedding is not None and prev_embedding is not None:
            similarity = utils.numpy_cosine_similarity(prev_embedding, new_embedding)
            if similarity > settings.REVISION_SIMILARITY_ACCEPTANCE:
                logger.warning(
                    "NANA: Ch %s revision attempt %s produced text too similar to previous (score: %.4f). Stopping revisions.",
                    novel_chapter_number,
                    attempt,
                    similarity,
                )
                return True  # Too similar, break loop
        return False  # Not too similar or similarity check failed

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
            text,
            novel_chapter_number,
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
        self._update_rich_display(
            step=f"Ch {novel_chapter_number} - Drafting Initial Text"
        )
        result = await self.service_layer.draft_chapter(
            self.plot_outline,
            novel_chapter_number,
            plot_point_focus,
            hybrid_context_for_draft,
            chapter_plan,
        )
        initial_draft_text = result.text
        initial_raw_llm_text = result.raw_llm_output
        draft_usage = None
        self._accumulate_tokens(
            f"Ch{novel_chapter_number}-{Stage.DRAFTING.value}", draft_usage
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
        fill_in_context = prereq_result.fill_in_context

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
        fill_in_context: str | None,
    ) -> str | None:
        """Finalize the chapter then refresh caches and context."""

        combined_fill_ins: list[str] = []
        if self.pending_fill_ins:
            combined_fill_ins.extend(self.pending_fill_ins)
            self.pending_fill_ins = []
        if fill_in_context:
            combined_fill_ins.append(fill_in_context)

        final_text_result, end_state = await self._finalize_and_save_chapter(
            novel_chapter_number,
            processed_text,
            processed_raw_llm,
            is_flawed,
            "\n".join(combined_fill_ins) or None,
        )

        self.last_chapter_end_state = end_state
        # The following logic is now encapsulated in _update_state_after_chapter_finalization
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
        """Updates caches and context after a chapter is finalized."""
        if novel_chapter_number % settings.PLOT_POINT_CHAPTER_SPAN == 0:
            pp_focus, pp_index = get_plot_point_info(
                self.plot_outline, novel_chapter_number
            )
            if pp_focus is not None and pp_index >= 0:
                await plot_queries.mark_plot_point_completed(pp_index)
                self.completed_plot_points.add(pp_focus)

        await self.refresh_plot_outline()
        if neo4j_manager.driver is not None:
            await self.refresh_knowledge_cache()
        else:
            logger.warning(
                "Neo4j driver not initialized. Skipping knowledge cache refresh."
            )

        next_hints = {"previous_chapter_end_state": end_state} if end_state else None
        self.next_chapter_context = await self.context_service.build_hybrid_context(
            self,
            novel_chapter_number + 1,
            None,
            next_hints,
            profile_name=ContextProfileName.DEFAULT,
        )
        self._store_pending_fill_ins()

    def _store_pending_fill_ins(self) -> None:
        """Stores pending fill-in chunks from the context service."""
        self.pending_fill_ins = [
            c.text for c in self.context_service.llm_fill_chunks if c.text
        ]

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
        return

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

    async def _initialize_run(self) -> bool:
        """Initialize the run by validating configuration and setting up state."""
        if not self._validate_critical_configs():
            self._update_rich_display(step="Critical Config Error - Halting")
            await self.display.stop()
            return False

        self.total_tokens_generated_this_run = 0
        self.run_start_time = time.time()
        self.display.start()

        if not await self._setup_db_and_kg_schema():
            return False

        try:
            await self.async_init_orchestrator()
        except Exception as exc:
            logger.critical(
                "NANA: Orchestrator async_init failed: %s",
                exc,
                exc_info=True,
            )
            return False
        return True

    async def _setup_db_and_kg_schema(self) -> bool:
        """Initializes Neo4j, creates schema, and loads KG schema."""
        try:
            async with neo4j_manager:
                await neo4j_manager.create_db_schema()
                logger.info("NANA: Neo4j connection and schema verified.")

                await self.kg_maintainer_agent.load_schema_from_db()
                logger.info("NANA: KG schema loaded into maintainer agent.")
                return True
        except Exception as exc:
            logger.critical(
                "NANA: Database or KG schema setup failed: %s",
                exc,
                exc_info=True,
            )
            # Potentially stop display and shutdown here if this is critical path
            # await self.display.stop()
            # await self.shutdown()
            return False

    async def _ensure_initial_setup(self) -> bool:
        """Perform initial setup if plot outline or title is missing."""
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
                "NANA: Core plot data missing or insufficient. Performing initial setup..."
            )
            if not await self.perform_initial_setup():
                logger.critical("NANA: Initial setup failed. Halting generation.")
                self._update_rich_display(step="Initial Setup Failed - Halting")
                return False
            self._update_novel_props_cache()

        return True

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
            self.total_tokens_generated_this_run,
        )
        self._update_rich_display(chapter_num=self.chapter_count, step="Run Finished")

    async def _setup_and_prepare_run(self) -> bool:
        """Combines initialization and ensuring initial setup."""
        if not await self._initialize_run():
            return False
        if not await self._ensure_initial_setup():
            return False
        return True

    async def run_novel_generation_loop(self):
        logger.info("--- NANA: Starting Novel Generation Run ---")

        if not await self._setup_and_prepare_run():
            # display.stop() and shutdown() are called within _initialize_run or _ensure_initial_setup if they fail.
            return

        try:
            await self._run_generation()
        except Exception as exc:
            logger.critical(
                "NANA: Unhandled exception in orchestrator main loop: %s",
                exc,
                exc_info=True,
            )
            self._update_rich_display(step="Critical Error in Main Loop")
        finally:
            await self.display.stop()
            await self.shutdown()

    async def run_ingestion_process(self, text_file: str) -> None:
        """Ingest existing text and populate the knowledge graph."""
        logger.info("--- NANA: Starting Ingestion Process ---")

        if not self._validate_critical_configs():
            await self.display.stop()
            return

        self.display.start()
        self.run_start_time = time.time()

        manager = IngestionManager(
            finalize_agent=self.finalize_agent,
            planner_agent=self.planner_agent,
            kg_maintainer=self.kg_maintainer_agent,
            file_manager=self.file_manager,
        )

        await manager.ingest(text_file)
        await self.refresh_plot_outline()
        if neo4j_manager.driver is not None:
            await self.refresh_knowledge_cache()
        else:
            logger.warning(
                "Neo4j driver not initialized. Skipping knowledge cache refresh."
            )

        self.chapter_count = await chapter_repository.load_chapter_count()
        await self.display.stop()
        await self.shutdown()
        logger.info("NANA: Ingestion process completed.")

    async def shutdown(self) -> None:
        """Close the Neo4j driver and the LLM service."""
        if neo4j_manager.driver is not None:
            await neo4j_manager.close()
        await llm_service.aclose()
