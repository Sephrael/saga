# orchestration/services/evaluation_revision_service.py
"""Service for managing the evaluation and revision cycle of chapter drafts."""

import asyncio
from typing import Any, Awaitable

import structlog
import utils
from agents.comprehensive_evaluator_agent import ComprehensiveEvaluatorAgent
from agents.drafting_agent import DraftingAgent # Not directly used here, but good for context
from agents.finalize_agent import FinalizeAgent # Not directly used here
from agents.kg_maintainer_agent import KGMaintainerAgent
from config import settings
from core.llm_interface import llm_service
from data_access import character_queries, world_queries # For root cause analysis
from initialization.models import PlotOutline
from kg_maintainer.models import (
    EvaluationResult,
    ProblemDetail,
    SceneDetail,
)
from processing.repetition_analyzer import RepetitionAnalyzer
from processing.revision_manager import RevisionManager
from orchestration.models import RevisionOutcome # This is a return type for the orchestrator
from orchestration.token_accountant import Stage # For accumulating tokens

# Assuming NANA_Orchestrator provides access to agents, managers, and token accumulation
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from orchestration.nana_orchestrator import NANA_Orchestrator

logger = structlog.get_logger(__name__)


class EvaluationRevisionService:
    def __init__(self, orchestrator: "NANA_Orchestrator"):
        """
        Initializes the EvaluationRevisionService.

        Args:
            orchestrator: The main NANA_Orchestrator instance to access agents,
                          managers, and configuration.
        """
        self._orchestrator = orchestrator
        # Expose necessary components from orchestrator for easier access
        self._evaluator_agent: ComprehensiveEvaluatorAgent = orchestrator.evaluator_agent
        self._revision_manager: RevisionManager = orchestrator.revision_manager
        self._repetition_analyzer: RepetitionAnalyzer = orchestrator.repetition_analyzer
        self._kg_maintainer_agent: KGMaintainerAgent = orchestrator.kg_maintainer_agent


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

    async def _run_evaluation_cycle(
        self,
        novel_chapter_number: int,
        attempt: int,
        current_text: str,
        plot_point_focus: str,
        plot_point_index: int,
        hybrid_context_for_draft: str,
        patched_spans: list[tuple[int, int]],
        plot_outline: PlotOutline, # Pass plot_outline explicitly
    ) -> tuple[
        EvaluationResult,
        list[ProblemDetail], # Continuity problems (currently not implemented fully in orchestrator)
        dict[str, int] | None, # Eval usage
        dict[str, int] | None, # Continuity usage
        list[ProblemDetail], # Repetition problems
    ]:
        self._orchestrator.token_manager._update_rich_display( # Orchestrator's display method
            chapter_num=novel_chapter_number,
            step=f"Ch {novel_chapter_number} - Evaluation Cycle {attempt} (Parallel)"
        )

        tasks_to_run: list[Awaitable[Any]] = []
        task_names: list[str] = []
        ignore_spans = patched_spans

        if settings.ENABLE_COMPREHENSIVE_EVALUATION:
            scoped_outline = utils.get_scoped_plot_outline(
                plot_outline, novel_chapter_number
            )
            tasks_to_run.append(
                self._evaluator_agent.evaluate_chapter_draft(
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

        # Placeholder for world continuity check if it were to be re-enabled more robustly
        # if settings.ENABLE_WORLD_CONTINUITY_CHECK:
        #     tasks_to_run.append(...)
        #     task_names.append("continuity")

        results = await asyncio.gather(*tasks_to_run)

        eval_result_obj: EvaluationResult | None = None
        eval_usage = None
        continuity_problems: list[ProblemDetail] = [] # Remains empty if no continuity check
        continuity_usage = None # Remains None

        result_idx = 0
        if "evaluation" in task_names:
            eval_result_obj, eval_usage = results[result_idx]
            result_idx += 1
        # if "continuity" in task_names:
        #     continuity_problems, continuity_usage = results[result_idx]

        evaluation_result = self._ensure_evaluation_result_object(eval_result_obj)

        repetition_probs = await self._repetition_analyzer.analyze(current_text)
        # evaluation_result.problems_found.extend(repetition_probs) # This is handled by the caller now
        # if repetition_probs:
        #     evaluation_result.needs_revision = True
        #     evaluation_result.reasons.append("Repetition issues detected")

        return (
            evaluation_result,
            continuity_problems,
            eval_usage,
            continuity_usage,
            repetition_probs, # Return repetition_probs separately
        )

    async def _execute_and_process_evaluation(
        self,
        novel_chapter_number: int,
        attempt: int,
        current_text: str,
        plot_point_focus: str,
        plot_point_index: int,
        hybrid_context_for_draft: str,
        patched_spans: list[tuple[int, int]],
        plot_outline: PlotOutline,
    ) -> tuple[EvaluationResult, list[ProblemDetail], bool]:
        """Runs evaluation and processes the results, returning eval data and if revision is needed."""
        (
            eval_result_obj,
            continuity_problems, # from _run_evaluation_cycle
            eval_usage,
            continuity_usage,
            repetition_problems, # from _run_evaluation_cycle
        ) = await self._run_evaluation_cycle(
            novel_chapter_number,
            attempt,
            current_text,
            plot_point_focus,
            plot_point_index,
            hybrid_context_for_draft,
            patched_spans,
            plot_outline,
        )

        # Token accumulation is handled by the orchestrator
        self._orchestrator._accumulate_tokens(
            f"Ch{novel_chapter_number}-{Stage.EVALUATION.value}-Attempt{attempt}",
            eval_usage,
            chapter_num=novel_chapter_number,
            current_step_for_display=f"Ch {novel_chapter_number} - Eval Attempt {attempt}"
        )
        self._orchestrator._accumulate_tokens(
            f"Ch{novel_chapter_number}-{Stage.CONTINUITY_CHECK.value}-Attempt{attempt}", # Stage name can be generic
            continuity_usage,
            chapter_num=novel_chapter_number,
            current_step_for_display=f"Ch {novel_chapter_number} - Continuity Attempt {attempt}"
        )

        evaluation_result = self._ensure_evaluation_result_object(eval_result_obj)

        # Debug output is handled by the orchestrator
        await self._orchestrator._save_debug_output(
            novel_chapter_number,
            f"evaluation_result_attempt_{attempt}",
            evaluation_result,
        )
        await self._orchestrator._save_debug_output(
            novel_chapter_number,
            f"continuity_problems_attempt_{attempt}",
            continuity_problems, # These are the ones from the eval cycle, could be empty
        )
        await self._orchestrator._save_debug_output(
            novel_chapter_number,
            f"repetition_problems_attempt_{attempt}",
            repetition_problems,
        )

        # Consolidate problems into the main evaluation_result
        if continuity_problems: # If continuity check was enabled and found problems
            logger.warning(
                "NANA: Ch %s (Attempt %s) - Consistency checker found %s issues.",
                novel_chapter_number,
                attempt,
                len(continuity_problems),
            )
            evaluation_result.problems_found.extend(continuity_problems)
            if not evaluation_result.needs_revision: # Ensure needs_revision is true
                evaluation_result.needs_revision = True
            unique_reasons = set(evaluation_result.reasons)
            unique_reasons.add("Continuity issues detected")
            evaluation_result.reasons = sorted(list(unique_reasons))

        if repetition_problems:
            evaluation_result.problems_found.extend(repetition_problems)
            if not evaluation_result.needs_revision: # Ensure needs_revision is true
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
        is_from_flawed_source_for_kg: bool,
        patched_spans: list[tuple[int, int]],
        continuity_problems: list[ProblemDetail], # Pass these along
        plot_outline: PlotOutline,
        # knowledge_cache needed for revision manager
        # Passed via orchestrator ref or directly if preferred
    ) -> tuple[str, str | None, list[tuple[int, int]], bool, bool]:
        # Fetch character and world data (can be cached in state_manager or fetched)
        # For now, assuming direct fetch as per original orchestrator logic
        char_profiles = await character_queries.get_character_profiles_from_db()
        world_building = await world_queries.get_world_building_from_db()

        revision_outcome_tuple, revision_usage = await self._revision_manager.revise_chapter(
            plot_outline,
            char_profiles, # From orchestrator's state_manager.knowledge_cache.characters
            world_building, # From orchestrator's state_manager.knowledge_cache.world
            current_text,
            novel_chapter_number,
            evaluation_result,
            hybrid_context_for_draft,
            chapter_plan,
            revision_cycle=attempt - 1,
            is_from_flawed_source=is_from_flawed_source_for_kg,
            already_patched_spans=patched_spans,
            continuity_problems=continuity_problems, # Pass continuity problems
        )
        self._orchestrator._accumulate_tokens( # Orchestrator's method
            f"Ch{novel_chapter_number}-{Stage.REVISION.value}-Attempt{attempt}",
            revision_usage,
            chapter_num=novel_chapter_number,
            current_step_for_display=f"Ch {novel_chapter_number} - Revision Attempt {attempt}"
        )

        if (
            revision_outcome_tuple # Check if not None
            and revision_outcome_tuple[0] # Check if text is not None or empty
            and len(revision_outcome_tuple[0]) > 50
            and len(revision_outcome_tuple[0]) >= len(current_text) * 0.5
        ):
            new_text, rev_raw_output, new_patched_spans = revision_outcome_tuple
            if new_text and new_text != current_text:
                is_too_similar = await self._check_revision_similarity(
                    novel_chapter_number, attempt, current_text, new_text
                )
                if is_too_similar:
                    return (
                        new_text,
                        rev_raw_output or current_raw_llm_output,
                        new_patched_spans,
                        True, # Revision successful (produced text)
                        True, # Should break loop
                    )

                logger.info(
                    "ERS: Ch %s - Revision attempt %s successful. New text length: %s. Re-evaluating.",
                    novel_chapter_number,
                    attempt,
                    len(new_text),
                )
                await self._orchestrator._save_debug_output( # Orchestrator's method
                    novel_chapter_number,
                    f"revised_text_attempt_{attempt}",
                    new_text,
                )
                return (
                    new_text,
                    rev_raw_output or current_raw_llm_output,
                    new_patched_spans,
                    True, # Revision successful
                    False, # Don't break loop yet
                )
            else:
                logger.error(
                    "ERS: Ch %s - Revision attempt %s did not change text or produced empty. Previous draft used.",
                    novel_chapter_number,
                    attempt,
                )
                return current_text, current_raw_llm_output, patched_spans, False, False
        else:
            logger.error(
                "ERS: Ch %s - Revision attempt %s failed to produce usable text. Previous draft used.",
                novel_chapter_number,
                attempt,
            )
            return current_text, current_raw_llm_output, patched_spans, False, False

    async def _check_revision_similarity(
        self,
        novel_chapter_number: int,
        attempt: int,
        current_text: str,
        new_text: str,
    ) -> bool:
        """Checks if the revised text is too similar to the current text."""
        new_embedding, prev_embedding = await asyncio.gather(
            llm_service.async_get_embedding(new_text),
            llm_service.async_get_embedding(current_text),
        )
        if new_embedding is not None and prev_embedding is not None:
            similarity = utils.numpy_cosine_similarity(prev_embedding, new_embedding)
            if similarity > settings.REVISION_SIMILARITY_ACCEPTANCE:
                logger.warning(
                    "ERS: Ch %s revision attempt %s produced text too similar (score: %.4f). Stopping.",
                    novel_chapter_number,
                    attempt,
                    similarity,
                )
                return True
        return False

    async def _handle_max_revisions_reached(
        self,
        novel_chapter_number: int,
        last_eval_result: EvaluationResult | None,
        plot_outline: PlotOutline,
    ) -> None:
        """Handles the scenario where max revisions are reached."""
        if last_eval_result is not None:
            # Fetch character and world data (can be cached or fetched)
            char_profiles = await character_queries.get_character_profiles_from_db()
            world_building = await world_queries.get_world_building_from_db()

            root_cause = self._revision_manager.identify_root_cause(
                [p.model_dump() for p in last_eval_result.problems_found],
                plot_outline,
                char_profiles,
                world_building,
            )
            if root_cause:
                logger.warning(
                    "ERS: Ch %s - Root cause analysis after max revisions: %s",
                    novel_chapter_number,
                    root_cause,
                )
                lower_cause = root_cause.lower()
                if "character profile" in lower_cause or "world element" in lower_cause:
                    # Delegate KG healing to the orchestrator's agent instance
                    await self._kg_maintainer_agent.heal_and_enrich_kg()


    async def run_revision_loop(
        self,
        novel_chapter_number: int,
        initial_text_to_process: str, # Renamed for clarity
        initial_raw_llm_output: str | None, # Renamed
        plot_point_focus: str,
        plot_point_index: int,
        hybrid_context_for_draft: str,
        chapter_plan: list[SceneDetail] | None,
        initial_patched_spans: list[tuple[int, int]], # Renamed
        initial_is_flawed: bool, # Renamed
        plot_outline: PlotOutline, # Pass plot_outline
    ) -> tuple[str | None, str | None, bool, list[tuple[int, int]]]: # Matches orchestrator return
        """
        Iteratively revises a chapter draft until it passes evaluation or max cycles are reached.
        This is the main entry point for this service from the orchestrator.
        """
        current_text = initial_text_to_process
        current_raw_llm_output = initial_raw_llm_output
        patched_spans = list(initial_patched_spans) # Ensure mutable copy
        is_from_flawed_source_for_kg = initial_is_flawed

        revisions_made = 0
        needs_revision = True
        last_eval_result: EvaluationResult | None = None

        while (
            needs_revision and revisions_made < settings.MAX_REVISION_CYCLES_PER_CHAPTER
        ):
            attempt = revisions_made + 1
            if current_text is None: # Should not happen if initial_text_to_process is valid
                logger.error(
                    "ERS: Ch %s - Text became None before revision cycle %s. Aborting.",
                    novel_chapter_number,
                    attempt,
                )
                return None, None, True, patched_spans # Error state

            # Perform pre-evaluation deduplication at the start of each cycle within the service.
            # The orchestrator's _prepare_text_for_evaluation can be removed.
            # This service now calls the orchestrator's perform_deduplication method.
            (
                current_text, # Update current_text with deduped version
                is_from_flawed_source_for_kg, # Update flaw status
            ) = await self._orchestrator.perform_deduplication(
                current_text, novel_chapter_number, f"pre_eval_cycle_{attempt}"
            )
            if current_text is None: # Should not happen if deduplication handles empty text gracefully
                 logger.error(
                    "ERS: Ch %s - Text became None after pre-evaluation deduplication in cycle %s. Aborting.",
                    novel_chapter_number,
                    attempt,
                )
                 return None, None, True, patched_spans

            (
                evaluation_result,
                continuity_problems, # Captured from evaluation cycle
                needs_revision_after_eval,
            ) = await self._execute_and_process_evaluation(
                novel_chapter_number,
                attempt,
                current_text,
                plot_point_focus,
                plot_point_index,
                hybrid_context_for_draft,
                patched_spans,
                plot_outline,
            )
            last_eval_result = evaluation_result
            needs_revision = needs_revision_after_eval

            if not needs_revision:
                logger.info(
                    "ERS: Ch %s draft passed evaluation (Attempt %s).",
                    novel_chapter_number,
                    attempt,
                )
                self._orchestrator.token_manager._update_rich_display(
                    chapter_num=novel_chapter_number,
                    step=f"Ch {novel_chapter_number} - Passed Evaluation"
                )
                is_from_flawed_source_for_kg = False # Passed, so not flawed from this cycle
                break

            is_from_flawed_source_for_kg = True # Needs revision, so mark as flawed
            logger.warning(
                "ERS: Ch %s draft (Attempt %s) needs revision. Reasons: %s",
                novel_chapter_number,
                attempt,
                "; ".join(evaluation_result.reasons),
            )
            # Display update for "Revision Attempt X" is handled by token accumulation in _execute_and_process_revision

            (
                revised_text,
                revised_raw_llm,
                new_patched_spans,
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
                is_from_flawed_source_for_kg=is_from_flawed_source_for_kg, # Pass current flaw status
                patched_spans=patched_spans,
                continuity_problems=continuity_problems, # Pass these along
                plot_outline=plot_outline,
            )

            current_text = revised_text
            current_raw_llm_output = revised_raw_llm
            patched_spans = new_patched_spans

            if should_break_loop:
                logger.warning("ERS: Ch %s - Breaking revision loop due to similarity or other condition.", novel_chapter_number)
                break # Exit revision loop

            if revision_successful:
                revisions_made += 1
            else: # Revision failed or didn't produce usable text
                revisions_made += 1
                # needs_revision remains true, loop continues or hits max attempts
                # No change to is_from_flawed_source_for_kg here, it's already true
                continue

        if needs_revision: # Max revisions hit or loop broken while still needing revision
            await self._handle_max_revisions_reached(
                novel_chapter_number, last_eval_result, plot_outline
            )
            # is_from_flawed_source_for_kg remains true

        # Final return based on the loop's outcome
        return (
            current_text,
            current_raw_llm_output,
            is_from_flawed_source_for_kg, # Reflects the final state
            patched_spans,
        )

    # The orchestrator will still handle:
    # - _handle_no_evaluation_fast_path (as it's a bypass of this service)
    # - _deduplicate_post_draft (initial deduplication)
    # - _prepare_text_for_evaluation (deduplication before this service's loop)
    # - _deduplicate_post_revision (final deduplication after this service's loop)
    # - Overall flow of calling draft, then this service, then finalization.
