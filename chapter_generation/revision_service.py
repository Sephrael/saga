"""Service for revising drafted chapters."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog
import utils
from config import settings
from core.llm_interface import llm_service
from data_access import character_queries, world_queries
from kg_maintainer.models import EvaluationResult

if TYPE_CHECKING:  # pragma: no cover - type hint import
    from orchestration.nana_orchestrator import NANA_Orchestrator

    from models import SceneDetail

logger = structlog.get_logger(__name__)


@dataclass
class RevisionResult:
    """Result of the revision loop."""

    text: str | None
    raw_llm_output: str | None
    is_flawed_source: bool
    patched_spans: list[tuple[int, int]]


class RevisionService:
    """Handle evaluation cycles and revisions for a chapter."""

    def __init__(self, orchestrator: NANA_Orchestrator) -> None:
        self.orchestrator = orchestrator

    async def run_revision_loop(
        self,
        chapter_number: int,
        current_text: str | None,
        current_raw_llm_output: str | None,
        plot_point_focus: str,
        plot_point_index: int,
        hybrid_context_for_draft: str,
        chapter_plan: list[SceneDetail] | None,
        patched_spans: list[tuple[int, int]],
        is_from_flawed_source_for_kg: bool,
    ) -> RevisionResult:
        revisions_made = 0
        needs_revision = True
        while (
            needs_revision and revisions_made < settings.MAX_REVISION_CYCLES_PER_CHAPTER
        ):
            attempt = revisions_made + 1
            if current_text is None:
                logger.error(
                    "NANA: Ch %s - Text became None before processing cycle %s. Aborting chapter.",
                    chapter_number,
                    attempt,
                )
                return RevisionResult(None, None, True, patched_spans)

            (
                eval_result_obj,
                continuity_problems,
                eval_usage,
                continuity_usage,
            ) = await self.orchestrator._run_evaluation_cycle(
                chapter_number,
                attempt,
                current_text,
                plot_point_focus,
                plot_point_index,
                hybrid_context_for_draft,
                patched_spans,
            )

            self.orchestrator._accumulate_tokens(
                f"Ch{chapter_number}-Evaluation-Attempt{attempt}",
                eval_usage,
            )
            self.orchestrator._accumulate_tokens(
                f"Ch{chapter_number}-ContinuityCheck-Attempt{attempt}",
                continuity_usage,
            )

            if isinstance(eval_result_obj, EvaluationResult):
                evaluation_result: EvaluationResult = eval_result_obj
            else:
                evaluation_result = EvaluationResult(**eval_result_obj)
            await self.orchestrator._save_debug_output(
                chapter_number,
                f"evaluation_result_attempt_{attempt}",
                evaluation_result,
            )
            await self.orchestrator._save_debug_output(
                chapter_number,
                f"continuity_problems_attempt_{attempt}",
                continuity_problems,
            )

            if continuity_problems:
                logger.warning(
                    "NANA: Ch %s (Attempt %s) - World Continuity Agent found %s issues.",
                    chapter_number,
                    attempt,
                    len(continuity_problems),
                )
                evaluation_result.problems_found.extend(continuity_problems)
                if not evaluation_result.needs_revision:
                    evaluation_result.needs_revision = True
                unique_reasons = set(evaluation_result.reasons)
                unique_reasons.add(
                    "Continuity issues identified by WorldContinuityAgent."
                )
                evaluation_result.reasons = sorted(list(unique_reasons))

            needs_revision = evaluation_result.needs_revision
            if not needs_revision:
                logger.info(
                    "NANA: Ch %s draft passed evaluation (Attempt %s). Text is considered good.",
                    chapter_number,
                    attempt,
                )
                self.orchestrator._update_rich_display(
                    step=f"Ch {chapter_number} - Passed Evaluation"
                )
                break

            is_from_flawed_source_for_kg = True
            logger.warning(
                "NANA: Ch %s draft (Attempt %s) needs revision. Reasons: %s",
                chapter_number,
                attempt,
                "; ".join(evaluation_result.reasons),
            )
            self.orchestrator._update_rich_display(
                step=f"Ch {chapter_number} - Revision Attempt {attempt}"
            )
            (
                revision_result,
                revision_usage,
            ) = await self.orchestrator.revision_manager.revise_chapter(
                self.orchestrator.plot_outline,
                await character_queries.get_character_profiles_from_db(),
                await world_queries.get_world_building_from_db(),
                current_text,
                chapter_number,
                evaluation_result,
                hybrid_context_for_draft,
                chapter_plan,
                is_from_flawed_source=is_from_flawed_source_for_kg,
                already_patched_spans=patched_spans,
            )
            self.orchestrator._accumulate_tokens(
                f"Ch{chapter_number}-Revision-Attempt{attempt}", revision_usage
            )
            if (
                revision_result
                and revision_result[0]
                and len(revision_result[0]) > 50
                and len(revision_result[0]) >= len(current_text) * 0.5
            ):
                new_text, rev_raw_output, patched_spans = revision_result
                if new_text and new_text != current_text:
                    new_embedding, prev_embedding = await asyncio.gather(
                        llm_service.async_get_embedding(new_text),
                        llm_service.async_get_embedding(current_text),
                    )
                    if new_embedding is not None and prev_embedding is not None:
                        similarity = utils.numpy_cosine_similarity(
                            prev_embedding,
                            new_embedding,
                        )
                        if similarity > settings.REVISION_SIMILARITY_ACCEPTANCE:
                            logger.warning(
                                "NANA: Ch %s revision attempt %s produced text too similar to previous (score: %.4f). Stopping revisions.",
                                chapter_number,
                                attempt,
                                similarity,
                            )
                            current_text = new_text
                            current_raw_llm_output = (
                                rev_raw_output or current_raw_llm_output
                            )
                            break
                    current_text = new_text
                    current_raw_llm_output = rev_raw_output or current_raw_llm_output
                    logger.info(
                        "NANA: Ch %s - Revision attempt %s successful. New text length: %s. Re-evaluating.",
                        chapter_number,
                        attempt,
                        len(current_text),
                    )
                    await self.orchestrator._save_debug_output(
                        chapter_number,
                        f"revised_text_attempt_{attempt}",
                        current_text,
                    )
                    revisions_made += 1
                else:
                    logger.error(
                        "NANA: Ch %s - Revision attempt %s failed to produce usable text. Proceeding with previous draft, marked as flawed.",
                        chapter_number,
                        attempt,
                    )
                    self.orchestrator._update_rich_display(
                        step=f"Ch {chapter_number} - Revision Failed (Retrying)"
                    )
                    revisions_made += 1
                    needs_revision = True
                    continue
            else:
                logger.error(
                    "NANA: Ch %s - Revision attempt %s failed to produce usable text.",
                    chapter_number,
                    attempt,
                )
                self.orchestrator._update_rich_display(
                    step=f"Ch {chapter_number} - Revision Failed (Retrying)"
                )
                revisions_made += 1
                needs_revision = True
                continue

        return RevisionResult(
            text=current_text,
            raw_llm_output=current_raw_llm_output,
            is_flawed_source=is_from_flawed_source_for_kg,
            patched_spans=patched_spans,
        )

    async def deduplicate_post_revision(
        self, chapter_number: int, text: str, is_flawed: bool
    ) -> tuple[str, bool]:
        (
            dedup_text_after_rev,
            removed_after_rev,
        ) = await self.orchestrator.perform_deduplication(
            text,
            chapter_number,
        )
        if removed_after_rev > 0:
            logger.info(
                "NANA: Ch %s - De-duplication after revisions removed %s characters.",
                chapter_number,
                removed_after_rev,
            )
            text = dedup_text_after_rev
            is_flawed = True
            await self.orchestrator._save_debug_output(
                chapter_number,
                "deduplicated_text_after_revision",
                text,
            )
        if len(text) < settings.MIN_ACCEPTABLE_DRAFT_LENGTH:
            logger.warning(
                "NANA: Final chosen text for Ch %s is short (%s chars). Marked as flawed for KG.",
                chapter_number,
                len(text),
            )
            is_flawed = True
        return text, is_flawed
