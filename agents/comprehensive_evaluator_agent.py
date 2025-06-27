# comprehensive_evaluator_agent.py
from typing import Any

import structlog
import utils  # MODIFIED: For spaCy functions
from config import settings
from core.llm_interface import llm_service  # MODIFIED
from data_access import chapter_queries
from processing.evaluation_helpers import (
    parse_llm_evaluation_output,
    perform_llm_comprehensive_evaluation,
)

from models import EvaluationResult, ProblemDetail

logger = structlog.get_logger(__name__)


class ComprehensiveEvaluatorAgent:
    def __init__(self, model_name: str = settings.EVALUATION_MODEL):
        self.model_name = model_name
        logger.info(
            f"ComprehensiveEvaluatorAgent initialized with model: {self.model_name}"
        )
        utils.load_spacy_model_if_needed()  # Ensure spaCy model is available

    async def evaluate_chapter_draft(
        self,
        plot_outline: dict[str, Any],
        character_names: list[str],
        world_item_ids_by_category: dict[str, list[str]],
        draft_text: str,
        chapter_number: int,
        plot_point_focus: str | None,
        plot_point_index: int,
        previous_chapters_context: str,
        ignore_spans: list[tuple[int, int]] | None | None = None,
    ) -> tuple[EvaluationResult, dict[str, int] | None]:
        processed_text = utils.remove_spans_from_text(draft_text, ignore_spans or [])
        logger.info(
            f"ComprehensiveEvaluatorAgent evaluating chapter {chapter_number} draft (length: {len(processed_text)} chars)..."
        )
        reasons_for_revision_summary: list[str] = []
        problem_details_list: list[ProblemDetail] = []
        needs_revision = False
        coherence_score: float | None = None
        total_usage_data: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        if not draft_text:
            needs_revision = True
            problem_details_list.append(
                ProblemDetail(
                    issue_category="meta",
                    problem_description="Draft is empty.",
                    quote_from_original_text="N/A - General Issue",
                    quote_char_start=None,
                    quote_char_end=None,
                    sentence_char_start=None,
                    sentence_char_end=None,
                    suggested_fix_focus="Generate content for the chapter.",
                )
            )
            reasons_for_revision_summary.append("Draft is empty.")
        elif len(draft_text) < settings.MIN_ACCEPTABLE_DRAFT_LENGTH:
            needs_revision = True
            problem_details_list.append(
                ProblemDetail(
                    issue_category="narrative_depth_and_length",
                    problem_description=f"Draft is too short ({len(draft_text)} chars). Minimum required: {settings.MIN_ACCEPTABLE_DRAFT_LENGTH}.",
                    quote_from_original_text="N/A - General Issue",
                    quote_char_start=None,
                    quote_char_end=None,
                    sentence_char_start=None,
                    sentence_char_end=None,
                    suggested_fix_focus=f"Expand content significantly across multiple scenes/sections to meet the {settings.MIN_ACCEPTABLE_DRAFT_LENGTH} character target. Focus on adding descriptive detail, character introspection, and dialogue.",
                )
            )
            reasons_for_revision_summary.append(
                f"Draft is too short ({len(draft_text)} chars). Minimum required: {settings.MIN_ACCEPTABLE_DRAFT_LENGTH}."
            )

        current_embedding_task = llm_service.async_get_embedding(draft_text)  # MODIFIED
        if chapter_number > 1:
            prev_embedding = await chapter_queries.get_embedding_from_db(
                chapter_number - 1
            )
            current_embedding = await current_embedding_task
            if current_embedding is not None and prev_embedding is not None:
                coherence_score = utils.numpy_cosine_similarity(
                    current_embedding, prev_embedding
                )
                logger.info(
                    f"Coherence score with previous chapter ({chapter_number - 1}): {coherence_score:.4f}"
                )
                if coherence_score < settings.REVISION_COHERENCE_THRESHOLD:
                    needs_revision = True
                    problem_details_list.append(
                        ProblemDetail(
                            issue_category="consistency",
                            problem_description=f"Low coherence with previous chapter (Score: {coherence_score:.4f}, Threshold: {settings.REVISION_COHERENCE_THRESHOLD}). The narrative flow or tone may be disjointed.",
                            quote_from_original_text="N/A - General Issue",
                            quote_char_start=None,
                            quote_char_end=None,
                            sentence_char_start=None,
                            sentence_char_end=None,
                            suggested_fix_focus="Review the transition from the previous chapter. Ensure stylistic, tonal, and narrative continuity. This might involve adjusting opening scenes or overall pacing.",
                        )
                    )
                    reasons_for_revision_summary.append(
                        f"Low coherence with previous chapter (Score: {coherence_score:.4f}, Threshold: {settings.REVISION_COHERENCE_THRESHOLD})."
                    )
            else:
                logger.warning(
                    f"Could not perform coherence check for ch {chapter_number} (missing current or previous embedding)."
                )
        else:
            logger.info("Skipping coherence check for Chapter 1.")
            await current_embedding_task

        (
            llm_eval_output_dict,
            llm_usage,
        ) = await perform_llm_comprehensive_evaluation(
            plot_outline,
            character_names,
            world_item_ids_by_category,
            processed_text,
            chapter_number,
            plot_point_focus,
            plot_point_index,
            previous_chapters_context,
            self.model_name,
        )
        if llm_usage:
            total_usage_data["prompt_tokens"] += llm_usage.get("prompt_tokens", 0)
            total_usage_data["completion_tokens"] += llm_usage.get(
                "completion_tokens", 0
            )
            total_usage_data["total_tokens"] += llm_usage.get("total_tokens", 0)

        llm_eval_text_output = llm_eval_output_dict.get(
            "problems_found_text_output", ""
        )
        parsed_problems_from_llm = await parse_llm_evaluation_output(
            llm_eval_text_output, chapter_number, draft_text
        )

        if parsed_problems_from_llm:
            problem_details_list.extend(parsed_problems_from_llm)
            needs_revision = True
            category_map_to_reason = {
                "consistency": "Consistency issues identified by LLM.",
                "plot_arc": "Plot Arc deviation identified by LLM.",
                "thematic_alignment": "Thematic Alignment issues identified by LLM.",
                "narrative_depth_and_length": "Narrative Depth/Length issues identified by LLM.",
                "repetition_and_redundancy": "Repetition/Redundancy issues identified by LLM.",
                "meta": "Meta/Uncategorized issues identified by LLM.",
            }
            for prob in parsed_problems_from_llm:
                reason = category_map_to_reason.get(prob.issue_category)
                if reason and reason not in reasons_for_revision_summary:
                    reasons_for_revision_summary.append(reason)
        elif (
            llm_eval_text_output.strip()
            and "no significant problems found" not in llm_eval_text_output.lower()
        ):
            logger.warning(
                f"LLM evaluation for Ch {chapter_number} provided text, but no problems were parsed. Text: '{llm_eval_text_output[:200]}...'"
            )
            problem_details_list.append(
                ProblemDetail(
                    issue_category="meta",
                    problem_description="LLM evaluation output was non-empty but could not be parsed into specific problems.",
                    quote_from_original_text="N/A - LLM Output Parsing",
                    quote_char_start=None,
                    quote_char_end=None,
                    sentence_char_start=None,
                    sentence_char_end=None,
                    suggested_fix_focus="Review LLM evaluation output and parsing logic. The output might not conform to the expected problem format.",
                )
            )
            if "LLM evaluation output unparsable." not in reasons_for_revision_summary:
                reasons_for_revision_summary.append("LLM evaluation output unparsable.")
            needs_revision = True

        unique_reasons_summary = sorted(list(set(reasons_for_revision_summary)))
        validated_problem_details_list: list[ProblemDetail] = []
        for prob_item in problem_details_list:
            if (
                prob_item.quote_from_original_text
                not in [
                    "N/A - General Issue",
                    "N/A - LLM Output Parsing",
                    "N/A - Malformed LLM Output",
                ]
                and prob_item.quote_char_start is None
                and prob_item.quote_from_original_text.strip()
                and draft_text.strip()
            ):
                logger.warning(
                    f"CompEvaluator: Problem quote TEXT for Ch {chapter_number} ('{prob_item.quote_from_original_text[:50]}...') present, "
                    f"but its offsets were NOT found by spaCy utils. Problem desc: {prob_item.problem_description}"
                )
            validated_problem_details_list.append(prob_item)

        logger.info(
            f"Evaluation for Ch {chapter_number} complete. Needs revision: {needs_revision}. Summary of reasons: {'; '.join(unique_reasons_summary) if unique_reasons_summary else 'None'}. Detailed problems found: {len(validated_problem_details_list)}"
        )

        final_eval_result = EvaluationResult(
            needs_revision=needs_revision,
            reasons=unique_reasons_summary,
            problems_found=validated_problem_details_list,
            coherence_score=coherence_score,
            consistency_issues=llm_eval_output_dict.get("legacy_consistency_issues"),
            plot_deviation_reason=llm_eval_output_dict.get("legacy_plot_arc_deviation"),
            thematic_issues=llm_eval_output_dict.get("legacy_thematic_issues"),
            narrative_depth_issues=llm_eval_output_dict.get(
                "legacy_narrative_depth_issues"
            ),
        )
        return final_eval_result, total_usage_data if total_usage_data[
            "total_tokens"
        ] > 0 else None
