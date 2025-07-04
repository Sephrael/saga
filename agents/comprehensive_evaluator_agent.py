# agents/comprehensive_evaluator_agent.py
"""Agent for analyzing chapter drafts and generating revision feedback."""

import json
from typing import Any

import structlog
import utils
from config import settings
from core.llm_interface import llm_service
from data_access import chapter_repository
from parsing import parse_problem_list
from processing.evaluation_helpers import (
    parse_llm_evaluation_output,
    perform_llm_comprehensive_evaluation,
)
from prompt_renderer import render_prompt

from models import EvaluationResult, ProblemDetail, SceneDetail

logger = structlog.get_logger(__name__)


class ComprehensiveEvaluatorAgent:
    """Evaluate chapter drafts and identify issues."""

    def __init__(self, model_name: str = settings.EVALUATION_MODEL):
        self.model_name = model_name
        logger.info(
            f"ComprehensiveEvaluatorAgent initialized with model: {self.model_name}"
        )
        utils.load_spacy_model_if_needed()  # Ensure spaCy model is available

    async def evaluate_chapter_draft(
        self,
        plot_outline: dict[str, Any],
        draft_text: str,
        chapter_number: int,
        plot_point_focus: str | None,
        plot_point_index: int,
        chapter_context: str,
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

        current_embedding_task = llm_service.async_get_embedding(draft_text)
        if chapter_number > 1:
            prev_embedding = await chapter_repository.get_embedding(chapter_number - 1)
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
            processed_text,
            chapter_number,
            plot_point_focus,
            plot_point_index,
            chapter_context,
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

    async def _parse_llm_consistency_output(
        self, json_text: str, chapter_number: int, original_draft_text: str
    ) -> list[ProblemDetail]:
        """Parse LLM JSON output for consistency problems."""
        problems = parse_problem_list(json_text, category="consistency")
        if not problems:
            logger.info(
                f"Consistency check for Ch {chapter_number} yielded no problems."
            )
            return []

        for i, prob in enumerate(problems):
            quote_text = prob["quote_from_original_text"]
            if (
                "N/A - General Issue" in quote_text
                or not quote_text.strip()
                or quote_text == "N/A"
            ):
                prob["quote_from_original_text"] = "N/A - General Issue"
            elif utils.spacy_manager.nlp is not None and original_draft_text.strip():
                offsets = await utils.find_quote_and_sentence_offsets_with_spacy(
                    original_draft_text, quote_text
                )
                if offsets:
                    q_start, q_end, s_start, s_end = offsets
                    prob["quote_char_start"] = q_start
                    prob["quote_char_end"] = q_end
                    prob["sentence_char_start"] = s_start
                    prob["sentence_char_end"] = s_end
                else:
                    logger.warning(
                        f"Ch {chapter_number} consistency problem {i + 1}: Could not find quote via spaCy: '{quote_text[:50]}...'"
                    )
            elif not original_draft_text.strip():
                logger.warning(
                    f"Ch {chapter_number} consistency problem {i + 1}: Original draft text is empty. Cannot find offsets for quote: '{quote_text[:50]}...'"
                )
            else:
                logger.info(
                    f"Ch {chapter_number} consistency problem {i + 1}: spaCy not available, quote offsets not determined for: '{quote_text[:50]}...'"
                )

        return problems

    async def check_scene_plan_consistency(
        self,
        plot_outline: dict[str, Any],
        scene_plan: list[SceneDetail],
        chapter_number: int,
        chapter_context: str,
    ) -> tuple[list[ProblemDetail], dict[str, int] | None]:
        """Validate a scene plan before drafting begins."""

        if not scene_plan:
            logger.warning(
                "ComprehensiveEvaluatorAgent: Scene plan consistency check skipped for Ch %s: empty plan.",
                chapter_number,
            )
            return [], None

        protagonist_name_str = plot_outline.get("protagonist_name", "The Protagonist")

        plot_points_summary_lines = (
            [
                f"- PP {i + 1}: {pp[:100]}..."
                for i, pp in enumerate(plot_outline.get("plot_points", []))
            ]
            if plot_outline.get("plot_points")
            else ["  - Not available"]
        )
        plot_points_summary_str = "\n".join(plot_points_summary_lines)

        few_shot_consistency_example_str = """**Ignore the narrative details in this example. It shows the required format only.**
[
  {
    "issue_category": "consistency",
    "problem_description": "The 'Sunstone' is described as glowing blue in this chapter, but the world building notes explicitly state all Sunstones are crimson red.",
    "quote_from_original_text": "She admired the brilliant blue glow of the Sunstone clutched in her hand.",
    "suggested_fix_focus": "Change the Sunstone's color to 'crimson red' to align with established world canon."
  },
  {
    "issue_category": "consistency",
    "problem_description": "Character Kael claims to have never met Elara before, but Previous Chapter Context (KG Fact) states \"Kael | mentored | Elara (Ch: 3)\".",
    "quote_from_original_text": "\"I do not believe we have crossed paths before, young one,\" Kael said, peering at Elara.",
    "suggested_fix_focus": "Adjust Kael's dialogue to acknowledge his prior mentorship of Elara, or introduce a reason for his feigned ignorance (e.g., memory loss, testing her)."
  }
]"""

        prompt = render_prompt(
            "world_continuity_agent/plan_consistency_check.j2",
            {
                "enable_no_think": True,
                "chapter_number": chapter_number,
                "novel_title": plot_outline.get("title", "Untitled Novel"),
                "protagonist_name_str": protagonist_name_str,
                "novel_genre": plot_outline.get("genre", "N/A"),
                "novel_theme": plot_outline.get("theme", "N/A"),
                "novel_protagonist": plot_outline.get("protagonist_name", "N/A"),
                "protagonist_arc": plot_outline.get("character_arc", "N/A"),
                "logline": plot_outline.get("logline", "N/A"),
                "plot_points_summary_str": plot_points_summary_str,
                "chapter_context": chapter_context,
                "scene_plan_json": json.dumps(scene_plan, ensure_ascii=False, indent=2),
                "few_shot_consistency_example_str": few_shot_consistency_example_str,
            },
        )

        logger.info(
            "Calling LLM (%s) for scene plan consistency check of chapter %s (expecting JSON)...",
            self.model_name,
            chapter_number,
        )
        cleaned_consistency_text, usage_data = await llm_service.async_call_llm(
            model_name=self.model_name,
            prompt=prompt,
            temperature=settings.TEMPERATURE_CONSISTENCY_CHECK,
            allow_fallback=True,
            stream_to_disk=False,
            auto_clean_response=True,
        )

        consistency_problems = await self._parse_llm_consistency_output(
            cleaned_consistency_text, chapter_number, json.dumps(scene_plan)
        )

        logger.info(
            "Scene plan consistency check for Ch %s found %s problems.",
            chapter_number,
            len(consistency_problems),
        )
        return consistency_problems, usage_data
