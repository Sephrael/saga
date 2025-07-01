# processing/evaluation_helpers.py
from typing import Any

import structlog
import utils
from config import settings
from core.llm_interface import llm_service
from prompt_renderer import render_prompt

from models import ProblemDetail
from processing.problem_parser import parse_problem_list

logger = structlog.get_logger(__name__)

# Standardized internal category names
INTERNAL_VALID_CATEGORIES = {
    "consistency",
    "plot_arc",
    "thematic_alignment",
    "narrative_depth_and_length",
    "repetition_and_redundancy",
    "meta",
}


def normalize_llm_category(llm_category_str: str) -> str:
    """Normalize category names from LLM output."""
    normalized = llm_category_str.lower().strip().replace(" ", "_")
    if "thematic" in normalized:
        return "thematic_alignment"
    if "narrative_depth" in normalized:
        return "narrative_depth_and_length"
    if "repetition" in normalized or "redundancy" in normalized:
        return "repetition_and_redundancy"
    if normalized in INTERNAL_VALID_CATEGORIES:
        return normalized
    return "meta"


async def parse_llm_evaluation_output(
    json_text: str, chapter_number: int, original_draft_text: str
) -> list[ProblemDetail]:
    """Parse LLM JSON output for chapter evaluation problems."""
    final_problems: list[ProblemDetail] = []

    parsed_data = parse_problem_list(json_text)
    if not parsed_data:
        logger.info("JSON evaluation for Ch %s yielded no problems.", chapter_number)
        return []

    for i, problem_dict in enumerate(parsed_data):
        if not isinstance(problem_dict, ProblemDetail):
            logger.warning(
                "Problem item %s in JSON list for Ch %s is not valid. Skipping. Item: %s",
                i + 1,
                chapter_number,
                problem_dict,
            )
            continue

        problem_meta = ProblemDetail(
            issue_category=normalize_llm_category(
                problem_dict.get("issue_category", "meta")
            ),
            problem_description=problem_dict.get(
                "problem_description", "N/A - Missing description from LLM"
            ),
            quote_from_original_text=problem_dict.get(
                "quote_from_original_text", "N/A - General Issue"
            ),
            quote_char_start=None,
            quote_char_end=None,
            sentence_char_start=None,
            sentence_char_end=None,
            suggested_fix_focus=problem_dict.get(
                "suggested_fix_focus", "N/A - Missing suggestion from LLM"
            ),
            rewrite_instruction=problem_dict.get("rewrite_instruction"),
            severity=problem_dict.get("severity"),
            related_spans=problem_dict.get("related_spans"),
        )

        if (
            problem_meta.issue_category == "meta"
            and str(problem_dict.get("issue_category", "")).lower().strip() != "meta"
        ):
            logger.warning(
                "LLM provided category '%s' in problem %s for Ch %s, which normalized to 'meta'.",
                problem_dict.get("issue_category"),
                i + 1,
                chapter_number,
            )

        quote_text_from_llm = problem_meta.quote_from_original_text
        if (
            "N/A - General Issue" in quote_text_from_llm
            or not quote_text_from_llm.strip()
            or quote_text_from_llm == "N/A"
        ):
            problem_meta.quote_from_original_text = "N/A - General Issue"
        elif utils.spacy_manager.nlp is not None and original_draft_text.strip():
            offsets_tuple = await utils.find_quote_and_sentence_offsets_with_spacy(
                original_draft_text, quote_text_from_llm
            )
            if offsets_tuple:
                q_start, q_end, s_start, s_end = offsets_tuple
                problem_meta.quote_char_start = q_start
                problem_meta.quote_char_end = q_end
                problem_meta.sentence_char_start = s_start
                problem_meta.sentence_char_end = s_end
            else:
                logger.warning(
                    "Ch %s problem %s: Could not find quote via spaCy: '%s'",
                    chapter_number,
                    i + 1,
                    quote_text_from_llm[:50],
                )
        elif not original_draft_text.strip():
            logger.warning(
                "Ch %s problem %s: Original draft text is empty. Cannot find offsets for quote: '%s'",
                chapter_number,
                i + 1,
                quote_text_from_llm[:50],
            )
        else:
            logger.info(
                "Ch %s problem %s: spaCy not available, quote offsets not determined for: '%s'",
                chapter_number,
                i + 1,
                quote_text_from_llm[:50],
            )

        final_problems.append(problem_meta)
    return final_problems


async def perform_llm_comprehensive_evaluation(
    plot_outline: dict[str, Any],
    draft_text: str,
    chapter_number: int,
    plot_point_focus: str | None,
    plot_point_index: int,
    chapter_context: str,
    model_name: str,
) -> tuple[dict[str, Any], dict[str, int] | None]:
    """Call the LLM to evaluate a chapter draft."""
    if not draft_text:
        logger.warning(
            "Comprehensive evaluation skipped for Ch %s: empty draft text.",
            chapter_number,
        )
        return {
            "problems_found_text_output": "Draft is empty.",
            "legacy_consistency_issues": "Skipped (empty draft)",
            "legacy_plot_arc_deviation": "Skipped (empty draft)",
            "legacy_thematic_issues": "Skipped (empty draft)",
            "legacy_narrative_depth_issues": "Skipped (empty draft)",
        }, None

    if plot_point_focus is None:
        logger.warning(
            "Plot point focus not available for Ch %s during comprehensive evaluation.",
            chapter_number,
        )

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

    few_shot_eval_example_str = """**Ignore the narrative details in this example. It shows the required format only.**
[
  {
    "issue_category": "CONSISTENCY",
    "problem_description": "Sága's corporeality is inconsistent with earlier chapters where it is clearly incorporeal.",
    "quote_from_original_text": "Saga darted through the crowd, shoving people aside with its own hands.",
    "suggested_fix_focus": "Rewrite this passage to describe Sága's actions without using physical verbs. Depict its interaction through its control of the environment or remote avatars, consistent with its 'incorporeal' nature."
  },
  {
    "issue_category": "CONSISTENCY",
    "problem_description": "Character Elara states she has never left her village, but her profile mentions she trained at the Royal Academy in the Capital.",
    "quote_from_original_text": "I've never seen anything beyond these village walls,\" Elara sighed, gazing at the distant mountains.",
    "suggested_fix_focus": "Adjust Elara's dialogue to align with her established backstory of training in the Capital, or reconcile this statement with her past (e.g., she's being metaphorical or hiding her past)."
  },
  {
    "issue_category": "PLOT_ARC",
    "problem_description": "The chapter focuses heavily on a minor side character's backstory, which doesn't significantly advance the intended plot point about finding the Sunstone.",
    "quote_from_original_text": "The old merchant then spent a long while recounting his youthful adventures in the spice trade, detailing three different voyages.",
    "suggested_fix_focus": "Reduce the side character's backstory significantly or tie it directly into how it helps or hinders the search for the Sunstone. Ensure the main plot point progression is central."
  },
  {
    "issue_category": "REPETITION_AND_REDUNDANCY",
    "problem_description": "The phrase 'the cost of loyalty' is repeated almost verbatim in three separate paragraphs, diminishing its impact.",
    "quote_from_original_text": "The cost of loyalty was not just in what he gave, but in what he lost.",
    "suggested_fix_focus": "Rephrase the concept in subsequent mentions. Explore different facets of this theme instead of restating the same sentence. For example, show the cost through a character's actions or a difficult choice, rather than repeating the phrase."
  },
  {
    "issue_category": "NARRATIVE_DEPTH_AND_LENGTH",
    "problem_description": "The confrontation with the antagonist feels rushed and lacks emotional impact. The protagonist's internal reaction to the antagonist's reveal is minimal.",
    "quote_from_original_text": "\"It was you all along!\" John exclaimed. The Baron merely smiled. Then they fought.",
    "suggested_fix_focus": "Expand on John's internal thoughts and feelings upon discovering the Baron's betrayal. Show, don't just tell, the emotional weight of this moment. Describe the fight with more detail and tension."
  }
]"""

    prompt = render_prompt(
        "comprehensive_evaluator_agent/evaluate_chapter.j2",
        {
            "enable_no_think": True,
            "chapter_number": chapter_number,
            "novel_title": plot_outline.get("title", "Untitled Novel"),
            "protagonist_name_str": protagonist_name_str,
            "min_length": settings.MIN_ACCEPTABLE_DRAFT_LENGTH,
            "novel_genre": plot_outline.get("genre", "N/A"),
            "novel_theme": plot_outline.get("theme", "N/A"),
            "novel_protagonist": plot_outline.get("protagonist_name", "N/A"),
            "protagonist_arc": plot_outline.get("character_arc", "N/A"),
            "logline": plot_outline.get("logline", "N/A"),
            "plot_points_summary_str": plot_points_summary_str,
            "chapter_context": chapter_context,
            "draft_text": draft_text,
            "few_shot_eval_example_str": few_shot_eval_example_str,
        },
    )

    logger.info(
        "Calling LLM (%s) for comprehensive evaluation of chapter %s (expecting JSON)...",
        model_name,
        chapter_number,
    )
    cleaned_evaluation_text, usage_data = await llm_service.async_call_llm(
        model_name=model_name,
        prompt=prompt,
        temperature=settings.TEMPERATURE_EVALUATION,
        allow_fallback=True,
        stream_to_disk=False,
        auto_clean_response=True,
    )

    no_issues_keywords = [
        "no significant problems found",
        "no issues found",
        "no problems found",
        "no revision needed",
        "no changes needed",
        "all clear",
        "looks good",
        "is fine",
        "is acceptable",
        "passes evaluation",
        "meets criteria",
        "therefore, no revision is needed",
    ]
    is_likely_no_issues_text = False
    if cleaned_evaluation_text.strip():
        normalized_eval_text = cleaned_evaluation_text.lower().strip().replace(".", "")
        for keyword in no_issues_keywords:
            normalized_keyword = keyword.lower().strip().replace(".", "")
            if normalized_keyword == normalized_eval_text or (
                len(normalized_eval_text) < len(normalized_keyword) + 20
                and normalized_keyword in normalized_eval_text
            ):
                is_likely_no_issues_text = True
                break
    eval_output_dict: dict[str, Any]
    if is_likely_no_issues_text:
        logger.info(
            "Heuristic: Evaluation for Ch %s appears to indicate 'no issues': '%s'",
            chapter_number,
            cleaned_evaluation_text[:100],
        )
        eval_output_dict = {
            "problems_found_text_output": cleaned_evaluation_text,
            "legacy_consistency_issues": None,
            "legacy_plot_arc_deviation": None,
            "legacy_thematic_issues": None,
            "legacy_narrative_depth_issues": None,
        }
    elif not cleaned_evaluation_text.strip():
        logger.error(
            "Comprehensive evaluation LLM for Ch %s returned empty text.",
            chapter_number,
        )
        eval_output_dict = {
            "problems_found_text_output": "Evaluation LLM call failed or returned empty.",
            "legacy_consistency_issues": "LLM call failed.",
            "legacy_plot_arc_deviation": "LLM call failed.",
            "legacy_thematic_issues": "LLM call failed.",
            "legacy_narrative_depth_issues": "LLM call failed.",
        }
    else:
        legacy_consistency = (
            "Potential consistency issues."
            if "consistency" in cleaned_evaluation_text.lower()
            else None
        )
        legacy_plot = (
            "Potential plot arc issues."
            if "plot_arc" in cleaned_evaluation_text.lower()
            else None
        )
        legacy_theme = (
            "Potential thematic issues."
            if "thematic" in cleaned_evaluation_text.lower()
            else None
        )
        legacy_depth = (
            "Potential narrative depth/length issues."
            if "narrative_depth" in cleaned_evaluation_text.lower()
            else None
        )
        logger.info(
            "Comprehensive evaluation for Ch %s complete. LLM output (first 200 chars): '%s'",
            chapter_number,
            cleaned_evaluation_text[:200],
        )
        eval_output_dict = {
            "problems_found_text_output": cleaned_evaluation_text,
            "legacy_consistency_issues": legacy_consistency,
            "legacy_plot_arc_deviation": legacy_plot,
            "legacy_thematic_issues": legacy_theme,
            "legacy_narrative_depth_issues": legacy_depth,
        }
    return eval_output_dict, usage_data
