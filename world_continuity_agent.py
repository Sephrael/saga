# world_continuity_agent.py
import logging
from typing import Dict, Any, List, Optional, Tuple

import config
from llm_interface import llm_service  # MODIFIED
import utils  # MODIFIED: For spaCy functions
from type import ProblemDetail
from data_access import kg_queries
from prompt_data_getters import (
    get_filtered_character_profiles_for_prompt_plain_text,
    get_filtered_world_data_for_prompt_plain_text,
)

# from parsing_utils import split_text_into_blocks,
# parse_key_value_block  # Removed
import json  # Added for JSON parsing

logger = logging.getLogger(__name__)

PROBLEM_DETAIL_KEY_MAP = {
    "issue_category": "issue_category",
    "problem_description": "problem_description",
    "quote_from_original": "quote_from_original_text",  # Maps to text field
    "suggested_fix_focus": "suggested_fix_focus",
}


class WorldContinuityAgent:
    def __init__(self, model_name: str = config.EVALUATION_MODEL):
        self.model_name = model_name
        logger.info(f"WorldContinuityAgent initialized with model: {self.model_name}")
        utils.load_spacy_model_if_needed()  # Ensure spaCy model is available

    async def _parse_llm_consistency_output(
        self, json_text: str, chapter_number: int, original_draft_text: str
    ) -> List[ProblemDetail]:
        """
        Parses LLM JSON output specifically for consistency problems.
        Expects a JSON array of problem objects.
        Populates character offsets for the quote and its containing sentence
        using spaCy.
        """
        final_problems: List[ProblemDetail] = []
        if not json_text or not json_text.strip():
            logger.info(
                f"Consistency check JSON for Ch {chapter_number} is empty."
                " No problems parsed."
            )
            return []

        try:
            parsed_data = json.loads(json_text)
            if not isinstance(parsed_data, list):
                if (
                    isinstance(parsed_data, dict)
                    and "status" in parsed_data
                    and (
                        "no significant consistency problems found"
                        in parsed_data["status"].lower()
                        or "no significant problems found"
                        in parsed_data["status"].lower()
                    )
                ):
                    logger.info(
                        f"JSON consistency check for Ch {chapter_number} "
                        f"indicates no problems: {parsed_data}"
                    )
                    return []
                if (
                    isinstance(parsed_data, dict)
                    and "problems" in parsed_data
                    and isinstance(parsed_data["problems"], list)
                ):
                    logger.info(
                        f"JSON consistency check for Ch {chapter_number} has"
                        " problems nested under 'problems' key."
                    )
                    parsed_data = parsed_data[
                        "problems"
                    ]  # Process the list of problems
                else:
                    logger.error(
                        f"LLM consistency output was not a JSON list of problems."
                        f" Received type: {type(parsed_data)}. Content:"
                        f" {json_text[:300]}"
                    )
                    final_problems.append(
                        {
                            "issue_category": "consistency",
                            "problem_description": (
                                "LLM output was not a list of consistency problems."
                            ),
                            "quote_from_original_text": (
                                "N/A - LLM Output Format Error"
                            ),
                            "quote_char_start": None,
                            "quote_char_end": None,
                            "sentence_char_start": None,
                            "sentence_char_end": None,
                            "suggested_fix_focus": (
                                "Ensure LLM outputs a JSON list of problem objects"
                                " for consistency check."
                            ),
                        }
                    )
                    return final_problems

            if not parsed_data:  # Empty list from JSON
                logger.info(
                    f"JSON consistency check for Ch {chapter_number} was an empty"
                    " list. No problems parsed."
                )
                return []

        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to decode JSON from LLM consistency output for Ch"
                f" {chapter_number}: {e}. Text: {json_text[:500]}..."
            )
            if (
                "no significant consistency problems found" in json_text.lower()
                or "no significant problems found" in json_text.lower()
            ):
                logger.info(
                    "JSON decode error for consistency check, but text indicates"
                    " no significant problems."
                )
                return []
            final_problems.append(
                {
                    "issue_category": "consistency",
                    "problem_description": (
                        f"Invalid JSON from LLM for consistency check: {e}"
                    ),
                    "quote_from_original_text": "N/A - Invalid JSON",
                    "quote_char_start": None,
                    "quote_char_end": None,
                    "sentence_char_start": None,
                    "sentence_char_end": None,
                    "suggested_fix_focus": (
                        "Review LLM output for JSON validity (consistency check)."
                    ),
                }
            )
            return final_problems

        for i, problem_dict in enumerate(parsed_data):
            if not isinstance(problem_dict, dict):
                logger.warning(
                    f"Consistency problem item {i + 1} in JSON list for Ch"
                    f" {chapter_number} is not a dictionary. Skipping. Item:"
                    f" {problem_dict}"
                )
                continue

            # PROBLEM_DETAIL_KEY_MAP can be used if LLM keys are different, but for now assume direct mapping
            # or LLM is prompted for these exact keys.
            problem_meta: ProblemDetail = {
                "issue_category": "consistency",  # Forced for this agent
                "problem_description": problem_dict.get(
                    "problem_description", "N/A - Missing description"
                ),
                "quote_from_original_text": problem_dict.get(
                    "quote_from_original_text", "N/A - General Issue"
                ),
                "quote_char_start": None,
                "quote_char_end": None,
                "sentence_char_start": None,
                "sentence_char_end": None,
                "suggested_fix_focus": problem_dict.get(
                    "suggested_fix_focus", "N/A - Missing suggestion"
                ),
            }

            # Verify LLM provided 'consistency' or warn if it didn't
            # (though we force it above)
            llm_category_raw = (
                str(problem_dict.get("issue_category", "consistency")).strip().lower()
            )
            if llm_category_raw != "consistency":
                logger.warning(
                    f"WorldContinuityAgent received non-consistency category"
                    f" '{llm_category_raw}' in problem {i + 1} for Ch"
                    f" {chapter_number}. It has been forced to 'consistency'."
                )

            quote_text_from_llm = problem_meta["quote_from_original_text"]
            if (
                "N/A - General Issue" in quote_text_from_llm
                or not quote_text_from_llm.strip()
                or quote_text_from_llm == "N/A"
            ):
                problem_meta["quote_from_original_text"] = "N/A - General Issue"
            elif utils.spacy_manager.nlp is not None and original_draft_text.strip():
                offsets_tuple = await utils.find_quote_and_sentence_offsets_with_spacy(
                    original_draft_text, quote_text_from_llm
                )
                if offsets_tuple:
                    q_start, q_end, s_start, s_end = offsets_tuple
                    problem_meta["quote_char_start"] = q_start
                    problem_meta["quote_char_end"] = q_end
                    problem_meta["sentence_char_start"] = s_start
                    problem_meta["sentence_char_end"] = s_end
                else:
                    logger.warning(
                        f"Ch {chapter_number} consistency problem {i + 1}: Could not find quote via spaCy: '{quote_text_from_llm[:50]}...'"
                    )
            elif not original_draft_text.strip():
                logger.warning(
                    f"Ch {chapter_number} consistency problem {i + 1}: Original"
                    f" draft text is empty for quote search: '",
                    f"{quote_text_from_llm[:50]}...'",
                )
            else:  # spaCy not loaded
                logger.info(
                    f"Ch {chapter_number} consistency problem {i + 1}: spaCy not"
                    f" available, quote offsets not determined for: '",
                    f"{quote_text_from_llm[:50]}...'",
                )

            final_problems.append(problem_meta)
        return final_problems

    async def check_consistency(
        self,
        novel_props: Dict[str, Any],
        draft_text: str,
        chapter_number: int,
        previous_chapters_context: str,
    ) -> Tuple[List[ProblemDetail], Optional[Dict[str, int]]]:
        if not draft_text:
            logger.warning(
                f"WorldContinuityAgent: Consistency check skipped for Ch {chapter_number}: empty draft text."
            )
            return [], None

        logger.info(
            f"WorldContinuityAgent performing focused consistency check for Chapter {chapter_number}..."
        )

        protagonist_name_str = novel_props.get("protagonist_name", "The Protagonist")
        char_profiles_plain_text = (
            await get_filtered_character_profiles_for_prompt_plain_text(
                novel_props, chapter_number - 1
            )
        )
        world_building_plain_text = await get_filtered_world_data_for_prompt_plain_text(
            novel_props, chapter_number - 1
        )

        plot_points_summary_lines = (
            [
                f"- PP {i + 1}: {pp[:100]}..."
                for i, pp in enumerate(novel_props.get("plot_points", []))
            ]
            if novel_props.get("plot_points")
            else ["  - Not available"]
        )
        plot_points_summary_str = "\n".join(plot_points_summary_lines)

        # This whole block will replace the existing few_shot_consistency_example_str
        few_shot_consistency_example_str = """
[
  {
    "issue_category": "consistency",
    "problem_description": "The 'Sunstone' is described as glowing blue in this"
    " chapter, but the world building notes explicitly state all Sunstones are"
    " crimson red.",
    "quote_from_original_text": "She admired the brilliant blue glow of the"
    " Sunstone clutched in her hand.",
    "suggested_fix_focus": "Change the Sunstone's color to 'crimson red' to"
    " align with established world canon."
  },
  {
    "issue_category": "consistency",
    "problem_description": "Character Kael claims to have never met Elara before,"
    " but Previous Chapter Context (KG Fact) states \"Kael | mentored | Elara"
    " (Ch: 3)\".",
    "quote_from_original_text": "\"I do not believe we have crossed paths"
    " before, young one,\" Kael said, peering at Elara.",
    "suggested_fix_focus": "Adjust Kael's dialogue to acknowledge his prior"
    " mentorship of Elara, or introduce a reason for his feigned ignorance"
    " (e.g., memory loss, testing her)."
  }
]
"""
        # Note: The user/developer needs to update the actual LLM prompt to request JSON.
        prompt_lines = []
        if config.ENABLE_LLM_NO_THINK_DIRECTIVE:
            prompt_lines.append("/no_think")

        prompt_lines.extend(
            [
                (
                    f"You are a World & Continuity Expert Editor for Chapter "
                    f'{chapter_number} of the novel "{novel_props.get("title", "Untitled Novel")}" '
                    f"(Protagonist: {protagonist_name_str})."
                ),
                "Your SOLE TASK is to identify specific CONSISTENCY issues in the **Complete Chapter Text** below. Focus on:",
                "- Contradictions with the Plot Outline summary.",
                "- Contradictions with Character Profiles (descriptions, established traits, known status, relationships).",
                "- Contradictions with World Building rules, descriptions, or established lore.",
                "- Contradictions or inconsistencies with the Previous Chapters Context (which includes semantic flow and Key Reliable KG Facts).",
                "- Internal inconsistencies of fact or established detail within THIS chapter's text.",
                "",
                "**Reference Information for CONSISTENCY Check (Summary Format):**",
                "  **Plot Outline Summary:**",
                "  ```text",
                f"  Title: {novel_props.get('title', 'N/A')}",
                f"  Genre: {novel_props.get('genre', 'N/A')}",
                f"  Theme: {novel_props.get('theme', 'N/A')}",
                f"  Protagonist: {novel_props.get('protagonist_name', 'N/A')} ({novel_props.get('character_arc', 'N/A')})",
                f"  Logline: {novel_props.get('logline', 'N/A')}",
                "  Key Plot Points (summary):",
                plot_points_summary_str,
                "  ```",
                "  **Character Profiles (Key Info - check 'prompt_notes' for provisional status):**",
                "  ```text",
                char_profiles_plain_text,
                "  ```",
                "  **World Building Notes (Key Info - check 'prompt_notes' for provisional status):**",
                "  ```text",
                world_building_plain_text,
                "  ```",
                "  **Previous Chapters Context (Includes Semantic Flow & Key Reliable KG Facts for Canon):**",
                "  --- PREVIOUS CONTEXT ---",
                (
                    previous_chapters_context
                    if previous_chapters_context.strip()
                    else "N/A (e.g., Chapter 1 or context retrieval failed)."
                ),
                "  --- END PREVIOUS CONTEXT ---",
                "",
                f"**Complete Chapter {chapter_number} Text (to analyze for consistency):**",
                "--- BEGIN COMPLETE CHAPTER TEXT ---",
                draft_text,
                "--- END COMPLETE CHAPTER TEXT ---",
                "",
                "**Output Format (CRITICAL - JSON ONLY):**",
                "If consistency problems are found, output a JSON array of problem objects.",
                'Each object MUST have these keys: "issue_category" (fixed to "consistency"), "problem_description", "quote_from_original_text", "suggested_fix_focus".',
                'The `quote_from_original_text` must be a VERBATIM quote (10-50 words) from the chapter text. If general or no quote applies, use "N/A - General Issue".',
                'If NO consistency problems are found, output an empty JSON array `[]` or a JSON object like `{"status": "No significant consistency problems found"}`.',
                "",
                "**Follow this example structure for your JSON output precisely:**",
                "```json",
                few_shot_consistency_example_str.strip(),
                "```",
                "",
                "Begin your JSON output now:",
            ]
        )
        prompt = "\n".join(prompt_lines)

        logger.info(
            f"Calling LLM ({self.model_name}) for World/Continuity consistency"
            f" check of chapter {chapter_number} (expecting JSON)..."
        )
        (
            cleaned_consistency_text,
            usage_data,
        ) = await llm_service.async_call_llm(
            model_name=self.model_name,
            prompt=prompt,
            temperature=config.TEMPERATURE_CONSISTENCY_CHECK,
            allow_fallback=True,
            stream_to_disk=False,
            auto_clean_response=True,
        )

        consistency_problems = await self._parse_llm_consistency_output(
            cleaned_consistency_text, chapter_number, draft_text
        )

        logger.info(
            f"World/Continuity consistency check for Ch {chapter_number} found"
            f" {len(consistency_problems)} problems."
        )
        return consistency_problems, usage_data

    async def suggest_canon_corrections(
        self, problems: List[ProblemDetail]
    ) -> List[str]:
        logger.warning(
            "suggest_canon_corrections method is a placeholder and not fully implemented."
        )
        suggestions = []
        for problem in problems:
            if problem.get("issue_category") == "consistency":
                suggestions.append(
                    f"For consistency problem '{problem.get('problem_description', 'N/A')}': Focus on '{problem.get('suggested_fix_focus', 'N/A')}' for canon alignment."
                )
        return suggestions

    async def query_kg_for_contradiction(
        self,
        entity1: str,
        entity2: Optional[str] = None,
        relation: Optional[str] = None,
        chapter_limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        logger.info(
            f"WorldContinuityAgent: Querying KG regarding potential"
            f" contradiction around '{entity1}'..."
        )
        facts = []
        facts.extend(
            await kg_queries.query_kg_from_db(
                subject=entity1,
                chapter_limit=chapter_limit,
                include_provisional=False,
            )
        )
        facts.extend(
            await kg_queries.query_kg_from_db(
                obj_val=entity1,
                chapter_limit=chapter_limit,
                include_provisional=False,
            )
        )

        if entity2 and relation:
            facts.extend(
                await kg_queries.query_kg_from_db(
                    subject=entity1,
                    predicate=relation,
                    obj_val=entity2,
                    chapter_limit=chapter_limit,
                    include_provisional=False,
                )
            )

        unique_facts_strs = set()
        formatted_facts = []
        for fact_dict in facts:
            fact_str = (
                f"{fact_dict.get('subject')} -{fact_dict.get('predicate')}->"
                f" {fact_dict.get('object')} (Ch: {fact_dict.get('chapter_added')}"
                f", Prov: {fact_dict.get('is_provisional')})"
            )
            if fact_str not in unique_facts_strs:
                formatted_facts.append(fact_dict)
                unique_facts_strs.add(fact_str)
        logger.info(
            f"Found {len(formatted_facts)} non-provisional KG facts related to"
            f" '{entity1}' (up to chapter {chapter_limit})."
        )
        return formatted_facts
