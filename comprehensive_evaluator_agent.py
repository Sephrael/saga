# comprehensive_evaluator_agent.py
import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple

import config
import llm_interface
import utils
from type import EvaluationResult, ProblemDetail
# from state_manager import state_manager # No longer directly used
from data_access import chapter_queries # For get_embedding_from_db
from prompt_data_getters import (
    get_filtered_character_profiles_for_prompt_plain_text,
    get_filtered_world_data_for_prompt_plain_text,
    get_reliable_kg_facts_for_drafting_prompt
)
from parsing_utils import split_text_into_blocks, parse_key_value_block

logger = logging.getLogger(__name__)

# Key map for parsing ProblemDetail from LLM plain text
PROBLEM_DETAIL_KEY_MAP = {
    "issue_category": "issue_category",
    "problem_description": "problem_description",
    "quote_from_original": "quote_from_original",
    "suggested_fix_focus": "suggested_fix_focus"
}

class ComprehensiveEvaluatorAgent:
    def __init__(self, model_name: str = config.EVALUATION_MODEL):
        self.model_name = model_name
        logger.info(f"ComprehensiveEvaluatorAgent initialized with model: {self.model_name}")

    def _parse_llm_evaluation_output(self, text: str, chapter_number: int) -> List[ProblemDetail]:
        """
        Parses LLM plain text output for chapter evaluation problems using generalized utilities.
        """
        problems_data: List[Dict[str, Any]] = []
        if not text or not text.strip() or "no significant problems found" in text.lower():
            logger.info(f"Plain text evaluation for Ch {chapter_number} is empty or indicates no problems. No problems parsed.")
            return []

        # Split by "---" separator for individual problem blocks
        problem_blocks_text = split_text_into_blocks(text, separator_regex_str=r'\n\s*---\s*\n')

        for block_num, block_content in enumerate(problem_blocks_text):
            if not block_content.strip():
                continue

            logger.debug(f"Parsing problem block {block_num+1} for Ch {chapter_number}:\n{block_content[:150]}...")

            parsed_problem_dict = parse_key_value_block(
                block_text_or_lines=block_content,
                key_map=PROBLEM_DETAIL_KEY_MAP,
                list_internal_keys=[] # No list keys in ProblemDetail
            )

            # Validate required keys
            required_keys_internal = set(PROBLEM_DETAIL_KEY_MAP.values())
            missing_keys = required_keys_internal - set(parsed_problem_dict.keys())

            if missing_keys:
                logger.warning(f"Could not parse all required fields from problem block {block_num+1} for Ch {chapter_number}. Missing: {missing_keys}. Block: '{block_content[:150]}...'")
                # Create a meta-problem indicating parsing failure for this block
                problems_data.append({
                    "issue_category": "meta",
                    "problem_description": f"Malformed problem block from LLM: {block_content}",
                    "quote_from_original": "N/A - Malformed LLM Output", # Critical for patching
                    "suggested_fix_focus": "Review LLM output and prompt for evaluation to ensure correct formatting."
                })
                continue
            
            # Sanitize quote_from_original: if empty or whitespace after parsing, set to "N/A - General Issue"
            if not parsed_problem_dict.get("quote_from_original", "").strip():
                parsed_problem_dict["quote_from_original"] = "N/A - General Issue"
                logger.debug(f"Problem block {block_num+1} for Ch {chapter_number} had empty quote, defaulted to 'N/A - General Issue'. Problem: {parsed_problem_dict['problem_description']}")


            # Validate issue_category
            category = str(parsed_problem_dict.get("issue_category", "meta")).strip().lower()
            valid_categories = ["consistency", "plot_arc", "thematic", "narrative_depth", "meta"]
            if category not in valid_categories:
                logger.warning(f"Parsed unknown issue category '{category}' in block {block_num+1} for Ch {chapter_number}. Defaulting to 'meta'.")
                parsed_problem_dict["issue_category"] = "meta"
            else:
                parsed_problem_dict["issue_category"] = category # Use validated category

            problems_data.append(parsed_problem_dict)

        return [prob for prob in problems_data if isinstance(prob, dict)] # type: ignore


    async def _perform_llm_comprehensive_evaluation(
        self,
        novel_props: Dict[str, Any],
        draft_text: str,
        chapter_number: int,
        plot_point_focus: Optional[str],
        plot_point_index: int,
        previous_chapters_context: str
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, int]]]:
        """
        Performs the LLM-based comprehensive evaluation.
        Returns evaluation details and LLM usage data.
        """
        if not draft_text:
            logger.warning(f"Comprehensive evaluation skipped for Ch {chapter_number}: empty draft text.")
            return {
                "problems_found_text_output": "Draft is empty.",
                "legacy_consistency_issues": "Skipped (empty draft)",
                "legacy_plot_arc_deviation": "Skipped (empty draft)",
                "legacy_thematic_issues": "Skipped (empty draft)",
                "legacy_narrative_depth_issues": "Skipped (empty draft)"
            }, None

        plot_point_focus_str = plot_point_focus if plot_point_focus is not None else "Not available for this chapter."
        if plot_point_focus is None:
            logger.warning(f"Plot point focus not available for Ch {chapter_number} during comprehensive evaluation.")

        novel_theme_str = novel_props.get('theme', 'Not specified')
        novel_genre_str = novel_props.get('genre', 'Not specified')
        protagonist_arc_str = novel_props.get('character_arc', 'Not specified')
        protagonist_name_str = novel_props.get('protagonist_name', 'The Protagonist')

        char_profiles_plain_text = await get_filtered_character_profiles_for_prompt_plain_text(novel_props, chapter_number - 1)
        world_building_plain_text = await get_filtered_world_data_for_prompt_plain_text(novel_props, chapter_number - 1)
        kg_check_results_text = await get_reliable_kg_facts_for_drafting_prompt(novel_props, chapter_number, None)


        prompt = f"""/no_think
You are a Master Editor evaluating Chapter {chapter_number} of a novel titled "{novel_props.get('title', 'Untitled Novel')}" (Protagonist: {protagonist_name_str}).
Analyze the **Complete Chapter Text** provided below.
Your task is to identify specific issues related to:
1.  **CONSISTENCY**: Contradictions with Plot Outline, Character Profiles, World Building, Key Reliable KG Facts, Previous Context, or internal inconsistencies within THIS chapter.
2.  **PLOT_ARC**: How well this chapter addresses or advances its Intended Plot Point: "{plot_point_focus_str}" (Plot Point #{plot_point_index + 1}).
3.  **THEMATIC_ALIGNMENT**: Alignment with the novel's core elements (Genre: {novel_genre_str}, Theme: {novel_theme_str}, Protagonist's Arc: {protagonist_arc_str}).
4.  **NARRATIVE_DEPTH_AND_LENGTH**: Sufficiency of descriptive detail, character introspection, dialogue development, pacing, and overall length (target: at least {config.MIN_ACCEPTABLE_DRAFT_LENGTH} characters).

**Reference Information for CONSISTENCY Check (Summary Format):**
  **Plot Outline Summary:**
  ```text
  Title: {novel_props.get('title', 'N/A')}
  Genre: {novel_props.get('genre', 'N/A')}
  Theme: {novel_props.get('theme', 'N/A')}
  Protagonist: {novel_props.get('protagonist_name', 'N/A')} ({novel_props.get('character_arc', 'N/A')})
  Logline: {novel_props.get('logline', 'N/A')}
  Key Plot Points (summary):
  {chr(10).join([f"- PP {i+1}: {pp[:100]}..." for i, pp in enumerate(novel_props.get('plot_points', []))]) if novel_props.get('plot_points') else "  - Not available"}
  ```
  **Character Profiles (Key Info - check 'prompt_notes' for provisional status):**
  ```text
  {char_profiles_plain_text}
  ```
  **World Building Notes (Key Info - check 'prompt_notes' for provisional status):**
  ```text
  {world_building_plain_text}
  ```
  {kg_check_results_text}
  **Previous Chapters Context (Semantic Flow & KG Facts for Canon):**
  --- PREVIOUS CONTEXT ---
  {previous_chapters_context if previous_chapters_context.strip() else "N/A (e.g., Chapter 1 or context retrieval failed)."}
  --- END PREVIOUS CONTEXT ---

**Complete Chapter {chapter_number} Text (to analyze):**
--- BEGIN COMPLETE CHAPTER TEXT ---
{draft_text}
--- END COMPLETE CHAPTER TEXT ---

**Output Format (CRITICAL):**
Provide your evaluation as plain text. If problems are found, list each problem individually using the following format (ensure keys like "ISSUE CATEGORY" are used, case can vary):

ISSUE CATEGORY: [consistency | plot_arc | thematic | narrative_depth | meta]
PROBLEM DESCRIPTION: [A concise description of the specific issue.]
QUOTE FROM ORIGINAL: [**A VERBATIM quote (10-50 words) from the "Complete Chapter Text" that clearly illustrates this specific problem.** If the issue is general (e.g., overall length, pervasive tone issue) and no single quote captures it, or if a quote is truly inapplicable, use "N/A - General Issue".]
SUGGESTED FIX FOCUS: [Brief guidance on what the revision for this specific quote/issue should focus on (e.g., "Clarify character's motivation", "Expand description of setting to enhance atmosphere").]
---

If multiple problems are found, separate each problem block with a line containing only "---".
If NO problems are found for a category or overall, output ONLY the phrase: "No significant problems found."
"""
        logger.info(f"Calling LLM ({self.model_name}) for comprehensive evaluation of chapter {chapter_number}...")
        raw_evaluation_text, usage_data = await llm_interface.async_call_llm(
            model_name=self.model_name,
            prompt=prompt,
            temperature=0.3,
            allow_fallback=True,
            stream_to_disk=False 
        )

        cleaned_evaluation_text = llm_interface.clean_model_response(raw_evaluation_text)

        no_issues_keywords = [
            "no significant problems found", "no issues found", "no problems found",
            "no revision needed", "no changes needed", "all clear", "looks good",
            "is fine", "is acceptable", "passes evaluation", "meets criteria",
            "therefore, no revision is needed"
        ]
        is_likely_no_issues_text = False
        if cleaned_evaluation_text.strip(): 
            normalized_eval_text = cleaned_evaluation_text.lower().strip().replace('.', '') 
            for keyword in no_issues_keywords:
                normalized_keyword = keyword.lower().strip().replace('.', '')
                if normalized_keyword == normalized_eval_text or \
                   (len(normalized_eval_text) < len(normalized_keyword) + 20 and normalized_keyword in normalized_eval_text): 
                     is_likely_no_issues_text = True
                     break

        eval_output_dict: Dict[str, Any]
        if is_likely_no_issues_text:
            logger.info(f"Heuristic: Evaluation for Ch {chapter_number} appears to indicate 'no issues': '{cleaned_evaluation_text[:100]}...'")
            eval_output_dict = {
                "problems_found_text_output": cleaned_evaluation_text, 
                "legacy_consistency_issues": None, "legacy_plot_arc_deviation": None,
                "legacy_thematic_issues": None, "legacy_narrative_depth_issues": None
            }
        elif not cleaned_evaluation_text.strip(): 
            logger.error(f"Comprehensive evaluation LLM for Ch {chapter_number} returned empty text after cleaning. Raw input: '{raw_evaluation_text[:200]}...'")
            eval_output_dict = {
                "problems_found_text_output": "Evaluation LLM call failed or returned empty.",
                "legacy_consistency_issues": "LLM call failed.", "legacy_plot_arc_deviation": "LLM call failed.",
                "legacy_thematic_issues": "LLM call failed.", "legacy_narrative_depth_issues": "LLM call failed."
            }
        else:
            legacy_consistency = "Potential consistency issues." if "consistency" in cleaned_evaluation_text.lower() else None
            legacy_plot = "Potential plot arc issues." if "plot_arc" in cleaned_evaluation_text.lower() else None
            legacy_theme = "Potential thematic issues." if "thematic" in cleaned_evaluation_text.lower() else None
            legacy_depth = "Potential narrative depth/length issues." if "narrative_depth" in cleaned_evaluation_text.lower() else None
            logger.info(f"Comprehensive evaluation for Ch {chapter_number} complete. LLM output (first 200 chars): '{cleaned_evaluation_text[:200]}...'")
            eval_output_dict = {
                "problems_found_text_output": cleaned_evaluation_text,
                "legacy_consistency_issues": legacy_consistency,
                "legacy_plot_arc_deviation": legacy_plot,
                "legacy_thematic_issues": legacy_theme,
                "legacy_narrative_depth_issues": legacy_depth
            }
        return eval_output_dict, usage_data


    async def evaluate_chapter_draft(
        self,
        novel_props: Dict[str, Any], 
        draft_text: str,
        chapter_number: int,
        plot_point_focus: Optional[str],
        plot_point_index: int,
        previous_chapters_context: str
    ) -> Tuple[EvaluationResult, Optional[Dict[str, int]]]:
        """
        Evaluates a chapter draft, incorporating pre-LLM checks and LLM-based comprehensive evaluation.
        Returns EvaluationResult and LLM usage data.
        """
        logger.info(f"ComprehensiveEvaluatorAgent evaluating chapter {chapter_number} draft (length: {len(draft_text)} chars)...")

        reasons_for_revision_summary: list[str] = []
        problem_details_list: List[ProblemDetail] = []
        needs_revision = False
        coherence_score: Optional[float] = None
        total_usage_data: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


        if not draft_text:
            needs_revision = True
            problem_details_list.append({
                "issue_category": "meta", "problem_description": "Draft is empty.",
                "quote_from_original": "N/A - General Issue", 
                "suggested_fix_focus": "Generate content for the chapter."
            })
            reasons_for_revision_summary.append("Draft is empty.")
        elif len(draft_text) < config.MIN_ACCEPTABLE_DRAFT_LENGTH:
            needs_revision = True
            problem_details_list.append({
                "issue_category": "narrative_depth",
                "problem_description": f"Draft is too short ({len(draft_text)} chars). Minimum required: {config.MIN_ACCEPTABLE_DRAFT_LENGTH}.",
                "quote_from_original": "N/A - General Issue", 
                "suggested_fix_focus": f"Expand content significantly across multiple scenes/sections to meet the {config.MIN_ACCEPTABLE_DRAFT_LENGTH} character target. Focus on adding descriptive detail, character introspection, and dialogue."
            })
            reasons_for_revision_summary.append(f"Draft is too short ({len(draft_text)} chars). Minimum required: {config.MIN_ACCEPTABLE_DRAFT_LENGTH}.")

        current_embedding_task = llm_interface.async_get_embedding(draft_text)
        if chapter_number > 1:
            prev_embedding = await chapter_queries.get_embedding_from_db(chapter_number - 1) # MODIFIED
            current_embedding = await current_embedding_task 
            if current_embedding is not None and prev_embedding is not None:
                coherence_score = utils.numpy_cosine_similarity(current_embedding, prev_embedding)
                logger.info(f"Coherence score with previous chapter ({chapter_number-1}): {coherence_score:.4f}")
                if coherence_score < config.REVISION_COHERENCE_THRESHOLD:
                    needs_revision = True
                    problem_details_list.append({
                        "issue_category": "consistency", 
                        "problem_description": f"Low coherence with previous chapter (Score: {coherence_score:.4f}, Threshold: {config.REVISION_COHERENCE_THRESHOLD}). The narrative flow or tone may be disjointed.",
                        "quote_from_original": "N/A - General Issue",
                        "suggested_fix_focus": "Review the transition from the previous chapter. Ensure stylistic, tonal, and narrative continuity. This might involve adjusting opening scenes or overall pacing."
                    })
                    reasons_for_revision_summary.append(f"Low coherence with previous chapter (Score: {coherence_score:.4f}, Threshold: {config.REVISION_COHERENCE_THRESHOLD}).")
            else:
                logger.warning(f"Could not perform coherence check for ch {chapter_number} (missing current or previous embedding).")
        else: 
            logger.info("Skipping coherence check for Chapter 1.")
            await current_embedding_task 

        llm_eval_output_dict, llm_usage = await self._perform_llm_comprehensive_evaluation(
            novel_props, draft_text, chapter_number, plot_point_focus, plot_point_index, previous_chapters_context
        )
        if llm_usage:
            total_usage_data["prompt_tokens"] += llm_usage.get("prompt_tokens", 0)
            total_usage_data["completion_tokens"] += llm_usage.get("completion_tokens", 0)
            total_usage_data["total_tokens"] += llm_usage.get("total_tokens", 0)

        llm_eval_text_output = llm_eval_output_dict.get("problems_found_text_output", "")
        parsed_problems_from_llm = self._parse_llm_evaluation_output(llm_eval_text_output, chapter_number)

        if parsed_problems_from_llm:
            problem_details_list.extend(parsed_problems_from_llm)
            needs_revision = True 
            category_map_to_reason = {
                "consistency": "Consistency issues identified by LLM.",
                "plot_arc": "Plot Arc deviation identified by LLM.",
                "thematic": "Thematic issues identified by LLM.",
                "narrative_depth": "Narrative Depth/Length issues identified by LLM.",
                "meta": "Meta/Uncategorized issues identified by LLM."
            }
            for prob in parsed_problems_from_llm:
                reason = category_map_to_reason.get(prob["issue_category"])
                if reason and reason not in reasons_for_revision_summary: 
                    reasons_for_revision_summary.append(reason)
        elif llm_eval_text_output.strip() and "no significant problems found" not in llm_eval_text_output.lower():
            logger.warning(f"LLM evaluation for Ch {chapter_number} provided text, but no problems were parsed. Text: '{llm_eval_text_output[:200]}...'")
            problem_details_list.append({
                "issue_category": "meta",
                "problem_description": "LLM evaluation output was non-empty but could not be parsed into specific problems.",
                "quote_from_original": "N/A - LLM Output Parsing",
                "suggested_fix_focus": "Review LLM evaluation output and parsing logic. The output might not conform to the expected problem format."
            })
            if "LLM evaluation output unparsable." not in reasons_for_revision_summary:
                 reasons_for_revision_summary.append("LLM evaluation output unparsable.")
            needs_revision = True 

        unique_reasons_summary = sorted(list(set(reasons_for_revision_summary)))
        validated_problem_details_list: List[ProblemDetail] = []
        for prob_item in problem_details_list:
            quote = prob_item["quote_from_original"]
            if not isinstance(quote, str):
                logger.warning(f"Problem quote for Ch {chapter_number} was not a string (type: {type(quote)}): '{str(quote)[:50]}...'. Converting. Problem desc: {prob_item['problem_description']}")
                prob_item["quote_from_original"] = "N/A - Invalid Quote Type" 
                quote = prob_item["quote_from_original"]
            non_general_quote_markers = ["N/A - General Issue", "N/A - LLM Output Parsing", "N/A - Parser Exception", "N/A - Invalid Quote Type"]
            if quote not in non_general_quote_markers and quote.strip() and draft_text: 
                if not (10 <= len(quote) <= 300): 
                     logger.warning(f"Problem quote for Ch {chapter_number} has unusual length ({len(quote)}): '{quote[:50]}...'")
                if quote not in draft_text:
                    logger.warning(
                        f"Problem quote for Ch {chapter_number} was NOT found VERBATIM in chapter text: '{quote[:50]}...'. "
                        f"This will prevent direct patching for this issue. Problem desc: {prob_item['problem_description']}"
                    )
            validated_problem_details_list.append(prob_item)

        logger.info(f"Evaluation for Ch {chapter_number} complete. Needs revision: {needs_revision}. Summary of reasons: {'; '.join(unique_reasons_summary) if unique_reasons_summary else 'None'}. Detailed problems found: {len(validated_problem_details_list)}")

        final_eval_result: EvaluationResult = {
            "needs_revision": needs_revision,
            "reasons": unique_reasons_summary,
            "problems_found": validated_problem_details_list, 
            "coherence_score": coherence_score,
            "consistency_issues": llm_eval_output_dict.get("legacy_consistency_issues"),
            "plot_deviation_reason": llm_eval_output_dict.get("legacy_plot_arc_deviation"),
            "thematic_issues": llm_eval_output_dict.get("legacy_thematic_issues"),
            "narrative_depth_issues": llm_eval_output_dict.get("legacy_narrative_depth_issues")
        }
        return final_eval_result, total_usage_data if total_usage_data["total_tokens"] > 0 else None