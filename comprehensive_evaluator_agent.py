      
# comprehensive_evaluator_agent.py
import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple

import config
import llm_interface
import utils # MODIFIED: For spaCy functions
from type import EvaluationResult, ProblemDetail
from data_access import chapter_queries
from prompt_data_getters import (
    get_filtered_character_profiles_for_prompt_plain_text,
    get_filtered_world_data_for_prompt_plain_text,
    get_reliable_kg_facts_for_drafting_prompt
)
from parsing_utils import split_text_into_blocks, parse_key_value_block

logger = logging.getLogger(__name__)

PROBLEM_DETAIL_KEY_MAP = {
    "issue_category": "issue_category",
    "problem_description": "problem_description",
    "quote_from_original": "quote_from_original_text", # Maps to text field
    "suggested_fix_focus": "suggested_fix_focus"
}

# Standardized internal category names
INTERNAL_VALID_CATEGORIES = {
    "consistency",
    "plot_arc",
    "thematic_alignment", # Standardized internal name
    "narrative_depth_and_length", # Standardized internal name
    "meta"
}

def _normalize_llm_category_to_internal(llm_category_str: str) -> str:
    """Normalizes category names from LLM output to internal standard names."""
    normalized = llm_category_str.lower().strip().replace(" ", "_")
    if "thematic" in normalized: # Catches "thematic_alignment" or "thematic"
        return "thematic_alignment"
    if "narrative_depth" in normalized: # Catches "narrative_depth_and_length" or "narrative_depth"
        return "narrative_depth_and_length"
    # For exact matches or other categories like consistency, plot_arc, meta
    if normalized in INTERNAL_VALID_CATEGORIES:
        return normalized
    return "meta" # Default if no clear mapping


class ComprehensiveEvaluatorAgent:
    def __init__(self, model_name: str = config.EVALUATION_MODEL):
        self.model_name = model_name
        logger.info(f"ComprehensiveEvaluatorAgent initialized with model: {self.model_name}")
        utils.load_spacy_model_if_needed() # Ensure spaCy model is available

    async def _parse_llm_evaluation_output(self, text: str, chapter_number: int, original_draft_text: str) -> List[ProblemDetail]: # Added original_draft_text
        """
        Parses LLM plain text output for chapter evaluation problems.
        Populates character offsets for the quote and its containing sentence using spaCy.
        """
        final_problems: List[ProblemDetail] = []

        if not text or not text.strip() or "no significant problems found" in text.lower():
            logger.info(f"Plain text evaluation for Ch {chapter_number} is empty or indicates no problems. No problems parsed.")
            return []

        problem_blocks_text = split_text_into_blocks(text, separator_regex_str=r'\n\s*---\s*\n')

        for block_num, block_content in enumerate(problem_blocks_text):
            if not block_content.strip():
                continue

            logger.debug(f"Parsing problem block {block_num+1} for Ch {chapter_number}:\n{block_content[:150]}...")

            parsed_problem_dict = parse_key_value_block(
                block_text_or_lines=block_content,
                key_map=PROBLEM_DETAIL_KEY_MAP,
                list_internal_keys=[]
            )

            required_keys_internal = set(PROBLEM_DETAIL_KEY_MAP.values())
            missing_keys = required_keys_internal - set(parsed_problem_dict.keys())

            problem_meta: ProblemDetail = { # Initialize with defaults
                "issue_category": "meta",
                "problem_description": "N/A",
                "quote_from_original_text": "N/A - Malformed LLM Output",
                "quote_char_start": None, "quote_char_end": None,
                "sentence_char_start": None, "sentence_char_end": None,
                "suggested_fix_focus": "Review LLM output."
            }

            if missing_keys:
                logger.warning(f"Could not parse all required fields from problem block {block_num+1} for Ch {chapter_number}. Missing: {missing_keys}. Block: '{block_content[:150]}...'")
                problem_meta["problem_description"] = f"Malformed problem block from LLM: {block_content}"
                final_problems.append(problem_meta)
                continue
            
            problem_meta["problem_description"] = parsed_problem_dict.get("problem_description", "N/A")
            problem_meta["quote_from_original_text"] = parsed_problem_dict.get("quote_from_original_text", "N/A - General Issue")
            problem_meta["suggested_fix_focus"] = parsed_problem_dict.get("suggested_fix_focus", "N/A")
            
            llm_category_raw = str(parsed_problem_dict.get("issue_category", "meta")).strip()
            internal_category = _normalize_llm_category_to_internal(llm_category_raw)
            problem_meta["issue_category"] = internal_category

            if internal_category == "meta" and llm_category_raw.lower().strip() != "meta":
                 logger.warning(f"LLM provided category '{llm_category_raw}' in block {block_num+1} for Ch {chapter_number}, which normalized to 'meta'. Check normalization if this is unexpected.")


            quote_text_from_llm = problem_meta["quote_from_original_text"]
            # Normalize "N/A..." variations to a single canonical form first
            if "N/A - General Issue" in quote_text_from_llm or not quote_text_from_llm.strip():
                problem_meta["quote_from_original_text"] = "N/A - General Issue"
                logger.debug(f"Problem block {block_num+1} for Ch {chapter_number} is 'N/A - General Issue' or empty quote.")
            elif utils.NLP_SPACY is not None and original_draft_text.strip():
                offsets_tuple = await utils.find_quote_and_sentence_offsets_with_spacy(original_draft_text, quote_text_from_llm)
                if offsets_tuple:
                    q_start, q_end, s_start, s_end = offsets_tuple
                    problem_meta["quote_char_start"] = q_start
                    problem_meta["quote_char_end"] = q_end
                    problem_meta["sentence_char_start"] = s_start
                    problem_meta["sentence_char_end"] = s_end
                    logger.debug(f"Ch {chapter_number} problem: Quote '{quote_text_from_llm[:30]}...' found at {q_start}-{q_end}, Sentence: {s_start}-{s_end}")
                else:
                    logger.warning(f"Ch {chapter_number} problem: Could not find quote via spaCy utils: '{quote_text_from_llm[:50]}...'. Offsets will be None.")
            elif not original_draft_text.strip():
                 logger.warning(f"Ch {chapter_number} problem: Original draft text is empty. Cannot find offsets for quote: '{quote_text_from_llm[:50]}...'")
            else: # spaCy not loaded
                logger.info(f"Ch {chapter_number} problem: spaCy not available, quote offsets not determined for: '{quote_text_from_llm[:50]}...'")
            
            final_problems.append(problem_meta)
        return final_problems


    async def _perform_llm_comprehensive_evaluation(
        self,
        novel_props: Dict[str, Any],
        draft_text: str,
        chapter_number: int,
        plot_point_focus: Optional[str],
        plot_point_index: int,
        previous_chapters_context: str
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, int]]]:
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

        plot_points_summary_lines = [
            f"- PP {i+1}: {pp[:100]}..." for i, pp in enumerate(novel_props.get('plot_points', []))
        ] if novel_props.get('plot_points') else ["  - Not available"]
        plot_points_summary_str = "\n".join(plot_points_summary_lines)

        # --- Few-Shot Example for Comprehensive Evaluation Output ---
        # The LLM should use these exact category names in its output.
        few_shot_eval_example_str = f"""
ISSUE CATEGORY: CONSISTENCY
PROBLEM DESCRIPTION: Character Elara states she has never left her village, but her profile mentions she trained at the Royal Academy in the Capital.
QUOTE FROM ORIGINAL: "I've never seen anything beyond these village walls," Elara sighed, gazing at the distant mountains.
SUGGESTED FIX FOCUS: Adjust Elara's dialogue to align with her established backstory of training in the Capital, or reconcile this statement with her past (e.g., she's being metaphorical or hiding her past).
---
ISSUE CATEGORY: PLOT_ARC
PROBLEM DESCRIPTION: The chapter focuses heavily on a minor side character's backstory, which doesn't significantly advance the intended plot point about finding the Sunstone.
QUOTE FROM ORIGINAL: The old merchant then spent a long while recounting his youthful adventures in the spice trade, detailing three different voyages.
SUGGESTED FIX FOCUS: Reduce the side character's backstory significantly or tie it directly into how it helps or hinders the search for the Sunstone. Ensure the main plot point progression is central.
---
ISSUE CATEGORY: NARRATIVE_DEPTH_AND_LENGTH
PROBLEM DESCRIPTION: The confrontation with the antagonist feels rushed and lacks emotional impact. The protagonist's internal reaction to the antagonist's reveal is minimal.
QUOTE FROM ORIGINAL: "It was you all along!" John exclaimed. The Baron merely smiled. Then they fought.
SUGGESTED FIX FOCUS: Expand on John's internal thoughts and feelings upon discovering the Baron's betrayal. Show, don't just tell, the emotional weight of this moment. Describe the fight with more detail and tension.
---
ISSUE CATEGORY: THEMATIC_ALIGNMENT
PROBLEM DESCRIPTION: A newly introduced magical system allowing instant teleportation undermines the established theme of 'arduous journeys have transformative power'.
QUOTE FROM ORIGINAL: With a flick of her wrist, Mariel teleported the entire party across the Obsidian Peaks.
SUGGESTED FIX FOCUS: Reconsider the instant teleportation. If kept, add significant costs, limitations, or consequences that align with or challenge the core theme in an interesting way, rather than negating it.
"""
        
        prompt_lines = [
            "/no_think",
            f"You are a Master Editor evaluating Chapter {chapter_number} of a novel titled \"{novel_props.get('title', 'Untitled Novel')}\" (Protagonist: {protagonist_name_str}).",
            "Analyze the **Complete Chapter Text** provided below.",
            "Your task is to identify specific issues related to these EXACT categories:", # Emphasize exact category names
            "1.  **CONSISTENCY**",
            "2.  **PLOT_ARC**",
            "3.  **THEMATIC_ALIGNMENT**", # Matches example
            f"4.  **NARRATIVE_DEPTH_AND_LENGTH** (target: at least {config.MIN_ACCEPTABLE_DRAFT_LENGTH} characters)", # Matches example
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
            kg_check_results_text,
            "  **Previous Chapters Context (Semantic Flow & KG Facts for Canon):**",
            "  --- PREVIOUS CONTEXT ---",
            previous_chapters_context if previous_chapters_context.strip() else "N/A (e.g., Chapter 1 or context retrieval failed).",
            "  --- END PREVIOUS CONTEXT ---",
            "",
            f"**Complete Chapter {chapter_number} Text (to analyze):**",
            "--- BEGIN COMPLETE CHAPTER TEXT ---",
            draft_text,
            "--- END COMPLETE CHAPTER TEXT ---",
            "",
            "**Output Format (CRITICAL - PLAIN TEXT ONLY):**",
            "Provide your evaluation as plain text. If problems are found, list each problem individually.",
            "For `ISSUE CATEGORY:`, use one of the EXACT category names listed in your task (CONSISTENCY, PLOT_ARC, THEMATIC_ALIGNMENT, NARRATIVE_DEPTH_AND_LENGTH).", # Re-emphasize
            "Then, use the EXACT keys: `PROBLEM DESCRIPTION:`, `QUOTE FROM ORIGINAL:`, `SUGGESTED FIX FOCUS:`.",
            "The `QUOTE FROM ORIGINAL:` must be a VERBATIM quote (10-50 words) from the chapter text. If general (e.g., overall length) or no quote applies, use \"N/A - General Issue\".",
            "Separate each problem block with a line containing only \"---\".",
            "If NO problems are found, output ONLY the phrase: \"No significant problems found.\"",
            "",
            "**Follow this example structure for your output precisely:**",
            "```plaintext",
            few_shot_eval_example_str.strip(),
            "```",
            "",
            "Begin your output now:"
        ]
        prompt = "\n".join(prompt_lines)

        logger.info(f"Calling LLM ({self.model_name}) for comprehensive evaluation of chapter {chapter_number}...")
        raw_evaluation_text, usage_data = await llm_interface.async_call_llm(
            model_name=self.model_name,
            prompt=prompt,
            temperature=config.TEMPERATURE_EVALUATION, 
            allow_fallback=True,
            stream_to_disk=False,
            frequency_penalty=config.FREQUENCY_PENALTY_EVALUATION, # Added
            presence_penalty=config.PRESENCE_PENALTY_EVALUATION    # Added
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
            # These legacy keys might still be useful for very quick high-level checks,
            # but the primary source of truth is the parsed ProblemDetail list.
            legacy_consistency = "Potential consistency issues." if "consistency" in cleaned_evaluation_text.lower() else None
            legacy_plot = "Potential plot arc issues." if "plot_arc" in cleaned_evaluation_text.lower() else None
            legacy_theme = "Potential thematic issues." if "thematic" in cleaned_evaluation_text.lower() else None # "thematic_alignment" should map
            legacy_depth = "Potential narrative depth/length issues." if "narrative_depth" in cleaned_evaluation_text.lower() else None # "narrative_depth_and_length" should map
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
                "quote_from_original_text": "N/A - General Issue",
                "quote_char_start": None, "quote_char_end": None,
                "sentence_char_start": None, "sentence_char_end": None,
                "suggested_fix_focus": "Generate content for the chapter."
            })
            reasons_for_revision_summary.append("Draft is empty.")
        elif len(draft_text) < config.MIN_ACCEPTABLE_DRAFT_LENGTH:
            needs_revision = True
            problem_details_list.append({
                "issue_category": "narrative_depth_and_length", # Standardized internal category
                "problem_description": f"Draft is too short ({len(draft_text)} chars). Minimum required: {config.MIN_ACCEPTABLE_DRAFT_LENGTH}.",
                "quote_from_original_text": "N/A - General Issue",
                "quote_char_start": None, "quote_char_end": None,
                "sentence_char_start": None, "sentence_char_end": None,
                "suggested_fix_focus": f"Expand content significantly across multiple scenes/sections to meet the {config.MIN_ACCEPTABLE_DRAFT_LENGTH} character target. Focus on adding descriptive detail, character introspection, and dialogue."
            })
            reasons_for_revision_summary.append(f"Draft is too short ({len(draft_text)} chars). Minimum required: {config.MIN_ACCEPTABLE_DRAFT_LENGTH}.")

        current_embedding_task = llm_interface.async_get_embedding(draft_text)
        if chapter_number > 1:
            prev_embedding = await chapter_queries.get_embedding_from_db(chapter_number - 1)
            current_embedding = await current_embedding_task
            if current_embedding is not None and prev_embedding is not None:
                coherence_score = utils.numpy_cosine_similarity(current_embedding, prev_embedding)
                logger.info(f"Coherence score with previous chapter ({chapter_number-1}): {coherence_score:.4f}")
                if coherence_score < config.REVISION_COHERENCE_THRESHOLD:
                    needs_revision = True
                    problem_details_list.append({
                        "issue_category": "consistency",
                        "problem_description": f"Low coherence with previous chapter (Score: {coherence_score:.4f}, Threshold: {config.REVISION_COHERENCE_THRESHOLD}). The narrative flow or tone may be disjointed.",
                        "quote_from_original_text": "N/A - General Issue",
                        "quote_char_start": None, "quote_char_end": None,
                        "sentence_char_start": None, "sentence_char_end": None,
                        "suggested_fix_focus": "Review the transition from the previous chapter. Ensure stylistic, tonal, and narrative continuity. This might involve adjusting opening scenes or overall pacing."
                    })
                    reasons_for_revision_summary.append(f"Low coherence with previous chapter (Score: {coherence_score:.4f}, Threshold: {config.REVISION_COHERENCE_THRESHOLD}).")
            else:
                logger.warning(f"Could not perform coherence check for ch {chapter_number} (missing current or previous embedding).")
        else:
            logger.info("Skipping coherence check for Chapter 1.")
            await current_embedding_task # Still await it if it's chapter 1 to ensure it's done

        llm_eval_output_dict, llm_usage = await self._perform_llm_comprehensive_evaluation(
            novel_props, draft_text, chapter_number, plot_point_focus, plot_point_index, previous_chapters_context
        )
        if llm_usage:
            total_usage_data["prompt_tokens"] += llm_usage.get("prompt_tokens", 0)
            total_usage_data["completion_tokens"] += llm_usage.get("completion_tokens", 0)
            total_usage_data["total_tokens"] += llm_usage.get("total_tokens", 0)

        llm_eval_text_output = llm_eval_output_dict.get("problems_found_text_output", "")
        # Pass original_draft_text to the parser
        parsed_problems_from_llm = await self._parse_llm_evaluation_output(llm_eval_text_output, chapter_number, draft_text)

        if parsed_problems_from_llm:
            problem_details_list.extend(parsed_problems_from_llm)
            needs_revision = True
            # Use INTERNAL_VALID_CATEGORIES for mapping to reasons if needed, or a more descriptive map
            category_map_to_reason = {
                "consistency": "Consistency issues identified by LLM.",
                "plot_arc": "Plot Arc deviation identified by LLM.",
                "thematic_alignment": "Thematic Alignment issues identified by LLM.",
                "narrative_depth_and_length": "Narrative Depth/Length issues identified by LLM.",
                "meta": "Meta/Uncategorized issues identified by LLM."
            }
            for prob in parsed_problems_from_llm:
                reason = category_map_to_reason.get(prob["issue_category"]) # prob["issue_category"] is now normalized
                if reason and reason not in reasons_for_revision_summary:
                    reasons_for_revision_summary.append(reason)
        elif llm_eval_text_output.strip() and "no significant problems found" not in llm_eval_text_output.lower():
            logger.warning(f"LLM evaluation for Ch {chapter_number} provided text, but no problems were parsed. Text: '{llm_eval_text_output[:200]}...'")
            problem_details_list.append({
                "issue_category": "meta",
                "problem_description": "LLM evaluation output was non-empty but could not be parsed into specific problems.",
                "quote_from_original_text": "N/A - LLM Output Parsing",
                "quote_char_start": None, "quote_char_end": None,
                "sentence_char_start": None, "sentence_char_end": None,
                "suggested_fix_focus": "Review LLM evaluation output and parsing logic. The output might not conform to the expected problem format."
            })
            if "LLM evaluation output unparsable." not in reasons_for_revision_summary:
                 reasons_for_revision_summary.append("LLM evaluation output unparsable.")
            needs_revision = True

        unique_reasons_summary = sorted(list(set(reasons_for_revision_summary)))
        validated_problem_details_list: List[ProblemDetail] = []
        for prob_item in problem_details_list:
            # Log if quote_char_start is still None for a non-general quote AFTER spaCy processing attempt.
            if prob_item["quote_from_original_text"] not in ["N/A - General Issue", "N/A - LLM Output Parsing", "N/A - Malformed LLM Output"] \
               and prob_item["quote_char_start"] is None \
               and prob_item["quote_from_original_text"].strip() and draft_text.strip(): # Ensure there was text to search in
                 logger.warning(
                    f"CompEvaluator: Problem quote TEXT for Ch {chapter_number} ('{prob_item['quote_from_original_text'][:50]}...') present, "
                    f"but its offsets were NOT found by spaCy utils. Problem desc: {prob_item['problem_description']}"
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

    