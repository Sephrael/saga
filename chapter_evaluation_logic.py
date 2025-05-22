# chapter_evaluation_logic.py
"""
Handles the evaluation of chapter drafts for consistency, plot arc alignment, etc.,
for the SAGA system.
"""
import logging
import json # Still used for dumping context data into prompts
import asyncio
import re
from typing import Optional, Dict, Any, List

import config
import llm_interface 
import utils 
from type import EvaluationResult, ProblemDetail 
from state_manager import state_manager
from prompt_data_getters import (
    get_filtered_character_profiles_for_prompt_plain_text, 
    get_filtered_world_data_for_prompt_plain_text,       
    get_reliable_kg_facts_for_drafting_prompt
)

logger = logging.getLogger(__name__)

def _parse_plain_text_evaluation(text: str, chapter_number: int) -> List[ProblemDetail]:
    """
    Parses LLM plain text output for chapter evaluation problems.
    Expected format per problem:
    ISSUE CATEGORY: <category_name>
    PROBLEM DESCRIPTION: <description_text>
    QUOTE FROM ORIGINAL: <quote_text>
    SUGGESTED FIX FOCUS: <fix_text>
    --- (separator)
    """
    problems: List[ProblemDetail] = []
    if not text or not text.strip():
        logger.warning(f"Plain text evaluation for Ch {chapter_number} is empty. No problems parsed.")
        return []

    # Normalize line endings and split by a clear problem separator
    # The LLM prompt now requests "---" as a separator.
    problem_blocks = re.split(r'\n\s*---\s*\n', text.strip(), flags=re.MULTILINE)

    for block_num, block in enumerate(problem_blocks):
        block = block.strip()
        if not block:
            continue

        try:
            category_match = re.search(r"^\s*ISSUE CATEGORY:\s*(.+)$", block, re.IGNORECASE | re.MULTILINE)
            description_match = re.search(r"^\s*PROBLEM DESCRIPTION:\s*(.+)$", block, re.IGNORECASE | re.MULTILINE)
            quote_match = re.search(r"^\s*QUOTE FROM ORIGINAL:\s*(.+)$", block, re.IGNORECASE | re.MULTILINE)
            fix_focus_match = re.search(r"^\s*SUGGESTED FIX FOCUS:\s*(.+)$", block, re.IGNORECASE | re.MULTILINE)

            if category_match and description_match and quote_match and fix_focus_match:
                category = category_match.group(1).strip().lower()
                # Validate category against expected values
                valid_categories = ["consistency", "plot_arc", "thematic", "narrative_depth", "meta"]
                if category not in valid_categories:
                    logger.warning(f"Parsed unknown issue category '{category}' in block {block_num+1} for Ch {chapter_number}. Defaulting to 'meta'. Block: {block[:100]}")
                    category = "meta"
                
                problem: ProblemDetail = {
                    "issue_category": category,
                    "problem_description": description_match.group(1).strip(),
                    "quote_from_original": quote_match.group(1).strip(),
                    "suggested_fix_focus": fix_focus_match.group(1).strip(),
                }
                problems.append(problem)
            else:
                logger.warning(f"Could not parse all required fields from problem block {block_num+1} for Ch {chapter_number}. Block: '{block[:150]}...'")
                # Fallback: try to capture the whole block as a meta-problem if any key is missing
                problems.append({
                    "issue_category": "meta",
                    "problem_description": f"Malformed problem block from LLM: {block}",
                    "quote_from_original": "N/A - Malformed LLM Output",
                    "suggested_fix_focus": "Review LLM output and prompt for evaluation."
                })

        except Exception as e:
            logger.error(f"Error parsing problem block {block_num+1} for Ch {chapter_number}: {e}. Block: '{block[:150]}...'", exc_info=True)
            problems.append({
                "issue_category": "meta",
                "problem_description": f"Exception during parsing LLM output: {e}. Content: {block}",
                "quote_from_original": "N/A - Parser Exception",
                "suggested_fix_focus": "Review parsing logic and LLM output for evaluation."
            })
            
    return problems


async def comprehensive_chapter_evaluation(
    agent, chapter_text: str, chapter_number: int, previous_chapters_context: str
) -> Dict[str, Any]: # This now returns a dict that will be processed into EvaluationResult
    """
    Performs a comprehensive evaluation of the chapter text using a single LLM call.
    The LLM is now expected to output structured plain text, not JSON.
    """
    if not chapter_text:
        logger.warning(f"Comprehensive evaluation skipped for Ch {chapter_number}: empty draft text.")
        # Still return the structure expected by the caller, now populated directly
        return {
            "problems_found_text_output": "Draft is empty.", # Store the raw text indicating no problems
            "legacy_consistency_issues": "Skipped (empty draft)",
            "legacy_plot_arc_deviation": "Skipped (empty draft)",
            "legacy_thematic_issues": "Skipped (empty draft)",
            "legacy_narrative_depth_issues": "Skipped (empty draft)"
        }

    plot_point_focus, plot_point_index = agent._get_plot_point_info(chapter_number)
    if plot_point_focus is None:
        plot_point_focus_str = "Not available for this chapter."
        logger.warning(f"Plot point focus not available for Ch {chapter_number} during comprehensive evaluation.")
    else:
        plot_point_focus_str = plot_point_focus

    novel_theme_str = agent.plot_outline.get('theme', 'Not specified')
    novel_genre_str = agent.plot_outline.get('genre', 'Not specified')
    protagonist_arc_str = agent.plot_outline.get('character_arc', 'Not specified')
    protagonist_name_str = agent.plot_outline.get('protagonist_name', 'The Protagonist')
    
    kg_check_results_text = await get_reliable_kg_facts_for_drafting_prompt(agent, chapter_number, None)

    # Use the new plain text formatters for prompt context
    # For Chapter 1, chapter_number - 1 will be 0. These getters need to handle it.
    char_profiles_plain_text = await get_filtered_character_profiles_for_prompt_plain_text(agent, chapter_number - 1)
    world_building_plain_text =  await get_filtered_world_data_for_prompt_plain_text(agent, chapter_number - 1)

    prompt = f"""/no_think
You are a Master Editor evaluating Chapter {chapter_number} of a novel titled "{agent.plot_outline.get('title', 'Untitled Novel')}" (Protagonist: {protagonist_name_str}).
Analyze the **Complete Chapter Text** provided below.
Your task is to identify specific issues related to:
1.  **CONSISTENCY**: Contradictions with Plot Outline, Character Profiles, World Building, Key Reliable KG Facts, Previous Context, or internal inconsistencies within THIS chapter.
2.  **PLOT_ARC**: How well this chapter addresses or advances its Intended Plot Point: "{plot_point_focus_str}" (Plot Point #{plot_point_index + 1}).
3.  **THEMATIC_ALIGNMENT**: Alignment with the novel's core elements (Genre: {novel_genre_str}, Theme: {novel_theme_str}, Protagonist's Arc: {protagonist_arc_str}).
4.  **NARRATIVE_DEPTH_AND_LENGTH**: Sufficiency of descriptive detail, character introspection, dialogue development, pacing, and overall length (target: at least {config.MIN_ACCEPTABLE_DRAFT_LENGTH} characters).

**Reference Information for CONSISTENCY Check (Summary Format):**
  **Plot Outline Summary:**
  ```text
  Title: {agent.plot_outline.get('title', 'N/A')}
  Genre: {agent.plot_outline.get('genre', 'N/A')}
  Theme: {agent.plot_outline.get('theme', 'N/A')}
  Protagonist: {agent.plot_outline.get('protagonist_name', 'N/A')} ({agent.plot_outline.get('character_arc', 'N/A')})
  Logline: {agent.plot_outline.get('logline', 'N/A')}
  Key Plot Points (summary):
  {chr(10).join([f"- PP {i+1}: {pp[:100]}..." for i, pp in enumerate(agent.plot_outline.get('plot_points', []))]) if agent.plot_outline.get('plot_points') else "  - Not available"}
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
{chapter_text}
--- END COMPLETE CHAPTER TEXT ---

**Output Format (CRITICAL):**
Provide your evaluation as plain text. If problems are found, list each problem individually using the following format:

ISSUE CATEGORY: [consistency | plot_arc | thematic | narrative_depth]
PROBLEM DESCRIPTION: [A concise description of the specific issue.]
QUOTE FROM ORIGINAL: [**A VERBATIM quote (10-50 words) from the "Complete Chapter Text" that clearly illustrates this specific problem.** If general (e.g., overall length), provide a representative short quote or "N/A - General Issue".]
SUGGESTED FIX FOCUS: [Brief guidance on what the revision for this specific quote should focus on (e.g., "Clarify character's motivation", "Expand description of setting").]
---

If multiple problems are found, separate each problem block with a line containing only "---".
If NO problems are found for a category or overall, output ONLY the phrase: "No significant problems found."

Example if issues are found:
ISSUE CATEGORY: consistency
PROBLEM DESCRIPTION: Character X acts out of character. Their profile says brave, but here they are cowardly.
QUOTE FROM ORIGINAL: X trembled and hid behind the rock, refusing to move.
SUGGESTED FIX FOCUS: Rewrite X's action to show bravery or internal conflict leading to hesitation, aligning with their profile.
---
ISSUE CATEGORY: narrative_depth
PROBLEM DESCRIPTION: The chapter is too short overall for the events covered and the target length.
QUOTE FROM ORIGINAL: N/A - General Issue
SUGGESTED FIX FOCUS: Identify several key scenes or descriptive passages throughout the chapter and expand them significantly.

Example if no issues are found:
No significant problems found.

Output ONLY the evaluation text as described.
"""
    logger.info(f"Calling LLM ({config.EVALUATION_MODEL}) for comprehensive (plain text) evaluation of chapter {chapter_number}...")
    raw_evaluation_text = await llm_interface.async_call_llm(
        model_name=config.EVALUATION_MODEL,
        prompt=prompt,
        temperature=0.3, 
        allow_fallback=True, 
        stream_to_disk=False 
    )
    
    cleaned_evaluation_text = llm_interface.clean_model_response(raw_evaluation_text)
    
    # --- HEURISTIC BLOCK for "no issues" text ---
    # This heuristic remains useful.
    no_issues_keywords = [
        "no significant problems found", "no issues found", "no problems found", 
        "no revision needed", "no changes needed", "all clear", "looks good", 
        "is fine", "is acceptable", "passes evaluation", "meets criteria",
        "no significant issues", "evaluation passed",
        "therefore, no revision is needed" 
    ]
    is_likely_no_issues_text = False
    if cleaned_evaluation_text.strip():
        # Check if the entire cleaned text is one of the "no issues" phrases
        # or if such a phrase is a dominant part of a very short response.
        normalized_eval_text = cleaned_evaluation_text.lower().strip().replace('.', '')
        for keyword in no_issues_keywords:
            normalized_keyword = keyword.lower().strip().replace('.', '')
            if normalized_keyword == normalized_eval_text:
                 is_likely_no_issues_text = True
                 break
            # If the response is very short and contains the keyword, consider it "no issues"
            if len(normalized_eval_text) < len(normalized_keyword) + 20 and normalized_keyword in normalized_eval_text:
                is_likely_no_issues_text = True
                break
    
    if is_likely_no_issues_text:
        logger.info(
            f"Heuristic: Evaluation for Ch {chapter_number} appears to be 'no issues' text: '{cleaned_evaluation_text[:100]}...'. "
            f"Returning empty problem list."
        )
        # The caller expects a dict; "problems_found_text_output" will store the actual LLM text.
        return {
            "problems_found_text_output": cleaned_evaluation_text, # Store the actual LLM output
            "legacy_consistency_issues": None, # No specific issues
            "legacy_plot_arc_deviation": None,
            "legacy_thematic_issues": None,
            "legacy_narrative_depth_issues": None
        }
    
    # If not "no issues", we return the raw text output for parsing by the caller
    # and some legacy fields for summarization.
    # The main parsing will happen in `evaluate_chapter_draft_logic` using `_parse_plain_text_evaluation`.
    
    # For legacy fields, we can do a quick scan of the raw text.
    # This is a simplification as detailed parsing is deferred.
    legacy_consistency = "Potential consistency issues detected." if "consistency" in cleaned_evaluation_text.lower() else None
    legacy_plot = "Potential plot arc issues detected." if "plot_arc" in cleaned_evaluation_text.lower() else None
    legacy_theme = "Potential thematic issues detected." if "thematic" in cleaned_evaluation_text.lower() else None
    legacy_depth = "Potential narrative depth/length issues detected." if "narrative_depth" in cleaned_evaluation_text.lower() else None

    if not cleaned_evaluation_text.strip(): # LLM returned empty or only whitespace
        logger.error(f"Comprehensive evaluation LLM for Ch {chapter_number} returned empty text after cleaning. Raw: '{raw_evaluation_text[:200]}...'")
        return {
            "problems_found_text_output": "Evaluation LLM call failed or returned empty.",
            "legacy_consistency_issues": "Evaluation LLM call failed.",
            "legacy_plot_arc_deviation": "Evaluation LLM call failed.",
            "legacy_thematic_issues": "Evaluation LLM call failed.",
            "legacy_narrative_depth_issues": "Evaluation LLM call failed."
        }
        
    logger.info(f"Comprehensive (plain text) evaluation for Ch {chapter_number} complete. LLM output: '{cleaned_evaluation_text[:200]}...'")
    return {
        "problems_found_text_output": cleaned_evaluation_text,
        "legacy_consistency_issues": legacy_consistency,
        "legacy_plot_arc_deviation": legacy_plot,
        "legacy_thematic_issues": legacy_theme,
        "legacy_narrative_depth_issues": legacy_depth
    }


async def evaluate_chapter_draft_logic(agent, draft_text: str, chapter_number: int, previous_chapters_context: str) -> EvaluationResult:
    """
    Evaluates a chapter draft.
    Comprehensive LLM evaluation now returns plain text, which is parsed here.
    """
    logger.info(f"Evaluating chapter {chapter_number} draft (length: {len(draft_text)} chars)...")
    
    reasons_for_revision_summary: list[str] = [] 
    problem_details_list: List[ProblemDetail] = [] 
    needs_revision = False
    coherence_score: Optional[float] = None
    
    if not draft_text: 
        needs_revision = True
        empty_draft_problem: ProblemDetail = { # Ensure it matches ProblemDetail type
            "issue_category": "meta", 
            "problem_description": "Draft is empty.", 
            "quote_from_original": "N/A - General Issue", # Changed from "" to align with other N/A 
            "suggested_fix_focus": "Generate content for the chapter."
        }
        problem_details_list.append(empty_draft_problem)
        reasons_for_revision_summary.append("Draft is empty.")
    elif len(draft_text) < config.MIN_ACCEPTABLE_DRAFT_LENGTH:
        needs_revision = True
        short_draft_problem: ProblemDetail = {
            "issue_category": "narrative_depth", 
            "problem_description": f"Draft is too short ({len(draft_text)} chars). Minimum required: {config.MIN_ACCEPTABLE_DRAFT_LENGTH}.", 
            "quote_from_original": "N/A - General Issue", 
            "suggested_fix_focus": f"Expand content significantly across multiple scenes/sections to meet the {config.MIN_ACCEPTABLE_DRAFT_LENGTH} character target. Focus on adding descriptive detail, character introspection, and dialogue."
        }
        problem_details_list.append(short_draft_problem)
        reasons_for_revision_summary.append(f"Draft is too short ({len(draft_text)} chars). Minimum required: {config.MIN_ACCEPTABLE_DRAFT_LENGTH}.")

    current_embedding_task = llm_interface.async_get_embedding(draft_text)
    if chapter_number > 1:
        prev_embedding = await state_manager.async_get_embedding_from_db(chapter_number - 1)
        current_embedding = await current_embedding_task 

        if current_embedding is not None and prev_embedding is not None:
            coherence_score = utils.numpy_cosine_similarity(current_embedding, prev_embedding)
            logger.info(f"Coherence score with previous chapter ({chapter_number-1}): {coherence_score:.4f}")
            if coherence_score < config.REVISION_COHERENCE_THRESHOLD:
                needs_revision = True
                coherence_problem: ProblemDetail = {
                    "issue_category": "consistency", 
                    "problem_description": f"Low coherence with previous chapter (Score: {coherence_score:.4f}, Threshold: {config.REVISION_COHERENCE_THRESHOLD}). The narrative flow or tone may be disjointed.",
                    "quote_from_original": "N/A - General Issue",
                    "suggested_fix_focus": "Review the transition from the previous chapter. Ensure stylistic, tonal, and narrative continuity. This might involve adjusting opening scenes or overall pacing."
                }
                problem_details_list.append(coherence_problem)
                reasons_for_revision_summary.append(f"Low coherence with previous chapter (Score: {coherence_score:.4f}, Threshold: {config.REVISION_COHERENCE_THRESHOLD}).")
        else:
            logger.warning(f"Could not perform coherence check for ch {chapter_number} (missing current or previous embedding).")
    else: 
        logger.info("Skipping coherence check for Chapter 1.")
        await current_embedding_task # Still need to await it if it's chapter 1

    llm_eval_output_dict = await comprehensive_chapter_evaluation(agent, draft_text, chapter_number, previous_chapters_context)
    
    llm_eval_text_output = llm_eval_output_dict.get("problems_found_text_output", "")
    parsed_problems_from_llm: List[ProblemDetail] = []

    if llm_eval_text_output and "no significant problems found" not in llm_eval_text_output.lower() and "skipped" not in llm_eval_text_output.lower() and "failed" not in llm_eval_text_output.lower() :
        parsed_problems_from_llm = _parse_plain_text_evaluation(llm_eval_text_output, chapter_number)
        if parsed_problems_from_llm:
            problem_details_list.extend(parsed_problems_from_llm)
            needs_revision = True 
            
            # Consolidate summary reasons
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
                    
        elif llm_eval_text_output.strip(): # LLM provided output, but parser found nothing, and it wasn't "no issues"
            logger.warning(f"LLM evaluation for Ch {chapter_number} provided text, but no problems were parsed by _parse_plain_text_evaluation. Text: '{llm_eval_text_output[:200]}...'")
            meta_problem: ProblemDetail = {
                "issue_category": "meta",
                "problem_description": "LLM evaluation output was non-empty but could not be parsed into specific problems.",
                "quote_from_original": "N/A - LLM Output Parsing",
                "suggested_fix_focus": "Review LLM evaluation output and parsing logic. The output might be malformed or indicate a general unparsable issue."
            }
            problem_details_list.append(meta_problem)
            if "LLM evaluation output unparsable." not in reasons_for_revision_summary:
                 reasons_for_revision_summary.append("LLM evaluation output unparsable.")
            needs_revision = True
            
    unique_reasons_summary = sorted(list(set(reasons_for_revision_summary)))

    validated_problem_details_list: List[ProblemDetail] = []
    for prob_item in problem_details_list:
        quote = prob_item["quote_from_original"]
        # Ensure quote is a string, even if empty
        if not isinstance(quote, str):
            logger.warning(f"Problem quote for Ch {chapter_number} was not a string ({type(quote)}): '{str(quote)[:50]}...'. Converting to 'N/A - Invalid Quote Type'. Problem: {prob_item['problem_description']}")
            prob_item["quote_from_original"] = "N/A - Invalid Quote Type"
            quote = prob_item["quote_from_original"]


        if quote not in ["N/A - General Issue", "N/A - LLM Output Parsing", "N/A - Parser Exception", "N/A - Invalid Quote Type"] and quote.strip() and draft_text:
            if not (10 <= len(quote) <= 300): 
                 logger.warning(f"Problem quote for Ch {chapter_number} has unusual length ({len(quote)} chars): '{quote[:50]}...'. Still including.")
            if quote not in draft_text:
                logger.warning(f"Problem quote for Ch {chapter_number} NOT FOUND VERBATIM in chapter text: '{quote[:50]}...'. This may fail patching. Problem: {prob_item['problem_description']}")
        validated_problem_details_list.append(prob_item)


    logger.info(f"Evaluation for Ch {chapter_number} complete. Needs revision: {needs_revision}. Summary reasons: {'; '.join(unique_reasons_summary) if unique_reasons_summary else 'None'}. Detailed problems: {len(validated_problem_details_list)}")
    
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
    return final_eval_result