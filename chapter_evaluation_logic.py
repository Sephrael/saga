# chapter_evaluation_logic.py
"""
Handles the evaluation of chapter drafts for consistency, plot arc alignment, etc.,
for the SAGA system.
"""
import logging
import json
import asyncio
from typing import Optional, Dict, Any, List

import config
import llm_interface
import utils 
from type import EvaluationResult, ProblemDetail # Updated import
from state_manager import state_manager
from prompt_data_getters import get_filtered_character_profiles_for_prompt, get_filtered_world_data_for_prompt

logger = logging.getLogger(__name__)

async def comprehensive_chapter_evaluation(
    agent, chapter_text: str, chapter_number: int, previous_chapters_context: str
) -> Dict[str, Any]: # This now returns a dict that will be processed into EvaluationResult
    """
    Performs a comprehensive evaluation of the chapter text using a single LLM call.
    Checks for consistency, plot arc alignment, thematic alignment, and narrative depth/length.
    Identifies specific quotes from the original text for each issue.
    'agent' is an instance of NovelWriterAgent.
    'previous_chapters_context' is the hybrid context or relevant prior context.
    Returns a dictionary based on the new structured JSON output from the LLM.
    """
    if not chapter_text:
        logger.warning(f"Comprehensive evaluation skipped for Ch {chapter_number}: empty draft text.")
        return {
            "problems_found": [{
                "issue_category": "meta", 
                "problem_description": "Draft is empty.", 
                "quote_from_original": "", 
                "suggested_fix_focus": "Generate content."
            }],
            "legacy_consistency_issues": "Skipped (empty draft)", # Retain for summarization in EvaluationResult
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
    
    kg_chapter_limit = chapter_number - 1
    protagonist_name = agent.plot_outline.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME)
    kg_loc_task = state_manager.async_get_most_recent_value(protagonist_name, "located_in", kg_chapter_limit, include_provisional=False)
    kg_status_task = state_manager.async_get_most_recent_value(protagonist_name, "status_is", kg_chapter_limit, include_provisional=False)
    kg_location, kg_status = await asyncio.gather(kg_loc_task, kg_status_task)
    kg_facts_for_prompt_list: list[str] = []
    if kg_location: kg_facts_for_prompt_list.append(f"- {protagonist_name}'s last reliably known location: {kg_location}.")
    if kg_status: kg_facts_for_prompt_list.append(f"- {protagonist_name}'s last reliably known status: {kg_status}.")
    kg_check_results_text = "**Key Reliable KG Facts (from pre-novel & previous chapters):**\n" + "\n".join(kg_facts_for_prompt_list) + "\n" if kg_facts_for_prompt_list else "**Key Reliable KG Facts:** None available or protagonist not tracked.\n"

    char_profiles_for_prompt = await get_filtered_character_profiles_for_prompt(agent, kg_chapter_limit)
    world_building_for_prompt =  await get_filtered_world_data_for_prompt(agent, kg_chapter_limit)

    prompt = f"""/no_think
You are a Master Editor evaluating Chapter {chapter_number} of a novel titled "{agent.plot_outline.get('title', 'Untitled Novel')}" (Protagonist: {protagonist_name_str}).
Analyze the **Complete Chapter Text** provided below.
Your task is to identify specific issues related to:
1.  **CONSISTENCY**: Contradictions with Plot Outline, Character Profiles, World Building, Key Reliable KG Facts, Previous Context, or internal inconsistencies within THIS chapter.
2.  **PLOT_ARC**: How well this chapter addresses or advances its Intended Plot Point: "{plot_point_focus_str}" (Plot Point #{plot_point_index + 1}).
3.  **THEMATIC_ALIGNMENT**: Alignment with the novel's core elements (Genre: {novel_genre_str}, Theme: {novel_theme_str}, Protagonist's Arc: {protagonist_arc_str}).
4.  **NARRATIVE_DEPTH_AND_LENGTH**: Sufficiency of descriptive detail, character introspection, dialogue development, pacing, and overall length (target: at least {config.MIN_ACCEPTABLE_DRAFT_LENGTH} characters).

**Reference Information for CONSISTENCY Check:**
  **Plot Outline Summary:**
  ```json
  {json.dumps(agent.plot_outline, indent=2, ensure_ascii=False, default=str, sort_keys=True)}
  ```
  **Character Profiles (Key Info - check 'prompt_notes' for provisional status):**
  ```json
  {json.dumps(char_profiles_for_prompt, indent=2, ensure_ascii=False, default=str, sort_keys=True)}
  ```
  **World Building Notes (Key Info - check 'prompt_notes' for provisional status):**
  ```json
  {json.dumps(world_building_for_prompt, indent=2, ensure_ascii=False, default=str, sort_keys=True)}
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
Provide your evaluation ONLY as a single, valid JSON object.
The JSON object *must* have a top-level key: `"problems_found"`.
`"problems_found"`: A JSON list of problem objects. Each problem object MUST have these keys:
    - `"issue_category"` (str): One of "consistency", "plot_arc", "thematic", "narrative_depth".
    - `"problem_description"` (str): A concise description of the specific issue.
    - `"quote_from_original"` (str): **A VERBATIM quote (10-50 words) from the "Complete Chapter Text" that clearly illustrates this specific problem.** This quote will be used for targeted revisions. If the problem is general (e.g., overall length), provide a representative short quote from a relevant section or indicate if not applicable with "N/A - General Issue".
    - `"suggested_fix_focus"` (str): Brief guidance on what the revision for this specific quote should focus on (e.g., "Clarify character's motivation", "Expand description of setting", "Ensure dialogue aligns with established personality", "Deepen internal conflict here", "Increase descriptive detail and emotional impact to expand this section").

If NO problems are found for a category or overall, `problems_found` should be an empty list `[]`.

Example of a valid JSON response if issues are found:
```json
{{
  "problems_found": [
    {{
      "issue_category": "consistency",
      "problem_description": "Character X acts out of character. Their profile says brave, but here they are cowardly.",
      "quote_from_original": "X trembled and hid behind the rock, refusing to move.",
      "suggested_fix_focus": "Rewrite X's action to show bravery or internal conflict leading to hesitation, aligning with their profile."
    }},
    {{
      "issue_category": "plot_arc",
      "problem_description": "The chapter introduces a side quest that doesn't relate to the main plot point of 'finding the artifact'.",
      "quote_from_original": "Suddenly, a villager asked for help finding his lost cat, and Y agreed.",
      "suggested_fix_focus": "Either remove this side quest or tie it directly to the artifact search."
    }},
    {{
      "issue_category": "narrative_depth",
      "problem_description": "The climax scene feels rushed and lacks emotional impact.",
      "quote_from_original": "The monster attacked. They fought. They won.",
      "suggested_fix_focus": "Expand this climax with more sensory details, character reactions, blow-by-blow action, and emotional payoff. Aim for significantly more text."
    }},
    {{
      "issue_category": "narrative_depth",
      "problem_description": "The chapter is too short overall for the events covered and the target length.",
      "quote_from_original": "N/A - General Issue", 
      "suggested_fix_focus": "Identify several key scenes or descriptive passages throughout the chapter and expand them significantly with more detail, introspection, and dialogue to meet the length target."
    }}
  ]
}}
```
Example of a valid JSON response if no issues are found:
```json
{{
  "problems_found": []
}}
```
Output ONLY the JSON object.
"""
    logger.info(f"Calling LLM ({config.EVALUATION_MODEL}) for comprehensive (detailed quote-based) evaluation of chapter {chapter_number}...")
    raw_evaluation = await llm_interface.async_call_llm(
        model_name=config.EVALUATION_MODEL,
        prompt=prompt,
        temperature=0.3, # Lower temperature for more precise quote extraction and structured output
        allow_fallback=True, 
        stream_to_disk=False 
    )
    
    parsed_result: Optional[Any] = await llm_interface.async_parse_llm_json_response(
        raw_evaluation, f"comprehensive quote-based evaluation for ch {chapter_number}", expect_type=dict
    )

    default_response = {
        "problems_found": [{
            "issue_category": "meta", 
            "problem_description": "Evaluation LLM call failed or returned invalid format.", 
            "quote_from_original": "N/A", 
            "suggested_fix_focus": "Review LLM evaluation prompt and response."
        }],
        "legacy_consistency_issues": "Evaluation LLM call failed or returned invalid format.",
        "legacy_plot_arc_deviation": "Evaluation LLM call failed or returned invalid format.",
        "legacy_thematic_issues": "Evaluation LLM call failed or returned invalid format.",
        "legacy_narrative_depth_issues": "Evaluation LLM call failed or returned invalid format."
    }

    if isinstance(parsed_result, dict) and "problems_found" in parsed_result and isinstance(parsed_result["problems_found"], list):
        validated_problems: List[ProblemDetail] = []
        problem_keys = {"issue_category", "problem_description", "quote_from_original", "suggested_fix_focus"}
        
        for prob_item_any in parsed_result["problems_found"]:
            if isinstance(prob_item_any, dict) and problem_keys.issubset(prob_item_any.keys()):
                # Basic type checks
                if all(isinstance(prob_item_any[k], str) for k in problem_keys):
                    # Further validation for quote_from_original if not "N/A - General Issue"
                    quote = prob_item_any["quote_from_original"]
                    if quote != "N/A - General Issue" and not (10 <= len(quote) <= 300): # Rough length check for a valid quote
                         logger.warning(f"Problem quote for Ch {chapter_number} has unusual length ({len(quote)} chars): '{quote[:50]}...'. Still including.")
                    
                    # Ensure quote is actually from original text (important for patching)
                    # This check can be computationally expensive for very long chapters.
                    # Consider sampling or skipping for performance if it becomes an issue.
                    # For now, let's assume the LLM follows instructions for VERBATIM.
                    # if quote != "N/A - General Issue" and quote not in chapter_text:
                    #    logger.warning(f"Problem quote for Ch {chapter_number} NOT FOUND VERBATIM in chapter text: '{quote[:50]}...'. This will fail patching.")
                    #    # Optionally, skip this problem or try to find a similar quote. For now, include it.

                    validated_problems.append(prob_item_any) # type: ignore
                else:
                    logger.warning(f"Problem item in evaluation for Ch {chapter_number} has incorrect value types: {prob_item_any}")
            else:
                logger.warning(f"Problem item in evaluation for Ch {chapter_number} is not a dict or missing keys: {prob_item_any}")

        # Populate legacy summary fields for existing logic/logging based on validated_problems
        final_result = {
            "problems_found": validated_problems,
            "legacy_consistency_issues": next((p["problem_description"] for p in validated_problems if p["issue_category"] == "consistency"), None),
            "legacy_plot_arc_deviation": next((p["problem_description"] for p in validated_problems if p["issue_category"] == "plot_arc"), None),
            "legacy_thematic_issues": next((p["problem_description"] for p in validated_problems if p["issue_category"] == "thematic"), None),
            "legacy_narrative_depth_issues": next((p["problem_description"] for p in validated_problems if p["issue_category"] == "narrative_depth"), None)
        }
        
        num_problems = len(validated_problems)
        problem_summary_str = ", ".join(f"{cat}: {count}" for cat, count in 
                                     {c: sum(1 for p in validated_problems if p['issue_category'] == c) for c in 
                                      ["consistency", "plot_arc", "thematic", "narrative_depth"] if sum(1 for p in validated_problems if p['issue_category'] == c) > 0}.items()
                                    ) if num_problems > 0 else "None"
        
        logger.info(f"Comprehensive (quote-based) evaluation for Ch {chapter_number} complete. Problems found: {num_problems} ({problem_summary_str}).")
        return final_result
    else:
        logger.error(f"Failed to parse comprehensive (quote-based) evaluation for Ch {chapter_number} into the expected dict/list structure. Raw: '{raw_evaluation[:500]}...'")
        await agent._save_debug_output(chapter_number, "comprehensive_quote_eval_parse_fail", raw_evaluation)
        return default_response


async def evaluate_chapter_draft_logic(agent, draft_text: str, chapter_number: int, previous_chapters_context: str) -> EvaluationResult:
    """
    Evaluates a chapter draft using comprehensive LLM evaluation and coherence score.
    'agent' is an instance of NovelWriterAgent.
    'previous_chapters_context' is the hybrid context or relevant prior context.
    """
    logger.info(f"Evaluating chapter {chapter_number} draft (length: {len(draft_text)} chars)...")
    
    reasons_for_revision_summary: list[str] = [] # High-level summary strings
    problem_details_list: List[ProblemDetail] = [] # Detailed problem objects
    needs_revision = False
    coherence_score: Optional[float] = None
    
    if not draft_text: 
        needs_revision = True
        empty_draft_problem = {
            "issue_category": "meta", 
            "problem_description": "Draft is empty.", 
            "quote_from_original": "", 
            "suggested_fix_focus": "Generate content for the chapter."
        }
        problem_details_list.append(empty_draft_problem) # type: ignore
        reasons_for_revision_summary.append("Draft is empty.")
    elif len(draft_text) < config.MIN_ACCEPTABLE_DRAFT_LENGTH:
        needs_revision = True
        short_draft_problem = {
            "issue_category": "narrative_depth", 
            "problem_description": f"Draft is too short ({len(draft_text)} chars). Minimum required: {config.MIN_ACCEPTABLE_DRAFT_LENGTH}.", 
            "quote_from_original": "N/A - General Issue", 
            "suggested_fix_focus": f"Expand content significantly across multiple scenes/sections to meet the {config.MIN_ACCEPTABLE_DRAFT_LENGTH} character target. Focus on adding descriptive detail, character introspection, and dialogue."
        }
        problem_details_list.append(short_draft_problem) # type: ignore
        reasons_for_revision_summary.append(f"Draft is too short ({len(draft_text)} chars). Minimum required: {config.MIN_ACCEPTABLE_DRAFT_LENGTH}.")
        # Note: Even if too short, we still run comprehensive_chapter_evaluation to find other issues.
        # The "N/A - General Issue" quote for overall length might be superseded by more specific depth issues found by LLM.

    current_embedding_task = llm_interface.async_get_embedding(draft_text)
    if chapter_number > 1:
        prev_embedding = await state_manager.async_get_embedding_from_db(chapter_number - 1)
        current_embedding = await current_embedding_task 

        if current_embedding is not None and prev_embedding is not None:
            coherence_score = utils.numpy_cosine_similarity(current_embedding, prev_embedding)
            logger.info(f"Coherence score with previous chapter ({chapter_number-1}): {coherence_score:.4f}")
            if coherence_score < config.REVISION_COHERENCE_THRESHOLD:
                needs_revision = True
                coherence_problem = {
                    "issue_category": "consistency", # Coherence is a form of consistency
                    "problem_description": f"Low coherence with previous chapter (Score: {coherence_score:.4f}, Threshold: {config.REVISION_COHERENCE_THRESHOLD}). The narrative flow or tone may be disjointed.",
                    "quote_from_original": "N/A - General Issue",
                    "suggested_fix_focus": "Review the transition from the previous chapter. Ensure stylistic, tonal, and narrative continuity. This might involve adjusting opening scenes or overall pacing."
                }
                problem_details_list.append(coherence_problem) # type: ignore
                reasons_for_revision_summary.append(f"Low coherence with previous chapter (Score: {coherence_score:.4f}, Threshold: {config.REVISION_COHERENCE_THRESHOLD}).")
        else:
            logger.warning(f"Could not perform coherence check for ch {chapter_number} (missing current or previous embedding).")
    else: 
        logger.info("Skipping coherence check for Chapter 1.")
        await current_embedding_task 

    # Comprehensive LLM-based evaluation (gets detailed problems with quotes)
    llm_eval_dict = await comprehensive_chapter_evaluation(agent, draft_text, chapter_number, previous_chapters_context)

    extracted_problems: List[ProblemDetail] = llm_eval_dict.get("problems_found", [])
    if extracted_problems:
        problem_details_list.extend(extracted_problems)
        needs_revision = True # Any problem from LLM evaluation flags for revision
        
        # Populate summary reasons from the detailed problems
        if any(p["issue_category"] == "consistency" for p in extracted_problems): reasons_for_revision_summary.append("Consistency issues identified by LLM.")
        if any(p["issue_category"] == "plot_arc" for p in extracted_problems): reasons_for_revision_summary.append("Plot Arc deviation identified by LLM.")
        if any(p["issue_category"] == "thematic" for p in extracted_problems): reasons_for_revision_summary.append("Thematic issues identified by LLM.")
        if any(p["issue_category"] == "narrative_depth" for p in extracted_problems): reasons_for_revision_summary.append("Narrative Depth/Length issues identified by LLM.")
            
    # Deduplicate reasons_for_revision_summary if necessary, though multiple sources for same category are fine
    unique_reasons_summary = sorted(list(set(reasons_for_revision_summary)))

    logger.info(f"Evaluation for Ch {chapter_number} complete. Needs revision: {needs_revision}. Summary reasons: {'; '.join(unique_reasons_summary) if unique_reasons_summary else 'None'}. Detailed problems: {len(problem_details_list)}")
    
    # Construct the final EvaluationResult object
    final_eval_result: EvaluationResult = {
        "needs_revision": needs_revision, 
        "reasons": unique_reasons_summary, 
        "problems_found": problem_details_list, # This is the new key part for patch-based revision
        "coherence_score": coherence_score, 
        "consistency_issues": llm_eval_dict.get("legacy_consistency_issues"), # Legacy summary for logging
        "plot_deviation_reason": llm_eval_dict.get("legacy_plot_arc_deviation"), # Legacy summary for logging
        "thematic_issues": llm_eval_dict.get("legacy_thematic_issues"), # Legacy summary for logging
        "narrative_depth_issues": llm_eval_dict.get("legacy_narrative_depth_issues") # Legacy summary for logging
    }
    return final_eval_result