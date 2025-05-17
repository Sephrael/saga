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
import utils # For numpy_cosine_similarity
from type import EvaluationResult
from state_manager import state_manager
from prompt_data_getters import get_filtered_character_profiles_for_prompt, get_filtered_world_data_for_prompt

logger = logging.getLogger(__name__)

async def comprehensive_chapter_evaluation(
    agent, chapter_text: str, chapter_number: int, previous_chapters_context: str
) -> Dict[str, Any]:
    """
    Performs a comprehensive evaluation of the chapter text using a single LLM call.
    Checks for consistency, plot arc alignment, and thematic alignment.
    'agent' is an instance of NovelWriterAgent.
    'previous_chapters_context' is the hybrid context or relevant prior context.
    Returns a dictionary with keys: "consistency_issues", "plot_arc_deviation", "thematic_issues".
    Values are strings describing issues, or None if no issues for that aspect.
    """
    if not chapter_text:
        logger.warning(f"Comprehensive evaluation skipped for Ch {chapter_number}: empty draft text.")
        return {
            "consistency_issues": "Skipped (empty draft)",
            "plot_arc_deviation": "Skipped (empty draft)",
            "thematic_issues": "Skipped (empty draft)"
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
    
    # For consistency check prompt context (similar to old check_draft_consistency_logic)
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
Analyze the **Complete Chapter Text** provided below and provide structured feedback on THREE key aspects:
1.  **CONSISTENCY**: Check for contradictions with Plot Outline, Character Profiles, World Building, Key Reliable KG Facts, Previous Context, or internal inconsistencies within THIS chapter.
2.  **PLOT_ARC**: Determine if this chapter clearly and substantially addresses or advances its Intended Plot Point: "{plot_point_focus_str}" (Plot Point #{plot_point_index + 1}).
3.  **THEMATIC_ALIGNMENT**: Assess alignment with the novel's core elements:
    - Genre: {novel_genre_str}
    - Central Theme: {novel_theme_str}
    - Protagonist's Arc: {protagonist_arc_str}

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
The JSON object *must* have these three top-level keys:
-   `"consistency_issues"`: A string describing specific contradictions or inconsistencies. If none, use `null`.
-   `"plot_arc_deviation"`: A string explaining how the chapter deviates from or fails to address the Intended Plot Point. If it aligns well, use `null`.
-   `"thematic_issues"`: A string describing significant thematic misalignments or deviations from genre/theme/arc. If it aligns well, use `null`.

Example of a valid JSON response if issues are found:
```json
{{
  "consistency_issues": "Character X acts out of character based on their profile (e.g., suddenly cowardly when described as brave). Location Y is described differently than in world notes.",
  "plot_arc_deviation": "The chapter focuses on a side quest not related to the main plot point of 'finding the artifact'.",
  "thematic_issues": "A humorous scene feels tonally inconsistent with the 'dystopian horror' genre. The protagonist's actions in this chapter contradict their stated character arc of 'learning empathy'."
}}
```
Example of a valid JSON response if no issues are found:
```json
{{
  "consistency_issues": null,
  "plot_arc_deviation": null,
  "thematic_issues": null
}}
```
Output ONLY the JSON object.
"""
    logger.info(f"Calling LLM ({config.EVALUATION_MODEL}) for comprehensive evaluation of chapter {chapter_number}...")
    raw_evaluation = await llm_interface.async_call_llm(
        model_name=config.EVALUATION_MODEL,
        prompt=prompt,
        temperature=0.5, # Lower temperature for more consistent evaluation
        allow_fallback=True # Evaluation is critical
    )
    
    parsed_result: Optional[Any] = await llm_interface.async_parse_llm_json_response(
        raw_evaluation, f"comprehensive evaluation for ch {chapter_number}", expect_type=dict
    )

    default_response = {
        "consistency_issues": "Evaluation LLM call failed or returned invalid format.",
        "plot_arc_deviation": "Evaluation LLM call failed or returned invalid format.",
        "thematic_issues": "Evaluation LLM call failed or returned invalid format."
    }

    if isinstance(parsed_result, dict):
        # Validate that the expected keys are present, even if null
        final_result = {
            "consistency_issues": parsed_result.get("consistency_issues"), # None if missing or null
            "plot_arc_deviation": parsed_result.get("plot_arc_deviation"), # None if missing or null
            "thematic_issues": parsed_result.get("thematic_issues")         # None if missing or null
        }
        if not all(key in parsed_result for key in ["consistency_issues", "plot_arc_deviation", "thematic_issues"]):
            logger.warning(f"Comprehensive evaluation for Ch {chapter_number} missing one or more core keys in response. Parsed: {parsed_result}")
            # Fill missing keys from default, but keep existing ones
            if "consistency_issues" not in parsed_result: final_result["consistency_issues"] = default_response["consistency_issues"]
            if "plot_arc_deviation" not in parsed_result: final_result["plot_arc_deviation"] = default_response["plot_arc_deviation"]
            if "thematic_issues" not in parsed_result: final_result["thematic_issues"] = default_response["thematic_issues"]
        
        logger.info(f"Comprehensive evaluation for Ch {chapter_number} complete. Issues - Consistency: {'Yes' if final_result['consistency_issues'] else 'No'}, Plot Arc: {'Yes' if final_result['plot_arc_deviation'] else 'No'}, Thematic: {'Yes' if final_result['thematic_issues'] else 'No'}.")
        return final_result
    else:
        logger.error(f"Failed to parse comprehensive evaluation for Ch {chapter_number} into a dict. Raw: '{raw_evaluation[:500]}...'")
        await agent._save_debug_output(chapter_number, "comprehensive_eval_parse_fail", raw_evaluation)
        return default_response


async def evaluate_chapter_draft_logic(agent, draft_text: str, chapter_number: int, previous_chapters_context: str) -> EvaluationResult:
    """
    Evaluates a chapter draft using comprehensive LLM evaluation and coherence score.
    'agent' is an instance of NovelWriterAgent.
    'previous_chapters_context' is the hybrid context or relevant prior context.
    """
    logger.info(f"Evaluating chapter {chapter_number} draft (length: {len(draft_text)} chars)...")
    
    reasons: list[str] = []
    needs_revision = False
    coherence_score: Optional[float] = None
    
    if not draft_text or len(draft_text) < config.MIN_ACCEPTABLE_DRAFT_LENGTH:
        needs_revision = True
        reasons.append(f"Draft is too short ({len(draft_text or '')} chars). Minimum required: {config.MIN_ACCEPTABLE_DRAFT_LENGTH}.")
        # If too short, comprehensive LLM eval might not be useful or could be skipped.
        # For now, proceed, but this is a point for future optimization.

    # Coherence check (embedding-based)
    current_embedding_task = llm_interface.async_get_embedding(draft_text)
    if chapter_number > 1:
        prev_embedding = await state_manager.async_get_embedding_from_db(chapter_number - 1)
        current_embedding = await current_embedding_task 

        if current_embedding is not None and prev_embedding is not None:
            coherence_score = utils.numpy_cosine_similarity(current_embedding, prev_embedding)
            logger.info(f"Coherence score with previous chapter ({chapter_number-1}): {coherence_score:.4f}")
            if coherence_score < config.REVISION_COHERENCE_THRESHOLD:
                needs_revision = True
                reasons.append(f"Low coherence with previous chapter (Score: {coherence_score:.4f}, Threshold: {config.REVISION_COHERENCE_THRESHOLD}).")
        else:
            logger.warning(f"Could not perform coherence check for ch {chapter_number} (missing current or previous embedding).")
    else: 
        logger.info("Skipping coherence check for Chapter 1.")
        await current_embedding_task # Ensure embedding is generated for Ch1 even if not used for coherence

    # Comprehensive LLM-based evaluation
    llm_eval_results = await comprehensive_chapter_evaluation(agent, draft_text, chapter_number, previous_chapters_context)

    consistency_issues_str = llm_eval_results.get("consistency_issues")
    plot_deviation_reason_str = llm_eval_results.get("plot_arc_deviation")
    thematic_issues_str = llm_eval_results.get("thematic_issues") # Added

    if consistency_issues_str:
        needs_revision = True
        reasons.append(f"Consistency issues identified:\n{consistency_issues_str}")
    
    if plot_deviation_reason_str:
        needs_revision = True
        reasons.append(f"Plot Arc Deviation: {plot_deviation_reason_str}")
    
    if thematic_issues_str: # Added thematic issues to revision trigger
        needs_revision = True
        reasons.append(f"Thematic Issues: {thematic_issues_str}")
            
    logger.info(f"Evaluation for Ch {chapter_number} complete. Needs revision: {needs_revision}.")
    return {
        "needs_revision": needs_revision, 
        "reasons": reasons, 
        "coherence_score": coherence_score, 
        "consistency_issues": consistency_issues_str, 
        "plot_deviation_reason": plot_deviation_reason_str,
        "thematic_issues": thematic_issues_str  # Added to match EvaluationResult type requirements
    }
