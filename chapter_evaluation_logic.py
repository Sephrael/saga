# chapter_evaluation_logic.py
"""
Handles the evaluation of chapter drafts for consistency, plot arc alignment, etc.,
for the SAGA system.
"""
import logging
import json
import asyncio
from typing import Optional

import config
import llm_interface
import utils # For numpy_cosine_similarity
from type import EvaluationResult # Assuming this is in type.py
# Import knowledge management logic for summarization, used in plot arc validation
from knowledge_management_logic import summarize_chapter_text_logic
# Import prompt data getters
from state_manager import state_manager
from prompt_data_getters import get_filtered_character_profiles_for_prompt, get_filtered_world_data_for_prompt

logger = logging.getLogger(__name__)

async def check_draft_consistency_logic(agent, chapter_draft_text: Optional[str], chapter_number: int, previous_chapters_context: str) -> Optional[str]:
    """Checks chapter draft for consistency against plot, characters, world, KG, and previous context.
    'agent' is an instance of NovelWriterAgent.
    Returns a string of issues, or None if consistent.
    """
    if not chapter_draft_text:
        logger.debug(f"Consistency check skipped for Ch {chapter_number}: empty draft text.")
        return None
    
    draft_snippet = chapter_draft_text[:config.KNOWLEDGE_UPDATE_SNIPPET_SIZE] 
    context_snippet = previous_chapters_context[:config.KNOWLEDGE_UPDATE_SNIPPET_SIZE // 2]
    
    protagonist_name = agent.plot_outline.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME)
    kg_chapter_limit = chapter_number - 1 

    kg_loc_task = state_manager.async_get_most_recent_value(protagonist_name, "located_in", kg_chapter_limit, include_provisional=False)
    kg_status_task = state_manager.async_get_most_recent_value(protagonist_name, "status_is", kg_chapter_limit, include_provisional=False)
    
    kg_location, kg_status = await asyncio.gather(kg_loc_task, kg_status_task)
    
    kg_facts_for_prompt: list[str] = []
    if kg_location: kg_facts_for_prompt.append(f"- {protagonist_name}'s last reliably known location: {kg_location}.")
    if kg_status: kg_facts_for_prompt.append(f"- {protagonist_name}'s last reliably known status: {kg_status}.")
        
    kg_check_results_text = "**Key Reliable KG Facts (from pre-novel & previous chapters):**\n" + "\n".join(kg_facts_for_prompt) + "\n" if kg_facts_for_prompt else "**Key Reliable KG Facts:** None available or protagonist not tracked.\n"

    char_profiles_for_prompt = await get_filtered_character_profiles_for_prompt(agent, kg_chapter_limit)
    world_building_for_prompt =  await get_filtered_world_data_for_prompt(agent, kg_chapter_limit)

    prompt = f"""/no_think
You are a Continuity Editor. Your task is to analyze the provided Draft Snippet for Chapter {chapter_number} for inconsistencies.
Compare the Draft Snippet against the following established information:
1. Plot Outline (overall story direction)
2. Character Profiles (character traits, status, history - note 'provisional' markers if present in prompt_notes)
3. World Building (locations, rules, lore - note 'provisional' markers if present in prompt_notes)
4. Key Reliable KG Facts (facts considered canon from previous chapters or pre-novel setup)
5. Previous Context (narrative flow and recent events)
6. The Draft Snippet's own internal consistency.

Your goal is to identify specific, objective contradictions or deviations from this established information.
Prioritize clear contradictions with facts from the Plot Outline, Character Profiles, World Building, and reliable KG Facts.

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
**Previous Context (Snippet from prior chapters):**
--- PREVIOUS CONTEXT ---
{context_snippet if context_snippet.strip() else "N/A (e.g., this is Chapter 1 or context retrieval failed)."}
--- END PREVIOUS CONTEXT ---

**Chapter {chapter_number} Draft Snippet (to analyze):**
--- DRAFT SNIPPET ---
{draft_snippet}
--- END DRAFT SNIPPET ---

**Analysis Task:**
List ONLY specific, objective contradictions, inconsistencies, or significant deviations found in the Draft Snippet.
DO NOT list items that are consistent or aligned with established information.
If NO inconsistencies are found, respond with the single word: None

IMPORTANT: Only list actual contradictions. DO NOT list things that are consistent.
"""
    response_raw = await llm_interface.async_call_llm(
        model_name=config.CONSISTENCY_CHECK_MODEL,
        prompt=prompt, 
        temperature=0.6, 
        max_tokens=config.MAX_CONSISTENCY_TOKENS
    ) 
    response_cleaned = llm_interface.clean_model_response(response_raw).strip()

    if not response_cleaned or response_cleaned.lower() == "none":
        logger.info(f"Consistency check passed for ch {chapter_number}. No issues reported by LLM.")
        return None
    
    logger.warning(f"Consistency issues reported for ch {chapter_number}:\n{response_cleaned}")
    return response_cleaned

async def validate_draft_plot_arc_logic(agent, chapter_draft_text: Optional[str], chapter_number: int) -> Optional[str]:
    """Validates if the chapter draft aligns with its intended plot point.
    'agent' is an instance of NovelWriterAgent.
    Returns a reason string if deviation, or None if aligned.
    """
    if not chapter_draft_text:
        logger.debug(f"Plot arc validation skipped for Ch {chapter_number}: empty draft text.")
        return None
            
    plot_point_focus, plot_point_index = agent._get_plot_point_info(chapter_number)
    if plot_point_focus is None:
        logger.warning(f"Plot arc validation skipped for ch {chapter_number}: No plot point focus available for this chapter.")
        return None 
    
    logger.info(f"Validating plot arc for ch {chapter_number} against Plot Point {plot_point_index + 1}: '{plot_point_focus[:100]}...'")
    
    summary = await summarize_chapter_text_logic(chapter_draft_text, chapter_number)
    validation_text_content = summary if summary and len(summary) > 50 else chapter_draft_text[:1500]
    
    if not validation_text_content.strip():
        logger.warning(f"Plot arc validation skipped for Ch {chapter_number}: no text content for validation (summary/snippet empty).")
        return None
            
    protagonist_name = agent.plot_outline.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME)
    prompt = f"""/no_think
You are a Story Analyst. Your task is to determine if the provided Chapter {chapter_number} Text (featuring protagonist: {protagonist_name}) successfully addresses its Intended Plot Point.

**Intended Plot Point for Chapter {chapter_number} (Plot Point #{plot_point_index + 1}):**
"{plot_point_focus}"

**Chapter {chapter_number} Text (Summary or Snippet):**
"{validation_text_content}"

**Evaluation Question:**
Does the provided Chapter Text (Summary/Snippet) clearly and substantially address or advance the Intended Plot Point?

**Response Format (CRITICAL):**
Respond with ONLY ONE of the following:
- `Yes` (if the chapter text aligns well with and advances the intended plot point)
- `No, because [specific reason]` (if the chapter text deviates, fails to address, or only superficially touches upon the intended plot point. Provide a concise, 1-2 sentence explanation for the deviation).

**Your Response:**"""
    validation_response_raw = await llm_interface.async_call_llm(
        model_name=config.INITIAL_SETUP_MODEL, 
        prompt=prompt, 
        temperature=0.6, 
        max_tokens=config.MAX_PLOT_VALIDATION_TOKENS
    ) 
    cleaned_plot_response = llm_interface.clean_model_response(validation_response_raw).strip()
    
    if cleaned_plot_response.lower().startswith("yes"):
        logger.info(f"Plot arc validation passed for ch {chapter_number}.")
        return None
    elif cleaned_plot_response.lower().startswith("no, because"):
        reason = cleaned_plot_response[len("no, because"):].strip()
        if not reason: reason = "LLM indicated deviation but provided no specific reason."
        logger.warning(f"Plot arc deviation identified for ch {chapter_number}: {reason}")
        return reason
    
    logger.warning(f"Plot arc validation for ch {chapter_number} produced an ambiguous response: '{cleaned_plot_response}'. Assuming alignment as a fallback.")
    return None

async def evaluate_chapter_draft_logic(agent, draft_text: str, chapter_number: int, previous_chapters_context: str) -> EvaluationResult:
    """Evaluates a chapter draft for coherence, consistency, and plot arc alignment.
    'agent' is an instance of NovelWriterAgent.
    """
    logger.info(f"Evaluating chapter {chapter_number} draft (length: {len(draft_text)} chars)...")
    
    reasons: list[str] = []
    needs_revision = False
    coherence_score: Optional[float] = None
    consistency_issues_str: Optional[str] = None 
    plot_deviation_reason_str: Optional[str] = None

    if len(draft_text) < config.MIN_ACCEPTABLE_DRAFT_LENGTH:
        needs_revision = True
        reasons.append(f"Draft is too short ({len(draft_text)} chars). Minimum required: {config.MIN_ACCEPTABLE_DRAFT_LENGTH}.")

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
        await current_embedding_task 

    eval_tasks = []
    if config.REVISION_CONSISTENCY_TRIGGER:
        eval_tasks.append(check_draft_consistency_logic(agent, draft_text, chapter_number, previous_chapters_context))
    else: 
        eval_tasks.append(asyncio.sleep(0, result=None)) 

    if config.PLOT_ARC_VALIDATION_TRIGGER:
        eval_tasks.append(validate_draft_plot_arc_logic(agent, draft_text, chapter_number))
    else: 
        eval_tasks.append(asyncio.sleep(0, result=None))
            
    consistency_result, plot_arc_result = await asyncio.gather(*eval_tasks)

    if config.REVISION_CONSISTENCY_TRIGGER and consistency_result:
        consistency_issues_str = consistency_result
        needs_revision = True
        reasons.append(f"Consistency issues identified:\n{consistency_issues_str}")
    
    if config.PLOT_ARC_VALIDATION_TRIGGER and plot_arc_result:
        plot_deviation_reason_str = plot_arc_result
        needs_revision = True
        reasons.append(f"Plot Arc Deviation: {plot_deviation_reason_str}")
            
    logger.info(f"Evaluation for Ch {chapter_number} complete. Needs revision: {needs_revision}.")
    return {
        "needs_revision": needs_revision, 
        "reasons": reasons, 
        "coherence_score": coherence_score, 
        "consistency_issues": consistency_issues_str, 
        "plot_deviation_reason": plot_deviation_reason_str
    }
