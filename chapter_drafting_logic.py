# chapter_drafting_logic.py
"""
Handles the generation of the initial chapter draft for the SAGA system.
Context data for prompts is now formatted as plain text.
"""
import logging
from typing import Tuple, Optional, List

import config
from llm_interface import llm_service, count_tokens, truncate_text_by_tokens
from type import SceneDetail
import utils
# No direct state_manager import needed here as orchestrator passes data
from prompt_data_getters import (
    get_filtered_character_profiles_for_prompt_plain_text, 
    get_filtered_world_data_for_prompt_plain_text,       
    # Hybrid context already provides KG facts in plain text format
)
from context_generation_logic import generate_hybrid_chapter_context_logic

logger = logging.getLogger(__name__)

def _format_scene_plan_for_prompt(
    chapter_plan: List[SceneDetail],
    model_name_for_tokens: str,
    max_tokens_budget: int,
) -> str:
    """Wrapper around utils.format_scene_plan_for_prompt for backward compatibility."""
    return utils.format_scene_plan_for_prompt(chapter_plan, model_name_for_tokens, max_tokens_budget)



async def generate_chapter_draft_logic(agent, chapter_number: int, plot_point_focus: Optional[str], hybrid_context: str, chapter_plan: Optional[List[SceneDetail]]) -> Tuple[Optional[str], Optional[str]]:
    """Generates the initial draft text for a chapter using HYBRID CONTEXT.
    Context data and scene plan are now formatted as plain text.
    'agent' here refers to the orchestrator or an object holding plot_outline etc.
    """
    if not plot_point_focus:
        plot_point_focus = "Continue the narrative logically, focusing on character development and plot progression based on previous events."
        logger.warning(f"Plot point focus was None for Ch {chapter_number} draft generation. Using generic fallback.")

    plan_section_for_prompt_parts: List[str] = []
    if config.ENABLE_AGENTIC_PLANNING:
        if chapter_plan and isinstance(chapter_plan, list):
            max_plan_tokens_for_prompt = config.MAX_CONTEXT_TOKENS // 3 
            plan_section_for_prompt_parts.append(_format_scene_plan_for_prompt(chapter_plan, config.DRAFTING_MODEL, max_plan_tokens_for_prompt))
            logger.info(f"Using detailed scene plan (plain text) for Ch {chapter_number} draft generation.")
        else:
            plan_section_for_prompt_parts.append(f"**Chapter Plan Note:** No detailed scene plan available. Rely on the Overall Plot Point Focus.\n**Overall Plot Point Focus for THIS Chapter:** {plot_point_focus}\n")
    else:
        plan_section_for_prompt_parts.append(f"**Chapter Plan Note:** Detailed agentic planning is disabled. Rely on the Overall Plot Point Focus.\n**Overall Plot Point Focus for THIS Chapter:** {plot_point_focus}\n")
    plan_section_for_prompt_str = "".join(plan_section_for_prompt_parts)

    char_profiles_plain_text = await get_filtered_character_profiles_for_prompt_plain_text(agent, chapter_number - 1)
    world_building_plain_text = await get_filtered_world_data_for_prompt_plain_text(agent, chapter_number - 1)

    plot_outline_data = getattr(agent, 'plot_outline', agent.get('plot_outline_full', agent.get('plot_outline', {})))

    prompt_lines = []
    if config.ENABLE_LLM_NO_THINK_DIRECTIVE:
        prompt_lines.append("/no_think")
    
    prompt_lines.extend([
        f"You are an expert novelist tasked with writing Chapter {chapter_number} of the novel titled \"{plot_outline_data.get('title', 'Untitled Novel')}\".",
        "**Story Bible / Core Information:**",
        f"  - Genre: {plot_outline_data.get('genre', 'N/A')}",
        f"  - Central Theme: {plot_outline_data.get('theme', 'N/A')}",
        f"  - Protagonist: {plot_outline_data.get('protagonist_name', 'N/A')}",
        f"  - Protagonist's Character Arc: {plot_outline_data.get('character_arc', 'N/A')}",
        "",
        plan_section_for_prompt_str,
        "",
        "**World Building Notes (Plain Text - pay attention to any 'prompt_notes' indicating provisional data):**",
        "```text",
        world_building_plain_text if world_building_plain_text.strip() else "No specific world building notes provided for this chapter's context.",
        "```",
        "**Character Profiles (Plain Text - pay attention to any 'prompt_notes' indicating provisional status):**",
        "```text",
        char_profiles_plain_text if char_profiles_plain_text.strip() else "No specific character profiles provided for this chapter's context.",
        "```",
        "**Hybrid Context (Semantic Context for Flow & KG Facts for Canon):**",
        "--- BEGIN HYBRID CONTEXT ---",
        hybrid_context if hybrid_context.strip() else "No previous context (e.g., this is Chapter 1 or context retrieval failed).",
        "--- END HYBRID CONTEXT ---",
        "",
        "**Writing Instructions:**",
        f"1. Write a compelling and engaging chapter, aiming for a substantial length of at least {config.MIN_ACCEPTABLE_DRAFT_LENGTH} characters of narrative text.",
        "2. If a **Detailed Scene Plan** is provided, adhere to it closely. For each scene, pay particular attention to its specified 'Summary', 'Key Dialogue Points', 'Setting Details', and **especially its 'Scene Focus Elements'**. Use the 'Scene Focus Elements' to guide you in elaborating, adding depth, and expanding the narrative to make each scene substantial and contribute to the overall chapter length target.",
        "3. If no detailed plan is available, focus on achieving the **Overall Plot Point Focus** for this chapter.",
        "4. Maintain consistency with all provided information (Story Bible, World Building, Character Profiles, Previous Context).",
        "   - **Crucially, the `KEY RELIABLE KG FACTS` section within the `HYBRID CONTEXT` provides established canon that MUST be respected.**",
        "   - The `SEMANTIC CONTEXT` section within the `HYBRID CONTEXT` should guide narrative flow, tone, and recall of recent events.",
        f"5. Ensure a smooth narrative flow and vivid prose suitable for the genre '{plot_outline_data.get('genre', 'story')}'.",
        "6. **Employ 'showing' over 'telling':** Use vivid descriptions, sensory details, character actions, internal monologues, and nuanced dialogue to convey information, atmosphere, and emotions.",
        "7. **Fully develop dialogue exchanges:** Allow characters to express themselves naturally, incorporating pauses, subtext, emotional reactions, and non-verbal cues. Don't shy away from longer conversations if they serve character or plot.",
        "8. **Thoroughly explore the protagonist's (and other key characters') thoughts, feelings, and internal reactions** to the unfolding events and interactions. Dedicate space to their internal processing.",
        "9. **Output ONLY the chapter text itself.** Do NOT include \"Chapter X\" headers, titles, author commentary, or any meta-discussion.",
        "",
        f"--- BEGIN CHAPTER {chapter_number} TEXT ---"
    ])
    prompt = "\n".join(prompt_lines)

    logger.info(f"Calling LLM ({config.DRAFTING_MODEL}) for Ch {chapter_number} draft. Target minimum length: {config.MIN_ACCEPTABLE_DRAFT_LENGTH} chars.")
    
    raw_llm_text_for_log, draft_usage = await llm_service.async_call_llm( 
        model_name=config.DRAFTING_MODEL,
        prompt=prompt,
        temperature=config.TEMPERATURE_DRAFTING, 
        max_tokens=None,
        allow_fallback=True,
        stream_to_disk=True,
        frequency_penalty=config.FREQUENCY_PENALTY_DRAFTING,
        presence_penalty=config.PRESENCE_PENALTY_DRAFTING,
        auto_clean_response=False 
    )

    if not raw_llm_text_for_log:
        logger.error(f"LLM returned no content for Ch {chapter_number} draft (primary and potential fallback failed).")
        return None, None 

    cleaned_text = llm_service.clean_model_response(raw_llm_text_for_log) # MODIFIED

    if not cleaned_text or len(cleaned_text) < 50:
        logger.error(f"Ch {chapter_number} draft has virtually no content after cleaning ({len(cleaned_text or '')} chars). Raw LLM output snippet: '{raw_llm_text_for_log[:200]}...'")
        return None, raw_llm_text_for_log 

    if len(cleaned_text) < config.MIN_ACCEPTABLE_DRAFT_LENGTH:
         logger.warning(
             f"Ch {chapter_number} draft is short ({len(cleaned_text)} chars) after cleaning, but will be passed for evaluation/revision. "
             f"Min required: {config.MIN_ACCEPTABLE_DRAFT_LENGTH}. "
             f"Snippet: '{cleaned_text[:200].replace(chr(10), ' ')}...'"
         )

    logger.info(f"Generated initial draft for ch {chapter_number} (Length: {len(cleaned_text)} chars).")
    return cleaned_text, raw_llm_text_for_log
