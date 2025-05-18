# chapter_drafting_logic.py
"""
Handles the generation of the initial chapter draft for the SAGA system.
"""
import logging
import json
from typing import Tuple, Optional, List

import config
import llm_interface
from type import SceneDetail 
from state_manager import state_manager
from prompt_data_getters import (
    get_filtered_character_profiles_for_prompt,
    get_filtered_world_data_for_prompt
)
from context_generation_logic import generate_hybrid_chapter_context_logic 

logger = logging.getLogger(__name__)

async def generate_chapter_draft_logic(agent, chapter_number: int, plot_point_focus: Optional[str], hybrid_context: str, chapter_plan: Optional[List[SceneDetail]]) -> Tuple[Optional[str], Optional[str]]:
    """Generates the initial draft text for a chapter using HYBRID CONTEXT.
    'agent' is an instance of NovelWriterAgent.
    'hybrid_context' contains both semantic context and KG facts.
    Returns (cleaned_text, raw_llm_text).
    """
    if not plot_point_focus:
        plot_point_focus = "Continue the narrative logically, focusing on character development and plot progression based on previous events."
        logger.warning(f"Plot point focus was None for Ch {chapter_number} draft generation. Using generic fallback.")
    
    plan_section_for_prompt = ""
    if config.ENABLE_AGENTIC_PLANNING:
        if chapter_plan and isinstance(chapter_plan, list): 
            try:
                plan_json_str = json.dumps(chapter_plan, indent=2, ensure_ascii=False)
                # Truncate plan if too long for the prompt, focusing on first few scenes
                max_plan_chars_for_prompt = config.MAX_CONTEXT_LENGTH // 4 # Example: 1/4th of total context for plan (was //5)
                if len(plan_json_str) > max_plan_chars_for_prompt:
                    # A more sophisticated truncation might try to preserve full scene objects
                    plan_json_str = plan_json_str[:max_plan_chars_for_prompt] + "\n... (plan truncated in prompt)"
                    logger.warning(f"Chapter plan for Ch {chapter_number} was truncated for the prompt.")
                
                plan_section_for_prompt = f"**Detailed Scene Plan (MUST BE FOLLOWED CLOSELY):**\n```json\n{plan_json_str}\n```\n"
                logger.info(f"Using detailed scene plan for Ch {chapter_number} draft generation.")
            except TypeError as e:
                logger.error(f"Could not serialize chapter plan to JSON for prompt: {e}. Plan: {chapter_plan}")
                plan_section_for_prompt = f"**Chapter Plan Note:** Error formatting plan. Rely on Plot Point Focus.\n**Plot Point Focus for THIS Chapter:** {plot_point_focus}\n"
        else: 
            plan_section_for_prompt = f"**Chapter Plan Note:** No detailed scene plan available. Rely on the Overall Plot Point Focus.\n**Overall Plot Point Focus for THIS Chapter:** {plot_point_focus}\n"
    else: 
        plan_section_for_prompt = f"**Chapter Plan Note:** Detailed agentic planning is disabled. Rely on the Overall Plot Point Focus.\n**Overall Plot Point Focus for THIS Chapter:** {plot_point_focus}\n"
        
    char_profiles_data = await get_filtered_character_profiles_for_prompt(agent, chapter_number - 1)
    char_profiles_json = json.dumps(char_profiles_data, indent=2, ensure_ascii=False, default=str)
    world_building_data = await get_filtered_world_data_for_prompt(agent, chapter_number - 1)
    world_building_json = json.dumps(world_building_data, indent=2, ensure_ascii=False, default=str)


    prompt = f"""/no_think
You are an expert novelist tasked with writing Chapter {chapter_number} of the novel titled "{agent.plot_outline.get('title', 'Untitled Novel')}".
**Story Bible / Core Information:**
  - Genre: {agent.plot_outline.get('genre', 'N/A')}
  - Central Theme: {agent.plot_outline.get('theme', 'N/A')}
  - Protagonist: {agent.plot_outline.get('protagonist_name', 'N/A')}
  - Protagonist's Character Arc: {agent.plot_outline.get('character_arc', 'N/A')}

{plan_section_for_prompt}

**World Building Notes (JSON format - pay attention to any 'prompt_notes' indicating provisional data from previous unrevised chapters):**
```json
{world_building_json}
```
**Character Profiles (JSON format - pay attention to any 'prompt_notes' indicating provisional data from previous unrevised chapters):**
```json
{char_profiles_json}
```
**Hybrid Context (Semantic Context for Flow & KG Facts for Canon):**
--- BEGIN HYBRID CONTEXT ---
{hybrid_context if hybrid_context.strip() else "No previous context (e.g., this is Chapter 1 or context retrieval failed)."}
--- END HYBRID CONTEXT ---

**Writing Instructions:**
1. Write a compelling and engaging chapter, aiming for a substantial length of at least {config.MIN_ACCEPTABLE_DRAFT_LENGTH} characters, ideally closer to {config.TARGET_DRAFT_LENGTH_UPPER_BOUND} characters of narrative text.
2. If a **Detailed Scene Plan** is provided, adhere to it closely. For each scene, pay particular attention to its specified 'summary', 'key_dialogue_points', 'setting_details', and **especially its 'scene_focus_elements'**. Use the 'scene_focus_elements' to guide you in elaborating, adding depth, and expanding the narrative to make each scene substantial and contribute to the overall chapter length target.
3. If no detailed plan is available, focus on achieving the **Overall Plot Point Focus** for this chapter.
4. Maintain consistency with all provided information (Story Bible, World Building, Character Profiles, Previous Context).
   - **Crucially, the `KEY RELIABLE KG FACTS` section within the `HYBRID CONTEXT` provides established canon that MUST be respected.**
   - The `SEMANTIC CONTEXT` section within the `HYBRID CONTEXT` should guide narrative flow, tone, and recall of recent events.
5. Ensure a smooth narrative flow and vivid prose suitable for the genre '{agent.plot_outline.get('genre', 'story')}'.
6. **Employ 'showing' over 'telling':** Use vivid descriptions, sensory details, character actions, internal monologues, and nuanced dialogue to convey information, atmosphere, and emotions.
7. **Fully develop dialogue exchanges:** Allow characters to express themselves naturally, incorporating pauses, subtext, emotional reactions, and non-verbal cues. Don't shy away from longer conversations if they serve character or plot.
8. **Thoroughly explore the protagonist's (and other key characters') thoughts, feelings, and internal reactions** to the unfolding events and interactions. Dedicate space to their internal processing.
9. **Output ONLY the chapter text itself.** Do NOT include "Chapter X" headers, titles, author commentary, or any meta-discussion.

--- BEGIN CHAPTER {chapter_number} TEXT ---
"""
    logger.info(f"Calling LLM ({config.DRAFTING_MODEL}) for Ch {chapter_number} draft. Target length: {config.MIN_ACCEPTABLE_DRAFT_LENGTH}-{config.TARGET_DRAFT_LENGTH_UPPER_BOUND} chars.")
    # Drafting is critical, allow fallback if primary drafting model fails
    raw_llm_text = await llm_interface.async_call_llm(
        model_name=config.DRAFTING_MODEL,
        prompt=prompt, 
        temperature=0.6,
        allow_fallback=True 
    )
    if not raw_llm_text:
        logger.error(f"LLM returned no content for Ch {chapter_number} draft (primary and potential fallback failed).")
        return None, None 
        
    cleaned_text = llm_interface.clean_model_response(raw_llm_text)
    if not cleaned_text or len(cleaned_text) < config.MIN_ACCEPTABLE_DRAFT_LENGTH:
         logger.error(f"Ch {chapter_number} draft is too short ({len(cleaned_text or '')} chars) after cleaning. Min required: {config.MIN_ACCEPTABLE_DRAFT_LENGTH}. Target: {config.MIN_ACCEPTABLE_DRAFT_LENGTH}-{config.TARGET_DRAFT_LENGTH_UPPER_BOUND}. Raw LLM output snippet: '{raw_llm_text[:200]}...'")
         return None, raw_llm_text 
         
    logger.info(f"Generated initial draft for ch {chapter_number} (Length: {len(cleaned_text)} chars).")
    return cleaned_text, raw_llm_text