# chapter_drafting_logic.py
"""
Handles the generation of the initial chapter draft for the SAGA system.
"""
import logging
import json
from typing import Tuple, Optional, List

import config
import llm_interface
from type import SceneDetail # Assuming this is in type.py
from state_manager import state_manager

logger = logging.getLogger(__name__)

async def generate_chapter_draft_logic(agent, chapter_number: int, plot_point_focus: Optional[str], context: str, chapter_plan: Optional[List[SceneDetail]]) -> Tuple[Optional[str], Optional[str]]:
    """Generates the initial draft text for a chapter.
    'agent' is an instance of NovelWriterAgent.
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
                plan_section_for_prompt = f"**Detailed Scene Plan (MUST BE FOLLOWED CLOSELY):**\n```json\n{plan_json_str}\n```\n"
                logger.info(f"Using detailed scene plan for Ch {chapter_number} draft generation.")
            except TypeError as e:
                logger.error(f"Could not serialize chapter plan to JSON for prompt: {e}. Plan: {chapter_plan}")
                plan_section_for_prompt = f"**Chapter Plan Note:** Error formatting plan. Rely on Plot Point Focus.\n**Plot Point Focus for THIS Chapter:** {plot_point_focus}\n"
        else: 
            plan_section_for_prompt = f"**Chapter Plan Note:** No detailed scene plan available. Rely on the Overall Plot Point Focus.\n**Overall Plot Point Focus for THIS Chapter:** {plot_point_focus}\n"
    else: 
        plan_section_for_prompt = f"**Chapter Plan Note:** Detailed agentic planning is disabled. Rely on the Overall Plot Point Focus.\n**Overall Plot Point Focus for THIS Chapter:** {plot_point_focus}\n"
        
    char_profiles_json = json.dumps(state_manager.get_filtered_character_profiles_for_prompt(agent, chapter_number - 1), indent=2, ensure_ascii=False, default=str)
    world_building_json = json.dumps(state_manager.get_filtered_world_data_for_prompt(agent, chapter_number - 1), indent=2, ensure_ascii=False, default=str)

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
**Context from Previous Chapters (Summaries/Snippets - note any 'Provisional' markers):**
--- BEGIN CONTEXT ---
{context if context.strip() else "No previous context (e.g., this is Chapter 1 or context retrieval failed)."}
--- END CONTEXT ---

**Writing Instructions:**
1. Write a compelling and engaging chapter, aiming for at least {config.MIN_ACCEPTABLE_DRAFT_LENGTH} characters.
2. If a **Detailed Scene Plan** is provided, adhere to it closely, fleshing out each scene.
3. If no detailed plan is available, focus on achieving the **Overall Plot Point Focus** for this chapter.
4. Maintain consistency with all provided information (Story Bible, World Building, Character Profiles, Previous Context).
5. Ensure a smooth narrative flow and vivid prose suitable for the genre '{agent.plot_outline.get('genre', 'story')}'.
6. **Output ONLY the chapter text itself.** Do NOT include "Chapter X" headers, titles, author commentary, or any meta-discussion.

--- BEGIN CHAPTER {chapter_number} TEXT ---
"""
    raw_llm_text = await llm_interface.async_call_llm(
        model_name=config.DRAFTING_MODEL,
        prompt=prompt, 
        temperature=0.6 
    )
    if not raw_llm_text:
        logger.error(f"LLM returned no content for Ch {chapter_number} draft.")
        return None, None 
        
    cleaned_text = llm_interface.clean_model_response(raw_llm_text)
    if not cleaned_text or len(cleaned_text) < config.MIN_ACCEPTABLE_DRAFT_LENGTH:
         logger.error(f"Ch {chapter_number} draft is too short ({len(cleaned_text or '')} chars) after cleaning. Min required: {config.MIN_ACCEPTABLE_DRAFT_LENGTH}. Raw LLM output snippet: '{raw_llm_text[:200]}...'")
         return None, raw_llm_text 
         
    logger.info(f"Generated initial draft for ch {chapter_number} (Length: {len(cleaned_text)} chars).")
    return cleaned_text, raw_llm_text
