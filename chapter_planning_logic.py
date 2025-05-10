# chapter_planning_logic.py
"""
Handles detailed chapter scene planning for the SAGA system.
"""
import logging
import json
import asyncio
from typing import List, Optional

import config
import llm_interface
from type import SceneDetail # Assuming this is in type.py
# Import prompt data getters
from prompt_data_getters import (
    get_character_state_snippet_for_prompt,
    get_world_state_snippet_for_prompt
)


logger = logging.getLogger(__name__)

async def plan_chapter_scenes_logic(agent, chapter_number: int) -> Optional[List[SceneDetail]]:
    """Asynchronously plans a chapter with detailed scenes if agentic planning is enabled.
    'agent' is an instance of NovelWriterAgent.
    """
    if not config.ENABLE_AGENTIC_PLANNING:
        logger.info(f"Agentic planning disabled by configuration. Skipping detailed planning for Chapter {chapter_number}.")
        return None 

    logger.info(f"Planning Chapter {chapter_number} with detailed scenes...")
    plot_point_focus, plot_point_index = agent._get_plot_point_info(chapter_number)
    if plot_point_focus is None: 
        logger.error(f"Cannot plan chapter {chapter_number}: No plot point focus available.")
        return None

    context_summary = ""
    if chapter_number > 1:
        prev_chap_data = await agent.db_manager.async_get_chapter_data_from_db(chapter_number - 1)
        if prev_chap_data:
            prev_summary = prev_chap_data.get('summary')
            prev_is_provisional = prev_chap_data.get('is_provisional', False)
            summary_prefix = "[Provisional Summary from Prev Ch] " if prev_is_provisional and prev_summary else "[Summary from Prev Ch] "
            if prev_summary: 
                context_summary += f"{summary_prefix}({chapter_number - 1}):\n{prev_summary[:1000].strip()}...\n"
            else: 
                prev_text = prev_chap_data.get('text', '')
                text_prefix = "[Provisional Text Snippet from Prev Ch] " if prev_is_provisional and prev_text else "[Text Snippet from Prev Ch] "
                if prev_text: 
                    context_summary += f"{text_prefix}({chapter_number - 1}):\n...{prev_text[-1000:].strip()}\n"
    
    protagonist_name = agent.plot_outline.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME)
    kg_chapter_limit = chapter_number - 1 
    
    kg_tasks = {
        "location": agent.db_manager.async_get_most_recent_value(protagonist_name, "located_in", kg_chapter_limit, include_provisional=False),
        "status": agent.db_manager.async_get_most_recent_value(protagonist_name, "status_is", kg_chapter_limit, include_provisional=False)
    }
    
    kg_results = await asyncio.gather(*kg_tasks.values())
    kg_fact_map = dict(zip(kg_tasks.keys(), kg_results))

    kg_facts_for_prompt: List[str] = []
    if kg_fact_map.get("location"): kg_facts_for_prompt.append(f"- {protagonist_name} is currently located in (reliable KG): {kg_fact_map['location']}.")
    if kg_fact_map.get("status"): kg_facts_for_prompt.append(f"- {protagonist_name}'s current status (reliable KG): {kg_fact_map['status']}.")
    
    kg_context_section = "**Relevant Reliable KG Facts (up to prev chapter/pre-novel):**\n" + "\n".join(kg_facts_for_prompt) + "\n" if kg_facts_for_prompt else ""

    character_state_snippet = get_character_state_snippet_for_prompt(agent, chapter_number)
    world_state_snippet = get_world_state_snippet_for_prompt(agent, chapter_number)

    prompt = f"""/no_think
You are a master plotter outlining **between 8 and 15 detailed scenes** for Chapter {chapter_number} of a novel.
**Novel Concept:**
  - Title: {agent.plot_outline.get('title', 'Untitled')}
  - Genre: {agent.plot_outline.get('genre', 'N/A')}
  - Theme: {agent.plot_outline.get('theme', 'N/A')}
  - Protagonist: {protagonist_name}
  - Protagonist's Arc: {agent.plot_outline.get('character_arc', 'N/A')}

**Mandatory Focus for THIS Chapter (Plot Point {plot_point_index + 1} of {len(agent.plot_outline.get('plot_points',[]))}):**
{plot_point_focus}

**Recent Context from Previous Chapter(s):**
{context_summary if context_summary else "This is the first chapter, or no prior summary is available."}
{kg_context_section}
**Current Character States (Key Characters, based on profiles and recent developments):**
{character_state_snippet} 

**Current World State (Relevant Locations/Elements, based on world-building):**
{world_state_snippet}

**Task:**
Create a detailed plan of 8 to 15 scenes for Chapter {chapter_number}. Each scene description in the plan MUST:
1. Directly advance or build towards the **Mandatory Focus** for this chapter.
2. Logically follow from the **Recent Context** and any provided **KG Facts**.
3. Involve relevant characters and world elements as appropriate.
4. Contribute to the **Protagonist's Arc** or the overall plot progression.
5. Be distinct from other scenes in this chapter plan.

**Output Format:**
Provide ONLY a single, valid JSON list of scene objects. Each object in the list must have the following keys:
  - `scene_number` (int): Sequential number for the scene within this chapter.
  - `summary` (str): A concise 1-2 sentence summary of what happens in the scene.
  - `characters_involved` (list[str]): A list of key character names involved in this scene.
  - `key_dialogue_points` (list[str]): 1-3 brief points outlining crucial dialogue or internal monologue.
  - `setting_details` (str): Brief description of the specific setting/location for this scene.
  - `contribution` (str): A short explanation of how this scene contributes to the chapter's goals or plot.

**Example JSON Scene (this would be one item in the list):**
```json
{{
  "scene_number": 1,
  "summary": "The protagonist, {protagonist_name}, discovers a cryptic message hidden in their family's old locket.",
  "characters_involved": ["{protagonist_name}"],
  "key_dialogue_points": ["'What could this symbol mean?'", "A sudden realization about a past event."],
  "setting_details": "A dusty, forgotten attic in the protagonist's ancestral home, late at night.",
  "contribution": "Serves as the inciting incident for this chapter's mystery/quest."
}}
```
Output the JSON list `[...]` directly. Do not include any other text, markdown, or explanation.
[
"""
    logger.info(f"Calling LLM ({config.PLANNING_MODEL}) for detailed scene plan for chapter {chapter_number}...")
    plan_raw = await llm_interface.async_call_llm(
        model_name=config.PLANNING_MODEL,
        prompt=prompt, 
        temperature=0.6, 
        max_tokens=config.MAX_PLANNING_TOKENS
    )
    
    parsed_plan: Optional[List[SceneDetail]] = await llm_interface.async_parse_llm_json_response(
        plan_raw, f"detailed scene plan for chapter {chapter_number}", expect_type=list
    )

    if parsed_plan and isinstance(parsed_plan, list) and len(parsed_plan) >= 1:
        valid_scenes: List[SceneDetail] = []
        required_scene_keys = {"scene_number", "summary", "characters_involved", "key_dialogue_points", "setting_details", "contribution"}
        for i, scene_item_any in enumerate(parsed_plan):
            if not isinstance(scene_item_any, dict):
                logger.warning(f"Scene item {i+1} in plan for ch {chapter_number} is not a dict. Skipping. Item: {scene_item_any}")
                continue
            
            scene_item = scene_item_any 
            
            if not required_scene_keys.issubset(scene_item.keys()):
                logger.warning(f"Scene {i+1} in plan for ch {chapter_number} has missing keys ({required_scene_keys - set(scene_item.keys())}). Skipping.")
                continue
            if not (isinstance(scene_item.get("scene_number"), int) and
                    isinstance(scene_item.get("summary"), str) and scene_item.get("summary", "").strip() and
                    isinstance(scene_item.get("characters_involved"), list) and
                    isinstance(scene_item.get("key_dialogue_points"), list) and
                    isinstance(scene_item.get("setting_details"), str) and scene_item.get("setting_details", "").strip() and
                    isinstance(scene_item.get("contribution"), str) and scene_item.get("contribution", "").strip()):
                logger.warning(f"Scene {i+1} in plan for ch {chapter_number} has invalid types or empty required strings. Skipping. Scene: {scene_item}")
                continue
            
            if not all(isinstance(c, str) for c in scene_item["characters_involved"]):
                 logger.warning(f"Scene {i+1} 'characters_involved' contains non-strings. Skipping. Scene: {scene_item}")
                 continue
            if not all(isinstance(d, str) for d in scene_item["key_dialogue_points"]):
                 logger.warning(f"Scene {i+1} 'key_dialogue_points' contains non-strings. Skipping. Scene: {scene_item}")
                 continue

            valid_scenes.append(scene_item) # type: ignore 
        
        if valid_scenes and len(valid_scenes) >= 1: 
            logger.info(f"Generated valid detailed scene plan for chapter {chapter_number} with {len(valid_scenes)} scenes.")
            return valid_scenes
        else:
            logger.error(f"All parsed scenes were invalid for chapter {chapter_number}. Raw LLM output: '{plan_raw[:500]}...'")
            await agent._save_debug_output(chapter_number, "detailed_plan_invalid_scenes", plan_raw)
            return None
    else:
        logger.error(f"Failed to generate/parse a valid JSON list for scene plan for chapter {chapter_number}. Raw LLM output: '{plan_raw[:500]}...'")
        await agent._save_debug_output(chapter_number, "detailed_plan_parse_fail", plan_raw)
        return None