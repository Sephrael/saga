# chapter_planning_logic.py
import logging
import json # Retained for dumping context to prompt, if complex structures are still easier this way for LLM to consume
import re
import asyncio
from typing import List, Optional, Any, Dict 

import config
import llm_interface
from type import SceneDetail 
from prompt_data_getters import (
    get_character_state_snippet_for_prompt, # Now returns plain text
    get_world_state_snippet_for_prompt,     # Now returns plain text
    get_reliable_kg_facts_for_drafting_prompt 
)
from state_manager import state_manager 

logger = logging.getLogger(__name__)

def _parse_plain_text_scene_plan(text: str, chapter_number: int) -> Optional[List[SceneDetail]]:
    """
    Parses plain text scene plan output from LLM.
    Expected format per scene:
    SCENE: <scene_number>
    SUMMARY: <summary_text>
    CHARACTERS_INVOLVED: <char1_text>, <char2_text>
    KEY_DIALOGUE_POINTS:
    - Point 1
    - Point 2
    SETTING_DETAILS: <setting_text>
    SCENE_FOCUS_ELEMENTS:
    - Focus 1
    - Focus 2
    CONTRIBUTION: <contribution_text>
    --- (separator or end of text)
    """
    scenes: List[SceneDetail] = []
    if not text or not text.strip():
        logger.warning(f"Plain text scene plan for Ch {chapter_number} is empty. No scenes parsed.")
        return None

    scene_blocks = re.split(r'\n\s*---\s*\n', text.strip(), flags=re.MULTILINE)
    if len(scene_blocks) == 1 and "SCENE:" not in scene_blocks[0].upper(): # If no "---" and no "SCENE:"
        # Try splitting by "SCENE:" keyword if it's a single block without ---
        scene_blocks = re.split(r'(?=^\s*SCENE:\s*\d+\s*$)', text.strip(), flags=re.IGNORECASE | re.MULTILINE)
        scene_blocks = [block for block in scene_blocks if block.strip()]


    current_scene: Optional[Dict[str, Any]] = None
    current_list_key: Optional[str] = None # For "key_dialogue_points" or "scene_focus_elements"

    key_map = {
        "scene": "scene_number",
        "summary": "summary",
        "characters_involved": "characters_involved", # expect comma-separated string
        "key_dialogue_points": "key_dialogue_points", # expect list
        "setting_details": "setting_details",
        "scene_focus_elements": "scene_focus_elements", # expect list
        "contribution": "contribution"
    }
    list_type_keys_internal = ["key_dialogue_points", "scene_focus_elements", "characters_involved_list_internal"] # characters_involved needs post-processing

    for block_num, block_content in enumerate(scene_blocks):
        block_content = block_content.strip()
        if not block_content: continue

        # Reset for new scene block potentially (though split should handle this)
        current_scene_data: Dict[str, Any] = {}
        active_list_key: Optional[str] = None
        active_list_values: List[str] = []

        lines = block_content.splitlines()
        for line_num, line_text in enumerate(lines):
            line = line_text.strip()
            if not line: continue

            # Finalize active list if current line doesn't continue it
            if active_list_key and not (line.startswith("- ") or line.startswith("* ")):
                if active_list_key in current_scene_data: # Should always be true if list was started
                    current_scene_data[active_list_key].extend(active_list_values)
                else:
                    current_scene_data[active_list_key] = active_list_values
                active_list_key = None
                active_list_values = []
            
            # Match "Key: Value"
            match = re.match(r"^\s*([A-Za-z0-9\s_]+):\s*(.*)$", line)
            if match:
                key_from_llm = match.group(1).strip().lower().replace(" ", "_")
                value_from_llm = match.group(2).strip()

                internal_key = key_map.get(key_from_llm)
                if internal_key:
                    if internal_key in list_type_keys_internal or key_from_llm in ["key_dialogue_points", "scene_focus_elements"]: # Handle list types
                        active_list_key = internal_key
                        active_list_values = []
                        if value_from_llm and not (value_from_llm.startswith("- ") or value_from_llm.startswith("* ")): # Content on same line
                            active_list_values.append(value_from_llm)
                        elif value_from_llm and (value_from_llm.startswith("- ") or value_from_llm.startswith("* ")): # List item on same line
                             active_list_values.append(value_from_llm[2:].strip())
                        # Initialize the list in the dict
                        if internal_key not in current_scene_data:
                            current_scene_data[internal_key] = []

                    elif internal_key == "characters_involved": # Special handling for comma-separated string
                        current_scene_data[internal_key] = [c.strip() for c in value_from_llm.split(',') if c.strip()]
                    elif internal_key == "scene_number":
                        try:
                            current_scene_data[internal_key] = int(value_from_llm)
                        except ValueError:
                            logger.warning(f"Invalid scene number '{value_from_llm}' in plan for Ch {chapter_number}, block {block_num+1}. Using sequential.")
                            current_scene_data[internal_key] = len(scenes) + 1
                    else: # Simple key-value
                        current_scene_data[internal_key] = value_from_llm
                else:
                    logger.debug(f"Scene plan parsing: Unknown key '{key_from_llm}' in Ch {chapter_number}, block {block_num+1}.")
            
            # Handle list items
            elif (line.startswith("- ") or line.startswith("* ")) and active_list_key:
                active_list_values.append(line[2:].strip())
            elif line.strip().lower().replace(":", "") in key_map and \
                 (key_map[line.strip().lower().replace(":", "")] in list_type_keys_internal or \
                  line.strip().lower().replace(":", "") in ["key_dialogue_points", "scene_focus_elements"]):
                # This is a list header on its own line
                active_list_key = key_map[line.strip().lower().replace(":", "")]
                active_list_values = []
                if active_list_key not in current_scene_data:
                     current_scene_data[active_list_key] = []


        # Finalize any list at the end of the block
        if active_list_key:
            if active_list_key in current_scene_data:
                current_scene_data[active_list_key].extend(active_list_values)
            else:
                current_scene_data[active_list_key] = active_list_values

        # Validate and add scene
        required_keys = {"scene_number", "summary", "characters_involved", "key_dialogue_points", "setting_details", "scene_focus_elements", "contribution"}
        if required_keys.issubset(current_scene_data.keys()):
            # Ensure list types are indeed lists
            for l_key in ["characters_involved", "key_dialogue_points", "scene_focus_elements"]:
                if not isinstance(current_scene_data.get(l_key), list):
                    logger.warning(f"Scene {current_scene_data.get('scene_number', 'N/A')} in Ch {chapter_number} has non-list for '{l_key}'. Correcting to empty list. Value: {current_scene_data.get(l_key)}")
                    current_scene_data[l_key] = [] # Correct to empty list

            scenes.append(current_scene_data) # type: ignore
        elif current_scene_data: # If some data was parsed but not a full scene
             logger.warning(f"Partial scene data parsed for Ch {chapter_number}, block {block_num+1}. Missing keys: {required_keys - set(current_scene_data.keys())}. Data: {current_scene_data}")


    if not scenes:
        logger.error(f"Failed to parse any valid scenes from plain text for Ch {chapter_number}. Raw text: '{text[:500]}...'")
        return None
    return scenes


async def plan_chapter_scenes_logic(agent, chapter_number: int) -> Optional[List[SceneDetail]]:
    if not config.ENABLE_AGENTIC_PLANNING:
        logger.info(f"Agentic planning disabled. Skipping detailed planning for Chapter {chapter_number}.")
        return None 

    logger.info(f"Planning Chapter {chapter_number} with detailed scenes...")
    plot_point_focus, plot_point_index = agent._get_plot_point_info(chapter_number)
    if plot_point_focus is None: 
        logger.error(f"Cannot plan chapter {chapter_number}: No plot point focus available.")
        return None

    context_summary = ""
    if chapter_number > 1:
        prev_chap_data = await state_manager.async_get_chapter_data_from_db(chapter_number - 1)
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
    kg_context_section = await get_reliable_kg_facts_for_drafting_prompt(agent, chapter_number, None)

    # Get plain text snippets
    character_state_snippet_plain_text = await get_character_state_snippet_for_prompt(agent, chapter_number)
    world_state_snippet_plain_text = await get_world_state_snippet_for_prompt(agent, chapter_number)

    future_plot_context = ""
    all_plot_points = agent.plot_outline.get('plot_points', [])
    if plot_point_index + 1 < len(all_plot_points):
        next_plot_point = all_plot_points[plot_point_index + 1]
        if isinstance(next_plot_point, str) and next_plot_point.strip():
            future_plot_context = f"\n**Anticipated Next Major Plot Point (for context, not this chapter's focus):**\n{next_plot_point.strip()}\n"

    prompt = f"""/no_think
You are a master plotter outlining **between {config.TARGET_SCENES_MIN} and {config.TARGET_SCENES_MAX} detailed scenes** for Chapter {chapter_number} of a novel.
**Novel Concept:**
  - Title: {agent.plot_outline.get('title', 'Untitled')}
  - Genre: {agent.plot_outline.get('genre', 'N/A')}
  - Theme: {agent.plot_outline.get('theme', 'N/A')}
  - Protagonist: {protagonist_name}
  - Protagonist's Arc: {agent.plot_outline.get('character_arc', 'N/A')}

**Mandatory Focus for THIS Chapter (Plot Point {plot_point_index + 1} of {len(agent.plot_outline.get('plot_points',[]))}):**
{plot_point_focus}
{future_plot_context}
**Recent Context from Previous Chapter(s):**
{context_summary if context_summary else "This is the first chapter, or no prior summary is available."}
{kg_context_section}
**Current Character States (Key Characters, based on profiles and recent developments - Plain Text):**
{character_state_snippet_plain_text} 

**Current World State (Relevant Locations/Elements, based on world-building - Plain Text):**
{world_state_snippet_plain_text}

**Task:**
Create a detailed plan of {config.TARGET_SCENES_MIN} to {config.TARGET_SCENES_MAX} scenes for Chapter {chapter_number}. 
For each scene in the plan, provide the following information using clear labels:
- `SCENE:` (Sequential number for the scene within this chapter)
- `SUMMARY:` (A concise 1-2 sentence summary of what happens in the scene)
- `CHARACTERS_INVOLVED:` (A comma-separated list of key character names)
- `KEY_DIALOGUE_POINTS:` (List 1-3 brief points outlining crucial dialogue or internal monologue, each on a new line starting with "- ")
- `SETTING_DETAILS:` (Brief description of the specific setting/location)
- `SCENE_FOCUS_ELEMENTS:` (List 1-2 specific aspects for targeted elaboration, each on a new line starting with "- ")
- `CONTRIBUTION:` (A short explanation of how this scene contributes to the chapter's goals or plot)

Separate each complete scene block with a line containing only "---".

**Example Scene Output Format:**
SCENE: 1
SUMMARY: The protagonist, {protagonist_name}, discovers a cryptic message hidden in their family's old locket.
CHARACTERS_INVOLVED: {protagonist_name}
KEY_DIALOGUE_POINTS:
- 'What could this symbol mean?'
- A sudden realization about a past event.
SETTING_DETAILS: A dusty, forgotten attic in the protagonist's ancestral home, late at night.
SCENE_FOCUS_ELEMENTS:
- Protagonist's mounting fear and obsessive curiosity
- Detailed sensory description of the attic and the locket
CONTRIBUTION: Serves as the inciting incident for this chapter's mystery/quest.
--- 
(Next scene would follow after the '---')

Output ONLY the scene plan text as described.
"""
    logger.info(f"Calling LLM ({config.PLANNING_MODEL}) for detailed scene plan for chapter {chapter_number} (target scenes: {config.TARGET_SCENES_MIN}-{config.TARGET_SCENES_MAX})...")
    plan_raw_text = await llm_interface.async_call_llm(
        model_name=config.PLANNING_MODEL,
        prompt=prompt, 
        temperature=0.6, 
        max_tokens=config.MAX_PLANNING_TOKENS,
        allow_fallback=True,
        stream_to_disk=True
    )
    
    cleaned_plan_text = llm_interface.clean_model_response(plan_raw_text)
    parsed_scenes: Optional[List[SceneDetail]] = _parse_plain_text_scene_plan(cleaned_plan_text, chapter_number)
    
    if parsed_scenes:
        valid_scenes_count = 0
        # Additional validation can be done here if _parse_plain_text_scene_plan isn't strict enough
        # For now, trust the parser if it returns a list.
        final_scenes = []
        for i, scene_item_any in enumerate(parsed_scenes): 
            # Ensure all required keys from SceneDetail are present and have basic correct types
            # The parser should ideally handle this. This is a double check.
            if not isinstance(scene_item_any, dict):
                logger.warning(f"Parsed scene item {i+1} for ch {chapter_number} is not a dict. Skipping. Item: {scene_item_any}")
                continue
            
            scene_item: Dict[str, Any] = scene_item_any
            required_scene_keys = {"scene_number", "summary", "characters_involved", "key_dialogue_points", "setting_details", "scene_focus_elements", "contribution"}
            if not required_scene_keys.issubset(scene_item.keys()):
                missing_keys = required_scene_keys - set(scene_item.keys())
                logger.warning(f"Scene {i+1} from parser for ch {chapter_number} has missing keys ({missing_keys}). Skipping. Scene: {scene_item}")
                continue
            
            # Type checks (parser should ideally enforce this, but good to double check structure)
            if not (isinstance(scene_item.get("scene_number"), int) and
                    isinstance(scene_item.get("summary"), str) and scene_item.get("summary", "").strip() and
                    isinstance(scene_item.get("characters_involved"), list) and 
                    isinstance(scene_item.get("key_dialogue_points"), list) and 
                    isinstance(scene_item.get("setting_details"), str) and scene_item.get("setting_details", "").strip() and
                    isinstance(scene_item.get("scene_focus_elements"), list) and 
                    isinstance(scene_item.get("contribution"), str) and scene_item.get("contribution", "").strip()):
                logger.warning(f"Scene {i+1} from parser for ch {chapter_number} has invalid types or empty required strings. Skipping. Scene: {scene_item}")
                continue
            final_scenes.append(scene_item) # type: ignore

        if final_scenes: 
            logger.info(f"Generated valid detailed scene plan for chapter {chapter_number} with {len(final_scenes)} scenes from plain text.")
            return final_scenes
        else:
            logger.error(f"Parsed list was empty or all scenes were invalid after parsing plain text for chapter {chapter_number}. Raw LLM output: '{plan_raw_text[:500]}...'")
            await agent._save_debug_output(chapter_number, "detailed_plan_invalid_or_empty_scenes_plain_text", plan_raw_text) 
            return None
    else:
        logger.error(f"Failed to parse a valid list of scenes from plain text for chapter {chapter_number}. Raw LLM output: '{plan_raw_text[:500]}...'")
        await agent._save_debug_output(chapter_number, "detailed_plan_parse_fail_plain_text", plan_raw_text)
        return None