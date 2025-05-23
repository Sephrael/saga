# chapter_planning_logic.py
import logging
import re
import asyncio
from typing import List, Optional, Any, Dict 

import config
import llm_interface
from type import SceneDetail 
from prompt_data_getters import (
    get_character_state_snippet_for_prompt, 
    get_world_state_snippet_for_prompt,     
    get_reliable_kg_facts_for_drafting_prompt 
)
from state_manager import state_manager 
from parsing_utils import split_text_into_blocks, parse_key_value_block # MODIFIED

logger = logging.getLogger(__name__)

# This key_map maps normalized LLM keys to SceneDetail internal keys
SCENE_PLAN_KEY_MAP = {
    "scene": "scene_number", # LLM "SCENE" or "scene" -> normalized "scene"
    "summary": "summary",
    "characters_involved": "characters_involved",
    "key_dialogue_points": "key_dialogue_points",
    "setting_details": "setting_details",
    "scene_focus_elements": "scene_focus_elements",
    "contribution": "contribution"
}
SCENE_PLAN_LIST_INTERNAL_KEYS = ["key_dialogue_points", "scene_focus_elements", "characters_involved"]
# Special handling for "characters_involved" to be comma-separated
SCENE_PLAN_SPECIAL_LIST_HANDLING = {
    "characters_involved": {"separator": ","}
}


def _parse_llm_scene_plan_output(text: str, chapter_number: int) -> Optional[List[SceneDetail]]:
    """
    Parses plain text scene plan output from LLM using generalized parsing utilities.
    """
    scenes_data: List[Dict[str, Any]] = [] # Store as dict first, then cast to SceneDetail
    if not text or not text.strip():
        logger.warning(f"Plain text scene plan for Ch {chapter_number} is empty. No scenes parsed.")
        return None

    # Try splitting by "---" first
    scene_blocks_text = split_text_into_blocks(text, separator_regex_str=r'\n\s*---\s*\n')
    
    # If "---" split results in one block (or none) and it doesn't seem to contain distinct SCENE headers,
    # try splitting by "SCENE: <number>" pattern.
    # This is a common LLM output pattern where scenes might not be "---" separated.
    if not scene_blocks_text or (len(scene_blocks_text) == 1 and "SCENE:" not in scene_blocks_text[0].upper()):
        # Regex to split *before* "SCENE: <number>"
        # (?=...) is a positive lookahead, ensuring the delimiter is kept with the next block.
        # We need to re-join the "SCENE: <number>" line with its content if re.split separates them.
        raw_split_by_header = re.split(r'(^\s*SCENE:\s*\d+\s*$)', text.strip(), flags=re.IGNORECASE | re.MULTILINE)
        
        # Process raw_split_by_header: pair header with its content block
        combined_blocks = []
        current_block_lines = []
        for i, part in enumerate(raw_split_by_header):
            part_stripped = part.strip()
            if not part_stripped: continue

            is_header = re.match(r'^\s*SCENE:\s*\d+\s*$', part_stripped, flags=re.IGNORECASE | re.MULTILINE)
            if is_header:
                if current_block_lines: # Finalize previous block
                    combined_blocks.append("\n".join(current_block_lines))
                    current_block_lines = []
                current_block_lines.append(part_stripped) # Start new block with header
            elif current_block_lines: # If a block has started, append content
                current_block_lines.append(part_stripped)
            # Else: content before the first header, usually preamble - ignore if not added to a block.

        if current_block_lines: # Add the last block
            combined_blocks.append("\n".join(current_block_lines))
        
        scene_blocks_text = [block for block in combined_blocks if block.strip()]


    if not scene_blocks_text:
        logger.warning(f"No scene blocks found for Ch {chapter_number} after attempting splits.")
        return None

    for block_num, block_content in enumerate(scene_blocks_text):
        if not block_content.strip():
            continue
        
        logger.debug(f"Parsing scene block {block_num+1} for Ch {chapter_number}:\n{block_content[:200]}...")
        
        parsed_scene_dict = parse_key_value_block(
            block_text_or_lines=block_content,
            key_map=SCENE_PLAN_KEY_MAP,
            list_internal_keys=SCENE_PLAN_LIST_INTERNAL_KEYS,
            special_list_handling=SCENE_PLAN_SPECIAL_LIST_HANDLING
        )

        # Validate required keys (as per SceneDetail)
        required_keys_internal = set(SCENE_PLAN_KEY_MAP.values())
        
        # Ensure 'scene_number' is int and present, otherwise assign sequentially
        if "scene_number" not in parsed_scene_dict or not isinstance(parsed_scene_dict["scene_number"], int):
            logger.warning(
                f"Scene block {block_num+1} in Ch {chapter_number} missing or invalid 'scene_number'. "
                f"Assigning sequential: {len(scenes_data) + 1}. Parsed value: {parsed_scene_dict.get('scene_number')}"
            )
            parsed_scene_dict["scene_number"] = len(scenes_data) + 1
        
        missing_keys = required_keys_internal - set(parsed_scene_dict.keys())
        if missing_keys:
            logger.warning(f"Partial scene data parsed for Ch {chapter_number}, block {block_num+1}. Missing keys: {missing_keys}. Data: {parsed_scene_dict}")
            # Fill missing keys with defaults to try and form a valid SceneDetail
            for req_key in required_keys_internal:
                if req_key not in parsed_scene_dict:
                    if req_key in SCENE_PLAN_LIST_INTERNAL_KEYS:
                        parsed_scene_dict[req_key] = []
                    elif req_key == "scene_number": # Should be handled above
                        pass 
                    else: # String keys
                        parsed_scene_dict[req_key] = "N/A - Missing from LLM output"
        
        # Final type validation for list fields
        for list_key in ["characters_involved", "key_dialogue_points", "scene_focus_elements"]:
            if not isinstance(parsed_scene_dict.get(list_key), list):
                logger.warning(f"Scene {parsed_scene_dict.get('scene_number', 'N/A')} in Ch {chapter_number} has non-list for '{list_key}'. Correcting. Value: {parsed_scene_dict.get(list_key)}")
                parsed_scene_dict[list_key] = [str(parsed_scene_dict.get(list_key))] if parsed_scene_dict.get(list_key) else []

        scenes_data.append(parsed_scene_dict)

    if not scenes_data:
        logger.error(f"Failed to parse any valid scenes from LLM output for Ch {chapter_number}. Raw text: '{text[:500]}...'")
        return None
        
    # Cast to SceneDetail after ensuring all keys are present
    # The parse_key_value_block and subsequent checks should ensure dicts are compliant
    # If strict SceneDetail typing is needed, further validation here or in caller.
    return [scene for scene in scenes_data if isinstance(scene, dict)] # type: ignore


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
For each scene in the plan, provide the following information using clear labels (case-insensitive keys are fine, but use these display names):
- `SCENE:` (Sequential number for the scene within this chapter)
- `SUMMARY:` (A concise 1-2 sentence summary of what happens in the scene)
- `CHARACTERS INVOLVED:` (A comma-separated list of key character names)
- `KEY DIALOGUE POINTS:` (List 1-3 brief points outlining crucial dialogue or internal monologue, each on a new line starting with "- ")
- `SETTING DETAILS:` (Brief description of the specific setting/location)
- `SCENE FOCUS ELEMENTS:` (List 1-2 specific aspects for targeted elaboration, each on a new line starting with "- ")
- `CONTRIBUTION:` (A short explanation of how this scene contributes to the chapter's goals or plot)

Separate each complete scene block with a line containing only "---". If you don't use "---", ensure each scene starts clearly with "SCENE: <number>".

**Example Scene Output Format:**
SCENE: 1
SUMMARY: The protagonist, {protagonist_name}, discovers a cryptic message hidden in their family's old locket.
CHARACTERS INVOLVED: {protagonist_name}
KEY DIALOGUE POINTS:
- 'What could this symbol mean?'
- A sudden realization about a past event.
SETTING DETAILS: A dusty, forgotten attic in the protagonist's ancestral home, late at night.
SCENE FOCUS ELEMENTS:
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
    parsed_scenes_list_of_dicts = _parse_llm_scene_plan_output(cleaned_plan_text, chapter_number) # MODIFIED call
    
    if parsed_scenes_list_of_dicts:
        final_scenes_typed: List[SceneDetail] = []
        for i, scene_dict in enumerate(parsed_scenes_list_of_dicts):
            # Basic validation for SceneDetail structure (parser should ensure this)
            if not isinstance(scene_dict, dict):
                logger.warning(f"Parsed scene item {i+1} for ch {chapter_number} is not a dict. Skipping. Item: {scene_dict}")
                continue
            
            required_scene_keys_internal = set(SCENE_PLAN_KEY_MAP.values())
            if not required_scene_keys_internal.issubset(scene_dict.keys()):
                missing_k = required_scene_keys_internal - set(scene_dict.keys())
                logger.warning(f"Scene {i+1} from parser for ch {chapter_number} has missing keys ({missing_k}). Skipping. Scene: {scene_dict}")
                continue
            
            # Type checks (parser should ideally enforce this, but good to double check structure)
            valid_types = (
                isinstance(scene_dict.get("scene_number"), int) and
                isinstance(scene_dict.get("summary"), str) and scene_dict.get("summary", "").strip() and
                isinstance(scene_dict.get("characters_involved"), list) and 
                isinstance(scene_dict.get("key_dialogue_points"), list) and 
                isinstance(scene_dict.get("setting_details"), str) and scene_dict.get("setting_details", "").strip() and
                isinstance(scene_dict.get("scene_focus_elements"), list) and 
                isinstance(scene_dict.get("contribution"), str) and scene_dict.get("contribution", "").strip()
            )
            if not valid_types:
                logger.warning(f"Scene {i+1} from parser for ch {chapter_number} has invalid types or empty required strings. Skipping. Scene: {scene_dict}")
                continue
            
            final_scenes_typed.append(scene_dict) # type: ignore [arg-type] # It's a dict, should conform to SceneDetail by now

        if final_scenes_typed: 
            logger.info(f"Generated valid detailed scene plan for chapter {chapter_number} with {len(final_scenes_typed)} scenes from plain text.")
            return final_scenes_typed
        else:
            logger.error(f"Parsed list was empty or all scenes were invalid after parsing plain text for chapter {chapter_number}. Raw LLM output: '{plan_raw_text[:500]}...'")
            await agent._save_debug_output(chapter_number, "detailed_plan_invalid_or_empty_scenes_plain_text", plan_raw_text) 
            return None
    else:
        logger.error(f"Failed to parse a valid list of scenes from plain text for chapter {chapter_number}. Raw LLM output: '{plan_raw_text[:500]}...'")
        await agent._save_debug_output(chapter_number, "detailed_plan_parse_fail_plain_text", plan_raw_text)
        return None