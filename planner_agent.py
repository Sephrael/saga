# planner_agent.py
import logging
import re
import asyncio
from typing import List, Optional, Any, Dict, Tuple

import config
from llm_interface import llm_service
from type import SceneDetail
from prompt_data_getters import (
    get_character_state_snippet_for_prompt,
    get_world_state_snippet_for_prompt,
    get_reliable_kg_facts_for_drafting_prompt
)
# from state_manager import state_manager # No longer directly used
from data_access import chapter_queries # For get_chapter_data_from_db
from parsing_utils import split_text_into_blocks, parse_key_value_block

logger = logging.getLogger(__name__)

SCENE_PLAN_KEY_MAP = {
    "scene": "scene_number", # LLM often uses "SCENE:" or "Scene Number:"
    "scene_number": "scene_number", # Explicitly map "scene_number" as well
    "summary": "summary",
    "characters_involved": "characters_involved",
    "key_dialogue_points": "key_dialogue_points",
    "setting_details": "setting_details",
    "scene_focus_elements": "scene_focus_elements",
    "contribution": "contribution"
}
SCENE_PLAN_LIST_INTERNAL_KEYS = ["key_dialogue_points", "scene_focus_elements", "characters_involved"]
SCENE_PLAN_SPECIAL_LIST_HANDLING = {
    "characters_involved": {"separator": ","}
}

class PlannerAgent:
    def __init__(self, model_name: str = config.PLANNING_MODEL):
        self.model_name = model_name
        logger.info(f"PlannerAgent initialized with model: {self.model_name}")

    def _parse_llm_scene_plan_output(self, text: str, chapter_number: int) -> Optional[List[SceneDetail]]:
        """
        Parses plain text scene plan output from LLM using generalized parsing utilities.
        """
        scenes_data: List[Dict[str, Any]] = []
        if not text or not text.strip():
            logger.warning(f"Plain text scene plan for Ch {chapter_number} is empty. No scenes parsed.")
            return None

        # Pre-process: Remove potential markdown code block wrappers if LLM added them
        cleaned_text_for_parsing = text
        code_block_match = re.match(r"^\s*```(?:plaintext)?\s*\n(.*?)\n\s*```\s*$", text, re.DOTALL | re.IGNORECASE)
        if code_block_match:
            logger.debug(f"Removing markdown code block wrapper from scene plan output for Ch {chapter_number}.")
            cleaned_text_for_parsing = code_block_match.group(1).strip()
        
        if not cleaned_text_for_parsing.strip():
            logger.warning(f"Scene plan text became empty after removing code block wrapper for Ch {chapter_number}.")
            return None

        # Primary split by "---"
        scene_blocks_text = split_text_into_blocks(cleaned_text_for_parsing, separator_regex_str=r'\n\s*---\s*\n')
        
        # Fallback split logic: if "---" isn't used effectively, try splitting by "SCENE:"
        first_block_is_scene_like = False
        if scene_blocks_text:
            first_block_is_scene_like = re.match(r"^\s*(SCENE|Scene Number):\s*\d+", scene_blocks_text[0].strip(), re.IGNORECASE) is not None

        if not scene_blocks_text or (len(scene_blocks_text) == 1 and not first_block_is_scene_like):
            logger.debug(f"Primary '---' split for Ch {chapter_number} yielded {len(scene_blocks_text)} blocks (first block scene-like: {first_block_is_scene_like}). Attempting fallback split by 'SCENE:' header.")
            individual_scene_matches = re.finditer(r"(?s)((?:SCENE|Scene Number):\s*\d+.*?)(?=(?:\s*(?:SCENE|Scene Number):\s*\d+|$))", cleaned_text_for_parsing, re.IGNORECASE | re.MULTILINE)
            scene_blocks_text = [match.group(1).strip() for match in individual_scene_matches]
            if scene_blocks_text:
                logger.info(f"Fallback split for Ch {chapter_number} by 'SCENE:' header yielded {len(scene_blocks_text)} potential scene blocks.")
            else:
                logger.warning(f"Fallback split by 'SCENE:' header for Ch {chapter_number} also yielded no blocks. Original text snippet: '{cleaned_text_for_parsing[:300]}...'")


        if not scene_blocks_text:
            logger.warning(f"No scene blocks found for Ch {chapter_number} after attempting primary and fallback splits.")
            return None

        for block_num, block_content in enumerate(scene_blocks_text):
            if not block_content.strip():
                continue
            logger.debug(f"Parsing scene block {block_num+1} for Ch {chapter_number}:\n{block_content[:200]}...")
            
            if not re.match(r"^\s*(SCENE|Scene Number):\s*\d+", block_content.strip(), re.IGNORECASE):
                logger.warning(f"Scene block {block_num+1} for Ch {chapter_number} does not start with a valid 'SCENE:' or 'Scene Number:' header. Skipping block. Content: '{block_content[:100]}...'")
                continue

            parsed_scene_dict = parse_key_value_block(
                block_text_or_lines=block_content,
                key_map=SCENE_PLAN_KEY_MAP,
                list_internal_keys=SCENE_PLAN_LIST_INTERNAL_KEYS,
                special_list_handling=SCENE_PLAN_SPECIAL_LIST_HANDLING
            )

            if "scene_number" not in parsed_scene_dict or not isinstance(parsed_scene_dict["scene_number"], int):
                scene_num_match = re.match(r"^\s*(?:SCENE|Scene Number):\s*(\d+)", block_content.strip(), re.IGNORECASE)
                if scene_num_match:
                    try:
                        parsed_scene_dict["scene_number"] = int(scene_num_match.group(1))
                        logger.info(f"Successfully regex-extracted scene_number {parsed_scene_dict['scene_number']} for block {block_num+1} Ch {chapter_number}.")
                    except ValueError:
                         logger.warning(f"Regex extracted non-integer scene_number '{scene_num_match.group(1)}' for block {block_num+1} Ch {chapter_number}.")
                
                if "scene_number" not in parsed_scene_dict or not isinstance(parsed_scene_dict["scene_number"], int):
                    assigned_scene_num = len(scenes_data) + 1
                    logger.warning(
                        f"Scene block {block_num+1} in Ch {chapter_number} missing or invalid 'scene_number' after primary parse and regex fallback. "
                        f"Assigning sequential: {assigned_scene_num}. Parsed value: {parsed_scene_dict.get('scene_number')}"
                    )
                    parsed_scene_dict["scene_number"] = assigned_scene_num
            
            missing_keys = set(SCENE_PLAN_KEY_MAP.values()) - set(parsed_scene_dict.keys())
            if missing_keys:
                logger.warning(f"Partial scene data parsed for Ch {chapter_number}, block {block_num+1} (Scene Num: {parsed_scene_dict.get('scene_number', 'N/A')}). Missing keys: {missing_keys}. Data: {parsed_scene_dict}")
                for req_key in missing_keys: 
                    if req_key in SCENE_PLAN_LIST_INTERNAL_KEYS:
                        parsed_scene_dict[req_key] = []
                    elif req_key == "scene_number": pass 
                    else: parsed_scene_dict[req_key] = "N/A - Missing from LLM output"
            
            for list_key in SCENE_PLAN_LIST_INTERNAL_KEYS:
                if not isinstance(parsed_scene_dict.get(list_key), list):
                    logger.warning(f"Scene {parsed_scene_dict.get('scene_number', 'N/A')} in Ch {chapter_number} has non-list for '{list_key}'. Correcting. Value: {parsed_scene_dict.get(list_key)}")
                    parsed_scene_dict[list_key] = [str(parsed_scene_dict.get(list_key))] if parsed_scene_dict.get(list_key) else []

            scenes_data.append(parsed_scene_dict)

        if not scenes_data:
            logger.error(f"Failed to parse any valid scenes from LLM output for Ch {chapter_number}. Cleaned text for parsing: '{cleaned_text_for_parsing[:500]}...'")
            return None
        
        scenes_data.sort(key=lambda x: x.get("scene_number", float('inf')))
        
        return [scene for scene in scenes_data if isinstance(scene, dict)] # type: ignore

    async def plan_chapter_scenes(
        self,
        novel_props: Dict[str, Any], 
        chapter_number: int, 
        plot_point_focus: Optional[str], 
        plot_point_index: int 
    ) -> Tuple[Optional[List[SceneDetail]], Optional[Dict[str, int]]]:
        """
        Generates a detailed scene plan for the chapter.
        Returns the plan and LLM usage data.
        """
        if not config.ENABLE_AGENTIC_PLANNING:
            logger.info(f"Agentic planning disabled. Skipping detailed planning for Chapter {chapter_number}.")
            return None, None

        logger.info(f"PlannerAgent planning Chapter {chapter_number} with detailed scenes...")
        if plot_point_focus is None:
            logger.error(f"Cannot plan chapter {chapter_number}: No plot point focus available.")
            return None, None

        context_summary_parts: List[str] = []
        if chapter_number > 1:
            prev_chap_data = await chapter_queries.get_chapter_data_from_db(chapter_number - 1)
            if prev_chap_data:
                prev_summary = prev_chap_data.get('summary')
                prev_is_provisional = prev_chap_data.get('is_provisional', False)
                summary_prefix = "[Provisional Summary from Prev Ch] " if prev_is_provisional and prev_summary else "[Summary from Prev Ch] "
                if prev_summary:
                    context_summary_parts.append(f"{summary_prefix}({chapter_number - 1}):\n{prev_summary[:1000].strip()}...\n")
                else:
                    prev_text = prev_chap_data.get('text', '')
                    text_prefix = "[Provisional Text Snippet from Prev Ch] " if prev_is_provisional and prev_text else "[Text Snippet from Prev Ch] "
                    if prev_text:
                        context_summary_parts.append(f"{text_prefix}({chapter_number - 1}):\n...{prev_text[-1000:].strip()}\n")
        
        context_summary_str = "".join(context_summary_parts)

        protagonist_name = novel_props.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME)
        kg_context_section = await get_reliable_kg_facts_for_drafting_prompt(novel_props, chapter_number, None) 
        character_state_snippet_plain_text = await get_character_state_snippet_for_prompt(novel_props, chapter_number)
        world_state_snippet_plain_text = await get_world_state_snippet_for_prompt(novel_props, chapter_number)

        future_plot_context_parts: List[str] = []
        all_plot_points = novel_props.get('plot_outline_full', {}).get('plot_points', [])
        total_plot_points_in_novel = len(all_plot_points)

        if plot_point_index + 1 < total_plot_points_in_novel:
            next_pp_text = all_plot_points[plot_point_index + 1]
            if isinstance(next_pp_text, str) and next_pp_text.strip():
                future_plot_context_parts.append(f"\n**Anticipated Next Major Plot Point (PP {plot_point_index + 2}/{total_plot_points_in_novel} - for context, not this chapter's focus):**\n{next_pp_text.strip()}\n")
            if plot_point_index + 2 < total_plot_points_in_novel:
                 next_next_pp_text = all_plot_points[plot_point_index + 2]
                 if isinstance(next_next_pp_text, str) and next_next_pp_text.strip():
                    future_plot_context_parts.append(f"**And Then (PP {plot_point_index + 3}/{total_plot_points_in_novel} - distant context):**\n{next_next_pp_text.strip()}\n")
        future_plot_context_str = "".join(future_plot_context_parts)

        few_shot_scene_plan_example_str = f"""
SCENE: 1
SUMMARY: Elara arrives at the Sunken Library, finding its entrance hidden and guarded by an ancient riddle.
CHARACTERS INVOLVED: Elara Vance
KEY DIALOGUE POINTS:
- Elara (internal): "This riddle... it speaks of starlight and shadow. What reflects both?"
- Elara (to herself, solving): "The water! The entrance must be beneath the lake's surface."
SETTING DETAILS: A mist-shrouded, unnaturally still lake. Crumbling, moss-covered ruins of a tower are visible on a small island in the center.
SCENE FOCUS ELEMENTS:
- Elara's deductive reasoning to solve the riddle.
- Building atmosphere of mystery and ancient magic around the library.
CONTRIBUTION: Introduces the challenge of accessing the Sunken Library and showcases Elara's intellect.
---
SCENE: 2
SUMMARY: Elara meets Master Kael, the library's ancient archivist, who tests her worthiness before revealing information about the Starfall Map.
CHARACTERS INVOLVED: Elara Vance, Master Kael
KEY DIALOGUE POINTS:
- Kael: "Many seek what is lost. Few understand its price. Why do you search, child of the shifting stars?"
- Elara: "I seek knowledge not for power, but to mend what was broken."
- Kael: "A noble sentiment. The map's first secret lies in the reflection of true north..."
SETTING DETAILS: Inside the Sunken Library's main chamber: vast, circular, dimly lit by glowing runes on the walls and bioluminescent moss. Water drips softly.
SCENE FOCUS ELEMENTS:
- The cryptic nature and wisdom of Master Kael.
- The initial reveal of a clue related to the Starfall Map.
CONTRIBUTION: Elara gains a crucial piece of information and a potential ally (or gatekeeper) in Kael, advancing the plot point about finding the map.
"""

        prompt_lines = []
        if config.ENABLE_LLM_NO_THINK_DIRECTIVE:
            prompt_lines.append("/no_think")
        
        prompt_lines.extend([
            f"You are a master plotter outlining **between {config.TARGET_SCENES_MIN} and {config.TARGET_SCENES_MAX} detailed scenes** for Chapter {chapter_number} of a novel.",
            "This chapter is part of a larger narrative arc.",
            "",
            "**Novel Concept:**",
            f"  - Title: {novel_props.get('title', 'Untitled')}",
            f"  - Genre: {novel_props.get('genre', 'N/A')}",
            f"  - Theme: {novel_props.get('theme', 'N/A')}",
            f"  - Protagonist: {protagonist_name}",
            f"  - Protagonist's Arc: {novel_props.get('character_arc', 'N/A')}",
            "",
            f"**Overall Narrative Context:** This chapter focuses on Plot Point {plot_point_index + 1} of {total_plot_points_in_novel} total major plot points in the novel.",
            "",
            f"**Mandatory Focus for THIS Chapter (Plot Point {plot_point_index + 1}):**",
            plot_point_focus,
            "",
            future_plot_context_str,
            "**Recent Context from Previous Chapter(s) (Semantic context & KG Facts):**",
            context_summary_str if context_summary_str else "This is the first chapter, or no prior summary is available.",
            kg_context_section,
            "**Current Character States (Key Characters, based on profiles and recent developments - Plain Text):**",
            character_state_snippet_plain_text,
            "",
            "**Current World State (Relevant Locations/Elements, based on world-building - Plain Text):**",
            world_state_snippet_plain_text,
            "",
            "**Task:**",
            f"Create a detailed plan of {config.TARGET_SCENES_MIN} to {config.TARGET_SCENES_MAX} scenes for Chapter {chapter_number}.",
            "These scenes should *primarily advance the Mandatory Focus Plot Point* for this chapter. Do NOT attempt to resolve future plot points in these scenes.",
            "For each scene in the plan, provide the following information using clear labels (case-insensitive keys are fine, but use these display names):",
            "- `SCENE:` (Sequential number for the scene within this chapter, or `Scene Number:`)",
            "- `SUMMARY:`",
            "- `CHARACTERS INVOLVED:` (Comma-separated list)",
            "- `KEY DIALOGUE POINTS:` (List, each on a new line starting with \"- \")",
            "- `SETTING DETAILS:`",
            "- `SCENE FOCUS ELEMENTS:` (List, each on a new line starting with \"- \")",
            "- `CONTRIBUTION:`",
            "",
            "Separate each complete scene block with a line containing only \"---\". If you don't use \"---\", ensure each scene starts clearly with \"SCENE: <number>\" or \"Scene Number: <number>\".",
            "",
            "**Follow this example structure for your output precisely:**",
            "```plaintext",
            few_shot_scene_plan_example_str.strip(),
            "```",
            "",
            "Output ONLY the scene plan text as described."
        ])
        prompt = "\n".join(prompt_lines)

        logger.info(f"Calling LLM ({self.model_name}) for detailed scene plan for chapter {chapter_number} (target scenes: {config.TARGET_SCENES_MIN}-{config.TARGET_SCENES_MAX}). Plot Point {plot_point_index+1}/{total_plot_points_in_novel}.")
        
        cleaned_plan_text_from_llm, usage_data = await llm_service.async_call_llm( 
            model_name=self.model_name,
            prompt=prompt,
            temperature=config.TEMPERATURE_PLANNING, 
            max_tokens=config.MAX_PLANNING_TOKENS,
            allow_fallback=True,
            stream_to_disk=True,
            frequency_penalty=config.FREQUENCY_PENALTY_PLANNING,
            presence_penalty=config.PRESENCE_PENALTY_PLANNING,
            auto_clean_response=True 
        )

        parsed_scenes_list_of_dicts = self._parse_llm_scene_plan_output(cleaned_plan_text_from_llm, chapter_number)

        if parsed_scenes_list_of_dicts:
            final_scenes_typed: List[SceneDetail] = []
            for i, scene_dict in enumerate(parsed_scenes_list_of_dicts):
                if not isinstance(scene_dict, dict):
                    logger.warning(f"Parsed scene item {i+1} for ch {chapter_number} is not a dict. Skipping. Item: {scene_dict}")
                    continue
                required_scene_keys_internal = set(SCENE_PLAN_KEY_MAP.values())
                if not required_scene_keys_internal.issubset(scene_dict.keys()):
                    missing_k = required_scene_keys_internal - set(scene_dict.keys())
                    logger.warning(f"Scene {scene_dict.get('scene_number', i+1)} from parser for ch {chapter_number} has missing keys ({missing_k}). Skipping. Scene: {scene_dict}")
                    continue
                
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
                    logger.warning(f"Scene {scene_dict.get('scene_number', i+1)} from parser for ch {chapter_number} has invalid types or empty required strings. Skipping. Scene: {scene_dict}")
                    continue
                
                final_scenes_typed.append(scene_dict) # type: ignore 
            
            if final_scenes_typed:
                logger.info(f"Generated valid detailed scene plan for chapter {chapter_number} with {len(final_scenes_typed)} scenes from plain text.")
                return final_scenes_typed, usage_data
            else:
                logger.error(f"Parsed list was empty or all scenes were invalid after parsing plain text for chapter {chapter_number}. Cleaned LLM output: '{cleaned_plan_text_from_llm[:500]}...'")
                return None, usage_data 
        else:
            logger.error(f"Failed to parse a valid list of scenes from plain text for chapter {chapter_number}. Cleaned LLM output: '{cleaned_plan_text_from_llm[:500]}...'")
            return None, usage_data