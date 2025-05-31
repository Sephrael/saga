# planner_agent.py
import logging
import re
import asyncio
from typing import List, Optional, Any, Dict, Tuple

import config
import llm_interface
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
    "scene": "scene_number",
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
        (Logic from chapter_planning_logic.py)
        """
        scenes_data: List[Dict[str, Any]] = []
        if not text or not text.strip():
            logger.warning(f"Plain text scene plan for Ch {chapter_number} is empty. No scenes parsed.")
            return None

        scene_blocks_text = split_text_into_blocks(text, separator_regex_str=r'\n\s*---\s*\n')

        if not scene_blocks_text or (len(scene_blocks_text) == 1 and "SCENE:" not in scene_blocks_text[0].upper()):
            # Fallback split if "---" is not used but "SCENE: <number>" is present
            raw_split_by_header = re.split(r'(^\s*SCENE:\s*\d+\s*$)', text.strip(), flags=re.IGNORECASE | re.MULTILINE)
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
                elif current_block_lines: # If it's not a header, and we have an active block, append to it
                    current_block_lines.append(part_stripped)
            if current_block_lines: # Append the last block
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

            required_keys_internal = set(SCENE_PLAN_KEY_MAP.values())
            if "scene_number" not in parsed_scene_dict or not isinstance(parsed_scene_dict["scene_number"], int):
                logger.warning(
                    f"Scene block {block_num+1} in Ch {chapter_number} missing or invalid 'scene_number'. "
                    f"Assigning sequential: {len(scenes_data) + 1}. Parsed value: {parsed_scene_dict.get('scene_number')}"
                )
                parsed_scene_dict["scene_number"] = len(scenes_data) + 1
            
            missing_keys = required_keys_internal - set(parsed_scene_dict.keys())
            if missing_keys:
                logger.warning(f"Partial scene data parsed for Ch {chapter_number}, block {block_num+1}. Missing keys: {missing_keys}. Data: {parsed_scene_dict}")
                # Populate missing keys with defaults to maintain structure
                for req_key in required_keys_internal:
                    if req_key not in parsed_scene_dict:
                        if req_key in SCENE_PLAN_LIST_INTERNAL_KEYS:
                            parsed_scene_dict[req_key] = []
                        elif req_key == "scene_number": pass # Already handled
                        else: parsed_scene_dict[req_key] = "N/A - Missing from LLM output"
            
            # Ensure list types are correct even if parser somehow missed it
            for list_key in ["characters_involved", "key_dialogue_points", "scene_focus_elements"]:
                if not isinstance(parsed_scene_dict.get(list_key), list):
                    logger.warning(f"Scene {parsed_scene_dict.get('scene_number', 'N/A')} in Ch {chapter_number} has non-list for '{list_key}'. Correcting. Value: {parsed_scene_dict.get(list_key)}")
                    parsed_scene_dict[list_key] = [str(parsed_scene_dict.get(list_key))] if parsed_scene_dict.get(list_key) else []

            scenes_data.append(parsed_scene_dict)

        if not scenes_data:
            logger.error(f"Failed to parse any valid scenes from LLM output for Ch {chapter_number}. Raw text: '{text[:500]}...'")
            return None
        
        return [scene for scene in scenes_data if isinstance(scene, dict)] # type: ignore

    async def plan_chapter_scenes(
        self,
        novel_props: Dict[str, Any], # Contains plot_outline_full, character_profiles, world_building
        chapter_number: int, # This is the novel's actual chapter number
        plot_point_focus: Optional[str], # The specific plot point for this chapter
        plot_point_index: int # 0-based index of this plot_point_focus in the overall plot_points list
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
        kg_context_section = await get_reliable_kg_facts_for_drafting_prompt(novel_props, chapter_number, None) # Chapter plan not available yet for KG facts
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

        prompt_lines = [
            "/no_think",
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
            "- `SCENE:` (Sequential number for the scene within this chapter)",
            "- `SUMMARY:` (A concise 1-2 sentence summary of what happens in the scene)",
            "- `CHARACTERS INVOLVED:` (A comma-separated list of key character names)",
            "- `KEY DIALOGUE POINTS:` (List 1-3 brief points outlining crucial dialogue or internal monologue, each on a new line starting with \"- \")",
            "- `SETTING DETAILS:` (Brief description of the specific setting/location)",
            "- `SCENE FOCUS ELEMENTS:` (List 1-2 specific aspects for targeted elaboration, each on a new line starting with \"- \")",
            "- `CONTRIBUTION:` (A short explanation of how this scene contributes to THIS chapter's Mandatory Focus Plot Point)",
            "",
            "Separate each complete scene block with a line containing only \"---\". If you don't use \"---\", ensure each scene starts clearly with \"SCENE: <number>\".",
            "",
            "Output ONLY the scene plan text as described."
        ]
        prompt = "\n".join(prompt_lines)

        logger.info(f"Calling LLM ({self.model_name}) for detailed scene plan for chapter {chapter_number} (target scenes: {config.TARGET_SCENES_MIN}-{config.TARGET_SCENES_MAX}). Plot Point {plot_point_index+1}/{total_plot_points_in_novel}.")
        plan_raw_text, usage_data = await llm_interface.async_call_llm(
            model_name=self.model_name,
            prompt=prompt,
            temperature=0.6,
            max_tokens=config.MAX_PLANNING_TOKENS,
            allow_fallback=True,
            stream_to_disk=True
        )

        cleaned_plan_text = llm_interface.clean_model_response(plan_raw_text)
        parsed_scenes_list_of_dicts = self._parse_llm_scene_plan_output(cleaned_plan_text, chapter_number)

        if parsed_scenes_list_of_dicts:
            final_scenes_typed: List[SceneDetail] = []
            for i, scene_dict in enumerate(parsed_scenes_list_of_dicts):
                if not isinstance(scene_dict, dict):
                    logger.warning(f"Parsed scene item {i+1} for ch {chapter_number} is not a dict. Skipping. Item: {scene_dict}")
                    continue
                required_scene_keys_internal = set(SCENE_PLAN_KEY_MAP.values())
                if not required_scene_keys_internal.issubset(scene_dict.keys()):
                    missing_k = required_scene_keys_internal - set(scene_dict.keys())
                    logger.warning(f"Scene {i+1} from parser for ch {chapter_number} has missing keys ({missing_k}). Skipping. Scene: {scene_dict}")
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
                    logger.warning(f"Scene {i+1} from parser for ch {chapter_number} has invalid types or empty required strings. Skipping. Scene: {scene_dict}")
                    continue
                
                final_scenes_typed.append(scene_dict) # type: ignore 
            if final_scenes_typed:
                logger.info(f"Generated valid detailed scene plan for chapter {chapter_number} with {len(final_scenes_typed)} scenes from plain text.")
                return final_scenes_typed, usage_data
            else:
                logger.error(f"Parsed list was empty or all scenes were invalid after parsing plain text for chapter {chapter_number}. Raw LLM output: '{plan_raw_text[:500]}...'")
                return None, usage_data 
        else:
            logger.error(f"Failed to parse a valid list of scenes from plain text for chapter {chapter_number}. Raw LLM output: '{plan_raw_text[:500]}...'")
            return None, usage_data