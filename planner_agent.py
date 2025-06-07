# planner_agent.py
# from parsing_utils import split_text_into_blocks, parse_key_value_block # Removed
import json  # Added for JSON parsing
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import config

# from state_manager import state_manager # No longer directly used
from data_access import chapter_queries  # For get_chapter_data_from_db
from llm_interface import llm_service
from prompt_renderer import render_prompt
from prompt_data_getters import (
    get_character_state_snippet_for_prompt,
    get_reliable_kg_facts_for_drafting_prompt,
    get_world_state_snippet_for_prompt,
)
from kg_maintainer.models import CharacterProfile, SceneDetail, WorldItem

logger = logging.getLogger(__name__)

SCENE_PLAN_KEY_MAP = {
    "scene": "scene_number",  # LLM often uses "SCENE:" or "Scene Number:"
    "scene_number": "scene_number",  # Explicitly map "scene_number" as well
    "summary": "summary",
    "characters_involved": "characters_involved",
    "key_dialogue_points": "key_dialogue_points",
    "setting_details": "setting_details",
    "scene_focus_elements": "scene_focus_elements",
    "contribution": "contribution",
}
SCENE_PLAN_LIST_INTERNAL_KEYS = [
    "key_dialogue_points",
    "scene_focus_elements",
    "characters_involved",
]
# SCENE_PLAN_SPECIAL_LIST_HANDLING removed as JSON should provide lists directly.


class PlannerAgent:
    def __init__(self, model_name: str = config.PLANNING_MODEL):
        self.model_name = model_name
        logger.info(f"PlannerAgent initialized with model: {self.model_name}")

    def _parse_llm_scene_plan_output(
        self, json_text: str, chapter_number: int
    ) -> Optional[List[SceneDetail]]:
        """
        Parses JSON scene plan output from LLM.
        Expects a JSON array of scene objects.
        """
        if not json_text or not json_text.strip():
            logger.warning(
                f"JSON scene plan for Ch {chapter_number} is empty. No scenes parsed."
            )
            return None

        try:
            parsed_data = json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to decode JSON scene plan for Ch {chapter_number}: {e}. Text: {json_text[:500]}..."
            )
            # Try to find a JSON array within the text if LLM wrapped it
            match = re.search(r"\[\s*\{.*\}\s*\]", json_text, re.DOTALL)
            if match:
                logger.info(
                    "Found a JSON array within the malformed JSON string. Attempting to parse that."
                )
                try:
                    parsed_data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    logger.error(
                        f"Still failed to parse extracted JSON array for Ch {chapter_number}."
                    )
                    return None
            else:
                return None

        if not isinstance(parsed_data, list):
            logger.warning(
                f"Parsed scene plan for Ch {chapter_number} is not a list as expected. Type: {type(parsed_data)}. Data: {str(parsed_data)[:300]}"
            )
            return None

        if not parsed_data:  # Empty list
            logger.warning(
                f"Parsed scene plan for Ch {chapter_number} is an empty list."
            )
            return None

        scenes_data: List[SceneDetail] = []
        for i, scene_item in enumerate(parsed_data):
            if not isinstance(scene_item, dict):
                logger.warning(
                    f"Scene item {i + 1} in Ch {chapter_number} is not a dictionary. Skipping. Item: {str(scene_item)[:100]}"
                )
                continue

            # Map keys from JSON to SceneDetail keys, defaulting to original if not in map
            # Prefer LLM to output keys matching SceneDetail directly.
            processed_scene_dict: Dict[str, Any] = {}
            for llm_key, value in scene_item.items():
                internal_key = SCENE_PLAN_KEY_MAP.get(
                    llm_key.lower().replace(" ", "_"), llm_key
                )
                processed_scene_dict[internal_key] = value

            # Validate essential keys and types
            scene_num = processed_scene_dict.get("scene_number")
            if not isinstance(scene_num, int):
                logger.warning(
                    f"Scene {i + 1} in Ch {chapter_number} has invalid or missing 'scene_number'. Assigning {i + 1}. Value: {scene_num}"
                )
                processed_scene_dict["scene_number"] = (
                    i + 1
                )  # Assign sequential if missing/invalid

            summary = processed_scene_dict.get("summary")
            if not isinstance(summary, str) or not summary.strip():
                logger.warning(
                    f"Scene {scene_num} in Ch {chapter_number} has invalid or missing 'summary'. Skipping."
                )
                continue

            # Ensure list types for specific fields
            for list_key in SCENE_PLAN_LIST_INTERNAL_KEYS:
                val = processed_scene_dict.get(list_key)
                if isinstance(val, str):  # If LLM gave comma-separated string
                    processed_scene_dict[list_key] = [
                        v.strip() for v in val.split(",") if v.strip()
                    ]
                elif not isinstance(val, list):
                    processed_scene_dict[list_key] = (
                        [str(val)] if val is not None else []
                    )

            # Fill missing optional keys with defaults
            for key_internal_name in SCENE_PLAN_KEY_MAP.values():
                if key_internal_name not in processed_scene_dict:
                    if key_internal_name in SCENE_PLAN_LIST_INTERNAL_KEYS:
                        processed_scene_dict[key_internal_name] = []
                    else:
                        processed_scene_dict[key_internal_name] = (
                            "N/A - Missing from LLM JSON"
                        )

            # Ensure all SceneDetail keys are present, even if some were not in SCENE_PLAN_KEY_MAP
            # This is important if SceneDetail has more fields than the map covers.
            # For now, assuming SCENE_PLAN_KEY_MAP covers all required fields of SceneDetail.

            scenes_data.append(processed_scene_dict)  # type: ignore

        if not scenes_data:
            logger.warning(f"No valid scenes parsed from JSON for Ch {chapter_number}.")
            return None

        scenes_data.sort(key=lambda x: x.get("scene_number", float("inf")))
        return scenes_data

    async def plan_chapter_scenes(
        self,
        plot_outline: Dict[str, Any],
        character_profiles: Dict[str, CharacterProfile],
        world_building: Dict[str, Dict[str, WorldItem]],
        chapter_number: int,
        plot_point_focus: Optional[str],
        plot_point_index: int,
    ) -> Tuple[Optional[List[SceneDetail]], Optional[Dict[str, int]]]:
        """
        Generates a detailed scene plan for the chapter.
        Returns the plan and LLM usage data.
        """
        if not config.ENABLE_AGENTIC_PLANNING:
            logger.info(
                f"Agentic planning disabled. Skipping detailed planning for Chapter {chapter_number}."
            )
            return None, None

        logger.info(
            f"PlannerAgent planning Chapter {chapter_number} with detailed scenes..."
        )
        if plot_point_focus is None:
            logger.error(
                f"Cannot plan chapter {chapter_number}: No plot point focus available."
            )
            return None, None

        context_summary_parts: List[str] = []
        if chapter_number > 1:
            prev_chap_data = await chapter_queries.get_chapter_data_from_db(
                chapter_number - 1
            )
            if prev_chap_data:
                prev_summary = prev_chap_data.get("summary")
                prev_is_provisional = prev_chap_data.get("is_provisional", False)
                summary_prefix = (
                    "[Provisional Summary from Prev Ch] "
                    if prev_is_provisional and prev_summary
                    else "[Summary from Prev Ch] "
                )
                if prev_summary:
                    context_summary_parts.append(
                        f"{summary_prefix}({chapter_number - 1}):\n{prev_summary[:1000].strip()}...\n"
                    )
                else:
                    prev_text = prev_chap_data.get("text", "")
                    text_prefix = (
                        "[Provisional Text Snippet from Prev Ch] "
                        if prev_is_provisional and prev_text
                        else "[Text Snippet from Prev Ch] "
                    )
                    if prev_text:
                        context_summary_parts.append(
                            f"{text_prefix}({chapter_number - 1}):\n...{prev_text[-1000:].strip()}\n"
                        )

        context_summary_str = "".join(context_summary_parts)

        protagonist_name = plot_outline.get(
            "protagonist_name", config.DEFAULT_PROTAGONIST_NAME
        )
        kg_context_section = await get_reliable_kg_facts_for_drafting_prompt(
            plot_outline, chapter_number, None
        )
        character_state_snippet_plain_text = (
            await get_character_state_snippet_for_prompt(
                character_profiles, plot_outline, chapter_number
            )
        )
        world_state_snippet_plain_text = await get_world_state_snippet_for_prompt(
            world_building, chapter_number
        )

        future_plot_context_parts: List[str] = []
        all_plot_points = plot_outline.get("plot_points", [])
        total_plot_points_in_novel = len(all_plot_points)

        if plot_point_index + 1 < total_plot_points_in_novel:
            next_pp_text = all_plot_points[plot_point_index + 1]
            if isinstance(next_pp_text, str) and next_pp_text.strip():
                future_plot_context_parts.append(
                    f"\n**Anticipated Next Major Plot Point (PP {plot_point_index + 2}/{total_plot_points_in_novel} - for context, not this chapter's focus):**\n{next_pp_text.strip()}\n"
                )
            if plot_point_index + 2 < total_plot_points_in_novel:
                next_next_pp_text = all_plot_points[plot_point_index + 2]
                if isinstance(next_next_pp_text, str) and next_next_pp_text.strip():
                    future_plot_context_parts.append(
                        f"**And Then (PP {plot_point_index + 3}/{total_plot_points_in_novel} - distant context):**\n{next_next_pp_text.strip()}\n"
                    )
        future_plot_context_str = "".join(future_plot_context_parts)

        # This whole block will replace the existing few_shot_scene_plan_example_str
        few_shot_scene_plan_example_str = """
[
  {
    "scene_number": 1,
    "summary": "Elara arrives at the Sunken Library, finding its entrance hidden and guarded by an ancient riddle.",
    "characters_involved": ["Elara Vance"],
    "key_dialogue_points": [
      "Elara (internal): \\"This riddle... it speaks of starlight and shadow. What reflects both?\\"",
      "Elara (to herself, solving): \\"The water! The entrance must be beneath the lake's surface.\\""
    ],
    "setting_details": "A mist-shrouded, unnaturally still lake. Crumbling, moss-covered ruins of a tower are visible on a small island in the center.",
    "scene_focus_elements": [
      "Elara's deductive reasoning to solve the riddle.",
      "Building atmosphere of mystery and ancient magic around the library."
    ],
    "contribution": "Introduces the challenge of accessing the Sunken Library and showcases Elara's intellect."
  },
  {
    "scene_number": 2,
    "summary": "Elara meets Master Kael, the library's ancient archivist, who tests her worthiness before revealing information about the Starfall Map.",
    "characters_involved": ["Elara Vance", "Master Kael"],
    "key_dialogue_points": [
      "Kael: \\"Many seek what is lost. Few understand its price. Why do you search, child of the shifting stars?\\"",
      "Elara: \\"I seek knowledge not for power, but to mend what was broken.\\"",
      "Kael: \\"A noble sentiment. The map's first secret lies in the reflection of true north...\\""
    ],
    "setting_details": "Inside the Sunken Library's main chamber: vast, circular, dimly lit by glowing runes on the walls and bioluminescent moss. Water drips softly.",
    "scene_focus_elements": [
      "The cryptic nature and wisdom of Master Kael.",
      "The initial reveal of a clue related to the Starfall Map."
    ],
    "contribution": "Elara gains a crucial piece of information and a potential ally (or gatekeeper) in Kael, advancing the plot point about finding the map."
  }
]
"""

        prompt = render_prompt(
            "planner_agent/scene_plan.j2",
            {
                "no_think": config.ENABLE_LLM_NO_THINK_DIRECTIVE,
                "target_scenes_min": config.TARGET_SCENES_MIN,
                "target_scenes_max": config.TARGET_SCENES_MAX,
                "chapter_number": chapter_number,
                "novel_title": plot_outline.get("title", "Untitled"),
                "novel_genre": plot_outline.get("genre", "N/A"),
                "novel_theme": plot_outline.get("theme", "N/A"),
                "protagonist_name": protagonist_name,
                "protagonist_arc": plot_outline.get("character_arc", "N/A"),
                "plot_point_index_plus1": plot_point_index + 1,
                "total_plot_points_in_novel": total_plot_points_in_novel,
                "plot_point_focus": plot_point_focus,
                "future_plot_context_str": future_plot_context_str,
                "context_summary_str": context_summary_str,
                "kg_context_section": kg_context_section,
                "character_state_snippet_plain_text": character_state_snippet_plain_text,
                "world_state_snippet_plain_text": world_state_snippet_plain_text,
                "few_shot_scene_plan_example_str": few_shot_scene_plan_example_str,
            },
        )
        logger.info(
            f"Calling LLM ({self.model_name}) for detailed scene plan for chapter {chapter_number} (target scenes: {config.TARGET_SCENES_MIN}-{config.TARGET_SCENES_MAX}, expecting JSON). Plot Point {plot_point_index + 1}/{total_plot_points_in_novel}."
        )

        (
            cleaned_plan_text_from_llm,
            usage_data,
        ) = await llm_service.async_call_llm(
            model_name=self.model_name,
            prompt=prompt,
            temperature=config.Temperatures.PLANNING,
            max_tokens=config.MAX_PLANNING_TOKENS,
            allow_fallback=True,
            stream_to_disk=True,
            frequency_penalty=config.FREQUENCY_PENALTY_PLANNING,
            presence_penalty=config.PRESENCE_PENALTY_PLANNING,
            auto_clean_response=True,
        )

        parsed_scenes_list_of_dicts = self._parse_llm_scene_plan_output(
            cleaned_plan_text_from_llm, chapter_number
        )

        if parsed_scenes_list_of_dicts:
            final_scenes_typed: List[SceneDetail] = []
            for i, scene_dict in enumerate(parsed_scenes_list_of_dicts):
                if not isinstance(scene_dict, dict):
                    logger.warning(
                        f"Parsed scene item {i + 1} for ch {chapter_number} is not a dict. Skipping. Item: {scene_dict}"
                    )
                    continue
                required_scene_keys_internal = set(SCENE_PLAN_KEY_MAP.values())
                if not required_scene_keys_internal.issubset(scene_dict.keys()):
                    missing_k = required_scene_keys_internal - set(scene_dict.keys())
                    logger.warning(
                        f"Scene {scene_dict.get('scene_number', i + 1)} from parser for ch {chapter_number} has missing keys ({missing_k}). Skipping. Scene: {scene_dict}"
                    )
                    continue

                valid_types = (
                    isinstance(scene_dict.get("scene_number"), int)
                    and isinstance(scene_dict.get("summary"), str)
                    and scene_dict.get("summary", "").strip()
                    and isinstance(scene_dict.get("characters_involved"), list)
                    and isinstance(scene_dict.get("key_dialogue_points"), list)
                    and isinstance(scene_dict.get("setting_details"), str)
                    and scene_dict.get("setting_details", "").strip()
                    and isinstance(scene_dict.get("scene_focus_elements"), list)
                    and isinstance(scene_dict.get("contribution"), str)
                    and scene_dict.get("contribution", "").strip()
                )
                if not valid_types:
                    logger.warning(
                        f"Scene {scene_dict.get('scene_number', i + 1)} from parser for ch {chapter_number} has invalid types or empty required strings. Skipping. Scene: {scene_dict}"
                    )
                    continue

                final_scenes_typed.append(scene_dict)  # type: ignore

            if final_scenes_typed:
                logger.info(
                    f"Generated valid detailed scene plan for chapter {chapter_number} with {len(final_scenes_typed)} scenes from plain text."
                )
                return final_scenes_typed, usage_data
            else:
                logger.error(
                    f"Parsed list was empty or all scenes were invalid after parsing plain text for chapter {chapter_number}. Cleaned LLM output: '{cleaned_plan_text_from_llm[:500]}...'"
                )
                return None, usage_data
        else:
            logger.error(
                f"Failed to parse a valid list of scenes from plain text for chapter {chapter_number}. Cleaned LLM output: '{cleaned_plan_text_from_llm[:500]}...'"
            )
            return None, usage_data
