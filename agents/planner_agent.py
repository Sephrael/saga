# agents/planner_agent.py
import json
import re
from typing import Any

import structlog
from config import settings
from core.llm_interface import llm_service
from prompt_renderer import render_prompt

from models import SceneDetail

logger = structlog.get_logger(__name__)

SCENE_PLAN_KEY_MAP = {
    "scene": "scene_number",
    "scene_number": "scene_number",
    "summary": "summary",
    "characters_involved": "characters_involved",
    "key_dialogue_points": "key_dialogue_points",
    "setting_details": "setting_details",
    "scene_focus_elements": "scene_focus_elements",
    "contribution": "contribution",
    # NEW: Add the directorial fields to the key map
    "scene_type": "scene_type",
    "pacing": "pacing",
    "character_arc_focus": "character_arc_focus",
    "relationship_development": "relationship_development",
}
SCENE_PLAN_LIST_INTERNAL_KEYS = [
    "key_dialogue_points",
    "scene_focus_elements",
    "characters_involved",
]


class PlannerAgent:
    """LLM-powered scene planner for chapter outlines."""

    def __init__(self, model_name: str = settings.PLANNING_MODEL):
        self.model_name = model_name
        logger.info(f"PlannerAgent initialized with model: {self.model_name}")

    def _parse_llm_scene_plan_output(
        self, json_text: str, chapter_number: int
    ) -> list[SceneDetail] | None:
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

        if not parsed_data:
            logger.warning(
                f"Parsed scene plan for Ch {chapter_number} is an empty list."
            )
            return None

        scenes_data: list[SceneDetail] = []
        for i, scene_item in enumerate(parsed_data):
            if not isinstance(scene_item, dict):
                logger.warning(
                    f"Scene item {i + 1} in Ch {chapter_number} is not a dictionary. Skipping. Item: {str(scene_item)[:100]}"
                )
                continue

            processed_scene_dict: dict[str, Any] = {}
            for llm_key, value in scene_item.items():
                internal_key = SCENE_PLAN_KEY_MAP.get(
                    llm_key.lower().replace(" ", "_"), llm_key
                )
                processed_scene_dict[internal_key] = value

            scene_num = processed_scene_dict.get("scene_number")
            if not isinstance(scene_num, int):
                logger.warning(
                    f"Scene {i + 1} in Ch {chapter_number} has invalid or missing 'scene_number'. Assigning {i + 1}. Value: {scene_num}"
                )
                processed_scene_dict["scene_number"] = i + 1

            summary = processed_scene_dict.get("summary")
            if not isinstance(summary, str) or not summary.strip():
                logger.warning(
                    f"Scene {scene_num} in Ch {chapter_number} has invalid or missing 'summary'. Skipping."
                )
                continue

            for list_key in SCENE_PLAN_LIST_INTERNAL_KEYS:
                val = processed_scene_dict.get(list_key)
                if isinstance(val, str):
                    processed_scene_dict[list_key] = [
                        v.strip() for v in val.split(",") if v.strip()
                    ]
                elif not isinstance(val, list):
                    processed_scene_dict[list_key] = (
                        [str(val)] if val is not None else []
                    )

            for key_internal_name in SCENE_PLAN_KEY_MAP.values():
                if key_internal_name not in processed_scene_dict:
                    if key_internal_name in SCENE_PLAN_LIST_INTERNAL_KEYS:
                        processed_scene_dict[key_internal_name] = []
                    else:
                        # For optional fields, this will be None.
                        processed_scene_dict[key_internal_name] = None

            scenes_data.append(processed_scene_dict)  # type: ignore

        if not scenes_data:
            logger.warning(f"No valid scenes parsed from JSON for Ch {chapter_number}.")
            return None

        scenes_data.sort(key=lambda x: x.get("scene_number", float("inf")))
        return scenes_data

    async def plan_chapter_scenes(
        self,
        plot_outline: dict[str, Any],
        chapter_number: int,
        plot_point_focus: str | None,
        plot_point_index: int,
        plot_point_progress_chapter: int,
        chapter_context: str,
        forbidden_locations: list[str] | None = None,
    ) -> tuple[list[SceneDetail] | None, dict[str, int] | None]:
        """Generate a detailed scene plan for a chapter.

        Args:
            plot_outline: Full plot outline for the novel.
            chapter_number: The sequential chapter number.
            plot_point_focus: The active major plot point.
            plot_point_index: Index of the active plot point in the outline.
            plot_point_progress_chapter: Chapter count within the current plot point span (1-based).
            chapter_context: Context string for planning.
            forbidden_locations: Locations or plot points that must be avoided.

        Returns:
            The planned scenes and token usage data.
        """
        if not settings.ENABLE_AGENTIC_PLANNING:
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

        context_summary_str = chapter_context

        protagonist_name = plot_outline.get(
            "protagonist_name", settings.DEFAULT_PROTAGONIST_NAME
        )

        future_plot_context_parts: list[str] = []
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

        few_shot_scene_plan_example_str = """
**Ignore the narrative details in this example. It shows the required format only.**
[
  {
    "scene_number": 1,
    "summary": "Elara arrives at the Sunken Library, finding its entrance hidden and guarded by an ancient riddle.",
    "characters_involved": ["Elara Vance"],
    "key_dialogue_points": [
      "Elara (internal): \\"This riddle... it speaks of starlight and shadow. What reflects both?\\"",
      "Elara (to herself, solving): \\"The water! The entrance must be beneath the lake's surface.\\""
    ],
    "setting_details": "A mist-shrouded, unnaturally still lake. Crumbling, moss-covered ruins of a tower are visible on a small island.",
    "scene_focus_elements": ["Elara's deductive reasoning", "Building atmosphere of mystery and ancient magic"],
    "contribution": "Introduces the challenge of accessing the Sunken Library and showcases Elara's intellect.",
    "scene_type": "ATMOSPHERE_BUILDING",
    "pacing": "SLOW",
    "character_arc_focus": "Establishes Elara's scholarly and determined nature when faced with a puzzle.",
    "relationship_development": null
  },
  {
    "scene_number": 2,
    "summary": "Elara meets Master Kael, the library's ancient archivist, who tests her worthiness before revealing information about the Starfall Map.",
    "characters_involved": ["Elara Vance", "Master Kael"],
    "key_dialogue_points": [
      "Kael: \\"Many seek what is lost. Few understand its price. Why do you search, child of the shifting stars?\\"",
      "Elara: \\"I seek knowledge not for power, but to mend what was broken.\\""
    ],
    "setting_details": "Inside the Sunken Library: vast, circular, dimly lit by glowing runes and bioluminescent moss.",
    "scene_focus_elements": ["The cryptic nature and wisdom of Master Kael", "The initial reveal of a clue for the Starfall Map"],
    "contribution": "Elara gains a crucial piece of information and a potential ally/gatekeeper in Kael, advancing the plot.",
    "scene_type": "DIALOGUE",
    "pacing": "MEDIUM",
    "character_arc_focus": "Elara must articulate her noble motivations, reinforcing her core identity.",
    "relationship_development": "The relationship between Elara and Kael is established as one of a student and a gatekeeper/mentor."
  },
  {
    "scene_number": 3,
    "summary": "As Elara leaves, she is ambushed by rival Seekers who try to steal the clue from her. She uses her wits and a minor magical artifact to escape.",
    "characters_involved": ["Elara Vance", "Rival Seeker (Thane)"],
    "key_dialogue_points": [
      "Thane: \\"The old man was a fool to trust you. The map belongs to the Crimson Hand!\\"",
      "Elara (activating artifact): \\"It belongs to those who would protect it!\\""
    ],
    "setting_details": "The narrow, crumbling causeway leading away from the library island.",
    "scene_focus_elements": ["Sudden danger and threat", "Elara's quick thinking under pressure", "First use of her 'Silvershard' artifact"],
    "contribution": "Introduces the antagonist faction and demonstrates that Elara is capable of defending herself.",
    "scene_type": "ACTION",
    "pacing": "FAST",
    "character_arc_focus": "Elara is forced from a purely intellectual challenge to a physical one, showing her resilience.",
    "relationship_development": "An antagonistic relationship with Thane and the Crimson Hand is established."
  }
]
"""
        prompt = render_prompt(
            "planner_agent/scene_plan.j2",
            {
                "enable_no_think": True,
                "target_scenes_min": settings.TARGET_SCENES_MIN,
                "target_scenes_max": settings.TARGET_SCENES_MAX,
                "chapter_number": chapter_number,
                "novel_title": plot_outline.get("title", "Untitled"),
                "novel_genre": plot_outline.get("genre", "N/A"),
                "novel_theme": plot_outline.get("theme", "N/A"),
                "protagonist_name": protagonist_name,
                "protagonist_arc": plot_outline.get("character_arc", "N/A"),
                "plot_point_index_plus1": plot_point_index + 1,
                "total_plot_points_in_novel": total_plot_points_in_novel,
                "plot_point_chapter_progress": plot_point_progress_chapter,
                "plot_point_chapter_span": settings.PLOT_POINT_CHAPTER_SPAN,
                "plot_point_focus": plot_point_focus,
                "future_plot_context_str": future_plot_context_str,
                "chapter_context": context_summary_str,
                "few_shot_scene_plan_example_str": few_shot_scene_plan_example_str,
                "forbidden_locations": forbidden_locations or [],
            },
        )
        logger.info(
            f"Calling LLM ({self.model_name}) for detailed scene plan for chapter {chapter_number} "
            f"(target scenes: {settings.TARGET_SCENES_MIN}-{settings.TARGET_SCENES_MAX}, expecting JSON). "
            f"Plot Point {plot_point_index + 1}/{total_plot_points_in_novel} "
            f"progress {plot_point_progress_chapter}/{settings.PLOT_POINT_CHAPTER_SPAN}."
        )

        (
            cleaned_plan_text_from_llm,
            usage_data,
        ) = await llm_service.async_call_llm(
            model_name=self.model_name,
            prompt=prompt,
            temperature=settings.TEMPERATURE_PLANNING,
            max_tokens=settings.MAX_PLANNING_TOKENS,
            allow_fallback=True,
            stream_to_disk=True,
            auto_clean_response=True,
        )

        parsed_scenes_list_of_dicts = self._parse_llm_scene_plan_output(
            cleaned_plan_text_from_llm, chapter_number
        )

        if parsed_scenes_list_of_dicts:
            final_scenes_typed: list[SceneDetail] = []
            for i, scene_dict in enumerate(parsed_scenes_list_of_dicts):
                if not isinstance(scene_dict, dict):
                    logger.warning(
                        f"Parsed scene item {i + 1} for ch {chapter_number} is not a dict. Skipping. Item: {scene_dict}"
                    )
                    continue

                # Basic validation for required fields
                if not scene_dict.get("summary"):
                    logger.warning(
                        f"Scene {scene_dict.get('scene_number', i + 1)} from parser for ch {chapter_number} has a missing summary. Skipping."
                    )
                    continue

                final_scenes_typed.append(scene_dict)  # type: ignore

            if final_scenes_typed:
                logger.info(
                    f"Generated valid detailed scene plan for chapter {chapter_number} with {len(final_scenes_typed)} scenes."
                )
                return final_scenes_typed, usage_data
            else:
                logger.error(
                    f"Parsed list was empty or all scenes were invalid after parsing for chapter {chapter_number}. Cleaned LLM output: '{cleaned_plan_text_from_llm[:500]}...'"
                )
                return None, usage_data
        else:
            logger.error(
                f"Failed to parse a valid list of scenes for chapter {chapter_number}. Cleaned LLM output: '{cleaned_plan_text_from_llm[:500]}...'"
            )
            return None, usage_data

    async def plan_continuation(
        self, summary_text: str, num_points: int = 5
    ) -> tuple[list[str] | None, dict[str, int] | None]:
        """Generate future plot points from a story summary."""
        prompt = render_prompt(
            "planner_agent/plan_continuation.j2",
            {"summary": summary_text, "num_points": num_points},
        )
        cleaned, usage = await llm_service.async_call_llm(
            model_name=self.model_name,
            prompt=prompt,
            temperature=settings.TEMPERATURE_PLANNING,
            max_tokens=settings.MAX_PLANNING_TOKENS,
            allow_fallback=True,
            stream_to_disk=False,
            auto_clean_response=True,
        )
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                return [str(p) for p in parsed if str(p).strip()], usage
        except json.JSONDecodeError:
            logger.error("Failed to parse continuation plan JSON: %s", cleaned)
        return None, usage
