# drafting_agent.py
import logging
from typing import Any, Dict, List, Optional, Tuple

import config
import utils
from llm_interface import count_tokens, llm_service, truncate_text_by_tokens
from prompt_renderer import render_prompt
from kg_maintainer.models import CharacterProfile, SceneDetail, WorldItem

logger = logging.getLogger(__name__)


class DraftingAgent:
    def __init__(self, model_name: str = config.DRAFTING_MODEL):
        # We now only need a single, capable model for drafting.
        self.drafting_model = model_name
        logger.info(
            f"DraftingAgent initialized with single-pass model: {self.drafting_model}"
        )

    async def draft_chapter(
        self,
        plot_outline: Dict[str, Any],
        character_profiles: Dict[str, CharacterProfile],
        world_building: Dict[str, Dict[str, WorldItem]],
        chapter_number: int,
        plot_point_focus: str,
        hybrid_context_for_draft: str,
        chapter_plan: Optional[List[SceneDetail]],
    ) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, int]]]:
        """
        Generates the initial draft for a chapter by drafting each scene sequentially and combining them.
        Returns: (draft_text, raw_llm_output, usage_data)
        """
        logger.info(
            f"DraftingAgent: Starting scene-by-scene draft for Chapter {chapter_number}..."
        )

        if not chapter_plan:
            logger.error(
                f"Cannot draft Chapter {chapter_number} scene-by-scene: No chapter plan provided."
            )
            return None, "Chapter plan was missing, cannot generate scenes.", None

        all_scenes_prose: List[str] = []
        all_raw_outputs: List[str] = []
        total_usage_data: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        # Context that is constant for all scenes in the chapter
        novel_title = plot_outline.get("title", "Untitled Novel")
        novel_genre = plot_outline.get("genre", "Unknown Genre")

        for scene_index, scene_detail in enumerate(chapter_plan):
            scene_number = scene_detail.get("scene_number", scene_index + 1)
            logger.info(f"Drafting Scene {scene_number} of Chapter {chapter_number}...")

            # Context from previously generated scenes in this chapter, truncated to a budget
            previous_scenes_prose = "\n\n".join(all_scenes_prose)
            max_tokens_for_prev_scenes = config.MAX_GENERATION_TOKENS // 2
            previous_scenes_prose_for_prompt = truncate_text_by_tokens(
                previous_scenes_prose,
                self.drafting_model,
                max_tokens_for_prev_scenes,
                truncation_marker="\n\n... (prose from earlier scenes in this chapter has been truncated)\n\n",
            )

            # Prepare the prompt for this specific scene using the new template
            prompt = render_prompt(
                "drafting_agent/draft_scene.j2",
                {
                    "no_think": config.ENABLE_LLM_NO_THINK_DIRECTIVE,
                    "chapter_number": chapter_number,
                    "novel_title": novel_title,
                    "novel_genre": novel_genre,
                    "scene_detail": scene_detail,
                    "hybrid_context_for_draft": hybrid_context_for_draft,
                    "previous_scenes_prose": previous_scenes_prose_for_prompt,
                    "min_length_per_scene": config.MIN_ACCEPTABLE_DRAFT_LENGTH
                    // len(chapter_plan),
                },
            )

            # Calculate max tokens for this scene's generation
            prompt_tokens = count_tokens(prompt, self.drafting_model)
            available_for_generation = (
                config.MAX_CONTEXT_TOKENS - prompt_tokens - 200
            )  # Safety buffer
            max_gen_tokens = min(
                config.MAX_GENERATION_TOKENS // 2, available_for_generation
            )

            if max_gen_tokens < 300:  # Not enough space for a meaningful scene
                error_msg = f"Insufficient token space for generating Scene {scene_number} in Ch {chapter_number}. Prompt tokens: {prompt_tokens}."
                logger.error(error_msg)
                # Return what we have so far, as this is a hard failure for the chapter.
                return (
                    "\n\n".join(all_scenes_prose) or None,
                    "\n\n---\n\n".join(all_raw_outputs),
                    total_usage_data,
                )

            # Call LLM for scene generation
            scene_prose, scene_usage_data = await llm_service.async_call_llm(
                model_name=self.drafting_model,
                prompt=prompt,
                temperature=config.Temperatures.DRAFTING,
                max_tokens=max_gen_tokens,
                allow_fallback=True,
                stream_to_disk=False,  # Avoid excessive I/O for smaller scene generations
                frequency_penalty=config.FREQUENCY_PENALTY_DRAFTING,
                presence_penalty=config.PRESENCE_PENALTY_DRAFTING,
                auto_clean_response=True,
            )

            if not scene_prose or not scene_prose.strip():
                logger.warning(
                    f"Drafting model failed to write prose for Scene {scene_number} of Chapter {chapter_number}. Skipping scene."
                )
                all_raw_outputs.append(f"--- SCENE {scene_number} FAILED TO GENERATE ---")
                continue

            all_scenes_prose.append(scene_prose)
            # Since auto_clean_response=True, the raw output is effectively the cleaned prose
            all_raw_outputs.append(scene_prose)

            if scene_usage_data:
                for key, value in scene_usage_data.items():
                    total_usage_data[key] = total_usage_data.get(key, 0) + value

        # Combine all parts
        final_draft_text = "\n\n".join(all_scenes_prose)
        final_raw_output = "\n\n---\n\n".join(all_raw_outputs)

        if not final_draft_text.strip():
            logger.error(
                f"Drafting failed for Chapter {chapter_number}: no scenes were successfully generated."
            )
            return None, final_raw_output, total_usage_data

        logger.info(
            f"DraftingAgent: Successfully generated draft for Chapter {chapter_number} from {len(all_scenes_prose)} scenes. Total Length: {len(final_draft_text)} characters."
        )

        return final_draft_text, final_raw_output, total_usage_data