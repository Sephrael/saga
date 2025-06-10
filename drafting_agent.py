# drafting_agent.py
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import config
from llm_interface import count_tokens, llm_service
from prompt_renderer import render_prompt
from kg_maintainer.models import CharacterProfile, SceneDetail, WorldItem
from utils import format_scene_plan_for_prompt

logger = logging.getLogger(__name__)


class DraftingAgent:
    def __init__(self, model_name: str = config.DRAFTING_MODEL):
        self.architect_model = config.MEDIUM_MODEL  # Use a reliable logical model
        self.artist_model = config.NARRATOR_MODEL  # This is the creative NARRATOR_MODEL
        logger.info(
            f"DraftingAgent initialized with Architect model: {self.architect_model} and Artist model: {self.artist_model}"
        )

    async def _plan_detailed_beats(
        self,
        plot_outline: Dict[str, Any],
        chapter_number: int,
        plot_point_focus: str,
        hybrid_context_for_draft: str,
        chapter_plan: Optional[List[SceneDetail]],
    ) -> Tuple[Optional[str], Optional[Dict[str, int]]]:
        """
        Step 1: The "Architect" plans the detailed beats of the chapter.
        This call uses a reliable model to generate a structured, detailed plan.
        """
        logger.info(
            f"DraftingAgent (Architect): Planning detailed beats for Chapter {chapter_number} with {self.architect_model}..."
        )

        max_tokens_for_plan_prompt = config.MAX_CONTEXT_TOKENS // 4
        scene_plan_prompt_text = (
            "No detailed scene plan available. Focus on the plot point."
        )
        if chapter_plan and config.ENABLE_AGENTIC_PLANNING:
            scene_plan_prompt_text = format_scene_plan_for_prompt(
                chapter_plan, self.architect_model, max_tokens_for_plan_prompt
            )

        prompt = render_prompt(
            "drafting_agent/plan_detailed_beats.j2", # A NEW PROMPT TEMPLATE IS NEEDED
            {
                "no_think": config.ENABLE_LLM_NO_THINK_DIRECTIVE,
                "chapter_number": chapter_number,
                "novel_title": plot_outline.get("title", "Untitled Novel"),
                "novel_genre": plot_outline.get("genre", "Unknown Genre"),
                "plot_point_focus": plot_point_focus,
                "scene_plan_prompt_text": scene_plan_prompt_text,
                "hybrid_context_for_draft": hybrid_context_for_draft,
            },
        )

        detailed_beats_json, usage_data = await llm_service.async_call_llm(
            model_name=self.architect_model,
            prompt=prompt,
            temperature=config.Temperatures.PLANNING, # Use a more controlled temperature
            max_tokens=config.MAX_PLANNING_TOKENS,
            allow_fallback=True,
            stream_to_disk=False,
            auto_clean_response=True,
        )

        # We expect this to be a JSON string, but we'll return the raw string
        # for the next step to use. We can add validation here if needed.
        if not detailed_beats_json or not detailed_beats_json.strip():
            logger.error(
                f"Architect model failed to produce detailed beats for Chapter {chapter_number}."
            )
            return None, usage_data

        return detailed_beats_json, usage_data

    async def _write_from_detailed_beats(
        self,
        plot_outline: Dict[str, Any],
        chapter_number: int,
        detailed_beats_plan: str,
    ) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, int]]]:
        """
        Step 2: The "Artist" writes the chapter prose from the detailed beats.
        This call uses the creative NARRATOR_MODEL.
        """
        logger.info(
            f"DraftingAgent (Artist): Writing Chapter {chapter_number} from detailed beats with {self.artist_model}..."
        )
        
        prompt = render_prompt(
            "drafting_agent/write_from_beats.j2", # A NEW PROMPT TEMPLATE IS NEEDED
            {
                "no_think": config.ENABLE_LLM_NO_THINK_DIRECTIVE,
                "chapter_number": chapter_number,
                "novel_title": plot_outline.get("title", "Untitled Novel"),
                "novel_genre": plot_outline.get("genre", "Unknown Genre"),
                "detailed_beats_plan": detailed_beats_plan,
                "min_length": config.MIN_ACCEPTABLE_DRAFT_LENGTH,
            },
        )

        prompt_tokens = count_tokens(prompt, self.artist_model)
        available_for_generation = config.MAX_CONTEXT_TOKENS - prompt_tokens - 200
        max_gen_tokens = config.MAX_GENERATION_TOKENS
        if available_for_generation < max_gen_tokens:
            if available_for_generation > 500:
                max_gen_tokens = available_for_generation
            else:
                logger.error(
                    f"Artist model has insufficient token space for generation in Ch {chapter_number}. Prompt tokens: {prompt_tokens}."
                )
                return None, "Prompt with detailed beats is too large.", None

        # This call can now be simplified: we expect raw prose, not JSON.
        draft_text, usage_data = await llm_service.async_call_llm(
            model_name=self.artist_model,
            prompt=prompt,
            temperature=config.Temperatures.DRAFTING,
            max_tokens=max_gen_tokens,
            allow_fallback=True,
            stream_to_disk=True,
            frequency_penalty=config.FREQUENCY_PENALTY_DRAFTING,
            presence_penalty=config.PRESENCE_PENALTY_DRAFTING,
            auto_clean_response=True, # We want the cleaned raw prose
        )

        if not draft_text or not draft_text.strip():
            logger.error(
                f"Artist model failed to write prose for Chapter {chapter_number} from detailed beats."
            )
            return None, draft_text, usage_data
            
        return draft_text, draft_text, usage_data

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
        Generates the initial draft for a chapter using the Architect/Artist pattern.
        Returns: (draft_text, raw_llm_output, cumulative_usage_data)
        """
        logger.info(
            f"DraftingAgent: Starting Architect/Artist process for Chapter {chapter_number}..."
        )
        cumulative_usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        def _add_usage(usage: Optional[Dict[str, int]]):
            if usage:
                cumulative_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                cumulative_usage["completion_tokens"] += usage.get(
                    "completion_tokens", 0
                )
                cumulative_usage["total_tokens"] += usage.get("total_tokens", 0)

        # Step 1: Architect plans the detailed beats
        detailed_beats, architect_usage = await self._plan_detailed_beats(
            plot_outline,
            chapter_number,
            plot_point_focus,
            hybrid_context_for_draft,
            chapter_plan,
        )
        _add_usage(architect_usage)

        if not detailed_beats:
            return (
                None,
                "Architect model failed to produce detailed beats.",
                cumulative_usage,
            )

        # Step 2: Artist writes the chapter from the detailed beats
        (
            final_draft,
            raw_artist_output,
            artist_usage,
        ) = await self._write_from_detailed_beats(
            plot_outline, chapter_number, detailed_beats
        )
        _add_usage(artist_usage)
        
        if not final_draft:
             return (
                None,
                raw_artist_output,
                cumulative_usage,
            )
            
        logger.info(
            f"DraftingAgent: Successfully generated draft for Chapter {chapter_number} via Architect/Artist pattern. Length: {len(final_draft)} characters."
        )

        # The raw LLM output to log can be the combination of both steps for full transparency
        full_raw_log = (
            f"--- ARCHITECT MODEL ({self.architect_model}) OUTPUT (DETAILED BEATS) ---\n"
            f"{detailed_beats}\n\n"
            f"--- ARTIST MODEL ({self.artist_model}) OUTPUT (PROSE) ---\n"
            f"{raw_artist_output}"
        )

        return final_draft, full_raw_log, cumulative_usage