# drafting_agent.py
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
        Generates the initial draft for a chapter using a single-pass, plan-then-write approach.
        Returns: (draft_text, raw_llm_output, usage_data)
        """
        logger.info(
            f"DraftingAgent: Starting single-pass draft for Chapter {chapter_number}..."
        )

        # Step 1: Prepare all context and render the new, unified prompt
        max_tokens_for_plan_prompt = config.MAX_CONTEXT_TOKENS // 4
        scene_plan_prompt_text = (
            "No detailed scene plan available. Focus on the plot point."
        )
        if chapter_plan and config.ENABLE_AGENTIC_PLANNING:
            scene_plan_prompt_text = format_scene_plan_for_prompt(
                chapter_plan, self.drafting_model, max_tokens_for_plan_prompt
            )

        # This now points to a new, single-pass prompt template
        prompt = render_prompt(
            "drafting_agent/draft_chapter_single_pass.j2",
            {
                "no_think": config.ENABLE_LLM_NO_THINK_DIRECTIVE,
                "chapter_number": chapter_number,
                "novel_title": plot_outline.get("title", "Untitled Novel"),
                "novel_genre": plot_outline.get("genre", "Unknown Genre"),
                "plot_point_focus": plot_point_focus,
                "scene_plan_prompt_text": scene_plan_prompt_text,
                "hybrid_context_for_draft": hybrid_context_for_draft,
                "min_length": config.MIN_ACCEPTABLE_DRAFT_LENGTH,
            },
        )

        # Step 2: Calculate token space for generation
        prompt_tokens = count_tokens(prompt, self.drafting_model)
        available_for_generation = config.MAX_CONTEXT_TOKENS - prompt_tokens - 200 # Safety buffer
        max_gen_tokens = config.MAX_GENERATION_TOKENS

        if available_for_generation < max_gen_tokens:
            if available_for_generation > 500: # Ensure we have a reasonable minimum
                max_gen_tokens = available_for_generation
            else:
                error_msg = f"Insufficient token space for generation in Ch {chapter_number}. Prompt tokens: {prompt_tokens}."
                logger.error(error_msg)
                return None, error_msg, None

        # Step 3: Make the single LLM call
        draft_text, usage_data = await llm_service.async_call_llm(
            model_name=self.drafting_model,
            prompt=prompt,
            temperature=config.Temperatures.DRAFTING,
            max_tokens=max_gen_tokens,
            allow_fallback=True,
            stream_to_disk=True,
            frequency_penalty=config.FREQUENCY_PENALTY_DRAFTING,
            presence_penalty=config.PRESENCE_PENALTY_DRAFTING,
            auto_clean_response=True, # We expect clean prose as the final output
        )

        if not draft_text or not draft_text.strip():
            error_msg = f"Drafting model failed to write prose for Chapter {chapter_number}."
            logger.error(error_msg)
            return None, draft_text or error_msg, usage_data
        
        logger.info(
            f"DraftingAgent: Successfully generated draft for Chapter {chapter_number} in a single pass. Length: {len(draft_text)} characters."
        )

        # The raw LLM output is now just the direct response from the single call.
        return draft_text, draft_text, usage_data