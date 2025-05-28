# chapter_drafting_logic.py
"""
Handles the generation of the initial chapter draft for the SAGA system.
Context data for prompts is now formatted as plain text.
"""
import logging
from typing import Tuple, Optional, List

import config
import llm_interface 
from type import SceneDetail
# No direct state_manager import needed here as orchestrator passes data
from prompt_data_getters import (
    get_filtered_character_profiles_for_prompt_plain_text, 
    get_filtered_world_data_for_prompt_plain_text,       
    # Hybrid context already provides KG facts in plain text format
)
from context_generation_logic import generate_hybrid_chapter_context_logic

logger = logging.getLogger(__name__)

def _format_scene_plan_for_prompt(chapter_plan: List[SceneDetail], model_name_for_tokens: str, max_tokens_budget: int) -> str:
    """ Formats the chapter plan (list of SceneDetail dicts) into plain text for the LLM prompt, respecting token limits. """
    if not chapter_plan:
        return "No detailed scene plan available."

    plan_text_lines = ["**Detailed Scene Plan (MUST BE FOLLOWED CLOSELY):**"]
    total_plan_text_so_far = "\n".join(plan_text_lines) + "\n" # Account for header

    for scene_idx, scene in enumerate(chapter_plan):
        scene_lines = [
            f"Scene Number: {scene.get('scene_number', 'N/A')}",
            f"  Summary: {scene.get('summary', 'N/A')}",
            f"  Characters Involved: {', '.join(scene.get('characters_involved', [])) if scene.get('characters_involved') else 'None'}",
            f"  Key Dialogue Points:"
        ]
        for point in scene.get('key_dialogue_points', []):
            scene_lines.append(f"    - {point}")
        scene_lines.append(f"  Setting Details: {scene.get('setting_details', 'N/A')}")
        scene_lines.append(f"  Scene Focus Elements:")
        for focus_el in scene.get('scene_focus_elements', []):
            scene_lines.append(f"    - {focus_el}")
        scene_lines.append(f"  Contribution: {scene.get('contribution', 'N/A')}")
        
        if scene_idx < len(chapter_plan) -1 : # Add separator if not the last scene
            scene_lines.append("-" * 20) 
        
        current_scene_text_segment = "\n".join(scene_lines) + "\n"
        
        # Check token count before adding this scene's text to the cumulative plan
        # This approach builds the plan incrementally and stops if the next segment exceeds the budget.
        prospective_total_plan_text = total_plan_text_so_far + current_scene_text_segment
        
        if llm_interface.count_tokens(prospective_total_plan_text, model_name_for_tokens) > max_tokens_budget:
            plan_text_lines.append("... (plan truncated in prompt due to token limit)")
            logger.warning(f"Chapter plan was token-truncated for the drafting prompt. Max tokens for plan: {max_tokens_budget}. Stopped before scene {scene.get('scene_number', 'N/A')}.")
            break 
        
        plan_text_lines.extend(scene_lines)
        total_plan_text_so_far = prospective_total_plan_text # Update cumulative text
        
    if len(plan_text_lines) <= 1 : # Only header means no scenes were added
        return "No detailed scene plan available or plan was too long to include any scenes."
        
    return "\n".join(plan_text_lines)


async def generate_chapter_draft_logic(agent, chapter_number: int, plot_point_focus: Optional[str], hybrid_context: str, chapter_plan: Optional[List[SceneDetail]]) -> Tuple[Optional[str], Optional[str]]:
    """Generates the initial draft text for a chapter using HYBRID CONTEXT.
    Context data and scene plan are now formatted as plain text.
    'agent' here refers to the orchestrator or an object holding plot_outline etc.
    """
    if not plot_point_focus:
        plot_point_focus = "Continue the narrative logically, focusing on character development and plot progression based on previous events."
        logger.warning(f"Plot point focus was None for Ch {chapter_number} draft generation. Using generic fallback.")

    plan_section_for_prompt = ""
    if config.ENABLE_AGENTIC_PLANNING:
        if chapter_plan and isinstance(chapter_plan, list):
            # Max tokens for plan segment: 1/3 of total context, or a fixed moderate amount like 8k if MAX_CONTEXT_TOKENS is huge.
            # Let's use a fraction of MAX_CONTEXT_TOKENS.
            max_plan_tokens_for_prompt = config.MAX_CONTEXT_TOKENS // 3 
            plan_section_for_prompt = _format_scene_plan_for_prompt(chapter_plan, config.DRAFTING_MODEL, max_plan_tokens_for_prompt)
            logger.info(f"Using detailed scene plan (plain text) for Ch {chapter_number} draft generation.")
        else:
            plan_section_for_prompt = f"**Chapter Plan Note:** No detailed scene plan available. Rely on the Overall Plot Point Focus.\n**Overall Plot Point Focus for THIS Chapter:** {plot_point_focus}\n"
    else:
        plan_section_for_prompt = f"**Chapter Plan Note:** Detailed agentic planning is disabled. Rely on the Overall Plot Point Focus.\n**Overall Plot Point Focus for THIS Chapter:** {plot_point_focus}\n"

    # Get plain text formatted context
    # For Chapter 1, chapter_number - 1 will be 0. The prompt data getters should handle this.
    # The 'agent' parameter passed to these getters is the orchestrator instance or novel_props dict.
    char_profiles_plain_text = await get_filtered_character_profiles_for_prompt_plain_text(agent, chapter_number - 1)
    world_building_plain_text = await get_filtered_world_data_for_prompt_plain_text(agent, chapter_number - 1)


    # Accessing plot_outline attributes from the 'agent' (orchestrator or props_dict)
    plot_outline_data = getattr(agent, 'plot_outline', agent.get('plot_outline_full', agent.get('plot_outline', {})))


    prompt = f"""/no_think
You are an expert novelist tasked with writing Chapter {chapter_number} of the novel titled "{plot_outline_data.get('title', 'Untitled Novel')}".
**Story Bible / Core Information:**
  - Genre: {plot_outline_data.get('genre', 'N/A')}
  - Central Theme: {plot_outline_data.get('theme', 'N/A')}
  - Protagonist: {plot_outline_data.get('protagonist_name', 'N/A')}
  - Protagonist's Character Arc: {plot_outline_data.get('character_arc', 'N/A')}

{plan_section_for_prompt}

**World Building Notes (Plain Text - pay attention to any 'prompt_notes' indicating provisional data):**
```text
{world_building_plain_text if world_building_plain_text.strip() else "No specific world building notes provided for this chapter's context."}
```
**Character Profiles (Plain Text - pay attention to any 'prompt_notes' indicating provisional status):**
```text
{char_profiles_plain_text if char_profiles_plain_text.strip() else "No specific character profiles provided for this chapter's context."}
```
**Hybrid Context (Semantic Context for Flow & KG Facts for Canon):**
--- BEGIN HYBRID CONTEXT ---
{hybrid_context if hybrid_context.strip() else "No previous context (e.g., this is Chapter 1 or context retrieval failed)."}
--- END HYBRID CONTEXT ---

**Writing Instructions:**
1. Write a compelling and engaging chapter, aiming for a substantial length of at least {config.MIN_ACCEPTABLE_DRAFT_LENGTH} characters of narrative text.
2. If a **Detailed Scene Plan** is provided, adhere to it closely. For each scene, pay particular attention to its specified 'Summary', 'Key Dialogue Points', 'Setting Details', and **especially its 'Scene Focus Elements'**. Use the 'Scene Focus Elements' to guide you in elaborating, adding depth, and expanding the narrative to make each scene substantial and contribute to the overall chapter length target.
3. If no detailed plan is available, focus on achieving the **Overall Plot Point Focus** for this chapter.
4. Maintain consistency with all provided information (Story Bible, World Building, Character Profiles, Previous Context).
   - **Crucially, the `KEY RELIABLE KG FACTS` section within the `HYBRID CONTEXT` provides established canon that MUST be respected.**
   - The `SEMANTIC CONTEXT` section within the `HYBRID CONTEXT` should guide narrative flow, tone, and recall of recent events.
5. Ensure a smooth narrative flow and vivid prose suitable for the genre '{plot_outline_data.get('genre', 'story')}'.
6. **Employ 'showing' over 'telling':** Use vivid descriptions, sensory details, character actions, internal monologues, and nuanced dialogue to convey information, atmosphere, and emotions.
7. **Fully develop dialogue exchanges:** Allow characters to express themselves naturally, incorporating pauses, subtext, emotional reactions, and non-verbal cues. Don't shy away from longer conversations if they serve character or plot.
8. **Thoroughly explore the protagonist's (and other key characters') thoughts, feelings, and internal reactions** to the unfolding events and interactions. Dedicate space to their internal processing.
9. **Output ONLY the chapter text itself.** Do NOT include "Chapter X" headers, titles, author commentary, or any meta-discussion.

--- BEGIN CHAPTER {chapter_number} TEXT ---
"""
    logger.info(f"Calling LLM ({config.DRAFTING_MODEL}) for Ch {chapter_number} draft. Target minimum length: {config.MIN_ACCEPTABLE_DRAFT_LENGTH} chars.")
    raw_llm_text, usage_data = await llm_interface.async_call_llm( # usage_data added here
        model_name=config.DRAFTING_MODEL,
        prompt=prompt,
        temperature=0.6, # Standard temperature for creative generation
        max_tokens=None, # Allow LLM to generate up to its own max or config.MAX_GENERATION_TOKENS if httpx client enforces it
        allow_fallback=True,
        stream_to_disk=True 
    )
    if not raw_llm_text:
        logger.error(f"LLM returned no content for Ch {chapter_number} draft (primary and potential fallback failed).")
        return None, None # No usage data to return if call failed this badly

    cleaned_text = llm_interface.clean_model_response(raw_llm_text)

    if not cleaned_text or len(cleaned_text) < 50: # Arbitrary threshold for "virtually no content"
        logger.error(f"Ch {chapter_number} draft has virtually no content after cleaning ({len(cleaned_text or '')} chars). Raw LLM output snippet: '{raw_llm_text[:200]}...'")
        # Return raw LLM text for debugging purposes if cleaning resulted in empty.
        return None, raw_llm_text # Return usage_data even if text is poor

    if len(cleaned_text) < config.MIN_ACCEPTABLE_DRAFT_LENGTH:
         logger.warning(
             f"Ch {chapter_number} draft is short ({len(cleaned_text)} chars) after cleaning, but will be passed for evaluation/revision. "
             f"Min required: {config.MIN_ACCEPTABLE_DRAFT_LENGTH}. "
             f"Snippet: '{cleaned_text[:200].replace(chr(10), ' ')}...'" # Replace newlines for single-line log
         )

    logger.info(f"Generated initial draft for ch {chapter_number} (Length: {len(cleaned_text)} chars).")
    # Return usage_data along with the text
    return cleaned_text, raw_llm_text