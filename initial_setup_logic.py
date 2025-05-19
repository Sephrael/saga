# initial_setup_logic.py
"""
Handles generation of initial plot outlines and world-building data for the SAGA system.
These functions are called by the NovelWriterAgent during its setup phase.
"""
import logging
import json
import random
from typing import Dict, Any, Optional # Added Optional

import config
import llm_interface

logger = logging.getLogger(__name__)

# Define type aliases for clarity, representing the structure of data these functions handle.
# These are essentially Dict[str, Any] but provide semantic meaning.
PlotOutlineData = Dict[str, Any]
WorldBuildingData = Dict[str, Any]

async def generate_plot_outline_logic(agent, default_protagonist_name: str, unhinged_mode: bool, **kwargs) -> PlotOutlineData:
    """Generates a new plot outline using an LLM.
    'agent' is an instance of NovelWriterAgent.
    Returns the generated plot outline data.
    """
    logger.info(f"Generating plot outline. Unhinged mode: {unhinged_mode}")
    
    base_elements_for_outline: Dict[str, Any] = {}
    if unhinged_mode:
        genre = kwargs.get("genre", random.choice(config.UNHINGED_GENRES))
        theme = kwargs.get("theme", random.choice(config.UNHINGED_THEMES))
        setting_archetype = kwargs.get("setting_archetype", random.choice(config.UNHINGED_SETTINGS_ARCHETYPES))
        protagonist_archetype = kwargs.get("protagonist_archetype", random.choice(config.UNHINGED_PROTAGONIST_ARCHETYPES))
        conflict_archetype = kwargs.get("conflict_archetype", random.choice(config.UNHINGED_CONFLICT_TYPES))
        
        prompt_core_elements = f"""
The novel is a '{genre}' story. Its central theme is '{theme}'.
The primary setting is inspired by: '{setting_archetype}'.
The protagonist is an archetype of: '{protagonist_archetype}'.
The main conflict revolves around: '{conflict_archetype}'.
Based on this combination, generate the following JSON fields:
1. `title`: A compelling title for the novel.
2. `protagonist_name`: A suitable name for the protagonist.
3. `protagonist_description`: A brief (1-2 sentences) description of the protagonist.
4. `setting`: A brief (1-2 sentences) description of the primary setting, expanding on the archetype.
5. `conflict`: A brief (1-2 sentences) summary of the main conflict.
6. `plot_points`: A JSON list of exactly 5 strings, representing major plot points from beginning to end.
7. `character_arc`: A string describing the protagonist's primary development arc through the story.
"""
        base_elements_for_outline = {
            "genre": genre, "theme": theme, 
            "setting_archetype_used": setting_archetype,
            "protagonist_archetype_used": protagonist_archetype,
            "conflict_archetype_used": conflict_archetype
        }
        required_keys = ["title", "protagonist_name", "protagonist_description", "setting", "conflict", "plot_points", "character_arc"]
    else: 
        genre = kwargs.get("genre", config.CONFIGURED_GENRE)
        theme = kwargs.get("theme", config.CONFIGURED_THEME)
        setting_description = kwargs.get("setting_description", config.CONFIGURED_SETTING_DESCRIPTION)
        prompt_core_elements = f"""
The novel is a '{genre}' story. Its central theme is '{theme}'.
The primary setting is: '{setting_description}'.
Based on these, generate the following JSON fields:
1. `title`: A compelling title for the novel.
2. `protagonist_name`: A suitable name for the protagonist (consider using '{default_protagonist_name}' or a variant if appropriate).
3. `protagonist_description`: A brief (1-2 sentences) description of the protagonist.
4. `plot_points`: A JSON list of exactly 5 strings, representing major plot points from beginning to end.
5. `character_arc`: A string describing the protagonist's primary development arc through the story.
6. `conflict`: A string summarizing the main conflict that drives the plot.
"""
        base_elements_for_outline = {"genre": genre, "theme": theme, "setting": setting_description}
        required_keys = ["title", "protagonist_name", "protagonist_description", "plot_points", "character_arc", "conflict"]

    prompt = f"""/no_think
You are a creative assistant specializing in narrative structure generation.
{prompt_core_elements}
Output ONLY the JSON object. Ensure the response is a single, valid JSON.
The `plot_points` field must be a JSON list containing exactly 5 string elements.
Example of expected JSON structure (keys might vary slightly based on mode):
{{
  "title": "string",
  "protagonist_name": "string",
  "protagonist_description": "string",
  "setting": "string (only if unhinged mode, otherwise inferred)", 
  "conflict": "string",
  "plot_points": ["string1", "string2", "string3", "string4", "string5"],
  "character_arc": "string"
}}
"""
    logger.info("Calling LLM for plot outline generation...")
    # Plot outline JSON is expected to be somewhat large.
    raw_outline_str = await llm_interface.async_call_llm(
        model_name=config.INITIAL_SETUP_MODEL,
        prompt=prompt, 
        temperature=0.6,
        stream_to_disk=True # Output can be a fairly large JSON
    )
    
    # Type `parsed_llm_response` as Any initially, as `async_parse_llm_json_response` might return various types or None
    parsed_llm_response: Any = await llm_interface.async_parse_llm_json_response(raw_outline_str, "plot outline generation")

    is_valid = False 
    # final_outline_data will hold the successfully parsed and validated outline, or be an empty dict.
    final_outline_data: PlotOutlineData = {} 

    if parsed_llm_response and isinstance(parsed_llm_response, dict):
        # Now we know parsed_llm_response is a dict, assign to a more specifically named variable
        parsed_outline: Dict[str, Any] = parsed_llm_response
        plot_points = parsed_outline.get("plot_points")
        
        # Validation logic
        if (all(key in parsed_outline and isinstance(parsed_outline[key], str) and parsed_outline[key].strip()
                for key in required_keys if key != "plot_points") and
            isinstance(plot_points, list) and len(plot_points) == 5 and
            all(isinstance(p, str) and p.strip() for p in plot_points)):
            is_valid = True
            final_outline_data = parsed_outline # Assign the validated dict
        else:
            missing_or_invalid = [
                key for key in required_keys 
                if key not in parsed_outline or 
                   (key != "plot_points" and (not isinstance(parsed_outline.get(key), str) or not str(parsed_outline.get(key, "")).strip())) or
                   (key == "plot_points" and (not isinstance(parsed_outline.get("plot_points"), list) or
                                              len(parsed_outline.get("plot_points", [])) != 5 or
                                              not all(isinstance(p, str) and p.strip() for p in parsed_outline.get("plot_points", []))))
            ]
            logger.warning(f"Generated plot outline failed validation. Missing/invalid keys: {missing_or_invalid}. Parsed: {parsed_outline}")

    if is_valid and final_outline_data: # final_outline_data is populated if is_valid
        agent.plot_outline = final_outline_data
        agent.plot_outline.update(base_elements_for_outline)
        agent.plot_outline.pop("is_default", None) 
        logger.info(f"Successfully generated plot outline: '{agent.plot_outline.get('title', 'N/A')}'")
    else:
        logger.error("Failed to generate a valid plot outline after LLM call and parsing. Applying default.")
        default_plot: PlotOutlineData = {
            "title": config.DEFAULT_PLOT_OUTLINE_TITLE,
            "protagonist_name": default_protagonist_name,
            "protagonist_description": f"Default protagonist: {default_protagonist_name}, a character facing challenges.",
            "plot_points": [f"Default Plot Point {i+1}: An event occurs." for i in range(5)],
            "character_arc": f"Default character arc: {default_protagonist_name} learns something important.",
            "setting": base_elements_for_outline.get("setting", "A generic place."), 
            "conflict": "Default conflict: The protagonist must overcome a significant obstacle.",
            "is_default": True 
        }
        # Merge relevant base elements into the default plot
        default_plot.update({k:v for k,v in base_elements_for_outline.items() if k in ["genre", "theme"]})
        if unhinged_mode: # Add unhinged specific keys if in that mode
            default_plot.update({
                k: base_elements_for_outline[k] 
                for k in ["setting_archetype_used", "protagonist_archetype_used", "conflict_archetype_used"] 
                if k in base_elements_for_outline
            })
        agent.plot_outline = default_plot
    
    agent.plot_outline.setdefault('protagonist_name', default_protagonist_name)
    await agent._save_all_json_state() 
    return agent.plot_outline


async def generate_world_building_logic(agent) -> WorldBuildingData:
    """Generates initial world-building data based on the plot outline.
    'agent' is an instance of NovelWriterAgent.
    Returns the generated world-building data.
    """
    if agent.world_building and not agent.world_building.get("is_default", False):
        # Check if data is already populated and non-default (original logic preserved for behavior)
        keys_minus_default = agent.world_building.keys() - {"is_default"}
        locations_data = agent.world_building.get("locations")
        has_multiple_other_keys = len(keys_minus_default) > 1
        has_multiple_locations = (
            "locations" in agent.world_building and 
            isinstance(locations_data, dict) and 
            len(locations_data) > 1
        )

        if has_multiple_other_keys or has_multiple_locations:
            logger.info("Skipping initial world-building: Data appears to be already populated and non-default.")
            return agent.world_building

    if not agent.plot_outline or not agent.plot_outline.get("setting"):
        logger.error("Cannot generate world-building: Plot outline or setting description is missing. Applying default world-building.")
        default_wb: WorldBuildingData = {
            "locations": {"Default Location": {"description": "A starting point for the story."}},
            "society": {"General Norms": {"description": "Basic societal structures and norms."}},
            "systems": {}, # Ensure all expected top-level keys are present
            "lore": {},
            "history": {},
            "is_default": True
        }
        agent.world_building = default_wb
        await agent._save_all_json_state()
        return agent.world_building

    prompt = f"""/no_think
You are a world-building assistant. Based on the provided novel concept, generate foundational world-building elements.
The output MUST be a single, valid JSON object.

Novel Concept:
Title: {agent.plot_outline.get('title', 'Untitled Novel')}
Genre: {agent.plot_outline.get('genre', 'Not specified')}
Theme: {agent.plot_outline.get('theme', 'Not specified')}
Setting Description (expand significantly on this): {agent.plot_outline.get('setting', 'A default setting')}
Main Conflict: {agent.plot_outline.get('conflict', 'A central conflict')}
Protagonist: {agent.plot_outline.get('protagonist_name', 'N/A')} ({agent.plot_outline.get('protagonist_description', 'N/A')})

Instructions:
1. Create detailed world-building elements. Focus on providing tangible details that can be used in the story.
2. Significantly expand on the provided setting description.
3. Structure the output JSON with top-level keys: "locations", "society", "systems" (e.g., technology, magic), "lore", and "history".
4. Under each top-level key, create sub-dictionaries where each key is the name of a specific element (e.g., a city name under "locations", a faction name under "society").
5. Each specific element's dictionary should at least contain a "description" field. Add other relevant fields as appropriate (e.g., "atmosphere" for locations, "goals" for factions, "rules" for systems).
6. Be creative and imaginative, aligning with the genre and theme.

**CRITICAL: Output ONLY the JSON object.**
Example Structure:
{{
  "locations": {{
    "Capital City": {{ "description": "The bustling heart of the kingdom...", "atmosphere": "Oppressive and gray" }},
    "Forbidden Forest": {{ "description": "An ancient forest no one dares enter..." }}
  }},
  "society": {{
    "Royal Guard": {{ "description": "The elite protectors of the throne.", "goals": ["Protect royalty"] }}
  }},
  "systems": {{
    "Aetheric Magic": {{ "description": "Magic drawn from the ambient aether.", "rules": ["Requires focus", "Weakens with distance"] }}
  }},
  "lore": {{
    "The Great Sundering": {{ "description": "A cataclysmic event that shaped the current world." }}
  }},
  "history": {{
    "Founding Era": {{ "description": "The period when the major kingdoms were established." }}
  }}
}}
"""
    logger.info("Generating initial world-building data via LLM...")
    # World building JSON can also be large.
    raw_world_data_str = await llm_interface.async_call_llm(
        model_name=config.INITIAL_SETUP_MODEL,
        prompt=prompt,
        temperature=0.6,
        stream_to_disk=True # Output can be a fairly large JSON
    )
    
    parsed_llm_response: Any = await llm_interface.async_parse_llm_json_response(raw_world_data_str, "initial world-building")

    is_valid = False 
    final_world_data: WorldBuildingData = {}

    if parsed_llm_response and isinstance(parsed_llm_response, dict):
        parsed_world_data: Dict[str, Any] = parsed_llm_response # Now known to be a dict
        expected_categories = ["locations", "society", "systems", "lore", "history"]
        
        # Check if at least one expected category is present, is a dict, and has content
        if any(cat in parsed_world_data and isinstance(parsed_world_data[cat], dict) and parsed_world_data[cat] 
               for cat in expected_categories):
            # Ensure all expected top-level keys are present in the final data, even if empty from LLM
            temp_data: WorldBuildingData = {}
            for cat in expected_categories:
                temp_data[cat] = parsed_world_data.get(cat, {}) # Default to empty dict if category missing
            final_world_data = temp_data
            is_valid = True
        else:
            logger.warning(f"Generated world-building lacks expected structure or content. Parsed: {parsed_world_data}")
    
    if is_valid and final_world_data: # final_world_data is populated if is_valid
        agent.world_building = final_world_data
        agent.world_building.pop("is_default", None) 
        logger.info("Successfully generated initial world-building data.")
    else:
        logger.error("Failed to generate valid world-building data. Applying default.")
        default_wb: WorldBuildingData = {
            "locations": {"Default Location": {"description": "A starting point."}},
            "society": {"General": {"description": "Basic societal norms."}},
            "systems": {},
            "lore": {},
            "history": {},
            "is_default": True 
        }
        agent.world_building = default_wb
        
    await agent._save_all_json_state() 
    return agent.world_building