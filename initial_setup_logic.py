# initial_setup_logic.py
"""
Handles generation of initial plot outlines and world-building data for the SAGA system.
These functions are called by the NovelWriterAgent during its setup phase.
"""
import logging
import json
import random
import os # For file existence check
from typing import Dict, Any, Optional 

import config
import llm_interface
from state_manager import state_manager # Import state_manager

logger = logging.getLogger(__name__)

# Define type aliases for clarity, representing the structure of data these functions handle.
# These are essentially Dict[str, Any] but provide semantic meaning.
PlotOutlineData = Dict[str, Any]
WorldBuildingData = Dict[str, Any]

def _create_default_plot(default_protagonist_name: str, base_elements: Dict[str, Any], unhinged: bool) -> PlotOutlineData:
    default_plot: PlotOutlineData = {
        "title": config.DEFAULT_PLOT_OUTLINE_TITLE,
        "protagonist_name": default_protagonist_name,
        "protagonist_description": f"Default protagonist: {default_protagonist_name}, a character facing challenges.",
        "plot_points": [f"Default Plot Point {i+1}: An event occurs." for i in range(5)],
        "character_arc": f"Default character arc: {default_protagonist_name} learns something important.",
        "setting": base_elements.get("setting", "A generic place."), 
        "conflict": "Default conflict: The protagonist must overcome a significant obstacle.",
        "is_default": True,
        "source": "default_fallback"
    }
    default_plot.update({k:v for k,v in base_elements.items() if k in ["genre", "theme"]})
    if unhinged:
        default_plot.update({
            k: base_elements[k] 
            for k in ["setting_archetype_used", "protagonist_archetype_used", "conflict_archetype_used"] 
            if k in base_elements
        })
    return default_plot

def _load_user_supplied_data() -> Optional[Dict[str, Any]]:
    """Loads and validates data from the user-supplied story elements file."""
    file_path = config.USER_STORY_ELEMENTS_FILE_PATH
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Basic validation (can be expanded)
            if not isinstance(data, dict) or \
               not isinstance(data.get("novel_concept"), dict) or \
               not isinstance(data.get("protagonist"), dict) or \
               not isinstance(data.get("plot_points"), list):
                logger.error(f"User-supplied file '{file_path}' is missing core structure (novel_concept, protagonist, plot_points).")
                return None
            
            logger.info(f"Successfully loaded user-supplied story data from '{file_path}'.")
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from user-supplied file '{file_path}': {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading user-supplied file '{file_path}': {e}", exc_info=True)
            return None
    return None

def _populate_agent_state_from_user_data(agent, user_data: Dict[str, Any]):
    """Populates agent's plot_outline, character_profiles, and world_building from user data."""
    plot_outline: PlotOutlineData = {}
    character_profiles: Dict[str, Any] = {}
    world_building: WorldBuildingData = {
        "locations": {}, "society": {}, "systems": {}, "lore": {}, "history": {},
        "user_supplied_data": True, # Mark as user-supplied
        "is_default": False # Not a default
    }

    # 1. Populate Plot Outline
    nc = user_data.get("novel_concept", {})
    plot_outline["title"] = nc.get("title", config.DEFAULT_PLOT_OUTLINE_TITLE)
    plot_outline["genre"] = nc.get("genre", config.CONFIGURED_GENRE)
    plot_outline["theme"] = nc.get("theme", config.CONFIGURED_THEME)
    plot_outline["logline"] = nc.get("logline", "")

    prot_data = user_data.get("protagonist", {})
    plot_outline["protagonist_name"] = prot_data.get("name", config.DEFAULT_PROTAGONIST_NAME)
    plot_outline["protagonist_description"] = prot_data.get("description", "")
    plot_outline["character_arc"] = prot_data.get("character_arc", "")
    
    ant_data = user_data.get("antagonist", {})
    if ant_data.get("name"):
        plot_outline["antagonist_name"] = ant_data["name"]
        plot_outline["antagonist_description"] = ant_data.get("description", "")
        plot_outline["antagonist_motivations"] = ant_data.get("motivations", "")

    conflict_data = user_data.get("conflict", {})
    plot_outline["conflict"] = conflict_data.get("summary", "")
    plot_outline["inciting_incident"] = conflict_data.get("inciting_incident", "")
    plot_outline["climax_event_preview"] = conflict_data.get("climax_event_preview", "")
    
    plot_outline["plot_points"] = user_data.get("plot_points", [])
    plot_outline["setting"] = user_data.get("setting", {}).get("primary_setting_description", "") # Used as main setting desc
    
    plot_outline["source"] = "user_supplied"
    plot_outline["is_default"] = False
    agent.plot_outline = plot_outline

    # 2. Populate Character Profiles
    if prot_data.get("name"):
        character_profiles[prot_data["name"]] = {
            "description": prot_data.get("description", ""),
            "traits": prot_data.get("traits", []),
            "status": prot_data.get("initial_status", "As described in initial setup."),
            "character_arc_summary": prot_data.get("character_arc", ""), # Add arc to profile too
            "role": "protagonist",
            "source": "user_supplied"
        }
    if ant_data.get("name"):
        character_profiles[ant_data["name"]] = {
            "description": ant_data.get("description", ""),
            "traits": ant_data.get("traits", []),
            "status": "As described in initial setup.",
            "motivations": ant_data.get("motivations", ""),
            "role": "antagonist",
            "source": "user_supplied"
        }
    for char_detail in user_data.get("other_key_characters", []):
        if char_detail.get("name"):
            character_profiles[char_detail["name"]] = {
                "description": char_detail.get("description", ""),
                "traits": char_detail.get("traits", []),
                "status": "As described in initial setup.",
                "role_in_story": char_detail.get("role_in_story", ""),
                "source": "user_supplied"
            }
    agent.character_profiles = character_profiles

    # 3. Populate World Building
    setting_data = user_data.get("setting", {})
    if setting_data.get("primary_setting_description"):
         world_building["_overview_"] = {"description": setting_data["primary_setting_description"]}

    for loc in setting_data.get("key_locations", []):
        if loc.get("name"):
            world_building["locations"][loc["name"]] = {
                "description": loc.get("description", ""),
                "atmosphere": loc.get("atmosphere", ""),
                "source": "user_supplied"
            }
    
    wd_details = user_data.get("world_details", {})
    if wd_details.get("magic_system_summary"):
        world_building["systems"]["Primary Magic System"] = { # Assuming one main system from summary
            "description": wd_details["magic_system_summary"],
            "rules": ["As described in summary"], # Placeholder
            "source": "user_supplied"
        }
    
    world_building["society"]["_factions_"] = {} # Sub-dictionary for factions
    for faction in wd_details.get("key_factions", []):
        if faction.get("name"):
            world_building["society"]["_factions_"][faction["name"]] = {
                "description": faction.get("description", ""),
                "goals": faction.get("goals", []),
                "source": "user_supplied"
            }
    
    world_building["lore"]["_key_lore_items_"] = {} # Sub-dictionary for lore
    for lore_item in wd_details.get("relevant_lore", []):
        if lore_item.get("name"):
            world_building["lore"]["_key_lore_items_"][lore_item["name"]] = {
                "description": lore_item.get("description", ""),
                "source": "user_supplied"
            }
    agent.world_building = world_building
    logger.info("Agent state (plot_outline, character_profiles, world_building) populated from user-supplied data.")


async def generate_plot_outline_logic(agent, default_protagonist_name: str, unhinged_mode: bool, **kwargs) -> PlotOutlineData:
    """Generates a new plot outline.
    Prioritizes user-supplied file, then unhinged/configured modes, then default.
    'agent' is an instance of NovelWriterAgent.
    Returns the generated plot outline data.
    """
    logger.info(f"Generating plot outline. Unhinged mode: {unhinged_mode}")
    
    # Attempt to load from user-supplied file first
    user_supplied_data = _load_user_supplied_data()
    if user_supplied_data:
        logger.info("Processing user-supplied story data for initial setup.")
        _populate_agent_state_from_user_data(agent, user_supplied_data)
        await agent._save_all_json_state()
        return agent.plot_outline # Already populated

    # Fallback to existing LLM generation logic if no user file
    logger.info("No valid user-supplied file found or processed. Proceeding with LLM/default generation.")
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
    raw_outline_str = await llm_interface.async_call_llm(
        model_name=config.INITIAL_SETUP_MODEL,
        prompt=prompt, 
        temperature=0.6,
        stream_to_disk=True
    )
    
    parsed_llm_response: Any = await llm_interface.async_parse_llm_json_response(raw_outline_str, "plot outline generation")

    is_valid = False 
    final_outline_data: PlotOutlineData = {} 

    if parsed_llm_response and isinstance(parsed_llm_response, dict):
        parsed_outline: Dict[str, Any] = parsed_llm_response
        plot_points = parsed_outline.get("plot_points")
        
        if (all(key in parsed_outline and isinstance(parsed_outline[key], str) and parsed_outline[key].strip()
                for key in required_keys if key != "plot_points") and
            isinstance(plot_points, list) and len(plot_points) == 5 and
            all(isinstance(p, str) and p.strip() for p in plot_points)):
            is_valid = True
            final_outline_data = parsed_outline
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

    if is_valid and final_outline_data:
        agent.plot_outline = final_outline_data
        agent.plot_outline.update(base_elements_for_outline)
        agent.plot_outline.pop("is_default", None) 
        agent.plot_outline["source"] = "llm_generated_unhinged" if unhinged_mode else "llm_generated_configured"
        logger.info(f"Successfully generated plot outline: '{agent.plot_outline.get('title', 'N/A')}'")
    else:
        logger.error("Failed to generate a valid plot outline after LLM call and parsing. Applying default.")
        agent.plot_outline = _create_default_plot(default_protagonist_name, base_elements_for_outline, unhinged_mode)
    
    agent.plot_outline.setdefault('protagonist_name', default_protagonist_name)
    # Initialize character_profiles and world_building if not populated by user file
    if not agent.character_profiles: agent.character_profiles = {}
    if not agent.world_building: agent.world_building = {}
    
    await agent._save_all_json_state() 
    return agent.plot_outline


async def generate_world_building_logic(agent) -> WorldBuildingData:
    """Generates initial world-building data based on the plot outline.
    Skips LLM generation if world_building data was already supplied by user.
    'agent' is an instance of NovelWriterAgent.
    Returns the generated world-building data.
    """
    # Check if world_building was populated by user-supplied data
    if agent.world_building and agent.world_building.get("user_supplied_data", False):
        logger.info("Skipping LLM world-building generation: Data was supplied by user file.")
        return agent.world_building

    if agent.world_building and not agent.world_building.get("is_default", False):
        keys_minus_default = agent.world_building.keys() - {"is_default", "user_supplied_data"}
        locations_data = agent.world_building.get("locations")
        has_multiple_other_keys = len(keys_minus_default) > 1
        has_multiple_locations = (
            "locations" in agent.world_building and 
            isinstance(locations_data, dict) and 
            len(locations_data) > 1
        )
        if has_multiple_other_keys or has_multiple_locations:
            logger.info("Skipping initial world-building: Data appears to be already populated and non-default/non-user-supplied.")
            return agent.world_building

    if not agent.plot_outline or not agent.plot_outline.get("setting"):
        logger.error("Cannot generate world-building: Plot outline or setting description is missing. Applying default world-building.")
        default_wb: WorldBuildingData = {
            "locations": {"Default Location": {"description": "A starting point for the story."}},
            "society": {"General Norms": {"description": "Basic societal structures and norms."}},
            "systems": {}, 
            "lore": {},
            "history": {},
            "is_default": True,
            "source": "default_fallback"
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
    raw_world_data_str = await llm_interface.async_call_llm(
        model_name=config.INITIAL_SETUP_MODEL,
        prompt=prompt,
        temperature=0.6,
        stream_to_disk=True
    )
    
    parsed_llm_response: Any = await llm_interface.async_parse_llm_json_response(raw_world_data_str, "initial world-building")

    is_valid = False 
    final_world_data: WorldBuildingData = {}

    if parsed_llm_response and isinstance(parsed_llm_response, dict):
        parsed_world_data: Dict[str, Any] = parsed_llm_response
        expected_categories = ["locations", "society", "systems", "lore", "history"]
        
        if any(cat in parsed_world_data and isinstance(parsed_world_data[cat], dict) and parsed_world_data[cat] 
               for cat in expected_categories):
            temp_data: WorldBuildingData = {}
            for cat in expected_categories:
                temp_data[cat] = parsed_world_data.get(cat, {})
            final_world_data = temp_data
            is_valid = True
        else:
            logger.warning(f"Generated world-building lacks expected structure or content. Parsed: {parsed_world_data}")
    
    if is_valid and final_world_data:
        agent.world_building = final_world_data
        agent.world_building.pop("is_default", None)
        agent.world_building.pop("user_supplied_data", None) # Clear this if LLM generated
        agent.world_building["source"] = "llm_generated"
        logger.info("Successfully generated initial world-building data via LLM.")
    else:
        logger.error("Failed to generate valid world-building data via LLM. Applying default.")
        default_wb: WorldBuildingData = {
            "locations": {"Default Location": {"description": "A starting point."}},
            "society": {"General": {"description": "Basic societal norms."}},
            "systems": {},
            "lore": {},
            "history": {},
            "is_default": True,
            "source": "default_fallback"
        }
        agent.world_building = default_wb
        
    await agent._save_all_json_state() 
    return agent.world_building