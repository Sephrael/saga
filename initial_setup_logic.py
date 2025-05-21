# initial_setup_logic.py
"""
Handles generation of initial plot outlines and world-building data for the SAGA system.
These functions are called by the NovelWriterAgent during its setup phase.
The generated Python dicts are then persisted to Neo4j by the agent.
"""
import logging
import json
import random
import os 
from typing import Dict, Any, Optional 

import config
import llm_interface
from state_manager import state_manager 

logger = logging.getLogger(__name__)

PlotOutlineData = Dict[str, Any]
WorldBuildingData = Dict[str, Any]

def _create_default_plot(default_protagonist_name: str, base_elements: Dict[str, Any], unhinged: bool) -> PlotOutlineData:
    default_plot: PlotOutlineData = {
        "title": config.DEFAULT_PLOT_OUTLINE_TITLE,
        "protagonist_name": default_protagonist_name,
        "protagonist_description": f"Default protagonist: {default_protagonist_name}, a character facing challenges.",
        "plot_points": [f"Default Plot Point {i+1}: An event occurs." for i in range(5)],
        "character_arc": f"Default character arc: {default_protagonist_name} learns something important.",
        "setting_description": base_elements.get("setting_description", base_elements.get("setting", "A generic place.")), # Use setting_description
        "conflict_summary": "Default conflict: The protagonist must overcome a significant obstacle.", # Use conflict_summary
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
    # Ensure essential keys are present for decomposed saving later
    for key in ["logline", "inciting_incident", "climax_event_preview", "antagonist_name", "antagonist_description", "antagonist_motivations"]:
        default_plot.setdefault(key, "")
    return default_plot

def _load_user_supplied_data() -> Optional[Dict[str, Any]]:
    """Loads and validates data from the user-supplied story elements file."""
    file_path = config.USER_STORY_ELEMENTS_FILE_PATH
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, dict) or \
               not isinstance(data.get("novel_concept"), dict) or \
               not isinstance(data.get("protagonist"), dict) or \
               not isinstance(data.get("plot_points"), list):
                logger.error(f"User-supplied file '{file_path}' is missing core structure.")
                return None
            logger.info(f"Successfully loaded user-supplied story data from '{file_path}'.")
            return data
        except Exception as e:
            logger.error(f"Error loading or parsing user-supplied file '{file_path}': {e}", exc_info=True)
            return None
    return None

def _populate_agent_state_from_user_data(agent, user_data: Dict[str, Any]):
    """Populates agent's Python dicts (plot_outline, character_profiles, world_building) from user data."""
    plot_outline: PlotOutlineData = {}
    character_profiles: Dict[str, Any] = {}
    # Initialize world_building with expected top-level categories for consistency, even if empty
    world_building: WorldBuildingData = {
        "locations": {}, "society": {}, "systems": {}, "lore": {}, "history": {}, "_overview_": {},
        "user_supplied_data": True, 
        "is_default": False,
        "source": "user_supplied"
    }

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
    plot_outline["antagonist_name"] = ant_data.get("name", "")
    plot_outline["antagonist_description"] = ant_data.get("description", "")
    plot_outline["antagonist_motivations"] = ant_data.get("motivations", "")

    conflict_data = user_data.get("conflict", {})
    plot_outline["conflict_summary"] = conflict_data.get("summary", "") # Use conflict_summary
    plot_outline["inciting_incident"] = conflict_data.get("inciting_incident", "")
    plot_outline["climax_event_preview"] = conflict_data.get("climax_event_preview", "")
    
    plot_outline["plot_points"] = user_data.get("plot_points", [])
    plot_outline["setting_description"] = user_data.get("setting", {}).get("primary_setting_description", "") # Use setting_description
    
    plot_outline["source"] = "user_supplied"
    plot_outline["is_default"] = False
    agent.plot_outline = plot_outline

    if prot_data.get("name"):
        character_profiles[prot_data["name"]] = {
            "description": prot_data.get("description", ""), "traits": prot_data.get("traits", []),
            "status": prot_data.get("initial_status", "As described"), "character_arc_summary": prot_data.get("character_arc", ""),
            "role": "protagonist", "source": "user_supplied", "relationships": prot_data.get("relationships", {})
        }
    if ant_data.get("name"):
        character_profiles[ant_data["name"]] = {
            "description": ant_data.get("description", ""), "traits": ant_data.get("traits", []),
            "status": "As described", "motivations": ant_data.get("motivations", ""),
            "role": "antagonist", "source": "user_supplied", "relationships": ant_data.get("relationships", {})
        }
    for char_detail in user_data.get("other_key_characters", []):
        if char_detail.get("name"):
            character_profiles[char_detail["name"]] = {
                "description": char_detail.get("description", ""), "traits": char_detail.get("traits", []),
                "status": "As described", "role_in_story": char_detail.get("role_in_story", ""),
                "source": "user_supplied", "relationships": char_detail.get("relationships", {})
            }
    agent.character_profiles = character_profiles

    setting_data = user_data.get("setting", {})
    if setting_data.get("primary_setting_description"):
         world_building["_overview_"]["description"] = setting_data["primary_setting_description"]

    for loc in setting_data.get("key_locations", []):
        if loc.get("name"):
            world_building["locations"][loc["name"]] = {
                "description": loc.get("description", ""), "atmosphere": loc.get("atmosphere", ""), "source": "user_supplied"
            }
    
    wd_details = user_data.get("world_details", {})
    if wd_details.get("magic_system_summary"):
        world_building["systems"]["Primary Magic System"] = {
            "description": wd_details["magic_system_summary"], "rules": ["As described in summary"], "source": "user_supplied"
        }
    
    # Ensure 'society' category and its '_factions_' sub-key exist
    if "society" not in world_building: world_building["society"] = {}
    if "_factions_" not in world_building["society"]: world_building["society"]["_factions_"] = {} # For compatibility if old code expects this

    for faction in wd_details.get("key_factions", []):
        if faction.get("name"):
            # Store factions directly under 'society' or a more specific 'factions' sub-category
            # For simplicity and to match prompt expectations, let's use 'factions' as a direct category if not already.
            if "factions" not in world_building: world_building["factions"] = {} # Add 'factions' category
            world_building["factions"][faction["name"]] = { # Storing under 'factions' category
                "description": faction.get("description", ""), "goals": faction.get("goals", []), "source": "user_supplied"
            }
    
    if "lore" not in world_building: world_building["lore"] = {}
    for lore_item in wd_details.get("relevant_lore", []):
        if lore_item.get("name"):
            world_building["lore"][lore_item["name"]] = {
                "description": lore_item.get("description", ""), "source": "user_supplied"
            }
    agent.world_building = world_building
    logger.info("Agent's Python dicts populated from user-supplied data.")


async def generate_plot_outline_logic(agent, default_protagonist_name: str, unhinged_mode: bool, **kwargs) -> PlotOutlineData:
    """Generates a new plot outline (Python dict).
    Prioritizes user-supplied file, then unhinged/configured modes, then default.
    """
    logger.info(f"Generating plot outline Python dict. Unhinged mode: {unhinged_mode}")
    
    user_supplied_data = _load_user_supplied_data()
    if user_supplied_data:
        logger.info("Processing user-supplied story data for initial setup dicts.")
        _populate_agent_state_from_user_data(agent, user_supplied_data)
        # No immediate save here; agent._save_all_json_state() will be called later by agent or main flow
        return agent.plot_outline 

    logger.info("No valid user-supplied file. Proceeding with LLM/default generation for Python dicts.")
    base_elements_for_outline: Dict[str, Any] = {}
    llm_fields_to_generate = [
        "`title`: A compelling title.",
        "`protagonist_name`: Suitable name.",
        "`protagonist_description`: Brief (1-2 sentences).",
        "`plot_points`: JSON list of 5 strings (major plot points).",
        "`character_arc`: Protagonist's development.",
        "`conflict_summary`: Main conflict driver.", # Changed from 'conflict' to 'conflict_summary'
        "`logline`: A concise one-sentence summary.",
        "`setting_description`: Brief (1-2 sentences) description of primary setting.", # Changed from 'setting'
        "`inciting_incident`: The event that kicks off the main plot.",
        "`climax_event_preview`: A hint at the story's climax."
        # Optional antagonist fields based on complexity desired
        # "`antagonist_name`: Name of the primary antagonist, if any.",
        # "`antagonist_description`: Brief description of the antagonist.",
        # "`antagonist_motivations`: What drives the antagonist."
    ]
    # Keys that are expected to be non-empty strings
    required_string_keys = ["title", "protagonist_name", "protagonist_description", "character_arc", "conflict_summary", "setting_description"]


    if unhinged_mode:
        genre = kwargs.get("genre", random.choice(config.UNHINGED_GENRES))
        theme = kwargs.get("theme", random.choice(config.UNHINGED_THEMES))
        setting_archetype = kwargs.get("setting_archetype", random.choice(config.UNHINGED_SETTINGS_ARCHETYPES))
        protagonist_archetype = kwargs.get("protagonist_archetype", random.choice(config.UNHINGED_PROTAGONIST_ARCHETYPES))
        conflict_archetype = kwargs.get("conflict_archetype", random.choice(config.UNHINGED_CONFLICT_TYPES))
        
        prompt_core_elements = f"""
Novel type: '{genre}', theme: '{theme}'.
Setting inspired by: '{setting_archetype}'.
Protagonist archetype: '{protagonist_archetype}'.
Main conflict around: '{conflict_archetype}'.
Generate JSON fields for: {', '.join(llm_fields_to_generate)}.
`setting_description` should expand on the archetype.
"""
        base_elements_for_outline = {
            "genre": genre, "theme": theme, "setting_archetype_used": setting_archetype,
            "protagonist_archetype_used": protagonist_archetype, "conflict_archetype_used": conflict_archetype
        }
    else: 
        genre = kwargs.get("genre", config.CONFIGURED_GENRE)
        theme = kwargs.get("theme", config.CONFIGURED_THEME)
        setting_description_input = kwargs.get("setting_description", config.CONFIGURED_SETTING_DESCRIPTION) # Renamed input var
        prompt_core_elements = f"""
Novel type: '{genre}', theme: '{theme}'.
Primary setting is: '{setting_description_input}'.
Generate JSON fields for: {', '.join(llm_fields_to_generate)}.
Protagonist name could be '{default_protagonist_name}'.
`setting_description` should be based on the input setting.
"""
        base_elements_for_outline = {"genre": genre, "theme": theme, "setting_description_input_to_llm": setting_description_input}


    prompt = f"""/no_think
You are a creative assistant for narrative structure.
{prompt_core_elements}
Output ONLY the JSON object. Ensure `plot_points` is a list of 5 strings.
Example JSON (keys might vary slightly):
{{
  "title": "string", "protagonist_name": "string", "protagonist_description": "string",
  "setting_description": "string", "conflict_summary": "string", 
  "plot_points": ["string1", "string2", "string3", "string4", "string5"],
  "character_arc": "string", "logline": "string", "inciting_incident": "string", "climax_event_preview": "string"
}}
"""
    logger.info("Calling LLM for plot outline generation (to Python dict)...")
    raw_outline_str = await llm_interface.async_call_llm(config.INITIAL_SETUP_MODEL, prompt, 0.7, stream_to_disk=True) # Increased temp slightly
    
    parsed_llm_response: Any = await llm_interface.async_parse_llm_json_response(raw_outline_str, "plot outline generation")

    is_valid = False 
    final_outline_data: PlotOutlineData = {} 

    if parsed_llm_response and isinstance(parsed_llm_response, dict):
        parsed_outline_dict: Dict[str, Any] = parsed_llm_response
        plot_points_value = parsed_outline_dict.get("plot_points")
        
        missing_or_invalid_keys = [
            key for key in required_string_keys  # Use the list of keys expected to be strings
            if not (key in parsed_outline_dict and 
                    isinstance(parsed_outline_dict[key], str) and 
                    parsed_outline_dict[key].strip())
        ]
        
        # Validate plot_points separately for its specific structure
        if not (isinstance(plot_points_value, list) and 
                len(plot_points_value) == 5 and 
                all(isinstance(p, str) and p.strip() for p in plot_points_value)):
            missing_or_invalid_keys.append("plot_points (structure/content issue)")

        if not missing_or_invalid_keys:
            is_valid = True
            final_outline_data = parsed_outline_dict
        else:
            logger.warning(f"LLM plot outline failed validation. Missing/invalid keys: {missing_or_invalid_keys}. Parsed: {parsed_outline_dict}")

    if is_valid and final_outline_data:
        agent.plot_outline = final_outline_data
        agent.plot_outline.update(base_elements_for_outline) # Add genre, theme etc.
        agent.plot_outline.pop("is_default", None) 
        agent.plot_outline["source"] = "llm_generated_unhinged" if unhinged_mode else "llm_generated_configured"
        logger.info(f"Successfully generated plot outline dict: '{agent.plot_outline.get('title', 'N/A')}'")
    else:
        logger.error("Failed to generate a valid plot outline dict. Applying default.")
        agent.plot_outline = _create_default_plot(default_protagonist_name, base_elements_for_outline, unhinged_mode)
    
    # Ensure protagonist_name is set, defaulting if necessary
    prot_name_from_outline = agent.plot_outline.get('protagonist_name')
    if not prot_name_from_outline or not isinstance(prot_name_from_outline, str) or not prot_name_from_outline.strip():
        agent.plot_outline['protagonist_name'] = default_protagonist_name
        logger.warning(f"Protagonist name from LLM outline was invalid or missing, set to default: {default_protagonist_name}")
    
    final_protagonist_name = agent.plot_outline['protagonist_name']

    # Initialize character_profiles if it's empty, and add the protagonist
    if not agent.character_profiles: 
        agent.character_profiles = {}
    
    if final_protagonist_name not in agent.character_profiles:
        # If user_supplied_data path was not taken, character_profiles would be empty or not contain this LLM-generated protagonist.
        # Create a basic profile for the protagonist.
        prot_desc_from_outline = agent.plot_outline.get('protagonist_description', f"The protagonist, {final_protagonist_name}.")
        character_arc_from_outline = agent.plot_outline.get('character_arc', "To be determined.")
        
        agent.character_profiles[final_protagonist_name] = {
            "description": prot_desc_from_outline,
            "traits": [], # LLM plot outline doesn't typically generate traits, can be added later
            "status": "Introduced", # Default status for a newly created character
            "character_arc_summary": character_arc_from_outline,
            "role": "protagonist",
            "source": agent.plot_outline.get("source", "llm_generated"), # Inherit source from plot
            "relationships": {} # Start with empty relationships
        }
        logger.info(f"Created initial profile for protagonist '{final_protagonist_name}' in agent.character_profiles.")

    if not agent.world_building: 
        agent.world_building = {"locations": {}, "society": {}, "systems": {}, "lore": {}, "history": {}, "_overview_": {}}

    # The agent's Python dict (self.plot_outline) is now populated.
    # Saving to Neo4j (decomposed) happens via agent._save_all_json_state() later.
    return agent.plot_outline


async def generate_world_building_logic(agent) -> WorldBuildingData:
    """Generates initial world-building data (Python dict) based on the plot outline.
    Skips LLM generation if world_building data was already user-supplied or looks populated.
    """
    if agent.world_building and agent.world_building.get("user_supplied_data", False):
        logger.info("Skipping LLM world-building dict generation: Data was user-supplied.")
        return agent.world_building

    # Check if world_building dict seems substantially populated already
    if agent.world_building and not agent.world_building.get("is_default", False):
        # Count meaningful categories (excluding meta keys and overview if it's just a placeholder)
        meaningful_categories_count = 0
        for cat, items in agent.world_building.items():
            if cat not in ["is_default", "user_supplied_data", "source", "_overview_"] and isinstance(items, dict) and items:
                meaningful_categories_count +=1
        if meaningful_categories_count > 1 or \
           (agent.world_building.get("_overview_", {}).get("description") and meaningful_categories_count >=1):
            logger.info("Skipping initial world-building dict generation: Data appears non-default and already populated.")
            return agent.world_building

    if not agent.plot_outline or not agent.plot_outline.get("setting_description"): # Check for setting_description
        logger.error("Cannot generate world-building dict: Plot outline or setting_description missing. Applying default.")
        default_wb: WorldBuildingData = {
            "locations": {"Default Location": {"description": "A starting point."}},
            "society": {"General Norms": {"description": "Basic societal structures."}},
            "systems": {}, "lore": {}, "history": {}, "_overview_": {"description": "A default world."},
            "is_default": True, "source": "default_fallback"
        }
        agent.world_building = default_wb
        return agent.world_building

    prompt = f"""/no_think
You are a world-building assistant. Based on the novel concept, generate world-building elements (Python dict).
Output MUST be a single, valid JSON object.

Novel Concept:
Title: {agent.plot_outline.get('title', 'Untitled')}
Genre: {agent.plot_outline.get('genre', 'N/A')}
Theme: {agent.plot_outline.get('theme', 'N/A')}
Setting Description (expand on this): {agent.plot_outline.get('setting_description', 'A default setting')}
Conflict: {agent.plot_outline.get('conflict_summary', 'A central conflict')}
Protagonist: {agent.plot_outline.get('protagonist_name', 'N/A')}

Instructions:
1. Create detailed world-building. Focus on tangible details.
2. Structure JSON with keys: "locations", "society", "systems" (e.g., tech, magic), "lore", "history", "_overview_".
3. Under each key (except _overview_), create sub-dictionaries (item name -> details dict).
4. Each item's details dict needs "description". Add other fields like "atmosphere" (locations), "goals" (list for factions), "rules" (list for systems), "key_elements" (list of notable sub-features or components).
5. "_overview_" should have a "description" of the overall world feel.
Output ONLY the JSON object. Example:
{{
  "_overview_": {{"description": "A dark, gritty world..."}},
  "locations": {{
    "Capital": {{"description": "...", "atmosphere": "Oppressive", "key_elements": ["The Spire", "Market District"]}}
  }},
  "society": {{
    "RoyalGuard": {{"description": "...", "goals": ["Protect the crown", "Maintain order"]}}
  }},
  "systems": {{
    "AetherMagic": {{"description": "...", "rules": ["Requires focus", "Drains vitality"], "key_elements": ["Aether Crystals", "Leylines"]}}
  }},
  "lore": {{"GreatWar": {{"description": "..."}}}},
  "history": {{"FoundingEra": {{"description": "..."}}}}
}}
"""
    logger.info("Generating initial world-building data (to Python dict) via LLM...")
    raw_world_data_str = await llm_interface.async_call_llm(config.INITIAL_SETUP_MODEL, prompt, 0.6, stream_to_disk=True)
    parsed_llm_response: Any = await llm_interface.async_parse_llm_json_response(raw_world_data_str, "initial world-building")

    is_valid = False 
    final_world_data: WorldBuildingData = {}

    if parsed_llm_response and isinstance(parsed_llm_response, dict):
        parsed_world_dict: Dict[str, Any] = parsed_llm_response
        expected_categories = ["locations", "society", "systems", "lore", "history", "_overview_"]
        # Check if at least one category has content or if overview is present
        if any(cat in parsed_world_dict and isinstance(parsed_world_dict[cat], dict) and (parsed_world_dict[cat] or cat == "_overview_")
               for cat in expected_categories):
            final_world_data = {cat: parsed_world_dict.get(cat, {}) for cat in expected_categories}
            is_valid = True
        else:
            logger.warning(f"Generated world-building dict lacks expected structure/content. Parsed: {parsed_world_dict}")
    
    if is_valid and final_world_data:
        agent.world_building = final_world_data
        agent.world_building.pop("is_default", None)
        agent.world_building.pop("user_supplied_data", None) 
        agent.world_building["source"] = "llm_generated"
        logger.info("Successfully generated initial world-building dict via LLM.")
    else:
        logger.error("Failed to generate valid world-building dict via LLM. Applying default.")
        default_wb: WorldBuildingData = {
            "locations": {"Default Location": {"description": "A starting point."}},
            "society": {"General": {"description": "Basic societal norms."}}, "systems": {}, "lore": {}, "history": {},
            "_overview_": {"description": "A default world setting."},
            "is_default": True, "source": "default_fallback"
        }
        agent.world_building = default_wb
        
    # Agent's Python dict (self.world_building) is populated.
    # Saving to Neo4j (decomposed) happens via agent._save_all_json_state() later.
    return agent.world_building