# initial_setup_logic.py
# MODIFIED: Added _get_prop and _get_nested_prop helpers for flexible agent/props access if needed,
# though this module typically modifies the 'agent' object directly.
import logging
import json
import random
import os
import re
from typing import Dict, Any, Optional, List, Tuple

import config
import llm_interface
from state_manager import state_manager
from parsing_utils import parse_key_value_block, parse_hierarchical_structured_text

logger = logging.getLogger(__name__)

PlotOutlineData = Dict[str, Any]
WorldBuildingData = Dict[str, Any]

PLOT_OUTLINE_KEY_MAP = {
    "title": "title", "protagonist_name": "protagonist_name", "protagonist_description": "protagonist_description",
    "plot_points": "plot_points", "character_arc": "character_arc", "conflict_summary": "conflict_summary",
    "logline": "logline", "setting_description": "setting_description", "inciting_incident": "inciting_incident",
    "climax_event_preview": "climax_event_preview", "antagonist_name": "antagonist_name",
    "antagonist_description": "antagonist_description", "antagonist_motivations": "antagonist_motivations"
}
PLOT_OUTLINE_LIST_INTERNAL_KEYS = ["plot_points"]

WORLD_CATEGORY_MAP_NORMALIZED_TO_INTERNAL = {
    "overview": "_overview_", "locations": "locations", "society": "society", "systems": "systems",
    "lore": "lore", "history": "history", "factions": "factions"
}
WORLD_CATEGORY_HEADER_PATTERN = re.compile(r"^\s*(?:Category\s*:\s*)?([A-Za-z\s_]+?):\s*$", re.IGNORECASE | re.MULTILINE)
WORLD_ITEM_HEADER_PATTERN = re.compile(r"^\s*([A-Za-z0-9\s'\-]+?)(?::\s*$|$)", re.MULTILINE) # Item name may or may not have colon
WORLD_DETAIL_KEY_MAP_NORMALIZED_TO_INTERNAL = {
    "description": "description", "atmosphere": "atmosphere", "modification_proposal": "modification_proposal",
    "goals": "goals", "rules": "rules", "key_elements": "key_elements", "traits": "traits"
}
WORLD_DETAIL_LIST_INTERNAL_KEYS = ["goals", "rules", "key_elements", "traits"]


def _create_default_plot(default_protagonist_name: str, base_elements: Dict[str, Any], unhinged: bool) -> PlotOutlineData:
    default_plot: PlotOutlineData = {
        "title": config.DEFAULT_PLOT_OUTLINE_TITLE, "protagonist_name": default_protagonist_name,
        "protagonist_description": f"Default protagonist: {default_protagonist_name}, a character facing challenges.",
        "plot_points": [f"Default Plot Point {i+1}: An event occurs." for i in range(5)], # Ensure 5 default plot points
        "character_arc": f"Default character arc: {default_protagonist_name} learns something important.",
        "setting_description": base_elements.get("setting_description", base_elements.get("setting", "A generic place.")),
        "conflict_summary": "Default conflict: The protagonist must overcome a significant obstacle.",
        "is_default": True, "source": "default_fallback"
    }
    default_plot.update({k:v for k,v in base_elements.items() if k in ["genre", "theme"]})
    if unhinged:
        default_plot.update({
            k: base_elements[k] for k in ["setting_archetype_used", "protagonist_archetype_used", "conflict_archetype_used"] if k in base_elements
        })
    # Ensure all keys from PLOT_OUTLINE_KEY_MAP are present, even if empty string
    for key_in_map in PLOT_OUTLINE_KEY_MAP.values():
        if key_in_map not in default_plot:
            default_plot[key_in_map] = [] if key_in_map in PLOT_OUTLINE_LIST_INTERNAL_KEYS else ""
    return default_plot

def _load_user_supplied_data() -> Optional[Dict[str, Any]]:
    file_path = config.USER_STORY_ELEMENTS_FILE_PATH
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
            # Basic validation: ensure presence of top-level keys expected for user data.
            # More detailed validation could be added (e.g., types of nested fields).
            if not isinstance(data, dict) or \
               not isinstance(data.get("novel_concept"), dict) or \
               not isinstance(data.get("protagonist"), dict) or \
               not isinstance(data.get("plot_points"), list):
                logger.error(f"User-supplied file '{file_path}' is missing one or more core structures: 'novel_concept', 'protagonist', or 'plot_points'.")
                return None
            logger.info(f"Successfully loaded user-supplied story data from '{file_path}'.")
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from user-supplied file '{file_path}': {e}", exc_info=True)
            return None
        except Exception as e: # Catch other potential errors like file IO issues
            logger.error(f"Unexpected error loading user-supplied file '{file_path}': {e}", exc_info=True)
            return None
    return None

def _populate_agent_state_from_user_data(agent: Any, user_data: Dict[str, Any]):
    """ Populates agent's state attributes (plot_outline, character_profiles, world_building) from user-supplied data.
        'agent' is typically the NANA_Orchestrator instance.
    """
    plot_outline: PlotOutlineData = {}
    character_profiles: Dict[str, Any] = {}
    world_building: WorldBuildingData = { # Initialize with standard categories
        "locations": {}, "society": {}, "systems": {}, "lore": {}, "history": {}, "_overview_": {}, "factions": {},
        "user_supplied_data": True, "is_default": False, "source": "user_supplied"
    }

    # Novel Concept & Plot Outline
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
    plot_outline["antagonist_name"] = ant_data.get("name", "") # Optional
    plot_outline["antagonist_description"] = ant_data.get("description", "") # Optional
    plot_outline["antagonist_motivations"] = ant_data.get("motivations", "") # Optional

    conflict_data = user_data.get("conflict", {})
    plot_outline["conflict_summary"] = conflict_data.get("summary", "")
    plot_outline["inciting_incident"] = conflict_data.get("inciting_incident", "")
    plot_outline["climax_event_preview"] = conflict_data.get("climax_event_preview", "")

    plot_outline["plot_points"] = user_data.get("plot_points", [])
    plot_outline["setting_description"] = user_data.get("setting", {}).get("primary_setting_description", "")
    plot_outline["source"] = "user_supplied"
    plot_outline["is_default"] = False
    agent.plot_outline = plot_outline # Modifies orchestrator's attribute

    # Character Profiles
    if prot_data.get("name"):
        character_profiles[prot_data["name"]] = {
            "description": prot_data.get("description", ""),
            "traits": prot_data.get("traits", []),
            "status": prot_data.get("initial_status", "As described"),
            "character_arc_summary": prot_data.get("character_arc", ""),
            "role": "protagonist", "source": "user_supplied",
            "relationships": prot_data.get("relationships", {})
        }
    if ant_data.get("name"): # Antagonist is optional
        character_profiles[ant_data["name"]] = {
            "description": ant_data.get("description", ""),
            "traits": ant_data.get("traits", []),
            "status": "As described",
            "motivations": ant_data.get("motivations", ""),
            "role": "antagonist", "source": "user_supplied",
            "relationships": ant_data.get("relationships", {})
        }
    for char_detail in user_data.get("other_key_characters", []):
        if char_detail.get("name"):
            character_profiles[char_detail["name"]] = {
                "description": char_detail.get("description", ""),
                "traits": char_detail.get("traits", []),
                "status": "As described",
                "role_in_story": char_detail.get("role_in_story", ""),
                "source": "user_supplied",
                "relationships": char_detail.get("relationships", {})
            }
    agent.character_profiles = character_profiles # Modifies orchestrator's attribute

    # World Building
    setting_data = user_data.get("setting", {})
    if setting_data.get("primary_setting_description"):
         world_building["_overview_"]["description"] = setting_data["primary_setting_description"]

    for loc in setting_data.get("key_locations", []):
        if loc.get("name"):
            world_building["locations"][loc["name"]] = {
                "description": loc.get("description", ""),
                "atmosphere": loc.get("atmosphere", ""),
                "source": "user_supplied"
            }
    wd_details = user_data.get("world_details", {})
    if wd_details.get("magic_system_summary"):
        world_building["systems"]["Primary Magic System"] = { # Example item name
            "description": wd_details["magic_system_summary"],
            "rules": ["As described in summary"], # Placeholder
            "source": "user_supplied"
        }
    if "factions" not in world_building: world_building["factions"] = {} # Ensure factions key exists
    for faction in wd_details.get("key_factions", []):
        if faction.get("name"):
            world_building["factions"][faction["name"]] = {
                "description": faction.get("description", ""),
                "goals": faction.get("goals", []),
                "source": "user_supplied"
            }
    if "lore" not in world_building: world_building["lore"] = {} # Ensure lore key exists
    for lore_item in wd_details.get("relevant_lore", []):
        if lore_item.get("name"):
            world_building["lore"][lore_item["name"]] = {
                "description": lore_item.get("description", ""),
                "source": "user_supplied"
            }
    agent.world_building = world_building # Modifies orchestrator's attribute
    logger.info("Agent state populated from user-supplied data.")


async def generate_plot_outline_logic(agent: Any, default_protagonist_name: str, unhinged_mode: bool, **kwargs) -> Tuple[PlotOutlineData, Optional[Dict[str, int]]]:
    """ Generates or loads plot outline. Modifies agent.plot_outline and potentially agent.character_profiles.
        'agent' is the NANA_Orchestrator instance.
        Returns the plot outline and LLM usage data.
    """
    logger.info(f"Generating plot outline. Unhinged mode: {unhinged_mode}")
    user_supplied_data = _load_user_supplied_data()
    if user_supplied_data:
        logger.info("Processing user-supplied data for initial setup.")
        _populate_agent_state_from_user_data(agent, user_supplied_data) # This sets agent.plot_outline etc.
        return agent.plot_outline, None # No LLM usage if from user data

    # If no user data, proceed with LLM/default generation
    logger.info("No valid user-supplied file found or processed. Proceeding with LLM/default generation for plot outline.")
    base_elements_for_outline: Dict[str, Any] = {} # For genre, theme, etc., to add to the generated/default outline
    current_plot_outline_key_map = {k.lower().replace(" ", "_"): v for k, v in PLOT_OUTLINE_KEY_MAP.items()}
    llm_fields_to_generate_text = "\n".join([f"- {k.replace('_', ' ').title()}" for k in current_plot_outline_key_map.keys()])
    required_string_keys_internal = ["title", "protagonist_name", "protagonist_description", "character_arc", "conflict_summary", "setting_description"]

    prompt_core_elements = ""
    if unhinged_mode:
        genre = kwargs.get("genre", random.choice(config.UNHINGED_GENRES))
        theme = kwargs.get("theme", random.choice(config.UNHINGED_THEMES))
        setting_archetype = kwargs.get("setting_archetype", random.choice(config.UNHINGED_SETTINGS_ARCHETYPES))
        protagonist_archetype = kwargs.get("protagonist_archetype", random.choice(config.UNHINGED_PROTAGONIST_ARCHETYPES))
        conflict_archetype = kwargs.get("conflict_archetype", random.choice(config.UNHINGED_CONFLICT_TYPES))
        prompt_core_elements = f"""You are crafting a novel concept.
Core Elements:
  - Genre: '{genre}'
  - Theme: '{theme}'
  - Setting Archetype: '{setting_archetype}'
  - Protagonist Archetype: '{protagonist_archetype}' (Ensure a specific name is generated for 'Protagonist Name')
  - Conflict Archetype: '{conflict_archetype}'

Based on these, generate the following plot outline fields:
{llm_fields_to_generate_text}"""
        base_elements_for_outline = {"genre": genre, "theme": theme, "setting_archetype_used": setting_archetype, "protagonist_archetype_used": protagonist_archetype, "conflict_archetype_used": conflict_archetype}
    else: # Configured mode
        genre = kwargs.get("genre", config.CONFIGURED_GENRE)
        theme = kwargs.get("theme", config.CONFIGURED_THEME)
        setting_description_input = kwargs.get("setting_description", config.CONFIGURED_SETTING_DESCRIPTION)
        prompt_core_elements = f"""You are crafting a novel concept.
Core Elements:
  - Genre: '{genre}'
  - Theme: '{theme}'
  - Setting Description: '{setting_description_input}'
  - Protagonist Name: '{default_protagonist_name}' (You can use this name or generate a new one if it fits better)

Based on these, generate the following plot outline fields:
{llm_fields_to_generate_text}"""
        base_elements_for_outline = {"genre": genre, "theme": theme, "setting_description_input_to_llm": setting_description_input}


    prompt = f"""/no_think
You are a creative assistant specializing in crafting compelling narrative structures.
{prompt_core_elements}

Please output ONLY the plot elements as plain text, using the specified field names.
Use the format:
FieldName: Value

For "Plot Points", use this EXACT format with each point on a new line prefixed by "- ":
Plot Points:
- First plot point description.
- Second plot point description.
- Third plot point description.
- Fourth plot point description.
- Fifth plot point description.

Example of full output:
Title: The Obsidian Labyrinth
Protagonist Name: Kaelen
Protagonist Description: A disgraced cartographer haunted by a past failure, seeking redemption.
Plot Points:
- Kaelen discovers a fragmented map hinting at the legendary Obsidian Labyrinth.
- He is pursued by a ruthless treasure hunter, Silas, who also seeks the Labyrinth's secrets.
- Kaelen must navigate a treacherous mountain pass, using his old cartography skills under pressure.
- Inside the Labyrinth, Kaelen confronts illusions reflecting his past trauma and overcomes them.
- Kaelen finds the Labyrinth's heart, choosing to seal its dangerous power rather than exploit it, finding peace.
Character Arc: Kaelen transforms from a guilt-ridden exile to a self-forgiven individual who values wisdom over renown.
Conflict Summary: Kaelen races against Silas to find the Obsidian Labyrinth, battling both external dangers and his internal demons, to decide the fate of its ancient power.
Logline: A disgraced cartographer seeking redemption must outwit a rival and conquer his past to secure a legendary labyrinth's dangerous secret.
Setting Description: The treacherous Dragon's Tooth mountains, leading to the hidden, reality-bending Obsidian Labyrinth.
Inciting Incident: An old colleague's dying message reveals the first clue to the Labyrinth's existence.
Climax Event Preview: At the Labyrinth's core, Kaelen faces Silas and makes a choice that defines his redemption, determining the Labyrinth's future.
Antagonist Name: Silas Vane
Antagonist Description: A notoriously cunning and amoral treasure hunter, driven by greed and a desire for power.
Antagonist Motivations: Believes the Labyrinth's power belongs to him and will stop at nothing to claim it.

Begin your output now using the requested field names:
"""
    logger.info("Calling LLM for plot outline generation (to plain text)...")
    raw_outline_text, usage_data = await llm_interface.async_call_llm(config.INITIAL_SETUP_MODEL, prompt, 0.7, stream_to_disk=True)
    cleaned_outline_text = llm_interface.clean_model_response(raw_outline_text)

    parsed_llm_response = parse_key_value_block(
        cleaned_outline_text, current_plot_outline_key_map, PLOT_OUTLINE_LIST_INTERNAL_KEYS
    )

    is_valid = False
    final_outline_data: PlotOutlineData = {}
    if parsed_llm_response:
        plot_points_value = parsed_llm_response.get("plot_points")
        missing_or_invalid_keys = [k for k in required_string_keys_internal if not (k in parsed_llm_response and isinstance(parsed_llm_response[k], str) and parsed_llm_response[k].strip())]
        
        if not (isinstance(plot_points_value, list) and len(plot_points_value) >= 3 and all(isinstance(p, str) and p.strip() for p in plot_points_value)):
            missing_or_invalid_keys.append("plot_points (structure/content issue: needs to be a list of at least 3 non-empty strings)")
        
        if not missing_or_invalid_keys:
            is_valid = True
            final_outline_data = parsed_llm_response
            # Ensure exactly 5 plot points, padding or truncating as needed
            if 'plot_points' in final_outline_data and isinstance(final_outline_data['plot_points'], list):
                current_pp_count = len(final_outline_data['plot_points'])
                if current_pp_count < 5:
                    final_outline_data['plot_points'].extend([f"Placeholder Plot Point {i+1} - expand further." for i in range(current_pp_count, 5)])
                elif current_pp_count > 5:
                    final_outline_data['plot_points'] = final_outline_data['plot_points'][:5]
        else:
            logger.warning(f"LLM generated plot outline failed validation after parsing. Missing/invalid keys: {missing_or_invalid_keys}. Parsed response: {parsed_llm_response}. Raw text snippet: '{cleaned_outline_text[:300]}...'")

    if is_valid and final_outline_data:
        agent.plot_outline = final_outline_data # Modifies orchestrator's attribute
        agent.plot_outline.update(base_elements_for_outline) # Add genre, theme etc.
        agent.plot_outline.pop("is_default", None) # Remove any default flag if generated
        agent.plot_outline["source"] = "llm_generated_unhinged" if unhinged_mode else "llm_generated_configured"
        logger.info(f"Successfully generated plot outline via LLM: '{agent.plot_outline.get('title', 'N/A')}'")
    else:
        logger.error("Failed to generate a valid plot outline via LLM. Applying default plot outline.")
        agent.plot_outline = _create_default_plot(default_protagonist_name, base_elements_for_outline, unhinged_mode)
        # If LLM failed, usage_data might still be relevant if the call was made, but quality is low.
        # If we fell back to default *before* an LLM call, usage_data would be None.
        # The current logic calls LLM then validates.

    # Ensure protagonist name from outline is valid and update/create profile
    prot_name_from_outline = agent.plot_outline.get('protagonist_name')
    if not prot_name_from_outline or not isinstance(prot_name_from_outline, str) or not prot_name_from_outline.strip():
        agent.plot_outline['protagonist_name'] = default_protagonist_name # Fallback to default if LLM failed this
        logger.warning(f"Protagonist name from LLM was invalid or missing. Set to default: {default_protagonist_name}")
    
    final_protagonist_name = agent.plot_outline['protagonist_name']

    # Initialize character_profiles on agent if it doesn't exist
    if not hasattr(agent, 'character_profiles') or agent.character_profiles is None:
        agent.character_profiles = {}

    if final_protagonist_name not in agent.character_profiles: # Create profile if not existing (e.g. from user data)
        prot_desc = agent.plot_outline.get('protagonist_description', f"The protagonist, {final_protagonist_name}.")
        char_arc = agent.plot_outline.get('character_arc', "To be determined.")
        agent.character_profiles[final_protagonist_name] = {
            "description": prot_desc, "traits": [], "status": "Introduced",
            "character_arc_summary": char_arc, "role": "protagonist",
            "source": agent.plot_outline.get("source", "llm_generated"),
            "relationships": {}
        }
        logger.info(f"Created initial character profile for protagonist '{final_protagonist_name}'.")
    
    # Initialize world_building on agent if it doesn't exist
    if not hasattr(agent, 'world_building') or agent.world_building is None:
        agent.world_building = {"locations": {}, "society": {}, "systems": {}, "lore": {}, "history": {}, "_overview_": {}, "factions": {}}

    return agent.plot_outline, usage_data


async def generate_world_building_logic(agent: Any) -> Tuple[WorldBuildingData, Optional[Dict[str, int]]]:
    """ Generates initial world-building data. Modifies agent.world_building.
        'agent' is the NANA_Orchestrator instance.
        Returns the world building data and LLM usage data.
    """
    # Check if world_building already exists and is non-default (e.g., from user data or previous run)
    if hasattr(agent, 'world_building') and agent.world_building:
        if agent.world_building.get("user_supplied_data", False):
            logger.info("Skipping LLM world-building generation: Data was user-supplied.")
            return agent.world_building, None
        # Check if it's substantially populated beyond default/metadata keys
        meaningful_categories_count = sum(1 for cat, items in agent.world_building.items()
                                          if cat not in ["is_default", "user_supplied_data", "source", "_overview_"] and isinstance(items, dict) and items)
        overview_has_content = isinstance(agent.world_building.get("_overview_"), dict) and agent.world_building["_overview_"].get("description")
        
        if meaningful_categories_count > 1 or (overview_has_content and meaningful_categories_count >=1):
            logger.info("Skipping initial world-building generation: Existing world_building data appears non-default and populated.")
            return agent.world_building, None

    if not hasattr(agent, 'plot_outline') or not agent.plot_outline or not agent.plot_outline.get("setting_description"):
        logger.error("Cannot generate world-building details: Plot outline or its setting_description is missing. Defaulting world_building.")
        default_wb: WorldBuildingData = {
            "locations": {"Default Location": {"description": "A starting point for adventures."}},
            "society": {"General Norms": {"description": "Basic societal structures."}},
            "systems": {}, "lore": {}, "history": {}, "factions": {},
            "_overview_": {"description": "A default world, ready to be shaped."},
            "is_default": True, "source": "default_fallback"
        }
        agent.world_building = default_wb # Modifies orchestrator's attribute
        return agent.world_building, None

    plot_title = agent.plot_outline.get('title', 'Untitled Novel')
    plot_genre = agent.plot_outline.get('genre', 'N/A')
    plot_setting_desc = agent.plot_outline.get('setting_description', 'A generic but intriguing setting.')

    prompt = f"""/no_think
You are an expert world-building assistant for novelists.
Based on the provided novel concept, generate detailed world-building elements as PLAIN TEXT.

**Novel Concept:**
  - Title: {plot_title}
  - Genre: {plot_genre}
  - Core Setting Idea: {plot_setting_desc}

**Instructions for Output:**
1.  Structure your output using clear category headers (e.g., `Overview:`, `Locations:`, `Society:`, `Systems:`, `Lore:`, `History:`, `Factions:`).
2.  For the `Overview:` category, provide a general description directly.
3.  For other categories (like `Locations`, `Factions`, etc.), list each item on its own line starting with the item's name (e.g., `The Whispering Woods:` or `The Sunken City`).
4.  Under each item, provide indented "Key: Value" pairs for its details. Use keys like `Description`, `Atmosphere`, `Goals`, `Rules`, `Key Elements`, `Traits`.
5.  For list-like details (e.g., `Goals` for a faction, `Rules` for a system), list each sub-item on a new line, prefixed with "- ".
6.  Ensure comprehensive yet concise details. Aim for 2-4 items per category where applicable (except Overview).

**Example Output Structure:**

Overview:
  Description: A sprawling desert planet where water is the most valuable currency, controlled by feuding city-states built around ancient wells. Technology is a mix of salvaged advanced tech and primitive ingenuity.

Locations:
  Oasis of Al-Nujum:
    Description: A legendary hidden oasis, said to be the source of all water.
    Atmosphere: Mystical, serene, heavily guarded by mythical creatures.
    Key Elements:
      - Crystal-clear spring
      - Ancient, glowing flora

  Dustwind Cantina:
    Description: A notorious gathering spot for smugglers, traders, and information brokers on the outskirts of a major city-state.
    Atmosphere: Rowdy, smoky, tense, filled with secrets.

Factions:
  The Aquifer Collective:
    Description: A powerful faction controlling the largest city-state and its wells.
    Goals:
      - Maintain absolute control over water distribution.
      - Suppress knowledge of alternative water sources.
    Traits: Authoritarian, technologically advanced (relatively), ruthless.

Systems:
  Sand-Navigation:
    Description: Methods used to traverse the vast, featureless deserts.
    Rules:
      - Relies on star patterns and subtle wind shifts.
      - Requires specialized gear to survive sandstorms.
    Key Elements:
      - Compass-like devices attuned to planetary magnetics.
      - Trained giant beetle mounts.

Begin your detailed world-building output now:
"""
    logger.info("Generating initial world-building data (to plain text) via LLM...")
    raw_world_data_text, usage_data = await llm_interface.async_call_llm(config.INITIAL_SETUP_MODEL, prompt, 0.6, stream_to_disk=True)
    cleaned_world_text = llm_interface.clean_model_response(raw_world_data_text)

    # Normalize keys for parsing (LLM might use "Description", "description", "DESCRIPTION")
    detail_key_map_normalized = {k.lower().replace(" ", "_"): v for k, v in WORLD_DETAIL_KEY_MAP_NORMALIZED_TO_INTERNAL.items()}

    parsed_llm_response = parse_hierarchical_structured_text(
        cleaned_world_text,
        WORLD_CATEGORY_HEADER_PATTERN,
        WORLD_ITEM_HEADER_PATTERN,
        detail_key_map_normalized,
        WORLD_DETAIL_LIST_INTERNAL_KEYS,
        overview_category_internal_key="_overview_" # Special handling for overview
    )

    is_valid = False
    final_world_data: WorldBuildingData = {}
    if parsed_llm_response:
        # Check if overview has content OR other categories have items
        overview_content = parsed_llm_response.get("_overview_", {}).get("description")
        other_categories_have_items = any(
            isinstance(items, dict) and items for cat, items in parsed_llm_response.items() if cat != "_overview_"
        )
        if overview_content or other_categories_have_items:
            final_world_data = parsed_llm_response
            is_valid = True
        else:
            logger.warning(f"Generated world-building parse resulted in no substantial content. Parsed structure: {parsed_llm_response}. Raw text snippet: '{cleaned_world_text[:300]}...'")

    if is_valid and final_world_data:
        # Ensure all standard categories exist in the final dict, even if empty
        for std_cat in ["locations", "society", "systems", "lore", "history", "factions", "_overview_"]:
            if std_cat not in final_world_data:
                final_world_data[std_cat] = {} if std_cat != "_overview_" else {"description": ""}
            elif std_cat == "_overview_" and not isinstance(final_world_data[std_cat], dict): # Ensure overview is a dict
                final_world_data[std_cat] = {"description": str(final_world_data[std_cat]) if final_world_data[std_cat] else ""}

        agent.world_building = final_world_data # Modifies orchestrator's attribute
        agent.world_building.pop("is_default", None) # Remove any default flag
        agent.world_building.pop("user_supplied_data", None) # Remove if it was set
        agent.world_building["source"] = "llm_generated"
        logger.info("Successfully generated initial world-building dictionary via LLM.")
    else:
        logger.error("Failed to generate a valid world-building dictionary via LLM. Applying default world_building.")
        default_wb: WorldBuildingData = {
            "locations": {"Default Location": {"description": "A starting point."}},
            "society": {"General Social Norms": {"description": "Basic societal structures."}},
            "systems": {}, "lore": {}, "history": {}, "factions": {},
            "_overview_": {"description": "A default world setting."},
            "is_default": True, "source": "default_fallback"
        }
        agent.world_building = default_wb
    return agent.world_building, usage_data