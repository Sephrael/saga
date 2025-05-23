# initial_setup_logic.py
import logging
import json 
import random
import os 
import re
from typing import Dict, Any, Optional, List

import config
import llm_interface
from state_manager import state_manager 
from parsing_utils import parse_key_value_block, parse_hierarchical_structured_text # MODIFIED

logger = logging.getLogger(__name__)

PlotOutlineData = Dict[str, Any]
WorldBuildingData = Dict[str, Any]

# --- Plot Outline ---
PLOT_OUTLINE_KEY_MAP = {
    "title": "title",
    "protagonist_name": "protagonist_name",
    "protagonist_description": "protagonist_description",
    "plot_points": "plot_points", 
    "character_arc": "character_arc",
    "conflict_summary": "conflict_summary",
    "logline": "logline",
    "setting_description": "setting_description",
    "inciting_incident": "inciting_incident",
    "climax_event_preview": "climax_event_preview",
    "antagonist_name": "antagonist_name",
    "antagonist_description": "antagonist_description",
    "antagonist_motivations": "antagonist_motivations"
}
PLOT_OUTLINE_LIST_INTERNAL_KEYS = ["plot_points"]

# --- World Building ---
# Maps normalized LLM category names to internal keys
WORLD_CATEGORY_MAP_NORMALIZED_TO_INTERNAL = { 
    "overview": "_overview_", "locations": "locations", "society": "society",
    "systems": "systems", "lore": "lore", "history": "history", "factions": "factions"
}
# Pattern to detect category headers like "Category: Locations" or "Locations:"
WORLD_CATEGORY_HEADER_PATTERN = re.compile(r"^\s*(?:Category\s*:\s*)?([A-Za-z\s_]+?):\s*$", re.IGNORECASE | re.MULTILINE)
# Pattern to detect item names, e.g., "Item Name:" or just "Item Name" on its own line if not indented
# This might need to be more robust; for now, assume item name is followed by a colon or is on a line by itself not looking like a detail.
WORLD_ITEM_HEADER_PATTERN = re.compile(r"^\s*([A-Za-z0-9\s'\-]+?)(?::\s*$|$)", re.MULTILINE) # Simpler: item name then optional colon

# For details within each world item (parsed by parse_key_value_block)
WORLD_DETAIL_KEY_MAP_NORMALIZED_TO_INTERNAL = {
    "description": "description", "atmosphere": "atmosphere", 
    "modification_proposal": "modification_proposal", # For dynamic updates
    "goals": "goals", "rules": "rules", "key_elements": "key_elements", "traits": "traits"
    # Add any other detail keys the LLM might use, normalized
}
WORLD_DETAIL_LIST_INTERNAL_KEYS = ["goals", "rules", "key_elements", "traits"]


def _create_default_plot(default_protagonist_name: str, base_elements: Dict[str, Any], unhinged: bool) -> PlotOutlineData:
    default_plot: PlotOutlineData = {
        "title": config.DEFAULT_PLOT_OUTLINE_TITLE,
        "protagonist_name": default_protagonist_name,
        "protagonist_description": f"Default protagonist: {default_protagonist_name}, a character facing challenges.",
        "plot_points": [f"Default Plot Point {i+1}: An event occurs." for i in range(5)],
        "character_arc": f"Default character arc: {default_protagonist_name} learns something important.",
        "setting_description": base_elements.get("setting_description", base_elements.get("setting", "A generic place.")),
        "conflict_summary": "Default conflict: The protagonist must overcome a significant obstacle.",
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
    for key in ["logline", "inciting_incident", "climax_event_preview", "antagonist_name", "antagonist_description", "antagonist_motivations"]:
        default_plot.setdefault(key, "")
    return default_plot

def _load_user_supplied_data() -> Optional[Dict[str, Any]]:
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
    plot_outline: PlotOutlineData = {}
    character_profiles: Dict[str, Any] = {}
    world_building: WorldBuildingData = {
        "locations": {}, "society": {}, "systems": {}, "lore": {}, "history": {}, "_overview_": {}, "factions": {}, # Added factions
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
    plot_outline["conflict_summary"] = conflict_data.get("summary", "")
    plot_outline["inciting_incident"] = conflict_data.get("inciting_incident", "")
    plot_outline["climax_event_preview"] = conflict_data.get("climax_event_preview", "")
    
    plot_outline["plot_points"] = user_data.get("plot_points", [])
    plot_outline["setting_description"] = user_data.get("setting", {}).get("primary_setting_description", "")
    
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
    
    if "factions" not in world_building: world_building["factions"] = {} 
    for faction in wd_details.get("key_factions", []):
        if faction.get("name"):
            world_building["factions"][faction["name"]] = { 
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
    logger.info(f"Generating plot outline Python dict. Unhinged mode: {unhinged_mode}")
    
    user_supplied_data = _load_user_supplied_data()
    if user_supplied_data:
        logger.info("Processing user-supplied story data for initial setup dicts.")
        _populate_agent_state_from_user_data(agent, user_supplied_data)
        return agent.plot_outline 

    logger.info("No valid user-supplied file. Proceeding with LLM/default generation for Python dicts.")
    base_elements_for_outline: Dict[str, Any] = {}
    
    # Key map keys are normalized LLM display keys
    current_plot_outline_key_map = {k.lower().replace(" ", "_"): v for k, v in PLOT_OUTLINE_KEY_MAP.items()}
    
    llm_fields_to_generate_text = "\n".join([f"- {k.replace('_', ' ').title()}" for k in current_plot_outline_key_map.keys()])
    required_string_keys_internal = ["title", "protagonist_name", "protagonist_description", "character_arc", "conflict_summary", "setting_description"]

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
Generate the following plot elements as plain text key-value pairs. For "Plot Points", list 5 distinct points as sub-items starting with "- ".
{llm_fields_to_generate_text}
"Setting Description" should expand on the archetype.
"""
        base_elements_for_outline = {
            "genre": genre, "theme": theme, "setting_archetype_used": setting_archetype,
            "protagonist_archetype_used": protagonist_archetype, "conflict_archetype_used": conflict_archetype
        }
    else: 
        genre = kwargs.get("genre", config.CONFIGURED_GENRE)
        theme = kwargs.get("theme", config.CONFIGURED_THEME)
        setting_description_input = kwargs.get("setting_description", config.CONFIGURED_SETTING_DESCRIPTION)
        prompt_core_elements = f"""
Novel type: '{genre}', theme: '{theme}'.
Primary setting is: '{setting_description_input}'.
Generate the following plot elements as plain text key-value pairs. For "Plot Points", list 5 distinct points as sub-items starting with "- ".
{llm_fields_to_generate_text}
Protagonist name could be '{default_protagonist_name}'.
"Setting Description" should be based on the input setting.
"""
        base_elements_for_outline = {"genre": genre, "theme": theme, "setting_description_input_to_llm": setting_description_input}

    prompt = f"""/no_think
You are a creative assistant for narrative structure.
{prompt_core_elements}
Output ONLY the plot elements as plain text. Use the format:
Key: Value
For "Plot Points", use:
Plot Points:
- Point 1 description
- Point 2 description
...

Example:
Title: The Crystal Key
Protagonist Name: Elara
Protagonist Description: A young scholar with a hidden destiny.
Setting Description: The ancient, magically-sealed city of Eldoria.
Conflict Summary: Elara must find the Crystal Key to save Eldoria from a creeping magical blight.
Plot Points:
- Elara discovers an ancient map hinting at the Key's location.
- She deciphers a crucial riddle under pressure from the blight's advance.
- Elara faces a guardian protecting the Key, testing her resolve.
- She makes a difficult sacrifice to obtain the Key.
- Elara uses the Key, and Eldoria is saved, but at a personal cost.
Character Arc: Elara grows from a timid scholar into a courageous leader.
Logline: A young scholar must find a legendary artifact to save her city, facing ancient guardians and personal sacrifice.
Inciting Incident: The magical blight first appears, threatening her home and family.
Climax Event Preview: Elara confronts the source of the blight in the city's sealed heart, a choice between personal loss and Eldoria's salvation.
Antagonist Name: Morwen (the Blight Witch)
Antagonist Description: A former guardian of Eldoria corrupted by forbidden magic.
Antagonist Motivations: Believes Eldoria's isolation caused its stagnation and seeks to 'liberate' it through destructive transformation.

Begin your output now:
"""
    logger.info("Calling LLM for plot outline generation (to plain text)...")
    raw_outline_text = await llm_interface.async_call_llm(config.INITIAL_SETUP_MODEL, prompt, 0.7, stream_to_disk=True)
    
    cleaned_outline_text = llm_interface.clean_model_response(raw_outline_text)
    # MODIFIED: Use new parsing utility
    parsed_llm_response = parse_key_value_block(
        cleaned_outline_text,
        key_map=current_plot_outline_key_map,
        list_internal_keys=PLOT_OUTLINE_LIST_INTERNAL_KEYS
    )

    is_valid = False 
    final_outline_data: PlotOutlineData = {} 

    if parsed_llm_response:
        plot_points_value = parsed_llm_response.get("plot_points")
        
        missing_or_invalid_keys = [
            key for key in required_string_keys_internal
            if not (key in parsed_llm_response and 
                    isinstance(parsed_llm_response[key], str) and 
                    parsed_llm_response[key].strip())
        ]
        
        if not (isinstance(plot_points_value, list) and 
                len(plot_points_value) >= 3 and 
                all(isinstance(p, str) and p.strip() for p in plot_points_value)):
            missing_or_invalid_keys.append("plot_points (structure/content issue)")

        if not missing_or_invalid_keys:
            is_valid = True
            final_outline_data = parsed_llm_response
            if 'plot_points' in final_outline_data and isinstance(final_outline_data['plot_points'], list):
                current_pp_count = len(final_outline_data['plot_points'])
                if current_pp_count < 5:
                    for i in range(current_pp_count + 1, 6):
                        final_outline_data['plot_points'].append(f"Placeholder Plot Point {i} - expand.")
                elif current_pp_count > 5:
                    final_outline_data['plot_points'] = final_outline_data['plot_points'][:5]
        else:
            logger.warning(f"LLM plot outline (plain text parsing) failed validation. Missing/invalid keys: {missing_or_invalid_keys}. Parsed: {parsed_llm_response}. Cleaned text for parsing: '{cleaned_outline_text[:300]}...'")

    if is_valid and final_outline_data:
        agent.plot_outline = final_outline_data
        agent.plot_outline.update(base_elements_for_outline)
        agent.plot_outline.pop("is_default", None) 
        agent.plot_outline["source"] = "llm_generated_unhinged" if unhinged_mode else "llm_generated_configured"
        logger.info(f"Successfully generated plot outline dict: '{agent.plot_outline.get('title', 'N/A')}'")
    else:
        logger.error("Failed to generate a valid plot outline dict from plain text. Applying default.")
        agent.plot_outline = _create_default_plot(default_protagonist_name, base_elements_for_outline, unhinged_mode)
    
    prot_name_from_outline = agent.plot_outline.get('protagonist_name')
    if not prot_name_from_outline or not isinstance(prot_name_from_outline, str) or not prot_name_from_outline.strip():
        agent.plot_outline['protagonist_name'] = default_protagonist_name
        logger.warning(f"Protagonist name from LLM outline was invalid or missing, set to default: {default_protagonist_name}")
    
    final_protagonist_name = agent.plot_outline['protagonist_name']

    if not agent.character_profiles: 
        agent.character_profiles = {}
    
    if final_protagonist_name not in agent.character_profiles:
        prot_desc_from_outline = agent.plot_outline.get('protagonist_description', f"The protagonist, {final_protagonist_name}.")
        character_arc_from_outline = agent.plot_outline.get('character_arc', "To be determined.")
        
        agent.character_profiles[final_protagonist_name] = {
            "description": prot_desc_from_outline, "traits": [], 
            "status": "Introduced", "character_arc_summary": character_arc_from_outline,
            "role": "protagonist", "source": agent.plot_outline.get("source", "llm_generated"),
            "relationships": {}
        }
        logger.info(f"Created initial profile for protagonist '{final_protagonist_name}' in agent.character_profiles.")

    if not agent.world_building: 
        agent.world_building = {"locations": {}, "society": {}, "systems": {}, "lore": {}, "history": {}, "_overview_": {}, "factions": {}}

    return agent.plot_outline


async def generate_world_building_logic(agent) -> WorldBuildingData:
    if agent.world_building and agent.world_building.get("user_supplied_data", False):
        logger.info("Skipping LLM world-building dict generation: Data was user-supplied.")
        return agent.world_building

    if agent.world_building and not agent.world_building.get("is_default", False):
        meaningful_categories_count = 0
        for cat, items in agent.world_building.items():
            if cat not in ["is_default", "user_supplied_data", "source", "_overview_"] and isinstance(items, dict) and items:
                meaningful_categories_count +=1
        if meaningful_categories_count > 1 or \
           (agent.world_building.get("_overview_", {}).get("description") and meaningful_categories_count >=1):
            logger.info("Skipping initial world-building dict generation: Data appears non-default and already populated.")
            return agent.world_building

    if not agent.plot_outline or not agent.plot_outline.get("setting_description"):
        logger.error("Cannot generate world-building dict: Plot outline or setting_description missing. Applying default.")
        default_wb: WorldBuildingData = {
            "locations": {"Default Location": {"description": "A starting point."}},
            "society": {"General Norms": {"description": "Basic societal structures."}},
            "systems": {}, "lore": {}, "history": {}, "factions": {}, 
            "_overview_": {"description": "A default world."},
            "is_default": True, "source": "default_fallback"
        }
        agent.world_building = default_wb
        return agent.world_building

    prompt = f"""/no_think
You are a world-building assistant. Based on the novel concept, generate world-building elements as PLAIN TEXT.
Output MUST follow this structured plain text format.

Novel Concept:
Title: {agent.plot_outline.get('title', 'Untitled')}
Genre: {agent.plot_outline.get('genre', 'N/A')}
Theme: {agent.plot_outline.get('theme', 'N/A')}
Setting Description (expand on this): {agent.plot_outline.get('setting_description', 'A default setting')}
Conflict: {agent.plot_outline.get('conflict_summary', 'A central conflict')}
Protagonist: {agent.plot_outline.get('protagonist_name', 'N/A')}

Instructions:
1. Create detailed world-building. Focus on tangible details.
2. Use top-level category headers EXACTLY like these: `Overview:`, `Locations:`, `Society:`, `Systems:`, `Lore:`, `History:`, `Factions:`. Each on its own line, ending with a colon.
3. Under each category (except Overview), list item names. Item names should be on their own line, typically followed by a colon.
4. For each item, provide details as indented "key: value" pairs. Example keys: "Description", "Atmosphere" (for locations), "Goals" (list for factions), "Rules" (list for systems), "Key Elements" (list of notable sub-features). Use Title Case for these keys.
5. For list-based details (like "Goals", "Rules", "Key Elements", "Traits"), list each entry on a new line, indented further and prefixed with "- ".
6. "Overview:" should just have a "Description:" of the overall world feel.

Example Output Format:
Overview:
  Description: A dark, gritty world where advanced technology coexists uneasily with primal magic. Resources are scarce, and factions vie for control.

Locations:
  Capital City Prime:
    Description: A sprawling metropolis encased in a protective energy dome. Gleaming towers pierce the smog-filled sky, while shadowy underlevels teem with black markets.
    Atmosphere: Oppressive, technologically advanced yet decaying.
    Key Elements:
      - The OmniCorp Spire (corporate HQ)
      - Sector 7 Slums
      - The Aetherium (magic research facility)
  Whispering Wastes:
    Description: A vast desert, magically scarred by an ancient war. Strange creatures and dangerous anomalies are common.
    Atmosphere: Desolate, mysterious, dangerous.

Society:
  OmniCorp Enforcers:
    Description: The corporate police force of Capital City Prime, known for their brutality and unwavering loyalty to OmniCorp.
    Goals:
      - Maintain OmniCorp's dominance.
      - Suppress dissent.
      - Control Aetherium distribution.
  Desert Nomads:
    Description: Hardy tribespeople who survive in the Whispering Wastes, possessing unique knowledge of its secrets.
    Traits:
      - Resilient
      - Secretive
      - Skilled in desert survival

Systems:
  Aether-Tech:
    Description: Technology powered by refined magical energy (Aether). Highly potent but volatile.
    Rules:
      - Requires Aetherium crystals to function.
      - Overuse can lead to Aether Sickness.
    Key Elements:
      - Personal Energy Shields
      - Aether-Powered Vehicles
      - Neural Implants

Lore:
  The Sundering War:
    Description: An ancient conflict between technologists and magic-users that reshaped the world and created the Whispering Wastes.

History:
  The Founding of Capital City Prime:
    Description: Established by OmniCorp survivors after The Sundering War, built upon the ruins of an older city.

Factions:
  OmniCorp:
    Description: The dominant hyper-corporation controlling most technology and resources.
    Goals:
      - Total market monopoly.
      - Advancement of Aether-Tech (under their control).
  The Free Scribes:
    Description: An underground network dedicated to preserving forbidden knowledge and history.
    Goals:
      - Resist OmniCorp's information control.
      - Uncover the truth about The Sundering War.

Begin your output now:
"""
    logger.info("Generating initial world-building data (to plain text) via LLM...")
    raw_world_data_text = await llm_interface.async_call_llm(config.INITIAL_SETUP_MODEL, prompt, 0.6, stream_to_disk=True)
    
    cleaned_world_text = llm_interface.clean_model_response(raw_world_data_text)
    
    # MODIFIED: Use new parsing utility
    # Normalize keys from WORLD_DETAIL_KEY_MAP_NORMALIZED_TO_INTERNAL for use in parse_hierarchical_structured_text
    detail_key_map_normalized = {k.lower().replace(" ", "_"): v for k, v in WORLD_DETAIL_KEY_MAP_NORMALIZED_TO_INTERNAL.items()}

    parsed_llm_response = parse_hierarchical_structured_text(
        cleaned_world_text,
        category_pattern=WORLD_CATEGORY_HEADER_PATTERN, # Regex for "CategoryName:"
        item_pattern=WORLD_ITEM_HEADER_PATTERN,         # Regex for "Item Name:" (or just "Item Name")
        detail_key_map=detail_key_map_normalized,
        detail_list_internal_keys=WORLD_DETAIL_LIST_INTERNAL_KEYS,
        overview_category_internal_key="_overview_" # Internal key for the overview section
    )
    
    is_valid = False 
    final_world_data: WorldBuildingData = {}

    if parsed_llm_response:
        # Check if _overview_ has content or any other category has items
        overview_content = parsed_llm_response.get("_overview_", {}).get("description")
        other_categories_have_items = any(
            isinstance(items, dict) and items 
            for cat, items in parsed_llm_response.items() if cat != "_overview_"
        )
        if overview_content or other_categories_have_items:
            final_world_data = parsed_llm_response
            is_valid = True
        else:
            logger.warning(f"Generated world-building (plain text parsing) lacks expected structure/content. Parsed: {parsed_llm_response}. Cleaned text for parsing: '{cleaned_world_text[:300]}...'")
    
    if is_valid and final_world_data:
        for std_cat in ["locations", "society", "systems", "lore", "history", "factions", "_overview_"]:
            if std_cat not in final_world_data:
                final_world_data[std_cat] = {} if std_cat != "_overview_" else {"description": ""}
            elif std_cat == "_overview_" and not isinstance(final_world_data[std_cat], dict):
                final_world_data[std_cat] = {"description": str(final_world_data[std_cat]) if final_world_data[std_cat] else ""}

        agent.world_building = final_world_data
        agent.world_building.pop("is_default", None)
        agent.world_building.pop("user_supplied_data", None) 
        agent.world_building["source"] = "llm_generated"
        logger.info("Successfully generated initial world-building dict via LLM (from plain text).")
    else:
        logger.error("Failed to generate valid world-building dict via LLM (from plain text). Applying default.")
        default_wb: WorldBuildingData = {
            "locations": {"Default Location": {"description": "A starting point."}},
            "society": {"General": {"description": "Basic societal norms."}}, "systems": {}, "lore": {}, "history": {},
            "factions": {}, "_overview_": {"description": "A default world setting."},
            "is_default": True, "source": "default_fallback"
        }
        agent.world_building = default_wb
        
    return agent.world_building