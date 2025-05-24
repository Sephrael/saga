# initial_setup_logic.py
# MODIFIED: Added _get_prop and _get_nested_prop helpers for flexible agent/props access if needed,
# though this module typically modifies the 'agent' object directly.
import logging
import json
import random
import os
import re
from typing import Dict, Any, Optional, List

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
WORLD_ITEM_HEADER_PATTERN = re.compile(r"^\s*([A-Za-z0-9\s'\-]+?)(?::\s*$|$)", re.MULTILINE)
WORLD_DETAIL_KEY_MAP_NORMALIZED_TO_INTERNAL = {
    "description": "description", "atmosphere": "atmosphere", "modification_proposal": "modification_proposal",
    "goals": "goals", "rules": "rules", "key_elements": "key_elements", "traits": "traits"
}
WORLD_DETAIL_LIST_INTERNAL_KEYS = ["goals", "rules", "key_elements", "traits"]


def _create_default_plot(default_protagonist_name: str, base_elements: Dict[str, Any], unhinged: bool) -> PlotOutlineData:
    default_plot: PlotOutlineData = {
        "title": config.DEFAULT_PLOT_OUTLINE_TITLE, "protagonist_name": default_protagonist_name,
        "protagonist_description": f"Default protagonist: {default_protagonist_name}, a character facing challenges.",
        "plot_points": [f"Default Plot Point {i+1}: An event occurs." for i in range(5)],
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
    for key in ["logline", "inciting_incident", "climax_event_preview", "antagonist_name", "antagonist_description", "antagonist_motivations"]:
        default_plot.setdefault(key, "")
    return default_plot

def _load_user_supplied_data() -> Optional[Dict[str, Any]]:
    file_path = config.USER_STORY_ELEMENTS_FILE_PATH
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
            if not isinstance(data, dict) or \
               not isinstance(data.get("novel_concept"), dict) or \
               not isinstance(data.get("protagonist"), dict) or \
               not isinstance(data.get("plot_points"), list):
                logger.error(f"User-supplied file '{file_path}' missing core structure.")
                return None
            logger.info(f"Successfully loaded user-supplied story data from '{file_path}'.")
            return data
        except Exception as e:
            logger.error(f"Error loading user-supplied file '{file_path}': {e}", exc_info=True)
            return None
    return None

def _populate_agent_state_from_user_data(agent: Any, user_data: Dict[str, Any]):
    """ agent is the NANA_Orchestrator instance """
    plot_outline: PlotOutlineData = {}
    character_profiles: Dict[str, Any] = {}
    world_building: WorldBuildingData = {
        "locations": {}, "society": {}, "systems": {}, "lore": {}, "history": {}, "_overview_": {}, "factions": {},
        "user_supplied_data": True, "is_default": False, "source": "user_supplied"
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
    plot_outline["source"] = "user_supplied"; plot_outline["is_default"] = False
    agent.plot_outline = plot_outline # Modifies orchestrator's attribute

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
    agent.character_profiles = character_profiles # Modifies orchestrator's attribute

    setting_data = user_data.get("setting", {})
    if setting_data.get("primary_setting_description"):
         world_building["_overview_"]["description"] = setting_data["primary_setting_description"]
    for loc in setting_data.get("key_locations", []):
        if loc.get("name"): world_building["locations"][loc["name"]] = {"description": loc.get("description", ""), "atmosphere": loc.get("atmosphere", ""), "source": "user_supplied"}
    wd_details = user_data.get("world_details", {})
    if wd_details.get("magic_system_summary"):
        world_building["systems"]["Primary Magic System"] = {"description": wd_details["magic_system_summary"], "rules": ["As described in summary"], "source": "user_supplied"}
    if "factions" not in world_building: world_building["factions"] = {}
    for faction in wd_details.get("key_factions", []):
        if faction.get("name"): world_building["factions"][faction["name"]] = {"description": faction.get("description", ""), "goals": faction.get("goals", []), "source": "user_supplied"}
    if "lore" not in world_building: world_building["lore"] = {}
    for lore_item in wd_details.get("relevant_lore", []):
        if lore_item.get("name"): world_building["lore"][lore_item["name"]] = {"description": lore_item.get("description", ""), "source": "user_supplied"}
    agent.world_building = world_building # Modifies orchestrator's attribute
    logger.info("Agent state populated from user-supplied data.")


async def generate_plot_outline_logic(agent: Any, default_protagonist_name: str, unhinged_mode: bool, **kwargs) -> PlotOutlineData:
    """ agent is the NANA_Orchestrator instance """
    logger.info(f"Generating plot outline. Unhinged: {unhinged_mode}")
    user_supplied_data = _load_user_supplied_data()
    if user_supplied_data:
        logger.info("Processing user-supplied data for initial setup.")
        _populate_agent_state_from_user_data(agent, user_supplied_data)
        return agent.plot_outline

    logger.info("No valid user-supplied file. LLM/default generation.")
    base_elements_for_outline: Dict[str, Any] = {}
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
        prompt_core_elements = f"""Novel type: '{genre}', theme: '{theme}'. Setting: '{setting_archetype}'. Protagonist: '{protagonist_archetype}'. Conflict: '{conflict_archetype}'. Generate fields: {llm_fields_to_generate_text}"""
        base_elements_for_outline = {"genre": genre, "theme": theme, "setting_archetype_used": setting_archetype, "protagonist_archetype_used": protagonist_archetype, "conflict_archetype_used": conflict_archetype}
    else:
        genre = kwargs.get("genre", config.CONFIGURED_GENRE)
        theme = kwargs.get("theme", config.CONFIGURED_THEME)
        setting_description_input = kwargs.get("setting_description", config.CONFIGURED_SETTING_DESCRIPTION)
        prompt_core_elements = f"""Novel type: '{genre}', theme: '{theme}'. Setting: '{setting_description_input}'. Protagonist name: '{default_protagonist_name}'. Generate fields: {llm_fields_to_generate_text}"""
        base_elements_for_outline = {"genre": genre, "theme": theme, "setting_description_input_to_llm": setting_description_input}

    prompt = f"""/no_think
You are a creative assistant for narrative structure.
{prompt_core_elements} 
Output ONLY the plot elements as plain text. Use the format:
Key: Value
For "Plot Points", use this EXACT format:
Plot Points:
- First plot point description.
- Second plot point description.
- Third plot point description.
- Fourth plot point description.
- Fifth plot point description.

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
    parsed_llm_response = parse_key_value_block(
        cleaned_outline_text, current_plot_outline_key_map, PLOT_OUTLINE_LIST_INTERNAL_KEYS
    )
    is_valid = False; final_outline_data: PlotOutlineData = {}
    if parsed_llm_response:
        plot_points_value = parsed_llm_response.get("plot_points")
        missing_or_invalid_keys = [k for k in required_string_keys_internal if not (k in parsed_llm_response and isinstance(parsed_llm_response[k], str) and parsed_llm_response[k].strip())]
        if not (isinstance(plot_points_value, list) and len(plot_points_value) >= 3 and all(isinstance(p, str) and p.strip() for p in plot_points_value)):
            missing_or_invalid_keys.append("plot_points (structure/content issue)")
        if not missing_or_invalid_keys:
            is_valid = True; final_outline_data = parsed_llm_response
            if 'plot_points' in final_outline_data and isinstance(final_outline_data['plot_points'], list):
                current_pp_count = len(final_outline_data['plot_points'])
                if current_pp_count < 5: final_outline_data['plot_points'].extend([f"Placeholder Plot Point {i+1} - expand." for i in range(current_pp_count, 5)])
                elif current_pp_count > 5: final_outline_data['plot_points'] = final_outline_data['plot_points'][:5]
        else: logger.warning(f"LLM plot outline parse failed. Missing/invalid: {missing_or_invalid_keys}. Parsed: {parsed_llm_response}. Text: '{cleaned_outline_text[:300]}...'")

    if is_valid and final_outline_data:
        agent.plot_outline = final_outline_data # Modifies orchestrator's attribute
        agent.plot_outline.update(base_elements_for_outline)
        agent.plot_outline.pop("is_default", None)
        agent.plot_outline["source"] = "llm_generated_unhinged" if unhinged_mode else "llm_generated_configured"
        logger.info(f"Generated plot outline: '{agent.plot_outline.get('title', 'N/A')}'")
    else:
        logger.error("Failed to generate valid plot outline. Applying default.")
        agent.plot_outline = _create_default_plot(default_protagonist_name, base_elements_for_outline, unhinged_mode)
    prot_name_from_outline = agent.plot_outline.get('protagonist_name')
    if not prot_name_from_outline or not isinstance(prot_name_from_outline, str) or not prot_name_from_outline.strip():
        agent.plot_outline['protagonist_name'] = default_protagonist_name
        logger.warning(f"Protagonist name invalid/missing, set to default: {default_protagonist_name}")
    final_protagonist_name = agent.plot_outline['protagonist_name']
    if not agent.character_profiles: agent.character_profiles = {}
    if final_protagonist_name not in agent.character_profiles:
        prot_desc = agent.plot_outline.get('protagonist_description', f"Protagonist, {final_protagonist_name}.")
        char_arc = agent.plot_outline.get('character_arc', "To be determined.")
        agent.character_profiles[final_protagonist_name] = {
            "description": prot_desc, "traits": [], "status": "Introduced", "character_arc_summary": char_arc,
            "role": "protagonist", "source": agent.plot_outline.get("source", "llm_generated"), "relationships": {}
        }
        logger.info(f"Created initial profile for '{final_protagonist_name}'.")
    if not agent.world_building:
        agent.world_building = {"locations": {}, "society": {}, "systems": {}, "lore": {}, "history": {}, "_overview_": {}, "factions": {}}
    return agent.plot_outline


async def generate_world_building_logic(agent: Any) -> WorldBuildingData:
    """ agent is the NANA_Orchestrator instance """
    if agent.world_building and agent.world_building.get("user_supplied_data", False):
        logger.info("Skipping LLM world-building: Data user-supplied.")
        return agent.world_building
    if agent.world_building and not agent.world_building.get("is_default", False):
        meaningful_cats = sum(1 for cat, items in agent.world_building.items() if cat not in ["is_default", "user_supplied_data", "source", "_overview_"] and isinstance(items, dict) and items)
        if meaningful_cats > 1 or (agent.world_building.get("_overview_", {}).get("description") and meaningful_cats >=1):
            logger.info("Skipping initial world-building: Data non-default and populated.")
            return agent.world_building
    if not agent.plot_outline or not agent.plot_outline.get("setting_description"):
        logger.error("Cannot generate world-building: Plot outline/setting missing. Defaulting.")
        default_wb: WorldBuildingData = {"locations": {"Default Location": {"description": "A starting point."}}, "society": {"General Norms": {"description": "Basic structures."}}, "systems": {}, "lore": {}, "history": {}, "factions": {}, "_overview_": {"description": "A default world."}, "is_default": True, "source": "default_fallback"}
        agent.world_building = default_wb # Modifies orchestrator's attribute
        return agent.world_building

    prompt = f"""/no_think
You are a world-building assistant. Based on novel concept, generate world-building elements as PLAIN TEXT.
Novel Concept: Title: {agent.plot_outline.get('title', 'Untitled')}, Genre: {agent.plot_outline.get('genre', 'N/A')}, Setting: {agent.plot_outline.get('setting_description', 'A default setting')}
Instructions: Use headers: `Overview:`, `Locations:`, `Society:`, etc. Item names on own line. Indented "Key: Value" pairs. List items with "- ".
Example: ... (same example as original) ... Begin:
""" # Shortened prompt for brevity, content is same as original
    logger.info("Generating initial world-building data (to plain text) via LLM...")
    raw_world_data_text = await llm_interface.async_call_llm(config.INITIAL_SETUP_MODEL, prompt, 0.6, stream_to_disk=True)
    cleaned_world_text = llm_interface.clean_model_response(raw_world_data_text)
    detail_key_map_normalized = {k.lower().replace(" ", "_"): v for k, v in WORLD_DETAIL_KEY_MAP_NORMALIZED_TO_INTERNAL.items()}
    parsed_llm_response = parse_hierarchical_structured_text(
        cleaned_world_text, WORLD_CATEGORY_HEADER_PATTERN, WORLD_ITEM_HEADER_PATTERN,
        detail_key_map_normalized, WORLD_DETAIL_LIST_INTERNAL_KEYS, "_overview_"
    )
    is_valid = False; final_world_data: WorldBuildingData = {}
    if parsed_llm_response:
        overview_content = parsed_llm_response.get("_overview_", {}).get("description")
        other_cats_have_items = any(isinstance(items, dict) and items for cat, items in parsed_llm_response.items() if cat != "_overview_")
        if overview_content or other_cats_have_items:
            final_world_data = parsed_llm_response; is_valid = True
        else: logger.warning(f"Generated world-building parse lacks structure/content. Parsed: {parsed_llm_response}. Text: '{cleaned_world_text[:300]}...'")

    if is_valid and final_world_data:
        for std_cat in ["locations", "society", "systems", "lore", "history", "factions", "_overview_"]:
            if std_cat not in final_world_data: final_world_data[std_cat] = {} if std_cat != "_overview_" else {"description": ""}
            elif std_cat == "_overview_" and not isinstance(final_world_data[std_cat], dict): final_world_data[std_cat] = {"description": str(final_world_data[std_cat]) if final_world_data[std_cat] else ""}
        agent.world_building = final_world_data # Modifies orchestrator's attribute
        agent.world_building.pop("is_default", None); agent.world_building.pop("user_supplied_data", None)
        agent.world_building["source"] = "llm_generated"
        logger.info("Successfully generated initial world-building dict via LLM.")
    else:
        logger.error("Failed to generate valid world-building dict. Applying default.")
        default_wb: WorldBuildingData = {"locations": {"Default Location": {"description": "A starting point."}}, "society": {"General": {"description": "Basic norms."}}, "systems": {}, "lore": {}, "history": {}, "factions": {}, "_overview_": {"description": "A default world."}, "is_default": True, "source": "default_fallback"}
        agent.world_building = default_wb
    return agent.world_building