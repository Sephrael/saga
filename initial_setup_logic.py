# initial_setup_logic.py
"""
Handles generation of initial plot outlines and world-building data for the SAGA system.
These functions are called by the NovelWriterAgent during its setup phase.
The generated Python dicts are then persisted to Neo4j by the agent.
LLM outputs are now expected in plain text, not JSON.
"""
import logging
import json # Retained for USER_STORY_ELEMENTS_FILE_PATH parsing
import random
import os 
import re
from typing import Dict, Any, Optional, List

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
    """Loads and validates data from the user-supplied story elements file (which is still JSON)."""
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
    """Populates agent's Python dicts from user data (which is JSON). This function's core logic remains."""
    plot_outline: PlotOutlineData = {}
    character_profiles: Dict[str, Any] = {}
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
    
    if "society" not in world_building: world_building["society"] = {}
    if "_factions_" not in world_building["society"]: world_building["society"]["_factions_"] = {} # DEPRECATED, use top-level "factions"

    if "factions" not in world_building: world_building["factions"] = {} # Top level key for factions
    for faction in wd_details.get("key_factions", []):
        if faction.get("name"):
            world_building["factions"][faction["name"]] = { # Ensure it goes into the top-level "factions"
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

def _parse_plain_text_key_value_list(text: str, keys_map: Dict[str, str], list_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Parses plain text with 'Key: Value' pairs and 'List Key:' followed by '- Item' lines.
    `keys_map` maps display keys (from LLM) to internal dict keys.
    `list_keys` specifies internal keys that should result in a list of strings.
    This version attempts to be more robust to leading non-key-value lines.
    """
    parsed_data: Dict[str, Any] = {}
    if list_keys is None: list_keys = []

    current_list_key: Optional[str] = None
    current_list_values: List[str] = []
    
    lines = text.splitlines()
    first_valid_line_found = False

    for line_num, line_content in enumerate(lines):
        line = line_content.strip()
        if not line:
            if first_valid_line_found: # Preserve blank lines once parsing started
                if current_list_key: # If in a list, a blank line might end it
                    parsed_data[current_list_key] = current_list_values
                    current_list_key = None
                    current_list_values = []
            continue # Skip empty lines or leading empty lines

        # Attempt to match key:value or list header
        # If not first_valid_line_found, we are looking for the first parsable line
        
        is_list_item = (line.startswith("- ") or line.startswith("* "))
        
        if current_list_key and is_list_item:
            current_list_values.append(line[2:].strip())
            first_valid_line_found = True
            continue
        
        # If we were collecting a list and this line is not a list item, finalize the list
        if current_list_key:
            parsed_data[current_list_key] = current_list_values
            current_list_key = None
            current_list_values = []
            # The current line might be a new key or list header, so it will be processed next.

        # Try to match "Key: Value"
        match = re.match(r"^\s*([^:]+?):\s*(.*)$", line) # Made key matching non-greedy
        
        if match:
            first_valid_line_found = True
            key_from_llm = match.group(1).strip()
            value_from_llm = match.group(2).strip()
            
            internal_key = None
            # Prioritize exact match from keys_map first
            if key_from_llm in keys_map:
                internal_key = keys_map[key_from_llm]
            else: # Fallback to case-insensitive
                for display_k, internal_k_candidate in keys_map.items():
                    if key_from_llm.lower() == display_k.lower():
                        internal_key = internal_k_candidate
                        break
            
            if internal_key:
                if internal_key in list_keys:
                    current_list_key = internal_key
                    current_list_values = []
                    if value_from_llm: 
                        # If value_from_llm itself looks like a list item, add its content
                        if value_from_llm.startswith("- ") or value_from_llm.startswith("* "):
                            current_list_values.append(value_from_llm[2:].strip())
                        else: # Treat as a single item if not formatted as list item
                            current_list_values.append(value_from_llm)
                else:
                    parsed_data[internal_key] = value_from_llm
            else:
                if first_valid_line_found: # Only log warning if we've already started parsing valid lines
                    logger.warning(f"Unknown key '{key_from_llm}' in LLM plain text output (line {line_num+1}): {line}")
        
        # Check if the line itself is a key for a list (e.g., "Plot Points:")
        # This is typically for when a list starts on the next line with "- " items
        elif not is_list_item: # and not match (already checked)
            key_from_llm_as_list_header = line.replace(":", "").strip()
            internal_key = None
            if key_from_llm_as_list_header in keys_map: # Exact match for list header
                internal_key = keys_map[key_from_llm_as_list_header]
            else: # Case-insensitive for list header
                for display_k, internal_k_candidate in keys_map.items():
                    if key_from_llm_as_list_header.lower() == display_k.lower():
                        internal_key = internal_k_candidate
                        break
            
            if internal_key and internal_key in list_keys:
                first_valid_line_found = True
                current_list_key = internal_key
                current_list_values = []
            elif first_valid_line_found: # If it's not a known key or list start, and we've started parsing
                 logger.debug(f"Line not parsed as key-value or list start (line {line_num+1}): '{line}'")

    # Finalize any pending list
    if current_list_key:
        parsed_data[current_list_key] = current_list_values
        
    return parsed_data


async def generate_plot_outline_logic(agent, default_protagonist_name: str, unhinged_mode: bool, **kwargs) -> PlotOutlineData:
    logger.info(f"Generating plot outline Python dict. Unhinged mode: {unhinged_mode}")
    
    user_supplied_data = _load_user_supplied_data()
    if user_supplied_data:
        logger.info("Processing user-supplied story data for initial setup dicts.")
        _populate_agent_state_from_user_data(agent, user_supplied_data)
        return agent.plot_outline 

    logger.info("No valid user-supplied file. Proceeding with LLM/default generation for Python dicts.")
    base_elements_for_outline: Dict[str, Any] = {}
    
    plot_outline_keys_map = {
        "Title": "title",
        "Protagonist Name": "protagonist_name",
        "Protagonist Description": "protagonist_description",
        "Plot Points": "plot_points", 
        "Character Arc": "character_arc",
        "Conflict Summary": "conflict_summary",
        "Logline": "logline",
        "Setting Description": "setting_description",
        "Inciting Incident": "inciting_incident",
        "Climax Event Preview": "climax_event_preview",
        "Antagonist Name": "antagonist_name",
        "Antagonist Description": "antagonist_description",
        "Antagonist Motivations": "antagonist_motivations"
    }
    list_type_keys_internal = ["plot_points"] 
    llm_fields_to_generate_text = "\n".join([f"- {k}" for k in plot_outline_keys_map.keys()])
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
    
    # Clean the response *before* parsing
    cleaned_outline_text = llm_interface.clean_model_response(raw_outline_text)
    parsed_llm_response = _parse_plain_text_key_value_list(cleaned_outline_text, plot_outline_keys_map, list_type_keys_internal)

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
        agent.world_building = {"locations": {}, "society": {}, "systems": {}, "lore": {}, "history": {}, "_overview_": {}}

    return agent.plot_outline


def _parse_plain_text_world_building(text: str) -> Dict[str, Any]:
    world_data: Dict[str, Any] = {"_overview_": {}}
    lines = text.splitlines()
    
    current_category_internal: Optional[str] = None
    current_item_name: Optional[str] = None
    current_item_details: Dict[str, Any] = {}
    
    category_map_display_to_internal = { # Maps display names (case-insensitive) to internal keys
        "overview": "_overview_", "locations": "locations", "society": "society",
        "systems": "systems", "lore": "lore", "history": "history", "factions": "factions"
    }
    list_detail_keys_internal = ["goals", "rules", "key_elements", "traits"] 

    active_list_key_internal: Optional[str] = None
    active_list_values: List[str] = []

    first_meaningful_line_found = False

    for line_num, line_content in enumerate(lines):
        line = line_content.strip()
        if not line: 
            if first_meaningful_line_found and active_list_key_internal: # Blank line might terminate a list
                if current_item_name and current_category_internal and current_category_internal != "_overview_":
                    current_item_details[active_list_key_internal] = active_list_values
                elif current_category_internal == "_overview_" and active_list_key_internal in world_data["_overview_"]:
                     world_data["_overview_"][active_list_key_internal] = active_list_values
                active_list_key_internal = None
                active_list_values = []
            continue # Skip empty lines

        first_meaningful_line_found = True

        # Finalize active list if current line doesn't continue it (e.g. new key, new item, new category)
        is_list_item_line = line.startswith("- ") or line.startswith("* ")
        if active_list_key_internal and not is_list_item_line:
            if current_item_name and current_category_internal and current_category_internal != "_overview_": # Item specific list
                 current_item_details[active_list_key_internal] = active_list_values
            elif current_category_internal == "_overview_": # Overview list (not typical but possible)
                 if active_list_key_internal in world_data["_overview_"]: # Check if key was initialized
                     world_data["_overview_"][active_list_key_internal].extend(active_list_values) # Extend if exists
                 else:
                      world_data["_overview_"][active_list_key_internal] = active_list_values # Assign if new
            active_list_key_internal = None
            active_list_values = []
        
        # Check for new Category
        # Category lines are typically like "Locations:" or "Overview:"
        cat_match = re.match(r"^\s*([A-Za-z\s_]+?):\s*$", line) # Ends with colon
        potential_cat_display_name = ""
        if cat_match:
            potential_cat_display_name = cat_match.group(1).strip()
        elif not re.search(r":\s*\S", line) and not is_list_item_line : # Doesn't have "Key: Value" and not a list item
            potential_cat_display_name = line.strip() # Could be a category name on its own

        if potential_cat_display_name:
            normalized_potential_cat = potential_cat_display_name.lower().replace(" ", "_")
            found_cat_internal = None
            for disp_cat, int_cat in category_map_display_to_internal.items():
                if disp_cat == normalized_potential_cat:
                    found_cat_internal = int_cat
                    break
            
            if found_cat_internal:
                if current_item_name and current_category_internal and current_category_internal != "_overview_": # Finalize previous item
                    if current_category_internal not in world_data: world_data[current_category_internal] = {}
                    world_data[current_category_internal][current_item_name] = current_item_details
                
                current_category_internal = found_cat_internal
                current_item_name = None # Reset item for new category
                current_item_details = {}
                if current_category_internal != "_overview_": # Ensure category dict exists
                    if current_category_internal not in world_data:
                        world_data[current_category_internal] = {}
                logger.debug(f"Switched to world category: {current_category_internal}")
                continue # Processed as category header

        if not current_category_internal: # Still waiting for the first valid category
            logger.debug(f"World parsing: Skipping line, no active category: '{line}'")
            continue

        # Check for new Item Name (if not in Overview)
        # Item names are typically followed by indented details, or are on their own line then details.
        if current_category_internal != "_overview_":
            item_name_colon_match = re.match(r"^\s*([A-Za-z0-9\s'\-]+?):\s*$", line)
            is_likely_detail_key_value = re.match(r"^\s\s+.*:\s*.", line) 
            
            if item_name_colon_match and not line.startswith("  "): 
                potential_item_name = item_name_colon_match.group(1).strip()
                # Avoid mistaking a detail key (like "Description:") as an item name
                normalized_potential_item_key = potential_item_name.lower().replace(" ","_")
                common_detail_keys = ["description", "atmosphere", "modification_proposal", "goals", "rules", "key_elements", "traits"]
                if normalized_potential_item_key not in common_detail_keys:
                    if current_item_name and current_category_internal: 
                        world_data[current_category_internal][current_item_name] = current_item_details
                    current_item_name = potential_item_name
                    current_item_details = {}
                    logger.debug(f"New world item (colon): {current_item_name} under {current_category_internal}")
                    continue
            elif not line.startswith("  ") and not re.match(r".*:\s*\S+", line) and not is_list_item_line:
                potential_item_name = line.strip()
                if potential_item_name: 
                    if current_item_name and current_category_internal:
                        world_data[current_category_internal][current_item_name] = current_item_details
                    current_item_name = potential_item_name
                    current_item_details = {}
                    logger.debug(f"New world item (standalone line): {current_item_name} under {current_category_internal}")
                    continue

        # Process Item Details (or Overview details)
        target_details_dict = current_item_details if current_category_internal != "_overview_" else world_data["_overview_"]
        
        if not current_item_name and current_category_internal != "_overview_": 
            logger.debug(f"World parsing: Skipping detail line, no active item name for category {current_category_internal}: '{line}'")
            continue

        if is_list_item_line: 
            if active_list_key_internal:
                active_list_values.append(line[2:].strip())
            else: logger.warning(f"World parsing: Orphaned list item '{line}' for item '{current_item_name or '_overview_'}'. No active list key.")
            continue

        detail_match = re.match(r"^\s*([A-Za-z0-9\s_()]+?):\s*(.*)$", line) 
        if detail_match:
            detail_key_raw = detail_match.group(1).strip()
            detail_value = detail_match.group(2).strip()
            detail_key_internal = detail_key_raw.lower().replace(" ", "_").replace("(", "").replace(")", "")

            if detail_key_internal in list_detail_keys_internal:
                active_list_key_internal = detail_key_internal
                active_list_values = []
                
                if detail_value:
                    if detail_value.startswith("- ") or detail_value.startswith("* "):
                        active_list_values.append(detail_value[2:].strip())
                    else: 
                        active_list_values.append(detail_value)
                if active_list_key_internal not in target_details_dict: # Initialize list in dict
                    target_details_dict[active_list_key_internal] = []

            else: 
                target_details_dict[detail_key_internal] = detail_value
        # Handle list header on its own line (e.g., "Goals:")
        elif line.strip().replace(":", "").lower().replace(" ", "_") in list_detail_keys_internal:
            active_list_key_internal = line.strip().replace(":", "").lower().replace(" ", "_")
            active_list_values = []
            if active_list_key_internal not in target_details_dict:
                 target_details_dict[active_list_key_internal] = []


    # Finalize last item and list
    if active_list_key_internal:
        target_dict_final = current_item_details if current_item_name and current_category_internal != "_overview_" else world_data["_overview_"]
        if active_list_key_internal in target_dict_final and isinstance(target_dict_final[active_list_key_internal], list):
             target_dict_final[active_list_key_internal].extend(active_list_values)
        else:
             target_dict_final[active_list_key_internal] = active_list_values

    if current_item_name and current_category_internal and current_category_internal != "_overview_":
        if current_category_internal not in world_data: world_data[current_category_internal] = {}
        world_data[current_category_internal][current_item_name] = current_item_details
        
    return world_data


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
2. Use top-level category headers EXACTLY like these: `Overview:`, `Locations:`, `Society:`, `Systems:`, `Lore:`, `History:`, `Factions:`. Each on its own line.
3. Under each category (except Overview), list item names. Item names should be on their own line, typically followed by a colon or indented details below.
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
    parsed_llm_response = _parse_plain_text_world_building(cleaned_world_text)
    
    is_valid = False 
    final_world_data: WorldBuildingData = {}

    if parsed_llm_response:
        if (parsed_llm_response.get("_overview_", {}).get("description") or
            any(isinstance(items, dict) and items for cat, items in parsed_llm_response.items() if cat != "_overview_")):
            final_world_data = parsed_llm_response
            is_valid = True
        else:
            logger.warning(f"Generated world-building (plain text parsing) lacks expected structure/content. Parsed: {parsed_llm_response}. Cleaned text for parsing: '{cleaned_world_text[:300]}...'")
    
    if is_valid and final_world_data:
        # Ensure all standard categories exist in the final dict, even if empty
        for std_cat in ["locations", "society", "systems", "lore", "history", "factions", "_overview_"]:
            if std_cat not in final_world_data:
                final_world_data[std_cat] = {} if std_cat != "_overview_" else {"description": ""}
            elif std_cat == "_overview_" and not isinstance(final_world_data[std_cat], dict): # Ensure _overview_ is dict
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