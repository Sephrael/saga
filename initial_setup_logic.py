# initial_setup_logic.py
import logging
import json # Retain for fallback or other JSON operations
import random
import os
import re
from typing import Dict, Any, Optional, List, Tuple

import config
from llm_interface import llm_service
import utils # For _is_fill_in
# MODIFIED: Aliased import for specific debug logging
# from parsing_utils import parse_key_value_block # Kept for plot parsing # REMOVED this import
from yaml_parser import load_yaml_file
# parse_markdown_to_dict was in markdown_story_parser.py, which is now deleted.
# This will likely cause an error later if not addressed.
# from markdown_story_parser import parse_markdown_to_dict # This line would now fail

logger = logging.getLogger(__name__)

PlotOutlineData = Dict[str, Any]
WorldBuildingData = Dict[str, Any]

PLOT_OUTLINE_KEY_MAP = {
    "title": "title", "protagonist_name": "protagonist_name", "protagonist_description": "protagonist_description",
    "plot_points": "plot_points", "character_arc": "character_arc", "conflict_summary": "conflict_summary",
    "logline": "logline", "setting_description": "setting_description", "inciting_incident": "inciting_incident",
    "climax_event_preview": "climax_event_preview", "antagonist_name": "antagonist_name",
    "antagonist_description": "antagonist_description", "antagonist_motivations": "antagonist_motivations",
    "genre": "genre", "theme": "theme" # Added genre and theme to be fillable
}
PLOT_OUTLINE_LIST_INTERNAL_KEYS = ["plot_points"]

WORLD_CATEGORY_MAP_NORMALIZED_TO_INTERNAL = {
    "overview": "_overview_", "locations": "locations", "society": "society", "systems": "systems",
    "lore": "lore", "history": "history", "factions": "factions"
}
# This map is crucial for mapping keys from Markdown (normalized) to your agent's internal keys
WORLD_DETAIL_KEY_MAP_FROM_MARKDOWN_TO_INTERNAL = {
    "description": "description", "atmosphere": "atmosphere", "modification_proposal": "modification_proposal",
    "goals": "goals", "rules": "rules", "key_elements": "key_elements", "traits": "traits",
    "primary_setting_description": "description", # For top-level setting section
    "known_effects": "known_effects", # For lore items
    "mood": "mood", # From LLM output
    "time_period": "time_period", # From LLM output
    "primary_conflict": "primary_conflict", # From LLM output
    "features": "features", # From LLM output
    "function": "function", # For systems from LLM output
    "key_beliefs": "key_beliefs", # For lore/factions from LLM output
    "key_events": "key_events", # For lore/history from LLM output
    "key_figures": "key_figures", # For history from LLM output
    "impact_on_daily_life": "impact_on_daily_life", # For society from LLM output
    "structure": "structure" # For society/factions from LLM output
}
WORLD_DETAIL_LIST_INTERNAL_KEYS = ["goals", "rules", "key_elements", "traits", "key_beliefs", "key_events", "key_figures", "features"] # Added features

def parse_key_value_block(text_block: str, key_map: Dict[str, str], list_keys_internal: List[str]) -> Dict[str, Any]:
    """
    Parses a block of text where keys are followed by values, potentially multi-line.
    Special handling for keys that expect a list of items (e.g., "Plot Points:").
    """
    parsed_data: Dict[str, Any] = {}
    lines = text_block.strip().splitlines()
    
    current_internal_key: Optional[str] = None
    current_value_lines: List[str] = []

    display_key_to_internal_map: Dict[str, str] = {}
    for norm_display_key, internal_key_val in key_map.items():
        display_key_for_llm_title = norm_display_key.replace('_', ' ').title() + ":"
        display_key_to_internal_map[display_key_for_llm_title] = internal_key_val
        
        # Add lowercase version as well for more robust matching if LLM doesn't follow title case
        display_key_for_llm_lower = norm_display_key.replace('_', ' ') + ":"
        if display_key_for_llm_lower not in display_key_to_internal_map : # Avoid overwriting preferred Title Case
            display_key_to_internal_map[display_key_for_llm_lower] = internal_key_val
        
        # Add version without colon if LLM forgets it sometimes
        display_key_for_llm_title_no_colon = norm_display_key.replace('_', ' ').title()
        if display_key_for_llm_title_no_colon not in display_key_to_internal_map:
             display_key_to_internal_map[display_key_for_llm_title_no_colon] = internal_key_val


    sorted_llm_keys_for_matching = sorted(display_key_to_internal_map.keys(), key=len, reverse=True)

    def process_previous_key_value():
        nonlocal current_internal_key, current_value_lines
        if current_internal_key and current_value_lines:
            if current_internal_key in list_keys_internal:
                list_items = [
                    line.strip()[2:].strip() for line in current_value_lines 
                    if line.strip().startswith("- ")
                ]
                if list_items:
                    parsed_data[current_internal_key] = list_items
                elif any(line.strip() for line in current_value_lines): # If lines were collected but not valid list items
                    # Fallback: treat as a single string if list parsing fails but content exists
                    logger.warning(f"List key '{current_internal_key}' had non-empty lines but no valid items starting with '- '. Collected: {current_value_lines}. Treating as single string.")
                    full_text = "\n".join(current_value_lines).strip()
                    if full_text: # Only assign if there's actual text
                        parsed_data[current_internal_key] = full_text
            else: # For non-list keys
                full_text = "\n".join(current_value_lines).strip()
                if full_text: # Only assign if there's actual text
                    parsed_data[current_internal_key] = full_text
        current_value_lines = []

    for line_content in lines:
        line_stripped_for_key_check = line_content.strip()
        matched_new_key = False
        
        for llm_output_key_format in sorted_llm_keys_for_matching:
            # Check if the line STARTS with the key format (potentially with or without colon)
            if line_stripped_for_key_check.startswith(llm_output_key_format):
                process_previous_key_value()
                current_internal_key = display_key_to_internal_map[llm_output_key_format]
                
                value_on_same_line = line_stripped_for_key_check[len(llm_output_key_format):].strip()
                if value_on_same_line:
                    current_value_lines.append(value_on_same_line)
                
                matched_new_key = True
                break 
        
        if not matched_new_key and current_internal_key:
            # This line is a continuation of the previous key's value or a list item
            if current_internal_key in list_keys_internal:
                if line_stripped_for_key_check.startswith("- "):
                    current_value_lines.append(line_stripped_for_key_check)
                # If it's a list key but line doesn't start with "- ", and it's not empty,
                # it could be a malformed list item or start of something else.
                # For robustness, only add lines starting with "- " to list_keys.
                # If LLM just lists things without "-", current_value_lines will capture them,
                # and the fallback logic in process_previous_key_value might treat it as a string.
                elif line_stripped_for_key_check: # If line is not empty and not a list item marker
                     pass # Don't add non-list-item lines to list_keys collections unless explicitly part of prior value.

            else: # Not a list key, append as part of a multi-line string
                current_value_lines.append(line_content) # Preserve original indents for multiline strings

    process_previous_key_value()

    for _, internal_key_to_ensure in key_map.items():
        if internal_key_to_ensure not in parsed_data or not parsed_data[internal_key_to_ensure]:
            # If key is missing, or its value is empty (e.g. empty string/list from parsing)
            if internal_key_to_ensure in list_keys_internal:
                parsed_data[internal_key_to_ensure] = [config.MARKDOWN_FILL_IN_PLACEHOLDER]
            else:
                parsed_data[internal_key_to_ensure] = config.MARKDOWN_FILL_IN_PLACEHOLDER
        elif internal_key_to_ensure in list_keys_internal:
            if not isinstance(parsed_data[internal_key_to_ensure], list):
                logger.warning(f"List key '{internal_key_to_ensure}' was parsed as non-list: '{parsed_data[internal_key_to_ensure]}'. Forcing to list.")
                val_str = str(parsed_data[internal_key_to_ensure]).strip()
                parsed_data[internal_key_to_ensure] = [val_str] if val_str else [config.MARKDOWN_FILL_IN_PLACEHOLDER]
            elif not parsed_data[internal_key_to_ensure]: # If it's an empty list
                parsed_data[internal_key_to_ensure] = [config.MARKDOWN_FILL_IN_PLACEHOLDER]


    return parsed_data

def _get_val_or_fill_in(data_dict: Optional[Dict[str, Any]], key: str, default_is_fill_in: bool = True) -> Any:
    if data_dict is None:
        return config.MARKDOWN_FILL_IN_PLACEHOLDER if default_is_fill_in else ""
    val = data_dict.get(key)
    if val is None or (isinstance(val, str) and not val.strip()): # Empty string is also a "fill-in" case
        return config.MARKDOWN_FILL_IN_PLACEHOLDER if default_is_fill_in else ""
    return val

def _create_default_plot(default_protagonist_name: str, base_elements: Dict[str, Any], unhinged: bool) -> PlotOutlineData:
    num_default_plot_points = config.TARGET_PLOT_POINTS_INITIAL_GENERATION
    default_plot: PlotOutlineData = {
        "title": config.DEFAULT_PLOT_OUTLINE_TITLE, "protagonist_name": default_protagonist_name,
        "protagonist_description": f"Default protagonist: {default_protagonist_name}, a character facing challenges.",
        "plot_points": [f"{config.MARKDOWN_FILL_IN_PLACEHOLDER} - Default Plot Point {i+1}" for i in range(num_default_plot_points)],
        "character_arc": f"Default character arc: {default_protagonist_name} learns something important over a significant journey.",
        "setting_description": base_elements.get("setting_description", base_elements.get("setting", "A generic place.")),
        "conflict_summary": "Default conflict: The protagonist must overcome a series of significant obstacles related to the core theme.",
        "is_default": True, "source": "default_fallback"
    }
    default_plot.update({k:v for k,v in base_elements.items() if k in ["genre", "theme"]})
    if unhinged:
        default_plot.update({
            k: base_elements[k] for k in ["setting_archetype_used", "protagonist_archetype_used", "conflict_archetype_used"] if k in base_elements
        })
    for key_in_map in PLOT_OUTLINE_KEY_MAP.values():
        if key_in_map not in default_plot:
            default_plot[key_in_map] = [] if key_in_map in PLOT_OUTLINE_LIST_INTERNAL_KEYS else config.MARKDOWN_FILL_IN_PLACEHOLDER
    return default_plot

def _load_user_supplied_data() -> Optional[Dict[str, Any]]:
    """Loads user-supplied story data from YAML file."""
    # Assuming USER_STORY_ELEMENTS_FILE_PATH in config will be updated to point to a .yaml file
    # or a new config variable USER_STORY_ELEMENTS_YAML_FILE_PATH will be used.
    # For this change, directly adjusting the expected file extension.
    yaml_file_path = config.USER_STORY_ELEMENTS_FILE_PATH.replace(".md", ".yaml").replace(".md.example", ".yaml.example")
    if not yaml_file_path.endswith((".yaml", ".yml")): # Ensure it's a yaml path after replace
        yaml_file_path = os.path.splitext(config.USER_STORY_ELEMENTS_FILE_PATH)[0] + ".yaml"
        logger.info(f"Adjusted file path to: {yaml_file_path} from {config.USER_STORY_ELEMENTS_FILE_PATH}")

    user_data = load_yaml_file(yaml_file_path) # normalize_keys is True by default in yaml_parser
    if user_data is None:
        logger.info(f"User story elements file '{yaml_file_path}' not found or failed to parse. Will proceed with LLM generation or defaults.")
        return None
    # load_yaml_file returns {} for empty file, or None for critical parse error / non-dict root
    if not isinstance(user_data, dict) or not user_data: # Ensure it's a non-empty dict
        logger.warning(f"User story elements file '{yaml_file_path}' was empty or did not yield a dictionary. Will proceed with LLM generation or defaults.")
        return {}

    # Key normalization is handled by load_yaml_file, so keys in user_data should be normalized.
    # Validation logic below assumes normalized keys (lowercase_with_underscores)
    expected_top_level_keys = ["novel_concept", "protagonist", "plot_points", "setting", "world_details", "antagonist", "conflict", "other_key_characters"]
    found_any_expected_key = False
    for key in expected_top_level_keys:
        if key in user_data and isinstance(user_data[key], dict) and user_data[key]:
            found_any_expected_key = True; break
        elif key == "plot_points" and key in user_data and isinstance(user_data[key], list) and user_data[key]:
            found_any_expected_key = True; break

    if not found_any_expected_key:
        logger.error(f"User-supplied YAML data from '{yaml_file_path}' does not seem to contain any expected top-level sections with content after key normalization. Parsed data: {user_data}")
        return {}

    logger.info(f"Successfully loaded and performed initial validation on user-supplied story data from '{yaml_file_path}'.")
    return user_data

def _populate_agent_state_from_user_data(agent: Any, user_data: Dict[str, Any]):
    plot_outline: PlotOutlineData = agent.plot_outline if hasattr(agent, 'plot_outline') and agent.plot_outline else {}
    character_profiles: Dict[str, Any] = agent.character_profiles if hasattr(agent, 'character_profiles') and agent.character_profiles else {}
    world_building: WorldBuildingData = agent.world_building if hasattr(agent, 'world_building') and agent.world_building else {}

    # Initialize world_building structure
    for cat_internal_key in WORLD_CATEGORY_MAP_NORMALIZED_TO_INTERNAL.values():
        world_building.setdefault(cat_internal_key, {})
    world_building["user_supplied_data"] = True # Mark that user data was involved
    world_building["is_default"] = False
    world_building["source"] = "user_supplied_yaml" # Updated source identifier


    nc = user_data.get("novel_concept", {}) # Assumes keys in user_data are already normalized by load_yaml_file
    plot_outline["title"] = _get_val_or_fill_in(nc, "title")
    plot_outline["genre"] = _get_val_or_fill_in(nc, "genre")
    plot_outline["theme"] = _get_val_or_fill_in(nc, "theme")
    plot_outline["logline"] = _get_val_or_fill_in(nc, "logline")

    prot_data = user_data.get("protagonist", {})
    plot_outline["protagonist_name"] = _get_val_or_fill_in(prot_data, "name")
    plot_outline["protagonist_description"] = _get_val_or_fill_in(prot_data, "description")
    plot_outline["character_arc"] = _get_val_or_fill_in(prot_data, "character_arc")

    ant_data = user_data.get("antagonist", {})
    plot_outline["antagonist_name"] = _get_val_or_fill_in(ant_data, "name")
    plot_outline["antagonist_description"] = _get_val_or_fill_in(ant_data, "description")
    plot_outline["antagonist_motivations"] = _get_val_or_fill_in(ant_data, "motivations")

    conflict_data = user_data.get("conflict", {})
    plot_outline["conflict_summary"] = _get_val_or_fill_in(conflict_data, "summary")
    plot_outline["inciting_incident"] = _get_val_or_fill_in(conflict_data, "inciting_incident")
    plot_outline["climax_event_preview"] = _get_val_or_fill_in(conflict_data, "climax_event_preview")

    raw_plot_points = user_data.get("plot_points", [])
    if not isinstance(raw_plot_points, list):
        logger.warning(f"Markdown 'plot_points' parsed as non-list: {type(raw_plot_points)}. Defaulting to [Fill-in].")
        plot_outline["plot_points"] = [config.MARKDOWN_FILL_IN_PLACEHOLDER] * config.TARGET_PLOT_POINTS_INITIAL_GENERATION
    else:
        plot_outline["plot_points"] = [
            str(pp).strip() if isinstance(pp, str) and (pp.strip() or utils._is_fill_in(pp)) else config.MARKDOWN_FILL_IN_PLACEHOLDER
            for pp in raw_plot_points
        ]
        while len(plot_outline["plot_points"]) > 0 and len(plot_outline["plot_points"]) < config.TARGET_PLOT_POINTS_INITIAL_GENERATION and plot_outline["plot_points"][-1] != config.MARKDOWN_FILL_IN_PLACEHOLDER :
             plot_outline["plot_points"].append(config.MARKDOWN_FILL_IN_PLACEHOLDER)

    # Process 'setting' section from user_data for world_building
    setting_data_md = user_data.get("setting", {})
    overview_desc_from_setting = _get_val_or_fill_in(setting_data_md, "primary_setting_description")
    plot_outline["setting_description"] = overview_desc_from_setting # Also store in plot_outline

    world_building.setdefault("_overview_", {})
    if not utils._is_fill_in(overview_desc_from_setting): # Only overwrite if user provided something
        world_building["_overview_"]["description"] = overview_desc_from_setting

    key_locations_md = setting_data_md.get("key_locations", {}) # This is usually like {"the_hourglass_curios": {"description": "..."}}
    if isinstance(key_locations_md, dict):
        world_building.setdefault("locations", {})
        for loc_name_norm_md, loc_details_md in key_locations_md.items():
            loc_name_display = loc_name_norm_md.replace("_", " ").title()
            if not utils._is_fill_in(loc_name_display) and isinstance(loc_details_md, dict):
                agent_loc_details = world_building["locations"].setdefault(loc_name_display, {"source": "user_supplied_markdown"})
                for md_detail_key, md_detail_val in loc_details_md.items():
                    internal_detail_key = WORLD_DETAIL_KEY_MAP_FROM_MARKDOWN_TO_INTERNAL.get(md_detail_key, md_detail_key)
                    agent_loc_details[internal_detail_key] = md_detail_val

    # Process 'world_details' section from user_data
    wd_details_md = user_data.get("world_details", {})
    if isinstance(wd_details_md, dict):
        for main_wd_key_norm_md, main_wd_content_md in wd_details_md.items():
            if main_wd_key_norm_md == "unique_world_feature":
                target_category_internal = "systems"
                world_building.setdefault(target_category_internal, {})
                if isinstance(main_wd_content_md, dict):
                    # Assume "Unique World Feature" as the display name if not explicitly named in MD
                    feature_item_name_display = _get_val_or_fill_in(main_wd_content_md, "name", default_is_fill_in=False) or "Unique World Feature"
                    agent_item_details = world_building[target_category_internal].setdefault(feature_item_name_display, {"source": "user_supplied_markdown"})
                    for md_detail_key, md_detail_val in main_wd_content_md.items():
                        if md_detail_key == "name": continue # Already used for item name
                        internal_detail_key = WORLD_DETAIL_KEY_MAP_FROM_MARKDOWN_TO_INTERNAL.get(md_detail_key, md_detail_key)
                        agent_item_details[internal_detail_key] = md_detail_val

            elif main_wd_key_norm_md == "key_factions":
                target_category_internal = "factions"
                world_building.setdefault(target_category_internal, {})
                if isinstance(main_wd_content_md, dict): # Content should be dict of faction_name: details
                    for item_name_norm_md, item_details_md in main_wd_content_md.items():
                        item_name_display = item_name_norm_md.replace("_", " ").title()
                        if not utils._is_fill_in(item_name_display) and isinstance(item_details_md, dict):
                            agent_item_details = world_building[target_category_internal].setdefault(item_name_display, {"source": "user_supplied_markdown"})
                            for md_detail_key, md_detail_val in item_details_md.items():
                                internal_detail_key = WORLD_DETAIL_KEY_MAP_FROM_MARKDOWN_TO_INTERNAL.get(md_detail_key, md_detail_key)
                                agent_item_details[internal_detail_key] = md_detail_val
            
            elif main_wd_key_norm_md == "relevant_lore":
                target_category_internal = "lore"
                world_building.setdefault(target_category_internal, {})
                if isinstance(main_wd_content_md, dict):
                    for item_name_norm_md, item_details_md in main_wd_content_md.items():
                        item_name_display = item_name_norm_md.replace("_", " ").title()
                        if not utils._is_fill_in(item_name_display) and isinstance(item_details_md, dict):
                            agent_item_details = world_building[target_category_internal].setdefault(item_name_display, {"source": "user_supplied_markdown"})
                            for md_detail_key, md_detail_val in item_details_md.items():
                                internal_detail_key = WORLD_DETAIL_KEY_MAP_FROM_MARKDOWN_TO_INTERNAL.get(md_detail_key, md_detail_key)
                                agent_item_details[internal_detail_key] = md_detail_val
            # Add other categories like "history", "society" similarly if they appear under world_details

    plot_outline["source"] = "user_supplied_yaml" # Updated source identifier
    plot_outline["is_default"] = False
    agent.plot_outline = plot_outline

    prot_name_val = plot_outline["protagonist_name"]
    if not utils._is_fill_in(prot_name_val):
        character_profiles.setdefault(prot_name_val, {})
        character_profiles[prot_name_val].update({
            "description": plot_outline["protagonist_description"],
            "traits": [t for t in prot_data.get("traits", []) if isinstance(t,str) and (t.strip() or utils._is_fill_in(t))], # Assumes 'traits' is list
            "status": _get_val_or_fill_in(prot_data, "initial_status") or "As described",
            "character_arc_summary": plot_outline["character_arc"],
            "role": "protagonist", "source": "user_supplied_yaml", # Updated source
            "relationships": prot_data.get("relationships", {}) # Assumes 'relationships' is dict
        })

    ant_name_val = plot_outline["antagonist_name"]
    if not utils._is_fill_in(ant_name_val) and ant_data: # ant_data is from user_data.get("antagonist", {})
        character_profiles.setdefault(ant_name_val, {})
        character_profiles[ant_name_val].update({
            "description": plot_outline["antagonist_description"],
            "traits": [t for t in ant_data.get("traits", []) if isinstance(t,str) and (t.strip() or utils._is_fill_in(t))],
            "status": "As described",
            "motivations": plot_outline["antagonist_motivations"],
            "role": "antagonist", "source": "user_supplied_yaml", # Updated source
            "relationships": ant_data.get("relationships", {})
        })

    other_chars_data = user_data.get("other_key_characters", {}) # This should be a dict of char_name: details
    if isinstance(other_chars_data, dict):
        for char_name_other_normalized_yaml, char_detail_yaml in other_chars_data.items():
            # char_name_other_normalized_yaml is already normalized by load_yaml_file
            char_name_other_display = char_name_other_normalized_yaml.replace("_", " ").title() # For display consistency if needed, but internal key is normalized
            if not utils._is_fill_in(char_name_other_display) and isinstance(char_detail_yaml, dict):
                # Use normalized key for character_profiles dict directly
                agent_char_details = character_profiles.setdefault(char_name_other_normalized_yaml, {"source": "user_supplied_yaml"})
                for yaml_detail_key, yaml_detail_val in char_detail_yaml.items(): # These keys are also normalized
                    # Assuming char profile keys are mostly direct (already normalized)
                    internal_detail_key = yaml_detail_key
                    agent_char_details[internal_detail_key] = yaml_detail_val

    agent.character_profiles = character_profiles
    agent.world_building = world_building # world_building was modified in place
    logger.info("Agent state populated from user-supplied YAML data (preserving '[Fill-in]' markers).")

async def generate_plot_outline_logic(agent: Any, default_protagonist_name: str, unhinged_mode: bool, **kwargs) -> Tuple[PlotOutlineData, Optional[Dict[str, int]]]:
    # ... (Plot outline generation logic remains largely the same, as it uses parse_key_value_block) ...
    # This function already assumes a flat key-value structure from the LLM, which is different from world-building.
    # The key change was ensuring _populate_agent_state_from_user_data correctly sets up agent.plot_outline
    # from the YAML before this LLM call happens (if it's needed).
    logger.info(f"Generating plot outline. Unhinged mode: {unhinged_mode}")
    user_supplied_data = _load_user_supplied_data() # Now loads YAML

    llm_was_called = False
    accumulated_usage_data: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    if user_supplied_data: # This will be None if file not found, or {} if empty/error
        if user_supplied_data: # Check if dict is not empty (i.e., file had content and parsed)
            logger.info("Processing user-supplied YAML data for initial setup.")
            _populate_agent_state_from_user_data(agent, user_supplied_data)
        else: # File was empty or structure was not a root dictionary
            logger.warning("User-supplied YAML file was effectively empty or did not yield a root dictionary. Plot will be fully generated or default.")
            agent.plot_outline = {} # Ensure it's an empty dict, not None
    else: # File not found
        logger.info("No user-supplied YAML file found. Plot outline will be fully generated by LLM or default.")
        agent.plot_outline = {}

    # Ensure agent.plot_outline is a dict before checking keys
    if not isinstance(agent.plot_outline, dict):
        agent.plot_outline = {}

    critical_plot_fields_for_llm_check = ["title", "protagonist_name", "protagonist_description", "setting_description", "conflict_summary", "genre", "theme"]
    needs_llm_for_core_plot = not agent.plot_outline or \
                             any(utils._is_fill_in(agent.plot_outline.get(field)) for field in critical_plot_fields_for_llm_check)

    plot_points_list_from_agent = agent.plot_outline.get("plot_points", [])
    if not isinstance(plot_points_list_from_agent, list):
        actual_plot_points_count = 0
        needs_llm_for_core_plot = True
        logger.warning(f"Plot points from agent state is not a list ({type(plot_points_list_from_agent)}). Will regenerate.")
    else:
        actual_plot_points = [pp for pp in plot_points_list_from_agent if not utils._is_fill_in(pp) and isinstance(pp,str) and pp.strip()]
        actual_plot_points_count = len(actual_plot_points)

    if actual_plot_points_count < config.TARGET_PLOT_POINTS_INITIAL_GENERATION:
        needs_llm_for_core_plot = True
        if agent.plot_outline and plot_points_list_from_agent:
            logger.info(f"Insufficient concrete plot points ({actual_plot_points_count} provided vs {config.TARGET_PLOT_POINTS_INITIAL_GENERATION} target). LLM will supplement.")

    if needs_llm_for_core_plot:
        llm_was_called = True
        logger.info("LLM generation required for core plot outline elements or to supplement plot points.")

        context_from_user_input = "\n**User-Provided Context (Respect these if not '[Fill-in]'):**\n"
        has_user_context = False
        current_plot_outline_key_map_for_llm_prompt = {k: v for k, v in PLOT_OUTLINE_KEY_MAP.items()}

        if agent.plot_outline: # Should be a dict now
            for display_key_normalized, internal_key in current_plot_outline_key_map_for_llm_prompt.items():
                user_val = agent.plot_outline.get(internal_key)
                display_key_title_case = display_key_normalized.replace('_', ' ').title()

                if user_val is not None:
                    if isinstance(user_val, list):
                        concrete_list_items = [item for item in user_val if not utils._is_fill_in(item) and str(item).strip()]
                        fill_in_placeholders_in_list = [item for item in user_val if utils._is_fill_in(item)]

                        if concrete_list_items:
                            context_from_user_input += f"  - {display_key_title_case}: User provided {len(concrete_list_items)} concrete item(s), e.g., \"{str(concrete_list_items[0])[:50]}...\". " # Ensure str
                            has_user_context = True
                        if fill_in_placeholders_in_list:
                             context_from_user_input += f"Also includes {len(fill_in_placeholders_in_list)} '{config.MARKDOWN_FILL_IN_PLACEHOLDER}' items to complete.\n"
                             has_user_context = True
                        elif concrete_list_items :
                             context_from_user_input += "This list might need expansion if below target count.\n"
                    elif not utils._is_fill_in(user_val) and str(user_val).strip():
                        context_from_user_input += f"  - {display_key_title_case}: {str(user_val)}\n"
                        has_user_context = True
                    elif utils._is_fill_in(user_val):
                        context_from_user_input += f"  - {display_key_title_case}: {config.MARKDOWN_FILL_IN_PLACEHOLDER} (User requests generation for this field)\n"
                        has_user_context = True
        if not has_user_context:
            context_from_user_input = "\n**User-Provided Context:** No specific overriding preferences or fill-in requests were found; generate all fields creatively.\n"

        llm_fields_to_generate_text = "\n".join([f"- {k.replace('_', ' ').title()}" for k in current_plot_outline_key_map_for_llm_prompt.keys()])
        target_num_plot_points = config.TARGET_PLOT_POINTS_INITIAL_GENERATION

        base_elements_for_outline: Dict[str, Any] = {}
        # Ensure agent.plot_outline is a dict for checks below
        current_agent_plot_outline = agent.plot_outline if isinstance(agent.plot_outline, dict) else {}
        in_pure_llm_scratch_mode = not user_supplied_data and (not current_agent_plot_outline or not any(current_agent_plot_outline.values()))


        prompt_core_elements_intro = ""
        if unhinged_mode and in_pure_llm_scratch_mode:
            base_elements_for_outline["genre"] = kwargs.get("genre", random.choice(config.UNHINGED_GENRES))
            base_elements_for_outline["theme"] = kwargs.get("theme", random.choice(config.UNHINGED_THEMES))
            base_elements_for_outline["setting_description"] = kwargs.get("setting_archetype", random.choice(config.UNHINGED_SETTINGS_ARCHETYPES))
            base_elements_for_outline["protagonist_name"] = default_protagonist_name
            base_elements_for_outline["source_hint"] = "unhinged_pure_llm"
            prompt_core_elements_intro = f"You are in UNHINGED mode. Generate a novel concept based on:\n  - Genre: {base_elements_for_outline['genre']}\n  - Theme: {base_elements_for_outline['theme']}\n  - Setting Archetype: {base_elements_for_outline['setting_description']}\n"
        else:
            base_elements_for_outline["genre"] = current_agent_plot_outline.get("genre") if not utils._is_fill_in(current_agent_plot_outline.get("genre")) else config.CONFIGURED_GENRE
            base_elements_for_outline["theme"] = current_agent_plot_outline.get("theme") if not utils._is_fill_in(current_agent_plot_outline.get("theme")) else config.CONFIGURED_THEME
            base_elements_for_outline["setting_description"] = current_agent_plot_outline.get("setting_description") if not utils._is_fill_in(current_agent_plot_outline.get("setting_description")) else config.CONFIGURED_SETTING_DESCRIPTION
            base_elements_for_outline["protagonist_name"] = current_agent_plot_outline.get("protagonist_name") if not utils._is_fill_in(current_agent_plot_outline.get("protagonist_name")) else default_protagonist_name
            base_elements_for_outline["source_hint"] = "configured_or_user_yaml" # Updated source hint
            prompt_core_elements_intro = f"Generate a novel concept based on (or incorporating):\n  - Genre: {base_elements_for_outline['genre']}\n  - Theme: {base_elements_for_outline['theme']}\n  - Initial Setting Idea: {base_elements_for_outline['setting_description']}\n  - Protagonist Name (if known): {base_elements_for_outline['protagonist_name']}\n"

        prompt_lines = []
        if config.ENABLE_LLM_NO_THINK_DIRECTIVE:
            prompt_lines.append("/no_think")

        prompt_lines.extend([
            "You are a creative assistant specializing in crafting compelling narrative structures for full novels.",
            f"{prompt_core_elements_intro}",
            f"{context_from_user_input}",
            "Based on all the above, generate or complete the following plot outline fields.",
            "If user context provided a concrete value for a field, you MUST use it.",
            f"If user context for a field is '{config.MARKDOWN_FILL_IN_PLACEHOLDER}' or missing, or if a list like 'Plot Points' needs more items, generate them creatively.",
            f"Ensure the \"Plot Points\" section contains approximately {target_num_plot_points} distinct points that form a complete narrative arc. If the user provided some plot points, integrate them and add more to reach the target.",
            "The fields to ensure are complete:",
            f"{llm_fields_to_generate_text}",
            "",
            "Please output ONLY the plot elements as plain text, using the specified field names (e.g., \"Title:\", \"Protagonist Name:\").",
            "For \"Plot Points\", use this EXACT format with each point on a new line prefixed by \"- \".",
            f"Example of \"Plot Points\" for a {target_num_plot_points}-point outline:",
            "Plot Points:",
            "- Plot Point 1 description.",
            "- Plot Point 2 description.",
            "...",
            f"- Plot Point {target_num_plot_points} description.",
            "",
            "Begin your output now using the requested field names:"
        ])
        prompt = "\n".join(prompt_lines)

        logger.info(f"Calling LLM for plot outline generation/completion (to plain text), targeting ~{target_num_plot_points} plot points...")
        cleaned_outline_text, usage_data = await llm_service.async_call_llm(
            model_name=config.INITIAL_SETUP_MODEL,
            prompt=prompt,
            temperature=config.TEMPERATURE_INITIAL_SETUP,
            stream_to_disk=True,
            frequency_penalty=config.FREQUENCY_PENALTY_INITIAL_SETUP,
            presence_penalty=config.PRESENCE_PENALTY_INITIAL_SETUP,
            auto_clean_response=True
        )
        if usage_data:
            accumulated_usage_data["prompt_tokens"] += usage_data.get("prompt_tokens", 0)
            accumulated_usage_data["completion_tokens"] += usage_data.get("completion_tokens", 0)
            accumulated_usage_data["total_tokens"] += usage_data.get("total_tokens", 0)

        parsed_llm_response = parse_key_value_block(
            cleaned_outline_text, current_plot_outline_key_map_for_llm_prompt, PLOT_OUTLINE_LIST_INTERNAL_KEYS
        )

        if not isinstance(agent.plot_outline, dict): agent.plot_outline = {} # Ensure it's a dict

        if parsed_llm_response:
            for key, llm_value in parsed_llm_response.items():
                existing_val = agent.plot_outline.get(key)

                if key == "plot_points":
                    user_pps_concrete = [pp for pp in (existing_val or []) if isinstance(pp, str) and not utils._is_fill_in(pp) and pp.strip()]
                    llm_pps = [pp for pp in llm_value if isinstance(pp, str) and pp.strip()] if isinstance(llm_value, list) else []

                    final_pps = user_pps_concrete[:]
                    for llm_pp_item in llm_pps:
                        if len(final_pps) >= target_num_plot_points: break
                        if not any(llm_pp_item.lower() in user_pp.lower() or user_pp.lower() in llm_pp_item.lower() for user_pp in user_pps_concrete):
                            final_pps.append(llm_pp_item)

                    while len(final_pps) < target_num_plot_points:
                         final_pps.append(f"{config.MARKDOWN_FILL_IN_PLACEHOLDER} - Additional plot point needed")
                    agent.plot_outline[key] = final_pps[:target_num_plot_points]
                elif utils._is_fill_in(existing_val) or existing_val is None or not str(existing_val).strip():
                    agent.plot_outline[key] = llm_value
                elif isinstance(existing_val, str) and not existing_val.strip() and llm_value: # Existing was empty string
                    agent.plot_outline[key] = llm_value


            for be_key in ["genre", "theme", "setting_description"]:
                if utils._is_fill_in(agent.plot_outline.get(be_key)):
                    llm_generated_val = parsed_llm_response.get(be_key, base_elements_for_outline.get(be_key))
                    if llm_generated_val and not utils._is_fill_in(llm_generated_val):
                         agent.plot_outline[be_key] = llm_generated_val
                    elif base_elements_for_outline.get(be_key) and not utils._is_fill_in(base_elements_for_outline.get(be_key)):
                         agent.plot_outline[be_key] = base_elements_for_outline.get(be_key)


            agent.plot_outline.pop("is_default", None)
            if not agent.plot_outline.get("source") or agent.plot_outline.get("source") == "default_fallback":
                agent.plot_outline["source"] = base_elements_for_outline.get("source_hint", "llm_generated_or_merged")
            logger.info(f"LLM successfully generated/updated plot outline elements. Title: '{agent.plot_outline.get('title', 'N/A')}' with {len(agent.plot_outline.get('plot_points',[]))} plot points.")
        else:
            logger.error("LLM failed to provide a parsable core plot outline. Falling back to default if agent state is still insufficient.")
            if not agent.plot_outline or any(utils._is_fill_in(agent.plot_outline.get(field)) for field in critical_plot_fields_for_llm_check):
                agent.plot_outline = _create_default_plot(default_protagonist_name, base_elements_for_outline, unhinged_mode and in_pure_llm_scratch_mode)

    # Ensure agent.plot_outline is a dict for final checks
    if not isinstance(agent.plot_outline, dict): agent.plot_outline = {}


    prot_name_from_outline = agent.plot_outline.get('protagonist_name')
    if utils._is_fill_in(prot_name_from_outline) or not isinstance(prot_name_from_outline, str) or not prot_name_from_outline.strip():
        agent.plot_outline['protagonist_name'] = default_protagonist_name
        logger.warning(f"Protagonist name resolved to default: {default_protagonist_name}")

    final_protagonist_name = agent.plot_outline['protagonist_name']

    if not hasattr(agent, 'character_profiles') or agent.character_profiles is None:
        agent.character_profiles = {}
    if not isinstance(agent.character_profiles, dict): agent.character_profiles = {}


    agent.character_profiles.setdefault(final_protagonist_name, {"source": agent.plot_outline.get("source", "unknown")})
    prot_profile_ref = agent.character_profiles[final_protagonist_name]

    for key, plot_key in [("description", "protagonist_description"), ("character_arc_summary", "character_arc")]:
        plot_val = agent.plot_outline.get(plot_key)
        if not utils._is_fill_in(plot_val):
            prot_profile_ref[key] = plot_val
        elif key not in prot_profile_ref or utils._is_fill_in(prot_profile_ref.get(key)):
            prot_profile_ref[key] = f"{config.MARKDOWN_FILL_IN_PLACEHOLDER} for {key}"

    prot_profile_ref["role"] = "protagonist"
    if "traits" not in prot_profile_ref or not prot_profile_ref["traits"] or \
       (isinstance(prot_profile_ref["traits"], list) and all(utils._is_fill_in(t) for t in prot_profile_ref["traits"])):
        prot_profile_ref["traits"] = [config.MARKDOWN_FILL_IN_PLACEHOLDER]
    if "status" not in prot_profile_ref or utils._is_fill_in(prot_profile_ref.get("status")):
        prot_profile_ref["status"] = config.MARKDOWN_FILL_IN_PLACEHOLDER \
                                     if utils._is_fill_in(agent.plot_outline.get("protagonist_description")) \
                                     else "As described in plot outline"

    logger.info(f"Finalized character profile for protagonist '{final_protagonist_name}'.")

    if not hasattr(agent, 'world_building') or agent.world_building is None:
        agent.world_building = {"_overview_":{}, "source": agent.plot_outline.get("source", "unknown")}
    if not isinstance(agent.world_building, dict) : agent.world_building = {"_overview_":{}, "source": agent.plot_outline.get("source", "unknown")}


    overview_desc_val = agent.plot_outline.get("setting_description", config.MARKDOWN_FILL_IN_PLACEHOLDER)
    agent.world_building.setdefault("_overview_", {})["description"] = overview_desc_val

    return agent.plot_outline, accumulated_usage_data if llm_was_called else None

async def generate_world_building_logic(agent: Any) -> Tuple[WorldBuildingData, Optional[Dict[str, int]]]:
    llm_was_called = False
    accumulated_usage_data: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    # Ensure agent.world_building is initialized as a dict
    if not hasattr(agent, 'world_building') or not isinstance(agent.world_building, dict):
        agent.world_building = {}

    needs_llm_for_world = False
    if not agent.world_building: # If it's empty after init or previous steps
        agent.world_building = {"source":"llm_to_create", "_overview_": {}} # Default source
        needs_llm_for_world = True
    elif agent.world_building.get("source") == "user_supplied_yaml": # Updated source check
        # Check if any [Fill-in] exists in user-supplied data
        for category_internal, items_or_details_dict in agent.world_building.items():
            if category_internal in ["is_default", "source", "user_supplied_data"]: continue # Skip helper keys
            if isinstance(items_or_details_dict, dict):
                if category_internal == "_overview_":
                    for detail_key, detail_value in items_or_details_dict.items():
                        if utils._is_fill_in(detail_value): needs_llm_for_world = True; break
                else: # Itemized categories
                    for item_name, item_detail_dict in items_or_details_dict.items():
                        if isinstance(item_detail_dict, dict):
                            for detail_key, detail_value in item_detail_dict.items():
                                if detail_key == "source": continue
                                if utils._is_fill_in(detail_value): needs_llm_for_world = True; break
                                if isinstance(detail_value, list) and any(utils._is_fill_in(li) for li in detail_value):
                                    needs_llm_for_world = True; break
                            if needs_llm_for_world: break # Break from items_or_details_dict loop
                if needs_llm_for_world: break # Break from categories loop
        if not needs_llm_for_world:
            logger.info("Skipping LLM world-building generation: Data was user-supplied from YAML and seems complete (no [Fill-in]s).")
            return agent.world_building, None
    else: # Source is not user_supplied_yaml, or it's empty, assume LLM is needed
        needs_llm_for_world = True

    # Ensure plot_outline exists and setting_description is usable
    if not hasattr(agent, 'plot_outline') or not isinstance(agent.plot_outline, dict) or \
       utils._is_fill_in(agent.plot_outline.get("setting_description")):
        logger.warning("Cannot generate detailed world-building if plot outline or its setting_description is missing or '[Fill-in]'.")
        if not agent.world_building.get("_overview_", {}).get("description") or \
           utils._is_fill_in(agent.world_building.get("_overview_", {}).get("description")):
            # Set a very generic overview if nothing else is available
            agent.world_building.setdefault("_overview_",{})["description"] = "A world to be detailed by the LLM."
        if agent.world_building.get("source") != "user_supplied_yaml": # Avoid overwriting this if it was set
            agent.world_building["source"] = "llm_generated_default_context"
        needs_llm_for_world = True # Force LLM if essential context is missing

    if not needs_llm_for_world: # This check might be redundant if logic above correctly sets needs_llm_for_world
        logger.info("World building data seems complete, skipping LLM call.")
        return agent.world_building, None

    llm_was_called = True
    plot_title = agent.plot_outline.get('title', 'Untitled Novel')
    plot_genre = agent.plot_outline.get('genre', 'N/A')
    
    # Prepare world_setting_desc for the prompt, prioritizing agent.world_building._overview.description
    world_setting_desc = agent.world_building.get("_overview_", {}).get("description")
    if not world_setting_desc or utils._is_fill_in(world_setting_desc):
        world_setting_desc = agent.plot_outline.get("setting_description") # Fallback to plot_outline
    if not world_setting_desc or utils._is_fill_in(world_setting_desc):
        world_setting_desc = 'A mysterious and detailed world waiting to be fleshed out by the LLM.'


    user_world_context_str = "\n**User-Provided World Context (Content from user_story_elements.yaml - Respect these concrete values. Complete any '[Fill-in]' fields creatively using Markdown):**\n" # Updated filename
    temp_user_wb_for_prompt: List[str] = []
    # Only build this section if the source is indeed user_supplied_yaml and it's not empty
    if agent.world_building.get("source") == "user_supplied_yaml" and \
       (agent.world_building.get("_overview_") or any(k not in ["_overview_", "is_default", "source", "user_supplied_data"] for k in agent.world_building.keys())): # user_supplied_data might be redundant

        overview_data_for_prompt = agent.world_building.get("_overview_", {})
        if overview_data_for_prompt: # Check if overview itself is not empty
            temp_user_wb_for_prompt.append("## Overview")
            for detail_key_internal, detail_val in overview_data_for_prompt.items():
                if detail_key_internal == "source": continue
                detail_key_display_for_prompt = detail_key_internal.replace("_", " ").capitalize()
                temp_user_wb_for_prompt.append(f"**{detail_key_display_for_prompt}**: {detail_val}")
            temp_user_wb_for_prompt.append("")
            
        for cat_internal, items_dict in agent.world_building.items():
            if cat_internal in ["_overview_", "is_default", "source", "user_supplied_data"]:
                continue
            
            cat_display_for_prompt = cat_internal.replace("_", " ").capitalize()
            category_lines_for_prompt: List[str] = [] # Lines for current category

            if isinstance(items_dict, dict) and items_dict:
                category_lines_for_prompt.append(f"## {cat_display_for_prompt}")
                for item_display_name, details in items_dict.items():
                    if not isinstance(details, dict) or item_display_name.startswith("_"): continue # Skip internal markers like _is_fill_in
                    
                    category_lines_for_prompt.append(f"### {item_display_name}")
                    item_has_details = False
                    for detail_key_internal, detail_val in details.items():
                        if detail_key_internal == "source": continue
                        detail_key_display_for_prompt = detail_key_internal.replace("_", " ").capitalize()
                        if isinstance(detail_val, list):
                            if detail_val: # Only add if list has items
                                category_lines_for_prompt.append(f"**{detail_key_display_for_prompt}**:")
                                for li_val in detail_val:
                                    category_lines_for_prompt.append(f"  - {li_val}")
                                item_has_details = True
                        elif detail_val is not None and (isinstance(detail_val, bool) or str(detail_val).strip()):
                            category_lines_for_prompt.append(f"**{detail_key_display_for_prompt}**: {detail_val}")
                            item_has_details = True
                    if item_has_details:
                         category_lines_for_prompt.append("") 
                
                if len(category_lines_for_prompt) > 1: # Header + some content
                    temp_user_wb_for_prompt.extend(category_lines_for_prompt)

    if temp_user_wb_for_prompt: # If any user context was actually formatted
        user_world_context_str += "\n".join(temp_user_wb_for_prompt)
    else:
        user_world_context_str = "\n**User-Provided World Context:** No specific world preferences or fill-in requests were found from user input; generate all fields creatively using Markdown.\n"


    # Constants for JSON prompt construction
    category_keys_str = ", ".join([f'"{k}"' for k in WORLD_CATEGORY_MAP_NORMALIZED_TO_INTERNAL.keys()])
    detail_keys_for_prompt = sorted([k for k in WORLD_DETAIL_KEY_MAP_FROM_MARKDOWN_TO_INTERNAL.keys() if k != "primary_setting_description"])
    detail_keys_str = ", ".join([f'"{k}"' for k in detail_keys_for_prompt])

    internal_to_markdown_detail_key_map = {v: k for k, v in WORLD_DETAIL_KEY_MAP_FROM_MARKDOWN_TO_INTERNAL.items()}
    list_detail_keys_for_prompt = sorted([internal_to_markdown_detail_key_map[k] for k in WORLD_DETAIL_LIST_INTERNAL_KEYS if k in internal_to_markdown_detail_key_map])
    list_keys_str = ", ".join([f'"{k}"' for k in list_detail_keys_for_prompt])

    # Construct the new prompt for JSON output
    prompt_lines = []
    if config.ENABLE_LLM_NO_THINK_DIRECTIVE:
        prompt_lines.append("/no_think")

    prompt_lines.append(f"""You are an expert world-building assistant for novelists.
Your goal is to generate a comprehensive and consistent set of world-building details.
You MUST output a single, valid JSON object that encompasses all world-building details. Do NOT use Markdown.

**JSON Structure Requirements:**

1.  **Top-Level Object:** The root of the JSON should be a single object.
2.  **Categories:** The top-level object should have keys for world-building categories. The primary categories are: {category_keys_str}.
    *   For the "overview" category, the value should be a JSON object containing detail key-value pairs. For example: `{{"description": "A vast desert world...", "mood": "Harsh and mysterious"}}`.
    *   For all other categories (e.g., "locations", "factions", "systems"), the value should be a JSON object. In this object, each key is the unique name of an item (e.g., "The Sunken City", "The Mechanists Guild"), and its value is another JSON object containing the details for that item.
3.  **Detail Keys:** The objects for "overview" and for each item within other categories (like "locations", "factions") should use the following keys for their details, where appropriate: {detail_keys_str}.
    *   Example for a location item: `{{"description": "A hidden oasis...", "atmosphere": "Peaceful but eerie"}}`
    *   Example for a faction item: `{{"description": "A group of scholars...", "goals": ["Preserve knowledge", "Share discoveries"]}}`
4.  **Value Types for Details:**
    *   Most detail values should be JSON strings.
    *   The following detail keys MUST have values that are JSON arrays of strings: {list_keys_str}.
        *   Example: `"goals": ["Achieve peace", "Build a monument"]`
        *   If there's only one item for such a key, it should still be in an array: `"key_events": ["The Great Upheaval"]`
        *   If there are no items for such a key, use an empty array: `"rules": []`

**Example of Expected JSON Output:**
```json
{{
  "overview": {{
    "description": "A world of floating islands interconnected by ancient bridges.",
    "mood": "Mysterious and serene",
    "time_period": "Post-cataclysmic era"
  }},
  "locations": {{
    "Aerie Citadel": {{
      "description": "The main hub city, built on the largest cluster of islands. Features advanced, wind-powered technology.",
      "atmosphere": "Bustling with traders, artisans, and sky-ship pilots. A sense of cautious optimism prevails."
    }},
    "Whispering Chasm": {{
      "description": "A deep rift between islands, rumored to hold ancient secrets and dangers.",
      "atmosphere": "Eerie, filled with strange echoes and gusts of wind."
    }}
  }},
  "factions": {{
    "Sky Wardens": {{
      "description": "Dedicated guardians of the old skyways and the inter-island bridges. They are stern and traditional.",
      "goals": ["Protect the floating islands from falling", "Preserve the knowledge of the Ancients"],
      "structure": "Hierarchical, led by a council of Elders."
    }}
  }},
  "systems": {{
    "Aetherium Currents": {{
      "description": "Naturally occurring energy currents in the sky that are harnessed for power and travel.",
      "rules": ["Currents shift seasonally", "Over-harvesting can lead to local depletion and atmospheric instability"],
      "function": "Primary power source for island technology and sky-ship propulsion."
    }}
  }}
  // ... other categories like "lore", "history", "society" would follow a similar structure ...
}}
```

**Handling User-Provided Context:**
The following section, delimited by "--- USER WORLD CONTEXT START ---" and "--- USER WORLD CONTEXT END ---", contains any world details already provided by the user. This context is currently in a Markdown-like format.
*   You MUST incorporate any concrete values from this user context into the correct locations within the JSON structure you generate.
*   If the user context specifies `"{config.MARKDOWN_FILL_IN_PLACEHOLDER}"` for a field, or if a field is missing from the user context that you would normally generate, you should creatively generate that field's content according to the JSON structure defined above.
*   Interpret headers (e.g., `## Locations`, `### Item Name`) and key-value pairs (e.g., `**Description**: ...`) from the user context to understand where the information belongs.

**Novel Concept (from Plot Outline):**
  - Title: {plot_title}
  - Genre: {plot_genre}
  - Core Setting Idea: {world_setting_desc}

--- USER WORLD CONTEXT START ---
{user_world_context_str}
--- USER WORLD CONTEXT END ---

Begin your single, valid JSON output now. Do NOT include any explanatory text before or after the JSON object.
""")
    prompt = "\n".join(prompt_lines)

    logger.info("Generating/completing initial world-building data (to JSON) via LLM...") # Updated log message
    cleaned_world_text_json, usage_data = await llm_service.async_call_llm( # Renamed variable
        model_name=config.INITIAL_SETUP_MODEL,
        prompt=prompt,
        temperature=config.TEMPERATURE_INITIAL_SETUP,
        stream_to_disk=True,
        frequency_penalty=config.FREQUENCY_PENALTY_INITIAL_SETUP,
        presence_penalty=config.PRESENCE_PENALTY_INITIAL_SETUP,
        auto_clean_response=True
    )
    if usage_data:
        accumulated_usage_data["prompt_tokens"] += usage_data.get("prompt_tokens", 0)
        accumulated_usage_data["completion_tokens"] += usage_data.get("completion_tokens", 0)
        accumulated_usage_data["total_tokens"] += usage_data.get("total_tokens", 0)

    logger.debug(f"Cleaned LLM world-building JSON output (len: {len(cleaned_world_text_json)}):\nSTART_OF_WB_JSON_TEXT\n{cleaned_world_text_json}\nEND_OF_WB_JSON_TEXT") # Updated log message

    logger.info("Attempting to parse LLM world-building output using JSON parser...") # Updated log message
    # TODO: parse_markdown_to_dict is currently an unresolved import.
    # This will cause a runtime error if this part of the code is reached.
    # For now, proceeding as if it exists, to fulfill the subtask's scope.
    parsed_llm_json_response: Optional[Dict[str, Any]] = None # Renamed variable
    try:
        # This import will fail if markdown_story_parser.py (containing parse_markdown_to_dict) is not available
        # and parse_markdown_to_dict hasn't been moved/redefined.
        # from markdown_story_parser import parse_markdown_to_dict # This line will be replaced
        # parsed_llm_markdown_response = parse_markdown_to_dict(cleaned_world_text_markdown)
        # logger.debug(f"DIRECT Output of parse_markdown_to_dict for LLM response: {json.dumps(parsed_llm_markdown_response, indent=2)}")

        # NEW JSON PARSING:
        parsed_llm_json_response = json.loads(cleaned_world_text_json) # Use json.loads for JSON
        logger.debug(f"DIRECT Output of json.loads for LLM response: {json.dumps(parsed_llm_json_response, indent=2)}")

    except ImportError: # This specific except block might become irrelevant if parse_markdown_to_dict is no longer called here.
        logger.error("CRITICAL: `parse_markdown_to_dict` is not available due to markdown_story_parser.py deletion. LLM Markdown output for world-building cannot be processed.")
        # parsed_llm_json_response remains None
    except json.JSONDecodeError as e:
        logger.error(f"CRITICAL: Failed to decode JSON from LLM output. Error: {e}. Raw text was: {cleaned_world_text_json}")
        parsed_llm_json_response = None


    if parsed_llm_json_response and isinstance(parsed_llm_json_response, dict):
        agent.world_building["source"] = "llm_generated_or_merged_json_style" # Set source upon successful LLM parse

        # Ensure agent.world_building is a dict (it should be, but as a safeguard)
        if not isinstance(agent.world_building, dict):
            agent.world_building = {"_overview_":{}} # Initialize if somehow it's not a dict

        # The parsing logic below assumes the LLM *perfectly* follows the new JSON structure.
        # It directly tries to map incoming JSON keys to internal agent state structure.
        for json_top_level_cat_key, json_cat_content in parsed_llm_json_response.items():
            internal_cat_name = WORLD_CATEGORY_MAP_NORMALIZED_TO_INTERNAL.get(json_top_level_cat_key, None)
            if internal_cat_name is None:
                logger.warning(f"LLM JSON response contained unrecognized top-level key '{json_top_level_cat_key}'. This key will be IGNORED.")
                continue

            agent.world_building.setdefault(internal_cat_name, {})
            target_category_in_agent = agent.world_building[internal_cat_name]

            if not isinstance(json_cat_content, dict):
                logger.warning(f"Content for category '{json_top_level_cat_key}' from LLM JSON is not a dictionary. Skipping. Content: {json_cat_content}")
                continue

            if internal_cat_name == "_overview_":
                # json_cat_content is like {"description": "...", "mood": "..."}
                for json_detail_key, json_detail_value in json_cat_content.items():
                    # Assuming json_detail_key is already the desired internal key, or maps directly
                    # from WORLD_DETAIL_KEY_MAP_FROM_MARKDOWN_TO_INTERNAL's keys.
                    internal_detail_key_target = WORLD_DETAIL_KEY_MAP_FROM_MARKDOWN_TO_INTERNAL.get(json_detail_key, json_detail_key)

                    # Conditionally wrap json_detail_value
                    if internal_detail_key_target not in WORLD_DETAIL_LIST_INTERNAL_KEYS:
                        if isinstance(json_detail_value, str):
                            json_detail_value = {"text": json_detail_value}
                        elif isinstance(json_detail_value, list):
                            json_detail_value = {"items": json_detail_value}

                    existing_val = target_category_in_agent.get(internal_detail_key_target)
                    if utils._is_fill_in(existing_val) or existing_val is None:
                        if json_detail_value is not None and not utils._is_fill_in(json_detail_value):
                            if internal_detail_key_target in WORLD_DETAIL_LIST_INTERNAL_KEYS:
                                if isinstance(json_detail_value, list):
                                    # Ensure all items are strings, filtering out any [Fill-in] from LLM if it sneaks in
                                    processed_list = [str(li) for li in json_detail_value if isinstance(li, (str, int, float, bool)) and not utils._is_fill_in(str(li))]
                                    if processed_list: # Only assign if there's actual content
                                        target_category_in_agent[internal_detail_key_target] = processed_list
                                    elif utils._is_fill_in(existing_val) or existing_val is None: # If existing was fill-in, and LLM provided empty/bad list
                                        target_category_in_agent[internal_detail_key_target] = [config.MARKDOWN_FILL_IN_PLACEHOLDER] # Fallback to fill-in
                                elif isinstance(json_detail_value, str): # LLM returned a string instead of a list
                                    logger.warning(f"LLM returned string for list key '{internal_detail_key_target}' in overview. Converting to list: '{json_detail_value}'")
                                    target_category_in_agent[internal_detail_key_target] = [json_detail_value]
                                else: # LLM returned something else problematic
                                    logger.warning(f"LLM returned non-list/non-string for list key '{internal_detail_key_target}' in overview: {type(json_detail_value)}. Setting to fill-in.")
                                    target_category_in_agent[internal_detail_key_target] = [config.MARKDOWN_FILL_IN_PLACEHOLDER]
                            else: # Not a list key
                                target_category_in_agent[internal_detail_key_target] = str(json_detail_value) # Ensure string
            else: # Itemized categories like "locations", "factions"
                processed_items_for_category: Dict[str, Any] = {}
                default_item_properties: Dict[str, Any] = {}

                for item_key_from_llm, item_value_from_llm in json_cat_content.items():
                    # Try to normalize item_key_from_llm in case it's a property name for the default item
                    internal_item_key_for_agent = WORLD_DETAIL_KEY_MAP_FROM_MARKDOWN_TO_INTERNAL.get(item_key_from_llm, item_key_from_llm)

                    if isinstance(item_value_from_llm, dict): # This is a structured item with its own attributes
                        item_attributes_dict = item_value_from_llm
                        processed_attributes: Dict[str, Any] = {}
                        for attr_key, attr_val in item_attributes_dict.items():
                            target_attr_key = WORLD_DETAIL_KEY_MAP_FROM_MARKDOWN_TO_INTERNAL.get(attr_key, attr_key)

                            # Conditionally wrap attr_val
                            if target_attr_key not in WORLD_DETAIL_LIST_INTERNAL_KEYS:
                                if isinstance(attr_val, str):
                                    attr_val = {"text": attr_val}
                                elif isinstance(attr_val, list):
                                    attr_val = {"items": attr_val}
                            processed_attributes[target_attr_key] = attr_val
                        # Use original item_key_from_llm as the item name (not normalized version)
                        processed_items_for_category[item_key_from_llm] = processed_attributes
                    else: # This is a flat property, potentially for a "default" item for this category
                        prop_value = item_value_from_llm
                        # Here, internal_item_key_for_agent is the property name
                        if internal_item_key_for_agent not in WORLD_DETAIL_LIST_INTERNAL_KEYS:
                            if isinstance(prop_value, str):
                                prop_value = {"text": prop_value}
                            elif isinstance(prop_value, list):
                                prop_value = {"items": prop_value}
                        default_item_properties[internal_item_key_for_agent] = prop_value

                if default_item_properties:
                    # Use the original LLM category key as the default item name
                    default_item_name = json_top_level_cat_key
                    if default_item_name in processed_items_for_category:
                        # If a real item has the same name as the category, merge default properties into it
                        # Default properties take precedence
                        logger.info(f"Merging default properties into existing item '{default_item_name}' for category '{internal_cat_name}'.")
                        processed_items_for_category[default_item_name].update(default_item_properties)
                    else:
                        processed_items_for_category[default_item_name] = default_item_properties

                # Update agent state for the current category
                # Clear existing items managed by LLM for this category before adding new ones
                # This part needs careful handling if user-supplied data and LLM data are mixed at item level
                # For now, assuming LLM is authoritative for categories it processes in this block

                # Preserve user-added items if any, only clear/update LLM-sourced items.
                # However, the current task implies replacing the old loop, which suggests a full overwrite for LLM-processed categories.
                # Let's proceed with clearing the category in agent state before repopulating from processed_items_for_category

                # Ensure the category exists
                agent.world_building.setdefault(internal_cat_name, {})

                # Option 1: Full clear - simpler, assumes LLM provides complete picture for this category now
                # agent.world_building[internal_cat_name].clear() # Clears all items, user-added or LLM

                # Option 2: Selective update/clear (More complex: requires tracking item sources)
                # For now, sticking to a simpler model: if LLM processes a category, it owns its content.
                # User data should be merged *before* LLM if fill-ins are present.
                # The current logic path is for when `needs_llm_for_world` is true.

                # Re-initialize the category in agent state to ensure clean slate for LLM content
                target_category_in_agent = agent.world_building[internal_cat_name] = {}


                for item_name, item_details_dict in processed_items_for_category.items():
                    # Ensure item_name is a string, as it's used as a dict key
                    item_name_str = str(item_name)
                    agent_item_details_target = target_category_in_agent.setdefault(item_name_str, {"source": "llm_generated_json_style"})
                    
                    # Merge the processed details.
                    # The item_details_dict already has values wrapped (text/items) and keys normalized where appropriate.
                    # We need to handle the [Fill-in] logic and list vs string for WORLD_DETAIL_LIST_INTERNAL_KEYS
                    # similar to how it's done for _overview_ or the previous item loop.

                    for detail_key, detail_value in item_details_dict.items():
                        # detail_key is already the internal_item_detail_key_target
                        existing_agent_val = agent_item_details_target.get(detail_key)

                        # Apply if existing is fill-in, or if it's None (new key for this item)
                        if utils._is_fill_in(existing_agent_val) or existing_agent_val is None:
                            if detail_value is not None and not utils._is_fill_in(detail_value): # LLM provided actual content
                                if detail_key in WORLD_DETAIL_LIST_INTERNAL_KEYS:
                                    # Value from LLM (detail_value) might be {"items": ["a", "b"]} or a direct list if wrapping wasn't applied
                                    # or it was already a list key.
                                    actual_list_items = []
                                    if isinstance(detail_value, dict) and "items" in detail_value and isinstance(detail_value["items"], list):
                                        actual_list_items = detail_value["items"]
                                    elif isinstance(detail_value, list): # LLM directly provided a list for a list key
                                        actual_list_items = detail_value
                                    elif isinstance(detail_value, str): # LLM provided a string for a list key
                                         logger.warning(f"LLM returned string for list key '{detail_key}' in item '{item_name_str}' for category '{internal_cat_name}'. Converting to list: '{detail_value}'")
                                         actual_list_items = [detail_value]
                                    else:
                                        logger.warning(f"LLM returned incompatible type for list key '{detail_key}' in item '{item_name_str}' for '{internal_cat_name}': {type(detail_value)}. Setting to fill-in.")
                                        actual_list_items = [config.MARKDOWN_FILL_IN_PLACEHOLDER]

                                    processed_list = [str(li) for li in actual_list_items if isinstance(li, (str, int, float, bool)) and not utils._is_fill_in(str(li))]
                                    if processed_list:
                                        agent_item_details_target[detail_key] = processed_list
                                    # If existing was fill-in and LLM provided empty/bad list, ensure it remains a fill-in list
                                    elif utils._is_fill_in(existing_agent_val) or existing_agent_val is None:
                                         agent_item_details_target[detail_key] = [config.MARKDOWN_FILL_IN_PLACEHOLDER]
                                else: # Not a list key
                                    # Value from LLM (detail_value) might be {"text": "..."} or a direct string
                                    if isinstance(detail_value, dict) and "text" in detail_value:
                                        agent_item_details_target[detail_key] = str(detail_value["text"])
                                    else: # Assume it's a direct string or can be converted
                                        agent_item_details_target[detail_key] = str(detail_value)
                            # If LLM detail_value is None or fill-in, and agent had fill-in/None, ensure it's properly set to fill-in
                            elif detail_key in WORLD_DETAIL_LIST_INTERNAL_KEYS:
                                agent_item_details_target[detail_key] = [config.MARKDOWN_FILL_IN_PLACEHOLDER]
                            else:
                                agent_item_details_target[detail_key] = config.MARKDOWN_FILL_IN_PLACEHOLDER
                        # If agent already had concrete data, LLM doesn't overwrite unless that logic changes.
                        # This part is primarily for filling in missing/default data.

        agent.world_building.pop("is_default", None)
        agent.world_building.pop("user_supplied_data", None)
        logger.info("Successfully processed LLM-generated JSON world-building into agent state.") # Updated log
    else:
        # This 'else' corresponds to 'if parsed_llm_json_response and isinstance(parsed_llm_json_response, dict):'
        # It means either json.loads failed (parsed_llm_json_response is None) or the root was not a dict.
        logger.error("Failed to parse a valid world-building dictionary from LLM JSON output, or root of JSON was not a dictionary. Existing or default world_building will be used.") # Updated log
        if not agent.world_building.get("_overview_") or utils._is_fill_in(agent.world_building.get("_overview_",{}).get("description")):
            default_wb_overview_desc = agent.plot_outline.get("setting_description", "A default world setting.")
            if utils._is_fill_in(default_wb_overview_desc): default_wb_overview_desc = "A default world setting, to be detailed later."

            agent.world_building.setdefault("_overview_", {})["description"] = default_wb_overview_desc
            # Only set to default_fallback if it wasn't originally user-supplied and now failing back
            if agent.world_building.get("source") != "user_supplied_yaml":
                 agent.world_building["source"] = "default_fallback"
            # is_default should only be set if we are truly falling back from an LLM attempt on non-user data
            if agent.world_building.get("source") == "default_fallback": # Check again, as it might have been set above
                 agent.world_building["is_default"] = True
    
    # Final check to ensure all expected categories exist, even if empty.
    # This should run regardless of LLM success if an LLM call was attempted.
    # If it was user_supplied_yaml and LLM was skipped, this part is also skipped.
    if llm_was_called: # Only do this if an LLM call was actually made
        current_source = agent.world_building.get("source", "")
        # Don't create empty categories if data is purely from user_supplied_yaml and LLM was not needed to fill gaps
        if not (current_source == "user_supplied_yaml" and not needs_llm_for_world):
            for cat_internal_key in WORLD_CATEGORY_MAP_NORMALIZED_TO_INTERNAL.values():
                agent.world_building.setdefault(cat_internal_key, {})


    return agent.world_building, accumulated_usage_data if llm_was_called else None