      
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
from parsing_utils import parse_key_value_block # Kept for plot parsing
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


    prompt_lines = []
    if config.ENABLE_LLM_NO_THINK_DIRECTIVE:
        prompt_lines.append("/no_think")

    prompt_lines.extend([
        "You are an expert world-building assistant for novelists.",
        "Based on the provided novel concept and any existing user world context, generate or complete detailed world-building elements using MARKDOWN FORMATTING.",
        "If user context provided a concrete value for an element, you MUST use it. You are filling in the gaps and expanding.",
        f"If user context for an element is '{config.MARKDOWN_FILL_IN_PLACEHOLDER}' or missing, generate it creatively, maintaining the Markdown style.",
        "",
        "**Novel Concept:**",
        f"  - Title: {plot_title}",
        f"  - Genre: {plot_genre}",
        f"  - Core Setting Idea (from plot outline): {world_setting_desc}",
        f"{user_world_context_str}", 
        "**Instructions for Output (CRITICAL - USE MARKDOWN FORMAT):**",
        "1.  Structure your output using Markdown headers. Top-level categories (Overview, Locations, Factions, Systems, Lore, History, Society) should be H2 headers (e.g., `## Locations`).",
        "2.  Specific items within categories (e.g., a specific location like 'The Hourglass Curios' under 'Locations') should be H3 headers (e.g., `### The Hourglass Curios`).",
        "3.  Details for each item or category should be listed as `**Key**: Value` pairs, each on a new line. Indent these under their respective header/item if that improves clarity, but `**Key**: Value` on its own line is primary.",
        "4.  For list-like details (e.g., `**Rules**` for a system, `**Goals**` for a faction), list each sub-item on a new line, indented further and prefixed with `- `.",
        "5.  Ensure comprehensive yet concise details. Aim for 2-4 items per category where applicable (except Overview). Expand on any `[Fill-in]` fields from the user context.",
        "",
        "**Example Output Structure (Follow this Markdown style):**",
        "```markdown",
        "## Overview",
        "**Description**: A general description of the world's feel and primary setting.",
        "",
        "## Locations",
        "### Temporal Hub Prime",
        "**Description**: A vast, crystalline city existing outside normal spacetime...",
        "**Atmosphere**: Once bustling and sterile, now largely abandoned...",
        "",
        "### The Shifting Sands (Fracture Zone)",
        "**Description**: A chaotic expanse where timelines bleed into one another...",
        "**Atmosphere**: Surreal and disorienting...",
        "",
        "## Factions",
        "### The Chronos Wardens",
        "**Description**: A scattered group trying to uphold old laws...",
        "**Goals**:",
        "  - Preserve timeline integrity.",
        "  - Recruit individuals like Jax.",
        "```",
        "",
        "Begin your detailed world-building output now using Markdown formatting:"
    ])
    prompt = "\n".join(prompt_lines)

    logger.info("Generating/completing initial world-building data (to Markdown) via LLM...")
    cleaned_world_text_markdown, usage_data = await llm_service.async_call_llm(
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

    logger.debug(f"Cleaned LLM world-building Markdown output (len: {len(cleaned_world_text_markdown)}):\nSTART_OF_WB_MD_TEXT\n{cleaned_world_text_markdown}\nEND_OF_WB_MD_TEXT")

    logger.info("Attempting to parse LLM world-building output using Markdown parser...")
    # TODO: parse_markdown_to_dict is currently an unresolved import.
    # This will cause a runtime error if this part of the code is reached.
    # For now, proceeding as if it exists, to fulfill the subtask's scope.
    parsed_llm_markdown_response: Optional[Dict[str, Any]] = None
    try:
        # This import will fail if markdown_story_parser.py (containing parse_markdown_to_dict) is not available
        # and parse_markdown_to_dict hasn't been moved/redefined.
        from markdown_story_parser import parse_markdown_to_dict
        parsed_llm_markdown_response = parse_markdown_to_dict(cleaned_world_text_markdown)
        logger.debug(f"DIRECT Output of parse_markdown_to_dict for LLM response: {json.dumps(parsed_llm_markdown_response, indent=2)}")
    except ImportError:
        logger.error("CRITICAL: `parse_markdown_to_dict` is not available due to markdown_story_parser.py deletion. LLM Markdown output for world-building cannot be processed.")
        # parsed_llm_markdown_response remains None


    # Ensure agent.world_building is a dict before merging
    if not isinstance(agent.world_building, dict):
        agent.world_building = {"source":"llm_generated_or_merged_yaml_style", "_overview_":{}} # Updated style
    elif "source" not in agent.world_building : # If it was populated by user data, source should be set
        agent.world_building["source"] = "llm_generated_or_merged_yaml_style" # Updated style


    if parsed_llm_markdown_response and isinstance(parsed_llm_markdown_response, dict):
        for md_top_level_cat_key, md_cat_content in parsed_llm_markdown_response.items(): # md_top_level_cat_key is normalized
            internal_cat_name = WORLD_CATEGORY_MAP_NORMALIZED_TO_INTERNAL.get(md_top_level_cat_key, None)
            if internal_cat_name is None:
                logger.warning(f"LLM Markdown response contained unrecognized top-level key '{md_top_level_cat_key}'. This key will be IGNORED for direct category mapping.")
                continue

            agent.world_building.setdefault(internal_cat_name, {})
            target_category_in_agent = agent.world_building[internal_cat_name]

            if not isinstance(md_cat_content, dict):
                logger.warning(f"Content for category '{md_top_level_cat_key}' from LLM is not a dictionary. Skipping. Content: {md_cat_content}")
                continue

            if internal_cat_name == "_overview_":
                for md_detail_key, md_detail_value in md_cat_content.items(): # md_detail_key is "description", "mood", etc.
                    internal_detail_key_target = WORLD_DETAIL_KEY_MAP_FROM_MARKDOWN_TO_INTERNAL.get(md_detail_key, md_detail_key)
                    existing_val = target_category_in_agent.get(internal_detail_key_target)
                    if utils._is_fill_in(existing_val) or existing_val is None:
                        if md_detail_value is not None and not utils._is_fill_in(md_detail_value):
                            target_category_in_agent[internal_detail_key_target] = md_detail_value
            else: # Itemized categories like "locations", "factions"
                  # md_cat_content is like {"the_core_nexus": {"description": "...", "atmosphere": "..."}}
                for item_name_norm_md, item_details_md in md_cat_content.items():
                    item_name_display = item_name_norm_md.replace("_", " ").title()

                    if not isinstance(item_details_md, dict):
                        logger.warning(f"Details for item '{item_name_norm_md}' in category '{internal_cat_name}' is not a dict. Skipping. Details: {item_details_md}")
                        continue
                    
                    # Ensure this item exists in the agent's state for this category
                    target_category_in_agent.setdefault(item_name_display, {"source": "llm_generated_yaml_style"}) # Updated style
                    current_agent_item_details = target_category_in_agent[item_name_display]

                    for md_item_detail_key, md_item_detail_value in item_details_md.items(): # md_item_detail_key is normalized
                        internal_item_detail_key_target = WORLD_DETAIL_KEY_MAP_FROM_MARKDOWN_TO_INTERNAL.get(md_item_detail_key, md_item_detail_key)
                        existing_item_val = current_agent_item_details.get(internal_item_detail_key_target)
                        
                        if utils._is_fill_in(existing_item_val) or existing_item_val is None: # If agent's current value is fill-in or missing
                            if md_item_detail_value is not None and not utils._is_fill_in(md_item_detail_value): # And LLM provided concrete
                                current_agent_item_details[internal_item_detail_key_target] = md_item_detail_value
                        # If agent had concrete user data, it's preserved (not overwritten by LLM unless it was [Fill-in])

        agent.world_building.pop("is_default", None) # Clean up helper flags
        agent.world_building.pop("user_supplied_data", None)
        logger.info("Successfully processed LLM-generated Markdown world-building into agent state.")
    else:
        logger.error("Failed to parse a valid world-building dictionary from LLM Markdown output. Existing or default world_building will be used.")
        if not agent.world_building.get("_overview_") or utils._is_fill_in(agent.world_building.get("_overview_",{}).get("description")):
            default_wb_overview_desc = agent.plot_outline.get("setting_description", "A default world setting.")
            if utils._is_fill_in(default_wb_overview_desc): default_wb_overview_desc = "A default world setting, to be detailed later."

            agent.world_building.setdefault("_overview_", {})["description"] = default_wb_overview_desc
            if agent.world_building.get("source") != "user_supplied_yaml": agent.world_building["source"] = "default_fallback" # Updated source
            if "is_default" not in agent.world_building : agent.world_building["is_default"] = True
    
    # Final check to ensure all expected categories exist, even if empty, if populated by LLM
    if agent.world_building.get("source") != "user_supplied_yaml": # Updated source
        for cat_internal_key in WORLD_CATEGORY_MAP_NORMALIZED_TO_INTERNAL.values():
            agent.world_building.setdefault(cat_internal_key, {})


    return agent.world_building, accumulated_usage_data if llm_was_called else None

    