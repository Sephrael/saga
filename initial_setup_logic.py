# initial_setup_logic.py
import logging
import json # Retain for fallback or other JSON operations
import random
import os
import re
from typing import Dict, Any, Optional, List, Tuple

import config
import llm_interface
import utils # For _is_fill_in
from parsing_utils import parse_key_value_block, parse_hierarchical_structured_text # Retain for LLM output parsing
from markdown_story_parser import load_and_parse_markdown_story_file # Use your new parser

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
# Pattern for world category headers, e.g., "Locations:" or "Category: Factions"
WORLD_CATEGORY_HEADER_PATTERN = re.compile(r"^\s*(?:Category\s*:\s*)?([A-Za-z\s_]+?):\s*$", re.IGNORECASE | re.MULTILINE)
# Pattern for world item headers, e.g., "The Old Mill:" or "The Sunken City" (colon optional if at end of line)
WORLD_ITEM_HEADER_PATTERN = re.compile(r"^\s*([A-Za-z0-9\s'\-]+?)(?::\s*$|$)", re.MULTILINE)
WORLD_DETAIL_KEY_MAP_NORMALIZED_TO_INTERNAL = {
    "description": "description", "atmosphere": "atmosphere", "modification_proposal": "modification_proposal",
    "goals": "goals", "rules": "rules", "key_elements": "key_elements", "traits": "traits"
}
WORLD_DETAIL_LIST_INTERNAL_KEYS = ["goals", "rules", "key_elements", "traits"]

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
    """Loads user-supplied story data from Markdown file."""
    user_data = load_and_parse_markdown_story_file(config.USER_STORY_ELEMENTS_FILE_PATH)
    if user_data is None: 
        logger.info(f"User story elements file '{config.USER_STORY_ELEMENTS_FILE_PATH}' not found. Will proceed with LLM generation or defaults.")
        return None
    if not user_data: 
        logger.warning(f"User story elements file '{config.USER_STORY_ELEMENTS_FILE_PATH}' was empty or could not be parsed. Will proceed with LLM generation or defaults.")
        return {} 

    expected_top_level_keys = ["novel_concept", "protagonist", "plot_points", "setting", "world_details", "antagonist", "conflict", "other_key_characters"]
    found_any_expected_key = False
    for key in expected_top_level_keys:
        if key in user_data and isinstance(user_data[key], dict) and user_data[key]: 
            found_any_expected_key = True; break
        elif key == "plot_points" and key in user_data and isinstance(user_data[key], list) and user_data[key]:
            found_any_expected_key = True; break
            
    if not found_any_expected_key:
        logger.error(f"User-supplied Markdown data from '{config.USER_STORY_ELEMENTS_FILE_PATH}' does not seem to contain any expected top-level sections with content. Parsed data: {user_data}")
        return {} 
        
    logger.info(f"Successfully loaded and performed initial validation on user-supplied story data from '{config.USER_STORY_ELEMENTS_FILE_PATH}'.")
    return user_data

def _populate_agent_state_from_user_data(agent: Any, user_data: Dict[str, Any]):
    plot_outline: PlotOutlineData = agent.plot_outline if hasattr(agent, 'plot_outline') and agent.plot_outline else {}
    character_profiles: Dict[str, Any] = agent.character_profiles if hasattr(agent, 'character_profiles') and agent.character_profiles else {}
    world_building: WorldBuildingData = agent.world_building if hasattr(agent, 'world_building') and agent.world_building else {}

    for cat in ["locations", "society", "systems", "lore", "history", "factions", "_overview_"]:
        world_building.setdefault(cat, {})
    world_building["user_supplied_data"] = True
    world_building["is_default"] = False
    world_building["source"] = "user_supplied_markdown"

    nc = user_data.get("novel_concept", {})
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

    setting_data_md = user_data.get("setting", {}) 
    plot_outline["setting_description"] = _get_val_or_fill_in(setting_data_md, "primary_setting_description")
    
    plot_outline["source"] = "user_supplied_markdown"
    plot_outline["is_default"] = False
    agent.plot_outline = plot_outline

    prot_name_val = plot_outline["protagonist_name"] 
    if not utils._is_fill_in(prot_name_val): 
        character_profiles.setdefault(prot_name_val, {}) 
        character_profiles[prot_name_val].update({
            "description": plot_outline["protagonist_description"],
            "traits": [t for t in prot_data.get("traits", []) if isinstance(t,str) and (t.strip() or utils._is_fill_in(t))],
            "status": _get_val_or_fill_in(prot_data, "initial_status") or "As described",
            "character_arc_summary": plot_outline["character_arc"],
            "role": "protagonist", "source": "user_supplied_markdown",
            "relationships": prot_data.get("relationships", {}) 
        })
    
    ant_name_val = plot_outline["antagonist_name"]
    if not utils._is_fill_in(ant_name_val) and ant_data:
        character_profiles.setdefault(ant_name_val, {})
        character_profiles[ant_name_val].update({
            "description": plot_outline["antagonist_description"],
            "traits": [t for t in ant_data.get("traits", []) if isinstance(t,str) and (t.strip() or utils._is_fill_in(t))],
            "status": "As described",
            "motivations": plot_outline["antagonist_motivations"],
            "role": "antagonist", "source": "user_supplied_markdown",
            "relationships": ant_data.get("relationships", {})
        })

    other_chars_data = user_data.get("other_key_characters", {})
    if isinstance(other_chars_data, dict):
        for char_name_other_raw, char_detail_raw in other_chars_data.items():
            char_name_other = str(char_name_other_raw)
            if not utils._is_fill_in(char_name_other) and isinstance(char_detail_raw, dict):
                character_profiles.setdefault(char_name_other, {})
                character_profiles[char_name_other].update({
                    "description": _get_val_or_fill_in(char_detail_raw, "description"),
                    "traits": [t for t in char_detail_raw.get("traits", []) if isinstance(t,str) and (t.strip() or utils._is_fill_in(t))],
                    "status": "As described", 
                    "role_in_story": _get_val_or_fill_in(char_detail_raw, "role_in_story"), 
                    "source": "user_supplied_markdown",
                    "relationships": char_detail_raw.get("relationships", {})
                })
    agent.character_profiles = character_profiles

    world_building.setdefault("_overview_", {})
    world_building["_overview_"]["description"] = plot_outline["setting_description"] 

    key_locations_md = setting_data_md.get("key_locations", {}) 
    if isinstance(key_locations_md, dict):
        world_building.setdefault("locations", {})
        for loc_name_raw, loc_details_raw in key_locations_md.items():
            loc_name = str(loc_name_raw) 
            if not utils._is_fill_in(loc_name) and isinstance(loc_details_raw, dict):
                world_building["locations"].setdefault(loc_name, {})
                world_building["locations"][loc_name].update({
                    "description": _get_val_or_fill_in(loc_details_raw, "description"),
                    "atmosphere": _get_val_or_fill_in(loc_details_raw, "atmosphere"),
                    "source": "user_supplied_markdown"
                })

    wd_details = user_data.get("world_details", {}) 
    unique_feature_md_item_name = "unique_world_feature" 
    unique_feature_data = wd_details.get(unique_feature_md_item_name, {})
    if isinstance(unique_feature_data, dict) and "description" in unique_feature_data :
        actual_feature_name_to_use = "Unique World Feature" 
        world_building.setdefault("systems", {})
        world_building["systems"].setdefault(actual_feature_name_to_use, {})
        world_building["systems"][actual_feature_name_to_use].update({
            "description": _get_val_or_fill_in(unique_feature_data, "description"),
            "rules": [r for r in unique_feature_data.get("rules", []) if isinstance(r,str) and (r.strip() or utils._is_fill_in(r))],
            "source": "user_supplied_markdown"
        })

    key_factions_md = wd_details.get("key_factions", {})
    if isinstance(key_factions_md, dict):
        world_building.setdefault("factions", {})
        for faction_name_raw, faction_details_raw in key_factions_md.items():
            faction_name = str(faction_name_raw)
            if not utils._is_fill_in(faction_name) and isinstance(faction_details_raw, dict):
                world_building["factions"].setdefault(faction_name, {})
                world_building["factions"][faction_name].update({
                    "description": _get_val_or_fill_in(faction_details_raw, "description"),
                    "goals": [g for g in faction_details_raw.get("goals", []) if isinstance(g,str) and (g.strip() or utils._is_fill_in(g))],
                    "source": "user_supplied_markdown"
                })
            
    relevant_lore_md = wd_details.get("relevant_lore", {})
    if isinstance(relevant_lore_md, dict):
        world_building.setdefault("lore", {})
        for lore_name_raw, lore_details_raw in relevant_lore_md.items():
            lore_name = str(lore_name_raw)
            if not utils._is_fill_in(lore_name) and isinstance(lore_details_raw, dict):
                world_building["lore"].setdefault(lore_name, {})
                world_building["lore"][lore_name].update({
                    "description": _get_val_or_fill_in(lore_details_raw, "description"),
                    "source": "user_supplied_markdown"
                })
                if "known_effects" in lore_details_raw: 
                    world_building["lore"][lore_name]["known_effects"] = _get_val_or_fill_in(lore_details_raw, "known_effects")

    agent.world_building = world_building
    logger.info("Agent state populated from user-supplied Markdown data (preserving '[Fill-in]' markers).")

async def generate_plot_outline_logic(agent: Any, default_protagonist_name: str, unhinged_mode: bool, **kwargs) -> Tuple[PlotOutlineData, Optional[Dict[str, int]]]:
    logger.info(f"Generating plot outline. Unhinged mode: {unhinged_mode}")
    user_supplied_data = _load_user_supplied_data() 
    
    llm_was_called = False
    accumulated_usage_data: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    if user_supplied_data: 
        if user_supplied_data: 
            logger.info("Processing user-supplied Markdown data for initial setup.")
            _populate_agent_state_from_user_data(agent, user_supplied_data)
        else: 
            logger.warning("User-supplied Markdown file was effectively empty or unparsable. Plot will be fully generated or default.")
            agent.plot_outline = {} 
    else: 
        logger.info("No user-supplied Markdown file found. Plot outline will be fully generated by LLM or default.")
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

        if agent.plot_outline:
            for display_key_normalized, internal_key in current_plot_outline_key_map_for_llm_prompt.items():
                user_val = agent.plot_outline.get(internal_key)
                display_key_title_case = display_key_normalized.replace('_', ' ').title()
                
                if user_val is not None: 
                    if isinstance(user_val, list): 
                        concrete_list_items = [item for item in user_val if not utils._is_fill_in(item) and str(item).strip()]
                        fill_in_placeholders_in_list = [item for item in user_val if utils._is_fill_in(item)]
                        
                        if concrete_list_items:
                            context_from_user_input += f"  - {display_key_title_case}: User provided {len(concrete_list_items)} concrete item(s), e.g., \"{concrete_list_items[0][:50]}...\". "
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
        in_pure_llm_scratch_mode = not user_supplied_data and (not agent.plot_outline or not any(agent.plot_outline.values()))

        prompt_core_elements_intro = ""
        if unhinged_mode and in_pure_llm_scratch_mode:
            base_elements_for_outline["genre"] = kwargs.get("genre", random.choice(config.UNHINGED_GENRES))
            base_elements_for_outline["theme"] = kwargs.get("theme", random.choice(config.UNHINGED_THEMES))
            base_elements_for_outline["setting_description"] = kwargs.get("setting_archetype", random.choice(config.UNHINGED_SETTINGS_ARCHETYPES))
            base_elements_for_outline["protagonist_name"] = default_protagonist_name 
            base_elements_for_outline["source_hint"] = "unhinged_pure_llm"
            prompt_core_elements_intro = f"You are in UNHINGED mode. Generate a novel concept based on:\n  - Genre: {base_elements_for_outline['genre']}\n  - Theme: {base_elements_for_outline['theme']}\n  - Setting Archetype: {base_elements_for_outline['setting_description']}\n"
        else: 
            base_elements_for_outline["genre"] = agent.plot_outline.get("genre") if agent.plot_outline and not utils._is_fill_in(agent.plot_outline.get("genre")) else config.CONFIGURED_GENRE
            base_elements_for_outline["theme"] = agent.plot_outline.get("theme") if agent.plot_outline and not utils._is_fill_in(agent.plot_outline.get("theme")) else config.CONFIGURED_THEME
            base_elements_for_outline["setting_description"] = agent.plot_outline.get("setting_description") if agent.plot_outline and not utils._is_fill_in(agent.plot_outline.get("setting_description")) else config.CONFIGURED_SETTING_DESCRIPTION
            base_elements_for_outline["protagonist_name"] = agent.plot_outline.get("protagonist_name") if agent.plot_outline and not utils._is_fill_in(agent.plot_outline.get("protagonist_name")) else default_protagonist_name
            base_elements_for_outline["source_hint"] = "configured_or_user_markdown"
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
        cleaned_outline_text, usage_data = await llm_interface.async_call_llm(
            model_name=config.INITIAL_SETUP_MODEL, 
            prompt=prompt, 
            temperature=config.TEMPERATURE_INITIAL_SETUP, 
            stream_to_disk=True,
            frequency_penalty=config.FREQUENCY_PENALTY_INITIAL_SETUP,
            presence_penalty=config.PRESENCE_PENALTY_INITIAL_SETUP
        )
        if usage_data:
            accumulated_usage_data["prompt_tokens"] += usage_data.get("prompt_tokens", 0)
            accumulated_usage_data["completion_tokens"] += usage_data.get("completion_tokens", 0)
            accumulated_usage_data["total_tokens"] += usage_data.get("total_tokens", 0)

        parsed_llm_response = parse_key_value_block(
            cleaned_outline_text, current_plot_outline_key_map_for_llm_prompt, PLOT_OUTLINE_LIST_INTERNAL_KEYS
        )

        if not agent.plot_outline: agent.plot_outline = {} 

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
            
            for be_key in ["genre", "theme", "setting_description"]:
                if utils._is_fill_in(agent.plot_outline.get(be_key)):
                    llm_generated_val = parsed_llm_response.get(be_key, base_elements_for_outline.get(be_key))
                    if llm_generated_val and not utils._is_fill_in(llm_generated_val):
                         agent.plot_outline[be_key] = llm_generated_val
                    elif base_elements_for_outline.get(be_key): 
                         agent.plot_outline[be_key] = base_elements_for_outline.get(be_key)

            agent.plot_outline.pop("is_default", None) 
            if not agent.plot_outline.get("source") or agent.plot_outline.get("source") == "default_fallback":
                agent.plot_outline["source"] = base_elements_for_outline.get("source_hint", "llm_generated_or_merged")
            logger.info(f"LLM successfully generated/updated plot outline elements. Title: '{agent.plot_outline.get('title', 'N/A')}' with {len(agent.plot_outline.get('plot_points',[]))} plot points.")
        else: 
            logger.error("LLM failed to provide a parsable core plot outline. Falling back to default if agent state is still insufficient.")
            if not agent.plot_outline or any(utils._is_fill_in(agent.plot_outline.get(field)) for field in critical_plot_fields_for_llm_check):
                agent.plot_outline = _create_default_plot(default_protagonist_name, base_elements_for_outline, unhinged_mode and in_pure_llm_scratch_mode)

    prot_name_from_outline = agent.plot_outline.get('protagonist_name')
    if utils._is_fill_in(prot_name_from_outline) or not isinstance(prot_name_from_outline, str) or not prot_name_from_outline.strip():
        agent.plot_outline['protagonist_name'] = default_protagonist_name
        logger.warning(f"Protagonist name resolved to default: {default_protagonist_name}")
    
    final_protagonist_name = agent.plot_outline['protagonist_name']

    if not hasattr(agent, 'character_profiles') or agent.character_profiles is None:
        agent.character_profiles = {}
    
    agent.character_profiles.setdefault(final_protagonist_name, {"source": agent.plot_outline.get("source", "unknown")})
    prot_profile_ref = agent.character_profiles[final_protagonist_name]

    for key, plot_key in [("description", "protagonist_description"), ("character_arc_summary", "character_arc")]:
        plot_val = agent.plot_outline.get(plot_key)
        if not utils._is_fill_in(plot_val):
            prot_profile_ref[key] = plot_val
        elif key not in prot_profile_ref or utils._is_fill_in(prot_profile_ref.get(key)): 
            prot_profile_ref[key] = f"{config.MARKDOWN_FILL_IN_PLACEHOLDER} for {key}"
    
    prot_profile_ref["role"] = "protagonist"
    if "traits" not in prot_profile_ref or not prot_profile_ref["traits"] or all(utils._is_fill_in(t) for t in prot_profile_ref["traits"]):
        prot_profile_ref["traits"] = [config.MARKDOWN_FILL_IN_PLACEHOLDER]
    if "status" not in prot_profile_ref or utils._is_fill_in(prot_profile_ref.get("status")):
        prot_profile_ref["status"] = config.MARKDOWN_FILL_IN_PLACEHOLDER \
                                     if utils._is_fill_in(agent.plot_outline.get("protagonist_description")) \
                                     else "As described in plot outline"

    logger.info(f"Finalized character profile for protagonist '{final_protagonist_name}'.")
    
    if not hasattr(agent, 'world_building') or agent.world_building is None:
        agent.world_building = {"_overview_":{}, "source": agent.plot_outline.get("source", "unknown")}
    
    overview_desc_val = agent.plot_outline.get("setting_description", config.MARKDOWN_FILL_IN_PLACEHOLDER)
    agent.world_building.setdefault("_overview_", {})["description"] = overview_desc_val

    return agent.plot_outline, accumulated_usage_data if llm_was_called else None

async def generate_world_building_logic(agent: Any) -> Tuple[WorldBuildingData, Optional[Dict[str, int]]]:
    llm_was_called = False
    accumulated_usage_data: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    needs_llm_for_world = False
    if not hasattr(agent, 'world_building') or not agent.world_building:
        agent.world_building = {"source":"llm_to_create", "_overview_": {}} 
        needs_llm_for_world = True
    elif agent.world_building.get("source") == "user_supplied_markdown":
        for category, items_or_details in agent.world_building.items():
            if category in ["is_default", "source", "user_supplied_data"]: continue
            if isinstance(items_or_details, dict):
                if category == "_overview_":
                    if utils._is_fill_in(items_or_details.get("description")) or not items_or_details.get("description"):
                        needs_llm_for_world = True; break
                else: 
                    for item_name, details in items_or_details.items():
                        if isinstance(details, dict):
                            if any(utils._is_fill_in(v) for k, v in details.items() if isinstance(v, str) and k != "source"):
                                needs_llm_for_world = True; break
                            for list_key in WORLD_DETAIL_LIST_INTERNAL_KEYS:
                                list_val = details.get(list_key, [])
                                if isinstance(list_val, list) and (not list_val or all(utils._is_fill_in(li) for li in list_val)):
                                     needs_llm_for_world = True; break
                            if needs_llm_for_world: break
            if needs_llm_for_world: break
        if not needs_llm_for_world:
            logger.info("Skipping LLM world-building generation: Data was user-supplied from Markdown and seems complete.")
            return agent.world_building, None
    else: 
        needs_llm_for_world = True

    if not hasattr(agent, 'plot_outline') or not agent.plot_outline or utils._is_fill_in(agent.plot_outline.get("setting_description")):
        logger.warning("Cannot generate detailed world-building if plot outline or its setting_description is missing or '[Fill-in]'. LLM will use generic context or previously set world description.")
        if "source" not in agent.world_building or not agent.world_building.get("_overview_", {}).get("description"):
             agent.world_building.setdefault("_overview_",{})["description"] = config.MARKDOWN_FILL_IN_PLACEHOLDER
             agent.world_building["source"] = "llm_generated_default_context"
        needs_llm_for_world = True

    if not needs_llm_for_world:
        logger.info("World building data seems complete, skipping LLM call.")
        return agent.world_building, None

    llm_was_called = True
    plot_title = agent.plot_outline.get('title', 'Untitled Novel')
    plot_genre = agent.plot_outline.get('genre', 'N/A')
    world_setting_desc = agent.world_building.get("_overview_",{}).get("description", config.MARKDOWN_FILL_IN_PLACEHOLDER)
    if utils._is_fill_in(world_setting_desc):
        world_setting_desc = agent.plot_outline.get("setting_description", 'A newly conceived world.')
    if utils._is_fill_in(world_setting_desc): 
        world_setting_desc = 'A mysterious and detailed world waiting to be fleshed out.'

    user_world_context_str = "\n**User-Provided World Context (Respect these if not '[Fill-in]', complete if '[Fill-in]'):**\n"
    has_world_user_context = False
    if agent.world_building:
        for cat, items_or_desc_val in agent.world_building.items():
            if cat in ["is_default", "source", "user_supplied_data"]: continue
            cat_display_name = cat.replace('_', ' ').title()
            if cat == "_overview_":
                if isinstance(items_or_desc_val, dict) and "description" in items_or_desc_val:
                    desc_val = items_or_desc_val["description"]
                    if utils._is_fill_in(desc_val):
                        user_world_context_str += f"  - {cat_display_name} Description: {config.MARKDOWN_FILL_IN_PLACEHOLDER} (User requests generation)\n"
                    else:
                        user_world_context_str += f"  - {cat_display_name} Description: {desc_val}\n"
                    has_world_user_context = True
            elif isinstance(items_or_desc_val, dict) and items_or_desc_val:
                temp_cat_lines = []
                for item_name, details in items_or_desc_val.items():
                    if isinstance(details, dict):
                        item_context_parts = []
                        has_fill_in_item_detail = False
                        for d_key, d_val in details.items():
                            if d_key == "source": continue
                            d_key_display = d_key.replace('_', ' ').title()
                            if utils._is_fill_in(d_val): 
                                has_fill_in_item_detail = True
                                item_context_parts.append(f"{d_key_display}: {config.MARKDOWN_FILL_IN_PLACEHOLDER}")
                            elif isinstance(d_val, list):
                                concrete_list_items = [li for li in d_val if not utils._is_fill_in(li) and str(li).strip()]
                                fill_in_list_items = [li for li in d_val if utils._is_fill_in(li)]
                                if concrete_list_items: item_context_parts.append(f"{d_key_display}: {concrete_list_items}")
                                if fill_in_list_items: item_context_parts.append(f"{d_key_display}: ({len(fill_in_list_items)} '{config.MARKDOWN_FILL_IN_PLACEHOLDER}' items)")
                                if concrete_list_items or fill_in_list_items: has_fill_in_item_detail = True # This was potential bug, should be based on fill_in_list_items mainly
                            elif str(d_val).strip(): 
                                item_context_parts.append(f"{d_key_display}: {d_val}")
                        
                        if item_context_parts:
                            item_str = f"    - {item_name}: " + "; ".join(item_context_parts)
                            if has_fill_in_item_detail : item_str += " (May require LLM completion for some details)"
                            temp_cat_lines.append(item_str + "\n")
                if temp_cat_lines:
                    user_world_context_str += f"  - Category: {cat_display_name}\n" + "".join(temp_cat_lines)
                    has_world_user_context = True
    if not has_world_user_context:
        user_world_context_str = "\n**User-Provided World Context:** No specific world preferences or fill-in requests were found; generate creatively.\n"
    
    prompt_lines = []
    if config.ENABLE_LLM_NO_THINK_DIRECTIVE:
        prompt_lines.append("/no_think")
    
    prompt_lines.extend([
        "You are an expert world-building assistant for novelists.",
        "Based on the provided novel concept and any existing user world context, generate or complete detailed world-building elements as PLAIN TEXT.",
        "If user context provided a concrete value for an element, you MUST use it.",
        f"If user context for an element is '{config.MARKDOWN_FILL_IN_PLACEHOLDER}' or missing, generate it creatively.",
        "",
        "**Novel Concept:**",
        f"  - Title: {plot_title}",
        f"  - Genre: {plot_genre}",
        f"  - Core Setting Idea: {world_setting_desc}",
        f"{user_world_context_str}",
        "**Instructions for Output:**",
        "1.  Structure your output using clear category headers (e.g., `Overview:`, `Locations:`, `Factions:`, `Systems:`, `Lore:`).",
        "2.  For the `Overview:` category, provide a general description directly under an appropriate key like `Description:`.",
        "3.  For other categories (like `Locations`, `Factions`), list each item on its own line starting with the item's name followed by a colon (e.g., `The Whispering Woods:` or `The Sunken City:`).",
        "4.  Under each item, provide indented \"Key: Value\" pairs for its details. Use keys like `Description`, `Atmosphere`, `Goals`, `Rules`, `Key Elements`, `Traits`.",
        "5.  For list-like details (e.g., `Goals` for a faction, `Rules` for a system), list each sub-item on a new line, prefixed with \"- \".",
        "6.  Ensure comprehensive yet concise details. Aim for 2-4 items per category where applicable (except Overview).",
        "",
        "Begin your detailed world-building output now:"
    ])
    prompt = "\n".join(prompt_lines)

    logger.info("Generating/completing initial world-building data (to plain text) via LLM...")
    # MODIFIED: cleaned_world_text directly from async_call_llm
    cleaned_world_text, usage_data = await llm_interface.async_call_llm(
        model_name=config.INITIAL_SETUP_MODEL, 
        prompt=prompt, 
        temperature=config.TEMPERATURE_INITIAL_SETUP, 
        stream_to_disk=True,
        frequency_penalty=config.FREQUENCY_PENALTY_INITIAL_SETUP,
        presence_penalty=config.PRESENCE_PENALTY_INITIAL_SETUP,
        auto_clean_response=True # Default
    )
    if usage_data:
        accumulated_usage_data["prompt_tokens"] += usage_data.get("prompt_tokens", 0)
        accumulated_usage_data["completion_tokens"] += usage_data.get("completion_tokens", 0)
        accumulated_usage_data["total_tokens"] += usage_data.get("total_tokens", 0)
    
    logger.debug(f"Cleaned LLM world-building output (len: {len(cleaned_world_text)}):\nSTART_OF_WB_TEXT\n{cleaned_world_text}\nEND_OF_WB_TEXT")

    normalized_detail_key_map = {k.lower().replace(" ", "_"): v for k, v in WORLD_DETAIL_KEY_MAP_NORMALIZED_TO_INTERNAL.items()}
    parsed_llm_response = parse_hierarchical_structured_text(
        cleaned_world_text, WORLD_CATEGORY_HEADER_PATTERN, WORLD_ITEM_HEADER_PATTERN,
        normalized_detail_key_map, WORLD_DETAIL_LIST_INTERNAL_KEYS,
        overview_category_internal_key="_overview_" 
    )
    logger.debug(f"Result of parse_hierarchical_structured_text: {parsed_llm_response}")


    if not agent.world_building: agent.world_building = {"source":"llm_generated_or_merged", "_overview_":{}}

    if parsed_llm_response:
        for category_llm_raw, items_llm in parsed_llm_response.items():
            category_llm = category_llm_raw.lower().replace(" ", "_")
            if category_llm not in WORLD_CATEGORY_MAP_NORMALIZED_TO_INTERNAL.values() and category_llm != "_overview_":
                logger.warning(f"LLM returned unknown world category '{category_llm_raw}'. Skipping.")
                continue
            
            agent.world_building.setdefault(category_llm, {}) 

            if isinstance(items_llm, dict):
                if category_llm == "_overview_":
                    existing_overview_desc = agent.world_building[category_llm].get("description")
                    if utils._is_fill_in(existing_overview_desc) or not str(existing_overview_desc).strip():
                        llm_overview_desc = items_llm.get("description", items_llm.get("Description")) 
                        if llm_overview_desc and not utils._is_fill_in(llm_overview_desc):
                             agent.world_building[category_llm]["description"] = llm_overview_desc
                else: 
                    for item_name_llm, details_llm in items_llm.items():
                        if not isinstance(details_llm, dict): continue
                        
                        existing_item_details = agent.world_building[category_llm].get(item_name_llm, {})
                        is_new_or_all_fill_in = not existing_item_details or \
                            all(utils._is_fill_in(v) for k,v in existing_item_details.items() if k != "source")
                        
                        if is_new_or_all_fill_in:
                            agent.world_building[category_llm][item_name_llm] = details_llm
                            agent.world_building[category_llm][item_name_llm]["source"] = "llm_generated"
                        else: 
                            for detail_key_llm_raw, detail_val_llm in details_llm.items():
                                internal_detail_key = WORLD_DETAIL_KEY_MAP_NORMALIZED_TO_INTERNAL.get(
                                    detail_key_llm_raw.lower().replace(" ","_"), detail_key_llm_raw 
                                )
                                if utils._is_fill_in(existing_item_details.get(internal_detail_key)) or existing_item_details.get(internal_detail_key) is None:
                                    existing_item_details[internal_detail_key] = detail_val_llm
                            agent.world_building[category_llm][item_name_llm] = existing_item_details

        agent.world_building.pop("is_default", None)
        agent.world_building.pop("user_supplied_data", None) 
        if "source" not in agent.world_building or agent.world_building["source"] != "user_supplied_markdown":
            agent.world_building["source"] = "llm_generated_or_merged"
        logger.info("Successfully generated/updated initial world-building dictionary via LLM.")
    else: 
        logger.error("Failed to generate/parse a valid world-building dictionary via LLM. Existing or default world_building will be used.")
        if not agent.world_building.get("_overview_") or utils._is_fill_in(agent.world_building.get("_overview_",{}).get("description")):
            default_wb_overview_desc = agent.plot_outline.get("setting_description", "A default world setting.")
            if utils._is_fill_in(default_wb_overview_desc): default_wb_overview_desc = "A default world setting, to be detailed later."
            
            agent.world_building.setdefault("_overview_", {})["description"] = default_wb_overview_desc
            if "source" not in agent.world_building: agent.world_building["source"] = "default_fallback"
            if "is_default" not in agent.world_building : agent.world_building["is_default"] = True
            
    return agent.world_building, accumulated_usage_data if llm_was_called else None