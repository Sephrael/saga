# prompt_data_getters.py
"""
Helper functions to prepare specific data snippets for LLM prompts in the SAGA system.
These functions typically filter or format parts of the agent's state,
increasingly by querying Neo4j directly for richer, graph-aware context.
MODIFIED: To handle 'agent_or_props' for more flexible data access.
"""
import logging
import re
import asyncio
import copy
from typing import Dict, List, Optional, Set, Tuple, Any

import config
# from state_manager import state_manager # No longer directly used
from data_access import character_queries, world_queries, kg_queries # MODIFIED
from type import SceneDetail

logger = logging.getLogger(__name__)

def _get_prop(agent_or_props: Any, key: str, default: Any = None) -> Any:
    """Helper to get a property from an agent-like object or a dictionary."""
    if isinstance(agent_or_props, dict):
        return agent_or_props.get(key, default)
    return getattr(agent_or_props, key, default)

def _get_nested_prop(agent_or_props: Any, primary_key: str, secondary_key: str, default: Any = None) -> Any:
    """Helper to get a nested property."""
    primary_data = _get_prop(agent_or_props, primary_key, {})
    if isinstance(primary_data, dict):
        return primary_data.get(secondary_key, default)
    return default


async def get_character_state_snippet_for_prompt(agent_or_props: Any, current_chapter_num_for_filtering: Optional[int] = None) -> str:
    """
    Creates a concise plain text string of key character states for prompts,
    fetching data directly from Neo4j.
    MODIFIED: Uses agent_or_props and new data_access functions.
    """
    text_output_lines: List[str] = []
    char_names_to_process: List[str] = []

    plot_outline_data = _get_prop(agent_or_props, 'plot_outline', _get_prop(agent_or_props, 'plot_outline_full', {}))
    protagonist_name = _get_prop(plot_outline_data, "protagonist_name")

    character_profiles_data = _get_prop(agent_or_props, 'character_profiles', {})
    all_known_char_names_from_state: List[str] = list(character_profiles_data.keys())

    if protagonist_name and protagonist_name in all_known_char_names_from_state:
        char_names_to_process.append(protagonist_name)

    for name in sorted(all_known_char_names_from_state):
        if name != protagonist_name and len(char_names_to_process) < config.PLANNING_CONTEXT_MAX_CHARACTERS_IN_SNIPPET:
            char_names_to_process.append(name)
        elif len(char_names_to_process) >= config.PLANNING_CONTEXT_MAX_CHARACTERS_IN_SNIPPET:
            break

    effective_filter_chapter = (current_chapter_num_for_filtering - 1) \
        if current_chapter_num_for_filtering is not None and current_chapter_num_for_filtering > 0 \
        else config.KG_PREPOPULATION_CHAPTER_NUM

    fetch_tasks = [
        character_queries.get_character_info_for_snippet_from_db(name, effective_filter_chapter) # MODIFIED
        for name in char_names_to_process
    ]

    if not fetch_tasks:
        return "No character profiles available or applicable to fetch for snippet."

    character_info_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

    for i, res_or_exc in enumerate(character_info_results):
        char_name_for_snippet = char_names_to_process[i]
        text_output_lines.append(f"Character: {char_name_for_snippet}")

        if isinstance(res_or_exc, Exception):
            logger.error(f"Error fetching snippet info for character '{char_name_for_snippet}' from Neo4j: {res_or_exc}")
            profile_fallback = character_profiles_data.get(char_name_for_snippet, {}) 
            desc_fb = (str(profile_fallback.get("description", "Error fetching description."))[:config.PLANNING_CONTEXT_MAX_CHARS_PER_PROFILE_DESC]).strip() + "..."
            status_fb = str(profile_fallback.get("status", "Unknown (error fetching status)"))
            dev_note_fb = "Error fetching development note."
            dev_keys_fb = sorted(
                [k for k in profile_fallback if k.startswith("development_in_chapter_") and int(k.split('_')[-1]) <= effective_filter_chapter],
                key=lambda x: int(x.split('_')[-1]), reverse=True
            )
            if dev_keys_fb:
                dev_note_fb = (str(profile_fallback.get(dev_keys_fb[0], "N/A"))[:config.PLANNING_CONTEXT_MAX_RECENT_DEV_PER_PROFILE]).strip() + "..."
            text_output_lines.append(f"  Description Snippet: {desc_fb}")
            text_output_lines.append(f"  Current Status: {status_fb} (Error in Neo4j fetch, using local state)")
            text_output_lines.append(f"  Most Recent Development Note: {dev_note_fb}")
        elif res_or_exc is None:
            logger.warning(f"Neo4j character snippet fetch for '{char_name_for_snippet}' was None. Using local state as fallback.")
            profile_fallback = character_profiles_data.get(char_name_for_snippet, {}) 
            desc_fb = (str(profile_fallback.get("description", "N/A"))[:config.PLANNING_CONTEXT_MAX_CHARS_PER_PROFILE_DESC]).strip() + "..."
            status_fb = str(profile_fallback.get("status", "Unknown"))
            dev_note_fb = "N/A"
            dev_keys_fb = sorted(
                [k for k in profile_fallback if k.startswith("development_in_chapter_") and int(k.split('_')[-1]) <= effective_filter_chapter],
                key=lambda x: int(x.split('_')[-1]), reverse=True
            )
            if dev_keys_fb:
                dev_note_fb = (str(profile_fallback.get(dev_keys_fb[0], "N/A"))[:config.PLANNING_CONTEXT_MAX_RECENT_DEV_PER_PROFILE]).strip() + "..."
            text_output_lines.append(f"  Description Snippet: {desc_fb}")
            text_output_lines.append(f"  Current Status: {status_fb} (Neo4j fetch was None, using local state)")
            text_output_lines.append(f"  Most Recent Development Note: {dev_note_fb}")
        else:
            profile_summary = res_or_exc
            provisional_note = " (Note: Some info may be provisional)" if profile_summary.get("is_provisional_overall") else ""
            desc_text = str(profile_summary.get('description', 'N/A'))
            status_text = str(profile_summary.get('current_status', 'Unknown'))
            dev_note_text = str(profile_summary.get('most_recent_development_note', 'N/A'))
            text_output_lines.append(f"  Description Snippet: {(desc_text[:config.PLANNING_CONTEXT_MAX_CHARS_PER_PROFILE_DESC]).strip()}...")
            text_output_lines.append(f"  Current Status: {status_text}{provisional_note}")
            text_output_lines.append(f"  Most Recent Development Note: {(dev_note_text[:config.PLANNING_CONTEXT_MAX_RECENT_DEV_PER_PROFILE]).strip()}...")
        text_output_lines.append("")

    return "\n".join(text_output_lines).strip() if text_output_lines else "No character profiles available or applicable."


async def get_world_state_snippet_for_prompt(agent_or_props: Any, current_chapter_num_for_filtering: Optional[int] = None) -> str:
    """
    Creates a concise plain text string of key world states for prompts,
    fetching data directly from Neo4j.
    MODIFIED: Uses agent_or_props and new data_access functions.
    """
    text_output_lines: List[str] = []
    any_provisional_in_world_snippet = False

    effective_filter_chapter = (current_chapter_num_for_filtering - 1) \
        if current_chapter_num_for_filtering is not None and current_chapter_num_for_filtering > 0 \
        else config.KG_PREPOPULATION_CHAPTER_NUM

    world_categories_to_fetch = {
        "locations": config.PLANNING_CONTEXT_MAX_LOCATIONS_IN_SNIPPET,
        "systems": config.PLANNING_CONTEXT_MAX_SYSTEMS_IN_SNIPPET,
        "factions": config.PLANNING_CONTEXT_MAX_FACTIONS_IN_SNIPPET,
    }

    fetch_tasks_world = []
    category_names_for_tasks = []

    for category_name, max_items in world_categories_to_fetch.items():
        if max_items > 0:
            fetch_tasks_world.append(
                world_queries.get_world_elements_for_snippet_from_db(category_name, effective_filter_chapter, max_items) # MODIFIED
            )
            category_names_for_tasks.append(category_name)

    if not fetch_tasks_world:
        return "No significant world-building data available or applicable to fetch for snippet."

    world_building_data = _get_prop(agent_or_props, 'world_building', {}) 
    world_info_results = await asyncio.gather(*fetch_tasks_world, return_exceptions=True)

    for i, res_or_exc in enumerate(world_info_results):
        category_name_result = category_names_for_tasks[i]
        category_title = f"Key {category_name_result.capitalize()}:"
        category_items_lines = []

        if isinstance(res_or_exc, Exception):
            logger.error(f"Error fetching snippet info for world category '{category_name_result}' from Neo4j: {res_or_exc}")
            category_items_lines.append("  - Error fetching data.")
        elif not res_or_exc: # Empty list from DB query
            logger.warning(f"Neo4j world snippet fetch for '{category_name_result}' empty. Using local state as fallback.")
            category_data_fallback = world_building_data.get(category_name_result, {})
            item_count_fb = 0
            for item_name_fb, item_details_fb in sorted(category_data_fallback.items()):
                if item_count_fb >= world_categories_to_fetch[category_name_result]: break
                if item_name_fb.startswith(("_", "source_quality_chapter_", "category_updated_in_chapter_")): continue
                desc_snip_fb = ""
                if isinstance(item_details_fb, dict) and item_details_fb.get("description"):
                    desc_snip_fb = f": {str(item_details_fb['description'])[:50].strip()}..."
                category_items_lines.append(f"  - {item_name_fb}{desc_snip_fb}")
                item_count_fb +=1
            if item_count_fb == 0:
                 category_items_lines.append("  - None notable or available (from local state).")
        else: # Results from DB
            items_list = res_or_exc
            category_has_provisional = False
            for item_data in items_list:
                name = str(item_data.get("name", "Unknown Item"))
                desc_snip = str(item_data.get("description_snippet", ""))
                prov_item_note = " (provisional)" if item_data.get("is_provisional") else ""
                display_desc_snip = desc_snip.replace(name, '', 1).strip() if desc_snip.startswith(name) else desc_snip
                display_desc_snip = (': ' + display_desc_snip) if display_desc_snip else ''
                category_items_lines.append(f"  - {name}{prov_item_note}{display_desc_snip}")
                if item_data.get("is_provisional"):
                    category_has_provisional = True
                    any_provisional_in_world_snippet = True
            if not items_list:
                 category_items_lines.append("  - None notable or available (from Neo4j).")
            if category_has_provisional:
                category_title += " (Note: Some items may be provisional)"

        text_output_lines.append(category_title)
        text_output_lines.extend(category_items_lines)
        text_output_lines.append("")

    if any_provisional_in_world_snippet and text_output_lines:
         text_output_lines.insert(0, "Overall World Note: Some world information might be provisional from unrevised prior chapters.\n")

    return "\n".join(text_output_lines).strip() if text_output_lines else "No significant world-building data available or applicable."

def _format_dict_for_plain_text_prompt(data: Dict[str, Any], indent_level: int = 0, name_override: Optional[str] = None) -> List[str]:
    lines = []
    indent = "  " * indent_level
    if name_override:
        lines.append(f"{indent}{name_override}:")
        indent_level +=1
        indent = "  " * indent_level
    priority_keys = ['description', 'status', 'traits', 'relationships', 'goals', 'rules', 'key_elements', 'atmosphere']
    sorted_keys = []
    remaining_keys = list(data.keys())
    for p_key in priority_keys:
        if p_key in remaining_keys:
            sorted_keys.append(p_key)
            remaining_keys.remove(p_key)
    sorted_keys.extend(sorted(remaining_keys))
    for key in sorted_keys:
        value = data[key]
        # Skip internal/metadata keys unless it's 'prompt_notes'
        if key.startswith(("source_quality_chapter_", "updated_in_chapter_", "added_in_chapter_", "is_provisional", "is_provisional_hint")) \
           and key != "prompt_notes": # Specifically allow prompt_notes
            continue
        if key == "prompt_notes_skip_this_key": # Another example of a skip key
            continue

        key_str = str(key).replace("_", " ").capitalize()
        if isinstance(value, dict):
            if value: # Only print header if dict is not empty
                lines.append(f"{indent}{key_str}:")
                lines.extend(_format_dict_for_plain_text_prompt(value, indent_level + 1))
        elif isinstance(value, list):
            if not value:
                lines.append(f"{indent}{key_str}: (empty list or N/A)")
            else:
                lines.append(f"{indent}{key_str}:")
                try: display_items = sorted([str(x) for x in value])
                except TypeError: display_items = [str(x) for x in value] # Fallback if not sortable
                for item in display_items:
                    if isinstance(item, dict): # Handle list of dicts
                         lines.extend(_format_dict_for_plain_text_prompt(item, indent_level + 2, name_override="- Item"))
                    else: lines.append(f"{indent}  - {str(item)}")
        elif value is not None and str(value).strip(): # Ensure value is not just whitespace
             lines.append(f"{indent}{key_str}: {str(value)}")
    return lines

def _add_provisional_notes_and_filter_developments(
    item_data_original: Dict[str, Any],
    up_to_chapter_inclusive: Optional[int],
    is_character: bool = True
) -> Dict[str, Any]:
    item_data = copy.deepcopy(item_data_original)
    prompt_notes_list = []
    effective_filter_chapter = config.KG_PREPOPULATION_CHAPTER_NUM if up_to_chapter_inclusive == 0 else up_to_chapter_inclusive
    
    dev_elaboration_prefix = "development_in_chapter_" if is_character else "elaboration_in_chapter_"
    added_prefix = "added_in_chapter_"
    
    keys_to_remove = []
    has_provisional_data_relevant_to_filter = False

    for key in list(item_data.keys()): # Iterate over copy of keys for safe removal
        # Filter future development/elaboration/added keys
        if key.startswith((dev_elaboration_prefix, added_prefix)):
            try:
                chap_num_of_key = int(key.split("_")[-1])
                if effective_filter_chapter is not None and chap_num_of_key > effective_filter_chapter:
                    keys_to_remove.append(key)
            except ValueError:
                logger.warning(f"Could not parse chapter from key '{key}' during filtering.")
        
        # Check for provisional data source notes
        if key.startswith("source_quality_chapter_"):
            try:
                chap_num_of_source = int(key.split("_")[-1])
                # Only consider provisional notes relevant up to the filter chapter
                if (effective_filter_chapter is None or chap_num_of_source <= effective_filter_chapter) and \
                   item_data[key] == "provisional_from_unrevised_draft":
                    has_provisional_data_relevant_to_filter = True
                    note = f"Data from Chapter {chap_num_of_source} may be provisional (from unrevised draft)."
                    if note not in prompt_notes_list:
                        prompt_notes_list.append(note)
            except ValueError:
                logger.warning(f"Could not parse chapter from source_quality key '{key}'.")

    for k_rem in keys_to_remove:
        item_data.pop(k_rem, None)
    
    if has_provisional_data_relevant_to_filter:
        item_data["is_provisional_hint"] = True # Internal hint for formatting
        
    if prompt_notes_list:
        item_data["prompt_notes"] = prompt_notes_list # This will be formatted by _format_dict_for_plain_text_prompt
        
    return item_data


async def _get_character_profiles_dict_with_notes(agent_or_props: Any, up_to_chapter_inclusive: Optional[int]) -> Dict[str, Any]:
    logger.debug(f"Internal: Getting character profiles dict with notes up to chapter {up_to_chapter_inclusive}.")
    processed_profiles: Dict[str, Any] = {}
    filter_chapter = config.KG_PREPOPULATION_CHAPTER_NUM if up_to_chapter_inclusive == 0 else up_to_chapter_inclusive
    character_profiles_data = _get_prop(agent_or_props, 'character_profiles', {})

    if not character_profiles_data:
        logger.warning("Character_profiles dictionary is empty.")
        return {}
    for char_name, profile_original in character_profiles_data.items():
        if not isinstance(profile_original, dict):
            logger.warning(f"Character profile for '{char_name}' is not a dict. Skipping.")
            continue
        processed_profiles[char_name] = _add_provisional_notes_and_filter_developments(
            profile_original, filter_chapter, is_character=True
        )
    return processed_profiles

async def get_filtered_character_profiles_for_prompt_plain_text(agent_or_props: Any, up_to_chapter_inclusive: Optional[int] = None) -> str:
    logger.info(f"Fetching and formatting filtered character profiles as PLAIN TEXT up to chapter {up_to_chapter_inclusive}.")
    profiles_dict_with_notes = await _get_character_profiles_dict_with_notes(agent_or_props, up_to_chapter_inclusive)
    if not profiles_dict_with_notes: return "No character profiles available."
    output_lines = ["Key Character Profiles:"]
    for char_name in sorted(profiles_dict_with_notes.keys()):
        profile_data = profiles_dict_with_notes[char_name]
        if not isinstance(profile_data, dict) or not profile_data: continue
        output_lines.append("")
        formatted_profile_lines = _format_dict_for_plain_text_prompt(profile_data, indent_level=1, name_override=char_name)
        output_lines.extend(formatted_profile_lines)
    return "\n".join(output_lines).strip()


async def _get_world_data_dict_with_notes(agent_or_props: Any, up_to_chapter_inclusive: Optional[int]) -> Dict[str, Any]:
    logger.debug(f"Internal: Getting world data dict with notes up to chapter {up_to_chapter_inclusive}.")
    processed_world_data: Dict[str, Any] = {}
    filter_chapter = config.KG_PREPOPULATION_CHAPTER_NUM if up_to_chapter_inclusive == 0 else up_to_chapter_inclusive
    world_building_data = _get_prop(agent_or_props, 'world_building', {})

    if not world_building_data:
        logger.warning("World_building dictionary is empty.")
        return {}
    for category_name, category_items_original in world_building_data.items():
        if not isinstance(category_items_original, dict) and category_name not in ["is_default", "source", "user_supplied_data"]:
            logger.warning(f"World category '{category_name}' content is not a dict (type: {type(category_items_original)}). Skipping formatting.")
            processed_world_data[category_name] = {} # Ensure key exists but empty
            continue
        if category_name in ["is_default", "source", "user_supplied_data"]:
             processed_world_data[category_name] = category_items_original # Copy metadata as is
             continue
        
        processed_category: Dict[str, Any] = {}
        if category_name == "_overview_": # Overview is a single dict of details
             processed_category = _add_provisional_notes_and_filter_developments(
                category_items_original, filter_chapter, is_character=False
            )
        else: # Other categories contain item_name: item_details pairs
            for item_name, item_data_original in category_items_original.items():
                if not isinstance(item_data_original, dict):
                    logger.warning(f"World item '{item_name}' in category '{category_name}' is not a dict. Skipping.")
                    continue
                processed_category[item_name] = _add_provisional_notes_and_filter_developments(
                    item_data_original, filter_chapter, is_character=False
                )
        processed_world_data[category_name] = processed_category
    return processed_world_data

async def get_filtered_world_data_for_prompt_plain_text(agent_or_props: Any, up_to_chapter_inclusive: Optional[int] = None) -> str:
    logger.info(f"Fetching and formatting filtered world data as PLAIN TEXT up to chapter {up_to_chapter_inclusive}.")
    world_data_dict_with_notes = await _get_world_data_dict_with_notes(agent_or_props, up_to_chapter_inclusive)
    if not world_data_dict_with_notes: return "No world-building data available."
    
    output_lines = []
    overview_data = world_data_dict_with_notes.get("_overview_")
    if overview_data and isinstance(overview_data, dict) and overview_data.get("description"): # Check if overview has content
        output_lines.append("World-Building Overview:")
        output_lines.extend(_format_dict_for_plain_text_prompt(overview_data, indent_level=1))
        output_lines.append("") # Add a blank line after overview
    
    sorted_categories = sorted([cat for cat in world_data_dict_with_notes.keys() if cat not in ["_overview_", "is_default", "source", "user_supplied_data"]])
    
    for category_name in sorted_categories:
        category_items = world_data_dict_with_notes[category_name]
        if not isinstance(category_items, dict) or not category_items: continue # Skip empty categories
        
        category_header_added = False
        for item_name in sorted(category_items.keys()):
            item_data = category_items[item_name]
            if not isinstance(item_data, dict) or not item_data: continue # Skip empty items

            if not category_header_added: # Add category header only if there's content
                output_lines.append(f"{category_name.replace('_', ' ').capitalize()}:")
                category_header_added = True

            item_lines = _format_dict_for_plain_text_prompt(item_data, indent_level=1, name_override=item_name)
            output_lines.extend(item_lines)
            if item_lines: output_lines.append("") # Blank line after each item's details
        
        # Clean up trailing blank line if it was the last thing added for this category
        if output_lines and output_lines[-1] == "" and category_header_added:
             pass # Keep the blank line to separate categories
        elif category_header_added: # If header was added but no items, or no blank line after last item
            output_lines.append("") # Ensure a blank line after a non-empty category

    if not output_lines: return "No significant world-building data available after filtering."
    return "\n".join(output_lines).strip()


async def heuristic_entity_spotter_for_kg(agent_or_props: Any, text_snippet: str) -> List[str]:
    character_profiles_data = _get_prop(agent_or_props, 'character_profiles', {})
    world_building_data = _get_prop(agent_or_props, 'world_building', {})

    entities = set(character_profiles_data.keys()) # Start with known character names
    
    # Regex for capitalized words/phrases (potential proper nouns)
    for match in re.finditer(r'\b([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]+){0,2})\b', text_snippet):
        entities.add(match.group(1).strip())
        
    # Add known world-building item names
    for category, items in world_building_data.items():
        if isinstance(items, dict) and category not in ["_overview_", "is_default", "source", "user_supplied_data"]:
            for item_name in items.keys():
                if isinstance(item_name, str) and item_name.strip(): # Ensure item_name is a valid string
                     entities.add(item_name.strip())

    common_false_positives = {"The", "A", "An", "Is", "It", "He", "She", "They", "Chapter", "Section", "I", "We", "You"}
    # Filter out common words and very short strings, unless they are known entities
    return sorted([e for e in list(entities)
                   if (len(e) > 2 or e in character_profiles_data or e in world_building_data.get(e, {})) and e not in common_false_positives])


async def get_reliable_kg_facts_for_drafting_prompt(
    agent_or_props: Any,
    chapter_number: int,
    chapter_plan: Optional[List[SceneDetail]] = None,
    max_facts_per_char: int = 2,
    max_total_facts: int = 7
) -> str:
    if chapter_number <= 0: return "No KG facts applicable for pre-first chapter."
    
    kg_chapter_limit = max(config.KG_PREPOPULATION_CHAPTER_NUM, chapter_number - 1)
    facts_for_prompt: List[str] = []

    plot_outline_data = _get_prop(agent_or_props, 'plot_outline', _get_prop(agent_or_props, 'plot_outline_full', {}))
    protagonist_name = _get_nested_prop(agent_or_props, 'plot_outline_full', 'protagonist_name', config.DEFAULT_PROTAGONIST_NAME) \
                       or _get_prop(plot_outline_data, 'protagonist_name', config.DEFAULT_PROTAGONIST_NAME)

    novel_id = config.MAIN_NOVEL_INFO_NODE_ID
    characters_of_interest: Set[str] = {protagonist_name} if protagonist_name else set()

    if chapter_plan and isinstance(chapter_plan, list):
        for scene_detail in chapter_plan:
            if isinstance(scene_detail, dict) and "characters_involved" in scene_detail and isinstance(scene_detail["characters_involved"], list):
                for char_name_in_plan in scene_detail["characters_involved"]:
                    if isinstance(char_name_in_plan, str) and char_name_in_plan.strip():
                        characters_of_interest.add(char_name_in_plan.strip())
    
    logger.debug(f"KG fact gathering for Ch {chapter_number} draft: Chars of interest: {characters_of_interest}, KG chapter limit: {kg_chapter_limit}")

    # Novel-level context
    novel_context_queries_params = [
        (f"MATCH (ni:NovelInfo {{id: $novel_id_param}}) RETURN ni.theme AS value, 'The novel\\'s theme' AS description", {"novel_id_param": novel_id}, "theme"),
        (f"MATCH (ni:NovelInfo {{id: $novel_id_param}}) RETURN ni.conflict_summary AS value, 'The main conflict' AS description", {"novel_id_param": novel_id}, "conflict_summary")
    ]
    for query, params, desc_key in novel_context_queries_params:
        if len(facts_for_prompt) >= max_total_facts: break
        try:
            res = await kg_queries.query_kg_from_db(subject=novel_id, predicate=desc_key) # Simplified query
            if res and res[0] and res[0].get('object'): # Assuming query_kg returns object
                facts_for_prompt.append(f"- {res[0]['description']} is: {res[0]['object']}.")
        except Exception as e: logger.warning(f"KG Query for novel context '{desc_key}' failed: {e}")

    # Character-specific facts
    for char_name in list(characters_of_interest)[:3]: # Limit characters to process for brevity
        if len(facts_for_prompt) >= max_total_facts: break
        facts_for_this_char = 0

        # Get status directly from Character node property
        char_node_status_query = "MATCH (c:Character:Entity {name: $char_name_param}) WHERE (c.is_provisional IS NULL OR c.is_provisional = FALSE) RETURN c.status AS status_value LIMIT 1"
        try:
            status_res = await kg_queries.query_kg_from_db(subject=char_name, predicate="has_status") # Example predicate
            if status_res and status_res[0] and status_res[0].get('object') and facts_for_this_char < max_facts_per_char:
                facts_for_prompt.append(f"- {char_name}'s status is: {status_res[0]['object']}.")
                facts_for_this_char += 1
        except Exception as e: logger.warning(f"KG Query for '{char_name}' status from node property failed: {e}")

        # Get status from DYNAMIC_REL if node property fails or to augment
        if facts_for_this_char < max_facts_per_char and len(facts_for_prompt) < max_total_facts:
            status_val_kg = await kg_queries.get_most_recent_value_from_db(char_name, "status_is", kg_chapter_limit, include_provisional=False)
            if status_val_kg:
                facts_for_prompt.append(f"- {char_name}'s status (from KG relation) is: {status_val_kg}.")
                facts_for_this_char += 1
        
        # Get location
        if facts_for_this_char < max_facts_per_char and len(facts_for_prompt) < max_total_facts:
            loc_val = await kg_queries.get_most_recent_value_from_db(char_name, "located_in", kg_chapter_limit, include_provisional=False)
            if loc_val:
                facts_for_prompt.append(f"- {char_name} is located in: {loc_val}.")
                facts_for_this_char += 1
        
        # Get a key relationship
        if facts_for_this_char < max_facts_per_char and len(facts_for_prompt) < max_total_facts:
            rel_query_results = await kg_queries.query_kg_from_db(
                subject=char_name, 
                chapter_limit=kg_chapter_limit, 
                include_provisional=False, 
                limit_results=1 # Get one most recent, non-provisional relation
            )
            # Filter for specific interesting relationship types client-side if needed
            interesting_rel_types = ['ally_of', 'enemy_of', 'mentor_of', 'protege_of', 'related_to']
            for rel_res in rel_query_results:
                if rel_res.get('predicate') in interesting_rel_types:
                    rel_type_display = rel_res['predicate'].replace('_', ' ')
                    facts_for_prompt.append(f"- {char_name} has a key relationship ({rel_type_display}) with: {rel_res.get('object')}.")
                    facts_for_this_char +=1
                    break # Take only one key relationship per character for this snippet
            
    if not facts_for_prompt:
        return "No specific reliable KG facts identified as highly relevant for this chapter's current focus from Neo4j."
    
    unique_facts = sorted(list(set(facts_for_prompt)))
    return "**Key Reliable KG Facts (from Neo4j - up to previous chapter/initial state):**\n" + "\n".join(unique_facts[:max_total_facts])