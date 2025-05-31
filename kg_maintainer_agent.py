# kg_maintainer_agent.py
import logging
import re
import asyncio
from typing import Dict, List, Optional, Any, Tuple

from async_lru import alru_cache

import config
import llm_interface
from core_db.base_db_manager import neo4j_manager # For execute_cypher_batch
from data_access import kg_queries # For add_kg_triple_to_db
from prompt_data_getters import (
    get_filtered_character_profiles_for_prompt_plain_text,
    heuristic_entity_spotter_for_kg,
    get_filtered_world_data_for_prompt_plain_text
)
from parsing_utils import parse_key_value_block, parse_hierarchical_structured_text, parse_kg_triples_from_text

logger = logging.getLogger(__name__)

# --- Parsing Constants (unchanged) ---
CHAR_UPDATE_KEY_MAP = {
    "description": "description", "traits": "traits", "status": "status",
    "relationships": "relationships", "modification_proposal": "modification_proposal",
    re.compile(r"development_in_chapter_\d+"): lambda match: match.group(0).lower()
}
CHAR_UPDATE_LIST_INTERNAL_KEYS = ["traits"]
CHAR_UPDATE_PKVB_SPECIAL_HANDLING = {
    "traits": {"separator": ","}
}
CHAR_UPDATE_RELATIONSHIP_POST_PARSING_HANDLING = { # Not directly used by parse_key_value_block, but for conceptual grouping
    "relationships": {"separator": ";", "item_format": "target:type"}
}

WORLD_UPDATE_CATEGORY_PATTERN = re.compile(r"^\s*Category:\s*([A-Za-z\s_]+?)\s*$", re.IGNORECASE | re.MULTILINE)
WORLD_UPDATE_ITEM_PATTERN = re.compile(r"^\s*Item:\s*([A-Za-z0-9\s'\-]+?)\s*$", re.IGNORECASE | re.MULTILINE)
WORLD_UPDATE_DETAIL_KEY_MAP = {
    "description": "description", "atmosphere": "atmosphere",
    "goals": "goals", "rules": "rules", "key_elements": "key_elements", "traits": "traits",
    "modification_proposal": "modification_proposal",
    re.compile(r"elaboration_in_chapter_\d+"): lambda match: match.group(0).lower()
}
WORLD_UPDATE_DETAIL_LIST_INTERNAL_KEYS = ["goals", "rules", "key_elements", "traits"]


# --- Summarization (Cached) ---
@alru_cache(maxsize=config.SUMMARY_CACHE_SIZE)
async def _llm_summarize_full_chapter_text_logic_internal(chapter_text_full_key: str, chapter_number: int) -> Tuple[str, Optional[Dict[str, int]]]:
    """ Summarizes chapter text using an LLM. Input `chapter_text_full_key` is the actual text. Returns summary and usage."""
    prompt_lines = [
        "/no_think",
        "You are a concise summarizer. Summarize the key events, character developments, and plot advancements from the following Chapter {chapter_number} text.",
        "The summary should be 1-3 sentences long and capture the most crucial information.",
        "Focus on what changed or was revealed.",
        "",
        "Full Chapter Text:",
        "--- BEGIN TEXT ---",
        chapter_text_full_key,
        "--- END TEXT ---",
        "",
        "Output ONLY the summary text. No extra commentary or \"Summary:\" prefix."
    ]
    prompt = "\n".join(prompt_lines)

    summary_raw, usage_data = await llm_interface.async_call_llm(
        model_name=config.SMALL_MODEL,
        prompt=prompt,
        temperature=0.6,
        max_tokens=config.MAX_SUMMARY_TOKENS,
        stream_to_disk=False
    )
    return llm_interface.clean_model_response(summary_raw).strip(), usage_data

# --- State Modification Helpers ---
def _apply_trait_modification(current_traits_list: List[str], modification_details_str: str) -> List[str]:
    traits_set = set(current_traits_list)
    for add_match in re.finditer(r"ADD\s+['\"]([^'\"]+)['\"]", modification_details_str, re.IGNORECASE):
        trait_to_add = add_match.group(1).strip()
        if trait_to_add: traits_set.add(trait_to_add)
    for remove_match in re.finditer(r"REMOVE\s+['\"]([^'\"]+)['\"]", modification_details_str, re.IGNORECASE):
        trait_to_remove = remove_match.group(1).strip()
        if trait_to_remove: traits_set.discard(trait_to_remove)
    return sorted(list(traits_set))

def _parse_modification_proposal_str(proposal_str: str) -> Optional[Tuple[str, str]]:
    """Parses 'MODIFY key: value' string. Returns (key, value_modification_str) or None."""
    if not isinstance(proposal_str, str) or not proposal_str.strip():
        return None
    match = re.match(r"MODIFY\s+([\w_]+)\s*:(.*)", proposal_str, re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip(), match.group(2).strip()

def _apply_state_modification_proposal_logic(
    target_dict: Dict[str, Any],
    proposal_str: str,
    item_name_for_log: str,
    item_type_for_log: str
) -> bool:
    """Applies a modification proposal to target_dict. Returns True if modified."""
    parsed_proposal = _parse_modification_proposal_str(proposal_str)
    if not parsed_proposal:
        logger.debug(f"Empty or invalid modification proposal for '{item_name_for_log}'. Proposal: '{proposal_str}'")
        return False

    key_name_from_proposal_upper, value_modification_str = parsed_proposal
    original_key_name = next(
        (k for k in target_dict if k.upper() == key_name_from_proposal_upper.upper()), # Case-insensitive match
        key_name_from_proposal_upper.lower() # Default to lowercase if no exact match
    )

    logger.debug(f"Applying modification proposal for '{item_name_for_log}' ({item_type_for_log}): Key='{original_key_name}', Modification='{value_modification_str}'")
    modified = False
    try:
        if original_key_name.lower() == "traits":
            if "traits" not in target_dict or not isinstance(target_dict["traits"], list):
                target_dict["traits"] = []
            old_traits = list(target_dict["traits"])
            target_dict["traits"] = _apply_trait_modification(target_dict["traits"], value_modification_str)
            if target_dict["traits"] != old_traits:
                modified = True
                logger.info(f"Applied trait modifications for '{item_name_for_log}'. New traits: {target_dict['traits']}")
        else:
            new_value_str = value_modification_str.strip("'\" ")
            if new_value_str:
                if target_dict.get(original_key_name) != new_value_str:
                    target_dict[original_key_name] = new_value_str
                    modified = True
                    logger.info(f"Applied modification to '{original_key_name}' for '{item_name_for_log}'. New value: '{new_value_str[:70]}...'")
            else:
                logger.warning(f"Modification proposal for '{original_key_name}' of '{item_name_for_log}' resulted in an empty new value. Proposal: '{proposal_str}'")
    except Exception as e:
        logger.error(f"Error applying modification proposal for '{item_name_for_log}': {e}. Key: {original_key_name}, Proposal: '{proposal_str}'", exc_info=True)
    return modified

# --- Character Profile Merging Helpers ---
def _initialize_new_character_profile_internal(
    char_name: str,
    char_update_data: Dict[str, Any],
    chapter_number: int,
    provisional_marker_key: str,
    dev_key: str
) -> Dict[str, Any]:
    new_profile: Dict[str, Any] = {
        "description": char_update_data.get("description", f"A character newly introduced in Chapter {chapter_number}."),
        "traits": sorted(list(set(t for t in char_update_data.get("traits", []) if isinstance(t, str) and t.strip()))),
        "relationships": char_update_data.get("relationships", {}),
        "status": char_update_data.get("status", "Newly introduced")
    }
    if dev_key in char_update_data and char_update_data[dev_key]:
        new_profile[dev_key] = char_update_data[dev_key]
    elif dev_key not in char_update_data :
         new_profile[dev_key] = f"Character '{char_name}' introduced or significantly involved in Chapter {chapter_number}."

    if provisional_marker_key in char_update_data:
        new_profile[provisional_marker_key] = char_update_data[provisional_marker_key]

    logger.info(f"Prepared new character profile for '{char_name}'.")
    return new_profile

def _update_existing_character_profile_fields_internal(
    existing_profile: Dict[str, Any],
    char_update_data: Dict[str, Any],
    dev_key: str,
    provisional_marker_key: str,
    all_character_profiles_for_lookup: Dict[str, Any] # Used to get char_name for logging if needed
) -> bool:
    """Updates existing profile fields. Returns True if any field was actually changed or added."""
    modified = False
    original_profile_copy = {k: (list(v) if isinstance(v, list) else dict(v) if isinstance(v, dict) else v) for k, v in existing_profile.items()}


    if provisional_marker_key in char_update_data and existing_profile.get(provisional_marker_key) != char_update_data[provisional_marker_key]:
        existing_profile[provisional_marker_key] = char_update_data[provisional_marker_key]
        modified = True

    for key, value_from_update in char_update_data.items():
        if key in ["modification_proposal", provisional_marker_key] or key.startswith("development_in_chapter_"):
            if key.startswith("development_in_chapter_") and value_from_update and isinstance(value_from_update, str) and value_from_update.strip():
                 if existing_profile.get(key) != value_from_update:
                    existing_profile[key] = value_from_update
                    modified = True
            continue

        if key == "traits" and isinstance(value_from_update, list):
            if "traits" not in existing_profile or not isinstance(existing_profile["traits"], list):
                existing_profile["traits"] = []
            original_traits_set = set(existing_profile["traits"])
            valid_new_traits = {t for t in value_from_update if isinstance(t, str) and t.strip()}
            new_traits_set = original_traits_set.union(valid_new_traits)
            if new_traits_set != original_traits_set or len(new_traits_set) != len(original_traits_set): # Check for actual change
                existing_profile["traits"] = sorted(list(new_traits_set))
                modified = True
        elif key == "relationships" and isinstance(value_from_update, dict):
            if not isinstance(existing_profile.get("relationships"), dict):
                existing_profile["relationships"] = {}
            for rel_target, rel_detail in value_from_update.items():
                if existing_profile["relationships"].get(rel_target) != rel_detail:
                    existing_profile["relationships"][rel_target] = rel_detail
                    modified = True
        elif isinstance(value_from_update, str) and value_from_update.strip():
            if existing_profile.get(key) != value_from_update:
                existing_profile[key] = value_from_update
                modified = True
        elif key not in existing_profile and value_from_update is not None: # Adding a new key with a non-None value
            existing_profile[key] = value_from_update
            modified = True
        # If value_from_update is None, we don't typically remove keys, just ignore.

    # Add/update development key if significant changes were made or if it was explicitly provided
    if dev_key in char_update_data and isinstance(char_update_data[dev_key], str) and char_update_data[dev_key].strip():
        if existing_profile.get(dev_key) != char_update_data[dev_key]:
            existing_profile[dev_key] = char_update_data[dev_key]
            modified = True
    elif modified and (not existing_profile.get(dev_key) or not str(existing_profile[dev_key]).strip()):
        # If other fields were modified, ensure a basic dev note exists
        char_name = next((k_name for k_name, profile_val in all_character_profiles_for_lookup.items() if profile_val is existing_profile), "UnknownCharacter")
        new_dev_note = f"Character '{char_name}' updated or significantly involved in Chapter {existing_profile.get(provisional_marker_key, 'N/A').split('_')[-1]}."
        if existing_profile.get(dev_key) != new_dev_note:
            existing_profile[dev_key] = new_dev_note
            modified = True # Count this as a modification as well
    
    return modified


def _merge_character_profile_updates_into_state_internal(
    character_profiles_dict_to_update: Dict[str, Any],
    updates_from_llm: Dict[str, Any],
    chapter_number: int,
    from_flawed_draft: bool
):
    if not updates_from_llm:
        logger.info(f"No character profile updates from LLM to merge for ch {chapter_number}.")
        return

    logger.info(f"Merging character profile updates for ch {chapter_number}. Characters in update: {list(updates_from_llm.keys())}")
    updated_chars_count, new_chars_count = 0, 0
    provisional_marker_key = f"source_quality_chapter_{chapter_number}"

    for char_name, char_update_data_original in updates_from_llm.items():
        if not isinstance(char_update_data_original, dict):
            logger.warning(f"Skipping invalid character update data for '{char_name}' (not a dict). Data: {char_update_data_original}")
            continue

        char_update_data = char_update_data_original.copy()
        if from_flawed_draft:
            char_update_data[provisional_marker_key] = "provisional_from_unrevised_draft"

        dev_key = f"development_in_chapter_{chapter_number}"
        modification_proposal = char_update_data.get("modification_proposal")

        if char_name not in character_profiles_dict_to_update:
            new_chars_count += 1
            new_profile = _initialize_new_character_profile_internal(char_name, char_update_data, chapter_number, provisional_marker_key, dev_key)
            character_profiles_dict_to_update[char_name] = new_profile
            if config.ENABLE_DYNAMIC_STATE_ADAPTATION and modification_proposal:
                _apply_state_modification_proposal_logic(character_profiles_dict_to_update[char_name], modification_proposal, char_name, "new character profile")
        else:
            existing_profile = character_profiles_dict_to_update[char_name]
            was_modified_by_proposal = False
            if config.ENABLE_DYNAMIC_STATE_ADAPTATION and modification_proposal:
                was_modified_by_proposal = _apply_state_modification_proposal_logic(existing_profile, modification_proposal, char_name, "existing character profile")

            was_modified_by_fields = _update_existing_character_profile_fields_internal(existing_profile, char_update_data, dev_key, provisional_marker_key, character_profiles_dict_to_update)
            
            if was_modified_by_proposal or was_modified_by_fields:
                updated_chars_count += 1

            # Ensure provisional marker is correctly set if from_flawed_draft
            if from_flawed_draft and existing_profile.get(provisional_marker_key) != "provisional_from_unrevised_draft":
                 existing_profile[provisional_marker_key] = "provisional_from_unrevised_draft"
                 if not (was_modified_by_proposal or was_modified_by_fields): # Avoid double counting if already modified
                     updated_chars_count +=1


    if updated_chars_count > 0 or new_chars_count > 0:
        logger.info(f"Character profile dict merge complete for ch {chapter_number}. Updated: {updated_chars_count}, New: {new_chars_count}.")
    else:
        logger.info(f"No character profiles were effectively updated or added to dict for ch {chapter_number}.")


# --- World Building Merging Helpers ---
def _robust_merge_world_item_data_logic_internal(
    target_dict: Dict[str, Any],
    update_dict: Dict[str, Any],
    item_name_for_log: str,
    chapter_num: int,
    from_flawed_draft_source: bool
) -> bool: # Returns True if target_dict was modified
    if not isinstance(target_dict, dict):
        logger.warning(f"World item '{item_name_for_log}' target_dict was not a dict. Initializing as new. Old: '{str(target_dict)[:100]}'")
        # This function modifies target_dict in place. If it's not a dict, this is problematic.
        # The caller should ensure target_dict is a dict before calling.
        # For safety, let's not proceed if target_dict is not a dict.
        return False

    item_was_modified_this_call = False
    provisional_marker_key = f"source_quality_chapter_{chapter_num}"

    if from_flawed_draft_source and target_dict.get(provisional_marker_key) != "provisional_from_unrevised_draft":
        target_dict[provisional_marker_key] = "provisional_from_unrevised_draft"
        item_was_modified_this_call = True

    if config.ENABLE_DYNAMIC_STATE_ADAPTATION and "modification_proposal" in update_dict:
        proposal = update_dict.pop("modification_proposal")
        if isinstance(proposal, str) and proposal.strip():
            if _apply_state_modification_proposal_logic(target_dict, proposal, item_name_for_log, "world item"):
                item_was_modified_this_call = True

    for key, value_from_update in update_dict.items():
        if key in [provisional_marker_key, "modification_proposal"] or \
           key.startswith(("updated_in_chapter_", "added_in_chapter_", "source_quality_chapter_")): # Prevent direct write of these metadata keys
            if key.startswith("elaboration_in_chapter_") and isinstance(value_from_update, str) and value_from_update.strip():
                 if target_dict.get(key) != value_from_update:
                    target_dict[key] = value_from_update
                    item_was_modified_this_call = True
            continue

        current_value_in_target = target_dict.get(key)

        if isinstance(value_from_update, dict):
            if not isinstance(current_value_in_target, dict):
                target_dict[key] = {}
                current_value_in_target = target_dict[key]
                item_was_modified_this_call = True # New sub-dict created

            # Recursive call for nested dicts
            if _robust_merge_world_item_data_logic_internal(
                current_value_in_target, value_from_update, f"{item_name_for_log}.{key}", chapter_num, from_flawed_draft_source
            ):
                item_was_modified_this_call = True

        elif isinstance(value_from_update, list):
            if not isinstance(current_value_in_target, list):
                target_dict[key] = []
                current_value_in_target = target_dict[key]
                item_was_modified_this_call = True # New list created

            initial_list_len = len(current_value_in_target)
            current_list_set = set(current_value_in_target) # For efficient "not in" check
            added_to_list = False
            for item_in_list_update in value_from_update:
                if item_in_list_update not in current_list_set:
                    current_value_in_target.append(item_in_list_update)
                    current_list_set.add(item_in_list_update) # Keep set in sync
                    added_to_list = True
            if added_to_list:
                item_was_modified_this_call = True
        elif value_from_update != current_value_in_target: # Simple value update
            target_dict[key] = value_from_update
            item_was_modified_this_call = True

    if item_was_modified_this_call:
        # Ensure an "updated_in_chapter_X" key is set if not marked as "added_in_chapter_X"
        # (added_in_chapter is set by the caller when item is first created)
        added_key_pattern = re.compile(r"added_in_chapter_(\d+)")
        is_already_marked_added = any(added_key_pattern.match(k) for k in target_dict)

        if not is_already_marked_added:
            update_marker = f"updated_in_chapter_{chapter_num}"
            if not target_dict.get(update_marker): # Set only if not already true
                 target_dict[update_marker] = True
                 # item_was_modified_this_call is already true if we are here

    return item_was_modified_this_call


def _merge_world_item_updates_into_state_internal(
    world_building_dict_to_update: Dict[str, Any],
    updates_from_llm: Dict[str, Any],
    chapter_number: int,
    from_flawed_draft: bool
):
    if not updates_from_llm:
        logger.info(f"No world-building updates from LLM to merge for ch {chapter_number}.")
        return

    logger.info(f"Merging world-building updates for ch {chapter_number}. Categories in update: {list(updates_from_llm.keys())}")
    items_affected_count = 0

    for category_key, category_updates_dict in updates_from_llm.items():
        if not isinstance(category_updates_dict, dict) or not category_updates_dict:
            logger.debug(f"Skipping empty or invalid update for world category '{category_key}' in ch {chapter_number}.")
            continue

        if category_key not in world_building_dict_to_update:
            world_building_dict_to_update[category_key] = {}
        elif not isinstance(world_building_dict_to_update[category_key], dict):
            logger.warning(f"Overwriting non-dictionary world category '{category_key}' with new dictionary structure for ch {chapter_number}.")
            world_building_dict_to_update[category_key] = {}

        target_category_dict = world_building_dict_to_update[category_key]

        if category_key == "_overview_":
            item_log_name = "_overview_"
            # For _overview_, target_category_dict is the item itself (which is world_building_dict_to_update[category_key])
            if _robust_merge_world_item_data_logic_internal(
                 target_category_dict, category_updates_dict, item_log_name, chapter_number, from_flawed_draft
            ):
                items_affected_count +=1
            continue

        for item_name, item_update_details in category_updates_dict.items():
            if not isinstance(item_update_details, dict):
                logger.warning(f"Skipping invalid item_details for '{item_name}' in cat '{category_key}' (not dict) for ch {chapter_number}. Data: {item_update_details}")
                continue

            item_log_name = f"{category_key}.{item_name}"
            existing_item_data = target_category_dict.get(item_name)
            item_created_this_call = False

            if existing_item_data is None:
                target_category_dict[item_name] = {} # Create new item entry
                existing_item_data = target_category_dict[item_name]
                existing_item_data[f"added_in_chapter_{chapter_number}"] = True # Mark as added
                item_created_this_call = True
                # items_affected_count will be incremented if robust_merge also finds modifications or if it's new

            if _robust_merge_world_item_data_logic_internal(
                existing_item_data, item_update_details, item_log_name, chapter_number, from_flawed_draft
            ) or item_created_this_call: # Count if modified OR if newly created
                items_affected_count += 1

    if items_affected_count > 0:
        logger.info(f"World-building dict merge complete for ch {chapter_number}. Approx {items_affected_count} items affected/added.")
    else:
        logger.info(f"No world-building dict items were effectively updated or added for ch {chapter_number}.")


# --- KG Pre-population Cypher Generation Helpers ---
def _generate_novel_info_cypher(plot_outline: Dict[str, Any], novel_id: str) -> List[Tuple[str, Dict[str, Any]]]:
    statements = []
    novel_props = {k: v for k, v in plot_outline.items() if not isinstance(v, (list, dict)) and v is not None}
    novel_props['id'] = novel_id
    statements.append(("MERGE (ni:NovelInfo {id: $id_val}) SET ni = $props_val",
                       {"id_val": novel_id, "props_val": novel_props}))
    return statements

def _generate_plot_points_cypher(plot_points_list: List[str], novel_id: str) -> List[Tuple[str, Dict[str, Any]]]:
    statements = []
    for i, desc in enumerate(plot_points_list):
        if isinstance(desc, str):
            pp_id = f"{novel_id}_pp_{i+1}"
            pp_props = {"id": pp_id, "sequence": i + 1, "description": desc, "status": "pending"}
            statements.append(("MERGE (pp:PlotPoint {id: $id_val}) SET pp = $props",
                               {"id_val": pp_id, "props": pp_props}))
            statements.append((
                "MATCH (ni:NovelInfo {id: $novel_id_val}) MATCH (pp:PlotPoint {id: $pp_id_val}) MERGE (ni)-[:HAS_PLOT_POINT]->(pp)",
                {"novel_id_val": novel_id, "pp_id_val": pp_id}
            ))
            if i > 0:
                prev_pp_id = f"{novel_id}_pp_{i}"
                statements.append((
                    "MATCH (prev_pp:PlotPoint {id: $prev_pp_id_val}) MATCH (curr_pp:PlotPoint {id: $curr_pp_id_val}) MERGE (prev_pp)-[:NEXT_PLOT_POINT]->(curr_pp)",
                    {"prev_pp_id_val": prev_pp_id, "curr_pp_id_val": pp_id}
                ))
    return statements

def _generate_character_node_cypher(char_name: str, char_props: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    return ("MERGE (c:Entity {name: $char_name_val}) SET c:Character SET c += $props_val",
            {"char_name_val": char_name, "props_val": char_props})

def _generate_traits_cypher(char_name: str, traits_list: List[str]) -> List[Tuple[str, Dict[str, Any]]]:
    statements = []
    for trait in traits_list:
        if isinstance(trait, str):
            statements.append((
                "MATCH (c:Character:Entity {name: $char_name_val}) MERGE (t:Trait {name: $trait_name_val}) MERGE (c)-[:HAS_TRAIT]->(t)",
                {"char_name_val": char_name, "trait_name_val": trait}
            ))
    return statements

def _generate_relationships_cypher(char_name: str, relationships_dict: Dict[str, Any], profile_source_info: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    statements = []
    for target_name, rel_detail in relationships_dict.items():
        rel_type = "RELATED_TO"
        rel_props_to_set = {"description": str(rel_detail)}

        if isinstance(rel_detail, dict):
            rel_type = str(rel_detail.get("type", rel_type)).upper().replace(" ", "_")
            rel_props_to_set = {k: v for k, v in rel_detail.items() if isinstance(v, (str, int, float, bool))}
            rel_props_to_set.setdefault("description", f"{rel_type} {target_name}")
        elif isinstance(rel_detail, str):
            rel_type = rel_detail.upper().replace(" ", "_")
            rel_props_to_set = {"description": rel_detail}

        # Ensure chapter_added and is_provisional are set for pre-population
        chap_added_val = profile_source_info.get(f"source_quality_chapter_{config.KG_PREPOPULATION_CHAPTER_NUM}", config.KG_PREPOPULATION_CHAPTER_NUM)
        rel_props_to_set.setdefault("chapter_added", chap_added_val if isinstance(chap_added_val, int) else config.KG_PREPOPULATION_CHAPTER_NUM)
        rel_props_to_set.setdefault("is_provisional", profile_source_info.get(f"source_quality_chapter_{config.KG_PREPOPULATION_CHAPTER_NUM}") == "provisional_from_unrevised_draft")
        
        props_for_set_clause = {k: v for k, v in rel_props_to_set.items() if k not in ['type', 'chapter_added']} # Already in MERGE pattern

        statements.append((
            """
            MATCH (c1:Character:Entity {name: $char_name1_val})
            MERGE (c2:Entity {name: $char_name2_val})
                ON CREATE SET c2:Character, c2.description = 'Auto-created via relationship from ' + $char_name1_val
                ON MATCH SET c2:Character
            MERGE (c1)-[r:DYNAMIC_REL {type: $rel_type_val, chapter_added: $chapter_added_val}]->(c2)
            SET r += $props_for_set_clause_val, r.last_updated = timestamp()
            """,
            {"char_name1_val": char_name, "char_name2_val": target_name,
             "rel_type_val": rel_type, "chapter_added_val": rel_props_to_set["chapter_added"],
             "props_for_set_clause_val": props_for_set_clause}
        ))
    return statements

def _generate_dev_events_cypher(char_name: str, profile_dict: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    statements = []
    for key, value_str in profile_dict.items():
        if key.startswith("development_in_chapter_") and isinstance(value_str, str):
            try:
                chap_num_int = int(key.split("_")[-1])
                dev_event_props = {"summary": value_str, "chapter_updated": chap_num_int}
                if profile_dict.get(f"source_quality_chapter_{chap_num_int}") == "provisional_from_unrevised_draft":
                    dev_event_props["is_provisional"] = True
                statements.append((
                    "MATCH (c:Character:Entity {name: $char_name_val}) CREATE (dev:DevelopmentEvent) SET dev = $props CREATE (c)-[:DEVELOPED_IN_CHAPTER]->(dev)",
                    {"char_name_val": char_name, "props": dev_event_props}
                ))
            except ValueError:
                logger.warning(f"Could not parse chapter from dev key: {key} for char {char_name}")
    return statements

def _generate_world_overview_cypher(overview_dict: Dict[str, Any], wc_id: str, overview_source_info: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    statements = []
    if "description" in overview_dict:
        overview_props = {"id": wc_id, "overview_description": str(overview_dict.get("description", ""))}
        if overview_source_info.get(f"source_quality_chapter_{config.KG_PREPOPULATION_CHAPTER_NUM}") == "provisional_from_unrevised_draft":
            overview_props["is_provisional"] = True
        statements.append(("MERGE (wc:WorldContainer {id: $id_val}) SET wc = $props_val",
                           {"id_val": wc_id, "props_val": overview_props}))
    return statements

def _generate_world_element_node_cypher(we_id_str: str, category_str: str, item_name_str: str, item_props_original: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    item_props_for_set = {k: v for k, v in item_props_original.items() if isinstance(v, (str, int, float, bool)) and v is not None}
    item_props_for_set.update({'id': we_id_str, 'name': item_name_str, 'category': category_str})

    created_chap_num = config.KG_PREPOPULATION_CHAPTER_NUM
    added_key = next((k for k in item_props_original if k.startswith("added_in_chapter_")), None)
    if added_key:
        try: created_chap_num = int(added_key.split("_")[-1])
        except ValueError: pass
    item_props_for_set['created_chapter'] = created_chap_num
    if item_props_original.get(f"source_quality_chapter_{created_chap_num}") == "provisional_from_unrevised_draft":
        item_props_for_set['is_provisional'] = True
    
    return ("MERGE (we:WorldElement {id: $id_val}) SET we = $props_val",
            {"id_val": we_id_str, "props_val": item_props_for_set})

def _generate_world_list_properties_cypher(we_id_str: str, list_prop_key_str: str, list_value: List[Any]) -> List[Tuple[str, Dict[str, Any]]]:
    statements = []
    for val_item in list_value:
        if isinstance(val_item, str):
            rel_name_base = list_prop_key_str.upper().rstrip('S')
            if list_prop_key_str == "key_elements": rel_name_base = "KEY_ELEMENT"
            elif list_prop_key_str == "traits": rel_name_base = "TRAIT_ASPECT"
            rel_name_final = f"HAS_{rel_name_base}"
            statements.append((
                f"MATCH (we:WorldElement {{id: $we_id_val}}) MERGE (v:ValueNode {{value: $val_item_val, type: $value_node_type_val}}) MERGE (we)-[:{rel_name_final}]->(v)",
                {"we_id_val": we_id_str, "val_item_val": val_item, "value_node_type_val": list_prop_key_str}
            ))
    return statements

def _generate_world_elaboration_events_cypher(we_id_str: str, item_name_str: str, details_dict: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    statements = []
    for key_str, value_val in details_dict.items():
        if key_str.startswith("elaboration_in_chapter_") and isinstance(value_val, str):
            try:
                chap_num_val = int(key_str.split("_")[-1])
                elab_props = {"summary": value_val, "chapter_updated": chap_num_val}
                if details_dict.get(f"source_quality_chapter_{chap_num_val}") == "provisional_from_unrevised_draft":
                    elab_props["is_provisional"] = True
                statements.append((
                    "MATCH (we:WorldElement {id: $we_id_val}) CREATE (we_elab:WorldElaborationEvent) SET we_elab = $props CREATE (we)-[:ELABORATED_IN_CHAPTER]->(we_elab)",
                    {"we_id_val": we_id_str, "props": elab_props}
                ))
            except ValueError:
                logger.warning(f"Could not parse chapter from world elab key: {key_str} for item {item_name_str}")
    return statements

async def _prepopulate_kg_from_dicts_internal(
    plot_outline: Dict[str, Any],
    character_profiles: Dict[str, Any],
    world_building: Dict[str, Any]
):
    logger.info("Starting Knowledge Graph pre-population directly from initial data dicts...")
    cypher_statements: List[Tuple[str, Dict[str, Any]]] = []
    novel_id = config.MAIN_NOVEL_INFO_NODE_ID

    if plot_outline:
        cypher_statements.extend(_generate_novel_info_cypher(plot_outline, novel_id))
        plot_points = plot_outline.get('plot_points', [])
        if isinstance(plot_points, list):
            cypher_statements.extend(_generate_plot_points_cypher(plot_points, novel_id))

    for char_name, profile in character_profiles.items():
        if not isinstance(profile, dict): continue
        char_props_for_set = {k: v for k, v in profile.items() if isinstance(v, (str, int, float, bool)) and v is not None}
        cypher_statements.append(_generate_character_node_cypher(char_name, char_props_for_set))
        if isinstance(profile.get("traits"), list):
            cypher_statements.extend(_generate_traits_cypher(char_name, profile["traits"]))
        if isinstance(profile.get("relationships"), dict):
            cypher_statements.extend(_generate_relationships_cypher(char_name, profile["relationships"], profile)) # Pass profile for source info
        cypher_statements.extend(_generate_dev_events_cypher(char_name, profile))

    for category, items_or_overview in world_building.items():
        if category == "_overview_":
            if isinstance(items_or_overview, dict):
                cypher_statements.extend(_generate_world_overview_cypher(items_or_overview, config.MAIN_WORLD_CONTAINER_NODE_ID, items_or_overview))
            continue
        if category in ["is_default", "source", "user_supplied_data"] or not isinstance(items_or_overview, dict):
            continue

        for item_name, details in items_or_overview.items():
            if not isinstance(details, dict) or item_name.startswith(("_", "source_quality_chapter_", "category_updated_in_chapter_")): continue
            we_id = f"{category}_{item_name}".replace(" ", "_").replace("'", "").lower()
            cypher_statements.append(_generate_world_element_node_cypher(we_id, category, item_name, details))
            for list_prop_key in ["goals", "rules", "key_elements", "traits"]:
                if isinstance(details.get(list_prop_key), list):
                    cypher_statements.extend(_generate_world_list_properties_cypher(we_id, list_prop_key, details[list_prop_key]))
            cypher_statements.extend(_generate_world_elaboration_events_cypher(we_id, item_name, details))

    if cypher_statements:
        try:
            await neo4j_manager.execute_cypher_batch(cypher_statements)
            logger.info(f"KG pre-population complete: Executed {len(cypher_statements)} Cypher statements directly from initial data.")
        except Exception as e:
            logger.error(f"Error during direct KG pre-population batch execution: {e}", exc_info=True)
    else:
        logger.info("No Cypher statements generated for KG pre-population from initial data.")


# --- Main Agent Class ---
class KGMaintainerAgent:
    def __init__(self, model_name: str = config.KNOWLEDGE_UPDATE_MODEL):
        self.model_name = model_name
        logger.info(f"KGMaintainerAgent initialized with model: {self.model_name}")

    async def summarize_chapter(self, chapter_text: Optional[str], chapter_number: int) -> Tuple[Optional[str], Optional[Dict[str, int]]]:
        if not chapter_text or len(chapter_text) < 50:
            logger.warning(f"Chapter {chapter_number} text too short for summarization ({len(chapter_text or '')} chars).")
            return None, None
        cleaned_summary, usage_data = await _llm_summarize_full_chapter_text_logic_internal(chapter_text, chapter_number)
        if cleaned_summary:
            logger.info(f"Generated summary for ch {chapter_number}: '{cleaned_summary[:100].strip()}...'")
            return cleaned_summary, usage_data
        logger.warning(f"Failed to generate a valid summary for ch {chapter_number} via LLM.")
        return None, usage_data

    def _parse_unified_character_updates(self, text_block: str, chapter_number: int) -> Dict[str, Any]:
        char_updates: Dict[str, Any] = {}
        character_block_starts = list(re.finditer(r"^\s*Character:\s*(.+)$", text_block, re.IGNORECASE | re.MULTILINE))

        for i, start_match in enumerate(character_block_starts):
            char_name = start_match.group(1).strip()
            if not char_name: continue

            block_start_index = start_match.end()
            block_end_index = character_block_starts[i+1].start() if i + 1 < len(character_block_starts) else len(text_block)
            individual_char_block_text = text_block[block_start_index:block_end_index].strip()

            if individual_char_block_text:
                parsed_char_data = parse_key_value_block(
                    individual_char_block_text,
                    CHAR_UPDATE_KEY_MAP,
                    CHAR_UPDATE_LIST_INTERNAL_KEYS,
                    special_list_handling=CHAR_UPDATE_PKVB_SPECIAL_HANDLING
                )

                if "relationships" in parsed_char_data and isinstance(parsed_char_data["relationships"], list):
                    rels_dict = {}
                    for rel_str_or_item in parsed_char_data["relationships"]:
                        if isinstance(rel_str_or_item, str) and ':' in rel_str_or_item:
                            parts = rel_str_or_item.split(":", 1)
                            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                                rels_dict[parts[0].strip()] = parts[1].strip()
                            else: logger.warning(f"Malformed relationship string for {char_name}: '{rel_str_or_item}'")
                        elif isinstance(rel_str_or_item, str) and rel_str_or_item.strip():
                            rels_dict[rel_str_or_item.strip()] = "related"
                    parsed_char_data["relationships"] = rels_dict

                dev_key_standard = f"development_in_chapter_{chapter_number}"
                specific_dev_key_from_llm = next((k for k in parsed_char_data if k.lower() == dev_key_standard.lower()), None)

                if specific_dev_key_from_llm and specific_dev_key_from_llm != dev_key_standard:
                    parsed_char_data[dev_key_standard] = parsed_char_data.pop(specific_dev_key_from_llm)
                elif not specific_dev_key_from_llm and any(k != "modification_proposal" for k in parsed_char_data):
                     parsed_char_data[dev_key_standard] = f"Character '{char_name}' appeared or was mentioned in Chapter {chapter_number}."

                char_updates[char_name] = parsed_char_data
        return char_updates

    def _parse_unified_world_updates(self, text_block: str, chapter_number: int) -> Dict[str, Any]:
        parsed_data = parse_hierarchical_structured_text(
            text_block,
            WORLD_UPDATE_CATEGORY_PATTERN,
            WORLD_UPDATE_ITEM_PATTERN,
            WORLD_UPDATE_DETAIL_KEY_MAP,
            WORLD_UPDATE_DETAIL_LIST_INTERNAL_KEYS,
            overview_category_internal_key="_overview_"
        )

        for category_name, items in parsed_data.items():
            elaboration_key_standard = f"elaboration_in_chapter_{chapter_number}"
            if category_name == "_overview_":
                if items and isinstance(items, dict) and any(k != "modification_proposal" for k in items):
                    specific_elab_key = next((k for k in items if k.lower() == elaboration_key_standard.lower()), None)
                    if not specific_elab_key:
                        items[elaboration_key_standard] = f"Overall world overview mentioned or updated in Chapter {chapter_number}."
                    elif specific_elab_key != elaboration_key_standard and items.get(specific_elab_key):
                        items[elaboration_key_standard] = items.pop(specific_elab_key)
            elif isinstance(items, dict):
                for item_name, item_details in items.items():
                    if isinstance(item_details, dict):
                        if any(k != "modification_proposal" for k in item_details):
                            specific_elab_key = next((k for k in item_details if k.lower() == elaboration_key_standard.lower()), None)
                            if not specific_elab_key:
                                 item_details[elaboration_key_standard] = f"Item '{item_name}' in category '{category_name}' was mentioned or interacted with in Chapter {chapter_number}."
                            elif specific_elab_key != elaboration_key_standard and item_details.get(specific_elab_key):
                                 item_details[elaboration_key_standard] = item_details.pop(specific_elab_key)
        return parsed_data

    async def _perform_unified_knowledge_extraction(
        self,
        novel_props: Dict[str, Any],
        chapter_text: str,
        chapter_number: int
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, int]]]:
        logger.info(f"KGMaintainerAgent performing unified knowledge extraction (plain text) for Chapter {chapter_number}...")
        if not chapter_text:
            logger.warning(f"Unified knowledge extraction skipped for Ch {chapter_number}: empty chapter text.")
            return {"character_updates": {}, "world_updates": {}, "knowledge_triples": []}, None

        protagonist_name = novel_props.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME)
        current_profiles_plain_text = await get_filtered_character_profiles_for_prompt_plain_text(novel_props, chapter_number - 1)
        current_world_plain_text = await get_filtered_world_data_for_prompt_plain_text(novel_props, chapter_number - 1)
        candidate_entities_list = await heuristic_entity_spotter_for_kg(novel_props, chapter_text)
        
        candidate_entities_text_parts: List[str] = []
        if candidate_entities_list:
            candidate_entities_text_parts.append("Candidate Entities (for KG triple focus):\n")
            candidate_entities_text_parts.extend([f"- {e}" for e in candidate_entities_list])
        else:
            candidate_entities_text_parts.append("Candidate Entities: None identified by heuristic.")
        candidate_entities_text_str = "\n".join(candidate_entities_text_parts)


        dynamic_instr_char_parts: List[str] = []
        dynamic_instr_world_parts: List[str] = []

        if config.ENABLE_DYNAMIC_STATE_ADAPTATION:
            dynamic_instr_char_parts.extend([
                f"For existing characters, if their traits, status, or core description needs modification based on THIS chapter's events, ",
                f"include a line like `Modification Proposal: MODIFY traits: ADD \"Determined\", REMOVE \"Hesitant\"`. ",
                f"Also specify all current `Traits` as a comma-separated list if changed, new `Status`, new `Description`. ",
                f"For NEW characters, provide `Description`, `Traits` (comma-separated), `Status`. ",
                f"For `Relationships`, list them as `Target Name: relationship type` (e.g., `John Doe: ally`) or on new lines under a `Relationships:` header, each as `- Target Name: type`.",
                f"Only include characters that are updated, newly introduced, or have a modification proposal."
            ])
            dynamic_instr_world_parts.extend([
                f"For existing world items, if their properties need modification, include a line like `Modification Proposal: MODIFY atmosphere: \"Now heavy with magical fallout\"`. ",
                f"Also provide the new full value for any changed properties. For NEW world items, provide all known properties. ",
                f"E.g., for locations: `Description: ...`, `Atmosphere: ...`. For factions: `Description: ...`, `Goals:` (followed by '- goal' lines). ",
                f"Only include world elements that are new, significantly changed by THIS chapter's events, or have a modification proposal."
            ])
        else:
            dynamic_instr_char_parts.append("Only include characters whose information is directly updated or those newly introduced in THIS chapter. Provide full description, traits (comma-separated), status for new chars.")
            dynamic_instr_world_parts.append("Only include world elements that are new or significantly changed by THIS chapter's events. Provide full details for new items.")
        
        dynamic_instr_char_str = "".join(dynamic_instr_char_parts)
        dynamic_instr_world_str = "".join(dynamic_instr_world_parts)

        common_predicates_str = ", ".join([
            "is_a", "located_in", "has_trait", "status_is", "feels", "knows", "believes", "wants",
            "interacted_with", "travelled_to", "discovered", "acquired", "lost", "used_item",
            "attacked", "helped", "part_of", "caused_by", "leads_to", "observed", "heard", "said",
            "thought_about", "decided_to", "has_goal", "has_feature", "related_to", "member_of",
            "leader_of", "enemy_of", "ally_of", "works_for", "has_ability", "possesses", "created_by",
            "has_description", "has_atmosphere", "has_rule", "has_history_event"
        ])

        example_output_structure_text_parts = [
            "### CHARACTER UPDATES ###",
            "Character: Elara Vance",
            "Description: Now appears more determined after facing the Shadow Beast.",
            "Traits: Brave, Resourceful, Determined",
            "Status: Wounded but resolute",
            f"Development in Chapter {chapter_number}: Confronted the Shadow Beast, discovered a hidden clue.",
            "Modification Proposal: MODIFY traits: ADD \"Determined\"",
            "",
            "Character: Gorok",
            "Description: A new troll encountered in the Whispering Woods.",
            "Traits: Grumpy, Strong",
            "Status: Wary of Elara",
            f"Development in Chapter {chapter_number}: Briefly fought Elara, then fled.",
            "",
            "### WORLD UPDATES ###",
            "Category: locations",
            "Item: Whispering Woods",
            "Description: The woods now feel colder, and shadows linger longer.",
            "Atmosphere: Ominous, Chilling",
            f"Elaboration in Chapter {chapter_number}: Site of the Shadow Beast encounter.",
            "Modification Proposal: MODIFY atmosphere: \"Ominous, Chilling\"",
            "",
            "Category: _overview_",
            "Description: The general mood of the realm has darkened slightly due to recent events.",
            f"Elaboration in Chapter {chapter_number}: Overall world overview mentioned or updated.",
            "",
            "### KG TRIPLES ###",
            "Elara Vance | confronted | Shadow Beast",
            "Elara Vance | status_is | Wounded but resolute",
            "Whispering Woods | has_atmosphere | Ominous",
            "Shadow Beast | located_in | Whispering Woods"
        ]
        example_output_structure_str = "\n".join(example_output_structure_text_parts)

        prompt_lines = [
            "/no_think",
            "You are a comprehensive literary analyst and knowledge engineer.",
            f"Analyze the **Complete Chapter {chapter_number} Text** (protagonist: {protagonist_name}) and extract information for three distinct knowledge bases.",
            "Output ONLY plain text, structured as described below.",
            "",
            "**Reference Information (Current State Before This Chapter - for context only, extract from THIS chapter's text):**",
            "  **Character Profiles Snapshot (Plain Text):**",
            "  ```text",
            current_profiles_plain_text if current_profiles_plain_text.strip() else "No character profiles provided as reference.",
            "  ```",
            "  **World Building Snapshot (Plain Text):**",
            "  ```text",
            current_world_plain_text if current_world_plain_text.strip() else "No world building data provided as reference.",
            "  ```",
            candidate_entities_text_str,
            "",
            f"**Complete Chapter {chapter_number} Text (Analyze this full text):**",
            "--- BEGIN COMPLETE CHAPTER TEXT ---",
            chapter_text,
            "--- END COMPLETE CHAPTER TEXT ---",
            "",
            "**Output Format (CRITICAL - PLAIN TEXT ONLY):**",
            "Use the following section headers EXACTLY:",
            "`### CHARACTER UPDATES ###`",
            "`### WORLD UPDATES ###`",
            "`### KG TRIPLES ###`",
            "",
            "**1. Under `### CHARACTER UPDATES ###`:**",
            "   List each character on a new line: `Character: [Character Name]`",
            "   Followed by indented key-value pairs for their profile data (use Title Case for keys like \"Description\", \"Traits\", \"Status\"):",
            "     `Description: [Full description if new or significantly changed]`",
            "     `Traits: [Comma-separated list, e.g., Brave, Cautious OR list with '-' prefix on new lines]`",
            "     `Status: [Current status]`",
            "     `Relationships:` (Optional, if any. List each as \"- Target Name: Relationship Type\", e.g., \"- Gorok: uneasy ally\")",
            f"     `Development in Chapter {chapter_number}: [Summary of their role/changes in THIS chapter]`",
            "     `Modification Proposal: [Optional: MODIFY key: value_change]`",
            f"   {dynamic_instr_char_str}",
            "",
            "**2. Under `### WORLD UPDATES ###`:**",
            "   Start each category with: `Category: [locations | society | systems | lore | history | factions | _overview_]`",
            "   For `_overview_`, provide: `Description: [Overall world feel change or new general description]` (and other relevant overview keys if applicable)",
            f"     `Elaboration in Chapter {chapter_number}: [Context from THIS chapter for overview]`",
            "   For other categories, list items: `Item: [Item Name]`",
            "   Followed by indented key-value pairs for item details (use Title Case for keys like \"Description\", \"Atmosphere\"):",
            "     `Description: [Full description if new or changed]`",
            "     `Atmosphere: [For locations]`",
            "     `Goals:` (For factions, list each goal on a new line starting with \"- \")",
            "       `- Goal 1`",
            f"     `Elaboration in Chapter {chapter_number}: [Context from THIS chapter for this item]`",
            "     `Modification Proposal: [Optional: MODIFY key: value_change]`",
            f"   {dynamic_instr_world_str}",
            "",
            "**3. Under `### KG TRIPLES ###`:**",
            "   List each factual triple. Choose ONE format and use it consistently for all triples in this section:",
            "   Format A (preferred): `Subject | Predicate | Object`",
            "   Format B: `Subject: [Subject Name], Predicate: [predicate_name], Object: [Object Name/Value]`",
            "   Format C: `- [Subject, Predicate, Object]`",
            f"   Use predicates from this suggested list where appropriate: {common_predicates_str}. Focus on NEW facts or significant CHANGES from THIS chapter.",
            "",
            "**Example of Expected Output Structure:**",
            "```",
            example_output_structure_str,
            "```",
            "",
            "Begin your output now:"
        ]
        prompt = "\n".join(prompt_lines)

        logger.info(f"Calling LLM ({self.model_name}) for unified knowledge extraction (Ch {chapter_number}).")
        raw_extraction_text, usage_data = await llm_interface.async_call_llm(
            model_name=self.model_name, prompt=prompt, temperature=0.5,
            max_tokens=config.MAX_KG_TRIPLE_TOKENS,
            allow_fallback=True, stream_to_disk=True
        )
        cleaned_extraction_text = llm_interface.clean_model_response(raw_extraction_text)

        sections = re.split(r"^\s*###\s*([\w\s]+?)\s*###\s*$", cleaned_extraction_text, flags=re.IGNORECASE | re.MULTILINE)

        parsed_sections: Dict[str, str] = {}
        current_section_name_normalized = None
        for i in range(1, len(sections)):
            if i % 2 == 1:
                header_name_raw = sections[i].strip().lower()
                if "character" in header_name_raw: current_section_name_normalized = "character_updates"
                elif "world" in header_name_raw: current_section_name_normalized = "world_updates"
                elif "kg triples" in header_name_raw: current_section_name_normalized = "kg_triples"
                else: current_section_name_normalized = None
            elif current_section_name_normalized:
                parsed_sections[current_section_name_normalized] = sections[i].strip()
                current_section_name_normalized = None

        char_updates_text = parsed_sections.get("character_updates", "")
        world_updates_text = parsed_sections.get("world_updates", "")
        kg_triples_text_block = parsed_sections.get("kg_triples", "")

        char_updates_dict = self._parse_unified_character_updates(char_updates_text, chapter_number)
        world_updates_dict = self._parse_unified_world_updates(world_updates_text, chapter_number)
        kg_triples_list = parse_kg_triples_from_text(kg_triples_text_block)

        final_extraction = {
            "character_updates": char_updates_dict,
            "world_updates": world_updates_dict,
            "knowledge_triples": kg_triples_list
        }
        logger.info(f"Unified knowledge extraction for Ch {chapter_number} complete. Chars updated/new: {len(final_extraction['character_updates'])}, World cats affected: {len(final_extraction['world_updates'])}, KG Triples extracted: {len(final_extraction['knowledge_triples'])}.")

        if not char_updates_dict and not world_updates_dict and not kg_triples_list and cleaned_extraction_text.strip():
            logger.warning(f"Unified extraction for Ch {chapter_number} yielded no structured data despite non-empty LLM response. Raw LLM output snippet: '{cleaned_extraction_text[:500]}...'")

        return final_extraction, usage_data

    async def extract_and_merge_knowledge(
        self,
        novel_props_mutable: Dict[str, Any],
        chapter_number: int,
        final_chapter_text: str,
        is_from_flawed_draft: bool
    ) -> Optional[Dict[str, int]]:
        """Extracts knowledge from chapter text and merges it into the agent's state and KG.
           Returns LLM usage data from the extraction step.
        """
        if not final_chapter_text:
            logger.warning(f"Skipping knowledge extraction and merge for ch {chapter_number}: Final chapter text is empty.")
            return None

        logger.info(f"KGMaintainerAgent starting knowledge extraction and merge for ch {chapter_number} (Source text from flawed draft: {is_from_flawed_draft})...")

        extraction_results, usage_data = await self._perform_unified_knowledge_extraction(
            novel_props_mutable,
            final_chapter_text,
            chapter_number
        )
        character_updates_dict = extraction_results.get("character_updates", {})
        world_updates_dict = extraction_results.get("world_updates", {})
        kg_triples_list = extraction_results.get("knowledge_triples", [])

        if character_updates_dict and isinstance(character_updates_dict, dict):
            _merge_character_profile_updates_into_state_internal(
                novel_props_mutable['character_profiles'],
                character_updates_dict, chapter_number, is_from_flawed_draft
            )
        else: logger.warning(f"No valid character updates extracted to merge for ch {chapter_number}.")

        if world_updates_dict and isinstance(world_updates_dict, dict):
            _merge_world_item_updates_into_state_internal(
                novel_props_mutable['world_building'],
                world_updates_dict, chapter_number, is_from_flawed_draft
            )
        else: logger.warning(f"No valid world-building updates extracted to merge for ch {chapter_number}.")

        if kg_triples_list and isinstance(kg_triples_list, list):
            added_count, skipped_count = 0, 0
            kg_add_tasks = []
            for triple_any in kg_triples_list:
                if isinstance(triple_any, list) and len(triple_any) == 3:
                    subj = str(triple_any[0]).strip() if triple_any[0] is not None else ""
                    pred = str(triple_any[1]).strip() if triple_any[1] is not None else ""
                    obj_val = str(triple_any[2]).strip() if triple_any[2] is not None else ""

                    if subj and pred and obj_val:
                        obj_truncated = obj_val[:500] + "..." if len(obj_val) > 503 else obj_val
                        kg_add_tasks.append(
                            kg_queries.add_kg_triple_to_db(subj, pred, obj_truncated, chapter_number, is_provisional=is_from_flawed_draft)
                        )
                        added_count += 1
                    else:
                        skipped_count += 1
                        logger.warning(f"Skipping invalid KG triple (empty component after strip) from extraction for ch {chapter_number}: {triple_any}")
                else:
                    skipped_count += 1
                    logger.warning(f"Skipping invalid KG triple format from extraction for ch {chapter_number}: {triple_any}")

            if kg_add_tasks:
                await asyncio.gather(*kg_add_tasks)
            logger.info(f"KG update for ch {chapter_number}: Attempted to add {added_count} triples, skipped {skipped_count} due to format/content.")
        else:
            logger.info(f"No KG triples extracted to add for ch {chapter_number}.")

        logger.info(f"Unified knowledge extraction and merge process complete for ch {chapter_number}.")
        return usage_data


    async def prepopulate_kg_from_initial_data(
        self,
        plot_outline: Dict[str, Any],
        character_profiles: Dict[str, Any],
        world_building: Dict[str, Any]
    ):
        """Populates the Knowledge Graph from the initial setup data."""
        await _prepopulate_kg_from_dicts_internal(plot_outline, character_profiles, world_building)