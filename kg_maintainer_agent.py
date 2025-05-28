# kg_maintainer_agent.py
import logging
import re
import asyncio
from typing import Dict, List, Optional, Any, Tuple

from async_lru import alru_cache

import config
import llm_interface
# from state_manager import state_manager # No longer directly used
from core_db.base_db_manager import neo4j_manager # For execute_cypher_batch
from data_access import kg_queries # For add_kg_triple_to_db
from prompt_data_getters import (
    get_filtered_character_profiles_for_prompt_plain_text,
    heuristic_entity_spotter_for_kg,
    get_filtered_world_data_for_prompt_plain_text
)
from parsing_utils import parse_key_value_block, parse_hierarchical_structured_text, parse_kg_triples_from_text

logger = logging.getLogger(__name__)

CHAR_UPDATE_KEY_MAP = {
    "description": "description", "traits": "traits", "status": "status",
    "relationships": "relationships", "modification_proposal": "modification_proposal",
    re.compile(r"development_in_chapter_\d+"): lambda match: match.group(0).lower() 
}
CHAR_UPDATE_LIST_INTERNAL_KEYS = ["traits"] # Relationships handled customly after

# For parse_key_value_block to correctly split single-line comma-separated "Traits: value1, value2"
CHAR_UPDATE_PKVB_SPECIAL_HANDLING = {
    "traits": {"separator": ","}
}
# For custom post-parsing of relationships if they are not list items
CHAR_UPDATE_RELATIONSHIP_POST_PARSING_HANDLING = {
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


@alru_cache(maxsize=config.SUMMARY_CACHE_SIZE)
async def _llm_summarize_full_chapter_text_logic_internal(chapter_text_full_key: str, chapter_number: int) -> Tuple[str, Optional[Dict[str, int]]]:
    """ Summarizes chapter text using an LLM. Input `chapter_text_full_key` is the actual text. Returns summary and usage."""
    prompt = f"""/no_think
You are a concise summarizer. Summarize the key events, character developments, and plot advancements from the following Chapter {chapter_number} text.
The summary should be 1-3 sentences long and capture the most crucial information.
Focus on what changed or was revealed.

Full Chapter Text:
--- BEGIN TEXT ---
{chapter_text_full_key}
--- END TEXT ---

Output ONLY the summary text. No extra commentary or "Summary:" prefix.
"""
    summary_raw, usage_data = await llm_interface.async_call_llm(
        model_name=config.SMALL_MODEL, 
        prompt=prompt,
        temperature=0.6,
        max_tokens=config.MAX_SUMMARY_TOKENS, 
        stream_to_disk=False 
    )
    return llm_interface.clean_model_response(summary_raw).strip(), usage_data

def _apply_trait_modification(current_traits_list: List[str], modification_details_str: str) -> List[str]:
    traits_set = set(current_traits_list)
    for add_match in re.finditer(r"ADD\s+['\"]([^'\"]+)['\"]", modification_details_str, re.IGNORECASE):
        trait_to_add = add_match.group(1).strip()
        if trait_to_add: traits_set.add(trait_to_add)
    for remove_match in re.finditer(r"REMOVE\s+['\"]([^'\"]+)['\"]", modification_details_str, re.IGNORECASE):
        trait_to_remove = remove_match.group(1).strip()
        if trait_to_remove: traits_set.discard(trait_to_remove) 
    return sorted(list(traits_set))

def _apply_state_modification_proposal_logic(
    target_dict: Dict[str, Any],
    proposal_str: str,
    item_name_for_log: str,
    item_type_for_log: str 
):
    if not isinstance(proposal_str, str) or not proposal_str.strip():
        logger.debug(f"Empty or invalid modification proposal for '{item_name_for_log}'. Proposal: '{proposal_str}'")
        return

    logger.debug(f"Applying modification proposal for '{item_name_for_log}' ({item_type_for_log}): '{proposal_str}'")
    
    match = re.match(r"MODIFY\s+([\w_]+)\s*:(.*)", proposal_str, re.IGNORECASE)
    if not match:
        logger.warning(f"Invalid modification proposal format for '{item_name_for_log}'. Proposal: '{proposal_str}'. Expected 'MODIFY key: value'.")
        return

    key_name_from_proposal_upper = match.group(1).strip().upper() 
    value_modification_str = match.group(2).strip()

    original_key_name = next(
        (k for k in target_dict if k.upper() == key_name_from_proposal_upper),
        key_name_from_proposal_upper.lower() 
    )
    
    try:
        if original_key_name.lower() == "traits": 
            if "traits" not in target_dict or not isinstance(target_dict["traits"], list):
                target_dict["traits"] = [] 
            target_dict["traits"] = _apply_trait_modification(target_dict["traits"], value_modification_str)
            logger.info(f"Applied trait modifications for '{item_name_for_log}'. New traits: {target_dict['traits']}")
        else: 
            new_value_str = value_modification_str.strip("'\" ")
            if new_value_str: 
                target_dict[original_key_name] = new_value_str
                logger.info(f"Applied modification to '{original_key_name}' for '{item_name_for_log}'. New value: '{new_value_str[:70]}...'")
            else:
                logger.warning(f"Modification proposal for '{original_key_name}' of '{item_name_for_log}' resulted in an empty new value. Proposal: '{proposal_str}'")
    except Exception as e:
        logger.error(f"Error applying modification proposal for '{item_name_for_log}': {e}. Key: {original_key_name}, Proposal: '{proposal_str}'", exc_info=True)


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
    all_character_profiles_for_lookup: Dict[str, Any] 
):
    if provisional_marker_key in char_update_data: 
        existing_profile[provisional_marker_key] = char_update_data[provisional_marker_key]

    for key, value in char_update_data.items():
        if key in ["modification_proposal", provisional_marker_key] or key.startswith("development_in_chapter_"):
            if key.startswith("development_in_chapter_") and value and isinstance(value, str) and value.strip():
                 existing_profile[key] = value
            continue 

        if key == "traits" and isinstance(value, list):
            if "traits" not in existing_profile or not isinstance(existing_profile["traits"], list): existing_profile["traits"] = []
            valid_new_traits = {t for t in value if isinstance(t, str) and t.strip()}
            existing_profile["traits"] = sorted(list(set(existing_profile["traits"]).union(valid_new_traits)))
        elif key == "relationships" and isinstance(value, dict):
            if not isinstance(existing_profile.get("relationships"), dict): existing_profile["relationships"] = {}
            existing_profile["relationships"].update(value) 
        elif key == "description" and isinstance(value, str) and value.strip(): existing_profile["description"] = value
        elif key == "status" and isinstance(value, str) and value.strip(): existing_profile["status"] = value
        elif key not in existing_profile and value is not None: 
            existing_profile[key] = value
        elif value is not None and existing_profile.get(key) != value : 
            existing_profile[key] = value
            
    if dev_key in char_update_data and isinstance(char_update_data[dev_key], str) and char_update_data[dev_key].strip():
        existing_profile[dev_key] = char_update_data[dev_key]
    elif any(k not in ["modification_proposal", provisional_marker_key] and v is not None for k,v in char_update_data.items()): 
        if dev_key not in existing_profile or not existing_profile[dev_key]: 
             char_name = next((k_name for k_name, profile_val in all_character_profiles_for_lookup.items() if profile_val is existing_profile), "UnknownCharacter")
             existing_profile[dev_key] = f"Character '{char_name}' updated or significantly involved in Chapter {existing_profile.get(provisional_marker_key, 'N').split('_')[-1]}."


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
            updated_chars_count += 1
            existing_profile = character_profiles_dict_to_update[char_name]
            if config.ENABLE_DYNAMIC_STATE_ADAPTATION and modification_proposal:
                _apply_state_modification_proposal_logic(existing_profile, modification_proposal, char_name, "existing character profile")
            
            _update_existing_character_profile_fields_internal(existing_profile, char_update_data, dev_key, provisional_marker_key, character_profiles_dict_to_update)
            
            if from_flawed_draft and existing_profile.get(provisional_marker_key) != "provisional_from_unrevised_draft":
                 existing_profile[provisional_marker_key] = "provisional_from_unrevised_draft"

    if updated_chars_count > 0 or new_chars_count > 0:
        logger.info(f"Character profile dict merge complete for ch {chapter_number}. Updated: {updated_chars_count}, New: {new_chars_count}.")
    else:
        logger.info(f"No character profiles were effectively updated or added to dict for ch {chapter_number}.")

def _robust_merge_world_item_data_logic_internal(
    target_dict: Dict[str, Any], 
    update_dict: Dict[str, Any], 
    item_name_for_log: str, 
    chapter_num: int,
    from_flawed_draft_source: bool 
) -> Dict[str, Any]:
    if not isinstance(target_dict, dict): 
        logger.warning(f"World item '{item_name_for_log}' target_dict was not a dict. Initializing as new. Old: '{str(target_dict)[:100]}'")
        current_item_data = {}
    else:
        current_item_data = target_dict 

    item_was_modified_this_call = False
    provisional_marker_key = f"source_quality_chapter_{chapter_num}"

    if from_flawed_draft_source and current_item_data.get(provisional_marker_key) != "provisional_from_unrevised_draft":
        current_item_data[provisional_marker_key] = "provisional_from_unrevised_draft"
        item_was_modified_this_call = True

    if config.ENABLE_DYNAMIC_STATE_ADAPTATION and "modification_proposal" in update_dict:
        proposal = update_dict.pop("modification_proposal") 
        if isinstance(proposal, str) and proposal.strip():
            _apply_state_modification_proposal_logic(current_item_data, proposal, item_name_for_log, "world item")
            item_was_modified_this_call = True
            
    for key, value_from_update in update_dict.items():
        if key in [provisional_marker_key, "modification_proposal"] or \
           key.startswith(("updated_in_chapter_", "added_in_chapter_", "source_quality_chapter_")):
            if key.startswith("elaboration_in_chapter_") and isinstance(value_from_update, str) and value_from_update.strip():
                 current_item_data[key] = value_from_update
                 item_was_modified_this_call = True
            continue

        current_value_in_target = current_item_data.get(key)

        if isinstance(value_from_update, dict): 
            if not isinstance(current_value_in_target, dict):
                current_item_data[key] = {} 
                current_value_in_target = current_item_data[key] # This now points to the new empty dict
                item_was_modified_this_call = True # Marked as modified because a new sub-dict was created
            # Recursive call for nested dicts - ensure correct provisional flag propagation
            _robust_merge_world_item_data_logic_internal( # Modifies current_value_in_target (which is current_item_data[key]) in-place
                current_value_in_target, value_from_update, f"{item_name_for_log}.{key}", chapter_num, from_flawed_draft_source
            )
            # Simplified: assume modification if sub-dict was processed. A more precise check would compare before/after.
            item_was_modified_this_call = True

        elif isinstance(value_from_update, list): 
            if not isinstance(current_value_in_target, list):
                current_item_data[key] = [] 
                current_value_in_target = current_item_data[key] # Update reference
                item_was_modified_this_call = True
            
            initial_list_len = len(current_value_in_target)
            for item_in_list_update in value_from_update:
                if item_in_list_update not in current_value_in_target: 
                    current_value_in_target.append(item_in_list_update)
            if len(current_value_in_target) > initial_list_len : item_was_modified_this_call = True
        
        elif value_from_update != current_value_in_target: 
            current_item_data[key] = value_from_update
            item_was_modified_this_call = True
            
    if item_was_modified_this_call:
        added_key = f"added_in_chapter_{chapter_num}"
        if not current_item_data.get(added_key): 
            current_item_data[f"updated_in_chapter_{chapter_num}"] = True
            
    return current_item_data


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
            existing_item_data = target_category_dict # For _overview_, target_category_dict is the item itself
            if not existing_item_data: # If target_category_dict was {} (e.g. first time for overview)
                 world_building_dict_to_update[category_key] = {} # Ensure it's a dict
                 existing_item_data = world_building_dict_to_update[category_key]

            _robust_merge_world_item_data_logic_internal( # Modifies existing_item_data in-place
                 existing_item_data, category_updates_dict, item_log_name, chapter_number, from_flawed_draft
            )
            # Check if it was marked as updated by the merge logic
            if existing_item_data.get(f"updated_in_chapter_{chapter_number}") or \
               (from_flawed_draft and existing_item_data.get(f"source_quality_chapter_{chapter_number}")):
                items_affected_count +=1
            continue


        for item_name, item_update_details in category_updates_dict.items():
            if not isinstance(item_update_details, dict):
                logger.warning(f"Skipping invalid item_details for '{item_name}' in cat '{category_key}' (not dict) for ch {chapter_number}. Data: {item_update_details}")
                continue

            item_log_name = f"{category_key}.{item_name}"
            existing_item_data = target_category_dict.get(item_name)

            if existing_item_data is None: 
                target_category_dict[item_name] = {} # Create new item entry
                existing_item_data = target_category_dict[item_name]
                existing_item_data[f"added_in_chapter_{chapter_number}"] = True # Mark as added
                items_affected_count +=1 # Count as affected because it's new
            
            _robust_merge_world_item_data_logic_internal( # Modifies existing_item_data in-place
                existing_item_data, item_update_details, item_log_name, chapter_number, from_flawed_draft
            )
            # Check if it was marked as updated and wasn't just added in this call
            if existing_item_data.get(f"updated_in_chapter_{chapter_number}") or \
               (from_flawed_draft and existing_item_data.get(f"source_quality_chapter_{chapter_number}")):
                if not existing_item_data.get(f"added_in_chapter_{chapter_number}"): # Don't double-count if it was just added
                     items_affected_count += 1
    
    if items_affected_count > 0:
        logger.info(f"World-building dict merge complete for ch {chapter_number}. Approx {items_affected_count} items affected/added.")
    else:
        logger.info(f"No world-building dict items were effectively updated or added for ch {chapter_number}.")


async def _prepopulate_kg_from_dicts_internal(
    plot_outline: Dict[str, Any],
    character_profiles: Dict[str, Any],
    world_building: Dict[str, Any]
):
    logger.info("Starting Knowledge Graph pre-population directly from initial data dicts...")
    cypher_statements: List[Tuple[str, Dict[str, Any]]] = []
    novel_id = config.MAIN_NOVEL_INFO_NODE_ID

    if plot_outline:
        novel_props = {k: v for k, v in plot_outline.items() if not isinstance(v, (list, dict)) and v is not None}
        novel_props['id'] = novel_id
        cypher_statements.append((f"MERGE (ni:NovelInfo {{id: $id}}) SET ni = $props", {"id": novel_id, "props": novel_props}))

        plot_points = plot_outline.get('plot_points', [])
        if isinstance(plot_points, list):
            for i, desc in enumerate(plot_points):
                if isinstance(desc, str):
                    pp_id = f"{novel_id}_pp_{i+1}"
                    cypher_statements.append((
                        f"MATCH (ni:NovelInfo {{id: '{novel_id}'}}) MERGE (pp:PlotPoint {{id: $pp_id}}) SET pp.sequence = $seq, pp.description = $desc MERGE (ni)-[:HAS_PLOT_POINT]->(pp)",
                        {"pp_id": pp_id, "seq": i + 1, "desc": desc}
                    ))
                    if i > 0: 
                        prev_pp_id = f"{novel_id}_pp_{i}"
                        cypher_statements.append((
                            f"MATCH (prev_pp:PlotPoint {{id: '{prev_pp_id}'}}) MATCH (curr_pp:PlotPoint {{id: '{pp_id}'}}) MERGE (prev_pp)-[:NEXT_PLOT_POINT]->(curr_pp)", {}
                        ))
    
    for char_name, profile in character_profiles.items():
        if not isinstance(profile, dict): continue
        
        char_props = {k: v for k, v in profile.items() if isinstance(v, (str, int, float, bool)) and v is not None}
        char_props_for_set = char_props.copy()
        char_props_for_set['name'] = char_name 

        cypher_statements.append(("MERGE (c:Character:Entity {name: $name}) SET c += $props", {"name": char_name, "props": char_props_for_set}))

        if isinstance(profile.get("traits"), list):
            for trait in profile["traits"]:
                if isinstance(trait, str):
                    cypher_statements.append(("MATCH (c:Character {name: $char_name}) MERGE (t:Trait {name: $trait_name}) MERGE (c)-[:HAS_TRAIT]->(t)", {"char_name": char_name, "trait_name": trait}))
        
        if isinstance(profile.get("relationships"), dict):
            for target_name, rel_detail in profile["relationships"].items():
                rel_type = "RELATED_TO"; rel_props_dict = {"description": str(rel_detail)}
                if isinstance(rel_detail, dict): 
                    rel_type = str(rel_detail.get("type", rel_type)).upper().replace(" ", "_")
                    rel_props_dict = {k:v for k,v in rel_detail.items() if isinstance(v, (str, int, float, bool))}
                    rel_props_dict.setdefault("description", f"{rel_type} {target_name}")
                elif isinstance(rel_detail, str): 
                    rel_type = rel_detail.upper().replace(" ", "_"); rel_props_dict = {"description": rel_detail}
                
                rel_props_dict.setdefault("chapter_added", config.KG_PREPOPULATION_CHAPTER_NUM)
                is_prov_initial = profile.get(f"source_quality_chapter_{config.KG_PREPOPULATION_CHAPTER_NUM}") == "provisional_from_unrevised_draft"
                rel_props_dict.setdefault("is_provisional", is_prov_initial)

                cypher_statements.append((
                    """
                    MATCH (c1:Character:Entity {name: $char_name1})
                    MERGE (c2:Entity {name: $char_name2}) 
                        ON CREATE SET c2:Character, c2.description = 'Auto-created via relationship from ' + $char_name1, c2.name = $char_name2
                        ON MATCH SET c2:Character 
                    MERGE (c1)-[r:DYNAMIC_REL {type:$rel_type_val}]->(c2)
                    SET r += $rel_props_val
                    """,
                    {"char_name1": char_name, "char_name2": target_name, "rel_type_val": rel_type, "rel_props_val": rel_props_dict}
                ))

    for category, items in world_building.items():
        if category == "_overview_" and isinstance(items, dict) and "description" in items:
            overview_props = {"id": config.MAIN_WORLD_CONTAINER_NODE_ID, "overview_description": items["description"]}
            if items.get(f"source_quality_chapter_{config.KG_PREPOPULATION_CHAPTER_NUM}") == "provisional_from_unrevised_draft": overview_props["is_provisional"] = True
            cypher_statements.append((f"MERGE (wc:WorldContainer {{id: $id}}) SET wc = $props", {"id": config.MAIN_WORLD_CONTAINER_NODE_ID, "props": overview_props}))
            continue

        if not isinstance(items, dict) or category in ["is_default", "source", "user_supplied_data"]:
            continue

        for item_name, details in items.items():
            if not isinstance(details, dict) or item_name.startswith(("_", "source_quality_chapter_")): continue
            
            we_id = f"{category}_{item_name}".replace(" ", "_").replace("'", "").lower()
            item_props = {k: v for k, v in details.items() if isinstance(v, (str, int, float, bool)) and v is not None}
            item_props.update({'id': we_id, 'name': item_name, 'category': category})

            created_chapter_num_initial = details.get(f"added_in_chapter_{config.KG_PREPOPULATION_CHAPTER_NUM}", config.KG_PREPOPULATION_CHAPTER_NUM)
            item_props['created_chapter'] = created_chapter_num_initial
            if details.get(f"source_quality_chapter_{created_chapter_num_initial}") == "provisional_from_unrevised_draft": item_props['is_provisional'] = True
            
            cypher_statements.append(("MERGE (we:WorldElement {id: $id}) SET we += $props", {"id": we_id, "props": item_props}))

            for list_prop_name in ["goals", "rules", "key_elements", "traits"]:
                if isinstance(details.get(list_prop_name), list):
                    for val_item in details[list_prop_name]:
                        if isinstance(val_item, str):
                            rel_name_base = list_prop_name.upper().rstrip('S')
                            if list_prop_name == "key_elements": rel_name_base = "KEY_ELEMENT"
                            elif list_prop_name == "traits": rel_name_base = "TRAIT_ASPECT"
                            rel_name_final = f"HAS_{rel_name_base}"
                            
                            query_str_for_list_prop = f"""
                                MATCH (we:WorldElement {{id: $we_id}})
                                MERGE (v:ValueNode {{value: $val_item_value, type: $value_node_type}})
                                MERGE (we)-[:{rel_name_final}]->(v)
                            """
                            cypher_statements.append((query_str_for_list_prop, 
                                                      {"we_id": we_id, "val_item_value": val_item, "value_node_type": list_prop_name}))
    
    if cypher_statements:
        try:
            await neo4j_manager.execute_cypher_batch(cypher_statements) # MODIFIED
            logger.info(f"KG pre-population complete: Executed {len(cypher_statements)} Cypher statements directly from initial data.")
        except Exception as e: logger.error(f"Error during direct KG pre-population batch execution: {e}", exc_info=True)
    else: logger.info("No Cypher statements generated for KG pre-population from initial data.")


class KGMaintainerAgent:
    def __init__(self, model_name: str = config.KNOWLEDGE_UPDATE_MODEL):
        self.model_name = model_name
        logger.info(f"KGMaintainerAgent initialized with model: {self.model_name}")

    async def summarize_chapter(self, chapter_text: Optional[str], chapter_number: int) -> Tuple[Optional[str], Optional[Dict[str, int]]]:
        """Generates a summary for a chapter text. Returns summary and LLM usage."""
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
                    special_list_handling=CHAR_UPDATE_PKVB_SPECIAL_HANDLING # MODIFIED: Use specific handling for PKVB
                )
                
                # Custom post-processing for relationships using CHAR_UPDATE_RELATIONSHIP_POST_PARSING_HANDLING
                if "relationships" in parsed_char_data and isinstance(parsed_char_data["relationships"], list):
                    rels_dict = {}
                    # Relationship lines might be parsed as single strings by parse_key_value_block if they have prefixes like "- "
                    # Or, if Relationships: Target:Type format is used on one line, it might be a single string.
                    for rel_str_or_item in parsed_char_data["relationships"]:
                        if isinstance(rel_str_or_item, str) and ':' in rel_str_or_item:
                            parts = rel_str_or_item.split(":", 1)
                            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                                rels_dict[parts[0].strip()] = parts[1].strip()
                            else: logger.warning(f"Malformed relationship string for {char_name}: '{rel_str_or_item}'")
                        elif isinstance(rel_str_or_item, str) and rel_str_or_item.strip(): # A single name, assume 'related'
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
        """Performs unified knowledge extraction. Returns extraction dict and LLM usage."""
        logger.info(f"KGMaintainerAgent performing unified knowledge extraction (plain text) for Chapter {chapter_number}...")
        if not chapter_text:
            logger.warning(f"Unified knowledge extraction skipped for Ch {chapter_number}: empty chapter text.")
            return {"character_updates": {}, "world_updates": {}, "knowledge_triples": []}, None

        protagonist_name = novel_props.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME)
        
        current_profiles_plain_text = await get_filtered_character_profiles_for_prompt_plain_text(novel_props, chapter_number - 1)
        current_world_plain_text = await get_filtered_world_data_for_prompt_plain_text(novel_props, chapter_number - 1)
        candidate_entities_list = await heuristic_entity_spotter_for_kg(novel_props, chapter_text)
        candidate_entities_text = "Candidate Entities (for KG triple focus):\n" + "\n".join([f"- {e}" for e in candidate_entities_list]) if candidate_entities_list else "Candidate Entities: None identified by heuristic."

        dynamic_instr_char = ""
        dynamic_instr_world = ""
        if config.ENABLE_DYNAMIC_STATE_ADAPTATION:
            dynamic_instr_char = (
                f"For existing characters, if their traits, status, or core description needs modification based on THIS chapter's events, "
                f"include a line like `Modification Proposal: MODIFY traits: ADD \"Determined\", REMOVE \"Hesitant\"`. "
                f"Also specify all current `Traits` as a comma-separated list if changed, new `Status`, new `Description`. "
                f"For NEW characters, provide `Description`, `Traits` (comma-separated), `Status`. "
                f"For `Relationships`, list them as `Target Name: relationship type` (e.g., `John Doe: ally`) or on new lines under a `Relationships:` header, each as `- Target Name: type`."
                f"Only include characters that are updated, newly introduced, or have a modification proposal."
            )
            dynamic_instr_world = (
                f"For existing world items, if their properties need modification, include a line like `Modification Proposal: MODIFY atmosphere: \"Now heavy with magical fallout\"`. "
                f"Also provide the new full value for any changed properties. For NEW world items, provide all known properties. "
                f"E.g., for locations: `Description: ...`, `Atmosphere: ...`. For factions: `Description: ...`, `Goals:` (followed by '- goal' lines). "
                f"Only include world elements that are new, significantly changed by THIS chapter's events, or have a modification proposal."
            )
        else: 
            dynamic_instr_char = "Only include characters whose information is directly updated or those newly introduced in THIS chapter. Provide full description, traits (comma-separated), status for new chars."
            dynamic_instr_world = "Only include world elements that are new or significantly changed by THIS chapter's events. Provide full details for new items."

        common_predicates_str = ", ".join([
            "is_a", "located_in", "has_trait", "status_is", "feels", "knows", "believes", "wants",
            "interacted_with", "travelled_to", "discovered", "acquired", "lost", "used_item",
            "attacked", "helped", "part_of", "caused_by", "leads_to", "observed", "heard", "said",
            "thought_about", "decided_to", "has_goal", "has_feature", "related_to", "member_of",
            "leader_of", "enemy_of", "ally_of", "works_for", "has_ability", "possesses", "created_by",
            "has_description", "has_atmosphere", "has_rule", "has_history_event"
        ])

        prompt = f"""/no_think
You are a comprehensive literary analyst and knowledge engineer.
Analyze the **Complete Chapter {chapter_number} Text** (protagonist: {protagonist_name}) and extract information for three distinct knowledge bases.
Output ONLY plain text, structured as described below.

**Reference Information (Current State Before This Chapter - for context only, extract from THIS chapter's text):**
  **Character Profiles Snapshot (Plain Text):**
  ```text
  {current_profiles_plain_text if current_profiles_plain_text.strip() else "No character profiles provided as reference."}
  ```
  **World Building Snapshot (Plain Text):**
  ```text
  {current_world_plain_text if current_world_plain_text.strip() else "No world building data provided as reference."}
  ```
  {candidate_entities_text}

**Complete Chapter {chapter_number} Text (Analyze this full text):**
--- BEGIN COMPLETE CHAPTER TEXT ---
{chapter_text}
--- END COMPLETE CHAPTER TEXT ---

**Output Format (CRITICAL - PLAIN TEXT ONLY):**
Use the following section headers EXACTLY:
`### CHARACTER UPDATES ###`
`### WORLD UPDATES ###`
`### KG TRIPLES ###`

**1. Under `### CHARACTER UPDATES ###`:**
   List each character on a new line: `Character: [Character Name]`
   Followed by indented key-value pairs for their profile data (use Title Case for keys like "Description", "Traits", "Status"):
     `Description: [Full description if new or significantly changed]`
     `Traits: [Comma-separated list, e.g., Brave, Cautious OR list with '-' prefix on new lines]`
     `Status: [Current status]`
     `Relationships:` (Optional, if any. List each as "- Target Name: Relationship Type", e.g., "- Gorok: uneasy ally")
     `Development in Chapter {chapter_number}: [Summary of their role/changes in THIS chapter]`
     `Modification Proposal: [Optional: MODIFY key: value_change]`
   {dynamic_instr_char}

**2. Under `### WORLD UPDATES ###`:**
   Start each category with: `Category: [locations | society | systems | lore | history | factions | _overview_]`
   For `_overview_`, provide: `Description: [Overall world feel change or new general description]` (and other relevant overview keys if applicable)
   For other categories, list items: `Item: [Item Name]`
   Followed by indented key-value pairs for item details (use Title Case for keys like "Description", "Atmosphere"):
     `Description: [Full description if new or changed]`
     `Atmosphere: [For locations]`
     `Goals:` (For factions, list each goal on a new line starting with "- ")
       `- Goal 1`
     `Elaboration in Chapter {chapter_number}: [Context from THIS chapter]`
     `Modification Proposal: [Optional: MODIFY key: value_change]`
   {dynamic_instr_world}

**3. Under `### KG TRIPLES ###`:**
   List each factual triple. Choose ONE format and use it consistently for all triples in this section:
   Format A (preferred): `Subject | Predicate | Object`
   Format B: `Subject: [Subject Name], Predicate: [predicate_name], Object: [Object Name/Value]`
   Format C: `- [Subject, Predicate, Object]`
   Use predicates from this suggested list where appropriate: {common_predicates_str}. Focus on NEW facts or significant CHANGES from THIS chapter.

Begin your output now:
"""
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
                            kg_queries.add_kg_triple_to_db(subj, pred, obj_truncated, chapter_number, is_provisional=is_from_flawed_draft) # MODIFIED
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
        await _prepopulate_kg_from_dicts_internal(plot_outline, character_profiles, world_building)


    async def heal_canon(self, identified_contradictions: List[Dict]) -> bool:
        logger.warning("heal_canon method is a placeholder and not fully implemented.")
        if not identified_contradictions:
            logger.info("No contradictions provided to heal_canon.")
            return True 

        for contradiction in identified_contradictions:
            logger.info(f"Placeholder: Attempting to address contradiction: {contradiction}")
        
        return False