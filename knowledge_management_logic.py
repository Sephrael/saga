# knowledge_management_logic.py
"""
Handles updates to knowledge bases (Python profiles, Knowledge Graph)
and summarization for the SAGA system.
LLM outputs for knowledge extraction are now plain text.
"""
import logging
import json # Retained for debug dumps, not for LLM prompt formatting/parsing
import re
import asyncio
from typing import Dict, List, Optional, Any, Tuple

from async_lru import alru_cache 

import config
import llm_interface

from prompt_data_getters import (
    get_filtered_character_profiles_for_prompt_plain_text, 
    heuristic_entity_spotter_for_kg,
    get_filtered_world_data_for_prompt_plain_text 
)
from state_manager import state_manager

logger = logging.getLogger(__name__)

# --- Summarization ---

@alru_cache(maxsize=config.SUMMARY_CACHE_SIZE)
async def llm_summarize_full_chapter_text_logic(chapter_text_full_key: str, chapter_number: int) -> str:
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
    summary_raw = await llm_interface.async_call_llm(
        model_name=config.SMALL_MODEL, 
        prompt=prompt,
        temperature=0.6,
        max_tokens=config.MAX_SUMMARY_TOKENS, 
        stream_to_disk=False 
    )
    return llm_interface.clean_model_response(summary_raw).strip()

async def summarize_chapter_text_logic(chapter_text: Optional[str], chapter_number: int) -> Optional[str]:
    if not chapter_text or len(chapter_text) < 50:
        logger.warning(f"Chapter {chapter_number} text too short for summarization ({len(chapter_text or '')} chars).")
        return None
            
    cleaned_summary = await llm_summarize_full_chapter_text_logic(chapter_text, chapter_number)

    if cleaned_summary: 
        logger.info(f"Generated summary for ch {chapter_number}: '{cleaned_summary[:100].strip()}...'")
        return cleaned_summary
    
    logger.warning(f"Failed to generate a valid summary for ch {chapter_number} via LLM.")
    return None

# --- JSON State Modification Proposal Logic (Operates on in-memory Python dicts) ---
# This logic remains as it applies proposals to Python dicts, which are still used internally.
# The `proposal_str` itself could come from LLM plain text parsing, not directly from LLM JSON.

def _apply_trait_modification(current_traits_list: List[str], modification_details_str: str) -> List[str]:
    traits_set = set(current_traits_list)
    for add_match in re.finditer(r"ADD\s+['\"]([^'\"]+)['\"]", modification_details_str, re.IGNORECASE):
        trait_to_add = add_match.group(1).strip()
        if trait_to_add: traits_set.add(trait_to_add)
    for remove_match in re.finditer(r"REMOVE\s+['\"]([^'\"]+)['\"]", modification_details_str, re.IGNORECASE):
        trait_to_remove = remove_match.group(1).strip()
        if trait_to_remove: traits_set.discard(trait_to_remove)
    return sorted(list(traits_set))

def apply_state_modification_proposal_logic(
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

# --- Character Profile Merging Logic ---
def _initialize_new_character_profile(
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
    if dev_key in char_update_data: new_profile[dev_key] = char_update_data[dev_key]
    if provisional_marker_key in char_update_data: new_profile[provisional_marker_key] = char_update_data[provisional_marker_key]
    logger.info(f"Prepared new character profile for '{char_name}'.")
    return new_profile

def _update_existing_character_profile_fields(
    existing_profile: Dict[str, Any],
    char_update_data: Dict[str, Any],
    dev_key: str,
    provisional_marker_key: str
):
    if provisional_marker_key in char_update_data: existing_profile[provisional_marker_key] = char_update_data[provisional_marker_key]
    for key, value in char_update_data.items():
        if key in ["modification_proposal", provisional_marker_key, dev_key]: continue
        if key == "traits" and isinstance(value, list):
            if "traits" not in existing_profile or not isinstance(existing_profile["traits"], list): existing_profile["traits"] = []
            valid_new_traits = {t for t in value if isinstance(t, str) and t.strip()}
            existing_profile["traits"] = sorted(list(set(existing_profile["traits"]).union(valid_new_traits)))
        elif key == "relationships" and isinstance(value, dict):
            if not isinstance(existing_profile.get("relationships"), dict): existing_profile["relationships"] = {}
            existing_profile["relationships"].update(value)
        elif key == "description" and isinstance(value, str) and value.strip(): existing_profile["description"] = value
        elif key == "status" and isinstance(value, str) and value.strip(): existing_profile["status"] = value
        elif key not in existing_profile and value is not None: existing_profile[key] = value 
    if dev_key in char_update_data and isinstance(char_update_data[dev_key], str) and char_update_data[dev_key].strip():
        existing_profile[dev_key] = char_update_data[dev_key]

def merge_character_profile_updates_logic(
    agent, updates_from_llm: Dict[str, Any], chapter_number: int, from_flawed_draft: bool
):
    """ Merges LLM updates (now parsed from plain text into dicts) into the agent's in-memory character_profiles dict. """
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
        if from_flawed_draft: char_update_data[provisional_marker_key] = "provisional_from_unrevised_draft"
        
        dev_key = f"development_in_chapter_{chapter_number}"
        
        modification_proposal = char_update_data.get("modification_proposal") # Key from parser might be "modification_proposal"
        
        if char_name not in agent.character_profiles:
            new_chars_count += 1
            new_profile = _initialize_new_character_profile(char_name, char_update_data, chapter_number, provisional_marker_key, dev_key)
            agent.character_profiles[char_name] = new_profile
            if config.ENABLE_DYNAMIC_STATE_ADAPTATION and modification_proposal:
                apply_state_modification_proposal_logic(agent.character_profiles[char_name], modification_proposal, char_name, "new character profile")
        else:
            updated_chars_count += 1
            existing_profile = agent.character_profiles[char_name]
            if config.ENABLE_DYNAMIC_STATE_ADAPTATION and modification_proposal:
                apply_state_modification_proposal_logic(existing_profile, modification_proposal, char_name, "existing character profile")
            _update_existing_character_profile_fields(existing_profile, char_update_data, dev_key, provisional_marker_key)
            if from_flawed_draft: existing_profile[provisional_marker_key] = "provisional_from_unrevised_draft"


    if updated_chars_count > 0 or new_chars_count > 0:
        logger.info(f"Character profile Python dict merge complete for ch {chapter_number}. Updated: {updated_chars_count}, New: {new_chars_count}.")
    else:
        logger.info(f"No character profiles were effectively updated or added to Python dicts for ch {chapter_number} after LLM analysis.")

# --- World Item Merging Logic ---
def robust_merge_world_item_data_logic(
    target_dict: Dict[str, Any], update_dict: Dict[str, Any], item_name_for_log: str, chapter_num: int, from_flawed_draft_source: bool
) -> Dict[str, Any]:
    if not isinstance(target_dict, dict):
        logger.warning(f"World item '{item_name_for_log}' target_dict was not a dict. Initializing as new. Old: '{str(target_dict)[:100]}'")
        current_item_data = {}
    else:
        current_item_data = target_dict
    
    item_was_modified_this_call = False
    provisional_marker_key = f"source_quality_chapter_{chapter_num}"
    
    if from_flawed_draft_source:
        current_item_data[provisional_marker_key] = "provisional_from_unrevised_draft"
        item_was_modified_this_call = True
        
    if config.ENABLE_DYNAMIC_STATE_ADAPTATION and "modification_proposal" in update_dict: # Key from parser might be "modification_proposal"
        proposal = update_dict.pop("modification_proposal")
        if isinstance(proposal, str) and proposal.strip():
            apply_state_modification_proposal_logic(current_item_data, proposal, item_name_for_log, "world item")
            item_was_modified_this_call = True
            
    for key, value_from_update in update_dict.items():
        if key in [provisional_marker_key, "modification_proposal"] or key.startswith(("updated_in_chapter_", "added_in_chapter_", "source_quality_chapter_")):
            if key.startswith("elaboration_in_chapter_") and isinstance(value_from_update, str) and value_from_update.strip():
                 current_item_data[key] = value_from_update 
                 item_was_modified_this_call = True
            continue
        
        current_value_in_target = current_item_data.get(key)
        if isinstance(value_from_update, dict): 
            merged_sub_dict = robust_merge_world_item_data_logic(
                current_value_in_target if isinstance(current_value_in_target, dict) else {},
                value_from_update, f"{item_name_for_log}.{key}", chapter_num, from_flawed_draft_source=False 
            )
            if merged_sub_dict != current_value_in_target: item_was_modified_this_call = True
            current_item_data[key] = merged_sub_dict
        elif isinstance(value_from_update, list): 
            if not isinstance(current_value_in_target, list):
                current_item_data[key] = []
                item_was_modified_this_call = True
            initial_list_len = len(current_item_data[key])
            for item_in_list_update in value_from_update:
                if item_in_list_update not in current_item_data[key]: current_item_data[key].append(item_in_list_update)
            if len(current_item_data[key]) > initial_list_len: item_was_modified_this_call = True
        elif value_from_update != current_value_in_target: 
            current_item_data[key] = value_from_update
            item_was_modified_this_call = True
            
    if item_was_modified_this_call and not current_item_data.get(f"added_in_chapter_{chapter_num}"):
        current_item_data[f"updated_in_chapter_{chapter_num}"] = True
        
    return current_item_data

def merge_world_item_updates_logic(
    agent, updates_from_llm: Dict[str, Any], chapter_number: int, from_flawed_draft: bool
):
    """ Merges LLM updates (now parsed from plain text into dicts) into the agent's in-memory world_building dict. """
    if not updates_from_llm:
        logger.info(f"No world-building updates from LLM to merge for ch {chapter_number}.")
        return
    logger.info(f"Merging world-building updates for ch {chapter_number}. Categories in update: {list(updates_from_llm.keys())}")
    items_affected_count = 0
    for category_key, category_updates_dict in updates_from_llm.items():
        if not isinstance(category_updates_dict, dict) or not category_updates_dict:
            logger.debug(f"Skipping empty or invalid update for world category '{category_key}' in ch {chapter_number}.")
            continue
        
        if category_key not in agent.world_building: agent.world_building[category_key] = {}
        elif not isinstance(agent.world_building[category_key], dict): 
            logger.warning(f"Overwriting non-dictionary world category '{category_key}' with new dictionary structure for ch {chapter_number}.")
            agent.world_building[category_key] = {}
            
        target_category_dict = agent.world_building[category_key]
        
        for item_name, item_update_details in category_updates_dict.items():
            if not isinstance(item_update_details, dict):
                logger.warning(f"Skipping invalid item_details for '{item_name}' in cat '{category_key}' (not dict) for ch {chapter_number}. Data: {item_update_details}")
                continue
            
            item_log_name = f"{category_key}.{item_name}"
            existing_item_data = target_category_dict.get(item_name)
            
            merged_item_data = robust_merge_world_item_data_logic(
                existing_item_data if existing_item_data is not None else {},
                item_update_details, item_log_name, chapter_number, from_flawed_draft
            )
            target_category_dict[item_name] = merged_item_data
            
            if existing_item_data is None: 
                merged_item_data[f"added_in_chapter_{chapter_number}"] = True 
                items_affected_count += 1
            elif merged_item_data.get(f"updated_in_chapter_{chapter_number}") or \
                 (from_flawed_draft and merged_item_data.get(f"source_quality_chapter_{chapter_number}")):
                items_affected_count += 1 

        if any(isinstance(v,dict) and (v.get(f"updated_in_chapter_{chapter_number}") or v.get(f"added_in_chapter_{chapter_number}")) 
               for v in target_category_dict.values()):
             target_category_dict[f"category_updated_in_chapter_{chapter_number}"] = True 

    if items_affected_count > 0:
        logger.info(f"World-building Python dict merge complete for ch {chapter_number}. Approx {items_affected_count} items affected/added.")
    else:
        logger.info(f"No world-building Python dict items were effectively updated or added for ch {chapter_number} after LLM analysis.")

# --- Unified Knowledge Extraction (Plain Text Parsing) ---

def _parse_plain_text_character_updates(text_block: str, chapter_number: int) -> Dict[str, Any]:
    char_updates: Dict[str, Any] = {}
    current_char_name: Optional[str] = None
    current_char_data: Dict[str, Any] = {}
    
    char_name_re = re.compile(r"^\s*Character:\s*(.+)$", re.IGNORECASE)
    key_value_re = re.compile(r"^\s*([A-Za-z0-9\s_()]+(?:\s*in\s*Chapter\s*\d+)?):\s*(.*)$", re.IGNORECASE) # Allow parentheses in keys e.g. Name (role)
    list_item_re = re.compile(r"^\s*-\s*(.+)$")

    active_list_key: Optional[str] = None
    active_list_values: List[str] = []

    def finalize_char():
        nonlocal current_char_name, current_char_data, active_list_key, active_list_values
        if active_list_key and active_list_values:
            # Special handling for relationships if it's a list of "Target: Type"
            if active_list_key == "relationships" and all([':' in v for v in active_list_values]):
                rels_dict = {}
                for rel_str in active_list_values:
                    parts = rel_str.split(":", 1)
                    if len(parts) == 2:
                        rels_dict[parts[0].strip()] = parts[1].strip()
                current_char_data[active_list_key] = rels_dict
            else:
                current_char_data[active_list_key] = active_list_values
            active_list_key = None
            active_list_values = []
        if current_char_name and current_char_data:
            dev_key = f"development_in_chapter_{chapter_number}"
            if dev_key not in current_char_data and any(k != "modification_proposal" for k in current_char_data):
                current_char_data[dev_key] = "Character appeared or was mentioned in this chapter."
            char_updates[current_char_name] = current_char_data
        current_char_name = None # type: ignore
        current_char_data = {}

    for line in text_block.splitlines():
        line = line.strip()
        if not line: continue

        char_match = char_name_re.match(line)
        if char_match:
            finalize_char()
            current_char_name = char_match.group(1).strip()
            current_char_data = {}
            active_list_key = None
            active_list_values = []
            continue

        if not current_char_name: continue

        list_item_match = list_item_re.match(line)
        if active_list_key and list_item_match:
            active_list_values.append(list_item_match.group(1).strip())
            continue
        elif active_list_key and not list_item_match: 
            if active_list_key == "relationships" and all([':' in v for v in active_list_values]):
                rels_dict = {}
                for rel_str in active_list_values:
                    parts = rel_str.split(":", 1)
                    if len(parts) == 2: rels_dict[parts[0].strip()] = parts[1].strip()
                current_char_data[active_list_key] = rels_dict
            else:
                current_char_data[active_list_key] = active_list_values
            active_list_key = None
            active_list_values = []

        kv_match = key_value_re.match(line)
        if kv_match:
            key_raw = kv_match.group(1).strip()
            value = kv_match.group(2).strip()
            
            key = key_raw.lower().replace(" ", "_").replace("(", "").replace(")", "") # Normalize
            if key.startswith("development_in_chapter_"): # Keep specific dev key format
                 key = key_raw.strip().replace(" ", "_")


            if key in ["traits", "relationships"]: 
                if value: # Content on same line
                    if key == "traits":
                        current_char_data[key] = [v.strip() for v in value.split(',') if v.strip()]
                    elif key == "relationships":
                        # Handle "Target1: Type1, Target2: Type2" or just "Target1, Target2"
                        rels = {}
                        pairs = value.split(',')
                        for pair_str in pairs:
                            target_name_part, rel_type_part = pair_str.strip(), "related" # Default
                            if ':' in pair_str:
                                target_name_part, rel_type_part = [p.strip() for p in pair_str.split(':', 1)]
                            elif '(' in pair_str and ')' in pair_str:
                                t_match = re.match(r"(.+?)\s*\((.+?)\)", pair_str)
                                if t_match:
                                    target_name_part, rel_type_part = t_match.group(1).strip(), t_match.group(2).strip()
                            if target_name_part:
                                rels[target_name_part] = rel_type_part
                        current_char_data[key] = rels
                else: # Start of a list/object to be populated by subsequent "- item" lines
                    active_list_key = key 
                    active_list_values = []
                    current_char_data[key] = {} if key == "relationships" else [] # Init as dict or list
            else:
                current_char_data[key] = value
    
    finalize_char()
    return char_updates

def _parse_plain_text_world_updates(text_block: str, chapter_number: int) -> Dict[str, Any]:
    world_updates: Dict[str, Any] = {}
    current_category: Optional[str] = None
    current_item_name: Optional[str] = None
    current_item_data: Dict[str, Any] = {}
    
    category_re = re.compile(r"^\s*Category:\s*(.+)$", re.IGNORECASE)
    item_re = re.compile(r"^\s*Item:\s*(.+)$", re.IGNORECASE)
    key_value_re = re.compile(r"^\s*([A-Za-z0-9\s_]+(?:\s*in\s*Chapter\s*\d+)?):\s*(.*)$", re.IGNORECASE)
    list_item_re = re.compile(r"^\s*-\s*(.+)$")

    active_list_key: Optional[str] = None
    active_list_values: List[str] = []
    
    valid_world_categories = ["locations", "society", "systems", "lore", "history", "factions", "_overview_"]

    def finalize_item():
        nonlocal current_category, current_item_name, current_item_data, active_list_key, active_list_values
        if active_list_key and active_list_values:
            current_item_data[active_list_key] = active_list_values
            active_list_key = None
            active_list_values = []
        if current_category and current_item_name and current_item_data:
            if current_category not in world_updates:
                world_updates[current_category] = {}
            dev_key = f"elaboration_in_chapter_{chapter_number}"
            if dev_key not in current_item_data and any(k != "modification_proposal" for k in current_item_data) :
                current_item_data[dev_key] = f"Item '{current_item_name}' in category '{current_category}' was mentioned or interacted with."
            world_updates[current_category][current_item_name] = current_item_data
        elif current_category == "_overview_" and current_item_data: # Overview has no item name
            world_updates["_overview_"] = current_item_data

        current_item_name = None # type: ignore
        current_item_data = {}

    for line in text_block.splitlines():
        line = line.strip()
        if not line: continue

        cat_match = category_re.match(line)
        if cat_match:
            finalize_item() 
            potential_category = cat_match.group(1).strip().lower().replace(" ", "_")
            if potential_category in valid_world_categories:
                current_category = potential_category
                if current_category == "_overview_": # Overview is special, doesn't have "items"
                    current_item_name = "_overview_item_" # Dummy name for logic flow
                    current_item_data = {} # Reset for overview
                else:
                    current_item_name = None # Reset for new category's items
                # current_item_data = {} # This was reset too early, moved it inside _overview_ check
            else:
                logger.warning(f"Unknown world category '{potential_category}' in unified extraction. Skipping.")
                current_category = None 
            active_list_key = None
            active_list_values = []
            continue

        if not current_category: continue 

        if current_category != "_overview_":
            item_match = item_re.match(line)
            if item_match:
                finalize_item() 
                current_item_name = item_match.group(1).strip()
                current_item_data = {}
                active_list_key = None
                active_list_values = []
                continue
        
        target_data_dict = current_item_data # For items or overview details
        if not current_item_name and current_category != "_overview_": continue # Wait for an item if not overview


        list_item_match = list_item_re.match(line)
        if active_list_key and list_item_match:
            active_list_values.append(list_item_match.group(1).strip())
            continue
        elif active_list_key and not list_item_match: 
            target_data_dict[active_list_key] = active_list_values
            active_list_key = None
            active_list_values = []
        
        kv_match = key_value_re.match(line)
        if kv_match:
            key_raw = kv_match.group(1).strip()
            value = kv_match.group(2).strip()
            key = key_raw.lower().replace(" ", "_")
            if key.startswith("elaboration_in_chapter_"):
                 key = key_raw.strip().replace(" ", "_")

            if key in ["goals", "rules", "key_elements", "traits"]: 
                active_list_key = key
                active_list_values = []
                target_data_dict[key] = [] 
                if value and not (value.startswith("- ") or value.startswith("* ")):
                     active_list_values.append(value)
                elif value and (value.startswith("- ") or value.startswith("* ")):
                     active_list_values.append(value[2:].strip())
            else:
                target_data_dict[key] = value
                
    finalize_item() 
    return world_updates

def _parse_plain_text_kg_triples(text_block: str) -> List[List[str]]:
    triples: List[List[str]] = []
    # Example formats:
    # Subject: Alice, Predicate: friend_of, Object: Bob
    # Alice | friend_of | Bob
    # - [Alice, friend_of, Bob]  (more direct if LLM can do this)
    # - Subject: Alice; Predicate: friend_of; Object: Bob

    # Try matching typical list format first if LLM uses it
    list_format_match = re.findall(r"^\s*-\s*\[\s*['\"]?([^,'\"\[\]]+?)['\"]?\s*,\s*['\"]?([^,'\"\[\]]+?)['\"]?\s*,\s*['\"]?([^,'\"\[\]]+?)['\"]?\s*\]", text_block, re.MULTILINE)
    for s, p, o in list_format_match:
        if s.strip() and p.strip() and o.strip():
            triples.append([s.strip(), p.strip(), o.strip()])
    if triples: # If we found some with this format, assume it's the primary one
        return triples

    # Fallback to other formats
    triple_re_sp_o = re.compile(r"^\s*(?:Subject:\s*(.+?)\s*,\s*Predicate:\s*(.+?)\s*,\s*Object:\s*(.+?)|Subject:\s*(.+?)\s*;\s*Predicate:\s*(.+?)\s*;\s*Object:\s*(.+?))\s*$", re.IGNORECASE | re.MULTILINE)
    pipe_re = re.compile(r"^\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*$", re.MULTILINE)

    for line in text_block.splitlines():
        line = line.strip()
        if not line: continue
        
        s, p, o = None, None, None
        match_spo = triple_re_sp_o.match(line)
        if match_spo:
            if match_spo.group(1) is not None: # Comma separated
                s, p, o = match_spo.group(1).strip(), match_spo.group(2).strip(), match_spo.group(3).strip()
            elif match_spo.group(4) is not None: # Semicolon separated
                s, p, o = match_spo.group(4).strip(), match_spo.group(5).strip(), match_spo.group(6).strip()
        else:
            match_pipe = pipe_re.match(line)
            if match_pipe:
                s, p, o = match_pipe.group(1).strip(), match_pipe.group(2).strip(), match_pipe.group(3).strip()
        
        if s and p and o:
            triples.append([s, p, o])
        elif line.count(',') == 2: # Last resort simple comma split
            parts = [part.strip() for part in line.split(',')]
            if len(parts) == 3 and all(parts):
                triples.append(parts)
            else: logger.warning(f"Could not parse line as KG triple: '{line}'")
        elif line.startswith("- ") and line.count(',') == 2 : # e.g. "- S, P, O"
             parts = [part.strip() for part in line[2:].split(',')]
             if len(parts) == 3 and all(parts):
                triples.append(parts)
             else: logger.warning(f"Could not parse line as KG triple: '{line}'")
        else:
             logger.warning(f"Could not parse line as KG triple: '{line}'")


    return triples


async def unified_knowledge_extraction(
    agent, 
    chapter_text: str, 
    chapter_number: int
) -> Dict[str, Any]:
    logger.info(f"Performing unified knowledge extraction (plain text) for Chapter {chapter_number}...")
    if not chapter_text:
        logger.warning(f"Unified knowledge extraction skipped for Ch {chapter_number}: empty chapter text.")
        return {"character_updates": {}, "world_updates": {}, "knowledge_triples": []}

    protagonist_name = agent.plot_outline.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME)
    
    # For Chapter 1, chapter_number - 1 will be 0. These getters must handle it.
    current_profiles_plain_text = await get_filtered_character_profiles_for_prompt_plain_text(agent, chapter_number - 1)
    current_world_plain_text = await get_filtered_world_data_for_prompt_plain_text(agent, chapter_number - 1)
    
    candidate_entities_list = await heuristic_entity_spotter_for_kg(agent, chapter_text)
    candidate_entities_text = "Candidate Entities (for KG triple focus):\n" + "\n".join([f"- {e}" for e in candidate_entities_list]) if candidate_entities_list else "Candidate Entities: None identified."

    dynamic_instr_char = (
        f"For existing characters, if their traits, status, or core description needs modification based on THIS chapter's events, "
        f"include a line like `Modification Proposal: MODIFY traits: ADD 'Determined', REMOVE 'Hesitant'`. "
        f"Also specify all current `Traits` as a comma-separated list if changed, new `Status`, new `Description`. "
        f"For NEW characters, provide `Description`, `Traits` (comma-separated), `Status`. "
        f"For `Relationships`, list them as `Target Name: relationship type` (e.g., `John Doe: ally`) or on new lines under a `Relationships:` header, each as `- Target Name: type`."
        f"Only include characters that are updated, newly introduced, or have a modification proposal."
    ) if config.ENABLE_DYNAMIC_STATE_ADAPTATION else "Only include characters whose information is directly updated or those newly introduced in THIS chapter. Provide full description, traits (comma-separated), status for new chars."

    dynamic_instr_world = (
        f"For existing world items, if their properties need modification, include a line like `Modification Proposal: MODIFY atmosphere: 'Now heavy with magical fallout'`. "
        f"Also provide the new full value for any changed properties. For NEW world items, provide all known properties. "
        f"E.g., for locations: `Description: ...`, `Atmosphere: ...`. For factions: `Description: ...`, `Goals:` (followed by '- goal' lines). "
        f"Only include world elements that are new, significantly changed by THIS chapter's events, or have a modification proposal."
    ) if config.ENABLE_DYNAMIC_STATE_ADAPTATION else "Only include world elements that are new or significantly changed by THIS chapter's events. Provide full details for new items."

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
  {current_profiles_plain_text}
  ```
  **World Building Snapshot (Plain Text):**
  ```text
  {current_world_plain_text}
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
   Followed by indented key-value pairs for their profile data:
     `Description: [Full description if new or significantly changed]`
     `Traits: [Comma-separated list, e.g., Brave, Cautious]`
     `Status: [Current status]`
     `Relationships:` (Optional, if any)
       `- [Target Character Name]: [Relationship Type, e.g., ally, enemy, mentor]`
     `Development in Chapter {chapter_number}: [Summary of their role/changes in THIS chapter]`
     `Modification Proposal: [Optional: MODIFY key: value_change]` (if ENABLE_DYNAMIC_STATE_ADAPTATION is true)
   {dynamic_instr_char}
   Example:
   Character: Elara
   Description: Now bears a scar from the fight.
   Traits: Brave, Determined, Scarred
   Status: Wounded but resolute
   Relationships:
     - Gorok: uneasy ally
   Development in Chapter {chapter_number}: Elara defeated the guardian and claimed the Sunstone.
   Modification Proposal: MODIFY traits: ADD 'Resourceful'

**2. Under `### WORLD UPDATES ###`:**
   Start each category with: `Category: [locations | society | systems | lore | history | factions | _overview_]`
   For `_overview_`, provide: `Description: [Overall world feel change]`
   For other categories, list items: `Item: [Item Name]`
   Followed by indented key-value pairs for item details:
     `Description: [Full description if new or changed]`
     `Atmosphere: [For locations]`
     `Goals:` (For factions, list each goal on a new line starting with "- ")
       `- Goal 1`
     `Elaboration in Chapter {chapter_number}: [Context from THIS chapter]`
     `Modification Proposal: [Optional: MODIFY key: value_change]` (if ENABLE_DYNAMIC_STATE_ADAPTATION is true)
   {dynamic_instr_world}
   Example:
   Category: locations
   Item: Sunstone Chamber
   Description: A vast cavern, now shimmering with residual energy.
   Atmosphere: Mystical, charged
   Elaboration in Chapter {chapter_number}: Elara found the Sunstone here after a difficult battle.

**3. Under `### KG TRIPLES ###`:**
   List each factual triple. Choose ONE format and use it consistently:
   Format A (preferred): `Subject | Predicate | Object`
   Format B: `Subject: [Subject Name], Predicate: [predicate_name], Object: [Object Name/Value]`
   Format C: `- [Subject, Predicate, Object]`
   Use predicates from this suggested list where appropriate: {common_predicates_str}. Focus on NEW facts from THIS chapter.
   Example (using Format A):
   Elara | travelled_to | Sunstone Chamber
   Sunstone Chamber | is_a | ancient ruin
   Sunstone | has_property | emits_light

Begin your output now:
"""
    
    logger.info(f"Calling LLM ({config.KNOWLEDGE_UPDATE_MODEL}) for unified knowledge extraction (Ch {chapter_number}).")
    raw_extraction_text = await llm_interface.async_call_llm(
        model_name=config.KNOWLEDGE_UPDATE_MODEL,
        prompt=prompt,
        temperature=0.5, 
        allow_fallback=True,
        stream_to_disk=True
    )

    cleaned_extraction_text = llm_interface.clean_model_response(raw_extraction_text)
    
    # Split the cleaned text into sections
    char_updates_text = ""
    world_updates_text = ""
    kg_triples_text = ""

    # Regex to find sections, allowing for optional whitespace around headers
    char_updates_match = re.search(r"###\s*CHARACTER UPDATES\s*###\s*(.*?)(?=\s*###\s*(?:WORLD UPDATES|KG TRIPLES)\s*###|$)", cleaned_extraction_text, re.IGNORECASE | re.DOTALL)
    world_updates_match = re.search(r"###\s*WORLD UPDATES\s*###\s*(.*?)(?=\s*###\s*KG TRIPLES\s*###|$)", cleaned_extraction_text, re.IGNORECASE | re.DOTALL)
    kg_triples_match = re.search(r"###\s*KG TRIPLES\s*###\s*(.*)$", cleaned_extraction_text, re.IGNORECASE | re.DOTALL)

    if char_updates_match:
        char_updates_text = char_updates_match.group(1).strip()
    if world_updates_match:
        world_updates_text = world_updates_match.group(1).strip()
    if kg_triples_match:
        kg_triples_text = kg_triples_match.group(1).strip()

    # Parse each section
    char_updates_dict = _parse_plain_text_character_updates(char_updates_text, chapter_number)
    world_updates_dict = _parse_plain_text_world_updates(world_updates_text, chapter_number)
    kg_triples_list = _parse_plain_text_kg_triples(kg_triples_text)

    final_extraction = {
        "character_updates": char_updates_dict,
        "world_updates": world_updates_dict,
        "knowledge_triples": kg_triples_list
    }
    
    logger.info(f"Unified knowledge extraction (plain text parsing) for Ch {chapter_number} complete. "
                f"Char updates: {len(final_extraction['character_updates'])}, "
                f"World updates categories: {len(final_extraction['world_updates'])}, "
                f"KG Triples: {len(final_extraction['knowledge_triples'])}.")
    if not char_updates_dict and not world_updates_dict and not kg_triples_list and cleaned_extraction_text.strip():
        logger.warning(f"Unified knowledge extraction for Ch {chapter_number} produced text, but parsing yielded no data. Raw cleaned text: '{cleaned_extraction_text[:500]}...'")
        await agent._save_debug_output(chapter_number, "unified_extraction_plain_text_parse_empty", cleaned_extraction_text)


    return final_extraction


# --- Knowledge Graph Pre-population ---
# This function already operates on agent's Python dicts, so its core logic is largely unaffected by LLM output changes.
# It translates Python dicts directly to Cypher.
async def prepopulate_kg_from_initial_data_logic(agent):
    logger.info("Starting Knowledge Graph pre-population directly from agent's initial data dicts...")
    
    cypher_statements: List[Tuple[str, Dict[str, Any]]] = []

    plot = agent.plot_outline
    novel_id = config.MAIN_NOVEL_INFO_NODE_ID
    if plot:
        novel_props = {k: v for k, v in plot.items() if not isinstance(v, (list, dict)) and v is not None}
        novel_props['id'] = novel_id
        cypher_statements.append((f"MERGE (ni:NovelInfo {{id: $id}}) SET ni = $props", {"id": novel_id, "props": novel_props}))

        plot_points = plot.get('plot_points', [])
        if isinstance(plot_points, list):
            for i, desc in enumerate(plot_points):
                if isinstance(desc, str):
                    pp_id = f"{novel_id}_pp_{i+1}"
                    cypher_statements.append((
                        f"""
                        MATCH (ni:NovelInfo {{id: '{novel_id}'}})
                        MERGE (pp:PlotPoint {{id: $pp_id}})
                        SET pp.sequence = $seq, pp.description = $desc
                        MERGE (ni)-[:HAS_PLOT_POINT]->(pp)
                        """,
                        {"pp_id": pp_id, "seq": i + 1, "desc": desc}
                    ))
                    if i > 0:
                        prev_pp_id = f"{novel_id}_pp_{i}"
                        cypher_statements.append((
                            f"""
                            MATCH (prev_pp:PlotPoint {{id: '{prev_pp_id}'}})
                            MATCH (curr_pp:PlotPoint {{id: '{pp_id}'}})
                            MERGE (prev_pp)-[:NEXT_PLOT_POINT]->(curr_pp)
                            """, {}
                        ))
    
    for char_name, profile in agent.character_profiles.items():
        if not isinstance(profile, dict): continue
        char_props = {k: v for k, v in profile.items() if isinstance(v, (str, int, float, bool)) and v is not None}
        char_props['name'] = char_name
        cypher_statements.append(("MERGE (c:Character {name: $name}) SET c += $props", {"name": char_name, "props": char_props}))

        if isinstance(profile.get("traits"), list):
            for trait in profile["traits"]:
                if isinstance(trait, str):
                    cypher_statements.append((
                        """
                        MATCH (c:Character {name: $char_name})
                        MERGE (t:Trait {name: $trait_name})
                        MERGE (c)-[:HAS_TRAIT]->(t)
                        """, {"char_name": char_name, "trait_name": trait}
                    ))
        
        if isinstance(profile.get("relationships"), dict):
            for target_name, rel_detail in profile["relationships"].items():
                rel_type = "RELATED_TO"
                rel_props_dict = {"description": str(rel_detail), "chapter_added": config.KG_PREPOPULATION_CHAPTER_NUM, "is_provisional": False}
                if isinstance(rel_detail, dict) and "type" in rel_detail: # If rel_detail is a dict with type
                    rel_type = rel_detail.pop("type", rel_type).upper().replace(" ", "_")
                    rel_props_dict.update(rel_detail) # Add other properties from dict
                elif isinstance(rel_detail, str): # If rel_detail is just a string description
                     pass # rel_props_dict already has description

                cypher_statements.append((
                    """
                    MATCH (c1:Character {name: $char_name1})
                    MERGE (c2:Character {name: $char_name2}) ON CREATE SET c2.description = 'Auto-created via relationship from ' + $char_name1
                    MERGE (c1)-[r:DYNAMIC_REL {type:$rel_type_val}]->(c2)
                    SET r += $rel_props_val, r.chapter_added = COALESCE($rel_props_val.chapter_added, r.chapter_added, $default_chap_add), r.is_provisional = COALESCE($rel_props_val.is_provisional, r.is_provisional, false)
                    """, 
                    {"char_name1": char_name, "char_name2": target_name, 
                     "rel_type_val": rel_type, "rel_props_val": rel_props_dict,
                     "default_chap_add": config.KG_PREPOPULATION_CHAPTER_NUM }
                ))

    for category, items in agent.world_building.items():
        if not isinstance(items, dict) or category in ["is_default", "source", "user_supplied_data", "_overview_"]:
            if category == "_overview_" and isinstance(items, dict) and "description" in items:
                cypher_statements.append((
                    f"MERGE (wc:WorldContainer {{id: $id}}) SET wc.overview_description = $desc",
                    {"id": config.MAIN_WORLD_CONTAINER_NODE_ID, "desc": items["description"]}
                ))
            continue

        for item_name, details in items.items():
            if not isinstance(details, dict) or item_name.startswith(("_", "source_quality_", "category_updated_")): continue
            
            we_id = f"{category}_{item_name}".replace(" ", "_").replace("'", "").lower()
            item_props = {k: v for k, v in details.items() if isinstance(v, (str, int, float, bool)) and v is not None}
            item_props.update({'id': we_id, 'name': item_name, 'category': category, 'created_chapter': config.KG_PREPOPULATION_CHAPTER_NUM})
            
            cypher_statements.append(("MERGE (we:WorldElement {id: $id}) SET we += $props", {"id": we_id, "props": item_props}))

            for list_prop_name in ["goals", "rules", "key_elements"]:
                if isinstance(details.get(list_prop_name), list):
                    for val_item in details[list_prop_name]:
                        if isinstance(val_item, str):
                            rel_name = f"HAS_{list_prop_name.upper().rstrip('S')}"
                            if list_prop_name == "key_elements": rel_name = "HAS_KEY_ELEMENT" # Align with save_world_building
                            cypher_statements.append((
                                f"""
                                MATCH (we:WorldElement {{id: $we_id}})
                                MERGE (v:ValueNode {{value: $val_item_value, type: '{list_prop_name}'}})
                                MERGE (we)-[:{rel_name}]->(v)
                                """, {"we_id": we_id, "val_item_value": val_item}
                            ))
    if cypher_statements:
        try:
            await state_manager.execute_cypher_batch(cypher_statements)
            logger.info(f"KG pre-population complete: Executed {len(cypher_statements)} Cypher statements directly from initial data.")
        except Exception as e:
            logger.error(f"Error during direct KG pre-population batch execution: {e}", exc_info=True)
    else:
        logger.info("No Cypher statements generated for KG pre-population from initial data.")


# --- Overall Knowledge Base Update Orchestration ---

async def update_all_knowledge_bases_logic(
    agent, 
    chapter_number: int,
    final_text: str,
    from_flawed_draft: bool 
):
    if not final_text:
        logger.warning(f"Skipping all knowledge base updates for ch {chapter_number}: Final text is missing or empty.")
        return
    
    logger.info(f"Updating all knowledge bases for ch {chapter_number} (Source from flawed draft: {from_flawed_draft})...")
    
    extraction_results = await unified_knowledge_extraction(agent, final_text, chapter_number)
    
    character_updates_dict = extraction_results.get("character_updates", {})
    world_updates_dict = extraction_results.get("world_updates", {})
    kg_triples_list = extraction_results.get("knowledge_triples", [])

    if character_updates_dict and isinstance(character_updates_dict, dict):
        merge_character_profile_updates_logic(agent, character_updates_dict, chapter_number, from_flawed_draft)
    else: logger.warning(f"No valid character updates (dict) from unified extraction for ch {chapter_number}.")

    if world_updates_dict and isinstance(world_updates_dict, dict):
        merge_world_item_updates_logic(agent, world_updates_dict, chapter_number, from_flawed_draft)
    else: logger.warning(f"No valid world-building updates (dict) from unified extraction for ch {chapter_number}.")

    if kg_triples_list and isinstance(kg_triples_list, list):
        added_count, skipped_count = 0, 0
        kg_add_tasks = []
        for triple_any in kg_triples_list:
            if isinstance(triple_any, list) and len(triple_any) == 3:
                subj, pred, obj_val = (str(triple_any[0]).strip() if triple_any[0] is not None else "",
                                   str(triple_any[1]).strip() if triple_any[1] is not None else "",
                                   str(triple_any[2]).strip() if triple_any[2] is not None else "")
                if subj and pred and obj_val:
                    obj_truncated = obj_val[:500] + "..." if len(obj_val) > 503 else obj_val 
                    kg_add_tasks.append(
                        state_manager.async_add_kg_triple(subj, pred, obj_truncated, chapter_number, is_provisional=from_flawed_draft)
                    )
                    added_count += 1
                else:
                    logger.warning(f"Skipping invalid KG triple (empty component) from ch {chapter_number}: {triple_any}")
                    skipped_count += 1
            else:
                logger.warning(f"Skipping invalid KG triple format (not list of 3) from ch {chapter_number}: {triple_any}")
                skipped_count += 1
        
        if kg_add_tasks: await asyncio.gather(*kg_add_tasks)
        logger.info(f"KG update from unified_extraction for ch {chapter_number}: Added {added_count} triples, skipped {skipped_count}.")
    else: logger.info(f"No KG triples from unified extraction for ch {chapter_number}.")

    logger.info(f"Unified knowledge extraction processing complete for ch {chapter_number}. In-memory dicts updated, KG triples sent to Neo4j.")