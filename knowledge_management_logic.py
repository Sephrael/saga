# knowledge_management_logic.py
"""
Handles updates to knowledge bases (JSON profiles, Knowledge Graph)
and summarization for the SAGA system.
"""
import logging
import json
import re
import asyncio
from typing import Dict, List, Optional, Any, Tuple

from async_lru import alru_cache # For caching LLM calls

import config
import llm_interface
from type import KnowledgeGraph, Entity, Relationship, Event, Location, Character, Faction

# Import thematic consistency check function
from thematic_consistency_checker import check_thematic_consistency_logic
# Import prompt data getters
from prompt_data_getters import (
    get_filtered_world_data_for_prompt,
    heuristic_entity_spotter_for_kg,
    get_filtered_character_profiles_for_prompt # Added for use in _build_character_update_prompt
)
from state_manager import state_manager

logger = logging.getLogger(__name__)

# --- Summarization ---

@alru_cache(maxsize=config.SUMMARY_CACHE_SIZE)
async def llm_summarize_chapter_snippet_logic(chapter_text_snippet_key: str, chapter_number: int) -> str:
    """Cached LLM call for summarizing a chapter snippet. Key is snippet to cache effectively."""
    prompt = f"""/no_think
You are a concise summarizer. Summarize the key events, character developments, and plot advancements from the following Chapter {chapter_number} text snippet.
The summary should be 1-3 sentences long and capture the most crucial information.
Focus on what changed or was revealed.

Chapter Text Snippet:
--- BEGIN TEXT ---
{chapter_text_snippet_key}
--- END TEXT ---

Output ONLY the summary text. No extra commentary or "Summary:" prefix.
"""
    summary_raw = await llm_interface.async_call_llm(
        model_name=config.SMALL_MODEL,
        prompt=prompt,
        temperature=0.6,
        max_tokens=config.MAX_SUMMARY_TOKENS
    )
    return llm_interface.clean_model_response(summary_raw).strip()

async def summarize_chapter_text_logic(chapter_text: Optional[str], chapter_number: int) -> Optional[str]:
    """ (Unused 'agent' parameter removed - no longer used in this function) """
    if not chapter_text or len(chapter_text) < 50:
        logger.warning(f"Chapter {chapter_number} text too short for summarization ({len(chapter_text or '')} chars).")
        return None
            
    snippet_for_summary = chapter_text[:config.KNOWLEDGE_UPDATE_SNIPPET_SIZE].strip()
    
    cleaned_summary = await llm_summarize_chapter_snippet_logic(snippet_for_summary, chapter_number)

    if cleaned_summary: 
        logger.info(f"Generated summary for ch {chapter_number}: '{cleaned_summary[:100].strip()}...'")
        return cleaned_summary
    
    logger.warning(f"Failed to generate a valid summary for ch {chapter_number} via LLM.")
    return None

# --- JSON State Modification Proposal Logic ---

def _apply_trait_modification(current_traits_list: List[str], modification_details_str: str) -> List[str]:
    """Applies ADD/REMOVE operations to a list of traits."""
    traits_set = set(current_traits_list)
    # Regex to find ADD "trait" or REMOVE "trait", case insensitive for ADD/REMOVE, sensitive for trait value
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
    item_type_for_log: str # e.g., "character profile", "world item"
):
    """Applies a modification proposal string to a dictionary (profile or world item)."""
    if not isinstance(proposal_str, str) or not proposal_str.strip():
        logger.debug(f"Empty or invalid modification proposal for '{item_name_for_log}'. Proposal: '{proposal_str}'")
        return

    logger.debug(f"Applying modification proposal for '{item_name_for_log}' ({item_type_for_log}): '{proposal_str}'")

    # Format: MODIFY key_name: new_value or MODIFY traits: ADD "X", REMOVE "Y"
    match = re.match(r"MODIFY\s+([\w_]+)\s*:(.*)", proposal_str, re.IGNORECASE)
    if not match:
        logger.warning(f"Invalid modification proposal format for '{item_name_for_log}'. Proposal: '{proposal_str}'. Expected 'MODIFY key: value'.")
        return

    key_name_from_proposal_upper = match.group(1).strip().upper()
    value_modification_str = match.group(2).strip() # This is the part after "key:"

    # Find the actual key in target_dict (case-insensitive match)
    # If not found, use the lowercased version of the key from the proposal
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
            # For other keys, assign the new value directly
            new_value_str = value_modification_str.strip("'\" ") # Remove potential quotes around the value
            if new_value_str: # Ensure the value is not empty after stripping
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
    """Initializes a new character profile dictionary."""
    new_profile: Dict[str, Any] = {
        "description": char_update_data.get("description", f"A character newly introduced in Chapter {chapter_number}."),
        "traits": sorted(list(set(t for t in char_update_data.get("traits", []) if isinstance(t, str) and t.strip()))),
        "relationships": char_update_data.get("relationships", {}),
        "status": char_update_data.get("status", "Newly introduced")
    }
    if dev_key in char_update_data:
        new_profile[dev_key] = char_update_data[dev_key]
    if provisional_marker_key in char_update_data: # Should be set by caller if needed
        new_profile[provisional_marker_key] = char_update_data[provisional_marker_key]
    
    logger.info(f"Prepared new character profile for '{char_name}'.")
    return new_profile

def _update_existing_character_profile_fields(
    existing_profile: Dict[str, Any],
    char_update_data: Dict[str, Any],
    dev_key: str,
    provisional_marker_key: str
):
    """Updates fields of an existing character profile based on LLM data, excluding modification proposals."""
    if provisional_marker_key in char_update_data:
        existing_profile[provisional_marker_key] = char_update_data[provisional_marker_key]

    for key, value in char_update_data.items():
        if key in ["modification_proposal", provisional_marker_key, dev_key]: # Handled separately or already applied
            continue

        if key == "traits" and isinstance(value, list):
            if "traits" not in existing_profile or not isinstance(existing_profile["traits"], list):
                existing_profile["traits"] = []
            valid_new_traits = {t for t in value if isinstance(t, str) and t.strip()}
            existing_profile["traits"] = sorted(list(set(existing_profile["traits"]).union(valid_new_traits)))
        elif key == "relationships" and isinstance(value, dict):
            if not isinstance(existing_profile.get("relationships"), dict):
                existing_profile["relationships"] = {}
            existing_profile["relationships"].update(value)
        elif key == "description" and isinstance(value, str) and value.strip():
            # Only update if not already handled by a modification_proposal
            existing_profile["description"] = value
        elif key == "status" and isinstance(value, str) and value.strip():
            existing_profile["status"] = value
        elif key not in existing_profile and value is not None: # Add new fields
            existing_profile[key] = value
        # If key exists and is not one of the above, it's generally not overwritten unless by modification_proposal
    
    # Ensure development key is present if other updates happened
    if dev_key in char_update_data and isinstance(char_update_data[dev_key], str) and char_update_data[dev_key].strip():
        existing_profile[dev_key] = char_update_data[dev_key]


def merge_character_profile_updates_logic(
    agent, # NovelWriterAgent instance
    updates_from_llm: Dict[str, Any],
    chapter_number: int,
    from_flawed_draft: bool
):
    """Merges LLM-proposed character updates into the agent's character_profiles."""
    if not updates_from_llm:
        logger.info(f"No character profile updates from LLM to merge for ch {chapter_number}.")
        return

    logger.info(f"Merging character profile JSON updates for ch {chapter_number}. Characters in update: {list(updates_from_llm.keys())}")
    updated_chars_count, new_chars_count = 0, 0
    
    # This key is used to mark the source quality IF the update comes from a flawed draft.
    # It's also added to the `char_update_data` before processing if `from_flawed_draft` is true.
    provisional_marker_key = f"source_quality_chapter_{chapter_number}"

    for char_name, char_update_data_original in updates_from_llm.items():
        if not isinstance(char_update_data_original, dict):
            logger.warning(f"Skipping invalid character update data for '{char_name}' (not a dict). Data: {char_update_data_original}")
            continue
        
        char_update_data = char_update_data_original.copy() # Work with a copy

        if from_flawed_draft: # Mark data originating from a flawed draft
            char_update_data[provisional_marker_key] = "provisional_from_unrevised_draft"

        dev_key = f"development_in_chapter_{chapter_number}"
        # Ensure dev_key is present if there are any updates beyond just a provisional marker
        is_substantive_update = len(char_update_data) > (1 if provisional_marker_key in char_update_data else 0)
        if dev_key not in char_update_data and is_substantive_update:
            char_update_data[dev_key] = "Character appeared or was mentioned in this chapter."

        modification_proposal = char_update_data.get("modification_proposal")

        if char_name not in agent.character_profiles:
            new_chars_count += 1
            new_profile = _initialize_new_character_profile(char_name, char_update_data, chapter_number, provisional_marker_key, dev_key)
            agent.character_profiles[char_name] = new_profile
            if config.ENABLE_DYNAMIC_STATE_ADAPTATION and modification_proposal:
                apply_state_modification_proposal_logic(
                    agent.character_profiles[char_name], modification_proposal, char_name, "new character profile"
                )
        else:
            updated_chars_count += 1
            existing_profile = agent.character_profiles[char_name]
            
            if config.ENABLE_DYNAMIC_STATE_ADAPTATION and modification_proposal:
                apply_state_modification_proposal_logic(
                    existing_profile, modification_proposal, char_name, "existing character profile"
                )
            
            # Apply other updates (description, status, traits, relationships, dev_key, provisional_marker)
            _update_existing_character_profile_fields(existing_profile, char_update_data, dev_key, provisional_marker_key)

    if updated_chars_count > 0 or new_chars_count > 0:
        logger.info(f"Character profile JSON merge complete for ch {chapter_number}. Updated: {updated_chars_count}, New: {new_chars_count}.")
    else:
        logger.info(f"No character profiles were effectively updated or added for ch {chapter_number} after LLM analysis.")


# --- World Item Merging Logic ---

def robust_merge_world_item_data_logic(
    target_dict: Dict[str, Any], # The existing data for the world item (or a new dict if item is new)
    update_dict: Dict[str, Any], # The new data from LLM for this item
    item_name_for_log: str,      # For logging purposes, e.g., "locations.CapitalCity"
    chapter_num: int,
    from_flawed_draft_source: bool # If the update_dict originates from a flawed draft
) -> Dict[str, Any]:
    """
    Recursively merges update_dict into target_dict for world items.
    If target_dict is not a dict, it's initialized as one.
    Returns the merged dictionary (which might be target_dict modified, or a new dict).
    """
    if not isinstance(target_dict, dict):
        logger.warning(f"World item '{item_name_for_log}' target_dict was not a dict. Initializing as a new dict. Old value: '{str(target_dict)[:100]}'")
        current_item_data = {} # Create a new dict if target_dict was not a dict
    else:
        current_item_data = target_dict # Modify in place
    
    item_was_modified_this_call = False
    provisional_marker_key = f"source_quality_chapter_{chapter_num}"

    # Apply provisional marker if the update source is flawed
    if from_flawed_draft_source: # This applies to the entire update_dict
        current_item_data[provisional_marker_key] = "provisional_from_unrevised_draft"
        item_was_modified_this_call = True

    # Handle modification proposal first
    if config.ENABLE_DYNAMIC_STATE_ADAPTATION and "modification_proposal" in update_dict:
        proposal = update_dict.pop("modification_proposal") # Remove after processing
        if isinstance(proposal, str) and proposal.strip():
            apply_state_modification_proposal_logic(current_item_data, proposal, item_name_for_log, "world item")
            item_was_modified_this_call = True

    for key, value_from_update in update_dict.items():
        # Skip already processed keys or internal tracking keys that shouldn't be merged as plain data
        if key in [provisional_marker_key, "modification_proposal"] or \
           key.startswith(("updated_in_chapter_", "added_in_chapter_", "source_quality_chapter_")): 
            if key.startswith("elaboration_in_chapter_") and isinstance(value_from_update, str) and value_from_update.strip():
                 current_item_data[key] = value_from_update # Keep elaborations
                 item_was_modified_this_call = True
            continue

        current_value_in_target = current_item_data.get(key)

        if isinstance(value_from_update, dict):
            # Pass current_value_in_target which might be None or not a dict, robust_merge will handle it
            merged_sub_dict = robust_merge_world_item_data_logic(
                current_value_in_target if isinstance(current_value_in_target, dict) else {}, # Ensure a dict is passed if creating new
                value_from_update, 
                f"{item_name_for_log}.{key}", 
                chapter_num, 
                from_flawed_draft_source=False 
            )
            if merged_sub_dict != current_value_in_target: 
                item_was_modified_this_call = True
            current_item_data[key] = merged_sub_dict
        elif isinstance(value_from_update, list):
            if not isinstance(current_value_in_target, list):
                current_item_data[key] = [] # Initialize if not a list
                item_was_modified_this_call = True
            
            initial_list_len = len(current_item_data[key])
            for item_in_list_update in value_from_update:
                if item_in_list_update not in current_item_data[key]: # Add only new, unique items
                    current_item_data[key].append(item_in_list_update)
            if len(current_item_data[key]) > initial_list_len:
                item_was_modified_this_call = True
        elif value_from_update != current_value_in_target: # For scalar values or if direct overwrite is intended
            current_item_data[key] = value_from_update
            item_was_modified_this_call = True
    
    if item_was_modified_this_call and not current_item_data.get(f"added_in_chapter_{chapter_num}"):
        current_item_data[f"updated_in_chapter_{chapter_num}"] = True
            
    return current_item_data


def merge_world_item_updates_logic(
    agent, # NovelWriterAgent instance
    updates_from_llm: Dict[str, Any], # Expected: {"category": {"item_name": {...updates...}}}
    chapter_number: int,
    from_flawed_draft: bool
):
    """Merges LLM-proposed world-building updates into the agent's world_building state."""
    if not updates_from_llm:
        logger.info(f"No world-building updates from LLM to merge for ch {chapter_number}.")
        return

    logger.info(f"Merging world-building JSON updates for ch {chapter_number}. Categories in update: {list(updates_from_llm.keys())}")
    items_affected_count = 0
    
    for category_key, category_updates_dict in updates_from_llm.items():
        if not isinstance(category_updates_dict, dict) or not category_updates_dict:
            logger.debug(f"Skipping empty or invalid update for world category '{category_key}' in ch {chapter_number}.")
            continue
        
        if category_key not in agent.world_building:
            agent.world_building[category_key] = {}
        elif not isinstance(agent.world_building[category_key], dict): 
            logger.warning(f"Overwriting non-dictionary world category '{category_key}' with new dictionary structure for ch {chapter_number}.")
            agent.world_building[category_key] = {}
        
        target_category_dict = agent.world_building[category_key]
        
        for item_name, item_update_details in category_updates_dict.items():
            if not isinstance(item_update_details, dict):
                logger.warning(f"Skipping invalid item_details for '{item_name}' in category '{category_key}' (not a dict) for ch {chapter_number}. Data: {item_update_details}")
                continue
            
            item_log_name = f"{category_key}.{item_name}"
            existing_item_data = target_category_dict.get(item_name)
            
            # robust_merge_world_item_data_logic handles if existing_item_data is None or not a dict
            # by returning a new merged dictionary. This new/modified dict must be assigned back.
            merged_item_data = robust_merge_world_item_data_logic(
                existing_item_data if existing_item_data is not None else {}, # Pass empty dict if new
                item_update_details, 
                item_log_name, 
                chapter_number, 
                from_flawed_draft
            )
            target_category_dict[item_name] = merged_item_data # Assign back

            if existing_item_data is None: # New item was added
                merged_item_data[f"added_in_chapter_{chapter_number}"] = True 
                items_affected_count += 1
            elif merged_item_data.get(f"updated_in_chapter_{chapter_number}") or \
                 (from_flawed_draft and merged_item_data.get(f"source_quality_chapter_{chapter_number}")):
                items_affected_count += 1
        
        if any(isinstance(v,dict) and (v.get(f"updated_in_chapter_{chapter_number}") or v.get(f"added_in_chapter_{chapter_number}")) 
               for v in target_category_dict.values()):
             target_category_dict[f"category_updated_in_chapter_{chapter_number}"] = True

    if items_affected_count > 0:
        logger.info(f"World-building JSON merge complete for ch {chapter_number}. Approximately {items_affected_count} items affected/added.")
    else:
        logger.info(f"No world-building JSON items were effectively updated or added for ch {chapter_number} after LLM analysis.")


# --- JSON Profile Update Orchestration (from Chapter Text) ---

async def _build_character_update_prompt(
    agent, text_snippet: str, chapter_number: int
) -> str:
    """Builds the prompt for LLM character profile updates."""
    protagonist_name = agent.plot_outline.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME)
    dynamic_instr_char = ""
    if config.ENABLE_DYNAMIC_STATE_ADAPTATION:
        dynamic_instr_char = (
            f"For existing characters, if their traits, status, or core description needs modification based on THIS chapter's events, "
            f"include a `\"modification_proposal\"` field. Example: `\"modification_proposal\": \"MODIFY traits: ADD 'Determined', REMOVE 'Hesitant'\"`. "
            f"Only include characters that are updated, newly introduced, or have a modification proposal."
        )
    else:
        dynamic_instr_char = "Only include characters whose information is directly updated or those newly introduced in THIS chapter."

    current_profiles_for_prompt = await get_filtered_character_profiles_for_prompt(agent, chapter_number - 1)

    return f"""/no_think
You are a meticulous literary analyst. Your task is to analyze the provided Chapter {chapter_number} Text Snippet (protagonist: {protagonist_name}) and identify updates for character profiles.
The output MUST be a single, valid JSON object representing character updates.

**Chapter {chapter_number} Text Snippet (focus on information revealed or changed IN THIS SNIPPET):**
--- BEGIN TEXT ---
{text_snippet}... (snippet may be truncated)
--- END TEXT ---
        
**Current Character Profiles (for reference - note 'prompt_notes' for provisional status):**
```json
{json.dumps(current_profiles_for_prompt, indent=2, ensure_ascii=False, default=str, sort_keys=True)}
```
**Character Update Instructions:**
1. Identify characters whose status, traits, relationships, or descriptions are explicitly updated or who are newly introduced in THIS chapter snippet.
2. For each such character, create an entry in the output JSON object (keyed by character name).
3. Each character entry should include relevant updated fields (e.g., "traits", "status", "description", "relationships").
4. Crucially, add a `development_in_chapter_{chapter_number}` key to each character entry, summarizing their role, actions, or significant changes in THIS chapter.
5. {dynamic_instr_char}
6. If no characters are updated or introduced, the output should be an empty JSON object `{{}}`.

**CRITICAL: Output ONLY the character updates JSON object as specified.**
Example Output Structure:
```json
{{
  "CharacterName": {{ 
    "description": "Updated description.", 
    "traits": ["NewTrait"], 
    "status": "Updated Status",
    "modification_proposal": "MODIFY traits: ADD 'Brave'", 
    "development_in_chapter_{chapter_number}": "They confronted the antagonist and revealed a new skill."
  }},
  "AnotherChar": {{
    "status": "Injured",
    "development_in_chapter_{chapter_number}": "Was wounded during the escape."
  }}
}}
```
"""

async def update_character_profiles_from_chapter_logic(
    agent, chapter_text: Optional[str], chapter_number: int, from_flawed_draft: bool
):
    """Updates character JSON profiles based on events in the chapter."""
    if not chapter_text or len(chapter_text) < 100: # Heuristic minimum length
        logger.info(f"Skipping character JSON knowledge update for ch {chapter_number}: Text too short or None.")
        return

    logger.info(f"Attempting character JSON profile update for ch {chapter_number} (Source from flawed draft: {from_flawed_draft}).")
    text_snippet = chapter_text[:config.KNOWLEDGE_UPDATE_SNIPPET_SIZE]
    
    prompt = await _build_character_update_prompt(agent, text_snippet, chapter_number)
    
    raw_analysis = await llm_interface.async_call_llm(
        model_name=config.KNOWLEDGE_UPDATE_MODEL,
        prompt=prompt,
        temperature=0.6
    )
    character_updates = await llm_interface.async_parse_llm_json_response(
        raw_analysis, f"character JSON profile update for ch {chapter_number}", expect_type=dict
    )

    if character_updates and isinstance(character_updates, dict):
        merge_character_profile_updates_logic(agent, character_updates, chapter_number, from_flawed_draft)
    else:
        logger.warning(f"LLM parsing for character JSON updates failed or returned no/invalid data for ch {chapter_number}. Raw: '{raw_analysis[:200] if raw_analysis else 'EMPTY'}'")


async def _build_world_update_prompt(
    agent, text_snippet: str, chapter_number: int
) -> str:
    """Builds the prompt for LLM world-building updates."""
    protagonist_name = agent.plot_outline.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME)
    dynamic_instr_world = ""
    if config.ENABLE_DYNAMIC_STATE_ADAPTATION:
        dynamic_instr_world = (
            f"For existing world items, if their properties need modification, include a `\"modification_proposal\"`. "
            f"Example: `\"modification_proposal\": \"MODIFY atmosphere: 'Now heavy with magical fallout'\"`. "
            f"Only include world elements (locations, society items, systems, lore, history) that are new, significantly changed by THIS chapter's events, or have a modification proposal."
        )
    else:
        dynamic_instr_world = "Only include world elements that are new or significantly changed by THIS chapter's events."

    current_world_for_prompt = await get_filtered_world_data_for_prompt(agent, chapter_number - 1)

    return f"""/no_think
You are a meticulous literary analyst. Your task is to analyze the provided Chapter {chapter_number} Text Snippet (protagonist: {protagonist_name}) and identify updates for world-building details.
The output MUST be a single, valid JSON object representing world-building updates.

**Chapter {chapter_number} Text Snippet (focus on information revealed or changed IN THIS SNIPPET):**
--- BEGIN TEXT ---
{text_snippet}... (snippet may be truncated)
--- END TEXT ---

**Current World Building Notes (for reference - note 'prompt_notes' for provisional status):**
```json
{json.dumps(current_world_for_prompt, indent=2, ensure_ascii=False, default=str, sort_keys=True)}
```
**World Building Update Instructions:**
1. Identify new or significantly changed locations, societal elements (factions, cultures), systems (magic, tech), lore, or historical details revealed in THIS chapter snippet.
2. For each, create an entry under the appropriate category (e.g., "locations", "society") within the output JSON object.
3. Each world element entry (e.g., a specific city under "locations") should contain its updated details (e.g., "description", "atmosphere", "rules", "goals").
4. Add an `elaboration_in_chapter_{chapter_number}` key to each world element entry, providing context or specifics from THIS chapter.
5. {dynamic_instr_world}
6. If no world elements are updated or introduced, the relevant category (e.g., "locations") should be an empty JSON object `{{}}`, or the entire output can be `{{}}`.

**CRITICAL: Output ONLY the world-building updates JSON object as specified.**
Example Output Structure:
```json
{{
  "locations": {{
    "NewDiscoveredCave": {{ 
      "description": "A dark, mysterious cave pulsating with strange energy.",
      "atmosphere": "Eerie and cold",
      "elaboration_in_chapter_{chapter_number}": "Discovered by the protagonist after deciphering ancient map."
    }}
  }},
  "systems": {{
    "AncientMagicSystem": {{
       "rules": "Previously unknown rule about requiring silver for casting.",
       "modification_proposal": "MODIFY description: 'Magic is now unstable during eclipses.'", 
       "elaboration_in_chapter_{chapter_number}": "A character failed to cast a spell during an eclipse, revealing this new property."
    }}
  }}
}}
```
"""

async def update_world_building_from_chapter_logic(
    agent, chapter_text: Optional[str], chapter_number: int, from_flawed_draft: bool
):
    """Updates world-building JSON files based on events in the chapter."""
    if not chapter_text or len(chapter_text) < 100: # Heuristic minimum length
        logger.info(f"Skipping world-building JSON knowledge update for ch {chapter_number}: Text too short or None.")
        return

    logger.info(f"Attempting world-building JSON update for ch {chapter_number} (Source from flawed draft: {from_flawed_draft}).")
    text_snippet = chapter_text[:config.KNOWLEDGE_UPDATE_SNIPPET_SIZE]

    prompt = await _build_world_update_prompt(agent, text_snippet, chapter_number)

    raw_analysis = await llm_interface.async_call_llm(
        model_name=config.KNOWLEDGE_UPDATE_MODEL,
        prompt=prompt,
        temperature=0.6
    )
    world_updates = await llm_interface.async_parse_llm_json_response(
        raw_analysis, f"world-building JSON update for ch {chapter_number}", expect_type=dict
    )

    if world_updates and isinstance(world_updates, dict):
        merge_world_item_updates_logic(agent, world_updates, chapter_number, from_flawed_draft)
    else:
        logger.warning(f"LLM parsing for world-building JSON updates failed or returned no/invalid data for ch {chapter_number}. Raw: '{raw_analysis[:200] if raw_analysis else 'EMPTY'}'")


# --- Knowledge Graph Management ---

@alru_cache(maxsize=config.KG_TRIPLE_EXTRACTION_CACHE_SIZE)
async def llm_extract_kg_triples_logic(
    protagonist_name_for_prompt: str, # Extracted protagonist name
    text_snippet_for_kg_key: str, # The text snippet itself, used as cache key
    chapter_number: int,
    candidate_entities_json_key: str # JSON string of candidate entities, used as cache key
) -> str: # Returns raw LLM response string
    """Cached LLM call for KG triple extraction."""
    common_predicates = [
        "is_a", "located_in", "has_trait", "status_is", "feels", "knows", "believes", "wants", 
        "interacted_with", "travelled_to", "discovered", "acquired", "lost", "used_item", 
        "attacked", "helped", "damaged", "repaired", "contains", "part_of", "caused_by", 
        "leads_to", "observed", "heard", "said", "thought_about", "decided_to", "has_goal", 
        "has_feature", "related_to", "member_of", "leader_of", "enemy_of", "ally_of", 
        "works_for", "has_ability", "possesses", "created_by"
    ] 
    
    candidate_entities_prompt_section = ""
    if candidate_entities_json_key and candidate_entities_json_key != "[]": # Check if not empty list string
        candidate_entities_prompt_section = (
            f"**Heuristically Identified Candidate Entities (Prioritize these for Subject/Object if relevant and present in the text snippet):**\n"
            f"```json\n{candidate_entities_json_key}\n```\n"
        )

    prompt = f"""/no_think
You are a Knowledge Graph Engineer. Your task is to extract factual (Subject, Predicate, Object) triples from the provided Text Snippet from Chapter {chapter_number} of a novel (protagonist: '{protagonist_name_for_prompt}').

**Chapter {chapter_number} Text Snippet:**
--- TEXT ---
{text_snippet_for_kg_key}
--- END TEXT ---

{candidate_entities_prompt_section}
**Instructions for Triple Extraction:**
1. Identify key entities (characters, locations, significant items, concepts) within the Text Snippet. Normalize names (e.g., "John Doe" not "John").
2. If Candidate Entities are provided, strongly consider them for Subjects and Objects if they are clearly mentioned and active in the snippet.
3. Use predicates from the Suggested Predicates list or create concise, descriptive alternatives if necessary. Predicates should be lowercase with underscores (verb_phrase_style).
4. Each triple must be a list of three non-empty strings: `["Subject", "predicate_name", "Object"]`.
5. Focus **ONLY** on information explicitly stated or very strongly implied within THIS Text Snippet. Do not infer beyond the text.
6. Prioritize facts about state changes, new relationships, key actions, discoveries, and significant attributes. Avoid trivial details.
7. **CRITICAL OUTPUT FORMAT:** Output ONLY a valid JSON list of lists (triples). If no meaningful facts can be extracted, output an empty JSON list `[]`.
8. **NO other text, markdown, explanations, or commentary.** The response must start with `[` and end with `]`.

**Suggested Predicates (use these or similar):**
{', '.join(common_predicates)}

**Example Output:**
`[["{protagonist_name_for_prompt}", "travelled_to", "Eclipse Spire"], ["Eclipse Spire", "is_a", "ancient ruin"], ["{protagonist_name_for_prompt}", "feels", "uneasy"]]`

JSON Output Only:
[
""" 
    return await llm_interface.async_call_llm(
        model_name=config.KNOWLEDGE_UPDATE_MODEL,
        prompt=prompt, 
        temperature=0.6, 
        max_tokens=config.MAX_KG_TRIPLE_TOKENS
    )

async def extract_and_store_kg_triples_logic(
    agent, # NovelWriterAgent instance, for _save_debug_output and plot_outline access
    chapter_text: Optional[str],
    chapter_number: int,
    from_flawed_draft: bool
):
    """Extracts KG triples from chapter text and adds them to the database via state_manager."""
    if not chapter_text:
        logger.warning(f"Skipping KG extraction for ch {chapter_number}: Chapter text is None or empty.")
        return
            
    logger.info(f"Extracting KG triples for ch {chapter_number} (Source from flawed draft: {from_flawed_draft})...")
    
    # Use a larger snippet for KG extraction as it might rely on broader context within the chapter
    text_snippet_for_kg = chapter_text[:config.KNOWLEDGE_UPDATE_SNIPPET_SIZE * 2].strip() 
    if len(text_snippet_for_kg) < len(chapter_text): # Log if truncated
        logger.debug(f"KG extraction for ch {chapter_number} will use truncated text ({len(text_snippet_for_kg)} chars out of {len(chapter_text)}).")

    candidate_entities = await heuristic_entity_spotter_for_kg(agent, text_snippet_for_kg)
    logger.debug(f"Candidate entities identified for KG extraction in Ch {chapter_number}: {candidate_entities[:10]}")
    candidate_entities_json_for_prompt = json.dumps(candidate_entities) # For cache key and prompt

    # Get protagonist_name for the cached LLM call
    protagonist_name = agent.plot_outline.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME)

    raw_triples_json_str = await llm_extract_kg_triples_logic(
        protagonist_name, text_snippet_for_kg, chapter_number, candidate_entities_json_for_prompt
    )
            
    parsed_triples = await llm_interface.async_parse_llm_json_response(
        raw_triples_json_str, f"KG triple extraction for chapter {chapter_number}", expect_type=list
    )
    
    if parsed_triples is None: # Parsing failed or LLM returned nothing usable
         logger.error(f"Failed to extract or parse any KG triples for ch {chapter_number}. Raw LLM output: {raw_triples_json_str[:200] if raw_triples_json_str else 'EMPTY'}")
         await agent._save_debug_output(chapter_number, "kg_extraction_final_fail_raw_llm", raw_triples_json_str or "EMPTY_RAW_TRIPLES_JSON")
         return
    
    if not parsed_triples: # LLM returned an empty list []
        logger.info(f"No KG triples were extracted by the LLM for ch {chapter_number}.")
        return
             
    added_count, skipped_count = 0, 0
    kg_add_tasks = []
    for triple_any in parsed_triples:
        if isinstance(triple_any, list) and len(triple_any) == 3:
            # Ensure all components are strings and stripped
            subj = str(triple_any[0]).strip() if triple_any[0] is not None else ""
            pred = str(triple_any[1]).strip() if triple_any[1] is not None else ""
            obj  = str(triple_any[2]).strip() if triple_any[2] is not None else ""
            
            if subj and pred and obj: # All components must be non-empty after stripping
                # state_manager handles the actual DB interaction
                kg_add_tasks.append(
                    state_manager.async_add_kg_triple(subj, pred, obj, chapter_number, is_provisional=from_flawed_draft)
                )
                added_count += 1
            else:
                logger.warning(f"Skipping invalid KG triple (empty component after strip) in ch {chapter_number}: Original: {triple_any}, Stripped: ['{subj}','{pred}','{obj}']")
                skipped_count += 1
        else:
            logger.warning(f"Skipping invalid KG triple format (not a list of 3) in ch {chapter_number}: {triple_any}")
            skipped_count += 1
    
    if kg_add_tasks:
        await asyncio.gather(*kg_add_tasks) # Execute all DB additions concurrently
    
    logger.info(f"KG update for ch {chapter_number}: Attempted to add {added_count} triples, skipped {skipped_count}. (Source Provisional: {from_flawed_draft})")


async def prepopulate_kg_from_initial_data_logic(agent): # NovelWriterAgent instance
    """Pre-populates the Knowledge Graph from the initial plot outline and world-building data."""
    logger.info("Starting Knowledge Graph pre-population from plot and world data...")

    # Prune data to keep the prompt manageable and focused
    pruned_plot = {
        "title": agent.plot_outline.get("title"), 
        "protagonist_name": agent.plot_outline.get("protagonist_name"),
        "genre": agent.plot_outline.get("genre"), 
        "theme": agent.plot_outline.get("theme"),
        "setting_description": agent.plot_outline.get("setting"), # Use 'setting' key from plot_outline
        "conflict_summary": agent.plot_outline.get("conflict"),
        "character_arc": agent.plot_outline.get("character_arc"), 
        "key_plot_points_summary": agent.plot_outline.get("plot_points", [])[:2] # First 2 plot points
    }
    pruned_world = {}
    for category, items in agent.world_building.items():
        if category == "is_default" or not isinstance(items, dict): continue # Skip internal flags or non-dict categories
        pruned_world[category] = {}
        for item_name, item_details in list(items.items())[:3]: # First 3 items per category
            if isinstance(item_details, dict):
                desc = item_details.get("description", item_details.get("text", "")) # Prefer 'description'
                if isinstance(desc, str) and desc.strip():
                     pruned_world[category][item_name] = {"description_snippet": desc[:200].strip() + "..."}
            elif isinstance(item_details, str) and item_details.strip(): # Handle cases where item_details is just a string
                pruned_world[category][item_name] = {"description_snippet": item_details[:200].strip() + "..."}

    
    combined_pruned_data = {"plot_summary": pruned_plot, "world_highlights": pruned_world}
    try:
        combined_data_json = json.dumps(combined_pruned_data, indent=2, ensure_ascii=False, default=str)
    except TypeError as e:
        logger.error(f"Error serializing pruned data for KG pre-population prompt: {e}. Data: {combined_pruned_data}")
        return
        
    protagonist_name = agent.plot_outline.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME)
    novel_title = agent.plot_outline.get("title", config.DEFAULT_PLOT_OUTLINE_TITLE)
    common_predicates_prepop = [
        "is_a", "has_title", "has_protagonist", "has_genre", "has_theme", "has_setting_description", 
        "has_conflict_summary", "has_character_arc", "has_description", "has_trait", "initial_status_is",
        "related_to", "located_in", "has_goal", "part_of", "member_of", "governed_by", 
        "known_for", "primary_setting_is", "key_element_is"
    ]

    prompt = f"""/no_think
You are a Knowledge Graph Engineer. Your task is to extract foundational (Subject, Predicate, Object) triples from the provided summarized Plot Outline and World Building Highlights for the novel titled '{novel_title}' (protagonist: '{protagonist_name}').
These triples will form the initial, canonical knowledge base before chapter generation begins.

**Input JSON Data (Summarized Plot & World Highlights):**
```json
{combined_data_json}
```
**Instructions for Triple Extraction:**
1. Analyze the input JSON. Keys within "plot_summary" and "world_highlights" often map to Subjects or Predicates. Values often map to Objects or provide descriptive text from which Objects can be extracted.
2. Extract core entities (the novel itself, protagonist, key locations, factions, concepts), their types (e.g., ["{protagonist_name}", "is_a", "protagonist"]), attributes, and key relationships.
3. Use predicates from the Suggested Predicates list or create concise, descriptive alternatives if necessary. Predicates should be lowercase with underscores.
4. For the novel itself, use "{novel_title}" (or its variable if name changes) as the Subject for facts like genre, theme, protagonist.
5. For the protagonist '{protagonist_name}', extract their initial description, core traits, and initial status (if implied).
6. For key locations, factions, etc., from "world_highlights", extract their names and core descriptions/properties.
7. All three components of a triple `["Subject", "predicate_name", "Object"]` MUST be non-empty strings.
8. **CRITICAL OUTPUT FORMAT:** Output ONLY a valid JSON list of lists (triples). If no meaningful facts, output `[]`.
9. **NO other text, markdown, explanations, or commentary.** The response must start with `[` and end with `]`.

**Suggested Predicates for Pre-population (use these or similar):**
{', '.join(common_predicates_prepop)}

**Example Output:**
`[["{novel_title}", "has_protagonist", "{protagonist_name}"], ["{protagonist_name}", "is_a", "protagonist"], ["{protagonist_name}", "initial_status_is", "seeking answers"], ["MainCity", "is_a", "capital city"], ["MainCity", "located_in", "PrimaryKingdom"]]`

JSON Output Only:
[
"""
    logger.info("Calling LLM for KG pre-population triple extraction...")
    raw_triples_json_str = await llm_interface.async_call_llm(
        model_name=config.KNOWLEDGE_UPDATE_MODEL, 
        prompt=prompt, 
        temperature=0.6, 
        max_tokens=config.MAX_PREPOP_KG_TOKENS
    )
    parsed_triples = await llm_interface.async_parse_llm_json_response(
        raw_triples_json_str, "KG pre-population triple extraction", expect_type=list
    )

    if parsed_triples is None:
        logger.error(f"Failed to extract/parse KG triples for pre-population. Raw LLM: {raw_triples_json_str[:500] if raw_triples_json_str else 'EMPTY'}")
        await agent._save_debug_output(config.KG_PREPOPULATION_CHAPTER_NUM, "kg_prepop_final_fail_raw_llm", raw_triples_json_str or "EMPTY_PREPOP_TRIPLES_JSON")
        return

    if not parsed_triples:
        logger.info("No KG triples were extracted by LLM for pre-population.")
        return

    added_count, skipped_count = 0, 0
    kg_add_tasks = []
    for triple_any in parsed_triples:
        if isinstance(triple_any, list) and len(triple_any) == 3:
            subj = str(triple_any[0]).strip() if triple_any[0] is not None else ""
            pred = str(triple_any[1]).strip() if triple_any[1] is not None else ""
            obj  = str(triple_any[2]).strip() if triple_any[2] is not None else ""
            if subj and pred and obj: 
                kg_add_tasks.append(
                    state_manager.async_add_kg_triple(subj, pred, obj, config.KG_PREPOPULATION_CHAPTER_NUM, is_provisional=False) # Pre-pop data is not provisional
                )
                added_count += 1
            else:
                logger.warning(f"Skipping invalid pre-population triple (empty component after strip): {triple_any}")
                skipped_count += 1
        else:
            logger.warning(f"Skipping invalid pre-population triple format (not list of 3): {triple_any}")
            skipped_count += 1
    
    if kg_add_tasks:
        await asyncio.gather(*kg_add_tasks)

    logger.info(f"KG pre-population complete: Added {added_count} foundational triples. Skipped {skipped_count} invalid triples.")
    if added_count == 0 and parsed_triples: # Log if LLM gave data but none was valid
        logger.warning("KG pre-population resulted in 0 valid triples added despite LLM returning data. Check LLM output and parsing.")


# --- Overall Knowledge Base Update Orchestration ---

async def update_all_knowledge_bases_logic(
    agent, # NovelWriterAgent instance
    chapter_number: int,
    final_text: str,
    from_flawed_draft: bool # Indicates if final_text comes from an unrevised/flawed draft
):
    """Updates JSON character/world profiles and the Knowledge Graph based on the finalized chapter."""
    if not final_text:
        logger.warning(f"Skipping all knowledge base updates for ch {chapter_number}: Final text is missing or empty.")
        return
    
    logger.info(f"Updating all knowledge bases for ch {chapter_number} (Source from flawed draft: {from_flawed_draft})...")
    
    # Create tasks for concurrent execution
    update_char_profiles_task = update_character_profiles_from_chapter_logic(
        agent, final_text, chapter_number, from_flawed_draft
    )
    update_world_profiles_task = update_world_building_from_chapter_logic(
        agent, final_text, chapter_number, from_flawed_draft
    )
    update_kg_task = extract_and_store_kg_triples_logic(
        agent, final_text, chapter_number, from_flawed_draft
    )
    
    # Add thematic consistency check
    thematic_check_task = check_thematic_consistency_logic(
        agent, chapter_number, final_text
    )
    
    try:
        # Gather results of all update tasks
        await asyncio.gather(
            update_char_profiles_task,
            update_world_profiles_task,
            update_kg_task,
            thematic_check_task
        )
        logger.info(f"All knowledge base updates (JSON profiles & KG) completed for ch {chapter_number}.")
    except Exception as e:
        logger.error(f"Error during concurrent knowledge base updates for ch {chapter_number}: {e}", exc_info=True)
        # Current behavior: log the error and save debug output.
        # For more critical systems, a more sophisticated error handling strategy might be needed,
        # such as retries for specific tasks or marking the chapter as needing knowledge reprocessing.
        await agent._save_debug_output(chapter_number, "knowledge_base_update_exception", str(e))