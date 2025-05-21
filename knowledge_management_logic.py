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

from prompt_data_getters import (
    get_filtered_character_profiles_for_prompt, # This will change to use Neo4j
    heuristic_entity_spotter_for_kg,
    get_filtered_world_data_for_prompt # This will change to use Neo4j
)
from state_manager import state_manager

logger = logging.getLogger(__name__)

# --- Summarization ---

@alru_cache(maxsize=config.SUMMARY_CACHE_SIZE)
async def llm_summarize_full_chapter_text_logic(chapter_text_full_key: str, chapter_number: int) -> str:
    """Cached LLM call for summarizing full chapter text. Key is the full text to cache effectively."""
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

def _apply_trait_modification(current_traits_list: List[str], modification_details_str: str) -> List[str]:
    """Applies ADD/REMOVE operations to a list of traits."""
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
    """Applies a modification proposal string to a dictionary (profile or world item).
       This modifies the agent's in-memory Python dictionary representation.
       The state_manager.save_* methods will later decompose these dicts to Neo4j.
    """
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
        key_name_from_proposal_upper.lower() # Default to lowercase if no case-insensitive match
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


# --- Character Profile Merging Logic (Operates on in-memory Python dicts) ---

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
        elif key not in existing_profile and value is not None: existing_profile[key] = value # Add new fields
    if dev_key in char_update_data and isinstance(char_update_data[dev_key], str) and char_update_data[dev_key].strip():
        existing_profile[dev_key] = char_update_data[dev_key]

def merge_character_profile_updates_logic(
    agent, updates_from_llm: Dict[str, Any], chapter_number: int, from_flawed_draft: bool
):
    """ Merges LLM updates into the agent's in-memory character_profiles dict. """
    if not updates_from_llm:
        logger.info(f"No character profile updates from LLM to merge for ch {chapter_number}.")
        return
    logger.info(f"Merging character profile JSON updates for ch {chapter_number}. Characters in update: {list(updates_from_llm.keys())}")
    updated_chars_count, new_chars_count = 0, 0
    provisional_marker_key = f"source_quality_chapter_{chapter_number}" # Marks if data from this chapter's update is flawed
    
    for char_name, char_update_data_original in updates_from_llm.items():
        if not isinstance(char_update_data_original, dict):
            logger.warning(f"Skipping invalid character update data for '{char_name}' (not a dict). Data: {char_update_data_original}")
            continue
        char_update_data = char_update_data_original.copy()
        if from_flawed_draft: char_update_data[provisional_marker_key] = "provisional_from_unrevised_draft"
        
        dev_key = f"development_in_chapter_{chapter_number}"
        is_substantive_update = any(k not in [provisional_marker_key, "modification_proposal"] for k in char_update_data)

        if dev_key not in char_update_data and is_substantive_update:
            char_update_data[dev_key] = "Character appeared or was mentioned in this chapter."
        
        modification_proposal = char_update_data.get("modification_proposal")
        
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
            # Ensure provisional marker is set on existing profile if update is from flawed draft
            if from_flawed_draft: existing_profile[provisional_marker_key] = "provisional_from_unrevised_draft"


    if updated_chars_count > 0 or new_chars_count > 0:
        logger.info(f"Character profile Python dict merge complete for ch {chapter_number}. Updated: {updated_chars_count}, New: {new_chars_count}.")
    else:
        logger.info(f"No character profiles were effectively updated or added to Python dicts for ch {chapter_number} after LLM analysis.")

# --- World Item Merging Logic (Operates on in-memory Python dicts) ---

def robust_merge_world_item_data_logic(
    target_dict: Dict[str, Any], update_dict: Dict[str, Any], item_name_for_log: str, chapter_num: int, from_flawed_draft_source: bool
) -> Dict[str, Any]:
    """ Robustly merges updates into a target world item dictionary. In-memory operation. """
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
        
    if config.ENABLE_DYNAMIC_STATE_ADAPTATION and "modification_proposal" in update_dict:
        proposal = update_dict.pop("modification_proposal")
        if isinstance(proposal, str) and proposal.strip():
            apply_state_modification_proposal_logic(current_item_data, proposal, item_name_for_log, "world item")
            item_was_modified_this_call = True
            
    for key, value_from_update in update_dict.items():
        if key in [provisional_marker_key, "modification_proposal"] or key.startswith(("updated_in_chapter_", "added_in_chapter_", "source_quality_chapter_")):
            if key.startswith("elaboration_in_chapter_") and isinstance(value_from_update, str) and value_from_update.strip():
                 current_item_data[key] = value_from_update # Specific elaboration for this chapter
                 item_was_modified_this_call = True
            continue
        
        current_value_in_target = current_item_data.get(key)
        if isinstance(value_from_update, dict): # Recursive merge for sub-dictionaries
            merged_sub_dict = robust_merge_world_item_data_logic(
                current_value_in_target if isinstance(current_value_in_target, dict) else {},
                value_from_update, f"{item_name_for_log}.{key}", chapter_num, from_flawed_draft_source=False # Provisional only at top
            )
            if merged_sub_dict != current_value_in_target: item_was_modified_this_call = True
            current_item_data[key] = merged_sub_dict
        elif isinstance(value_from_update, list): # Union for lists
            if not isinstance(current_value_in_target, list):
                current_item_data[key] = []
                item_was_modified_this_call = True
            initial_list_len = len(current_item_data[key])
            for item_in_list_update in value_from_update:
                if item_in_list_update not in current_item_data[key]: current_item_data[key].append(item_in_list_update)
            if len(current_item_data[key]) > initial_list_len: item_was_modified_this_call = True
        elif value_from_update != current_value_in_target: # Direct update for other types
            current_item_data[key] = value_from_update
            item_was_modified_this_call = True
            
    if item_was_modified_this_call and not current_item_data.get(f"added_in_chapter_{chapter_num}"):
        # Mark as updated in this chapter if any modification occurred and it wasn't just added
        current_item_data[f"updated_in_chapter_{chapter_num}"] = True
        
    return current_item_data

def merge_world_item_updates_logic(
    agent, updates_from_llm: Dict[str, Any], chapter_number: int, from_flawed_draft: bool
):
    """ Merges LLM updates into the agent's in-memory world_building dict. """
    if not updates_from_llm:
        logger.info(f"No world-building updates from LLM to merge for ch {chapter_number}.")
        return
    logger.info(f"Merging world-building JSON updates for ch {chapter_number}. Categories in update: {list(updates_from_llm.keys())}")
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
            
            if existing_item_data is None: # Item was newly added
                merged_item_data[f"added_in_chapter_{chapter_number}"] = True 
                items_affected_count += 1
            elif merged_item_data.get(f"updated_in_chapter_{chapter_number}") or \
                 (from_flawed_draft and merged_item_data.get(f"source_quality_chapter_{chapter_number}")):
                items_affected_count += 1 # Item was updated or marked provisional

        if any(isinstance(v,dict) and (v.get(f"updated_in_chapter_{chapter_number}") or v.get(f"added_in_chapter_{chapter_number}")) 
               for v in target_category_dict.values()):
             target_category_dict[f"category_updated_in_chapter_{chapter_number}"] = True # Mark category as updated

    if items_affected_count > 0:
        logger.info(f"World-building Python dict merge complete for ch {chapter_number}. Approx {items_affected_count} items affected/added.")
    else:
        logger.info(f"No world-building Python dict items were effectively updated or added for ch {chapter_number} after LLM analysis.")

# --- Unified Knowledge Extraction ---

async def unified_knowledge_extraction(
    agent, 
    chapter_text: str, 
    chapter_number: int
) -> Dict[str, Any]:
    """
    Extracts all knowledge updates (character profiles, world-building, KG triples)
    from the full chapter text in a single LLM call.
    The prompt needs to guide the LLM to output data that can be mapped to fine-grained graph updates.
    """
    logger.info(f"Performing unified knowledge extraction for Chapter {chapter_number}...")
    if not chapter_text:
        logger.warning(f"Unified knowledge extraction skipped for Ch {chapter_number}: empty chapter text.")
        return {"character_updates": {}, "world_updates": {}, "knowledge_triples": []}

    protagonist_name = agent.plot_outline.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME)
    
    # For context, use the agent's current in-memory Python dicts.
    # The get_filtered_*_for_prompt functions will eventually query Neo4j for richer context.
    # For this iteration, we assume they work on agent's dicts or are adapted.
    current_profiles_for_prompt_dict = await get_filtered_character_profiles_for_prompt(agent, chapter_number - 1)
    current_world_for_prompt_dict = await get_filtered_world_data_for_prompt(agent, chapter_number - 1)
    
    candidate_entities = await heuristic_entity_spotter_for_kg(agent, chapter_text)
    
    dynamic_instr_char = (
        f"For existing characters, if their traits, status, or core description needs modification based on THIS chapter's events, "
        f"include a `\"modification_proposal\"` field in their update object. Example: `\"modification_proposal\": \"MODIFY traits: ADD 'Determined', REMOVE 'Hesitant'\"`. "
        f"Also specify all current `\"traits\"` as a list if changed, new `\"status\"`, new `\"description\"`. "
        f"For NEW characters, provide `\"description\"`, `\"traits\"` (list), `\"status\"`. "
        f"Include `\"relationships\"` as an object where keys are target character names and values are relationship descriptions (e.g., `\"allied with\"`, `\"distrusts\"`)."
        f"Only include characters that are updated, newly introduced, or have a modification proposal."
    ) if config.ENABLE_DYNAMIC_STATE_ADAPTATION else "Only include characters whose information is directly updated or those newly introduced in THIS chapter. Provide full description, traits, status for new chars."

    dynamic_instr_world = (
        f"For existing world items, if their properties need modification, include a `\"modification_proposal\"`. "
        f"Example: `\"modification_proposal\": \"MODIFY atmosphere: 'Now heavy with magical fallout'\"`. "
        f"Also provide the new full value for any changed properties. For NEW world items, provide all known properties. "
        f"E.g., for locations: `\"description\"`, `\"atmosphere\"`. For factions: `\"description\"`, `\"goals\"` (list). "
        f"Only include world elements (locations, society items, systems, lore, history) that are new, significantly changed by THIS chapter's events, or have a modification proposal."
    ) if config.ENABLE_DYNAMIC_STATE_ADAPTATION else "Only include world elements that are new or significantly changed by THIS chapter's events. Provide full details for new items."

    common_predicates = [
        "is_a", "located_in", "has_trait", "status_is", "feels", "knows", "believes", "wants", 
        "interacted_with", "travelled_to", "discovered", "acquired", "lost", "used_item", 
        "attacked", "helped", "damaged", "repaired", "contains", "part_of", "caused_by", 
        "leads_to", "observed", "heard", "said", "thought_about", "decided_to", "has_goal", 
        "has_feature", "related_to", "member_of", "leader_of", "enemy_of", "ally_of", 
        "works_for", "has_ability", "possesses", "created_by", "has_description", "has_atmosphere",
        "has_rule", "has_history_event"
    ]

    prompt = f"""/no_think
You are a comprehensive literary analyst and knowledge engineer.
Analyze the **Complete Chapter {chapter_number} Text** (protagonist: {protagonist_name}) and extract information for three distinct knowledge bases:
1.  **CHARACTER_UPDATES**: Identify changes to existing character profiles or new characters introduced.
2.  **WORLD_UPDATES**: Identify new or significantly changed world-building elements.
3.  **KG_TRIPLES**: Extract factual (Subject, Predicate, Object) triples.

**Reference Information (Current State Before This Chapter - for context only, extract from THIS chapter's text):**
  **Character Profiles Snapshot:**
  ```json
  {json.dumps(current_profiles_for_prompt_dict, indent=2, ensure_ascii=False, default=str, sort_keys=True)}
  ```
  **World Building Snapshot:**
  ```json
  {json.dumps(current_world_for_prompt_dict, indent=2, ensure_ascii=False, default=str, sort_keys=True)}
  ```
  **Heuristically Identified Candidate Entities:**
  ```json
  {json.dumps(candidate_entities)}
  ```

**Complete Chapter {chapter_number} Text (Analyze this full text):**
--- BEGIN COMPLETE CHAPTER TEXT ---
{chapter_text}
--- END COMPLETE CHAPTER TEXT ---

**Output Format (CRITICAL):**
Provide your analysis ONLY as a single, valid JSON object with top-level keys: `"character_updates"`, `"world_updates"`, `"knowledge_triples"`.

**1. `character_updates` (JSON Object):**
   - Keyed by character name. Each value is an object with fields like "description", "traits" (list), "status", "relationships" (object), etc.
   - Add `development_in_chapter_{chapter_number}` string summarizing their role/changes in THIS chapter.
   - {dynamic_instr_char}
   - Example: `{{"CharacterName": {{"status": "Updated", "traits": ["Brave", "Tired"], "development_in_chapter_{chapter_number}": "Action."}}}}`

**2. `world_updates` (JSON Object):**
   - Keyed by category (e.g., "locations", "society", "systems", "lore", "history").
   - Each category contains item names as keys, with their updated details as objects.
   - Add `elaboration_in_chapter_{chapter_number}` string providing context from THIS chapter.
   - {dynamic_instr_world}
   - Example: `{{"locations": {{"NewCave": {{"description": "Dark.", "atmosphere": "Eerie", "elaboration_in_chapter_{chapter_number}": "Found."}}}}}}`

**3. `knowledge_triples` (JSON List of Lists):**
   - Each triple: `["Subject", "predicate_name", "Object"]`. Focus on THIS chapter.
   - Suggested Predicates: {', '.join(common_predicates)}
   - Example: `[["{protagonist_name}", "travelled_to", "Eclipse Spire"], ["Eclipse Spire", "is_a", "ancient ruin"]]`

Output ONLY the JSON object.
"""
    
    logger.info(f"Calling LLM ({config.KNOWLEDGE_UPDATE_MODEL}) for unified knowledge extraction (Ch {chapter_number}).")
    raw_extraction_json = await llm_interface.async_call_llm(
        model_name=config.KNOWLEDGE_UPDATE_MODEL,
        prompt=prompt,
        temperature=0.5, # Slightly lower temp for more factual extraction
        allow_fallback=True,
        stream_to_disk=True
    )

    parsed_result: Optional[Any] = await llm_interface.async_parse_llm_json_response(
        raw_extraction_json, f"unified knowledge extraction for ch {chapter_number}", expect_type=dict
    )

    default_response = {"character_updates": {}, "world_updates": {}, "knowledge_triples": []}
    if isinstance(parsed_result, dict):
        final_extraction = {
            "character_updates": parsed_result.get("character_updates", {}),
            "world_updates": parsed_result.get("world_updates", {}),
            "knowledge_triples": parsed_result.get("knowledge_triples", [])
        }
        if not isinstance(final_extraction["character_updates"], dict): final_extraction["character_updates"] = {}
        if not isinstance(final_extraction["world_updates"], dict): final_extraction["world_updates"] = {}
        if not isinstance(final_extraction["knowledge_triples"], list): final_extraction["knowledge_triples"] = []
        
        logger.info(f"Unified knowledge extraction for Ch {chapter_number} complete. "
                    f"Char updates: {len(final_extraction['character_updates'])}, "
                    f"World updates categories: {len(final_extraction['world_updates'])}, "
                    f"KG Triples: {len(final_extraction['knowledge_triples'])}.")
        return final_extraction
    else:
        logger.error(f"Failed to parse unified knowledge extraction for Ch {chapter_number} into a dict. Raw: '{raw_extraction_json[:500]}...'")
        await agent._save_debug_output(chapter_number, "unified_extraction_parse_fail", raw_extraction_json)
        return default_response


# --- Knowledge Graph Pre-population ---

def _prepare_prepopulation_data_summary(agent) -> Dict[str, Any]:
    """Prepares a structured summary of Python dicts (plot, characters, world) for KG pre-population prompt."""
    summary = {}
    plot_summary = {
        "title": agent.plot_outline.get("title"), "genre": agent.plot_outline.get("genre"),
        "theme": agent.plot_outline.get("theme"), "logline": agent.plot_outline.get("logline"),
        "protagonist_name": agent.plot_outline.get("protagonist_name"),
        "protagonist_description": agent.plot_outline.get("protagonist_description"),
        "protagonist_character_arc": agent.plot_outline.get("character_arc"),
        "antagonist_name": agent.plot_outline.get("antagonist_name"),
        "antagonist_description": agent.plot_outline.get("antagonist_description"),
        "antagonist_motivations": agent.plot_outline.get("antagonist_motivations"),
        "setting_description": agent.plot_outline.get("setting"),
        "conflict_summary": agent.plot_outline.get("conflict"),
        "inciting_incident": agent.plot_outline.get("inciting_incident"),
        "climax_event_preview": agent.plot_outline.get("climax_event_preview"),
        "key_plot_points_summary": agent.plot_outline.get("plot_points", [])
    }
    summary["novel_concept_and_plot"] = {k: v for k, v in plot_summary.items() if v}

    char_summary = {}
    for char_name, profile in agent.character_profiles.items():
        if isinstance(profile, dict):
            char_summary[char_name] = {
                "description": profile.get("description", ""), "role": profile.get("role", profile.get("role_in_story", "N/A")),
                "initial_status": profile.get("status", ""), "traits_preview": profile.get("traits", []),
                "motivations": profile.get("motivations", ""), "relationships_preview": profile.get("relationships", {}) # Added relationships
            }
    if char_summary: summary["key_characters"] = char_summary
    
    world_highlights = {}
    overview = agent.world_building.get("_overview_")
    if overview and isinstance(overview, dict) and overview.get("description"):
        world_highlights["overall_setting_description"] = overview["description"]

    for category, items in agent.world_building.items():
        if category in ["_overview_", "is_default", "user_supplied_data", "source"] or not isinstance(items, dict): continue
        category_highlights = {}
        item_count = 0
        for item_name, item_details in items.items():
            if item_name.startswith(("_", "source_quality_chapter_", "category_updated_in_chapter_")): continue
            if item_count >= 5: break 
            if isinstance(item_details, dict):
                desc = item_details.get("description", "")
                if desc:
                    details_to_add = {"description_snippet": desc[:150] + "..."}
                    if item_details.get("atmosphere"): details_to_add["atmosphere_snippet"] = str(item_details["atmosphere"])[:100]
                    if item_details.get("goals"): details_to_add["goals_preview"] = item_details["goals"][:3]
                    if item_details.get("rules"): details_to_add["rules_preview"] = item_details["rules"][:3]
                    category_highlights[item_name] = details_to_add
                item_count +=1
        if category_highlights: world_highlights[category] = category_highlights
    if world_highlights: summary["world_highlights"] = world_highlights
    return summary


async def prepopulate_kg_from_initial_data_logic(agent):
    """
    Takes the agent's initial Python dicts (plot_outline, character_profiles, world_building)
    and directly translates them into fine-grained Neo4j nodes and relationships.
    This bypasses LLM extraction for pre-population and uses direct Cypher generation.
    """
    logger.info("Starting Knowledge Graph pre-population directly from agent's initial data dicts...")
    
    cypher_statements: List[Tuple[str, Dict[str, Any]]] = []

    # 1. Prepopulate from Plot Outline
    plot = agent.plot_outline
    novel_id = config.MAIN_NOVEL_INFO_NODE_ID
    if plot:
        # NovelInfo node
        novel_props = {k: v for k, v in plot.items() if not isinstance(v, (list, dict)) and v is not None}
        novel_props['id'] = novel_id
        cypher_statements.append((f"MERGE (ni:NovelInfo {{id: $id}}) SET ni = $props", {"id": novel_id, "props": novel_props}))

        # PlotPoints
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
    
    # 2. Prepopulate from Character Profiles
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
                rel_props = {"description": str(rel_detail), "chapter_added": config.KG_PREPOPULATION_CHAPTER_NUM, "is_provisional": False}
                if isinstance(rel_detail, dict) and "type" in rel_detail:
                    rel_type = rel_detail.pop("type", rel_type).upper()
                    rel_props.update(rel_detail)

                cypher_statements.append((
                    """
                    MATCH (c1:Character {name: $char_name1})
                    MERGE (c2:Character {name: $char_name2}) ON CREATE SET c2.description = 'Auto-created via relationship'
                    MERGE (c1)-[r:DYNAMIC_REL {type:$rel_type, chapter_added:$chap_add, is_provisional: $is_prov}]->(c2)
                    SET r += $rel_props 
                    """, # Using DYNAMIC_REL with type property
                    {"char_name1": char_name, "char_name2": target_name, "rel_type": rel_type, 
                     "chap_add": config.KG_PREPOPULATION_CHAPTER_NUM, "is_prov": False, "rel_props": rel_props }
                ))

    # 3. Prepopulate from World Building
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
                            cypher_statements.append((
                                f"""
                                MATCH (we:WorldElement {{id: $we_id}})
                                MERGE (v:ValueNode {{value: $val_item_value, type: '{list_prop_name}'}}) // Add type to ValueNode
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
    """
    Updates agent's in-memory Python dicts and adds explicit KG triples to Neo4j
    based on unified LLM extraction.
    The agent's _save_all_json_state method will then persist the updated Python dicts
    into the decomposed Neo4j structure.
    """
    if not final_text:
        logger.warning(f"Skipping all knowledge base updates for ch {chapter_number}: Final text is missing or empty.")
        return
    
    logger.info(f"Updating all knowledge bases for ch {chapter_number} (Source from flawed draft: {from_flawed_draft})...")
    
    extraction_results = await unified_knowledge_extraction(agent, final_text, chapter_number)
    
    character_updates_dict = extraction_results.get("character_updates", {})
    world_updates_dict = extraction_results.get("world_updates", {})
    kg_triples_list = extraction_results.get("knowledge_triples", [])

    # Process updates for in-memory Python dicts
    if character_updates_dict and isinstance(character_updates_dict, dict):
        merge_character_profile_updates_logic(agent, character_updates_dict, chapter_number, from_flawed_draft)
    else: logger.warning(f"No valid character updates (dict) from unified extraction for ch {chapter_number}.")

    if world_updates_dict and isinstance(world_updates_dict, dict):
        merge_world_item_updates_logic(agent, world_updates_dict, chapter_number, from_flawed_draft)
    else: logger.warning(f"No valid world-building updates (dict) from unified extraction for ch {chapter_number}.")

    # Add explicit KG triples to Neo4j
    if kg_triples_list and isinstance(kg_triples_list, list):
        added_count, skipped_count = 0, 0
        kg_add_tasks = []
        for triple_any in kg_triples_list:
            if isinstance(triple_any, list) and len(triple_any) == 3:
                subj, pred, obj_val = (str(triple_any[0]).strip() if triple_any[0] is not None else "",
                                   str(triple_any[1]).strip() if triple_any[1] is not None else "",
                                   str(triple_any[2]).strip() if triple_any[2] is not None else "")
                if subj and pred and obj_val:
                    # Truncate long objects for KG stability, though properties can be long
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
    # The updated Python dicts (agent.character_profiles, agent.world_building) will be saved
    # to Neo4j (decomposed) when agent._save_all_json_state() is called.