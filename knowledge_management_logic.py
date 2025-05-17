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
from type import KnowledgeGraph, Entity, Relationship, Event, Location, Character, Faction # Keep for potential future structured KG

from prompt_data_getters import (
    get_filtered_world_data_for_prompt,
    heuristic_entity_spotter_for_kg,
    get_filtered_character_profiles_for_prompt
)
from state_manager import state_manager

logger = logging.getLogger(__name__)

# --- Summarization ---

@alru_cache(maxsize=config.SUMMARY_CACHE_SIZE)
async def llm_summarize_full_chapter_text_logic(chapter_text_full_key: str, chapter_number: int) -> str:
    """Cached LLM call for summarizing full chapter text. Key is the full text to cache effectively."""
    # Note: If chapter_text_full_key is very long, it might exceed practical limits for a cache key
    # or LLM input limits depending on the model.
    # This implements the "full text" change as requested.
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
        model_name=config.SMALL_MODEL, # Usually a smaller model for summarization
        prompt=prompt,
        temperature=0.6,
        max_tokens=config.MAX_SUMMARY_TOKENS # Max output tokens for the summary
    )
    return llm_interface.clean_model_response(summary_raw).strip()

async def summarize_chapter_text_logic(chapter_text: Optional[str], chapter_number: int) -> Optional[str]:
    if not chapter_text or len(chapter_text) < 50:
        logger.warning(f"Chapter {chapter_number} text too short for summarization ({len(chapter_text or '')} chars).")
        return None
            
    # Per request, use full text for summarization LLM call.
    # The effectiveness of caching with full text as key needs to be monitored.
    cleaned_summary = await llm_summarize_full_chapter_text_logic(chapter_text, chapter_number)

    if cleaned_summary: 
        logger.info(f"Generated summary for ch {chapter_number}: '{cleaned_summary[:100].strip()}...'")
        return cleaned_summary
    
    logger.warning(f"Failed to generate a valid summary for ch {chapter_number} via LLM.")
    return None

# --- JSON State Modification Proposal Logic (Remains the same) ---

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


# --- Character Profile Merging Logic (Remains the same) ---

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
    if dev_key in char_update_data:
        new_profile[dev_key] = char_update_data[dev_key]
    if provisional_marker_key in char_update_data:
        new_profile[provisional_marker_key] = char_update_data[provisional_marker_key]
    logger.info(f"Prepared new character profile for '{char_name}'.")
    return new_profile

def _update_existing_character_profile_fields(
    existing_profile: Dict[str, Any],
    char_update_data: Dict[str, Any],
    dev_key: str,
    provisional_marker_key: str
):
    if provisional_marker_key in char_update_data:
        existing_profile[provisional_marker_key] = char_update_data[provisional_marker_key]
    for key, value in char_update_data.items():
        if key in ["modification_proposal", provisional_marker_key, dev_key]: continue
        if key == "traits" and isinstance(value, list):
            if "traits" not in existing_profile or not isinstance(existing_profile["traits"], list): existing_profile["traits"] = []
            valid_new_traits = {t for t in value if isinstance(t, str) and t.strip()}
            existing_profile["traits"] = sorted(list(set(existing_profile["traits"]).union(valid_new_traits)))
        elif key == "relationships" and isinstance(value, dict):
            if not isinstance(existing_profile.get("relationships"), dict): existing_profile["relationships"] = {}
            existing_profile["relationships"].update(value)
        elif key == "description" and isinstance(value, str) and value.strip():
            existing_profile["description"] = value
        elif key == "status" and isinstance(value, str) and value.strip():
            existing_profile["status"] = value
        elif key not in existing_profile and value is not None:
            existing_profile[key] = value
    if dev_key in char_update_data and isinstance(char_update_data[dev_key], str) and char_update_data[dev_key].strip():
        existing_profile[dev_key] = char_update_data[dev_key]

def merge_character_profile_updates_logic(
    agent, updates_from_llm: Dict[str, Any], chapter_number: int, from_flawed_draft: bool
):
    if not updates_from_llm:
        logger.info(f"No character profile updates from LLM to merge for ch {chapter_number}.")
        return
    logger.info(f"Merging character profile JSON updates for ch {chapter_number}. Characters in update: {list(updates_from_llm.keys())}")
    updated_chars_count, new_chars_count = 0, 0
    provisional_marker_key = f"source_quality_chapter_{chapter_number}"
    for char_name, char_update_data_original in updates_from_llm.items():
        if not isinstance(char_update_data_original, dict):
            logger.warning(f"Skipping invalid character update data for '{char_name}' (not a dict). Data: {char_update_data_original}")
            continue
        char_update_data = char_update_data_original.copy()
        if from_flawed_draft: char_update_data[provisional_marker_key] = "provisional_from_unrevised_draft"
        dev_key = f"development_in_chapter_{chapter_number}"
        is_substantive_update = len(char_update_data) > (1 if provisional_marker_key in char_update_data else 0)
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
    if updated_chars_count > 0 or new_chars_count > 0:
        logger.info(f"Character profile JSON merge complete for ch {chapter_number}. Updated: {updated_chars_count}, New: {new_chars_count}.")
    else:
        logger.info(f"No character profiles were effectively updated or added for ch {chapter_number} after LLM analysis.")

# --- World Item Merging Logic (Remains the same) ---

def robust_merge_world_item_data_logic(
    target_dict: Dict[str, Any], update_dict: Dict[str, Any], item_name_for_log: str, chapter_num: int, from_flawed_draft_source: bool
) -> Dict[str, Any]:
    if not isinstance(target_dict, dict):
        logger.warning(f"World item '{item_name_for_log}' target_dict was not a dict. Initializing as a new dict. Old value: '{str(target_dict)[:100]}'")
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
                logger.warning(f"Skipping invalid item_details for '{item_name}' in category '{category_key}' (not a dict) for ch {chapter_number}. Data: {item_update_details}")
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
        logger.info(f"World-building JSON merge complete for ch {chapter_number}. Approximately {items_affected_count} items affected/added.")
    else:
        logger.info(f"No world-building JSON items were effectively updated or added for ch {chapter_number} after LLM analysis.")

# --- Unified Knowledge Extraction ---

async def unified_knowledge_extraction(
    agent, # NovelWriterAgent instance
    chapter_text: str, 
    chapter_number: int
    # `from_flawed_draft` is handled by the caller (`update_all_knowledge_bases_logic`)
    # when processing the results of this function.
) -> Dict[str, Any]:
    """
    Extracts all knowledge updates (character profiles, world-building, KG triples)
    from the full chapter text in a single LLM call.
    """
    logger.info(f"Performing unified knowledge extraction for Chapter {chapter_number}...")
    if not chapter_text:
        logger.warning(f"Unified knowledge extraction skipped for Ch {chapter_number}: empty chapter text.")
        return {"character_updates": {}, "world_updates": {}, "knowledge_triples": []}

    protagonist_name = agent.plot_outline.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME)
    
    # Prepare context for character and world updates within the unified prompt
    current_profiles_for_prompt = await get_filtered_character_profiles_for_prompt(agent, chapter_number - 1)
    current_world_for_prompt = await get_filtered_world_data_for_prompt(agent, chapter_number - 1)
    candidate_entities = await heuristic_entity_spotter_for_kg(agent, chapter_text) # Use full text
    candidate_entities_json_for_prompt = json.dumps(candidate_entities)
    
    dynamic_instr_char = (
        f"For existing characters, if their traits, status, or core description needs modification based on THIS chapter's events, "
        f"include a `\"modification_proposal\"` field. Example: `\"modification_proposal\": \"MODIFY traits: ADD 'Determined', REMOVE 'Hesitant'\"`. "
        f"Only include characters that are updated, newly introduced, or have a modification proposal."
    ) if config.ENABLE_DYNAMIC_STATE_ADAPTATION else "Only include characters whose information is directly updated or those newly introduced in THIS chapter."

    dynamic_instr_world = (
        f"For existing world items, if their properties need modification, include a `\"modification_proposal\"`. "
        f"Example: `\"modification_proposal\": \"MODIFY atmosphere: 'Now heavy with magical fallout'\"`. "
        f"Only include world elements (locations, society items, systems, lore, history) that are new, significantly changed by THIS chapter's events, or have a modification proposal."
    ) if config.ENABLE_DYNAMIC_STATE_ADAPTATION else "Only include world elements that are new or significantly changed by THIS chapter's events."

    common_predicates = [
        "is_a", "located_in", "has_trait", "status_is", "feels", "knows", "believes", "wants", 
        "interacted_with", "travelled_to", "discovered", "acquired", "lost", "used_item", 
        "attacked", "helped", "damaged", "repaired", "contains", "part_of", "caused_by", 
        "leads_to", "observed", "heard", "said", "thought_about", "decided_to", "has_goal", 
        "has_feature", "related_to", "member_of", "leader_of", "enemy_of", "ally_of", 
        "works_for", "has_ability", "possesses", "created_by"
    ]

    prompt = f"""/no_think
You are a comprehensive literary analyst and knowledge engineer.
Analyze the **Complete Chapter {chapter_number} Text** (protagonist: {protagonist_name}) and extract information for three distinct knowledge bases:
1.  **CHARACTER_UPDATES**: Identify changes to existing character profiles or new characters introduced.
2.  **WORLD_UPDATES**: Identify new or significantly changed world-building elements (locations, society, systems, lore, history).
3.  **KG_TRIPLES**: Extract factual (Subject, Predicate, Object) triples representing key facts, events, and relationships.

**Reference Information (Current State Before This Chapter):**
  **Character Profiles (for CHARACTER_UPDATES context):**
  ```json
  {json.dumps(current_profiles_for_prompt, indent=2, ensure_ascii=False, default=str, sort_keys=True)}
  ```
  **World Building Notes (for WORLD_UPDATES context):**
  ```json
  {json.dumps(current_world_for_prompt, indent=2, ensure_ascii=False, default=str, sort_keys=True)}
  ```
  **Heuristically Identified Candidate Entities (for KG_TRIPLES Subject/Object consideration):**
  ```json
  {candidate_entities_json_for_prompt}
  ```

**Complete Chapter {chapter_number} Text (Analyze this full text):**
--- BEGIN COMPLETE CHAPTER TEXT ---
{chapter_text}
--- END COMPLETE CHAPTER TEXT ---

**Output Format (CRITICAL):**
Provide your analysis ONLY as a single, valid JSON object.
The JSON object *must* have these three top-level keys: `"character_updates"`, `"world_updates"`, and `"knowledge_triples"`.

**1. `character_updates` (JSON Object):**
   - Keyed by character name.
   - Each character entry includes relevant updated fields (e.g., "description", "traits", "status", "relationships").
   - Add `development_in_chapter_{chapter_number}` summarizing their role/changes in THIS chapter.
   - {dynamic_instr_char}
   - Example: `{{"CharacterName": {{"status": "Updated", "development_in_chapter_{chapter_number}": "Action."}}}}`
   - If no character updates, use an empty object: `{{}}`

**2. `world_updates` (JSON Object):**
   - Keyed by category (e.g., "locations", "society", "systems", "lore", "history").
   - Each category contains item names as keys, with their updated details.
   - Add `elaboration_in_chapter_{chapter_number}` providing context from THIS chapter.
   - {dynamic_instr_world}
   - Example: `{{"locations": {{"NewCave": {{"description": "Dark.", "elaboration_in_chapter_{chapter_number}": "Found."}}}}}}`
   - If no world updates for a category, use an empty object for that category: `{{"locations": {{}}, "systems": {{...}}}}`
   - If no world updates at all, `world_updates` can be `{{}}`.

**3. `knowledge_triples` (JSON List of Lists):**
   - Each triple is a list: `["Subject", "predicate_name", "Object"]`.
   - Focus on information explicitly stated or very strongly implied in THIS chapter text.
   - Use suggested predicates or concise alternatives (lowercase_with_underscores).
   - Suggested Predicates: {', '.join(common_predicates)}
   - Example: `[["{protagonist_name}", "travelled_to", "Eclipse Spire"], ["Eclipse Spire", "is_a", "ancient ruin"]]`
   - If no triples, use an empty list: `[]`

**Example of a complete valid JSON response:**
```json
{{
  "character_updates": {{
    "Alice": {{
      "status": "Determined",
      "modification_proposal": "MODIFY traits: ADD 'Resilient'",
      "development_in_chapter_{chapter_number}": "Alice overcame a major obstacle."
    }}
  }},
  "world_updates": {{
    "locations": {{
      "Shadowfen": {{
        "description": "A newly discovered swamp, treacherous and dark.",
        "atmosphere": "Oppressive",
        "elaboration_in_chapter_{chapter_number}": "The protagonist ventured into Shadowfen for the first time."
      }}
    }},
    "systems": {{}}
  }},
  "knowledge_triples": [
    ["Alice", "entered", "Shadowfen"],
    ["Shadowfen", "is_a", "swamp"],
    ["Alice", "felt", "anxious"]
  ]
}}
```
Output ONLY the JSON object.
"""
    
    logger.info(f"Calling LLM ({config.KNOWLEDGE_UPDATE_MODEL}) for unified knowledge extraction (Ch {chapter_number}).")
    raw_extraction_json = await llm_interface.async_call_llm(
        model_name=config.KNOWLEDGE_UPDATE_MODEL,
        prompt=prompt,
        temperature=0.6,
        allow_fallback=True # Knowledge extraction is important
    )

    parsed_result: Optional[Any] = await llm_interface.async_parse_llm_json_response(
        raw_extraction_json, f"unified knowledge extraction for ch {chapter_number}", expect_type=dict
    )

    default_response = {
        "character_updates": {}, 
        "world_updates": {}, 
        "knowledge_triples": []
    }

    if isinstance(parsed_result, dict):
        # Ensure all keys are present, default to empty if missing from LLM response
        final_extraction = {
            "character_updates": parsed_result.get("character_updates", {}),
            "world_updates": parsed_result.get("world_updates", {}),
            "knowledge_triples": parsed_result.get("knowledge_triples", [])
        }
        # Basic type validation for sub-structures
        if not isinstance(final_extraction["character_updates"], dict):
            logger.warning(f"Unified extraction: 'character_updates' is not a dict. Defaulting to empty. Raw: {final_extraction['character_updates']}")
            final_extraction["character_updates"] = {}
        if not isinstance(final_extraction["world_updates"], dict):
            logger.warning(f"Unified extraction: 'world_updates' is not a dict. Defaulting to empty. Raw: {final_extraction['world_updates']}")
            final_extraction["world_updates"] = {}
        if not isinstance(final_extraction["knowledge_triples"], list):
            logger.warning(f"Unified extraction: 'knowledge_triples' is not a list. Defaulting to empty. Raw: {final_extraction['knowledge_triples']}")
            final_extraction["knowledge_triples"] = []
        
        logger.info(f"Unified knowledge extraction for Ch {chapter_number} complete. "
                    f"Char updates: {len(final_extraction['character_updates'])}, "
                    f"World updates categories: {len(final_extraction['world_updates'])}, "
                    f"KG Triples: {len(final_extraction['knowledge_triples'])}.")
        return final_extraction
    else:
        logger.error(f"Failed to parse unified knowledge extraction for Ch {chapter_number} into a dict. Raw: '{raw_extraction_json[:500]}...'")
        await agent._save_debug_output(chapter_number, "unified_extraction_parse_fail", raw_extraction_json)
        return default_response


# --- Knowledge Graph Pre-population (Remains largely the same, but uses full plot/world from agent state) ---

async def prepopulate_kg_from_initial_data_logic(agent): # NovelWriterAgent instance
    logger.info("Starting Knowledge Graph pre-population from plot and world data...")
    
    # Using full agent.plot_outline and agent.world_building as context for this pre-population.
    # Pruning is still good for the prompt, but it's based on the complete data.
    pruned_plot = {
        "title": agent.plot_outline.get("title"), 
        "protagonist_name": agent.plot_outline.get("protagonist_name"),
        "genre": agent.plot_outline.get("genre"), 
        "theme": agent.plot_outline.get("theme"),
        "setting_description": agent.plot_outline.get("setting"),
        "conflict_summary": agent.plot_outline.get("conflict"),
        "character_arc": agent.plot_outline.get("character_arc"), 
        "key_plot_points_summary": agent.plot_outline.get("plot_points", [])[:2]
    }
    pruned_world = {}
    for category, items in agent.world_building.items():
        if category == "is_default" or not isinstance(items, dict): continue
        pruned_world[category] = {}
        for item_name, item_details in list(items.items())[:3]: 
            if isinstance(item_details, dict):
                desc = item_details.get("description", item_details.get("text", ""))
                if isinstance(desc, str) and desc.strip():
                     pruned_world[category][item_name] = {"description_snippet": desc[:200].strip() + "..."}
            elif isinstance(item_details, str) and item_details.strip():
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
    if not isinstance(parsed_triples, list):
        logger.error(f"parsed_triples is not a list as expected for KG pre-population. Got type: {type(parsed_triples)}. Raw LLM: {raw_triples_json_str[:500] if raw_triples_json_str else 'EMPTY'}")
        await agent._save_debug_output(config.KG_PREPOPULATION_CHAPTER_NUM, "kg_prepop_invalid_parsed_triples_type", f"Expected list, got {type(parsed_triples)}")
        return
    if not hasattr(parsed_triples, '__iter__'):
        logger.error(f"parsed_triples (expected to be a list) is not iterable for KG pre-population. Got type: {type(parsed_triples)}")
        await agent._save_debug_output(config.KG_PREPOPULATION_CHAPTER_NUM, "kg_prepop_non_iterable_parsed_triples", str(type(parsed_triples)))
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
                kg_add_tasks.append(state_manager.async_add_kg_triple(subj, pred, obj, config.KG_PREPOPULATION_CHAPTER_NUM, is_provisional=False))
                added_count += 1
            else: logger.warning(f"Skipping invalid pre-population triple (empty component after strip): {triple_any}")
        else: logger.warning(f"Skipping invalid pre-population triple format (not list of 3): {triple_any}")
    if kg_add_tasks: await asyncio.gather(*kg_add_tasks)
    logger.info(f"KG pre-population complete: Added {added_count} foundational triples. Skipped {skipped_count} invalid triples.")
    if added_count == 0 and parsed_triples:
        logger.warning("KG pre-population resulted in 0 valid triples added despite LLM returning data. Check LLM output and parsing.")

# --- Overall Knowledge Base Update Orchestration ---

async def update_all_knowledge_bases_logic(
    agent, # NovelWriterAgent instance
    chapter_number: int,
    final_text: str,
    from_flawed_draft: bool # Indicates if final_text comes from an unrevised/flawed draft
):
    """
    Updates JSON character/world profiles and the Knowledge Graph based on the finalized chapter,
    using a unified LLM call for extraction.
    """
    if not final_text:
        logger.warning(f"Skipping all knowledge base updates for ch {chapter_number}: Final text is missing or empty.")
        return
    
    logger.info(f"Updating all knowledge bases for ch {chapter_number} (Source from flawed draft: {from_flawed_draft})...")
    
    # Perform unified knowledge extraction
    extraction_results = await unified_knowledge_extraction(agent, final_text, chapter_number)
    
    character_updates = extraction_results.get("character_updates", {})
    world_updates = extraction_results.get("world_updates", {})
    kg_triples = extraction_results.get("knowledge_triples", [])

    # Process character profile updates
    if character_updates and isinstance(character_updates, dict):
        merge_character_profile_updates_logic(agent, character_updates, chapter_number, from_flawed_draft)
    else:
        logger.warning(f"No valid character updates received from unified extraction for ch {chapter_number}.")

    # Process world-building updates
    if world_updates and isinstance(world_updates, dict):
        merge_world_item_updates_logic(agent, world_updates, chapter_number, from_flawed_draft)
    else:
        logger.warning(f"No valid world-building updates received from unified extraction for ch {chapter_number}.")

    # Process Knowledge Graph triples
    if kg_triples and isinstance(kg_triples, list):
        added_count, skipped_count = 0, 0
        kg_add_tasks = []
        for triple_any in kg_triples:
            if isinstance(triple_any, list) and len(triple_any) == 3:
                subj = str(triple_any[0]).strip() if triple_any[0] is not None else ""
                pred = str(triple_any[1]).strip() if triple_any[1] is not None else ""
                obj  = str(triple_any[2]).strip() if triple_any[2] is not None else ""
                if subj and pred and obj:
                    kg_add_tasks.append(
                        state_manager.async_add_kg_triple(subj, pred, obj, chapter_number, is_provisional=from_flawed_draft)
                    )
                    added_count += 1
                else:
                    logger.warning(f"Skipping invalid KG triple (empty component after strip) from unified_extraction in ch {chapter_number}: Original: {triple_any}")
                    skipped_count += 1
            else:
                logger.warning(f"Skipping invalid KG triple format (not a list of 3) from unified_extraction in ch {chapter_number}: {triple_any}")
                skipped_count += 1
        
        if kg_add_tasks:
            await asyncio.gather(*kg_add_tasks)
        logger.info(f"KG update from unified_extraction for ch {chapter_number}: Attempted to add {added_count} triples, skipped {skipped_count}.")
    else:
        logger.info(f"No KG triples received from unified extraction for ch {chapter_number}.")

    # Thematic consistency check is now part of evaluate_chapter_draft_logic, so no separate call here.
    
    logger.info(f"All knowledge base updates from unified extraction completed for ch {chapter_number}.")