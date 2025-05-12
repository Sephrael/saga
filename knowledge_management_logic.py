# knowledge_management_logic.py
"""
Handles updates to knowledge bases (JSON profiles, Knowledge Graph)
and summarization for the SAGA system.
"""
import logging
import json
import re
import asyncio
from typing import Dict, List, Optional, Any

from async_lru import alru_cache # For caching LLM calls

import config
import llm_interface
# Import prompt data getters
from prompt_data_getters import (
    get_filtered_character_profiles_for_prompt,
    get_filtered_world_data_for_prompt,
    heuristic_entity_spotter_for_kg
)
from state_manager import state_manager

logger = logging.getLogger(__name__)

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
        model_name=config.SUMMARIZATION_MODEL,
        prompt=prompt,
        temperature=0.6,
        max_tokens=config.MAX_SUMMARY_TOKENS
    )
    return llm_interface.clean_model_response(summary_raw).strip()

async def summarize_chapter_text_logic(agent, chapter_text: Optional[str], chapter_number: int) -> Optional[str]:
    """ 'agent' is an instance of NovelWriterAgent (unused here but kept for consistency if needed later)."""
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

def apply_state_modification_proposal_logic(agent, target_dict: Dict[str, Any], proposal_str: str, item_name_for_log: str, item_type_for_log: str):
    """Applies a modification proposal string to a dictionary (profile or world item).
    'agent' is an instance of NovelWriterAgent (unused here but kept for consistency).
    """
    if not isinstance(proposal_str, str) or not proposal_str.strip():
        logger.debug(f"Empty or invalid modification proposal for '{item_name_for_log}'. Proposal: '{proposal_str}'")
        return
    
    logger.debug(f"Applying modification proposal for '{item_name_for_log}' ({item_type_for_log}): '{proposal_str}'")
    
    proposal_norm = proposal_str.strip().upper()
    key_to_modify_match = re.match(r"MODIFY\s+([\w_]+)\s*:(.*)", proposal_norm, re.IGNORECASE)

    if not key_to_modify_match:
        logger.warning(f"Invalid modification proposal format for '{item_name_for_log}'. Proposal: '{proposal_str}'. Expected 'MODIFY key: value'.")
        return

    key_name_upper = key_to_modify_match.group(1).strip()
    original_key_name = next((k for k in target_dict if k.upper() == key_name_upper), key_name_upper.lower())
    
    value_modification_str_original_case = proposal_str[key_to_modify_match.end(1)+1:].strip() 

    try:
        if original_key_name.lower() == "traits": 
            if "traits" not in target_dict or not isinstance(target_dict["traits"], list):
                target_dict["traits"] = [] 
            
            current_traits_set = set(target_dict["traits"])
            for add_match in re.finditer(r"ADD\s+['\"]([^'\"]+)['\"]", value_modification_str_original_case, re.IGNORECASE):
                trait_to_add = add_match.group(1).strip()
                if trait_to_add: current_traits_set.add(trait_to_add)
            for remove_match in re.finditer(r"REMOVE\s+['\"]([^'\"]+)['\"]", value_modification_str_original_case, re.IGNORECASE):
                trait_to_remove = remove_match.group(1).strip()
                if trait_to_remove: current_traits_set.discard(trait_to_remove)
            
            target_dict["traits"] = sorted(list(current_traits_set))
            logger.info(f"Applied trait modifications for '{item_name_for_log}'. New traits: {target_dict['traits']}")
        else: 
            new_value_str = value_modification_str_original_case.strip("'\" ") 
            if new_value_str: 
                target_dict[original_key_name] = new_value_str 
                logger.info(f"Applied modification to '{original_key_name}' for '{item_name_for_log}'. New value: '{new_value_str[:70]}...'")
            else:
                logger.warning(f"Modification proposal for '{original_key_name}' of '{item_name_for_log}' resulted in an empty new value. Proposal: '{proposal_str}'")
    except Exception as e:
        logger.error(f"Error applying modification proposal for '{item_name_for_log}': {e}. Proposal: '{proposal_str}'", exc_info=True)

def merge_character_profile_updates_logic(agent, updates_from_llm: Dict[str, Any], chapter_number: int, from_flawed_draft: bool):
    """Merges LLM-proposed character updates into the agent's character_profiles.
    'agent' is an instance of NovelWriterAgent.
    """
    if not updates_from_llm:
        logger.info(f"No character profile updates from LLM to merge for ch {chapter_number}.")
        return
    
    logger.info(f"Merging character profile JSON updates for ch {chapter_number}. Characters in update: {list(updates_from_llm.keys())}")
    updated_chars_count, new_chars_count = 0, 0
    provisional_marker_key = f"source_quality_chapter_{chapter_number}" 

    for char_name, char_update_data in updates_from_llm.items():
        if not isinstance(char_update_data, dict):
            logger.warning(f"Skipping invalid character update data for '{char_name}' (not a dict). Data: {char_update_data}")
            continue
        
        char_update = char_update_data.copy()
        
        if from_flawed_draft:
            char_update[provisional_marker_key] = "provisional_from_unrevised_draft"

        dev_key = f"development_in_chapter_{chapter_number}"
        if dev_key not in char_update and (len(char_update) > (1 if provisional_marker_key in char_update else 0)):
             char_update[dev_key] = "Character appeared or was mentioned in this chapter."
        
        if char_name not in agent.character_profiles:
            new_chars_count += 1
            logger.info(f"Adding new character '{char_name}' based on ch {chapter_number} analysis.")
            agent.character_profiles[char_name] = {
                "description": char_update.get("description", f"A character newly introduced in Chapter {chapter_number}."),
                "traits": sorted(list(set(t for t in char_update.get("traits", []) if isinstance(t, str) and t.strip()))),
                "relationships": char_update.get("relationships", {}), 
                "status": char_update.get("status", "Newly introduced")
            }
            if dev_key in char_update: agent.character_profiles[char_name][dev_key] = char_update[dev_key]
            if provisional_marker_key in char_update: agent.character_profiles[char_name][provisional_marker_key] = char_update[provisional_marker_key]
            
            if config.ENABLE_DYNAMIC_STATE_ADAPTATION and "modification_proposal" in char_update:
                apply_state_modification_proposal_logic(agent, agent.character_profiles[char_name], char_update["modification_proposal"], char_name, "new character profile")
        
        else: 
            updated_chars_count += 1
            logger.debug(f"Updating existing character '{char_name}' based on ch {chapter_number} analysis.")
            existing_profile = agent.character_profiles[char_name]
            
            if provisional_marker_key in char_update: 
                existing_profile[provisional_marker_key] = char_update[provisional_marker_key]
            
            if config.ENABLE_DYNAMIC_STATE_ADAPTATION and "modification_proposal" in char_update:
                apply_state_modification_proposal_logic(agent, existing_profile, char_update["modification_proposal"], char_name, "existing character profile")
            
            for key, value in char_update.items():
                if key in ["modification_proposal", provisional_marker_key]: continue 
                
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
                    if not (config.ENABLE_DYNAMIC_STATE_ADAPTATION and "modification_proposal" in char_update and "MODIFY DESCRIPTION" in char_update["modification_proposal"].upper()):
                        existing_profile["description"] = value 
                elif key == dev_key and isinstance(value, str) and value.strip():
                    existing_profile[key] = value 
                elif key == "status" and isinstance(value, str) and value.strip():
                    existing_profile["status"] = value 
                elif key not in existing_profile and value is not None: 
                    existing_profile[key] = value

    if updated_chars_count > 0 or new_chars_count > 0:
        logger.info(f"Character profile JSON merge complete for ch {chapter_number}. Updated: {updated_chars_count}, New: {new_chars_count}.")
    else:
        logger.info(f"No character profiles were effectively updated or added for ch {chapter_number} after LLM analysis.")

def robust_merge_world_item_data_logic(agent, target_dict: Dict[str, Any], update_dict: Dict[str, Any], item_name_for_log: str, chapter_num: int, from_flawed_draft_source: bool) -> Dict[str, Any]:
    """Recursively merges update_dict into target_dict for world items.
    'agent' is an instance of NovelWriterAgent.
    """
    current_item_data = target_dict.copy() if isinstance(target_dict, dict) else {}
    if not isinstance(target_dict, dict) and target_dict is not None: 
        current_item_data['description'] = str(target_dict) 
        logger.warning(f"World item '{item_name_for_log}' was not a dict. Converted to dict, old value saved as 'description'.")
    
    item_was_modified_this_call = False
    provisional_marker_key = f"source_quality_chapter_{chapter_num}"

    if provisional_marker_key in update_dict:
        current_item_data[provisional_marker_key] = update_dict[provisional_marker_key]
        item_was_modified_this_call = True 

    if config.ENABLE_DYNAMIC_STATE_ADAPTATION and "modification_proposal" in update_dict:
        proposal = update_dict.pop("modification_proposal") 
        if isinstance(proposal, str) and proposal.strip():
            apply_state_modification_proposal_logic(agent, current_item_data, proposal, item_name_for_log, "world item")
            item_was_modified_this_call = True
    
    for key, value_from_update in update_dict.items():
        if key in [provisional_marker_key, "modification_proposal"] or key.startswith(("updated_in_chapter_", "added_in_chapter_", "elaboration_in_chapter_")):
            if key.startswith("elaboration_in_chapter_") and isinstance(value_from_update, str) and value_from_update.strip():
                current_item_data[key] = value_from_update 
                item_was_modified_this_call = True
            continue

        current_value_in_target = current_item_data.get(key)

        if isinstance(value_from_update, dict):
            if not isinstance(current_value_in_target, dict):
                current_item_data[key] = {}
                item_was_modified_this_call = True 
            merged_sub_dict = robust_merge_world_item_data_logic(agent, current_item_data[key], value_from_update, f"{item_name_for_log}.{key}", chapter_num, from_flawed_draft_source)
            if merged_sub_dict != current_item_data[key]: 
                item_was_modified_this_call = True
            current_item_data[key] = merged_sub_dict
        elif isinstance(value_from_update, list):
            if not isinstance(current_value_in_target, list):
                current_item_data[key] = []
                item_was_modified_this_call = True 
            
            initial_list_len = len(current_item_data[key])
            for item_in_list_update in value_from_update:
                if item_in_list_update not in current_item_data[key]:
                    current_item_data[key].append(item_in_list_update)
            if len(current_item_data[key]) > initial_list_len:
                item_was_modified_this_call = True
        elif value_from_update != current_value_in_target: 
            current_item_data[key] = value_from_update
            item_was_modified_this_call = True
    
    if item_was_modified_this_call and not current_item_data.get(f"added_in_chapter_{chapter_num}"):
        current_item_data[f"updated_in_chapter_{chapter_num}"] = True
            
    return current_item_data

def merge_world_item_updates_logic(agent, updates_from_llm: Dict[str, Any], chapter_number: int, from_flawed_draft: bool):
    """Merges LLM-proposed world-building updates into the agent's world_building state.
    'agent' is an instance of NovelWriterAgent.
    """
    if not updates_from_llm:
        logger.info(f"No world-building updates from LLM to merge for ch {chapter_number}.")
        return

    logger.info(f"Merging world-building JSON updates for ch {chapter_number}. Categories in update: {list(updates_from_llm.keys())}")
    items_affected_count = 0
    provisional_marker_key = f"source_quality_chapter_{chapter_number}"

    for category_key, category_updates_dict in updates_from_llm.items():
        if not isinstance(category_updates_dict, dict) or not category_updates_dict:
            logger.debug(f"Skipping empty or invalid update for world category '{category_key}' in ch {chapter_number}.")
            continue
        
        if category_key not in agent.world_building:
            agent.world_building[category_key] = {}
        elif not isinstance(agent.world_building[category_key], dict): 
            logger.warning(f"Overwriting non-dictionary world category '{category_key}' with new dictionary structure.")
            agent.world_building[category_key] = {}
        
        target_category_dict = agent.world_building[category_key]
        
        for item_name, item_update_details in category_updates_dict.items():
            if not isinstance(item_update_details, dict):
                logger.warning(f"Skipping invalid item_details for '{item_name}' in category '{category_key}' (not a dict). Data: {item_update_details}")
                continue
            
            item_log_name = f"{category_key}.{item_name}"
            update_copy = item_update_details.copy() 

            if from_flawed_draft:
                update_copy[provisional_marker_key] = "provisional_from_unrevised_draft"
            
            existing_item_data = target_category_dict.get(item_name)
            
            if existing_item_data is None: 
                logger.info(f"Adding new world item '{item_log_name}' from ch {chapter_number} analysis.")
                new_item_data = robust_merge_world_item_data_logic(agent, {}, update_copy, item_log_name, chapter_number, from_flawed_draft)
                new_item_data[f"added_in_chapter_{chapter_number}"] = True 
                target_category_dict[item_name] = new_item_data
                items_affected_count +=1
            elif isinstance(existing_item_data, dict): 
                logger.debug(f"Updating existing world item '{item_log_name}' from ch {chapter_number} analysis.")
                updated_item_data = robust_merge_world_item_data_logic(agent, existing_item_data, update_copy, item_log_name, chapter_number, from_flawed_draft)
                target_category_dict[item_name] = updated_item_data
                if updated_item_data.get(f"updated_in_chapter_{chapter_number}") or update_copy.get(provisional_marker_key):
                    items_affected_count +=1
            else: 
                logger.warning(f"Existing world item '{item_log_name}' is not a dictionary. Overwriting with new data from ch {chapter_number}.")
                new_item_data = robust_merge_world_item_data_logic(agent, {}, update_copy, item_log_name, chapter_number, from_flawed_draft)
                new_item_data[f"added_in_chapter_{chapter_number}"] = True 
                target_category_dict[item_name] = new_item_data
                items_affected_count += 1
        
        if any(isinstance(v,dict) and (v.get(f"updated_in_chapter_{chapter_number}") or v.get(f"added_in_chapter_{chapter_number}")) 
               for v in target_category_dict.values()):
             target_category_dict[f"category_updated_in_chapter_{chapter_number}"] = True

    if items_affected_count > 0:
        logger.info(f"World-building JSON merge complete for ch {chapter_number}. Approximately {items_affected_count} items affected/added.")
    else:
        logger.info(f"No world-building JSON items were effectively updated or added for ch {chapter_number} after LLM analysis.")


async def update_json_profiles_from_chapter_logic(agent, chapter_text: Optional[str], chapter_number: int, from_flawed_draft: bool):
    """Updates character and world-building JSON files based on events in the chapter.
    'agent' is an instance of NovelWriterAgent.
    """
    if not chapter_text or len(chapter_text) < 100:
        logger.info(f"Skipping JSON knowledge update for ch {chapter_number}: Text too short or None.")
        return

    known_char_names = list(agent.character_profiles.keys())
    known_loc_names = list(agent.world_building.get("locations", {}).keys())
    
    text_lower = chapter_text.lower()
    mentioned_entities = [name for name in known_char_names if name.lower() in text_lower]
    mentioned_entities.extend(name for name in known_loc_names if name.lower() in text_lower)
    
    if not mentioned_entities and chapter_number > 3 : 
         logger.info(f"JSON knowledge update for ch {chapter_number}: No known characters or locations mentioned significantly (heuristic). Will still attempt LLM update if configured.")

    logger.info(f"Attempting combined JSON (character/world) update for ch {chapter_number} (Source from flawed draft: {from_flawed_draft}). Mentions found (heuristic): {mentioned_entities[:5]}")
    text_snippet = chapter_text[:config.KNOWLEDGE_UPDATE_SNIPPET_SIZE] 
    protagonist_name = agent.plot_outline.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME)
    
    dynamic_instr_char, dynamic_instr_world = "", ""
    if config.ENABLE_DYNAMIC_STATE_ADAPTATION:
        dynamic_instr_char = """For existing characters, if their traits, status, or core description needs modification based on THIS chapter's events, include a `"modification_proposal"` field. Example: `"modification_proposal": "MODIFY traits: ADD 'Determined', REMOVE 'Hesitant'"`. Only include characters that are updated, newly introduced, or have a modification proposal."""
        dynamic_instr_world = """For existing world items, if their properties need modification, include a `"modification_proposal"`. Example: `"modification_proposal": "MODIFY atmosphere: 'Now heavy with magical fallout'"`. Only include world elements (locations, society items, systems, lore, history) that are new, significantly changed by THIS chapter's events, or have a modification proposal."""
    else:
        dynamic_instr_char = "Only include characters whose information is directly updated or those newly introduced in THIS chapter."
        dynamic_instr_world = "Only include world elements that are new or significantly changed by THIS chapter's events."
        
    current_profiles_for_prompt = await get_filtered_character_profiles_for_prompt(agent, chapter_number - 1)
    current_world_for_prompt = await get_filtered_world_data_for_prompt(agent, chapter_number - 1)

    prompt = f"""/no_think
You are a meticulous literary analyst. Your task is to analyze the provided Chapter {chapter_number} Text Snippet (protagonist: {protagonist_name}) and identify updates for character profiles AND world-building details.
The output MUST be a single, valid JSON object with two top-level keys: "character_updates" and "world_building_updates".

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
2. For each such character, create an entry in the "character_updates" object (keyed by character name).
3. Each character entry should include relevant updated fields (e.g., "traits", "status", "description", "relationships").
4. Crucially, add a `development_in_chapter_{chapter_number}` key to each character entry, summarizing their role, actions, or significant changes in THIS chapter.
5. {dynamic_instr_char}
6. If no characters are updated or introduced, the value of "character_updates" should be an empty JSON object `{{}}`.

**Current World Building Notes (for reference - note 'prompt_notes' for provisional status):**
```json
{json.dumps(current_world_for_prompt, indent=2, ensure_ascii=False, default=str, sort_keys=True)}
```
**World Building Update Instructions:**
1. Identify new or significantly changed locations, societal elements (factions, cultures), systems (magic, tech), lore, or historical details revealed in THIS chapter snippet.
2. For each, create an entry under the appropriate category (e.g., "locations", "society") within the "world_building_updates" object.
3. Each world element entry should contain its updated details (e.g., "description", "atmosphere", "rules", "goals").
4. Add an `elaboration_in_chapter_{chapter_number}` key to each world element entry, providing context or specifics from THIS chapter.
5. {dynamic_instr_world}
6. If no world elements are updated or introduced, the relevant category (e.g., "locations") should be an empty JSON object `{{}}`, or the entire "world_building_updates" can be `{{}}`.

**CRITICAL: Output ONLY the combined JSON object as specified.**
Example Output Structure:
```json
{{
  "character_updates": {{
    "CharacterName": {{ 
      "description": "Updated description.", 
      "traits": ["NewTrait"], 
      "status": "Updated Status",
      "modification_proposal": "MODIFY traits: ADD 'Brave'", 
      "development_in_chapter_{chapter_number}": "They confronted the antagonist and revealed a new skill."
    }}
  }},
  "world_building_updates": {{
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
}}
```
"""
    raw_analysis = await llm_interface.async_call_llm(
        model_name=config.KNOWLEDGE_UPDATE_MODEL,
        prompt=prompt, 
        temperature=0.6 
    )
    combined_updates = await llm_interface.async_parse_llm_json_response(
        raw_analysis, f"combined character/world JSON update for ch {chapter_number}"
    )

    if not combined_updates or not isinstance(combined_updates, dict):
        logger.warning(f"LLM parsing for combined char/world JSON updates failed or returned no data for ch {chapter_number}. Raw LLM: {raw_analysis[:200] if raw_analysis else 'EMPTY'}")
        return

    char_updates = combined_updates.get("character_updates")
    if char_updates and isinstance(char_updates, dict):
        merge_character_profile_updates_logic(agent, char_updates, chapter_number, from_flawed_draft)
    else:
        logger.info(f"No 'character_updates' field found or it's not a dictionary in combined response for ch {chapter_number}.")

    world_updates = combined_updates.get("world_building_updates")
    if world_updates and isinstance(world_updates, dict):
        merge_world_item_updates_logic(agent, world_updates, chapter_number, from_flawed_draft)
    else:
        logger.info(f"No 'world_building_updates' field found or it's not a dictionary in combined response for ch {chapter_number}.")


@alru_cache(maxsize=config.KG_TRIPLE_EXTRACTION_CACHE_SIZE)
async def llm_extract_kg_triples_logic(agent, text_snippet_for_kg_key: str, chapter_number: int, candidate_entities_json_key: str) -> str:
    """Cached LLM call for KG triple extraction.
    'agent' is an instance of NovelWriterAgent.
    """
    protagonist_name = agent.plot_outline.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME)
    common_predicates = [
        "is_a", "located_in", "has_trait", "status_is", "feels", "knows", "believes", "wants", 
        "interacted_with", "travelled_to", "discovered", "acquired", "lost", "used_item", 
        "attacked", "helped", "damaged", "repaired", "contains", "part_of", "caused_by", 
        "leads_to", "observed", "heard", "said", "thought_about", "decided_to", "has_goal", 
        "has_feature", "related_to", "member_of", "leader_of", "enemy_of", "ally_of", 
        "works_for", "has_ability", "possesses", "created_by"
    ] 
    
    candidate_entities_prompt_section = ""
    if candidate_entities_json_key and candidate_entities_json_key != "[]": 
        candidate_entities_prompt_section = f"**Heuristically Identified Candidate Entities (Prioritize these for Subject/Object if relevant and present in the text snippet):**\n```json\n{candidate_entities_json_key}\n```\n"

    prompt = f"""/no_think
You are a Knowledge Graph Engineer. Your task is to extract factual (Subject, Predicate, Object) triples from the provided Text Snippet from Chapter {chapter_number} of a novel (protagonist: '{protagonist_name}').

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
`[["{protagonist_name}", "travelled_to", "Eclipse Spire"], ["Eclipse Spire", "is_a", "ancient ruin"], ["{protagonist_name}", "feels", "uneasy"]]`

JSON Output Only:
[
""" 
    return await llm_interface.async_call_llm(
        model_name=config.KNOWLEDGE_UPDATE_MODEL,
        prompt=prompt, 
        temperature=0.6, 
        max_tokens=config.MAX_KG_TRIPLE_TOKENS
    )

async def extract_and_store_kg_triples_logic(agent, chapter_text: Optional[str], chapter_number: int, from_flawed_draft: bool):
    """Extracts KG triples from chapter text and adds them to the database.
    'agent' is an instance of NovelWriterAgent.
    """
    if not chapter_text:
        logger.warning(f"Skipping KG extraction for ch {chapter_number}: Chapter text is None or empty.")
        return
            
    logger.info(f"Extracting KG triples for ch {chapter_number} (Source from flawed draft: {from_flawed_draft})...")
    
    text_snippet_for_kg = chapter_text[:config.KNOWLEDGE_UPDATE_SNIPPET_SIZE * 2].strip() 
    if len(text_snippet_for_kg) < len(chapter_text):
        logger.warning(f"KG extraction for ch {chapter_number} will use truncated text ({len(text_snippet_for_kg)} chars out of {len(chapter_text)}).")

    candidate_entities = await heuristic_entity_spotter_for_kg(agent, text_snippet_for_kg)
    logger.debug(f"Candidate entities identified for KG extraction in Ch {chapter_number}: {candidate_entities[:10]}")
    candidate_entities_json_for_prompt = json.dumps(candidate_entities) 

    raw_triples_json_str = await llm_extract_kg_triples_logic(
        agent, text_snippet_for_kg, chapter_number, candidate_entities_json_for_prompt
    )
            
    parsed_triples = await llm_interface.async_parse_llm_json_response(
        raw_triples_json_str, f"KG triple extraction for chapter {chapter_number}", expect_type=list
    )
    
    if parsed_triples is None: 
         logger.error(f"Failed to extract or parse any KG triples for ch {chapter_number} after all attempts. Raw LLM output: {raw_triples_json_str[:200] if raw_triples_json_str else 'EMPTY'}")
         await agent._save_debug_output(chapter_number, "kg_extraction_final_fail_raw_llm", raw_triples_json_str or "EMPTY_RAW_TRIPLES_JSON")
         return
    
    if not parsed_triples: 
        logger.info(f"No KG triples were extracted by the LLM for ch {chapter_number}.")
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
        await asyncio.gather(*kg_add_tasks) 
    
    logger.info(f"KG update for ch {chapter_number}: Attempted to add {added_count} triples, skipped {skipped_count}. (Source Provisional: {from_flawed_draft})")


async def prepopulate_kg_from_initial_data_logic(agent):
    """Pre-populates the Knowledge Graph from the initial plot outline and world-building data.
    'agent' is an instance of NovelWriterAgent.
    """
    logger.info("Starting Knowledge Graph pre-population from plot and world data...")

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
    
    combined_pruned_data = {"plot_summary": pruned_plot, "world_highlights": pruned_world}
    try:
        combined_data_json = json.dumps(combined_pruned_data, indent=2, ensure_ascii=False, default=str)
    except TypeError as e:
        logger.error(f"Error serializing pruned data for KG pre-population prompt: {e}")
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
        logger.error(f"Failed to extract/parse KG triples for pre-population after all attempts. Raw LLM: {raw_triples_json_str[:500] if raw_triples_json_str else 'EMPTY'}")
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
                    state_manager.async_add_kg_triple(subj, pred, obj, config.KG_PREPOPULATION_CHAPTER_NUM, is_provisional=False) 
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
    if added_count == 0 and parsed_triples: 
        logger.warning("KG pre-population resulted in 0 valid triples added despite LLM returning data. Check LLM output and parsing.")


async def update_all_knowledge_bases_logic(agent, chapter_number: int, final_text: str, from_flawed_draft: bool):
    """Updates JSON character/world profiles and the Knowledge Graph based on the finalized chapter.
    'agent' is an instance of NovelWriterAgent.
    """
    if not final_text:
        logger.warning(f"Skipping knowledge base update for ch {chapter_number}: Final text is missing or empty.")
        return
    logger.info(f"Updating knowledge bases for ch {chapter_number} (Source from flawed draft: {from_flawed_draft})...")
    
    update_json_task = await update_json_profiles_from_chapter_logic(agent, final_text, chapter_number, from_flawed_draft)
    update_kg_task = await extract_and_store_kg_triples_logic(agent, final_text, chapter_number, from_flawed_draft)
    
    try:
        await asyncio.gather(update_json_task, update_kg_task)
        logger.info(f"Knowledge base updates (JSON profiles & KG) completed for ch {chapter_number}.")
    except Exception as e:
        logger.error(f"Error during concurrent knowledge base update for ch {chapter_number}: {e}", exc_info=True)