# prompt_data_getters.py
"""
Helper functions to prepare specific data snippets for LLM prompts in the SAGA system.
These functions typically filter or format parts of the agent's state.
"""
import logging
import json
import re
import asyncio
import builtins # Import the builtins module
from typing import Dict, List, Optional, Set, Tuple # Added Set and Tuple

import config
from state_manager import state_manager
from type import JsonStateData, SceneDetail # Added SceneDetail

logger = logging.getLogger(__name__)

def get_character_state_snippet_for_prompt(agent, current_chapter_num_for_filtering: Optional[int] = None) -> str:
    """Creates a concise JSON string of key character states for prompts.
    'agent' is an instance of NovelWriterAgent.
    """
    snippet_data: Dict[str, Dict[str, str]] = {}
    char_count = 0
    
    protagonist_name = agent.plot_outline.get("protagonist_name")
    sorted_char_names: List[str] = []
    if protagonist_name and protagonist_name in agent.character_profiles:
        sorted_char_names.append(protagonist_name)
    
    # Use builtins.sorted here as well if standard sorted causes issues,
    # though the Pylance error was not reported for this specific line.
    # For consistency and to be safe, one might change it, but let's stick to reported lines.
    for name in sorted(agent.character_profiles.keys()): 
        if name != protagonist_name:
            sorted_char_names.append(name)
        
    effective_filter_chapter = (current_chapter_num_for_filtering - 1) \
        if current_chapter_num_for_filtering is not None and current_chapter_num_for_filtering > 0 \
        else config.KG_PREPOPULATION_CHAPTER_NUM

    for name in sorted_char_names:
        if char_count >= config.PLANNING_CONTEXT_MAX_CHARACTERS_IN_SNIPPET:
            break
        
        profile = agent.character_profiles.get(name, {})
        if not isinstance(profile, dict): continue 

        provisional_note = ""
        # Line 72 (original): Use builtins.any
        if builtins.any(key.startswith("source_quality_chapter_") and 
               int(key.split('_')[-1]) <= effective_filter_chapter and
               profile.get(key) == "provisional_from_unrevised_draft"
               for key in profile):
             provisional_note = " (Note: Some info may be provisional based on unrevised prior chapters)"

        # Line 78 (original): Use builtins.sorted
        dev_notes_keys = builtins.sorted(
            [k for k in profile if k.startswith("development_in_chapter_") and int(k.split('_')[-1]) <= effective_filter_chapter], 
            key=lambda x: int(x.split('_')[-1]), 
            reverse=True
        )
        recent_dev_note_text = profile.get(dev_notes_keys[0], "N/A") if dev_notes_keys else "No specific development notes prior to this chapter."
        
        snippet_data[name] = {
            "description_snippet": profile.get("description", "No description available.")[:config.PLANNING_CONTEXT_MAX_CHARS_PER_PROFILE_DESC].strip() + "...",
            "current_status": profile.get("status", "Unknown") + provisional_note,
            "most_recent_development_note": recent_dev_note_text[:config.PLANNING_CONTEXT_MAX_RECENT_DEV_PER_PROFILE].strip() + "..."
        }
        char_count += 1
            
    return json.dumps(snippet_data, indent=2, ensure_ascii=False, default=str) if snippet_data else "No character profiles available or applicable."

def get_world_state_snippet_for_prompt(agent, current_chapter_num_for_filtering: Optional[int] = None) -> str:
    """Creates a concise JSON string of key world states for prompts.
    'agent' is an instance of NovelWriterAgent.
    """
    snippet_data: Dict[str, any] = {}
    
    effective_filter_chapter = (current_chapter_num_for_filtering - 1) \
        if current_chapter_num_for_filtering is not None and current_chapter_num_for_filtering > 0 \
        else config.KG_PREPOPULATION_CHAPTER_NUM

    def get_provisional_note_for_category(category_dict: Dict[str, any], chapter_limit: int) -> str:
        # If Pylance errors occurred with 'any' here, would apply builtins.any too.
        if builtins.any(key.startswith("source_quality_chapter_") and 
               int(key.split('_')[-1]) <= chapter_limit and
               category_dict.get(key) == "provisional_from_unrevised_draft"
               for key in category_dict): # Assuming category_dict can have such keys directly
             return " (Note: Some category info may be provisional)"
        
        for item_data in category_dict.values():
            if isinstance(item_data, dict) and \
               builtins.any(key.startswith("source_quality_chapter_") and 
                   int(key.split('_')[-1]) <= chapter_limit and
                   item_data.get(key) == "provisional_from_unrevised_draft"
                   for key in item_data): # Assuming item_data can have such keys
                return " (Note: Some items within this category may have provisional info)"
        return ""

    world_categories_for_snippet = {
        "locations": config.PLANNING_CONTEXT_MAX_LOCATIONS_IN_SNIPPET,
        "systems": config.PLANNING_CONTEXT_MAX_SYSTEMS_IN_SNIPPET,
    }
    
    for category_name, max_items in world_categories_for_snippet.items():
        category_data = agent.world_building.get(category_name, {})
        if isinstance(category_data, dict) and category_data:
            prov_note = get_provisional_note_for_category(category_data, effective_filter_chapter)
            item_snippets = []
            for item_name, item_details in list(category_data.items())[:max_items]:
                if item_name.startswith(("source_quality_chapter_", "category_updated_in_chapter_")): continue 
                desc_snippet = ""
                if isinstance(item_details, dict) and item_details.get("description"):
                    desc_snippet = f": {str(item_details['description'])[:50].strip()}..."
                item_snippets.append(f"{item_name}{desc_snippet}")

            if item_snippets:
                snippet_data[f"key_{category_name}{prov_note}"] = item_snippets
    
    society_data = agent.world_building.get("society", {})
    if isinstance(society_data, dict):
        factions_data = society_data.get("Key Factions", society_data.get("factions", {})) 
        if isinstance(factions_data, dict) and factions_data:
            prov_note_factions = get_provisional_note_for_category(factions_data, effective_filter_chapter)
            faction_names = [name for name in list(factions_data.keys()) if not name.startswith("source_quality_chapter_")][:config.PLANNING_CONTEXT_MAX_FACTIONS_IN_SNIPPET]
            if faction_names:
                 snippet_data[f"key_factions{prov_note_factions}"] = faction_names
                 
    return json.dumps(snippet_data, indent=2, ensure_ascii=False, default=str) if snippet_data else "No significant world-building data available or applicable."

async def get_filtered_character_profiles_for_prompt(agent, up_to_chapter_inclusive: Optional[int] = None) -> JsonStateData:
    """Retrieves character profiles from the ORM and adds 'prompt_notes' for provisional data up to a chapter.
    'agent' is an instance of NovelWriterAgent.
    """
    profiles = agent.character_profiles 
    profiles_copy = json.loads(json.dumps(profiles)) 

    if up_to_chapter_inclusive is None: 
        return profiles_copy 

    for char_name, profile_data in profiles_copy.items():
        if not isinstance(profile_data, dict): continue 

        provisional_notes_for_char: List[str] = []
        for i in range(1, up_to_chapter_inclusive + 1): 
            prov_key = f"source_quality_chapter_{i}"
            if profile_data.get(prov_key) == "provisional_from_unrevised_draft":
                provisional_notes_for_char.append(f"Information for this character updated in Chapter {i} was marked as provisional (derived from an unrevised draft).")
    
        if provisional_notes_for_char:
            if "prompt_notes" not in profile_data: profile_data["prompt_notes"] = []
            if not isinstance(profile_data["prompt_notes"], list): profile_data["prompt_notes"] = []
            for note in provisional_notes_for_char:
                if note not in profile_data["prompt_notes"]:
                    profile_data["prompt_notes"].append(note)
    return profiles_copy


async def get_filtered_world_data_for_prompt(agent, up_to_chapter_inclusive: Optional[int] = None) -> JsonStateData:
    """Retrieves world building data from the ORM and adds 'prompt_notes' for provisional data up to a chapter.
    'agent' is an instance of NovelWriterAgent.
    """
    world_data = agent.world_building
    world_data_copy = json.loads(json.dumps(world_data))

    if up_to_chapter_inclusive is None:
        return world_data_copy

    for category_name, category_items in world_data_copy.items():
        if not isinstance(category_items, dict): continue

        category_provisional_notes: List[str] = []
        for i in range(1, up_to_chapter_inclusive + 1):
            cat_prov_key = f"source_quality_chapter_{i}" 
            if category_items.get(cat_prov_key) == "provisional_from_unrevised_draft":
                category_provisional_notes.append(f"The category '{category_name}' had information updated in Chapter {i} marked as provisional.")
        
        if category_provisional_notes:
            if "prompt_notes" not in category_items: category_items["prompt_notes"] = []
            if not isinstance(category_items["prompt_notes"], list): category_items["prompt_notes"] = []
            for note in category_provisional_notes:
                if note not in category_items["prompt_notes"]:
                    category_items["prompt_notes"].append(note)

        for item_name, item_data in category_items.items():
            if not isinstance(item_data, dict): continue 
            
            item_provisional_notes: List[str] = []
            for i in range(1, up_to_chapter_inclusive + 1):
                item_prov_key = f"source_quality_chapter_{i}"
                if item_data.get(item_prov_key) == "provisional_from_unrevised_draft":
                    item_provisional_notes.append(f"The world item '{item_name}' (category: '{category_name}') had information updated in Chapter {i} marked as provisional.")
            
            if item_provisional_notes:
                if "prompt_notes" not in item_data: item_data["prompt_notes"] = []
                if not isinstance(item_data["prompt_notes"], list): item_data["prompt_notes"] = []
                for note in item_provisional_notes:
                    if note not in item_data["prompt_notes"]:
                        item_data["prompt_notes"].append(note)
    return world_data_copy


async def heuristic_entity_spotter_for_kg(agent, text_snippet: str) -> List[str]:
    """Basic heuristic to spot potential entities (proper nouns) in text, including known characters.
    'agent' is an instance of NovelWriterAgent.
    """
    entities = set(agent.character_profiles.keys()) 
    
    for match in re.finditer(r'\b([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]+){0,2})\b', text_snippet):
        entities.add(match.group(1).strip())
    
    common_false_positives = {"The", "A", "An", "Is", "It", "He", "She", "They"} 
    
    # If sorted() was causing Pylance issues elsewhere, builtins.sorted could be used here too.
    return sorted([e for e in list(entities) 
                   if (len(e) > 3 or e in agent.character_profiles) and e not in common_false_positives])

async def get_reliable_kg_facts_for_drafting_prompt(
    agent, 
    chapter_number: int, 
    chapter_plan: Optional[List[SceneDetail]] = None,
    max_facts: int = 5 # Max number of distinct facts to return
) -> str:
    """
    Fetches a few highly relevant, reliable (non-provisional) KG facts for the drafting prompt.
    Focuses on the protagonist and other key characters potentially involved in the chapter.
    'agent' is an instance of NovelWriterAgent.
    """
    if chapter_number <= 0: return "No KG facts applicable for pre-first chapter."

    kg_chapter_limit = chapter_number - 1
    facts_for_prompt: List[str] = []
    
    protagonist_name = agent.plot_outline.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME)
    
    characters_of_interest: Set[str] = {protagonist_name}
    if chapter_plan and isinstance(chapter_plan, list):
        for scene_detail in chapter_plan:
            if isinstance(scene_detail, dict) and "characters_involved" in scene_detail and isinstance(scene_detail["characters_involved"], list):
                for char_name_in_plan in scene_detail["characters_involved"]: # Renamed to avoid conflict
                    if isinstance(char_name_in_plan, str) and char_name_in_plan.strip():
                        characters_of_interest.add(char_name_in_plan.strip())
    
    logger.debug(f"KG fact gathering for Ch {chapter_number} draft: Characters of interest: {characters_of_interest}")

    char_predicates = ["located_in", "status_is", "has_goal", "feels"] 

    # Store tasks and their associated metadata (origin_char, origin_pred)
    task_metadata_list: List[Tuple[str, str]] = []
    coroutines_to_gather = []

    for char_name in list(characters_of_interest)[:3]: 
        for pred in char_predicates:
            coroutines_to_gather.append(
                state_manager.async_get_most_recent_value(
                    char_name, pred, kg_chapter_limit, include_provisional=False
                )
            )
            task_metadata_list.append((char_name, pred)) # Store metadata

    if coroutines_to_gather:
        kg_results = await asyncio.gather(*coroutines_to_gather, return_exceptions=True) # Added return_exceptions
        
        for i, value_or_exception in enumerate(kg_results):
            if isinstance(value_or_exception, Exception):
                logger.error(f"Error fetching KG value for {task_metadata_list[i]}: {value_or_exception}")
                continue # Skip this result if there was an error

            value = value_or_exception # Now it's the actual value
            if value: # value can be None if not found, or the string if found
                origin_char, origin_pred = task_metadata_list[i]
                fact_str = f"- {origin_char} {origin_pred.replace('_', ' ')}: {value}."
                if fact_str not in facts_for_prompt: 
                    facts_for_prompt.append(fact_str)
                if len(facts_for_prompt) >= max_facts:
                    break
    
    if not facts_for_prompt:
        return "No specific reliable KG facts identified for key characters in this chapter."
        
    return "**Key Reliable KG Facts (from pre-novel & previous reliable chapter states):**\n" + "\n".join(facts_for_prompt)