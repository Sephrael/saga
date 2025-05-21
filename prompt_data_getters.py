# prompt_data_getters.py
"""
Helper functions to prepare specific data snippets for LLM prompts in the SAGA system.
These functions typically filter or format parts of the agent's state,
increasingly by querying Neo4j directly for richer, graph-aware context.
"""
import logging
import json
import re
import asyncio
from typing import Dict, List, Optional, Set, Tuple, Any, Iterable 

import config
from state_manager import state_manager
from type import JsonStateData, SceneDetail 

logger = logging.getLogger(__name__)

async def get_character_state_snippet_for_prompt(agent, current_chapter_num_for_filtering: Optional[int] = None) -> str:
    """
    Creates a concise JSON string of key character states for prompts,
    now fetching data directly from Neo4j.
    'agent' is an instance of NovelWriterAgent.
    """
    snippet_data: Dict[str, Dict[str, str]] = {}
    char_names_to_process: List[str] = []
    
    protagonist_name = agent.plot_outline.get("protagonist_name") 
    
    # Get character names from the agent's current in-memory profiles as the pool
    # This ensures we're only querying for characters the agent is currently "aware" of.
    # A full graph scan for all :Character nodes might be too broad for a snippet.
    all_known_char_names_from_agent_dict: List[str] = list(agent.character_profiles.keys())
    
    if protagonist_name and protagonist_name in all_known_char_names_from_agent_dict:
        char_names_to_process.append(protagonist_name)
    
    for name in sorted(all_known_char_names_from_agent_dict): 
        if name != protagonist_name and len(char_names_to_process) < config.PLANNING_CONTEXT_MAX_CHARACTERS_IN_SNIPPET:
            char_names_to_process.append(name)
        elif len(char_names_to_process) >= config.PLANNING_CONTEXT_MAX_CHARACTERS_IN_SNIPPET:
            break
        
    effective_filter_chapter = (current_chapter_num_for_filtering - 1) \
        if current_chapter_num_for_filtering is not None and current_chapter_num_for_filtering > 0 \
        else config.KG_PREPOPULATION_CHAPTER_NUM

    fetch_tasks = [
        state_manager.get_character_info_for_snippet(name, effective_filter_chapter) 
        for name in char_names_to_process
    ]
    
    if not fetch_tasks:
        return "No character profiles available or applicable to fetch for snippet."

    character_info_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

    for i, res_or_exc in enumerate(character_info_results):
        char_name_for_snippet = char_names_to_process[i]
        if isinstance(res_or_exc, Exception):
            logger.error(f"Error fetching snippet info for character '{char_name_for_snippet}' from Neo4j: {res_or_exc}")
            # Fallback to in-memory agent.character_profiles dict if Neo4j fails
            profile_fallback = agent.character_profiles.get(char_name_for_snippet, {})
            desc_fb = (profile_fallback.get("description", "Error fetching description.")[:config.PLANNING_CONTEXT_MAX_CHARS_PER_PROFILE_DESC]).strip() + "..."
            status_fb = profile_fallback.get("status", "Unknown (error)")
            dev_note_fb = "Error fetching development note."
            # Attempt to find a recent dev note from dict if possible for fallback
            dev_keys_fb = sorted([k for k in profile_fallback if k.startswith("development_in_chapter_") and int(k.split('_')[-1]) <= effective_filter_chapter], key=lambda x: int(x.split('_')[-1]), reverse=True)
            if dev_keys_fb: dev_note_fb = (profile_fallback.get(dev_keys_fb[0], "N/A")[:config.PLANNING_CONTEXT_MAX_RECENT_DEV_PER_PROFILE]).strip() + "..."
            
            snippet_data[char_name_for_snippet] = {
                "description_snippet": desc_fb,
                "current_status": status_fb,
                "most_recent_development_note": dev_note_fb
            }
            continue
        
        profile_summary = res_or_exc 
        if not profile_summary: # Neo4j returned None (e.g. character not found in detail query)
             # Fallback to in-memory agent.character_profiles dict
            profile_fallback = agent.character_profiles.get(char_name_for_snippet, {})
            desc_fb = (profile_fallback.get("description", "N/A")[:config.PLANNING_CONTEXT_MAX_CHARS_PER_PROFILE_DESC]).strip() + "..."
            status_fb = profile_fallback.get("status", "Unknown")
            dev_note_fb = "N/A"
            dev_keys_fb = sorted([k for k in profile_fallback if k.startswith("development_in_chapter_") and int(k.split('_')[-1]) <= effective_filter_chapter], key=lambda x: int(x.split('_')[-1]), reverse=True)
            if dev_keys_fb: dev_note_fb = (profile_fallback.get(dev_keys_fb[0], "N/A")[:config.PLANNING_CONTEXT_MAX_RECENT_DEV_PER_PROFILE]).strip() + "..."

            snippet_data[char_name_for_snippet] = {
                "description_snippet": desc_fb,
                "current_status": status_fb + " (Data potentially from agent memory)",
                "most_recent_development_note": dev_note_fb
            }
            logger.warning(f"Neo4j character snippet fetch for '{char_name_for_snippet}' was None, using agent memory as fallback.")
            continue


        provisional_note = " (Note: Some info may be provisional)" if profile_summary.get("is_provisional_overall") else ""
        
        snippet_data[char_name_for_snippet] = {
            "description_snippet": (profile_summary.get("description", "N/A")[:config.PLANNING_CONTEXT_MAX_CHARS_PER_PROFILE_DESC]).strip() + "...",
            "current_status": profile_summary.get("current_status", "Unknown") + provisional_note,
            "most_recent_development_note": (profile_summary.get("most_recent_development_note", "N/A")[:config.PLANNING_CONTEXT_MAX_RECENT_DEV_PER_PROFILE]).strip() + "..."
        }
            
    return json.dumps(snippet_data, indent=2, ensure_ascii=False, default=str) if snippet_data else "No character profiles available or applicable."


async def get_world_state_snippet_for_prompt(agent, current_chapter_num_for_filtering: Optional[int] = None) -> str:
    """
    Creates a concise JSON string of key world states for prompts,
    now fetching data directly from Neo4j.
    'agent' is an instance of NovelWriterAgent.
    """
    snippet_data: Dict[str, Any] = {}
    
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
                state_manager.get_world_elements_for_snippet(category_name, effective_filter_chapter, max_items)
            )
            category_names_for_tasks.append(category_name)
    
    if not fetch_tasks_world:
        return "No significant world-building data available or applicable to fetch for snippet."

    world_info_results = await asyncio.gather(*fetch_tasks_world, return_exceptions=True)
    
    any_provisional_in_world_snippet = False

    for i, res_or_exc in enumerate(world_info_results):
        category_name_result = category_names_for_tasks[i]
        
        if isinstance(res_or_exc, Exception):
            logger.error(f"Error fetching snippet info for world category '{category_name_result}' from Neo4j: {res_or_exc}")
            snippet_data[f"key_{category_name_result}_error"] = ["Error fetching data."]
            continue

        items_list = res_or_exc 
        if not items_list: # Neo4j returned empty list for this category
            # Fallback to in-memory agent.world_building for this category
            category_data_fallback = agent.world_building.get(category_name_result, {})
            item_snippets_fallback = []
            item_count_fb = 0
            for item_name_fb, item_details_fb in category_data_fallback.items():
                if item_count_fb >= world_categories_to_fetch[category_name_result]: break
                if item_name_fb.startswith(("_", "source_quality_chapter_", "category_updated_in_chapter_")): continue
                
                desc_snip_fb = ""
                if isinstance(item_details_fb, dict) and item_details_fb.get("description"):
                    desc_snip_fb = f": {str(item_details_fb['description'])[:50].strip()}..."
                item_snippets_fallback.append(f"{item_name_fb}{desc_snip_fb}")
                item_count_fb +=1
            if item_snippets_fallback:
                snippet_data[f"key_{category_name_result}_fallback"] = item_snippets_fallback
            logger.warning(f"Neo4j world snippet fetch for '{category_name_result}' empty, using agent memory as fallback.")
            continue
        
        item_display_list = []
        category_has_provisional = False
        for item_data in items_list:
            name = item_data.get("name", "Unknown Item")
            desc_snip = item_data.get("description_snippet", "") 
            item_display_list.append(f"{name}{desc_snip.replace(name, '').strip() if desc_snip.startswith(name) else (': ' + desc_snip if desc_snip else '')}")
            if item_data.get("is_provisional"):
                category_has_provisional = True
                any_provisional_in_world_snippet = True
        
        prov_note_cat = " (Note: Some items may be provisional)" if category_has_provisional else ""
        snippet_data[f"key_{category_name_result}{prov_note_cat}"] = item_display_list

    if any_provisional_in_world_snippet and "overall_world_note" not in snippet_data :
         snippet_data["overall_world_note"] = "Some world information might be provisional from unrevised prior chapters."
                
    return json.dumps(snippet_data, indent=2, ensure_ascii=False, default=str) if snippet_data else "No significant world-building data available or applicable."


async def get_filtered_character_profiles_for_prompt(agent, up_to_chapter_inclusive: Optional[int] = None) -> JsonStateData:
    """
    Retrieves character profiles by querying Neo4j and reassembles them into a dictionary.
    Adds 'prompt_notes' for provisional data up to a specified chapter.
    """
    logger.info(f"Fetching filtered character profiles from Neo4j up to chapter {up_to_chapter_inclusive}.")
    
    profiles_from_graph = await state_manager.get_character_profiles() 
    profiles_copy = json.loads(json.dumps(profiles_from_graph)) # Deep copy

    if up_to_chapter_inclusive is None: 
        return profiles_copy 

    for char_name, profile_data in profiles_copy.items():
        if not isinstance(profile_data, dict): continue 

        provisional_notes_for_char: List[str] = []
        # Check for `source_quality_chapter_X` properties if they are still being saved
        # on the Character node or reassembled into the dict by get_character_profiles.
        for i in range(1, up_to_chapter_inclusive + 1): 
            prov_key = f"source_quality_chapter_{i}"
            if profile_data.get(prov_key) == "provisional_from_unrevised_draft":
                provisional_notes_for_char.append(f"Information for this character updated in Chapter {i} was marked as provisional (derived from an unrevised draft).")
        
        provisional_rels_query = """
        MATCH (c:Character {name: $char_name})
        CALL {
            WITH c // c is now explicitly in scope for the subquery
            MATCH (c)-[r:DYNAMIC_REL {is_provisional: true}]->(o:Entity)
            WHERE r.chapter_added <= $chapter_limit
            RETURN r.type AS rel_type, r.chapter_added AS rel_chapter, o.name AS target_name
            UNION
            WITH c // c is now explicitly in scope for the subquery
            MATCH (s:Entity)-[r:DYNAMIC_REL {is_provisional: true}]->(c)
            WHERE r.chapter_added <= $chapter_limit
            RETURN r.type AS rel_type, r.chapter_added AS rel_chapter, s.name AS target_name
        }
        RETURN COLLECT(DISTINCT "Relationship type '" + rel_type + "' with '" + target_name + "' from chapter " + toString(rel_chapter) + " is provisional.") AS provisional_rel_notes
        LIMIT 1
        """
        try:
            params = {"char_name": char_name, "chapter_limit": up_to_chapter_inclusive}
            prov_rel_results = await state_manager._execute_read_query(provisional_rels_query, params)
            if prov_rel_results and prov_rel_results[0] and prov_rel_results[0].get("provisional_rel_notes"):
                provisional_notes_for_char.extend(prov_rel_results[0]["provisional_rel_notes"])
        except Exception as e:
            logger.error(f"Error querying provisional relationships for {char_name}: {e}")


        if provisional_notes_for_char:
            if "prompt_notes" not in profile_data: profile_data["prompt_notes"] = []
            # Ensure prompt_notes is a list (it might be None or other type from graph if schema varies)
            if not isinstance(profile_data["prompt_notes"], list): profile_data["prompt_notes"] = []
            
            unique_new_notes = set(provisional_notes_for_char) - set(profile_data["prompt_notes"])
            profile_data["prompt_notes"].extend(list(unique_new_notes))

    return profiles_copy


async def get_filtered_world_data_for_prompt(agent, up_to_chapter_inclusive: Optional[int] = None) -> JsonStateData:
    """
    Retrieves world building data by querying Neo4j and reassembles it.
    Adds 'prompt_notes' for provisional data up to a specified chapter.
    """
    logger.info(f"Fetching filtered world data from Neo4j up to chapter {up_to_chapter_inclusive}.")
    
    world_data_from_graph = await state_manager.get_world_building() 
    world_data_copy = json.loads(json.dumps(world_data_from_graph)) # Deep copy

    if up_to_chapter_inclusive is None:
        return world_data_copy

    for category_name, category_items in world_data_copy.items():
        if not isinstance(category_items, dict): continue
        
        for item_name, item_data in category_items.items():
            if not isinstance(item_data, dict): continue 
            
            item_provisional_notes: List[str] = []
            # Check for `source_quality_chapter_X` properties on the item_data dict
            for i in range(1, up_to_chapter_inclusive + 1):
                item_prov_key = f"source_quality_chapter_{i}"
                if item_data.get(item_prov_key) == "provisional_from_unrevised_draft":
                    item_provisional_notes.append(f"World item '{item_name}' ({category_name}) data from Ch {i} is provisional (unrevised source).")
            
            we_id = f"{category_name}_{item_name}".replace(" ", "_").replace("'", "").lower() 
            provisional_world_rels_query = """
            MATCH (we:WorldElement {id: $we_id}) 
            CALL {
                WITH we 
                MATCH (we)-[r:DYNAMIC_REL {is_provisional: true}]->(o:Entity)
                WHERE r.chapter_added <= $chapter_limit
                RETURN r.type AS rel_type, r.chapter_added AS rel_chapter, o.name AS target_name
                UNION
                WITH we 
                MATCH (s:Entity)-[r:DYNAMIC_REL {is_provisional: true}]->(we)
                WHERE r.chapter_added <= $chapter_limit
                RETURN r.type AS rel_type, r.chapter_added AS rel_chapter, s.name AS target_name
            }
            RETURN COLLECT(DISTINCT "Relationship type '" + rel_type + "' involving '" + we.name + "' ('" + target_name + "') from chapter " + toString(rel_chapter) + " is provisional.") AS provisional_rel_notes
            LIMIT 1
            """
            try:
                params = {"we_id": we_id, "chapter_limit": up_to_chapter_inclusive}
                prov_world_rel_results = await state_manager._execute_read_query(provisional_world_rels_query, params)
                if prov_world_rel_results and prov_world_rel_results[0] and prov_world_rel_results[0].get("provisional_rel_notes"):
                    item_provisional_notes.extend(prov_world_rel_results[0]["provisional_rel_notes"])
            except Exception as e:
                 logger.error(f"Error querying provisional relationships for world item {we_id} ({item_name}): {e}")

            if item_provisional_notes:
                if "prompt_notes" not in item_data: item_data["prompt_notes"] = []
                if not isinstance(item_data["prompt_notes"], list): item_data["prompt_notes"] = []
                
                unique_new_notes = set(item_provisional_notes) - set(item_data["prompt_notes"])
                item_data["prompt_notes"].extend(list(unique_new_notes))
    return world_data_copy


async def heuristic_entity_spotter_for_kg(agent, text_snippet: str) -> List[str]:
    """Basic heuristic to spot potential entities (proper nouns) in text, including known characters."""
    # Known character names from agent's in-memory dict (assumed to be reasonably up-to-date from Neo4j load)
    entities = set(agent.character_profiles.keys()) 
    
    for match in re.finditer(r'\b([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]+){0,2})\b', text_snippet):
        entities.add(match.group(1).strip())
    
    common_false_positives = {"The", "A", "An", "Is", "It", "He", "She", "They", "Chapter", "Section"} 
    
    return sorted([e for e in list(entities) 
                   if (len(e) > 2 or e in agent.character_profiles) and e not in common_false_positives])


async def get_reliable_kg_facts_for_drafting_prompt(
    agent, 
    chapter_number: int, 
    chapter_plan: Optional[List[SceneDetail]] = None,
    max_facts_per_char: int = 2, 
    max_total_facts: int = 7 
) -> str:
    """
    Fetches relevant, reliable (non-provisional) KG facts for the drafting prompt using Cypher.
    Focuses on characters involved in the plan and general novel context.
    """
    if chapter_number <= 0: return "No KG facts applicable for pre-first chapter."

    kg_chapter_limit = chapter_number - 1 
    facts_for_prompt: List[str] = []
    
    protagonist_name = agent.plot_outline.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME)
    novel_id = config.MAIN_NOVEL_INFO_NODE_ID # Use the constant ID for NovelInfo
    
    characters_of_interest: Set[str] = {protagonist_name}
    if chapter_plan and isinstance(chapter_plan, list):
        for scene_detail in chapter_plan:
            if isinstance(scene_detail, dict) and "characters_involved" in scene_detail and isinstance(scene_detail["characters_involved"], list):
                for char_name_in_plan in scene_detail["characters_involved"]:
                    if isinstance(char_name_in_plan, str) and char_name_in_plan.strip():
                        characters_of_interest.add(char_name_in_plan.strip())
    
    logger.debug(f"KG fact gathering for Ch {chapter_number} draft: Characters of interest: {characters_of_interest}")

    # 1. Get general novel context facts (Theme, Conflict Summary)
    novel_context_queries_params = [
        (f"MATCH (ni:NovelInfo {{id: $novelId}}) RETURN ni.theme AS value, 'The novel\\'s theme' AS description", {"novelId": novel_id}, "theme"),
        (f"MATCH (ni:NovelInfo {{id: $novelId}}) RETURN ni.conflict_summary AS value, 'The main conflict' AS description", {"novelId": novel_id}, "conflict")
    ]
    for query, params, desc_key in novel_context_queries_params:
        if len(facts_for_prompt) >= max_total_facts: break
        try:
            res = await state_manager._execute_read_query(query, params) 
            if res and res[0] and res[0].get('value'): 
                facts_for_prompt.append(f"- {res[0]['description']} is: {res[0]['value']}.")
        except Exception as e: 
            logger.warning(f"KG Query for novel context '{desc_key}' failed: {e}")

    # 2. Get facts about characters of interest
    for char_name in list(characters_of_interest)[:3]: 
        if len(facts_for_prompt) >= max_total_facts: break
        facts_for_this_char = 0
        
        # Fetch status directly from Character node property if it's stored there and reliable
        char_node_query = "MATCH (c:Character {name: $char_name}) RETURN c.status AS status_val, c.source_quality_chapter_0 AS initial_prov_status" # Assuming status_is not reliable KG fact
        # The `source_quality_chapter_0` would indicate if the initial status from prepopulation was provisional.
        # For this getter, we want *reliable* facts. So we need to ensure status isn't from a provisional source.
        # A more robust way: Check if status was set by a :DYNAMIC_REL that is NOT provisional.
        # Or, if status is a direct property, assume it's reliable if not explicitly marked provisional.
        # For simplicity here, if status is a direct node property, we assume it's reliable unless a specific "provisional_status" flag exists.
        # The current `async_get_most_recent_value` for "status_is" *already* filters by `is_provisional = FALSE`.
        
        status_val = await state_manager.async_get_most_recent_value(char_name, "status_is", kg_chapter_limit, include_provisional=False)
        if status_val and facts_for_this_char < max_facts_per_char and len(facts_for_prompt) < max_total_facts:
            facts_for_prompt.append(f"- {char_name}'s status is: {status_val}.")
            facts_for_this_char += 1
            
        loc_val = await state_manager.async_get_most_recent_value(char_name, "located_in", kg_chapter_limit, include_provisional=False)
        if loc_val and facts_for_this_char < max_facts_per_char and len(facts_for_prompt) < max_total_facts:
            facts_for_prompt.append(f"- {char_name} is located in: {loc_val}.")
            facts_for_this_char += 1
            
        if facts_for_this_char < max_facts_per_char and len(facts_for_prompt) < max_total_facts:
            ally_query = """
            MATCH (c:Entity {name: $char_name})-[r:DYNAMIC_REL {type: 'ally_of'}]->(ally:Entity)
            WHERE r.chapter_added <= $chapter_limit AND r.is_provisional = FALSE
            RETURN ally.name AS ally_name ORDER BY r.chapter_added DESC, r.confidence DESC LIMIT 1 
            """ 
            try:
                ally_res = await state_manager._execute_read_query(ally_query, {"char_name": char_name, "chapter_limit": kg_chapter_limit})
                if ally_res and ally_res[0] and ally_res[0].get('ally_name'):
                    facts_for_prompt.append(f"- {char_name} is an ally of: {ally_res[0]['ally_name']}.")
                    facts_for_this_char +=1
            except Exception as e:
                logger.warning(f"KG Query for '{char_name}' ally failed: {e}")
    
    if not facts_for_prompt:
        return "No specific reliable KG facts identified as highly relevant for this chapter's current focus from Neo4j."
        
    unique_facts = sorted(list(set(facts_for_prompt)))

    return "**Key Reliable KG Facts (from Neo4j - up to previous chapter):**\n" + "\n".join(unique_facts[:max_total_facts])