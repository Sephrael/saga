# novel_logic.py
"""
Contains the core logic for the novel generation agent, including state management,
chapter writing, analysis, revision, and knowledge updates.
Integrates a knowledge graph (KG) for improved consistency and context.
Designed to be run asynchronously.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Copyright 2025 Dennis Lewis
"""

import os
import json
import re
import numpy as np
import logging
import random
from typing import Dict, List, Optional, Tuple, Any, TypedDict, Union
import asyncio
from async_lru import alru_cache

import config
import utils
import llm_interface
from database_manager import DatabaseManager

logger = logging.getLogger(__name__)

# Type Hinting
class EvaluationResult(TypedDict):
    needs_revision: bool
    reasons: List[str]
    coherence_score: Optional[float]
    consistency_issues: Optional[str]
    plot_deviation_reason: Optional[str]

class SceneDetail(TypedDict):
    scene_number: int
    summary: str
    characters_involved: List[str]
    key_dialogue_points: List[str]
    setting_details: str
    contribution: str # How this scene contributes to the chapter/plot

JsonStateData = Dict[str, Any]

class NovelWriterAgent:
    """
    Manages the state and orchestrates the asynchronous process of generating a novel.
    """

    def __init__(self):
        logger.info("Initializing NovelWriterAgent...")
        self.db_manager = DatabaseManager(config.DATABASE_FILE)
        self.plot_outline: JsonStateData = {}
        self.character_profiles: JsonStateData = {}
        self.world_building: JsonStateData = {}
        self.chapter_count: int = 0
        self._load_initial_state_sync() # Synchronous loading in constructor
        logger.info(f"NovelWriterAgent initialized. Current chapter count: {self.chapter_count}")

    def _load_json_file(self, file_path: str, attribute_name: str) -> JsonStateData:
        """Loads a single JSON file, handling errors and returning a dictionary."""
        data: JsonStateData = {}
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                if isinstance(loaded_data, dict):
                    data = loaded_data
                    logger.info(f"Successfully loaded {attribute_name.replace('_', ' ')} from {file_path}")
                else:
                    logger.warning(f"{file_path} content is not a dictionary. Ignoring and using empty data.")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON from {file_path}: {e}. Using empty data.", exc_info=True)
            except Exception as e:
                logger.error(f"Unexpected error loading {file_path}: {e}. Using empty data.", exc_info=True)
        else:
            logger.info(f"No {attribute_name.replace('_', ' ')} file found ('{file_path}'). Using empty data.")
        return data

    def _load_initial_state_sync(self):
        """Loads existing agent state synchronously. Called from __init__."""
        logger.info("Attempting to load existing agent state...")
        self.chapter_count = self.db_manager.load_chapter_count() # Sync DB call
        logger.info(f"Loaded chapter count from database: {self.chapter_count}")

        self.plot_outline = self._load_json_file(config.PLOT_OUTLINE_FILE, "plot_outline")
        self.character_profiles = self._load_json_file(config.CHARACTER_PROFILES_FILE, "character_profiles")
        self.world_building = self._load_json_file(config.WORLD_BUILDER_FILE, "world_building")
        
        logger.info("Finished loading initial state.")

    def _save_single_json_state_sync(self, file_path: str, data_dict: JsonStateData, dict_name: str):
        """Synchronous helper to save a single JSON state file."""
        if not data_dict or not isinstance(data_dict, dict):
            logger.debug(f"Skipping save for {file_path}, data empty or not a dict.")
            return False

        data_to_save = json.loads(json.dumps(data_dict)) # Deep copy for manipulation

        # Logic to conditionally remove "is_default" flag
        is_default_flag_value = data_to_save.get("is_default", False)
        is_content_truly_default = False
        if file_path == config.PLOT_OUTLINE_FILE:
            is_content_truly_default = (
                data_to_save.get("title") == config.DEFAULT_PLOT_OUTLINE_TITLE and
                len(data_to_save.get("plot_points", [])) <= 5 and # Allows for default 5 plot points
                data_to_save.get("protagonist_name") == config.DEFAULT_PROTAGONIST_NAME
            )
        elif file_path == config.CHARACTER_PROFILES_FILE:
            is_content_truly_default = (
                not data_to_save or (
                    len(data_to_save) == 1 and config.DEFAULT_PROTAGONIST_NAME in data_to_save and
                    data_to_save[config.DEFAULT_PROTAGONIST_NAME].get("description", "").startswith("Default:")
                )
            )
        elif file_path == config.WORLD_BUILDER_FILE:
            locations = data_to_save.get("locations", {})
            is_content_truly_default = (
                len(locations) <= 1 and # Allow 0 or 1 default location
                ("Default Location" in locations if locations else True) and
                len(data_to_save.keys()) <= 3 # "locations", "society", "is_default" or similar
            )

        if is_default_flag_value and not is_content_truly_default:
            if "is_default" in data_to_save:
                del data_to_save["is_default"]
                logger.debug(f"'is_default' flag removed from {dict_name} as content is no longer default.")
        elif not is_default_flag_value and "is_default" in data_to_save:
             del data_to_save["is_default"] # Remove if false, it's implied
             logger.debug(f"'is_default: false' flag removed from {dict_name}.")
        elif is_default_flag_value and is_content_truly_default:
            logger.debug(f"'is_default: true' retained for {dict_name} as content appears default.")
        
        # Always remove is_default if it's False, it's implied. Only save if True and actually default.
        if "is_default" in data_to_save and not data_to_save["is_default"]:
            del data_to_save["is_default"]

        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved JSON state for {dict_name} to {file_path}.")
            return True
        except Exception as e:
            logger.error(f"Failed to save JSON to {file_path}: {e}", exc_info=True)
            return False

    async def _save_all_json_state(self):
        """Asynchronously saves all JSON state files (plot, characters, world)."""
        logger.debug("Saving agent JSON state (plot, characters, world)...")
        loop = asyncio.get_event_loop()
        
        # Use run_in_executor for each synchronous file save operation
        # These can run "concurrently" in threads if multiple files are saved.
        tasks = [
            loop.run_in_executor(None, self._save_single_json_state_sync, config.PLOT_OUTLINE_FILE, self.plot_outline, "plot_outline"),
            loop.run_in_executor(None, self._save_single_json_state_sync, config.CHARACTER_PROFILES_FILE, self.character_profiles, "character_profiles"),
            loop.run_in_executor(None, self._save_single_json_state_sync, config.WORLD_BUILDER_FILE, self.world_building, "world_building"),
        ]
        results = await asyncio.gather(*tasks)
        saved_count = sum(1 for r in results if r) # Count successful saves

        if saved_count > 0:
            logger.info(f"JSON state saved for {saved_count} file(s).")
        else:
            logger.info("No JSON state files were updated/saved.")

    async def generate_plot_outline(self, default_protagonist_name: str, unhinged_mode: bool, **kwargs) -> JsonStateData:
        """Generates a new plot outline using an LLM."""
        logger.info(f"Generating plot outline. Unhinged mode: {unhinged_mode}")
        
        base_elements_for_outline: Dict[str, Any] = {}
        if unhinged_mode:
            genre = kwargs.get("genre", random.choice(config.UNHINGED_GENRES))
            theme = kwargs.get("theme", random.choice(config.UNHINGED_THEMES))
            setting_archetype = kwargs.get("setting_archetype", random.choice(config.UNHINGED_SETTINGS_ARCHETYPES))
            protagonist_archetype = kwargs.get("protagonist_archetype", random.choice(config.UNHINGED_PROTAGONIST_ARCHETYPES))
            conflict_archetype = kwargs.get("conflict_archetype", random.choice(config.UNHINGED_CONFLICT_TYPES))
            
            prompt_core_elements = f"""
The novel is a '{genre}' story. Its central theme is '{theme}'.
The primary setting is inspired by: '{setting_archetype}'.
The protagonist is an archetype of: '{protagonist_archetype}'.
The main conflict revolves around: '{conflict_archetype}'.
Based on this combination, generate the following JSON fields:
1. `title`: A compelling title for the novel.
2. `protagonist_name`: A suitable name for the protagonist.
3. `protagonist_description`: A brief (1-2 sentences) description of the protagonist.
4. `setting`: A brief (1-2 sentences) description of the primary setting, expanding on the archetype.
5. `conflict`: A brief (1-2 sentences) summary of the main conflict.
6. `plot_points`: A JSON list of exactly 5 strings, representing major plot points from beginning to end.
7. `character_arc`: A string describing the protagonist's primary development arc through the story.
"""
            base_elements_for_outline = {
                "genre": genre, "theme": theme, 
                "setting_archetype_used": setting_archetype,
                "protagonist_archetype_used": protagonist_archetype,
                "conflict_archetype_used": conflict_archetype
            }
            required_keys = ["title", "protagonist_name", "protagonist_description", "setting", "conflict", "plot_points", "character_arc"]
        else: 
            genre = kwargs.get("genre", config.CONFIGURED_GENRE)
            theme = kwargs.get("theme", config.CONFIGURED_THEME)
            setting_description = kwargs.get("setting_description", config.CONFIGURED_SETTING_DESCRIPTION)
            prompt_core_elements = f"""
The novel is a '{genre}' story. Its central theme is '{theme}'.
The primary setting is: '{setting_description}'.
Based on these, generate the following JSON fields:
1. `title`: A compelling title for the novel.
2. `protagonist_name`: A suitable name for the protagonist (consider using '{default_protagonist_name}' or a variant if appropriate).
3. `protagonist_description`: A brief (1-2 sentences) description of the protagonist.
4. `plot_points`: A JSON list of exactly 5 strings, representing major plot points from beginning to end.
5. `character_arc`: A string describing the protagonist's primary development arc through the story.
6. `conflict`: A string summarizing the main conflict that drives the plot.
""" # Note: 'setting' is implicitly defined for standard mode from `setting_description`.
            base_elements_for_outline = {"genre": genre, "theme": theme, "setting": setting_description}
            required_keys = ["title", "protagonist_name", "protagonist_description", "plot_points", "character_arc", "conflict"]

        prompt = f"""/no_think
You are a creative assistant specializing in narrative structure generation.
{prompt_core_elements}
Output ONLY the JSON object. Ensure the response is a single, valid JSON.
The `plot_points` field must be a JSON list containing exactly 5 string elements.
Example of expected JSON structure (keys might vary slightly based on mode):
{{
  "title": "string",
  "protagonist_name": "string",
  "protagonist_description": "string",
  "setting": "string (only if unhinged mode, otherwise inferred)", 
  "conflict": "string",
  "plot_points": ["string1", "string2", "string3", "string4", "string5"],
  "character_arc": "string"
}}
"""
        logger.info("Calling LLM for plot outline generation...")
        raw_outline_str = await llm_interface.async_call_llm(
            model_name=config.INITIAL_SETUP_MODEL,
            prompt=prompt, 
            temperature=0.6 # Higher temperature for creativity
        )
        parsed_outline = await llm_interface.async_parse_llm_json_response(raw_outline_str, "plot outline generation")

        is_valid = False
        if parsed_outline and isinstance(parsed_outline, dict):
            plot_points = parsed_outline.get("plot_points")
            # Check all required keys are present and non-empty strings (or list for plot_points)
            if (all(key in parsed_outline and isinstance(parsed_outline[key], str) and parsed_outline[key].strip()
                    for key in required_keys if key != "plot_points") and
                isinstance(plot_points, list) and len(plot_points) == 5 and
                all(isinstance(p, str) and p.strip() for p in plot_points)):
                is_valid = True
            else:
                missing_or_invalid = [
                    key for key in required_keys 
                    if key not in parsed_outline or 
                       (key != "plot_points" and (not isinstance(parsed_outline[key], str) or not parsed_outline[key].strip())) or
                       (key == "plot_points" and (not isinstance(parsed_outline.get("plot_points"), list) or
                                                  len(parsed_outline.get("plot_points", [])) != 5 or
                                                  not all(isinstance(p, str) and p.strip() for p in parsed_outline.get("plot_points", []))))
                ]
                logger.warning(f"Generated plot outline failed validation. Missing/invalid keys: {missing_or_invalid}. Parsed: {parsed_outline}")

        if is_valid and isinstance(parsed_outline, dict): # Type guard for mypy
            self.plot_outline = parsed_outline
            self.plot_outline.update(base_elements_for_outline)
            # 'is_default' will be handled by _save_all_json_state based on content
            logger.info(f"Successfully generated plot outline: '{self.plot_outline.get('title', 'N/A')}'")
        else:
            logger.error("Failed to generate a valid plot outline after LLM call and parsing. Applying default.")
            self.plot_outline = {
                "title": config.DEFAULT_PLOT_OUTLINE_TITLE,
                "protagonist_name": default_protagonist_name,
                "protagonist_description": f"Default protagonist: {default_protagonist_name}, a character facing challenges.",
                "plot_points": [f"Default Plot Point {i+1}: An event occurs." for i in range(5)],
                "character_arc": f"Default character arc: {default_protagonist_name} learns something important.",
                "setting": base_elements_for_outline.get("setting", "A generic place."),
                "conflict": "Default conflict: The protagonist must overcome a significant obstacle.",
                "is_default": True # Explicitly mark as default
            }
            self.plot_outline.update(base_elements_for_outline) # Ensure genre/theme are still set
        
        self.plot_outline.setdefault('protagonist_name', default_protagonist_name) # Ensure protagonist name is always set
        await self._save_all_json_state()
        return self.plot_outline

    async def generate_world_building(self) -> JsonStateData:
        """Generates initial world-building data based on the plot outline."""
        if self.world_building and not self.world_building.get("is_default", False): # is_default might not exist yet
             # A more robust check: if it has substantial keys beyond just a default marker
            if len(self.world_building.keys() - {"is_default"}) > 1 or \
               ("locations" in self.world_building and len(self.world_building["locations"]) > 1):
                logger.info("Skipping initial world-building: Data appears to be already populated and non-default.")
                return self.world_building

        if not self.plot_outline or not self.plot_outline.get("setting"):
            logger.error("Cannot generate world-building: Plot outline or setting description is missing. Applying default world-building.")
            self.world_building = {
                "locations": {"Default Location": {"description": "A starting point for the story."}},
                "society": {"General Norms": {"description": "Basic societal structures and norms."}},
                "is_default": True
            }
            await self._save_all_json_state()
            return self.world_building

        prompt = f"""/no_think
You are a world-building assistant. Based on the provided novel concept, generate foundational world-building elements.
The output MUST be a single, valid JSON object.

Novel Concept:
Title: {self.plot_outline.get('title', 'Untitled Novel')}
Genre: {self.plot_outline.get('genre', 'Not specified')}
Theme: {self.plot_outline.get('theme', 'Not specified')}
Setting Description (expand significantly on this): {self.plot_outline.get('setting', 'A default setting')}
Main Conflict: {self.plot_outline.get('conflict', 'A central conflict')}
Protagonist: {self.plot_outline.get('protagonist_name', 'N/A')} ({self.plot_outline.get('protagonist_description', 'N/A')})

Instructions:
1. Create detailed world-building elements. Focus on providing tangible details that can be used in the story.
2. Significantly expand on the provided setting description.
3. Structure the output JSON with top-level keys: "locations", "society", "systems" (e.g., technology, magic), "lore", and "history".
4. Under each top-level key, create sub-dictionaries where each key is the name of a specific element (e.g., a city name under "locations", a faction name under "society").
5. Each specific element's dictionary should at least contain a "description" field. Add other relevant fields as appropriate (e.g., "atmosphere" for locations, "goals" for factions, "rules" for systems).
6. Be creative and imaginative, aligning with the genre and theme.

**CRITICAL: Output ONLY the JSON object.**
Example Structure:
{{
  "locations": {{
    "Capital City": {{ "description": "The bustling heart of the kingdom...", "atmosphere": "Oppressive and gray" }},
    "Forbidden Forest": {{ "description": "An ancient forest no one dares enter..." }}
  }},
  "society": {{
    "Royal Guard": {{ "description": "The elite protectors of the throne.", "goals": ["Protect royalty"] }}
  }},
  "systems": {{
    "Aetheric Magic": {{ "description": "Magic drawn from the ambient aether.", "rules": ["Requires focus", "Weakens with distance"] }}
  }},
  "lore": {{
    "The Great Sundering": {{ "description": "A cataclysmic event that shaped the current world." }}
  }},
  "history": {{
    "Founding Era": {{ "description": "The period when the major kingdoms were established." }}
  }}
}}
"""
        logger.info("Generating initial world-building data via LLM...")
        raw_world_data_str = await llm_interface.async_call_llm(
            model_name=config.INITIAL_SETUP_MODEL,
            prompt=prompt,
            temperature=0.6 # Moderately creative
        )
        parsed_world_data = await llm_interface.async_parse_llm_json_response(raw_world_data_str, "initial world-building")

        is_valid = False
        if parsed_world_data and isinstance(parsed_world_data, dict):
            # Check for presence of at least one main category with some content
            expected_categories = ["locations", "society", "systems", "lore", "history"]
            if any(cat in parsed_world_data and isinstance(parsed_world_data[cat], dict) and parsed_world_data[cat] 
                   for cat in expected_categories):
                self.world_building = parsed_world_data
                # 'is_default' will be handled by _save_all_json_state
                logger.info("Successfully generated initial world-building data.")
                is_valid = True
            else:
                logger.warning(f"Generated world-building lacks expected structure or content. Parsed: {parsed_world_data}")
        
        if not is_valid:
            logger.error("Failed to generate valid world-building data. Applying default.")
            self.world_building = {
                "locations": {"Default Location": {"description": "A starting point."}},
                "society": {"General": {"description": "Basic societal norms."}},
                "is_default": True
            }
        await self._save_all_json_state()
        return self.world_building

    async def _plan_chapter(self, chapter_number: int) -> Optional[List[SceneDetail]]:
        """Asynchronously plans a chapter with detailed scenes if agentic planning is enabled."""
        if not config.ENABLE_AGENTIC_PLANNING:
            logger.info(f"Agentic planning disabled by configuration. Skipping detailed planning for Chapter {chapter_number}.")
            return None # Return None to indicate no detailed plan, drafting will rely on plot point focus.

        logger.info(f"Planning Chapter {chapter_number} with detailed scenes...")
        plot_point_focus, plot_point_index = self._get_plot_point_info(chapter_number)
        if plot_point_focus is None: # This should ideally not happen if plot_points are managed well
            logger.error(f"Cannot plan chapter {chapter_number}: No plot point focus available.")
            return None

        context_summary = ""
        if chapter_number > 1:
            prev_chap_data = await self.db_manager.async_get_chapter_data_from_db(chapter_number - 1)
            if prev_chap_data:
                prev_summary = prev_chap_data.get('summary')
                prev_is_provisional = prev_chap_data.get('is_provisional', False)
                summary_prefix = "[Provisional Summary from Prev Ch] " if prev_is_provisional and prev_summary else "[Summary from Prev Ch] "
                if prev_summary: 
                    context_summary += f"{summary_prefix}({chapter_number - 1}):\n{prev_summary[:1000].strip()}...\n"
                else: # Fallback to text snippet if summary is missing
                    prev_text = prev_chap_data.get('text', '')
                    text_prefix = "[Provisional Text Snippet from Prev Ch] " if prev_is_provisional and prev_text else "[Text Snippet from Prev Ch] "
                    if prev_text: 
                        context_summary += f"{text_prefix}({chapter_number - 1}):\n...{prev_text[-1000:].strip()}\n"
        
        protagonist_name = self.plot_outline.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME)
        kg_chapter_limit = chapter_number - 1 # Facts up to the end of the previous chapter
        
        # Gather KG facts concurrently
        kg_tasks = {
            "location": self.db_manager.async_get_most_recent_value(protagonist_name, "located_in", kg_chapter_limit, include_provisional=False),
            "status": self.db_manager.async_get_most_recent_value(protagonist_name, "status_is", kg_chapter_limit, include_provisional=False)
        }
        # Add more KG queries here if needed for planning context
        
        kg_results = await asyncio.gather(*kg_tasks.values())
        kg_fact_map = dict(zip(kg_tasks.keys(), kg_results))

        kg_facts_for_prompt: List[str] = []
        if kg_fact_map.get("location"): kg_facts_for_prompt.append(f"- {protagonist_name} is currently located in (reliable KG): {kg_fact_map['location']}.")
        if kg_fact_map.get("status"): kg_facts_for_prompt.append(f"- {protagonist_name}'s current status (reliable KG): {kg_fact_map['status']}.")
        
        kg_context_section = "**Relevant Reliable KG Facts (up to prev chapter/pre-novel):**\n" + "\n".join(kg_facts_for_prompt) + "\n" if kg_facts_for_prompt else ""

        prompt = f"""/no_think
You are a master plotter outlining **between 8 and 15 detailed scenes** for Chapter {chapter_number} of a novel.
**Novel Concept:**
  - Title: {self.plot_outline.get('title', 'Untitled')}
  - Genre: {self.plot_outline.get('genre', 'N/A')}
  - Theme: {self.plot_outline.get('theme', 'N/A')}
  - Protagonist: {protagonist_name}
  - Protagonist's Arc: {self.plot_outline.get('character_arc', 'N/A')}

**Mandatory Focus for THIS Chapter (Plot Point {plot_point_index + 1} of {len(self.plot_outline.get('plot_points',[]))}):**
{plot_point_focus}

**Recent Context from Previous Chapter(s):**
{context_summary if context_summary else "This is the first chapter, or no prior summary is available."}
{kg_context_section}
**Current Character States (Key Characters, based on profiles and recent developments):**
{self._get_relevant_character_state_snippet(chapter_number)} 

**Current World State (Relevant Locations/Elements, based on world-building):**
{self._get_relevant_world_state_snippet(chapter_number)}

**Task:**
Create a detailed plan of 8 to 15 scenes for Chapter {chapter_number}. Each scene description in the plan MUST:
1. Directly advance or build towards the **Mandatory Focus** for this chapter.
2. Logically follow from the **Recent Context** and any provided **KG Facts**.
3. Involve relevant characters and world elements as appropriate.
4. Contribute to the **Protagonist's Arc** or the overall plot progression.
5. Be distinct from other scenes in this chapter plan.

**Output Format:**
Provide ONLY a single, valid JSON list of scene objects. Each object in the list must have the following keys:
  - `scene_number` (int): Sequential number for the scene within this chapter.
  - `summary` (str): A concise 1-2 sentence summary of what happens in the scene.
  - `characters_involved` (list[str]): A list of key character names involved in this scene.
  - `key_dialogue_points` (list[str]): 1-3 brief points outlining crucial dialogue or internal monologue.
  - `setting_details` (str): Brief description of the specific setting/location for this scene.
  - `contribution` (str): A short explanation of how this scene contributes to the chapter's goals or plot.

**Example JSON Scene (this would be one item in the list):**
```json
{{
  "scene_number": 1,
  "summary": "The protagonist, {protagonist_name}, discovers a cryptic message hidden in their family's old locket.",
  "characters_involved": ["{protagonist_name}"],
  "key_dialogue_points": ["'What could this symbol mean?'", "A sudden realization about a past event."],
  "setting_details": "A dusty, forgotten attic in the protagonist's ancestral home, late at night.",
  "contribution": "Serves as the inciting incident for this chapter's mystery/quest."
}}
```
Output the JSON list `[...]` directly. Do not include any other text, markdown, or explanation.
[
""" # The trailing [ is a common technique to guide LLMs to start a list.
        logger.info(f"Calling LLM ({config.PLANNING_MODEL}) for detailed scene plan for chapter {chapter_number}...")
        plan_raw = await llm_interface.async_call_llm(
            model_name=config.PLANNING_MODEL,
            prompt=prompt, 
            temperature=0.6, 
            max_tokens=config.MAX_PLANNING_TOKENS
        )
        
        parsed_plan: Optional[List[SceneDetail]] = await llm_interface.async_parse_llm_json_response(
            plan_raw, f"detailed scene plan for chapter {chapter_number}", expect_type=list
        )

        if parsed_plan and isinstance(parsed_plan, list) and len(parsed_plan) >= 1: # Min 1 scene
            valid_scenes: List[SceneDetail] = []
            required_scene_keys = {"scene_number", "summary", "characters_involved", "key_dialogue_points", "setting_details", "contribution"}
            for i, scene_item_any in enumerate(parsed_plan):
                if not isinstance(scene_item_any, dict):
                    logger.warning(f"Scene item {i+1} in plan for ch {chapter_number} is not a dict. Skipping. Item: {scene_item_any}")
                    continue
                
                scene_item = scene_item_any # Now known to be a dict
                
                # Validate structure and types
                if not required_scene_keys.issubset(scene_item.keys()):
                    logger.warning(f"Scene {i+1} in plan for ch {chapter_number} has missing keys ({required_scene_keys - set(scene_item.keys())}). Skipping.")
                    continue
                if not (isinstance(scene_item.get("scene_number"), int) and
                        isinstance(scene_item.get("summary"), str) and scene_item.get("summary", "").strip() and
                        isinstance(scene_item.get("characters_involved"), list) and
                        isinstance(scene_item.get("key_dialogue_points"), list) and
                        isinstance(scene_item.get("setting_details"), str) and scene_item.get("setting_details", "").strip() and
                        isinstance(scene_item.get("contribution"), str) and scene_item.get("contribution", "").strip()):
                    logger.warning(f"Scene {i+1} in plan for ch {chapter_number} has invalid types or empty required strings. Skipping. Scene: {scene_item}")
                    continue
                
                # Ensure list elements are strings for characters_involved and key_dialogue_points
                if not all(isinstance(c, str) for c in scene_item["characters_involved"]):
                     logger.warning(f"Scene {i+1} 'characters_involved' contains non-strings. Skipping. Scene: {scene_item}")
                     continue
                if not all(isinstance(d, str) for d in scene_item["key_dialogue_points"]):
                     logger.warning(f"Scene {i+1} 'key_dialogue_points' contains non-strings. Skipping. Scene: {scene_item}")
                     continue

                valid_scenes.append(scene_item) # type: ignore # SceneDetail type compatibility
            
            if valid_scenes and len(valid_scenes) >= 1: # Re-check after validation
                logger.info(f"Generated valid detailed scene plan for chapter {chapter_number} with {len(valid_scenes)} scenes.")
                return valid_scenes
            else:
                logger.error(f"All parsed scenes were invalid for chapter {chapter_number}. Raw LLM output: '{plan_raw[:500]}...'")
                await self._save_debug_output(chapter_number, "detailed_plan_invalid_scenes", plan_raw)
                return None
        else:
            logger.error(f"Failed to generate/parse a valid JSON list for scene plan for chapter {chapter_number}. Raw LLM output: '{plan_raw[:500]}...'")
            await self._save_debug_output(chapter_number, "detailed_plan_parse_fail", plan_raw)
            return None

    async def write_chapter(self, chapter_number: int) -> Optional[str]:
        """Orchestrates the full asynchronous process of writing a single chapter."""
        logger.info(f"=== Starting Chapter {chapter_number} Generation ===")
        if not (self.plot_outline and 
                self.plot_outline.get("plot_points") and 
                self.plot_outline.get("protagonist_name")):
            logger.error(f"Cannot write Ch {chapter_number}: Plot outline, plot points, or protagonist name missing.")
            return None
        if chapter_number <= 0: 
            logger.error(f"Cannot write Ch {chapter_number}: Chapter number must be positive.")
            return None

        chapter_plan: Optional[List[SceneDetail]] = await self._plan_chapter(chapter_number)
        
        # If planning failed and it's enabled, we might halt or proceed with caution.
        # For now, if chapter_plan is None, _generate_draft will rely on plot_point_focus.
        if config.ENABLE_AGENTIC_PLANNING and chapter_plan is None:
            logger.warning(f"Ch {chapter_number}: Agentic planning was enabled but failed to produce a plan. Proceeding with plot point focus only.")
            # Optionally, one could decide to halt here if a plan is strictly required.

        context_for_draft = await self._get_context(chapter_number)
        plot_point_focus, _ = self._get_plot_point_info(chapter_number)
        if plot_point_focus is None: # Should be caught earlier, but good to double check
            logger.error(f"Ch {chapter_number} generation halted: no plot point focus.")
            return None

        initial_draft_text, initial_raw_llm_text = await self._generate_draft(
            chapter_number, plot_point_focus, context_for_draft, chapter_plan
        )

        if not initial_draft_text:
            logger.error(f"Failed to generate initial draft for ch {chapter_number}.")
            await self._save_debug_output(chapter_number, "initial_draft_fail_raw_llm", initial_raw_llm_text or "")
            return None

        evaluation = await self._evaluate_draft(initial_draft_text, chapter_number, context_for_draft)
        current_text = initial_draft_text
        final_raw_output_log = f"--- INITIAL DRAFT (RAW LLM OUTPUT) ---\n{initial_raw_llm_text}\n\n"
        proceeded_with_flaws = False 

        if evaluation["needs_revision"]:
            revision_reason_str = "\n- ".join(evaluation["reasons"])
            logger.warning(f"Ch {chapter_number} flagged for revision. Reason(s):\n- {revision_reason_str}")
            
            revised_text_tuple = await self._revise_chapter(
                current_text, chapter_number, revision_reason_str, context_for_draft, chapter_plan
            )
            if revised_text_tuple:
                revised_text, raw_revision_llm_output = revised_text_tuple
                logger.info(f"Revision successful for ch {chapter_number}. Re-evaluating revised draft...")
                revised_evaluation = await self._evaluate_draft(revised_text, chapter_number, context_for_draft)
                
                if revised_evaluation["needs_revision"]:
                    logger.error(f"Revised draft for ch {chapter_number} STILL FAILED evaluation. Reasons:\n- " + "\n- ".join(revised_evaluation["reasons"]))
                    proceeded_with_flaws = True # Still proceed, but mark as flawed
                else:
                    logger.info(f"Revised draft for ch {chapter_number} passed re-evaluation.")
                
                current_text = revised_text 
                final_raw_output_log += f"--- REVISION (Reason: {revision_reason_str}) (RAW LLM OUTPUT) ---\n{raw_revision_llm_output}\n\n"
            else: # Revision attempt failed (e.g., LLM error, too similar)
                logger.error(f"Revision attempt failed for ch {chapter_number}. Proceeding with the original (flawed) draft.")
                proceeded_with_flaws = True
                final_raw_output_log += f"--- REVISION ATTEMPT FAILED (Reason: {revision_reason_str}) ---\n\n"
        else:
            logger.info(f"Initial draft for ch {chapter_number} passed evaluation.")

        # Finalize and update knowledge bases
        if not await self._finalize_chapter_core(chapter_number, current_text, final_raw_output_log, proceeded_with_flaws):
             logger.error(f"=== Finished Ch {chapter_number} WITH ERRORS during core finalization (DB/File save) ===")
             return None # Do not proceed to knowledge updates if core finalization failed

        await self._update_knowledge_bases(chapter_number, current_text, proceeded_with_flaws)
        
        self.chapter_count = max(self.chapter_count, chapter_number) # Update in-memory count
        await self._save_all_json_state() # Save potentially updated plot/char/world JSONs
        
        status_message = "Successfully" if not proceeded_with_flaws else "With Flaws (Accepted after failed revision or due to issues)"
        logger.info(f"=== Finished Ch {chapter_number} {status_message} ===")
        return current_text

    def _get_plot_point_info(self, chapter_number: int) -> Tuple[Optional[str], int]:
        """Retrieves the plot point focus for a given chapter number."""
        plot_points = self.plot_outline.get("plot_points", [])
        if not isinstance(plot_points, list) or not plot_points:
            logger.warning(f"No plot points defined in plot outline for chapter {chapter_number}.")
            return None, -1
        
        if chapter_number <= 0: 
            logger.warning(f"Invalid chapter number {chapter_number} for plot point lookup.")
            return None, -1
        
        # Plot points are 0-indexed, chapter numbers are 1-indexed.
        # Simple mapping: Chapter 1 -> plot_points[0], Chapter 2 -> plot_points[1], etc.
        # If chapter_number exceeds number of plot points, use the last one.
        plot_point_index = min(chapter_number - 1, len(plot_points) - 1)
        
        if 0 <= plot_point_index < len(plot_points):
            return plot_points[plot_point_index], plot_point_index
        
        logger.warning(f"Could not determine plot point for chapter {chapter_number} from {len(plot_points)} available points.")
        return None, -1 # Should not be reached if logic above is correct

    async def _generate_draft(self, chapter_number: int, plot_point_focus: Optional[str], context: str, chapter_plan: Optional[List[SceneDetail]]) -> Tuple[Optional[str], Optional[str]]:
        """Generates the initial draft text for a chapter."""
        if not plot_point_focus: # Should ideally always be set by _get_plot_point_info
            plot_point_focus = "Continue the narrative logically, focusing on character development and plot progression based on previous events."
            logger.warning(f"Plot point focus was None for Ch {chapter_number} draft generation. Using generic fallback.")
        
        plan_section_for_prompt = ""
        if config.ENABLE_AGENTIC_PLANNING:
            if chapter_plan and isinstance(chapter_plan, list): 
                try:
                    plan_json_str = json.dumps(chapter_plan, indent=2, ensure_ascii=False)
                    plan_section_for_prompt = f"**Detailed Scene Plan (MUST BE FOLLOWED CLOSELY):**\n```json\n{plan_json_str}\n```\n"
                    logger.info(f"Using detailed scene plan for Ch {chapter_number} draft generation.")
                except TypeError as e:
                    logger.error(f"Could not serialize chapter plan to JSON for prompt: {e}. Plan: {chapter_plan}")
                    plan_section_for_prompt = f"**Chapter Plan Note:** Error formatting plan. Rely on Plot Point Focus.\n**Plot Point Focus for THIS Chapter:** {plot_point_focus}\n"
            else: # No plan, or planning disabled and plan is None
                plan_section_for_prompt = f"**Chapter Plan Note:** No detailed scene plan available. Rely on the Overall Plot Point Focus.\n**Overall Plot Point Focus for THIS Chapter:** {plot_point_focus}\n"
        else: # Planning disabled
            plan_section_for_prompt = f"**Chapter Plan Note:** Detailed agentic planning is disabled. Rely on the Overall Plot Point Focus.\n**Overall Plot Point Focus for THIS Chapter:** {plot_point_focus}\n"
            
        # Filter profiles and world state up to the *previous* chapter for context.
        char_profiles_json = json.dumps(self._get_filtered_profiles_for_prompt(chapter_number - 1), indent=2, ensure_ascii=False, default=str)
        world_building_json = json.dumps(self._get_filtered_world_for_prompt(chapter_number - 1), indent=2, ensure_ascii=False, default=str)

        prompt = f"""/no_think
You are an expert novelist tasked with writing Chapter {chapter_number} of the novel titled "{self.plot_outline.get('title', 'Untitled Novel')}".
**Story Bible / Core Information:**
  - Genre: {self.plot_outline.get('genre', 'N/A')}
  - Central Theme: {self.plot_outline.get('theme', 'N/A')}
  - Protagonist: {self.plot_outline.get('protagonist_name', 'N/A')}
  - Protagonist's Character Arc: {self.plot_outline.get('character_arc', 'N/A')}

{plan_section_for_prompt}
**World Building Notes (JSON format - pay attention to any 'prompt_notes' indicating provisional data from previous unrevised chapters):**
```json
{world_building_json}
```
**Character Profiles (JSON format - pay attention to any 'prompt_notes' indicating provisional data from previous unrevised chapters):**
```json
{char_profiles_json}
```
**Context from Previous Chapters (Summaries/Snippets - note any 'Provisional' markers):**
--- BEGIN CONTEXT ---
{context if context.strip() else "No previous context (e.g., this is Chapter 1 or context retrieval failed)."}
--- END CONTEXT ---

**Writing Instructions:**
1. Write a compelling and engaging chapter, aiming for at least {config.MIN_ACCEPTABLE_DRAFT_LENGTH} characters.
2. If a **Detailed Scene Plan** is provided, adhere to it closely, fleshing out each scene.
3. If no detailed plan is available, focus on achieving the **Overall Plot Point Focus** for this chapter.
4. Maintain consistency with all provided information (Story Bible, World Building, Character Profiles, Previous Context).
5. Ensure a smooth narrative flow and vivid prose suitable for the genre '{self.plot_outline.get('genre', 'story')}'.
6. **Output ONLY the chapter text itself.** Do NOT include "Chapter X" headers, titles, author commentary, or any meta-discussion.

--- BEGIN CHAPTER {chapter_number} TEXT ---
"""
        raw_llm_text = await llm_interface.async_call_llm(
            model_name=config.DRAFTING_MODEL,
            prompt=prompt, 
            temperature=0.6 # Balanced temperature for drafting
        )
        if not raw_llm_text:
            logger.error(f"LLM returned no content for Ch {chapter_number} draft.")
            return None, None # Return None for both cleaned and raw text
            
        cleaned_text = llm_interface.clean_model_response(raw_llm_text)
        if not cleaned_text or len(cleaned_text) < config.MIN_ACCEPTABLE_DRAFT_LENGTH:
             logger.error(f"Ch {chapter_number} draft is too short ({len(cleaned_text or '')} chars) after cleaning. Min required: {config.MIN_ACCEPTABLE_DRAFT_LENGTH}. Raw LLM output snippet: '{raw_llm_text[:200]}...'")
             return None, raw_llm_text # Return None for cleaned, but provide raw for debugging
             
        logger.info(f"Generated initial draft for ch {chapter_number} (Length: {len(cleaned_text)} chars).")
        return cleaned_text, raw_llm_text

    async def _evaluate_draft(self, draft_text: str, chapter_number: int, previous_chapters_context: str) -> EvaluationResult:
        """Evaluates a chapter draft for coherence, consistency, and plot arc alignment."""
        logger.info(f"Evaluating chapter {chapter_number} draft (length: {len(draft_text)} chars)...")
        
        reasons: List[str] = []
        needs_revision = False
        coherence_score: Optional[float] = None
        consistency_issues_str: Optional[str] = None # Renamed from consistency_issues
        plot_deviation_reason_str: Optional[str] = None # Renamed from plot_deviation_reason

        # 1. Check minimum length
        if len(draft_text) < config.MIN_ACCEPTABLE_DRAFT_LENGTH:
            needs_revision = True
            reasons.append(f"Draft is too short ({len(draft_text)} chars). Minimum required: {config.MIN_ACCEPTABLE_DRAFT_LENGTH}.")

        # 2. Coherence Check (if not Chapter 1)
        # Start embedding fetch early, can run concurrently with other checks
        current_embedding_task = llm_interface.async_get_embedding(draft_text)
        
        if chapter_number > 1:
            prev_embedding = await self.db_manager.async_get_embedding_from_db(chapter_number - 1)
            current_embedding = await current_embedding_task # Await here as prev_embedding is needed

            if current_embedding is not None and prev_embedding is not None:
                coherence_score = utils.numpy_cosine_similarity(current_embedding, prev_embedding)
                logger.info(f"Coherence score with previous chapter ({chapter_number-1}): {coherence_score:.4f}")
                if coherence_score < config.REVISION_COHERENCE_THRESHOLD:
                    needs_revision = True
                    reasons.append(f"Low coherence with previous chapter (Score: {coherence_score:.4f}, Threshold: {config.REVISION_COHERENCE_THRESHOLD}).")
            else:
                logger.warning(f"Could not perform coherence check for ch {chapter_number} (missing current or previous embedding).")
        else: 
            logger.info("Skipping coherence check for Chapter 1.")
            await current_embedding_task # Ensure it completes even if not used for coherence

        # 3. Consistency and Plot Arc Validation (can run these concurrently)
        eval_tasks = []
        if config.REVISION_CONSISTENCY_TRIGGER:
            eval_tasks.append(self._check_consistency(draft_text, chapter_number, previous_chapters_context))
        else: # Placeholder if disabled
            eval_tasks.append(asyncio.sleep(0, result=None)) 

        if config.PLOT_ARC_VALIDATION_TRIGGER:
            eval_tasks.append(self._validate_plot_arc(draft_text, chapter_number))
        else: # Placeholder if disabled
            eval_tasks.append(asyncio.sleep(0, result=None))
            
        # Await results of consistency and plot arc checks
        consistency_result, plot_arc_result = await asyncio.gather(*eval_tasks)

        if config.REVISION_CONSISTENCY_TRIGGER and consistency_result:
            consistency_issues_str = consistency_result
            needs_revision = True
            reasons.append(f"Consistency issues identified:\n{consistency_issues_str}")
        
        if config.PLOT_ARC_VALIDATION_TRIGGER and plot_arc_result:
            plot_deviation_reason_str = plot_arc_result
            needs_revision = True
            reasons.append(f"Plot Arc Deviation: {plot_deviation_reason_str}")
            
        logger.info(f"Evaluation for Ch {chapter_number} complete. Needs revision: {needs_revision}.")
        return {
            "needs_revision": needs_revision, 
            "reasons": reasons, 
            "coherence_score": coherence_score, 
            "consistency_issues": consistency_issues_str, 
            "plot_deviation_reason": plot_deviation_reason_str
        }

    async def _revise_chapter(self, original_text: str, chapter_number: int, revision_reason: str, context_from_previous: str, chapter_plan: Optional[List[SceneDetail]]) -> Optional[Tuple[str, str]]:
        """Attempts to revise a chapter based on evaluation feedback."""
        if not original_text or not revision_reason:
            logger.error(f"Revision for ch {chapter_number} cannot proceed: missing original text or revision reason.")
            return None
        
        clean_reason = llm_interface.clean_model_response(revision_reason).strip()
        if not clean_reason:
            logger.error(f"Revision reason for ch {chapter_number} is empty after cleaning. Cannot proceed with revision.")
            return None
            
        logger.warning(f"Attempting revision for chapter {chapter_number}. Reason(s):\n{clean_reason}")
        
        # Truncate context and original text for prompt to avoid excessive length
        context_limit = config.MAX_CONTEXT_LENGTH // 4  # Allow more space for original text and plan
        original_text_limit = config.MAX_CONTEXT_LENGTH // 2
        
        context_snippet = context_from_previous[:context_limit].strip() + ("..." if len(context_from_previous) > context_limit else "")
        original_snippet = original_text[:original_text_limit].strip() + ("..." if len(original_text) > original_text_limit else "")
        
        plan_focus_section = ""
        plot_point_focus, _ = self._get_plot_point_info(chapter_number) # Always get plot point focus

        if config.ENABLE_AGENTIC_PLANNING and chapter_plan:
            try:
                plan_json_str = json.dumps(chapter_plan, indent=2, ensure_ascii=False)
                plan_snippet_for_prompt = plan_json_str[:(config.MAX_CONTEXT_LENGTH // 4)] # Max 1/4 for plan
                if len(plan_json_str) > len(plan_snippet_for_prompt):
                    plan_snippet_for_prompt += "\n... (plan truncated)"
                plan_focus_section = f"**Original Detailed Scene Plan (Target - align with this while fixing issues):**\n```json\n{plan_snippet_for_prompt}\n```\n"
            except TypeError: # Should not happen if chapter_plan is List[SceneDetail]
                 plan_focus_section = f"**Original Chapter Focus (Target):**\n{plot_point_focus or 'Not specified.'}\n"
        else: # Planning disabled or no plan available
            plan_focus_section = f"**Original Chapter Focus (Target):**\n{plot_point_focus or 'Not specified.'}\n"
            
        protagonist_name = self.plot_outline.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME)
        prompt = f"""/no_think
You are a skilled revising author tasked with rewriting Chapter {chapter_number} of a novel featuring protagonist {protagonist_name}.
**Critique/Reason(s) for Revision (These issues MUST be addressed comprehensively):**
--- FEEDBACK START ---
{clean_reason}
--- FEEDBACK END ---

{plan_focus_section}
**Context from Previous Chapters (for flow and continuity):**
--- BEGIN CONTEXT ---
{context_snippet if context_snippet else "No previous context (e.g., Chapter 1)."}
--- END CONTEXT ---

**Original Draft Snippet (for reference ONLY - your main goal is to address the critique and align with the plan/focus):**
--- BEGIN ORIGINAL DRAFT SNIPPET ---
{original_snippet}
--- END ORIGINAL DRAFT SNIPPET ---

**Revision Instructions:**
1. **PRIORITY:** Thoroughly address all issues listed in the **Critique/Reason(s) for Revision**.
2. **Rewrite the ENTIRE chapter text.** Do not just patch the original.
3. Align the rewritten chapter with the **Original Detailed Scene Plan** (if provided) or the **Original Chapter Focus**.
4. Ensure the revised chapter flows smoothly with the **Context from Previous Chapters**.
5. Maintain the established tone, style, and genre ('{self.plot_outline.get('genre', 'story')}') of the novel.
6. The revised chapter should be substantial, aiming for at least {config.MIN_ACCEPTABLE_DRAFT_LENGTH} characters.
7. **Output ONLY the rewritten chapter text.** No "Chapter X" headers, titles, or meta-commentary.

--- BEGIN REVISED CHAPTER {chapter_number} TEXT ---
"""
        revised_raw_llm_output = await llm_interface.async_call_llm(
            model_name=config.REVISION_MODEL,
            prompt=prompt, 
            temperature=0.6 # Slightly lower temp for more controlled revision
        ) 
        if not revised_raw_llm_output:
            logger.error(f"Revision LLM call failed for ch {chapter_number} (returned empty).")
            return None
            
        revised_cleaned_text = llm_interface.clean_model_response(revised_raw_llm_output)
        if not revised_cleaned_text or len(revised_cleaned_text) < config.MIN_ACCEPTABLE_DRAFT_LENGTH:
            logger.error(f"Revised draft for ch {chapter_number} is too short ({len(revised_cleaned_text or '')} chars) after cleaning. Min required: {config.MIN_ACCEPTABLE_DRAFT_LENGTH}.")
            await self._save_debug_output(chapter_number, "revision_fail_short_raw_llm", revised_raw_llm_output)
            return None
        
        # Similarity check to ensure the revision is meaningfully different
        original_embedding_task = llm_interface.async_get_embedding(original_text)
        revised_embedding_task = llm_interface.async_get_embedding(revised_cleaned_text)
        original_embedding, revised_embedding = await asyncio.gather(original_embedding_task, revised_embedding_task)

        if original_embedding is not None and revised_embedding is not None:
            similarity_score = utils.numpy_cosine_similarity(original_embedding, revised_embedding)
            logger.info(f"Revision similarity score with original draft: {similarity_score:.4f}")
            if similarity_score >= config.REVISION_SIMILARITY_ACCEPTANCE:
                logger.warning(f"Revision for ch {chapter_number} rejected: Too similar to original (Score: {similarity_score:.4f} >= Threshold: {config.REVISION_SIMILARITY_ACCEPTANCE}).")
                await self._save_debug_output(chapter_number, "revision_rejected_similar_raw_llm", revised_raw_llm_output)
                return None # Indicate revision failed due to similarity
        else:
            logger.warning(f"Could not get embeddings for revision similarity check of ch {chapter_number}. Accepting revision by default.")
            
        logger.info(f"Revision for ch {chapter_number} accepted (Length: {len(revised_cleaned_text)} chars).")
        return revised_cleaned_text, revised_raw_llm_output

    async def _finalize_chapter_core(self, chapter_number: int, final_text: str, raw_llm_log_for_db: str, from_flawed_draft: bool) -> bool:
        """Core finalization: summarize, embed, save to DB and files."""
        logger.info(f"Finalizing chapter {chapter_number} (From flawed draft: {from_flawed_draft}). Text length: {len(final_text)}.")
        if not final_text:
            logger.error(f"Cannot finalize ch {chapter_number}: Final text is missing or empty.")
            return False

        # These can run concurrently
        summary_task = self._summarize_chapter(final_text, chapter_number)
        embedding_task = llm_interface.async_get_embedding(final_text)
        summary, final_embedding = await asyncio.gather(summary_task, embedding_task)

        if final_embedding is None:
            logger.error(f"CRITICAL: Failed to generate embedding for final text of Chapter {chapter_number}. This may impact future context.")
            # Decide if this is fatal. For now, proceed but log heavily.

        try:
            await self.db_manager.async_save_chapter_data(
                chapter_number, final_text, raw_llm_log_for_db, summary, final_embedding, from_flawed_draft
            )
        except Exception as e: # Catch any exception from DB save
            logger.error(f"Database save failed for chapter {chapter_number}: {e}", exc_info=True)
            return False # Indicate failure

        # File I/O (chapter text, raw log)
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, self._save_chapter_text_files_sync, chapter_number, final_text, raw_llm_log_for_db)
            logger.info(f"Saved chapter text and raw LLM log files for ch {chapter_number}.")
        except IOError as e:
            logger.error(f"Failed writing chapter text/log files for ch {chapter_number}: {e}", exc_info=True)
            # This might not be fatal if DB save succeeded, but it's a problem.
        
        logger.info(f"Core finalization complete for ch {chapter_number}.")
        return True

    def _save_chapter_text_files_sync(self, chapter_number: int, final_text: str, raw_llm_log: str):
        """Synchronous helper for saving chapter text and raw LLM log files."""
        chapter_file_path = os.path.join(config.CHAPTERS_DIR, f"chapter_{chapter_number:04d}.txt") # Padded for sorting
        log_file_path = os.path.join(config.CHAPTER_LOGS_DIR, f"chapter_{chapter_number:04d}_raw_llm_log.txt")

        with open(chapter_file_path, 'w', encoding='utf-8') as f:
            f.write(final_text)
        with open(log_file_path, 'w', encoding='utf-8') as f:
            f.write(raw_llm_log) # Save the accumulated raw LLM interactions

    async def _update_knowledge_bases(self, chapter_number: int, final_text: str, from_flawed_draft: bool):
        """Updates JSON character/world profiles and the Knowledge Graph based on the finalized chapter."""
        if not final_text:
            logger.warning(f"Skipping knowledge base update for ch {chapter_number}: Final text is missing or empty.")
            return
        logger.info(f"Updating knowledge bases for ch {chapter_number} (Source from flawed draft: {from_flawed_draft})...")
        
        # These two operations can run concurrently
        update_json_task = self._update_character_and_world_json_from_chapter(final_text, chapter_number, from_flawed_draft)
        update_kg_task = self._extract_and_update_kg(final_text, chapter_number, from_flawed_draft)
        
        try:
            await asyncio.gather(update_json_task, update_kg_task)
            logger.info(f"Knowledge base updates (JSON profiles & KG) completed for ch {chapter_number}.")
        except Exception as e:
            logger.error(f"Error during concurrent knowledge base update for ch {chapter_number}: {e}", exc_info=True)
            # Log error but continue, as chapter text is saved. Individual methods should log their own specifics.

    async def _get_context(self, current_chapter_number: int) -> str:
        """Constructs context for the current chapter from previous summaries/text and KG."""
        if current_chapter_number <= 1:
            return "" # No context for the first chapter
        logger.debug(f"Retrieving and constructing context for Chapter {current_chapter_number}...")
        
        plot_point_focus, plot_point_index = self._get_plot_point_info(current_chapter_number)
        context_query_text = plot_point_focus if plot_point_focus else f"Narrative context relevant to events leading up to chapter {current_chapter_number}."
        
        if plot_point_focus:
            logger.info(f"Context query for ch {current_chapter_number} based on Plot Point {plot_point_index + 1}: '{context_query_text[:100]}...'")
        else:
            logger.warning(f"No specific plot point found for ch {current_chapter_number}. Using generic context query.")
            
        query_embedding = await llm_interface.async_get_embedding(context_query_text)
        
        # Fallback if query embedding fails: use sequential previous chapter summaries
        if query_embedding is None:
            logger.warning("Failed to generate embedding for context query. Falling back to sequential previous chapter summaries/text.")
            context_parts: List[str] = []
            total_chars = 0
            # Get last few chapters sequentially
            for i in range(max(1, current_chapter_number - config.CONTEXT_CHAPTER_COUNT), current_chapter_number):
                if total_chars >= config.MAX_CONTEXT_LENGTH: break
                chap_data = await self.db_manager.async_get_chapter_data_from_db(i)
                if chap_data:
                    content = (chap_data.get('summary') or chap_data.get('text', '')).strip()
                    is_prov = chap_data.get('is_provisional', False)
                    ctype = "Provisional Summary" if chap_data.get('summary') and is_prov else \
                            "Summary" if chap_data.get('summary') else \
                            "Provisional Text Snippet" if is_prov else "Text Snippet"
                    
                    if content:
                        prefix = f"[Fallback Context from Chapter {i} ({ctype})]:\n"; suffix = "\n---\n"
                        available_space = config.MAX_CONTEXT_LENGTH - total_chars - (len(prefix) + len(suffix))
                        if available_space <= 0: break
                        
                        truncated_content = content[:available_space]
                        context_parts.append(f"{prefix}{truncated_content}{suffix}")
                        total_chars += len(prefix) + len(truncated_content) + len(suffix)
            
            final_context = "\n".join(reversed(context_parts)).strip() # Most recent fallback context first
            logger.info(f"Constructed fallback context: {len(final_context)} chars.")
            return final_context
            
        # Semantic search using query embedding
        past_embeddings = await self.db_manager.async_get_all_past_embeddings(current_chapter_number)
        if not past_embeddings:
            logger.info("No past embeddings found for semantic context search.")
            return "" # No embeddings to compare against
        
        similarities = sorted(
            [(chap_num, utils.numpy_cosine_similarity(query_embedding, emb)) 
             for chap_num, emb in past_embeddings if emb is not None], 
            key=lambda item: item[1], 
            reverse=True
        )
        if not similarities:
            logger.info("No valid similarities found with past embeddings.")
            return ""
        
        # Select top N similar chapters, always include the immediate previous chapter if not already top N
        top_n_indices = [cs[0] for cs in similarities[:config.CONTEXT_CHAPTER_COUNT]]
        logger.info(f"Top {len(top_n_indices)} relevant chapters for context (semantic search): {top_n_indices} (Scores: {[f'{s:.3f}' for _, s in similarities[:config.CONTEXT_CHAPTER_COUNT]]})")
        
        # Ensure immediate previous chapter is included if it exists
        immediate_prev_chap_num = current_chapter_number - 1
        if immediate_prev_chap_num > 0 and immediate_prev_chap_num not in top_n_indices:
            # Add it, then re-sort and potentially trim if it exceeds CONTEXT_CHAPTER_COUNT
            # For simplicity, just add it; the fetching loop will handle MAX_CONTEXT_LENGTH
            top_n_indices.append(immediate_prev_chap_num)
            logger.debug(f"Added immediate previous chapter {immediate_prev_chap_num} to context list.")
            
        chapters_to_fetch = sorted(list(set(top_n_indices)), reverse=True) # Fetch most relevant/recent first
        logger.debug(f"Final list of chapters to fetch for context: {chapters_to_fetch}")
        
        context_parts: List[str] = []
        total_chars = 0
        
        # Fetch chapter data (summaries or text snippets)
        chap_data_tasks = {
            chap_num: self.db_manager.async_get_chapter_data_from_db(chap_num) 
            for chap_num in chapters_to_fetch
        }
        chap_data_results_list = await asyncio.gather(*chap_data_tasks.values())
        chap_data_map = dict(zip(chap_data_tasks.keys(), chap_data_results_list))

        for chap_num in chapters_to_fetch: # Process in order of relevance (most recent of the relevant ones first)
            if total_chars >= config.MAX_CONTEXT_LENGTH: break
            chap_data = chap_data_map.get(chap_num)
            if chap_data:
                content = (chap_data.get('summary') or chap_data.get('text', '')).strip()
                is_prov = chap_data.get('is_provisional', False)
                ctype = "Provisional Summary" if chap_data.get('summary') and is_prov else \
                        "Summary" if chap_data.get('summary') else \
                        "Provisional Text Snippet" if is_prov else "Text Snippet"
                
                if content:
                    prefix = f"[Context from Chapter {chap_num} ({ctype})]:\n"; suffix = "\n---\n"
                    available_space = config.MAX_CONTEXT_LENGTH - total_chars - (len(prefix) + len(suffix))
                    if available_space <= 0: break # No space left for this content

                    truncated_content = content[:available_space]
                    context_parts.append(f"{prefix}{truncated_content}{suffix}")
                    total_chars += len(prefix) + len(truncated_content) + len(suffix)
                    logger.debug(f"Added context from ch {chap_num} ({ctype}), {len(truncated_content)} chars. Total context chars: {total_chars}.")
            else:
                logger.warning(f"Could not retrieve chapter data for ch {chap_num} during context construction.")
                
        final_context = "\n".join(reversed(context_parts)).strip() # Present context chronologically (oldest relevant to newest relevant)
        logger.info(f"Constructed final semantic context: {len(final_context)} chars from chapters {chapters_to_fetch}.")
        return final_context

    @alru_cache(maxsize=config.SUMMARY_CACHE_SIZE)
    async def _summarize_chapter_llm_call(self, chapter_text_snippet_key: str, chapter_number: int) -> str:
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
            temperature=0.6, # Lower temperature for factual summary
            max_tokens=config.MAX_SUMMARY_TOKENS
        )
        return llm_interface.clean_model_response(summary_raw).strip()

    async def _summarize_chapter(self, chapter_text: Optional[str], chapter_number: int) -> Optional[str]:
        if not chapter_text or len(chapter_text) < 50: # Arbitrary short length, too small to summarize meaningfully
            logger.warning(f"Chapter {chapter_number} text too short for summarization ({len(chapter_text or '')} chars).")
            return None
            
        # Use a fixed-size snippet for the cache key and LLM call to improve cache hits
        # and manage token usage for summarization.
        snippet_for_summary = chapter_text[:config.KNOWLEDGE_UPDATE_SNIPPET_SIZE].strip()
        
        cleaned_summary = await self._summarize_chapter_llm_call(snippet_for_summary, chapter_number)

        if cleaned_summary: 
            logger.info(f"Generated summary for ch {chapter_number}: '{cleaned_summary[:100].strip()}...'")
            return cleaned_summary
        
        logger.warning(f"Failed to generate a valid summary for ch {chapter_number} via LLM.")
        return None

    async def _check_consistency(self, chapter_draft_text: Optional[str], chapter_number: int, previous_chapters_context: str) -> Optional[str]:
        """Checks chapter draft for consistency against plot, characters, world, KG, and previous context."""
        if not chapter_draft_text:
            logger.debug(f"Consistency check skipped for Ch {chapter_number}: empty draft text.")
            return None
        
        draft_snippet = chapter_draft_text[:config.KNOWLEDGE_UPDATE_SNIPPET_SIZE] # Use a snippet for token economy
        context_snippet = previous_chapters_context[:config.KNOWLEDGE_UPDATE_SNIPPET_SIZE // 2]
        
        protagonist_name = self.plot_outline.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME)
        kg_chapter_limit = chapter_number - 1 # Facts established up to the *previous* chapter

        # Gather relevant KG facts (non-provisional only for consistency check)
        # These can be fetched concurrently
        kg_loc_task = self.db_manager.async_get_most_recent_value(protagonist_name, "located_in", kg_chapter_limit, include_provisional=False)
        kg_status_task = self.db_manager.async_get_most_recent_value(protagonist_name, "status_is", kg_chapter_limit, include_provisional=False)
        # Add more critical KG fact queries here as needed
        
        kg_location, kg_status = await asyncio.gather(kg_loc_task, kg_status_task)
        
        kg_facts_for_prompt: List[str] = []
        if kg_location: kg_facts_for_prompt.append(f"- {protagonist_name}'s last reliably known location: {kg_location}.")
        if kg_status: kg_facts_for_prompt.append(f"- {protagonist_name}'s last reliably known status: {kg_status}.")
            
        kg_check_results_text = "**Key Reliable KG Facts (from pre-novel & previous chapters):**\n" + "\n".join(kg_facts_for_prompt) + "\n" if kg_facts_for_prompt else "**Key Reliable KG Facts:** None available or protagonist not tracked.\n"

        # Filtered profiles and world, noting provisional data up to previous chapter
        char_profiles_for_prompt = self._get_filtered_profiles_for_prompt(kg_chapter_limit)
        world_building_for_prompt = self._get_filtered_world_for_prompt(kg_chapter_limit)

        prompt = f"""/no_think
You are a Continuity Editor. Your task is to analyze the provided Draft Snippet for Chapter {chapter_number} for inconsistencies.
Compare the Draft Snippet against the following established information:
1. Plot Outline (overall story direction)
2. Character Profiles (character traits, status, history - note 'provisional' markers if present in prompt_notes)
3. World Building (locations, rules, lore - note 'provisional' markers if present in prompt_notes)
4. Key Reliable KG Facts (facts considered canon from previous chapters or pre-novel setup)
5. Previous Context (narrative flow and recent events)
6. The Draft Snippet's own internal consistency.

Your goal is to identify specific, objective contradictions or deviations from this established information.
Prioritize clear contradictions with facts from the Plot Outline, Character Profiles, World Building, and reliable KG Facts.

**Plot Outline Summary:**
```json
{json.dumps(self.plot_outline, indent=2, ensure_ascii=False, default=str, sort_keys=True)}
```
**Character Profiles (Key Info - check 'prompt_notes' for provisional status):**
```json
{json.dumps(char_profiles_for_prompt, indent=2, ensure_ascii=False, default=str, sort_keys=True)}
```
**World Building Notes (Key Info - check 'prompt_notes' for provisional status):**
```json
{json.dumps(world_building_for_prompt, indent=2, ensure_ascii=False, default=str, sort_keys=True)}
```
{kg_check_results_text}
**Previous Context (Snippet from prior chapters):**
--- PREVIOUS CONTEXT ---
{context_snippet if context_snippet.strip() else "N/A (e.g., this is Chapter 1 or context retrieval failed)."}
--- END PREVIOUS CONTEXT ---

**Chapter {chapter_number} Draft Snippet (to analyze):**
--- DRAFT SNIPPET ---
{draft_snippet}
--- END DRAFT SNIPPET ---

**Analysis Task:**
List ONLY specific, objective contradictions, inconsistencies, or significant deviations found in the Draft Snippet when compared to the provided established information.
If NO inconsistencies are found, respond with the single word: None
Otherwise, list each issue clearly and concisely, like:
- Character X acts out of established trait Y.
- Location Z is described differently than in world-building notes.
- Event A contradicts a reliable KG fact about B.

**Identified Inconsistencies (or "None"):**
"""
        response_raw = await llm_interface.async_call_llm(
            model_name=config.CONSISTENCY_CHECK_MODEL,
            prompt=prompt, 
            temperature=0.6, # Lower temp for factual analysis
            max_tokens=config.MAX_CONSISTENCY_TOKENS
        ) 
        response_cleaned = llm_interface.clean_model_response(response_raw).strip()

        if not response_cleaned or response_cleaned.lower() == "none":
            logger.info(f"Consistency check passed for ch {chapter_number}. No issues reported by LLM.")
            return None
        
        logger.warning(f"Consistency issues reported for ch {chapter_number}:\n{response_cleaned}")
        return response_cleaned

    async def _validate_plot_arc(self, chapter_draft_text: Optional[str], chapter_number: int) -> Optional[str]:
        """Validates if the chapter draft aligns with its intended plot point."""
        if not chapter_draft_text:
            logger.debug(f"Plot arc validation skipped for Ch {chapter_number}: empty draft text.")
            return None
            
        plot_point_focus, plot_point_index = self._get_plot_point_info(chapter_number)
        if plot_point_focus is None:
            logger.warning(f"Plot arc validation skipped for ch {chapter_number}: No plot point focus available for this chapter.")
            return None 
        
        logger.info(f"Validating plot arc for ch {chapter_number} against Plot Point {plot_point_index + 1}: '{plot_point_focus[:100]}...'")
        
        # Use chapter summary for validation if available and substantial, otherwise use a snippet of the draft.
        summary = await self._summarize_chapter(chapter_draft_text, chapter_number)
        validation_text_content = summary if summary and len(summary) > 50 else chapter_draft_text[:1500] # Use first 1500 chars if no good summary
        
        if not validation_text_content.strip():
            logger.warning(f"Plot arc validation skipped for Ch {chapter_number}: no text content for validation (summary/snippet empty).")
            return None
            
        protagonist_name = self.plot_outline.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME)
        prompt = f"""/no_think
You are a Story Analyst. Your task is to determine if the provided Chapter {chapter_number} Text (featuring protagonist: {protagonist_name}) successfully addresses its Intended Plot Point.

**Intended Plot Point for Chapter {chapter_number} (Plot Point #{plot_point_index + 1}):**
"{plot_point_focus}"

**Chapter {chapter_number} Text (Summary or Snippet):**
"{validation_text_content}"

**Evaluation Question:**
Does the provided Chapter Text (Summary/Snippet) clearly and substantially address or advance the Intended Plot Point?

**Response Format (CRITICAL):**
Respond with ONLY ONE of the following:
- `Yes` (if the chapter text aligns well with and advances the intended plot point)
- `No, because [specific reason]` (if the chapter text deviates, fails to address, or only superficially touches upon the intended plot point. Provide a concise, 1-2 sentence explanation for the deviation).

**Your Response:**"""
        validation_response_raw = await llm_interface.async_call_llm(
            model_name=config.INITIAL_SETUP_MODEL, # Can use a smaller/faster model for this focused task
            prompt=prompt, 
            temperature=0.6, # Low temp for focused decision
            max_tokens=config.MAX_PLOT_VALIDATION_TOKENS
        ) 
        cleaned_plot_response = llm_interface.clean_model_response(validation_response_raw).strip()
        
        if cleaned_plot_response.lower().startswith("yes"):
            logger.info(f"Plot arc validation passed for ch {chapter_number}.")
            return None
        elif cleaned_plot_response.lower().startswith("no, because"):
            reason = cleaned_plot_response[len("no, because"):].strip()
            if not reason: reason = "LLM indicated deviation but provided no specific reason."
            logger.warning(f"Plot arc deviation identified for ch {chapter_number}: {reason}")
            return reason
        
        logger.warning(f"Plot arc validation for ch {chapter_number} produced an ambiguous response: '{cleaned_plot_response}'. Assuming alignment as a fallback.")
        return None # Fallback: assume alignment if response is not clearly "No, because..."

    async def _update_character_and_world_json_from_chapter(self, chapter_text: Optional[str], chapter_number: int, from_flawed_draft: bool):
        """Updates character and world-building JSON files based on events in the chapter."""
        if not chapter_text or len(chapter_text) < 100: # Arbitrary threshold for meaningful content
            logger.info(f"Skipping JSON knowledge update for ch {chapter_number}: Text too short or None.")
            return

        # Heuristic: Check for mentions of known entities to decide if an update is likely needed.
        # This is a simple optimization and might miss new entities.
        known_char_names = list(self.character_profiles.keys())
        known_loc_names = list(self.world_building.get("locations", {}).keys())
        
        text_lower = chapter_text.lower()
        mentioned_entities = [name for name in known_char_names if name.lower() in text_lower]
        mentioned_entities.extend(name for name in known_loc_names if name.lower() in text_lower)
        
        # If no known entities are mentioned, and it's not an early chapter (where new entities are common),
        # we might skip the LLM call for updates to save resources. This is configurable.
        # For now, always attempt update, but log the heuristic.
        if not mentioned_entities and chapter_number > 3 : # Example: skip if no known mentions after Ch 3
             logger.info(f"JSON knowledge update for ch {chapter_number}: No known characters or locations mentioned significantly (heuristic). Will still attempt LLM update if configured.")
             # Potentially add a config flag here to truly skip if desired.

        logger.info(f"Attempting combined JSON (character/world) update for ch {chapter_number} (Source from flawed draft: {from_flawed_draft}). Mentions found (heuristic): {mentioned_entities[:5]}")
        text_snippet = chapter_text[:config.KNOWLEDGE_UPDATE_SNIPPET_SIZE] # Use a consistent snippet size
        protagonist_name = self.plot_outline.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME)
        
        # Dynamic instructions based on configuration
        dynamic_instr_char, dynamic_instr_world = "", ""
        if config.ENABLE_DYNAMIC_STATE_ADAPTATION:
            dynamic_instr_char = """For existing characters, if their traits, status, or core description needs modification based on THIS chapter's events, include a `"modification_proposal"` field. Example: `"modification_proposal": "MODIFY traits: ADD 'Determined', REMOVE 'Hesitant'"`. Only include characters that are updated, newly introduced, or have a modification proposal."""
            dynamic_instr_world = """For existing world items, if their properties need modification, include a `"modification_proposal"`. Example: `"modification_proposal": "MODIFY atmosphere: 'Now heavy with magical fallout'"`. Only include world elements (locations, society items, systems, lore, history) that are new, significantly changed by THIS chapter's events, or have a modification proposal."""
        else:
            dynamic_instr_char = "Only include characters whose information is directly updated or those newly introduced in THIS chapter."
            dynamic_instr_world = "Only include world elements that are new or significantly changed by THIS chapter's events."
            
        # Filtered profiles and world up to *previous* chapter state
        current_profiles_for_prompt = self._get_filtered_profiles_for_prompt(chapter_number - 1)
        current_world_for_prompt = self._get_filtered_world_for_prompt(chapter_number - 1)

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
      "modification_proposal": "MODIFY traits: ADD 'Brave'", // If dynamic adaptation enabled
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
         "modification_proposal": "MODIFY description: 'Magic is now unstable during eclipses.'", // If dynamic adaptation
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
            temperature=0.6 # More factual for updates
        )
        combined_updates = await llm_interface.async_parse_llm_json_response(
            raw_analysis, f"combined character/world JSON update for ch {chapter_number}"
        )

        if not combined_updates or not isinstance(combined_updates, dict):
            logger.warning(f"LLM parsing for combined char/world JSON updates failed or returned no data for ch {chapter_number}. Raw LLM: {raw_analysis[:200] if raw_analysis else 'EMPTY'}")
            return

        char_updates = combined_updates.get("character_updates")
        if char_updates and isinstance(char_updates, dict):
            self._merge_character_updates(char_updates, chapter_number, from_flawed_draft)
        else:
            logger.info(f"No 'character_updates' field found or it's not a dictionary in combined response for ch {chapter_number}.")

        world_updates = combined_updates.get("world_building_updates")
        if world_updates and isinstance(world_updates, dict):
            self._merge_world_building_updates(world_updates, chapter_number, from_flawed_draft)
        else:
            logger.info(f"No 'world_building_updates' field found or it's not a dictionary in combined response for ch {chapter_number}.")

    def _merge_character_updates(self, updates_from_llm: Dict[str, Any], chapter_number: int, from_flawed_draft: bool):
        """Merges LLM-proposed character updates into the agent's character_profiles."""
        if not updates_from_llm:
            logger.info(f"No character profile updates from LLM to merge for ch {chapter_number}.")
            return
        
        logger.info(f"Merging character profile JSON updates for ch {chapter_number}. Characters in update: {list(updates_from_llm.keys())}")
        updated_chars_count, new_chars_count = 0, 0
        provisional_marker_key = f"source_quality_chapter_{chapter_number}" # Marks data derived from this chapter

        for char_name, char_update_data in updates_from_llm.items():
            if not isinstance(char_update_data, dict):
                logger.warning(f"Skipping invalid character update data for '{char_name}' (not a dict). Data: {char_update_data}")
                continue
            
            # Make a copy to avoid modifying the input dict if it's used elsewhere
            char_update = char_update_data.copy()
            
            # Add provisional marker if data comes from a flawed draft
            if from_flawed_draft:
                char_update[provisional_marker_key] = "provisional_from_unrevised_draft"

            # Ensure development note for the current chapter exists if there are other updates
            dev_key = f"development_in_chapter_{chapter_number}"
            if dev_key not in char_update and (len(char_update) > (1 if provisional_marker_key in char_update else 0)):
                 # Add a default development note if other fields were updated but no specific note was provided
                 char_update[dev_key] = "Character appeared or was mentioned in this chapter."
            
            # Handle new character
            if char_name not in self.character_profiles:
                new_chars_count += 1
                logger.info(f"Adding new character '{char_name}' based on ch {chapter_number} analysis.")
                self.character_profiles[char_name] = {
                    "description": char_update.get("description", f"A character newly introduced in Chapter {chapter_number}."),
                    "traits": sorted(list(set(t for t in char_update.get("traits", []) if isinstance(t, str) and t.strip()))),
                    "relationships": char_update.get("relationships", {}), 
                    "status": char_update.get("status", "Newly introduced")
                }
                # Add development note and provisional marker to the new profile
                if dev_key in char_update: self.character_profiles[char_name][dev_key] = char_update[dev_key]
                if provisional_marker_key in char_update: self.character_profiles[char_name][provisional_marker_key] = char_update[provisional_marker_key]
                
                # Apply modification proposal if present (even for new characters, it might set initial state)
                if config.ENABLE_DYNAMIC_STATE_ADAPTATION and "modification_proposal" in char_update:
                    self._apply_modification_proposal(self.character_profiles[char_name], char_update["modification_proposal"], char_name, "new character profile")
            
            # Handle existing character
            else: 
                updated_chars_count += 1
                logger.debug(f"Updating existing character '{char_name}' based on ch {chapter_number} analysis.")
                existing_profile = self.character_profiles[char_name]
                
                if provisional_marker_key in char_update: # Mark existing profile if new info is provisional
                    existing_profile[provisional_marker_key] = char_update[provisional_marker_key]
                
                # Apply modification proposal first, as it might alter keys that are then updated
                if config.ENABLE_DYNAMIC_STATE_ADAPTATION and "modification_proposal" in char_update:
                    self._apply_modification_proposal(existing_profile, char_update["modification_proposal"], char_name, "existing character profile")
                
                # Merge other fields from char_update into existing_profile
                for key, value in char_update.items():
                    if key in ["modification_proposal", provisional_marker_key]: continue # Already handled
                    
                    if key == "traits" and isinstance(value, list):
                        if "traits" not in existing_profile or not isinstance(existing_profile["traits"], list):
                            existing_profile["traits"] = []
                        valid_new_traits = {t for t in value if isinstance(t, str) and t.strip()}
                        existing_profile["traits"] = sorted(list(set(existing_profile["traits"]).union(valid_new_traits)))
                    elif key == "relationships" and isinstance(value, dict):
                         if not isinstance(existing_profile.get("relationships"), dict):
                             existing_profile["relationships"] = {}
                         existing_profile["relationships"].update(value) # Simple dict update, could be smarter
                    elif key == "description" and isinstance(value, str) and value.strip():
                        # Only update if not handled by a specific "MODIFY DESCRIPTION" proposal
                        if not (config.ENABLE_DYNAMIC_STATE_ADAPTATION and "modification_proposal" in char_update and "MODIFY DESCRIPTION" in char_update["modification_proposal"].upper()):
                            existing_profile["description"] = value # Overwrite description
                    elif key == dev_key and isinstance(value, str) and value.strip():
                        existing_profile[key] = value # Add/update chapter-specific development
                    elif key == "status" and isinstance(value, str) and value.strip():
                        existing_profile["status"] = value # Overwrite status
                    elif key not in existing_profile and value is not None: # Add new keys if they don't exist
                        existing_profile[key] = value

        if updated_chars_count > 0 or new_chars_count > 0:
            logger.info(f"Character profile JSON merge complete for ch {chapter_number}. Updated: {updated_chars_count}, New: {new_chars_count}.")
        else:
            logger.info(f"No character profiles were effectively updated or added for ch {chapter_number} after LLM analysis.")

    def _merge_world_building_updates(self, updates_from_llm: Dict[str, Any], chapter_number: int, from_flawed_draft: bool):
        """Merges LLM-proposed world-building updates into the agent's world_building state."""
        if not updates_from_llm:
            logger.info(f"No world-building updates from LLM to merge for ch {chapter_number}.")
            return

        logger.info(f"Merging world-building JSON updates for ch {chapter_number}. Categories in update: {list(updates_from_llm.keys())}")
        items_affected_count = 0
        provisional_marker_key = f"source_quality_chapter_{chapter_number}" # Marks data derived from this chapter

        for category_key, category_updates_dict in updates_from_llm.items():
            if not isinstance(category_updates_dict, dict) or not category_updates_dict:
                logger.debug(f"Skipping empty or invalid update for world category '{category_key}' in ch {chapter_number}.")
                continue
            
            # Ensure the category exists in self.world_building
            if category_key not in self.world_building:
                self.world_building[category_key] = {}
            elif not isinstance(self.world_building[category_key], dict): # If it exists but isn't a dict (e.g., old format)
                logger.warning(f"Overwriting non-dictionary world category '{category_key}' with new dictionary structure.")
                self.world_building[category_key] = {}
            
            target_category_dict = self.world_building[category_key]
            
            # Mark category itself if updates are provisional (though item-level is more granular)
            # if from_flawed_draft:
            #     target_category_dict[provisional_marker_key] = "provisional_from_unrevised_draft"

            for item_name, item_update_details in category_updates_dict.items():
                if not isinstance(item_update_details, dict):
                    logger.warning(f"Skipping invalid item_details for '{item_name}' in category '{category_key}' (not a dict). Data: {item_update_details}")
                    continue
                
                item_log_name = f"{category_key}.{item_name}"
                update_copy = item_update_details.copy() # Work with a copy

                if from_flawed_draft:
                    update_copy[provisional_marker_key] = "provisional_from_unrevised_draft"
                
                existing_item_data = target_category_dict.get(item_name)
                
                if existing_item_data is None: # New world item
                    logger.info(f"Adding new world item '{item_log_name}' from ch {chapter_number} analysis.")
                    new_item_data = self._robust_merge_world_item_recursive({}, update_copy, item_log_name, chapter_number, from_flawed_draft)
                    new_item_data[f"added_in_chapter_{chapter_number}"] = True # Mark when it was added
                    target_category_dict[item_name] = new_item_data
                    items_affected_count +=1
                elif isinstance(existing_item_data, dict): # Existing world item, merge updates
                    logger.debug(f"Updating existing world item '{item_log_name}' from ch {chapter_number} analysis.")
                    updated_item_data = self._robust_merge_world_item_recursive(existing_item_data, update_copy, item_log_name, chapter_number, from_flawed_draft)
                    target_category_dict[item_name] = updated_item_data
                    # Check if it was actually modified by this call (beyond just existing)
                    if updated_item_data.get(f"updated_in_chapter_{chapter_number}") or update_copy.get(provisional_marker_key):
                        items_affected_count +=1
                else: # Existing item is not a dict, problematic state. Overwrite with new dict.
                    logger.warning(f"Existing world item '{item_log_name}' is not a dictionary. Overwriting with new data from ch {chapter_number}.")
                    new_item_data = self._robust_merge_world_item_recursive({}, update_copy, item_log_name, chapter_number, from_flawed_draft)
                    new_item_data[f"added_in_chapter_{chapter_number}"] = True # Mark as "added" in this form
                    target_category_dict[item_name] = new_item_data
                    items_affected_count += 1
            
            # Mark category as updated if any of its items were affected in this chapter
            if any(isinstance(v,dict) and (v.get(f"updated_in_chapter_{chapter_number}") or v.get(f"added_in_chapter_{chapter_number}")) 
                   for v in target_category_dict.values()):
                 target_category_dict[f"category_updated_in_chapter_{chapter_number}"] = True

        if items_affected_count > 0:
            logger.info(f"World-building JSON merge complete for ch {chapter_number}. Approximately {items_affected_count} items affected/added.")
        else:
            logger.info(f"No world-building JSON items were effectively updated or added for ch {chapter_number} after LLM analysis.")

    def _apply_modification_proposal(self, target_dict: Dict[str, Any], proposal_str: str, item_name_for_log: str, item_type_for_log: str):
        """Applies a modification proposal string to a dictionary (profile or world item)."""
        if not isinstance(proposal_str, str) or not proposal_str.strip():
            logger.debug(f"Empty or invalid modification proposal for '{item_name_for_log}'. Proposal: '{proposal_str}'")
            return
        
        logger.debug(f"Applying modification proposal for '{item_name_for_log}' ({item_type_for_log}): '{proposal_str}'")
        
        # Example proposal: "MODIFY traits: ADD 'Brave', REMOVE 'Cowardly'; MODIFY status: 'Injured'"
        # This regex is simplified; a more robust parser might be needed for complex proposals.
        # It captures "MODIFY key: value_part"
        # And for traits, "ADD 'Trait'" or "REMOVE 'Trait'"
        
        # Simple strategy: MODIFY <key>: <new_value_string>
        # For traits: MODIFY traits: ADD 'x', REMOVE 'y'
        
        # Normalize proposal for easier parsing
        proposal_norm = proposal_str.strip().upper()
        key_to_modify_match = re.match(r"MODIFY\s+([\w_]+)\s*:(.*)", proposal_norm, re.IGNORECASE)

        if not key_to_modify_match:
            logger.warning(f"Invalid modification proposal format for '{item_name_for_log}'. Proposal: '{proposal_str}'. Expected 'MODIFY key: value'.")
            return

        key_name_upper = key_to_modify_match.group(1).strip()
        # Find original case key in target_dict
        original_key_name = next((k for k in target_dict if k.upper() == key_name_upper), key_name_upper.lower())
        
        value_modification_str_original_case = proposal_str[key_to_modify_match.end(1)+1:].strip() # Get the part after "MODIFY key:"

        try:
            if original_key_name.lower() == "traits": 
                if "traits" not in target_dict or not isinstance(target_dict["traits"], list):
                    target_dict["traits"] = [] # Ensure 'traits' is a list
                
                current_traits_set = set(target_dict["traits"])
                # Process ADD operations
                for add_match in re.finditer(r"ADD\s+['\"]([^'\"]+)['\"]", value_modification_str_original_case, re.IGNORECASE):
                    trait_to_add = add_match.group(1).strip()
                    if trait_to_add: current_traits_set.add(trait_to_add)
                # Process REMOVE operations
                for remove_match in re.finditer(r"REMOVE\s+['\"]([^'\"]+)['\"]", value_modification_str_original_case, re.IGNORECASE):
                    trait_to_remove = remove_match.group(1).strip()
                    if trait_to_remove: current_traits_set.discard(trait_to_remove)
                
                target_dict["traits"] = sorted(list(current_traits_set))
                logger.info(f"Applied trait modifications for '{item_name_for_log}'. New traits: {target_dict['traits']}")
            else: # General key modification
                new_value_str = value_modification_str_original_case.strip("'\" ") # Remove potential quotes around the whole value
                if new_value_str: # Only update if there's a non-empty value
                    target_dict[original_key_name] = new_value_str 
                    logger.info(f"Applied modification to '{original_key_name}' for '{item_name_for_log}'. New value: '{new_value_str[:70]}...'")
                else:
                    logger.warning(f"Modification proposal for '{original_key_name}' of '{item_name_for_log}' resulted in an empty new value. Proposal: '{proposal_str}'")
        except Exception as e:
            logger.error(f"Error applying modification proposal for '{item_name_for_log}': {e}. Proposal: '{proposal_str}'", exc_info=True)

    def _robust_merge_world_item_recursive(self, target_dict: Dict[str, Any], update_dict: Dict[str, Any], item_name_for_log: str, chapter_num: int, from_flawed_draft_source: bool) -> Dict[str, Any]:
        """Recursively merges update_dict into target_dict for world items."""
        # Ensure target_dict is a dict; if not (e.g. old data was just a string), initialize it.
        current_item_data = target_dict.copy() if isinstance(target_dict, dict) else {}
        if not isinstance(target_dict, dict) and target_dict is not None: # If it was not None and not a dict
            current_item_data['description'] = str(target_dict) # Preserve old string value as description
            logger.warning(f"World item '{item_name_for_log}' was not a dict. Converted to dict, old value saved as 'description'.")
        
        item_was_modified_this_call = False
        provisional_marker_key = f"source_quality_chapter_{chapter_num}"

        # Handle top-level provisional marker from update_dict
        if provisional_marker_key in update_dict:
            current_item_data[provisional_marker_key] = update_dict[provisional_marker_key]
            item_was_modified_this_call = True # Marking as provisional is a modification

        # Apply modification proposal if present
        if config.ENABLE_DYNAMIC_STATE_ADAPTATION and "modification_proposal" in update_dict:
            proposal = update_dict.pop("modification_proposal") # Remove after processing
            if isinstance(proposal, str) and proposal.strip():
                self._apply_modification_proposal(current_item_data, proposal, item_name_for_log, "world item")
                item_was_modified_this_call = True
        
        # Iterate through keys in the update dictionary
        for key, value_from_update in update_dict.items():
            # Skip special keys already handled or to be ignored in direct merge
            if key in [provisional_marker_key, "modification_proposal"] or key.startswith(("updated_in_chapter_", "added_in_chapter_", "elaboration_in_chapter_")):
                if key.startswith("elaboration_in_chapter_") and isinstance(value_from_update, str) and value_from_update.strip():
                    current_item_data[key] = value_from_update # Keep elaborations
                    item_was_modified_this_call = True
                continue

            current_value_in_target = current_item_data.get(key)

            if isinstance(value_from_update, dict):
                # If target's value for key is not a dict, or doesn't exist, initialize it
                if not isinstance(current_value_in_target, dict):
                    current_item_data[key] = {}
                    item_was_modified_this_call = True # Structure changed
                # Recursive merge for nested dictionaries
                merged_sub_dict = self._robust_merge_world_item_recursive(current_item_data[key], value_from_update, f"{item_name_for_log}.{key}", chapter_num, from_flawed_draft_source)
                if merged_sub_dict != current_item_data[key]: # Check if sub-merge actually changed anything
                    item_was_modified_this_call = True
                current_item_data[key] = merged_sub_dict
            elif isinstance(value_from_update, list):
                # For lists, typically append unique items or replace. Here, append unique.
                if not isinstance(current_value_in_target, list):
                    current_item_data[key] = []
                    item_was_modified_this_call = True # Structure changed
                
                initial_list_len = len(current_item_data[key])
                for item_in_list_update in value_from_update:
                    if item_in_list_update not in current_item_data[key]:
                        current_item_data[key].append(item_in_list_update)
                if len(current_item_data[key]) > initial_list_len:
                    item_was_modified_this_call = True
            elif value_from_update != current_value_in_target: # Simple value update
                current_item_data[key] = value_from_update
                item_was_modified_this_call = True
        
        # If any modification happened during this call, mark it as updated in this chapter
        # unless it was just added in this chapter (which is handled by the caller)
        if item_was_modified_this_call and not current_item_data.get(f"added_in_chapter_{chapter_num}"):
            current_item_data[f"updated_in_chapter_{chapter_num}"] = True
            
        return current_item_data

    def _heuristic_entity_spotter(self, text_snippet: str) -> List[str]:
        """Basic heuristic to spot potential entities (proper nouns) in text, including known characters."""
        entities = set(self.character_profiles.keys()) # Start with known characters
        
        # Regex for capitalized words/phrases (potential proper nouns)
        # This is a simple heuristic and may have false positives/negatives.
        # Looks for sequences of 1-3 capitalized words.
        for match in re.finditer(r'\b([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]+){0,2})\b', text_snippet):
            entities.add(match.group(1).strip())
        
        # Filter out very short "entities" unless they are known characters,
        # and common words that might be capitalized at sentence starts.
        # This filtering needs to be tuned based on observed model outputs.
        common_false_positives = {"The", "A", "An", "Is", "It", "He", "She", "They"} # Example
        
        return sorted([e for e in list(entities) 
                       if (len(e) > 3 or e in self.character_profiles) and e not in common_false_positives])


    @alru_cache(maxsize=config.KG_TRIPLE_EXTRACTION_CACHE_SIZE)
    async def _extract_kg_triples_llm_call(self, text_snippet_for_kg_key: str, chapter_number: int, candidate_entities_json_key: str) -> str:
        """Cached LLM call for KG triple extraction. Keys are inputs to ensure cache effectiveness."""
        protagonist_name = self.plot_outline.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME)
        common_predicates = [
            "is_a", "located_in", "has_trait", "status_is", "feels", "knows", "believes", "wants", 
            "interacted_with", "travelled_to", "discovered", "acquired", "lost", "used_item", 
            "attacked", "helped", "damaged", "repaired", "contains", "part_of", "caused_by", 
            "leads_to", "observed", "heard", "said", "thought_about", "decided_to", "has_goal", 
            "has_feature", "related_to", "member_of", "leader_of", "enemy_of", "ally_of", 
            "works_for", "has_ability", "possesses", "created_by"
        ] # Expanded list
        
        candidate_entities_prompt_section = ""
        if candidate_entities_json_key and candidate_entities_json_key != "[]": # Check if not empty list string
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
""" # Trailing [ to guide LLM
        return await llm_interface.async_call_llm(
            model_name=config.KNOWLEDGE_UPDATE_MODEL,
            prompt=prompt, 
            temperature=0.6, # Lower temperature for factual extraction
            max_tokens=config.MAX_KG_TRIPLE_TOKENS
        )

    async def _extract_and_update_kg(self, chapter_text: Optional[str], chapter_number: int, from_flawed_draft: bool):
        """Extracts KG triples from chapter text and adds them to the database."""
        if not chapter_text:
            logger.warning(f"Skipping KG extraction for ch {chapter_number}: Chapter text is None or empty.")
            return
            
        logger.info(f"Extracting KG triples for ch {chapter_number} (Source from flawed draft: {from_flawed_draft})...")
        
        # Use a larger snippet for KG extraction than for summarization, but still capped
        text_snippet_for_kg = chapter_text[:config.KNOWLEDGE_UPDATE_SNIPPET_SIZE * 2].strip() 
        if len(text_snippet_for_kg) < len(chapter_text):
            logger.warning(f"KG extraction for ch {chapter_number} will use truncated text ({len(text_snippet_for_kg)} chars out of {len(chapter_text)}).")

        # Heuristic: Pre-identify candidate entities to guide the LLM
        candidate_entities = self._heuristic_entity_spotter(text_snippet_for_kg)
        logger.debug(f"Candidate entities identified for KG extraction in Ch {chapter_number}: {candidate_entities[:10]}")
        candidate_entities_json_for_prompt = json.dumps(candidate_entities) # Pass as JSON string for cache key

        # Use cached LLM call for extraction
        raw_triples_json_str = await self._extract_kg_triples_llm_call(
            text_snippet_for_kg, chapter_number, candidate_entities_json_for_prompt
        )
            
        parsed_triples = await llm_interface.async_parse_llm_json_response(
            raw_triples_json_str, f"KG triple extraction for chapter {chapter_number}", expect_type=list
        )
        
        if parsed_triples is None: # This means parsing failed even after LLM correction
             logger.error(f"Failed to extract or parse any KG triples for ch {chapter_number} after all attempts. Raw LLM output: {raw_triples_json_str[:200] if raw_triples_json_str else 'EMPTY'}")
             await self._save_debug_output(chapter_number, "kg_extraction_final_fail_raw_llm", raw_triples_json_str or "EMPTY_RAW_TRIPLES_JSON")
             return
        
        if not parsed_triples: # Empty list [] is a valid response meaning no triples found
            logger.info(f"No KG triples were extracted by the LLM for ch {chapter_number}.")
            return
             
        added_count, skipped_count = 0, 0
        # Batch DB additions if db_manager supports it, or add concurrently
        kg_add_tasks = []
        for triple_any in parsed_triples:
            if isinstance(triple_any, list) and len(triple_any) == 3:
                # Ensure all components are strings and non-empty after stripping
                subj = str(triple_any[0]).strip() if triple_any[0] is not None else ""
                pred = str(triple_any[1]).strip() if triple_any[1] is not None else ""
                obj  = str(triple_any[2]).strip() if triple_any[2] is not None else ""
                
                if subj and pred and obj:
                    # Schedule async DB add operation
                    kg_add_tasks.append(
                        self.db_manager.async_add_kg_triple(subj, pred, obj, chapter_number, is_provisional=from_flawed_draft)
                    )
                    added_count += 1
                else:
                    logger.warning(f"Skipping invalid KG triple (empty component after strip) in ch {chapter_number}: Original: {triple_any}, Stripped: ['{subj}','{pred}','{obj}']")
                    skipped_count += 1
            else:
                logger.warning(f"Skipping invalid KG triple format (not a list of 3) in ch {chapter_number}: {triple_any}")
                skipped_count += 1
        
        if kg_add_tasks:
            await asyncio.gather(*kg_add_tasks) # Execute all valid DB additions concurrently
        
        logger.info(f"KG update for ch {chapter_number}: Attempted to add {added_count} triples, skipped {skipped_count}. (Source Provisional: {from_flawed_draft})")

    async def _prepopulate_knowledge_graph(self):
        """Pre-populates the Knowledge Graph from the initial plot outline and world-building data."""
        logger.info("Starting Knowledge Graph pre-population from plot and world data...")
        if not self.plot_outline or self.plot_outline.get("is_default", True):
            logger.warning("Skipping KG pre-population: Plot outline is missing or default.")
            return
        if not self.world_building or self.world_building.get("is_default", True):
            logger.warning("Skipping KG pre-population: World building data is missing or default.")
            return

        # Create a pruned/summarized version of plot and world for the prompt
        pruned_plot = {
            "title": self.plot_outline.get("title"), 
            "protagonist_name": self.plot_outline.get("protagonist_name"),
            "genre": self.plot_outline.get("genre"), 
            "theme": self.plot_outline.get("theme"),
            "setting_description": self.plot_outline.get("setting"), 
            "conflict_summary": self.plot_outline.get("conflict"),
            "character_arc": self.plot_outline.get("character_arc"), 
            "key_plot_points_summary": self.plot_outline.get("plot_points", [])[:2] # First 2 plot points
        }
        pruned_world = {}
        for category, items in self.world_building.items():
            if category == "is_default" or not isinstance(items, dict): continue # Skip marker or non-dict categories
            pruned_world[category] = {}
            # Take a few items from each category, and summarize their descriptions
            for item_name, item_details in list(items.items())[:3]: # First 3 items per category
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
            
        protagonist_name = self.plot_outline.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME)
        novel_title = self.plot_outline.get("title", config.DEFAULT_PLOT_OUTLINE_TITLE)
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
            model_name=config.KNOWLEDGE_UPDATE_MODEL, # Use a capable model
            prompt=prompt, 
            temperature=0.6, # Low temp for factual extraction from given data
            max_tokens=config.MAX_PREPOP_KG_TOKENS
        )
        parsed_triples = await llm_interface.async_parse_llm_json_response(
            raw_triples_json_str, "KG pre-population triple extraction", expect_type=list
        )

        if parsed_triples is None:
            logger.error(f"Failed to extract/parse KG triples for pre-population after all attempts. Raw LLM: {raw_triples_json_str[:500] if raw_triples_json_str else 'EMPTY'}")
            await self._save_debug_output(config.KG_PREPOPULATION_CHAPTER_NUM, "kg_prepop_final_fail_raw_llm", raw_triples_json_str or "EMPTY_PREPOP_TRIPLES_JSON")
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
                        self.db_manager.async_add_kg_triple(subj, pred, obj, config.KG_PREPOPULATION_CHAPTER_NUM, is_provisional=False) # Pre-pop facts are not provisional
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
        if added_count == 0 and parsed_triples: # LLM returned triples, but all were invalid
            logger.warning("KG pre-population resulted in 0 valid triples added despite LLM returning data. Check LLM output and parsing.")

    def _get_relevant_character_state_snippet(self, current_chapter_num_for_filtering: Optional[int] = None) -> str:
        """Creates a concise JSON string of key character states for prompts."""
        snippet_data: Dict[str, Dict[str, str]] = {}
        char_count = 0
        
        # Prioritize protagonist
        protagonist_name = self.plot_outline.get("protagonist_name")
        sorted_char_names: List[str] = []
        if protagonist_name and protagonist_name in self.character_profiles:
            sorted_char_names.append(protagonist_name)
        
        # Add other characters, ensuring protagonist (if present) is first
        for name in sorted(self.character_profiles.keys()):
            if name != protagonist_name:
                sorted_char_names.append(name)
            
        # Determine effective chapter for filtering provisional/development notes (up to end of previous chapter)
        effective_filter_chapter = (current_chapter_num_for_filtering - 1) \
            if current_chapter_num_for_filtering is not None and current_chapter_num_for_filtering > 0 \
            else config.KG_PREPOPULATION_CHAPTER_NUM # For Ch1 planning, use pre-pop state

        for name in sorted_char_names:
            if char_count >= config.PLANNING_CONTEXT_MAX_CHARACTERS_IN_SNIPPET:
                break
            
            profile = self.character_profiles.get(name, {})
            if not isinstance(profile, dict): continue # Skip if profile somehow isn't a dict

            provisional_note = ""
            # Check if any source_quality_chapter_X key up to effective_filter_chapter indicates provisional
            if any(key.startswith("source_quality_chapter_") and 
                   int(key.split('_')[-1]) <= effective_filter_chapter and
                   profile.get(key) == "provisional_from_unrevised_draft"
                   for key in profile):
                 provisional_note = " (Note: Some info may be provisional based on unrevised prior chapters)"

            # Get most recent development note up to effective_filter_chapter
            dev_notes_keys = sorted(
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

    def _get_relevant_world_state_snippet(self, current_chapter_num_for_filtering: Optional[int] = None) -> str:
        """Creates a concise JSON string of key world states for prompts."""
        snippet_data: Dict[str, Any] = {}
        
        effective_filter_chapter = (current_chapter_num_for_filtering - 1) \
            if current_chapter_num_for_filtering is not None and current_chapter_num_for_filtering > 0 \
            else config.KG_PREPOPULATION_CHAPTER_NUM

        def get_provisional_note_for_category(category_dict: Dict[str, Any], chapter_limit: int) -> str:
            # Check if the category itself or any item within it (up to chapter_limit) is marked provisional
            if any(key.startswith("source_quality_chapter_") and 
                   int(key.split('_')[-1]) <= chapter_limit and
                   category_dict.get(key) == "provisional_from_unrevised_draft"
                   for key in category_dict):
                 return " (Note: Some category info may be provisional)"
            
            for item_data in category_dict.values():
                if isinstance(item_data, dict) and \
                   any(key.startswith("source_quality_chapter_") and 
                       int(key.split('_')[-1]) <= chapter_limit and
                       item_data.get(key) == "provisional_from_unrevised_draft"
                       for key in item_data):
                    return " (Note: Some items within this category may have provisional info)"
            return ""

        world_categories_for_snippet = {
            "locations": config.PLANNING_CONTEXT_MAX_LOCATIONS_IN_SNIPPET,
            # "society": config.PLANNING_CONTEXT_MAX_FACTIONS_IN_SNIPPET, # Assuming factions are main part of society
            "systems": config.PLANNING_CONTEXT_MAX_SYSTEMS_IN_SNIPPET,
        }
        
        for category_name, max_items in world_categories_for_snippet.items():
            category_data = self.world_building.get(category_name, {})
            if isinstance(category_data, dict) and category_data:
                prov_note = get_provisional_note_for_category(category_data, effective_filter_chapter)
                # Get item names, potentially with a short description snippet
                item_snippets = []
                for item_name, item_details in list(category_data.items())[:max_items]:
                    if item_name.startswith(("source_quality_chapter_", "category_updated_in_chapter_")): continue # Skip metadata keys
                    desc_snippet = ""
                    if isinstance(item_details, dict) and item_details.get("description"):
                        desc_snippet = f": {str(item_details['description'])[:50].strip()}..."
                    item_snippets.append(f"{item_name}{desc_snippet}")

                if item_snippets:
                    snippet_data[f"key_{category_name}{prov_note}"] = item_snippets
        
        # Special handling for factions if they are nested under "society"
        society_data = self.world_building.get("society", {})
        if isinstance(society_data, dict):
            factions_data = society_data.get("Key Factions", society_data.get("factions", {})) # Common names for factions
            if isinstance(factions_data, dict) and factions_data:
                prov_note_factions = get_provisional_note_for_category(factions_data, effective_filter_chapter)
                faction_names = [name for name in list(factions_data.keys()) if not name.startswith("source_quality_chapter_")][:config.PLANNING_CONTEXT_MAX_FACTIONS_IN_SNIPPET]
                if faction_names:
                     snippet_data[f"key_factions{prov_note_factions}"] = faction_names
                     
        return json.dumps(snippet_data, indent=2, ensure_ascii=False, default=str) if snippet_data else "No significant world-building data available or applicable."

    def _get_filtered_profiles_for_prompt(self, up_to_chapter_inclusive: Optional[int] = None) -> JsonStateData:
        """Creates a copy of character profiles, adding 'prompt_notes' for provisional data up to a chapter."""
        if not self.character_profiles: return {}
        profiles_copy = json.loads(json.dumps(self.character_profiles)) # Deep copy
        
        if up_to_chapter_inclusive is None: # If no chapter limit, return full copy
            return profiles_copy

        for char_name, profile_data in profiles_copy.items():
            if not isinstance(profile_data, dict): continue # Should not happen with good data
            
            provisional_notes_for_char: List[str] = []
            # Check for provisional markers from chapter 1 up to and including up_to_chapter_inclusive
            for i in range(1, up_to_chapter_inclusive + 1): 
                prov_key = f"source_quality_chapter_{i}"
                if profile_data.get(prov_key) == "provisional_from_unrevised_draft":
                    provisional_notes_for_char.append(f"Information for this character updated in Chapter {i} was marked as provisional (derived from an unrevised draft).")
            
            if provisional_notes_for_char:
                if "prompt_notes" not in profile_data: profile_data["prompt_notes"] = []
                # Add unique notes
                for note in provisional_notes_for_char:
                    if note not in profile_data["prompt_notes"]:
                        profile_data["prompt_notes"].append(note)
        return profiles_copy

    def _get_filtered_world_for_prompt(self, up_to_chapter_inclusive: Optional[int] = None) -> JsonStateData:
        """Creates a copy of world_building, adding 'prompt_notes' for provisional data up to a chapter."""
        if not self.world_building: return {}
        world_copy = json.loads(json.dumps(self.world_building)) # Deep copy

        if up_to_chapter_inclusive is None:
            return world_copy

        for category_name, category_items in world_copy.items():
            if not isinstance(category_items, dict): continue

            # Check category-level provisional markers
            category_provisional_notes: List[str] = []
            for i in range(1, up_to_chapter_inclusive + 1):
                cat_prov_key = f"source_quality_chapter_{i}" 
                if category_items.get(cat_prov_key) == "provisional_from_unrevised_draft":
                    category_provisional_notes.append(f"The category '{category_name}' had information updated in Chapter {i} marked as provisional.")
            
            if category_provisional_notes:
                if "prompt_notes" not in category_items: category_items["prompt_notes"] = []
                for note in category_provisional_notes:
                    if note not in category_items["prompt_notes"]:
                         category_items["prompt_notes"].append(note)

            # Check item-level provisional markers
            for item_name, item_data in category_items.items():
                if not isinstance(item_data, dict): continue # Skip non-dict items like prompt_notes itself
                
                item_provisional_notes: List[str] = []
                for i in range(1, up_to_chapter_inclusive + 1):
                    item_prov_key = f"source_quality_chapter_{i}"
                    if item_data.get(item_prov_key) == "provisional_from_unrevised_draft":
                        item_provisional_notes.append(f"The world item '{item_name}' (category: '{category_name}') had information updated in Chapter {i} marked as provisional.")
                
                if item_provisional_notes:
                    if "prompt_notes" not in item_data: item_data["prompt_notes"] = []
                    for note in item_provisional_notes:
                        if note not in item_data["prompt_notes"]:
                            item_data["prompt_notes"].append(note)
        return world_copy

    async def _save_debug_output(self, chapter_number: int, stage_description: str, content: Any):
        """Saves content to a debug file, useful for inspecting LLM outputs or intermediate states."""
        if content is None: return # Don't save if no content
        
        content_str = str(content) if not isinstance(content, str) else content
        if not content_str.strip(): return # Don't save empty strings
            
        try:
            # Sanitize stage_description for use in filename
            safe_stage_desc = "".join(c if c.isalnum() or c in ['_', '-'] else "_" for c in stage_description)
            file_name = f"chapter_{chapter_number:04d}_{safe_stage_desc}.txt"
            file_path = os.path.join(config.DEBUG_OUTPUTS_DIR, file_name)
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._save_debug_output_sync_io, file_path, content_str, chapter_number, stage_description)
        except Exception as e: # Catch broad exceptions during the async part (e.g., path creation)
            logger.error(f"Failed to initiate save for debug output (Ch {chapter_number}, Stage '{stage_description}'): {e}", exc_info=True)

    def _save_debug_output_sync_io(self, file_path: str, content_str: str, chapter_number: int, stage_description: str):
        """Synchronous I/O part of saving debug output."""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True) # Ensure dir exists
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content_str)
            logger.debug(f"Saved debug output for Ch {chapter_number}, Stage '{stage_description}' to {file_path}")
        except IOError as e:
            logger.error(f"IOError saving debug output for Ch {chapter_number}, Stage '{stage_description}' to {file_path}: {e}", exc_info=True)
        except Exception as e: # Catch other errors during sync file write
            logger.error(f"Unexpected error saving debug output for Ch {chapter_number}, Stage '{stage_description}' to {file_path}: {e}", exc_info=True)