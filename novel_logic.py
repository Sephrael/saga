# novel_logic.py
"""
Contains the core logic for the novel generation agent, including state management,
chapter writing, analysis, revision, and knowledge updates.
Integrates a knowledge graph (KG) for improved consistency and context.

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
import random # For unhinged mode
from typing import Dict, List, Optional, Tuple, Any, TypedDict, Union

# Import components from other modules
import config
import utils
import llm_interface
from database_manager import DatabaseManager

# Initialize logger for this module
logger = logging.getLogger(__name__)

# Type Hinting for Evaluation Results
class EvaluationResult(TypedDict):
    needs_revision: bool
    reasons: List[str]
    coherence_score: Optional[float]
    consistency_issues: Optional[str] 
    plot_deviation_reason: Optional[str]

# Type Hinting for Scene Plan
class SceneDetail(TypedDict):
    scene_number: int
    summary: str 
    characters_involved: List[str]
    key_dialogue_points: List[str] 
    setting_details: str 
    contribution: str 

class NovelWriterAgent:
    """
    Manages the state and orchestrates the process of generating a novel
    chapter by chapter, interacting with LLMs, a database, and a knowledge graph.
    """

    def __init__(self):
        """Initializes the agent, loads state, and sets up components."""
        logger.info("Initializing NovelWriterAgent...")
        self.db_manager = DatabaseManager(config.DATABASE_FILE)
        self.plot_outline: Dict[str, Any] = {}
        self.character_profiles: Dict[str, Any] = {}
        self.world_building: Dict[str, Any] = {}
        self.chapter_count: int = 0
        self._load_existing_state()
        logger.info(f"NovelWriterAgent initialized. Current chapter count: {self.chapter_count}")

    def _load_existing_state(self):
        logger.info("Attempting to load existing agent state...")
        self.chapter_count = self.db_manager.load_chapter_count()
        logger.info(f"Loaded chapter count from database: {self.chapter_count}")
        for file_path, attr_name in [
            (config.PLOT_OUTLINE_FILE, "plot_outline"),
            (config.CHARACTER_PROFILES_FILE, "character_profiles"),
            (config.WORLD_BUILDER_FILE, "world_building")]:
            data: Dict[str, Any] = {}
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
                    if not isinstance(data, dict):
                        logger.warning(f"{file_path} not a dict. Ignoring."); data = {}
                    else: logger.info(f"Successfully loaded {attr_name.replace('_', ' ')} from {file_path}")
                except Exception as e: logger.error(f"Failed to load/decode {file_path}: {e}", exc_info=True)
            else: logger.info(f"No {attr_name.replace('_', ' ')} file found ('{file_path}').")
            setattr(self, attr_name, data)
        logger.info("Finished loading existing state.")

    def _save_json_state(self):
        logger.debug("Saving agent JSON state (plot, characters, world)...")
        state_saved_count = 0
        try:
            for file_path, data_dict_attr_name in [
                (config.PLOT_OUTLINE_FILE, "plot_outline"),
                (config.CHARACTER_PROFILES_FILE, "character_profiles"),
                (config.WORLD_BUILDER_FILE, "world_building")]:
                data_dict = getattr(self, data_dict_attr_name)
                if data_dict and isinstance(data_dict, dict):
                    data_to_save = json.loads(json.dumps(data_dict)) 
                    
                    if "is_default" in data_to_save:
                        is_default_flag_value = data_to_save["is_default"]
                        is_content_truly_default = False
                        if file_path == config.PLOT_OUTLINE_FILE:
                            is_content_truly_default = (data_to_save.get("title") == config.DEFAULT_PLOT_OUTLINE_TITLE and
                                                       len(data_to_save.get("plot_points", [])) <= 5 and
                                                       data_to_save.get("protagonist_name") == config.DEFAULT_PROTAGONIST_NAME)
                        elif file_path == config.CHARACTER_PROFILES_FILE:
                            is_content_truly_default = (len(data_to_save) == 0 or (
                                len(data_to_save) == 1 and config.DEFAULT_PROTAGONIST_NAME in data_to_save and
                                data_to_save[config.DEFAULT_PROTAGONIST_NAME].get("description", "").startswith("Default:")))
                        elif file_path == config.WORLD_BUILDER_FILE:
                            is_content_truly_default = (len(data_to_save.get("locations", {})) == 1 and
                                                       "Default Location" in data_to_save.get("locations", {}) and
                                                       len(data_to_save.keys()) <= 3)
                        if not is_default_flag_value or (is_default_flag_value and not is_content_truly_default):
                            if "is_default" in data_to_save: del data_to_save["is_default"]; logger.debug(f"'is_default' flag removed from {data_dict_attr_name}.")
                        elif is_default_flag_value and is_content_truly_default: logger.debug(f"'is_default: true' retained for {data_dict_attr_name}.")
                    
                    try:
                        os.makedirs(os.path.dirname(file_path), exist_ok=True)
                        with open(file_path, 'w', encoding='utf-8') as f: json.dump(data_to_save, f, indent=2, ensure_ascii=False)
                        state_saved_count += 1
                    except Exception as e: logger.error(f"Failed to save JSON to {file_path}: {e}", exc_info=True)
                else: logger.debug(f"Skipping save for {file_path}, data empty/not dict.")
            if state_saved_count > 0: logger.info(f"JSON state saved for {state_saved_count} file(s).")
            else: logger.info("No JSON state files updated/saved.")
        except Exception as e: logger.error(f"Unexpected error during JSON state saving: {e}", exc_info=True)


    def generate_plot_outline(self, default_protagonist_name: str, unhinged_mode: bool, **kwargs) -> Dict[str, Any]:
        logger.info(f"Generating plot outline. Unhinged mode: {unhinged_mode}")
        if unhinged_mode:
            genre = kwargs.get("genre", "fantasy") 
            theme = kwargs.get("theme", "the nature of good and evil")
            setting_archetype = kwargs.get("setting_archetype", "a generic medieval kingdom")
            protagonist_archetype = kwargs.get("protagonist_archetype", "a reluctant hero")
            conflict_archetype = kwargs.get("conflict_archetype", "an external threat to the kingdom")
            
            prompt_core_elements = f"""
The novel is a '{genre}' story. Its central theme is '{theme}'.
The primary setting is inspired by: '{setting_archetype}'.
The protagonist is an archetype of: '{protagonist_archetype}'.
The main conflict revolves around: '{conflict_archetype}'.
Based on this combination, generate:
1. `title`: Compelling title.
2. `protagonist_name`: Suitable name.
3. `protagonist_description`: Brief (1-2 sentences) description.
4. `setting`: Brief (1-2 sentences) description of primary setting.
5. `conflict`: Brief (1-2 sentences) summary of main conflict.
6. `plot_points`: List of exactly 5 strings (major plot points).
7. `character_arc`: Protagonist's primary development arc string.
"""
            base_elements_for_outline = {
                "genre": genre, "theme": theme, 
                "setting_archetype_used": setting_archetype,
                "protagonist_archetype_used": protagonist_archetype,
                "conflict_archetype_used": conflict_archetype
            }
        else: 
            genre = kwargs.get("genre", config.CONFIGURED_GENRE)
            theme = kwargs.get("theme", config.CONFIGURED_THEME)
            setting_description = kwargs.get("setting_description", config.CONFIGURED_SETTING_DESCRIPTION)
            prompt_core_elements = f"""
The novel is a '{genre}' story. Its central theme is '{theme}'.
The primary setting is: '{setting_description}'.
Based on these, generate:
1. `title`: Compelling title.
2. `protagonist_name`: Suitable name.
3. `protagonist_description`: Brief (1-2 sentences) description.
4. `plot_points`: List of exactly 5 strings (major plot points).
5. `character_arc`: Protagonist's primary development arc string.
6. `conflict`: String summarizing main conflict.
"""
            base_elements_for_outline = {"genre": genre, "theme": theme, "setting": setting_description}

        prompt = f"""/no_think
        
        You are a creative assistant for narrative structure.
        {prompt_core_elements}
Output ONLY the JSON object. No intro/markdown/meta-commentary.
`plot_points` must be a JSON list of 5 strings.
Example (keys vary by mode): {{ "title": "string", ... }}
"""
        logger.info("Calling LLM for plot outline generation...")
        raw_outline_str = llm_interface.call_llm(
            model_name=config.INITIAL_SETUP_MODEL,
            prompt=prompt, 
            temperature=0.7
        )
        parsed_outline = llm_interface.parse_llm_json_response(raw_outline_str, "plot outline")

        required_keys_standard = ["title", "protagonist_name", "protagonist_description", "plot_points", "character_arc", "conflict"]
        required_keys_unhinged = required_keys_standard + ["setting"] 
        required_keys = required_keys_unhinged if unhinged_mode else required_keys_standard
        
        is_valid = False
        if parsed_outline and isinstance(parsed_outline, dict):
            plot_points = parsed_outline.get("plot_points")
            if (all(key in parsed_outline and parsed_outline[key] for key in required_keys) and
                isinstance(plot_points, list) and len(plot_points) == 5 and
                all(isinstance(p, str) and p.strip() for p in plot_points)):
                is_valid = True
                self.plot_outline = parsed_outline
                self.plot_outline.update(base_elements_for_outline)
                self.plot_outline["is_default"] = False
                logger.info(f"Successfully generated plot outline: '{self.plot_outline.get('title', 'N/A')}'")
            else:
                 logger.warning(f"Generated plot outline failed validation. Parsed: {parsed_outline}")

        if not is_valid:
            logger.error("Failed to generate valid plot outline. Applying default.")
            self.plot_outline = {
                "title": config.DEFAULT_PLOT_OUTLINE_TITLE,
                "protagonist_name": default_protagonist_name,
                "protagonist_description": "Default protagonist: A character facing challenges.",
                "plot_points": [f"Default Plot Point {i+1}" for i in range(5)],
                "character_arc": "Default character arc: The protagonist learns something.",
                "setting": "Default setting: A generic place.",
                "conflict": "Default conflict: The protagonist overcomes an obstacle.",
                "is_default": True
            }
            self.plot_outline.update(base_elements_for_outline)
        
        self.plot_outline.setdefault('protagonist_name', default_protagonist_name)
        self._save_json_state()
        return self.plot_outline

    def generate_world_building(self):
        if self.world_building and not self.world_building.get("is_default", False):
            logger.info("Skipping initial world-building: Data seems populated.")
            return

        if not self.plot_outline or not self.plot_outline.get("setting"):
            logger.error("Cannot generate world-building: Plot outline/setting missing. Applying default.")
            self.world_building = {
                "locations": {"Default Location": {"description": "A starting point."}},
                "society": {"General": {"description": "Basic societal norms."}},
                "is_default": True
            }
            self._save_json_state()
            return

        prompt = f"""/no_think
        
        You are a world-building assistant. Generate foundational elements based on this novel concept. Output MUST be a single, valid JSON object.
        Novel Concept:
        Title: {self.plot_outline.get('title', 'Untitled')}
        Genre: {self.plot_outline.get('genre', 'undefined')}
        Theme: {self.plot_outline.get('theme', 'undefined')}
        Setting Description (Expand on this): {self.plot_outline.get('setting', 'default setting')}
        Main Conflict: {self.plot_outline.get('conflict', 'default conflict')}
        Protagonist: {self.plot_outline.get('protagonist_name', 'N/A')} ({self.plot_outline.get('protagonist_description', 'N/A')})
        Instructions:
        1. Create detailed world-building for locations, society, systems (tech/magic), lore, history.
        2. Be creative, provide tangible detail, significantly expand on setting description.
        3. **CRITICAL: Output ONLY the JSON object.**
        4. Use categories: "locations", "society", "systems", "lore", "history". Items within should be dicts (e.g., with "description").
        Example Structure: {{ "locations": {{ "Loc1": {{ "description": "..." }} }}, ... }}
        """
        logger.info("Generating initial world-building data via LLM...")
        raw_world_data_str = llm_interface.call_llm(
            model_name=config.INITIAL_SETUP_MODEL,
            prompt=prompt, 
            temperature=0.6
        )
        parsed_world_data = llm_interface.parse_llm_json_response(raw_world_data_str, "initial world-building")

        is_valid = False
        if parsed_world_data and isinstance(parsed_world_data, dict):
            if any(k in parsed_world_data and isinstance(parsed_world_data[k], dict) and parsed_world_data[k] for k in ["locations", "society", "systems", "lore", "history"]):
                self.world_building = parsed_world_data
                self.world_building["is_default"] = False
                logger.info("Successfully generated initial world-building data.")
                is_valid = True
            else: logger.warning(f"Generated world-building lacks expected structure. Parsed: {parsed_world_data}")

        if not is_valid:
            logger.error("Failed to generate valid world-building. Applying default.")
            self.world_building = {
                "locations": {"Default Location": {"description": "A starting point."}},
                "society": {"General": {"description": "Basic societal norms."}},
                "is_default": True
            }
        self._save_json_state()

    def _plan_chapter(self, chapter_number: int) -> Optional[Union[str, List[SceneDetail]]]:
        if not config.ENABLE_AGENTIC_PLANNING:
            return "Agentic planning disabled by configuration."
        logger.info(f"Planning Chapter {chapter_number} with detailed scenes...")
        plot_point_focus, plot_point_index = self._get_plot_point_info(chapter_number)
        if plot_point_focus is None:
            logger.error(f"Cannot plan chapter {chapter_number}: no plot point focus.")
            return None

        context_summary = ""
        if chapter_number > 1:
            prev_chap_data = self.db_manager.get_chapter_data_from_db(chapter_number - 1)
            if prev_chap_data:
                prev_summary = prev_chap_data.get('summary')
                prev_is_provisional = prev_chap_data.get('is_provisional', False)
                summary_prefix = "[Provisional Summary from Prev Ch] " if prev_is_provisional and prev_summary else "[Summary from Prev Ch] "
                if prev_summary: context_summary += f"{summary_prefix} ({chapter_number - 1}):\n{prev_summary[:1000]}...\n"
                else:
                    prev_text = prev_chap_data.get('text', '')
                    text_prefix = "[Provisional Text Snippet from Prev Ch] " if prev_is_provisional and prev_text else "[Text Snippet from Prev Ch] "
                    if prev_text: context_summary += f"{text_prefix} ({chapter_number - 1}):\n...{prev_text[-1000:]}\n"
        
        kg_facts = []
        protagonist_name = self.plot_outline.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME)
        kg_chapter_limit = chapter_number - 1 
        current_loc = self.db_manager.get_most_recent_value(protagonist_name, "located_in", kg_chapter_limit, include_provisional=False)
        current_status = self.db_manager.get_most_recent_value(protagonist_name, "status_is", kg_chapter_limit, include_provisional=False)
        if current_loc: kg_facts.append(f"- {protagonist_name} is currently located in (reliable KG): {current_loc}.")
        if current_status: kg_facts.append(f"- {protagonist_name}'s current status (reliable KG): {current_status}.")
        kg_context_section = "**Relevant Reliable KG Facts (up to prev chapter/pre-novel):**\n" + "\n".join(kg_facts) + "\n" if kg_facts else ""

        prompt = f"""/no_think
        
        You are a master plotter outlining **8-15 detailed scenes** for Chapter {chapter_number}.
        **Novel Concept:** Title: {self.plot_outline.get('title', 'Untitled')}, Genre: {self.plot_outline.get('genre', 'N/A')}, Theme: {self.plot_outline.get('theme', 'N/A')}, Protagonist: {protagonist_name}, Arc: {self.plot_outline.get('character_arc', 'N/A')}
        **Mandatory Focus for THIS Chapter (Plot Point {plot_point_index + 1}):** {plot_point_focus}
        **Recent Context:** {context_summary if context_summary else "This is the first chapter or no prior summary."}
        {kg_context_section}
        **Current Character States (Key Characters):** {self._get_relevant_character_state_snippet(chapter_number)} 
        **Current World State (Relevant Locations/Elements):** {self._get_relevant_world_state_snippet(chapter_number)}
        **Task:** Create 8-15 detailed scenes for Chapter {chapter_number}. Each scene MUST:
        1. Advance **Mandatory Focus**. 2. Follow **Recent Context** & **KG Facts**. 3. Involve relevant characters/world elements. 4. Contribute to **Protagonist Arc** or plot. 5. Be distinct.
        **Output Format:** ONLY a single, valid JSON list of scene objects. Each object keys: `scene_number` (int), `summary` (str, 1-2 sentences), `characters_involved` (list[str]), `key_dialogue_points` (list[str], 1-3 brief points), `setting_details` (str), `contribution` (str).
        **Example JSON Scene (part of list):**
        ```json
        {{ "scene_number": 1, "summary": "Protagonist finds cryptic message.", "characters_involved": ["{protagonist_name}"], "key_dialogue_points": ["'What is this?'"], "setting_details": "Dusty attic.", "contribution": "Inciting incident for chapter."}}
        ```
        Output JSON list `[...]` only.
        [
        """
        logger.info(f"Calling LLM ({config.PLANNING_MODEL}) for detailed scene plan for chapter {chapter_number}...")
        plan_raw = llm_interface.call_llm(
            model_name=config.PLANNING_MODEL,
            prompt=prompt, 
            temperature=0.65, 
            max_tokens=config.MAX_PLANNING_TOKENS
        )
        
        parsed_plan: Optional[List[SceneDetail]] = llm_interface.parse_llm_json_response(
            plan_raw, f"detailed scene plan for chapter {chapter_number}", expect_type=list
        )

        if parsed_plan and isinstance(parsed_plan, list) and len(parsed_plan) >= 1:
            valid_scenes = []
            required_scene_keys = {"scene_number", "summary", "characters_involved", "key_dialogue_points", "setting_details", "contribution"}
            for i, scene_item in enumerate(parsed_plan):
                if isinstance(scene_item, dict) and required_scene_keys.issubset(scene_item.keys()):
                    if not all(isinstance(scene_item[k], str) for k in ["summary", "setting_details", "contribution"]): continue
                    if not isinstance(scene_item["characters_involved"], list) or not isinstance(scene_item["key_dialogue_points"], list): continue
                    valid_scenes.append(scene_item)
                else: logger.warning(f"Scene {i+1} in plan for ch {chapter_number} has missing keys. Skipping.")
            
            if valid_scenes and len(valid_scenes) >= 1:
                logger.info(f"Generated detailed scene plan for chapter {chapter_number} with {len(valid_scenes)} scenes.")
                return valid_scenes
            else:
                logger.error(f"Failed: All parsed scenes invalid for chapter {chapter_number}. Raw: '{plan_raw[:500]}...'")
                self._save_debug_output(chapter_number, "detailed_plan_invalid_scenes", plan_raw)
                return None
        else:
            logger.error(f"Failed to generate/parse valid scene plan (JSON list) for chapter {chapter_number}. Raw: '{plan_raw[:500]}...'")
            self._save_debug_output(chapter_number, "detailed_plan_parse_fail", plan_raw)
            return None

    def write_chapter(self, chapter_number: int) -> Optional[str]:
        logger.info(f"=== Starting Chapter {chapter_number} Generation ===")
        if not self.plot_outline or not self.plot_outline.get("plot_points") or not self.plot_outline.get("protagonist_name"):
            logger.error(f"Cannot write Ch {chapter_number}: Plot outline/points/protagonist missing.")
            return None
        if chapter_number <= 0: 
            logger.error(f"Cannot write Ch {chapter_number}: Chapter number must be positive.")
            return None

        chapter_plan_obj: Optional[Union[str, List[SceneDetail]]] = self._plan_chapter(chapter_number)
        
        if config.ENABLE_AGENTIC_PLANNING:
            if chapter_plan_obj is None:
                logger.error(f"Ch {chapter_number} generation halted: planning failure (plan is None).")
                return None
            if not isinstance(chapter_plan_obj, list) and chapter_plan_obj != "Agentic planning disabled by configuration.":
                logger.error(f"Ch {chapter_number} generation halted: invalid plan type {type(chapter_plan_obj)}.")
                return None

        context_for_draft = self._get_context(chapter_number)
        plot_point_focus, _ = self._get_plot_point_info(chapter_number)

        initial_draft_text, initial_raw_text = self._generate_draft(
            chapter_number, plot_point_focus, context_for_draft, chapter_plan_obj
        )

        if not initial_draft_text:
            logger.error(f"Failed to generate initial draft for ch {chapter_number}.")
            self._save_debug_output(chapter_number, "initial_raw_fail_after_clean", initial_raw_text or "")
            return None

        evaluation = self._evaluate_draft(initial_draft_text, chapter_number, context_for_draft)
        current_text, final_raw_output_log = initial_draft_text, f"--- INITIAL DRAFT RAW ---\n{initial_raw_text}\n\n"
        proceeded_with_flaws = False 

        if evaluation["needs_revision"]:
            revision_reason_str = "\n- ".join(evaluation["reasons"])
            logger.warning(f"Ch {chapter_number} flagged for revision. Reason(s):\n- {revision_reason_str}")
            revised_text_tuple = self._revise_chapter(
                current_text, chapter_number, revision_reason_str, context_for_draft, chapter_plan_obj
            )
            if revised_text_tuple:
                revised_text, raw_revision_output = revised_text_tuple
                logger.info(f"Revision successful for ch {chapter_number}. Evaluating revised draft...")
                revised_evaluation = self._evaluate_draft(revised_text, chapter_number, context_for_draft)
                if revised_evaluation["needs_revision"]:
                    logger.error(f"Revised draft for ch {chapter_number} STILL failed. Reasons:\n- " + "\n- ".join(revised_evaluation["reasons"]))
                    proceeded_with_flaws = True 
                else: logger.info(f"Revised draft for ch {chapter_number} passed evaluation.")
                current_text = revised_text 
                final_raw_output_log += f"--- REVISION (Reason: {revision_reason_str}) ---\n{raw_revision_output}\n\n"
            else:
                logger.error(f"Revision failed for ch {chapter_number}. Proceeding with original flawed draft.")
                proceeded_with_flaws = True
                final_raw_output_log += f"--- REVISION FAILED (Reason: {revision_reason_str}) ---\n\n"
        else: logger.info(f"Initial draft for ch {chapter_number} passed evaluation.")

        if not self._finalize_chapter_core(chapter_number, current_text, final_raw_output_log, proceeded_with_flaws):
             logger.error(f"=== Finished Ch {chapter_number} With Errors During Core Finalization ===")
             return None

        self._update_knowledge_bases(chapter_number, current_text, proceeded_with_flaws)
        
        self.chapter_count = max(self.chapter_count, chapter_number)
        self._save_json_state() 
        logger.info(f"=== Finished Ch {chapter_number} Successfully (Proceeded with flaws: {proceeded_with_flaws}) ===")
        return current_text

    def _get_plot_point_info(self, chapter_number: int) -> Tuple[Optional[str], int]:
        plot_points = self.plot_outline.get("plot_points", [])
        if chapter_number <= 0: 
            logger.warning(f"Invalid chapter number {chapter_number} for plot point lookup.")
            return None, -1
        
        plot_point_index = min(chapter_number - 1, len(plot_points) - 1) if plot_points else -1
        if 0 <= plot_point_index < len(plot_points):
            return plot_points[plot_point_index], plot_point_index
        else:
            logger.warning(f"Could not find plot point for chapter {chapter_number}.")
            return None, -1

    def _generate_draft(self, chapter_number: int, plot_point_focus: Optional[str], context: str, chapter_plan: Optional[Union[str, List[SceneDetail]]]) -> Tuple[Optional[str], Optional[str]]:
        if not plot_point_focus:
            plot_point_focus = "Continue narrative logically, focusing on character development and plot progression."
        
        plan_section_for_prompt = ""
        if config.ENABLE_AGENTIC_PLANNING:
            if isinstance(chapter_plan, str): 
                plan_section_for_prompt = f"**Chapter Plan Note:**\n{chapter_plan}\n"
            elif isinstance(chapter_plan, list) and chapter_plan: 
                try:
                    plan_json_str = json.dumps(chapter_plan, indent=2, ensure_ascii=False)
                    plan_section_for_prompt = f"**Detailed Scene Plan (MUST FOLLOW CLOSELY):**\n```json\n{plan_json_str}\n```\n"
                    logger.info(f"Using detailed scene plan for Ch {chapter_number} draft.")
                except TypeError as e:
                    logger.error(f"Could not serialize chapter plan to JSON: {e}. Plan: {chapter_plan}")
                    plan_section_for_prompt = "**Chapter Plan Note:** Error formatting plan. Rely on Plot Point Focus.\n"
            else: plan_section_for_prompt = "**Chapter Plan Note:** No detailed plan. Rely on Plot Point Focus.\n"
        else: plan_section_for_prompt = "**Chapter Plan Note:** Agentic planning disabled. Rely on Plot Point Focus.\n"
            
        char_profiles_json = json.dumps(self._get_filtered_profiles_for_prompt(chapter_number -1), indent=2, ensure_ascii=False, default=str)
        world_building_json = json.dumps(self._get_filtered_world_for_prompt(chapter_number -1), indent=2, ensure_ascii=False, default=str)

        prompt = f"""/no_think
        
        You are an expert novelist writing Chapter {chapter_number} of "{self.plot_outline.get('title', 'Untitled Novel')}".
        **Story Bible:** Genre: {self.plot_outline.get('genre', 'N/A')}, Theme: {self.plot_outline.get('theme', 'N/A')}, Protagonist: {self.plot_outline.get('protagonist_name', 'N/A')}, Arc: {self.plot_outline.get('character_arc', 'N/A')}
        **Overall Plot Point Focus for THIS Chapter:** {plot_point_focus}
        {plan_section_for_prompt}
        **World Building (JSON - Note provisional info):** ```json\n{world_building_json}\n```
        **Character Profiles (JSON - Note provisional info):** ```json\n{char_profiles_json}\n```
        **Context from Previous Chapters (Note provisional summaries):**\n--- BEGIN CONTEXT ---\n{context if context else "No previous context."}\n--- END CONTEXT ---
        **Instructions:** Write compelling chapter (target {config.MIN_ACCEPTABLE_DRAFT_LENGTH}+ chars). Follow detailed scene plan if provided, otherwise Plot Point Focus. Maintain consistency. Smooth flow. Vivid prose for genre '{self.plot_outline.get('genre', 'story')}'. **Output ONLY chapter text.** No "Chapter X" headers or meta-commentary.
        --- BEGIN CHAPTER {chapter_number} TEXT ---
        """
        raw_text = llm_interface.call_llm(
            model_name=config.DRAFTING_MODEL,
            prompt=prompt, 
            temperature=0.65
        )
        if not raw_text: return None, None
        cleaned_text = llm_interface.clean_model_response(raw_text)
        if not cleaned_text or len(cleaned_text) < config.MIN_ACCEPTABLE_DRAFT_LENGTH:
             logger.error(f"Ch {chapter_number} draft too short ({len(cleaned_text or '')} chars). Raw: '{raw_text[:200]}...'")
             return None, raw_text
        logger.info(f"Generated initial draft for ch {chapter_number} (Len: {len(cleaned_text)}).")
        return cleaned_text, raw_text

    def _evaluate_draft(self, draft_text: str, chapter_number: int, previous_chapters_context: str) -> EvaluationResult:
        logger.info(f"Evaluating chapter {chapter_number} draft...")
        reasons: List[str] = []
        needs_revision = False
        coherence_score, consistency_issues, plot_deviation_reason = None, None, None

        if chapter_number > 1:
            current_embedding = llm_interface.get_embedding(draft_text)
            prev_embedding = self.db_manager.get_embedding_from_db(chapter_number - 1)
            if current_embedding is not None and prev_embedding is not None:
                coherence_score = utils.numpy_cosine_similarity(current_embedding, prev_embedding)
                logger.info(f"Coherence with prev ch ({chapter_number-1}): {coherence_score:.4f}")
                if coherence_score < config.REVISION_COHERENCE_THRESHOLD:
                    needs_revision = True; reasons.append(f"Low coherence (Score: {coherence_score:.4f}).")
            else: logger.warning(f"Could not perform coherence check for ch {chapter_number}.")
        else: logger.info("Skipping coherence check for Chapter 1.")

        if config.REVISION_CONSISTENCY_TRIGGER:
            consistency_issues = self._check_consistency(draft_text, chapter_number, previous_chapters_context)
            if consistency_issues: needs_revision = True; reasons.append(f"Consistency issues:\n{consistency_issues}")
        
        if config.PLOT_ARC_VALIDATION_TRIGGER:
            plot_deviation_reason = self._validate_plot_arc(draft_text, chapter_number)
            if plot_deviation_reason: needs_revision = True; reasons.append(f"Plot Arc Deviation: {plot_deviation_reason}")

        if len(draft_text) < config.MIN_ACCEPTABLE_DRAFT_LENGTH:
            needs_revision = True; reasons.append(f"Draft too short ({len(draft_text)} chars).")
            
        logger.info(f"Evaluation for Ch {chapter_number}. Needs revision: {needs_revision}.")
        return {"needs_revision": needs_revision, "reasons": reasons, "coherence_score": coherence_score, "consistency_issues": consistency_issues, "plot_deviation_reason": plot_deviation_reason}

    def _revise_chapter(self, original_text: str, chapter_number: int, reason: str, context_from_previous: str, chapter_plan: Optional[Union[str, List[SceneDetail]]]) -> Optional[Tuple[str, str]]:
        if not original_text or not reason: logger.error(f"Revision for ch {chapter_number} missing text/reason."); return None
        clean_reason = llm_interface.clean_model_response(reason).strip()
        if not clean_reason: logger.error(f"Revision reason for ch {chapter_number} empty."); return None
            
        logger.warning(f"Attempting revision for chapter {chapter_number}. Reason:\n{clean_reason}")
        context_limit, original_text_limit = config.MAX_CONTEXT_LENGTH // 3, config.MAX_CONTEXT_LENGTH // 3
        context_snippet = context_from_previous[:context_limit]
        original_snippet = original_text[:original_text_limit]
        
        plan_focus_section = ""
        if config.ENABLE_AGENTIC_PLANNING:
            if isinstance(chapter_plan, str): plan_focus_section = f"**Original Plan Note (Target):**\n{chapter_plan}\n"
            elif isinstance(chapter_plan, list) and chapter_plan:
                try:
                    plan_json_str = json.dumps(chapter_plan, indent=2, ensure_ascii=False)
                    plan_snippet_for_prompt = plan_json_str[:(config.MAX_CONTEXT_LENGTH // 4)]
                    if len(plan_json_str) > len(plan_snippet_for_prompt): plan_snippet_for_prompt += "\n... (plan truncated)"
                    plan_focus_section = f"**Original Scene Plan (Target - align with this while fixing issues):**\n```json\n{plan_snippet_for_prompt}\n```\n"
                except TypeError:
                     plot_point_focus, _ = self._get_plot_point_info(chapter_number)
                     plan_focus_section = f"**Original Chapter Focus (Target):**\n{plot_point_focus}\n"
            else: 
                plot_point_focus, _ = self._get_plot_point_info(chapter_number)
                plan_focus_section = f"**Original Chapter Focus (Target):**\n{plot_point_focus}\n"
        else:
            plot_point_focus, _ = self._get_plot_point_info(chapter_number)
            plan_focus_section = f"**Original Chapter Focus (Target):**\n{plot_point_focus}\n"
            
        protagonist_name = self.plot_outline.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME)
        prompt = f"""/no_think
        
        You are a skilled revising author rewriting Chapter {chapter_number} (protagonist: {protagonist_name}).
        **Critique/Reason(s) for Revision (MUST address):**\n--- FEEDBACK START ---\n{clean_reason}\n--- FEEDBACK END ---\n
        {plan_focus_section}
        **Context from Previous Chapters:**\n--- BEGIN CONTEXT ---\n{context_snippet if context_snippet else "No previous context."}\n--- END CONTEXT ---
        **Original Draft Snippet (Reference ONLY - focus on critique & plan):**\n--- BEGIN ORIGINAL DRAFT ---\n{original_snippet}\n--- END ORIGINAL DRAFT ---
        **Instructions:** 1. **PRIORITY:** Fix critique issues. 2. **Rewrite ENTIRE chapter**, aligning with Original Plan/Focus. 3. Ensure flow with Context. 4. Preserve tone/style. 5. Aim for {config.MIN_ACCEPTABLE_DRAFT_LENGTH}+ chars. 6. **Output ONLY rewritten chapter text.**
        --- BEGIN REVISED CHAPTER {chapter_number} TEXT ---
        """
        revised_raw = llm_interface.call_llm(
            model_name=config.REVISION_MODEL,
            prompt=prompt, 
            temperature=0.6
        ) 
        if not revised_raw: logger.error(f"Revision call failed for ch {chapter_number}."); return None
            
        revised_cleaned = llm_interface.clean_model_response(revised_raw)
        if not revised_cleaned or len(revised_cleaned) < config.MIN_ACCEPTABLE_DRAFT_LENGTH:
            logger.error(f"Revision for ch {chapter_number} too short ({len(revised_cleaned or '')} chars)."); self._save_debug_output(chapter_number, "revision_raw_fail_short", revised_raw); return None
            
        original_embedding = llm_interface.get_embedding(original_text)
        revised_embedding = llm_interface.get_embedding(revised_cleaned)
        if original_embedding is not None and revised_embedding is not None:
            similarity_score = utils.numpy_cosine_similarity(original_embedding, revised_embedding)
            logger.info(f"Revision similarity: {similarity_score:.4f}")
            if similarity_score >= config.REVISION_SIMILARITY_ACCEPTANCE:
                logger.warning(f"Revision for ch {chapter_number} rejected: Too similar (Score: {similarity_score:.4f})."); self._save_debug_output(chapter_number, "revision_raw_rejected_similar", revised_raw); return None
            logger.info(f"Revision for ch {chapter_number} accepted (Similarity: {similarity_score:.4f}).")
            return revised_cleaned, revised_raw
        else:
            logger.warning(f"Could not get embeddings for revision similarity check of ch {chapter_number}. Accepting."); return revised_cleaned, revised_raw

    def _finalize_chapter_core(self, chapter_number: int, final_text: str, raw_log: str, from_flawed_draft: bool) -> bool:
        logger.info(f"Finalizing chapter {chapter_number} (From flawed draft: {from_flawed_draft})...")
        if not final_text: logger.error(f"Cannot finalize ch {chapter_number}: Final text missing."); return False

        summary = self._summarize_chapter(final_text, chapter_number)
        final_embedding = llm_interface.get_embedding(final_text)
        if final_embedding is None: logger.error(f"CRITICAL: Failed embedding final Ch {chapter_number}.")

        try:
            self.db_manager.save_chapter_data(chapter_number, final_text, raw_log, summary, final_embedding, from_flawed_draft)
        except Exception as e: logger.error(f"DB save failed for ch {chapter_number}: {e}", exc_info=True); return False

        try:
            with open(os.path.join(config.OUTPUT_DIR, "chapters", f"chapter_{chapter_number}.txt"), 'w', encoding='utf-8') as f: f.write(final_text)
            with open(os.path.join(config.OUTPUT_DIR, "chapter_logs", f"chapter_{chapter_number}_raw_log.txt"), 'w', encoding='utf-8') as f: f.write(raw_log)
            logger.info(f"Saved chapter text & raw log for ch {chapter_number}.")
        except IOError as e: logger.error(f"Failed writing chapter text/log for ch {chapter_number}: {e}", exc_info=True)
        
        logger.info(f"Core finalization complete for ch {chapter_number}.")
        return True

    def _update_knowledge_bases(self, chapter_number: int, final_text: str, from_flawed_draft: bool):
        if not final_text: logger.warning(f"Skipping knowledge base update for ch {chapter_number}: Final text missing."); return
        logger.info(f"Updating knowledge bases for ch {chapter_number} (From flawed draft: {from_flawed_draft})...")
        try:
            self._update_character_and_world_json_from_chapter(final_text, chapter_number, from_flawed_draft)
            logger.info(f"JSON knowledge bases (char/world) updated for ch {chapter_number}.")
            
            self._extract_and_update_kg(final_text, chapter_number, from_flawed_draft)
            logger.info(f"Knowledge Graph updated for ch {chapter_number}.")
        except Exception as e: logger.error(f"Error during knowledge base update for ch {chapter_number}: {e}", exc_info=True)

    def _get_context(self, current_chapter_number: int) -> str:
        if current_chapter_number <= 1: return ""
        logger.debug(f"Retrieving context for Chapter {current_chapter_number}...")
        plot_point_focus, plot_point_index = self._get_plot_point_info(current_chapter_number)
        context_query_text = plot_point_focus if plot_point_focus else f"Narrative context for chapter {current_chapter_number}."
        
        if plot_point_focus: logger.info(f"Context query for ch {current_chapter_number} from Plot Point {plot_point_index + 1}: '{context_query_text[:100]}...'")
        else: logger.warning(f"No plot point for ch {current_chapter_number}. Generic context query.")
            
        query_embedding = llm_interface.get_embedding(context_query_text)
        
        if query_embedding is None:
            logger.warning("Failed embedding context query. Falling back to prev ch summary/text.")
            prev_chap_db_data = self.db_manager.get_chapter_data_from_db(current_chapter_number - 1)
            if prev_chap_db_data:
                is_prov = prev_chap_db_data.get('is_provisional', False)
                fallback_content = prev_chap_db_data.get('summary') or prev_chap_db_data.get('text', '')
                if fallback_content:
                    prefix = "[Provisional Fallback] " if is_prov else "[Fallback] "
                    logger.info(f"Using fallback context: {prefix} ({len(fallback_content)} chars).")
                    return (prefix + fallback_content)[:config.MAX_CONTEXT_LENGTH]
            logger.warning("Fallback context retrieval failed."); return ""
            
        past_embeddings = self.db_manager.get_all_past_embeddings(current_chapter_number)
        if not past_embeddings: return ""
        
        similarities = sorted([(chap_num, utils.numpy_cosine_similarity(query_embedding, emb)) for chap_num, emb in past_embeddings if emb is not None], key=lambda item: item[1], reverse=True)
        if not similarities: return ""
        
        top_n_indices = [cs[0] for cs in similarities[:config.CONTEXT_CHAPTER_COUNT]]
        logger.info(f"Top {len(top_n_indices)} relevant chapters for context: {top_n_indices} (Scores: {[f'{s:.3f}' for _, s in similarities[:config.CONTEXT_CHAPTER_COUNT]]})")
        
        if (current_chapter_number - 1) > 0 and (current_chapter_number - 1) not in top_n_indices:
            top_n_indices.append(current_chapter_number - 1) 
            
        context_parts, total_chars, chapters_to_fetch = [], 0, sorted(list(set(top_n_indices)))
        logger.debug(f"Fetching context data for chapters: {chapters_to_fetch}")
        
        for chap_num in chapters_to_fetch:
            if total_chars >= config.MAX_CONTEXT_LENGTH: break
            chap_data_row = self.db_manager.get_chapter_data_from_db(chap_num) 
            if chap_data_row:
                content = (chap_data_row.get('summary') or chap_data_row.get('text', '')).strip()
                is_prov = chap_data_row.get('is_provisional', False)
                content_type = "Provisional Summary" if chap_data_row.get('summary') and is_prov else "Summary" if chap_data_row.get('summary') else "Provisional Text Snippet" if is_prov else "Text Snippet"
                
                if content:
                    prefix = f"[From Chapter {chap_num} ({content_type})]:\n"; suffix = "\n---\n"
                    formatting_chars = len(prefix) + len(suffix) + 10 
                    content_limit = max(0, config.MAX_CONTEXT_LENGTH - total_chars - formatting_chars)
                    if content_limit > 0:
                        truncated_content = content[:content_limit]
                        context_parts.append(f"{prefix}{truncated_content}{suffix}")
                        total_chars += len(truncated_content) + formatting_chars
                        logger.debug(f"Added context from ch {chap_num} ({content_type}), {len(truncated_content)} chars. Total: {total_chars}.")
            else: logger.warning(f"Could not retrieve chapter data for {chap_num} for context.")
                
        final_context = "\n".join(context_parts).strip()
        logger.info(f"Constructed final semantic context: {len(final_context)} chars from chapters {chapters_to_fetch}.")
        return final_context

    def _summarize_chapter(self, chapter_text: Optional[str], chapter_number: int) -> Optional[str]:
        if not chapter_text or len(chapter_text) < 50: return None
        snippet = chapter_text[:config.KNOWLEDGE_UPDATE_SNIPPET_SIZE] # Snippet size for summary can be smaller
        prompt = f"""/no_think
        
        Summarize Chapter {chapter_number} (1-3 sentences), crucial plot advancements, character decisions, or revelations. Be succinct.
        Chapter Text Snippet:\n--- BEGIN TEXT ---\n{snippet}\n--- END TEXT ---\nOutput ONLY summary text.
        """
        summary_raw = llm_interface.call_llm(
            model_name=config.SUMMARIZATION_MODEL,
            prompt=prompt, 
            temperature=0.6, 
            max_tokens=config.MAX_SUMMARY_TOKENS
        )
        cleaned_summary = llm_interface.clean_model_response(summary_raw).strip()
        if cleaned_summary: logger.info(f"Generated summary for ch {chapter_number}: '{cleaned_summary[:100]}...'"); return cleaned_summary
        else: logger.warning(f"Failed to generate valid summary for ch {chapter_number}."); return None

    def _check_consistency(self, chapter_draft_text: Optional[str], chapter_number: int, previous_chapters_context: str) -> Optional[str]:
        if not chapter_draft_text: return None
        
        draft_snippet = chapter_draft_text[:config.KNOWLEDGE_UPDATE_SNIPPET_SIZE]
        context_snippet = previous_chapters_context[:config.KNOWLEDGE_UPDATE_SNIPPET_SIZE // 2]
        kg_facts_for_consistency_prompt: List[str] = []
        protagonist_name = self.plot_outline.get("protagonist_name")
        kg_chapter_limit = chapter_number - 1

        if protagonist_name:
            logger.debug(f"Gathering reliable KG facts up to chapter_added={kg_chapter_limit} for Ch {chapter_number} consistency...")
            loc = self.db_manager.get_most_recent_value(protagonist_name, "located_in", kg_chapter_limit, include_provisional=False)
            if loc: kg_facts_for_consistency_prompt.append(f"- {protagonist_name} last reliably at: {loc}.")
            status = self.db_manager.get_most_recent_value(protagonist_name, "status_is", kg_chapter_limit, include_provisional=False)
            if status: kg_facts_for_consistency_prompt.append(f"- {protagonist_name}'s last reliable status: {status}.")
            
        kg_check_results_text = "**Key Reliable KG Facts (from pre-novel & prev chapters):**\n" + "\n".join(kg_facts_for_consistency_prompt) + "\n" if kg_facts_for_consistency_prompt else "**Key Reliable KG Facts:** None available.\n"

        char_profiles_for_prompt = self._get_filtered_profiles_for_prompt(kg_chapter_limit)
        world_building_for_prompt = self._get_filtered_world_for_prompt(kg_chapter_limit)

        prompt = f"""/no_think
        
        Continuity Editor: Analyze Chapter {chapter_number} Draft Snippet.
        Compare against: Plot Outline, Character Profiles (canon, note provisional), World Building (canon, note provisional), {kg_check_results_text} (crucial established facts), Previous Context (flow), Draft's internal consistency.
        List ONLY specific, objective contradictions/deviations. Prioritize facts from Plot, Profiles, World, KG. If NO issues, respond ONLY: None
        **Plot Outline:** ```json\n{json.dumps(self.plot_outline, indent=2, ensure_ascii=False, default=str)}\n```
        **Character Profiles:** ```json\n{json.dumps(char_profiles_for_prompt, indent=2, ensure_ascii=False, default=str)}\n```
        **World Building:** ```json\n{json.dumps(world_building_for_prompt, indent=2, ensure_ascii=False, default=str)}\n```
        **Previous Context (Snippet):**\n--- PREVIOUS CONTEXT ---\n{context_snippet if context_snippet else "N/A (e.g., Ch 1)."}\n--- END PREVIOUS CONTEXT ---
        **Chapter {chapter_number} Draft (to analyze):**\n--- DRAFT ---\n{draft_snippet}\n--- END DRAFT ---
        **Inconsistencies (or "None"):**
        """
        response_raw = llm_interface.call_llm(
            model_name=config.CONSISTENCY_CHECK_MODEL,
            prompt=prompt, 
            temperature=0.5, # Lower temp for factual analysis
            max_tokens=config.MAX_CONSISTENCY_TOKENS
        ) 
        response_cleaned = llm_interface.clean_model_response(response_raw).strip()

        if not response_cleaned or response_cleaned.lower() == "none": logger.info(f"Consistency check passed for ch {chapter_number}."); return None
        else: logger.warning(f"Consistency issues for ch {chapter_number}:\n{response_cleaned}"); return response_cleaned

    def _validate_plot_arc(self, chapter_draft_text: Optional[str], chapter_number: int) -> Optional[str]:
        if not chapter_draft_text: return None
        plot_point_focus, plot_point_index = self._get_plot_point_info(chapter_number)
        if plot_point_focus is None: logger.warning(f"Plot arc validation skipped for ch {chapter_number}: No plot point focus."); return None 
        
        logger.info(f"Validating plot arc for ch {chapter_number} against Plot Point {plot_point_index + 1}: '{plot_point_focus[:100]}...'")
        summary = self._summarize_chapter(chapter_draft_text, chapter_number) # Uses SUMMARIZATION_MODEL
        validation_text = summary if summary and len(summary) > 50 else chapter_draft_text[:1500]
        if not validation_text: return None
            
        protagonist_name = self.plot_outline.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME)
        prompt = f"""/no_think
        
        Story Analyst: Does Chapter {chapter_number} Text (protagonist: {protagonist_name}) address its Intended Plot Point?
        **Intended Plot Point ({plot_point_index + 1} for Ch {chapter_number}):** "{plot_point_focus}"
        **Chapter {chapter_number} Text (Summary/Snippet):** "{validation_text}"
        **Evaluation:** Does Chapter Text align with Intended Plot Point?
        **CRITICAL:** Respond ONLY `Yes` (aligns) OR `No, because...` (deviates, 1-2 sentence explanation).
        Response:"""
        validation_response_raw = llm_interface.call_llm(
            model_name=config.INITIAL_SETUP_MODEL, # Can use a medium model for this
            prompt=prompt, 
            temperature=0.5, 
            max_tokens=config.MAX_PLOT_VALIDATION_TOKENS
        ) 
        cleaned_plot_response = llm_interface.clean_model_response(validation_response_raw).strip()
        
        if cleaned_plot_response.lower().startswith("yes"): logger.info(f"Plot arc validation passed for ch {chapter_number}."); return None
        elif cleaned_plot_response.lower().startswith("no, because"):
            reason = cleaned_plot_response[len("no, because"):].strip() or "LLM indicated deviation, no specific reason."
            logger.warning(f"Plot arc deviation for ch {chapter_number}: {reason}"); return reason
        else:
            logger.warning(f"Plot arc validation for ch {chapter_number} ambiguous: '{cleaned_plot_response}'. Assuming alignment."); return None 

    def _update_character_and_world_json_from_chapter(self, chapter_text: Optional[str], chapter_number: int, from_flawed_draft: bool):
        if not chapter_text or len(chapter_text) < 100: # Basic heuristic
            logger.info(f"Skipping JSON knowledge update for ch {chapter_number}: Text too short or None.")
            return

        # Heuristic: Check for mentions of known characters or locations
        known_char_names = list(self.character_profiles.keys())
        known_loc_names = list(self.world_building.get("locations", {}).keys())
        
        mentioned_entities = []
        for name in known_char_names:
            if name.lower() in chapter_text.lower(): # Simple case-insensitive check
                mentioned_entities.append(name)
        for name in known_loc_names:
            if name.lower() in chapter_text.lower():
                 mentioned_entities.append(name)
        
        if not mentioned_entities and chapter_number > 1 : # Allow updates for Ch1 even if no prior known entities
             # A more sophisticated check could be for NEW entities, but this is simpler for now.
             logger.info(f"Skipping JSON knowledge update for ch {chapter_number}: No known characters or locations mentioned significantly (heuristic).")
             return

        logger.info(f"Attempting combined JSON (char/world) update for ch {chapter_number} (Flawed draft: {from_flawed_draft}). Mentions: {mentioned_entities[:5]}")
        text_snippet = chapter_text[:config.KNOWLEDGE_UPDATE_SNIPPET_SIZE]
        protagonist_name = self.plot_outline.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME)
        
        dynamic_instructions_char = ""
        dynamic_instructions_world = ""
        if config.ENABLE_DYNAMIC_STATE_ADAPTATION:
            dynamic_instructions_char = """For existing characters, propose modifications to `traits` or `description` using `"modification_proposal"` field. Ex: `"modification_proposal": "MODIFY traits: ADD 'Determined'"`. Only include characters updated, newly introduced, or with a modification proposal."""
            dynamic_instructions_world = """For existing world items, propose modifications using `"modification_proposal"`. Ex: `"modification_proposal": "MODIFY atmosphere: 'Now heavy.'"`. Only include world elements (locations, society items, etc.) that are new, significantly changed, or have a modification proposal."""
        else:
            dynamic_instructions_char = "Only include characters updated or newly introduced."
            dynamic_instructions_world = "Only include world elements new or significantly changed."
            
        current_profiles_for_prompt = self._get_filtered_profiles_for_prompt(chapter_number -1)
        current_world_for_prompt = self._get_filtered_world_for_prompt(chapter_number -1)

        prompt = f"""/no_think
        
        You are a literary analyst. Analyze Chapter {chapter_number} snippet (protagonist: {protagonist_name}) for updates to character profiles AND world-building details.
        Output MUST be a single, valid JSON object with two top-level keys: "character_updates" and "world_building_updates".

        **Chapter Text Snippet:**\n--- BEGIN TEXT ---\n{text_snippet}...\n--- END TEXT ---\n
        
        **Current Character Profiles (reference):**\n```json\n{json.dumps(current_profiles_for_prompt, indent=2, ensure_ascii=False, default=str)}\n```
        **Character Update Instructions:**
        1. Identify characters updated/introduced in THIS chapter.
        2. For each, note new traits, relationship changes, status, description.
        3. Add `development_in_chapter_{chapter_number}` key summarizing their role/change in THIS chapter.
        4. {dynamic_instructions_char}
        5. The value of "character_updates" should be a JSON object where keys are character names. If no char updates, use `{{}}`.

        **Current World Building Notes (reference):**\n```json\n{json.dumps(current_world_for_prompt, indent=2, ensure_ascii=False, default=str)}\n```
        **World Building Update Instructions:**
        1. Identify new/changed locations, society elements, systems, lore, history from THIS chapter.
        2. For each, provide relevant details (description, atmosphere, goals, rules, text, etc.).
        3. Add `elaboration_in_chapter_{chapter_number}` key for context from THIS chapter.
        4. {dynamic_instructions_world}
        5. The value of "world_building_updates" should be a JSON object with top-level keys like "locations", "society", etc. If no world updates, use `{{}}`.

        **CRITICAL: Output ONLY the combined JSON object.**
        Example Output:
        ```json
        {{
          "character_updates": {{
            "Char1": {{ "traits": ["NewTrait"], "modification_proposal": "MODIFY status: 'Injured'", "development_in_chapter_{chapter_number}": "Fought bravely."}}
          }},
          "world_building_updates": {{
            "locations": {{ "NewCave": {{ "description": "A dark, mysterious cave.", "elaboration_in_chapter_{chapter_number}": "Discovered by Char1."}} }},
            "systems": {{ "AncientMagic": {{ "modification_proposal": "MODIFY rules: 'Now weaker near iron.'" }} }}
          }}
        }}
        ```
        """
        raw_analysis = llm_interface.call_llm(
            model_name=config.KNOWLEDGE_UPDATE_MODEL,
            prompt=prompt, 
            temperature=0.5 # More factual
        )
        combined_updates = llm_interface.parse_llm_json_response(raw_analysis, f"combined char/world update for ch {chapter_number}")

        if not combined_updates or not isinstance(combined_updates, dict):
            logger.warning(f"LLM found no combined char/world JSON updates in ch {chapter_number} or parse failed.")
            return

        char_updates = combined_updates.get("character_updates")
        if char_updates and isinstance(char_updates, dict):
            self._merge_character_updates(char_updates, chapter_number, from_flawed_draft)
        else:
            logger.info(f"No character_updates found in combined response for ch {chapter_number}.")

        world_updates = combined_updates.get("world_building_updates")
        if world_updates and isinstance(world_updates, dict):
            self._merge_world_building_updates(world_updates, chapter_number, from_flawed_draft)
        else:
            logger.info(f"No world_building_updates found in combined response for ch {chapter_number}.")


    def _merge_character_updates(self, updates: Dict[str, Any], chapter_number: int, from_flawed_draft: bool):
        # This is the merging logic previously in _update_character_profiles
        if not updates: logger.info(f"No character profile updates to merge for ch {chapter_number}."); return
        
        logger.info(f"Merging character profile JSON updates for ch {chapter_number} for characters: {list(updates.keys())}")
        updated_chars_count, new_chars_count = 0, 0
        provisional_marker_key = f"source_quality_chapter_{chapter_number}"

        for char_name, char_update_data in updates.items():
            if not isinstance(char_update_data, dict): continue
            
            char_update = char_update_data.copy()
            if from_flawed_draft: char_update[provisional_marker_key] = "provisional_from_unrevised_draft"

            dev_key = f"development_in_chapter_{chapter_number}"
            if dev_key not in char_update and (len(char_update) > (1 if from_flawed_draft else 0) or \
               (len(char_update) == (1 if from_flawed_draft else 0) and "modification_proposal" not in char_update)):
                 char_update[dev_key] = "Character appeared/mentioned." 
            
            if char_name not in self.character_profiles:
                new_chars_count += 1
                logger.info(f"Adding new character '{char_name}' from ch {chapter_number}.")
                self.character_profiles[char_name] = {
                    "description": char_update.get("description", f"Introduced in Ch {chapter_number}."),
                    "traits": sorted(list(set(t for t in char_update.get("traits", []) if isinstance(t, str) and t.strip()))),
                    "relationships": char_update.get("relationships", {}), 
                    "status": char_update.get("status", "Newly introduced")
                }
                self.character_profiles[char_name][dev_key] = char_update.get(dev_key, f"Introduced in Ch {chapter_number}.")
                if from_flawed_draft: self.character_profiles[char_name][provisional_marker_key] = char_update[provisional_marker_key]
                if config.ENABLE_DYNAMIC_STATE_ADAPTATION and "modification_proposal" in char_update:
                    self._apply_modification_proposal(self.character_profiles[char_name], char_update["modification_proposal"], char_name, "character profile")
            else: 
                updated_chars_count += 1
                logger.debug(f"Updating existing character '{char_name}' from ch {chapter_number}.")
                existing_profile = self.character_profiles[char_name]
                if from_flawed_draft: existing_profile[provisional_marker_key] = char_update[provisional_marker_key]
                if config.ENABLE_DYNAMIC_STATE_ADAPTATION and "modification_proposal" in char_update:
                    self._apply_modification_proposal(existing_profile, char_update["modification_proposal"], char_name, "character profile")
                
                for key, value in char_update.items():
                    if key == "modification_proposal" or key == provisional_marker_key: continue 
                    if key == "traits" and isinstance(value, list):
                        if "traits" not in existing_profile or not isinstance(existing_profile["traits"], list): existing_profile["traits"] = []
                        existing_profile["traits"] = sorted(list(set(existing_profile["traits"]).union(set(t for t in value if isinstance(t, str) and t.strip()))))
                    elif key == "relationships" and isinstance(value, dict):
                         if not isinstance(existing_profile.get("relationships"), dict): existing_profile["relationships"] = {}
                         existing_profile["relationships"].update(value)
                    elif key == "description" and isinstance(value, str) and value.strip():
                         if not (config.ENABLE_DYNAMIC_STATE_ADAPTATION and "modification_proposal" in char_update and "MODIFY DESCRIPTION" in char_update["modification_proposal"].upper()):
                            existing_profile["description"] = value.strip()
                    elif key == dev_key and isinstance(value, str) and value.strip(): existing_profile[key] = value.strip() 
                    elif key == "status" and isinstance(value, str) and value.strip(): existing_profile["status"] = value.strip()
                    elif key not in existing_profile and value is not None: existing_profile[key] = value

        if updated_chars_count > 0 or new_chars_count > 0: logger.info(f"Character profile JSON merge done. Updated: {updated_chars_count}, New: {new_chars_count}.")
        else: logger.info(f"No character profiles effectively updated/added in ch {chapter_number}.")


    def _merge_world_building_updates(self, updates: Dict[str, Any], chapter_number: int, from_flawed_draft: bool):
        # This is the merging logic previously in _update_world_building
        if not updates: logger.info(f"No world-building updates to merge for ch {chapter_number}."); return

        logger.info(f"Merging world-building JSON updates for ch {chapter_number} for categories: {list(updates.keys())}")
        items_affected_count = 0
        provisional_marker_key = f"source_quality_chapter_{chapter_number}"

        for category_key, category_updates_dict_raw in updates.items():
            if not isinstance(category_updates_dict_raw, dict) or not category_updates_dict_raw: continue
            category_updates_dict = category_updates_dict_raw.copy() 
            if category_key not in self.world_building: self.world_building[category_key] = {}
            if not isinstance(self.world_building[category_key], dict):
                logger.warning(f"Overwriting non-dict world category '{category_key}'."); self.world_building[category_key] = {}
            
            target_category_dict = self.world_building[category_key]
            if from_flawed_draft: target_category_dict[provisional_marker_key] = "provisional_from_unrevised_draft"

            for item_name, item_update_details_raw in category_updates_dict.items():
                if not isinstance(item_update_details_raw, dict): logger.warning(f"Skipping invalid item_details for '{item_name}' in '{category_key}'."); continue
                
                item_update_details = item_update_details_raw.copy()
                item_log_name = f"{category_key}.{item_name}"
                if from_flawed_draft: item_update_details[provisional_marker_key] = "provisional_from_unrevised_draft"
                existing_item_data = target_category_dict.get(item_name)
                
                if existing_item_data is None: 
                    logger.info(f"Adding new world item '{item_log_name}'.")
                    new_item = self._robust_merge_world_item({}, item_update_details, item_log_name, chapter_number, from_flawed_draft)
                    new_item[f"added_in_chapter_{chapter_number}"] = True 
                    target_category_dict[item_name] = new_item; items_affected_count +=1
                else: 
                    updated_item = self._robust_merge_world_item(existing_item_data, item_update_details, item_log_name, chapter_number, from_flawed_draft)
                    target_category_dict[item_name] = updated_item
                    if updated_item.get(f"updated_in_chapter_{chapter_number}") or updated_item.get(f"added_in_chapter_{chapter_number}"): items_affected_count +=1
            
            if any(isinstance(v,dict) and (v.get(f"updated_in_chapter_{chapter_number}") or v.get(f"added_in_chapter_{chapter_number}")) for v in target_category_dict.values()):
                 target_category_dict[f"updated_in_chapter_{chapter_number}"] = True 

        if items_affected_count > 0: logger.info(f"World-building JSON merge done. {items_affected_count} items affected.")
        else: logger.info(f"No world-building JSON items effectively updated/added in ch {chapter_number}.")


    def _apply_modification_proposal(self, profile_or_item: Dict[str, Any], proposal: str, item_name: str, item_type_for_log: str):
        if not isinstance(proposal, str) or not proposal.strip(): logger.debug(f"Empty proposal for '{item_name}'."); return
        logger.debug(f"Applying modification proposal for '{item_name}' ({item_type_for_log}): '{proposal}'")
        proposal_norm = proposal.strip().upper()
        try:
            match_modify_key = re.match(r"MODIFY\s+([\w_]+)\s*:", proposal_norm)
            if not match_modify_key: logger.warning(f"Invalid proposal format for '{item_name}': '{proposal}'"); return
            
            key_to_modify_upper = match_modify_key.group(1).strip()
            original_key = next((k for k in profile_or_item if k.upper() == key_to_modify_upper), key_to_modify_upper.lower())
            action_details_original_case = proposal[match_modify_key.end():].strip()

            if original_key.lower() == "traits": 
                if "traits" not in profile_or_item or not isinstance(profile_or_item["traits"], list): profile_or_item["traits"] = []
                current_traits_set = set(profile_or_item["traits"])
                for match_add in re.finditer(r"ADD\s+['\"]([^'\"]+)['\"]", action_details_original_case, re.IGNORECASE):
                    trait_to_add = match_add.group(1).strip()
                    if trait_to_add: current_traits_set.add(trait_to_add)
                for match_remove in re.finditer(r"REMOVE\s+['\"]([^'\"]+)['\"]", action_details_original_case, re.IGNORECASE):
                    trait_to_remove = match_remove.group(1).strip()
                    if trait_to_remove: current_traits_set.discard(trait_to_remove)
                profile_or_item["traits"] = sorted(list(current_traits_set))
                logger.info(f"Applied trait modifications for '{item_name}'. New: {profile_or_item['traits']}")
            else: 
                new_value_str = action_details_original_case.strip("'\" ")
                if new_value_str: 
                    profile_or_item[original_key] = new_value_str 
                    logger.info(f"Applied modification to '{original_key}' for '{item_name}'. New: '{new_value_str[:50]}...'")
                else: logger.warning(f"Modification proposal for '{original_key}' of '{item_name}' had empty new value. Proposal: '{proposal}'")
        except Exception as e: logger.error(f"Error applying modification proposal for '{item_name}': {e}. Proposal: '{proposal}'", exc_info=True)

    def _robust_merge_world_item(self, target_item: Any, update_details: Dict[str, Any], item_name_for_log: str, chapter_num: int, from_flawed_draft_source: bool) -> Dict[str, Any]:
        current_item_dict: Dict[str, Any]
        provisional_marker_key = f"source_quality_chapter_{chapter_num}"

        if not isinstance(target_item, dict):
            logger.warning(f"World item '{item_name_for_log}' not dict. Converting. Original '{str(target_item)[:50]}...' to 'description'.")
            current_item_dict = {"description": str(target_item)}
            current_item_dict[f"updated_in_chapter_{chapter_num}"] = True 
        else: current_item_dict = target_item.copy() 

        item_was_modified_this_call = False
        if from_flawed_draft_source:
            current_item_dict[provisional_marker_key] = update_details.get(provisional_marker_key, "provisional_from_unrevised_draft")
            item_was_modified_this_call = True

        if config.ENABLE_DYNAMIC_STATE_ADAPTATION and "modification_proposal" in update_details:
            proposal = update_details.pop("modification_proposal") 
            if isinstance(proposal, str) and proposal.strip():
                self._apply_modification_proposal(current_item_dict, proposal, item_name_for_log, "world item")
                item_was_modified_this_call = True
        
        for key, value in update_details.items():
            if key.startswith(("updated_in_chapter_", "added_in_chapter_")) or key == provisional_marker_key or key == "modification_proposal": continue
            target_value = current_item_dict.get(key)
            if isinstance(value, dict): 
                if not isinstance(target_value, dict): current_item_dict[key] = {}; current_item_dict[key][f"added_in_chapter_{chapter_num}"] = True
                current_item_dict[key] = self._robust_merge_world_item(current_item_dict[key], value, f"{item_name_for_log}.{key}", chapter_num, from_flawed_draft_source)
                if current_item_dict[key].get(f"updated_in_chapter_{chapter_num}") or current_item_dict[key].get(f"added_in_chapter_{chapter_num}"): item_was_modified_this_call = True
            elif isinstance(value, list): 
                if not isinstance(target_value, list): current_item_dict[key] = []
                initial_list_len = len(current_item_dict[key])
                for item_in_list_update in value:
                    if item_in_list_update not in current_item_dict[key]: current_item_dict[key].append(item_in_list_update)
                if len(current_item_dict[key]) > initial_list_len: item_was_modified_this_call = True
            elif value != target_value: current_item_dict[key] = value; item_was_modified_this_call = True
        
        if item_was_modified_this_call and not current_item_dict.get(f"added_in_chapter_{chapter_num}"):
            current_item_dict[f"updated_in_chapter_{chapter_num}"] = True
        return current_item_dict

    def _extract_and_update_kg(self, chapter_text: Optional[str], chapter_number: int, from_flawed_draft: bool):
        if not chapter_text: logger.warning(f"Skipping KG extraction for ch {chapter_number}: Text None."); return
            
        logger.info(f"Extracting KG triples for ch {chapter_number} (Flawed draft: {from_flawed_draft})...")
        text_snippet = chapter_text[:config.KNOWLEDGE_UPDATE_SNIPPET_SIZE * 2] # Slightly larger snippet for KG
        if len(text_snippet) < len(chapter_text): logger.warning(f"KG extraction using truncated text ({len(text_snippet)} chars) for ch {chapter_number}.")
            
        protagonist_name = self.plot_outline.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME)
        common_predicates = ["is_a", "located_in", "has_trait", "status_is", "feels", "knows", "believes", "wants", "interacted_with", "travelled_to", "discovered", "acquired", "lost", "used_item", "attacked", "helped", "damaged", "repaired", "contains", "part_of", "caused_by", "leads_to", "observed", "heard", "said", "thought_about", "decided_to", "has_goal", "has_feature", "related_to", "member_of", "leader_of", "enemy_of", "ally_of", "works_for", "has_ability"]
        
        prompt = f"""/no_think
        
        KG Engineer: Extract (Subject, Predicate, Object) triples from Ch {chapter_number} Text Snippet (protagonist: '{protagonist_name}').
        **Ch {chapter_number} Text Snippet:**\n--- TEXT ---\n{text_snippet}\n--- END TEXT ---\n
        **Instructions:** 1. Identify key entities (normalize names). 2. Use suggested predicates or concise alternatives. 3. Extract `["Subject", "predicate", "Object"]` (all non-empty strings). 4. Focus ONLY on info from THIS text. 5. Prioritize state changes & key events. 6. **CRITICAL OUTPUT:** ONLY JSON list of lists. `[]` if no facts. 7. **NO extra text/markdown.** Start `[` end `]`.
        **Suggested Predicates:** {', '.join(common_predicates)}
        **Example:** `[["{protagonist_name}", "travelled_to", "Eclipse Spire"], ...]`
        JSON Output Only:
        [
        """
        raw_triples_json = llm_interface.call_llm(
            model_name=config.KNOWLEDGE_UPDATE_MODEL,
            prompt=prompt, 
            temperature=0.4, # Lower temp for factual extraction
            max_tokens=config.MAX_KG_TRIPLE_TOKENS
        ) 
        parsed_triples = llm_interface.parse_llm_json_response(raw_triples_json, f"KG triple extraction for chapter {chapter_number}", expect_type=list)
        
        if parsed_triples is None:
             logger.error(f"Failed to extract/parse KG triples for ch {chapter_number}. Raw: {raw_triples_json[:200] if raw_triples_json else 'EMPTY'}")
             self._save_debug_output(chapter_number, "kg_extraction_raw_fail_final", raw_triples_json or "EMPTY"); return
             
        added_count, skipped_count = 0, 0
        for triple in parsed_triples:
            if isinstance(triple, list) and len(triple) == 3:
                subj, pred, obj = [str(t).strip() if t is not None else "" for t in triple]
                if subj and pred and obj: self.db_manager.add_kg_triple(subj, pred, obj, chapter_number, is_provisional=from_flawed_draft); added_count += 1
                else: logger.warning(f"Skipping invalid triple (empty) in ch {chapter_number}: {triple}"); skipped_count += 1
            else: logger.warning(f"Skipping invalid triple format in KG for ch {chapter_number}: {triple}"); skipped_count += 1
        logger.info(f"Added {added_count} KG triples from ch {chapter_number}. Skipped {skipped_count}. (Source Provisional: {from_flawed_draft})")

    def _prepopulate_knowledge_graph(self):
        logger.info("Starting KG pre-population...")
        if not self.plot_outline or self.plot_outline.get("is_default", True): logger.warning("Skipping KG pre-pop: Plot outline missing/default."); return
        if not self.world_building or self.world_building.get("is_default", True): logger.warning("Skipping KG pre-pop: World building missing/default."); return

        # Pruned Plot Outline for prompt
        pruned_plot = {
            "title": self.plot_outline.get("title"),
            "protagonist_name": self.plot_outline.get("protagonist_name"),
            "genre": self.plot_outline.get("genre"),
            "theme": self.plot_outline.get("theme"),
            "setting_description": self.plot_outline.get("setting"), # Use the specific setting description
            "conflict_summary": self.plot_outline.get("conflict"),
            "character_arc": self.plot_outline.get("character_arc"),
            "key_plot_points_summary": self.plot_outline.get("plot_points", [])[:2] # Maybe first 2 plot points
        }
        
        # Pruned World Building for prompt
        pruned_world = {}
        for category, items in self.world_building.items():
            if category == "is_default": continue
            if isinstance(items, dict):
                pruned_world[category] = {}
                for item_name, item_details in list(items.items())[:5]: # Limit items per category
                    if isinstance(item_details, dict) and "description" in item_details:
                        pruned_world[category][item_name] = {"description": str(item_details["description"])[:200] + "..."} # Truncate desc
                    elif isinstance(item_details, dict) and "text" in item_details: # For lore
                         pruned_world[category][item_name] = {"text": str(item_details["text"])[:200] + "..."}
        
        combined_pruned_data = {
            "plot_summary": pruned_plot,
            "world_highlights": pruned_world
        }
        try: combined_data_json = json.dumps(combined_pruned_data, indent=2, ensure_ascii=False, default=str)
        except TypeError as e: logger.error(f"Error serializing pruned data for KG pre-pop: {e}"); return
            
        protagonist_name = self.plot_outline.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME)
        novel_title = self.plot_outline.get("title", config.DEFAULT_PLOT_OUTLINE_TITLE)
        common_predicates = ["is_a", "has_title", "has_protagonist", "has_genre", "has_theme", "has_setting_description", "has_conflict_summary", "has_character_arc", "has_description", "has_trait", "related_to", "located_in", "has_goal", "part_of", "member_of", "governed_by", "known_for", "primary_setting_is"]

        prompt = f"""/no_think
        
        KG Engineer: Extract foundational (Subject, Predicate, Object) triples from summarized Plot & World data for novel '{novel_title}' (protagonist: '{protagonist_name}').
        **Input JSON Data (Summarized Plot & World Highlights):**
        ```json
        {combined_data_json}
        ```
        **Instructions:** 1. Analyze JSON. Keys often subjects/predicates. Values objects/descriptions. 2. Extract core entities, types, attributes, relationships for foundational canon. 3. Use suggested predicates or concise alternatives. 4. For novel itself, use "{novel_title}" as subject for genre, theme, protagonist. 5. For '{protagonist_name}', extract description, core traits, initial status. 6. For locations, factions, etc., extract names, descriptions. 7. All triple components `["S", "P", "O"]` MUST be non-empty strings. 8. **CRITICAL OUTPUT:** ONLY JSON list of lists. `[]` if no facts. 9. **NO EXTRA TEXT/MARKDOWN.** Start `[` end `]`.
        **Suggested Predicates:** {', '.join(common_predicates)}
        **Example:** `[["{novel_title}", "has_protagonist", "{protagonist_name}"], ...]`
        JSON Output Only:
        [
        """
        logger.info("Calling LLM for KG pre-population triple extraction...")
        raw_triples_json = llm_interface.call_llm(
            model_name=config.KNOWLEDGE_UPDATE_MODEL, # Medium model should be fine
            prompt=prompt, 
            temperature=0.4, 
            max_tokens=config.MAX_PREPOP_KG_TOKENS
        )
        parsed_triples = llm_interface.parse_llm_json_response(raw_triples_json, "KG pre-population triple extraction", expect_type=list)

        if parsed_triples is None:
            logger.error(f"Failed to extract/parse KG triples for pre-population. Raw: {raw_triples_json[:500] if raw_triples_json else 'EMPTY'}")
            self._save_debug_output(config.KG_PREPOPULATION_CHAPTER_NUM, "kg_prepop_raw_fail_final", raw_triples_json or "EMPTY"); return

        added_count, skipped_count = 0, 0
        for triple in parsed_triples:
            if isinstance(triple, list) and len(triple) == 3:
                subj, pred, obj = [str(t).strip() if t is not None else "" for t in triple]
                if subj and pred and obj: self.db_manager.add_kg_triple(subj, pred, obj, config.KG_PREPOPULATION_CHAPTER_NUM, is_provisional=False); added_count += 1
                else: logger.warning(f"Skipping invalid pre-pop triple (empty): {triple}"); skipped_count += 1
            else: logger.warning(f"Skipping invalid pre-pop triple format: {triple}"); skipped_count += 1
        logger.info(f"KG pre-pop: Added {added_count} foundational triples. Skipped {skipped_count}.")
        if added_count == 0 and parsed_triples: logger.warning("KG pre-pop 0 valid triples despite LLM data.")


    def _get_relevant_character_state_snippet(self, current_chapter_num_for_filtering: Optional[int] = None) -> str:
        snippet_data, count = {}, 0
        sorted_char_names = []
        protagonist_name = self.plot_outline.get("protagonist_name")
        if protagonist_name and protagonist_name in self.character_profiles: sorted_char_names.append(protagonist_name)
        for name in sorted(self.character_profiles.keys()):
            if name != protagonist_name: sorted_char_names.append(name)
            
        for name in sorted_char_names:
            if count >= config.PLANNING_CONTEXT_MAX_CHARACTERS_IN_SNIPPET: break
            profile = self.character_profiles.get(name, {})
            is_provisional_note = ""
            effective_filter_chapter = (current_chapter_num_for_filtering -1) if current_chapter_num_for_filtering is not None and current_chapter_num_for_filtering > 0 else config.KG_PREPOPULATION_CHAPTER_NUM
            if any(key.startswith("source_quality_chapter_") and int(key.split('_')[-1]) <= effective_filter_chapter for key in profile):
                 is_provisional_note = " (Note: Some info may be provisional)"

            dev_notes_keys = sorted([k for k in profile if k.startswith("development_in_chapter_") and int(k.split('_')[-1]) <= effective_filter_chapter], key=lambda x: int(x.split('_')[-1]), reverse=True)
            recent_dev_note = profile.get(dev_notes_keys[0], "N/A") if dev_notes_keys else "N/A"
            
            snippet_data[name] = {
                "desc_snippet": profile.get("description", "")[:config.PLANNING_CONTEXT_MAX_CHARS_PER_PROFILE_DESC] + "...",
                "status": profile.get("status", "Unknown") + is_provisional_note,
                "recent_dev_note": recent_dev_note[:config.PLANNING_CONTEXT_MAX_RECENT_DEV_PER_PROFILE] + "..."
            }
            count += 1
        return json.dumps(snippet_data, indent=2, ensure_ascii=False, default=str) if snippet_data else "No character profiles."

    def _get_relevant_world_state_snippet(self, current_chapter_num_for_filtering: Optional[int] = None) -> str:
        snippet_data = {}
        effective_filter_chapter = (current_chapter_num_for_filtering -1) if current_chapter_num_for_filtering is not None and current_chapter_num_for_filtering > 0 else config.KG_PREPOPULATION_CHAPTER_NUM
        def get_provisional_note(item_dict: Dict[str, Any], chapter_limit: int) -> str:
            if any(key.startswith("source_quality_chapter_") and int(key.split('_')[-1]) <= chapter_limit for key in item_dict): return " (Note: Some info may be provisional)"
            return ""

        if "locations" in self.world_building and isinstance(self.world_building["locations"], dict):
             loc_provisional_note = get_provisional_note(self.world_building["locations"], effective_filter_chapter)
             snippet_data["locations_overview" + loc_provisional_note] = list(self.world_building["locations"].keys())[:config.PLANNING_CONTEXT_MAX_LOCATIONS_IN_SNIPPET]
        
        society_data = self.world_building.get("society", {})
        if isinstance(society_data, dict):
            soc_provisional_note = get_provisional_note(society_data, effective_filter_chapter)
            if "Key Factions" in society_data and isinstance(society_data["Key Factions"], dict):
                 snippet_data["key_factions" + soc_provisional_note] = list(society_data["Key Factions"].keys())[:config.PLANNING_CONTEXT_MAX_FACTIONS_IN_SNIPPET]
        
        if "systems" in self.world_building and isinstance(self.world_building["systems"], dict):
             sys_provisional_note = get_provisional_note(self.world_building["systems"], effective_filter_chapter)
             snippet_data["key_systems" + sys_provisional_note] = list(self.world_building["systems"].keys())[:config.PLANNING_CONTEXT_MAX_SYSTEMS_IN_SNIPPET]
             
        return json.dumps(snippet_data, indent=2, ensure_ascii=False, default=str) if snippet_data else "No world-building data."

    def _get_filtered_profiles_for_prompt(self, up_to_chapter: Optional[int] = None) -> Dict[str, Any]:
        profiles_copy = json.loads(json.dumps(self.character_profiles)) 
        if up_to_chapter is None: return profiles_copy
        for char_name, profile_data in profiles_copy.items():
            provisional_notes_for_char = []
            for i in range(1, up_to_chapter + 1): 
                prov_key = f"source_quality_chapter_{i}"
                if profile_data.get(prov_key) == "provisional_from_unrevised_draft": provisional_notes_for_char.append(f"Info for this char updated in ch {i} was provisional.")
            if provisional_notes_for_char:
                if "prompt_notes" not in profile_data: profile_data["prompt_notes"] = []
                profile_data["prompt_notes"].extend(list(set(provisional_notes_for_char))) 
        return profiles_copy

    def _get_filtered_world_for_prompt(self, up_to_chapter: Optional[int] = None) -> Dict[str, Any]:
        world_copy = json.loads(json.dumps(self.world_building)) 
        if up_to_chapter is None: return world_copy
        for category, items in world_copy.items():
            if isinstance(items, dict):
                category_provisional_notes = []
                for i in range(1, up_to_chapter + 1):
                    cat_prov_key = f"source_quality_chapter_{i}" 
                    if items.get(cat_prov_key) == "provisional_from_unrevised_draft": category_provisional_notes.append(f"Category '{category}' info updated in ch {i} was provisional.")
                if category_provisional_notes:
                    if "prompt_notes" not in items: items["prompt_notes"] = []
                    items["prompt_notes"].extend(list(set(category_provisional_notes)))
                for item_name, item_data in items.items():
                    if isinstance(item_data, dict):
                        item_provisional_notes = []
                        for i in range(1, up_to_chapter + 1):
                            prov_key = f"source_quality_chapter_{i}"
                            if item_data.get(prov_key) == "provisional_from_unrevised_draft": item_provisional_notes.append(f"Item '{item_name}' ('{category}') info updated in ch {i} was provisional.")
                        if item_provisional_notes:
                            if "prompt_notes" not in item_data: item_data["prompt_notes"] = []
                            item_data["prompt_notes"].extend(list(set(item_provisional_notes)))
        return world_copy

    def _save_debug_output(self, chapter_number: int, stage: str, content: Any):
        if content is None: return
        content_str = str(content) if not isinstance(content, str) else content
        try:
            debug_dir = os.path.join(config.OUTPUT_DIR, "debug_outputs") 
            os.makedirs(debug_dir, exist_ok=True)
            safe_stage = "".join(c if c.isalnum() or c in ['_', '-'] else "_" for c in stage)
            file_path = os.path.join(debug_dir, f"chapter_{chapter_number}_{safe_stage}.txt")
            with open(file_path, 'w', encoding='utf-8') as f: f.write(content_str)
            logger.debug(f"Saved debug output for ch {chapter_number} stage '{stage}' to {file_path}")
        except Exception as e: logger.error(f"Failed to save debug output for ch {chapter_number} stage '{stage}': {e}", exc_info=True)
