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
from typing import Dict, List, Optional, Tuple, Any, TypedDict

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
    consistency_issues: Optional[str] # This will now include KG-based issues
    plot_deviation_reason: Optional[str]

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
                    data_to_save = data_dict.copy()
                    # Handle 'is_default' cleanup (as before)
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
                    
                    # Remove our temporary provisional flags from the saved JSON if they exist at top level
                    # Note: Deeper provisional flags within nested structures would need more complex cleanup
                    # or be kept as part of the state.
                    provisional_key_pattern = re.compile(r"source_quality_chapter_\d+")
                    keys_to_delete = [k for k in data_to_save if provisional_key_pattern.match(k)]
                    for k_del in keys_to_delete:
                        del data_to_save[k_del]
                        logger.debug(f"Removed temporary provisional flag '{k_del}' from {data_dict_attr_name} before saving.")

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
        # (This method remains largely the same as your last version, no direct impact from these changes)
        logger.info(f"Generating plot outline. Unhinged mode: {unhinged_mode}")
        # ... (prompt preparation based on mode) ...
        if unhinged_mode:
            genre = kwargs.get("genre", "fantasy") 
            theme = kwargs.get("theme", "the nature of good and evil")
            setting_archetype = kwargs.get("setting_archetype", "a generic medieval kingdom")
            protagonist_archetype = kwargs.get("protagonist_archetype", "a reluctant hero")
            conflict_archetype = kwargs.get("conflict_archetype", "an external threat to the kingdom")
            
            prompt_core_elements = f"""
The novel is a '{genre}' story.
Its central theme is '{theme}'.
The primary setting is inspired by: '{setting_archetype}'.
The protagonist is an archetype of: '{protagonist_archetype}'.
The main conflict revolves around: '{conflict_archetype}'.

Based on this unusual combination, please generate the following:
1.  `title`: A compelling and fitting title for this unique novel.
2.  `protagonist_name`: A suitable name for the protagonist.
3.  `protagonist_description`: A brief (1-2 sentences) description of the protagonist, fleshing out the archetype within this specific story.
4.  `setting`: A brief (1-2 sentences) description of the novel's primary setting, expanding on the archetype.
5.  `conflict`: A brief (1-2 sentences) summary of the main conflict, making it specific to this story.
6.  `plot_points`: A list of exactly 5 strings, each describing a major plot point.
7.  `character_arc`: A string describing the protagonist's primary development arc.
"""
            base_elements_for_outline = {
                "genre": genre, "theme": theme, 
                "setting_archetype_used": setting_archetype,
                "protagonist_archetype_used": protagonist_archetype,
                "conflict_archetype_used": conflict_archetype
            }
        else: # Standard mode
            genre = kwargs.get("genre", config.CONFIGURED_GENRE)
            theme = kwargs.get("theme", config.CONFIGURED_THEME)
            setting_description = kwargs.get("setting_description", config.CONFIGURED_SETTING_DESCRIPTION)
            prompt_core_elements = f"""
The novel is a '{genre}' story.
Its central theme is '{theme}'.
The primary setting is: '{setting_description}'.

Based on these core elements, please generate the following:
1.  `title`: A compelling title for the novel.
2.  `protagonist_name`: A suitable name for the protagonist.
3.  `protagonist_description`: A brief (1-2 sentences) description of the protagonist.
4.  `plot_points`: A list of exactly 5 strings, each describing a major plot point.
5.  `character_arc`: A string describing the protagonist's primary development arc.
6.  `conflict`: A string summarizing the main internal and/or external conflict.
"""
            base_elements_for_outline = {"genre": genre, "theme": theme, "setting": setting_description}

        prompt = f"""You are a creative assistant specializing in narrative structure.
        {prompt_core_elements}

Output ONLY the JSON object adhering strictly to the requested keys.
Do not include any introductory text, explanations, markdown formatting, or meta-commentary.
Ensure the `plot_points` value is a JSON list of 5 strings.
Example Structure (keys will vary slightly based on mode above):
{{
  "title": "string",
  "protagonist_name": "string",
  "protagonist_description": "string",
  "plot_points": ["Plot Point 1: ...", "Plot Point 2: ...", "Plot Point 3: ...", "Plot Point 4: ...", "Plot Point 5: ..."],
  "character_arc": "string",
  "setting": "string", 
  "conflict": "string"
}}
"""
        logger.info("Calling LLM for plot outline generation...")
        raw_outline_str = llm_interface.call_llm(prompt, temperature=0.7)
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
                logger.info(f"Successfully generated and validated plot outline for title: '{self.plot_outline.get('title', 'N/A')}'")
            else:
                 logger.warning(f"Generated plot outline failed validation. Missing/empty keys or incorrect plot_points. Parsed: {parsed_outline}")

        if not is_valid:
            logger.error("Failed to generate a valid plot outline. Applying default structure.")
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
        # (This method remains largely the same, no direct impact from these changes)
        if self.world_building and not self.world_building.get("is_default", False):
            logger.info("Skipping initial world-building generation: Data seems populated and not default.")
            return

        if not self.plot_outline or not self.plot_outline.get("setting"):
            logger.error("Cannot generate world-building: Plot outline (especially setting description) is missing. Applying minimal default.")
            self.world_building = {
                "locations": {"Default Location": {"description": "A starting point."}},
                "society": {"General": {"description": "Basic societal norms apply."}},
                "is_default": True
            }
            self._save_json_state()
            return

        prompt = f"""You are a world-building assistant tasked with generating foundational elements based on a novel concept. Your output MUST be a single, valid JSON object.
        Novel Concept:
        Title: {self.plot_outline.get('title', 'Untitled')}
        Genre: {self.plot_outline.get('genre', 'undefined')}
        Theme: {self.plot_outline.get('theme', 'undefined')}
        Setting Description (Expand on this): {self.plot_outline.get('setting', 'default setting')}
        Main Conflict: {self.plot_outline.get('conflict', 'default conflict')}
        Protagonist: {self.plot_outline.get('protagonist_name', 'N/A')} ({self.plot_outline.get('protagonist_description', 'N/A')})
        Instructions:
        1. Create detailed world-building elements covering locations, society, unique systems (tech/magic), lore, and history relevant to the concept.
        2. Be creative and provide enough detail to make the world feel tangible. Expand significantly on the provided setting description.
        3. **CRITICAL: Output ONLY the JSON object.**
        4. **JSON Syntax Rules:** Ensure valid JSON.
        5. Use clear top-level categories: "locations", "society", "systems", "lore", "history".
           Items within these categories (e.g., a specific location, a faction, a historical event) should be dictionaries themselves, typically containing a "description" key and other relevant attributes.
        Example Structure:
        ```json
        {{
          "locations": {{
            "Location Name 1": {{ "description": "string", "atmosphere": "string", "relevance": "string" }}
          }},
          "society": {{
            "Governance": {{ "description": "string" }},
            "Key Factions": {{ "Faction 1": {{ "description": "string", "goal": "string" }} }}
          }},
          "systems": {{ "Key System Name": {{ "name": "string", "rules": "string" }} }},
          "lore": {{ "Key Myth": {{ "text": "string", "origin": "string" }} }},
          "history": {{ "Foundational Event": {{ "description": "string" }} }}
        }}
        ```
        """
        logger.info("Generating initial world-building data via LLM...")
        raw_world_data_str = llm_interface.call_llm(prompt, temperature=0.6)
        parsed_world_data = llm_interface.parse_llm_json_response(raw_world_data_str, "initial world-building")

        is_valid = False
        if parsed_world_data and isinstance(parsed_world_data, dict):
            if any(k in parsed_world_data and isinstance(parsed_world_data[k], dict) and parsed_world_data[k] for k in ["locations", "society", "systems", "lore", "history"]):
                self.world_building = parsed_world_data
                self.world_building["is_default"] = False
                logger.info("Successfully generated and validated initial world-building data.")
                is_valid = True
            else: logger.warning(f"Generated world-building JSON lacks expected structure/content. Parsed: {parsed_world_data}")

        if not is_valid:
            logger.error("Failed to generate valid world-building data. Applying default structure.")
            self.world_building = {
                "locations": {"Default Location": {"description": "A starting point."}},
                "society": {"General": {"description": "Basic societal norms."}},
                "is_default": True
            }
        self._save_json_state()

    def _plan_chapter(self, chapter_number: int) -> Optional[str]:
        if not config.ENABLE_AGENTIC_PLANNING:
            return "Agentic planning disabled by configuration."
        logger.info(f"Planning Chapter {chapter_number}...")
        plot_point_focus, plot_point_index = self._get_plot_point_info(chapter_number)
        if plot_point_focus is None:
            logger.error(f"Cannot plan ch {chapter_number}: no plot point.")
            return None

        context_summary = ""
        if chapter_number > 1:
            # Consider fetching provisional status of previous chapter here if needed for context quality
            prev_chap_data = self.db_manager.get_chapter_data_from_db(chapter_number - 1)
            if prev_chap_data:
                prev_summary = prev_chap_data.get('summary')
                # prev_is_provisional = prev_chap_data.get('is_provisional', False) # If you fetch this from DB
                # summary_prefix = "[Provisional Summary] " if prev_is_provisional and prev_summary else ""
                if prev_summary:
                    context_summary += f"Summary of Previous Chapter ({chapter_number - 1}):\n{prev_summary[:1000]}...\n"
                else:
                    prev_text = prev_chap_data.get('text', '')
                    if prev_text:
                        context_summary += f"End Snippet of Previous Chapter ({chapter_number - 1}):\n...{prev_text[-1000:]}\n"
        
        kg_facts = []
        protagonist_name = self.plot_outline.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME)
        # For planning, we might prefer non-provisional facts for stability
        current_loc = self.db_manager.get_most_recent_value(protagonist_name, "located_in", chapter_number - 1, include_provisional=False)
        current_status = self.db_manager.get_most_recent_value(protagonist_name, "status_is", chapter_number - 1, include_provisional=False)

        if current_loc: kg_facts.append(f"- {protagonist_name} is currently located in (reliable KG): {current_loc}.")
        if current_status: kg_facts.append(f"- {protagonist_name}'s current status (reliable KG): {current_status}.")
        
        kg_context_section = "**Relevant Reliable Facts from Knowledge Graph (up to end of previous chapter):**\n" + "\n".join(kg_facts) + "\n" if kg_facts else ""

        prompt = f"""You are a master plotter outlining the key narrative beats for Chapter {chapter_number} of a novel.
        **Overall Novel Concept:**
        * Title: {self.plot_outline.get('title', 'Untitled')}
        * Genre: {self.plot_outline.get('genre', 'N/A')}
        * Theme: {self.plot_outline.get('theme', 'N/A')}
        * Protagonist: {protagonist_name}
        * Protagonist Arc Goal: {self.plot_outline.get('character_arc', 'N/A')}
        **Mandatory Focus for THIS Chapter (Plot Point {plot_point_index + 1}):**
        {plot_point_focus}
        **Recent Context (End of Previous Chapter/Summary):**
        {context_summary if context_summary else "This is the first chapter."}
        {kg_context_section}
        **Current Character States (Key Characters Only, from JSON profiles - note if any info is provisional):**
        {self._get_relevant_character_state_snippet()} 
        **Current World State (Relevant Locations/Elements, from JSON world-building - note if any info is provisional):**
        {self._get_relevant_world_state_snippet()}
        **Task:** Outline 3-5 essential scenes/beats for Chapter {chapter_number} that:
        1. Directly address the **Mandatory Focus**. 2. Logically follow **Recent Context** & **Reliable KG Facts**. 3. Involve relevant characters/world elements. 4. Contribute to **Protagonist Arc Goal**.
        **Output Format:** ONLY a numbered list of scenes/beats (1 sentence per beat). Start with "1.".
        """
        logger.info(f"Calling LLM to generate plan for chapter {chapter_number}...")
        plan_raw = llm_interface.call_llm(prompt, temperature=0.6, max_tokens=config.MAX_PLANNING_TOKENS)
        plan_cleaned = llm_interface.clean_model_response(plan_raw).strip()
        if plan_cleaned and re.match(r"^\s*1\.", plan_cleaned):
            logger.info(f"Successfully generated plan for chapter {chapter_number}:\n{plan_cleaned}")
            return plan_cleaned
        else:
            logger.error(f"Failed to generate valid plan for ch {chapter_number}. Response: '{plan_raw[:200]}...'")
            return None

    def write_chapter(self, chapter_number: int) -> Optional[str]:
        logger.info(f"=== Starting Chapter {chapter_number} Generation ===")
        if not self.plot_outline or not self.plot_outline.get("plot_points") or not self.plot_outline.get("protagonist_name"):
            logger.error(f"Cannot write chapter {chapter_number}: Plot outline, plot points, or protagonist_name are missing.")
            return None
        if chapter_number <= 0:
            logger.error(f"Cannot write chapter {chapter_number}: Chapter number must be positive.")
            return None

        chapter_plan = self._plan_chapter(chapter_number)
        if config.ENABLE_AGENTIC_PLANNING and chapter_plan is None:
            logger.error(f"Chapter {chapter_number} generation halted due to planning failure.")
            return None

        context_for_draft = self._get_context(chapter_number)
        plot_point_focus, _ = self._get_plot_point_info(chapter_number)

        initial_draft_text, initial_raw_text = self._generate_draft(
            chapter_number, plot_point_focus, context_for_draft, chapter_plan
        )

        if not initial_draft_text:
            logger.error(f"Failed to generate initial draft for chapter {chapter_number}.")
            self._save_debug_output(chapter_number, "initial_raw_fail_after_clean", initial_raw_text or "")
            return None

        evaluation = self._evaluate_draft(initial_draft_text, chapter_number, context_for_draft)
        current_text, final_raw_output_log = initial_draft_text, f"--- INITIAL DRAFT RAW ---\n{initial_raw_text}\n\n"
        
        proceeded_with_flaws = False 

        if evaluation["needs_revision"]:
            revision_reason_str = "\n- ".join(evaluation["reasons"])
            logger.warning(f"Chapter {chapter_number} flagged for revision. Reason(s):\n- {revision_reason_str}")
            revised_text_tuple = self._revise_chapter(
                current_text, chapter_number, revision_reason_str, context_for_draft, chapter_plan
            )
            if revised_text_tuple:
                revised_text, raw_revision_output = revised_text_tuple
                logger.info(f"Revision successful for chapter {chapter_number}. Evaluating revised draft...")
                revised_evaluation = self._evaluate_draft(revised_text, chapter_number, context_for_draft)
                if revised_evaluation["needs_revision"]:
                    logger.error(f"Revised draft for ch {chapter_number} STILL failed evaluation. Reasons:\n- " + "\n- ".join(revised_evaluation["reasons"]))
                    proceeded_with_flaws = True 
                else:
                    logger.info(f"Revised draft for chapter {chapter_number} passed evaluation.")
                current_text = revised_text 
                final_raw_output_log += f"--- REVISION ATTEMPT (Reason: {revision_reason_str}) ---\n{raw_revision_output}\n\n"
            else:
                logger.error(f"Revision failed for chapter {chapter_number}. Proceeding with original draft despite issues.")
                proceeded_with_flaws = True
                final_raw_output_log += f"--- REVISION FAILED (Reason: {revision_reason_str}) ---\n\n"
        else:
            logger.info(f"Initial draft for chapter {chapter_number} passed evaluation.")

        if not self._finalize_chapter_core(chapter_number, current_text, final_raw_output_log, proceeded_with_flaws):
             logger.error(f"=== Finished Chapter {chapter_number} Generation With Errors During Core Finalization ===")
             return None

        self._update_knowledge_bases(chapter_number, current_text, proceeded_with_flaws)
        
        self.chapter_count = max(self.chapter_count, chapter_number)
        self._save_json_state() 
        logger.info(f"=== Finished Chapter {chapter_number} Generation Successfully (Proceeded with flaws: {proceeded_with_flaws}) ===")
        return current_text

    def _get_plot_point_info(self, chapter_number: int) -> Tuple[Optional[str], int]:
        plot_points = self.plot_outline.get("plot_points", [])
        plot_point_index = min(chapter_number - 1, len(plot_points) - 1) if plot_points else -1
        if 0 <= plot_point_index < len(plot_points):
            return plot_points[plot_point_index], plot_point_index
        else:
            logger.warning(f"Could not find plot point for chapter {chapter_number}.")
            return None, -1

    def _generate_draft(self, chapter_number: int, plot_point_focus: Optional[str], context: str, plan: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        if not plot_point_focus:
            plot_point_focus = "Continue the narrative logically."
        plan_section = ""
        if plan and config.ENABLE_AGENTIC_PLANNING:
            plan_section = f"**Chapter Plan / Key Beats:**\n{plan}\n" if plan != "Agentic planning disabled by configuration." else f"**Note:** {plan}\n"
        
        # Note: For _get_filtered_profiles_for_prompt and _get_filtered_world_for_prompt,
        # you'd need to decide how to handle provisional flags if they are present in self.character_profiles etc.
        # For simplicity, current implementation passes them as is.
        char_profiles_json = json.dumps(self._get_filtered_profiles_for_prompt(chapter_number -1), indent=2, ensure_ascii=False, default=str)
        world_building_json = json.dumps(self._get_filtered_world_for_prompt(chapter_number -1), indent=2, ensure_ascii=False, default=str)

        prompt = f"""You are an expert novelist continuing Chapter {chapter_number} of "{self.plot_outline.get('title', 'Untitled Novel')}".
        **Story Bible:** Genre: {self.plot_outline.get('genre', 'N/A')}, Theme: {self.plot_outline.get('theme', 'N/A')}, Protagonist: {self.plot_outline.get('protagonist_name', 'N/A')}, Arc: {self.plot_outline.get('character_arc', 'N/A')}, Setting: {self.plot_outline.get('setting', 'N/A')}, Conflict: {self.plot_outline.get('conflict', 'N/A')}
        **Focus for THIS Chapter:** {plot_point_focus}
        {plan_section}
        **Current World Building Notes (JSON - Note any provisional info):** ```json\n{world_building_json}\n```
        **Current Character Profiles (JSON - Note any provisional info):** ```json\n{char_profiles_json}\n```
        **Context from Previous Relevant Chapters (Note any provisional summaries):**\n--- BEGIN CONTEXT ---\n{context if context else "No previous context."}\n--- END CONTEXT ---
        **Instructions:** Write a compelling chapter (1000-2000 words). Follow plan/focus. Maintain consistency with non-provisional data. Smooth flow. Vivid prose for genre '{self.plot_outline.get('genre', 'story')}'. **Output ONLY chapter text.**
        --- BEGIN CHAPTER {chapter_number} ---
        """
        raw_text = llm_interface.call_llm(prompt, temperature=0.6)
        if not raw_text:
            return None, None
        cleaned_text = llm_interface.clean_model_response(raw_text)
        if not cleaned_text or len(cleaned_text) < config.MIN_ACCEPTABLE_DRAFT_LENGTH:
             logger.error(f"Ch {chapter_number} draft too short ({len(cleaned_text or '')} chars).")
             return None, raw_text
        logger.info(f"Generated initial draft for ch {chapter_number} (Len: {len(cleaned_text)}).")
        return cleaned_text, raw_text

    def _evaluate_draft(self, draft_text: str, chapter_number: int, previous_chapters_context: str) -> EvaluationResult:
        logger.info(f"Performing evaluation checks for chapter {chapter_number} draft...")
        reasons: List[str] = []
        needs_revision = False
        coherence_score, consistency_issues, plot_deviation_reason = None, None, None

        if chapter_number > 1:
            current_embedding = llm_interface.get_embedding(draft_text)
            prev_embedding = self.db_manager.get_embedding_from_db(chapter_number - 1)
            if current_embedding is not None and prev_embedding is not None:
                coherence_score = utils.numpy_cosine_similarity(current_embedding, prev_embedding)
                logger.info(f"Coherence score with previous chapter ({chapter_number-1}): {coherence_score:.4f}")
                if coherence_score < config.REVISION_COHERENCE_THRESHOLD:
                    needs_revision = True
                    reasons.append(f"Low coherence (Score: {coherence_score:.4f}).")
            else:
                logger.warning(f"Could not perform coherence check for ch {chapter_number}.")
        else:
            logger.info("Skipping coherence check for Chapter 1.")

        if config.REVISION_CONSISTENCY_TRIGGER:
            consistency_issues = self._check_consistency(draft_text, chapter_number, previous_chapters_context)
            if consistency_issues:
                needs_revision = True
                reasons.append(f"Consistency issues:\n{consistency_issues}")
        
        if config.PLOT_ARC_VALIDATION_TRIGGER:
            plot_deviation_reason = self._validate_plot_arc(draft_text, chapter_number)
            if plot_deviation_reason:
                needs_revision = True
                reasons.append(f"Plot Arc Deviation: {plot_deviation_reason}")

        if len(draft_text) < config.MIN_ACCEPTABLE_DRAFT_LENGTH:
            needs_revision = True
            reasons.append(f"Draft too short ({len(draft_text)} chars).")
            
        logger.info(f"Evaluation complete for Chapter {chapter_number}. Needs revision: {needs_revision}.")
        return {"needs_revision": needs_revision, "reasons": reasons, "coherence_score": coherence_score, "consistency_issues": consistency_issues, "plot_deviation_reason": plot_deviation_reason}

    def _revise_chapter(self, original_text: str, chapter_number: int, reason: str, context_from_previous: str, plan: Optional[str]) -> Optional[Tuple[str, str]]:
        if not original_text or not reason:
            logger.error(f"Revision for ch {chapter_number} missing text/reason.")
            return None
        clean_reason = llm_interface.clean_model_response(reason).strip()
        if not clean_reason:
            logger.error(f"Revision reason for ch {chapter_number} empty.")
            return None
            
        logger.warning(f"Attempting revision for chapter {chapter_number}. Reason:\n{clean_reason}")
        context_limit, original_text_limit, plan_limit = config.MAX_CONTEXT_LENGTH // 3, config.MAX_CONTEXT_LENGTH // 3, config.MAX_CONTEXT_LENGTH // 3
        context_snippet = context_from_previous[:context_limit]
        original_snippet = original_text[:original_text_limit]
        
        plan_focus_section = ""
        if plan and config.ENABLE_AGENTIC_PLANNING and plan != "Agentic planning disabled by configuration.":
            plan_focus_section = f"**Original Chapter Plan (Target for Revision):**\n{plan[:plan_limit]}\n"
        else:
            plot_point_focus, _ = self._get_plot_point_info(chapter_number)
            plan_focus_section = f"**Original Chapter Focus (Target for Revision):**\n{plot_point_focus}\n"
            
        protagonist_name = self.plot_outline.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME)
        prompt = f"""You are a skilled revising author rewriting Chapter {chapter_number} (protagonist: {protagonist_name}) to correct issues.
        **Critique / Reason(s) for Revision (MUST be addressed):**\n--- FEEDBACK START ---\n{clean_reason}\n--- FEEDBACK END ---\n{plan_focus_section}
        **Context from Previous Relevant Chapters (Note any provisional summaries):**\n--- BEGIN CONTEXT ---\n{context_snippet if context_snippet else "No previous context."}\n--- END CONTEXT ---
        **Original Draft Snippet of Chapter {chapter_number} (Reference ONLY - DO NOT simply copy/tweak):**\n--- BEGIN ORIGINAL DRAFT SNIPPET ---\n{original_snippet}\n--- END ORIGINAL DRAFT SNIPPET ---
        **Instructions:** 1. **PRIORITY:** Fix issues from critique. 2. **Rewrite ENTIRE chapter.** 3. Ensure logical flow with Context. 4. Preserve tone/style but prioritize fixes. 5. Aim for length >= {config.MIN_ACCEPTABLE_DRAFT_LENGTH} chars. 6. **Output ONLY rewritten chapter text.**
        --- BEGIN REVISED CHAPTER {chapter_number} ---
        """
        revised_raw = llm_interface.call_llm(prompt, temperature=0.6) # Potentially adjust temperature for revision
        if not revised_raw:
            logger.error(f"Revision call failed for ch {chapter_number}.")
            return None
            
        revised_cleaned = llm_interface.clean_model_response(revised_raw)
        if not revised_cleaned or len(revised_cleaned) < config.MIN_ACCEPTABLE_DRAFT_LENGTH:
            logger.error(f"Revision for ch {chapter_number} too short ({len(revised_cleaned or '')} chars).")
            self._save_debug_output(chapter_number, "revision_raw_fail_short", revised_raw)
            return None
            
        original_embedding = llm_interface.get_embedding(original_text)
        revised_embedding = llm_interface.get_embedding(revised_cleaned)
        if original_embedding is not None and revised_embedding is not None:
            similarity_score = utils.numpy_cosine_similarity(original_embedding, revised_embedding)
            logger.info(f"Revision similarity score: {similarity_score:.4f}")
            if similarity_score >= config.REVISION_SIMILARITY_ACCEPTANCE:
                logger.warning(f"Revision for ch {chapter_number} rejected: Too similar (Score: {similarity_score:.4f}).")
                self._save_debug_output(chapter_number, "revision_raw_rejected_similar", revised_raw)
                return None
            logger.info(f"Revision for ch {chapter_number} accepted (Similarity: {similarity_score:.4f}).")
            return revised_cleaned, revised_raw
        else:
            logger.warning(f"Could not get embeddings for revision similarity check of ch {chapter_number}. Accepting revision.")
            return revised_cleaned, revised_raw

    def _finalize_chapter_core(self, chapter_number: int, final_text: str, raw_log: str, from_flawed_draft: bool) -> bool:
        logger.info(f"Starting core finalization process for chapter {chapter_number} (From flawed draft: {from_flawed_draft})...")
        if not final_text:
            logger.error(f"Cannot finalize chapter {chapter_number}: Final text is missing.")
            return False

        summary = self._summarize_chapter(final_text, chapter_number)
        final_embedding = llm_interface.get_embedding(final_text)
        if final_embedding is None:
            logger.error(f"CRITICAL: Failed to generate embedding for final version of chapter {chapter_number}.")

        try:
            self.db_manager.save_chapter_data(
                chapter_number,
                final_text,
                raw_log,
                summary,
                final_embedding,
                from_flawed_draft # Pass the flag
            )
        except Exception as e:
            logger.error(f"Database save failed for chapter {chapter_number}: {e}", exc_info=True)
            return False

        try:
            with open(os.path.join(config.OUTPUT_DIR, f"chapter_{chapter_number}.txt"), 'w', encoding='utf-8') as f:
                f.write(final_text)
            with open(os.path.join(config.OUTPUT_DIR, f"chapter_{chapter_number}_raw_log.txt"), 'w', encoding='utf-8') as f:
                f.write(raw_log)
            logger.info(f"Saved chapter text & raw log for chapter {chapter_number}.")
        except IOError as e:
            logger.error(f"Failed to write chapter text/log files for chapter {chapter_number}: {e}", exc_info=True)
        
        logger.info(f"Core finalization complete for chapter {chapter_number}.")
        return True

    def _update_knowledge_bases(self, chapter_number: int, final_text: str, from_flawed_draft: bool):
        if not final_text:
            logger.warning(f"Skipping knowledge base update for ch {chapter_number}: Final text is missing.")
            return
        logger.info(f"Updating knowledge bases (JSON & KG) for chapter {chapter_number} (From flawed draft: {from_flawed_draft})...")
        try:
            self._update_character_profiles(final_text, chapter_number, from_flawed_draft)
            self._update_world_building(final_text, chapter_number, from_flawed_draft) 
            logger.info(f"JSON knowledge bases updated for chapter {chapter_number}.")
            
            self._extract_and_update_kg(final_text, chapter_number, from_flawed_draft)
            logger.info(f"Knowledge Graph updated for chapter {chapter_number}.")
        except Exception as e:
            logger.error(f"Error occurred during knowledge base update for chapter {chapter_number}: {e}", exc_info=True)

    def _get_context(self, current_chapter_number: int) -> str:
        if current_chapter_number <= 1:
            return ""
        logger.debug(f"Retrieving context for Chapter {current_chapter_number}...")
        plot_point_focus, plot_point_index = self._get_plot_point_info(current_chapter_number)
        context_query_text = plot_point_focus if plot_point_focus else f"Narrative context for chapter {current_chapter_number}."
        
        if plot_point_focus:
            logger.info(f"Context query for ch {current_chapter_number} from Plot Point {plot_point_index + 1}: '{context_query_text[:100]}...'")
        else:
            logger.warning(f"No plot point for ch {current_chapter_number}. Generic context query.")
            
        query_embedding = llm_interface.get_embedding(context_query_text)
        
        if query_embedding is None:
            logger.warning("Failed embedding context query. Falling back to prev chapter summary/text.")
            # Fetch previous chapter data, check its provisional status
            prev_chap_db_data = self.db_manager.get_chapter_data_from_db(current_chapter_number - 1) # This needs to be extended to return provisional flag
            if prev_chap_db_data:
                # Assuming get_chapter_data_from_db is modified to return a dict that includes 'is_provisional'
                # For now, we can't easily get this without changing get_chapter_data_from_db
                # So the fallback won't explicitly mark provisional, but the log indicates fallback.
                fallback_context = prev_chap_db_data.get('summary') or prev_chap_db_data.get('text', '')
                if fallback_context:
                    logger.info(f"Using fallback context ({len(fallback_context)} chars).")
                    return fallback_context[:config.MAX_CONTEXT_LENGTH]
            logger.warning("Fallback context failed.")
            return ""
            
        past_embeddings = self.db_manager.get_all_past_embeddings(current_chapter_number)
        if not past_embeddings: return ""
        
        similarities = sorted(
            [(chap_num, utils.numpy_cosine_similarity(query_embedding, emb)) for chap_num, emb in past_embeddings if emb is not None],
            key=lambda item: item[1],
            reverse=True
        )
        if not similarities: return ""
        
        top_n_indices = [cs[0] for cs in similarities[:config.CONTEXT_CHAPTER_COUNT]]
        logger.info(f"Top {len(top_n_indices)} relevant chapters for context: {top_n_indices} (Scores: {[f'{s:.3f}' for _, s in similarities[:config.CONTEXT_CHAPTER_COUNT]]})")
        
        if (current_chapter_number - 1) > 0 and (current_chapter_number - 1) not in top_n_indices:
            top_n_indices.append(current_chapter_number - 1) # Ensure immediate previous chapter is included
            
        context_parts, total_chars, chapters_to_fetch = [], 0, sorted(list(set(top_n_indices)))
        logger.debug(f"Fetching context data for chapters: {chapters_to_fetch}")
        
        for chap_num in chapters_to_fetch:
            if total_chars >= config.MAX_CONTEXT_LENGTH: break
            # Modify get_chapter_data_from_db to return the row, which includes 'is_provisional'
            chap_data_row = self.db_manager.get_chapter_data_from_db(chap_num) # Assuming this returns a dict-like Row
            if chap_data_row:
                content = (chap_data_row.get('summary') or chap_data_row.get('text', '')).strip()
                is_prov = chap_data_row.get('is_provisional', False)
                content_type = "Provisional Summary" if chap_data_row.get('summary') and is_prov else \
                               "Summary" if chap_data_row.get('summary') else \
                               "Provisional Text Snippet" if is_prov else "Text Snippet"
                
                if content:
                    prefix = f"[From Chapter {chap_num} ({content_type})]:\n"
                    suffix = "\n---\n"
                    formatting_chars = len(prefix) + len(suffix) + 10 # conservative estimate
                    content_limit = max(0, config.MAX_CONTEXT_LENGTH - total_chars - formatting_chars)
                    
                    if content_limit > 0:
                        truncated_content = content[:content_limit]
                        context_parts.append(f"{prefix}{truncated_content}{suffix}")
                        total_chars += len(truncated_content) + formatting_chars
                        logger.debug(f"Added context from ch {chap_num} ({content_type}), {len(truncated_content)} chars. Total: {total_chars}.")
            else:
                logger.warning(f"Could not retrieve chapter data for {chap_num} for context.")
                
        final_context = "\n".join(context_parts).strip()
        logger.info(f"Constructed final semantic context: {len(final_context)} chars from chapters {chapters_to_fetch}.")
        return final_context

    def _summarize_chapter(self, chapter_text: Optional[str], chapter_number: int) -> Optional[str]:
        if not chapter_text or len(chapter_text) < 50:
            return None
        snippet = chapter_text[:config.KNOWLEDGE_UPDATE_SNIPPET_SIZE]
        prompt = f"""Summarize Chapter {chapter_number} (1-3 sentences), capturing crucial plot advancements, character decisions, or revelations. Be succinct.
        Chapter Text Snippet:\n--- BEGIN TEXT ---\n{snippet}\n--- END TEXT ---\nOutput ONLY summary text.
        """
        summary_raw = llm_interface.call_llm(prompt, temperature=0.6, max_tokens=config.MAX_SUMMARY_TOKENS if hasattr(config, 'MAX_SUMMARY_TOKENS') else 4096) # Added fallback for max_tokens
        cleaned_summary = llm_interface.clean_model_response(summary_raw).strip()
        if cleaned_summary:
            logger.info(f"Generated summary for ch {chapter_number}: '{cleaned_summary[:100]}...'")
            return cleaned_summary
        else:
            logger.warning(f"Failed to generate valid summary for ch {chapter_number}.")
            return None

    def _check_consistency(self, chapter_draft_text: Optional[str], chapter_number: int, previous_chapters_context: str) -> Optional[str]:
        if not chapter_draft_text: return None
        
        draft_snippet = chapter_draft_text[:config.KNOWLEDGE_UPDATE_SNIPPET_SIZE]
        context_snippet = previous_chapters_context[:config.KNOWLEDGE_UPDATE_SNIPPET_SIZE // 2]
        
        kg_facts_for_consistency_prompt: List[str] = []
        protagonist_name = self.plot_outline.get("protagonist_name")

        if chapter_number > 1 and protagonist_name:
            logger.debug(f"Gathering reliable KG facts for Chapter {chapter_number} consistency check...")
            # Fetch reliable (non-provisional) KG facts from previous chapters
            loc = self.db_manager.get_most_recent_value(protagonist_name, "located_in", chapter_number - 1, include_provisional=False)
            if loc: kg_facts_for_consistency_prompt.append(f"- {protagonist_name} was last reliably known to be at: {loc}.")
            
            status = self.db_manager.get_most_recent_value(protagonist_name, "status_is", chapter_number - 1, include_provisional=False)
            if status: kg_facts_for_consistency_prompt.append(f"- {protagonist_name}'s last reliable status was: {status}.")
            
            # Add more key entities/attributes as needed, e.g., key allies, items.
            # Example:
            # key_item = self.plot_outline.get("key_macguffin_name")
            # if key_item:
            #    item_holder = self.db_manager.get_most_recent_value(key_item, "possessed_by", chapter_number - 1, include_provisional=False)
            #    if item_holder: kg_facts_for_consistency_prompt.append(f"- {key_item} was last reliably possessed by: {item_holder}.")

        kg_check_results_text = "**Key Reliable KG Facts (from prev chapters):**\n" + "\n".join(kg_facts_for_consistency_prompt) + "\n" if kg_facts_for_consistency_prompt else "**Key Reliable KG Facts:** None available for comparison.\n"

        # Prepare JSON context, noting that filtering for provisional data within these complex structures is hard.
        # The prompt will instruct the LLM to prioritize plot/profile/world data.
        # For future: _get_filtered_profiles_for_prompt and _get_filtered_world_for_prompt could be enhanced.
        char_profiles_for_prompt = self._get_filtered_profiles_for_prompt(chapter_number - 1)
        world_building_for_prompt = self._get_filtered_world_for_prompt(chapter_number -1)

        prompt = f"""You are a continuity editor. Analyze Chapter {chapter_number} Draft Snippet.
        Compare against: 
        1. Plot Outline.
        2. Character Profiles (consider this as established canon).
        3. World Building (consider this as established canon).
        4. {kg_check_results_text} (Pay close attention to these established facts from reliable past knowledge).
        5. Context from Previous Chapters (consider this for narrative flow).
        6. Internal consistency within the Draft itself.

        List ONLY specific, objective contradictions, factual inconsistencies, or significant deviations. 
        Prioritize issues contradicting established facts from Plot Outline, Character Profiles, World Building, or the provided Reliable KG Facts.
        If NO significant inconsistencies, respond ONLY with: None

        **Plot Outline:** ```json\n{json.dumps(self.plot_outline, indent=2, ensure_ascii=False, default=str)}\n```
        **Character Profiles:** ```json\n{json.dumps(char_profiles_for_prompt, indent=2, ensure_ascii=False, default=str)}\n```
        **World Building:** ```json\n{json.dumps(world_building_for_prompt, indent=2, ensure_ascii=False, default=str)}\n```
        **Context from Previous Chapters (Snippet):**\n--- BEGIN PREVIOUS CONTEXT ---\n{context_snippet if context_snippet else "N/A (e.g., Ch 1)."}\n--- END PREVIOUS CONTEXT ---
        **Chapter {chapter_number} Draft Text Snippet (to analyze):**\n--- BEGIN DRAFT ---\n{draft_snippet}\n--- END DRAFT ---
        **List specific inconsistencies (or "None"):**
        """
        response_raw = llm_interface.call_llm(prompt, temperature=0.6, max_tokens=config.MAX_CONSISTENCY_TOKENS if hasattr(config, 'MAX_CONSISTENCY_TOKENS') else 8192) # Added fallback
        response_cleaned = llm_interface.clean_model_response(response_raw).strip()

        if not response_cleaned or response_cleaned.lower() == "none":
            logger.info(f"Consistency check passed for ch {chapter_number}.")
            return None
        else:
            logger.warning(f"Consistency issues for ch {chapter_number}:\n{response_cleaned}")
            return response_cleaned

    def _validate_plot_arc(self, chapter_draft_text: Optional[str], chapter_number: int) -> Optional[str]:
        if not chapter_draft_text: return None
        plot_point_focus, plot_point_index = self._get_plot_point_info(chapter_number)
        if plot_point_focus is None:
            return None 
        
        logger.info(f"Validating plot arc for ch {chapter_number} against Plot Point {plot_point_index + 1}: '{plot_point_focus[:100]}...'")
        summary = self._summarize_chapter(chapter_draft_text, chapter_number)
        validation_text = summary if summary and len(summary) > 50 else chapter_draft_text[:1500]
        if not validation_text: return None
            
        protagonist_name = self.plot_outline.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME)
        prompt = f"""You are a story structure analyst. Determine if Chapter {chapter_number} Text (protagonist: {protagonist_name}) addresses the core of its Intended Plot Point.
        **Intended Plot Point (Plot Point {plot_point_index + 1} for Chapter {chapter_number}):** "{plot_point_focus}"
        **Chapter {chapter_number} Text (Summary or Snippet to analyze):** "{validation_text}"
        **Evaluation:** Does Chapter Text content/events align with Intended Plot Point?
        **CRITICAL INSTRUCTION:** Respond ONLY with `Yes` (if aligns) OR `No, because...` (if deviates, 1-2 sentence concise explanation).
        Response:"""
        validation_response_raw = llm_interface.call_llm(prompt, temperature=0.6, max_tokens=config.MAX_PLOT_VALIDATION_TOKENS if hasattr(config, 'MAX_PLOT_VALIDATION_TOKENS') else 8192) # Added fallback
        cleaned_plot_response = llm_interface.clean_model_response(validation_response_raw).strip()
        
        if cleaned_plot_response.lower().startswith("yes"): # Accept "Yes." or "Yes, because..."
            logger.info(f"Plot arc validation passed for ch {chapter_number}.")
            return None
        elif cleaned_plot_response.lower().startswith("no, because"):
            reason = cleaned_plot_response[len("no, because"):].strip() or "LLM indicated deviation, no specific reason."
            logger.warning(f"Plot arc deviation for ch {chapter_number}: {reason}")
            return reason
        else:
            logger.warning(f"Plot arc validation for ch {chapter_number} ambiguous response: '{cleaned_plot_response}'. Assuming alignment for safety, but review prompt/response.")
            return None # Or treat ambiguity as failure depending on strictness.

    def _update_character_profiles(self, chapter_text: Optional[str], chapter_number: int, from_flawed_draft: bool):
        if not chapter_text: return
        text_snippet = chapter_text[:config.KNOWLEDGE_UPDATE_SNIPPET_SIZE]
        protagonist_name = self.plot_outline.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME)
        
        dynamic_instructions = ""
        if config.ENABLE_DYNAMIC_STATE_ADAPTATION:
            dynamic_instructions = """3. **Dynamic Adaptation:** For existing characters, propose modifications to `traits` or `description` using `"modification_proposal"` field if warranted. Example: `"modification_proposal": "MODIFY traits: ADD 'Determined'"`.
4. **Crucially:** Only include characters updated, newly introduced, or with a modification proposal in output JSON."""
        else:
            dynamic_instructions = "3. **Crucially:** Only include characters updated or newly introduced in output JSON."
            
        # Pass existing profiles (potentially filtered for provisional if desired)
        current_profiles_for_prompt = self._get_filtered_profiles_for_prompt(chapter_number -1)

        prompt = f"""You are a literary analyst. Analyze Chapter {chapter_number} snippet (protagonist: {protagonist_name}) for updates to character profiles. Output MUST be a single, valid JSON object.
        **Chapter Text Snippet:**\n--- BEGIN TEXT ---\n{text_snippet}...\n--- END TEXT ---\n
        **Current Character Profiles (reference - note if any data is provisional from past chapters):**\n```json\n{json.dumps(current_profiles_for_prompt, indent=2, ensure_ascii=False, default=str)}\n```
        **Instructions:** 1. Identify characters updated/introduced. 2. Note new traits, relationship changes, status, description. Add `development_in_chapter_{chapter_number}` key summarizing role/change.
        {dynamic_instructions}
        5. **CRITICAL: Output ONLY JSON.** If no updates, output `{{}}`.
        **Example Output (only updated chars):**\n```json\n{{ "Char1": {{ "traits": ["New"], "modification_proposal": "MODIFY status: 'Injured'", "development_in_chapter_{chapter_number}": "Fought bravely."}} }}\n```
        """
        logger.info(f"Analyzing Chapter {chapter_number} to update character profiles JSON (Dynamic: {config.ENABLE_DYNAMIC_STATE_ADAPTATION}, From Flawed Draft: {from_flawed_draft})...")
        raw_analysis = llm_interface.call_llm(prompt, temperature=0.6)
        updates = llm_interface.parse_llm_json_response(raw_analysis, f"character profile update for chapter {chapter_number}")
        
        if not updates or not isinstance(updates, dict) or not updates:
            logger.info(f"LLM found no character profile JSON updates in ch {chapter_number}.")
            return
            
        logger.info(f"Merging character profile JSON updates for ch {chapter_number} for characters: {list(updates.keys())}")
        updated_chars_count, new_chars_count = 0, 0
        
        provisional_marker_key = f"source_quality_chapter_{chapter_number}"

        for char_name, char_update_data in updates.items():
            if not isinstance(char_update_data, dict): continue
            
            # Make a copy to add provisional marker without affecting other loops if `updates` is reused
            char_update = char_update_data.copy()

            if from_flawed_draft:
                char_update[provisional_marker_key] = "provisional_from_unrevised_draft"

            dev_key = f"development_in_chapter_{chapter_number}"
            if dev_key not in char_update and (len(char_update) > 1 or (len(char_update) == 1 and "modification_proposal" not in char_update)):
                 char_update[dev_key] = "Character appeared/mentioned in chapter." # More generic
            
            if char_name not in self.character_profiles:
                new_chars_count += 1
                logger.info(f"Adding new character '{char_name}' from ch {chapter_number}.")
                self.character_profiles[char_name] = {
                    "description": char_update.get("description", f"Introduced in Ch {chapter_number}."),
                    "traits": sorted(list(set(t for t in char_update.get("traits", []) if isinstance(t, str) and t.strip()))),
                    "relationships": char_update.get("relationships", {}), # Ensure it's a dict
                    "status": char_update.get("status", "Newly introduced")
                }
                # Add dev key and provisional marker to the newly created profile
                self.character_profiles[char_name][dev_key] = char_update.get(dev_key, f"Introduced in Ch {chapter_number}.")
                if from_flawed_draft:
                    self.character_profiles[char_name][provisional_marker_key] = char_update[provisional_marker_key]

                if config.ENABLE_DYNAMIC_STATE_ADAPTATION and "modification_proposal" in char_update:
                    self._apply_modification_proposal(self.character_profiles[char_name], char_update["modification_proposal"], char_name, "character profile")
            else: 
                updated_chars_count += 1
                logger.debug(f"Updating existing character '{char_name}' from ch {chapter_number}.")
                existing_profile = self.character_profiles[char_name]
                
                if from_flawed_draft: # Mark existing profile as having updates from a flawed source for this chapter
                    existing_profile[provisional_marker_key] = char_update[provisional_marker_key]

                if config.ENABLE_DYNAMIC_STATE_ADAPTATION and "modification_proposal" in char_update:
                    self._apply_modification_proposal(existing_profile, char_update["modification_proposal"], char_name, "character profile")
                
                for key, value in char_update.items():
                    if key == "modification_proposal" or key == provisional_marker_key: continue # Skip these markers
                    
                    if key == "traits" and isinstance(value, list):
                        if "traits" not in existing_profile or not isinstance(existing_profile["traits"], list): existing_profile["traits"] = []
                        existing_profile["traits"] = sorted(list(set(existing_profile["traits"]).union(set(t for t in value if isinstance(t, str) and t.strip()))))
                    elif key == "relationships" and isinstance(value, dict):
                         if not isinstance(existing_profile.get("relationships"), dict): existing_profile["relationships"] = {}
                         existing_profile["relationships"].update(value)
                    elif key == "description" and isinstance(value, str) and value.strip():
                         # Avoid overwriting description if a dynamic proposal is also trying to modify it
                         if not (config.ENABLE_DYNAMIC_STATE_ADAPTATION and "modification_proposal" in char_update and "MODIFY DESCRIPTION" in char_update["modification_proposal"].upper()):
                            existing_profile["description"] = value.strip()
                    elif key == dev_key and isinstance(value, str) and value.strip():
                        existing_profile[key] = value.strip() # Add development note for current chapter
                    elif key == "status" and isinstance(value, str) and value.strip():
                        existing_profile["status"] = value.strip()
                    # Handle other keys as needed, or add them if not present
                    elif key not in existing_profile and value is not None:
                         existing_profile[key] = value


        if updated_chars_count > 0 or new_chars_count > 0:
            logger.info(f"Character profile JSON merge complete. Updated: {updated_chars_count}, New: {new_chars_count}.")
        else:
            logger.info(f"No character profiles effectively updated/added in ch {chapter_number}.")

    def _apply_modification_proposal(self, profile_or_item: Dict[str, Any], proposal: str, item_name: str, item_type_for_log: str):
        # (This method remains largely the same)
        if not isinstance(proposal, str) or not proposal.strip():
            logger.debug(f"Empty proposal for '{item_name}'.")
            return
        logger.debug(f"Applying modification proposal for '{item_name}' ({item_type_for_log}): '{proposal}'")
        proposal_norm = proposal.strip().upper()
        try:
            match_modify_key = re.match(r"MODIFY\s+([\w_]+)\s*:", proposal_norm)
            if not match_modify_key:
                logger.warning(f"Invalid proposal format for '{item_name}': '{proposal}'")
                return
            
            key_to_modify_upper = match_modify_key.group(1).strip()
            # Find original key case-insensitively or use lowercase as fallback
            original_key = next((k for k in profile_or_item if k.upper() == key_to_modify_upper), key_to_modify_upper.lower())
            
            action_details_original_case = proposal[match_modify_key.end():].strip()

            if original_key.lower() == "traits": # Special handling for traits list
                if "traits" not in profile_or_item or not isinstance(profile_or_item["traits"], list):
                    profile_or_item["traits"] = []
                current_traits_set = set(profile_or_item["traits"])
                for match_add in re.finditer(r"ADD\s+['\"]([^'\"]+)['\"]", action_details_original_case, re.IGNORECASE):
                    trait_to_add = match_add.group(1).strip()
                    if trait_to_add: current_traits_set.add(trait_to_add)
                for match_remove in re.finditer(r"REMOVE\s+['\"]([^'\"]+)['\"]", action_details_original_case, re.IGNORECASE):
                    trait_to_remove = match_remove.group(1).strip()
                    if trait_to_remove: current_traits_set.discard(trait_to_remove)
                profile_or_item["traits"] = sorted(list(current_traits_set))
                logger.info(f"Applied trait modifications for '{item_name}'. New traits: {profile_or_item['traits']}")
            else: # Generic key modification
                new_value_str = action_details_original_case.strip("'\" ")
                if new_value_str: # Ensure there's actually a value to set
                    profile_or_item[original_key] = new_value_str 
                    logger.info(f"Applied modification to '{original_key}' for '{item_name}'. New value: '{new_value_str[:50]}...'")
                else:
                    logger.warning(f"Modification proposal for '{original_key}' of '{item_name}' resulted in an empty new value. Proposal: '{proposal}'")
        except Exception as e:
            logger.error(f"Error applying modification proposal for '{item_name}': {e}. Proposal: '{proposal}'", exc_info=True)

    def _update_world_building(self, chapter_text: Optional[str], chapter_number: int, from_flawed_draft: bool):
        if not chapter_text: return
        text_snippet = chapter_text[:config.KNOWLEDGE_UPDATE_SNIPPET_SIZE]
        dynamic_instructions = ""
        if config.ENABLE_DYNAMIC_STATE_ADAPTATION:
            dynamic_instructions = """6. **Dynamic Adaptation:** Propose modifications to existing items using `"modification_proposal"`. Example: `"modification_proposal": "MODIFY atmosphere: 'Now heavy.'"`. For items like history events, modify their "description" key: `"modification_proposal": "MODIFY description: 'New detail.'"`.
7. **CRITICAL: Output ONLY JSON of new/updated elements.**"""
        else:
            dynamic_instructions = "6. **CRITICAL: Output ONLY JSON of new/updated elements."

        current_world_for_prompt = self._get_filtered_world_for_prompt(chapter_number-1)

        prompt = f"""You are a world-building analyst. Examine Chapter {chapter_number} snippet for new info or significant changes to existing world elements. Output MUST be a single, valid JSON object.
        **Chapter Text Snippet:**\n--- BEGIN TEXT ---\n{text_snippet}...\n--- END TEXT ---\n
        **Current World Building Notes (reference - note if any data is provisional from past chapters):**\n```json\n{json.dumps(current_world_for_prompt, indent=2, ensure_ascii=False, default=str)}\n```
        **Instructions:** 1. Identify new/changed locations (ensure dicts). 2. Note new/changed society, factions (ensure dicts). 3. Extract new/changed systems, tech, magic (ensure dicts). 4. Capture new/changed lore, history (ensure items are dicts with "description" or "text"). 5. Focus on THIS chapter. Add `elaboration_in_chapter_{chapter_number}`.
        {dynamic_instructions}
        8. Use valid JSON. If no updates, output `{{}}`.
        **Example Update (only new/changed):**\n```json\n{{ "locations": {{ "Existing Loc": {{ "modification_proposal": "MODIFY atmosphere: 'Heavy.'", "elaboration_in_chapter_{chapter_number}": "Inscription found."}} }}, "history": {{ "EventX": {{ "modification_proposal": "MODIFY description: 'New detail.'" }} }} }}\n```
        """
        logger.info(f"Analyzing Chapter {chapter_number} to update world-building JSON (Dynamic: {config.ENABLE_DYNAMIC_STATE_ADAPTATION}, From Flawed Draft: {from_flawed_draft})...")
        raw_analysis = llm_interface.call_llm(prompt, temperature=0.6)
        updates = llm_interface.parse_llm_json_response(raw_analysis, f"world-building update for chapter {chapter_number}")
        
        if not updates or not isinstance(updates, dict) or not updates:
            logger.info(f"LLM found no world-building JSON updates in ch {chapter_number}.")
            return
            
        logger.info(f"Merging world-building JSON updates for ch {chapter_number} for categories: {list(updates.keys())}")
        
        items_affected_count = 0
        provisional_marker_key = f"source_quality_chapter_{chapter_number}"

        for category_key, category_updates_dict_raw in updates.items():
            if not isinstance(category_updates_dict_raw, dict) or not category_updates_dict_raw: continue
            
            category_updates_dict = category_updates_dict_raw.copy() # Work with a copy

            if category_key not in self.world_building: self.world_building[category_key] = {}
            if not isinstance(self.world_building[category_key], dict):
                logger.warning(f"Overwriting non-dict world category '{category_key}'.")
                self.world_building[category_key] = {}
            
            target_category_dict = self.world_building[category_key]
            
            # If the entire category update is from a flawed draft, mark it.
            # This is a coarse-grained provisional marker for the category for this chapter.
            if from_flawed_draft:
                 target_category_dict[provisional_marker_key] = "provisional_from_unrevised_draft"


            for item_name, item_update_details_raw in category_updates_dict.items():
                if not isinstance(item_update_details_raw, dict):
                    logger.warning(f"Skipping invalid item_details for '{item_name}' in '{category_key}'.")
                    continue
                
                item_update_details = item_update_details_raw.copy()
                item_log_name = f"{category_key}.{item_name}"
                
                # Add provisional marker to the update details if source is flawed
                # This will be handled by _robust_merge_world_item or applied after
                if from_flawed_draft:
                    item_update_details[provisional_marker_key] = "provisional_from_unrevised_draft"

                existing_item_data = target_category_dict.get(item_name)
                
                if existing_item_data is None: # New item
                    logger.info(f"Adding new world item '{item_log_name}'.")
                    new_item = self._robust_merge_world_item({}, item_update_details, item_log_name, chapter_number, from_flawed_draft)
                    new_item[f"added_in_chapter_{chapter_number}"] = True # Mark when it was added
                    target_category_dict[item_name] = new_item
                    items_affected_count +=1
                else: # Existing item
                    updated_item = self._robust_merge_world_item(existing_item_data, item_update_details, item_log_name, chapter_number, from_flawed_draft)
                    target_category_dict[item_name] = updated_item
                    if updated_item.get(f"updated_in_chapter_{chapter_number}") or updated_item.get(f"added_in_chapter_{chapter_number}"):
                        items_affected_count +=1
            
            # Check if any item within the category was marked as updated/added this chapter
            if any(isinstance(v,dict) and (v.get(f"updated_in_chapter_{chapter_number}") or v.get(f"added_in_chapter_{chapter_number}")) for v in target_category_dict.values()):
                 target_category_dict[f"updated_in_chapter_{chapter_number}"] = True # Mark category as updated

        if items_affected_count > 0:
            logger.info(f"World-building JSON merge completed. {items_affected_count} items/sub-items affected.")
        else:
            logger.info(f"No world-building JSON items effectively updated/added in ch {chapter_number}.")

    def _robust_merge_world_item(self, target_item: Any, update_details: Dict[str, Any], item_name_for_log: str, chapter_num: int, from_flawed_draft_source: bool) -> Dict[str, Any]:
        current_item_dict: Dict[str, Any]
        provisional_marker_key = f"source_quality_chapter_{chapter_num}"

        if not isinstance(target_item, dict):
            logger.warning(f"World item '{item_name_for_log}' not dict (type: {type(target_item)}). Converting. Original '{str(target_item)[:50]}...' to 'description'.")
            current_item_dict = {"description": str(target_item)}
            current_item_dict[f"updated_in_chapter_{chapter_num}"] = True 
        else:
            current_item_dict = target_item.copy() # Work on a copy

        item_was_modified_this_call = False

        # Apply provisional flag from this specific update source
        if from_flawed_draft_source:
            current_item_dict[provisional_marker_key] = update_details.get(provisional_marker_key, "provisional_from_unrevised_draft")
            item_was_modified_this_call = True


        if config.ENABLE_DYNAMIC_STATE_ADAPTATION and "modification_proposal" in update_details:
            proposal = update_details.pop("modification_proposal") # Remove after processing
            if isinstance(proposal, str) and proposal.strip():
                self._apply_modification_proposal(current_item_dict, proposal, item_name_for_log, "world item")
                item_was_modified_this_call = True
        
        for key, value in update_details.items():
            # Skip markers we've already handled or are purely informational for this merge
            if key.startswith("updated_in_chapter_") or key.startswith("added_in_chapter_") or key == provisional_marker_key or key == "modification_proposal":
                continue
            
            target_value = current_item_dict.get(key)
            
            if isinstance(value, dict): # Nested dictionary
                if not isinstance(target_value, dict):
                    current_item_dict[key] = {} # Initialize if not a dict
                    current_item_dict[key][f"added_in_chapter_{chapter_num}"] = True
                current_item_dict[key] = self._robust_merge_world_item(current_item_dict[key], value, f"{item_name_for_log}.{key}", chapter_num, from_flawed_draft_source)
                if current_item_dict[key].get(f"updated_in_chapter_{chapter_num}") or current_item_dict[key].get(f"added_in_chapter_{chapter_num}"):
                    item_was_modified_this_call = True
            elif isinstance(value, list): # List
                if not isinstance(target_value, list):
                    current_item_dict[key] = []
                
                initial_list_len = len(current_item_dict[key])
                # Add only new items to the list to avoid duplicates, preserving order somewhat
                for item_in_list_update in value:
                    if item_in_list_update not in current_item_dict[key]:
                        current_item_dict[key].append(item_in_list_update)
                if len(current_item_dict[key]) > initial_list_len:
                    item_was_modified_this_call = True
            elif value != target_value: # Simple value update
                current_item_dict[key] = value
                item_was_modified_this_call = True
        
        if item_was_modified_this_call and not current_item_dict.get(f"added_in_chapter_{chapter_num}"):
             # If it wasn't newly added this chapter but was modified, mark it as updated this chapter.
            current_item_dict[f"updated_in_chapter_{chapter_num}"] = True
            
        return current_item_dict

    def _extract_and_update_kg(self, chapter_text: Optional[str], chapter_number: int, from_flawed_draft: bool):
        if not chapter_text:
            logger.warning(f"Skipping KG extraction for ch {chapter_number}: Text None.")
            return
            
        logger.info(f"Extracting KG triples for chapter {chapter_number} (From flawed draft: {from_flawed_draft})...")
        text_snippet = chapter_text[:config.KNOWLEDGE_UPDATE_SNIPPET_SIZE * 4] # Use a larger snippet for KG
        if len(text_snippet) < len(chapter_text):
            logger.warning(f"KG extraction using truncated text ({len(text_snippet)} chars) for ch {chapter_number}.")
            
        protagonist_name = self.plot_outline.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME)
        common_predicates = ["is_a", "located_in", "has_trait", "status_is", "feels", "knows", "believes", "wants", "interacted_with", "travelled_to", "discovered", "acquired", "lost", "used_item", "attacked", "helped", "damaged", "repaired", "contains", "part_of", "caused_by", "leads_to", "observed", "heard", "said", "thought_about", "decided_to"]
        
        prompt = f"""You are a KG Engineer. Extract factual (Subject, Predicate, Object) triples from Chapter {chapter_number} Text Snippet (protagonist: '{protagonist_name}').
        **Chapter {chapter_number} Text Snippet:**\n--- BEGIN TEXT ---\n{text_snippet}\n--- END TEXT ---\n
        **Instructions:** 1. Identify key entities (normalize names). 2. Use suggested predicates or concise alternatives. 3. Extract facts as `["Subject", "predicate", "Object"]` (all non-empty strings). 4. Focus ONLY on info from THIS text. 5. Prioritize state changes & key events. 6. **CRITICAL OUTPUT:** ONLY a single, valid JSON list of lists. `[]` if no facts. 7. **NO extra text/markdown.** Start `[` end `]`.
        **Suggested Predicates:** {', '.join(common_predicates)}
        **Example CORRECT Output (actual output is just the list):**
        ```json
        [["{protagonist_name}", "travelled_to", "Eclipse Spire"], ["{protagonist_name}", "status_is", "Conflicted"]]
        ```
        JSON Output Only:
        [
        """
        raw_triples_json = llm_interface.call_llm(prompt, temperature=0.5, max_tokens=config.MAX_KG_TRIPLE_TOKENS if hasattr(config, 'MAX_KG_TRIPLE_TOKENS') else 2000) # Added fallback
        parsed_triples = llm_interface.parse_llm_json_response(raw_triples_json, f"KG triple extraction for chapter {chapter_number}", expect_type=list)
        
        if parsed_triples is None:
             logger.error(f"Failed to extract/parse KG triples for ch {chapter_number} after all attempts. Raw: {raw_triples_json[:200] if raw_triples_json else 'EMPTY'}")
             self._save_debug_output(chapter_number, "kg_extraction_raw_fail_final", raw_triples_json or "EMPTY")
             return
             
        added_count, skipped_count = 0, 0
        for triple in parsed_triples:
            if isinstance(triple, list) and len(triple) == 3:
                subj, pred, obj = [str(t).strip() if t is not None else "" for t in triple]
                if subj and pred and obj:
                    self.db_manager.add_kg_triple(subj, pred, obj, chapter_number, is_provisional=from_flawed_draft)
                    added_count += 1
                else:
                    logger.warning(f"Skipping invalid triple (empty component) in ch {chapter_number}: {triple}")
                    skipped_count += 1
            else:
                logger.warning(f"Skipping invalid triple format in KG for ch {chapter_number}: {triple}")
                skipped_count += 1
        logger.info(f"Added {added_count} KG triples from ch {chapter_number}. Skipped {skipped_count}. (Source Provisional: {from_flawed_draft})")

    def _get_relevant_character_state_snippet(self, current_chapter_num_for_filtering: Optional[int] = None) -> str:
        # This method now needs to be aware of provisional flags if they are stored in self.character_profiles
        snippet_data, count = {}, 0
        sorted_char_names = sorted(self.character_profiles.keys())
        protagonist_name = self.plot_outline.get("protagonist_name")
        if protagonist_name and protagonist_name in sorted_char_names:
            sorted_char_names.insert(0, sorted_char_names.pop(sorted_char_names.index(protagonist_name)))
            
        for name in sorted_char_names:
            if count >= config.PLANNING_CONTEXT_MAX_CHARACTERS_IN_SNIPPET: break
            profile = self.character_profiles.get(name, {})
            
            # Check for provisional status based on the latest update for this character
            # This requires iterating through source_quality_chapter_X keys if they exist
            is_provisional_note = ""
            if current_chapter_num_for_filtering is not None:
                # Find the most recent provisional flag for this character up to current_chapter_num_for_filtering
                # This is a simplification; a more robust way would be to check the specific fields being used.
                provisional_keys = sorted([k for k in profile if k.startswith("source_quality_chapter_") and int(k.split('_')[-1]) < current_chapter_num_for_filtering], reverse=True)
                if provisional_keys and profile.get(provisional_keys[0]) == "provisional_from_unrevised_draft":
                    is_provisional_note = " (Note: Some info may be provisional)"


            dev_notes_keys = sorted([k for k in profile if k.startswith("development_in_chapter_")], reverse=True)
            recent_dev_note = profile.get(dev_notes_keys[0], "N/A") if dev_notes_keys else "N/A"
            
            snippet_data[name] = {
                "desc_snippet": profile.get("description", "")[:config.PLANNING_CONTEXT_MAX_CHARS_PER_PROFILE_DESC] + "...",
                "status": profile.get("status", "Unknown") + is_provisional_note,
                "recent_dev_note": recent_dev_note[:config.PLANNING_CONTEXT_MAX_RECENT_DEV_PER_PROFILE] + "..."
            }
            count += 1
        return json.dumps(snippet_data, indent=2, ensure_ascii=False, default=str) if snippet_data else "No character profiles."

    def _get_relevant_world_state_snippet(self, current_chapter_num_for_filtering: Optional[int] = None) -> str:
        # Similar to character snippet, this needs to be aware of provisional flags.
        snippet_data = {}
        # This is a simplification. Ideally, you'd check provisional status of specific items.
        # For now, we'll just pass the data. The LLM prompt should mention to be wary of provisional data.
        
        if "locations" in self.world_building and isinstance(self.world_building["locations"], dict):
             # Check if category itself was marked provisional recently
             loc_provisional_note = ""
             if current_chapter_num_for_filtering and self.world_building["locations"].get(f"source_quality_chapter_{current_chapter_num_for_filtering-1}"):
                 loc_provisional_note = " (Note: Some location info may be provisional)"
             snippet_data["locations_overview" + loc_provisional_note] = list(self.world_building["locations"].keys())[:config.PLANNING_CONTEXT_MAX_LOCATIONS_IN_SNIPPET]
        
        society_data = self.world_building.get("society", {})
        if isinstance(society_data, dict) and "Key Factions" in society_data and isinstance(society_data["Key Factions"], dict):
             snippet_data["key_factions"] = list(society_data["Key Factions"].keys())[:config.PLANNING_CONTEXT_MAX_FACTIONS_IN_SNIPPET]
        
        if "systems" in self.world_building and isinstance(self.world_building["systems"], dict):
             snippet_data["key_systems"] = list(self.world_building["systems"].keys())[:config.PLANNING_CONTEXT_MAX_SYSTEMS_IN_SNIPPET]
             
        return json.dumps(snippet_data, indent=2, ensure_ascii=False, default=str) if snippet_data else "No world-building data."

    def _get_filtered_profiles_for_prompt(self, up_to_chapter: Optional[int] = None) -> Dict[str, Any]:
        """
        Prepares character profiles for inclusion in a prompt.
        Conceptually, this is where you would filter or annotate provisional data.
        For now, it returns a copy, but you can extend it.
        """
        # Create a deep copy to avoid modifying original
        profiles_copy = json.loads(json.dumps(self.character_profiles)) # Simple deep copy

        if up_to_chapter is None: # If no chapter limit, return all
            return profiles_copy

        # Iterate and add notes about provisional data for the prompt
        for char_name, profile_data in profiles_copy.items():
            provisional_notes_for_char = []
            for i in range(1, up_to_chapter + 1):
                prov_key = f"source_quality_chapter_{i}"
                if profile_data.get(prov_key) == "provisional_from_unrevised_draft":
                    provisional_notes_for_char.append(f"Info from ch {i} is provisional.")
            
            if provisional_notes_for_char:
                if "prompt_notes" not in profile_data:
                    profile_data["prompt_notes"] = []
                profile_data["prompt_notes"].extend(provisional_notes_for_char)
        return profiles_copy

    def _get_filtered_world_for_prompt(self, up_to_chapter: Optional[int] = None) -> Dict[str, Any]:
        """
        Prepares world building data for inclusion in a prompt.
        Conceptually, this is where you would filter or annotate provisional data.
        For now, it returns a copy.
        """
        world_copy = json.loads(json.dumps(self.world_building)) # Simple deep copy

        if up_to_chapter is None:
            return world_copy
            
        for category, items in world_copy.items():
            if isinstance(items, dict):
                # Check category-level provisional flag
                cat_prov_key = f"source_quality_chapter_{up_to_chapter}" # Check up to the most recent relevant chapter
                if items.get(cat_prov_key) == "provisional_from_unrevised_draft":
                     if "prompt_notes" not in items: items["prompt_notes"] = []
                     items["prompt_notes"].append(f"Category '{category}' data from ch {up_to_chapter} is provisional.")

                for item_name, item_data in items.items():
                    if isinstance(item_data, dict):
                        provisional_notes_for_item = []
                        for i in range(1, up_to_chapter + 1):
                            prov_key = f"source_quality_chapter_{i}"
                            if item_data.get(prov_key) == "provisional_from_unrevised_draft":
                                provisional_notes_for_item.append(f"Item '{item_name}' info from ch {i} is provisional.")
                        if provisional_notes_for_item:
                            if "prompt_notes" not in item_data: item_data["prompt_notes"] = []
                            item_data["prompt_notes"].extend(provisional_notes_for_item)
        return world_copy


    def _save_debug_output(self, chapter_number: int, stage: str, content: Any):
        if content is None: return
        content_str = str(content) if not isinstance(content, str) else content
        try:
            debug_dir = os.path.join(config.OUTPUT_DIR, "debug")
            os.makedirs(debug_dir, exist_ok=True)
            safe_stage = "".join(c if c.isalnum() or c in ['_', '-'] else "_" for c in stage)
            file_path = os.path.join(debug_dir, f"chapter_{chapter_number}_{safe_stage}.txt")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content_str)
            logger.debug(f"Saved debug output for ch {chapter_number} stage '{stage}' to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save debug output for ch {chapter_number} stage '{stage}': {e}", exc_info=True)
