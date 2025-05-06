# novel_logic.py
"""
Contains the core logic for the novel generation agent, including state management,
chapter writing, analysis, revision, and knowledge updates.
Integrates a knowledge graph (KG) for improved consistency and context.
"""

import os
import json
import re
import numpy as np
import logging
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
    consistency_issues: Optional[str] # Now potentially includes KG feedback
    plot_deviation_reason: Optional[str] # Now potentially includes KG feedback

class NovelWriterAgent:
    """
    Manages the state and orchestrates the process of generating a novel
    chapter by chapter, interacting with LLMs, a database, and a knowledge graph.
    """

    def __init__(self):
        """Initializes the agent, loads state, and sets up components."""
        logger.info("Initializing NovelWriterAgent...")
        # Instantiate the database manager (handles chapters, embeddings, KG)
        self.db_manager = DatabaseManager(config.DATABASE_FILE)
        # Initialize state variables
        self.plot_outline: Dict[str, Any] = {}
        self.character_profiles: Dict[str, Any] = {} # Still used for descriptive/trait info
        self.world_building: Dict[str, Any] = {}   # Still used for descriptive/lore info
        self.chapter_count: int = 0
        # Load existing state (including chapter count)
        self._load_existing_state()
        logger.info(f"NovelWriterAgent initialized. Current chapter count: {self.chapter_count}")

    # --- State Management (Unchanged) ---
    def _load_existing_state(self):
        """Loads saved state from JSON files and chapter count from the database."""
        logger.info("Attempting to load existing agent state...")
        self.chapter_count = self.db_manager.load_chapter_count()
        logger.info(f"Loaded chapter count from database: {self.chapter_count}")

        for file_path, attr_name in [
            (config.PLOT_OUTLINE_FILE, "plot_outline"),
            (config.CHARACTER_PROFILES_FILE, "character_profiles"),
            (config.WORLD_BUILDER_FILE, "world_building")
        ]:
            data: Dict[str, Any] = {}
            logger.debug(f"Attempting to load JSON state from: {file_path}")
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        loaded_data = json.load(f)
                    if isinstance(loaded_data, dict):
                        data = loaded_data
                        logger.info(f"Successfully loaded {attr_name.replace('_', ' ')} from {file_path}")
                    else:
                        logger.warning(f"File {file_path} did not contain a valid JSON dictionary. Ignoring file content for {attr_name}.")
                except (json.JSONDecodeError, IOError) as e:
                    logger.error(f"Failed to load or decode {file_path}: {e}. Starting with empty data for {attr_name}.", exc_info=True)
                except Exception as e:
                    logger.error(f"An unexpected error occurred loading {file_path}: {e}", exc_info=True)
            else:
                logger.info(f"No {attr_name.replace('_', ' ')} file found ('{file_path}'). State will be empty or generated if needed.")
            setattr(self, attr_name, data)
        logger.info("Finished loading existing state.")


    def _save_json_state(self):
        """Saves the current state (plot, characters, world) to JSON files."""
        logger.debug("Saving agent JSON state (plot, characters, world)...")
        state_saved_count = 0
        try:
            for file_path, data in [
                (config.PLOT_OUTLINE_FILE, self.plot_outline),
                (config.CHARACTER_PROFILES_FILE, self.character_profiles),
                (config.WORLD_BUILDER_FILE, self.world_building)
            ]:
                if data and isinstance(data, dict):
                    logger.debug(f"Attempting to save data to {file_path}")
                    try:
                        os.makedirs(os.path.dirname(file_path), exist_ok=True)
                        with open(file_path, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                        logger.debug(f"Successfully saved data to {file_path}")
                        state_saved_count += 1
                    except (IOError, TypeError) as e:
                        logger.error(f"Failed to save JSON state to {file_path}: {e}", exc_info=True)
                    except Exception as e:
                         logger.error(f"An unexpected error occurred saving JSON to {file_path}: {e}", exc_info=True)
                else:
                    logger.debug(f"Skipping save for {file_path}, data is empty or not a dictionary.")
            if state_saved_count > 0:
                logger.info(f"JSON state saved successfully for {state_saved_count} file(s).")
            else:
                logger.info("No JSON state files were updated or saved in this cycle.")
        except Exception as e:
            logger.error(f"An unexpected error occurred during JSON state saving process: {e}", exc_info=True)

    # --- High-Level Generation Functions (Unchanged) ---
    def generate_plot_outline(self, genre: str, theme: str, protagonist: str) -> Dict[str, Any]:
        """Generates a structured plot outline using the LLM."""
        # (Existing code - no changes needed)
        prompt = f"""/no_think
        
        You are a creative assistant specializing in narrative structure. Generate a detailed plot outline for a '{genre}' novel with the central theme of '{theme}'.
The protagonist is described as: {protagonist}.

The outline must include the following keys with appropriate string or list-of-string values:
1.  `title`: A compelling title for the novel.
2.  `plot_points`: A list of exactly 5 strings, each describing a major plot point (e.g., Inciting Incident, Rising Action points, Climax, Falling Action/Resolution).
3.  `character_arc`: A string describing the protagonist's primary development arc throughout the story.
4.  `setting`: A string briefly describing the novel's primary setting.
5.  `conflict`: A string summarizing the main internal and/or external conflict driving the story.

Output ONLY the JSON object adhering strictly to this structure. Do not include any introductory text, explanations, markdown formatting, or meta-commentary like <think> blocks. Ensure the `plot_points` value is a JSON list of 5 strings.
Example Structure:
{{
  "title": "string",
  "genre": "{genre}",
  "theme": "{theme}",
  "protagonist_description": "{protagonist}",
  "plot_points": ["Plot Point 1: ...", "Plot Point 2: ...", "Plot Point 3: ...", "Plot Point 4: ...", "Plot Point 5: ..."],
  "character_arc": "string",
  "setting": "string",
  "conflict": "string"
}}
"""
        logger.info("Generating plot outline via LLM...")
        raw_outline_str = llm_interface.call_llm(prompt, temperature=0.6)
        parsed_outline = llm_interface.parse_llm_json_response(raw_outline_str, "plot outline")

        required_keys = ["title", "plot_points", "character_arc", "setting", "conflict"]
        is_valid = False
        if parsed_outline and isinstance(parsed_outline, dict):
            plot_points = parsed_outline.get("plot_points")
            if (all(key in parsed_outline for key in required_keys) and
                isinstance(plot_points, list) and
                len(plot_points) == 5 and
                all(isinstance(p, str) and p.strip() for p in plot_points)):
                is_valid = True
                self.plot_outline = parsed_outline
                logger.info(f"Successfully generated and validated plot outline for title: '{self.plot_outline.get('title', 'N/A')}'")
            else:
                 logger.warning(f"Generated plot outline failed validation. Missing keys, incorrect plot_points format/length, or empty strings found. Parsed data: {parsed_outline}")

        if not is_valid:
            logger.error("Failed to generate a valid plot outline. Applying default structure.")
            self.plot_outline = {
                "title": "Untitled Novel", "genre": genre, "theme": theme, "protagonist_description": protagonist,
                "plot_points": [
                    "Default Point 1: Setup - Introduce protagonist and world.",
                    "Default Point 2: Inciting Incident - Event that disrupts the status quo.",
                    "Default Point 3: Rising Action - Complications and character development.",
                    "Default Point 4: Climax - Peak of the conflict.",
                    "Default Point 5: Resolution - Aftermath and conclusion."
                ],
                "character_arc": "Default character arc: Protagonist faces challenges and undergoes some change.",
                "setting": "Default setting: A place relevant to the story.",
                "conflict": "Default conflict: The primary struggle the protagonist faces."
            }

        self.plot_outline['genre'] = genre
        self.plot_outline['theme'] = theme
        self.plot_outline['protagonist_description'] = protagonist
        self._save_json_state()
        return self.plot_outline

    def generate_world_building(self):
        """Generates initial world-building elements based on the existing plot outline."""
        # (Existing code - no changes needed)
        if self.world_building and not ("Default Location" in self.world_building.get("locations", {}) and len(self.world_building.keys()) <= 2):
            logger.info("Skipping initial world-building generation: Data seems already populated.")
            return

        if not self.plot_outline or not self.plot_outline.get("setting"):
            logger.error("Cannot generate world-building: Plot outline (especially setting description) is missing.")
            return

        prompt = f"""/no_think
        
        You are a world-building assistant tasked with generating foundational elements based on a novel concept. Your output MUST be a single, valid JSON object.

Novel Concept:
Title: {self.plot_outline.get('title', 'Untitled')}
Genre: {self.plot_outline.get('genre', 'undefined')}
Theme: {self.plot_outline.get('theme', 'undefined')}
Setting Description (Expand on this): {self.plot_outline.get('setting', 'default setting')}
Main Conflict: {self.plot_outline.get('conflict', 'default conflict')}
Protagonist: {self.plot_outline.get('protagonist_description', 'N/A')}

Instructions:
1. Create detailed world-building elements covering locations, society, unique systems (tech/magic), lore, and history relevant to the concept.
2. Be creative and provide enough detail to make the world feel tangible. Expand significantly on the provided setting description.
3. **CRITICAL: Output ONLY the JSON object.** Do not include any text before or after the JSON.
4. **JSON Syntax Rules:** Ensure valid JSON (double quotes for keys/strings, no trailing commas, balanced braces/brackets).
5. Use clear top-level categories: "locations", "society", "systems", "lore", "history". Populate these with nested dictionaries or lists.

Example Structure (Adhere to this format and syntax rules):
```json
{{
  "locations": {{
    "Location Name 1": {{ "description": "string", "atmosphere": "string", "relevance": "string" }},
    "Location Name 2": {{ "description": "string", "details": ["detail1", "detail2"] }}
  }},
  "society": {{
    "Governance": "string",
    "Key Factions": {{
      "Faction 1": {{ "description": "string", "goal": "string" }}
    }},
    "Customs": ["string1", "string2"]
  }},
  "systems": {{
    "Key System Name": {{ "name": "string", "rules": "string", "limitations": "string" }}
  }},
  "lore": {{
    "Key Myth": "string",
    "Creation Story Snippet": "string"
  }},
  "history": {{
    "Foundational Event": "string",
    "Recent Conflict": "string"
  }}
}}
```
"""
        logger.info("Generating initial world-building data via LLM...")
        raw_world_data_str = llm_interface.call_llm(prompt, temperature=0.6)
        parsed_world_data = llm_interface.parse_llm_json_response(raw_world_data_str, "initial world-building")

        is_valid = False
        if parsed_world_data and isinstance(parsed_world_data, dict):
            if any(k in parsed_world_data and parsed_world_data[k] for k in ["locations", "society", "systems", "lore", "history"]):
                self.world_building = parsed_world_data
                logger.info("Successfully generated and validated initial world-building data.")
                is_valid = True
            else:
                 logger.warning(f"Generated world-building JSON seems empty or lacks expected structure, even after potential retry. Parsed data: {parsed_world_data}")

        if not is_valid:
            logger.error("Failed to generate valid world-building data. Applying default structure.")
            self.world_building = {
                "locations": {"Default Location": {"description": "A starting point based on the outline setting.", "relevance": "Initial setting"}},
                "society": {"General": "Basic societal norms apply unless specified otherwise."},
                "systems": {"Note": "No specific unique systems defined initially."},
                "lore": {"Background": "Minimal initial lore provided."},
                "history": {"Origin": "History related to the setting and conflict."}
            }
        self._save_json_state()

    # --- Chapter Planning Step (Now includes KG query) ---
    def _plan_chapter(self, chapter_number: int) -> Optional[str]:
        """
        Generates a brief plan for the chapter, incorporating KG facts.
        """
        if not config.ENABLE_AGENTIC_PLANNING:
            logger.info("Agentic planning is disabled in config. Skipping planning step.")
            return "Agentic planning disabled by configuration."

        logger.info(f"Planning Chapter {chapter_number}...")

        plot_point_focus, plot_point_index = self._get_plot_point_info(chapter_number)
        if plot_point_focus is None:
             logger.error(f"Cannot plan chapter {chapter_number}: Could not determine plot point.")
             return None

        # --- Gather Context (including KG facts) ---
        context = ""
        # Get previous chapter summary/snippet
        if chapter_number > 1:
            prev_chap_data = self.db_manager.get_chapter_data_from_db(chapter_number - 1)
            if prev_chap_data:
                prev_summary = prev_chap_data.get('summary')
                if prev_summary:
                    context += f"Summary of Previous Chapter ({chapter_number - 1}):\n{prev_summary[:1000]}...\n"
                else:
                     prev_text = prev_chap_data.get('text', '')
                     if prev_text: context += f"End Snippet of Previous Chapter ({chapter_number - 1}):\n...{prev_text[-1000:]}\n"

        # --- Query KG for relevant current state ---
        kg_facts = []
        # Example KG queries: Protagonist location and status
        protagonist_name = self.plot_outline.get("protagonist_name", "Sága") # Assume protagonist name or default
        current_loc = self.db_manager.get_most_recent_value(protagonist_name, "located_in", chapter_number - 1)
        current_status = self.db_manager.get_most_recent_value(protagonist_name, "status_is", chapter_number - 1)

        if current_loc: kg_facts.append(f"- {protagonist_name} is currently located in: {current_loc}.")
        if current_status: kg_facts.append(f"- {protagonist_name}'s current status: {current_status}.")
        # Add more queries as needed (e.g., relationships, key item locations)

        kg_context_section = "**Relevant Facts from Knowledge Graph:**\n" + "\n".join(kg_facts) + "\n" if kg_facts else ""
        # --- End KG Query ---

        prompt = f"""/no_think
        
        You are a master plotter outlining the key narrative beats for Chapter {chapter_number} of a novel.

**Overall Novel Concept:**
* Title: {self.plot_outline.get('title', 'Untitled')}
* Genre: {self.plot_outline.get('genre', 'N/A')}
* Theme: {self.plot_outline.get('theme', 'N/A')}
* Protagonist Arc Goal: {self.plot_outline.get('character_arc', 'N/A')}

**Mandatory Focus for THIS Chapter (Plot Point {plot_point_index + 1}):**
{plot_point_focus}

**Recent Context (End of Previous Chapter/Summary):**
{context if context else "This is the first chapter, focus on establishing the initial state and the plot point."}

{kg_context_section}
**Current Character States (Key Characters Only):**
{self._get_relevant_character_state_snippet()}

**Current World State (Relevant Locations/Elements):**
{self._get_relevant_world_state_snippet()}

**Task:**
Based *only* on the information above (especially Recent Context and KG Facts), outline 3-5 essential scenes or narrative beats required for Chapter {chapter_number}. These beats MUST:
1.  Directly address and progress the **Mandatory Focus (Plot Point {plot_point_index + 1})**.
2.  Logically follow from the **Recent Context** and **KG Facts**.
3.  Involve relevant characters/world elements consistent with their current states.
4.  Contribute towards the overall **Protagonist Arc Goal**.

**Output Format:**
Output ONLY a numbered list of the key scenes/beats (1 sentence description per beat). Do not include any introduction, commentary, or explanation. Start directly with "1.".

Example:
1. Scene where Character A (currently in Location X) investigates the strange signal mentioned previously.
2. Character A encounters Character B, leading to conflict based on their differing goals and current status.
3. The chapter ends with Character A making a crucial discovery related to the plot point.
"""

        logger.info(f"Calling LLM to generate plan for chapter {chapter_number}...")
        plan_raw = llm_interface.call_llm(
            prompt,
            temperature=0.6,
            max_tokens=config.MAX_PLANNING_TOKENS
        )
        plan_cleaned = llm_interface.clean_model_response(plan_raw).strip()

        if plan_cleaned and re.match(r"^\s*1\.", plan_cleaned):
            logger.info(f"Successfully generated plan for chapter {chapter_number}:\n{plan_cleaned}")
            return plan_cleaned
        else:
            logger.error(f"Failed to generate a valid plan for chapter {chapter_number}. Response: '{plan_raw[:200]}...'")
            return None


    # --- Refactored Core Chapter Writing Logic ---

    def write_chapter(self, chapter_number: int) -> Optional[str]:
        """
        Orchestrates the full process for generating a single chapter:
        Plan -> Generate -> Evaluate -> Revise (if needed) -> Finalize -> Update KG.
        """
        logger.info(f"=== Starting Chapter {chapter_number} Generation ===")

        # --- Pre-checks ---
        if not self.plot_outline or not self.plot_outline.get("plot_points"):
            logger.error(f"Cannot write chapter {chapter_number}: Plot outline or plot points are missing.")
            return None
        if chapter_number <= 0:
             logger.error(f"Cannot write chapter {chapter_number}: Chapter number must be positive.")
             return None

        # --- 1. Plan Chapter (incorporates KG) ---
        chapter_plan = self._plan_chapter(chapter_number)
        if config.ENABLE_AGENTIC_PLANNING and chapter_plan is None:
            logger.error(f"Chapter {chapter_number} generation halted due to planning failure.")
            return None

        # --- 2. Prepare Context & Generate Initial Draft ---
        logger.debug(f"Preparing context for chapter {chapter_number} draft...")
        context = self._get_context(chapter_number) # Context now potentially uses KG
        plot_point_focus, _ = self._get_plot_point_info(chapter_number)

        logger.info(f"Generating initial draft for chapter {chapter_number}.")
        initial_draft_text, initial_raw_text = self._generate_draft(
            chapter_number, plot_point_focus, context, chapter_plan
        )

        if not initial_draft_text:
            logger.error(f"Failed to generate initial draft for chapter {chapter_number}.")
            self._save_debug_output(chapter_number, "initial_raw_fail_after_clean", initial_raw_text or "")
            return None

        # --- 3. Evaluate Draft (incorporates KG checks) ---
        logger.info(f"Evaluating initial draft for chapter {chapter_number}...")
        evaluation = self._evaluate_draft(initial_draft_text, chapter_number)
        current_text = initial_draft_text
        final_raw_output_log = f"--- INITIAL DRAFT RAW ---\n{initial_raw_text}\n\n"

        # --- 4. Revise Draft if Necessary (incorporates KG feedback) ---
        if evaluation["needs_revision"]:
            revision_reason_str = "\n- ".join(evaluation["reasons"])
            logger.warning(f"Chapter {chapter_number} flagged for revision. Reason(s):\n- {revision_reason_str}")

            revised_text_tuple = self._revise_chapter(
                current_text, chapter_number, revision_reason_str, context, chapter_plan
            )

            if revised_text_tuple:
                revised_text, raw_revision_output = revised_text_tuple
                logger.info(f"Revision successful for chapter {chapter_number}. Evaluating revised draft...")

                # --- Re-Evaluate the Revised Draft ---
                revised_evaluation = self._evaluate_draft(revised_text, chapter_number)
                if revised_evaluation["needs_revision"]:
                     logger.error(f"Revised draft for chapter {chapter_number} STILL failed evaluation. Reasons:\n- " + "\n- ".join(revised_evaluation["reasons"]) + "\nProceeding with the revised draft despite issues.")
                else:
                     logger.info(f"Revised draft for chapter {chapter_number} passed evaluation.")

                current_text = revised_text
                final_raw_output_log += f"--- REVISION ATTEMPT (Reason: {revision_reason_str}) ---\n{raw_revision_output}\n\n"
            else:
                logger.error(f"Revision failed for chapter {chapter_number}. Proceeding with the original draft despite evaluation issues.")
                final_raw_output_log += f"--- REVISION FAILED (Reason: {revision_reason_str}) ---\n\n"
        else:
            logger.info(f"Initial draft for chapter {chapter_number} passed evaluation.")

        # --- 5. Finalize Chapter (Summarize, Embed, Save Text/Embedding) ---
        logger.info(f"Finalizing chapter {chapter_number} text and embedding...")
        finalization_success = self._finalize_chapter_core(chapter_number, current_text, final_raw_output_log)

        if not finalization_success:
             logger.error(f"=== Finished Chapter {chapter_number} Generation With Errors During Core Finalization ===")
             return None # Stop if core saving failed

        # --- 6. Update Knowledge (JSON Profiles/World + NEW KG Extraction) ---
        logger.info(f"Updating knowledge bases for chapter {chapter_number}...")
        self._update_knowledge_bases(chapter_number, current_text)

        # --- 7. Save Final State ---
        self.chapter_count = max(self.chapter_count, chapter_number)
        self._save_json_state() # Save potentially updated JSON files

        logger.info(f"=== Finished Chapter {chapter_number} Generation Successfully ===")
        return current_text


    # --- Internal Helper Methods for write_chapter ---

    def _get_plot_point_info(self, chapter_number: int) -> Tuple[Optional[str], int]:
        """Gets the plot point text and index for the given chapter number."""
        # (Existing code - unchanged)
        plot_points = self.plot_outline.get("plot_points", [])
        plot_point_index = min(chapter_number - 1, len(plot_points) - 1) if plot_points else -1
        if 0 <= plot_point_index < len(plot_points):
            return plot_points[plot_point_index], plot_point_index
        else:
            logger.warning(f"Could not find plot point for chapter {chapter_number} in outline.")
            return None, -1

    def _generate_draft(self, chapter_number: int, plot_point_focus: Optional[str], context: str, plan: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        """Generates the initial chapter draft using the LLM."""
        # (Existing code - unchanged, uses plan if provided)
        if not plot_point_focus:
             plot_point_focus = "Continue the narrative logically, developing from previous events."

        plan_section = ""
        if plan and config.ENABLE_AGENTIC_PLANNING:
            if plan != "Agentic planning disabled by configuration.":
                 plan_section = f"**Chapter Plan / Key Beats:**\n{plan}\n"
            else:
                 plan_section = f"**Note:** {plan}\n"

        prompt = f"""/no_think
        
        You are an expert novelist continuing a story. Write Chapter {chapter_number} of the novel titled "{self.plot_outline.get('title', 'Untitled Novel')}".

**Story Bible:**
* Genre: {self.plot_outline.get('genre', 'N/A')}
* Theme: {self.plot_outline.get('theme', 'N/A')}
* Protagonist Arc: {self.plot_outline.get('character_arc', 'N/A')}
* Setting: {self.plot_outline.get('setting', 'N/A')}
* Main Conflict: {self.plot_outline.get('conflict', 'N/A')}

**Focus for THIS Chapter:**
{plot_point_focus}

{plan_section}
**Current World Building Notes:**
```json
{json.dumps(self.world_building, indent=2, ensure_ascii=False)}
```

**Current Character Profiles:**
```json
{json.dumps(self.character_profiles, indent=2, ensure_ascii=False)}
```

**Context from Previous Relevant Chapters:**
--- BEGIN CONTEXT ---
{context if context else "No specific context from previous chapters available (either Chapter 1 or retrieval failed)."}
--- END CONTEXT ---

**Instructions for Writing Chapter {chapter_number}:**
1.  Write a compelling chapter (approx 1000-2000 words).
2.  Crucially, follow the **Chapter Plan / Key Beats** (if provided) and address the **Focus for THIS Chapter**.
3.  Maintain character voice/consistency and world details. Avoid info-dumping.
4.  Ensure smooth narrative flow and logical progression.
5.  Employ vivid descriptions and engaging prose for the '{self.plot_outline.get('genre', 'story')}' genre.
6.  **Output ONLY the chapter text.** No headers, titles, notes, or JSON. Start directly with the text. Use standard paragraph breaks.
--- BEGIN CHAPTER {chapter_number} ---
"""
        logger.debug(f"Generating initial draft for chapter {chapter_number}. Prompt length approx {len(prompt)} chars.")
        raw_text = llm_interface.call_llm(prompt, temperature=0.6)

        if not raw_text:
            return None, None

        cleaned_text = llm_interface.clean_model_response(raw_text)
        if not cleaned_text or len(cleaned_text) < config.MIN_ACCEPTABLE_DRAFT_LENGTH:
             logger.error(f"Chapter {chapter_number} resulted in empty or too short text ({len(cleaned_text or '')} chars < Min: {config.MIN_ACCEPTABLE_DRAFT_LENGTH}) after cleaning raw output.")
             return None, raw_text

        logger.info(f"Generated initial draft for chapter {chapter_number} (Cleaned Length: {len(cleaned_text)} chars).")
        logger.debug(f"Chapter {chapter_number} Initial Draft Snippet: {cleaned_text[:200].strip()}...")
        return cleaned_text, raw_text


    def _evaluate_draft(self, draft_text: str, chapter_number: int) -> EvaluationResult:
        """Performs multiple checks (including KG) on the draft."""
        logger.info(f"Performing evaluation checks for chapter {chapter_number} draft...")
        reasons: List[str] = []
        needs_revision = False
        coherence_score = None
        consistency_issues = None
        plot_deviation_reason = None

        # --- Coherence Check ---
        if chapter_number > 1:
            logger.debug(f"Checking coherence for chapter {chapter_number}...")
            current_embedding = llm_interface.get_embedding(draft_text)
            prev_embedding = self.db_manager.get_embedding_from_db(chapter_number - 1)
            if current_embedding is not None and prev_embedding is not None:
                coherence_score = utils.numpy_cosine_similarity(current_embedding, prev_embedding)
                logger.info(f"Coherence score with previous chapter ({chapter_number-1}): {coherence_score:.4f}")
                if coherence_score < config.REVISION_COHERENCE_THRESHOLD:
                    needs_revision = True
                    reasons.append(f"Low coherence with previous chapter (Score: {coherence_score:.4f} < Threshold: {config.REVISION_COHERENCE_THRESHOLD}).")
            else:
                logger.warning(f"Could not perform coherence check for chapter {chapter_number} (missing current or previous embedding).")
        else:
             logger.info("Skipping coherence check for Chapter 1.")

        # --- Consistency Check (Now includes KG) ---
        if config.REVISION_CONSISTENCY_TRIGGER:
            logger.debug(f"Checking consistency for chapter {chapter_number}...")
            # This function now internally queries KG and uses LLM for analysis
            consistency_issues = self._check_consistency(draft_text, chapter_number)
            if consistency_issues:
                needs_revision = True
                reasons.append(f"Consistency issues identified:\n{consistency_issues}")
                logger.warning(f"Consistency issues found for Chapter {chapter_number}.")

        # --- Plot Arc Validation (Now includes KG) ---
        if config.PLOT_ARC_VALIDATION_TRIGGER:
            logger.debug(f"Validating plot arc alignment for chapter {chapter_number}...")
            # This function now internally queries KG for plot conditions
            plot_deviation_reason = self._validate_plot_arc(draft_text, chapter_number)
            if plot_deviation_reason:
                needs_revision = True
                reasons.append(f"Plot Arc Deviation: {plot_deviation_reason}")
                logger.warning(f"Plot arc deviation found for Chapter {chapter_number}.")

        # --- Basic Length Check ---
        if len(draft_text) < config.MIN_ACCEPTABLE_DRAFT_LENGTH:
             needs_revision = True
             reasons.append(f"Generated draft is too short ({len(draft_text)} chars < Minimum: {config.MIN_ACCEPTABLE_DRAFT_LENGTH}).")
             logger.warning(f"Draft for Chapter {chapter_number} is too short.")


        logger.info(f"Evaluation complete for Chapter {chapter_number}. Needs revision: {needs_revision}.")
        return {
            "needs_revision": needs_revision,
            "reasons": reasons,
            "coherence_score": coherence_score,
            "consistency_issues": consistency_issues,
            "plot_deviation_reason": plot_deviation_reason
        }


    def _revise_chapter(self, original_text: str, chapter_number: int, reason: str, context: str, plan: Optional[str]) -> Optional[Tuple[str, str]]:
        """Attempts to revise a chapter based on feedback (including KG facts)."""
        # (Using the previously refined prompt from immersive id="saga_novel_logic_update_v1")
        # --- Input Validation ---
        if not original_text or not reason:
            logger.error(f"Revision requested for chapter {chapter_number}, but original text or reason is missing. Skipping.")
            return None
        clean_reason = llm_interface.clean_model_response(reason).strip()
        if not clean_reason:
            logger.error(f"Revision reason for chapter {chapter_number} was empty after cleaning. Skipping.")
            return None

        logger.warning(f"Attempting revision for chapter {chapter_number}.")
        logger.debug(f"Revision Reason Provided:\n{clean_reason}") # Log the specific reason

        # --- Prepare Prompt for Revision ---
        context_limit = config.MAX_CONTEXT_LENGTH // 3
        original_text_limit = config.MAX_CONTEXT_LENGTH // 3
        plan_limit = config.MAX_CONTEXT_LENGTH // 3
        context_snippet = context[:context_limit]
        original_snippet = original_text[:original_text_limit]

        plan_focus_section = ""
        if plan and config.ENABLE_AGENTIC_PLANNING and plan != "Agentic planning disabled by configuration.":
            plan_focus_section = f"**Original Chapter Plan / Key Beats (Target for Revision):**\n{plan[:plan_limit]}\n"
        else:
            plot_point_focus, _ = self._get_plot_point_info(chapter_number)
            plan_focus_section = f"**Original Chapter Focus (Plot Point - Target for Revision):**\n{plot_point_focus}\n"

        # --- Using the Refined Prompt ---
        prompt = f"""/no_think
        
        You are a skilled revising author tasked with significantly rewriting Chapter {chapter_number} to correct specific issues identified in the feedback. Your primary goal is to address the feedback points directly, requiring substantial changes to the original narrative if necessary.

**Critique / Reason(s) for Revision (MUST be addressed):**
--- FEEDBACK START ---
{clean_reason}
--- FEEDBACK END ---

{plan_focus_section}
**Context from Previous Relevant Chapters (Maintain continuity with this):**
--- BEGIN CONTEXT ---
{context_snippet if context_snippet else "No previous context provided."}
--- END CONTEXT ---

**Original Draft Snippet of Chapter {chapter_number} (Reference ONLY - DO NOT simply copy/tweak. Significant changes are likely needed based on feedback):**
--- BEGIN ORIGINAL DRAFT SNIPPET ---
{original_snippet}
--- END ORIGINAL DRAFT SNIPPET ---

**Instructions for Rewriting Chapter {chapter_number}:**
1.  **PRIORITY:** Thoroughly analyze the **Critique / Reason(s) for Revision**. Your rewrite MUST directly and substantially fix these issues. If plot deviation is noted, the rewrite MUST center on the **Original Chapter Plan / Focus**. If specific factual inconsistencies (potentially from Knowledge Graph checks) are mentioned, ensure the rewrite corrects them. Do not be afraid to discard parts of the original draft that conflict with the feedback.
2.  **Rewrite the ENTIRE chapter.** Do not just edit the original. Create a new version from the ground up that incorporates the necessary corrections and focuses on the target plan/plot point.
3.  Ensure the revised chapter logically follows the **Context** and maintains established character/world consistency *unless* the feedback requires correcting an inconsistency.
4.  Preserve the overall narrative tone and style, but prioritize fixing the core issues over mimicking the original text exactly. Significant structural or content changes may be necessary.
5.  Aim for appropriate length (at least {config.MIN_ACCEPTABLE_DRAFT_LENGTH} characters) and detail.
6.  **Output ONLY the fully rewritten chapter text.** No headers, titles, explanations, apologies, or meta-commentary. Start directly with the first sentence.
--- BEGIN REVISED CHAPTER {chapter_number} ---
"""
        # --- Call LLM for Revision ---
        logger.info(f"Calling LLM to revise chapter {chapter_number} with enhanced prompt...")
        revised_raw = llm_interface.call_llm(prompt, temperature=0.6)

        if not revised_raw:
            logger.error(f"Revision call failed for chapter {chapter_number} (LLM returned empty response).")
            return None

        # --- Process and Validate Revision ---
        revised_cleaned = llm_interface.clean_model_response(revised_raw)
        if not revised_cleaned or len(revised_cleaned) < config.MIN_ACCEPTABLE_DRAFT_LENGTH:
            logger.error(f"Revision for chapter {chapter_number} failed: generated text was empty or too short ({len(revised_cleaned or '')} chars < Min: {config.MIN_ACCEPTABLE_DRAFT_LENGTH}) after cleaning.")
            self._save_debug_output(chapter_number, "revision_raw_fail_too_short", revised_raw)
            return None

        # --- Similarity Check ---
        logger.debug(f"Performing similarity check between original and revised chapter {chapter_number}...")
        original_embedding = llm_interface.get_embedding(original_text)
        revised_embedding = llm_interface.get_embedding(revised_cleaned)

        if original_embedding is not None and revised_embedding is not None:
            similarity_score = utils.numpy_cosine_similarity(original_embedding, revised_embedding)
            logger.info(f"Revision similarity score with original: {similarity_score:.4f}")
            if similarity_score >= config.REVISION_SIMILARITY_ACCEPTANCE:
                logger.warning(f"Revision for chapter {chapter_number} rejected: Too similar to original (Score: {similarity_score:.4f} >= {config.REVISION_SIMILARITY_ACCEPTANCE}).")
                self._save_debug_output(chapter_number, "revision_raw_rejected_too_similar", revised_raw)
                return None
            else:
                logger.info(f"Revision for chapter {chapter_number} accepted (Similarity: {similarity_score:.4f}).")
                return revised_cleaned, revised_raw
        else:
            logger.warning(f"Could not get valid embeddings for revision similarity check of chapter {chapter_number}. Accepting revision.")
            return revised_cleaned, revised_raw


    def _finalize_chapter_core(self, chapter_number: int, final_text: str, raw_log: str) -> bool:
        """Handles core final steps: summarization, embedding, saving text/embedding."""
        logger.info(f"Starting core finalization process for chapter {chapter_number}...")

        if not final_text:
            logger.error(f"Cannot finalize chapter {chapter_number}: Final text is missing.")
            return False

        # Generate Summary
        summary = self._summarize_chapter(final_text, chapter_number)

        # Generate Embedding
        final_embedding = llm_interface.get_embedding(final_text)
        if final_embedding is None:
            logger.error(f"CRITICAL: Failed to generate embedding for final version of chapter {chapter_number}. Saving without embedding.")
            # Proceed without embedding, but log critical failure

        # Save Final Chapter Data to DB (Text, Raw Log, Summary, Embedding)
        logger.info(f"Saving final text/embedding data for chapter {chapter_number} to database...")
        try:
            self.db_manager.save_chapter_data(
                chapter_number,
                final_text,
                raw_log,
                summary,
                final_embedding # May be None
            )
        except Exception as e:
             logger.error(f"Database save failed for chapter {chapter_number}: {e}", exc_info=True)
             return False # Treat DB save failure as critical for finalization

        # Save chapter text to flat files
        try:
            final_txt_path = os.path.join(config.OUTPUT_DIR, f"chapter_{chapter_number}.txt")
            with open(final_txt_path, 'w', encoding='utf-8') as f:
                f.write(final_text)
            logger.info(f"Saved final chapter text to {final_txt_path}")

            raw_log_path = os.path.join(config.OUTPUT_DIR, f"chapter_{chapter_number}_raw_log.txt")
            with open(raw_log_path, 'w', encoding='utf-8') as f:
                f.write(raw_log)
            logger.info(f"Saved raw generation log to {raw_log_path}")
        except IOError as e:
            logger.error(f"Failed to write chapter text/log files for chapter {chapter_number}: {e}", exc_info=True)
            # Log error but don't fail finalization for this

        logger.info(f"Core finalization complete for chapter {chapter_number}.")
        return True

    def _update_knowledge_bases(self, chapter_number: int, final_text: str):
        """Updates JSON profiles/world AND extracts/updates the Knowledge Graph."""
        if not final_text:
            logger.warning(f"Skipping knowledge base update for chapter {chapter_number}: Final text is missing.")
            return

        logger.info(f"Updating knowledge bases (JSON & KG) for chapter {chapter_number}...")
        try:
            # 1. Update JSON (Descriptive info, traits - existing logic)
            self._update_character_profiles(final_text, chapter_number)
            self._update_world_building(final_text, chapter_number)
            logger.info(f"JSON knowledge bases updated for chapter {chapter_number}.")

            # 2. Extract and Update Knowledge Graph (Factual triples)
            self._extract_and_update_kg(final_text, chapter_number)
            logger.info(f"Knowledge Graph updated for chapter {chapter_number}.")

        except Exception as e:
            logger.error(f"Error occurred during knowledge base update for chapter {chapter_number}: {e}", exc_info=True)


    # --- Context, Summarization (Unchanged) ---
    def _get_context(self, current_chapter_number: int) -> str:
        """Retrieves relevant context using semantic similarity."""
        # (Existing code - unchanged)
        if current_chapter_number <= 1: return ""
        logger.debug(f"Retrieving context for Chapter {current_chapter_number}...")
        context_query_text = None
        plot_point_focus, plot_point_index = self._get_plot_point_info(current_chapter_number)
        if plot_point_focus:
             context_query_text = plot_point_focus
             logger.info(f"Context query for chapter {current_chapter_number} based on Plot Point {plot_point_index + 1}: '{context_query_text[:100]}...'")
        else:
             logger.warning(f"Could not determine plot point for chapter {current_chapter_number}. Using generic context query.")
             context_query_text = f"Narrative context for chapter {current_chapter_number} following chapter {current_chapter_number - 1}."
        query_embedding = llm_interface.get_embedding(context_query_text)
        if query_embedding is None:
            logger.warning("Failed to get embedding for context query. Falling back to previous chapter summary/text.")
            last_chap_data = self.db_manager.get_chapter_data_from_db(current_chapter_number - 1)
            if last_chap_data:
                fallback_context = last_chap_data.get('summary') or last_chap_data.get('text', '')
                if fallback_context:
                     logger.info(f"Using fallback context (previous chapter summary/text) - {len(fallback_context)} chars.")
                     return fallback_context[:config.MAX_CONTEXT_LENGTH]
            logger.warning("Fallback context failed: Could not retrieve previous chapter data.")
            return ""
        past_embeddings = self.db_manager.get_all_past_embeddings(current_chapter_number)
        if not past_embeddings: return ""
        similarities: List[Tuple[int, float]] = []
        for chapter_num, embedding in past_embeddings:
            if embedding is not None: similarities.append((chapter_num, utils.numpy_cosine_similarity(query_embedding, embedding)))
        if not similarities: return ""
        similarities.sort(key=lambda item: item[1], reverse=True)
        top_n_indices = [chap_num for chap_num, sim in similarities[:config.CONTEXT_CHAPTER_COUNT]]
        similarity_scores = [f'{s:.3f}' for _, s in similarities[:config.CONTEXT_CHAPTER_COUNT]]
        logger.info(f"Top {len(top_n_indices)} relevant chapters for context: {top_n_indices} (Scores: {similarity_scores})")
        previous_chapter_num = current_chapter_number - 1
        if previous_chapter_num not in top_n_indices and previous_chapter_num > 0:
             logger.debug(f"Adding previous chapter {previous_chapter_num} to context list.")
             top_n_indices.append(previous_chapter_num)
        context_parts = []
        total_chars = 0
        chapters_to_fetch = sorted(list(set(top_n_indices)))
        logger.debug(f"Fetching context data for chapters: {chapters_to_fetch}")
        for chap_num in chapters_to_fetch:
            if total_chars >= config.MAX_CONTEXT_LENGTH: break
            chap_data = self.db_manager.get_chapter_data_from_db(chap_num)
            if chap_data:
                content = chap_data.get('summary') or chap_data.get('text', '')
                content_type = "Summary" if chap_data.get('summary') else "Text Snippet"
                if content:
                    content = content.strip()
                    chars_available = config.MAX_CONTEXT_LENGTH - total_chars
                    formatting_chars = len(f"[From Chapter {chap_num} ({content_type})]:\n\n---\n") + 5
                    content_limit = max(0, chars_available - formatting_chars)
                    truncated_content = content[:content_limit]
                    if truncated_content:
                        context_parts.append(f"[From Chapter {chap_num} ({content_type})]:\n{truncated_content}\n---")
                        added_len = len(truncated_content) + formatting_chars
                        total_chars += added_len
                        logger.debug(f"Added context from chapter {chap_num} ({content_type}), {len(truncated_content)} chars. Total: {total_chars} chars.")
            else:
                logger.warning(f"Could not retrieve chapter data for {chap_num} while building context.")
        final_context = "\n".join(context_parts).strip()
        logger.info(f"Constructed final semantic context: {len(final_context)} chars from chapters {chapters_to_fetch}.")
        return final_context

    def _summarize_chapter(self, chapter_text: Optional[str], chapter_number: int) -> Optional[str]:
        """Generates a brief summary (1-3 sentences) of the chapter text."""
        # (Existing code - unchanged)
        if not chapter_text or len(chapter_text) < 50: return None
        snippet_for_summary = chapter_text[:config.KNOWLEDGE_UPDATE_SNIPPET_SIZE]
        prompt = f"""/no_think
        
        You are an expert at concise summarization. Read the following chapter text (Chapter {chapter_number}) and provide a very brief summary (strictly 1-3 sentences) capturing only the most crucial plot advancements, character decisions, or revelations. Be extremely succinct.

Chapter Text Snippet:
--- BEGIN TEXT ---
{snippet_for_summary}
--- END TEXT ---

Output ONLY the summary text. Do not include any introduction, commentary, or labels like "Summary:".
"""
        logger.info(f"Generating summary for chapter {chapter_number}...")
        summary_raw = llm_interface.call_llm(prompt, temperature=0.6, max_tokens=150)
        cleaned_summary = llm_interface.clean_model_response(summary_raw).strip()
        if cleaned_summary:
            logger.info(f"Generated summary for chapter {chapter_number}: '{cleaned_summary[:100]}...'")
            return cleaned_summary
        else:
            logger.warning(f"Failed to generate a valid summary for chapter {chapter_number}.")
            return None

    # --- Consistency/Validation Helpers (Now integrate KG queries) ---

    def _check_consistency(self, chapter_text: Optional[str], chapter_number: int) -> Optional[str]:
        """Checks chapter text for consistency against JSON state AND KG facts."""
        if not chapter_text: return None
        text_snippet = chapter_text[:config.KNOWLEDGE_UPDATE_SNIPPET_SIZE]

        # --- Perform KG Queries for Specific Checks ---
        kg_consistency_checks = []
        # Example: Check protagonist location consistency
        protagonist_name = self.plot_outline.get("protagonist_name", "Sága")
        # Find location mentioned in the current chapter snippet (simplified regex example)
        current_mention_match = re.search(r"(?:in|at|near|entered|reached)\s+the\s+([\w\s]+)(?:Valley|Spire|Expanse|Nexus|Wastes)", chapter_text, re.IGNORECASE)
        if current_mention_match:
            mentioned_loc = current_mention_match.group(1).strip()
            # Query KG for location in *previous* chapter state
            previous_loc = self.db_manager.get_most_recent_value(protagonist_name, "located_in", chapter_number - 1)
            if previous_loc and previous_loc.lower() != mentioned_loc.lower():
                 # Basic check: If location mentioned differs from last known KG location
                 # More complex logic could check adjacency, travel time etc.
                 kg_consistency_checks.append(f"- KG Check: Chapter mentions {protagonist_name} near '{mentioned_loc}', but last known KG location was '{previous_loc}' (in chapter <= {chapter_number - 1}). Verify travel/transition.")
            elif not previous_loc and chapter_number > 1:
                 kg_consistency_checks.append(f"- KG Check: Chapter mentions {protagonist_name} near '{mentioned_loc}', but no previous location found in KG. Verify starting location.")

        # Add more KG queries here (e.g., check character status, item possession)
        # ...

        kg_check_results_text = "**Knowledge Graph Checks:**\n" + "\n".join(kg_consistency_checks) + "\n" if kg_consistency_checks else "**Knowledge Graph Checks:**\nNo specific inconsistencies flagged by KG queries.\n"
        # --- End KG Queries ---

        prompt = f"""/no_think
        
        You are a meticulous continuity editor. Analyze the following draft text snippet for Chapter {chapter_number}.
Compare it rigorously against the provided Plot Outline, Character Profiles, World Building notes, AND the results of Knowledge Graph consistency checks.
Identify and list ONLY specific, objective contradictions, factual inconsistencies, or significant deviations from established information.
Prioritize issues flagged by the Knowledge Graph Checks.
Do NOT list stylistic choices, minor omissions, or subjective interpretations. Focus on clear errors.
If NO significant inconsistencies are found, respond ONLY with the single word: None

**Plot Outline:**
```json
{json.dumps(self.plot_outline, indent=2, ensure_ascii=False)}
```

**Character Profiles:**
```json
{json.dumps(self.character_profiles, indent=2, ensure_ascii=False)}
```

**World Building:**
```json
{json.dumps(self.world_building, indent=2, ensure_ascii=False)}
```

{kg_check_results_text}
**Chapter {chapter_number} Draft Text Snippet:**
--- BEGIN DRAFT ---
{text_snippet}
--- END DRAFT ---

**List specific inconsistencies below (or respond "None"):**
"""
        logger.info(f"Checking consistency for chapter {chapter_number} via LLM (with KG checks)...")
        response_raw = llm_interface.call_llm(prompt, temperature=0.6, max_tokens=500)
        response_cleaned = llm_interface.clean_model_response(response_raw).strip()

        if not response_cleaned or response_cleaned.lower() == "none":
            logger.info(f"Consistency check passed for chapter {chapter_number}.")
            return None
        else:
            # Prepend KG issues if LLM didn't explicitly include them
            final_issues = response_cleaned
            if kg_consistency_checks and not all(check[2:] in final_issues for check in kg_consistency_checks): # Basic check if KG issues are in response
                 final_issues = "KG Check Issues:\n" + "\n".join(kg_consistency_checks) + "\nLLM Analysis:\n" + response_cleaned
            logger.warning(f"Consistency issues reported for chapter {chapter_number}:\n{final_issues}")
            return final_issues


    def _validate_plot_arc(self, chapter_text: Optional[str], chapter_number: int) -> Optional[str]:
        """Validates chapter alignment with plot point, potentially using KG facts."""
        if not chapter_text: return None
        plot_point_focus, plot_point_index = self._get_plot_point_info(chapter_number)
        if plot_point_focus is None: return None

        logger.info(f"Validating plot arc for chapter {chapter_number} against Plot Point {plot_point_index + 1}: '{plot_point_focus[:100]}...'")
        summary = self._summarize_chapter(chapter_text, chapter_number)
        validation_text = summary if summary else chapter_text[:1500]
        if not validation_text: return None

        # --- Perform KG Queries for Plot Point Conditions ---
        kg_plot_checks = []
        # Example: If plot point 3 requires confronting the Iron Collective
        if plot_point_index == 2: # Plot point 3 (0-indexed)
             protagonist_name = self.plot_outline.get("protagonist_name", "Sága")
             # Check if an interaction triple exists *within this chapter's extracted KG data*
             # Note: This requires KG extraction to happen *before* final validation, or querying based on text analysis.
             # Simpler approach for now: Check if the entity is mentioned prominently.
             # A more advanced approach would query KG triples added *by this chapter*.
             if "Iron Collective" not in chapter_text: # Simplified check
                  kg_plot_checks.append("- KG Check: Plot Point 3 requires confronting 'Iron Collective', but entity not prominently mentioned in chapter text.")

        # Add more KG checks based on specific plot point requirements...
        # ...

        kg_check_results_text = "**Knowledge Graph Plot Checks:**\n" + "\n".join(kg_plot_checks) + "\n" if kg_plot_checks else ""
        # --- End KG Queries ---


        prompt = f"""/no_think
        
        You are a strict story structure analyst. Your task is to determine if the provided chapter text successfully addresses the core elements of the intended plot point, considering any relevant Knowledge Graph checks.

**Intended Plot Point (Plot Point {plot_point_index + 1}):**
"{plot_point_focus}"

{kg_check_results_text}
**Chapter {chapter_number} Text (Summary or Snippet):**
"{validation_text}"

**Evaluation:**
Does the chapter text align with the core idea of the intended plot point, satisfying any KG checks?
**CRITICAL INSTRUCTION:** You MUST respond with **ONLY ONE** of the following two formats:
1.  If the chapter aligns sufficiently: Respond with the single word `Yes`
2.  If the chapter significantly deviates or misses the main focus (including KG check failures): Respond with `No, because...` followed by a VERY concise explanation (1 sentence maximum, mention KG failure if applicable).

**ONLY these two response formats are acceptable.**
Response:"""

        logger.info(f"Calling LLM for plot arc validation (with KG checks) for chapter {chapter_number}...")
        validation_response_raw = llm_interface.call_llm(prompt, temperature=0.6, max_tokens=100)
        cleaned_plot_response = llm_interface.clean_model_response(validation_response_raw).strip()

        if cleaned_plot_response.lower() == "yes":
            logger.info(f"Plot arc validation passed for chapter {chapter_number} (LLM responded 'Yes').")
            return None
        elif cleaned_plot_response.lower().startswith("no, because"):
            reason = cleaned_plot_response[len("no, because"):].strip()
            if not reason: reason = "LLM indicated deviation but provided no specific reason."
            # Prepend KG issues if they were flagged but not mentioned by LLM
            if kg_plot_checks and not any(check[2:20] in reason for check in kg_plot_checks): # Basic check
                 reason = "KG Plot Check Failed: " + "; ".join(k[2:] for k in kg_plot_checks) + ". LLM Reason: " + reason
            logger.warning(f"Plot arc deviation identified for chapter {chapter_number}: {reason}")
            return reason
        else:
            logger.warning(f"Plot arc validation for chapter {chapter_number} returned an ambiguous/incorrect format: '{cleaned_plot_response}'. Assuming alignment.")
            return None


    # --- Knowledge Update Functions (JSON part separated, KG part added) ---

    def _update_character_profiles(self, chapter_text: Optional[str], chapter_number: int):
        """Updates character profiles JSON based on chapter text analysis."""
        # (Existing logic for JSON updates - including dynamic adaptation if enabled)
        if not chapter_text: return
        text_snippet = chapter_text[:config.KNOWLEDGE_UPDATE_SNIPPET_SIZE]

        dynamic_instructions = ""
        if config.ENABLE_DYNAMIC_STATE_ADAPTATION:
            dynamic_instructions = """
3.  **Dynamic Adaptation:** For existing characters, explicitly propose modifications to `traits` or `description` using a `"modification_proposal"` field if warranted. Example: `"modification_proposal": "MODIFY traits: ADD 'Determined'"`.
4.  **Crucially:** Only include characters updated, newly introduced, or with a modification proposal."""
        else:
             dynamic_instructions = "3. **Crucially:** Only include characters updated or newly introduced."

        prompt = f"""/no_think
        
        You are a literary analyst tracking characters. Analyze Chapter {chapter_number} snippet for updates to character profiles (traits, relationships, status, description). Your output MUST be a single, valid JSON object.

**Chapter Text Snippet:**
--- BEGIN TEXT ---
{text_snippet}...
--- END TEXT ---

**Current Character Profiles:**
```json
{json.dumps(self.character_profiles, indent=2, ensure_ascii=False)}
```

**Instructions:**
1.  Identify every character updated or introduced in the snippet.
2.  For each: Note new traits, relationship changes, status updates, description changes. Add a `development_in_chapter_{chapter_number}` key summarizing their role/change.
{dynamic_instructions}
5.  **CRITICAL: Output ONLY the JSON object.** Use valid JSON syntax. If no updates, output `{{}}`.

**Example Output Structure (Include only updated chars):**
```json
{{
  "Character Name 1": {{
    "traits": ["New Trait Observed"],
    "relationships": {{ "Other Character": "Relationship became strained" }},
    "modification_proposal": "MODIFY traits: ADD 'Determined', REMOVE 'Hesitant'",
    "development_in_chapter_{chapter_number}": "Made a critical decision...",
    "status": "Major change observed"
  }},
  "New Character Name": {{ ... }}
}}
```
"""
        logger.info(f"Analyzing Chapter {chapter_number} to update character profiles JSON (Dynamic: {config.ENABLE_DYNAMIC_STATE_ADAPTATION})...")
        raw_analysis = llm_interface.call_llm(prompt, temperature=0.6)
        updates = llm_interface.parse_llm_json_response(raw_analysis, f"character profile update for chapter {chapter_number}")

        if not updates or not isinstance(updates, dict) or not updates:
            logger.info(f"LLM analysis found no character profile JSON updates in chapter {chapter_number}.")
            return

        logger.info(f"Merging character profile JSON updates for chapter {chapter_number} for characters: {list(updates.keys())}")
        updated_chars_count = 0
        new_chars_count = 0

        for char_name, char_update in updates.items():
            if not isinstance(char_update, dict): continue
            dev_key = f"development_in_chapter_{chapter_number}"
            if dev_key not in char_update and len(char_update) > 1: char_update[dev_key] = "Character appeared or was mentioned."

            if char_name not in self.character_profiles:
                new_chars_count += 1
                logger.info(f"Adding new character '{char_name}' to JSON profiles based on chapter {chapter_number}.")
                self.character_profiles[char_name] = {
                    "description": char_update.get("description", f"Introduced in Chapter {chapter_number}."),
                    "traits": sorted(list(set(t for t in char_update.get("traits", []) if isinstance(t, str)))) if isinstance(char_update.get("traits"), list) else [],
                    "relationships": char_update.get("relationships", {}) if isinstance(char_update.get("relationships"), dict) else {},
                    dev_key: char_update.get(dev_key, f"Introduced in Chapter {chapter_number}."),
                    "status": char_update.get("status", "Newly introduced")
                }
                if config.ENABLE_DYNAMIC_STATE_ADAPTATION and "modification_proposal" in char_update:
                    self._apply_modification_proposal(self.character_profiles[char_name], char_update["modification_proposal"], char_name)
            else:
                updated_chars_count += 1
                logger.debug(f"Updating existing character '{char_name}' in JSON profiles based on chapter {chapter_number}.")
                existing_profile = self.character_profiles[char_name]
                if config.ENABLE_DYNAMIC_STATE_ADAPTATION and "modification_proposal" in char_update:
                    self._apply_modification_proposal(existing_profile, char_update["modification_proposal"], char_name)
                for key, value in char_update.items():
                    if key == "modification_proposal": continue
                    if key == "traits" and isinstance(value, list):
                        if "traits" not in existing_profile or not isinstance(existing_profile["traits"], list): existing_profile["traits"] = []
                        existing_traits = set(existing_profile["traits"])
                        new_traits = set(trait for trait in value if isinstance(trait, str))
                        existing_profile["traits"] = sorted(list(existing_traits.union(new_traits)))
                    elif key == "relationships" and isinstance(value, dict):
                         if not isinstance(existing_profile.get("relationships"), dict): existing_profile["relationships"] = {}
                         existing_profile["relationships"].update(value)
                    elif key == "description" and isinstance(value, str) and value:
                         if not (config.ENABLE_DYNAMIC_STATE_ADAPTATION and "modification_proposal" in char_update and "MODIFY description" in char_update["modification_proposal"]):
                             existing_profile["description"] = value
                    elif key == dev_key and isinstance(value, str) and value:
                         existing_profile[key] = value
                    elif key == "status" and isinstance(value, str) and value:
                         existing_profile["status"] = value
        logger.info(f"Character profile JSON merge complete. Updated: {updated_chars_count}, New: {new_chars_count}.")


    def _apply_modification_proposal(self, profile: Dict[str, Any], proposal: str, name: str):
        """Parses and applies modification proposals from LLM analysis to JSON profiles."""
        # (Existing code - unchanged)
        if not isinstance(proposal, str): return
        logger.debug(f"Applying modification proposal for '{name}': '{proposal}'")
        try:
            match_key = re.match(r"MODIFY\s+(\w+):", proposal, re.IGNORECASE)
            if not match_key: return
            key_to_modify = match_key.group(1).strip().lower()
            action_details = proposal[match_key.end():].strip()

            if key_to_modify == "traits":
                if "traits" not in profile or not isinstance(profile["traits"], list): profile["traits"] = []
                current_traits = set(profile["traits"])
                for match in re.findall(r"ADD\s+['\"]([^'\"]+)['\"]", action_details, re.IGNORECASE):
                    logger.debug(f"Applying ADD trait '{match}' for {name}")
                    current_traits.add(match)
                for match in re.findall(r"REMOVE\s+['\"]([^'\"]+)['\"]", action_details, re.IGNORECASE):
                     logger.debug(f"Applying REMOVE trait '{match}' for {name}")
                     current_traits.discard(match)
                profile["traits"] = sorted(list(current_traits))
                logger.info(f"Applied trait modifications for '{name}'. New traits: {profile['traits']}")
            elif key_to_modify == "description":
                new_desc = action_details.strip("'\" ")
                if new_desc:
                    profile["description"] = new_desc
                    logger.info(f"Applied description modification for '{name}'.")
            else:
                logger.warning(f"Unknown modification key '{key_to_modify}' in proposal for '{name}'.")
        except Exception as e:
             logger.error(f"Error applying modification proposal for '{name}': {e}. Proposal: '{proposal}'", exc_info=True)


    def _update_world_building(self, chapter_text: Optional[str], chapter_number: int):
        """Updates world building JSON based on chapter text analysis."""
        # (Existing logic for JSON updates - including dynamic adaptation if enabled)
        if not chapter_text: return
        text_snippet = chapter_text[:config.KNOWLEDGE_UPDATE_SNIPPET_SIZE]

        dynamic_instructions = ""
        if config.ENABLE_DYNAMIC_STATE_ADAPTATION:
            dynamic_instructions = """
6.  **Dynamic Adaptation:** Propose modifications to existing items using a `"modification_proposal"` field. Example: `"modification_proposal": "MODIFY atmosphere: 'Now feels heavy.'"`.
7.  **CRITICAL: Output ONLY the JSON object containing ONLY new or updated elements.**"""
        else:
            dynamic_instructions = "6. **CRITICAL: Output ONLY the JSON object containing ONLY new or updated elements."

        prompt = f"""/no_think
        
        You are a world-building analyst. Examine Chapter {chapter_number} snippet and identify ONLY new info or significant elaborations on existing world elements based on current notes. Your output MUST be a single, valid JSON object.

**Chapter Text Snippet:**
--- BEGIN TEXT ---
{text_snippet}...
--- END TEXT ---

**Current World Building Notes:**
```json
{json.dumps(self.world_building, indent=2, ensure_ascii=False)}
```

**Instructions:**
1.  Identify new locations or significant new details about existing ones.
2.  Note newly revealed info about society, factions, history, etc.
3.  Extract new details about systems, tech, magic, flora/fauna.
4.  Capture new lore or historical context.
5.  Focus ONLY on info from THIS chapter. Add standard updates under relevant keys (e.g., `"description"`, `"elaboration_in_chapter_{chapter_number}"`).
{dynamic_instructions}
8.  Use valid JSON syntax. If no updates, output `{{}}`.

**Example Update Output Structure (only new/changed info):**
```json
{{
  "locations": {{
    "Existing Location": {{
      "modification_proposal": "MODIFY atmosphere: 'Now feels heavy.'",
      "elaboration_in_chapter_{chapter_number}": "Found a hidden passage.",
      "updated_in_chapter_{chapter_number}": true
    }},
    "Newly Discovered Area": {{ ... "added_in_chapter_{chapter_number}": true }}
  }}
}}
```
"""
        logger.info(f"Analyzing Chapter {chapter_number} to update world-building JSON (Dynamic: {config.ENABLE_DYNAMIC_STATE_ADAPTATION})...")
        raw_analysis = llm_interface.call_llm(prompt, temperature=0.6)
        updates = llm_interface.parse_llm_json_response(raw_analysis, f"world-building update for chapter {chapter_number}")

        if not updates or not isinstance(updates, dict) or not updates:
            logger.info(f"LLM analysis found no world-building JSON updates in chapter {chapter_number}.")
            return

        logger.info(f"Merging world-building JSON updates for chapter {chapter_number} for categories: {list(updates.keys())}")

        # --- Recursive Merge Logic (Handles modifications) ---
        def merge_dicts_recursive(target: Dict[str, Any], source: Dict[str, Any], context_name=""):
            if config.ENABLE_DYNAMIC_STATE_ADAPTATION and "modification_proposal" in source:
                proposal = source.get("modification_proposal")
                if isinstance(proposal, str):
                    logger.debug(f"Applying world modification proposal for '{context_name}': '{proposal}'")
                    try:
                        parts = proposal.split(":", 1)
                        if len(parts) == 2:
                            action_key, value_str = parts[0].strip().lower(), parts[1].strip().strip("'\" ")
                            action_parts = action_key.split(" ", 1)
                            if len(action_parts) == 2:
                                action, key = action_parts[0], action_parts[1]
                                if action == "modify":
                                    if key in target: target[key] = value_str; logger.info(f"Applied MODIFICATION to '{context_name}.{key}'")
                                    else: logger.warning(f"Cannot MODIFY non-existent key '{key}' in '{context_name}'")
                                elif action == "add":
                                    if key not in target: target[key] = []
                                    if isinstance(target.get(key), list):
                                        if value_str not in target[key]: target[key].append(value_str); logger.info(f"Applied ADD to list '{context_name}.{key}'")
                                    else: logger.warning(f"Cannot ADD to non-list key '{key}' in '{context_name}'")
                    except Exception as e_mod: logger.error(f"Error applying world mod proposal for '{context_name}': {e_mod}", exc_info=True)
                if "modification_proposal" in source: del source["modification_proposal"]

            for key, value in source.items():
                if key.startswith("updated_in_chapter_") or key.startswith("added_in_chapter_"): continue
                target_value = target.get(key)
                if isinstance(target_value, dict) and isinstance(value, dict):
                    merge_dicts_recursive(target_value, value, f"{context_name}.{key}" if context_name else key)
                    target_value[f"updated_in_chapter_{chapter_number}"] = True
                elif isinstance(value, dict):
                    value[f"added_in_chapter_{chapter_number}"] = True
                    target[key] = value
                elif isinstance(target_value, list) and isinstance(value, list):
                    items_added = False
                    for item in value:
                        if item not in target[key]: target[key].append(item); items_added = True
                    if items_added: target[f"updated_in_chapter_{chapter_number}"] = True
                elif value is not None and value != target_value:
                    if not (isinstance(value, str) and not value and isinstance(target_value, str) and target_value):
                        target[key] = value
                        target[f"updated_in_chapter_{chapter_number}"] = True
        # --- End Merge Logic ---

        categories_updated = []
        for category, cat_updates in updates.items():
            if category not in self.world_building:
                 if isinstance(cat_updates, dict) and cat_updates:
                      logger.info(f"Adding new world-building category: {category}")
                      cat_updates[f"added_in_chapter_{chapter_number}"] = True
                      self.world_building[category] = cat_updates
                      categories_updated.append(category)
                 continue
            if isinstance(self.world_building.get(category), dict) and isinstance(cat_updates, dict) and cat_updates:
                logger.debug(f"Merging world-building JSON updates for category: {category}")
                merge_dicts_recursive(self.world_building[category], cat_updates, category)
                self.world_building[category][f"updated_in_chapter_{chapter_number}"] = True
                categories_updated.append(category)
            else:
                 logger.warning(f"Skipping merge for world-building category '{category}': Invalid format or empty updates.")
        if categories_updated: logger.info(f"World-building JSON merge completed. Categories updated: {categories_updated}.")
        else: logger.info(f"No world-building JSON categories were merged.")


    # --- NEW: Knowledge Graph Extraction ---

    def _extract_and_update_kg(self, chapter_text: Optional[str], chapter_number: int):
        """
        Analyzes the finalized chapter text using an LLM to extract factual triples
        (subject, predicate, object) and adds them to the knowledge graph database.
        Includes stricter prompt and more robust parsing, accounting for mandatory think blocks
        and potential extraneous text/fences around the JSON list.
        """
        if not chapter_text:
            logger.warning(f"Skipping KG extraction for chapter {chapter_number}: Text is None.")
            return

        logger.info(f"Extracting Knowledge Graph triples for chapter {chapter_number}...")
        # Use a larger snippet, potentially the whole chapter if feasible, for better fact extraction
        text_snippet = chapter_text # Using full text, adjust if needed
        if len(text_snippet) > config.KNOWLEDGE_UPDATE_SNIPPET_SIZE * 3: # Example limit
             text_snippet = chapter_text[:config.KNOWLEDGE_UPDATE_SNIPPET_SIZE * 3]
             logger.warning(f"KG extraction using truncated text snippet ({len(text_snippet)} chars) for chapter {chapter_number}.")


        # Define a set of useful predicates (can be expanded)
        common_predicates = [
            "is_a", "located_in", "has_trait", "status_is", "feels", "knows",
            "interacted_with", "travelled_to", "discovered", "attacked", "helped",
            "damaged", "repaired", "contains", "part_of", "caused_by", "leads_to"
        ]

        # --- REFINED KG EXTRACTION PROMPT ---
        # Prompt remains the same as the previous version, emphasizing JSON list output
        prompt = f"""/no_think
        
        You are a Knowledge Graph Engineer. Extract factual relationships from the provided chapter text as (Subject, Predicate, Object) triples. Focus on concrete facts, events, states, and relationships relevant to plot, character, and world state.

**Chapter {chapter_number} Text Snippet:**
--- BEGIN TEXT ---
{text_snippet}
--- END TEXT ---

**Instructions:**
1.  Identify key entities (characters, locations, factions, items, concepts). Normalize names (e.g., "Sága").
2.  Identify relationships/properties. Use predicates from the suggested list or create concise alternatives.
3.  Extract facts as triples: `["Subject", "predicate", "Object"]`.
4.  Focus ONLY on information explicitly stated or strongly implied in THIS text.
5.  Prioritize facts relevant to state changes (location, status, relationships) and key events.
6.  **CRITICAL OUTPUT FORMAT:** Respond ONLY with a single, valid JSON list of lists. Each inner list MUST be a triple `["Subject", "predicate", "Object"]`.
7.  **DO NOT include ANY text, commentary, explanations, or markdown formatting (like ```json) before or after the JSON list.** The response MUST start directly with `[` and end directly with `]`.
8.  If no significant facts are found, output an empty JSON list: `[]`.

**Suggested Predicates:** {', '.join(common_predicates)}

**Example of CORRECT Output:**
```json
[["Sága", "travelled_to", "Eclipse Spire"], ["Sága", "status_is", "Conflicted"], ["Larkin", "is_a", "Subroutine"]]
```
(Your actual output should just be the JSON list, starting with `[` and ending with `]`, not the markdown ```json block)

JSON Output Only:
[
""" # Added opening bracket to guide the model

        # Call LLM for extraction
        raw_triples_json = llm_interface.call_llm(prompt, temperature=0.6, max_tokens=1500)

        # --- More Robust Parsing Logic ---
        parsed_triples = None
        if raw_triples_json:
            # 1. Clean the raw response (removes mandatory <think> block)
            cleaned_response = llm_interface.clean_model_response(raw_triples_json).strip()
            logger.debug(f"KG Extraction: Cleaned response snippet: {cleaned_response[:150]}...")

            # 2. Use regex to find the main JSON list structure `[...]`
            # This pattern looks for the outermost brackets, allowing for nested structures.
            # It handles potential leading/trailing text or markdown fences.
            # re.DOTALL makes '.' match newlines.
            json_match = re.search(r'(\[.*\])', cleaned_response, re.DOTALL)
            json_string_to_parse = None

            if json_match:
                json_string_to_parse = json_match.group(1).strip()
                logger.debug("KG Extraction: Found potential JSON list using regex `(\[.*\])`.")
                # Optional: Add a check for balanced brackets as a basic sanity check
                if json_string_to_parse.count('[') != json_string_to_parse.count(']'):
                    logger.warning("KG Extraction: Potential JSON list found by regex has unbalanced brackets. Parsing might fail.")
            else:
                logger.warning(f"KG Extraction: Could not find a `[...]` structure in the cleaned response. Snippet: {cleaned_response[:100]}...")
                # Try parsing the whole cleaned response as a last resort, though unlikely to succeed
                json_string_to_parse = cleaned_response

            # 3. Attempt to parse the identified JSON string
            if json_string_to_parse:
                try:
                    # Attempt parsing
                    parsed_data = json.loads(json_string_to_parse)
                    if isinstance(parsed_data, list):
                        parsed_triples = parsed_data
                        logger.debug(f"Successfully parsed JSON list for KG triples for chapter {chapter_number}.")
                    else:
                        logger.warning(f"Parsed JSON data for KG triples was not a list for chapter {chapter_number}. Type: {type(parsed_data)}")
                except json.JSONDecodeError as e:
                    # Log the specific error and the problematic string
                    error_snippet = json_string_to_parse[:200] # Show beginning of string
                    error_snippet_end = json_string_to_parse[-200:] # Show end of string
                    logger.error(f"Failed to parse JSON string for KG triples for chapter {chapter_number}: {e}. String snippet (start): {error_snippet}... String snippet (end): ...{error_snippet_end}")
            else:
                 logger.error(f"Could not identify a JSON string to parse for KG triples in chapter {chapter_number}.")


        if parsed_triples is None:
             logger.error(f"Failed to extract or parse KG triples list for chapter {chapter_number}. Raw response snippet: {raw_triples_json[:200]}")
             # Save the failed extraction attempt for debugging
             self._save_debug_output(chapter_number, "kg_extraction_raw_fail", raw_triples_json or "EMPTY")
             return # Stop KG update if parsing failed

        # Add extracted triples to the database
        added_count = 0
        skipped_count = 0
        for triple in parsed_triples:
            # Validate triple structure before adding
            if isinstance(triple, list) and len(triple) == 3 and all(isinstance(t, str) for t in triple):
                # Further check: Ensure strings are not empty after stripping
                subj, pred, obj = [t.strip() for t in triple]
                if subj and pred and obj:
                    # Add triple to DB (method handles validation and avoids duplicates for the chapter)
                    self.db_manager.add_kg_triple(subj, pred, obj, chapter_number)
                    added_count += 1
                else:
                    # Log warnings for triples with empty strings after stripping
                    logger.warning(f"Skipping invalid triple with empty component(s) after stripping in chapter {chapter_number}: {triple}")
                    skipped_count += 1
            else:
                logger.warning(f"Skipping invalid triple format in KG extraction result for chapter {chapter_number}: {triple}")
                skipped_count += 1

        logger.info(f"Added {added_count} KG triples extracted from chapter {chapter_number}. Skipped {skipped_count} invalid/empty triples.")


    # --- Helper for Planning Snippets (Unchanged) ---
    def _get_relevant_character_state_snippet(self) -> str:
        """Creates a concise snippet of relevant character states for planning prompts."""
        # (Existing code - unchanged)
        snippet = {}
        count = 0
        sorted_chars = sorted(self.character_profiles.keys())
        for name in sorted_chars:
            if count >= 5: break
            profile = self.character_profiles[name]
            dev_keys = sorted([k for k in profile if k.startswith("development_in_chapter_")], reverse=True)
            recent_dev_note = profile.get(dev_keys[0], "N/A") if dev_keys else "N/A"
            snippet[name] = {
                "desc": profile.get("description", "")[:100] + "...",
                "status": profile.get("status", "Unknown"),
                "recent_dev": recent_dev_note[:150] + "..."
            }
            count += 1
        return json.dumps(snippet, indent=2, ensure_ascii=False) if snippet else "No character profiles available."

    def _get_relevant_world_state_snippet(self) -> str:
        """Creates a concise snippet of relevant world states for planning prompts."""
        # (Existing code - unchanged)
        snippet = {}
        if "locations" in self.world_building and isinstance(self.world_building["locations"], dict):
             snippet["locations"] = list(self.world_building["locations"].keys())[:3]
        if "society" in self.world_building and isinstance(self.world_building.get("society"), dict) and "Key Factions" in self.world_building["society"] and isinstance(self.world_building["society"]["Key Factions"], dict):
             snippet["factions"] = list(self.world_building["society"]["Key Factions"].keys())[:2]
        if "systems" in self.world_building and isinstance(self.world_building["systems"], dict):
             snippet["systems"] = list(self.world_building["systems"].keys())[:2]
        return json.dumps(snippet, indent=2, ensure_ascii=False) if snippet else "No world-building data available."


    def _save_debug_output(self, chapter_number: int, stage: str, content: str):
        """Helper to save raw text during debugging stages."""
        # (Existing code - unchanged)
        if not content: return
        try:
            debug_dir = os.path.join(config.OUTPUT_DIR, "debug")
            os.makedirs(debug_dir, exist_ok=True)
            safe_stage = "".join(c if c.isalnum() else "_" for c in stage)
            file_path = os.path.join(debug_dir, f"chapter_{chapter_number}_{safe_stage}.txt")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.debug(f"Saved debug output for chapter {chapter_number} stage '{stage}' to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save debug output for chapter {chapter_number} stage '{stage}': {e}", exc_info=True)

