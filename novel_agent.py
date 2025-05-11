# novel_agent.py
"""
Contains the main NovelWriterAgent class for the Saga Novel Generation system.
This class orchestrates the novel writing process, managing state and delegating
specific tasks (like setup, planning, drafting, evaluation, revision, knowledge updates,
and context building) to specialized logic modules.

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
import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any

import config
import llm_interface
from database_manager import DatabaseManager
from state_manager import state_manager
from type import JsonStateData, EvaluationResult, SceneDetail

# Import a BUNCH of functions from the new modules
from initial_setup_logic import generate_plot_outline_logic, generate_world_building_logic
from chapter_planning_logic import plan_chapter_scenes_logic
from chapter_drafting_logic import generate_chapter_draft_logic
from chapter_evaluation_logic import evaluate_chapter_draft_logic
from chapter_revision_logic import revise_chapter_draft_logic
from knowledge_management_logic import (
    update_all_knowledge_bases_logic,
    summarize_chapter_text_logic,
    prepopulate_kg_from_initial_data_logic
)
from context_generation_logic import generate_chapter_context_logic


logger = logging.getLogger(__name__)

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

        # Try to load from ORM first, fall back to JSON files if needed
        self.plot_outline = state_manager.get_plot_outline()
        if not self.plot_outline:
            logger.info("Plot outline not found in ORM database. Trying JSON file.")
            self.plot_outline = self._load_json_file(config.PLOT_OUTLINE_FILE, "plot_outline")
            if self.plot_outline:
                # If we loaded from JSON, save to ORM for future use
                state_manager.save_plot_outline(self.plot_outline)
                logger.info("Saved plot outline from JSON to ORM database.")

        self.character_profiles = state_manager.get_character_profiles()
        if not self.character_profiles:
            logger.info("Character profiles not found in ORM database. Trying JSON file.")
            self.character_profiles = self._load_json_file(config.CHARACTER_PROFILES_FILE, "character_profiles")
            if self.character_profiles:
                # If we loaded from JSON, save to ORM for future use
                state_manager.save_character_profiles(self.character_profiles)
                logger.info("Saved character profiles from JSON to ORM database.")

        self.world_building = state_manager.get_world_building()
        if not self.world_building:
            logger.info("World building not found in ORM database. Trying JSON file.")
            self.world_building = self._load_json_file(config.WORLD_BUILDER_FILE, "world_building")
            if self.world_building:
                # If we loaded from JSON, save to ORM for future use
                state_manager.save_world_building(self.world_building)
                logger.info("Saved world building from JSON to ORM database.")
        
        logger.info("Finished loading initial state.")

    def _save_single_json_state_sync(self, file_path: str, data_dict: JsonStateData, dict_name: str):
        """Synchronous helper to save a single JSON state file."""
        if not data_dict or not isinstance(data_dict, dict):
            logger.debug(f"Skipping save for {file_path}, data empty or not a dict.")
            return False

        data_to_save = json.loads(json.dumps(data_dict)) # Deep copy for manipulation

        if not data_to_save.get("is_default"): 
            data_to_save.pop("is_default", None)

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
        """Asynchronously saves all state data (plot, characters, world) to both ORM and JSON files."""
        logger.debug("Saving agent state (plot, characters, world)...")
        loop = asyncio.get_event_loop()
        
        # Save to ORM database
        orm_tasks = [
            loop.run_in_executor(None, state_manager.save_plot_outline, self.plot_outline),
            loop.run_in_executor(None, state_manager.save_character_profiles, self.character_profiles),
            loop.run_in_executor(None, state_manager.save_world_building, self.world_building)
        ]
        
        # Also save to JSON files for backward compatibility
        json_tasks = [
            loop.run_in_executor(None, self._save_single_json_state_sync, config.PLOT_OUTLINE_FILE, self.plot_outline, "plot_outline"),
            loop.run_in_executor(None, self._save_single_json_state_sync, config.CHARACTER_PROFILES_FILE, self.character_profiles, "character_profiles"),
            loop.run_in_executor(None, self._save_single_json_state_sync, config.WORLD_BUILDER_FILE, self.world_building, "world_building"),
        ]
        
        # Run all tasks concurrently
        all_tasks = orm_tasks + json_tasks
        results = await asyncio.gather(*all_tasks)
        
        # Count successful saves (first half of results are ORM, second half are JSON)
        orm_saved_count = sum(1 for r in results[:3] if r)
        json_saved_count = sum(1 for r in results[3:] if r)

        if orm_saved_count > 0:
            logger.info(f"State saved to ORM database for {orm_saved_count} object(s).")
        else:
            logger.warning("No state objects were saved to ORM database.")
            
        if json_saved_count > 0:
            logger.info(f"State saved to JSON files for {json_saved_count} file(s).")
        else:
            logger.info("No JSON state files were updated/saved.")

    async def generate_plot_outline(self, default_protagonist_name: str, unhinged_mode: bool, **kwargs) -> JsonStateData:
        """Generates a new plot outline."""
        # This method now calls the logic from initial_setup_logic.py
        return await generate_plot_outline_logic(self, default_protagonist_name, unhinged_mode, **kwargs)

    async def generate_world_building(self) -> JsonStateData:
        """Generates initial world-building data."""
        # This method now calls the logic from initial_setup_logic.py
        return await generate_world_building_logic(self)

    async def _prepopulate_knowledge_graph(self):
        """Pre-populates the Knowledge Graph from initial plot and world data."""
        # This method now calls the logic from knowledge_management_logic.py
        await prepopulate_kg_from_initial_data_logic(self)


    def _get_plot_point_info(self, chapter_number: int) -> Tuple[Optional[str], int]:
        """Retrieves the plot point focus for a given chapter number."""
        plot_points = self.plot_outline.get("plot_points", [])
        if not isinstance(plot_points, list) or not plot_points:
            logger.warning(f"No plot points defined in plot outline for chapter {chapter_number}.")
            return None, -1
        
        if chapter_number <= 0: 
            logger.warning(f"Invalid chapter number {chapter_number} for plot point lookup.")
            return None, -1
        
        plot_point_index = min(chapter_number - 1, len(plot_points) - 1)
        
        if 0 <= plot_point_index < len(plot_points):
            return plot_points[plot_point_index], plot_point_index
        
        logger.warning(f"Could not determine plot point for chapter {chapter_number} from {len(plot_points)} available points.")
        return None, -1


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

        chapter_plan: Optional[List[SceneDetail]] = await plan_chapter_scenes_logic(self, chapter_number)
        
        if config.ENABLE_AGENTIC_PLANNING and chapter_plan is None:
            logger.warning(f"Ch {chapter_number}: Agentic planning was enabled but failed to produce a plan. Proceeding with plot point focus only.")

        context_for_draft = await generate_chapter_context_logic(self, chapter_number)
        plot_point_focus, _ = self._get_plot_point_info(chapter_number)
        if plot_point_focus is None: 
            logger.error(f"Ch {chapter_number} generation halted: no plot point focus.")
            return None

        initial_draft_text, initial_raw_llm_text = await generate_chapter_draft_logic(
            self, chapter_number, plot_point_focus, context_for_draft, chapter_plan
        )

        if not initial_draft_text:
            logger.error(f"Failed to generate initial draft for ch {chapter_number}.")
            await self._save_debug_output(chapter_number, "initial_draft_fail_raw_llm", initial_raw_llm_text or "")
            return None

        evaluation = await evaluate_chapter_draft_logic(self, initial_draft_text, chapter_number, context_for_draft)
        current_text = initial_draft_text
        final_raw_output_log = f"--- INITIAL DRAFT (RAW LLM OUTPUT) ---\n{initial_raw_llm_text}\n\n"
        proceeded_with_flaws = False 

        if evaluation["needs_revision"]:
            revision_reason_str = "\n- ".join(evaluation["reasons"])
            logger.warning(f"Ch {chapter_number} flagged for revision. Reason(s):\n- {revision_reason_str}")
            
            revised_text_tuple = await revise_chapter_draft_logic(
                self, current_text, chapter_number, revision_reason_str, context_for_draft, chapter_plan
            )
            if revised_text_tuple:
                revised_text, raw_revision_llm_output = revised_text_tuple
                logger.info(f"Revision successful for ch {chapter_number}. Re-evaluating revised draft...")
                revised_evaluation = await evaluate_chapter_draft_logic(self, revised_text, chapter_number, context_for_draft)
                
                if revised_evaluation["needs_revision"]:
                    logger.error(f"Revised draft for ch {chapter_number} STILL FAILED evaluation. Reasons:\n- " + "\n- ".join(revised_evaluation["reasons"]))
                    proceeded_with_flaws = True
                else:
                    logger.info(f"Revised draft for ch {chapter_number} passed re-evaluation.")
                
                current_text = revised_text 
                final_raw_output_log += f"--- REVISION (Reason: {evaluation['reasons']}) (RAW LLM OUTPUT) ---\n{raw_revision_llm_output}\n\n" # Use original evaluation reasons for log.
            else: 
                logger.error(f"Revision attempt failed for ch {chapter_number}. Proceeding with the original (flawed) draft.")
                proceeded_with_flaws = True
                final_raw_output_log += f"--- REVISION ATTEMPT FAILED (Reason: {evaluation['reasons']}) ---\n\n"
        else:
            logger.info(f"Initial draft for ch {chapter_number} passed evaluation.")

        if not await self._finalize_chapter_core(chapter_number, current_text, final_raw_output_log, proceeded_with_flaws):
             logger.error(f"=== Finished Ch {chapter_number} WITH ERRORS during core finalization (DB/File save) ===")
             return None

        await update_all_knowledge_bases_logic(self, chapter_number, current_text, proceeded_with_flaws)
        
        self.chapter_count = max(self.chapter_count, chapter_number) 
        await self._save_all_json_state() 
        
        status_message = "Successfully" if not proceeded_with_flaws else "With Flaws (Accepted after failed revision or due to issues)"
        logger.info(f"=== Finished Ch {chapter_number} {status_message} ===")
        return current_text

    async def _finalize_chapter_core(self, chapter_number: int, final_text: str, raw_llm_log_for_db: str, from_flawed_draft: bool) -> bool:
        """Core finalization: summarize, embed, save to DB and files."""
        logger.info(f"Finalizing chapter {chapter_number} (From flawed draft: {from_flawed_draft}). Text length: {len(final_text)}.")
        if not final_text:
            logger.error(f"Cannot finalize ch {chapter_number}: Final text is missing or empty.")
            return False

        # These can run concurrently
        summary_task = summarize_chapter_text_logic(self, final_text, chapter_number) # from knowledge_management_logic
        # Embedding directly uses llm_interface
        embedding_task = asyncio.create_task(llm_interface.async_get_embedding(final_text)) # Renamed to avoid conflict
        
        summary, final_embedding = await asyncio.gather(summary_task, embedding_task)

        if final_embedding is None:
            logger.error(f"CRITICAL: Failed to generate embedding for final text of Chapter {chapter_number}. This may impact future context.")

        try:
            await self.db_manager.async_save_chapter_data(
                chapter_number, final_text, raw_llm_log_for_db, summary, final_embedding, from_flawed_draft
            )
        except Exception as e: 
            logger.error(f"Database save failed for chapter {chapter_number}: {e}", exc_info=True)
            return False 

        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, self._save_chapter_text_files_sync, chapter_number, final_text, raw_llm_log_for_db)
            logger.info(f"Saved chapter text and raw LLM log files for ch {chapter_number}.")
        except IOError as e:
            logger.error(f"Failed writing chapter text/log files for ch {chapter_number}: {e}", exc_info=True)
        
        logger.info(f"Core finalization complete for ch {chapter_number}.")
        return True

    def _save_chapter_text_files_sync(self, chapter_number: int, final_text: str, raw_llm_log: str):
        """Synchronous helper for saving chapter text and raw LLM log files."""
        chapter_file_path = os.path.join(config.CHAPTERS_DIR, f"chapter_{chapter_number:04d}.txt") 
        log_file_path = os.path.join(config.CHAPTER_LOGS_DIR, f"chapter_{chapter_number:04d}_raw_llm_log.txt")

        with open(chapter_file_path, 'w', encoding='utf-8') as f:
            f.write(final_text)
        with open(log_file_path, 'w', encoding='utf-8') as f:
            f.write(raw_llm_log)

    async def _save_debug_output(self, chapter_number: int, stage_description: str, content: Any):
        """Saves content to a debug file, useful for inspecting LLM outputs or intermediate states."""
        if content is None: return 
        
        content_str = str(content) if not isinstance(content, str) else content
        if not content_str.strip(): return 
            
        try:
            safe_stage_desc = "".join(c if c.isalnum() or c in ['_', '-'] else "_" for c in stage_description)
            file_name = f"chapter_{chapter_number:04d}_{safe_stage_desc}.txt"
            file_path = os.path.join(config.DEBUG_OUTPUTS_DIR, file_name)
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._save_debug_output_sync_io, file_path, content_str, chapter_number, stage_description)
        except Exception as e: 
            logger.error(f"Failed to initiate save for debug output (Ch {chapter_number}, Stage '{stage_description}'): {e}", exc_info=True)

    def _save_debug_output_sync_io(self, file_path: str, content_str: str, chapter_number: int, stage_description: str):
        """Synchronous I/O part of saving debug output."""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True) 
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content_str)
            logger.debug(f"Saved debug output for Ch {chapter_number}, Stage '{stage_description}' to {file_path}")
        except IOError as e:
            logger.error(f"IOError saving debug output for Ch {chapter_number}, Stage '{stage_description}' to {file_path}: {e}", exc_info=True)
        except Exception as e: 
            logger.error(f"Unexpected error saving debug output for Ch {chapter_number}, Stage '{stage_description}' to {file_path}: {e}", exc_info=True)
