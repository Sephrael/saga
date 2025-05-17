# novel_agent.py
import os
import json
import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any # JsonStateData might be specific, keep generic Dict for attributes

import config
import llm_interface
from state_manager import state_manager # Use the singleton instance
from type import JsonStateData, EvaluationResult, SceneDetail # Keep JsonStateData in imports in case it's used correctly elsewhere or by sub-logics

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
from context_generation_logic import generate_hybrid_chapter_context_logic


logger = logging.getLogger(__name__)

class NovelWriterAgent:
    def __init__(self):
        logger.info("Initializing NovelWriterAgent instance...")
        self.plot_outline: Dict[str, Any] = {}
        self.character_profiles: Dict[str, Any] = {}
        self.world_building: Dict[str, Any] = {}
        self.chapter_count: int = 0
        logger.info("NovelWriterAgent instance created. Call async_init() to load/prepare state.")

    async def async_init(self):
        """Asynchronously initializes agent state by loading from ORM and ensuring DB schema."""
        logger.info("NovelWriterAgent async_init started...")
        
        self.chapter_count = await state_manager.async_load_chapter_count()
        logger.info(f"Loaded chapter count from ORM: {self.chapter_count}")

        load_tasks = {
            "plot": state_manager.get_plot_outline(),
            "chars": state_manager.get_character_profiles(),
            "world": state_manager.get_world_building()
        }
        results = await asyncio.gather(*load_tasks.values(), return_exceptions=True)
        loaded_data = dict(zip(load_tasks.keys(), results))

        for key, value in loaded_data.items():
            if isinstance(value, Exception):
                logger.error(f"Error loading {key} during async_init: {value}", exc_info=value)
                # Ensure attributes are initialized to empty dicts on error, consistent with their type
                if key == "plot": self.plot_outline = {}
                elif key == "chars": self.character_profiles = {}
                elif key == "world": self.world_building = {}
            else:
                # Value should be a dict. Assigning Dict[str, Any] to Dict[str, Any] is fine.
                if key == "plot": self.plot_outline = value if isinstance(value, dict) else {}
                elif key == "chars": self.character_profiles = value if isinstance(value, dict) else {}
                elif key == "world": self.world_building = value if isinstance(value, dict) else {}
        
        if not self.plot_outline:
            logger.warning("Plot outline is empty after ORM load. Expected during initial setup or if save failed previously.")
        if not self.character_profiles:
            logger.warning("Character profiles are empty after ORM load. Expected during initial setup or if save failed previously.")
        if not self.world_building:
            logger.warning("World building is empty after ORM load. Expected during initial setup or if save failed previously.")
        
        logger.info("NovelWriterAgent async_init complete.")

    async def _save_all_json_state(self):
        logger.debug("Saving agent state (plot, characters, world) to ORM...")
        
        tasks = [
            state_manager.save_plot_outline(self.plot_outline),
            state_manager.save_character_profiles(self.character_profiles),
            state_manager.save_world_building(self.world_building)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = 0
        for i, res in enumerate(results):
            item_name = ["plot_outline", "character_profiles", "world_building"][i]
            if isinstance(res, Exception):
                logger.error(f"Failed to save {item_name} to ORM: {res}", exc_info=res)
            elif res is True:
                success_count += 1
            else: # Should not happen if save methods return bool or raise
                logger.warning(f"Unexpected return value from save_{item_name}: {res}")

        if success_count == len(tasks):
            logger.info(f"All ({success_count}) state objects saved to ORM successfully.")
        else:
            logger.warning(f"Only {success_count}/{len(tasks)} state objects saved to ORM successfully.")

    # Changed return type from JsonStateData to Dict[str, Any]
    async def generate_plot_outline(self, default_protagonist_name: str, unhinged_mode: bool, **kwargs) -> Dict[str, Any]:
        return await generate_plot_outline_logic(self, default_protagonist_name, unhinged_mode, **kwargs)

    # Changed return type from JsonStateData to Dict[str, Any]
    async def generate_world_building(self) -> Dict[str, Any]:
        return await generate_world_building_logic(self)

    async def _prepopulate_knowledge_graph(self):
        await prepopulate_kg_from_initial_data_logic(self)

    def _get_plot_point_info(self, chapter_number: int) -> Tuple[Optional[str], int]:
        plot_points = self.plot_outline.get("plot_points", [])
        if not isinstance(plot_points, list) or not plot_points:
            logger.warning(f"No plot points defined in plot outline for chapter {chapter_number}.")
            return None, -1
        
        if chapter_number <= 0: 
            logger.warning(f"Invalid chapter number {chapter_number} for plot point lookup.")
            return None, -1
        
        plot_point_index = min(chapter_number - 1, len(plot_points) - 1)
        
        if 0 <= plot_point_index < len(plot_points):
            # Ensure the plot point itself is a string, or handle appropriately if it can be complex
            plot_point = plot_points[plot_point_index]
            if isinstance(plot_point, str): # Or check for the expected structure of a plot point
                return plot_point, plot_point_index
            else:
                logger.warning(f"Plot point at index {plot_point_index} for chapter {chapter_number} is not a string: {type(plot_point)}")
                # Depending on expected structure, you might return str(plot_point) or handle error
                return str(plot_point) if plot_point is not None else None, plot_point_index
        
        logger.warning(f"Could not determine plot point for chapter {chapter_number} from {len(plot_points)} available points.")
        return None, -1

    async def write_chapter(self, chapter_number: int) -> Optional[str]:
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
            logger.warning(f"Ch {chapter_number}: Agentic planning enabled but failed to produce a plan. Proceeding with plot point focus only.")

        # Generate hybrid context using the chapter_plan
        hybrid_context_for_draft = await generate_hybrid_chapter_context_logic(self, chapter_number, chapter_plan) 
        
        plot_point_focus, _ = self._get_plot_point_info(chapter_number)
        if plot_point_focus is None: 
            logger.error(f"Ch {chapter_number} generation halted: no plot point focus.")
            return None

        initial_draft_text, initial_raw_llm_text = await generate_chapter_draft_logic(
            self, chapter_number, plot_point_focus, hybrid_context_for_draft, chapter_plan
        )

        if not initial_draft_text:
            logger.error(f"Failed to generate initial draft for ch {chapter_number}.")
            await self._save_debug_output(chapter_number, "initial_draft_fail_raw_llm", initial_raw_llm_text or "")
            return None
        
        # The `evaluate_chapter_draft_logic` now uses the `hybrid_context_for_draft` (passed as `previous_chapters_context`)
        # for its comprehensive evaluation, including the consistency check aspects that previously needed context.
        evaluation = await evaluate_chapter_draft_logic(self, initial_draft_text, chapter_number, hybrid_context_for_draft)
        current_text = initial_draft_text
        final_raw_output_log = f"--- INITIAL DRAFT (RAW LLM OUTPUT) ---\n{initial_raw_llm_text}\n\n"
        proceeded_with_flaws = False 

        if evaluation["needs_revision"]:
            revision_reason_str = "\n- ".join(evaluation["reasons"])
            logger.warning(f"Ch {chapter_number} flagged for revision. Reason(s):\n- {revision_reason_str}")
            
            revised_text_tuple = await revise_chapter_draft_logic(
                self, current_text, chapter_number, revision_reason_str, hybrid_context_for_draft, chapter_plan
            )
            if revised_text_tuple:
                revised_text, raw_revision_llm_output = revised_text_tuple
                logger.info(f"Revision successful for ch {chapter_number}. Re-evaluating revised draft...")
                revised_evaluation = await evaluate_chapter_draft_logic(self, revised_text, chapter_number, hybrid_context_for_draft)
                
                if revised_evaluation["needs_revision"]:
                    logger.error(f"Revised draft for ch {chapter_number} STILL FAILED evaluation. Reasons:\n- " + "\n- ".join(revised_evaluation["reasons"]))
                    proceeded_with_flaws = True
                else:
                    logger.info(f"Revised draft for ch {chapter_number} passed re-evaluation.")
                
                current_text = revised_text 
                final_raw_output_log += f"--- REVISION (Reason: {evaluation['reasons']}) (RAW LLM OUTPUT) ---\n{raw_revision_llm_output}\n\n"
            else: 
                logger.error(f"Revision attempt failed for ch {chapter_number}. Proceeding with the original (flawed) draft.")
                proceeded_with_flaws = True
                final_raw_output_log += f"--- REVISION ATTEMPT FAILED (Reason: {evaluation['reasons']}) ---\n\n"
        else:
            logger.info(f"Initial draft for ch {chapter_number} passed evaluation.")

        if not await self._finalize_chapter_core(chapter_number, current_text, final_raw_output_log, proceeded_with_flaws):
             logger.error(f"=== Finished Ch {chapter_number} WITH ERRORS during core finalization (DB/File save or embedding) ===")
             return None 

        # update_all_knowledge_bases_logic now uses unified extraction with full text.
        await update_all_knowledge_bases_logic(self, chapter_number, current_text, proceeded_with_flaws)
        
        self.chapter_count = max(self.chapter_count, chapter_number) 
        await self._save_all_json_state() 
        
        status_message = "Successfully" if not proceeded_with_flaws else "With Flaws (Accepted after failed revision or due to issues)"
        logger.info(f"=== Finished Ch {chapter_number} {status_message} ===")
        return current_text

    async def _finalize_chapter_core(self, chapter_number: int, final_text: str, raw_llm_log_for_db: str, from_flawed_draft: bool) -> bool:
        logger.info(f"Finalizing chapter {chapter_number} (From flawed draft: {from_flawed_draft}). Text length: {len(final_text)}.")
        if not final_text:
            logger.error(f"Cannot finalize ch {chapter_number}: Final text is missing or empty.")
            return False

        # summarize_chapter_text_logic now takes full text for its summarization LLM call
        summary_task = summarize_chapter_text_logic(final_text, chapter_number)
        embedding_task = asyncio.create_task(llm_interface.async_get_embedding(final_text))
        
        summary, final_embedding = await asyncio.gather(summary_task, embedding_task)

        if final_embedding is None:
            logger.error(f"CRITICAL: Failed to generate embedding for final text of Chapter {chapter_number}. Finalization cannot proceed.")
            return False

        try:
            await state_manager.async_save_chapter_data(
                chapter_number, final_text, raw_llm_log_for_db, summary, final_embedding, from_flawed_draft
            )
        except Exception as e: 
            logger.error(f"ORM save failed for chapter {chapter_number}: {e}", exc_info=True)
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
        chapter_file_path = os.path.join(config.CHAPTERS_DIR, f"chapter_{chapter_number:04d}.txt") 
        log_file_path = os.path.join(config.CHAPTER_LOGS_DIR, f"chapter_{chapter_number:04d}_raw_llm_log.txt")

        os.makedirs(os.path.dirname(chapter_file_path), exist_ok=True)
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        with open(chapter_file_path, 'w', encoding='utf-8') as f:
            f.write(final_text)
        with open(log_file_path, 'w', encoding='utf-8') as f:
            f.write(raw_llm_log)

    async def _save_debug_output(self, chapter_number: int, stage_description: str, content: Any):
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
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True) 
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content_str)
            logger.debug(f"Saved debug output for Ch {chapter_number}, Stage '{stage_description}' to {file_path}")
        except IOError as e:
            logger.error(f"IOError saving debug output for Ch {chapter_number}, Stage '{stage_description}' to {file_path}: {e}", exc_info=True)
        except Exception as e: 
            logger.error(f"Unexpected error saving debug output for Ch {chapter_number}, Stage '{stage_description}' to {file_path}: {e}", exc_info=True)