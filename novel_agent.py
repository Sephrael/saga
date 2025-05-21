# novel_agent.py
import os
import json
import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any 

import config
import llm_interface
from state_manager import state_manager 
from type import JsonStateData, EvaluationResult, SceneDetail, ProblemDetail 

from initial_setup_logic import generate_plot_outline_logic, generate_world_building_logic
from chapter_planning_logic import plan_chapter_scenes_logic 
from chapter_drafting_logic import generate_chapter_draft_logic
from chapter_evaluation_logic import evaluate_chapter_draft_logic 
from chapter_revision_logic import revise_chapter_draft_logic # Updated name to revise_chapter_draft_logic
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
        """Asynchronously initializes agent state by loading from Neo4j."""
        logger.info("NovelWriterAgent async_init started...")
        
        self.chapter_count = await state_manager.async_load_chapter_count()
        logger.info(f"Loaded chapter count from Neo4j: {self.chapter_count}")

        # Load decomposed structures. state_manager now handles reassembly into dicts.
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
                # Ensure defaults are set if loading fails
                if key == "plot": self.plot_outline = {}
                elif key == "chars": self.character_profiles = {}
                elif key == "world": self.world_building = {}
            else:
                if key == "plot": self.plot_outline = value if isinstance(value, dict) else {}
                elif key == "chars": self.character_profiles = value if isinstance(value, dict) else {}
                elif key == "world": self.world_building = value if isinstance(value, dict) else {}
        
        if not self.plot_outline:
            logger.warning("Plot outline is empty after Neo4j load. Expected during initial setup or if save failed previously.")
        if not self.character_profiles:
            logger.warning("Character profiles are empty after Neo4j load. Expected during initial setup or if save failed previously.")
        if not self.world_building:
            logger.warning("World building is empty after Neo4j load. Expected during initial setup or if save failed previously.")
        
        logger.info("NovelWriterAgent async_init complete.")

    async def _save_all_json_state(self):
        logger.info("Saving agent state (plot, characters, world) to Neo4j (decomposed)...")
        # state_manager.save_* methods now handle decomposition.
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
                logger.error(f"Failed to save {item_name} to Neo4j (decomposed): {res}", exc_info=res)
            elif res is True: 
                success_count += 1
                logger.info(f"Successfully saved {item_name} to Neo4j (decomposed).")
            else: 
                logger.warning(f"Unexpected return value from save_{item_name} (decomposed): {res}")

        if success_count == len(tasks):
            logger.info(f"All ({success_count}) state objects saved to Neo4j (decomposed) successfully.")
        else:
            logger.warning(f"Only {success_count}/{len(tasks)} state objects saved to Neo4j (decomposed) successfully.")

    async def generate_plot_outline(self, default_protagonist_name: str, unhinged_mode: bool, **kwargs) -> Dict[str, Any]:
        # Logic to generate self.plot_outline as a Python dict remains the same.
        # The change is in how _save_all_json_state (called within or after) persists it.
        generated_outline = await generate_plot_outline_logic(self, default_protagonist_name, unhinged_mode, **kwargs)
        # self.plot_outline is updated by generate_plot_outline_logic
        # It will also call _save_all_json_state internally if it modifies the agent's state.
        return generated_outline


    async def generate_world_building(self) -> Dict[str, Any]:
        # Logic to generate self.world_building as a Python dict remains the same.
        generated_world_data = await generate_world_building_logic(self)
        # self.world_building is updated by generate_world_building_logic
        # It will also call _save_all_json_state internally.
        return generated_world_data

    async def _prepopulate_knowledge_graph(self):
        # This will now call the updated prepopulate_kg_from_initial_data_logic,
        # which takes the agent's Python dicts and creates fine-grained Neo4j data.
        await prepopulate_kg_from_initial_data_logic(self)

    def _get_plot_point_info(self, chapter_number: int) -> Tuple[Optional[str], int]:
        # This method still reads from self.plot_outline (Python dict)
        plot_points = self.plot_outline.get("plot_points", [])
        if not isinstance(plot_points, list) or not plot_points:
            logger.warning(f"No plot points defined in plot outline for chapter {chapter_number}.")
            return None, -1
        
        if chapter_number <= 0: 
            logger.warning(f"Invalid chapter number {chapter_number} for plot point lookup.")
            return None, -1
        
        plot_point_index = min(chapter_number - 1, len(plot_points) - 1)
        
        if 0 <= plot_point_index < len(plot_points):
            plot_point = plot_points[plot_point_index]
            if isinstance(plot_point, str): 
                return plot_point, plot_point_index
            else:
                logger.warning(f"Plot point at index {plot_point_index} for chapter {chapter_number} is not a string: {type(plot_point)}")
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

        hybrid_context_for_draft = await generate_hybrid_chapter_context_logic(self, chapter_number, chapter_plan) 
        
        plot_point_focus, _ = self._get_plot_point_info(chapter_number)
        if plot_point_focus is None: 
            logger.error(f"Ch {chapter_number} generation halted: no plot point focus.")
            return None

        initial_draft_text, initial_raw_llm_text = await generate_chapter_draft_logic(
            self, chapter_number, plot_point_focus, hybrid_context_for_draft, chapter_plan
        )

        if not initial_draft_text or not initial_raw_llm_text: 
            logger.error(f"Failed to generate initial draft for ch {chapter_number}.")
            await self._save_debug_output(chapter_number, "initial_draft_fail_raw_llm", initial_raw_llm_text or "Initial draft LLM output was None")
            return None
        
        current_text_to_process = initial_draft_text
        current_raw_llm_output = initial_raw_llm_text
        is_from_flawed_source = False 

        # Initial evaluation
        evaluation = await evaluate_chapter_draft_logic(self, current_text_to_process, chapter_number, hybrid_context_for_draft)

        if evaluation["needs_revision"]:
            logger.warning(f"Ch {chapter_number} initial draft ({len(current_text_to_process)} chars) flagged for revision. Reasons: {'; '.join(evaluation['reasons'])}")
            
            revised_text_tuple = await revise_chapter_draft_logic(
                agent=self, 
                original_text=current_text_to_process, 
                chapter_number=chapter_number, 
                evaluation_result=evaluation, 
                hybrid_context_for_revision=hybrid_context_for_draft, 
                chapter_plan=chapter_plan
            )

            if revised_text_tuple:
                revised_text, raw_revision_llm_output = revised_text_tuple
                logger.info(f"Revision attempted for ch {chapter_number}. Re-evaluating revised draft ({len(revised_text)} chars)...")
                
                revised_evaluation = await evaluate_chapter_draft_logic(self, revised_text, chapter_number, hybrid_context_for_draft)
                
                if revised_evaluation["needs_revision"]:
                    logger.error(f"Revised draft for ch {chapter_number} STILL FAILED evaluation. Reasons: {'; '.join(revised_evaluation['reasons'])}")
                    if len(current_text_to_process) >= len(revised_text) and len(current_text_to_process) >= config.MIN_ACCEPTABLE_DRAFT_LENGTH // 2:
                        logger.warning(f"Reverting to original draft ({len(current_text_to_process)} chars) as revised draft ({len(revised_text)} chars) also failed evaluation and original is longer/better.")
                        is_from_flawed_source = True 
                    else:
                        logger.warning(f"Proceeding with revised draft ({len(revised_text)} chars) despite failing re-evaluation, as it's preferred over original ({len(current_text_to_process)} chars) or original was too short.")
                        current_text_to_process = revised_text
                        current_raw_llm_output = raw_revision_llm_output
                        is_from_flawed_source = True 
                else: 
                    current_text_to_process = revised_text
                    current_raw_llm_output = raw_revision_llm_output
                    logger.info(f"Revised draft for ch {chapter_number} passed re-evaluation.")
                    is_from_flawed_source = False 
            else: 
                logger.error(f"Revision attempt failed for ch {chapter_number} (no text produced by revision logic). Proceeding with the original (flawed) draft.")
                is_from_flawed_source = True 
        else:
            logger.info(f"Initial draft for ch {chapter_number} passed evaluation.")
            is_from_flawed_source = False

        if len(current_text_to_process) < config.MIN_ACCEPTABLE_DRAFT_LENGTH:
             logger.warning(f"Final chosen text for Ch {chapter_number} is too short ({len(current_text_to_process)} chars). Will be marked as 'from_flawed_draft' for knowledge updates.")
             is_from_flawed_source = True


        if not await self._finalize_chapter_core(chapter_number, current_text_to_process, current_raw_llm_output, is_from_flawed_source):
             logger.error(f"=== Finished Ch {chapter_number} WITH ERRORS during core finalization (Neo4j/File save or embedding) ===")
             return None 

        # Knowledge base updates will operate on the agent's Python dicts,
        # which are then saved (decomposed) by _save_all_json_state.
        await update_all_knowledge_bases_logic(self, chapter_number, current_text_to_process, is_from_flawed_source)
        
        self.chapter_count = max(self.chapter_count, chapter_number) 
        await self._save_all_json_state() # Persist the (potentially updated) dicts as decomposed graph data
        
        status_message = "Successfully" if not is_from_flawed_source else "With Flaws (Marked as flawed due to evaluation or length issues)"
        logger.info(f"=== Finished Ch {chapter_number} {status_message} ===")
        return current_text_to_process

    async def _finalize_chapter_core(self, chapter_number: int, final_text: str, raw_llm_log_for_db: str, from_flawed_draft: bool) -> bool:
        logger.info(f"Finalizing chapter {chapter_number} (From flawed draft: {from_flawed_draft}). Text length: {len(final_text)}.")
        if not final_text:
            logger.error(f"Cannot finalize ch {chapter_number}: Final text is missing or empty.")
            return False
        
        effective_raw_llm_log = raw_llm_log_for_db if raw_llm_log_for_db is not None else "Raw LLM output was not available for this version."

        summary_task = summarize_chapter_text_logic(final_text, chapter_number)
        embedding_task = asyncio.create_task(llm_interface.async_get_embedding(final_text))
        
        summary, final_embedding = await asyncio.gather(summary_task, embedding_task)

        if final_embedding is None:
            logger.error(f"CRITICAL: Failed to generate embedding for final text of Chapter {chapter_number}. Finalization cannot proceed.")
            return False

        try:
            await state_manager.async_save_chapter_data(
                chapter_number, final_text, effective_raw_llm_log, summary, final_embedding, from_flawed_draft
            )
        except Exception as e: 
            logger.error(f"Neo4j save failed for chapter {chapter_number}: {e}", exc_info=True)
            return False 

        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, self._save_chapter_text_files_sync, chapter_number, final_text, effective_raw_llm_log)
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

        try:
            with open(chapter_file_path, 'w', encoding='utf-8') as f:
                f.write(final_text)
            with open(log_file_path, 'w', encoding='utf-8') as f:
                f.write(raw_llm_log if raw_llm_log is not None else "Raw LLM log content was None.")
        except IOError as e:
            logger.error(f"IOError saving chapter text/log files for ch {chapter_number}: {e}", exc_info=True)


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