# nana_orchestrator.py
import logging
import os
import random
import asyncio
from typing import Dict, Any, Optional, List, Tuple

import config
import llm_interface
from state_manager import state_manager
from type import EvaluationResult, SceneDetail, ProblemDetail, AgentStateData

# Import Agent classes
from comprehensive_evaluator_agent import ComprehensiveEvaluatorAgent
from planner_agent import PlannerAgent
from drafting_agent import DraftingAgent
from kg_maintainer_agent import KGMaintainerAgent
from world_continuity_agent import WorldContinuityAgent

# Import logic for initial setup and context generation
from initial_setup_logic import generate_plot_outline_logic, generate_world_building_logic
from context_generation_logic import generate_hybrid_chapter_context_logic
from chapter_revision_logic import revise_chapter_draft_logic # Revision still uses this logic

logger = logging.getLogger(__name__)

class NANA_Orchestrator:
    def __init__(self):
        logger.info("Initializing NANA Orchestrator...")
        # Agent instances
        self.planner_agent = PlannerAgent()
        self.drafting_agent = DraftingAgent()
        self.evaluator_agent = ComprehensiveEvaluatorAgent() # Using the comprehensive one for now
        self.world_continuity_agent = WorldContinuityAgent()
        self.kg_maintainer_agent = KGMaintainerAgent()

        # Core novel state (mirrors NovelWriterAgent's attributes)
        self.plot_outline: Dict[str, Any] = {}
        self.character_profiles: Dict[str, Any] = {} # This will be updated by KGMaintainerAgent
        self.world_building: Dict[str, Any] = {}   # This will be updated by KGMaintainerAgent
        self.chapter_count: int = 0
        # novel_props will be a dictionary to pass to agents, containing the above 3 + title etc.
        self.novel_props_cache: Dict[str, Any] = {}
        logger.info("NANA Orchestrator initialized.")

    def _update_novel_props_cache(self):
        """Updates the convenience cache of novel properties."""
        self.novel_props_cache = {
            "title": self.plot_outline.get("title", config.DEFAULT_PLOT_OUTLINE_TITLE),
            "genre": self.plot_outline.get("genre", config.CONFIGURED_GENRE),
            "theme": self.plot_outline.get("theme", config.CONFIGURED_THEME),
            "protagonist_name": self.plot_outline.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME),
            "character_arc": self.plot_outline.get("character_arc", "N/A"),
            "logline": self.plot_outline.get("logline", "N/A"),
            "plot_points": self.plot_outline.get("plot_points", []),
            # These are critical: agents need current character_profiles and world_building state
            "character_profiles": self.character_profiles,
            "world_building": self.world_building,
            "plot_outline_full": self.plot_outline # For agents that need the whole outline
        }
        # The prompt_data_getters take an 'agent' like object.
        # They will need to be adapted to take this novel_props_cache or
        # the specific sub-dictionaries (like self.character_profiles).
        # For now, they expect an object with .character_profiles, .world_building, .plot_outline
        # So, we can pass `self` (the orchestrator instance) to them.

    async def async_init_orchestrator(self):
        """Loads initial state from Neo4j."""
        logger.info("NANA Orchestrator async_init_orchestrator started...")
        self.chapter_count = await state_manager.async_load_chapter_count()
        logger.info(f"Loaded chapter count from Neo4j: {self.chapter_count}")

        load_tasks = {
            "plot": state_manager.get_plot_outline(),
            "chars": state_manager.get_character_profiles(),
            "world": state_manager.get_world_building()
        }
        results = await asyncio.gather(*load_tasks.values(), return_exceptions=True)
        loaded_data = dict(zip(load_tasks.keys(), results))

        for key, value in loaded_data.items():
            if isinstance(value, Exception):
                logger.error(f"Error loading {key} during orchestrator init: {value}", exc_info=value)
                if key == "plot": self.plot_outline = {}
                elif key == "chars": self.character_profiles = {}
                elif key == "world": self.world_building = {}
            else:
                if key == "plot": self.plot_outline = value if isinstance(value, dict) else {}
                elif key == "chars": self.character_profiles = value if isinstance(value, dict) else {}
                elif key == "world": self.world_building = value if isinstance(value, dict) else {}
        self._update_novel_props_cache()
        logger.info("NANA Orchestrator async_init_orchestrator complete.")

    async def _save_core_novel_state_to_neo4j(self):
        logger.info("NANA: Saving core novel state (plot, characters, world) to Neo4j...")
        # The KGMaintainerAgent now handles updating character_profiles and world_building in memory.
        # So, we save what's in the orchestrator's attributes.
        tasks = [
            state_manager.save_plot_outline(self.plot_outline),
            state_manager.save_character_profiles(self.character_profiles),
            state_manager.save_world_building(self.world_building)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        success_count = 0
        for i, res in enumerate(results):
            item_name = ["plot_outline", "character_profiles", "world_building"][i]
            if isinstance(res, Exception): logger.error(f"Failed to save {item_name} to Neo4j: {res}", exc_info=res)
            elif res is True: success_count += 1
            else: logger.warning(f"Unexpected return value from save_{item_name}: {res}")
        if success_count == len(tasks): logger.info("All core state objects saved to Neo4j successfully.")
        else: logger.warning(f"Only {success_count}/{len(tasks)} core state objects saved successfully.")


    async def perform_initial_setup(self):
        """Handles the initial generation of plot, world, and characters if not already present."""
        logger.info("NANA performing initial setup...")
        print("\n--- NANA: Initializing Plot, Characters, and World ---")

        generation_params: Dict[str, Any] = {}
        if config.UNHINGED_PLOT_MODE and not os.path.exists(config.USER_STORY_ELEMENTS_FILE_PATH):
            generation_params.update({
                "genre": random.choice(config.UNHINGED_GENRES), "theme": random.choice(config.UNHINGED_THEMES),
                "setting_archetype": random.choice(config.UNHINGED_SETTINGS_ARCHETYPES),
                "protagonist_archetype": random.choice(config.UNHINGED_PROTAGONIST_ARCHETYPES),
                "conflict_archetype": random.choice(config.UNHINGED_CONFLICT_TYPES)
            })
        elif not os.path.exists(config.USER_STORY_ELEMENTS_FILE_PATH):
            generation_params.update({
                "genre": config.CONFIGURED_GENRE, "theme": config.CONFIGURED_THEME,
                "setting_description": config.CONFIGURED_SETTING_DESCRIPTION
            })

        # generate_plot_outline_logic and generate_world_building_logic expect an 'agent-like' object
        # to update self.plot_outline, self.character_profiles, self.world_building
        # Here, 'self' (the orchestrator) acts as that agent-like object for these setup functions.
        await generate_plot_outline_logic(self, config.DEFAULT_PROTAGONIST_NAME,
                                          config.UNHINGED_PLOT_MODE if not os.path.exists(config.USER_STORY_ELEMENTS_FILE_PATH) else False,
                                          **generation_params)
        plot_source = self.plot_outline.get("source", "unknown")
        print(f"   Plot Outline initialized/loaded (source: {plot_source}). Title: '{self.plot_outline.get('title', 'N/A')}'")

        await generate_world_building_logic(self)
        world_source = self.world_building.get("source", "unknown")
        print(f"   World Building initialized/loaded (source: {world_source}).")

        self._update_novel_props_cache() # Update cache after initial setup
        await self._save_core_novel_state_to_neo4j()
        print("   Initial plot, character, and world data saved to Neo4j.")

        if not self.plot_outline or self.plot_outline.get("is_default"):
            logger.warning("Initial setup resulted in a default or empty plot outline. This might impact generation quality.")
        return True

    async def _prepopulate_kg_if_needed(self):
        logger.info("NANA: Checking if KG pre-population is needed...")
        # This logic is similar to main.py's version
        plot_source = self.plot_outline.get("source", "")
        is_user_or_llm_plot = plot_source == "user_supplied" or plot_source.startswith("llm_generated")
        if not is_user_or_llm_plot:
            logger.info(f"Skipping KG pre-population: Plot outline is default or source is unclear ('{plot_source}').")
            return

        # Check if already pre-populated (simplified check)
        pp_check_query = f"MATCH (ni:NovelInfo {{id: '{config.MAIN_NOVEL_INFO_NODE_ID}'}})-[:HAS_PLOT_POINT]->(:PlotPoint) RETURN count(*) AS pp_count"
        pp_result = await state_manager._execute_read_query(pp_check_query) # Direct use for this specific check
        if pp_result and pp_result[0] and pp_result[0]['pp_count'] > 0:
            logger.info("Found existing NovelInfo with plot points. Assuming KG already pre-populated. Skipping explicit pre-population.")
            return

        print("\n--- NANA: Pre-populating Knowledge Graph from Initial Data ---")
        await self.kg_maintainer_agent.prepopulate_kg_from_initial_data(
            self.plot_outline, self.character_profiles, self.world_building
        )
        print("   Knowledge Graph pre-population step complete.")

    def _get_plot_point_info_for_chapter(self, chapter_number: int) -> Tuple[Optional[str], int]:
        """Helper to get plot point focus and index for a chapter."""
        plot_points = self.plot_outline.get("plot_points", [])
        if not isinstance(plot_points, list) or not plot_points: return None, -1
        if chapter_number <= 0: return None, -1
        plot_point_index = min(chapter_number - 1, len(plot_points) - 1)
        if 0 <= plot_point_index < len(plot_points):
            plot_point = plot_points[plot_point_index]
            return str(plot_point) if plot_point is not None else None, plot_point_index
        return None, -1

    async def _save_chapter_text_and_log(self, chapter_number: int, final_text: str, raw_llm_log: Optional[str]):
        """Saves chapter text and raw LLM log to files."""
        # This is synchronous I/O, run in executor
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, self._save_chapter_files_sync_io, chapter_number, final_text, raw_llm_log or "N/A")
            logger.info(f"Saved chapter text and raw LLM log files for ch {chapter_number}.")
        except IOError as e:
            logger.error(f"Failed writing chapter text/log files for ch {chapter_number}: {e}", exc_info=True)

    def _save_chapter_files_sync_io(self, chapter_number: int, final_text: str, raw_llm_log: str):
        chapter_file_path = os.path.join(config.CHAPTERS_DIR, f"chapter_{chapter_number:04d}.txt")
        log_file_path = os.path.join(config.CHAPTER_LOGS_DIR, f"chapter_{chapter_number:04d}_raw_llm_log.txt")
        os.makedirs(os.path.dirname(chapter_file_path), exist_ok=True)
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        with open(chapter_file_path, 'w', encoding='utf-8') as f: f.write(final_text)
        with open(log_file_path, 'w', encoding='utf-8') as f: f.write(raw_llm_log)

    async def _save_debug_output(self, chapter_number: int, stage_description: str, content: Any):
        """Saves debug output content to a file."""
        # This is synchronous I/O, run in executor
        if content is None: return
        content_str = str(content) if not isinstance(content, str) else content
        if not content_str.strip(): return
        loop = asyncio.get_event_loop()
        try:
            safe_stage_desc = "".join(c if c.isalnum() or c in ['_', '-'] else "_" for c in stage_description)
            file_name = f"chapter_{chapter_number:04d}_{safe_stage_desc}.txt"
            file_path = os.path.join(config.DEBUG_OUTPUTS_DIR, file_name)
            await loop.run_in_executor(None, self._save_debug_output_sync_io, file_path, content_str)
            logger.debug(f"Saved debug output for Ch {chapter_number}, Stage '{stage_description}' to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save debug output (Ch {chapter_number}, Stage '{stage_description}'): {e}", exc_info=True)

    def _save_debug_output_sync_io(self, file_path: str, content_str: str):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f: f.write(content_str)


    async def run_chapter_generation_process(self, chapter_number: int) -> Optional[str]:
        """Manages the generation of a single chapter through the agent swarm."""
        logger.info(f"=== NANA: Starting Chapter {chapter_number} Generation ===")
        if not (self.plot_outline and self.plot_outline.get("plot_points") and self.plot_outline.get("protagonist_name")):
            logger.error(f"NANA: Cannot write Ch {chapter_number}: Plot outline, plot points, or protagonist name missing.")
            return None

        plot_point_focus, plot_point_index = self._get_plot_point_info_for_chapter(chapter_number)
        if plot_point_focus is None:
            logger.error(f"NANA: Ch {chapter_number} generation halted: no plot point focus.")
            return None

        # 0. (Optional) World Continuity Agent pre-check (not implemented for pre-draft)
        # This agent could analyze the plan against existing KG to flag potential issues *before* drafting.

        # 1. Planner Agent: Generate scene plan
        # The planner needs novel_props, chapter_number, plot_point_focus, plot_point_index
        chapter_plan: Optional[List[SceneDetail]] = await self.planner_agent.plan_chapter_scenes(
            self.novel_props_cache, chapter_number, plot_point_focus, plot_point_index
        )
        if config.ENABLE_AGENTIC_PLANNING and chapter_plan is None:
            logger.warning(f"NANA: Ch {chapter_number}: Planning Agent failed. Proceeding with plot point focus only.")

        # 2. Generate Hybrid Context (using orchestrator's current state for prompt_data_getters)
        # The context_generation_logic uses an 'agent-like' object for its getters. `self` can serve this role.
        hybrid_context_for_draft = await generate_hybrid_chapter_context_logic(self, chapter_number, chapter_plan)

        # 3. Drafting Agent: Generate initial draft
        initial_draft_text, initial_raw_llm_text = await self.drafting_agent.draft_chapter(
            self.novel_props_cache, chapter_number, plot_point_focus, hybrid_context_for_draft, chapter_plan
        )
        if not initial_draft_text:
            logger.error(f"NANA: Drafting Agent failed for Ch {chapter_number}.")
            await self._save_debug_output(chapter_number, "initial_draft_fail_raw_llm", initial_raw_llm_text or "None")
            return None
        
        current_text_to_process = initial_draft_text
        current_raw_llm_output = initial_raw_llm_text
        is_from_flawed_source = False # Tracks if the text comes from a draft that failed evaluation

        # 4. Evaluation Loop (Simplified: one round of comprehensive eval + revision)
        max_revision_attempts = 1 # Could be configurable
        for attempt in range(max_revision_attempts + 1):
            # 4a. Comprehensive Evaluator Agent
            evaluation_result: EvaluationResult = await self.evaluator_agent.evaluate_chapter_draft(
                self.novel_props_cache, current_text_to_process, chapter_number,
                plot_point_focus, plot_point_index, hybrid_context_for_draft
            )

            # 4b. (Optional) World Continuity Agent check on current draft
            continuity_problems: List[ProblemDetail] = await self.world_continuity_agent.check_consistency(
                 self.novel_props_cache, current_text_to_process, chapter_number, hybrid_context_for_draft
            )
            if continuity_problems:
                logger.warning(f"NANA: Ch {chapter_number} - World Continuity Agent found {len(continuity_problems)} issues.")
                evaluation_result["problems_found"].extend(continuity_problems)
                if not evaluation_result["needs_revision"]: # If comprehensive eval was ok, but continuity found issues
                    evaluation_result["needs_revision"] = True
                    evaluation_result["reasons"].append("Continuity issues identified.")
                # Deduplicate reasons if necessary (not done here for simplicity)

            if not evaluation_result["needs_revision"]:
                logger.info(f"NANA: Ch {chapter_number} draft passed evaluation (Attempt {attempt+1}).")
                is_from_flawed_source = False
                break # Exit revision loop
            else: # Needs revision
                is_from_flawed_source = True # Mark as flawed if it needed revision
                logger.warning(f"NANA: Ch {chapter_number} draft (Attempt {attempt+1}) needs revision. Reasons: {'; '.join(evaluation_result['reasons'])}")
                if attempt >= max_revision_attempts:
                    logger.error(f"NANA: Ch {chapter_number} - Max revision attempts reached. Proceeding with current (flawed) draft.")
                    break

                # 4c. Revision (using chapter_revision_logic directly for now)
                # A dedicated RevisionAgent could encapsulate this.
                # The revision logic needs an 'agent-like' object for _get_plot_point_info.
                # For now, it can use `self` from orchestrator.
                revised_text_tuple = await revise_chapter_draft_logic(
                    self, current_text_to_process, chapter_number,
                    evaluation_result, hybrid_context_for_draft, chapter_plan
                )
                if revised_text_tuple and revised_text_tuple[0]:
                    current_text_to_process, rev_raw_output = revised_text_tuple
                    current_raw_llm_output = rev_raw_output if rev_raw_output else current_raw_llm_output
                    logger.info(f"NANA: Ch {chapter_number} - Revision attempt {attempt+1} successful. Re-evaluating.")
                else:
                    logger.error(f"NANA: Ch {chapter_number} - Revision attempt {attempt+1} failed. Proceeding with previous draft.")
                    break # Exit revision loop, use current (flawed) draft
        # End of evaluation/revision loop

        if len(current_text_to_process) < config.MIN_ACCEPTABLE_DRAFT_LENGTH:
             logger.warning(f"NANA: Final chosen text for Ch {chapter_number} is too short ({len(current_text_to_process)} chars). Marked as flawed.")
             is_from_flawed_source = True

        # 5. Finalize Chapter in Neo4j (text, summary, embedding)
        chapter_summary = await self.kg_maintainer_agent.summarize_chapter(current_text_to_process, chapter_number)
        final_embedding = await llm_interface.async_get_embedding(current_text_to_process)

        if final_embedding is None:
            logger.error(f"NANA CRITICAL: Failed to generate embedding for final text of Chapter {chapter_number}. Cannot save chapter.")
            return None

        await state_manager.async_save_chapter_data(
            chapter_number, current_text_to_process, current_raw_llm_output or "N/A",
            chapter_summary, final_embedding, is_from_flawed_source
        )
        await self._save_chapter_text_and_log(chapter_number, current_text_to_process, current_raw_llm_output)

        # 6. KG Maintainer Agent: Extract and merge knowledge from the FINAL chapter text
        # This modifies the orchestrator's self.character_profiles and self.world_building
        await self.kg_maintainer_agent.extract_and_merge_knowledge(
            self.novel_props_cache, # Pass the dict that holds mutable character_profiles and world_building
            chapter_number,
            current_text_to_process,
            is_from_flawed_source
        )
        # Update orchestrator's direct attributes from the modified cache
        self.character_profiles = self.novel_props_cache['character_profiles']
        self.world_building = self.novel_props_cache['world_building']


        # 7. Save updated agent state (plot, characters, world) from orchestrator memory to Neo4j
        self.chapter_count = max(self.chapter_count, chapter_number)
        await self._save_core_novel_state_to_neo4j() # Saves the now-updated self.character_profiles etc.

        status_message = "Successfully" if not is_from_flawed_source else "With Flaws"
        logger.info(f"=== NANA: Finished Ch {chapter_number} {status_message} ===")
        return current_text_to_process


    async def run_novel_generation_loop(self):
        """Main loop for orchestrating novel generation."""
        logger.info("--- NANA: Starting Novel Generation Run ---")
        try:
            await state_manager.connect()
            await state_manager.create_db_and_tables()
            logger.info("NANA: State_manager initialized and Neo4j connection/schema verified.")

            await self.async_init_orchestrator() # Load existing state

            if not self.plot_outline: # Check if initial setup is truly needed
                if not await self.perform_initial_setup():
                    logger.critical("NANA: Initial setup failed. Halting.")
                    return
            await self._prepopulate_kg_if_needed()
            self._update_novel_props_cache() # Ensure cache is up-to-date after setup/load

            print("\n--- NANA: Starting Novel Writing Process ---")
            start_chapter = self.chapter_count + 1
            end_chapter = start_chapter + config.CHAPTERS_PER_RUN if config.CHAPTERS_PER_RUN > 0 else start_chapter

            print(f"NANA: Current Chapter Count (at start of run): {self.chapter_count}")
            if start_chapter < end_chapter:
                print(f"NANA: Targeting Chapters: {start_chapter} to {end_chapter - 1} in this run.")
            else:
                print(f"NANA: CHAPTERS_PER_RUN ({config.CHAPTERS_PER_RUN}) results in no new chapters. Current: {self.chapter_count}.")

            chapters_successfully_written = 0
            for i in range(start_chapter, end_chapter):
                print(f"\n--- NANA: Attempting Chapter {i} ---")
                try:
                    chapter_text = await self.run_chapter_generation_process(i)
                    if chapter_text:
                        chapters_successfully_written += 1
                        print(f"NANA: Chapter {i}: Successfully generated (Length: {len(chapter_text)} chars).")
                        print(f"   Snippet: {chapter_text[:200].replace(chr(10), ' ')}...")
                    else:
                        print(f"NANA: Chapter {i}: Failed to generate or save. Check logs.")
                        # Decide if to break or continue on chapter failure
                        # break
                except Exception as e:
                    logger.critical(f"NANA: Critical error during chapter {i} writing process: {e}", exc_info=True)
                    break

            print(f"\n--- NANA: Novel writing process finished for this run ---")
            final_chapter_count_from_db = await state_manager.async_load_chapter_count()
            print(f"NANA: Successfully wrote {chapters_successfully_written} chapter(s).")
            print(f"NANA: Current total chapters in database: {final_chapter_count_from_db}")
            logger.info(f"--- NANA: Saga Novel Generation Run Finished. Final Neo4j chapter count: {final_chapter_count_from_db} ---")

        except Exception as e:
            logger.critical(f"NANA: Unhandled exception in orchestrator main loop: {e}", exc_info=True)
        finally:
            await state_manager.close()
            logger.info("NANA: Neo4j driver successfully closed on application exit.")


def setup_logging_nana(): # Renamed to avoid conflict if main.py still exists
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT
    )
    if config.LOG_FILE:
        try:
            log_dir = os.path.dirname(config.LOG_FILE)
            if log_dir: os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(config.LOG_FILE, maxBytes=10**6, backupCount=5, mode='a', encoding='utf-8')
            file_handler.setLevel(config.LOG_LEVEL)
            formatter = logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT)
            file_handler.setFormatter(formatter)
            logging.getLogger().addHandler(file_handler)
            logging.info(f"File logging enabled. Log file: {config.LOG_FILE}")
        except Exception as e:
            logging.error(f"Failed to configure file logging to {config.LOG_FILE}: {e}", exc_info=True)
            # No sys.exit here, orchestrator will handle failures
    logging.info(f"NANA Logging setup complete. Level: {logging.getLevelName(config.LOG_LEVEL)}")


if __name__ == "__main__":
    setup_logging_nana()
    orchestrator = NANA_Orchestrator()
    try:
        asyncio.run(orchestrator.run_novel_generation_loop())
    except KeyboardInterrupt:
        logger.info("NANA Orchestrator shutting down gracefully...")
    except Exception as main_err:
        logger.critical(f"NANA Orchestrator unhandled main exception: {main_err}", exc_info=True)
        # Consider sys.exit(1) here if this is the main entry point