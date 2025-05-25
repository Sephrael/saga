# nana_orchestrator.py
import logging
import logging.handlers
import os
import random
import asyncio
import time # For Rich display updates
from typing import Dict, Any, Optional, List, Tuple

import config
import llm_interface
# from state_manager import state_manager # No longer used directly
from core_db.base_db_manager import neo4j_manager # Use the new manager
from data_access import ( # Import specific query functions
    plot_queries,
    character_queries,
    world_queries,
    chapter_queries,
    kg_queries
)
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
from chapter_revision_logic import revise_chapter_draft_logic 

# Rich imports for live display
try:
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text
    from rich.console import Group
    from rich.table import Table
    from rich.logging import RichHandler 
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # Fallback stubs if rich is not available
    class Live: 
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def update(self, *args, **kwargs): pass
        def stop(self): pass
    class Text: 
        def __init__(self, *args, **kwargs): self.plain = ""
    class Group: 
        def __init__(self, *args, **kwargs): pass
    class Panel: 
        def __init__(self, *args, **kwargs): pass
    class RichHandler: 
        def __init__(self, *args, **kwargs): pass


logger = logging.getLogger(__name__)

class NANA_Orchestrator:
    def __init__(self):
        logger.info("Initializing NANA Orchestrator...")
        # Agent instances
        self.planner_agent = PlannerAgent()
        self.drafting_agent = DraftingAgent()
        self.evaluator_agent = ComprehensiveEvaluatorAgent()
        self.world_continuity_agent = WorldContinuityAgent()
        self.kg_maintainer_agent = KGMaintainerAgent()

        # Core novel state
        self.plot_outline: Dict[str, Any] = {}
        self.character_profiles: Dict[str, Any] = {}
        self.world_building: Dict[str, Any] = {}
        self.chapter_count: int = 0

        self.novel_props_cache: Dict[str, Any] = {}
        self.total_tokens_generated_this_run: int = 0 

        # Rich display elements
        self.rich_live: Optional[Live] = None
        self.rich_status_group: Optional[Group] = None
        self.status_text_novel_title: Text = Text("Novel: N/A")
        self.status_text_current_chapter: Text = Text("Current Chapter: N/A")
        self.status_text_current_step: Text = Text("Current Step: Initializing...")
        self.status_text_tokens_generated: Text = Text("Tokens Generated (this run): 0")
        self.status_text_elapsed_time: Text = Text("Elapsed Time: 0s")
        self.run_start_time: float = 0.0

        if RICH_AVAILABLE and config.ENABLE_RICH_PROGRESS:
            self.rich_status_group = Group(
                self.status_text_novel_title,
                self.status_text_current_chapter,
                self.status_text_current_step,
                self.status_text_tokens_generated,
                self.status_text_elapsed_time
            )
            self.rich_live = Live(
                Panel(self.rich_status_group, title="SAGA NANA Progress", border_style="blue", expand=True), 
                refresh_per_second=config.RICH_REFRESH_PER_SECOND,
                transient=False,
                redirect_stdout=True, 
                redirect_stderr=True 
            )
        else:
            logger.info("Rich library not available or ENABLE_RICH_PROGRESS is False. Progress will be shown via standard logs.")

        logger.info("NANA Orchestrator initialized.")

    def _update_rich_display(self, chapter_num: Optional[int] = None, step: Optional[str] = None):
        if not (RICH_AVAILABLE and config.ENABLE_RICH_PROGRESS and self.rich_live):
            return

        if chapter_num is not None:
            self.status_text_current_chapter.plain = f"Current Chapter: {chapter_num}"
        if step is not None:
            self.status_text_current_step.plain = f"Current Step: {step}"
        
        self.status_text_novel_title.plain = f"Novel: {self.plot_outline.get('title', 'N/A')}"
        self.status_text_tokens_generated.plain = f"Tokens Generated (this run): {self.total_tokens_generated_this_run:,}" 
        
        elapsed_seconds = time.time() - self.run_start_time
        self.status_text_elapsed_time.plain = f"Elapsed Time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_seconds))}"


    def _accumulate_tokens(self, operation_name: str, usage_data: Optional[Dict[str, int]]):
        if usage_data and isinstance(usage_data.get("completion_tokens"), int):
            completed_tokens = usage_data["completion_tokens"]
            self.total_tokens_generated_this_run += completed_tokens
            if not (RICH_AVAILABLE and config.ENABLE_RICH_PROGRESS):
                 logger.info(
                    f"NANA Activity: Tokens from '{operation_name}': {completed_tokens}. "
                    f"Total generated this run: {self.total_tokens_generated_this_run}"
                )
            self._update_rich_display() 
        elif usage_data and isinstance(usage_data.get("total_tokens"), int) and not isinstance(usage_data.get("completion_tokens"), int):
            logger.warning(f"NANA Activity: '{operation_name}' - 'completion_tokens' missing or not int in usage_data. Tokens not added. Usage: {usage_data}")
        else:
            logger.debug(f"NANA Activity: '{operation_name}' - No valid usage_data or completion_tokens provided for token accumulation.")


    def _update_novel_props_cache(self):
        self.novel_props_cache = {
            "title": self.plot_outline.get("title", config.DEFAULT_PLOT_OUTLINE_TITLE),
            "genre": self.plot_outline.get("genre", config.CONFIGURED_GENRE),
            "theme": self.plot_outline.get("theme", config.CONFIGURED_THEME),
            "protagonist_name": self.plot_outline.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME),
            "character_arc": self.plot_outline.get("character_arc", "N/A"),
            "logline": self.plot_outline.get("logline", "N/A"),
            "plot_points": self.plot_outline.get("plot_points", []),
            "character_profiles": self.character_profiles,
            "world_building": self.world_building,
            "plot_outline_full": self.plot_outline
        }
        self._update_rich_display()


    async def async_init_orchestrator(self):
        logger.info("NANA Orchestrator async_init_orchestrator started...")
        self._update_rich_display(step="Initializing Orchestrator")
        self.chapter_count = await chapter_queries.load_chapter_count_from_db() # MODIFIED
        logger.info(f"Loaded chapter count from Neo4j: {self.chapter_count}")

        load_tasks = {
            "plot": plot_queries.get_plot_outline_from_db(), # MODIFIED
            "chars": character_queries.get_character_profiles_from_db(), # MODIFIED
            "world": world_queries.get_world_building_from_db() # MODIFIED
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
        self._update_rich_display(step="Orchestrator Initialized")


    async def _save_core_novel_state_to_neo4j(self):
        logger.info("NANA: Saving core novel state (plot, characters, world) to Neo4j...")
        tasks = [
            plot_queries.save_plot_outline_to_db(self.plot_outline), # MODIFIED
            character_queries.save_character_profiles_to_db(self.character_profiles), # MODIFIED
            world_queries.save_world_building_to_db(self.world_building) # MODIFIED
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
        self._update_rich_display(step="Performing Initial Setup")
        logger.info("NANA performing initial setup...")
        logger.info("\n--- NANA: Initializing Plot, Characters, and World ---")

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

        _, plot_usage = await generate_plot_outline_logic(self, config.DEFAULT_PROTAGONIST_NAME,
                                          config.UNHINGED_PLOT_MODE if not os.path.exists(config.USER_STORY_ELEMENTS_FILE_PATH) else False,
                                          **generation_params)
        self._accumulate_tokens("InitialSetup-PlotOutline", plot_usage)
        plot_source = self.plot_outline.get("source", "unknown")
        logger.info(f"   Plot Outline initialized/loaded (source: {plot_source}). Title: '{self.plot_outline.get('title', 'N/A')}'")
        self._update_rich_display(step="Plot Outline Generated")


        _, world_usage = await generate_world_building_logic(self)
        self._accumulate_tokens("InitialSetup-WorldBuilding", world_usage)
        world_source = self.world_building.get("source", "unknown")
        logger.info(f"   World Building initialized/loaded (source: {world_source}).")
        self._update_rich_display(step="World Building Generated")


        self._update_novel_props_cache() 
        await self._save_core_novel_state_to_neo4j()
        logger.info("   Initial plot, character, and world data saved to Neo4j.")
        self._update_rich_display(step="Initial State Saved")


        if not self.plot_outline or self.plot_outline.get("is_default"):
            logger.warning("Initial setup resulted in a default or empty plot outline. This might impact generation quality.")
        return True

    async def _prepopulate_kg_if_needed(self):
        self._update_rich_display(step="Pre-populating KG (if needed)")
        logger.info("NANA: Checking if KG pre-population is needed...")
        plot_source = self.plot_outline.get("source", "")
        is_user_or_llm_plot = plot_source == "user_supplied" or plot_source.startswith("llm_generated")
        if not is_user_or_llm_plot:
            logger.info(f"Skipping KG pre-population: Plot outline is default or source is unclear ('{plot_source}').")
            return

        pp_check_query = f"MATCH (ni:NovelInfo {{id: '{config.MAIN_NOVEL_INFO_NODE_ID}'}})-[:HAS_PLOT_POINT]->(:PlotPoint) RETURN count(*) AS pp_count"
        pp_result = await neo4j_manager.execute_read_query(pp_check_query) # MODIFIED
        if pp_result and pp_result[0] and pp_result[0]['pp_count'] > 0:
            logger.info("Found existing NovelInfo with plot points. Assuming KG already pre-populated. Skipping explicit pre-population.")
            return

        logger.info("\n--- NANA: Pre-populating Knowledge Graph from Initial Data ---")
        await self.kg_maintainer_agent.prepopulate_kg_from_initial_data(
            self.plot_outline, self.character_profiles, self.world_building
        )
        logger.info("   Knowledge Graph pre-population step complete.")
        self._update_rich_display(step="KG Pre-population Complete")


    def _get_plot_point_info_for_chapter(self, chapter_number: int) -> Tuple[Optional[str], int]:
        plot_points = self.plot_outline.get("plot_points", [])
        if not isinstance(plot_points, list) or not plot_points: return None, -1
        if chapter_number <= 0: return None, -1
        plot_point_index = min(chapter_number - 1, len(plot_points) - 1)
        if 0 <= plot_point_index < len(plot_points):
            plot_point = plot_points[plot_point_index]
            return str(plot_point) if plot_point is not None else None, plot_point_index
        return None, -1

    async def _save_chapter_text_and_log(self, chapter_number: int, final_text: str, raw_llm_log: Optional[str]):
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
        logger.info(f"=== NANA: Starting Chapter {chapter_number} Generation ===")
        self._update_rich_display(chapter_num=chapter_number, step="Starting Chapter")
        if not (self.plot_outline and self.plot_outline.get("plot_points") and self.plot_outline.get("protagonist_name")):
            logger.error(f"NANA: Cannot write Ch {chapter_number}: Plot outline, plot points, or protagonist name missing from orchestrator state.")
            self._update_rich_display(step=f"Ch {chapter_number} Failed - Missing Plot Outline")
            return None

        plot_point_focus, plot_point_index = self._get_plot_point_info_for_chapter(chapter_number)
        if plot_point_focus is None:
            logger.error(f"NANA: Ch {chapter_number} generation halted: no plot point focus could be determined.")
            self._update_rich_display(step=f"Ch {chapter_number} Failed - No Plot Point Focus")
            return None

        self._update_rich_display(step=f"Ch {chapter_number} - Planning Scenes")
        chapter_plan_result, plan_usage = await self.planner_agent.plan_chapter_scenes(
            self.novel_props_cache, chapter_number, plot_point_focus, plot_point_index
        )
        self._accumulate_tokens(f"Ch{chapter_number}-Planning", plan_usage)
        chapter_plan: Optional[List[SceneDetail]] = chapter_plan_result
        if config.ENABLE_AGENTIC_PLANNING and chapter_plan is None:
            logger.warning(f"NANA: Ch {chapter_number}: Planning Agent failed or planning disabled. Proceeding with plot point focus only for drafting.")
        await self._save_debug_output(chapter_number, "scene_plan", chapter_plan if chapter_plan else "No plan generated.")
        self._update_rich_display(step=f"Ch {chapter_number} - Generating Hybrid Context")

        hybrid_context_for_draft = await generate_hybrid_chapter_context_logic(self.novel_props_cache, chapter_number, chapter_plan)
        await self._save_debug_output(chapter_number, "hybrid_context_for_draft", hybrid_context_for_draft)
        self._update_rich_display(step=f"Ch {chapter_number} - Drafting Initial Text")

        # Pass self (orchestrator) instead of novel_props_cache to drafting_agent if it needs direct attribute access
        initial_draft_text, initial_raw_llm_text, draft_usage = await self.drafting_agent.draft_chapter(
            self, # Pass orchestrator instance (agent)
            chapter_number, 
            plot_point_focus, 
            hybrid_context_for_draft, 
            chapter_plan
        )
        self._accumulate_tokens(f"Ch{chapter_number}-Drafting", draft_usage) # Accumulate tokens from drafting
        if not initial_draft_text:
            logger.error(f"NANA: Drafting Agent failed for Ch {chapter_number}. No initial draft produced.")
            await self._save_debug_output(chapter_number, "initial_draft_fail_raw_llm", initial_raw_llm_text or "Drafting Agent returned None for raw output.")
            self._update_rich_display(step=f"Ch {chapter_number} Failed - No Initial Draft")
            return None
        await self._save_debug_output(chapter_number, "initial_draft", initial_draft_text)
        
        current_text_to_process = initial_draft_text
        current_raw_llm_output = initial_raw_llm_text
        is_from_flawed_source = False 

        max_revision_attempts = 2 
        for attempt in range(max_revision_attempts + 1): 
            logger.info(f"NANA: Ch {chapter_number} - Evaluation Cycle, Attempt {attempt + 1}")
            self._update_rich_display(step=f"Ch {chapter_number} - Evaluation Cycle {attempt + 1}")
            
            eval_result_obj, eval_usage = await self.evaluator_agent.evaluate_chapter_draft(
                self.novel_props_cache, current_text_to_process, chapter_number,
                plot_point_focus, plot_point_index, hybrid_context_for_draft 
            )
            self._accumulate_tokens(f"Ch{chapter_number}-Evaluation-Attempt{attempt+1}", eval_usage)
            evaluation_result: EvaluationResult = eval_result_obj
            await self._save_debug_output(chapter_number, f"evaluation_result_attempt_{attempt+1}", evaluation_result)

            self._update_rich_display(step=f"Ch {chapter_number} - Continuity Check {attempt + 1}")
            continuity_problems, continuity_usage = await self.world_continuity_agent.check_consistency(
                 self.novel_props_cache, current_text_to_process, chapter_number, hybrid_context_for_draft
            )
            self._accumulate_tokens(f"Ch{chapter_number}-ContinuityCheck-Attempt{attempt+1}", continuity_usage)
            await self._save_debug_output(chapter_number, f"continuity_problems_attempt_{attempt+1}", continuity_problems)

            if continuity_problems:
                logger.warning(f"NANA: Ch {chapter_number} (Attempt {attempt+1}) - World Continuity Agent found {len(continuity_problems)} issues.")
                evaluation_result["problems_found"].extend(continuity_problems) 
                if not evaluation_result["needs_revision"]: 
                    evaluation_result["needs_revision"] = True
                    evaluation_result["reasons"].append("Continuity issues identified by WorldContinuityAgent.")
                else: 
                    if "Continuity issues identified by WorldContinuityAgent." not in evaluation_result["reasons"]:
                        evaluation_result["reasons"].append("Continuity issues identified by WorldContinuityAgent.")
            
            if not evaluation_result["needs_revision"]:
                logger.info(f"NANA: Ch {chapter_number} draft passed evaluation (Attempt {attempt+1}). Text is considered good.")
                is_from_flawed_source = False 
                self._update_rich_display(step=f"Ch {chapter_number} - Passed Evaluation")
                break 
            else: 
                logger.warning(f"NANA: Ch {chapter_number} draft (Attempt {attempt+1}) needs revision. Reasons: {'; '.join(evaluation_result.get('reasons',[]))}")
                
                if attempt >= max_revision_attempts:
                    logger.error(f"NANA: Ch {chapter_number} - Max revision attempts ({max_revision_attempts}) reached. Proceeding with current draft, marked as flawed.")
                    is_from_flawed_source = True 
                    self._update_rich_display(step=f"Ch {chapter_number} - Max Revisions Reached (Flawed)")
                    break 

                self._update_rich_display(step=f"Ch {chapter_number} - Revision Attempt {attempt + 1}")
                revision_tuple_result, revision_usage = await revise_chapter_draft_logic(
                    self, # Pass orchestrator instance (agent)
                    current_text_to_process, 
                    chapter_number,
                    evaluation_result, 
                    hybrid_context_for_draft, 
                    chapter_plan 
                )
                self._accumulate_tokens(f"Ch{chapter_number}-Revision-Attempt{attempt+1}", revision_usage)
                
                if revision_tuple_result and revision_tuple_result[0] and len(revision_tuple_result[0]) > 50: 
                    current_text_to_process, rev_raw_output = revision_tuple_result
                    current_raw_llm_output = rev_raw_output if rev_raw_output else current_raw_llm_output 
                    logger.info(f"NANA: Ch {chapter_number} - Revision attempt {attempt+1} successful. New text length: {len(current_text_to_process)}. Re-evaluating in next loop iteration.")
                    await self._save_debug_output(chapter_number, f"revised_text_attempt_{attempt+1}", current_text_to_process)
                else:
                    logger.error(f"NANA: Ch {chapter_number} - Revision attempt {attempt+1} failed to produce usable text. Proceeding with previous draft, marked as flawed.")
                    is_from_flawed_source = True 
                    self._update_rich_display(step=f"Ch {chapter_number} - Revision Failed (Flawed)")
                    break 
        
        if len(current_text_to_process) < config.MIN_ACCEPTABLE_DRAFT_LENGTH:
             logger.warning(f"NANA: Final chosen text for Ch {chapter_number} is too short ({len(current_text_to_process)} chars). This might impact quality and will be marked as flawed.")
             is_from_flawed_source = True 

        self._update_rich_display(step=f"Ch {chapter_number} - Summarizing")
        chapter_summary, summary_usage = await self.kg_maintainer_agent.summarize_chapter(current_text_to_process, chapter_number)
        self._accumulate_tokens(f"Ch{chapter_number}-Summarization", summary_usage)
        await self._save_debug_output(chapter_number, "final_summary", chapter_summary)
        final_embedding = await llm_interface.async_get_embedding(current_text_to_process)

        if final_embedding is None:
            logger.error(f"NANA CRITICAL: Failed to generate embedding for final text of Chapter {chapter_number}. Cannot save chapter to Neo4j state. Text saved to file system only.")
            await self._save_chapter_text_and_log(chapter_number, current_text_to_process, current_raw_llm_output)
            self._update_rich_display(step=f"Ch {chapter_number} Failed - No Embedding")
            return None 

        self._update_rich_display(step=f"Ch {chapter_number} - Saving to DB")
        await chapter_queries.save_chapter_data_to_db( # MODIFIED
            chapter_number, current_text_to_process, current_raw_llm_output or "N/A",
            chapter_summary, final_embedding, is_from_flawed_source
        )
        await self._save_chapter_text_and_log(chapter_number, current_text_to_process, current_raw_llm_output)

        self._update_rich_display(step=f"Ch {chapter_number} - Updating KG")
        kg_merge_usage = await self.kg_maintainer_agent.extract_and_merge_knowledge(
            self.novel_props_cache, 
            chapter_number,
            current_text_to_process,
            is_from_flawed_source 
        )
        self._accumulate_tokens(f"Ch{chapter_number}-KGExtractionMerge", kg_merge_usage)
        self.character_profiles = self.novel_props_cache['character_profiles']
        self.world_building = self.novel_props_cache['world_building']
        self._update_novel_props_cache() 

        self.chapter_count = max(self.chapter_count, chapter_number) 
        await self._save_core_novel_state_to_neo4j() 

        status_message = "Successfully Generated" if not is_from_flawed_source else "Generated (Marked with Flaws)"
        logger.info(f"=== NANA: Finished Chapter {chapter_number} - {status_message} ===")
        self._update_rich_display(step=f"Ch {chapter_number} - {status_message}")
        return current_text_to_process


    async def run_novel_generation_loop(self):
        logger.info("--- NANA: Starting Novel Generation Run ---")
        self.total_tokens_generated_this_run = 0 
        self.run_start_time = time.time()

        if self.rich_live: self.rich_live.start()

        try:
            await neo4j_manager.connect() # MODIFIED
            await neo4j_manager.create_db_schema()  # MODIFIED (renamed from create_db_and_tables)
            logger.info("NANA: Neo4j connection and schema verified.")

            await self.async_init_orchestrator() 

            if not self.plot_outline: 
                if not await self.perform_initial_setup():
                    logger.critical("NANA: Initial setup failed. Halting generation.")
                    self._update_rich_display(step="Initial Setup Failed - Halting")
                    return
            
            await self._prepopulate_kg_if_needed()
            self._update_novel_props_cache() 

            logger.info("\n--- NANA: Starting Novel Writing Process ---")
            start_chapter = self.chapter_count + 1
            end_chapter_exclusive = start_chapter + config.CHAPTERS_PER_RUN if config.CHAPTERS_PER_RUN > 0 else start_chapter

            logger.info(f"NANA: Current Chapter Count (at start of run): {self.chapter_count}")
            if start_chapter < end_chapter_exclusive:
                logger.info(f"NANA: Targeting Chapters: {start_chapter} to {end_chapter_exclusive - 1} in this run.")
            else:
                logger.info(f"NANA: CHAPTERS_PER_RUN ({config.CHAPTERS_PER_RUN}) results in no new chapters to generate. Current count: {self.chapter_count}.")

            chapters_successfully_written_this_run = 0
            for i in range(start_chapter, end_chapter_exclusive):
                logger.info(f"\n--- NANA: Attempting Chapter {i} ---")
                self._update_rich_display(chapter_num=i, step="Starting Chapter Loop")
                try:
                    chapter_text_result = await self.run_chapter_generation_process(i)
                    if chapter_text_result: 
                        chapters_successfully_written_this_run += 1
                        logger.info(f"NANA: Chapter {i}: Processed. Final text length: {len(chapter_text_result)} chars.")
                        logger.info(f"   Snippet: {chapter_text_result[:200].replace(chr(10), ' ')}...")
                    else:
                        logger.error(f"NANA: Chapter {i}: Failed to generate or save. Check logs for critical errors.")
                except Exception as e:
                    logger.critical(f"NANA: Critical unhandled error during chapter {i} writing process: {e}", exc_info=True)
                    self._update_rich_display(step=f"Critical Error Ch {i} - Halting")
                    break 

            logger.info(f"\n--- NANA: Novel writing process finished for this run ---")
            final_chapter_count_from_db = await chapter_queries.load_chapter_count_from_db()  # MODIFIED
            logger.info(f"NANA: Successfully processed {chapters_successfully_written_this_run} chapter(s) in this run.")
            logger.info(f"NANA: Current total chapters in database: {final_chapter_count_from_db}")
            logger.info(f"NANA: Total LLM tokens generated this run: {self.total_tokens_generated_this_run}")
            logger.info(f"--- NANA: Saga Novel Generation Run Finished. Final Neo4j chapter count: {final_chapter_count_from_db} ---")
            self._update_rich_display(chapter_num=final_chapter_count_from_db, step="Run Finished")


        except Exception as e:
            logger.critical(f"NANA: Unhandled exception in orchestrator main loop: {e}", exc_info=True)
            self._update_rich_display(step="Critical Error in Main Loop")
        finally:
            if self.rich_live: 
                await asyncio.sleep(0.1) 
                self.rich_live.stop() 
            await neo4j_manager.close() # MODIFIED
            logger.info("NANA: Neo4j driver successfully closed on application exit.")


def setup_logging_nana():
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format=config.LOG_FORMAT, 
        datefmt=config.LOG_DATE_FORMAT,
        handlers=[] 
    )
    
    root_logger = logging.getLogger()

    if config.LOG_FILE:
        try:
            log_dir = os.path.dirname(config.LOG_FILE)
            if log_dir: os.makedirs(log_dir, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                config.LOG_FILE, maxBytes=10*1024*1024, backupCount=5, mode='a', encoding='utf-8'
            )
            file_handler.setLevel(config.LOG_LEVEL) 
            formatter = logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler) 
            root_logger.info(f"File logging enabled. Log file: {config.LOG_FILE}")
        except Exception as e:
            root_logger.error(f"Failed to configure file logging to {config.LOG_FILE}: {e}", exc_info=True)

    if RICH_AVAILABLE and config.ENABLE_RICH_PROGRESS:
        rich_handler = RichHandler(
            level=config.LOG_LEVEL, 
            rich_tracebacks=True, 
            show_path=False, 
            markup=True, 
            show_time=True, 
            show_level=True 
        )
        root_logger.addHandler(rich_handler)
    else:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(config.LOG_LEVEL)
        stream_formatter = logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT)
        stream_handler.setFormatter(stream_formatter)
        root_logger.addHandler(stream_handler)
        if not (RICH_AVAILABLE and config.ENABLE_RICH_PROGRESS): # Log only if Rich isn't also logging it
             root_logger.info("Standard stream logging handler enabled for console.")


    logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)
    logging.getLogger("neo4j").setLevel(logging.WARNING) 
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
            
    root_logger.info(f"NANA Logging setup complete. Application Log Level: {logging.getLevelName(config.LOG_LEVEL)}.")


if __name__ == "__main__":
    setup_logging_nana()
    orchestrator = NANA_Orchestrator()
    try:
        asyncio.run(orchestrator.run_novel_generation_loop())
    except KeyboardInterrupt:
        logger.info("NANA Orchestrator shutting down gracefully due to KeyboardInterrupt...")
        if orchestrator.rich_live and orchestrator.rich_live.is_started: # type: ignore
            orchestrator._update_rich_display(step="Shutdown (KeyboardInterrupt)")
            orchestrator.rich_live.stop() # type: ignore
    except Exception as main_err:
        logger.critical(f"NANA Orchestrator encountered an unhandled main exception: {main_err}", exc_info=True)
        if orchestrator.rich_live and orchestrator.rich_live.is_started: # type: ignore
            orchestrator._update_rich_display(step=f"FATAL ERROR: {str(main_err)[:50]}...")
            orchestrator.rich_live.stop() # type: ignore
    finally:
        if neo4j_manager.driver is not None: # MODIFIED
            logger.info("Ensuring Neo4j driver is closed from main entry point.")
            async def _close_driver_main():
                 await neo4j_manager.close() # MODIFIED
            try:
                if asyncio.get_event_loop().is_running():
                    asyncio.create_task(_close_driver_main()) 
                else:
                    asyncio.run(_close_driver_main())
            except RuntimeError as e: 
                 logger.debug(f"Could not explicitly close driver from main (event loop might be closed or other issue): {e}")