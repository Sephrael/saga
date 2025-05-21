# main.py
from logging.handlers import RotatingFileHandler
import logging
import sys
import os 
import random 
import asyncio
from typing import Dict, Any

from novel_agent import NovelWriterAgent
import config 
from state_manager import state_manager 

RUN_WITH_ASYNCIO_RUN = True 

def setup_logging():
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT
    )
    if config.LOG_FILE:
        try:
            log_dir = os.path.dirname(config.LOG_FILE)
            if log_dir: os.makedirs(log_dir, exist_ok=True)
            
            file_handler = RotatingFileHandler(config.LOG_FILE, maxBytes=10**6, backupCount=5, mode='a', encoding='utf-8')
            file_handler.setLevel(config.LOG_LEVEL)
            formatter = logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT)
            file_handler.setFormatter(formatter)
            logging.getLogger().addHandler(file_handler)
            logging.info(f"File logging enabled. Log file: {config.LOG_FILE}")
        except Exception as e:
            logging.error(f"Failed to configure file logging to {config.LOG_FILE}: {e}", exc_info=True)
            print(f"CRITICAL: Could not set up log file at '{config.LOG_FILE}'. Error: {e}", file=sys.stderr)
            sys.exit(1)
    logging.info(f"Logging setup complete. Level: {logging.getLevelName(config.LOG_LEVEL)}")


async def perform_initial_setup(agent: NovelWriterAgent) -> bool:
    logger = logging.getLogger(__name__)
    logger.info("Performing initial setup (populating agent's Python dicts)...")

    print("\n--- Initializing Plot, Characters, and World (in-memory dicts) ---")
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
    
    try:
        # These calls populate agent's Python dicts (self.plot_outline, etc.)
        await agent.generate_plot_outline(
            default_protagonist_name=config.DEFAULT_PROTAGONIST_NAME,
            unhinged_mode=config.UNHINGED_PLOT_MODE if not os.path.exists(config.USER_STORY_ELEMENTS_FILE_PATH) else False,
            **generation_params
        )
        plot_source = agent.plot_outline.get("source", "unknown")
        print(f"   Plot Outline Python dict initialized/loaded (source: {plot_source}). Title: '{agent.plot_outline.get('title', 'N/A')}'")

        await agent.generate_world_building()
        world_source = agent.world_building.get("source", "unknown")
        is_user_supplied_world = agent.world_building.get("user_supplied_data", False)
        print(f"   World Building Python dict initialized/loaded (source: {world_source}, user_supplied: {is_user_supplied_world}).")

        # Now, explicitly save the populated Python dicts to Neo4j in decomposed form
        await agent._save_all_json_state()
        print("   Initial plot, character, and world Python dicts saved to Neo4j (decomposed).")

    except Exception as e:
        logger.critical(f"Critical error during initial setup (Python dict population or save to Neo4j): {e}", exc_info=True)
        return False 
    
    if not agent.plot_outline or agent.plot_outline.get("is_default"):
        logger.warning("Initial setup resulted in a default or empty plot outline dict. This might impact generation quality.")
    
    return True 

async def prepopulate_kg_if_needed(agent: NovelWriterAgent):
    logger = logging.getLogger(__name__)
    
    plot_source = agent.plot_outline.get("source", "")
    is_user_or_llm_plot = plot_source == "user_supplied" or plot_source.startswith("llm_generated")

    if not is_user_or_llm_plot:
        logger.info(f"Skipping KG pre-population: Plot outline is default or source is unclear ('{plot_source}'). Pre-population relies on detailed initial data.")
        return

    if state_manager.driver is None:
        logger.warning("Neo4j driver not connected. Attempting connect for KG pre-population check.")
        try: await state_manager.connect()
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j, cannot check for existing KG pre-population: {e}")
            return

    # Check if any :NovelInfo node exists, as a proxy for prepopulation
    # A more robust check would count specific types of prepopulated nodes/rels.
    check_query = f"MATCH (ni:NovelInfo {{id: '{config.MAIN_NOVEL_INFO_NODE_ID}'}}) RETURN count(ni) as count"
    try:
        result = await state_manager._execute_read_query(check_query)
        if result and result[0] and result[0]['count'] > 0:
            logger.info(f"Found existing NovelInfo node. Assuming KG already pre-populated or managed. Skipping explicit pre-population step.")
            # If we want to ensure plot_points are there from a previous save:
            # plot_points_count_query = f"MATCH (:NovelInfo {{id: '{config.MAIN_NOVEL_INFO_NODE_ID}'}})-[:HAS_PLOT_POINT]->(:PlotPoint) RETURN count(*) as pp_count"
            # pp_res = await state_manager._execute_read_query(plot_points_count_query)
            # if pp_res and pp_res[0]['pp_count'] > 0:
            #     logger.info(f"Found {pp_res[0]['pp_count']} plot points. KG seems pre-populated.")
            #     return
            # else:
            #     logger.info("NovelInfo node exists, but no plot points. Proceeding with pre-population.")
            return # Simplified: if NovelInfo exists, assume prepopulated for now
    except Exception as e:
        logger.error(f"Error checking for existing KG pre-population: {e}. Will attempt pre-population.", exc_info=True)
        
    print("\n--- Pre-populating Knowledge Graph from Initial Agent Data (Python Dicts) ---")
    try:
        await agent._prepopulate_knowledge_graph() # This now uses the direct Cypher generation method
        print("Knowledge Graph pre-population step from agent's Python dicts complete.")
    except Exception as e:
        logger.error(f"Error during Knowledge Graph pre-population from agent dicts: {e}", exc_info=True)

async def run_novel_generation_async():
    logger = logging.getLogger(__name__) 
    logger.info(f"--- Starting Saga Novel Generation (Execution Mode: Async, Neo4j Decomposed) ---")

    try:
        await state_manager.connect()
        await state_manager.create_db_and_tables() 
        logger.info("state_manager initialized and Neo4j connection/schema verified.")

        agent = NovelWriterAgent()
        await agent.async_init() # Loads state from Neo4j into agent's Python dicts
        logger.info("NovelWriterAgent initialized and state loaded (Python dicts from Neo4j).")
    except Exception as e:
        logger.critical(f"Failed to initialize state_manager or NovelWriterAgent: {e}", exc_info=True)
        print(f"\nFATAL: Could not initialize. Check logs. Error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        if not await perform_initial_setup(agent): # Populates agent dicts, then saves to Neo4j decomposed
            logger.critical("Initial setup (Python dict population or save to Neo4j) failed. Halting.")
            sys.exit(1)

        await prepopulate_kg_if_needed(agent) # Prepopulates KG directly from agent's Python dicts

        print("\n--- Starting Novel Writing Process ---")
        start_chapter = agent.chapter_count + 1
        end_chapter = start_chapter + config.CHAPTERS_PER_RUN if config.CHAPTERS_PER_RUN > 0 else start_chapter

        print(f"Current Chapter Count (from Neo4j at agent init): {agent.chapter_count}")
        if start_chapter < end_chapter :
            print(f"Targeting Chapters: {start_chapter} to {end_chapter - 1} in this run.")
        else:
            print(f"CHAPTERS_PER_RUN ({config.CHAPTERS_PER_RUN}) results in no new chapters. Current: {agent.chapter_count}.")
            logger.info(f"CHAPTERS_PER_RUN is {config.CHAPTERS_PER_RUN}, skipping chapter writing loop.")


        chapters_successfully_written = 0
        for i in range(start_chapter, end_chapter):
            print(f"\n--- Attempting Chapter {i} ---")
            try:
                # agent.write_chapter internally uses agent's Python dicts for its logic,
                # then saves the final chapter text and updates to knowledge bases (Python dicts + KG triples).
                # Finally, it calls _save_all_json_state to persist the updated Python dicts to Neo4j (decomposed).
                chapter_text = await agent.write_chapter(i)
                if chapter_text:
                    chapters_successfully_written += 1
                    print(f"Chapter {i}: Successfully generated (Length: {len(chapter_text)} chars).")
                    print(f"   Snippet: {chapter_text[:200].replace(chr(10), ' ')}...")
                else:
                    print(f"Chapter {i}: Failed to generate or save. Check logs.")
            except Exception as e:
                logger.critical(f"Critical error during chapter {i} writing process: {e}", exc_info=True)
                break 

        print(f"\n--- Novel writing process finished for this run ---")
        final_chapter_count_from_db = await state_manager.async_load_chapter_count()
        print(f"Successfully wrote {chapters_successfully_written} chapter(s).")
        print(f"Current total chapters in database: {final_chapter_count_from_db}")
        logger.info(f"--- Saga Novel Generation Run Finished. Final Neo4j chapter count: {final_chapter_count_from_db} ---")

    finally:
        await state_manager.close() 
        logger.info("Neo4j driver successfully closed on application exit.")


if __name__ == "__main__":
    setup_logging() 
    if RUN_WITH_ASYNCIO_RUN:
        try:
            asyncio.run(run_novel_generation_async())
        except KeyboardInterrupt:
            logging.getLogger(__name__).info("Shutting down gracefully...")
        except Exception as main_err: 
            logging.getLogger(__name__).critical(f"Unhandled main exception: {main_err}", exc_info=True)
            sys.exit(1) # Ensure exit on unhandled error in main async run
    else: # Fallback for environments where asyncio.run() is not suitable
        logger = logging.getLogger(__name__)
        logger.warning("RUN_WITH_ASYNCIO_RUN is False. "
                       "This script expects 'run_novel_generation_async()' to be called from an existing event loop if not run directly.")
        # Consider either exiting or providing a clear message if this path is not intended for production
        # sys.exit(1)