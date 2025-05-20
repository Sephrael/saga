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
from state_manager import state_manager # Import the global instance

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
    logger.info("Performing initial setup...")

    # generate_plot_outline_logic now handles user file, LLM gen, or default.
    # It populates agent.plot_outline, agent.character_profiles, and potentially agent.world_building.
    print("\n--- Initializing Plot, Characters, and World ---")
    generation_params: Dict[str, Any] = {}
    if config.UNHINGED_PLOT_MODE and not os.path.exists(config.USER_STORY_ELEMENTS_FILE_PATH): # Unhinged only if no user file
        generation_params.update({
            "genre": random.choice(config.UNHINGED_GENRES), "theme": random.choice(config.UNHINGED_THEMES),
            "setting_archetype": random.choice(config.UNHINGED_SETTINGS_ARCHETYPES),
            "protagonist_archetype": random.choice(config.UNHINGED_PROTAGONIST_ARCHETYPES),
            "conflict_archetype": random.choice(config.UNHINGED_CONFLICT_TYPES)
        })
    elif not os.path.exists(config.USER_STORY_ELEMENTS_FILE_PATH): # Configured mode if no user file and not unhinged
        generation_params.update({
            "genre": config.CONFIGURED_GENRE, "theme": config.CONFIGURED_THEME,
            "setting_description": config.CONFIGURED_SETTING_DESCRIPTION
        })
    
    try:
        # This call will populate plot_outline, character_profiles, and potentially world_building
        # if a user file is found and processed.
        await agent.generate_plot_outline(
            default_protagonist_name=config.DEFAULT_PROTAGONIST_NAME,
            unhinged_mode=config.UNHINGED_PLOT_MODE if not os.path.exists(config.USER_STORY_ELEMENTS_FILE_PATH) else False,
            **generation_params
        )
        
        # Log how the plot outline was sourced
        plot_source = agent.plot_outline.get("source", "unknown")
        if plot_source == "user_supplied":
            print(f"Loaded story elements from user file: '{config.USER_STORY_ELEMENTS_FILE_PATH}'")
            print(f"   Novel Title: '{agent.plot_outline.get('title', 'N/A')}'")
        elif plot_source.startswith("llm_generated"):
            print(f"Generated Plot Outline via LLM for: '{agent.plot_outline.get('title', 'N/A')}'")
        elif plot_source == "default_fallback":
            print(f"Plot Outline defaulted for: '{agent.plot_outline.get('title', 'N/A')}'")
        else:
            print(f"Plot Outline initialized for: '{agent.plot_outline.get('title', 'N/A')}' (source: {plot_source})")

        # Now call generate_world_building. It will skip LLM generation if
        # world_building was already populated from a user file.
        await agent.generate_world_building()
        world_source = agent.world_building.get("source", agent.world_building.get("user_supplied_data"))
        if world_source == "user_supplied_data" or agent.world_building.get("user_supplied_data"): # Redundant check for clarity
            print("   World-building data also loaded from user file.")
        elif world_source == "llm_generated":
            print("   Generated initial world-building data via LLM.")
        elif world_source == "default_fallback":
            print("   World-building data defaulted.")
        else:
             logger.info("   World-building data seems pre-existing or source unclear, using as-is.")


    except Exception as e:
        logger.critical(f"Critical error during initial setup (plot/character/world generation): {e}", exc_info=True)
        return False 
    
    if not agent.plot_outline or agent.plot_outline.get("is_default"):
        logger.warning("Initial setup resulted in a default or empty plot outline. This might impact generation quality.")
        # Not returning False, as system might still run with defaults.
    
    return True 

async def prepopulate_kg_if_needed(agent: NovelWriterAgent):
    logger = logging.getLogger(__name__)
    
    # Check if initial setup was from user file or resulted in default, which might affect KG pre-population decision
    plot_source = agent.plot_outline.get("source", "")
    is_user_or_llm_plot = plot_source == "user_supplied" or plot_source.startswith("llm_generated")

    if not is_user_or_llm_plot:
        logger.info(f"Skipping KG pre-population: Plot outline is default or source is unclear ('{plot_source}').")
        return

    # Check if the Neo4j connection is established before querying
    if state_manager.driver is None:
        logger.warning("Neo4j driver not connected. Attempting to connect for KG pre-population check.")
        try:
            await state_manager.connect()
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j, cannot check for existing KG pre-population: {e}")
            return

    existing_prepop_facts = await state_manager.async_query_kg(
        chapter_limit=config.KG_PREPOPULATION_CHAPTER_NUM, 
        include_provisional=True 
    )
    prepop_facts_at_zero = [f for f in existing_prepop_facts if f.get('chapter_added') == config.KG_PREPOPULATION_CHAPTER_NUM]

    if prepop_facts_at_zero:
        logger.info(f"Found {len(prepop_facts_at_zero)} existing KG triples at chapter {config.KG_PREPOPULATION_CHAPTER_NUM}. Assuming KG pre-populated.")
        return
        
    print("\n--- Pre-populating Knowledge Graph from Initial Data ---")
    try:
        await agent._prepopulate_knowledge_graph()
        print("Knowledge Graph pre-population step complete.")
    except Exception as e:
        logger.error(f"Error during Knowledge Graph pre-population: {e}", exc_info=True)

async def run_novel_generation_async():
    logger = logging.getLogger(__name__) 
    logger.info(f"--- Starting Saga Novel Generation (Execution Mode: Async) ---")

    try:
        # Initialize state_manager and connect to Neo4j
        await state_manager.connect()
        await state_manager.create_db_and_tables() # This will now create Neo4j constraints/indexes
        logger.info("state_manager initialized and Neo4j connection/constraints verified.")

        agent = NovelWriterAgent()
        await agent.async_init() # Load agent's state from Neo4j
        logger.info("NovelWriterAgent initialized and state loaded successfully.")
    except Exception as e:
        logger.critical(f"Failed to initialize state_manager or NovelWriterAgent: {e}", exc_info=True)
        print(f"\nFATAL: Could not initialize. Check logs. Error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        if not await perform_initial_setup(agent):
            logger.critical("Initial setup (plot/world) failed. Halting generation.")
            sys.exit(1)

        await prepopulate_kg_if_needed(agent)

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
                chapter_text = await agent.write_chapter(i)
                if chapter_text:
                    chapters_successfully_written += 1
                    print(f"Chapter {i}: Successfully generated (Length: {len(chapter_text)} chars).")
                    print(f"   Snippet: {chapter_text[:200].replace(chr(10), ' ')}...")
                else:
                    print(f"Chapter {i}: Failed to generate or save. Check logs.")
                    # break # Optionally stop on first failure
            except Exception as e:
                logger.critical(f"Critical error during chapter {i} writing process: {e}", exc_info=True)
                break 

        print(f"\n--- Novel writing process finished for this run ---")
        final_chapter_count_from_db = await state_manager.async_load_chapter_count()
        print(f"Successfully wrote {chapters_successfully_written} chapter(s).")
        print(f"Current total chapters in database: {final_chapter_count_from_db}")
        logger.info(f"--- Saga Novel Generation Run Finished. Final Neo4j chapter count: {final_chapter_count_from_db} ---")

    finally:
        await state_manager.close() # Ensure Neo4j driver is closed
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
    else:
        logger = logging.getLogger(__name__)
        logger.warning("RUN_WITH_ASYNCIO_RUN is False. "
                       "This script expects 'run_novel_generation_async()' to be called from an existing event loop if not run directly.")
        print("Script not run with asyncio.run(). If you intended to run the novel generation, "
              "ensure RUN_WITH_ASYNCIO_RUN is True or call run_novel_generation_async() from an event loop.")
        pass