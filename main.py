# main.py
"""
Main execution script for the Saga Novel Generation system.
Initializes logging, creates the NovelWriterAgent, ensures necessary
setup (plot outline, world-building), and runs the chapter generation loop.
Includes an option to run in asynchronous mode.

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

import logging
import sys
import os 
import random 
import asyncio # For running async operations

from novel_logic import NovelWriterAgent
import config 

# --- Configuration for Async Mode ---
# Set to True to run the novel generation using asynchronous methods where available.
# This requires novel_logic.py and other components to support async operations.
# Ensure you have httpx and aiosqlite installed: pip install httpx aiosqlite
RUN_ASYNCHRONOUSLY = True # Toggle this to switch between sync/async execution
# ------------------------------------

def setup_logging():
    log_level_name = config.LOG_LEVEL.upper()
    log_level = getattr(logging, log_level_name, logging.INFO) 

    logging.basicConfig(
        level=log_level,
        format=config.LOG_FORMAT,
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    if config.LOG_FILE:
        try:
            log_dir = os.path.dirname(config.LOG_FILE)
            if log_dir: os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(config.LOG_FILE, mode='a', encoding='utf-8')
            file_handler.setLevel(log_level)
            formatter = logging.Formatter(config.LOG_FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
            file_handler.setFormatter(formatter)
            logging.getLogger().addHandler(file_handler)
            logging.info(f"File logging enabled. Log file: {config.LOG_FILE}")
        except Exception as e:
            logging.error(f"Failed to configure file logging to {config.LOG_FILE}: {e}", exc_info=True)
            print(f"Warning: Could not set up log file at '{config.LOG_FILE}'. Error: {e}", file=sys.stderr)
    logging.info(f"Logging setup complete. Level: {log_level_name}")

async def run_novel_generation_async(): # Asynchronous version of the main runner
    """
    Main async function to initialize the agent and orchestrate the novel writing process.
    """
    setup_logging() # Logging setup is synchronous
    logger = logging.getLogger(__name__)
    logger.info(f"--- Starting Saga Novel Generation (Async Mode: {RUN_ASYNCHRONOUSLY}) ---")

    try:
        agent = NovelWriterAgent(use_async_db=RUN_ASYNCHRONOUSLY) # Pass async preference
        logger.info("NovelWriterAgent initialized successfully.")
    except Exception as e:
        logger.critical(f"Failed to initialize NovelWriterAgent: {e}", exc_info=True)
        print(f"\nFATAL: Could not initialize the agent. Check logs. Error: {e}", file=sys.stderr)
        sys.exit(1)

    logger.info("Checking for existing plot outline...")
    is_default_outline = agent.plot_outline.get("is_default", True) or \
                         not agent.plot_outline.get("title") or \
                         not agent.plot_outline.get("protagonist_name") or \
                         len(agent.plot_outline.get("plot_points", [])) < 5

    if is_default_outline:
        print("\n--- Generating New Plot Outline ---")
        logger.info("No valid plot outline found or outline appears default. Generating new one.")
        
        generation_params = {}
        if config.UNHINGED_PLOT_MODE:
            logger.info("Unhinged plot mode ENABLED. Randomizing core elements.")
            print("--- UNHINGED PLOT MODE: Generating randomized core elements ---")
            generation_params["genre"] = random.choice(config.UNHINGED_GENRES)
            generation_params["theme"] = random.choice(config.UNHINGED_THEMES)
            generation_params["setting_archetype"] = random.choice(config.UNHINGED_SETTINGS_ARCHETYPES)
            generation_params["protagonist_archetype"] = random.choice(config.UNHINGED_PROTAGONIST_ARCHETYPES)
            generation_params["conflict_archetype"] = random.choice(config.UNHINGED_CONFLICT_TYPES)
            logger.info(f"Randomized elements for unhinged mode: {generation_params}")
        else:
            logger.info("Standard plot mode. Using configured genre, theme, setting.")
            generation_params["genre"] = config.CONFIGURED_GENRE
            generation_params["theme"] = config.CONFIGURED_THEME
            generation_params["setting_description"] = config.CONFIGURED_SETTING_DESCRIPTION

        try:
            # generate_plot_outline is currently synchronous, but could be made async
            # If it were async: await agent.generate_plot_outline(...)
            outline = agent.generate_plot_outline(
                default_protagonist_name=config.DEFAULT_PROTAGONIST_NAME,
                unhinged_mode=config.UNHINGED_PLOT_MODE,
                **generation_params
            )
            print(f"Generated Outline for: '{outline.get('title', 'N/A')}' (Protagonist: {outline.get('protagonist_name', 'N/A')})")
            print(f"Genre: {outline.get('genre', 'N/A')}, Theme: {outline.get('theme', 'N/A')}")
            logger.info(f"Generated plot outline titled: '{outline.get('title', 'N/A')}'")
        except Exception as e:
            logger.critical(f"Critical error during plot outline generation: {e}", exc_info=True)
            print(f"\nFATAL: Error generating plot outline: {e}. Cannot continue.", file=sys.stderr)
            sys.exit(1)
    else:
        print("\n--- Using Existing Plot Outline ---")
        print(f"Loaded outline for: '{agent.plot_outline.get('title', 'N/A')}' (Protagonist: {agent.plot_outline.get('protagonist_name', 'N/A')})")
        logger.info(f"Using existing plot outline: '{agent.plot_outline.get('title', 'N/A')}'")

    logger.info("Checking for existing world-building data...")
    is_default_world = agent.world_building.get("is_default", True) or \
                       (len(agent.world_building.keys()) <= 3 and "Default Location" in agent.world_building.get("locations", {}))

    if is_default_world:
        print("\n--- Generating Initial World-Building Data ---")
        logger.info("World-building data appears default or missing. Generating initial data based on plot outline.")
        try:
            # generate_world_building is currently synchronous
            # If it were async: await agent.generate_world_building()
            agent.generate_world_building()
            print("Generated/Refreshed initial world-building data.")
            logger.info("Initial world-building data generation complete.")
        except Exception as e:
            logger.error(f"Error generating world building: {e}", exc_info=True)
            print(f"\nWarning: Error generating world building: {e}. Proceeding with potentially default data.")
    else:
        print("\n--- Using Existing World-Building Data ---")
        logger.info("Using existing world-building data.")

    if not agent.plot_outline.get("is_default", True) and not agent.world_building.get("is_default", True):
        is_kg_prepopulated = False
        if agent.chapter_count == 0: 
            # kg_at_zero = agent.db_manager.query_kg(...) # sync original
            # For async, this would be:
            kg_at_zero = await agent.db_manager.async_query_kg(
                subject=None, predicate=None, obj=None, 
                chapter_limit=config.KG_PREPOPULATION_CHAPTER_NUM, 
                include_provisional=True 
            ) if RUN_ASYNCHRONOUSLY else agent.db_manager.query_kg(
                 subject=None, predicate=None, obj=None, 
                chapter_limit=config.KG_PREPOPULATION_CHAPTER_NUM, 
                include_provisional=True
            )

            kg_at_zero_specific = [triple for triple in kg_at_zero if triple.get('chapter_added') == config.KG_PREPOPULATION_CHAPTER_NUM]
            if kg_at_zero_specific:
                is_kg_prepopulated = True
                logger.info(f"Found {len(kg_at_zero_specific)} existing KG triples at chapter {config.KG_PREPOPULATION_CHAPTER_NUM}. Assuming KG already pre-populated.")
        
        if not is_kg_prepopulated and agent.chapter_count == 0: 
            print("\n--- Pre-populating Knowledge Graph from Plot and World Data ---")
            logger.info("Attempting to pre-populate Knowledge Graph.")
            try:
                await agent._prepopulate_knowledge_graph() # _prepopulate_knowledge_graph is now async
                print("Knowledge Graph pre-population step complete.")
                logger.info("Knowledge Graph pre-population successful.")
            except Exception as e:
                logger.error(f"Error during Knowledge Graph pre-population: {e}", exc_info=True)
                print(f"\nWarning: Error pre-populating Knowledge Graph: {e}. KG might be incomplete.")
        elif agent.chapter_count > 0:
            logger.info("Skipping KG pre-population: Novel already has chapters.")


    print("\n--- Starting Novel Writing Process ---")
    # agent.chapter_count can be loaded async too if load_chapter_count is made async
    # For now, assuming it's loaded synchronously during agent init.
    start_chapter = agent.chapter_count + 1
    end_chapter = start_chapter + config.CHAPTERS_PER_RUN

    print(f"Current Chapter Count (from DB): {agent.chapter_count}")
    if config.CHAPTERS_PER_RUN > 0:
        print(f"Targeting Chapters: {start_chapter} to {end_chapter - 1} in this run.")
        logger.info(f"Starting chapter writing loop from {start_chapter} to {end_chapter - 1}.")
    else:
        print("CHAPTERS_PER_RUN is set to 0 in config. No chapters will be written.")
        logger.info("CHAPTERS_PER_RUN is 0, skipping chapter writing loop.")

    for i in range(start_chapter, end_chapter):
        print(f"\n--- Attempting Chapter {i} ---")
        logger.info(f"--- Starting Generation for Chapter {i} ---")
        try:
            # write_chapter is now async
            chapter_text = await agent.write_chapter(i)
            if chapter_text:
                print(f"Chapter {i}: Successfully generated and saved (Length: {len(chapter_text)} chars).")
                snippet = ' '.join(chapter_text[:250].splitlines()).strip()
                print(f"Chapter {i} Snippet: {snippet}...")
                logger.info(f"--- Successfully completed Chapter {i} ---")
            else:
                print(f"Chapter {i}: Failed to generate or save. Check logs for details.")
                logger.error(f"Chapter {i} generation failed. See previous log messages for reasons.")
        except Exception as e:
            logger.critical(f"Critical error during chapter {i} writing process: {e}", exc_info=True)
            print(f"\n!!! Critical Error during chapter {i} writing: {e} !!! Halting generation.", file=sys.stderr)
            break 

    print(f"\n--- Novel writing process finished for this run ---")
    print(f"Final Agent Chapter Count (in DB): {agent.chapter_count}") # agent.chapter_count updated in write_chapter
    print(f"Check the '{config.OUTPUT_DIR}' directory for JSON state files, chapter text files, logs, and the database ('{os.path.basename(config.DATABASE_FILE)}').")
    logger.info(f"--- Saga Novel Generation Run Finished. Final chapter count: {agent.chapter_count} ---")


# Synchronous version for compatibility or if async is not desired.
# This is mostly your original run_novel_generation function.
def run_novel_generation_sync():
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info(f"--- Starting Saga Novel Generation (Async Mode: {RUN_ASYNCHRONOUSLY}) ---")

    try:
        agent = NovelWriterAgent(use_async_db=False) # Explicitly false for sync
        logger.info("NovelWriterAgent initialized successfully.")
    except Exception as e:
        logger.critical(f"Failed to initialize NovelWriterAgent: {e}", exc_info=True)
        sys.exit(1)

    is_default_outline = agent.plot_outline.get("is_default", True) or \
                         not agent.plot_outline.get("title") or \
                         not agent.plot_outline.get("protagonist_name") or \
                         len(agent.plot_outline.get("plot_points", [])) < 5

    if is_default_outline:
        print("\n--- Generating New Plot Outline ---")
        generation_params = {}
        if config.UNHINGED_PLOT_MODE:
            generation_params["genre"] = random.choice(config.UNHINGED_GENRES)
            generation_params["theme"] = random.choice(config.UNHINGED_THEMES)
            # ... (rest of unhinged params)
        else:
            generation_params["genre"] = config.CONFIGURED_GENRE # ... (rest of standard params)
        
        outline = agent.generate_plot_outline( # Synchronous call
            default_protagonist_name=config.DEFAULT_PROTAGONIST_NAME,
            unhinged_mode=config.UNHINGED_PLOT_MODE,
            **generation_params
        )
        print(f"Generated Outline for: '{outline.get('title', 'N/A')}'")
    else:
        print(f"\n--- Using Existing Plot Outline: '{agent.plot_outline.get('title', 'N/A')}' ---")


    is_default_world = agent.world_building.get("is_default", True) or \
                       (len(agent.world_building.keys()) <= 3 and "Default Location" in agent.world_building.get("locations", {}))
    if is_default_world:
        print("\n--- Generating Initial World-Building Data ---")
        agent.generate_world_building() # Synchronous call
    else:
        print("\n--- Using Existing World-Building Data ---")


    if not agent.plot_outline.get("is_default", True) and not agent.world_building.get("is_default", True):
        is_kg_prepopulated = False
        if agent.chapter_count == 0:
            kg_at_zero = agent.db_manager.query_kg(
                subject=None, predicate=None, obj=None, 
                chapter_limit=config.KG_PREPOPULATION_CHAPTER_NUM, 
                include_provisional=True
            )
            kg_at_zero_specific = [triple for triple in kg_at_zero if triple.get('chapter_added') == config.KG_PREPOPULATION_CHAPTER_NUM]
            if kg_at_zero_specific: is_kg_prepopulated = True
        
        if not is_kg_prepopulated and agent.chapter_count == 0:
            print("\n--- Pre-populating Knowledge Graph ---")
            # _prepopulate_knowledge_graph became async.
            # For a truly synchronous path, you'd need a sync version or run this part async.
            # This illustrates the divergence if not fully converting.
            # For now, we'll assume if RUN_ASYNCHRONOUSLY is false, we might skip this or it needs a sync alternative.
            # Or, we can run just this part using asyncio.run() if it's standalone enough.
            try:
                asyncio.run(agent._prepopulate_knowledge_graph()) # Run this specific async method
                print("Knowledge Graph pre-population step complete (run synchronously via asyncio.run).")
            except Exception as e:
                 logger.error(f"Error during KG pre-population (sync main): {e}", exc_info=True)
        elif agent.chapter_count > 0:
            logger.info("Skipping KG pre-population: Novel already has chapters.")


    print("\n--- Starting Novel Writing Process ---")
    start_chapter = agent.chapter_count + 1
    end_chapter = start_chapter + config.CHAPTERS_PER_RUN

    for i in range(start_chapter, end_chapter):
        print(f"\n--- Attempting Chapter {i} ---")
        # write_chapter is now async. For a sync path, this also needs adjustment.
        # This highlights the difficulty of a partial async conversion.
        # We'll run it using asyncio.run for this demonstration of a "sync" path.
        try:
            chapter_text = asyncio.run(agent.write_chapter(i))
            if chapter_text:
                print(f"Chapter {i}: Successfully generated (Length: {len(chapter_text)} chars).")
            else:
                print(f"Chapter {i}: Failed to generate.")
        except Exception as e:
            logger.critical(f"Critical error during chapter {i} (sync main): {e}", exc_info=True)
            break
            
    print(f"\n--- Novel writing process finished. Final count: {agent.chapter_count} ---")


if __name__ == "__main__":
    if RUN_ASYNCHRONOUSLY:
        asyncio.run(run_novel_generation_async())
    else:
        # The synchronous path is more complex now due to async methods in NovelWriterAgent.
        # A pure synchronous path would require not making NovelWriterAgent methods async,
        # or having duplicate sync/async logic paths throughout.
        # The `run_novel_generation_sync` above attempts to call async methods using `asyncio.run`
        # for individual operations, which is not ideal for performance but demonstrates a way.
        print("Running in a hybrid mode. Some async operations will be run synchronously using asyncio.run().")
        print("For true synchronous execution, NovelWriterAgent methods should not be async.")
        run_novel_generation_sync()
