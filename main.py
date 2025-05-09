# main.py
"""
Main execution script for the Saga Novel Generation system.
Initializes logging, creates the NovelWriterAgent, ensures necessary
setup (plot outline, world-building), and runs the chapter generation loop.
Primarily designed for asynchronous execution.

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
import asyncio

from novel_logic import NovelWriterAgent # JsonStateData removed as it's internal to novel_logic
import config 

# --- Configuration for Execution Mode ---
# Set to True to run the main loop using asyncio.run().
# If False, it's implied that this script might be part of a larger async application.
# The NovelWriterAgent is now designed to be async-first.
RUN_WITH_ASYNCIO_RUN = True 
# ------------------------------------

def setup_logging():
    """Configures logging for the application."""
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT
    )
    # Configure file handler if LOG_FILE is set
    if config.LOG_FILE:
        try:
            log_dir = os.path.dirname(config.LOG_FILE)
            if log_dir: os.makedirs(log_dir, exist_ok=True)
            
            file_handler = logging.FileHandler(config.LOG_FILE, mode='a', encoding='utf-8')
            file_handler.setLevel(config.LOG_LEVEL)
            formatter = logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT)
            file_handler.setFormatter(formatter)
            logging.getLogger().addHandler(file_handler)
            logging.info(f"File logging enabled. Log file: {config.LOG_FILE}")
        except Exception as e:
            # Log to console if file logging setup fails
            logging.error(f"Failed to configure file logging to {config.LOG_FILE}: {e}", exc_info=True)
            print(f"CRITICAL: Could not set up log file at '{config.LOG_FILE}'. Error: {e}", file=sys.stderr)
            sys.exit(1) # Exit if crucial logging to file fails
            
    logging.info(f"Logging setup complete. Level: {logging.getLevelName(config.LOG_LEVEL)}")


async def perform_initial_setup(agent: NovelWriterAgent) -> bool:
    """Handles the initial setup for plot outline and world-building."""
    logger = logging.getLogger(__name__)

    # --- Plot Outline Setup ---
    logger.info("Checking for existing plot outline...")
    # Check if plot outline is default or missing essential elements
    is_default_outline = agent.plot_outline.get("is_default", True) or \
                         not agent.plot_outline.get("title") or \
                         agent.plot_outline.get("title") == config.DEFAULT_PLOT_OUTLINE_TITLE or \
                         not agent.plot_outline.get("protagonist_name") or \
                         len(agent.plot_outline.get("plot_points", [])) < 5

    if is_default_outline:
        print("\n--- Generating New Plot Outline ---")
        logger.info("No valid plot outline found or outline appears default. Generating a new one.")
        
        generation_params: Dict[str, Any] = {}
        if config.UNHINGED_PLOT_MODE:
            logger.info("Unhinged plot mode ENABLED. Randomizing core elements for plot generation.")
            print("--- UNHINGED PLOT MODE: Generating randomized core elements for plot ---")
            generation_params.update({
                "genre": random.choice(config.UNHINGED_GENRES),
                "theme": random.choice(config.UNHINGED_THEMES),
                "setting_archetype": random.choice(config.UNHINGED_SETTINGS_ARCHETYPES),
                "protagonist_archetype": random.choice(config.UNHINGED_PROTAGONIST_ARCHETYPES),
                "conflict_archetype": random.choice(config.UNHINGED_CONFLICT_TYPES)
            })
            logger.info(f"Randomized elements for unhinged plot mode: {generation_params}")
        else:
            logger.info("Standard plot mode. Using configured genre, theme, and setting description.")
            generation_params.update({
                "genre": config.CONFIGURED_GENRE,
                "theme": config.CONFIGURED_THEME,
                "setting_description": config.CONFIGURED_SETTING_DESCRIPTION
            })

        try:
            outline = await agent.generate_plot_outline(
                default_protagonist_name=config.DEFAULT_PROTAGONIST_NAME,
                unhinged_mode=config.UNHINGED_PLOT_MODE,
                **generation_params
            )
            print(f"Generated Plot Outline for: '{outline.get('title', 'N/A')}' (Protagonist: {outline.get('protagonist_name', 'N/A')})")
            print(f"Genre: {outline.get('genre', 'N/A')}, Theme: {outline.get('theme', 'N/A')}")
            logger.info(f"Generated plot outline titled: '{outline.get('title', 'N/A')}'")
        except Exception as e:
            logger.critical(f"Critical error during plot outline generation: {e}", exc_info=True)
            print(f"\nFATAL: Error generating plot outline: {e}. Cannot continue.", file=sys.stderr)
            return False # Indicate setup failure
    else:
        print("\n--- Using Existing Plot Outline ---")
        print(f"Loaded outline for: '{agent.plot_outline.get('title', 'N/A')}' (Protagonist: {agent.plot_outline.get('protagonist_name', 'N/A')})")
        logger.info(f"Using existing plot outline: '{agent.plot_outline.get('title', 'N/A')}'")

    # --- World-Building Setup ---
    logger.info("Checking for existing world-building data...")
    # Check if world-building is default or minimal
    is_default_world = agent.world_building.get("is_default", True) or \
                       (len(agent.world_building.keys() - {"is_default"}) <= 1 and # e.g. only 'locations' or 'society'
                        ("Default Location" in agent.world_building.get("locations", {}) or \
                         "General" in agent.world_building.get("society", {})) # Simplified check
                       )


    if is_default_world:
        print("\n--- Generating Initial World-Building Data ---")
        logger.info("World-building data appears default or minimal. Generating initial data based on plot outline.")
        try:
            await agent.generate_world_building()
            print("Generated/Refreshed initial world-building data.")
            logger.info("Initial world-building data generation complete.")
        except Exception as e:
            logger.error(f"Error generating world building: {e}", exc_info=True)
            print(f"\nWarning: Error generating world building: {e}. Proceeding with potentially default or incomplete data.")
            # Not returning False here, as system might still function with basic world data.
    else:
        print("\n--- Using Existing World-Building Data ---")
        logger.info("Using existing world-building data.")
    
    return True # Indicate successful setup (or proceeding despite minor warnings)


async def prepopulate_kg_if_needed(agent: NovelWriterAgent):
    """Pre-populates the Knowledge Graph if it's a new novel."""
    logger = logging.getLogger(__name__)
    
    # KG pre-population only makes sense if plot and world are non-default.
    # The is_default flag is now handled more dynamically during save by the agent.
    # Check for substantial content instead.
    plot_is_substantial = agent.plot_outline.get("title") != config.DEFAULT_PLOT_OUTLINE_TITLE
    world_is_substantial = len(agent.world_building.keys() - {"is_default"}) > 1 or \
                           not ("Default Location" in agent.world_building.get("locations", {}))


    if not plot_is_substantial or not world_is_substantial:
        logger.info("Skipping KG pre-population: Plot outline or world-building appears default or lacks substantial content.")
        return

    if agent.chapter_count > 0:
        logger.info(f"Skipping KG pre-population: Novel already has {agent.chapter_count} chapters.")
        return

    # Check if KG already has facts for chapter 0 (pre-population chapter)
    existing_prepop_facts = await agent.db_manager.async_query_kg(
        chapter_limit=config.KG_PREPOPULATION_CHAPTER_NUM, 
        include_provisional=True 
    )
    prepop_facts_at_zero = [f for f in existing_prepop_facts if f.get('chapter_added') == config.KG_PREPOPULATION_CHAPTER_NUM]

    if prepop_facts_at_zero:
        logger.info(f"Found {len(prepop_facts_at_zero)} existing KG triples at chapter {config.KG_PREPOPULATION_CHAPTER_NUM}. Assuming KG already pre-populated.")
        return
        
    print("\n--- Pre-populating Knowledge Graph from Plot and World Data ---")
    logger.info("Attempting to pre-populate Knowledge Graph as it's a new novel with no chapter 0 facts.")
    try:
        await agent._prepopulate_knowledge_graph() # Accessing protected member for this specific setup task
        print("Knowledge Graph pre-population step complete.")
        logger.info("Knowledge Graph pre-population successful.")
    except Exception as e:
        logger.error(f"Error during Knowledge Graph pre-population: {e}", exc_info=True)
        print(f"\nWarning: Error pre-populating Knowledge Graph: {e}. KG might be incomplete.")


async def run_novel_generation_async():
    """
    Main asynchronous function to initialize the agent and orchestrate the novel writing process.
    """
    logger = logging.getLogger(__name__) 
    logger.info(f"--- Starting Saga Novel Generation (Execution Mode: Async) ---")

    try:
        agent = NovelWriterAgent() 
        logger.info("NovelWriterAgent initialized successfully.")
    except Exception as e:
        logger.critical(f"Failed to initialize NovelWriterAgent: {e}", exc_info=True)
        print(f"\nFATAL: Could not initialize the agent. Check logs. Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not await perform_initial_setup(agent):
        logger.critical("Initial setup (plot/world) failed. Halting generation.")
        sys.exit(1)

    await prepopulate_kg_if_needed(agent)

    print("\n--- Starting Novel Writing Process ---")
    start_chapter = agent.chapter_count + 1
    
    if config.CHAPTERS_PER_RUN <= 0:
        end_chapter = start_chapter 
    else:
        end_chapter = start_chapter + config.CHAPTERS_PER_RUN

    print(f"Current Chapter Count (from DB at agent init): {agent.chapter_count}")
    if start_chapter < end_chapter:
        print(f"Targeting Chapters: {start_chapter} to {end_chapter - 1} in this run.")
        logger.info(f"Starting chapter writing loop from chapter {start_chapter} up to (but not including) {end_chapter}.")
    else:
        print(f"CHAPTERS_PER_RUN is configured to {config.CHAPTERS_PER_RUN}. No new chapters will be written in this run.")
        logger.info(f"CHAPTERS_PER_RUN is {config.CHAPTERS_PER_RUN}, skipping chapter writing loop.")

    chapters_successfully_written = 0
    for i in range(start_chapter, end_chapter):
        print(f"\n--- Attempting Chapter {i} ---")
        logger.info(f"--- Starting Generation for Chapter {i} ---")
        try:
            chapter_text = await agent.write_chapter(i)
            if chapter_text:
                chapters_successfully_written += 1
                print(f"Chapter {i}: Successfully generated and saved (Length: {len(chapter_text)} chars).")
                # Display a snippet of the generated chapter
                snippet_lines = chapter_text.splitlines()
                snippet = ' '.join(line.strip() for line in snippet_lines if line.strip())[:250]
                print(f"Chapter {i} Snippet: {snippet}...")
                logger.info(f"--- Successfully completed Chapter {i} ---")
            else:
                # write_chapter should log its own errors. This is a fallback.
                print(f"Chapter {i}: Failed to generate or save. Check logs for details.")
                logger.error(f"Chapter {i} generation failed or returned no text. See previous log messages for reasons.")
                # Decide if we should break the loop on failure. For now, try next chapter.
        except Exception as e:
            logger.critical(f"Critical error during chapter {i} writing process: {e}", exc_info=True)
            print(f"\n!!! CRITICAL ERROR during chapter {i} writing: {e} !!! Halting generation for this run.", file=sys.stderr)
            break # Halt this run if a chapter writing process has an unhandled exception

    print(f"\n--- Novel writing process finished for this run ---")
    # agent.chapter_count should be updated within write_chapter and saved to JSON state there.
    # For final verification, we could reload it from DB or rely on the in-memory agent state.
    # The agent's in-memory chapter_count is updated when a chapter is successfully written.
    final_chapter_count_in_agent = agent.chapter_count
    
    # Optionally, reload from DB for absolute certainty for the final message
    final_chapter_count_from_db = await agent.db_manager.async_load_chapter_count()

    print(f"Successfully wrote {chapters_successfully_written} chapter(s) in this run.")
    print(f"Agent's in-memory chapter count after run: {final_chapter_count_in_agent}")
    print(f"Current total chapters in database: {final_chapter_count_from_db}")
    print(f"Check the '{config.BASE_OUTPUT_DIR}' directory for JSON state files, chapter text files, logs, and the database ('{os.path.basename(config.DATABASE_FILE)}').")
    logger.info(f"--- Saga Novel Generation Run Finished. Final chapter count (DB): {final_chapter_count_from_db} ---")


if __name__ == "__main__":
    setup_logging() # Setup logging first
    if RUN_WITH_ASYNCIO_RUN:
        try:
            asyncio.run(run_novel_generation_async())
        except KeyboardInterrupt:
            logging.getLogger(__name__).warning("Novel generation process interrupted by user (KeyboardInterrupt).")
            print("\nProcess interrupted by user. Exiting.")
        except Exception as main_err: # Catch any other unexpected errors from the async run
            logging.getLogger(__name__).critical(f"Unhandled exception in main async execution: {main_err}", exc_info=True)
            print(f"\nFATAL UNHANDLED EXCEPTION: {main_err}. Check logs.", file=sys.stderr)
            sys.exit(1)
    else:
        # This case is for when main.py might be imported and run_novel_generation_async()
        # is called by an existing asyncio event loop.
        # For direct execution, RUN_WITH_ASYNCIO_RUN = True is expected.
        logger = logging.getLogger(__name__)
        logger.warning("RUN_WITH_ASYNCIO_RUN is False. "
                       "This script expects 'run_novel_generation_async()' to be called from an existing event loop if not run directly.")
        print("Script not run with asyncio.run(). If you intended to run the novel generation, "
              "ensure RUN_WITH_ASYNCIO_RUN is True or call run_novel_generation_async() from an event loop.")