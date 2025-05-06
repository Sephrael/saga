# main.py
"""
Main execution script for the Saga Novel Generation system.
Initializes logging, creates the NovelWriterAgent, ensures necessary
setup (plot outline, world-building), and runs the chapter generation loop.
"""

import logging
import sys
import os # Needed for file handler path

# Import necessary components from other modules
from novel_logic import NovelWriterAgent
import config # To setup logging and access defaults/constants

def setup_logging():
    """Configures logging based on settings in config.py"""
    log_level_name = config.LOG_LEVEL.upper()
    log_level = getattr(logging, log_level_name, logging.INFO) # Default to INFO if invalid level name

    # Base configuration for logging (console)
    logging.basicConfig(
        level=log_level,
        format=config.LOG_FORMAT,
        datefmt='%Y-%m-%d %H:%M:%S' # Added date format
    )

    # --- Optional File Handler ---
    if config.LOG_FILE:
        try:
            # Ensure log directory exists
            log_dir = os.path.dirname(config.LOG_FILE)
            if log_dir:
                 os.makedirs(log_dir, exist_ok=True)

            # Create file handler
            file_handler = logging.FileHandler(config.LOG_FILE, mode='a', encoding='utf-8') # Append mode
            file_handler.setLevel(log_level) # Set same level or different if needed
            # Create formatter and add it to the handler
            formatter = logging.Formatter(config.LOG_FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
            file_handler.setFormatter(formatter)
            # Add the handler to the root logger
            logging.getLogger().addHandler(file_handler)
            logging.info(f"File logging enabled. Log file: {config.LOG_FILE}")
        except Exception as e:
            logging.error(f"Failed to configure file logging to {config.LOG_FILE}: {e}", exc_info=True)
            print(f"Warning: Could not set up log file at '{config.LOG_FILE}'. Error: {e}", file=sys.stderr)

    logging.info(f"Logging setup complete. Level: {log_level_name}")

def run_novel_generation():
    """
    Main function to initialize the agent and orchestrate the novel writing process.
    """
    # Setup logging as the first step
    setup_logging()
    # Get a logger specific to this main module
    logger = logging.getLogger(__name__)

    logger.info("--- Starting Saga Novel Generation ---")

    # --- Initialize Agent ---
    try:
        # Creating an instance of the agent automatically loads state and initializes DB
        agent = NovelWriterAgent()
        logger.info("NovelWriterAgent initialized successfully.")
    except Exception as e:
        # Catch critical errors during initialization (e.g., DB connection failure)
        logger.critical(f"Failed to initialize NovelWriterAgent: {e}", exc_info=True)
        # Print error to console as logging might not be fully working if init failed early
        print(f"\nFATAL: Could not initialize the agent. Check logs. Error: {e}", file=sys.stderr)
        sys.exit(1) # Exit if agent cannot be initialized

    # --- Step 1: Ensure Plot Outline Exists ---
    logger.info("Checking for existing plot outline...")
    # Check if the loaded outline seems like a default or empty one
    is_default_outline = (
        not agent.plot_outline or # No outline loaded
        agent.plot_outline.get('title') == "Untitled Novel" or # Default title
        not agent.plot_outline.get("plot_points") or # Missing plot points key
        # Check if plot_points list exists, is not empty, and contains the default text
        (isinstance(agent.plot_outline.get("plot_points"), list) and
         len(agent.plot_outline["plot_points"]) > 0 and
         "Default Point 1" in agent.plot_outline["plot_points"][0]) or
        # Check if plot points list has fewer than the expected number (e.g., 5)
        len(agent.plot_outline.get("plot_points", [])) < 5
    )

    if is_default_outline:
        print("\n--- Generating New Plot Outline ---")
        logger.info("No valid plot outline found or outline appears default. Generating new one.")
        try:
            # Generate outline using defaults from config file
            outline = agent.generate_plot_outline(
                genre=config.DEFAULT_GENRE,
                theme=config.DEFAULT_THEME,
                protagonist=config.DEFAULT_PROTAGONIST
            )
            print(f"Generated Outline for: '{outline.get('title', 'N/A')}'")
            logger.info(f"Generated plot outline titled: '{outline.get('title', 'N/A')}'")
        except Exception as e:
            # Handle critical errors during outline generation
            logger.critical(f"Critical error during plot outline generation: {e}", exc_info=True)
            print(f"\nFATAL: Error generating plot outline: {e}. Cannot continue.", file=sys.stderr)
            sys.exit(1) # Exit if outline generation fails
    else:
        # Use the existing outline loaded by the agent
        print("\n--- Using Existing Plot Outline ---")
        print(f"Loaded outline for: '{agent.plot_outline.get('title', 'N/A')}'")
        logger.info(f"Using existing plot outline: '{agent.plot_outline.get('title', 'N/A')}'")

    # --- Step 2: Ensure World Building Exists ---
    logger.info("Checking for existing world-building data...")
    # Check if world-building looks minimal/default
    is_default_world = (
        not agent.world_building or # No world building loaded
        # Check if only the default location exists and few other keys
        ("Default Location" in agent.world_building.get("locations", {}) and len(agent.world_building.keys()) <= 2)
    )

    if is_default_world:
        print("\n--- Generating Initial World-Building Data ---")
        logger.info("World-building data appears default or missing. Generating initial data based on plot outline.")
        try:
            # Generate world-building based on the current plot outline
            agent.generate_world_building()
            print("Generated/Refreshed initial world-building data.")
            logger.info("Initial world-building data generation complete.")
        except Exception as e:
            # Log error but allow proceeding with potentially default world-building
            logger.error(f"Error generating world building: {e}", exc_info=True)
            print(f"\nWarning: Error generating world building: {e}. Proceeding with potentially default data.")
    else:
        # Use existing world-building data
        print("\n--- Using Existing World-Building Data ---")
        logger.info("Using existing world-building data.")

    # --- Step 3: Write Chapters ---
    print("\n--- Starting Novel Writing Process ---")
    # Determine the starting chapter number based on the loaded count
    start_chapter = agent.chapter_count + 1
    # Determine the end chapter number based on how many chapters to write per run (from config)
    # The loop will go up to (end_chapter - 1)
    end_chapter = start_chapter + config.CHAPTERS_PER_RUN

    print(f"Current Chapter Count (from DB): {agent.chapter_count}")
    if config.CHAPTERS_PER_RUN > 0:
        print(f"Targeting Chapters: {start_chapter} to {end_chapter - 1} in this run.")
        logger.info(f"Starting chapter writing loop from {start_chapter} to {end_chapter - 1}.")
    else:
        print("CHAPTERS_PER_RUN is set to 0 in config. No chapters will be written.")
        logger.info("CHAPTERS_PER_RUN is 0, skipping chapter writing loop.")

    # Loop through the target chapter numbers for this run
    for i in range(start_chapter, end_chapter):
        print(f"\n--- Attempting Chapter {i} ---")
        logger.info(f"--- Starting Generation for Chapter {i} ---")
        try:
            # Call the agent's method to write the chapter
            # This method encapsulates generation, validation, revision, updates, and saving
            chapter_text = agent.write_chapter(i)

            # Check if chapter writing was successful
            if chapter_text:
                # Print success message and snippet to console
                print(f"Chapter {i}: Successfully generated and saved (Length: {len(chapter_text)} chars).")
                # Provide a clean snippet without extra internal newlines for console readability
                snippet = ' '.join(chapter_text[:250].splitlines()).strip()
                print(f"Chapter {i} Snippet: {snippet}...")
                logger.info(f"--- Successfully completed Chapter {i} ---")
            else:
                # Print failure message to console
                print(f"Chapter {i}: Failed to generate or save. Check logs for details.")
                logger.error(f"Chapter {i} generation failed. See previous log messages for reasons.")
                # Optional: Decide whether to stop the entire run on a single chapter failure
                # print("Halting generation due to chapter failure.")
                # break # Uncomment to stop on failure

        except Exception as e:
            # Catch any unexpected critical errors during the writing of a specific chapter
            logger.critical(f"Critical error during chapter {i} writing process: {e}", exc_info=True)
            print(f"\n!!! Critical Error during chapter {i} writing: {e} !!! Halting generation.", file=sys.stderr)
            break # Stop the loop immediately on critical errors

    # --- Completion ---
    print(f"\n--- Novel writing process finished for this run ---")
    # Display the final chapter count after the run
    print(f"Final Agent Chapter Count (in DB): {agent.chapter_count}")
    print(f"Check the '{config.OUTPUT_DIR}' directory for JSON state files, chapter text files, logs, and the database ('{os.path.basename(config.DATABASE_FILE)}').")
    logger.info(f"--- Saga Novel Generation Run Finished. Final chapter count: {agent.chapter_count} ---")

# Standard Python entry point check
if __name__ == "__main__":
    run_novel_generation()
