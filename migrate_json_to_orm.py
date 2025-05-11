#!/usr/bin/env python3
"""
Script to migrate data from JSON files to the ORM database.
This script uses the migrate_from_json method in the StateManager class
to migrate the data from JSON files to the ORM database.

Usage:
    python migrate_json_to_orm.py

This script will:
1. Load data from JSON files (plot_outline.json, character_profiles.json, world_building.json)
2. Convert the data to ORM models
3. Save the data to the database
"""

import logging
import os
import sys
from state_manager import state_manager
import config

# Configure logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format=config.LOG_FORMAT,
    datefmt=config.LOG_DATE_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.LOG_FILE) if config.LOG_FILE else logging.NullHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main function to run the migration."""
    logger.info("Starting migration from JSON files to ORM database...")
    
    # Check if JSON files exist
    json_files = [
        config.PLOT_OUTLINE_FILE,
        config.CHARACTER_PROFILES_FILE,
        config.WORLD_BUILDER_FILE
    ]
    
    missing_files = [f for f in json_files if not os.path.exists(f)]
    if missing_files:
        logger.warning(f"The following JSON files are missing: {', '.join(missing_files)}")
        if len(missing_files) == len(json_files):
            logger.error("All JSON files are missing. Nothing to migrate.")
            return
    
    # Run the migration
    try:
        state_manager.migrate_from_json()
        logger.info("Migration completed successfully.")
    except Exception as e:
        logger.error(f"Error during migration: {e}", exc_info=True)
        return
    
    # Verify the migration
    logger.info("Verifying migration...")
    plot_outline = state_manager.get_plot_outline()
    character_profiles = state_manager.get_character_profiles()
    world_building = state_manager.get_world_building()
    
    if plot_outline:
        logger.info(f"Plot outline migrated successfully. Contains {len(plot_outline)} keys.")
    else:
        logger.warning("Plot outline not found in ORM database after migration.")
    
    if character_profiles:
        logger.info(f"Character profiles migrated successfully. Contains {len(character_profiles)} characters.")
    else:
        logger.warning("Character profiles not found in ORM database after migration.")
    
    if world_building:
        logger.info(f"World building migrated successfully. Contains {len(world_building)} elements.")
    else:
        logger.warning("World building not found in ORM database after migration.")
    
    logger.info("Migration verification completed.")

if __name__ == "__main__":
    main()
