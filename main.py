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
    logger.info("Checking for existing plot outline...")
    
    plot_outline_data = agent.plot_outline 
    should_regenerate_plot = False

    if not plot_outline_data: 
        should_regenerate_plot = True
        logger.info("Plot outline data is empty from ORM. Will generate new.")
    elif plot_outline_data.get("is_default") is True: 
        should_regenerate_plot = True
        logger.info("Plot outline was previously marked as default. Will regenerate.")
    elif not all(k in plot_outline_data for k in ["title", "protagonist_name", "plot_points"]):
        should_regenerate_plot = True
        logger.info("Plot outline missing essential keys (title, protagonist_name, plot_points). Will regenerate.")
    else:
        logger.info(f"Using existing plot outline: '{plot_outline_data.get('title', 'N/A')}'")


    if should_regenerate_plot:
        print("\n--- Generating New Plot Outline ---")
        logger.info("Generating a new plot outline.")
        generation_params: Dict[str, Any] = {}
        if config.UNHINGED_PLOT_MODE:
            generation_params.update({
                "genre": random.choice(config.UNHINGED_GENRES), "theme": random.choice(config.UNHINGED_THEMES),
                "setting_archetype": random.choice(config.UNHINGED_SETTINGS_ARCHETYPES),
                "protagonist_archetype": random.choice(config.UNHINGED_PROTAGONIST_ARCHETYPES),
                "conflict_archetype": random.choice(config.UNHINGED_CONFLICT_TYPES)
            })
        else:
            generation_params.update({
                "genre": config.CONFIGURED_GENRE, "theme": config.CONFIGURED_THEME,
                "setting_description": config.CONFIGURED_SETTING_DESCRIPTION
            })
        try:
            outline = await agent.generate_plot_outline(
                default_protagonist_name=config.DEFAULT_PROTAGONIST_NAME,
                unhinged_mode=config.UNHINGED_PLOT_MODE,
                **generation_params
            )
            print(f"Generated Plot Outline for: '{outline.get('title', 'N/A')}'")
        except Exception as e:
            logger.critical(f"Critical error during plot outline generation: {e}", exc_info=True)
            return False 
    else:
        print("\n--- Using Existing Plot Outline ---")
        print(f"Loaded outline for: '{agent.plot_outline.get('title', 'N/A')}'")

    logger.info("Checking for existing world-building data...")
    world_building_data = agent.world_building
    should_regenerate_world = False
    if not world_building_data:
        should_regenerate_world = True
        logger.info("World-building data is empty from ORM. Will generate new.")
    elif world_building_data.get("is_default") is True:
        should_regenerate_world = True
        logger.info("World-building was previously marked as default. Will regenerate.")
    elif not world_building_data.get("locations"): 
        should_regenerate_world = True
        logger.info("World-building data seems minimal (e.g., no locations). Will regenerate.")
    else:
        logger.info("Using existing world-building data.")

    if should_regenerate_world:
        print("\n--- Generating Initial World-Building Data ---")
        try:
            await agent.generate_world_building()
            print("Generated/Refreshed initial world-building data.")
        except Exception as e:
            logger.error(f"Error generating world building: {e}. Proceeding with potentially minimal/default world data.", exc_info=True)
            # Not returning False, allowing the process to continue if plot is fine.
    else:
        print("\n--- Using Existing World-Building Data ---")
    
    return True 

async def prepopulate_kg_if_needed(agent: NovelWriterAgent):
    logger = logging.getLogger(__name__)
    
    if agent.plot_outline.get("is_default", False) or agent.world_building.get("is_default", False):
        logger.info("Skipping KG pre-population: Plot outline or world-building is default.")
        return

    if agent.chapter_count > 0:
        logger.info(f"Skipping KG pre-population: Novel already has {agent.chapter_count} chapters.")
        return

    existing_prepop_facts = await state_manager.async_query_kg(
        chapter_limit=config.KG_PREPOPULATION_CHAPTER_NUM, 
        include_provisional=True 
    )
    prepop_facts_at_zero = [f for f in existing_prepop_facts if f.get('chapter_added') == config.KG_PREPOPULATION_CHAPTER_NUM]

    if prepop_facts_at_zero:
        logger.info(f"Found {len(prepop_facts_at_zero)} existing KG triples at chapter {config.KG_PREPOPULATION_CHAPTER_NUM}. Assuming KG pre-populated.")
        return
        
    print("\n--- Pre-populating Knowledge Graph from Plot and World Data ---")
    try:
        await agent._prepopulate_knowledge_graph()
        print("Knowledge Graph pre-population step complete.")
    except Exception as e:
        logger.error(f"Error during Knowledge Graph pre-population: {e}", exc_info=True)

async def run_novel_generation_async():
    logger = logging.getLogger(__name__) 
    logger.info(f"--- Starting Saga Novel Generation (Execution Mode: Async) ---")

    try:
        # Initialize state_manager and create DB tables first
        await state_manager.create_db_and_tables()
        logger.info("state_manager initialized and database tables checked/created.")

        agent = NovelWriterAgent()
        await agent.async_init() # Load agent's state from DB
        logger.info("NovelWriterAgent initialized and state loaded successfully.")
    except Exception as e:
        logger.critical(f"Failed to initialize state_manager or NovelWriterAgent: {e}", exc_info=True)
        print(f"\nFATAL: Could not initialize. Check logs. Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not await perform_initial_setup(agent):
        logger.critical("Initial setup (plot/world) failed. Halting generation.")
        sys.exit(1)

    await prepopulate_kg_if_needed(agent)

    print("\n--- Starting Novel Writing Process ---")
    start_chapter = agent.chapter_count + 1
    end_chapter = start_chapter + config.CHAPTERS_PER_RUN if config.CHAPTERS_PER_RUN > 0 else start_chapter

    print(f"Current Chapter Count (from ORM at agent init): {agent.chapter_count}")
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
    logger.info(f"--- Saga Novel Generation Run Finished. Final DB chapter count: {final_chapter_count_from_db} ---")

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
```

**7. `state_manager.py`**
   - Comments regarding unused relationships and `drop_all` are kept as they are informational or potentially useful for development. No code changes made here based on placeholders.

**8. `type.py`**
   - `timestamp` in `Event` remains `Optional[str]`. Comments are kept as reminders for potential future evolution.

**9. `thematic_consistency_checker.py` (Implemented)**
   - Implemented a basic LLM-based thematic consistency check.

```python
# thematic_consistency_checker.py
"""
Handles thematic consistency checking for chapter drafts.
"""
import logging
import json
from typing import Optional

import config
import llm_interface

logger = logging.getLogger(__name__)

async def check_thematic_consistency_logic(agent, chapter_number: int, chapter_text: str) -> Optional[str]:
    """
    Checks the chapter text for thematic consistency against the novel's core elements.
    'agent' is an instance of NovelWriterAgent.
    Returns a string describing thematic issues, or None if consistent.
    """
    logger.info(f"Performing thematic consistency check for Chapter {chapter_number}.")

    if not chapter_text:
        logger.warning(f"Thematic consistency check skipped for Ch {chapter_number}: empty chapter text.")
        return None

    novel_theme = agent.plot_outline.get('theme', 'Not specified')
    novel_genre = agent.plot_outline.get('genre', 'Not specified')
    protagonist_arc = agent.plot_outline.get('character_arc', 'Not specified')
    protagonist_name = agent.plot_outline.get('protagonist_name', 'The Protagonist')

    # Use a snippet of the chapter text to keep the prompt manageable
    text_snippet = chapter_text[:config.THEMATIC_CONSISTENCY_CHAPTER_SNIPPET_SIZE].strip()
    if len(chapter_text) > len(text_snippet):
        text_snippet += "..."
        logger.debug(f"Using truncated text snippet for thematic check (Ch {chapter_number}, {len(text_snippet)} chars).")


    prompt = f"""/no_think
You are a Literary Analyst specializing in thematic consistency.
Your task is to evaluate if the provided Chapter Snippet aligns with the novel's established thematic elements.

**Novel's Core Thematic Elements:**
- **Genre:** {novel_genre}
- **Central Theme:** {novel_theme}
- **Protagonist ({protagonist_name})'s Arc:** {protagonist_arc}

**Chapter {chapter_number} Snippet to Analyze:**
--- BEGIN SNIPPET ---
{text_snippet}
--- END SNIPPET ---

**Analysis Instructions:**
1.  Read the Chapter Snippet carefully.
2.  Compare its content, tone, character actions/dialogue, and plot developments against the Novel's Core Thematic Elements provided above.
3.  Consider if the snippet:
    - Reinforces or subtly explores the Central Theme.
    - Adheres to the established Genre conventions and atmosphere.
    - Contributes appropriately to the Protagonist's Arc (if the protagonist is featured).
    - Introduces elements that seem out of place or contradictory to these core elements.

**Output Format (CRITICAL):**
-   If you find **significant thematic inconsistencies** or deviations, describe them clearly and concisely in 1-3 bullet points. Focus on specific examples from the snippet if possible.
-   If the Chapter Snippet **aligns well** with the thematic elements or has no significant inconsistencies, respond with the single word: **None**

**Your Analysis (or "None"):**
"""

    logger.debug(f"Calling LLM for thematic consistency check (Ch {chapter_number}). Model: {config.THEMATIC_CONSISTENCY_MODEL}")
    raw_response = await llm_interface.async_call_llm(
        model_name=config.THEMATIC_CONSISTENCY_MODEL,
        prompt=prompt,
        temperature=0.5, # Moderately deterministic for analysis
        max_tokens=config.MAX_THEMATIC_CONSISTENCY_TOKENS
    )

    if not raw_response:
        logger.warning(f"Thematic consistency check for Ch {chapter_number} received no response from LLM. Assuming consistency as fallback.")
        return None

    cleaned_response = llm_interface.clean_model_response(raw_response).strip()

    if not cleaned_response:
        logger.warning(f"Thematic consistency check for Ch {chapter_number} resulted in empty string after cleaning. Assuming consistency.")
        return None
        
    if cleaned_response.lower() == "none":
        logger.info(f"Thematic consistency check passed for Chapter {chapter_number}. No issues reported.")
        return None
    else:
        logger.warning(f"Thematic inconsistency reported for Chapter {chapter_number}:\n{cleaned_response}")
        # Add to evaluation reasons in chapter_evaluation_logic if this is to trigger revision
        # Currently, this function just returns the issues. The calling function (update_all_knowledge_bases_logic)
        # doesn't use the return value to trigger revisions.
        # If it should trigger revisions, it needs to be integrated into evaluate_chapter_draft_logic.
        # For now, it serves as a logged warning/analysis.
        # To make it trigger revision, you'd modify chapter_evaluation_logic to call this
        # and append its result to 'reasons' if not None.
        return cleaned_response # Return the issues string