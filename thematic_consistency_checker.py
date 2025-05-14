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