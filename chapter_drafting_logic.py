# chapter_drafting_logic.py
"""
DEPRECATED: The logic for chapter drafting has been streamlined and moved
directly into `drafting_agent.py`. This file is kept to avoid import errors
in older parts of the system but should not be used for new development.
"""

import logging

logger = logging.getLogger(__name__)

logger.warning(
    "The 'chapter_drafting_logic.py' module is deprecated. "
    "Drafting logic is now handled by the DraftingAgent class in 'drafting_agent.py'."
)
