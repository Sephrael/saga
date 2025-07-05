# orchestration/services/deduplication_service.py
"""Service for performing text deduplication."""

import structlog

from config import settings
from processing.text_deduplicator import TextDeduplicator

logger = structlog.get_logger(__name__)

class DeduplicationService:
    def __init__(self):
        # Potentially initialize the TextDeduplicator here if its settings are static
        # or allow passing a pre-configured instance.
        # For now, creating it on-the-fly in the method.
        pass

    async def perform_deduplication(
        self, text_to_dedup: str, chapter_number: int, context_description: str = "general"
    ) -> tuple[str, int]:
        """
        Performs de-duplication on the given text.

        Args:
            text_to_dedup: The text string to de-duplicate.
            chapter_number: The chapter number, for logging purposes.
            context_description: A string describing the context of this deduplication call (e.g., "post-draft", "pre-evaluation").

        Returns:
            A tuple containing the de-duplicated text and the number of characters removed.
        """
        logger.info(
            f"DeduplicationService: Performing de-duplication for Chapter {chapter_number} ({context_description})..."
        )
        if not text_to_dedup or not text_to_dedup.strip():
            logger.info(
                f"De-duplication for Chapter {chapter_number} ({context_description}): Input text is empty. No action taken."
            )
            return text_to_dedup, 0
        try:
            deduper = TextDeduplicator(
                similarity_threshold=settings.DEDUPLICATION_SEMANTIC_THRESHOLD,
                use_semantic_comparison=settings.DEDUPLICATION_USE_SEMANTIC,
                min_segment_length_chars=settings.DEDUPLICATION_MIN_SEGMENT_LENGTH,
            )
            deduplicated_text, chars_removed = await deduper.deduplicate(
                text_to_dedup, segment_level="sentence"
            )
            if chars_removed > 0:
                method = (
                    "semantic"
                    if settings.DEDUPLICATION_USE_SEMANTIC
                    else "normalized string"
                )
                logger.info(
                    f"De-duplication for Chapter {chapter_number} ({context_description}) removed {chars_removed} characters using {method} matching."
                )
            else:
                logger.info(
                    f"De-duplication for Chapter {chapter_number} ({context_description}): No significant duplicates found."
                )
            return deduplicated_text, chars_removed
        except Exception as e:
            logger.error(
                f"Error during de-duplication for Chapter {chapter_number} ({context_description}): {e}",
                exc_info=True,
            )
            return text_to_dedup, 0

    # If there were other helper methods specifically for deduplication, they would go here.
    # For example, if _save_debug_output was only used by deduplication,
    # it might be moved here, but it seems more general.
    # The Orchestrator's _save_debug_output can be called from the orchestrator itself
    # after getting the results from this service.
