"""Utilities for detecting and removing duplicate text segments."""

from __future__ import annotations

import logging

import utils
from config import settings

logger = logging.getLogger(__name__)


class TextDeduplicator:
    """Detects duplicate text segments using ``utils.deduplicate_text_segments``."""

    def __init__(
        self,
        similarity_threshold: float = settings.DEDUPLICATION_SEMANTIC_THRESHOLD,
        use_semantic_comparison: bool = settings.DEDUPLICATION_USE_SEMANTIC,
        min_segment_length_chars: int = settings.DEDUPLICATION_MIN_SEGMENT_LENGTH,
        prefer_newer: bool = False,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.use_semantic_comparison = use_semantic_comparison
        self.min_segment_length_chars = min_segment_length_chars
        self.prefer_newer = prefer_newer

    async def deduplicate(
        self, original_text: str, segment_level: str = "paragraph"
    ) -> tuple[str, int]:
        """Remove near-duplicate segments from ``original_text``."""
        return await utils.deduplicate_text_segments(
            original_text,
            segment_level=segment_level,
            similarity_threshold=self.similarity_threshold,
            use_semantic_comparison=self.use_semantic_comparison,
            min_segment_length_chars=self.min_segment_length_chars,
            prefer_newer=self.prefer_newer,
        )
