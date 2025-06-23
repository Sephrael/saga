"""Utilities for constructing and parsing chapter-based KG property keys."""

from typing import Optional

ELABORATION_PREFIX = "elaboration_in_chapter_"
DEVELOPMENT_PREFIX = "development_in_chapter_"
SOURCE_QUALITY_PREFIX = "source_quality_chapter_"
ADDED_PREFIX = "added_in_chapter_"
UPDATED_PREFIX = "updated_in_chapter_"


def elaboration_key(chapter: int) -> str:
    """Return the elaboration key for ``chapter``."""
    return f"{ELABORATION_PREFIX}{chapter}"


def development_key(chapter: int) -> str:
    """Return the development key for ``chapter``."""
    return f"{DEVELOPMENT_PREFIX}{chapter}"


def source_quality_key(chapter: int) -> str:
    """Return the source quality key for ``chapter``."""
    return f"{SOURCE_QUALITY_PREFIX}{chapter}"


def added_key(chapter: int) -> str:
    """Return the added key for ``chapter``."""
    return f"{ADDED_PREFIX}{chapter}"


def updated_key(chapter: int) -> str:
    """Return the updated key for ``chapter``."""
    return f"{UPDATED_PREFIX}{chapter}"


def parse_elaboration_key(key: str) -> Optional[int]:
    """Return the chapter from an elaboration key if parsable."""
    if key.startswith(ELABORATION_PREFIX):
        suffix = key[len(ELABORATION_PREFIX) :]
        if suffix.isdigit():
            return int(suffix)
    return None
