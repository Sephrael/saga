"""Utilities for constructing and parsing chapter-based KG property keys."""

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


def parse_elaboration_key(key: str) -> int | None:
    """Return the chapter from an elaboration key if parsable."""
    if key.startswith(ELABORATION_PREFIX):
        suffix = key[len(ELABORATION_PREFIX) :]
        if suffix.isdigit():
            return int(suffix)
    return None


def parse_development_key(key: str) -> int | None:
    """Return the chapter from a development key if parsable."""
    if key.startswith(DEVELOPMENT_PREFIX):
        suffix = key[len(DEVELOPMENT_PREFIX) :]
        if suffix.isdigit():
            return int(suffix)
    return None


def parse_source_quality_key(key: str) -> int | None:
    """Return the chapter from a source quality key if parsable."""
    if key.startswith(SOURCE_QUALITY_PREFIX):
        suffix = key[len(SOURCE_QUALITY_PREFIX) :]
        if suffix.isdigit():
            return int(suffix)
    return None


def parse_added_key(key: str) -> int | None:
    """Return the chapter from an added key if parsable."""
    if key.startswith(ADDED_PREFIX):
        suffix = key[len(ADDED_PREFIX) :]
        if suffix.isdigit():
            return int(suffix)
    return None


def parse_updated_key(key: str) -> int | None:
    """Return the chapter from an updated key if parsable."""
    if key.startswith(UPDATED_PREFIX):
        suffix = key[len(UPDATED_PREFIX) :]
        if suffix.isdigit():
            return int(suffix)
    return None
