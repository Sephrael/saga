# kg_constants.py
"""Constants used for property names and the canonical schema in the knowledge graph."""

from config import settings

# Chapter-based property prefixes
ELABORATION_PREFIX = settings.ELABORATION_PREFIX
DEVELOPMENT_PREFIX = settings.DEVELOPMENT_PREFIX
SOURCE_QUALITY_PREFIX = settings.SOURCE_QUALITY_PREFIX
ADDED_PREFIX = settings.ADDED_PREFIX
UPDATED_PREFIX = settings.UPDATED_PREFIX

# Relationship and node property names
KG_REL_CHAPTER_ADDED = settings.KG_REL_CHAPTER_ADDED
KG_NODE_CREATED_CHAPTER = settings.KG_NODE_CREATED_CHAPTER
KG_NODE_CHAPTER_UPDATED = settings.KG_NODE_CHAPTER_UPDATED
KG_IS_PROVISIONAL = settings.KG_IS_PROVISIONAL

# --- Canonical Schema Definition ---

# Set of known node labels used in the knowledge graph.
# The KGMaintainerAgent will be instructed to use these labels.
# The base_db_manager ensures these label tokens exist on startup.
NODE_LABELS = {
    "Entity",  # Base label for all nodes
    "NovelInfo",
    "Chapter",
    "Character",
    "WorldElement",
    "WorldContainer",
    "PlotPoint",
    "Trait",
    "ValueNode",  # For literal-like values that need to be nodes
    "DevelopmentEvent",
    "WorldElaborationEvent",
    # Add other entity types as they become concepts, e.g., "Item", "Organization", "Concept"
    "Location",
    "Faction",
    "System",  # e.g. Magic System
    "Lore",
    "History",
}


# Set of known relationship types used in the knowledge graph. This helps
# normalize variations and informs downstream consumers of valid types.
RELATIONSHIP_TYPES = {
    # Structural Relationships
    "HAS_PLOT_POINT",
    "NEXT_PLOT_POINT",
    "HAS_CHARACTER",
    "HAS_WORLD_META",
    "CONTAINS_ELEMENT",
    # Character-related Relationships
    "HAS_TRAIT",
    "DEVELOPED_IN_CHAPTER",
    "KNOWS",
    "ALLY_OF",
    "ENEMY_OF",
    "FRIEND_OF",
    "RIVAL_OF",
    "MENTOR_OF",
    "PROTEGE_OF",
    "WORKS_FOR",
    "MEMBER_OF",
    "LOVES",
    "HATES",
    "IS_DEAD",
    "IS_REMEMBERED_AS",
    "WAS_FRIEND_OF",
    # WorldElement-related Relationships
    "HAS_GOAL",
    "HAS_RULE",
    "HAS_KEY_ELEMENT",
    "HAS_TRAIT_ASPECT",
    "ELABORATED_IN_CHAPTER",
    # Generic & Dynamic Relationships
    "IS_A",
    "PART_OF",
    "LOCATED_IN",
    "NEAR",
    "HAS_STATUS",
    "HAS_ABILITY",
    "OWNS",
    "RELATED_TO",  # Generic fallback
    "DYNAMIC_REL",  # For KG triples where the relationship type is in a property
}


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
