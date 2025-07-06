# data_access/utils/world_utils.py
"""Utility functions for world data processing and manipulation."""

from typing import Any

import kg_constants as kg_keys
import structlog
from config import settings
from kg_maintainer.models import (
    WorldItem,  # If WorldItem is used by utils, otherwise remove
)

import utils  # Assuming this is the main project utils

logger = structlog.get_logger(__name__)

# This global will be managed by the WorldQueryService or a dedicated cache manager
WORLD_NAME_TO_ID_CACHE: dict[str, str] = {}


def normalize_for_id(text: str) -> str:
    """Wraps the main utils._normalize_for_id for specific use in world data if needed,
    or just rely on direct usage of utils._normalize_for_id.
    For now, let's assume direct usage of utils._normalize_for_id is fine.
    If specific world-related normalization was different, it would go here.
    """
    return utils._normalize_for_id(text)


def build_world_element_id(
    category: str, item_name: str, details: dict[str, Any]
) -> str:
    """
    Return a stable ID for a world element.
    Moved from data_access/world_queries.py
    """
    if isinstance(details.get("id"), str) and details.get("id", "").strip():
        return str(details["id"])
    norm_cat = normalize_for_id(category) or "unknown_category"
    norm_name = normalize_for_id(item_name) or "unknown_name"
    return f"{norm_cat}_{norm_name}"


def resolve_world_name_from_cache(name: str) -> str | None:
    """
    Return canonical world item ID for a display name if known in the cache.
    Moved from data_access/world_queries.py (original: resolve_world_name)
    The cache (WORLD_NAME_TO_ID_CACHE) will be populated by the WorldQueryService.
    """
    if not name:
        return None
    return WORLD_NAME_TO_ID_CACHE.get(normalize_for_id(name))


def update_world_name_to_id_cache(item_name: str, item_id: str) -> None:
    """Updates the local WORLD_NAME_TO_ID_CACHE."""
    if item_name and item_id:
        WORLD_NAME_TO_ID_CACHE[normalize_for_id(item_name)] = item_id


def clear_world_name_to_id_cache() -> None:
    """Clears the local WORLD_NAME_TO_ID_CACHE."""
    WORLD_NAME_TO_ID_CACHE.clear()


def get_world_item_by_name_from_data(
    world_data: dict[str, dict[str, WorldItem]], name: str
) -> WorldItem | None:
    """
    Retrieve a WorldItem from pre-loaded world_data using a fuzzy name lookup via cache.
    Moved from data_access/world_queries.py (original: get_world_item_by_name)
    """
    item_id = resolve_world_name_from_cache(name)
    if not item_id:
        return None
    for items_in_category in world_data.values():
        if not isinstance(items_in_category, dict):
            continue
        for item in items_in_category.values():
            if isinstance(item, WorldItem) and item.id == item_id:
                return item
    return None


# --- Data processing helpers ---


def process_elaborations_for_item(
    elaborations: list[dict[str, Any]],
    chapter_limit: int | None,
    item_detail_dict: dict[str, Any],
) -> int:
    """
    Apply elaboration records to ``item_detail_dict`` and return the count of applied elaborations.
    Moved from data_access/world_queries.py (_process_elaborations)
    """
    count = 0
    for elab_rec in elaborations:
        chapter_val = elab_rec.get("chapter")
        summary_val = elab_rec.get("summary")
        if chapter_val is None or summary_val is None:
            continue
        if chapter_limit is not None and chapter_val > chapter_limit:
            continue

        elab_key = kg_keys.elaboration_key(chapter_val)  # kg_constants
        item_detail_dict[elab_key] = summary_val

        if elab_rec.get("prov"):  # Check for provisional flag from DB
            # Use KG_IS_PROVISIONAL from kg_constants for consistency if it's the direct field name
            # For now, assuming "prov" is the direct boolean field from the query
            item_detail_dict[
                kg_keys.source_quality_key(chapter_val)
            ] = (  # kg_constants
                "provisional_from_unrevised_draft"
            )
        count += 1
    return count


def extract_core_world_element_fields(
    we_node: dict[str, Any],
) -> tuple[str | None, str | None, str | None]:
    """
    Extract category, item_name, and we_id from the world element node.
    Moved from data_access/world_queries.py (_extract_core_we_fields)
    """
    category = we_node.get("category")
    item_name = we_node.get("name")
    we_id = we_node.get("id")
    if not all([category, item_name, we_id]):
        # Logger is available here if needed, or this function can raise an error/return sentinel
        logger.debug(  # Changed to debug to be less noisy if this is common for partial data
            "Skipping WorldElement in util function due to missing core fields (id, name, or category): %s",
            we_node.get("id", "N/A"),
        )
        return None, None, None
    return category, item_name, we_id


def initialize_item_detail_dict_from_node(
    we_node: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    """
    Initialize item_detail_dict from we_node and process creation chapter.
    Moved from data_access/world_queries.py (_initialize_item_detail_dict)
    """
    item_detail_dict = dict(we_node)
    item_detail_dict.pop("created_ts", None)
    item_detail_dict.pop("updated_ts", None)

    # Use constants from kg_keys
    created_chapter_num_val = item_detail_dict.pop(
        kg_keys.KG_NODE_CREATED_CHAPTER, settings.KG_PREPOPULATION_CHAPTER_NUM
    )
    created_chapter_num = int(created_chapter_num_val)  # Ensure it's an int
    item_detail_dict["created_chapter"] = created_chapter_num
    item_detail_dict[kg_keys.added_key(created_chapter_num)] = True

    # Handle provisional status consistently
    is_provisional_at_creation = item_detail_dict.pop(kg_keys.KG_IS_PROVISIONAL, False)
    item_detail_dict["is_provisional"] = bool(
        is_provisional_at_creation
    )  # Ensure boolean
    if is_provisional_at_creation:
        item_detail_dict[kg_keys.source_quality_key(created_chapter_num)] = (
            "provisional_from_unrevised_draft"
        )
    return item_detail_dict, created_chapter_num


def populate_list_attributes_for_item(
    record: dict[str, Any], item_detail_dict: dict[str, Any]
) -> None:
    """
    Populate list attributes (goals, rules, key_elements, traits) in item_detail_dict.
    Moved from data_access/world_queries.py (_populate_list_attributes)
    """
    list_attrs = ["goals", "rules", "key_elements", "traits"]
    for attr in list_attrs:
        # Ensure values are strings and filter out None before sorting
        values = record.get(attr)
        if not values:
            values = []
        attr_values = [str(v) for v in values if v is not None]
        item_detail_dict[attr] = sorted(attr_values)


def should_include_world_item(
    created_chapter_num: int,
    actual_elaborations_count: int,  # Number of elaborations *within the chapter_limit*
    chapter_limit: int | None,
    item_name: str,
    we_id: str,
) -> bool:
    """
    Determine if the world item should be included based on chapter limits and relevant elaborations.
    Moved from data_access/world_queries.py (_should_include_world_item)
    """
    if chapter_limit is None:
        return True  # No limit, include all
    if created_chapter_num <= chapter_limit:
        return True  # Created within limit
    # If created after limit, only include if it has relevant elaborations up to the limit
    if actual_elaborations_count > 0:
        return True

    logger.debug(
        "WorldElement '%s' (id: %s) created in chapter %s with no elaborations up to chapter %s, excluding.",
        item_name,
        we_id,
        created_chapter_num,
        chapter_limit,
    )
    return False


# Default categories, moved from world_queries
DEFAULT_WORLD_CATEGORIES = [
    "locations",
    "society",
    "systems",
    "lore",
    "history",
    "factions",
]

# Ensure WorldItem is imported if used in type hints here.
# from kg_maintainer.models import WorldItem
# It's used by get_world_item_by_name_from_data
# Ensure kg_constants are imported as kg_keys
# import kg_constants as kg_keys
# Ensure settings are imported
# from config import settings
# Ensure main utils are imported
# import utils
