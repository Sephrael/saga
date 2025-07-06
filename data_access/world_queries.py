# data_access/world_queries.py
"""
Facade for world data access, delegating to specific services.
This module will be significantly slimmed down.
"""

from typing import Any

import structlog
from kg_maintainer.models import WorldItem  # Required for type hints

from .services.world_persistence_service import WorldPersistenceService

# Import services
from .services.world_query_service import WorldQueryService
from .utils import (
    world_utils,  # For direct access to cache utilities if needed by facade users
)

logger = structlog.get_logger(__name__)

# Initialize services. These could be singletons or managed elsewhere if preferred.
# For now, direct instantiation. WorldPersistenceService might need WorldQueryService.
_world_query_service = WorldQueryService()
_world_persistence_service = WorldPersistenceService(
    world_query_service=_world_query_service
)


# --- Public API exposed by this module ---


async def get_world_building_from_db(
    chapter_limit: int | None = None,
) -> dict[str, dict[str, WorldItem]]:
    """
    Delegates to WorldQueryService to load world elements grouped by category.
    """
    return await _world_query_service.get_world_building_data(chapter_limit)


async def get_world_item_by_id(item_id: str) -> WorldItem | None:
    """
    Delegates to WorldQueryService to retrieve a single WorldItem by its ID.
    """
    return await _world_query_service.get_world_item_by_id(item_id)


async def get_all_world_item_ids_by_category() -> dict[str, list[str]]:
    """
    Delegates to WorldQueryService to get all world item IDs grouped by category.
    """
    return await _world_query_service.get_all_world_item_ids_by_category()


async def get_world_elements_for_snippet_from_db(
    category: str, chapter_limit: int, item_limit: int
) -> list[dict[str, Any]]:
    """
    Delegates to WorldQueryService to get a subset of world elements for prompt context.
    """
    return await _world_query_service.get_world_elements_for_snippet(
        category, chapter_limit, item_limit
    )


async def find_thin_world_elements_for_enrichment() -> list[dict[str, Any]]:
    """
    Delegates to WorldQueryService to find 'thin' WorldElement nodes.
    """
    return await _world_query_service.find_thin_world_elements_for_enrichment()


async def sync_world_items(
    world_items: dict[str, dict[str, WorldItem]],  # This is dict of WorldItem objects
    chapter_number: int,
    full_sync: bool = False,
) -> bool:
    """
    Persist world element data to Neo4j.
    Delegates to WorldPersistenceService.
    If full_sync, converts WorldItem objects to dicts first.
    """
    if full_sync:
        # Convert WorldItem objects to dictionaries for full sync
        world_data_as_dict: dict[str, Any] = {}
        for cat, items_in_cat in world_items.items():
            if cat == "_overview_":  # Handle overview separately if it's a WorldItem
                if isinstance(items_in_cat, dict) and "_overview_" in items_in_cat:
                    overview_item = items_in_cat["_overview_"]
                    if isinstance(overview_item, WorldItem):
                        world_data_as_dict["_overview_"] = overview_item.to_dict()
                    elif isinstance(overview_item, dict):  # Already a dict
                        world_data_as_dict["_overview_"] = overview_item
                elif isinstance(
                    items_in_cat, WorldItem
                ):  # If _overview_ value is directly the item
                    world_data_as_dict["_overview_"] = items_in_cat.to_dict()
            elif isinstance(items_in_cat, dict):
                world_data_as_dict[cat] = {
                    name: item.to_dict() if isinstance(item, WorldItem) else item
                    for name, item in items_in_cat.items()
                }
        return await _world_persistence_service.sync_full_world_state_to_db(
            world_data_as_dict
        )
    else:
        return await _world_persistence_service.sync_world_items_incremental(
            world_items, chapter_number
        )


async def sync_full_state_from_object_to_db(world_data_as_dict: dict[str, Any]) -> bool:
    """
    Persist the entire world-building state (already as dicts) to Neo4j.
    Delegates to WorldPersistenceService.
    """
    # This function assumes world_data_as_dict is already in the correct dictionary format.
    return await _world_persistence_service.sync_full_world_state_to_db(
        world_data_as_dict
    )


async def fix_missing_world_element_core_fields() -> int:
    """
    Delegates to WorldPersistenceService to populate missing core fields.
    """
    return (
        await _world_persistence_service.fix_missing_world_element_core_fields_in_db()
    )


async def remove_world_element_trait_aspect(element_id: str, trait_value: str) -> bool:
    """
    Delegates to WorldPersistenceService to remove a trait aspect.
    """
    return await _world_persistence_service.remove_world_element_trait_aspect_from_db(
        element_id, trait_value
    )


# Utility functions from world_utils that might be exposed directly if needed by other modules
# For now, assume users of world_queries will use the service methods above,
# and services will use world_utils internally.
# If resolve_world_name or get_world_item_by_name (from data) are still needed externally,
# they can call the world_utils versions.


def resolve_world_name(name: str) -> str | None:
    """
    Resolves a world item name to its ID using the utility cache.
    Note: The cache in world_utils is populated by service calls like get_world_building_from_db.
    """
    return world_utils.resolve_world_name_from_cache(name)


def get_world_item_by_name(
    world_data: dict[str, dict[str, WorldItem]], name: str
) -> WorldItem | None:
    """
    Retrieves a WorldItem from pre-loaded world_data using the utility function.
    """
    return world_utils.get_world_item_by_name_from_data(world_data, name)


# The global WORLD_NAME_TO_ID is now managed within world_utils.py (WORLD_NAME_TO_ID_CACHE)
# The DEFAULT_WORLD_CATEGORIES is now in world_utils.py
# All private helper functions (_load_world_container, _build_world_elements_query, etc.)
# have been moved into the respective services or world_utils.py.
