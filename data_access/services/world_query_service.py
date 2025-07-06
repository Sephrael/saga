# data_access/services/world_query_service.py
"""Service for fetching world-building data from the database."""

from typing import Any

import kg_constants as kg_keys  # For KG_NODE_CREATED_CHAPTER, KG_IS_PROVISIONAL etc.
import structlog
from async_lru import alru_cache
from config import settings  # For cache sizes, default chapter numbers
from core.db_manager import neo4j_manager
from kg_maintainer.models import WorldItem

# Assuming world_utils.py is in the same directory or accessible via path
from ..utils import world_utils

logger = structlog.get_logger(__name__)


class WorldQueryService:
    def __init__(self):
        # This service might hold a reference to neo4j_manager if it were not a singleton
        # For now, direct usage of neo4j_manager is fine.
        # The WORLD_NAME_TO_ID cache is now in world_utils.py and managed there,
        # but this service will be responsible for populating it during full loads.
        pass

    async def _load_world_container_overview(self, wc_id: str) -> WorldItem | None:
        """
        Return the world container ``WorldItem`` (overview) from Neo4j.
        Moved from data_access/world_queries.py (_load_world_container)
        """
        overview_query = "MATCH (wc:WorldContainer:Entity {id: $wc_id_param}) RETURN wc"
        overview_res_list = await neo4j_manager.execute_read_query(
            overview_query, {"wc_id_param": wc_id}
        )
        if not overview_res_list or not overview_res_list[0].get("wc"):
            logger.warning("World container overview node not found for ID: %s", wc_id)
            return None

        wc_node = overview_res_list[0]["wc"]
        overview_data_dict = dict(wc_node)
        overview_data_dict.pop("created_ts", None)
        overview_data_dict.pop("updated_ts", None)

        # Handle provisional status for the overview
        if overview_data_dict.get(kg_keys.KG_IS_PROVISIONAL):
            overview_data_dict[
                kg_keys.source_quality_key(settings.KG_PREPOPULATION_CHAPTER_NUM)
            ] = "provisional_from_unrevised_draft"

        # Ensure 'id' field is present for from_dict if it expects it
        # The wc_node itself has an 'id', but it might be removed by dict(wc_node) if not careful
        overview_data_dict["id"] = wc_id

        return WorldItem.from_dict("_overview_", "_overview_", overview_data_dict)

    def _build_world_elements_query_cypher(self, chapter_filter: str) -> str:
        """
        Return the Cypher query used to fetch world elements.
        Moved from data_access/world_queries.py (_build_world_elements_query)
        This could be part of a dedicated CypherBuilder class if it grows more complex.
        """
        # Ensure KG_NODE_CHAPTER_UPDATED and KG_IS_PROVISIONAL are correctly referenced
        return f"""
        MATCH (we:WorldElement:Entity)
        WHERE (we.is_deleted IS NULL OR we.is_deleted = FALSE) {chapter_filter}

        OPTIONAL MATCH (we)-[g:HAS_GOAL]->(goal:ValueNode:Entity {{type: 'goals'}})
        WITH we,
            [v IN collect(DISTINCT coalesce(goal.value, '')) WHERE trim(v) <> ""] AS goals

        OPTIONAL MATCH (we)-[ru:HAS_RULE]->(rule:ValueNode:Entity {{type: 'rules'}})
        WITH we,
            goals,
            [v IN collect(DISTINCT coalesce(rule.value, '')) WHERE trim(v) <> ""] AS rules

        OPTIONAL MATCH (we)-[ke:HAS_KEY_ELEMENT]->(kelem:ValueNode:Entity {{type: 'key_elements'}})
        WITH we,
            goals,
            rules,
            [v IN collect(DISTINCT coalesce(kelem.value, '')) WHERE trim(v) <> ""] AS key_elements

        OPTIONAL MATCH (we)-[tr:HAS_TRAIT_ASPECT]->(trait:ValueNode:Entity {{type: 'traits'}})
        WITH we,
            goals,
            rules,
            key_elements,
            [v IN collect(DISTINCT coalesce(trait.value, '')) WHERE trim(v) <> ""] AS traits

        OPTIONAL MATCH (we)-[:ELABORATED_IN_CHAPTER]->(elab:WorldElaborationEvent:Entity)
        WHERE ($limit IS NULL OR elab.{kg_keys.KG_NODE_CHAPTER_UPDATED} <= $limit) AND elab.summary IS NOT NULL
        WITH we,
            goals,
            rules,
            key_elements,
            traits,
            collect(DISTINCT {{
                chapter: elab.{kg_keys.KG_NODE_CHAPTER_UPDATED},
                summary: elab.summary,
                prov: coalesce(elab.{kg_keys.KG_IS_PROVISIONAL}, false)
            }}) AS elaborations

        RETURN we, goals, rules, key_elements, traits, elaborations
        ORDER BY we.category, we.name
        """

    async def _fetch_world_elements_from_db(
        self, chapter_limit: int | None
    ) -> list[dict[str, Any]]:
        """
        Fetch raw world element data from the database.
        Moved from data_access/world_queries.py (_fetch_world_elements_from_db)
        """
        we_params: dict[str, Any] = {"limit": chapter_limit}
        chapter_filter = ""
        if chapter_limit is not None:
            # Use constant for KG_NODE_CREATED_CHAPTER
            chapter_filter = (
                f"AND (we.{kg_keys.KG_NODE_CREATED_CHAPTER} IS NULL "
                f"OR we.{kg_keys.KG_NODE_CREATED_CHAPTER} <= $limit)"
            )
        query = self._build_world_elements_query_cypher(chapter_filter)
        return await neo4j_manager.execute_read_query(query, we_params)

    def _update_world_data_from_record(
        self,
        record: dict[str, Any],
        world_data: dict[str, dict[str, WorldItem]],
        chapter_limit: int | None,
    ) -> None:
        """
        Update ``world_data`` using a single query record, utilizing world_utils.
        Moved from data_access/world_queries.py (_update_world_data_from_record)
        """
        we_node = record.get("we")
        if not we_node:
            return

        category, item_name, we_id = world_utils.extract_core_world_element_fields(
            we_node
        )
        if not category or not item_name or not we_id:
            return

        item_detail_dict, created_chapter_num = (
            world_utils.initialize_item_detail_dict_from_node(we_node)
        )
        world_utils.populate_list_attributes_for_item(record, item_detail_dict)

        actual_elaborations_count = world_utils.process_elaborations_for_item(
            record.get("elaborations", []), chapter_limit, item_detail_dict
        )

        # Ensure 'id' is present before WorldItem creation if from_dict expects it.
        # initialize_item_detail_dict_from_node should handle this if we_node contains 'id'.
        # If we_id is the canonical one, ensure it's used.
        item_detail_dict["id"] = we_id

        if world_utils.should_include_world_item(
            created_chapter_num,
            actual_elaborations_count,
            chapter_limit,
            item_name,
            we_id,
        ):
            world_data.setdefault(category, {})[item_name] = WorldItem.from_dict(
                category, item_name, item_detail_dict
            )
            # Update the cache in world_utils
            world_utils.update_world_name_to_id_cache(item_name, we_id)

    @alru_cache(maxsize=settings.WORLD_QUERY_CACHE_SIZE)  # Keep cache decorator
    async def get_world_building_data(
        self, chapter_limit: int | None = None
    ) -> dict[str, dict[str, WorldItem]]:
        """
        Load world elements grouped by category from Neo4j.
        Primary method for fetching all world data.
        Replaces get_world_building_from_db from world_queries.py.
        """
        logger.info(
            "WorldQueryService: Loading world building data from Neo4j%s...",
            f" up to chapter {chapter_limit}" if chapter_limit is not None else "",
        )
        world_data: dict[str, dict[str, WorldItem]] = {}
        wc_id_param = (
            settings.MAIN_WORLD_CONTAINER_NODE_ID
        )  # Assuming this setting is available

        # Clear and prepare the name-to-ID cache for this fetch operation
        world_utils.clear_world_name_to_id_cache()

        # Load and cache overview item
        overview_item = await self._load_world_container_overview(wc_id_param)
        if overview_item:
            world_data.setdefault("_overview_", {})["_overview_"] = overview_item
            # The overview doesn't typically go into WORLD_NAME_TO_ID_CACHE unless explicitly needed by that name
            # world_utils.update_world_name_to_id_cache("_overview_", overview_item.id)

        # Fetch and process world elements
        # The original fix_missing_world_element_core_fields might be called by persistence service or a maintenance task.
        # For querying, we assume data is relatively clean or handled by extraction utils.

        we_results = await self._fetch_world_elements_from_db(chapter_limit)

        if not we_results:
            # Populate default categories if no elements found
            for cat_key in world_utils.DEFAULT_WORLD_CATEGORIES:
                world_data.setdefault(cat_key, {})
            logger.info(
                "WorldQueryService: No WorldElements found in Neo4j%s.",
                f" up to chapter {chapter_limit}" if chapter_limit is not None else "",
            )
            return world_data

        for record in we_results:
            self._update_world_data_from_record(record, world_data, chapter_limit)

        num_elements_loaded = sum(
            len(items) for cat, items in world_data.items() if cat != "_overview_"
        )
        logger.info(
            "WorldQueryService: Loaded %d elements from Neo4j%s.",
            num_elements_loaded,
            f" up to chapter {chapter_limit}" if chapter_limit is not None else "",
        )
        return world_data

    async def _fetch_world_element_node_by_id(
        self, item_id: str
    ) -> dict[str, Any] | None:
        """
        Fetch the WorldElement node by ID. Tries original ID then resolved name from cache.
        Moved from data_access/world_queries.py (_fetch_world_element_node)
        """
        query = (
            "MATCH (we:WorldElement:Entity {id: $id})"
            " WHERE we.is_deleted IS NULL OR we.is_deleted = FALSE"
            " RETURN we"
        )
        results = await neo4j_manager.execute_read_query(query, {"id": item_id})
        if results and results[0].get("we"):
            return results[0]["we"]

        # Try resolving name if item_id itself might be a name (less ideal for "by_id" func)
        # This part might be less relevant if callers always use canonical IDs.
        # For robustness, keeping a lookup if item_id itself is not found.
        alt_id = world_utils.resolve_world_name_from_cache(item_id)
        if (
            alt_id and alt_id != item_id
        ):  # If item_id was a name and resolved to a different ID
            logger.debug(
                "Attempting lookup for item '%s' using resolved ID '%s'",
                item_id,
                alt_id,
            )
            results = await neo4j_manager.execute_read_query(query, {"id": alt_id})
            if results and results[0].get("we"):
                return results[0]["we"]
        return None

    async def _fetch_and_process_list_properties_for_item(
        self, item_id: str, item_detail: dict[str, Any]
    ) -> None:
        """
        Fetch and process list-based properties (goals, rules, etc.) for a single item.
        Moved from data_access/world_queries.py (_fetch_and_process_list_properties)
        """
        list_prop_map = {
            "goals": "HAS_GOAL",
            "rules": "HAS_RULE",
            "key_elements": "HAS_KEY_ELEMENT",
            "traits": "HAS_TRAIT_ASPECT",
        }
        for list_prop_key, rel_name_internal in list_prop_map.items():
            list_values_query = f"""
            MATCH (:WorldElement:Entity {{id: $we_id_param}})-[:{rel_name_internal}]->(v:ValueNode:Entity {{type: $value_node_type_param}})
            WHERE v.value IS NOT NULL AND trim(v.value) <> ""
            RETURN v.value AS item_value
            ORDER BY v.value ASC
            """
            list_val_res = await neo4j_manager.execute_read_query(
                list_values_query,
                {"we_id_param": item_id, "value_node_type_param": list_prop_key},
            )
            item_detail[list_prop_key] = sorted(
                [
                    res_item["item_value"]
                    for res_item in list_val_res
                    if res_item
                    and res_item.get("item_value")
                    is not None  # Redundant due to WHERE clause but safe
                ]
            )

    async def _fetch_and_process_elaborations_for_item(
        self, item_id: str, item_detail: dict[str, Any]
    ) -> None:
        """
        Fetch and process elaboration data for a single item.
        Moved from data_access/world_queries.py (_fetch_and_process_elaborations)
        """
        elab_query = f"""
        MATCH (:WorldElement:Entity {{id: $we_id_param}})-[:ELABORATED_IN_CHAPTER]->(elab:WorldElaborationEvent:Entity)
        RETURN elab.summary AS summary,
               elab.{kg_keys.KG_NODE_CHAPTER_UPDATED} AS chapter,
               elab.{kg_keys.KG_IS_PROVISIONAL} AS is_provisional
        ORDER BY elab.{kg_keys.KG_NODE_CHAPTER_UPDATED} ASC
        """
        elab_results = await neo4j_manager.execute_read_query(
            elab_query, {"we_id_param": item_id}
        )
        if elab_results:
            world_utils.process_elaborations_for_item(elab_results, None, item_detail)

    @alru_cache(maxsize=settings.WORLD_QUERY_CACHE_SIZE)  # Keep cache decorator
    async def get_world_item_by_id(self, item_id: str) -> WorldItem | None:
        """
        Retrieve a single ``WorldItem`` from Neo4j by its ID.
        Replaces get_world_item_by_id from world_queries.py.
        """
        logger.info("WorldQueryService: Loading world item '%s' from Neo4j...", item_id)

        we_node = await self._fetch_world_element_node_by_id(item_id)
        if not we_node:
            logger.info("WorldQueryService: No world item found for id '%s'.", item_id)
            return None

        item_detail, category, item_name = (
            world_utils.extract_core_world_element_fields(we_node)
        )
        if not category or not item_name:
            # Error already logged by extract_core_world_element_fields
            return None

        # Initialize details further (created_chapter, provisional status)
        item_detail_from_node, _ = world_utils.initialize_item_detail_dict_from_node(
            we_node
        )
        # Merge/update item_detail with these processed fields
        item_detail.update(item_detail_from_node)

        await self._fetch_and_process_list_properties_for_item(item_id, item_detail)
        await self._fetch_and_process_elaborations_for_item(item_id, item_detail)

        item_detail["id"] = (
            item_id  # Ensure the original/looked-up ID is part of the final dict
        )
        return WorldItem.from_dict(category, item_name, item_detail)

    @alru_cache(maxsize=settings.WORLD_QUERY_CACHE_SIZE)  # Keep cache decorator
    async def get_all_world_item_ids_by_category(self) -> dict[str, list[str]]:
        """
        Return all world item IDs grouped by category.
        Moved from data_access/world_queries.py.
        """
        query = (
            "MATCH (we:WorldElement:Entity) "
            "WHERE we.is_deleted IS NULL OR we.is_deleted = FALSE "
            "RETURN we.category AS category, we.id AS id"
        )
        results = await neo4j_manager.execute_read_query(query)
        mapping: dict[str, list[str]] = {}
        for record in results:
            category = record.get("category")
            item_id = record.get("id")
            if category and item_id:
                mapping.setdefault(category, []).append(item_id)
        return mapping

    async def get_world_elements_for_snippet(
        self, category: str, chapter_limit: int, item_limit: int
    ) -> list[dict[str, Any]]:
        """
        Return a subset of world elements for prompt context.
        Moved from data_access/world_queries.py (get_world_elements_for_snippet_from_db)
        """
        # Ensure constants are correctly referenced
        query = f"""
        MATCH (we:WorldElement:Entity {{category: $category_param}})
        WHERE (we.{kg_keys.KG_NODE_CREATED_CHAPTER} IS NULL OR we.{kg_keys.KG_NODE_CREATED_CHAPTER} <= $chapter_limit_param)
          AND (we.is_deleted IS NULL OR we.is_deleted = FALSE)

        OPTIONAL MATCH (we)-[:ELABORATED_IN_CHAPTER]->(elab:WorldElaborationEvent:Entity)
        WHERE elab.{kg_keys.KG_NODE_CHAPTER_UPDATED} <= $chapter_limit_param
          AND elab.{kg_keys.KG_IS_PROVISIONAL} = TRUE

        WITH we, COUNT(elab) AS provisional_elaborations_count
        WITH we, ( we.{kg_keys.KG_IS_PROVISIONAL} = TRUE OR provisional_elaborations_count > 0 ) AS is_item_provisional_overall

        RETURN we.name AS name,
               we.description AS description,
               is_item_provisional_overall AS is_provisional
        ORDER BY we.name ASC
        LIMIT $item_limit_param
        """
        params = {
            "category_param": category,
            "chapter_limit_param": chapter_limit,
            "item_limit_param": item_limit,
        }
        items = []
        try:
            results = await neo4j_manager.execute_read_query(query, params)
            if results:
                for record in results:
                    desc_val = record.get("description")
                    desc = str(desc_val) if desc_val is not None else ""
                    items.append(
                        {
                            "name": record.get("name"),
                            "description_snippet": (
                                desc[:50].strip() + "..."
                                if len(desc) > 50
                                else desc.strip()
                            ),
                            "is_provisional": record.get("is_provisional", False),
                        }
                    )
        except Exception as e:
            logger.error(
                f"WorldQueryService: Error fetching world elements for snippet (cat {category}): {e}",
                exc_info=True,
            )
        return items

    async def find_thin_world_elements_for_enrichment(self) -> list[dict[str, Any]]:
        """
        Finds WorldElement nodes that are considered 'thin' (e.g., missing description).
        Moved from data_access/world_queries.py.
        """
        query = """
        MATCH (we:WorldElement)
        WHERE (we.description IS NULL OR we.description = '')
          AND (we.is_deleted IS NULL OR we.is_deleted = FALSE)
        RETURN we.id AS id, we.name AS name, we.category as category
        LIMIT 20 // Limit to avoid overwhelming the LLM in one cycle
        """
        try:
            results = await neo4j_manager.execute_read_query(query)
            return results if results else []
        except Exception as e:
            logger.error(
                f"WorldQueryService: Error finding thin world elements: {e}",
                exc_info=True,
            )
            return []


# Ensure WorldItem is imported for type hints
# from kg_maintainer.models import WorldItem
# Ensure settings are imported for cache sizes etc.
# from config import settings
# Ensure kg_constants are imported
# import kg_constants as kg_keys
# Ensure neo4j_manager is available
# from core.db_manager import neo4j_manager
# Ensure world_utils are correctly imported relative to this service's location
# from ..utils import world_utils
