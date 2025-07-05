# data_access/services/world_persistence_service.py
"""Service for persisting world-building data to the database."""

from typing import Any
import structlog

from core.db_manager import neo4j_manager
from kg_maintainer.models import WorldItem
from data_access.cypher_builders.world_cypher import generate_world_element_node_cypher # Assuming this path
import kg_constants as kg_keys # For KG_NODE_CREATED_CHAPTER, KG_IS_PROVISIONAL etc.
from config import settings # For MAIN_NOVEL_INFO_NODE_ID etc.

from ..utils import world_utils # Path to world_utils

logger = structlog.get_logger(__name__)

# Forward declaration for type hinting if WorldQueryService is used
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .world_query_service import WorldQueryService


class WorldPersistenceService:
    def __init__(self, world_query_service: "WorldQueryService | None" = None):
        """
        Initializes the WorldPersistenceService.

        Args:
            world_query_service: An instance of WorldQueryService, needed for
                                 operations like fetching existing IDs during full sync.
                                 Can be None if only partial sync methods are used that don't require it.
        """
        self._world_query_service = world_query_service

    async def sync_world_items_incremental(
        self,
        world_items: dict[str, dict[str, WorldItem]],
        chapter_number: int,
    ) -> bool:
        """
        Persist world element data to Neo4j incrementally.
        Based on the non-full_sync part of original sync_world_items.
        """
        # Clear and repopulate the cache based on the items being synced.
        # This assumes world_items contains the current source of truth for names/IDs.
        world_utils.clear_world_name_to_id_cache()
        for _cat, items_in_category in world_items.items():
            if not isinstance(items_in_category, dict):
                continue
            for item_obj in items_in_category.values():
                if isinstance(item_obj, WorldItem) and item_obj.name and item_obj.id:
                    world_utils.update_world_name_to_id_cache(item_obj.name, item_obj.id)

        statements: list[tuple[str, dict[str, Any]]] = []
        count = 0
        for category_name, items_in_category in world_items.items():
            if not isinstance(items_in_category, dict):
                logger.debug(f"Skipping non-dict items for category {category_name} in incremental sync.")
                continue
            for item_name, item_obj in items_in_category.items():
                if not isinstance(item_obj, WorldItem):
                    logger.debug(f"Skipping non-WorldItem '{item_name}' in category '{category_name}'.")
                    continue
                # generate_world_element_node_cypher is expected to return a list of (query, params) tuples
                item_statements = generate_world_element_node_cypher(item_obj, chapter_number)
                statements.extend(item_statements)
                if item_statements: # Only increment if statements were generated
                    count += 1

        if not statements:
            logger.info("No statements generated for incremental world item sync for chapter %d.", chapter_number)
            return True # No error, just nothing to do

        try:
            await neo4j_manager.execute_cypher_batch(statements)
            logger.info(
                "WorldPersistenceService: Persisted %d world element updates for chapter %d.",
                count,
                chapter_number,
            )
            return True
        except Exception as exc:
            logger.error(
                "WorldPersistenceService: Error persisting incremental world updates for chapter %d: %s",
                chapter_number,
                exc,
                exc_info=True,
            )
            return False

    # --- Full Sync Logic ---

    def _collect_input_world_element_ids(self, world_data_dict: dict[str, Any]) -> set[str]:
        """
        Gather all world element IDs present in the input ``world_data_dict``.
        Moved from data_access/world_queries.py
        """
        ids: set[str] = set()
        for category, items in world_data_dict.items():
            if category == "_overview_" or not isinstance(items, dict):
                continue
            for name, details in items.items():
                # Skip special keys that might be present at the category level
                if name.startswith(("_", kg_keys.SOURCE_QUALITY_PREFIX, "category_updated_in_chapter_")):
                    continue
                if isinstance(details, dict): # Ensure details is a dict
                    # Use world_utils to build ID
                    we_id = world_utils.build_world_element_id(category, name, details)
                    if we_id: # Ensure an ID was successfully built
                        ids.add(we_id)
        return ids

    async def _fetch_existing_world_element_ids_from_db(self) -> set[str]:
        """
        Return IDs of non-deleted world elements stored in Neo4j.
        Moved from data_access/world_queries.py (_fetch_existing_world_element_ids)
        """
        records = await neo4j_manager.execute_read_query(
            "MATCH (we:WorldElement:Entity)"
            " WHERE we.is_deleted IS NULL OR we.is_deleted = FALSE"
            " RETURN we.id AS id"
        )
        return {rec["id"] for rec in records if rec and rec.get("id")}

    def _generate_world_container_statements(
        self, overview_details: dict[str, Any], wc_id: str, novel_id: str
    ) -> list[tuple[str, dict[str, Any]]]:
        """Moved from data_access/world_queries.py"""
        if not isinstance(overview_details, dict):
            return []
        wc_props: dict[str, Any] = {
            "id": wc_id,
            "overview_description": str(overview_details.get("description", "")),
            # Use kg_keys constants
            kg_keys.KG_IS_PROVISIONAL: overview_details.get(
                kg_keys.source_quality_key(settings.KG_PREPOPULATION_CHAPTER_NUM)
            )
            == "provisional_from_unrevised_draft",
        }
        for key, val in overview_details.items():
            if isinstance(val, (str, int, float, bool)) and key not in wc_props:
                wc_props[key] = val
        return [
            (
                """
            MERGE (wc:Entity {id: $id_val})
            ON CREATE SET wc:WorldContainer, wc = $props, wc.created_ts = timestamp()
            ON MATCH SET  wc:WorldContainer, wc += $props, wc.updated_ts = timestamp()
            """, # Use += on match to preserve other properties not in $props
                {"id_val": wc_id, "props": wc_props},
            ),
            (
                """
            MATCH (ni:NovelInfo:Entity {id: $novel_id_val})
            MATCH (wc:WorldContainer:Entity {id: $wc_id_val})
            MERGE (ni)-[:HAS_WORLD_META]->(wc)
            """,
                {"novel_id_val": novel_id, "wc_id_val": wc_id},
            ),
        ]

    def _prepare_world_element_props_for_sync(
        self, category: str, name: str, details: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Build property dictionary for a ``WorldElement`` node for full sync.
        Moved from data_access/world_queries.py (_prepare_world_element_props)
        """
        we_id = world_utils.build_world_element_id(category, name, details)
        we_props = {
            "id": we_id,
            "name": name,
            "category": category,
        }
        # Use kg_keys constants
        created_chap_val = details.get(
            kg_keys.KG_NODE_CREATED_CHAPTER, # Check this first
            details.get("created_chapter", settings.KG_PREPOPULATION_CHAPTER_NUM),
        )
        created_chap_int = int(created_chap_val)
        we_props[kg_keys.KG_NODE_CREATED_CHAPTER] = created_chap_int

        is_prov_bool = False
        sq_key = kg_keys.source_quality_key(created_chap_int)
        if details.get(sq_key) == "provisional_from_unrevised_draft":
            is_prov_bool = True
        elif details.get(kg_keys.KG_IS_PROVISIONAL) is True:
            is_prov_bool = True
        elif details.get("is_provisional") is True: # Generic fallback
            is_prov_bool = True

        we_props[kg_keys.KG_IS_PROVISIONAL] = is_prov_bool
        we_props["is_deleted"] = bool(details.get("is_deleted", False)) # Ensure it's set

        for key, val in details.items():
            if (
                isinstance(val, (str, int, float, bool))
                and key not in we_props # Avoid overwriting already set core properties
                and not key.startswith(kg_keys.ELABORATION_PREFIX)
                and not key.startswith(kg_keys.ADDED_PREFIX)
                and not key.startswith(kg_keys.SOURCE_QUALITY_PREFIX)
                and key not in {"goals", "rules", "key_elements", "traits", "id", "name", "category", "created_chapter", "is_provisional", "is_deleted"}
            ):
                we_props[key] = val
        return we_props

    def _generate_list_value_statements_for_sync(
        self, we_id: str, details: dict[str, Any]
    ) -> list[tuple[str, dict[str, Any]]]:
        """Moved from data_access/world_queries.py (_generate_list_value_statements)"""
        statements: list[tuple[str, dict[str, Any]]] = []
        list_prop_map = {
            "goals": "HAS_GOAL", "rules": "HAS_RULE",
            "key_elements": "HAS_KEY_ELEMENT", "traits": "HAS_TRAIT_ASPECT",
        }
        for prop_key, rel_name in list_prop_map.items():
            current_values = {
                str(v).strip() for v in details.get(prop_key, [])
                if isinstance(v, str) and str(v).strip()
            }
            # Delete old relationships to values not in the current list
            statements.append(
                (
                    f"""
                MATCH (we:WorldElement:Entity {{id: $we_id_val}})-[r:{rel_name}]->(v:ValueNode:Entity {{type: $value_node_type}})
                WHERE NOT v.value IN $current_values_list
                DETACH DELETE r
                // Optionally delete v if it becomes an orphan, handled by _cleanup_orphan_value_nodes later
                """,
                    {
                        "we_id_val": we_id, "value_node_type": prop_key,
                        "current_values_list": list(current_values) if current_values else ["__dummy_value_to_prevent_deleting_all__"] # Handle empty list
                    },
                )
            )
            if current_values: # Create/merge new relationships
                statements.append(
                    (
                        f"""
                    MATCH (we:WorldElement:Entity {{id: $we_id_val}})
                    UNWIND $current_values_list AS item_value_str
                    MERGE (v:Entity:ValueNode {{value: item_value_str, type: $value_node_type}})
                       ON CREATE SET v.created_ts = timestamp()
                    MERGE (we)-[:{rel_name}]->(v)
                    """,
                        {
                            "we_id_val": we_id, "value_node_type": prop_key,
                            "current_values_list": list(current_values),
                        },
                    )
                )
        return statements

    def _generate_elaboration_statements_for_sync(
        self, we_id: str, details: dict[str, Any], item_name_for_log: str
    ) -> list[tuple[str, dict[str, Any]]]:
        """Moved from data_access/world_queries.py (_generate_elaboration_statements)"""
        statements: list[tuple[str, dict[str, Any]]] = [
            ( # Delete all existing elaborations for this world element first
                """
            MATCH (we:WorldElement:Entity {id: $we_id_val})-[r:ELABORATED_IN_CHAPTER]->(elab:WorldElaborationEvent:Entity)
            DETACH DELETE elab, r
            """, # Using DETACH DELETE elab, r will also remove the relationship r
                {"we_id_val": we_id},
            )
        ]
        for key, value in details.items():
            if (
                key.startswith(kg_keys.ELABORATION_PREFIX)
                and isinstance(value, str) and value.strip()
            ):
                try:
                    chap_num = kg_keys.parse_elaboration_key(key)
                    if chap_num is None: continue

                    elab_is_prov = False
                    sq_key = kg_keys.source_quality_key(chap_num)
                    if details.get(sq_key) == "provisional_from_unrevised_draft":
                        elab_is_prov = True

                    elab_id = f"elab_{we_id}_ch{chap_num}_{hash(value.strip())}" # More robust ID
                    elab_props = {
                        "id": elab_id, "summary": value.strip(),
                        kg_keys.KG_NODE_CHAPTER_UPDATED: chap_num,
                        kg_keys.KG_IS_PROVISIONAL: elab_is_prov,
                    }
                    statements.append(
                        (
                            """
                        MATCH (we:WorldElement:Entity {id: $we_id_val})
                        CREATE (elab:Entity:WorldElaborationEvent)
                        SET elab = $props, elab.created_ts = timestamp()
                        CREATE (we)-[:ELABORATED_IN_CHAPTER]->(elab)
                        """,
                            {"we_id_val": we_id, "props": elab_props},
                        )
                    )
                except ValueError: # From parse_elaboration_key
                    logger.warning("Could not parse chapter for world elab key: %s for item %s", key, item_name_for_log)
        return statements

    def _cleanup_orphan_value_nodes_statement(self) -> tuple[str, dict[str, Any]]:
        """Moved from data_access/world_queries.py (_cleanup_orphan_value_nodes)"""
        return (
            """
        MATCH (v:ValueNode:Entity)
        WHERE NOT EXISTS((:WorldElement:Entity)-[]->(v))
          AND NOT EXISTS((:CharacterProfile:Entity)-[]->(v)) // Also check character links
          AND NOT EXISTS((:Entity)-[:DYNAMIC_REL]->(v)) // And generic dynamic rels
        DETACH DELETE v
        """, {},
        )

    async def _generate_orphan_deletion_statements_for_sync(
        self, world_data_dict: dict[str, Any]
    ) -> list[tuple[str, dict[str, Any]]]:
        """Moved from data_access/world_queries.py (_generate_orphan_deletion_statements)"""
        input_ids = self._collect_input_world_element_ids(world_data_dict)
        try:
            # This could call a method on WorldQueryService if it's passed in
            if self._world_query_service:
                # This specific method isn't on WorldQueryService yet, but a similar one could be
                # For now, directly using the original logic:
                existing_ids = await self._fetch_existing_world_element_ids_from_db()
            else: # Fallback or raise error if query service not available
                logger.warning("WorldQueryService not available for orphan check, fetching IDs directly.")
                existing_ids = await self._fetch_existing_world_element_ids_from_db()

        except Exception as exc:
            logger.error(f"Failed to retrieve existing WorldElement IDs for orphan check: {exc}", exc_info=True)
            raise

        orphan_ids = existing_ids - input_ids
        if orphan_ids:
            return [
                (
                    """
                MATCH (we:WorldElement:Entity) WHERE we.id IN $we_ids_to_delete
                SET we.is_deleted = TRUE, we.updated_ts = timestamp()
                """, {"we_ids_to_delete": list(orphan_ids)},
                )
            ]
        return []

    def _generate_world_element_statements_for_sync(
        self, world_data_dict: dict[str, Any], wc_id: str
    ) -> list[tuple[str, dict[str, Any]]]:
        """Moved from data_access/world_queries.py (_generate_world_element_statements)"""
        statements: list[tuple[str, dict[str, Any]]] = []
        for category, items in world_data_dict.items():
            if category == "_overview_" or not isinstance(items, dict):
                continue
            for name, details in items.items():
                if not isinstance(details, dict) or name.startswith(("_", kg_keys.SOURCE_QUALITY_PREFIX)):
                    continue

                we_id = world_utils.build_world_element_id(category, name, details)
                props = self._prepare_world_element_props_for_sync(category, name, details)

                statements.append( # Merge WorldElement node
                    (
                        """
                    MERGE (we:Entity {id: $id_val})
                    ON CREATE SET we:WorldElement, we = $props, we.created_ts = timestamp()
                    ON MATCH SET  we:WorldElement, we += $props, we.updated_ts = timestamp()
                    """, {"id_val": we_id, "props": props},
                    )
                )
                statements.append( # Link to WorldContainer
                    (
                        """
                    MATCH (wc:WorldContainer:Entity {id: $wc_id_val})
                    MATCH (we:WorldElement:Entity {id: $we_id_val})
                    MERGE (wc)-[:CONTAINS_ELEMENT]->(we)
                    """, {"wc_id_val": wc_id, "we_id_val": we_id},
                    )
                )
                statements.extend(self._generate_list_value_statements_for_sync(we_id, details))
                statements.extend(self._generate_elaboration_statements_for_sync(we_id, details, name))
        return statements

    async def _build_all_sync_statements(
        self, world_data_dict: dict[str, Any], wc_id: str, novel_id: str
    ) -> list[tuple[str, dict[str, Any]]]:
        """Moved from data_access/world_queries.py (_build_sync_statements)"""
        statements: list[tuple[str, dict[str, Any]]] = []
        statements.extend(
            self._generate_world_container_statements(
                world_data_dict.get("_overview_", {}), wc_id, novel_id
            )
        )
        try:
            orphan_statements = await self._generate_orphan_deletion_statements_for_sync(world_data_dict)
            statements.extend(orphan_statements)
        except Exception as exc:
            logger.error(f"Failed to generate orphan deletion statements during full sync build: {exc}", exc_info=True)
            # Decide on error handling: continue without orphan deletion or raise
            pass
        statements.extend(self._generate_world_element_statements_for_sync(world_data_dict, wc_id))
        statements.append(self._cleanup_orphan_value_nodes_statement())
        return statements

    async def sync_full_world_state_to_db(self, world_data_as_dict: dict[str, Any]) -> bool:
        """
        Persist the entire world-building state (from dict) to Neo4j.
        Replaces sync_full_state_from_object_to_db.
        """
        logger.info("WorldPersistenceService: Synchronizing full world building data to Neo4j...")
        novel_id = settings.MAIN_NOVEL_INFO_NODE_ID
        wc_id = settings.MAIN_WORLD_CONTAINER_NODE_ID

        # Clear and repopulate the name-to-ID cache for this full sync
        world_utils.clear_world_name_to_id_cache()
        for category, items in world_data_as_dict.items():
            if category == "_overview_" or not isinstance(items, dict):
                continue
            for name, details in items.items():
                if isinstance(details, dict):
                    item_id = world_utils.build_world_element_id(category, name, details)
                    world_utils.update_world_name_to_id_cache(name, item_id)

        # Overview special case for cache
        overview_details = world_data_as_dict.get("_overview_", {})
        if isinstance(overview_details, dict) and overview_details.get("id"):
             world_utils.update_world_name_to_id_cache("_overview_", overview_details["id"])
        elif overview_details: # Fallback if ID not in details but wc_id is known
            world_utils.update_world_name_to_id_cache("_overview_", wc_id)


        all_statements = await self._build_all_sync_statements(world_data_as_dict, wc_id, novel_id)

        try:
            if all_statements:
                await neo4j_manager.execute_cypher_batch(all_statements)
            logger.info("WorldPersistenceService: Successfully synchronized full world building data.")
            return True
        except Exception as e:
            logger.error(f"WorldPersistenceService: Error synchronizing full world data: {e}", exc_info=True)
            return False

    async def fix_missing_world_element_core_fields_in_db(self) -> int:
        """
        Populate missing ``id``, ``name``, or ``category`` on WorldElements in the DB.
        Moved from data_access/world_queries.py (fix_missing_world_element_core_fields)
        """
        query = """
        MATCH (we:WorldElement)
        WHERE (we.is_deleted IS NULL OR we.is_deleted = FALSE)
          AND (
            we.id IS NULL OR trim(we.id) = "" OR
            we.name IS NULL OR trim(we.name) = "" OR
            we.category IS NULL OR trim(we.category) = ""
          )
        RETURN elementId(we) AS nid, we.id AS id, we.name AS name, we.category AS category
        """
        try:
            results = await neo4j_manager.execute_read_query(query)
        except Exception as exc:
            logger.error("Error fetching WorldElements missing core fields: %s", exc, exc_info=True)
            return 0
        if not results: return 0

        statements: list[tuple[str, dict[str, Any]]] = []
        for rec in results:
            neo_id, w_id, name, category = rec.get("nid"), rec.get("id"), rec.get("name"), rec.get("category")
            if neo_id is None: continue

            w_id = str(w_id).strip() if isinstance(w_id, str) else None
            name = str(name).strip() if isinstance(name, str) else None
            category = str(category).strip() if isinstance(category, str) else None

            props_to_set: dict[str, Any] = {}
            current_name, current_category, current_id = name, category, w_id

            if not current_name:
                current_name = (w_id.split("_", 1)[-1].replace("_", " ").title() if w_id else "Unnamed Element")
                props_to_set["name"] = current_name
            if not current_category:
                current_category = (w_id.split("_")[0] if w_id else "unknown_category")
                props_to_set["category"] = current_category
            if not current_id and current_name and current_category: # Must have name/cat to build ID
                props_to_set["id"] = world_utils.build_world_element_id(current_category, current_name, {}) # Pass empty details

            if props_to_set:
                statements.append(
                    ("MATCH (we:WorldElement) WHERE elementId(we) = $nid SET we += $props, we.updated_ts = timestamp()",
                     {"nid": neo_id, "props": props_to_set})
                )
        if not statements: return 0
        try:
            await neo4j_manager.execute_cypher_batch(statements)
            logger.info("WorldPersistenceService: Fixed missing core fields for %d WorldElements.", len(statements))
        except Exception as exc:
            logger.error("Error updating WorldElements with missing core fields: %s", exc, exc_info=True)
            return 0
        return len(statements)

    async def remove_world_element_trait_aspect_from_db(self, element_id: str, trait_value: str) -> bool:
        """
        Remove a trait aspect from a world element in the DB.
        Moved from data_access/world_queries.py (remove_world_element_trait_aspect)
        """
        query = (
            "MATCH (:WorldElement:Entity {id: $we_id})-"
            "[r:HAS_TRAIT_ASPECT]->(v:ValueNode:Entity {type: 'traits', value: $trait})"
            " DELETE r"
        )
        cleanup_query = ( # To remove the ValueNode if it becomes an orphan
            "MATCH (v:ValueNode:Entity {type: 'traits', value: $trait})"
            " WHERE NOT EXISTS((:WorldElement:Entity)-[:HAS_TRAIT_ASPECT]->(v))"
            " DETACH DELETE v"
        )
        try:
            await neo4j_manager.execute_write_query(query, {"we_id": element_id, "trait": trait_value})
            await neo4j_manager.execute_write_query(cleanup_query, {"trait": trait_value})
            logger.info("Removed trait '%s' from world element %s.", trait_value, element_id)
            return True
        except Exception as exc:
            logger.error("Error removing trait '%s' from world element %s: %s", trait_value, element_id, exc, exc_info=True)
            return False

# Ensure imports are correct based on file structure
# from core.db_manager import neo4j_manager
# from kg_maintainer.models import WorldItem
# from ..cypher_builders.world_cypher import generate_world_element_node_cypher (adjust path if needed)
# import kg_constants as kg_keys
# from config import settings
# from ..utils import world_utils
# from .world_query_service import WorldQueryService (for type hint)
