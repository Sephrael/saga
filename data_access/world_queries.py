# data_access/world_queries.py
from typing import Any

import kg_constants as kg_keys
import structlog
import utils
from async_lru import alru_cache  # type: ignore
from config import settings
from core.db_manager import neo4j_manager
from kg_constants import (
    KG_IS_PROVISIONAL,
    KG_NODE_CHAPTER_UPDATED,
    KG_NODE_CREATED_CHAPTER,
)
from kg_maintainer.models import WorldItem

from .cypher_builders.world_cypher import generate_world_element_node_cypher

logger = structlog.get_logger(__name__)

# Mapping from normalized world item names to canonical IDs
WORLD_NAME_TO_ID: dict[str, str] = {}

# Default categories used when no world elements exist
DEFAULT_WORLD_CATEGORIES = [
    "locations",
    "society",
    "systems",
    "lore",
    "history",
    "factions",
]


async def _load_world_container(wc_id: str) -> WorldItem | None:
    """Return the world container ``WorldItem`` from Neo4j."""

    overview_query = "MATCH (wc:WorldContainer:Entity {id: $wc_id_param}) RETURN wc"
    overview_res_list = await neo4j_manager.execute_read_query(
        overview_query, {"wc_id_param": wc_id}
    )
    if not overview_res_list or not overview_res_list[0].get("wc"):
        return None

    wc_node = overview_res_list[0]["wc"]
    overview_data_dict = dict(wc_node)
    overview_data_dict.pop("created_ts", None)
    overview_data_dict.pop("updated_ts", None)

    if overview_data_dict.get(KG_IS_PROVISIONAL):
        overview_data_dict[
            kg_keys.source_quality_key(settings.KG_PREPOPULATION_CHAPTER_NUM)
        ] = "provisional_from_unrevised_draft"

    return WorldItem.from_dict("_overview_", "_overview_", overview_data_dict)


def _build_world_elements_query(chapter_filter: str) -> str:
    """Return the cypher query used to fetch world elements."""

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
    WITH we,
        goals,
        rules,
        key_elements,
        traits,

        [e IN collect(DISTINCT CASE
            WHEN ($limit IS NULL OR elab.{KG_NODE_CHAPTER_UPDATED} <= $limit) AND elab.summary IS NOT NULL THEN {{
                chapter: elab.{KG_NODE_CHAPTER_UPDATED},
                summary: elab.summary,
                prov: coalesce(elab.{KG_IS_PROVISIONAL}, false)
            }}
            ELSE NULL END) WHERE e IS NOT NULL] AS elaborations

    RETURN we, goals, rules, key_elements, traits, elaborations
    ORDER BY we.category, we.name
    """


def _process_elaborations(
    elaborations: list[dict[str, Any]],
    chapter_limit: int | None,
    item_detail_dict: dict[str, Any],
) -> int:
    """Apply elaboration records to ``item_detail_dict`` and return the count."""

    count = 0
    for elab_rec in elaborations:
        chapter_val = elab_rec.get("chapter")
        summary_val = elab_rec.get("summary")
        if chapter_val is None or summary_val is None:
            continue
        if chapter_limit is not None and chapter_val > chapter_limit:
            continue
        elab_key = kg_keys.elaboration_key(chapter_val)
        item_detail_dict[elab_key] = summary_val
        if elab_rec.get("prov"):
            item_detail_dict[kg_keys.source_quality_key(chapter_val)] = (
                "provisional_from_unrevised_draft"
            )
        count += 1
    return count


def _extract_core_we_fields(
    we_node: dict[str, Any],
) -> tuple[str | None, str | None, str | None]:
    """Extract category, item_name, and we_id from the we_node."""
    category = we_node.get("category")
    item_name = we_node.get("name")
    we_id = we_node.get("id")
    if not all([category, item_name, we_id]):
        logger.warning(
            "Skipping WorldElement with missing core fields (id, name, or category): %s",
            we_node,
        )
        return None, None, None
    return category, item_name, we_id


def _initialize_item_detail_dict(we_node: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Initialize item_detail_dict from we_node and process creation chapter."""
    item_detail_dict = dict(we_node)
    item_detail_dict.pop("created_ts", None)
    item_detail_dict.pop("updated_ts", None)

    created_chapter_num = item_detail_dict.pop(
        KG_NODE_CREATED_CHAPTER, settings.KG_PREPOPULATION_CHAPTER_NUM
    )
    item_detail_dict["created_chapter"] = int(created_chapter_num)
    item_detail_dict[kg_keys.added_key(created_chapter_num)] = True

    is_provisional_at_creation = item_detail_dict.pop(KG_IS_PROVISIONAL, False)
    item_detail_dict["is_provisional"] = is_provisional_at_creation
    if is_provisional_at_creation:
        item_detail_dict[kg_keys.source_quality_key(created_chapter_num)] = (
            "provisional_from_unrevised_draft"
        )
    return item_detail_dict, created_chapter_num


def _populate_list_attributes(
    record: dict[str, Any], item_detail_dict: dict[str, Any]
) -> None:
    """Populate list attributes (goals, rules, key_elements, traits) in item_detail_dict."""
    list_attrs = ["goals", "rules", "key_elements", "traits"]
    for attr in list_attrs:
        item_detail_dict[attr] = sorted(
            [v for v in record.get(attr, []) if v is not None]
        )


def _should_include_world_item(
    created_chapter_num: int,
    actual_elaborations_count: int,
    chapter_limit: int | None,
    item_name: str,  # For logging
    we_id: str,  # For logging
) -> bool:
    """Determine if the world item should be included based on chapter limits and elaborations."""
    if chapter_limit is None:
        return True
    if created_chapter_num <= chapter_limit:
        return True
    if actual_elaborations_count > 0:
        return True

    # Log exclusion if item is created after chapter_limit and has no relevant elaborations
    logger.debug(
        "WorldElement '%s' (id: %s) created in chapter %s with no elaborations up to chapter %s, excluding.",
        item_name,
        we_id,
        created_chapter_num,
        chapter_limit,
    )
    return False


def _update_world_data_from_record(
    record: dict[str, Any],
    world_data: dict[str, dict[str, WorldItem]],
    chapter_limit: int | None,
) -> None:
    """Update ``world_data`` using a single query record."""
    we_node = record.get("we")
    if not we_node:
        return

    category, item_name, we_id = _extract_core_we_fields(we_node)
    if not category or not item_name or not we_id:
        return

    item_detail_dict, created_chapter_num = _initialize_item_detail_dict(we_node)
    _populate_list_attributes(record, item_detail_dict)

    actual_elaborations_count = _process_elaborations(
        record.get("elaborations", []), chapter_limit, item_detail_dict
    )
    item_detail_dict["id"] = we_id  # Ensure ID is present before creating WorldItem

    if _should_include_world_item(
        created_chapter_num, actual_elaborations_count, chapter_limit, item_name, we_id
    ):
        world_data.setdefault(category, {})[item_name] = WorldItem.from_dict(
            category, item_name, item_detail_dict
        )
        WORLD_NAME_TO_ID[utils._normalize_for_id(item_name)] = we_id


def resolve_world_name(name: str) -> str | None:
    """Return canonical world item ID for a display name if known."""
    if not name:
        return None
    return WORLD_NAME_TO_ID.get(utils._normalize_for_id(name))


def get_world_item_by_name(
    world_data: dict[str, dict[str, WorldItem]], name: str
) -> WorldItem | None:
    """Retrieve a WorldItem from cached data using a fuzzy name lookup."""
    item_id = resolve_world_name(name)
    if not item_id:
        return None
    for items in world_data.values():
        if not isinstance(items, dict):
            continue
        for item in items.values():
            if isinstance(item, WorldItem) and item.id == item_id:
                return item
    return None


async def sync_world_items(
    world_items: dict[str, dict[str, WorldItem]],
    chapter_number: int,
    full_sync: bool = False,
) -> bool:
    """Persist world element data to Neo4j."""
    WORLD_NAME_TO_ID.clear()
    for _cat, items in world_items.items():
        if not isinstance(items, dict):
            continue
        for item in items.values():
            if isinstance(item, WorldItem):
                WORLD_NAME_TO_ID[utils._normalize_for_id(item.name)] = item.id
    if full_sync:
        world_dict = {
            cat: {name: item.to_dict() for name, item in items.items()}
            for cat, items in world_items.items()
        }
        return await sync_full_state_from_object_to_db(world_dict)

    statements: list[tuple[str, dict[str, Any]]] = []
    count = 0
    for category_items in world_items.values():
        if not isinstance(category_items, dict):
            continue
        for item_obj in category_items.values():
            statements.extend(
                generate_world_element_node_cypher(item_obj, chapter_number)
            )
            count += 1

    try:
        if statements:
            await neo4j_manager.execute_cypher_batch(statements)
        logger.info(
            "Persisted %d world element updates for chapter %d.",
            count,
            chapter_number,
        )
        return True
    except Exception as exc:  # pragma: no cover - log and return failure
        logger.error(
            "Error persisting world element updates for chapter %d: %s",
            chapter_number,
            exc,
            exc_info=True,
        )
        return False


def _build_world_element_id(
    category: str, item_name: str, details: dict[str, Any]
) -> str:
    """Return a stable ID for a world element."""
    if isinstance(details.get("id"), str) and details.get("id", "").strip():
        return str(details["id"])
    norm_cat = utils._normalize_for_id(category) or "unknown_category"
    norm_name = utils._normalize_for_id(item_name) or "unknown_name"
    return f"{norm_cat}_{norm_name}"


def _collect_input_world_element_ids(world_data: dict[str, Any]) -> set[str]:
    """Gather all world element IDs present in the input ``world_data``."""
    ids: set[str] = set()
    for category, items in world_data.items():
        if category == "_overview_" or not isinstance(items, dict):
            continue
        for name, details in items.items():
            if name.startswith(
                ("_", kg_keys.SOURCE_QUALITY_PREFIX, "category_updated_in_chapter_")
            ):
                continue
            if isinstance(details, dict):
                we_id = _build_world_element_id(category, name, details)
                if we_id:
                    ids.add(we_id)
    return ids


async def _fetch_existing_world_element_ids() -> set[str]:
    """Return IDs of non-deleted world elements stored in Neo4j."""
    records = await neo4j_manager.execute_read_query(
        "MATCH (we:WorldElement:Entity)"
        " WHERE we.is_deleted IS NULL OR we.is_deleted = FALSE"
        " RETURN we.id AS id"
    )
    return {rec["id"] for rec in records if rec and rec.get("id")}


def _generate_world_container_statements(
    overview_details: dict[str, Any], wc_id: str, novel_id: str
) -> list[tuple[str, dict[str, Any]]]:
    """Create cypher statements for the world container node."""
    if not isinstance(overview_details, dict):
        return []
    wc_props: dict[str, Any] = {
        "id": wc_id,
        "overview_description": str(overview_details.get("description", "")),
        KG_IS_PROVISIONAL: overview_details.get(
            kg_keys.source_quality_key(settings.KG_PREPOPULATION_CHAPTER_NUM)
        )
        == "provisional_from_unrevised_draft",
    }
    for key, val in overview_details.items():
        if isinstance(val, str | int | float | bool) and key not in wc_props:
            wc_props[key] = val
    return [
        (
            """
        MERGE (wc:Entity {id: $id_val})
        ON CREATE SET wc:WorldContainer, wc = $props, wc.created_ts = timestamp()
        ON MATCH SET  wc:WorldContainer, wc = $props, wc.updated_ts = timestamp()
        """,
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


def _prepare_world_element_props(
    category: str, name: str, details: dict[str, Any]
) -> dict[str, Any]:
    """Build property dictionary for a ``WorldElement`` node."""
    we_props = {
        "id": _build_world_element_id(category, name, details),
        "name": name,
        "category": category,
    }
    created_chap_val = details.get(
        KG_NODE_CREATED_CHAPTER,
        details.get("created_chapter", settings.KG_PREPOPULATION_CHAPTER_NUM),
    )
    # Ensure created_chap_val is an int for internal logic and provisional checks
    created_chap_int = int(created_chap_val)
    we_props[KG_NODE_CREATED_CHAPTER] = created_chap_int

    is_prov_bool = False
    # Use the integer form for generating the source quality key
    sq_key = kg_keys.source_quality_key(created_chap_int)
    if details.get(sq_key) == "provisional_from_unrevised_draft":
        is_prov_bool = True
    elif details.get(KG_IS_PROVISIONAL) is True: # Check direct KG_IS_PROVISIONAL key
        is_prov_bool = True
    elif details.get("is_provisional") is True: # Check generic is_provisional key
        is_prov_bool = True

    we_props[KG_IS_PROVISIONAL] = is_prov_bool # Store as boolean
    we_props["is_deleted"] = False # Store as boolean

    # Populate other properties, ensuring they are of appropriate types for Neo4j
    for key, val in details.items():
        if (
            isinstance(val, (str, int, float, bool)) # Allowed scalar types
            and key not in we_props # Avoid overwriting already set properties
            and not key.startswith(kg_keys.ELABORATION_PREFIX)
            and not key.startswith(kg_keys.ADDED_PREFIX)
            and not key.startswith(kg_keys.SOURCE_QUALITY_PREFIX)
            and key not in { # Set of keys managed by other logic
                "goals", "rules", "key_elements", "traits", # List properties
                "id", "name", "category", "created_chapter", "is_provisional" # Core/handled props
            }
        ):
            we_props[key] = val # Assign directly
    return we_props


def _generate_list_value_statements(
    we_id: str, details: dict[str, Any]
) -> list[tuple[str, dict[str, Any]]]:
    """Create cypher statements for list properties like goals or rules."""
    statements: list[tuple[str, dict[str, Any]]] = []
    list_prop_map = {
        "goals": "HAS_GOAL",
        "rules": "HAS_RULE",
        "key_elements": "HAS_KEY_ELEMENT",
        "traits": "HAS_TRAIT_ASPECT",
    }
    for prop_key, rel_name in list_prop_map.items():
        current_values = {
            str(v).strip()
            for v in details.get(prop_key, [])
            if isinstance(v, str) and str(v).strip()
        }
        statements.append(
            (
                f"""
            MATCH (we:WorldElement:Entity {{id: $we_id_val}})-[r:{rel_name}]->(v:ValueNode:Entity {{type: $value_node_type}})
            WHERE NOT v.value IN $current_values_list
            DELETE r
            """,
                {
                    "we_id_val": we_id,
                    "value_node_type": prop_key,
                    "current_values_list": list(current_values),
                },
            )
        )
        if current_values:
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
                        "we_id_val": we_id,
                        "value_node_type": prop_key,
                        "current_values_list": list(current_values),
                    },
                )
            )
    return statements


def _generate_elaboration_statements(
    we_id: str, details: dict[str, Any], item_name: str
) -> list[tuple[str, dict[str, Any]]]:
    """Create cypher statements for ``WorldElaborationEvent`` nodes."""
    statements: list[tuple[str, dict[str, Any]]] = [
        (
            """
        MATCH (we:WorldElement:Entity {id: $we_id_val})-[r:ELABORATED_IN_CHAPTER]->(elab:WorldElaborationEvent:Entity)
        DETACH DELETE elab, r
        """,
            {"we_id_val": we_id},
        )
    ]
    for key, value in details.items():
        if (
            key.startswith(kg_keys.ELABORATION_PREFIX)
            and isinstance(value, str)
            and value.strip()
        ):
            try:
                chap_num = kg_keys.parse_elaboration_key(key)
                if chap_num is None:
                    logger.warning(
                        "Could not parse chapter number from world elab key: %s for item %s",
                        key,
                        item_name,
                    )
                    continue
                elab_is_prov = False
                sq_key = kg_keys.source_quality_key(chap_num)
                if details.get(sq_key) == "provisional_from_unrevised_draft":
                    elab_is_prov = True
                elab_props = {
                    "id": f"elab_{we_id}_ch{chap_num}_{hash(value.strip())}",
                    "summary": value.strip(),
                    KG_NODE_CHAPTER_UPDATED: chap_num,
                    KG_IS_PROVISIONAL: elab_is_prov,
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
            except ValueError:
                logger.warning(
                    "Could not parse chapter for world elab key: %s for item %s",
                    key,
                    item_name,
                )
    return statements


def _cleanup_orphan_value_nodes() -> tuple[str, dict[str, Any]]:
    """Return cypher to remove orphan ``ValueNode`` records."""
    return (
        """
    MATCH (v:ValueNode:Entity)
    WHERE NOT EXISTS((:WorldElement:Entity)-[]->(v)) AND NOT EXISTS((:Entity)-[:DYNAMIC_REL]->(v))
    DETACH DELETE v
    """,
        {},
    )


def _generate_world_element_statements(
    world_data: dict[str, Any], wc_id: str
) -> list[tuple[str, dict[str, Any]]]:
    """Create cypher statements for all ``WorldElement`` nodes."""
    statements: list[tuple[str, dict[str, Any]]] = []
    for category, items in world_data.items():
        if category == "_overview_" or not isinstance(items, dict):
            continue
        for name, details in items.items():
            if not isinstance(details, dict) or name.startswith(
                ("_", kg_keys.SOURCE_QUALITY_PREFIX, "category_updated_in_chapter_")
            ):
                continue
            we_id = _build_world_element_id(category, name, details)
            props = _prepare_world_element_props(category, name, details)
            statements.append(
                (
                    """
                MERGE (we:Entity {id: $id_val})
                ON CREATE SET we:WorldElement, we = $props, we.created_ts = timestamp()
                ON MATCH SET  we:WorldElement, we += $props, we.updated_ts = timestamp()
                """,
                    {"id_val": we_id, "props": props},
                )
            )
            statements.append(
                (
                    """
                MATCH (wc:WorldContainer:Entity {id: $wc_id_val})
                MATCH (we:WorldElement:Entity {id: $we_id_val})
                MERGE (wc)-[:CONTAINS_ELEMENT]->(we)
                """,
                    {"wc_id_val": wc_id, "we_id_val": we_id},
                )
            )
            statements.extend(_generate_list_value_statements(we_id, details))
            statements.extend(_generate_elaboration_statements(we_id, details, name))
    return statements


async def sync_full_state_from_object_to_db(world_data: dict[str, Any]) -> bool:
    """Persist the entire world-building state to Neo4j."""
    logger.info("Synchronizing world building data to Neo4j (non-destructive)...")

    novel_id = settings.MAIN_NOVEL_INFO_NODE_ID
    wc_id = settings.MAIN_WORLD_CONTAINER_NODE_ID

    # _build_sync_statements is now async, so it needs to be awaited
    all_statements = await _build_sync_statements(world_data, wc_id, novel_id)

    try:
        if all_statements:
            await neo4j_manager.execute_cypher_batch(all_statements)
        logger.info("Successfully synchronized world building data to Neo4j.")
        return True
    except Exception as e:  # pragma: no cover - log and return failure
        logger.error(f"Error synchronizing world building data: {e}", exc_info=True)
        return False


async def _build_sync_statements(
    world_data: dict[str, Any], wc_id: str, novel_id: str
) -> list[tuple[str, dict[str, Any]]]:
    """Build all Cypher statements for the sync operation."""
    statements: list[tuple[str, dict[str, Any]]] = []

    # 1. World Container Statements
    statements.extend(
        _generate_world_container_statements(
            world_data.get("_overview_", {}), wc_id, novel_id
        )
    )

    # 2. Orphaned World Element Deletion Statements
    try:
        orphan_statements = await _generate_orphan_deletion_statements(world_data)
        statements.extend(orphan_statements)
    except Exception as exc:
        logger.error(
            f"Failed to generate orphan deletion statements: {exc}", exc_info=True
        )
        # Depending on policy, you might want to raise here or return empty list
        # For now, we'll log and continue, potentially leaving orphans.
        pass

    # 3. World Element Creation/Update Statements
    statements.extend(_generate_world_element_statements(world_data, wc_id))

    # 4. Cleanup Orphan Value Nodes
    statements.append(_cleanup_orphan_value_nodes())

    return statements


async def _generate_orphan_deletion_statements(
    world_data: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """Generate statements to mark orphaned WorldElements as deleted."""
    input_ids = _collect_input_world_element_ids(world_data)
    try:
        existing_ids = await _fetch_existing_world_element_ids()
    except Exception as exc:  # pragma: no cover
        logger.error(
            f"Failed to retrieve existing WorldElement IDs from DB for orphan check: {exc}",
            exc_info=True,
        )
        raise  # Re-raise to be handled by the caller

    orphan_ids = existing_ids - input_ids
    if orphan_ids:
        return [
            (
                """
            MATCH (we:WorldElement:Entity)
            WHERE we.id IN $we_ids_to_delete
            SET we.is_deleted = TRUE
            """,
                {"we_ids_to_delete": list(orphan_ids)},
            )
        ]
    return []


async def _fetch_world_element_node(item_id: str) -> dict[str, Any] | None:
    """Fetch the WorldElement node, trying original ID and then resolved name."""
    query = (
        "MATCH (we:WorldElement:Entity {id: $id})"
        " WHERE we.is_deleted IS NULL OR we.is_deleted = FALSE"
        " RETURN we"
    )
    results = await neo4j_manager.execute_read_query(query, {"id": item_id})
    if results and results[0].get("we"):
        return results[0]["we"]

    alt_id = resolve_world_name(item_id)
    if alt_id and alt_id != item_id:
        results = await neo4j_manager.execute_read_query(query, {"id": alt_id})
        if results and results[0].get("we"):
            return results[0]["we"]
    return None


def _process_world_element_core_fields(
    we_node: dict[str, Any], item_id: str
) -> tuple[dict[str, Any], str | None, str | None]:
    """Process core fields from the Neo4j node into item_detail dict."""
    category = we_node.get("category")
    item_name = we_node.get("name")
    if not category or not item_name:
        logger.warning("WorldElement missing category or name for id '%s'.", item_id)
        return {}, None, None

    item_detail: dict[str, Any] = dict(we_node)
    item_detail.pop("created_ts", None)
    item_detail.pop("updated_ts", None)

    created_chapter_num = item_detail.pop(
        KG_NODE_CREATED_CHAPTER, settings.KG_PREPOPULATION_CHAPTER_NUM
    )
    item_detail["created_chapter"] = int(created_chapter_num)
    item_detail[kg_keys.added_key(created_chapter_num)] = True

    is_provisional_at_creation = item_detail.pop(KG_IS_PROVISIONAL, False)
    item_detail["is_provisional"] = is_provisional_at_creation
    if is_provisional_at_creation:
        item_detail[kg_keys.source_quality_key(created_chapter_num)] = (
            "provisional_from_unrevised_draft"
        )
    return item_detail, category, item_name


async def _fetch_and_process_list_properties(
    item_id: str, item_detail: dict[str, Any]
) -> None:
    """Fetch and process list-based properties (goals, rules, etc.)."""
    list_prop_map = {
        "goals": "HAS_GOAL",
        "rules": "HAS_RULE",
        "key_elements": "HAS_KEY_ELEMENT",
        "traits": "HAS_TRAIT_ASPECT",
    }
    for list_prop_key, rel_name_internal in list_prop_map.items():
        list_values_query = f"""
        MATCH (:WorldElement:Entity {{id: $we_id_param}})-[:{rel_name_internal}]->(v:ValueNode:Entity {{type: $value_node_type_param}})
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
                if res_item and res_item.get("item_value") is not None
            ]
        )


async def _fetch_and_process_elaborations(
    item_id: str, item_detail: dict[str, Any]
) -> None:
    """Fetch and process elaboration data."""
    elab_query = f"""
    MATCH (:WorldElement:Entity {{id: $we_id_param}})-[:ELABORATED_IN_CHAPTER]->(elab:WorldElaborationEvent:Entity)
    RETURN elab.summary AS summary, elab.{KG_NODE_CHAPTER_UPDATED} AS chapter, elab.{KG_IS_PROVISIONAL} AS is_provisional
    ORDER BY elab.chapter_updated ASC
    """
    elab_results = await neo4j_manager.execute_read_query(
        elab_query, {"we_id_param": item_id}
    )
    if elab_results:
        for elab_rec in elab_results:
            chapter_val = elab_rec.get("chapter")
            summary_val = elab_rec.get("summary")
            if chapter_val is not None and summary_val is not None:
                elab_key = kg_keys.elaboration_key(chapter_val)
                item_detail[elab_key] = summary_val
                if elab_rec.get(KG_IS_PROVISIONAL):  # Direct check on the field from DB
                    item_detail[kg_keys.source_quality_key(chapter_val)] = (
                        "provisional_from_unrevised_draft"
                    )


@alru_cache(maxsize=settings.WORLD_QUERY_CACHE_SIZE)
async def get_world_item_by_id(item_id: str) -> WorldItem | None:
    """Retrieve a single ``WorldItem`` from Neo4j by its ID or fall back to name."""
    logger.info("Loading world item '%s' from Neo4j...", item_id)

    we_node = await _fetch_world_element_node(item_id)
    if not we_node:
        logger.info("No world item found for id '%s'.", item_id)
        return None

    item_detail, category, item_name = _process_world_element_core_fields(
        we_node, item_id
    )
    if not category or not item_name:  # Check if processing failed
        return None  # Error already logged in _process_world_element_core_fields

    await _fetch_and_process_list_properties(item_id, item_detail)
    await _fetch_and_process_elaborations(item_id, item_detail)

    item_detail["id"] = (
        item_id  # Ensure the original/resolved ID is part of the final dict
    )
    return WorldItem.from_dict(category, item_name, item_detail)


@alru_cache(maxsize=settings.WORLD_QUERY_CACHE_SIZE)
async def get_all_world_item_ids_by_category() -> dict[str, list[str]]:
    """Return all world item IDs grouped by category."""
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


@alru_cache(maxsize=settings.WORLD_QUERY_CACHE_SIZE)
async def get_world_building_from_db(
    chapter_limit: int | None = None,
) -> dict[str, dict[str, WorldItem]]:
    """Load world elements grouped by category from Neo4j."""

    logger.info(
        "Loading decomposed world building data from Neo4j%s...",
        f" up to chapter {chapter_limit}" if chapter_limit is not None else "",
    )
    world_data: dict[str, dict[str, WorldItem]] = {}
    wc_id_param = settings.MAIN_WORLD_CONTAINER_NODE_ID

    await fix_missing_world_element_core_fields()
    WORLD_NAME_TO_ID.clear()

    await _load_and_cache_overview_item(wc_id_param, world_data)

    we_results = await _fetch_world_elements_from_db(chapter_limit)

    if not we_results:
        _populate_default_categories(world_data)
        logger.info(
            "No WorldElements found in Neo4j%s.",
            f" up to chapter {chapter_limit}" if chapter_limit is not None else "",
        )
        return world_data

    for record in we_results:
        _update_world_data_from_record(record, world_data, chapter_limit)

    _log_loaded_elements_count(world_data, chapter_limit)
    return world_data


async def _load_and_cache_overview_item(
    wc_id: str, world_data: dict[str, dict[str, WorldItem]]
) -> None:
    """Load the overview item and add it to world_data and WORLD_NAME_TO_ID."""
    overview_item = await _load_world_container(wc_id)
    if overview_item:
        world_data.setdefault("_overview_", {})["_overview_"] = overview_item
        WORLD_NAME_TO_ID[utils._normalize_for_id("_overview_")] = (
            utils._normalize_for_id("_overview_")
        )


async def _fetch_world_elements_from_db(
    chapter_limit: int | None,
) -> list[dict[str, Any]]:
    """Fetch raw world element data from the database."""
    we_params: dict[str, Any] = {"limit": chapter_limit}
    chapter_filter = ""
    if chapter_limit is not None:
        chapter_filter = (
            f"AND (we.{KG_NODE_CREATED_CHAPTER} IS NULL "
            f"OR we.{KG_NODE_CREATED_CHAPTER} <= $limit)"
        )
    query = _build_world_elements_query(chapter_filter)
    return await neo4j_manager.execute_read_query(query, we_params)


def _populate_default_categories(world_data: dict[str, dict[str, WorldItem]]) -> None:
    """Populate world_data with default categories if no elements are found."""
    for cat_key in DEFAULT_WORLD_CATEGORIES:
        world_data.setdefault(cat_key, {})


def _log_loaded_elements_count(
    world_data: dict[str, dict[str, WorldItem]], chapter_limit: int | None
) -> None:
    """Log the number of world elements loaded."""
    num_elements_loaded = sum(
        len(items) for cat, items in world_data.items() if cat != "_overview_"
    )
    logger.info(
        "Successfully loaded and recomposed world building data (%d elements) from Neo4j%s.",
        num_elements_loaded,
        f" up to chapter {chapter_limit}" if chapter_limit is not None else "",
    )


async def get_world_elements_for_snippet_from_db(
    category: str, chapter_limit: int, item_limit: int
) -> list[dict[str, Any]]:
    """Return a subset of world elements for prompt context."""

    query = f"""
    MATCH (we:WorldElement:Entity {{category: $category_param}})
    WHERE (we.{KG_NODE_CREATED_CHAPTER} IS NULL OR we.{KG_NODE_CREATED_CHAPTER} <= $chapter_limit_param)

    OPTIONAL MATCH (we)-[:ELABORATED_IN_CHAPTER]->(elab:WorldElaborationEvent:Entity)
    WHERE elab.{KG_NODE_CHAPTER_UPDATED} <= $chapter_limit_param AND elab.{KG_IS_PROVISIONAL} = TRUE
    
    WITH we, COUNT(elab) AS provisional_elaborations_count
    WITH we, ( we.{KG_IS_PROVISIONAL} = TRUE OR provisional_elaborations_count > 0 ) AS is_item_provisional_overall
    
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
                desc = (
                    str(desc_val) if desc_val is not None else ""
                )  # Ensure desc is a string

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
            f"Error fetching world elements for snippet (cat {category}): {e}",
            exc_info=True,
        )
    return items


async def find_thin_world_elements_for_enrichment() -> list[dict[str, Any]]:
    """Finds WorldElement nodes that are considered 'thin' (e.g., missing description)."""
    query = """
    MATCH (we:WorldElement)
    WHERE (we.description IS NULL OR we.description = '') AND (we.is_deleted IS NULL OR we.is_deleted = FALSE)
    RETURN we.id AS id, we.name AS name, we.category as category
    LIMIT 20 // Limit to avoid overwhelming the LLM in one cycle
    """
    try:
        results = await neo4j_manager.execute_read_query(query)
        return results if results else []
    except Exception as e:
        logger.error(f"Error finding thin world elements: {e}", exc_info=True)
        return []


async def fix_missing_world_element_core_fields() -> int:
    """Populate missing ``id``, ``name``, or ``category`` on WorldElements."""

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
    except Exception as exc:  # pragma: no cover - narrow DB errors
        logger.error(
            "Error fetching WorldElements missing core fields: %s",
            exc,
            exc_info=True,
        )
        return 0

    if not results:
        return 0

    statements: list[tuple[str, dict[str, Any]]] = []

    for rec in results:
        neo_id = rec.get("nid")
        if neo_id is None:
            continue

        w_id = rec.get("id")
        name = rec.get("name")
        category = rec.get("category")

        if isinstance(w_id, str):
            w_id = w_id.strip() or None
        if isinstance(name, str):
            name = name.strip() or None
        if isinstance(category, str):
            category = category.strip() or None

        props: dict[str, Any] = {}

        if not name and isinstance(w_id, str):
            name_part = w_id.split("_", 1)[-1]
            props["name"] = name_part.replace("_", " ").title()
            name = props["name"]
        elif not name:
            props["name"] = "Unnamed Element"
            name = props["name"]

        if not category:
            if isinstance(w_id, str):
                props["category"] = w_id.split("_")[0]
            else:
                props["category"] = "unknown_category"
            category = props["category"]

        if not w_id and name:
            props["id"] = (
                f"{utils._normalize_for_id(category)}_{utils._normalize_for_id(name)}"
            )

        if props:
            statements.append(
                (
                    "MATCH (we:WorldElement) WHERE elementId(we) = $nid SET we += $props",
                    {"nid": neo_id, "props": props},
                )
            )

    if not statements:
        return 0

    try:
        await neo4j_manager.execute_cypher_batch(statements)
        logger.info("Filled missing core fields for %d WorldElements.", len(statements))
    except Exception as exc:  # pragma: no cover - narrow DB errors
        logger.error(
            "Error updating WorldElements with missing core fields: %s",
            exc,
            exc_info=True,
        )
        return 0

    return len(statements)


async def remove_world_element_trait_aspect(element_id: str, trait_value: str) -> bool:
    """Remove a trait aspect from a world element."""
    query = (
        "MATCH (:WorldElement {id: $we_id})-"
        "[r:HAS_TRAIT_ASPECT]->(v:ValueNode {type: 'traits', value: $trait})"
        " DELETE r"
    )
    cleanup_query = (
        "MATCH (v:ValueNode {type: 'traits', value: $trait})"
        " WHERE NOT EXISTS((:WorldElement)-[:HAS_TRAIT_ASPECT]->(v))"
        " DETACH DELETE v"
    )
    try:
        await neo4j_manager.execute_write_query(
            query, {"we_id": element_id, "trait": trait_value}
        )
        await neo4j_manager.execute_write_query(cleanup_query, {"trait": trait_value})
        return True
    except Exception as exc:  # pragma: no cover - log but return False
        logger.error(
            "Error removing trait '%s' from world element %s: %s",
            trait_value,
            element_id,
            exc,
        )
        return False
