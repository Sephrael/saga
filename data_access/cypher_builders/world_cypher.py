# data_access/cypher_builders/world_cypher.py
import json
from typing import Any

import kg_constants as kg_keys
import structlog
from config import settings
from kg_constants import (
    KG_IS_PROVISIONAL,
    KG_NODE_CREATED_CHAPTER,
)
from kg_maintainer.models import WorldItem

logger = structlog.get_logger(__name__)


def _prepare_node_properties(
    item: WorldItem, chapter_number_for_delta: int
) -> dict[str, Any]:
    """Prepares the properties dictionary for the main WorldElement node."""
    node_props = {
        "name": item.name,
        "category": item.category,
        KG_NODE_CREATED_CHAPTER: item.created_chapter,
        "is_deleted": False,  # Default to False
    }

    # Determine provisional status
    current_chapter_source_quality_key = kg_keys.source_quality_key(
        chapter_number_for_delta
    )
    if (
        isinstance(item.properties, dict)
        and item.properties.get(current_chapter_source_quality_key)
        == "provisional_from_unrevised_draft"
    ):
        node_props[KG_IS_PROVISIONAL] = True
    else:
        # Fallback to item's own provisional status if not set by current chapter's source quality
        node_props[KG_IS_PROVISIONAL] = item.is_provisional

    # Add other properties, serializing complex types
    if isinstance(item.properties, dict):
        # Keys that are handled by relationships or are metadata, not direct node properties
        skipped_property_keys = {
            "goals",
            "rules",
            "key_elements",
            "traits",  # Handled by relationships
            kg_keys.elaboration_key(0).rsplit("_", 1)[0],  # Elaboration prefix
            kg_keys.source_quality_key(0).rsplit("_", 1)[0],  # Source quality prefix
            kg_keys.added_key(0).rsplit("_", 1)[0],  # Added prefix
        }
        # Also skip keys already set (name, category, created_chapter, is_deleted, KG_IS_PROVISIONAL)
        already_set_keys = set(node_props.keys())

        for key, value in item.properties.items():
            # Check if key starts with any of the prefixes in skipped_property_keys
            is_skipped_prefix = any(
                key.startswith(prefix)
                for prefix in skipped_property_keys
                if prefix.endswith("PREFIX")
            )  # Simplified check, assumes prefixes are identifiable

            if (
                key in already_set_keys
                or key in skipped_property_keys
                or is_skipped_prefix
                or key.startswith(kg_keys.ELABORATION_PREFIX)
                or key.startswith(kg_keys.SOURCE_QUALITY_PREFIX)
                or key.startswith(kg_keys.ADDED_PREFIX)
            ):
                continue

            if isinstance(value, (str, int, float, bool)):
                node_props[key] = value
            elif isinstance(value, (list, dict)):
                try:
                    node_props[key] = json.dumps(value, ensure_ascii=False)
                except TypeError:
                    logger.warning(
                        "Could not JSON serialize property '%s' for WorldElement '%s'. Skipping.",
                        key,
                        item.id,
                    )
    return node_props


_LIST_PROPERTIES_TO_RELATE = {
    "goals": "HAS_GOAL",
    "rules": "HAS_RULE",
    "key_elements": "HAS_KEY_ELEMENT",
    "traits": "HAS_TRAIT_ASPECT",
}


def _generate_list_property_relationship_statements(
    item_id: str,
    item_properties: dict[str, Any] | None,
) -> list[tuple[str, dict[str, Any]]]:
    """Generates Cypher for creating ValueNodes and relationships for list-based properties."""
    statements: list[tuple[str, dict[str, Any]]] = []
    if not isinstance(item_properties, dict):
        return statements

    for prop_key, rel_type in _LIST_PROPERTIES_TO_RELATE.items():
        list_value = item_properties.get(prop_key, [])
        if not isinstance(list_value, list):
            # Log or handle if a property expected to be a list isn't
            logger.debug(
                "Property '%s' for item '%s' is not a list, skipping relationship generation.",
                prop_key,
                item_id,
            )
            continue

        for value_str_unstripped in list_value:
            if isinstance(value_str_unstripped, str):
                value_str = value_str_unstripped.strip()
                if value_str:
                    statements.append(
                        (
                            f"""
                            MATCH (
                                we:WorldElement:Entity {{id: $we_id}}
                            )
                            MERGE (
                                v:Entity:ValueNode {{
                                    value: $value_str,
                                    type: $prop_key
                                }}
                            )
                              ON CREATE SET v.created_ts = timestamp()
                            MERGE (we)-[:{rel_type}]->(v)
                            """,
                            {
                                "we_id": item_id,
                                "value_str": value_str,
                                "prop_key": prop_key,
                            },
                        )
                    )
    return statements


def _generate_elaboration_event_statement(
    item: WorldItem,
    chapter_number_for_delta: int,
    node_is_provisional: bool,  # Pass the determined provisional status of the main node
) -> tuple[str, dict[str, Any]] | None:
    """Generates Cypher for creating an WorldElaborationEvent node and relationship if applicable."""
    elab_event_key = kg_keys.elaboration_key(chapter_number_for_delta)

    if not isinstance(item.properties, dict) or elab_event_key not in item.properties:
        return None

    elab_summary = item.properties[elab_event_key]
    if not isinstance(elab_summary, str) or not elab_summary.strip():
        return None

    elab_event_id = f"elab_{item.id}_ch{chapter_number_for_delta}_{hash(elab_summary)}"
    elab_props = {
        "id": elab_event_id,
        "summary": elab_summary,
        "chapter_updated": chapter_number_for_delta,
        KG_IS_PROVISIONAL: node_is_provisional,  # Use the passed provisional status
    }
    statement = (
        """
        MATCH (we:WorldElement:Entity {id: $we_id})
        MERGE (elab:Entity {id: $elab_event_id})
            ON CREATE SET
                elab:WorldElaborationEvent,
                elab = $props,
                elab.created_ts = timestamp()
            ON MATCH SET
                elab:WorldElaborationEvent,
                elab = $props,
                elab.updated_ts = timestamp()
        MERGE (we)-[:ELABORATED_IN_CHAPTER]->(elab)
        """,
        {
            "we_id": item.id,
            "elab_event_id": elab_event_id,
            "props": elab_props,
        },
    )
    return statement


def generate_world_element_node_cypher(
    item: WorldItem, chapter_number_for_delta: int = 0
) -> list[tuple[str, dict[str, Any]]]:
    """Create Cypher statements for a world element update."""
    statements: list[tuple[str, dict[str, Any]]] = []

    # 1. Prepare main node properties
    node_props = _prepare_node_properties(item, chapter_number_for_delta)

    # 2. Statement for the WorldElement node itself
    statements.append(
        (
            """
            MERGE (we:Entity {id: $id})
            ON CREATE SET
                we:WorldElement,
                we += $props,
                we.created_ts = timestamp()
            ON MATCH SET
                we:WorldElement, /* Ensure label is present on match */
                we += $props,    /* Update properties on match */
                we.updated_ts = timestamp()
            """,
            {"id": item.id, "props": node_props},
        )
    )

    # 3. Statement to link to the main WorldContainer
    statements.append(
        (
            """
            MATCH (wc:WorldContainer:Entity {id: $wc_id})
            MATCH (we:WorldElement:Entity {id: $we_id})
            MERGE (wc)-[:CONTAINS_ELEMENT]->(we)
            """,
            {
                "wc_id": settings.MAIN_WORLD_CONTAINER_NODE_ID,
                "we_id": item.id,
            },
        )
    )

    # 4. Generate statements for list-based properties (goals, rules, etc.)
    list_prop_statements = _generate_list_property_relationship_statements(
        item.id, item.properties
    )
    statements.extend(list_prop_statements)

    # 5. Generate statement for elaboration event, if any
    # The provisional status of the main node is needed for the elaboration event
    node_is_provisional = node_props.get(KG_IS_PROVISIONAL, False)
    elaboration_statement = _generate_elaboration_event_statement(
        item, chapter_number_for_delta, node_is_provisional
    )
    if elaboration_statement:
        statements.append(elaboration_statement)

    return statements
