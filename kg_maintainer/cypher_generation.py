# kg_maintainer/cypher_generation.py
from typing import Dict, List, Tuple, Any
import json
import logging

import config  # For MAIN_NOVEL_INFO_NODE_ID, MAIN_WORLD_CONTAINER_NODE_ID
from .models import CharacterProfile, WorldItem
from kg_constants import (
    KG_NODE_CREATED_CHAPTER,
    KG_IS_PROVISIONAL,
    KG_REL_CHAPTER_ADDED,
)

logger = logging.getLogger(__name__)


def generate_character_node_cypher(
    profile: CharacterProfile, chapter_number_for_delta: int
) -> List[Tuple[str, Dict[str, Any]]]:
    """Create Cypher statements to persist or update a single character profile's core attributes and direct relationships."""
    statements: List[Tuple[str, Dict[str, Any]]] = []

    props_from_profile = profile.to_dict()

    basic_props = {
        k: v
        for k, v in props_from_profile.items()
        if isinstance(v, (str, int, float, bool))
        and k not in ["name", "traits", "relationships"]
        and not k.startswith("development_in_chapter_")
        and not k.startswith(
            "source_quality_chapter_"
        )  # These are handled by events/flags
    }
    if isinstance(profile.updates, dict):
        for k_update, v_update in profile.updates.items():
            if (
                isinstance(v_update, (str, int, float, bool))
                and not k_update.startswith("development_in_chapter_")
                and not k_update.startswith("source_quality_chapter_")
            ):
                if k_update not in basic_props:
                    basic_props[k_update] = v_update

    # Ensure is_provisional is set based on the current chapter's source quality if available
    current_chapter_source_quality_key = (
        f"source_quality_chapter_{chapter_number_for_delta}"
    )
    if (
        isinstance(profile.updates, dict)
        and profile.updates.get(current_chapter_source_quality_key)
        == "provisional_from_unrevised_draft"
    ):
        basic_props[KG_IS_PROVISIONAL] = True
    elif (
        KG_IS_PROVISIONAL not in basic_props
    ):  # Default if not specified for this update
        basic_props[KG_IS_PROVISIONAL] = False

    statements.append(
        (
            """
            MERGE (c:Entity {name: $name})
            ON CREATE SET
                c:Character,
                c += $props,
                c.created_ts = timestamp()
            ON MATCH SET
                c:Character,
                c += $props,
                c.updated_ts = timestamp()
            """,
            {"name": profile.name, "props": basic_props},
        )
    )

    statements.append(
        (
            """
            MATCH (ni:NovelInfo:Entity {id: $novel_id})
            MATCH (c:Character:Entity {name: $name})
            MERGE (ni)-[:HAS_CHARACTER]->(c)
            """,
            {"novel_id": config.MAIN_NOVEL_INFO_NODE_ID, "name": profile.name},
        )
    )

    if profile.traits:
        for trait_name in profile.traits:
            if isinstance(trait_name, str) and trait_name.strip():
                statements.append(
                    (
                        """
                        MATCH (c:Character:Entity {name: $name})
                        MERGE (t:Entity {name: $trait_name})
                            ON CREATE SET t:Trait, t.created_ts = timestamp()
                            ON MATCH SET t:Trait
                        MERGE (c)-[:HAS_TRAIT]->(t)
                        """,
                        {"name": profile.name, "trait_name": trait_name.strip()},
                    )
                )

    # Development Events from profile.updates
    # This should ONLY process the development event for the CURRENT chapter_number_for_delta
    dev_event_key_for_current_chapter = (
        f"development_in_chapter_{chapter_number_for_delta}"
    )
    if (
        isinstance(profile.updates, dict)
        and dev_event_key_for_current_chapter in profile.updates
    ):
        dev_event_summary = profile.updates[dev_event_key_for_current_chapter]
        if isinstance(dev_event_summary, str) and dev_event_summary.strip():
            dev_event_id = f"dev_{profile.name}_ch{chapter_number_for_delta}_{hash(dev_event_summary)}"
            dev_event_props = {
                "id": dev_event_id,
                "summary": dev_event_summary,
                "chapter_updated": chapter_number_for_delta,  # Use the passed chapter number
                KG_IS_PROVISIONAL: basic_props.get(
                    KG_IS_PROVISIONAL, False
                ),  # Inherit provisional status
            }
            statements.append(
                (
                    """
                    MATCH (c:Character:Entity {name: $name})
                    MERGE (dev:Entity {id: $dev_event_id})
                        ON CREATE SET dev:DevelopmentEvent, dev = $props, dev.created_ts = timestamp()
                        ON MATCH SET  dev:DevelopmentEvent, dev = $props, dev.updated_ts = timestamp()
                    MERGE (c)-[:DEVELOPED_IN_CHAPTER]->(dev)
                    """,
                    {
                        "name": profile.name,
                        "dev_event_id": dev_event_id,
                        "props": dev_event_props,
                    },
                )
            )

    # Explicit relationships from profile.relationships (assumed to be updates for current chapter_number_for_delta)
    if profile.relationships:
        for target_char_name, rel_detail in profile.relationships.items():
            if isinstance(target_char_name, str) and target_char_name.strip():
                rel_type_str = "RELATED_TO"
                rel_cypher_props = {}

                if isinstance(rel_detail, str) and rel_detail.strip():
                    rel_cypher_props["description"] = rel_detail.strip()
                    if rel_detail.isupper() and " " not in rel_detail:
                        rel_type_str = rel_detail
                elif isinstance(rel_detail, dict):
                    rel_type_str = (
                        str(rel_detail.get("type", rel_type_str))
                        .upper()
                        .replace(" ", "_")
                    )
                    for k_rel, v_rel in rel_detail.items():
                        if isinstance(v_rel, (str, int, float, bool)):
                            rel_cypher_props[k_rel] = v_rel
                    rel_cypher_props.pop("type", None)

                # Tag relationship with current chapter metadata
                rel_cypher_props[KG_REL_CHAPTER_ADDED] = chapter_number_for_delta
                rel_cypher_props[KG_IS_PROVISIONAL] = basic_props.get(
                    KG_IS_PROVISIONAL, False
                )
                rel_cypher_props["source_profile_managed"] = (
                    True  # Mark as managed by character profile system
                )

                statements.append(
                    (
                        """
                        MATCH (c1:Character:Entity {name: $source_name})
                        MERGE (c2:Entity {name: $target_name})
                            ON CREATE SET c2:Character, c2.description = 'Auto-created via relationship from ' + $source_name, c2.created_ts = timestamp()
                            ON MATCH SET c2:Character
                        
                        MERGE (c1)-[r:DYNAMIC_REL {type: $rel_type_str, chapter_added: $chapter_num_delta}]->(c2)
                            ON CREATE SET r = $rel_props, r.created_ts = timestamp()
                            ON MATCH SET  r += $rel_props, r.updated_ts = timestamp()
                        """,
                        {
                            "source_name": profile.name,
                            "target_name": target_char_name.strip(),
                            "rel_type_str": rel_type_str,
                            "chapter_num_delta": chapter_number_for_delta,  # For unique MERGE key
                            "rel_props": rel_cypher_props,
                        },
                    )
                )
    return statements


def generate_world_element_node_cypher(
    item: WorldItem, chapter_number_for_delta: int
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Create Cypher statements to persist or update a single world element.
    """
    statements: List[Tuple[str, Dict[str, Any]]] = []

    node_props = {
        "name": item.name,
        "category": item.category,
        # KG_NODE_CREATED_CHAPTER is set when the item is first seen.
        # Updates might affect KG_IS_PROVISIONAL for the *current chapter's context*.
        KG_NODE_CREATED_CHAPTER: item.created_chapter,  # This should be its original creation chapter
    }

    current_chapter_source_quality_key = (
        f"source_quality_chapter_{chapter_number_for_delta}"
    )
    if (
        isinstance(item.properties, dict)
        and item.properties.get(current_chapter_source_quality_key)
        == "provisional_from_unrevised_draft"
    ):
        node_props[KG_IS_PROVISIONAL] = True
    elif (
        KG_IS_PROVISIONAL not in node_props
    ):  # Default if not specified for this update context
        # The item.is_provisional itself reflects the overall status, not specific to this delta.
        # For the node itself, its core provisional status is important.
        node_props[KG_IS_PROVISIONAL] = item.is_provisional

    if isinstance(item.properties, dict):
        for key, value in item.properties.items():
            if (
                isinstance(value, (str, int, float, bool))
                and key not in node_props
                and not key.startswith("elaboration_in_chapter_")
                and not key.startswith("source_quality_chapter_")
                and not key.startswith("added_in_chapter_")
                and key not in ["goals", "rules", "key_elements", "traits"]
            ):
                node_props[key] = value
            elif isinstance(value, (list, dict)) and key not in [
                "goals",
                "rules",
                "key_elements",
                "traits",
            ]:
                try:
                    node_props[key] = json.dumps(value, ensure_ascii=False)
                except TypeError:
                    logger.warning(
                        f"Could not JSON serialize property '{key}' for WorldElement '{item.id}'. Skipping."
                    )

    statements.append(
        (
            """
            MERGE (we:Entity {id: $id})
            ON CREATE SET
                we:WorldElement,
                we += $props,
                we.created_ts = timestamp()
            ON MATCH SET
                we:WorldElement,
                we += $props,
                we.updated_ts = timestamp()
            """,
            {"id": item.id, "props": node_props},
        )
    )

    statements.append(
        (
            """
            MATCH (wc:WorldContainer:Entity {id: $wc_id})
            MATCH (we:WorldElement:Entity {id: $we_id})
            MERGE (wc)-[:CONTAINS_ELEMENT]->(we)
            """,
            {"wc_id": config.MAIN_WORLD_CONTAINER_NODE_ID, "we_id": item.id},
        )
    )

    list_properties_to_relate = {
        "goals": "HAS_GOAL",
        "rules": "HAS_RULE",
        "key_elements": "HAS_KEY_ELEMENT",
        "traits": "HAS_TRAIT_ASPECT",
    }

    for prop_key, rel_type in list_properties_to_relate.items():
        list_value = []
        if isinstance(item.properties, dict):
            list_value = item.properties.get(prop_key, [])

        if isinstance(list_value, list):
            for value_str_unstripped in list_value:
                if isinstance(value_str_unstripped, str):
                    value_str = value_str_unstripped.strip()
                    if value_str:
                        # MERGE ValueNode based on its value and type, not a separate ID.
                        statements.append(
                            (
                                f"""
                                MATCH (we:WorldElement:Entity {{id: $we_id}})
                                MERGE (v:Entity:ValueNode {{value: $value_str, type: $prop_key}})
                                  ON CREATE SET v.created_ts = timestamp()
                                  // ON MATCH SET can be omitted if no other props of ValueNode change
                                MERGE (we)-[:{rel_type}]->(v)
                                """,
                                {
                                    "we_id": item.id,
                                    "value_str": value_str,  # This is the constrained property
                                    "prop_key": prop_key,  # This is the other constrained property
                                },
                            )
                        )

    # Handle Elaboration Events for the current chapter_number_for_delta
    elab_event_key_for_current_chapter = (
        f"elaboration_in_chapter_{chapter_number_for_delta}"
    )
    if (
        isinstance(item.properties, dict)
        and elab_event_key_for_current_chapter in item.properties
    ):
        elab_summary = item.properties[elab_event_key_for_current_chapter]
        if isinstance(elab_summary, str) and elab_summary.strip():
            elab_event_id = (
                f"elab_{item.id}_ch{chapter_number_for_delta}_{hash(elab_summary)}"
            )
            elab_props = {
                "id": elab_event_id,
                "summary": elab_summary,
                "chapter_updated": chapter_number_for_delta,  # Use passed chapter number
                KG_IS_PROVISIONAL: node_props.get(
                    KG_IS_PROVISIONAL, False
                ),  # Inherit from node's provisional status for this delta
            }
            statements.append(
                (
                    """
                    MATCH (we:WorldElement:Entity {id: $we_id})
                    MERGE (elab:Entity {id: $elab_event_id})
                        ON CREATE SET elab:WorldElaborationEvent, elab = $props, elab.created_ts = timestamp()
                        ON MATCH SET  elab:WorldElaborationEvent, elab = $props, elab.updated_ts = timestamp()
                    MERGE (we)-[:ELABORATED_IN_CHAPTER]->(elab)
                    """,
                    {
                        "we_id": item.id,
                        "elab_event_id": elab_event_id,
                        "props": elab_props,
                    },
                )
            )
    return statements
