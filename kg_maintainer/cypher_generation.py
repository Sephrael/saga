"""Generate Cypher statements for persisting KG data."""

from typing import Dict, List, Tuple, Any
import json
import logging

import config
from .models import CharacterProfile, WorldItem

logger = logging.getLogger(__name__)


def generate_character_node_cypher(profile: CharacterProfile) -> List[Tuple[str, Dict[str, Any]]]:
    """Create Cypher statements to persist a character profile."""
    statements: List[Tuple[str, Dict[str, Any]]] = []
    props = profile.to_dict()
    basic_props = {k: v for k, v in props.items() if isinstance(v, (str, int, float, bool))}
    statements.append(
        (
            "MERGE (c:Entity {name: $name}) SET c:Character SET c += $props",
            {"name": profile.name, "props": basic_props},
        )
    )

    statements.append(
        (
            "MATCH (ni:NovelInfo:Entity {id: $novel_id})"
            " MATCH (c:Character:Entity {name: $name})"
            " MERGE (ni)-[:HAS_CHARACTER]->(c)",
            {"novel_id": config.MAIN_NOVEL_INFO_NODE_ID, "name": profile.name},
        )
    )

    if profile.traits:
        for trait in profile.traits:
            statements.append(
                (
                    "MATCH (c:Character:Entity {name: $name}) MERGE (t:Entity:Trait {name: $trait}) MERGE (c)-[:HAS_TRAIT]->(t)",
                    {"name": profile.name, "trait": trait},
                )
            )
    return statements


def generate_world_element_node_cypher(item: WorldItem) -> List[Tuple[str, Dict[str, Any]]]:
    """Return Cypher statements to persist a world element and link it.

    Neo4j enforces a global uniqueness constraint on ``Entity.name``. To avoid
    conflicts when a world element shares its name with another entity, we
    merge on ``name`` and store ``id`` as a property. A relationship to the main
    ``WorldContainer`` is also ensured.
    """

    props = item.to_dict()
    props.update({"name": item.name, "category": item.category})

    safe_props: Dict[str, Any] = {}
    for key, value in props.items():
        if isinstance(value, (str, int, float, bool)):
            safe_props[key] = value
        else:
            safe_props[key] = json.dumps(value, ensure_ascii=False)

    statements = [
        (
            "MERGE (we:Entity {name: $name}) SET we:WorldElement SET we += $props",
            {"name": item.name, "props": safe_props},
        ),
        (
            "MATCH (wc:WorldContainer:Entity {id: $wc_id})"
            " MATCH (we:WorldElement:Entity {name: $name})"
            " MERGE (wc)-[:CONTAINS_ELEMENT]->(we)",
            {"wc_id": config.MAIN_WORLD_CONTAINER_NODE_ID, "name": item.name},
        ),
    ]

    return statements

def generate_world_element_node_cypher(item: WorldItem) -> Tuple[str, Dict[str, Any]]:
    """Create Cypher for a single world element node."""
    props = item.to_dict()
    props.update({"name": item.name, "category": item.category})
    return (
        "MERGE (we:Entity {id: $id}) SET we:WorldElement SET we = $props",
        {"id": item.id, "props": props},
    )
