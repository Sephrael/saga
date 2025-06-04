"""Generate Cypher statements for persisting KG data."""

from typing import Dict, List, Tuple, Any
import json
import logging
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
    if profile.traits:
        for trait in profile.traits:
            statements.append(
                (
                    "MATCH (c:Character:Entity {name: $name}) MERGE (t:Entity:Trait {name: $trait}) MERGE (c)-[:HAS_TRAIT]->(t)",
                    {"name": profile.name, "trait": trait},
                )
            )
    return statements


def generate_world_element_node_cypher(item: WorldItem) -> Tuple[str, Dict[str, Any]]:
    """Create Cypher for a single world element node.

    Neo4j enforces a global uniqueness constraint on ``Entity.name``. To avoid
    conflicts when a world element shares its name with another entity
    (e.g. a character), we merge on ``name`` and simply set ``id`` as a
    property.
    """

    props = item.to_dict()
    props.update({"name": item.name, "category": item.category})

    safe_props: Dict[str, Any] = {}
    for key, value in props.items():
        if isinstance(value, (str, int, float, bool)):
            safe_props[key] = value
        else:
            safe_props[key] = json.dumps(value, ensure_ascii=False)

    return (
        "MERGE (we:Entity {name: $name}) SET we:WorldElement SET we += $props",
        {"name": item.name, "props": safe_props},
    )
