"""Generate Cypher statements for persisting KG data."""

from typing import Dict, List, Tuple, Any
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
    """Create Cypher for a single world element node."""
    props = item.to_dict()
    props.update({"name": item.name, "category": item.category})
    return (
        "MERGE (we:Entity {id: $id}) SET we:WorldElement SET we = $props",
        {"id": item.id, "props": props},
    )
