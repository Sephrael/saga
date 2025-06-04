"""Dataclasses representing structured knowledge graph entities."""

from dataclasses import dataclass, field
from typing import Dict, List, Any

from kg_constants import (
    KG_IS_PROVISIONAL,
    KG_NODE_CREATED_CHAPTER,
)

@dataclass
class CharacterProfile:
    """Structured information about a character."""
    name: str
    description: str = ""
    traits: List[str] = field(default_factory=list)
    relationships: Dict[str, Any] = field(default_factory=dict)
    status: str = ""
    updates: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "CharacterProfile":
        description = data.get("description", "")
        traits = list(data.get("traits", []))
        relationships = dict(data.get("relationships", {}))
        status = data.get("status", "")
        remaining = {k: v for k, v in data.items() if k not in {"description", "traits", "relationships", "status"}}
        return cls(name, description, traits, relationships, status, remaining)

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "description": self.description,
            "traits": self.traits,
            "relationships": self.relationships,
            "status": self.status,
        }
        data.update(self.updates)
        return data


@dataclass
class WorldItem:
    """Structured information about a world element."""
    id: str
    category: str
    name: str
    created_chapter: int = 0
    is_provisional: bool = False
    properties: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, category: str, name: str, data: Dict[str, Any]) -> "WorldItem":
        item_id = data.get("id", f"{category}_{name}")
        created_chapter = int(data.get(KG_NODE_CREATED_CHAPTER, 0))
        is_provisional = bool(data.get(KG_IS_PROVISIONAL, False))
        props = {
            k: v
            for k, v in data.items()
            if k not in {"id", KG_NODE_CREATED_CHAPTER, KG_IS_PROVISIONAL}
        }
        return cls(item_id, category, name, created_chapter, is_provisional, props)

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "id": self.id,
            KG_NODE_CREATED_CHAPTER: self.created_chapter,
            KG_IS_PROVISIONAL: self.is_provisional,
        }
        data.update(self.properties)
        return data
