# kg_maintainer/models.py
import re  # For normalization in ID generation
from dataclasses import dataclass, field
from typing import Dict, List, Any

from kg_constants import (
    KG_IS_PROVISIONAL,
    KG_NODE_CREATED_CHAPTER,
)


def _normalize_for_id(text: str) -> str:
    """Normalize a string for use in an ID."""
    """Lowercase, convert spaces to underscores, remove problematic chars."""
    if not isinstance(text, str):  # Handle potential non-string input
        text = str(text)
    text = text.strip().lower()
    text = re.sub(r"['\"()]", "", text)
    # Remove apostrophes, quotes, parentheses
    text = re.sub(r"\s+", "_", text)
    # Replace whitespace with underscore
    text = re.sub(r"[^a-z0-9_]", "", text)
    # Keep only alphanumeric and underscore
    return text


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
        remaining = {
            k: v
            for k, v in data.items()
            if k not in {"description", "traits", "relationships", "status"}
        }
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

    id: str  # Canonical ID derived from normalized category and name
    category: str  # Display/canonical category
    name: str  # Display/canonical name
    created_chapter: int = 0
    is_provisional: bool = False
    properties: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls, category: str, name: str, data: Dict[str, Any]
    ) -> "WorldItem":
        """
        Creates a WorldItem.
        'category' and 'name' are the intended display/canonical values.
        The 'id' is always generated deterministically from normalized
        versions of these.
        Any 'id' in the 'data' dict is ignored.
        """
        if (
            not category
            or not isinstance(category, str)
            or not category.strip()
        ):
            raise ValueError("WorldItem category must be a non-empty string.")
        if not name or not isinstance(name, str) or not name.strip():
            raise ValueError(
                "WorldItem name must be a non-empty string "
                f"(for category '{category}')."
            )

        # Generate canonical ID from normalized category and name.
        # The 'name' and 'category' stored on the instance are the ones
        # passed as arguments.
        normalized_id_category = _normalize_for_id(category)
        normalized_id_name = _normalize_for_id(name)

        # Ensure no empty parts in ID after normalization
        if not normalized_id_category:
            normalized_id_category = "unknown_category"
        if not normalized_id_name:
            normalized_id_name = "unknown_name"

        item_id = f"{normalized_id_category}_{normalized_id_name}"

        created_chapter = int(data.get(KG_NODE_CREATED_CHAPTER, 0))
        is_provisional = bool(data.get(KG_IS_PROVISIONAL, False))

        # Properties from 'data', excluding any 'id' passed in, and KG flags
        props = {
            k: v
            for k, v in data.items()
            if k not in {"id", KG_NODE_CREATED_CHAPTER, KG_IS_PROVISIONAL}
        }

        return cls(
            item_id,
            category,
            name,
            created_chapter,
            is_provisional,
            props,
        )

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "id": self.id,
            KG_NODE_CREATED_CHAPTER: self.created_chapter,
            KG_IS_PROVISIONAL: self.is_provisional,
            "created_chapter": self.created_chapter,  # convenience duplicate
            "is_provisional": self.is_provisional,  # convenience duplicate
            # name and category are top-level attributes, not in properties
            # dict for this method's output
        }
        data.update(self.properties)
        return data
