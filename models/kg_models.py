"""Core data models for characters and world elements."""

from __future__ import annotations

from typing import Any

import kg_constants as kg_keys
import utils
from kg_constants import KG_IS_PROVISIONAL, KG_NODE_CREATED_CHAPTER
from pydantic import BaseModel, Field


class CharacterProfile(BaseModel):
    """Structured information about a character."""

    name: str
    description: str = ""
    traits: list[str] = Field(default_factory=list)
    relationships: dict[str, Any] = Field(default_factory=dict)
    status: str = "Unknown"
    updates: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> CharacterProfile:
        """Create a ``CharacterProfile`` from a raw dictionary."""

        known_fields = cls.model_fields.keys()
        profile_data = {k: v for k, v in data.items() if k in known_fields}
        updates_data = {k: v for k, v in data.items() if k not in known_fields}
        if "updates" in profile_data:
            updates_data.update(profile_data["updates"])
        profile_data["updates"] = updates_data
        return cls(name=name, **profile_data)

    def to_dict(self) -> dict[str, Any]:
        """Convert the profile to a flat dictionary."""

        data = self.model_dump(exclude={"name"})
        updates_data = data.pop("updates", {})
        data.update(updates_data)
        return data

    def get_development_summary(self, up_to_chapter: int | None = None) -> list[str]:
        """Return development notes up to a chapter number."""

        notes: list[str] = []
        prefix = kg_keys.DEVELOPMENT_PREFIX
        for key, val in sorted(self.updates.items()):
            if not key.startswith(prefix):
                continue
            chap = kg_keys.parse_development_key(key)
            if chap is None:
                continue
            if up_to_chapter is None or chap <= up_to_chapter:
                if isinstance(val, str):
                    notes.append(val)
        return notes


class WorldItem(BaseModel):
    """Structured information about a world element."""

    id: str
    category: str
    name: str
    created_chapter: int = 0
    is_provisional: bool = False
    properties: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, category: str, name: str, data: dict[str, Any]) -> WorldItem:
        """Create a ``WorldItem`` from a raw dictionary."""

        if not category or not isinstance(category, str) or not category.strip():
            raise ValueError("WorldItem category must be a non-empty string.")
        if not name or not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"WorldItem name must be a non-empty string (for category '{category}')."
            )

        normalized_id_category = utils._normalize_for_id(category)
        normalized_id_name = utils._normalize_for_id(name)
        item_id = f"{normalized_id_category}_{normalized_id_name}"

        created_chapter = int(data.get(KG_NODE_CREATED_CHAPTER, 0))
        is_provisional = bool(data.get(KG_IS_PROVISIONAL, False))

        props = {
            k: v
            for k, v in data.items()
            if k
            not in {
                "id",
                "category",
                "name",
                KG_NODE_CREATED_CHAPTER,
                KG_IS_PROVISIONAL,
            }
        }

        return cls(
            id=item_id,
            category=category,
            name=name,
            created_chapter=created_chapter,
            is_provisional=is_provisional,
            properties=props,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the item to a flat dictionary."""

        data = self.model_dump(exclude={"id", "category", "name"})
        properties_data = data.pop("properties", {})
        data.update(properties_data)
        return data

    def get_elaboration_summary(self, up_to_chapter: int | None = None) -> list[str]:
        """Return elaboration notes up to a chapter number."""

        notes: list[str] = []
        prefix = kg_keys.ELABORATION_PREFIX
        for key, val in sorted(self.properties.items()):
            if not key.startswith(prefix):
                continue
            chap = kg_keys.parse_elaboration_key(key)
            if chap is None:
                continue
            if up_to_chapter is None or chap <= up_to_chapter:
                if isinstance(val, str):
                    notes.append(val)
        return notes
