# kg_maintainer/models.py
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict

import utils
from kg_constants import (
    KG_IS_PROVISIONAL,
    KG_NODE_CREATED_CHAPTER,
)


class SceneDetail(TypedDict):
    scene_number: int
    summary: str
    characters_involved: List[str]
    key_dialogue_points: List[str]
    setting_details: str
    scene_focus_elements: List[str]
    contribution: str


class ProblemDetail(TypedDict):
    """Detailed information about a specific problem found during evaluation."""

    issue_category: str
    problem_description: str
    quote_from_original_text: str
    quote_char_start: Optional[int]
    quote_char_end: Optional[int]
    sentence_char_start: Optional[int]
    sentence_char_end: Optional[int]
    suggested_fix_focus: str


class EvaluationResult(TypedDict):
    needs_revision: bool
    reasons: List[str]
    problems_found: List[ProblemDetail]
    coherence_score: Optional[float]
    consistency_issues: Optional[str]
    plot_deviation_reason: Optional[str]
    thematic_issues: Optional[str]
    narrative_depth_issues: Optional[str]


class PatchInstruction(TypedDict):
    """Instruction for applying a single patch to the chapter text."""

    original_problem_quote_text: str
    target_char_start: Optional[int]
    target_char_end: Optional[int]
    replace_with: str
    reason_for_change: str


class AgentStateData(TypedDict):
    plot_outline: dict
    character_profiles: dict
    world_building: dict


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

    def get_development_summary(self, up_to_chapter: Optional[int] = None) -> List[str]:
        """Return development notes up to a given chapter number."""
        notes: List[str] = []
        prefix = "development_in_chapter_"
        for key, val in sorted(self.updates.items()):
            if not key.startswith(prefix):
                continue
            try:
                chap = int(key.split("_")[-1])
            except ValueError:
                continue
            if up_to_chapter is None or chap <= up_to_chapter:
                if isinstance(val, str):
                    notes.append(val)
        return notes


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
    def from_dict(cls, category: str, name: str, data: Dict[str, Any]) -> "WorldItem":
        """
        Creates a WorldItem.
        'category' and 'name' are the intended display/canonical values.
        The 'id' is always generated deterministically from normalized
        versions of these.
        Any 'id' in the 'data' dict is ignored.
        """
        if not category or not isinstance(category, str) or not category.strip():
            raise ValueError("WorldItem category must be a non-empty string.")
        if not name or not isinstance(name, str) or not name.strip():
            raise ValueError(
                "WorldItem name must be a non-empty string "
                f"(for category '{category}')."
            )

        # Generate canonical ID from normalized category and name.
        # The 'name' and 'category' stored on the instance are the ones
        # passed as arguments.
        normalized_id_category = utils._normalize_for_id(category)
        normalized_id_name = utils._normalize_for_id(name)

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

    def get_elaboration_summary(self, up_to_chapter: Optional[int] = None) -> List[str]:
        """Return elaboration notes up to a given chapter number."""
        notes: List[str] = []
        prefix = "elaboration_in_chapter_"
        for key, val in sorted(self.properties.items()):
            if not key.startswith(prefix):
                continue
            try:
                chap = int(key.split("_")[-1])
            except ValueError:
                continue
            if up_to_chapter is None or chap <= up_to_chapter:
                if isinstance(val, str):
                    notes.append(val)
        return notes
