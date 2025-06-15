# kg_maintainer/models.py
"""Data models for SAGA's state and knowledge graph objects."""

from typing import Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field

import utils
from kg_constants import (
    KG_IS_PROVISIONAL,
    KG_NODE_CREATED_CHAPTER,
)


class SceneDetail(TypedDict, total=False):
    """
    A detailed plan for a single scene in a chapter.
    This is the contract between the PlannerAgent and DraftingAgent.
    """

    scene_number: int
    summary: str
    characters_involved: List[str]
    key_dialogue_points: List[str]
    setting_details: str
    scene_focus_elements: List[str]
    contribution: str
    # NEW: Directorial fields for controlling narrative texture and variety.
    scene_type: str  # e.g., 'ACTION', 'DIALOGUE', 'INTROSPECTION', 'REVELATION', 'ATMOSPHERE_BUILDING', 'TRANSITION'
    pacing: str  # e.g., 'SLOW', 'MEDIUM', 'FAST', 'URGENT'
    character_arc_focus: Optional[
        str
    ]  # e.g., "Isabelle confronts her past trauma, moving from fear to defiance."
    relationship_development: Optional[
        str
    ]  # e.g., "The trust between Isabelle and Lysander is tested by a new revelation."


class ProblemDetail(TypedDict, total=False):
    """Detailed information about a specific problem found during evaluation."""

    issue_category: str
    problem_description: str
    quote_from_original_text: str
    quote_char_start: Optional[int]
    quote_char_end: Optional[int]
    sentence_char_start: Optional[int]
    sentence_char_end: Optional[int]
    suggested_fix_focus: str


class EvaluationResult(TypedDict, total=False):
    """Structured result from the evaluator agent."""

    needs_revision: bool
    reasons: List[str]
    problems_found: List[ProblemDetail]
    coherence_score: Optional[float]
    consistency_issues: Optional[str]
    plot_deviation_reason: Optional[str]
    thematic_issues: Optional[str]
    narrative_depth_issues: Optional[str]


class PatchInstruction(TypedDict, total=False):
    """Instruction for applying a single patch to the chapter text."""

    original_problem_quote_text: str
    target_char_start: Optional[int]
    target_char_end: Optional[int]
    replace_with: str
    reason_for_change: str


class AgentStateData(TypedDict):
    """Legacy type hint for core state objects."""

    plot_outline: dict
    character_profiles: dict
    world_building: dict


class CharacterProfile(BaseModel):
    """Structured information about a character, using Pydantic for validation."""

    name: str
    description: str = ""
    traits: List[str] = Field(default_factory=list)
    relationships: Dict[str, Any] = Field(default_factory=dict)
    status: str = "Unknown"
    updates: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "CharacterProfile":
        """Creates a CharacterProfile from a dictionary, maintaining compatibility."""
        # Pydantic automatically handles mapping, but we can keep this for explicit control.
        # It also handles separating known fields from 'updates' if we used extra='allow'.
        # For simplicity and compatibility, we'll manually handle the 'updates' field.
        known_fields = cls.model_fields.keys()
        profile_data = {k: v for k, v in data.items() if k in known_fields}
        updates_data = {k: v for k, v in data.items() if k not in known_fields}

        # Combine the explicitly defined 'updates' dict with any extra fields
        if "updates" in profile_data:
            updates_data.update(profile_data["updates"])
        profile_data["updates"] = updates_data

        return cls(name=name, **profile_data)

    def to_dict(self) -> Dict[str, Any]:
        """Converts the CharacterProfile to a dictionary for serialization."""
        data = self.model_dump(exclude={"name"})
        # Flatten the 'updates' dict into the main dict for compatibility with old format
        updates_data = data.pop("updates", {})
        data.update(updates_data)
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
            except (ValueError, IndexError):
                continue
            if up_to_chapter is None or chap <= up_to_chapter:
                if isinstance(val, str):
                    notes.append(val)
        return notes


class WorldItem(BaseModel):
    """Structured information about a world element, using Pydantic for validation."""

    id: str  # Canonical ID derived from normalized category and name
    category: str  # Display/canonical category
    name: str  # Display/canonical name
    created_chapter: int = 0
    is_provisional: bool = False
    properties: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, category: str, name: str, data: Dict[str, Any]) -> "WorldItem":
        """
        Creates a WorldItem from a dictionary, generating a canonical ID.
        This classmethod is the primary entry point for creating WorldItem objects.
        """
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

    def to_dict(self) -> Dict[str, Any]:
        """Converts the WorldItem to a dictionary for serialization."""
        # Pydantic's model_dump is the modern way to do this.
        data = self.model_dump(exclude={"id", "category", "name"})
        # For compatibility with any old code expecting properties at the top level
        properties_data = data.pop("properties", {})
        data.update(properties_data)
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
            except (ValueError, IndexError):
                continue
            if up_to_chapter is None or chap <= up_to_chapter:
                if isinstance(val, str):
                    notes.append(val)
        return notes
