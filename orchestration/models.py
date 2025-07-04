# orchestration/models.py
"""Shared dataclasses for orchestration services."""

from dataclasses import dataclass, field

from kg_maintainer.models import CharacterProfile, WorldItem


@dataclass
class RevisionOutcome:
    """Final text after processing and whether it is marked flawed."""

    text: str | None
    raw_llm_output: str | None
    is_flawed: bool


@dataclass
class KnowledgeCache:
    """In-memory cache for KG data used during chapter generation."""

    characters: dict[str, CharacterProfile] = field(default_factory=dict)
    world: dict[str, dict[str, WorldItem]] = field(default_factory=dict)
