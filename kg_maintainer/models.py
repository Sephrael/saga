"""Compatibility layer exposing data models from :mod:`models`."""

from typing import TypedDict

from models import (
    CharacterProfile,
    EvaluationResult,
    PatchInstruction,
    ProblemDetail,
    SceneDetail,
    WorldItem,
)


class AgentStateData(TypedDict):
    """Legacy type hint for core state objects."""

    plot_outline: dict
    character_profiles: dict
    world_building: dict


__all__ = [
    "AgentStateData",
    "CharacterProfile",
    "EvaluationResult",
    "PatchInstruction",
    "ProblemDetail",
    "SceneDetail",
    "WorldItem",
]
