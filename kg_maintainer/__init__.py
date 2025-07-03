# kg_maintainer/__init__.py
"""Package consolidating KG maintainer utilities."""

import importlib

from parsing import (
    __name__ as _parsing_name,
)
from parsing import (
    parse_unified_character_updates,
    parse_unified_world_updates,
)

from . import merge as merge
from . import models as models

parsing = importlib.import_module(_parsing_name)

from .merge import (
    merge_character_profile_updates,
    merge_world_item_updates,
)
from .models import (
    AgentStateData,
    CharacterProfile,
    EvaluationResult,
    PatchInstruction,
    ProblemDetail,
    SceneDetail,
    WorldItem,
)

__all__ = [
    "AgentStateData",
    "CharacterProfile",
    "EvaluationResult",
    "PatchInstruction",
    "ProblemDetail",
    "SceneDetail",
    "WorldItem",
    "parse_unified_character_updates",
    "parse_unified_world_updates",
    "merge_character_profile_updates",
    "merge_world_item_updates",
    "merge",
    "models",
    "parsing",
]
