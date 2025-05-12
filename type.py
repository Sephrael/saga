# type.py
"""
Common type hints used across the Saga Novel Generation system.
"""
from typing import Dict, List, Optional, Tuple, Any, TypedDict

# For JSON state files like plot_outline, character_profiles, world_building
JsonStateData = Dict[str, Any]

# For chapter evaluation results
class EvaluationResult(TypedDict):
    needs_revision: bool
    reasons: List[str]
    coherence_score: Optional[float]
    consistency_issues: Optional[str]
    plot_deviation_reason: Optional[str]

# For detailed scene plans
class SceneDetail(TypedDict):
    scene_number: int
    summary: str
    characters_involved: List[str]
    key_dialogue_points: List[str]
    setting_details: str
    contribution: str # How this scene contributes to the chapter/plot