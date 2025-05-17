# type.py
from typing import TypedDict, List, Union, Any, Dict, Optional

class SceneDetail(TypedDict):
    scene_number: int
    summary: str # Was: plot: str
    characters_involved: List[str] # Was: characters: list[str]
    key_dialogue_points: List[str] # New
    setting_details: str # Was: setting: str
    contribution: str # New

class EvaluationResult(TypedDict):
    needs_revision: bool
    reasons: List[str] # Was: feedback: str
    coherence_score: Optional[float] # New
    consistency_issues: Optional[str] # New
    plot_deviation_reason: Optional[str] # New
    thematic_issues: Optional[str] # New, added for comprehensive evaluation

class JsonStateData(TypedDict): # This seems to be a general container, might not be strictly enforced everywhere
    plot_outline: dict
    character_profiles: dict
    world_building: dict

ChapterPlan = List[SceneDetail]

# Define JsonType as a Union of common JSON data types
JsonType = Union[Dict[str, Any], List[Any], str, int, float, bool, None]