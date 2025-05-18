# type.py
from typing import TypedDict, List, Union, Any, Dict, Optional

class SceneDetail(TypedDict):
    scene_number: int
    summary: str # Was: plot: str
    characters_involved: List[str] # Was: characters: list[str]
    key_dialogue_points: List[str] # New
    setting_details: str # Was: setting: str
    scene_focus_elements: List[str] # NEW: For targeted elaboration to increase scene length/depth
    contribution: str # New

class EvaluationResult(TypedDict):
    needs_revision: bool
    reasons: List[str] # Was: feedback: str
    coherence_score: Optional[float] # New
    consistency_issues: Optional[str] # New
    plot_deviation_reason: Optional[str] # New
    thematic_issues: Optional[str] # New, added for comprehensive evaluation
    narrative_depth_issues: Optional[str] # NEW: Specific feedback on narrative depth and length

class JsonStateData(TypedDict): # This seems to be a general container, might not be strictly enforced everywhere
    plot_outline: dict
    character_profiles: dict
    world_building: dict

ChapterPlan = List[SceneDetail]

# Define JsonType as a Union of common JSON data types
JsonType = Union[Dict[str, Any], List[Any], str, int, float, bool, None]