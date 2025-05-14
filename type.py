# type.py
from typing import TypedDict, List, Union, Any, Dict

class SceneDetail(TypedDict):
    scene_number: int
    characters: list[str]
    setting: str
    plot: str

class EvaluationResult(TypedDict):
    needs_revision: bool
    feedback: str

class JsonStateData(TypedDict):
    plot_outline: dict
    character_profiles: dict
    world_building: dict

ChapterPlan = List[SceneDetail]

# Define JsonType as a Union of common JSON data types
JsonType = Union[Dict[str, Any], List[Any], str, int, float, bool, None]
