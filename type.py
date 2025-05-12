# type.py
from typing import TypedDict, List

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
