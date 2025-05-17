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

# Added for knowledge_management_logic.py, adjust as needed
class KnowledgeGraph(TypedDict):
    entities: Dict[str, 'Entity']
    relationships: List['Relationship']
    events: List['Event']

class Entity(TypedDict, total=False):
    id: str
    type: str
    name: str
    description: Optional[str]
    attributes: Dict[str, Any]
    # Specific entity types might extend this
    status: Optional[str] # For characters
    traits: Optional[List[str]] # For characters
    goals: Optional[List[str]] # For factions/characters

class Relationship(TypedDict):
    source_id: str
    target_id: str
    type: str
    description: Optional[str]

class Event(TypedDict):
    id: str
    type: str
    description: str
    involved_entities: List[str]
    location: Optional[str]
    timestamp: Optional[str] # Or a more specific datetime type

class Location(Entity): # Example specific entity
    atmosphere: Optional[str]

class Character(Entity): # Example specific entity
    # status, traits already in Entity with total=False
    relationships: Optional[Dict[str, str]] # e.g., {"CharacterName": "Ally"}

class Faction(Entity): # Example specific entity
    # goals already in Entity
    members: Optional[List[str]]