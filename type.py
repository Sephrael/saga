# type.py
from typing import TypedDict, List, Union, Any, Dict, Optional

class SceneDetail(TypedDict):
    scene_number: int
    summary: str 
    characters_involved: List[str] 
    key_dialogue_points: List[str] 
    setting_details: str 
    scene_focus_elements: List[str] 
    contribution: str 

class ProblemDetail(TypedDict):
    """Detailed information about a specific problem found during evaluation."""
    issue_category: str # e.g., "consistency", "plot_arc", "thematic", "narrative_depth"
    problem_description: str
    quote_from_original: str # Verbatim quote from the original text illustrating the problem
    suggested_fix_focus: str # Guidance for the LLM on how to fix it

class EvaluationResult(TypedDict):
    needs_revision: bool
    reasons: List[str] # High-level summary of reasons for revision
    problems_found: List[ProblemDetail] 
    coherence_score: Optional[float] 
    consistency_issues: Optional[str] 
    plot_deviation_reason: Optional[str] 
    thematic_issues: Optional[str] 
    narrative_depth_issues: Optional[str] 

class PatchInstruction(TypedDict):
    """Instruction for applying a single patch to the chapter text."""
    search_text: str 
    replace_with: str 
    original_quote_ref: str 
    reason_for_change: str 

# Renamed JsonStateData to reflect that it's the agent's internal Python dict state,
# not necessarily derived from JSON LLM output anymore.
class AgentStateData(TypedDict): 
    plot_outline: dict
    character_profiles: dict
    world_building: dict

ChapterPlan = List[SceneDetail]

# JsonType is less relevant if we are not parsing generic JSON from LLMs.
# It might still be used for API payloads or user-supplied JSON files.
JsonType = Union[Dict[str, Any], List[Any], str, int, float, bool, None]