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
    problems_found: List[ProblemDetail] # NEW: Detailed list of specific problems with quotes
    coherence_score: Optional[float] 
    consistency_issues: Optional[str] # Summary string, still useful for overall logging
    plot_deviation_reason: Optional[str] # Summary string
    thematic_issues: Optional[str] # Summary string
    narrative_depth_issues: Optional[str] # Summary string

class PatchInstruction(TypedDict):
    """Instruction for applying a single patch to the chapter text."""
    search_text: str # The exact text to find (from ProblemDetail.quote_from_original)
    replace_with: str # The new text to substitute
    original_quote_ref: str # Reference to the original quote for traceability
    reason_for_change: str # Brief explanation linked to the ProblemDetail

class JsonStateData(TypedDict): 
    plot_outline: dict
    character_profiles: dict
    world_building: dict

ChapterPlan = List[SceneDetail]

JsonType = Union[Dict[str, Any], List[Any], str, int, float, bool, None]