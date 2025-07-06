# models/agent_models.py
"""TypedDict structures used for inter-agent communication."""

from __future__ import annotations

from typing import Any, TypedDict

from pydantic import BaseModel, ConfigDict, Field


class SceneDetail(TypedDict, total=False):
    """A detailed plan for a single scene."""

    scene_number: int
    summary: str
    characters_involved: list[str]
    key_dialogue_points: list[str]
    setting_details: str
    scene_focus_elements: list[str]
    contribution: str
    scene_type: str
    pacing: str
    character_arc_focus: str | None
    relationship_development: str | None


class AgentBaseModel(BaseModel):
    """Base model supporting mapping style access."""

    model_config = ConfigDict(from_attributes=True, extra="allow")

    def __getitem__(self, item: str) -> Any:  # pragma: no cover - convenience
        return getattr(self, item)

    def __setitem__(
        self, key: str, value: Any
    ) -> None:  # pragma: no cover - convenience
        setattr(self, key, value)

    def get(
        self, item: str, default: Any = None
    ) -> Any:  # pragma: no cover - convenience
        return getattr(self, item, default)


class ProblemDetail(AgentBaseModel):
    """Information about a problem found during evaluation."""

    issue_category: str
    problem_description: str
    quote_from_original_text: str
    quote_char_start: int | None = None
    quote_char_end: int | None = None
    sentence_char_start: int | None = None
    sentence_char_end: int | None = None
    suggested_fix_focus: str
    rewrite_instruction: str | None = None
    severity: str | None = None
    related_spans: list[tuple[int, int]] | None = None


class EvaluationResult(AgentBaseModel):
    """Structured result from the evaluator agent."""

    needs_revision: bool
    reasons: list[str]
    problems_found: list[ProblemDetail]
    coherence_score: float | None = None
    consistency_issues: str | None = None
    plot_deviation_reason: str | None = None
    thematic_issues: str | None = None
    narrative_depth_issues: str | None = None


class PatchInstruction(AgentBaseModel):
    """Instruction for applying a single patch to chapter text."""

    original_problem_quote_text: str
    target_char_start: int | None = None
    target_char_end: int | None = None
    replace_with: str
    reason_for_change: str


class CharacterState(AgentBaseModel):
    """State information for a character at chapter end."""

    name: str
    status: str
    location: str
    immediate_goal: str | None = None


class ChapterEndState(AgentBaseModel):
    """Snapshot of the world at the end of a chapter."""

    chapter_number: int
    character_states: list[CharacterState]
    unresolved_cliffhanger: str | None = None
    key_world_changes: dict[str, str] = Field(default_factory=dict)
