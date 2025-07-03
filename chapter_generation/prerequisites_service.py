# chapter_generation/prerequisites_service.py
"""Gather required planning and context before drafting."""

from __future__ import annotations

from dataclasses import dataclass

from models import SceneDetail


@dataclass
class PrerequisiteData:
    """Data required before drafting a chapter."""

    plot_point_focus: str | None
    plot_point_index: int
    chapter_plan: list[SceneDetail] | None
    hybrid_context_for_draft: str | None
    fill_in_context: str | None = None
