# chapter_generation/context_kg_utils.py
"""Utility functions for context-related KG operations."""

from typing import Any


async def get_reliable_kg_facts_for_drafting_prompt(
    plot_outline: dict[str, Any],
    chapter_number: int,
    chapter_plan: list[dict[str, Any]] | None = None,
) -> str:
    """Return KG facts relevant to the chapter focus."""
    return ""


async def get_kg_reasoning_guidance_for_prompt(
    plot_outline: dict[str, Any],
    chapter_number: int,
    chapter_plan: list[dict[str, Any]] | None = None,
    max_guidelines: int = 5,
) -> str:
    """Return KG reasoning hints for the chapter."""
    return ""
