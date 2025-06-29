# chapter_generation/revision_service.py
"""Service for revising drafted chapters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RevisionResult:
    """Result of the revision loop."""

    text: str | None
    raw_llm_output: str | None
    is_flawed_source: bool
    patched_spans: list[tuple[int, int]]
