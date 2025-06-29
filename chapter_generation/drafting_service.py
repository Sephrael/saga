"""Service for drafting initial chapter text."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass
class DraftResult:
    """Initial draft and raw output from the DraftingAgent."""

    text: str | None
    raw_llm_output: str | None

    def __iter__(self) -> Iterable[str | None]:
        """Allow tuple-style unpacking of the draft result."""
        yield self.text
        yield self.raw_llm_output
