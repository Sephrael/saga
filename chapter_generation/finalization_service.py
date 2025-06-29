"""Service for finalizing chapters and persisting results."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FinalizationServiceResult:
    """Outcome of finalization."""

    text: str | None
