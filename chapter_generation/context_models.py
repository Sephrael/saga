# chapter_generation/context_models.py
"""Data models used for context generation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from models.agent_models import SceneDetail


class ContextProfileName(str, Enum):
    """Supported context profiles."""

    DEFAULT = "default"
    ALTERNATE = "alternate"


@dataclass
class ContextRequest:
    """Parameters describing the desired context."""

    chapter_number: int
    plot_focus: str | None
    plot_outline: dict[str, Any]
    chapter_plan: list[SceneDetail] | None = None
    agent_hints: dict[str, Any] | None = None
    profile_name: ContextProfileName = ContextProfileName.DEFAULT


@dataclass
class ProviderSettings:
    """Configuration for an individual provider instance."""

    provider: Any
    max_tokens: int | None = None
    detail_level: str | None = None


@dataclass
class ProfileConfiguration:
    """List of providers and token limit for a profile."""

    providers: list[ProviderSettings]
    max_tokens: int


@dataclass
class ContextChunk:
    """A chunk of context returned by a provider."""

    text: str
    tokens: int
    provenance: dict[str, Any]
    source: str
    from_llm_fill: bool = False
    """Whether this chunk's content was filled in by an LLM."""
