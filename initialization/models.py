# initialization/models.py
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from models import CharacterProfile as BaseCharacterProfile
from models import WorldItem
from models.agent_models import AgentBaseModel


class PlotOutline(AgentBaseModel):
    """Structured representation of the plot outline."""

    title: str | None = None
    protagonist_name: str | None = None
    genre: str | None = None
    setting: str | None = None
    theme: str | None = None
    logline: str | None = None
    inciting_incident: str | None = None
    central_conflict: str | None = None
    stakes: str | None = None
    plot_points: list[str] = Field(default_factory=list)
    narrative_style: str | None = None
    tone: str | None = None
    pacing: str | None = None
    is_default: bool | None = None
    source: str | None = None

    def setdefault(
        self, key: str, default: Any
    ) -> Any:  # pragma: no cover - convenience
        val = self.get(key)
        if val is None:
            self[key] = default
            return default
        return val


class CharacterProfile(BaseCharacterProfile):
    """Alias for character profiles used during initialization."""


class WorldBuilding(BaseModel):
    """Container for world building items keyed by category."""

    model_config = ConfigDict(extra="allow")

    data: dict[str, dict[str, WorldItem]] = Field(default_factory=dict)
    is_default: bool | None = None
    source: str | None = None

    def __getitem__(
        self, key: str
    ) -> dict[str, WorldItem]:  # pragma: no cover - convenience
        return self.data[key]

    def __setitem__(
        self, key: str, value: Any
    ) -> None:  # pragma: no cover - convenience
        if key == "is_default":
            self.is_default = bool(value)
        elif key == "source":
            self.source = str(value)
        else:
            self.data[key] = value

    def get(
        self, key: str, default: Any = None
    ) -> Any:  # pragma: no cover - convenience
        if key in {"is_default", "source"}:
            return getattr(self, key, default)
        return self.data.get(key, default)

    def items(
        self,
    ) -> Iterable[tuple[str, dict[str, WorldItem]]]:  # pragma: no cover - convenience
        return self.data.items()

    def __contains__(self, key: str) -> bool:  # pragma: no cover - convenience
        return key in self.data

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"is_default": self.is_default, "source": self.source}
        result.update(self.data)
        return result


__all__ = ["PlotOutline", "CharacterProfile", "WorldBuilding", "WorldItem"]
