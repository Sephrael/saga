# models/user_input_models.py
"""User-facing models for providing story input data."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .kg_models import CharacterProfile, WorldItem


class NovelConceptModel(BaseModel):
    """High level concept for the novel."""

    title: str = Field(..., min_length=1)
    genre: str | None = None
    setting: str | None = None
    logline: str | None = None
    theme: str | None = None


class RelationshipModel(BaseModel):
    """Relationship details for a character."""

    name: str | None = None
    status: str | None = None
    details: str | None = None


class ProtagonistModel(BaseModel):
    """Primary character information."""

    name: str
    description: str | None = None
    traits: list[str] = []
    motivation: str | None = None
    role: str | None = None
    relationships: dict[str, RelationshipModel] = {}


class CharacterGroupModel(BaseModel):
    """Container for characters provided in the YAML."""

    protagonist: ProtagonistModel | None = None
    antagonist: ProtagonistModel | None = None
    supporting_characters: list[ProtagonistModel] = []


class KeyLocationModel(BaseModel):
    """A single location within the setting."""

    name: str
    description: str | None = None
    atmosphere: str | None = None


class SettingModel(BaseModel):
    """Setting information for the story world."""

    primary_setting_overview: str | None = None
    key_locations: list[KeyLocationModel] = []


class PlotElementsModel(BaseModel):
    """Major plot elements provided by the user."""

    inciting_incident: str | None = None
    plot_points: list[str] = []
    central_conflict: str | None = None
    stakes: str | None = None


class UserStoryInputModel(BaseModel):
    """Top level structure for ``user_story_elements.yaml``."""

    model_config = ConfigDict(extra="allow")

    novel_concept: NovelConceptModel | None = None
    protagonist: ProtagonistModel | None = None
    antagonist: ProtagonistModel | None = None
    characters: CharacterGroupModel | None = None
    plot_elements: PlotElementsModel | None = None
    setting: SettingModel | None = None
    world_details: dict[str, Any] | None = None
    other_key_characters: dict[str, ProtagonistModel] | None = None
    conflict: dict[str, Any] | None = None
    style_and_tone: dict[str, Any] | None = None
    world_specifics: dict[str, Any] | None = None
    symbolism: list[dict[str, str]] | None = None


def _add_character_profile(
    characters: dict[str, CharacterProfile], info: ProtagonistModel, role: str
) -> None:
    """Create and store a ``CharacterProfile`` from the supplied model."""
    cp = CharacterProfile(name=info.name)
    cp.description = info.description or ""
    cp.traits = info.traits
    cp.relationships = {
        rel_key: rel.model_dump(exclude_none=True)
        for rel_key, rel in info.relationships.items()
    }
    cp.status = "As described"
    cp.updates["role"] = role
    if hasattr(info, "motivation"):
        cp.updates["motivation"] = info.motivation or ""
    characters[cp.name] = cp


def _add_world_items_from_setting(
    setting: SettingModel, world_items: dict[str, dict[str, WorldItem]]
) -> None:
    """Populate ``world_items`` from the ``SettingModel``."""
    world_items.setdefault("_overview_", {})["_overview_"] = WorldItem.from_dict(
        "_overview_",
        "_overview_",
        {"description": setting.primary_setting_overview or ""},
    )
    for loc in setting.key_locations:
        world_items.setdefault("locations", {})[loc.name] = WorldItem.from_dict(
            "locations",
            loc.name,
            {"description": loc.description or "", "atmosphere": loc.atmosphere or ""},
        )


def _add_world_details(
    details: dict[str, Any], world_items: dict[str, dict[str, WorldItem]]
) -> None:
    """Add world items from ``details`` dict."""
    for category, items in details.items():
        world_items.setdefault(category, {})
        if isinstance(items, dict):
            for item_name, item_details in items.items():
                world_items[category][item_name] = WorldItem.from_dict(
                    category, item_name, item_details
                )


def _extract_main_character(model: UserStoryInputModel) -> ProtagonistModel | None:
    """Return the primary protagonist from ``model``."""
    return model.protagonist or (
        model.characters.protagonist if model.characters else None
    )


def _handle_antagonist(
    model: UserStoryInputModel, characters: dict[str, CharacterProfile]
) -> None:
    """Add antagonist character profile if present."""
    antagonist = model.antagonist or (
        model.characters.antagonist if model.characters else None
    )
    if antagonist:
        _add_character_profile(characters, antagonist, antagonist.role or "antagonist")


def _add_supporting_characters(
    model: UserStoryInputModel, characters: dict[str, CharacterProfile]
) -> None:
    """Add supporting and other key characters from ``model``."""
    if model.other_key_characters:
        for _name, info in model.other_key_characters.items():
            _add_character_profile(characters, info, "other_key_character")

    if model.characters and model.characters.supporting_characters:
        for info in model.characters.supporting_characters:
            _add_character_profile(
                characters, info, info.role or "supporting_character"
            )


def _apply_plot_elements(
    plot_outline: dict[str, Any], elements: PlotElementsModel
) -> None:
    """Merge plot element data into ``plot_outline``."""
    plot_outline["inciting_incident"] = elements.inciting_incident
    plot_outline["plot_points"] = elements.plot_points
    plot_outline["central_conflict"] = elements.central_conflict
    plot_outline["stakes"] = elements.stakes


def _merge_style_and_tone(
    plot_outline: dict[str, Any], style_data: dict[str, Any]
) -> None:
    """Merge style and tone information into ``plot_outline``."""
    for key in ("narrative_style", "tone", "pacing"):
        if key in style_data:
            plot_outline[key] = style_data[key]


def user_story_to_objects(
    model: UserStoryInputModel,
) -> tuple[
    dict[str, Any], dict[str, CharacterProfile], dict[str, dict[str, WorldItem]]
]:
    """Convert ``UserStoryInputModel`` to internal dataclass objects."""

    plot_outline: dict[str, Any] = {}
    characters: dict[str, CharacterProfile] = {}
    world_items: dict[str, dict[str, WorldItem]] = {}

    if model.novel_concept:
        plot_outline.update(model.novel_concept.model_dump(exclude_none=True))
        if model.novel_concept.setting is not None:
            plot_outline["setting"] = model.novel_concept.setting

    main_char_model = _extract_main_character(model)
    if main_char_model:
        plot_outline["protagonist_name"] = main_char_model.name
        _add_character_profile(
            characters, main_char_model, main_char_model.role or "protagonist"
        )

    _handle_antagonist(model, characters)
    _add_supporting_characters(model, characters)

    if model.plot_elements:
        _apply_plot_elements(plot_outline, model.plot_elements)

    if model.setting:
        _add_world_items_from_setting(model.setting, world_items)

    if model.style_and_tone:
        _merge_style_and_tone(plot_outline, model.style_and_tone)

    if model.world_details:
        _add_world_details(model.world_details, world_items)

    return plot_outline, characters, world_items if world_items else []
