# kg_maintainer/merge.py
"""Helpers for merging parsed updates into existing knowledge graph state."""

import structlog
from utils import kg_property_keys as kg_keys

from .models import CharacterProfile, WorldItem

logger = structlog.get_logger(__name__)


def initialize_new_character_profile(
    char_name: str, char_update: CharacterProfile, chapter_number: int
) -> CharacterProfile:
    """Create a new ``CharacterProfile`` from parsed updates.

    Args:
        char_name: The character's name.
        char_update: Parsed attributes for the character.
        chapter_number: Chapter where the character first appears.

    Returns:
        The initialized character profile.
    """
    provisional_key = kg_keys.source_quality_key(chapter_number)
    dev_key = kg_keys.development_key(chapter_number)
    data = char_update.to_dict()
    new_profile = CharacterProfile(
        name=char_name,
        description=data.get(
            "description",
            f"A character newly introduced in Chapter {chapter_number}.",
        ),
        traits=sorted(
            {t for t in data.get("traits", []) if isinstance(t, str) and t.strip()}
        ),
        relationships=data.get("relationships", {}),
        status=data.get("status", "Newly introduced"),
        updates={
            dev_key: data.get(
                dev_key,
                (f"Character '{char_name}' introduced in Chapter {chapter_number}."),
            )
        },
    )
    if provisional_key in data:
        new_profile.updates[provisional_key] = data[provisional_key]
    return new_profile


def merge_character_profile_updates(
    profiles: dict[str, CharacterProfile],
    updates: dict[str, CharacterProfile],
    chapter_number: int,
    from_flawed_draft: bool,
) -> None:
    """Merge parsed character updates into existing profiles.

    Args:
        profiles: Current character profiles keyed by name.
        updates: Newly parsed updates for the chapter.
        chapter_number: The chapter number being processed.
        from_flawed_draft: Whether updates came from an unrevised draft.

    Returns:
        ``None``. Profiles are modified in place.
    """
    provisional_key = kg_keys.source_quality_key(chapter_number)
    for name, update in updates.items():
        data = update.to_dict()
        if from_flawed_draft:
            data[provisional_key] = "provisional_from_unrevised_draft"
        dev_key = kg_keys.development_key(chapter_number)
        if name not in profiles:
            profiles[name] = initialize_new_character_profile(
                name, update, chapter_number
            )
            continue
        profile = profiles[name]
        prof_dict = profile.to_dict()
        modified = False
        for key, val in data.items():
            if key in {"modification_proposal", provisional_key} or (
                key.startswith(kg_keys.DEVELOPMENT_PREFIX)
            ):
                continue
            if key == "traits" and isinstance(val, list):
                new_traits = sorted(
                    set(profile.traits).union(
                        {t for t in val if isinstance(t, str) and t.strip()}
                    )
                )
                if new_traits != profile.traits:
                    profile.traits = new_traits
                    modified = True
            elif key == "relationships" and isinstance(val, dict):
                for target, rel in val.items():
                    if profile.relationships.get(target) != rel:
                        profile.relationships[target] = rel
                        modified = True
            elif isinstance(val, str) and val.strip() and prof_dict.get(key) != val:
                profile.updates[key] = val
                modified = True
        if dev_key in data and isinstance(data[dev_key], str):
            profile.updates[dev_key] = data[dev_key]
            modified = True
        if from_flawed_draft:
            profile.updates[provisional_key] = "provisional_from_unrevised_draft"
        if modified:
            logger.debug("Profile for %s modified", name)


def merge_world_item_updates(
    world: dict[str, dict[str, WorldItem]],
    updates: dict[str, dict[str, WorldItem]],
    chapter_number: int,
    from_flawed_draft: bool,
) -> None:
    """Merge parsed world item updates into the in-memory world state.

    Args:
        world: Existing world items keyed by category then name.
        updates: Updates parsed from the latest chapter.
        chapter_number: The chapter number being processed.
        from_flawed_draft: Whether updates are from an unrevised draft.

    Returns:
        ``None``. The ``world`` dictionary is modified in place.
    """
    provisional_key = kg_keys.source_quality_key(chapter_number)
    for category, cat_updates in updates.items():
        if category not in world:
            world[category] = {}
        for name, update in cat_updates.items():
            data = update.to_dict()
            if from_flawed_draft:
                data[provisional_key] = "provisional_from_unrevised_draft"
            if name not in world[category]:
                world[category][name] = update
                world[category][name].properties.setdefault(
                    kg_keys.added_key(chapter_number), True
                )
                continue
            item = world[category][name]
            item_props = item.to_dict()
            for key, val in data.items():
                if key in {provisional_key, "modification_proposal"} or (
                    key.startswith(
                        (
                            kg_keys.UPDATED_PREFIX,
                            kg_keys.ADDED_PREFIX,
                            kg_keys.SOURCE_QUALITY_PREFIX,
                        )
                    )
                ):
                    if (
                        key.startswith(kg_keys.ELABORATION_PREFIX)
                        and isinstance(val, str)
                        and val.strip()
                    ):
                        item.properties[key] = val
                    continue
                cur_val = item_props.get(key)
                if isinstance(val, list):
                    cur_list = item.properties.get(key, [])
                    for elem in val:
                        if elem not in cur_list:
                            cur_list.append(elem)
                    item.properties[key] = cur_list
                elif isinstance(val, dict):
                    sub = item.properties.get(key, {})
                    if not isinstance(sub, dict):
                        sub = {}
                    sub.update(val)
                    item.properties[key] = sub
                elif cur_val != val:
                    item.properties[key] = val
            item.properties.setdefault(
                kg_keys.updated_key(chapter_number),
                True,
            )
