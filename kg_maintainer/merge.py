# kg_maintainer/merge.py
"""Helpers for merging parsed updates into existing knowledge graph state."""

import logging
from typing import Dict

from .models import CharacterProfile, WorldItem

logger = logging.getLogger(__name__)


def initialize_new_character_profile(
    char_name: str, char_update: CharacterProfile, chapter_number: int
) -> CharacterProfile:
    """Create a new character profile from parsed updates."""
    provisional_key = f"source_quality_chapter_{chapter_number}"
    dev_key = f"development_in_chapter_{chapter_number}"
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
    profiles: Dict[str, CharacterProfile],
    updates: Dict[str, CharacterProfile],
    chapter_number: int,
    from_flawed_draft: bool,
) -> None:
    """Merge character updates into existing profile dictionary."""
    provisional_key = f"source_quality_chapter_{chapter_number}"
    for name, update in updates.items():
        data = update.to_dict()
        if from_flawed_draft:
            data[provisional_key] = "provisional_from_unrevised_draft"
        dev_key = f"development_in_chapter_{chapter_number}"
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
                key.startswith("development_in_chapter_")
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
    world: Dict[str, Dict[str, WorldItem]],
    updates: Dict[str, Dict[str, WorldItem]],
    chapter_number: int,
    from_flawed_draft: bool,
) -> None:
    """Merge world item updates into the current world dictionary."""
    provisional_key = f"source_quality_chapter_{chapter_number}"
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
                    f"added_in_chapter_{chapter_number}", True
                )
                continue
            item = world[category][name]
            item_props = item.to_dict()
            for key, val in data.items():
                if key in {provisional_key, "modification_proposal"} or (
                    key.startswith(
                        (
                            "updated_in_chapter_",
                            "added_in_chapter_",
                            "source_quality_chapter_",
                        )
                    )
                ):
                    if (
                        key.startswith("elaboration_in_chapter_")
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
                f"updated_in_chapter_{chapter_number}",
                True,
            )
