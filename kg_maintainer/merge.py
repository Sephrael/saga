# kg_maintainer/merge.py
"""Helpers for merging parsed updates into existing knowledge graph state."""

import kg_constants as kg_keys
import structlog

from .models import CharacterProfile, WorldItem

logger = structlog.get_logger(__name__)


def initialize_new_character_profile(
    char_name: str,
    char_update: CharacterProfile,
    chapter_number: int,
    from_flawed_draft: bool = False,
) -> CharacterProfile:
    """Create a new ``CharacterProfile`` from parsed updates.

    Args:
        char_name: The character's name.
        char_update: Parsed attributes for the character.
        chapter_number: Chapter where the character first appears.
        from_flawed_draft: Whether the character was extracted from an unrevised
            draft.

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
    if from_flawed_draft:
        new_profile.updates[provisional_key] = "provisional_from_unrevised_draft"
    elif provisional_key in data:
        new_profile.updates[provisional_key] = data[provisional_key]
    return new_profile


def _merge_traits(
    profile_traits: list[str], update_traits: list[any]
) -> list[str] | None:
    """Merges trait lists and returns new list if changed, else None."""
    new_traits_set = set(profile_traits)
    added = False
    for t in update_traits:
        if isinstance(t, str) and t.strip() and t not in new_traits_set:
            new_traits_set.add(t)
            added = True
    if added:
        return sorted(list(new_traits_set))
    return None


def _merge_relationships(
    profile_relationships: dict[str, str], update_relationships: dict[any, any]
) -> bool:
    """Merges relationship dicts. Returns True if changed."""
    modified = False
    for target, rel_val in update_relationships.items():
        if not isinstance(target, str) or not isinstance(rel_val, str):
            continue  # Skip malformed entries
        if profile_relationships.get(target) != rel_val:
            profile_relationships[target] = rel_val
            modified = True
    return modified


def _merge_generic_attributes(
    profile: CharacterProfile,
    update_data: dict[str, any],
    existing_profile_dict: dict[str, any],
) -> bool:
    """Merges generic string attributes. Returns True if modified."""
    modified = False
    # Define keys that are handled by specialized mergers or should be skipped
    skipped_keys = {
        "traits",
        "relationships",
        "modification_proposal",
        kg_keys.source_quality_key(0).rsplit("_", 1)[0],  # Match prefix
        kg_keys.DEVELOPMENT_PREFIX.rstrip("_"),  # Match prefix
    }

    for key, val in update_data.items():
        # Skip if key is handled by other functions or is a prefixed development/source key
        if (
            key in skipped_keys
            or key.startswith(kg_keys.DEVELOPMENT_PREFIX)
            or key.startswith(kg_keys.SOURCE_QUALITY_PREFIX)
        ):
            continue

        if isinstance(val, str) and val.strip():
            # Check against the original dict representation for direct attributes
            # For other attributes, they are stored in profile.updates
            if hasattr(profile, key):
                if getattr(profile, key) != val:
                    setattr(profile, key, val)
                    modified = True
            elif (
                existing_profile_dict.get(key) != val
            ):  # Check if it's a dynamic field in updates
                profile.updates[key] = val
                modified = True
            elif profile.updates.get(key) != val:  # Fallback to updates dict
                profile.updates[key] = val
                modified = True
    return modified


def merge_character_profile_updates(
    profiles: dict[str, CharacterProfile],
    updates: dict[str, CharacterProfile],
    chapter_number: int,
    from_flawed_draft: bool,
) -> None:
    """Merge parsed character updates into existing profiles."""
    provisional_key = kg_keys.source_quality_key(chapter_number)
    dev_key = kg_keys.development_key(chapter_number)

    for name, update_obj in updates.items():
        update_data = update_obj.to_dict()
        if from_flawed_draft:
            # Ensure this key is present for flawed drafts if not already set by parser
            update_data.setdefault(provisional_key, "provisional_from_unrevised_draft")

        if name not in profiles:
            profiles[name] = initialize_new_character_profile(
                name,
                update_obj,  # Pass the object itself
                chapter_number,
                from_flawed_draft=from_flawed_draft,
            )
            continue

        profile = profiles[name]
        overall_modified = False

        # Merge traits
        if "traits" in update_data and isinstance(update_data["traits"], list):
            merged_traits = _merge_traits(profile.traits, update_data["traits"])
            if merged_traits is not None:
                profile.traits = merged_traits
                overall_modified = True

        # Merge relationships
        if "relationships" in update_data and isinstance(
            update_data["relationships"], dict
        ):
            if _merge_relationships(
                profile.relationships, update_data["relationships"]
            ):
                overall_modified = True

        # Merge generic attributes (description, status, etc.)
        # This needs the original profile dict for comparison if attributes are not direct properties
        # However, CharacterProfile attributes are direct, so we pass profile.to_dict()
        # for consistent checking of existing values if needed by the helper,
        # though _merge_generic_attributes primarily uses hasattr.
        if _merge_generic_attributes(profile, update_data, profile.to_dict()):
            overall_modified = True

        # Handle development log explicitly
        if dev_key in update_data and isinstance(update_data[dev_key], str):
            if profile.updates.get(dev_key) != update_data[dev_key]:
                profile.updates[dev_key] = update_data[dev_key]
                overall_modified = True

        # Handle provisional key explicitly if it was part of the update_data
        if provisional_key in update_data and isinstance(
            update_data[provisional_key], str
        ):
            if profile.updates.get(provisional_key) != update_data[provisional_key]:
                profile.updates[provisional_key] = update_data[provisional_key]
                overall_modified = True
        elif (
            from_flawed_draft
        ):  # Ensure it's set if from flawed draft, even if not in original update_data
            if (
                profile.updates.get(provisional_key)
                != "provisional_from_unrevised_draft"
            ):
                profile.updates[provisional_key] = "provisional_from_unrevised_draft"
                overall_modified = True

        if overall_modified:
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
                world_item = update
                if from_flawed_draft:
                    world_item.properties[provisional_key] = (
                        "provisional_from_unrevised_draft"
                    )
                world[category][name] = world_item
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
