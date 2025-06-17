from kg_maintainer import (
    CharacterProfile,
    WorldItem,
    merge_character_profile_updates,
    merge_world_item_updates,
)
from kg_maintainer.merge import initialize_new_character_profile


def test_merge_character_profile_updates():
    current = {"Alice": CharacterProfile(name="Alice", traits=["brave"])}
    updates = {"Alice": CharacterProfile(name="Alice", traits=["brave", "smart"])}
    merge_character_profile_updates(current, updates, 1, False)
    assert current["Alice"].traits == ["brave", "smart"]


def test_merge_world_item_updates():
    current = {
        "Places": {
            "City": WorldItem.from_dict(
                "Places",
                "City",
                {"description": "Old"},
            )
        }
    }
    updates = {
        "Places": {
            "City": WorldItem.from_dict(
                "Places",
                "City",
                {"description": "New"},
            )
        }
    }
    merge_world_item_updates(current, updates, 1, False)
    assert current["Places"]["City"].properties["description"] == "New"


def test_initialize_new_character_profile_defaults():
    update = CharacterProfile(name="Eve")
    profile = initialize_new_character_profile("Eve", update, 2)
    assert profile.description == ""
    assert profile.updates["development_in_chapter_2"].startswith("Character")


def test_merge_character_profile_updates_new_from_flawed():
    profiles = {}
    updates = {"Zoe": CharacterProfile(name="Zoe", traits=["kind"])}
    merge_character_profile_updates(profiles, updates, 1, True)
    prof = profiles["Zoe"]
    assert prof.traits == ["kind"]
    assert "source_quality_chapter_1" not in prof.updates


def test_merge_world_item_updates_new_item_flawed():
    current = {}
    updates = {"Things": {"Book": WorldItem.from_dict("Things", "Book", {"info": "x"})}}
    merge_world_item_updates(current, updates, 3, True)
    item = current["Things"]["Book"]
    assert item.properties["added_in_chapter_3"]
    assert "source_quality_chapter_3" not in item.properties


def test_merge_world_item_updates_merge_complex():
    current = {
        "Places": {
            "Town": WorldItem.from_dict(
                "Places", "Town", {"features": ["old"], "data": {"a": 1}}
            )
        }
    }
    updates = {
        "Places": {
            "Town": WorldItem.from_dict(
                "Places",
                "Town",
                {"features": ["new", "old"], "data": {"b": 2}, "desc": "Town desc"},
            )
        }
    }
    merge_world_item_updates(current, updates, 2, False)
    item = current["Places"]["Town"]
    assert "Town desc" == item.properties["desc"]
    assert set(item.properties["features"]) == {"old", "new"}
    assert item.properties["data"] == {"a": 1, "b": 2}
    assert item.properties["updated_in_chapter_2"]
