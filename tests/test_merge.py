from kg_maintainer import (
    CharacterProfile,
    WorldItem,
    merge_character_profile_updates,
    merge_world_item_updates,
)


def test_merge_character_profile_updates():
    current = {"Alice": CharacterProfile("Alice", traits=["brave"])}
    updates = {"Alice": CharacterProfile("Alice", traits=["brave", "smart"])}
    merge_character_profile_updates(current, updates, 1, False)
    assert current["Alice"].traits == ["brave", "smart"]


def test_merge_world_item_updates():
    current = {"Places": {"City": WorldItem("Places_City", "Places", "City", properties={"description": "Old"})}}
    updates = {"Places": {"City": WorldItem("Places_City", "Places", "City", properties={"description": "New"})}}
    merge_world_item_updates(current, updates, 1, False)
    assert current["Places"]["City"].properties["description"] == "New"
