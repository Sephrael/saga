from kg_maintainer import (
    parse_unified_character_updates,
    parse_unified_world_updates,
    CharacterProfile,
)


def test_parse_character_updates_simple():
    text = """Character: Alice\nDescription: Brave hero\nTraits:\n- smart"""
    result = parse_unified_character_updates(text, 1)
    assert "Alice" in result
    prof = result["Alice"]
    assert isinstance(prof, CharacterProfile)


def test_parse_world_updates_simple():
    text = "Category: Places\nCity:\n    Description: Large city"
    result = parse_unified_world_updates(text, 1)
    assert isinstance(result, dict)

