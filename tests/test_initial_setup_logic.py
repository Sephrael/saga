import pytest
from unittest.mock import patch, AsyncMock
import json

# Assuming initial_setup_logic.py is in the parent directory or PYTHONPATH is set up
# For this environment, it's in the root.
from initial_setup_logic import (
    generate_world_building_logic,
    WORLD_CATEGORY_MAP_NORMALIZED_TO_INTERNAL,
    WORLD_DETAIL_KEY_MAP_FROM_MARKDOWN_TO_INTERNAL,
    WORLD_DETAIL_LIST_INTERNAL_KEYS
)
import config # For config.MARKDOWN_FILL_IN_PLACEHOLDER and other config values if needed

# Minimal Mock Agent class
class MockAgent:
    def __init__(self):
        self.plot_outline = {
            "title": "Test Novel",
            "genre": "Fantasy",
            "setting_description": "A land of high mountains and deep valleys."
        }
        # World building can start empty or with some pre-existing user data
        self.world_building = {}

@pytest.fixture
def agent_instance():
    """Provides a fresh MockAgent instance for each test."""
    return MockAgent()

@pytest.mark.asyncio
async def test_valid_json_output_from_llm(agent_instance):
    """Test successful parsing of valid JSON output from LLM."""
    valid_json_payload = {
        "overview": {
            "description": "A vast desert planet with twin suns.",
            "mood": "Harsh and unforgiving"
        },
        "locations": {
            "Oasis City": {
                "description": "A bustling hub built around the last known water source.",
                "atmosphere": "Lively but tense",
                "features": ["Large marketplace", "Protected wells"]
            }
        },
        "factions": {
            "Desert Nomads": {
                "description": "Tribes that roam the great sands.",
                "goals": ["Survive the harsh conditions", "Find new water sources"],
                "structure": "Tribal councils"
            }
        }
    }
    llm_response_str = json.dumps(valid_json_payload)
    mock_llm_output = (llm_response_str, {"prompt_tokens": 10, "completion_tokens": 100, "total_tokens": 110})

    with patch('initial_setup_logic.llm_service.async_call_llm', AsyncMock(return_value=mock_llm_output)) as mocked_llm_call:
        await generate_world_building_logic(agent_instance)

    wb = agent_instance.world_building
    assert wb['_overview_']['description'] == "A vast desert planet with twin suns."
    assert wb['_overview_']['mood'] == "Harsh and unforgiving"

    assert "Oasis City" in wb['locations']
    assert wb['locations']['Oasis City']['description'] == "A bustling hub built around the last known water source."
    assert wb['locations']['Oasis City']['atmosphere'] == "Lively but tense"
    assert wb['locations']['Oasis City']['features'] == ["Large marketplace", "Protected wells"]

    assert "Desert Nomads" in wb['factions']
    assert wb['factions']['Desert Nomads']['description'] == "Tribes that roam the great sands."
    assert wb['factions']['Desert Nomads']['goals'] == ["Survive the harsh conditions", "Find new water sources"]
    assert wb['factions']['Desert Nomads']['structure'] == "Tribal councils"

    assert wb.get("source") == "llm_generated_or_merged_json_style"
    mocked_llm_call.assert_called_once()

@pytest.mark.asyncio
async def test_invalid_json_output_decode_error(agent_instance):
    """Test handling of JSONDecodeError when LLM output is malformed."""
    malformed_json_str = '{"overview": {"description": "A broken world..."' # Missing closing brace
    mock_llm_output = (malformed_json_str, {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15})

    initial_setting_desc = agent_instance.plot_outline["setting_description"]

    with patch('initial_setup_logic.llm_service.async_call_llm', AsyncMock(return_value=mock_llm_output)):
        await generate_world_building_logic(agent_instance)

    wb = agent_instance.world_building
    # Expect fallback to default data
    assert wb['_overview_']['description'] == initial_setting_desc # Or a generic default if plot_outline is also empty
    assert wb.get("source") == "default_fallback"
    assert wb.get("is_default") is True
    # Check that other categories are present and empty (due to setdefault in final part of function)
    for cat_internal_key in WORLD_CATEGORY_MAP_NORMALIZED_TO_INTERNAL.values():
        if cat_internal_key != "_overview_": # overview is handled above
             assert cat_internal_key in wb
             assert wb[cat_internal_key] == {}


@pytest.mark.asyncio
async def test_llm_provides_string_for_expected_list(agent_instance):
    """Test when LLM returns a string for a field expected to be a list."""
    json_payload_string_for_list = {
        "overview": {"description": "Test"},
        "factions": {
            "Guardians": {
                "description": "Defenders of the realm.",
                "goals": "Protect the innocent" # Should be a list
            }
        }
    }
    llm_response_str = json.dumps(json_payload_string_for_list)
    mock_llm_output = (llm_response_str, {"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60})

    with patch('initial_setup_logic.llm_service.async_call_llm', AsyncMock(return_value=mock_llm_output)):
        await generate_world_building_logic(agent_instance)

    wb = agent_instance.world_building
    assert "Guardians" in wb['factions']
    assert wb['factions']['Guardians']['goals'] == ["Protect the innocent"] # Should be converted to list

@pytest.mark.asyncio
async def test_llm_provides_integer_for_expected_list(agent_instance):
    """Test when LLM returns an integer for a field expected to be a list."""
    json_payload_int_for_list = {
        "overview": {"description": "Test"},
        "systems": {
            "Magic System": {
                "description": "A complex system of arcane energies.",
                "rules": 12345 # Should be a list of strings
            }
        }
    }
    llm_response_str = json.dumps(json_payload_int_for_list)
    mock_llm_output = (llm_response_str, {"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60})

    with patch('initial_setup_logic.llm_service.async_call_llm', AsyncMock(return_value=mock_llm_output)):
        await generate_world_building_logic(agent_instance)

    wb = agent_instance.world_building
    assert "Magic System" in wb['systems']
    # The code should replace this with a fill-in placeholder list
    assert wb['systems']['Magic System']['rules'] == [config.MARKDOWN_FILL_IN_PLACEHOLDER]

@pytest.mark.asyncio
async def test_llm_output_missing_category(agent_instance):
    """Test when LLM output is valid JSON but misses an expected category."""
    json_payload_missing_category = {
        "overview": {
            "description": "A world primarily of water.",
            "mood": "Serene and vast"
        },
        "locations": {
            "Floating Market": {
                "description": "A market built on interconnected rafts.",
                "features": ["Exotic goods", "Boat travel only"]
            }
        }
        # "factions" category is missing
    }
    llm_response_str = json.dumps(json_payload_missing_category)
    mock_llm_output = (llm_response_str, {"prompt_tokens": 10, "completion_tokens": 80, "total_tokens": 90})

    with patch('initial_setup_logic.llm_service.async_call_llm', AsyncMock(return_value=mock_llm_output)):
        await generate_world_building_logic(agent_instance)

    wb = agent_instance.world_building
    assert wb['_overview_']['description'] == "A world primarily of water."
    assert "Floating Market" in wb['locations']

    # Check that all categories defined in WORLD_CATEGORY_MAP_NORMALIZED_TO_INTERNAL exist
    # For missing ones like 'factions', they should be initialized as empty dicts by the end of the function
    # if the source is not user_supplied_yaml (which it isn't in this LLM-only flow)
    for cat_norm, cat_internal in WORLD_CATEGORY_MAP_NORMALIZED_TO_INTERNAL.items():
        assert cat_internal in wb
        if cat_norm not in json_payload_missing_category:
            assert wb[cat_internal] == {} # Should be initialized as empty

@pytest.mark.asyncio
async def test_llm_output_empty_list_for_list_field(agent_instance):
    """Test when LLM provides an empty list for a field that expects a list."""
    json_payload = {
        "overview": {"description": "Test"},
        "factions": {
            "Silent Monks": {
                "description": "They speak no words.",
                "goals": [] # Empty list provided by LLM
            }
        }
    }
    llm_response_str = json.dumps(json_payload)
    mock_llm_output = (llm_response_str, {"prompt_tokens":10, "completion_tokens":50, "total_tokens":60})

    with patch('initial_setup_logic.llm_service.async_call_llm', AsyncMock(return_value=mock_llm_output)):
        await generate_world_building_logic(agent_instance)

    wb = agent_instance.world_building
    # The current logic, if an empty list is provided by LLM for a list key,
    # and the existing value was [Fill-in] or None, it will replace it with [config.MARKDOWN_FILL_IN_PLACEHOLDER]
    # This is because `processed_list` will be empty, and it hits the `elif utils._is_fill_in(existing_item_val) or existing_item_val is None:`
    assert wb['factions']['Silent Monks']['goals'] == [config.MARKDOWN_FILL_IN_PLACEHOLDER]

@pytest.mark.asyncio
async def test_existing_user_data_preserved_and_llm_fills_gaps(agent_instance):
    """Test that existing user-supplied data is preserved and LLM fills in missing parts."""
    agent_instance.world_building = {
        "source": "user_supplied_yaml", # Mark as user-supplied
        "_overview_": {
            "description": "User's original world description.",
            "mood": config.MARKDOWN_FILL_IN_PLACEHOLDER # User wants LLM to fill this
        },
        "locations": {
            "User Keep": {
                "description": "A sturdy keep from user data.",
                "atmosphere": "Ancient and strong"
            }
        },
        "factions": { # User wants this whole category filled by LLM
            # This implies that if "factions" itself is missing, or present but empty,
            # or contains only fill-in markers, LLM should populate it.
            # For this test, let's assume "factions" category is requested for fill-in by its absence
            # or by an explicit [Fill-in] marker if the structure supported that at category level.
            # The current logic will create it if absent in agent.world_building then populate.
        }
    }
    # Ensure all categories are at least setdefault in agent.world_building before LLM call
    # This mimics the behavior if _populate_agent_state_from_user_data ran
    for cat_key in WORLD_CATEGORY_MAP_NORMALIZED_TO_INTERNAL.values():
        agent_instance.world_building.setdefault(cat_key, {})


    llm_json_payload = {
        "overview": { # LLM attempts to overwrite description, but provides mood
            "description": "LLM's newer world description.",
            "mood": "Mystical and vibrant (from LLM)"
        },
        "locations": { # LLM adds a new location and tries to modify User Keep
            "User Keep": {
                "description": "LLM tries to change this.", # Should be ignored
                "atmosphere": "LLM tries to change this too.", # Should be ignored
                "features": ["New LLM feature"] # This is a new field, should be added
            },
            "LLM Lava Caves": {
                "description": "Dangerous lava caves from LLM.",
                "atmosphere": "Hot and oppressive"
            }
        },
        "factions": { # LLM generates this category
            "Sun Scorched Clan": {
                "description": "A clan adapted to extreme heat.",
                "goals": ["Find the legendary Sunstone", "Appease the Sun God"]
            }
        }
    }
    llm_response_str = json.dumps(llm_json_payload)
    mock_llm_output = (llm_response_str, {"prompt_tokens": 20, "completion_tokens": 150, "total_tokens": 170})

    with patch('initial_setup_logic.llm_service.async_call_llm', AsyncMock(return_value=mock_llm_output)):
        await generate_world_building_logic(agent_instance)

    wb = agent_instance.world_building

    # Overview: User's description preserved, LLM's mood used for [Fill-in]
    assert wb['_overview_']['description'] == "User's original world description."
    assert wb['_overview_']['mood'] == "Mystical and vibrant (from LLM)"

    # Locations: User Keep's original details preserved, new field 'features' added by LLM
    assert "User Keep" in wb['locations']
    assert wb['locations']['User Keep']['description'] == "A sturdy keep from user data."
    assert wb['locations']['User Keep']['atmosphere'] == "Ancient and strong"
    assert wb['locations']['User Keep']['features'] == ["New LLM feature"]

    # New location from LLM added
    assert "LLM Lava Caves" in wb['locations']
    assert wb['locations']['LLM Lava Caves']['description'] == "Dangerous lava caves from LLM."

    # Factions: Generated by LLM
    assert "Sun Scorched Clan" in wb['factions']
    assert wb['factions']['Sun Scorched Clan']['description'] == "A clan adapted to extreme heat."
    assert wb['factions']['Sun Scorched Clan']['goals'] == ["Find the legendary Sunstone", "Appease the Sun God"]

    # Source should indicate it was merged/processed, not purely user_supplied anymore if LLM was called
    assert wb.get("source") == "llm_generated_or_merged_json_style"

    # is_default and user_supplied_data flags should be removed after processing
    assert "is_default" not in wb
    assert "user_supplied_data" not in wb
