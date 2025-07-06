from unittest.mock import AsyncMock

import pytest
from data_access import world_queries

import utils


@pytest.mark.asyncio
async def test_fix_missing_world_element_core_fields(monkeypatch):
    sample = [
        {"nid": 1, "id": None, "name": "Guardian system", "category": "systems"},
        {"nid": 2, "id": "systems_life_pod", "name": None, "category": "systems"},
        {
            "nid": 3,
            "id": "lore_alien_structure",
            "name": "Alien structure",
            "category": None,
        },
    ]

    monkeypatch.setattr(
        world_queries.neo4j_manager,
        "execute_read_query",
        AsyncMock(return_value=sample),
    )
    captured = []

    async def fake_batch(statements):
        captured.extend(statements)

    monkeypatch.setattr(
        world_queries.neo4j_manager,
        "execute_cypher_batch",
        AsyncMock(side_effect=fake_batch),
    )

    updated = await world_queries.fix_missing_world_element_core_fields()

    assert updated == 3
    assert (
        captured[0][1]["props"]["id"]
        == f"systems_{utils._normalize_for_id('Guardian system')}"
    )
    assert captured[1][1]["props"]["name"] == "Life Pod"
    assert captured[2][1]["props"]["category"] == "lore"


@pytest.mark.asyncio
async def test_fix_missing_world_element_core_fields_defaults(monkeypatch):
    sample = [{"nid": 4, "id": None, "name": "Feeds", "category": None}]
    monkeypatch.setattr(
        world_queries.neo4j_manager,
        "execute_read_query",
        AsyncMock(return_value=sample),
    )
    captured = []

    async def fake_batch(statements):
        captured.extend(statements)

    monkeypatch.setattr(
        world_queries.neo4j_manager,
        "execute_cypher_batch",
        AsyncMock(side_effect=fake_batch),
    )

    updated = await world_queries.fix_missing_world_element_core_fields()

    assert updated == 1
    props = captured[0][1]["props"]
    assert props["category"] == "unknown_category"
    assert props["id"].startswith("unknown_category_")


@pytest.mark.asyncio
async def test_fix_missing_world_element_core_fields_blank(monkeypatch):
    sample = [{"nid": 5, "id": " ", "name": "Hope", "category": ""}]
    monkeypatch.setattr(
        world_queries.neo4j_manager,
        "execute_read_query",
        AsyncMock(return_value=sample),
    )
    captured = []

    async def fake_batch(statements):
        captured.extend(statements)

    monkeypatch.setattr(
        world_queries.neo4j_manager,
        "execute_cypher_batch",
        AsyncMock(side_effect=fake_batch),
    )

    updated = await world_queries.fix_missing_world_element_core_fields()

    assert updated == 1
    props = captured[0][1]["props"]
    assert props["category"] == "unknown_category"
    assert props["id"].startswith("unknown_category_")


@pytest.mark.asyncio
async def test_get_world_building_runs_healer(monkeypatch):
    sample_we = [
        {
            "we": {
                "id": "places_city",
                "name": "City",
                "category": "places",
                "created_ts": 1,
                world_queries.KG_NODE_CREATED_CHAPTER: 1,
            }
        }
    ]

    async def fake_read(query, params=None):
        if "RETURN wc" in query:
            return [{"wc": {"overview_description": "desc"}}]
        if "RETURN we" in query:
            return sample_we
        return []

    monkeypatch.setattr(
        world_queries.neo4j_manager,
        "execute_read_query",
        AsyncMock(side_effect=fake_read),
    )

    healer_called = AsyncMock(return_value=0)
    monkeypatch.setattr(
        world_queries,
        "fix_missing_world_element_core_fields",
        healer_called,
    )

    world_queries.WORLD_NAME_TO_ID.clear()
    await world_queries.get_world_building_from_db()
    healer_called.assert_awaited_once()
