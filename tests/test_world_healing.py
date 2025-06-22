from unittest.mock import AsyncMock

import pytest

import utils
from data_access import world_queries


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
