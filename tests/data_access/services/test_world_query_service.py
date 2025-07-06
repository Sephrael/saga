# tests/data_access/services/test_world_query_service.py
from unittest.mock import AsyncMock

import kg_constants as kg_keys
import pytest
from core.db_manager import neo4j_manager
from data_access.services.world_query_service import WorldQueryService
from data_access.utils import world_utils

import utils


@pytest.mark.asyncio
async def test_get_world_building_data_populates_cache(monkeypatch):
    service = WorldQueryService()

    sample_records = [
        {
            "we": {
                "id": "locations_city",
                "name": "City",
                "category": "locations",
                kg_keys.KG_NODE_CREATED_CHAPTER: 1,
            },
            "goals": [],
            "rules": [],
            "key_elements": [],
            "traits": [],
            "elaborations": [],
        },
        {
            "we": {
                "id": "factions_band",
                "name": "Band",
                "category": "factions",
                kg_keys.KG_NODE_CREATED_CHAPTER: 1,
            },
            "goals": [],
            "rules": [],
            "key_elements": [],
            "traits": [],
            "elaborations": [],
        },
    ]

    async def fake_read(query, params=None):
        if "WorldContainer" in query:
            return [
                {
                    "wc": {
                        "id": "wc1",
                        "overview_description": "desc",
                    }
                }
            ]
        if "WorldElement" in query:
            return sample_records
        return []

    mock = AsyncMock(side_effect=fake_read)
    monkeypatch.setattr(neo4j_manager, "execute_read_query", mock)

    world_utils.clear_world_name_to_id_cache()
    service.get_world_building_data.cache_clear()

    data1 = await service.get_world_building_data()
    call_count = mock.await_count
    data2 = await service.get_world_building_data()

    assert call_count == mock.await_count
    assert {"locations", "factions"}.issubset(data1.keys())
    norm_city = utils._normalize_for_id("City")
    norm_band = utils._normalize_for_id("Band")
    assert world_utils.WORLD_NAME_TO_ID_CACHE[norm_city] == "locations_city"
    assert world_utils.WORLD_NAME_TO_ID_CACHE[norm_band] == "factions_band"
    assert data1 is data2

    service.get_world_building_data.cache_clear()
