from unittest.mock import AsyncMock

import pytest

from data_access import kg_queries
from kg_maintainer.models import SceneDetail
from prompt_data_getters import get_reliable_kg_facts_for_drafting_prompt


@pytest.mark.asyncio
async def test_get_reliable_kg_facts_adds_path(monkeypatch):
    async def fake_get_property(key):
        return None

    monkeypatch.setattr(
        kg_queries,
        "get_novel_info_property_from_db",
        AsyncMock(side_effect=fake_get_property),
    )
    monkeypatch.setattr(
        kg_queries,
        "get_most_recent_value_from_db",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        kg_queries,
        "query_kg_from_db",
        AsyncMock(return_value=[]),
    )

    async def fake_path(src, tgt, max_depth=4):
        return [
            {"subject": src, "predicate": "KNOWS", "object": tgt},
        ]

    monkeypatch.setattr(
        kg_queries,
        "get_shortest_path_triples_between_entities",
        AsyncMock(side_effect=fake_path),
    )

    outline = {"protagonist_name": "Alice"}
    plan = [SceneDetail(scene_number=1, summary="", characters_involved=["Bob"])]
    result = await get_reliable_kg_facts_for_drafting_prompt(outline, 2, plan)
    assert "Path Alice -KNOWS-> Bob" in result
