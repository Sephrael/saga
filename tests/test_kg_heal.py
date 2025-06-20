from unittest.mock import AsyncMock

import pytest

from data_access import kg_queries


@pytest.mark.asyncio
async def test_normalize_existing_relationship_types(monkeypatch):
    async def fake_read(query, params=None):
        return [{"t": "knows"}, {"t": "Is_Friend_of"}, {"t": None}]

    executed = []

    async def fake_batch(statements):
        executed.extend(statements)

    monkeypatch.setattr(
        kg_queries.neo4j_manager, "execute_read_query", AsyncMock(side_effect=fake_read)
    )
    monkeypatch.setattr(
        kg_queries.neo4j_manager,
        "execute_cypher_batch",
        AsyncMock(side_effect=fake_batch),
    )

    await kg_queries.normalize_existing_relationship_types()

    assert executed
    normalized = {params["new"] for _, params in executed}
    assert "KNOWS" in normalized
    assert "IS_FRIEND_OF" in normalized


@pytest.mark.asyncio
async def test_promote_dynamic_relationships(monkeypatch):
    async def fake_types():
        return ["KNOWS", "ALLY_OF"]

    async def fake_write(query, params=None):
        assert params == {"valid_types": ["KNOWS", "ALLY_OF"]}
        return [{"promoted": 2}]

    monkeypatch.setattr(
        kg_queries,
        "get_defined_relationship_types",
        AsyncMock(side_effect=fake_types),
    )
    monkeypatch.setattr(
        kg_queries.neo4j_manager,
        "execute_write_query",
        AsyncMock(side_effect=fake_write),
    )

    promoted = await kg_queries.promote_dynamic_relationships()
    assert promoted == 2


@pytest.mark.asyncio
async def test_deduplicate_relationships(monkeypatch):
    async def fake_write(query, params=None):
        return [{"removed": 1}]

    monkeypatch.setattr(
        kg_queries.neo4j_manager,
        "execute_write_query",
        AsyncMock(side_effect=fake_write),
    )

    removed = await kg_queries.deduplicate_relationships()
    assert removed == 1
