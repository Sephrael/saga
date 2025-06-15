import pytest
from unittest.mock import AsyncMock

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
