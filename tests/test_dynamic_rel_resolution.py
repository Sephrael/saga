import json
from unittest.mock import AsyncMock

import pytest

from agents.kg_maintainer_agent import KGMaintainerAgent
from core.llm_interface import llm_service
from data_access import kg_queries


@pytest.mark.asyncio
async def test_resolve_dynamic_relationships(monkeypatch):
    agent = KGMaintainerAgent()

    monkeypatch.setattr(
        kg_queries,
        "get_dynamic_rels_for_resolution",
        AsyncMock(
            return_value=[
                {
                    "rel_id": 1,
                    "subject_name": "Alice",
                    "object_name": "Bob",
                    "current_type": "met",
                }
            ]
        ),
    )

    updated: list = []

    async def fake_update(rid, new_type):
        updated.append((rid, new_type))

    monkeypatch.setattr(
        kg_queries, "update_dynamic_rel_type", AsyncMock(side_effect=fake_update)
    )

    async def fake_call(**_):
        return json.dumps({"resolved_type": "MET_AT"}), None

    monkeypatch.setattr(llm_service, "async_call_llm", fake_call)

    await agent._resolve_dynamic_relationships()
    assert updated == [(1, "MET_AT")]
