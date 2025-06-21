from unittest.mock import AsyncMock

import pytest

from agents import world_continuity_agent
from agents.world_continuity_agent import WorldContinuityAgent
from core.llm_interface import llm_service
from data_access import character_queries, world_queries


@pytest.mark.asyncio
async def test_check_consistency_passes_id_mapping(monkeypatch):
    agent = WorldContinuityAgent()

    monkeypatch.setattr(
        character_queries, "get_character_profiles_from_db", AsyncMock(return_value=[])
    )
    mapping = {"places": ["id1"]}
    monkeypatch.setattr(
        world_queries,
        "get_all_world_item_ids_by_category",
        AsyncMock(return_value=mapping),
    )

    received: dict = {}

    async def fake_get_world(ids_by_cat, up_to):
        received["ids"] = ids_by_cat
        return ""

    monkeypatch.setattr(
        world_continuity_agent,
        "get_filtered_world_data_for_prompt_plain_text",
        fake_get_world,
    )

    monkeypatch.setattr(world_continuity_agent, "render_prompt", lambda *a, **k: "")
    monkeypatch.setattr(
        llm_service, "async_call_llm", AsyncMock(return_value=("[]", {}))
    )

    await agent.check_consistency({"plot_points": []}, "draft", 1, "")

    assert received["ids"] == mapping


@pytest.mark.asyncio
async def test_check_scene_plan_consistency(monkeypatch):
    agent = WorldContinuityAgent()

    monkeypatch.setattr(
        character_queries, "get_character_profiles_from_db", AsyncMock(return_value=[])
    )
    mapping = {"places": ["id2"]}
    monkeypatch.setattr(
        world_queries,
        "get_all_world_item_ids_by_category",
        AsyncMock(return_value=mapping),
    )

    received: dict = {}

    async def fake_get_world(ids_by_cat, up_to):
        received["ids"] = ids_by_cat
        return ""

    monkeypatch.setattr(
        world_continuity_agent,
        "get_filtered_world_data_for_prompt_plain_text",
        fake_get_world,
    )

    monkeypatch.setattr(world_continuity_agent, "render_prompt", lambda *a, **k: "")
    monkeypatch.setattr(
        llm_service, "async_call_llm", AsyncMock(return_value=("[]", {}))
    )

    await agent.check_scene_plan_consistency(
        {"plot_points": []},
        [{"scene_number": 1}],
        1,
    )

    assert received["ids"] == mapping
