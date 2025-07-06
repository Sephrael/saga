# tests/test_world_continuity_agent.py
from unittest.mock import AsyncMock

import pytest
from agents import comprehensive_evaluator_agent
from agents.comprehensive_evaluator_agent import ComprehensiveEvaluatorAgent
from core.llm_interface import llm_service
from data_access import character_queries, world_queries


@pytest.mark.asyncio
async def test_check_scene_plan_consistency(monkeypatch):
    agent = ComprehensiveEvaluatorAgent()

    monkeypatch.setattr(
        character_queries,
        "get_character_profiles_from_db",
        AsyncMock(side_effect=Exception("should not call")),
    )
    monkeypatch.setattr(
        world_queries,
        "get_all_world_item_ids_by_category",
        AsyncMock(side_effect=Exception("should not call")),
    )

    monkeypatch.setattr(
        comprehensive_evaluator_agent, "render_prompt", lambda *a, **k: ""
    )
    monkeypatch.setattr(
        llm_service, "async_call_llm", AsyncMock(return_value=("[]", {}))
    )

    await agent.check_scene_plan_consistency(
        {"plot_points": []},
        [{"scene_number": 1}],
        1,
        "ctx",
    )
