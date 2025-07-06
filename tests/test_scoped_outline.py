# tests/test_scoped_outline.py
from unittest.mock import AsyncMock

import pytest
from agents.comprehensive_evaluator_agent import ComprehensiveEvaluatorAgent
from core.llm_interface import llm_service
from data_access import chapter_queries

import utils


@pytest.mark.asyncio
async def test_evaluation_prompt_uses_scoped_plot_points(monkeypatch):
    monkeypatch.setattr(utils, "load_spacy_model_if_needed", lambda: None)
    monkeypatch.setattr(
        llm_service, "async_get_embedding", AsyncMock(return_value=[0.0])
    )
    monkeypatch.setattr(
        chapter_queries, "get_embedding_from_db", AsyncMock(return_value=[0.0])
    )
    monkeypatch.setattr(
        "processing.evaluation_helpers.parse_llm_evaluation_output",
        AsyncMock(return_value=[]),
    )

    captured = {}

    async def fake_llm(*args, **kwargs):
        captured["prompt"] = kwargs.get("prompt") or args[1]
        return "[]", {}

    monkeypatch.setattr(llm_service, "async_call_llm", fake_llm)

    agent = ComprehensiveEvaluatorAgent()
    outline = {"plot_points": ["pp1", "pp2", "pp3", "pp4"]}
    scoped = utils.get_scoped_plot_outline(outline, 3)
    await agent.evaluate_chapter_draft(scoped, "draft", 3, "pp2", 1, "ctx")

    assert "pp2" in captured["prompt"]
    assert "pp3" in captured["prompt"]
    assert "pp1" not in captured["prompt"]
    assert "- PP 4: pp4" not in captured["prompt"]
