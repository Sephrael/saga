import pytest
from agents.world_continuity_agent import WorldContinuityAgent
from processing.evaluation_helpers import parse_llm_evaluation_output


@pytest.mark.asyncio
async def test_eval_agent_parsing_valid(monkeypatch):
    data = '[{"issue_category": "plot_arc", "problem_description": "d", "quote_from_original_text": "q", "suggested_fix_focus": "f"}]'
    problems = await parse_llm_evaluation_output(data, 1, "text")
    assert problems[0]["issue_category"] == "plot_arc"


@pytest.mark.asyncio
async def test_eval_agent_parsing_empty():
    result = await parse_llm_evaluation_output("", 1, "text")
    assert result == []


@pytest.mark.asyncio
async def test_consistency_agent_parsing_invalid(monkeypatch):
    agent = WorldContinuityAgent()
    problems = await agent._parse_llm_consistency_output("notjson", 1, "text")
    assert problems[0]["issue_category"] == "consistency"
