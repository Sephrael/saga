import asyncio
import pytest

from comprehensive_evaluator_agent import ComprehensiveEvaluatorAgent
from world_continuity_agent import WorldContinuityAgent


@pytest.mark.asyncio
async def test_eval_agent_parsing_valid(monkeypatch):
    agent = ComprehensiveEvaluatorAgent()
    monkeypatch.setattr(
        agent, "_parse_llm_evaluation_output", agent._parse_llm_evaluation_output
    )
    data = '[{"issue_category": "plot_arc", "problem_description": "d", "quote_from_original_text": "q", "suggested_fix_focus": "f"}]'
    problems = await agent._parse_llm_evaluation_output(data, 1, "text")
    assert problems[0]["issue_category"] == "plot_arc"


@pytest.mark.asyncio
async def test_eval_agent_parsing_empty():
    agent = ComprehensiveEvaluatorAgent()
    result = await agent._parse_llm_evaluation_output("", 1, "text")
    assert result == []


@pytest.mark.asyncio
async def test_consistency_agent_parsing_invalid(monkeypatch):
    agent = WorldContinuityAgent()
    problems = await agent._parse_llm_consistency_output("notjson", 1, "text")
    assert problems[0]["issue_category"] == "consistency"
