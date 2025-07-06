# tests/test_evaluation_parsing.py
import pytest
from agents.comprehensive_evaluator_agent import ComprehensiveEvaluatorAgent
from processing.evaluation_helpers import parse_llm_evaluation_output


@pytest.mark.asyncio
async def test_eval_agent_parsing_valid(monkeypatch):
    data = '[{"issue_category": "plot_arc", "problem_description": "d", "quote_from_original_text": "q", "suggested_fix_focus": "f", "rewrite_instruction": "do it"}]'
    problems = await parse_llm_evaluation_output(data, 1, "text")
    assert problems[0]["issue_category"] == "plot_arc"
    assert problems[0]["rewrite_instruction"] == "do it"


@pytest.mark.asyncio
async def test_eval_agent_parsing_empty():
    result = await parse_llm_evaluation_output("", 1, "text")
    assert result == []


@pytest.mark.asyncio
async def test_consistency_agent_parsing_invalid(monkeypatch):
    agent = ComprehensiveEvaluatorAgent()
    problems = await agent._parse_llm_consistency_output("notjson", 1, "text")
    assert problems[0]["issue_category"] == "consistency"
