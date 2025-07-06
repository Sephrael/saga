# tests/test_planner_agent_json_plan.py
from agents.planner_agent import PlannerAgent


def test_load_json_plan_valid():
    agent = PlannerAgent()
    data = agent._load_json_plan('[{"scene_number": 1, "summary": "s"}]', 1)
    assert data == [{"scene_number": 1, "summary": "s"}]


def test_load_json_plan_fallback():
    agent = PlannerAgent()
    text = 'garbage [{"scene_number": 1, "summary": "s"}] trailing'
    data = agent._load_json_plan(text, 1)
    assert data == [{"scene_number": 1, "summary": "s"}]


def test_load_json_plan_invalid():
    agent = PlannerAgent()
    assert agent._load_json_plan("invalid", 1) is None


def test_load_json_plan_non_list():
    agent = PlannerAgent()
    assert agent._load_json_plan('{"a": 1}', 1) is None


def test_load_json_plan_empty_list():
    agent = PlannerAgent()
    assert agent._load_json_plan("[]", 1) is None
