# tests/test_patch_validation_agent.py
import pytest
from agents.patch_validation_agent import PatchValidationAgent
from core.llm_interface import llm_service


@pytest.mark.asyncio
async def test_validation_rejects_no(monkeypatch):
    async def fake_call(*_a, **_k):
        return "NO\nBad patch", None

    monkeypatch.setattr(llm_service, "async_call_llm", fake_call)
    agent = PatchValidationAgent()
    ok, reason, _ = await agent.validate_patch("ctx", {"replace_with": "x"}, [])
    assert not ok


@pytest.mark.asyncio
async def test_validation_accepts_yes(monkeypatch):
    async def fake_call(*_a, **_k):
        return "YES\nLooks good", None

    monkeypatch.setattr(llm_service, "async_call_llm", fake_call)
    agent = PatchValidationAgent()
    ok, reason, _ = await agent.validate_patch("ctx", {"replace_with": "x"}, [])
    assert ok


@pytest.mark.asyncio
async def test_validation_failure_reason(monkeypatch):
    async def fake_call(*_a, **_k):
        return "NO\nMissing dragon", None

    monkeypatch.setattr(llm_service, "async_call_llm", fake_call)
    agent = PatchValidationAgent()
    problems = [
        {
            "problem_description": "lacking dragon",
            "rewrite_instruction": "mention the dragon",
        }
    ]
    ok, reason, _ = await agent.validate_patch(
        "ctx", {"replace_with": "A hero wins."}, problems
    )
    assert not ok and reason == "Missing dragon"
