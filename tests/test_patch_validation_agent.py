import pytest
from agents.patch_validation_agent import PatchValidationAgent
from config import settings
from core.llm_interface import llm_service


@pytest.mark.asyncio
async def test_validation_rejects_below_threshold(monkeypatch):
    async def fake_call(*_a, **_k):
        return f"{settings.PATCH_VALIDATION_THRESHOLD - 1} low", None

    monkeypatch.setattr(llm_service, "async_call_llm", fake_call)
    agent = PatchValidationAgent()
    ok, reason, _ = await agent.validate_patch("ctx", {"replace_with": "x"}, [])
    assert not ok


@pytest.mark.asyncio
async def test_validation_accepts_at_threshold(monkeypatch):
    async def fake_call(*_a, **_k):
        return f"{settings.PATCH_VALIDATION_THRESHOLD} ok", None

    monkeypatch.setattr(llm_service, "async_call_llm", fake_call)
    agent = PatchValidationAgent()
    ok, reason, _ = await agent.validate_patch("ctx", {"replace_with": "x"}, [])
    assert ok


@pytest.mark.asyncio
async def test_rewrite_instruction_missing_keyword(monkeypatch):
    async def fake_call(*_a, **_k):
        return f"{settings.PATCH_VALIDATION_THRESHOLD + 10} good", None

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
    assert not ok


@pytest.mark.asyncio
async def test_rewrite_instruction_keywords_present(monkeypatch):
    async def fake_call(*_a, **_k):
        return f"{settings.PATCH_VALIDATION_THRESHOLD + 10} good", None

    monkeypatch.setattr(llm_service, "async_call_llm", fake_call)
    agent = PatchValidationAgent()
    problems = [
        {
            "problem_description": "lacking dragon",
            "rewrite_instruction": "mention the dragon",
        }
    ]
    ok, reason, _ = await agent.validate_patch(
        "ctx", {"replace_with": "The dragon appears."}, problems
    )
    assert ok
