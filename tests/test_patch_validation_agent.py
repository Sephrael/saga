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
    ok, _ = await agent.validate_patch("ctx", {"replace_with": "x"}, [])
    assert not ok


@pytest.mark.asyncio
async def test_validation_accepts_at_threshold(monkeypatch):
    async def fake_call(*_a, **_k):
        return f"{settings.PATCH_VALIDATION_THRESHOLD} ok", None

    monkeypatch.setattr(llm_service, "async_call_llm", fake_call)
    agent = PatchValidationAgent()
    ok, _ = await agent.validate_patch("ctx", {"replace_with": "x"}, [])
    assert ok
