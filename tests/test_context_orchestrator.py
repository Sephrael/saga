import pytest
from chapter_generation.context_orchestrator import ContextOrchestrator, ContextRequest
from chapter_generation.context_providers import ContextChunk, ContextProvider
from config import settings


class DummyProvider(ContextProvider):
    def __init__(self, text: str, source: str) -> None:
        self.text = text
        self.source = source

    async def get_context(self, request: ContextRequest) -> ContextChunk:
        return ContextChunk(
            text=self.text, tokens=len(self.text), provenance={}, source=self.source
        )


@pytest.mark.asyncio
async def test_orchestrator_truncates(monkeypatch):
    monkeypatch.setattr(settings, "MAX_CONTEXT_TOKENS", 5)
    orch = ContextOrchestrator(
        [DummyProvider("abc", "A"), DummyProvider("defghij", "B")]
    )
    req = ContextRequest(1, None, {})
    out = await orch.build_context(req)
    assert "A" in out
    assert "B" not in out or "defghij" not in out


@pytest.mark.asyncio
async def test_agent_hints_cache_key():
    orch = ContextOrchestrator([DummyProvider("abc", "A")])
    req = ContextRequest(
        1,
        None,
        {},
        agent_hints={"notes": ["a", {"b": 1}]},
    )
    out1 = await orch.build_context(req)
    out2 = await orch.build_context(req)
    assert out1 == out2
