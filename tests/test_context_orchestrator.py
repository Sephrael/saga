import asyncio

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


class SlowProvider(ContextProvider):
    """Provider that sleeps before returning."""

    def __init__(self, delay: float, text: str, source: str) -> None:
        self.delay = delay
        self.text = text
        self.source = source

    async def get_context(self, request: ContextRequest) -> ContextChunk:
        await asyncio.sleep(self.delay)
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


@pytest.mark.asyncio
async def test_providers_run_concurrently():
    orch = ContextOrchestrator(
        [SlowProvider(0.05, "a", "A"), SlowProvider(0.05, "b", "B")]
    )
    req = ContextRequest(1, None, {})
    events: list[str] = []
    for prov in orch.providers:
        original = prov.get_context

        async def wrapper(
            request: ContextRequest, *, _orig=original, _prov=prov
        ) -> ContextChunk:
            events.append(f"start-{_prov.source}")
            chunk = await _orig(request)
            events.append(f"end-{_prov.source}")
            return chunk

        prov.get_context = wrapper  # type: ignore[assignment]
    await orch.build_context(req)
    assert events[0].startswith("start") and events[1].startswith("start")
