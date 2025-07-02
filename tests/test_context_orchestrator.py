import asyncio

import pytest
from chapter_generation.context_models import (
    ContextChunk,
    ContextProfileName,
    ContextRequest,
    ProfileConfiguration,
    ProviderSettings,
)
from chapter_generation.context_orchestrator import ContextOrchestrator
from chapter_generation.context_providers import ContextProvider


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
async def test_orchestrator_truncates():
    profiles = {
        ContextProfileName.DEFAULT: ProfileConfiguration(
            providers=[
                ProviderSettings(DummyProvider("abc", "A")),
                ProviderSettings(DummyProvider("defghij", "B")),
            ],
            max_tokens=5,
        )
    }
    orch = ContextOrchestrator(profiles)
    req = ContextRequest(1, None, {})
    out = await orch.build_context(req)
    assert "A" in out
    assert "B" not in out or "defghij" not in out


@pytest.mark.asyncio
async def test_agent_hints_cache_key():
    profiles = {
        ContextProfileName.DEFAULT: ProfileConfiguration(
            providers=[ProviderSettings(DummyProvider("abc", "A"))],
            max_tokens=50,
        )
    }
    orch = ContextOrchestrator(profiles)
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
    profiles = {
        ContextProfileName.DEFAULT: ProfileConfiguration(
            providers=[
                ProviderSettings(SlowProvider(0.05, "a", "A")),
                ProviderSettings(SlowProvider(0.05, "b", "B")),
            ],
            max_tokens=50,
        )
    }
    orch = ContextOrchestrator(profiles)
    req = ContextRequest(1, None, {})
    events: list[str] = []
    for prov in [ps.provider for ps in profiles[ContextProfileName.DEFAULT].providers]:
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


@pytest.mark.asyncio
async def test_profile_selection():
    profiles = {
        ContextProfileName.DEFAULT: ProfileConfiguration(
            providers=[ProviderSettings(DummyProvider("a", "A"))],
            max_tokens=50,
        ),
        ContextProfileName.ALTERNATE: ProfileConfiguration(
            providers=[ProviderSettings(DummyProvider("b", "B"))],
            max_tokens=50,
        ),
    }
    orch = ContextOrchestrator(profiles)
    req = ContextRequest(1, None, {}, profile_name=ContextProfileName.ALTERNATE)
    ctx = await orch.build_context(req)
    assert ctx.startswith("[B]")
