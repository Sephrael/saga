# tests/test_context_orchestrator.py
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

    async def get_context(
        self, request: ContextRequest, settings: ProviderSettings | None = None
    ) -> ContextChunk:
        return ContextChunk(
            text=self.text, tokens=len(self.text), provenance={}, source=self.source
        )


class SlowProvider(ContextProvider):
    """Provider that sleeps before returning."""

    def __init__(self, delay: float, text: str, source: str) -> None:
        self.delay = delay
        self.text = text
        self.source = source

    async def get_context(
        self, request: ContextRequest, settings: ProviderSettings | None = None
    ) -> ContextChunk:
        await asyncio.sleep(self.delay)
        return ContextChunk(
            text=self.text, tokens=len(self.text), provenance={}, source=self.source
        )


class ConfigAwareProvider(ContextProvider):
    def __init__(self) -> None:
        self.last_setting = None
        self.source = "cfg"

    async def get_context(
        self, request: ContextRequest, settings: ProviderSettings | None = None
    ) -> ContextChunk:
        self.last_setting = settings.max_tokens if settings else None
        text = "x" * 10
        if settings and settings.max_tokens:
            text = text[: settings.max_tokens]
        return ContextChunk(
            text=text, tokens=len(text), provenance={}, source=self.source
        )


class CountingProvider(ContextProvider):
    def __init__(self, text: str, source: str) -> None:
        self.text = text
        self.source = source
        self.call_count = 0

    async def get_context(
        self, request: ContextRequest, settings: ProviderSettings | None = None
    ) -> ContextChunk:
        self.call_count += 1
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
            request: ContextRequest,
            settings: ProviderSettings | None = None,
            *,
            _orig=original,
            _prov=prov,
        ) -> ContextChunk:
            events.append(f"start-{_prov.source}")
            chunk = await _orig(request, settings)
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


@pytest.mark.asyncio
async def test_provider_settings_passed():
    provider = ConfigAwareProvider()
    profiles = {
        ContextProfileName.DEFAULT: ProfileConfiguration(
            providers=[ProviderSettings(provider, max_tokens=5)],
            max_tokens=50,
        )
    }
    orch = ContextOrchestrator(profiles)
    req = ContextRequest(1, None, {})
    await orch.build_context(req)
    assert provider.last_setting == 5


@pytest.mark.asyncio
async def test_cache_miss_on_profile_change():
    p1 = CountingProvider("a", "A")
    p2 = CountingProvider("b", "B")
    profiles = {
        ContextProfileName.DEFAULT: ProfileConfiguration(
            providers=[ProviderSettings(p1)],
            max_tokens=50,
        ),
        ContextProfileName.ALTERNATE: ProfileConfiguration(
            providers=[ProviderSettings(p2)],
            max_tokens=50,
        ),
    }
    orch = ContextOrchestrator(profiles)
    req = ContextRequest(1, None, {}, profile_name=ContextProfileName.DEFAULT)
    await orch.build_context(req)
    assert p1.call_count == 1
    req.profile_name = ContextProfileName.ALTERNATE
    await orch.build_context(req)
    assert p2.call_count == 1
    req.profile_name = ContextProfileName.DEFAULT
    await orch.build_context(req)
    assert p1.call_count == 1


@pytest.mark.asyncio
async def test_cache_miss_on_provider_setting_change():
    provider = CountingProvider("abc", "A")
    profiles = {
        ContextProfileName.DEFAULT: ProfileConfiguration(
            providers=[ProviderSettings(provider, max_tokens=5)],
            max_tokens=50,
        )
    }
    orch = ContextOrchestrator(profiles)
    req = ContextRequest(1, None, {})
    await orch.build_context(req)
    assert provider.call_count == 1
    profiles[ContextProfileName.DEFAULT].providers[0].max_tokens = 6
    await orch.build_context(req)
    assert provider.call_count == 2
