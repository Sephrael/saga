# chapter_generation/context_orchestrator.py
"""Orchestrates context generation from multiple providers."""

from __future__ import annotations

import asyncio
import importlib
import json
import time
from collections.abc import Mapping
from typing import Any

import structlog
from config import settings
from core.llm_interface import count_tokens, truncate_text_by_tokens

from models.agent_models import SceneDetail

from . import context_kg_utils
from .context_models import (
    ContextChunk,
    ContextProfileName,
    ContextRequest,
    ProfileConfiguration,
    ProviderSettings,
)

logger = structlog.get_logger(__name__)


class TTLCache:
    """Simple TTL-based LRU cache."""

    def __init__(self, maxsize: int, ttl: float) -> None:
        self.maxsize = maxsize
        self.ttl = ttl
        self._data: dict[tuple, tuple[float, str]] = {}

    def get(self, key: tuple) -> str | None:
        item = self._data.get(key)
        if not item:
            return None
        ts, value = item
        if time.time() - ts > self.ttl:
            del self._data[key]
            return None
        return value

    def set(self, key: tuple, value: str) -> None:
        if len(self._data) >= self.maxsize:
            oldest = sorted(self._data.items(), key=lambda x: x[1][0])[0][0]
            del self._data[oldest]
        self._data[key] = (time.time(), value)


class ContextOrchestrator:
    """Gather and merge context from configured providers."""

    def __init__(
        self, profiles: Mapping[ContextProfileName, ProfileConfiguration]
    ) -> None:
        self.profiles = profiles
        self.cache = TTLCache(settings.CONTEXT_CACHE_SIZE, settings.CONTEXT_CACHE_TTL)
        self.llm_fill_chunks: list[ContextChunk] = []

    async def build_context(self, request: ContextRequest) -> str:
        """Return an ordered context string for the request."""
        agent_key = None
        if request.agent_hints:
            try:
                agent_key = json.dumps(request.agent_hints, sort_keys=True, default=str)
            except Exception as exc:  # pragma: no cover - log and continue
                logger.warning(
                    "Failed to serialize agent hints for cache key", error=exc
                )

        profile = self.profiles.get(
            request.profile_name, self.profiles.get(ContextProfileName.DEFAULT)
        )
        if profile is None:
            raise ValueError(f"Unknown context profile: {request.profile_name}")

        provider_conf = tuple(
            (
                type(ps.provider).__module__,
                type(ps.provider).__qualname__,
                ps.max_tokens,
                ps.detail_level,
            )
            for ps in profile.providers
        )

        cache_key = (
            request.profile_name.value,
            provider_conf,
            request.chapter_number,
            request.plot_focus,
            agent_key,
        )
        cached = self.cache.get(cache_key)
        if cached:
            logger.debug("Context cache hit", key=cache_key)
            return cached

        chunks: list[ContextChunk] = []
        tasks = [ps.provider.get_context(request, ps) for ps in profile.providers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for ps, res in zip(profile.providers, results, strict=True):
            provider = ps.provider
            if isinstance(res, Exception):  # pragma: no cover - log and skip
                logger.warning(
                    "Context provider error", provider=provider.source, error=res
                )
                continue
            if isinstance(res, ContextChunk):
                chunks.append(res)
            else:
                logger.warning(
                    "Invalid context provider result",
                    provider=provider.source,
                    result_type=type(res).__name__,
                )

        self.llm_fill_chunks = [c for c in chunks if c.from_llm_fill]

        merged: list[str] = []
        token_total = 0
        max_tokens = profile.max_tokens
        for chunk in chunks:
            if not chunk.text:
                continue
            prefix = f"[{chunk.source}]\n"
            part = prefix + chunk.text
            tokens = count_tokens(part, settings.DRAFTING_MODEL)
            if token_total + tokens > max_tokens:
                remaining = max_tokens - token_total
                if remaining <= 0:
                    break
                part = truncate_text_by_tokens(part, settings.DRAFTING_MODEL, remaining)
                merged.append(part)
                token_total += remaining
                break
            merged.append(part)
            token_total += tokens

        final_context = "\n---\n".join(merged)
        self.cache.set(cache_key, final_context)
        logger.info("Built context", tokens=token_total)
        return final_context

    async def build_hybrid_context(
        self,
        agent_or_props: Any,
        current_chapter_number: int,
        chapter_plan: list[SceneDetail] | None,
        agent_hints: dict[str, Any] | None = None,
        profile_name: ContextProfileName = ContextProfileName.DEFAULT,
        missing_entities: list[str] | None = None,
    ) -> str:
        """Build context and enrich it with facts about missing entities."""
        if isinstance(agent_or_props, dict):
            plot_outline = agent_or_props.get(
                "plot_outline_full", agent_or_props.get("plot_outline", {})
            )
            plot_focus = None
        else:
            plot_outline = getattr(
                agent_or_props, "plot_outline_full", None
            ) or getattr(agent_or_props, "plot_outline", {})
            plot_focus = getattr(agent_or_props, "plot_point_focus", None)

        if hasattr(plot_outline, "model_dump"):
            try:
                plot_outline = plot_outline.model_dump(exclude_none=True)
            except Exception as exc:  # pragma: no cover - log and continue
                logger.warning(
                    "Failed to dump plot outline to dict", error=exc, exc_info=True
                )

        missing_lines: list[str] = []
        if missing_entities:
            try:
                missing_lines = await context_kg_utils.get_facts_for_entities(
                    missing_entities
                )
            except Exception:  # pragma: no cover - log and continue
                logger.error(
                    "Failed to lookup missing entities: %s",
                    missing_entities,
                    exc_info=True,
                )

        request = ContextRequest(
            chapter_number=current_chapter_number,
            plot_focus=plot_focus,
            plot_outline=plot_outline,
            chapter_plan=chapter_plan,
            agent_hints=agent_hints,
            profile_name=profile_name,
        )
        base_context = await self.build_context(request)

        if missing_lines:
            prefix = "[KG_LOOKUP]\n" + "\n".join(missing_lines)
            if base_context:
                return prefix + "\n---\n" + base_context
            return prefix
        return base_context


def create_from_settings() -> ContextOrchestrator:
    """Instantiate an orchestrator using provider config in settings."""
    profiles: dict[ContextProfileName, ProfileConfiguration] = {}
    for name, conf in settings.CONTEXT_PROFILES.items():
        provider_instances = []
        for dotted in conf.get("providers", []):
            module_name, class_name = dotted.rsplit(".", 1)
            module = importlib.import_module(module_name)
            provider_cls = getattr(module, class_name)
            provider_instances.append(provider_cls())

        providers = [ProviderSettings(provider=p) for p in provider_instances]
        max_tokens = conf.get("max_tokens", settings.MAX_CONTEXT_TOKENS)
        profiles[ContextProfileName(name)] = ProfileConfiguration(
            providers=providers,
            max_tokens=max_tokens,
        )
    return ContextOrchestrator(profiles)
