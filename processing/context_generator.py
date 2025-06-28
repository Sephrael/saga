from __future__ import annotations

import importlib
from typing import Any

from chapter_generation import ContextOrchestrator, ContextRequest
from config import settings

from models import SceneDetail

provider_instances = []
for dotted in settings.CONTEXT_PROVIDERS:
    module_name, class_name = dotted.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    provider_instances.append(cls())
context_service = ContextOrchestrator(provider_instances)


async def _generate_semantic_chapter_context_logic(
    agent_or_props: Any, current_chapter_number: int
) -> str:
    """Generate semantic context via the orchestrator."""
    outline = (
        agent_or_props.get("plot_outline_full", agent_or_props.get("plot_outline", {}))
        if isinstance(agent_or_props, dict)
        else getattr(agent_or_props, "plot_outline_full", None)
        or getattr(agent_or_props, "plot_outline", {})
    )
    request = ContextRequest(
        chapter_number=current_chapter_number,
        plot_focus=None,
        plot_outline=outline,
    )
    return await context_service.build_context(request)


async def generate_hybrid_chapter_context_logic(
    agent_or_props: Any,
    current_chapter_number: int,
    chapter_plan: list[SceneDetail] | None,
) -> str:
    """Generate full context via the orchestrator."""
    outline = (
        agent_or_props.get("plot_outline_full", agent_or_props.get("plot_outline", {}))
        if isinstance(agent_or_props, dict)
        else getattr(agent_or_props, "plot_outline_full", None)
        or getattr(agent_or_props, "plot_outline", {})
    )
    request = ContextRequest(
        chapter_number=current_chapter_number,
        plot_focus=None,
        plot_outline=outline,
        chapter_plan=chapter_plan,
    )
    return await context_service.build_context(request)
