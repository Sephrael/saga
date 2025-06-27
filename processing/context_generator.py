from __future__ import annotations

from typing import Any

from chapter_generation.context_service import ContextService

from models import SceneDetail

context_service = ContextService()


async def _generate_semantic_chapter_context_logic(
    agent_or_props: Any, current_chapter_number: int
) -> str:
    """Backward compatible wrapper for ContextService.get_semantic_context."""
    return await context_service.get_semantic_context(
        agent_or_props, current_chapter_number
    )


async def generate_hybrid_chapter_context_logic(
    agent_or_props: Any,
    current_chapter_number: int,
    chapter_plan: list[SceneDetail] | None,
) -> str:
    """Backward compatible wrapper for ContextService.build_hybrid_context."""
    return await context_service.build_hybrid_context(
        agent_or_props, current_chapter_number, chapter_plan
    )
