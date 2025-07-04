# initialization/bootstrappers/plot_bootstrapper.py
import asyncio
from collections.abc import Coroutine
from typing import Any

import structlog
import utils
from config import settings

from initialization.models import PlotOutline

from .common import bootstrap_field


def _needs_bootstrap(value: Any) -> bool:
    """Return ``True`` if ``value`` is empty or a fill-in marker."""
    return not value or utils._is_fill_in(value)


def _fields_to_bootstrap(plot_outline: PlotOutline) -> dict[str, bool]:
    """Return mapping of plot fields that require bootstrapping."""
    return {
        "title": _needs_bootstrap(plot_outline.get("title")),
        "protagonist_name": _needs_bootstrap(plot_outline.get("protagonist_name")),
        "genre": _needs_bootstrap(plot_outline.get("genre")),
        "setting": _needs_bootstrap(plot_outline.get("setting")),
        "theme": _needs_bootstrap(plot_outline.get("theme")),
        "logline": _needs_bootstrap(plot_outline.get("logline")),
        "inciting_incident": _needs_bootstrap(plot_outline.get("inciting_incident")),
        "central_conflict": _needs_bootstrap(plot_outline.get("central_conflict")),
        "stakes": _needs_bootstrap(plot_outline.get("stakes")),
        "narrative_style": _needs_bootstrap(plot_outline.get("narrative_style")),
        "tone": _needs_bootstrap(plot_outline.get("tone")),
        "pacing": _needs_bootstrap(plot_outline.get("pacing")),
    }


def _calculate_needed_plot_points(points: list[str]) -> int:
    """Return count of plot points that must be generated."""
    fill_in_count = sum(1 for p in points if utils._is_fill_in(p))
    return max(
        0,
        settings.TARGET_PLOT_POINTS_INITIAL_GENERATION - (len(points) - fill_in_count),
    )


def _build_bootstrap_tasks(plot_outline: PlotOutline) -> dict[str, Coroutine]:
    """Return mapping of field names to bootstrap tasks."""
    tasks: dict[str, Coroutine] = {}
    for field, needed in _fields_to_bootstrap(plot_outline).items():
        if needed:
            tasks[field] = bootstrap_field(
                field,
                plot_outline,
                "bootstrapper/fill_plot_field.j2",
            )

    needed_plot_points = _calculate_needed_plot_points(
        plot_outline.get("plot_points", []) or []
    )
    if needed_plot_points > 0:
        tasks["plot_points"] = bootstrap_field(
            "plot_points",
            plot_outline,
            "bootstrapper/fill_plot_points.j2",
            is_list=True,
            list_count=needed_plot_points,
        )
    return tasks


def _apply_bootstrap_results(
    plot_outline: PlotOutline,
    results: list[tuple[Any, dict[str, int] | None]],
    keys: list[str],
    usage_data: dict[str, int],
) -> None:
    """Merge bootstrap results back into ``plot_outline`` and ``usage_data``."""
    for i, (value, usage) in enumerate(results):
        field = keys[i]
        if usage:
            for k, v in usage.items():
                usage_data[k] = usage_data.get(k, 0) + v
        if field == "plot_points":
            final_points = [
                p
                for p in plot_outline.get("plot_points", [])
                if not utils._is_fill_in(p)
            ]
            final_points.extend(value)
            plot_outline["plot_points"] = final_points[
                : settings.TARGET_PLOT_POINTS_INITIAL_GENERATION
            ]
        elif value:
            plot_outline[field] = value

    if usage_data["total_tokens"] > 0:
        plot_outline["is_default"] = False
        plot_outline["source"] = "bootstrapped"


logger = structlog.get_logger(__name__)


def create_default_plot(default_protagonist_name: str) -> PlotOutline:
    """Create a default plot outline with placeholders."""
    num_points = settings.TARGET_PLOT_POINTS_INITIAL_GENERATION
    return PlotOutline(
        title=settings.DEFAULT_PLOT_OUTLINE_TITLE,
        protagonist_name=default_protagonist_name,
        genre=settings.CONFIGURED_GENRE,
        setting=settings.CONFIGURED_SETTING_DESCRIPTION,
        theme=settings.CONFIGURED_THEME,
        logline=settings.FILL_IN,
        inciting_incident=settings.FILL_IN,
        central_conflict=settings.FILL_IN,
        stakes=settings.FILL_IN,
        plot_points=[f"{settings.FILL_IN}" for _ in range(num_points)],
        narrative_style=settings.FILL_IN,
        tone=settings.FILL_IN,
        pacing=settings.FILL_IN,
        is_default=True,
        source="default_fallback",
    )


async def bootstrap_plot_outline(
    plot_outline: PlotOutline,
) -> tuple[PlotOutline, dict[str, int] | None]:
    """Fill missing plot fields via LLM."""
    tasks = _build_bootstrap_tasks(plot_outline)
    if not tasks:
        return plot_outline, None

    usage_data = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    results = await asyncio.gather(*tasks.values())
    _apply_bootstrap_results(plot_outline, results, list(tasks.keys()), usage_data)
    return plot_outline, usage_data if usage_data["total_tokens"] > 0 else None
