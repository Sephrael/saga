import logging
from typing import Any

from config import settings

logger = logging.getLogger(__name__)


def get_plot_point_info(
    plot_outline: dict[str, Any], chapter_number: int
) -> tuple[str | None, int]:
    """Return plot point text and index for the chapter."""
    plot_points = plot_outline.get("plot_points", [])
    if not isinstance(plot_points, list) or not plot_points or chapter_number <= 0:
        logger.error(
            "No plot points available or invalid chapter number (%s).",
            chapter_number,
        )
        return None, -1

    plot_point_index = chapter_number - 1
    if 0 <= plot_point_index < len(plot_points):
        plot_point_item = plot_points[plot_point_index]
        plot_point_text = (
            plot_point_item.get("description")
            if isinstance(plot_point_item, dict)
            else str(plot_point_item)
        )
        if isinstance(plot_point_text, str) and plot_point_text.strip():
            return plot_point_text, plot_point_index
        logger.warning(
            "Plot point at index %s for chapter %s is empty or invalid. Using placeholder.",
            plot_point_index,
            chapter_number,
        )
        return settings.FILL_IN, plot_point_index

    logger.error(
        "Plot point index %s is out of bounds for plot_points list (len: %s) for chapter %s.",
        plot_point_index,
        len(plot_points),
        chapter_number,
    )
    return None, -1
