# data_access/chapter_queries.py
"""Wrapper functions around :class:`ChapterRepository`."""

from __future__ import annotations

from typing import Any

import numpy as np
from structlog import get_logger

from .repository import chapter_repository

logger = get_logger(__name__)


async def load_chapter_count_from_db() -> int:
    """Return the number of chapters stored in the database."""
    return await chapter_repository.load_chapter_count()


async def save_chapter_data_to_db(
    chapter_number: int,
    text: str,
    raw_llm_output: str,
    summary: str | None,
    embedding_array: np.ndarray | None,
    is_provisional: bool = False,
    end_state: dict[str, Any] | None = None,
) -> None:
    """Persist chapter text and metadata using :class:`ChapterRepository`."""
    await chapter_repository.save_chapter_data(
        chapter_number,
        text,
        raw_llm_output,
        summary,
        embedding_array,
        is_provisional,
        end_state,
    )


async def get_chapter_data_from_db(chapter_number: int) -> dict[str, Any] | None:
    """Return stored chapter data for ``chapter_number``."""
    return await chapter_repository.get_chapter_data(chapter_number)


async def get_chapters_data_from_db(
    start_number: int, end_number: int
) -> list[dict[str, Any]]:
    """Return data for chapters ``start_number`` through ``end_number - 1``."""
    return await chapter_repository.get_chapter_data_range(start_number, end_number)


async def get_embedding_from_db(chapter_number: int) -> np.ndarray | None:
    """Return the text embedding for ``chapter_number`` if present."""
    return await chapter_repository.get_embedding(chapter_number)


async def find_similar_chapters_in_db(
    query_embedding: np.ndarray,
    limit: int,
    current_chapter_to_exclude: int | None = None,
    chapter_limit: int | None = None,
) -> list[dict[str, Any]]:
    """Return chapters with embeddings most similar to ``query_embedding``."""
    return await chapter_repository.find_similar_chapters(
        query_embedding,
        limit,
        current_chapter_to_exclude,
        chapter_limit,
    )
