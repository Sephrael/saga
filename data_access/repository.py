# data_access/repository.py
"""Repository abstractions for Neo4j access."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import numpy as np
from config import settings
from core.db_manager import Neo4jManagerSingleton, neo4j_manager
from core.llm_interface import llm_service

__all__ = [
    "BaseRepository",
    "ChapterRepository",
    "chapter_repository",
]


class BaseRepository:
    """Base repository providing simple database helpers."""

    def __init__(self, db: Neo4jManagerSingleton | None = None) -> None:
        self.db = db or neo4j_manager

    async def read(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a read query."""
        return await self.db.execute_read_query(query, parameters)

    async def write(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a write query."""
        return await self.db.execute_write_query(query, parameters)

    async def batch(self, statements: Sequence[tuple[str, dict[str, Any]]]) -> None:
        """Execute a batch of Cypher statements."""
        await self.db.execute_cypher_batch(list(statements))


class ChapterRepository(BaseRepository):
    """Data access methods for chapters."""

    async def load_chapter_count(self) -> int:
        """Return the number of chapters stored in the database."""
        query = f"MATCH (c:{settings.NEO4J_VECTOR_NODE_LABEL}) RETURN count(c) AS chapter_count"
        try:
            result = await self.read(query)
            return result[0]["chapter_count"] if result and result[0] else 0
        except Exception:
            return 0

    async def save_chapter_data(
        self,
        chapter_number: int,
        text: str,
        raw_llm_output: str,
        summary: str | None,
        embedding_array: np.ndarray | None,
        is_provisional: bool = False,
        end_state: dict[str, Any] | None = None,
    ) -> None:
        """Persist chapter text and metadata to Neo4j."""
        if chapter_number < 0:
            return

        if embedding_array is None:
            embedding_array = await llm_service.async_get_embedding(text)
        embedding_list = self.db.embedding_to_list(embedding_array)
        query = f"""
        MERGE (c:{settings.NEO4J_VECTOR_NODE_LABEL} {{number: $chapter_number_param}})
        SET c.text = $text_param,
            c.raw_llm_output = $raw_llm_output_param,
            c.summary = $summary_param,
            c.is_provisional = $is_provisional_param,
            c.end_state_json = $end_state_param,
            c.{settings.NEO4J_VECTOR_PROPERTY_NAME} = $text_embedding_param,
            c.last_updated = timestamp()
        """
        parameters = {
            "chapter_number_param": chapter_number,
            "text_param": text,
            "raw_llm_output_param": raw_llm_output,
            "summary_param": summary if summary is not None else "",
            "is_provisional_param": is_provisional,
            "end_state_param": json.dumps(end_state) if end_state is not None else "",
            "text_embedding_param": embedding_list,
        }
        await self.write(query, parameters)

    async def get_chapter_data(self, chapter_number: int) -> dict[str, Any] | None:
        """Retrieve stored chapter data."""
        if chapter_number < 0:
            return None
        query = f"""
        MATCH (c:{settings.NEO4J_VECTOR_NODE_LABEL} {{number: $chapter_number_param}})
        RETURN c.text AS text,
               c.raw_llm_output AS raw_llm_output,
               c.summary AS summary,
               c.is_provisional AS is_provisional,
               c.end_state_json AS end_state_json
        """
        try:
            result = await self.read(query, {"chapter_number_param": chapter_number})
            if result and result[0]:
                return {
                    "text": result[0].get("text"),
                    "summary": result[0].get("summary"),
                    "is_provisional": result[0].get("is_provisional", False),
                    "raw_llm_output": result[0].get("raw_llm_output"),
                    "end_state_json": result[0].get("end_state_json"),
                }
        except Exception:
            return None
        return None

    async def get_embedding(self, chapter_number: int) -> np.ndarray | None:
        """Return the text embedding for a chapter."""
        if chapter_number < 0:
            return None
        query = f"""
        MATCH (c:{settings.NEO4J_VECTOR_NODE_LABEL} {{number: $chapter_number_param}})
        WHERE c.{settings.NEO4J_VECTOR_PROPERTY_NAME} IS NOT NULL
        RETURN c.{settings.NEO4J_VECTOR_PROPERTY_NAME} AS text_embedding
        """
        try:
            result = await self.read(query, {"chapter_number_param": chapter_number})
            if result and result[0] and result[0].get("text_embedding"):
                embedding_list = result[0]["text_embedding"]
                return self.db.list_to_embedding(embedding_list)
        except Exception:
            return None
        return None

    async def find_similar_chapters(
        self,
        query_embedding: np.ndarray,
        limit: int,
        current_chapter_to_exclude: int | None = None,
        chapter_limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return chapters with embeddings most similar to the query vector."""
        if query_embedding is None or query_embedding.size == 0:
            return []
        query_embedding_list = self.db.embedding_to_list(query_embedding)
        conditions: list[str] = []
        params_dict: dict[str, Any] = {
            "index_name_param": settings.NEO4J_VECTOR_INDEX_NAME,
            "limit_param": limit + 1
            if current_chapter_to_exclude is not None
            else limit,
            "queryVector_param": query_embedding_list,
        }
        if current_chapter_to_exclude is not None:
            conditions.append("c.number <> $current_chapter_to_exclude_param")
            params_dict["current_chapter_to_exclude_param"] = current_chapter_to_exclude
        if chapter_limit is not None:
            conditions.append("c.number <= $chapter_limit_param")
            params_dict["chapter_limit_param"] = chapter_limit
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        similarity_query = f"""
        CALL db.index.vector.queryNodes($index_name_param, $limit_param, $queryVector_param)
        YIELD node AS c, score
        {where_clause}
        RETURN c.number AS chapter_number,
               c.summary AS summary,
               c.text AS text,
               c.is_provisional AS is_provisional,
               score
        ORDER BY score DESC
        """
        results = await self.read(similarity_query, params_dict)
        similar: list[dict[str, Any]] = []
        for record in results:
            if len(similar) >= limit:
                break
            similar.append(
                {
                    "chapter_number": record.get("chapter_number"),
                    "summary": record.get("summary"),
                    "text": record.get("text"),
                    "is_provisional": record.get("is_provisional", False),
                    "score": record.get("score"),
                }
            )
        return similar


chapter_repository = ChapterRepository()
