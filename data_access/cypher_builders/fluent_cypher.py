# data_access/cypher_builders/fluent_cypher.py
"""Fluent builder for assembling Cypher queries."""

from __future__ import annotations


class CypherBuilder:
    """Simple fluent interface for constructing Cypher queries."""

    def __init__(self) -> None:
        self._parts: list[str] = []

    def raw(self, text: str) -> CypherBuilder:
        self._parts.append(text.strip())
        return self

    def match(self, clause: str) -> CypherBuilder:
        return self.raw(f"MATCH {clause}")

    def merge(self, clause: str) -> CypherBuilder:
        return self.raw(f"MERGE {clause}")

    def create(self, clause: str) -> CypherBuilder:
        return self.raw(f"CREATE {clause}")

    def set(self, clause: str) -> CypherBuilder:
        return self.raw(f"SET {clause}")

    def where(self, clause: str) -> CypherBuilder:
        return self.raw(f"WHERE {clause}")

    def return_(self, clause: str) -> CypherBuilder:
        return self.raw(f"RETURN {clause}")

    def build(self) -> str:
        return "\n".join(self._parts)
