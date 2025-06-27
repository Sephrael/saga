# core_db/base_db_manager.py
from typing import Any

import numpy as np
import structlog
from config import settings
from kg_constants import NODE_LABELS, RELATIONSHIP_TYPES
from neo4j import (  # type: ignore
    AsyncDriver,
    AsyncGraphDatabase,
    AsyncManagedTransaction,
)
from neo4j.exceptions import ServiceUnavailable  # type: ignore

logger = structlog.get_logger(__name__)


class Neo4jManagerSingleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._initialized_flag = False
        return cls._instance

    def __init__(self):
        if self._initialized_flag:
            return

        self.logger = structlog.get_logger(__name__)
        self.driver: AsyncDriver | None = None
        self._initialized_flag = True
        self.logger.info(
            "Neo4jManagerSingleton initialized. Call connect() to establish connection."
        )

    async def __aenter__(self) -> "Neo4jManagerSingleton":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def connect(self):
        if self.driver:
            self.logger.info(
                "Existing driver instance found. Attempting to close it before creating a new connection."
            )
            try:
                await self.driver.close()
            except Exception as e_close:
                self.logger.warning(
                    f"Error closing existing driver (it might have been already closed or invalid): {e_close}"
                )
            finally:
                self.driver = None

        try:
            self.driver = AsyncGraphDatabase.driver(
                settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
            )
            await self.driver.verify_connectivity()
            self.logger.info(f"Successfully connected to Neo4j at {settings.NEO4J_URI}")
        except ServiceUnavailable as e:
            self.logger.critical(
                f"Neo4j connection failed: {e}. Ensure the Neo4j database is running and accessible."
            )
            self.driver = None
            raise
        except Exception as e:
            self.logger.critical(
                f"Unexpected error during Neo4j connection: {e}", exc_info=True
            )
            self.driver = None
            raise

    async def close(self):
        if self.driver:
            try:
                await self.driver.close()
                self.logger.info("Neo4j driver closed.")
            except Exception as e:
                self.logger.error(
                    f"Error while closing Neo4j driver: {e}", exc_info=True
                )
            finally:
                self.driver = None
        else:
            self.logger.info("No active Neo4j driver to close (driver was None).")

    async def _ensure_connected(self):
        if self.driver is None:
            self.logger.info("Driver is None, attempting to connect.")
            await self.connect()

        if self.driver is None:
            raise ConnectionError("Neo4j driver not initialized or connection failed.")

    async def _execute_query_tx(
        self,
        tx: AsyncManagedTransaction,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        self.logger.debug(f"Executing Cypher query: {query} with params: {parameters}")
        result_cursor = await tx.run(query, parameters)
        return await result_cursor.data()

    async def execute_read_query(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        await self._ensure_connected()
        async with self.driver.session(database=settings.NEO4J_DATABASE) as session:  # type: ignore
            return await session.execute_read(self._execute_query_tx, query, parameters)

    async def execute_write_query(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        await self._ensure_connected()
        async with self.driver.session(database=settings.NEO4J_DATABASE) as session:  # type: ignore
            return await session.execute_write(
                self._execute_query_tx, query, parameters
            )

    async def execute_cypher_batch(
        self, cypher_statements_with_params: list[tuple[str, dict[str, Any]]]
    ):
        if not cypher_statements_with_params:
            self.logger.info("execute_cypher_batch: No statements to execute.")
            return

        await self._ensure_connected()
        async with self.driver.session(database=settings.NEO4J_DATABASE) as session:  # type: ignore
            tx: AsyncManagedTransaction | None = None
            try:
                tx = await session.begin_transaction()
                for query, params in cypher_statements_with_params:
                    self.logger.debug(f"Batch Cypher: {query} with params {params}")
                    await tx.run(query, params)  # type: ignore
                await tx.commit()  # type: ignore
                self.logger.info(
                    f"Successfully executed batch of {len(cypher_statements_with_params)} Cypher statements."
                )
            except Exception as e:
                self.logger.error(
                    f"Error in Cypher batch execution: {e}. Rolling back.",
                    exc_info=True,
                )
                if tx is not None:
                    try:
                        # Check if transaction is not already closed and has a rollback method
                        if (
                            hasattr(tx, "rollback")
                            and callable(tx.rollback)
                            and not tx.closed()
                        ):  # type: ignore
                            await tx.rollback()  # type: ignore
                        elif tx.closed():  # type: ignore
                            self.logger.warning(
                                "Transaction was already closed, cannot explicitly rollback."
                            )
                        else:
                            self.logger.warning(
                                "Transaction object does not have callable rollback or is not in expected state."
                            )
                    except Exception as rb_exc:
                        self.logger.error(
                            f"Exception during explicit transaction rollback: {rb_exc}",
                            exc_info=True,
                        )
                raise

    async def _run_schema_operations(
        self, queries: list[str], description: str
    ) -> None:
        ops: list[tuple[str, dict[str, Any]]] = [(q, {}) for q in queries]
        try:
            await self.execute_cypher_batch(ops)
            self.logger.info(f"Successfully executed {description} batch.")
        except Exception as e:
            self.logger.error(
                f"Error during {description} batch execution: {e}", exc_info=True
            )
            self.logger.warning(
                "Attempting to apply operations individually as a fallback."
            )
            for query_text in queries:
                try:
                    await self.execute_write_query(query_text)
                    self.logger.info(
                        f"Fallback: Successfully applied operation: '{query_text[:100]}...'"
                    )
                except Exception as individual_e:
                    self.logger.warning(
                        f"Fallback: Failed to apply operation '{query_text[:100]}...': {individual_e}"
                    )

    async def _create_constraints(self) -> None:
        queries = [
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT novelInfo_id_unique IF NOT EXISTS FOR (n:NovelInfo) REQUIRE n.id IS UNIQUE",
            f"CREATE CONSTRAINT chapter_number_unique IF NOT EXISTS FOR (c:{settings.NEO4J_VECTOR_NODE_LABEL}) REQUIRE c.number IS UNIQUE",
            "CREATE CONSTRAINT character_name_unique IF NOT EXISTS FOR (char:Character) REQUIRE char.name IS UNIQUE",
            "CREATE CONSTRAINT worldElement_id_unique IF NOT EXISTS FOR (we:WorldElement) REQUIRE we.id IS UNIQUE",
            "CREATE CONSTRAINT worldContainer_id_unique IF NOT EXISTS FOR (wc:WorldContainer) REQUIRE wc.id IS UNIQUE",
            "CREATE CONSTRAINT trait_name_unique IF NOT EXISTS FOR (t:Trait) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT plotPoint_id_unique IF NOT EXISTS FOR (pp:PlotPoint) REQUIRE pp.id IS UNIQUE",
            "CREATE CONSTRAINT valueNode_value_type_unique IF NOT EXISTS FOR (vn:ValueNode) REQUIRE (vn.value, vn.type) IS UNIQUE",
            "CREATE CONSTRAINT developmentEvent_id_unique IF NOT EXISTS FOR (dev:DevelopmentEvent) REQUIRE dev.id IS UNIQUE",
            "CREATE CONSTRAINT worldElaborationEvent_id_unique IF NOT EXISTS FOR (elab:WorldElaborationEvent) REQUIRE elab.id IS UNIQUE",
        ]
        await self._run_schema_operations(queries, "constraint")

    async def _create_indexes(self) -> None:
        queries = [
            "CREATE INDEX entity_name_property_idx IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_is_provisional_idx IF NOT EXISTS FOR (e:Entity) ON (e.is_provisional)",
            "CREATE INDEX entity_is_deleted_idx IF NOT EXISTS FOR (e:Entity) ON (e.is_deleted)",
            "CREATE INDEX plotPoint_sequence IF NOT EXISTS FOR (pp:PlotPoint) ON (pp.sequence)",
            "CREATE INDEX developmentEvent_chapter_updated IF NOT EXISTS FOR (d:DevelopmentEvent) ON (d.chapter_updated)",
            "CREATE INDEX worldElaborationEvent_chapter_updated IF NOT EXISTS FOR (we:WorldElaborationEvent) ON (we.chapter_updated)",
            "CREATE INDEX dynamicRel_chapter_added IF NOT EXISTS FOR ()-[r:DYNAMIC_REL]-() ON (r.chapter_added)",
            "CREATE INDEX dynamicRel_type IF NOT EXISTS FOR ()-[r:DYNAMIC_REL]-() ON (r.type)",
            "CREATE INDEX dynamicRel_is_provisional IF NOT EXISTS FOR ()-[r:DYNAMIC_REL]-() ON (r.is_provisional)",
            "CREATE INDEX dynamicRel_source_profile_managed IF NOT EXISTS FOR ()-[r:DYNAMIC_REL]-() ON (r.source_profile_managed)",
            "CREATE INDEX worldElement_category IF NOT EXISTS FOR (we:WorldElement) ON (we.category)",
            "CREATE INDEX worldElement_name_property_idx IF NOT EXISTS FOR (we:WorldElement) ON (we.name)",
            "CREATE INDEX entity_description_idx IF NOT EXISTS FOR (e:Entity) ON (e.description)",
            "CREATE INDEX entity_created_chapter_idx IF NOT EXISTS FOR (e:Entity) ON (e.created_chapter)",
            "CREATE INDEX plotPoint_description_idx IF NOT EXISTS FOR (pp:PlotPoint) ON (pp.description)",
            "CREATE INDEX valueNode_value_idx IF NOT EXISTS FOR (vn:ValueNode) ON (vn.value)",
            f"CREATE INDEX chapter_is_provisional IF NOT EXISTS FOR (c:{settings.NEO4J_VECTOR_NODE_LABEL}) ON (c.is_provisional)",
        ]
        await self._run_schema_operations(queries, "index")

    async def _create_vector_index(self) -> None:
        query = f"""
        CREATE VECTOR INDEX {settings.NEO4J_VECTOR_INDEX_NAME} IF NOT EXISTS
        FOR (c:{settings.NEO4J_VECTOR_NODE_LABEL}) ON (c.{settings.NEO4J_VECTOR_PROPERTY_NAME})
        OPTIONS {{indexConfig: {{
            `vector.dimensions`: {settings.NEO4J_VECTOR_DIMENSIONS},
            `vector.similarity_function`: '{settings.NEO4J_VECTOR_SIMILARITY_FUNCTION}'
        }}}}
        """
        await self._run_schema_operations([query], "vector index")

    async def create_db_schema(self) -> None:
        """Create and verify Neo4j indexes and constraints.

        The method issues idempotent `CREATE ... IF NOT EXISTS` statements for
        all core indexes, constraints, and relationship type tokens. Any batch
        failures fall back to executing operations individually.
        """
        self.logger.info(
            "Creating/verifying Neo4j schema elements (batch execution)..."
        )

        await self._create_constraints()
        await self._create_indexes()
        await self._create_vector_index()

        relationship_type_queries = [
            (
                f"CREATE (a:__RelTypePlaceholder)-[:{rel_type}]->"
                f"(b:__RelTypePlaceholder) WITH a,b DETACH DELETE a,b"
            )
            for rel_type in RELATIONSHIP_TYPES
        ]
        node_label_queries = [
            f"CREATE (a:`{label}`) WITH a DELETE a" for label in NODE_LABELS
        ]

        await self._run_schema_operations(
            relationship_type_queries + node_label_queries, "schema token"
        )

        self.logger.info(
            "Neo4j schema (indexes, constraints, labels, relationship types, vector index) verification process complete."
        )

    def embedding_to_list(self, embedding: np.ndarray | None) -> list[float] | None:
        if embedding is None:
            return None
        if not isinstance(embedding, np.ndarray):
            self.logger.warning(
                f"Attempting to convert non-numpy array to list for Neo4j: {type(embedding)}"
            )
            if hasattr(embedding, "tolist"):
                return embedding.tolist()  # type: ignore
            self.logger.error(
                f"Cannot convert type {type(embedding)} to list for Neo4j."
            )
            return None
        return embedding.astype(np.float32).tolist()

    def list_to_embedding(
        self, embedding_list: list[float | int] | None
    ) -> np.ndarray | None:
        if embedding_list is None:
            return None
        try:
            return np.array(embedding_list, dtype=settings.EMBEDDING_DTYPE)
        except Exception as e:
            self.logger.error(
                f"Error converting list to numpy embedding: {e}", exc_info=True
            )
            return None


neo4j_manager = Neo4jManagerSingleton()
