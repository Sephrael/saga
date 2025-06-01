# core_db/base_db_manager.py
import logging
import json
from typing import Optional, List, Dict, Any, Tuple, Union
import numpy as np
import asyncio

from neo4j import AsyncGraphDatabase, AsyncSession, AsyncManagedTransaction # type: ignore
from neo4j.exceptions import ServiceUnavailable # type: ignore

import config

logger = logging.getLogger(__name__)

class Neo4jManagerSingleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Neo4jManagerSingleton, cls).__new__(cls)
            cls._instance._initialized_flag = False
        return cls._instance

    def __init__(self):
        if self._initialized_flag:
            return
        
        self.logger = logging.getLogger(__name__) # Use the module's logger
        self.driver: Optional[AsyncGraphDatabase] = None
        self._initialized_flag = True
        self.logger.info("Neo4jManagerSingleton initialized. Call connect() to establish connection.")

    async def connect(self):
        if self.driver is None:
            try:
                self.driver = AsyncGraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
                await self.driver.verify_connectivity()
                self.logger.info(f"Successfully connected to Neo4j at {config.NEO4J_URI}")
            except ServiceUnavailable as e:
                self.logger.critical(f"Neo4j connection failed: {e}. Ensure the Neo4j database is running and accessible.")
                self.driver = None
                raise
            except Exception as e:
                self.logger.critical(f"Unexpected error during Neo4j connection: {e}", exc_info=True)
                self.driver = None
                raise

    async def close(self):
        if self.driver:
            await self.driver.close()
            self.driver = None
            self.logger.info("Neo4j driver closed.")

    async def _execute_query_tx(self, tx: AsyncManagedTransaction, query: str, parameters: Optional[Dict] = None):
        self.logger.debug(f"Executing Cypher query: {query} with params: {parameters}")
        result = await tx.run(query, parameters)
        return await result.data()

    async def execute_read_query(self, query: str, parameters: Optional[Dict] = None):
        if self.driver is None: await self.connect()
        if self.driver is None: raise ConnectionError("Neo4j driver not initialized.")
        async with self.driver.session(database=config.NEO4J_DATABASE) as session: # type: ignore
            return await session.execute_read(self._execute_query_tx, query, parameters)

    async def execute_write_query(self, query: str, parameters: Optional[Dict] = None):
        if self.driver is None: await self.connect()
        if self.driver is None: raise ConnectionError("Neo4j driver not initialized.")
        async with self.driver.session(database=config.NEO4J_DATABASE) as session: # type: ignore
            return await session.execute_write(self._execute_query_tx, query, parameters)
            
    async def execute_cypher_batch(self, cypher_statements_with_params: List[Tuple[str, Dict[str, Any]]]):
        if not cypher_statements_with_params:
            self.logger.info("execute_cypher_batch: No statements to execute.")
            return

        if self.driver is None: await self.connect()
        if self.driver is None: raise ConnectionError("Neo4j driver not initialized.")

        async with self.driver.session(database=config.NEO4J_DATABASE) as session: # type: AsyncSession # type: ignore
            tx: AsyncManagedTransaction = await session.begin_transaction()
            try:
                for query, params in cypher_statements_with_params:
                    self.logger.debug(f"Batch Cypher: {query} with params {params}")
                    await tx.run(query, params)
                await tx.commit()
                self.logger.info(f"Successfully executed batch of {len(cypher_statements_with_params)} Cypher statements.")
            except Exception as e:
                self.logger.error(f"Error in Cypher batch execution: {e}. Rolling back.", exc_info=True)
                if not tx.closed():
                    await tx.rollback()
                raise

    async def create_db_schema(self):
        self.logger.info("Creating/verifying Neo4j indexes and constraints, including vector index (using batch execution)...")
        
        core_constraints_queries = [
            "CREATE CONSTRAINT novelInfo_id_unique IF NOT EXISTS FOR (n:NovelInfo) REQUIRE n.id IS UNIQUE",
            f"CREATE CONSTRAINT chapter_number_unique IF NOT EXISTS FOR (c:{config.NEO4J_VECTOR_NODE_LABEL}) REQUIRE c.number IS UNIQUE",
            "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE CONSTRAINT character_name_unique IF NOT EXISTS FOR (char:Character) REQUIRE char.name IS UNIQUE",
            "CREATE CONSTRAINT worldElement_id_unique IF NOT EXISTS FOR (we:WorldElement) REQUIRE we.id IS UNIQUE",
            "CREATE CONSTRAINT worldContainer_id_unique IF NOT EXISTS FOR (wc:WorldContainer) REQUIRE wc.id IS UNIQUE",
            "CREATE CONSTRAINT trait_name_unique IF NOT EXISTS FOR (t:Trait) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT plotPoint_id_unique IF NOT EXISTS FOR (pp:PlotPoint) REQUIRE pp.id IS UNIQUE",
            "CREATE CONSTRAINT valueNode_value_type_unique IF NOT EXISTS FOR (vn:ValueNode) REQUIRE (vn.value, vn.type) IS UNIQUE",
        ]
        index_queries = [
            "CREATE INDEX plotPoint_sequence IF NOT EXISTS FOR (pp:PlotPoint) ON (pp.sequence)",
            "CREATE INDEX statusEvent_chapter_updated IF NOT EXISTS FOR (s:StatusEvent) ON (s.chapter_updated)",
            "CREATE INDEX developmentEvent_chapter_updated IF NOT EXISTS FOR (d:DevelopmentEvent) ON (d.chapter_updated)",
            "CREATE INDEX worldElaborationEvent_chapter_updated IF NOT EXISTS FOR (we:WorldElaborationEvent) ON (we.chapter_updated)",
            "CREATE INDEX dynamicRel_chapter_added IF NOT EXISTS FOR ()-[r:DYNAMIC_REL]-() ON (r.chapter_added)",
            "CREATE INDEX dynamicRel_type IF NOT EXISTS FOR ()-[r:DYNAMIC_REL]-() ON (r.type)",
            "CREATE INDEX worldElement_category IF NOT EXISTS FOR (we:WorldElement) ON (we.category)",
            "CREATE INDEX worldElement_name IF NOT EXISTS FOR (we:WorldElement) ON (we.name)",
            f"CREATE INDEX chapter_is_provisional IF NOT EXISTS FOR (c:{config.NEO4J_VECTOR_NODE_LABEL}) ON (c.is_provisional)",
            "CREATE INDEX dynamicRel_is_provisional IF NOT EXISTS FOR ()-[r:DYNAMIC_REL]-() ON (r.is_provisional)",
        ]

        vector_index_query = f"""
        CREATE VECTOR INDEX {config.NEO4J_VECTOR_INDEX_NAME} IF NOT EXISTS
        FOR (c:{config.NEO4J_VECTOR_NODE_LABEL}) ON (c.{config.NEO4J_VECTOR_PROPERTY_NAME})
        OPTIONS {{indexConfig: {{
            `vector.dimensions`: {config.NEO4J_VECTOR_DIMENSIONS},
            `vector.similarity_function`: '{config.NEO4J_VECTOR_SIMILARITY_FUNCTION}'
        }}}}
        """
        
        all_schema_ops_queries = core_constraints_queries + index_queries + [vector_index_query]
        
        # Prepare for batch execution: List of (query, params_dict) tuples
        # For schema operations, params_dict is usually empty.
        schema_statements_with_params: List[Tuple[str, Dict[str, Any]]] = [
            (query, {}) for query in all_schema_ops_queries
        ]

        try:
            await self.execute_cypher_batch(schema_statements_with_params)
            self.logger.info(f"Successfully executed batch of {len(schema_statements_with_params)} schema operations (indexes, constraints, vector index).")
        except Exception as e:
            # If batch fails, it's hard to know which one failed without more granular error reporting from the driver/DB for batch.
            # We log the general error. Individual retries or checks might be needed for robustness in a production setting.
            self.logger.error(f"Error during batch schema operation execution: {e}. Some schema elements might not be created/verified.", exc_info=True)
            # Fallback or individual attempts could be added here if critical
            self.logger.warning(
                "Attempting to apply schema operations individually as a fallback due to batch failure. "
                "This may result in some operations failing if they were the cause or if they depend on failed ones."
            )
            for query in all_schema_ops_queries:
                try:
                    await self.execute_write_query(query) # No params needed for these
                    self.logger.info(f"Fallback: Successfully applied schema operation: '{query[:100]}...'")
                except Exception as individual_e:
                    self.logger.warning(f"Fallback: Failed to apply schema operation '{query[:100]}...': {individual_e} (This might be okay if it already exists or due to initial batch failure).")

        self.logger.info("Neo4j schema (indexes, constraints, vector index) verification process complete.")


    def embedding_to_list(self, embedding: Optional[np.ndarray]) -> Optional[List[float]]:
        if embedding is None:
            return None
        if not isinstance(embedding, np.ndarray):
            self.logger.warning(f"Attempting to convert non-numpy array to list for Neo4j: {type(embedding)}")
            if hasattr(embedding, 'tolist'):
                return embedding.tolist() # type: ignore
            return None
        return embedding.astype(np.float32).tolist()

    def list_to_embedding(self, embedding_list: Optional[List[Union[float, int]]]) -> Optional[np.ndarray]:
        if embedding_list is None:
            return None
        try:
            return np.array(embedding_list, dtype=config.EMBEDDING_DTYPE)
        except Exception as e:
            self.logger.error(f"Error converting list to numpy embedding: {e}", exc_info=True)
            return None

neo4j_manager = Neo4jManagerSingleton()