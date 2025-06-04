# core_db/base_db_manager.py
import logging
import json
from typing import Optional, List, Dict, Any, Tuple, Union
import numpy as np
import asyncio

from neo4j import AsyncGraphDatabase, AsyncSession, AsyncManagedTransaction, AsyncDriver # type: ignore
from neo4j.exceptions import ServiceUnavailable, Neo4jError # type: ignore

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
        
        self.logger = logging.getLogger(__name__)
        self.driver: Optional[AsyncDriver] = None
        self._initialized_flag = True
        self.logger.info("Neo4jManagerSingleton initialized. Call connect() to establish connection.")

    async def connect(self):
        if self.driver:
            self.logger.info("Existing driver instance found. Attempting to close it before creating a new connection.")
            try:
                await self.driver.close()
            except Exception as e_close:
                self.logger.warning(f"Error closing existing driver (it might have been already closed or invalid): {e_close}")
            finally:
                self.driver = None

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
            try:
                await self.driver.close()
                self.logger.info("Neo4j driver closed.")
            except Exception as e:
                self.logger.error(f"Error while closing Neo4j driver: {e}", exc_info=True)
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

    async def _execute_query_tx(self, tx: AsyncManagedTransaction, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        self.logger.debug(f"Executing Cypher query: {query} with params: {parameters}")
        result_cursor = await tx.run(query, parameters)
        return await result_cursor.data()

    async def execute_read_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        await self._ensure_connected()
        async with self.driver.session(database=config.NEO4J_DATABASE) as session: # type: ignore
            return await session.execute_read(self._execute_query_tx, query, parameters)

    async def execute_write_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        await self._ensure_connected()
        async with self.driver.session(database=config.NEO4J_DATABASE) as session: # type: ignore
            return await session.execute_write(self._execute_query_tx, query, parameters)
            
    async def execute_cypher_batch(self, cypher_statements_with_params: List[Tuple[str, Dict[str, Any]]]):
        if not cypher_statements_with_params:
            self.logger.info("execute_cypher_batch: No statements to execute.")
            return

        await self._ensure_connected()
        async with self.driver.session(database=config.NEO4J_DATABASE) as session: # type: ignore
            tx: Optional[AsyncManagedTransaction] = None
            try:
                tx = await session.begin_transaction()
                for query, params in cypher_statements_with_params:
                    self.logger.debug(f"Batch Cypher: {query} with params {params}")
                    await tx.run(query, params) # type: ignore
                await tx.commit() # type: ignore
                self.logger.info(f"Successfully executed batch of {len(cypher_statements_with_params)} Cypher statements.")
            except Exception as e:
                self.logger.error(f"Error in Cypher batch execution: {e}. Rolling back.", exc_info=True)
                if tx is not None:
                    try:
                        # Check if transaction is not already closed and has a rollback method
                        if hasattr(tx, 'rollback') and callable(tx.rollback) and not tx.closed(): # type: ignore
                             await tx.rollback() # type: ignore
                        elif tx.closed(): # type: ignore
                             self.logger.warning("Transaction was already closed, cannot explicitly rollback.")
                        else:
                             self.logger.warning("Transaction object does not have callable rollback or is not in expected state.")
                    except Exception as rb_exc:
                        self.logger.error(f"Exception during explicit transaction rollback: {rb_exc}", exc_info=True)
                raise

    async def create_db_schema(self):
        self.logger.info("Creating/verifying Neo4j indexes and constraints (batch execution)...")
        
        global_entity_name_constraint_name = "entity_name_unique"
        drop_global_entity_name_constraint = f"DROP CONSTRAINT {global_entity_name_constraint_name} IF EXISTS"
        
        worldelement_name_index_name = "worldElement_name_index" 
        drop_worldelement_name_plain_index = f"DROP INDEX {worldelement_name_index_name} IF EXISTS"
        drop_worldelement_name_unique_prop_idx = "DROP INDEX worldelement_name_unique IF EXISTS"
        # Also drop the constraint we are removing
        drop_worldelement_name_constraint = "DROP CONSTRAINT worldElement_name_unique IF EXISTS"


        core_constraints_queries = [
            "CREATE CONSTRAINT novelInfo_id_unique IF NOT EXISTS FOR (n:NovelInfo) REQUIRE n.id IS UNIQUE",
            f"CREATE CONSTRAINT chapter_number_unique IF NOT EXISTS FOR (c:{config.NEO4J_VECTOR_NODE_LABEL}) REQUIRE c.number IS UNIQUE",
            "CREATE CONSTRAINT character_name_unique IF NOT EXISTS FOR (char:Character) REQUIRE char.name IS UNIQUE",
            "CREATE CONSTRAINT worldElement_id_unique IF NOT EXISTS FOR (we:WorldElement) REQUIRE we.id IS UNIQUE",
            # "CREATE CONSTRAINT worldElement_name_unique IF NOT EXISTS FOR (we:WorldElement) REQUIRE we.name IS UNIQUE", # REMOVED - id is sufficient unique key
            "CREATE CONSTRAINT worldContainer_id_unique IF NOT EXISTS FOR (wc:WorldContainer) REQUIRE wc.id IS UNIQUE",
            "CREATE CONSTRAINT trait_name_unique IF NOT EXISTS FOR (t:Trait) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT plotPoint_id_unique IF NOT EXISTS FOR (pp:PlotPoint) REQUIRE pp.id IS UNIQUE",
            "CREATE CONSTRAINT valueNode_value_type_unique IF NOT EXISTS FOR (vn:ValueNode) REQUIRE (vn.value, vn.type) IS UNIQUE",
            "CREATE CONSTRAINT developmentEvent_id_unique IF NOT EXISTS FOR (dev:DevelopmentEvent) REQUIRE dev.id IS UNIQUE",
            "CREATE CONSTRAINT worldElaborationEvent_id_unique IF NOT EXISTS FOR (elab:WorldElaborationEvent) REQUIRE elab.id IS UNIQUE",
        ]
        
        index_queries = [
            "CREATE INDEX plotPoint_sequence IF NOT EXISTS FOR (pp:PlotPoint) ON (pp.sequence)",
            "CREATE INDEX developmentEvent_chapter_updated IF NOT EXISTS FOR (d:DevelopmentEvent) ON (d.chapter_updated)",
            "CREATE INDEX worldElaborationEvent_chapter_updated IF NOT EXISTS FOR (we:WorldElaborationEvent) ON (we.chapter_updated)",
            "CREATE INDEX dynamicRel_chapter_added IF NOT EXISTS FOR ()-[r:DYNAMIC_REL]-() ON (r.chapter_added)",
            "CREATE INDEX dynamicRel_type IF NOT EXISTS FOR ()-[r:DYNAMIC_REL]-() ON (r.type)",
            "CREATE INDEX worldElement_category IF NOT EXISTS FOR (we:WorldElement) ON (we.category)",
            # Add index on WorldElement.name as it's still queried often, just not unique globally
            "CREATE INDEX worldElement_name_property_idx IF NOT EXISTS FOR (we:WorldElement) ON (we.name)",
            f"CREATE INDEX chapter_is_provisional IF NOT EXISTS FOR (c:{config.NEO4J_VECTOR_NODE_LABEL}) ON (c.is_provisional)",
            "CREATE INDEX dynamicRel_is_provisional IF NOT EXISTS FOR ()-[r:DYNAMIC_REL]-() ON (r.is_provisional)",
            "CREATE INDEX entity_is_provisional IF NOT EXISTS FOR (e:Entity) ON (e.is_provisional)", 
        ]

        vector_index_query = f"""
        CREATE VECTOR INDEX {config.NEO4J_VECTOR_INDEX_NAME} IF NOT EXISTS
        FOR (c:{config.NEO4J_VECTOR_NODE_LABEL}) ON (c.{config.NEO4J_VECTOR_PROPERTY_NAME})
        OPTIONS {{indexConfig: {{
            `vector.dimensions`: {config.NEO4J_VECTOR_DIMENSIONS},
            `vector.similarity_function`: '{config.NEO4J_VECTOR_SIMILARITY_FUNCTION}'
        }}}}
        """
        
        pre_drop_statements_with_params: List[Tuple[str, Dict[str, Any]]] = [
            (drop_global_entity_name_constraint, {"constraint_name": global_entity_name_constraint_name}),
            (drop_worldelement_name_plain_index, {"index_name": worldelement_name_index_name}),
            (drop_worldelement_name_unique_prop_idx, {"index_name": "worldelement_name_unique"}),
            (drop_worldelement_name_constraint, {}), # Drop the problematic constraint
        ]
        
        self.logger.info("Executing pre-emptive drop statements for potentially conflicting schema...")
        for query, params in pre_drop_statements_with_params:
            try:
                await self.execute_write_query(query, params if params else {}) # Ensure params is not None
                self.logger.info(f"Successfully executed pre-emptive drop: '{query}'")
            except Neo4jError as e_drop:
                log_message = (
                    f"Could not execute pre-emptive drop '{query}' (this may be fine if it didn't exist or permission issue): {e_drop}. "
                )
                if query == drop_global_entity_name_constraint:
                     log_message += (
                        f"If you are encountering 'ConstraintValidationFailed' errors related to 'Entity.name', "
                        f"ensure that no global uniqueness constraint exists on (:Entity {{name}}), or that if it does, "
                        f"its name is '{global_entity_name_constraint_name}' so this script can attempt to drop it. "
                        f"Otherwise, you may need to manually drop it using 'DROP CONSTRAINT YourConstraintName'. "
                        f"You can find constraint names with 'SHOW CONSTRAINTS'."
                    )
                     self.logger.error(log_message) 
                elif query == drop_worldelement_name_constraint:
                    self.logger.info(f"Successfully dropped (or it didn't exist) 'worldElement_name_unique' constraint: {query}")
                else:
                    self.logger.warning(log_message)
            except Exception as e_unexpected_drop:
                self.logger.error(f"Unexpected error during pre-emptive drop '{query}': {e_unexpected_drop}", exc_info=True)


        all_schema_ops_queries = core_constraints_queries + index_queries + [vector_index_query]
        
        schema_statements_with_params: List[Tuple[str, Dict[str, Any]]] = [(query, {}) for query in all_schema_ops_queries]

        try:
            await self.execute_cypher_batch(schema_statements_with_params)
            self.logger.info(f"Successfully executed batch of {len(schema_statements_with_params)} schema operations.")
        except Exception as e:
            self.logger.error(f"Error during batch schema operation execution: {e}. Some schema elements might not be created/verified.", exc_info=True)
            self.logger.warning("Attempting to apply schema operations individually as a fallback.")
            for query_text in all_schema_ops_queries:
                try:
                    await self.execute_write_query(query_text)
                    self.logger.info(f"Fallback: Successfully applied schema operation: '{query_text[:100]}...'")
                except Exception as individual_e:
                    self.logger.warning(f"Fallback: Failed to apply schema operation '{query_text[:100]}...': {individual_e}")

        self.logger.info("Neo4j schema (indexes, constraints, vector index) verification process complete.")

    def embedding_to_list(self, embedding: Optional[np.ndarray]) -> Optional[List[float]]:
        if embedding is None: return None
        if not isinstance(embedding, np.ndarray):
            self.logger.warning(f"Attempting to convert non-numpy array to list for Neo4j: {type(embedding)}")
            if hasattr(embedding, 'tolist'): return embedding.tolist() # type: ignore
            self.logger.error(f"Cannot convert type {type(embedding)} to list for Neo4j.")
            return None
        return embedding.astype(np.float32).tolist()

    def list_to_embedding(self, embedding_list: Optional[List[Union[float, int]]]) -> Optional[np.ndarray]:
        if embedding_list is None: return None
        try:
            return np.array(embedding_list, dtype=config.EMBEDDING_DTYPE)
        except Exception as e:
            self.logger.error(f"Error converting list to numpy embedding: {e}", exc_info=True)
            return None

neo4j_manager = Neo4jManagerSingleton()