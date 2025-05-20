# state_manager.py
import logging
import json
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import base64 # For encoding/decoding embeddings
import asyncio

# Neo4j specific imports
from neo4j import AsyncGraphDatabase, AsyncSession, AsyncManagedTransaction # type: ignore
from neo4j.exceptions import ServiceUnavailable, ClientError # type: ignore

import config # For NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, EMBEDDING_DTYPE, etc.

logger = logging.getLogger(__name__)

class state_managerSingleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(state_managerSingleton, cls).__new__(cls)
            cls._instance._initialized_flag = False
        return cls._instance

    def __init__(self):
        if self._initialized_flag:
            return
        
        self.logger = logging.getLogger(__name__)
        self.driver: Optional[AsyncGraphDatabase] = None
        self._initialized_flag = True
        self.logger.info("Neo4j state_managerSingleton initialized. Call connect() to establish connection.")

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

    async def _execute_query(self, query: str, parameters: Optional[Dict] = None, write: bool = False):
        if self.driver is None:
            await self.connect() # Attempt to connect if not already
            if self.driver is None:
                self.logger.error("Neo4j driver not initialized. Cannot execute query.")
                return None

        async with self.driver.session() as session: # type: AsyncSession
            try:
                if write:
                    return await session.write_transaction(self._run_query_in_tx, query, parameters)
                else:
                    return await session.read_transaction(self._run_query_in_tx, query, parameters)
            except ClientError as e:
                self.logger.error(f"Neo4j Cypher error: {e.code} - {e.message}. Query: {query}, Params: {parameters}")
                raise
            except Exception as e:
                self.logger.error(f"Error executing Neo4j query: {e}. Query: {query}, Params: {parameters}", exc_info=True)
                raise

    @staticmethod
    async def _run_query_in_tx(tx: AsyncManagedTransaction, query: str, parameters: Optional[Dict] = None):
        result = await tx.run(query, parameters)
        return await result.data()


    async def create_db_and_tables(self):
        # In Neo4j, this is more about creating indexes and constraints
        self.logger.info("Creating/verifying Neo4j indexes and constraints...")
        queries = [
            "CREATE CONSTRAINT IF NOT EXISTS ON (c:Chapter) ASSERT c.number IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS ON (e:Embedding) ASSERT e.chapter_number IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS ON (e:Entity) ASSERT e.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS ON (n:NovelOutline) ASSERT n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS ON (cp:CharacterProfiles) ASSERT cp.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS ON (wb:WorldBuilding) ASSERT wb.id IS UNIQUE",
            "CREATE INDEX IF NOT EXISTS FOR ()-[r:RELATIONSHIP]-() ON (r.chapter_added)"
        ]
        for query in queries:
            try:
                await self._execute_query(query, write=True)
            except Exception as e:
                self.logger.warning(f"Failed to create constraint/index for '{query}': {e}")
        self.logger.info("Neo4j indexes and constraints verified.")

    def _serialize_embedding(self, embedding: np.ndarray) -> Tuple[str, str, str]:
        embedding_to_save = embedding.astype(config.EMBEDDING_DTYPE)
        if embedding_to_save.ndim == 0:
             embedding_to_save = embedding_to_save.reshape(1)
        
        # Convert to bytes, then base64 encode for storage in Neo4j string property
        return base64.b64encode(embedding_to_save.tobytes()).decode('utf-8'), \
               str(embedding_to_save.dtype), \
               json.dumps(list(embedding_to_save.shape))

    def _deserialize_embedding(self, b64_blob: str, dtype_str: str, shape_str: str) -> Optional[np.ndarray]:
        try:
            blob_bytes = base64.b64decode(b64_blob)
            shape = tuple(json.loads(shape_str))
            dtype = np.dtype(dtype_str)
            return np.frombuffer(blob_bytes, dtype=dtype).reshape(shape)
        except Exception as e:
            self.logger.error(f"Error deserializing embedding (shape: {shape_str}, dtype: {dtype_str}): {e}", exc_info=True)
            return None

    async def save_plot_outline(self, plot_data: Dict[str, Any]) -> bool:
        json_value = json.dumps(plot_data, ensure_ascii=False)
        query = """
        MERGE (n:NovelOutline {id: 'main_plot'})
        SET n.data = $json_value
        RETURN n.id
        """
        try:
            result = await self._execute_query(query, {"json_value": json_value}, write=True)
            self.logger.info("Saved plot outline to Neo4j.")
            return result is not None
        except Exception:
            return False

    async def get_plot_outline(self) -> Dict[str, Any]:
        query = """
        MATCH (n:NovelOutline {id: 'main_plot'})
        RETURN n.data AS data
        """
        try:
            result = await self._execute_query(query, write=False)
            if result and result[0] and result[0]["data"]:
                return json.loads(result[0]["data"])
            return {}
        except Exception:
            return {}

    async def save_character_profiles(self, profiles_data: Dict[str, Any]) -> bool:
        json_data = json.dumps(profiles_data, ensure_ascii=False)
        query = """
        MERGE (cp:CharacterProfiles {id: '_all_profiles_'})
        SET cp.data = $json_data
        RETURN cp.id
        """
        try:
            result = await self._execute_query(query, {"json_data": json_data}, write=True)
            self.logger.info("Saved character profiles to Neo4j.")
            return result is not None
        except Exception:
            return False

    async def get_character_profiles(self) -> Dict[str, Any]:
        query = """
        MATCH (cp:CharacterProfiles {id: '_all_profiles_'})
        RETURN cp.data AS data
        """
        try:
            result = await self._execute_query(query, write=False)
            if result and result[0] and result[0]["data"]:
                return json.loads(result[0]["data"])
            return {}
        except Exception:
            return {}

    async def save_world_building(self, world_data: Dict[str, Any]) -> bool:
        json_data = json.dumps(world_data, ensure_ascii=False)
        query = """
        MERGE (wb:WorldBuilding {id: 'main_world'})
        SET wb.data = $json_data
        RETURN wb.id
        """
        try:
            result = await self._execute_query(query, {"json_data": json_data}, write=True)
            self.logger.info("Saved world building to Neo4j.")
            return result is not None
        except Exception:
            return False

    async def get_world_building(self) -> Dict[str, Any]:
        query = """
        MATCH (wb:WorldBuilding {id: 'main_world'})
        RETURN wb.data AS data
        """
        try:
            result = await self._execute_query(query, write=False)
            if result and result[0] and result[0]["data"]:
                return json.loads(result[0]["data"])
            return {}
        except Exception:
            return {}

    async def async_load_chapter_count(self) -> int:
        query = """
        MATCH (c:Chapter)
        RETURN MAX(c.number) AS max_chap_num
        """
        try:
            result = await self._execute_query(query, write=False)
            max_chap_num = result[0]["max_chap_num"] if result and result[0] else 0
            count = max_chap_num if max_chap_num is not None else 0
            self.logger.info(f"Neo4j loaded chapter count: {count}")
            return count
        except Exception:
            self.logger.error("Failed to load chapter count from Neo4j.", exc_info=True)
            return 0


    async def async_save_chapter_data(self, chapter_number: int, text: str, raw_llm_output: str, summary: Optional[str], embedding_array: Optional[np.ndarray], is_provisional: bool = False):
        if chapter_number <= 0:
            self.logger.error(f"Neo4j: Cannot save chapter data for invalid chapter_number: {chapter_number}.")
            return

        embedding_b64, embedding_dtype, embedding_shape = None, None, None
        if embedding_array is not None and isinstance(embedding_array, np.ndarray) and embedding_array.size > 0:
            embedding_b64, embedding_dtype, embedding_shape = self._serialize_embedding(embedding_array)

        query = """
        MERGE (c:Chapter {number: $chapter_number})
        SET c.text = $text,
            c.raw_text = $raw_text,
            c.summary = $summary,
            c.is_provisional = $is_provisional
        """
        parameters = {
            "chapter_number": chapter_number,
            "text": text,
            "raw_text": raw_llm_output,
            "summary": summary if summary is not None else "",
            "is_provisional": is_provisional
        }

        try:
            await self._execute_query(query, parameters, write=True)
            self.logger.info(f"Neo4j: Successfully saved chapter data for chapter {chapter_number}.")

            # Handle embedding separately or in the same transaction
            if embedding_b64:
                embedding_query = """
                MERGE (e:Embedding {chapter_number: $chapter_number})
                SET e.embedding_b64 = $embedding_b64,
                    e.dtype = $dtype,
                    e.shape = $shape
                WITH e
                MATCH (c:Chapter {number: $chapter_number})
                MERGE (c)-[:HAS_EMBEDDING]->(e)
                """
                embedding_params = {
                    "chapter_number": chapter_number,
                    "embedding_b64": embedding_b64,
                    "dtype": embedding_dtype,
                    "shape": embedding_shape
                }
                await self._execute_query(embedding_query, embedding_params, write=True)
                self.logger.debug(f"Neo4j: Saved embedding for chapter {chapter_number}.")
            else: # If no new embedding, ensure old one is removed if exists
                remove_embedding_query = """
                MATCH (c:Chapter {number: $chapter_number})-[r:HAS_EMBEDDING]->(e:Embedding)
                DETACH DELETE r, e
                """
                await self._execute_query(remove_embedding_query, {"chapter_number": chapter_number}, write=True)
                self.logger.debug(f"Neo4j: Removed embedding for chapter {chapter_number} as none was provided.")

        except Exception as e:
            self.logger.error(f"Neo4j: Error saving chapter data for chapter {chapter_number}: {e}", exc_info=True)


    async def async_get_chapter_data_from_db(self, chapter_number: int) -> Optional[Dict[str, Any]]:
        if chapter_number <= 0: return None
        query = """
        MATCH (c:Chapter {number: $chapter_number})
        RETURN c.text AS text, c.raw_text AS raw_text, c.summary AS summary, c.is_provisional AS is_provisional
        """
        try:
            result = await self._execute_query(query, {"chapter_number": chapter_number}, write=False)
            if result and result[0]:
                self.logger.debug(f"Neo4j: Data found for chapter {chapter_number}.")
                return {
                    "text": result[0]["text"],
                    "summary": result[0]["summary"],
                    "is_provisional": result[0]["is_provisional"],
                    "raw_text": result[0]["raw_text"]
                }
            self.logger.debug(f"Neo4j: No data found for chapter {chapter_number}.")
            return None
        except Exception:
            self.logger.error(f"Neo4j: Error getting chapter data for {chapter_number}.", exc_info=True)
            return None

    async def async_get_embedding_from_db(self, chapter_number: int) -> Optional[np.ndarray]:
        if chapter_number <= 0: return None
        query = """
        MATCH (c:Chapter {number: $chapter_number})-[:HAS_EMBEDDING]->(e:Embedding)
        RETURN e.embedding_b64 AS embedding_b64, e.dtype AS dtype, e.shape AS shape
        """
        try:
            result = await self._execute_query(query, {"chapter_number": chapter_number}, write=False)
            if result and result[0]:
                return self._deserialize_embedding(result[0]["embedding_b64"], result[0]["dtype"], result[0]["shape"])
            self.logger.debug(f"Neo4j: No embedding found for chapter {chapter_number}.")
            return None
        except Exception:
            self.logger.error(f"Neo4j: Error getting embedding for {chapter_number}.", exc_info=True)
            return None

    async def async_get_all_past_embeddings(self, current_chapter_number: int) -> List[Tuple[int, np.ndarray]]:
        embeddings_list: List[Tuple[int, np.ndarray]] = []
        query = """
        MATCH (c:Chapter)-[:HAS_EMBEDDING]->(e:Embedding)
        WHERE c.number < $current_chapter_number AND c.number > 0
        RETURN c.number AS chapter_number, e.embedding_b64 AS embedding_b64, e.dtype AS dtype, e.shape AS shape
        ORDER BY c.number DESC
        """
        try:
            results = await self._execute_query(query, {"current_chapter_number": current_chapter_number}, write=False)
            if results:
                for record in results:
                    deserialized_emb = self._deserialize_embedding(record["embedding_b64"], record["dtype"], record["shape"])
                    if deserialized_emb is not None:
                        embeddings_list.append((record["chapter_number"], deserialized_emb))
            self.logger.info(f"Neo4j: Retrieved {len(embeddings_list)} past embeddings.")
            return embeddings_list
        except Exception:
            self.logger.error(f"Neo4j: Error getting all past embeddings.", exc_info=True)
            return []
        
    async def async_add_kg_triple(self, subject: str, predicate: str, obj_val: str, chapter_added: int, confidence: float = 1.0, is_provisional: bool = False):
        subj_s, pred_s, obj_s = subject.strip(), predicate.strip(), obj_val.strip()
        if not all([subj_s, pred_s, obj_s]) or chapter_added < config.KG_PREPOPULATION_CHAPTER_NUM:
            self.logger.warning(f"Neo4j: Invalid KG triple for add: S='{subj_s}', P='{pred_s}', O='{obj_s}', Chap={chapter_added}")
            return

        query = """
        MERGE (s:Entity {name: $subject})
        MERGE (o:Entity {name: $object})
        MERGE (s)-[r:RELATIONSHIP {type: $predicate, chapter_added: $chapter_added, is_provisional: $is_provisional}]->(o)
        ON CREATE SET r.confidence = $confidence
        ON MATCH SET r.confidence = CASE WHEN $confidence > r.confidence THEN $confidence ELSE r.confidence END, 
                     r.is_provisional = $is_provisional
        RETURN s.name, type(r), o.name
        """
        parameters = {
            "subject": subj_s,
            "predicate": pred_s,
            "object": obj_s,
            "chapter_added": chapter_added,
            "confidence": confidence,
            "is_provisional": is_provisional
        }
        try:
            await self._execute_query(query, parameters, write=True)
            self.logger.debug(f"Neo4j: Added KG triple for Ch {chapter_added}: ({subj_s}, {pred_s}, {obj_s}).")
        except Exception:
            self.logger.error(f"Neo4j: Error adding KG triple: ({subj_s}, {pred_s}, {obj_s}).", exc_info=True)

    async def async_query_kg(self, subject: Optional[str] = None, predicate: Optional[str] = None, obj_val: Optional[str] = None, chapter_limit: Optional[int] = None, include_provisional: bool = True) -> List[Dict[str, Any]]:
        conditions = []
        parameters = {}
        
        query = "MATCH (s:Entity)-[r:RELATIONSHIP]->(o:Entity)"

        if subject is not None:
            conditions.append("s.name = $subject")
            parameters["subject"] = subject.strip()
        if predicate is not None:
            conditions.append("r.type = $predicate")
            parameters["predicate"] = predicate.strip()
        if obj_val is not None:
            conditions.append("o.name = $object")
            parameters["object"] = obj_val.strip()
        if chapter_limit is not None:
            conditions.append("r.chapter_added <= $chapter_limit")
            parameters["chapter_limit"] = chapter_limit
        if not include_provisional:
            conditions.append("r.is_provisional = FALSE")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " RETURN s.name AS subject, r.type AS predicate, o.name AS object, r.chapter_added AS chapter_added, r.confidence AS confidence, r.is_provisional AS is_provisional"
        query += " ORDER BY r.chapter_added DESC, r.confidence DESC"

        try:
            results = await self._execute_query(query, parameters, write=False)
            triples_list: List[Dict[str, Any]] = []
            if results:
                for record in results:
                    triples_list.append({
                        "subject": record["subject"],
                        "predicate": record["predicate"],
                        "object": record["object"],
                        "chapter_added": record["chapter_added"],
                        "confidence": record["confidence"],
                        "is_provisional": record["is_provisional"]
                    })
            self.logger.debug(f"Neo4j: KG query returned {len(triples_list)} results.")
            return triples_list
        except Exception:
            self.logger.error(f"Neo4j: Error querying KG.", exc_info=True)
            return []

    async def async_get_most_recent_value(self, subject: str, predicate: str, chapter_limit: Optional[int] = None, include_provisional: bool = False) -> Optional[str]:
        if not subject.strip() or not predicate.strip():
            self.logger.warning(f"Neo4j: get_most_recent_value: empty subject or predicate. S='{subject}', P='{predicate}'")
            return None
        
        results = await self.async_query_kg(subject=subject, predicate=predicate, chapter_limit=chapter_limit, include_provisional=include_provisional)
        if results:
            value = str(results[0]["object"]) 
            self.logger.debug(f"Neo4j: Found most recent value: '{value}' from Ch {results[0]['chapter_added']}")
            return value
        self.logger.debug(f"Neo4j: No value found for ({subject}, {predicate}) up to Ch {chapter_limit}")
        return None

state_manager = state_managerSingleton()