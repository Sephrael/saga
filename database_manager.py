# database_manager.py
"""
Manages all interactions with the SQLite database for the Saga project.
Handles connection, schema initialization, and CRUD operations for
chapter text, summaries, embeddings, and the knowledge graph.
Includes asynchronous versions of database operations using aiosqlite.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Copyright 2025 Dennis Lewis
"""

import sqlite3
import json
import numpy as np
import logging
import os
from typing import Optional, Dict, List, Tuple, Any, Union
import aiosqlite # For asynchronous SQLite operations

import config

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Handles all interactions with the SQLite database."""

    def __init__(self, db_path: str = config.DATABASE_FILE):
        self.db_path = db_path
        self._ensure_db_directory()
        self._init_db_schema() # Synchronous initialization

    def _ensure_db_directory(self):
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            try:
                os.makedirs(db_dir, exist_ok=True)
                logger.info(f"Created directory for database: {db_dir}")
            except OSError as e:
                logger.error(f"Failed to create directory {db_dir}: {e}", exc_info=True)
                raise

    def _get_sync_connection(self) -> sqlite3.Connection:
        """Establishes and returns a synchronous database connection."""
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON;")
            logger.debug(f"Synchronous DB connection established to {self.db_path}")
            return conn
        except sqlite3.Error as e:
            logger.error(f"Synchronous DB connection error to {self.db_path}: {e}", exc_info=True)
            raise

    async def _get_async_connection(self) -> aiosqlite.Connection:
        """Establishes and returns an asynchronous database connection."""
        try:
            conn = await aiosqlite.connect(self.db_path, timeout=10.0)
            conn.row_factory = aiosqlite.Row
            await conn.execute("PRAGMA foreign_keys = ON;")
            logger.debug(f"Asynchronous DB connection established to {self.db_path} (id: {id(conn)})")
            return conn
        except sqlite3.Error as e: # aiosqlite.Error inherits from sqlite3.Error
            logger.error(f"Asynchronous DB connection error to {self.db_path}: {e}", exc_info=True)
            raise

    def _init_db_schema(self): # Stays synchronous for initial setup
        logger.info(f"Initializing/Verifying database schema at {self.db_path}...")
        try:
            with self._get_sync_connection() as conn:
                cursor = conn.cursor()
                # Chapters Table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chapters (
                        chapter_number INTEGER PRIMARY KEY,
                        text TEXT,
                        raw_text TEXT, -- For storing raw LLM output before cleaning, useful for debugging
                        summary TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        is_provisional BOOLEAN DEFAULT FALSE 
                    )
                """)
                # Embeddings Table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS embeddings (
                        chapter_number INTEGER PRIMARY KEY,
                        embedding_blob BLOB NOT NULL,
                        dtype TEXT NOT NULL,
                        shape TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (chapter_number) REFERENCES chapters (chapter_number) ON DELETE CASCADE
                    )
                """)
                # Knowledge Graph Table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS knowledge_graph (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        subject TEXT NOT NULL,
                        predicate TEXT NOT NULL,
                        object TEXT NOT NULL,
                        chapter_added INTEGER NOT NULL, -- Chapter where this fact was introduced/confirmed
                        confidence REAL DEFAULT 1.0, 
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        is_provisional BOOLEAN DEFAULT FALSE -- If true, fact might be from unrevised draft
                    )
                """)
                logger.debug("Verified/Created tables: 'chapters', 'embeddings', 'knowledge_graph'.")

                # Indices
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_chapter ON embeddings (chapter_number);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_subject ON knowledge_graph (subject);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_predicate ON knowledge_graph (predicate);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_object ON knowledge_graph (object);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_spo ON knowledge_graph (subject, predicate, object);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_chapter ON knowledge_graph (chapter_added);")
                logger.debug("Verified/Created indices.")
                
                # Schema migrations (idempotent)
                table_columns = {}
                for table_name in ["chapters", "knowledge_graph"]:
                    cursor.execute(f"PRAGMA table_info({table_name});")
                    table_columns[table_name] = [row['name'] for row in cursor.fetchall()]

                if "is_provisional" not in table_columns["chapters"]:
                    logger.info("Altering 'chapters' table to add 'is_provisional' column.")
                    cursor.execute("ALTER TABLE chapters ADD COLUMN is_provisional BOOLEAN DEFAULT FALSE;")
                
                if "raw_text" not in table_columns["chapters"]: # Added raw_text for debugging LLM outputs
                    logger.info("Altering 'chapters' table to add 'raw_text' column.")
                    cursor.execute("ALTER TABLE chapters ADD COLUMN raw_text TEXT;")

                if "is_provisional" not in table_columns["knowledge_graph"]:
                    logger.info("Altering 'knowledge_graph' table to add 'is_provisional' column.")
                    cursor.execute("ALTER TABLE knowledge_graph ADD COLUMN is_provisional BOOLEAN DEFAULT FALSE;")
                
                conn.commit()
            logger.info("Database schema initialization/verification complete.")
        except sqlite3.Error as e:
            logger.error(f"Database schema initialization error: {e}", exc_info=True)
            raise

    def _serialize_embedding(self, embedding: np.ndarray) -> Tuple[bytes, str, str]:
        embedding_to_save = embedding.astype(config.EMBEDDING_DTYPE)
        if embedding_to_save.ndim == 0: # Handle scalar array
             embedding_to_save = embedding_to_save.reshape(1)
        return embedding_to_save.tobytes(), str(embedding_to_save.dtype), json.dumps(embedding_to_save.shape)

    def _deserialize_embedding(self, blob: bytes, dtype_str: str, shape_str: str) -> Optional[np.ndarray]:
        try:
            shape = tuple(json.loads(shape_str))
            dtype = np.dtype(dtype_str)
            return np.frombuffer(blob, dtype=dtype).reshape(shape)
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError) as e:
            logger.error(f"Error deserializing embedding (shape: {shape_str}, dtype: {dtype_str}): {e}", exc_info=True)
            return None

    def load_chapter_count(self) -> int: # Sync
        logger.debug("Loading highest chapter count from database...")
        try:
            with self._get_sync_connection() as conn:
                cursor = conn.cursor()
                # Ensure we only count chapters > 0, as 0 might be for pre-population
                cursor.execute("SELECT MAX(chapter_number) FROM chapters WHERE chapter_number > 0")
                result = cursor.fetchone()
                count = result[0] if result and result[0] is not None else 0
                logger.info(f"Loaded chapter count from database: {count}")
                return count
        except sqlite3.Error as e:
            logger.error(f"Failed to load chapter count from DB: {e}. Assuming 0.", exc_info=True)
            return 0
    
    async def async_load_chapter_count(self) -> int: # Async
        logger.debug("Async loading highest chapter count from database...")
        conn = None
        try:
            conn = await self._get_async_connection()
            async with conn.execute("SELECT MAX(chapter_number) FROM chapters WHERE chapter_number > 0") as cursor:
                result = await cursor.fetchone()
            count = result[0] if result and result[0] is not None else 0
            logger.info(f"Async loaded chapter count from database: {count}")
            return count
        except sqlite3.Error as e:
            logger.error(f"Async failed to load chapter count from DB: {e}. Assuming 0.", exc_info=True)
            return 0
        finally:
            if conn: await conn.close()

    def _prepare_embedding_for_save(self, embedding: Optional[np.ndarray], chapter_number: int) -> Tuple[Optional[bytes], Optional[str], Optional[str]]:
        if embedding is not None and isinstance(embedding, np.ndarray) and embedding.size > 0:
            try:
                blob, dtype_str, shape_str = self._serialize_embedding(embedding)
                logger.debug(f"Serialized embedding for chapter {chapter_number} (Shape: {shape_str}, Dtype: {dtype_str})")
                return blob, dtype_str, shape_str
            except Exception as e:
                logger.error(f"Error serializing embedding for chapter {chapter_number}: {e}", exc_info=True)
        else:
            logger.warning(f"Chapter {chapter_number} has no valid embedding. Embedding will not be saved/updated.")
        return None, None, None

    def save_chapter_data(self, chapter_number: int, text: str, raw_llm_output: str, summary: Optional[str], embedding: Optional[np.ndarray], is_provisional: bool = False): # Sync
        if chapter_number <= 0:
            logger.error(f"Cannot save chapter data for invalid chapter_number: {chapter_number}.")
            return
        logger.info(f"Saving data for chapter {chapter_number} (Provisional: {is_provisional}).")
        
        embedding_blob, dtype_str, shape_str = self._prepare_embedding_for_save(embedding, chapter_number)

        try:
            with self._get_sync_connection() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO chapters (chapter_number, text, raw_text, summary, is_provisional) VALUES (?, ?, ?, ?, ?)",
                    (chapter_number, text, raw_llm_output, summary if summary is not None else "", is_provisional)
                )
                if embedding_blob and dtype_str and shape_str:
                    conn.execute(
                        "INSERT OR REPLACE INTO embeddings (chapter_number, embedding_blob, dtype, shape) VALUES (?, ?, ?, ?)",
                        (chapter_number, embedding_blob, dtype_str, shape_str)
                    )
                else: # If no valid embedding, ensure any old one is removed
                    conn.execute("DELETE FROM embeddings WHERE chapter_number = ?", (chapter_number,))
                conn.commit()
            logger.info(f"Successfully saved chapter data for chapter {chapter_number}.")
        except sqlite3.Error as e:
            logger.error(f"DB error saving chapter {chapter_number}: {e}", exc_info=True)

    async def async_save_chapter_data(self, chapter_number: int, text: str, raw_llm_output: str, summary: Optional[str], embedding: Optional[np.ndarray], is_provisional: bool = False): # Async
        if chapter_number <= 0:
            logger.error(f"Async: Cannot save chapter data for invalid chapter_number: {chapter_number}.")
            return
        logger.info(f"Async: Saving data for chapter {chapter_number} (Provisional: {is_provisional}).")

        embedding_blob, dtype_str, shape_str = self._prepare_embedding_for_save(embedding, chapter_number)
        conn = None
        try:
            conn = await self._get_async_connection()
            await conn.execute(
                "INSERT OR REPLACE INTO chapters (chapter_number, text, raw_text, summary, is_provisional) VALUES (?, ?, ?, ?, ?)",
                (chapter_number, text, raw_llm_output, summary if summary is not None else "", is_provisional)
            )
            if embedding_blob and dtype_str and shape_str:
                await conn.execute(
                    "INSERT OR REPLACE INTO embeddings (chapter_number, embedding_blob, dtype, shape) VALUES (?, ?, ?, ?)",
                    (chapter_number, embedding_blob, dtype_str, shape_str)
                )
            else:
                await conn.execute("DELETE FROM embeddings WHERE chapter_number = ?", (chapter_number,))
            await conn.commit()
            logger.info(f"Async: Successfully saved chapter data for chapter {chapter_number}.")
        except sqlite3.Error as e:
            logger.error(f"Async: DB error saving chapter {chapter_number}: {e}", exc_info=True)
            if conn: await conn.rollback()
        finally:
            if conn: await conn.close()
            
    def get_embedding_from_db(self, chapter_number: int) -> Optional[np.ndarray]: # Sync
        if chapter_number <= 0: return None
        logger.debug(f"Retrieving embedding for chapter {chapter_number}.")
        try:
            with self._get_sync_connection() as conn:
                cursor = conn.execute("SELECT embedding_blob, dtype, shape FROM embeddings WHERE chapter_number = ?", (chapter_number,))
                result = cursor.fetchone()
            if result and result["embedding_blob"]:
                embedding = self._deserialize_embedding(result["embedding_blob"], result["dtype"], result["shape"])
                if embedding is not None: logger.debug(f"Retrieved embedding for chapter {chapter_number}.")
                return embedding
            logger.debug(f"No embedding found for chapter {chapter_number}.")
        except sqlite3.Error as e:
            logger.error(f"DB error retrieving embedding for chapter {chapter_number}: {e}", exc_info=True)
        return None

    async def async_get_embedding_from_db(self, chapter_number: int) -> Optional[np.ndarray]: # Async
        if chapter_number <= 0: return None
        logger.debug(f"Async: Retrieving embedding for chapter {chapter_number}.")
        conn = None
        try:
            conn = await self._get_async_connection()
            async with conn.execute("SELECT embedding_blob, dtype, shape FROM embeddings WHERE chapter_number = ?", (chapter_number,)) as cursor:
                result = await cursor.fetchone()
            if result and result["embedding_blob"]:
                embedding = self._deserialize_embedding(result["embedding_blob"], result["dtype"], result["shape"])
                if embedding is not None: logger.debug(f"Async: Retrieved embedding for chapter {chapter_number}.")
                return embedding
            logger.debug(f"Async: No embedding found for chapter {chapter_number}.")
        except sqlite3.Error as e:
            logger.error(f"Async: DB error retrieving embedding for chapter {chapter_number}: {e}", exc_info=True)
        finally:
            if conn: await conn.close()
        return None

    def get_all_past_embeddings(self, current_chapter_number: int) -> List[Tuple[int, np.ndarray]]: # Sync
        logger.debug(f"Retrieving all past embeddings before chapter {current_chapter_number}.")
        embeddings: List[Tuple[int, np.ndarray]] = []
        try:
            with self._get_sync_connection() as conn:
                cursor = conn.execute(
                    "SELECT chapter_number, embedding_blob, dtype, shape FROM embeddings WHERE chapter_number < ? AND chapter_number > 0 ORDER BY chapter_number DESC",
                    (current_chapter_number,)
                )
                results = cursor.fetchall()
            for row in results:
                embedding = self._deserialize_embedding(row["embedding_blob"], row["dtype"], row["shape"])
                if embedding is not None: embeddings.append((row["chapter_number"], embedding))
        except sqlite3.Error as e:
            logger.error(f"DB error retrieving past embeddings: {e}", exc_info=True)
        logger.info(f"Retrieved {len(embeddings)} past embeddings.")
        return embeddings

    async def async_get_all_past_embeddings(self, current_chapter_number: int) -> List[Tuple[int, np.ndarray]]: # Async
        logger.debug(f"Async: Retrieving all past embeddings before chapter {current_chapter_number}.")
        embeddings: List[Tuple[int, np.ndarray]] = []
        conn = None
        try:
            conn = await self._get_async_connection()
            async with conn.execute(
                "SELECT chapter_number, embedding_blob, dtype, shape FROM embeddings WHERE chapter_number < ? AND chapter_number > 0 ORDER BY chapter_number DESC",
                (current_chapter_number,)
            ) as cursor:
                results = await cursor.fetchall()
            for row in results:
                embedding = self._deserialize_embedding(row["embedding_blob"], row["dtype"], row["shape"])
                if embedding is not None: embeddings.append((row["chapter_number"], embedding))
        except sqlite3.Error as e:
            logger.error(f"Async: DB error retrieving past embeddings: {e}", exc_info=True)
        finally:
            if conn: await conn.close()
        logger.info(f"Async: Retrieved {len(embeddings)} past embeddings.")
        return embeddings

    def get_chapter_data_from_db(self, chapter_number: int) -> Optional[Dict[str, Any]]: # Sync
        if chapter_number <= 0: return None
        logger.debug(f"Retrieving data for chapter {chapter_number}.")
        try:
            with self._get_sync_connection() as conn:
                cursor = conn.execute("SELECT text, summary, is_provisional FROM chapters WHERE chapter_number = ?", (chapter_number,))
                result = cursor.fetchone()
            if result:
                data = dict(result)
                logger.debug(f"Found data for chapter {chapter_number} (Provisional: {data.get('is_provisional')}).")
                return data
            logger.debug(f"No data found for chapter {chapter_number}.")
        except sqlite3.Error as e:
            logger.error(f"DB error retrieving data for chapter {chapter_number}: {e}", exc_info=True)
        return None

    async def async_get_chapter_data_from_db(self, chapter_number: int) -> Optional[Dict[str, Any]]: # Async
        if chapter_number <= 0: return None
        logger.debug(f"Async: Retrieving data for chapter {chapter_number}.")
        conn = None
        try:
            conn = await self._get_async_connection()
            async with conn.execute("SELECT text, summary, is_provisional FROM chapters WHERE chapter_number = ?", (chapter_number,)) as cursor:
                result = await cursor.fetchone()
            if result:
                data = dict(result)
                logger.debug(f"Async: Found data for chapter {chapter_number} (Provisional: {data.get('is_provisional')}).")
                return data
            logger.debug(f"Async: No data found for chapter {chapter_number}.")
        except sqlite3.Error as e:
            logger.error(f"Async: DB error retrieving data for chapter {chapter_number}: {e}", exc_info=True)
        finally:
            if conn: await conn.close()
        return None

    def add_kg_triple(self, subject: str, predicate: str, obj: str, chapter_added: int, confidence: float = 1.0, is_provisional: bool = False): # Sync
        subj, pred, o = subject.strip(), predicate.strip(), obj.strip()
        if not all([subj, pred, o]) or chapter_added < config.KG_PREPOPULATION_CHAPTER_NUM:
            logger.warning(f"Invalid KG triple for add: S='{subj}', P='{pred}', O='{o}', Chap={chapter_added}")
            return
        logger.debug(f"Adding KG triple: ({subj}, {pred}, {o}) from Ch {chapter_added} (Prov: {is_provisional})")
        try:
            with self._get_sync_connection() as conn:
                # Check for existence before inserting to avoid duplicates for the same chapter_added
                cursor = conn.execute(
                    "SELECT id FROM knowledge_graph WHERE subject = ? AND predicate = ? AND object = ? AND chapter_added = ?",
                    (subj, pred, o, chapter_added)
                )
                if cursor.fetchone():
                    logger.debug(f"Triple ({subj}, {pred}, {o}) for Ch {chapter_added} already exists. Skipping.")
                    return

                conn.execute(
                    "INSERT INTO knowledge_graph (subject, predicate, object, chapter_added, confidence, is_provisional) VALUES (?, ?, ?, ?, ?, ?)",
                    (subj, pred, o, chapter_added, confidence, is_provisional)
                )
                conn.commit()
                logger.debug(f"Added KG triple for Ch {chapter_added}.")
        except sqlite3.Error as e:
            logger.error(f"DB error adding KG triple: {e}", exc_info=True)

    async def async_add_kg_triple(self, subject: str, predicate: str, obj: str, chapter_added: int, confidence: float = 1.0, is_provisional: bool = False): # Async
        subj, pred, o = subject.strip(), predicate.strip(), obj.strip()
        if not all([subj, pred, o]) or chapter_added < config.KG_PREPOPULATION_CHAPTER_NUM:
            logger.warning(f"Async: Invalid KG triple for add: S='{subj}', P='{pred}', O='{o}', Chap={chapter_added}")
            return
        logger.debug(f"Async: Adding KG triple: ({subj}, {pred}, {o}) from Ch {chapter_added} (Prov: {is_provisional})")
        conn = None
        try:
            conn = await self._get_async_connection()
            async with conn.execute(
                "SELECT id FROM knowledge_graph WHERE subject = ? AND predicate = ? AND object = ? AND chapter_added = ?",
                (subj, pred, o, chapter_added)
            ) as cursor:
                exists = await cursor.fetchone()
            
            if exists:
                logger.debug(f"Async: Triple ({subj}, {pred}, {o}) for Ch {chapter_added} already exists. Skipping.")
                return

            await conn.execute(
                "INSERT INTO knowledge_graph (subject, predicate, object, chapter_added, confidence, is_provisional) VALUES (?, ?, ?, ?, ?, ?)",
                (subj, pred, o, chapter_added, confidence, is_provisional)
            )
            await conn.commit()
            logger.debug(f"Async: Added KG triple for Ch {chapter_added}.")
        except sqlite3.Error as e:
            logger.error(f"Async: DB error adding KG triple: {e}", exc_info=True)
            if conn: await conn.rollback()
        finally:
            if conn: await conn.close()

    def _build_kg_query(self, subject: Optional[str], predicate: Optional[str], obj: Optional[str], chapter_limit: Optional[int], include_provisional: bool) -> Tuple[str, List[Any]]:
        query_parts = ["SELECT id, subject, predicate, object, chapter_added, confidence, timestamp, is_provisional FROM knowledge_graph WHERE 1=1"]
        params: List[Any] = []
        if subject is not None: query_parts.append("AND subject = ?"); params.append(subject.strip())
        if predicate is not None: query_parts.append("AND predicate = ?"); params.append(predicate.strip())
        if obj is not None: query_parts.append("AND object = ?"); params.append(obj.strip())
        if chapter_limit is not None and isinstance(chapter_limit, int) and chapter_limit >= config.KG_PREPOPULATION_CHAPTER_NUM:
            query_parts.append("AND chapter_added <= ?"); params.append(chapter_limit)
        if not include_provisional: query_parts.append("AND is_provisional = FALSE")
        query_parts.append("ORDER BY chapter_added DESC, confidence DESC, timestamp DESC")
        return " ".join(query_parts), params

    def query_kg(self, subject: Optional[str] = None, predicate: Optional[str] = None, obj: Optional[str] = None, chapter_limit: Optional[int] = None, include_provisional: bool = True) -> List[Dict[str, Any]]: # Sync
        final_query, params = self._build_kg_query(subject, predicate, obj, chapter_limit, include_provisional)
        logger.debug(f"Executing KG query: {final_query} with params: {params}")
        results: List[Dict[str, Any]] = []
        try:
            with self._get_sync_connection() as conn:
                cursor = conn.execute(final_query, params)
                results = [dict(row) for row in cursor.fetchall()]
            logger.debug(f"KG query returned {len(results)} results.")
        except sqlite3.Error as e:
            logger.error(f"DB error querying KG: {e}", exc_info=True)
        return results

    async def async_query_kg(self, subject: Optional[str] = None, predicate: Optional[str] = None, obj: Optional[str] = None, chapter_limit: Optional[int] = None, include_provisional: bool = True) -> List[Dict[str, Any]]: # Async
        final_query, params = self._build_kg_query(subject, predicate, obj, chapter_limit, include_provisional)
        logger.debug(f"Async: Executing KG query: {final_query} with params: {params}")
        results_list: List[Dict[str, Any]] = []
        conn = None
        try:
            conn = await self._get_async_connection()
            async with conn.execute(final_query, params) as cursor:
                rows = await cursor.fetchall()
                results_list = [dict(row) for row in rows]
            logger.debug(f"Async: KG query returned {len(results_list)} results.")
        except sqlite3.Error as e:
            logger.error(f"Async: DB error querying KG: {e}", exc_info=True)
        finally:
            if conn: await conn.close()
        return results_list

    def get_most_recent_value(self, subject: str, predicate: str, chapter_limit: Optional[int] = None, include_provisional: bool = False) -> Optional[str]: # Sync
        if not subject.strip() or not predicate.strip():
            logger.warning(f"get_most_recent_value: empty subject or predicate. S='{subject}', P='{predicate}'")
            return None
        logger.debug(f"Getting most recent KG value for ({subject}, {predicate}) up to Ch {chapter_limit} (Prov: {include_provisional})")
        results = self.query_kg(subject=subject, predicate=predicate, chapter_limit=chapter_limit, include_provisional=include_provisional)
        if results:
            value = str(results[0]["object"])
            logger.debug(f"Found most recent value: '{value}' from Ch {results[0]['chapter_added']}")
            return value
        logger.debug(f"No value found for ({subject}, {predicate}) up to Ch {chapter_limit}")
        return None

    async def async_get_most_recent_value(self, subject: str, predicate: str, chapter_limit: Optional[int] = None, include_provisional: bool = False) -> Optional[str]: # Async
        if not subject.strip() or not predicate.strip():
            logger.warning(f"Async: get_most_recent_value: empty subject or predicate. S='{subject}', P='{predicate}'")
            return None
        logger.debug(f"Async: Getting most recent KG value for ({subject}, {predicate}) up to Ch {chapter_limit} (Prov: {include_provisional})")
        results = await self.async_query_kg(subject=subject, predicate=predicate, chapter_limit=chapter_limit, include_provisional=include_provisional)
        if results:
            value = str(results[0]["object"])
            logger.debug(f"Async: Found most recent value: '{value}' from Ch {results[0]['chapter_added']}")
            return value
        logger.debug(f"Async: No value found for ({subject}, {predicate}) up to Ch {chapter_limit}")
        return None