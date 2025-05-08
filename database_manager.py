# database_manager.py
"""
Manages all interactions with the SQLite database for the Saga project.
Handles connection, schema initialization, and CRUD operations for
chapter text, summaries, embeddings, and the knowledge graph.

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
from typing import Optional, Dict, List, Tuple, Any

# Import configuration for DB path and embedding dtype
import config

# Initialize logger for this module
logger = logging.getLogger(__name__)


class DatabaseManager:
    """Handles all interactions with the SQLite database."""

    def __init__(self, db_path: str = config.DATABASE_FILE):
        """
        Initializes the DatabaseManager.

        Args:
            db_path: The path to the SQLite database file. Defaults to config.DATABASE_FILE.
        """
        self.db_path = db_path
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            try:
                os.makedirs(db_dir, exist_ok=True)
                logger.info(f"Created directory for database: {db_dir}")
            except OSError as e:
                logger.error(f"Failed to create directory {db_dir}: {e}", exc_info=True)
                raise  # Re-raise as this could be critical

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """
        Establishes and returns a database connection.
        Sets row_factory to sqlite3.Row for dictionary-like row access.
        Includes a timeout and enables foreign keys.
        """
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON;")
            logger.debug(f"Database connection established to {self.db_path}")
            return conn
        except sqlite3.Error as e:
            logger.error(
                f"Database connection error to {self.db_path}: {e}", exc_info=True
            )
            raise

    def _init_db(self):
        """
        Initializes the SQLite database schema, including the knowledge_graph table.
        Creates tables and indices if they don't already exist.
        """
        logger.info(f"Initializing/Verifying database schema at {self.db_path}...")
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chapters (
                        chapter_number INTEGER PRIMARY KEY,
                        text TEXT,
                        raw_text TEXT,
                        summary TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        is_provisional BOOLEAN DEFAULT FALSE -- ****** NEW COLUMN ******
                    )
                """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS embeddings (
                        chapter_number INTEGER PRIMARY KEY,
                        embedding_blob BLOB NOT NULL,
                        dtype TEXT NOT NULL,
                        shape TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (chapter_number) REFERENCES chapters (chapter_number) ON DELETE CASCADE
                    )
                """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS knowledge_graph (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        subject TEXT NOT NULL,
                        predicate TEXT NOT NULL,
                        object TEXT NOT NULL,
                        chapter_added INTEGER NOT NULL,
                        confidence REAL DEFAULT 1.0, 
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        is_provisional BOOLEAN DEFAULT FALSE -- ****** NEW COLUMN ******
                    )
                """
                )
                logger.debug(
                    "Verified/Created tables: 'chapters', 'embeddings', 'knowledge_graph'."
                )

                # Indices
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_embeddings_chapter ON embeddings (chapter_number);"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_kg_subject ON knowledge_graph (subject);"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_kg_predicate ON knowledge_graph (predicate);"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_kg_object ON knowledge_graph (object);"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_kg_spo ON knowledge_graph (subject, predicate, object);"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_kg_chapter ON knowledge_graph (chapter_added);"
                )
                logger.debug("Verified/Created indices.")
                conn.commit()
                
                try:
                    cursor.execute("SELECT is_provisional FROM chapters LIMIT 1;")
                except sqlite3.OperationalError:
                    logger.info("Altering chapters table to add is_provisional column.")
                    cursor.execute("ALTER TABLE chapters ADD COLUMN is_provisional BOOLEAN DEFAULT FALSE;")
                
                try:
                    cursor.execute("SELECT is_provisional FROM knowledge_graph LIMIT 1;")
                except sqlite3.OperationalError:
                    logger.info("Altering knowledge_graph table to add is_provisional column.")
                    cursor.execute("ALTER TABLE knowledge_graph ADD COLUMN is_provisional BOOLEAN DEFAULT FALSE;")
                
                conn.commit() # Commit schema changes including potential ALTER TABLE

            logger.info("Database schema initialization/verification complete.")
        except sqlite3.Error as e:
            logger.error(f"Database schema initialization error: {e}", exc_info=True)
            raise

    def _deserialize_embedding(
        self, blob: bytes, dtype_str: str, shape_str: str
    ) -> Optional[np.ndarray]:
        """Helper to deserialize an embedding from database components."""
        try:
            shape = tuple(json.loads(shape_str))
            dtype = np.dtype(
                dtype_str
            )  # Use config.EMBEDDING_DTYPE if dtype_str is problematic
            embedding = np.frombuffer(blob, dtype=dtype).reshape(shape)
            return embedding
        except (
            json.JSONDecodeError,
            TypeError,
            ValueError,
            AttributeError,
        ) as e:  # Added AttributeError for np.dtype
            logger.error(
                f"Error deserializing embedding (shape: {shape_str}, dtype: {dtype_str}): {e}",
                exc_info=True,
            )
            return None

    def load_chapter_count(self) -> int:
        """Loads the current highest chapter number recorded in the database."""
        logger.debug("Loading highest chapter count from database...")
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT MAX(chapter_number) FROM chapters")
                result = cursor.fetchone()
                count = result[0] if result and result[0] is not None else 0
                logger.info(f"Loaded chapter count from database: {count}")
                return count
        except sqlite3.Error as e:
            logger.error(
                f"Failed to load chapter count from DB: {e}. Assuming 0.", exc_info=True
            )
            return 0

    def save_chapter_data(
        self,
        chapter_number: int,
        text: str,
        raw_text: str,
        summary: Optional[str],
        embedding: Optional[np.ndarray],
        is_provisional: bool = False # ****** NEW ARGUMENT ******
    ):
        logger.info(
            f"Attempting to save data for chapter {chapter_number} to database (Provisional: {is_provisional})."
        )
        
        embedding_blob, dtype_str, shape_str = None, "", ""
        if (
            embedding is not None
            and isinstance(embedding, np.ndarray)
            and embedding.size > 0
        ):
            try:
                # Ensure correct dtype before serialization
                embedding_to_save = embedding.astype(config.EMBEDDING_DTYPE)
                if embedding_to_save.ndim == 0:
                    embedding_to_save = embedding_to_save.reshape(
                        1
                    )  # Handle 0-dim arrays

                embedding_blob = embedding_to_save.tobytes()
                dtype_str = str(embedding_to_save.dtype)
                shape_str = json.dumps(embedding_to_save.shape)
                logger.debug(
                    f"Serialized embedding for chapter {chapter_number} (Shape: {shape_str}, Dtype: {dtype_str})"
                )
            except Exception as e:
                logger.error(
                    f"Error serializing embedding for chapter {chapter_number}: {e}",
                    exc_info=True,
                )
                embedding_blob = None  # Ensure it's None if serialization fails
        else:
            logger.warning(
                f"Chapter {chapter_number} has no valid embedding provided or embedding is empty. Embedding will not be saved."
            )

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT OR REPLACE INTO chapters (chapter_number, text, raw_text, summary, is_provisional) VALUES (?, ?, ?, ?, ?)", # Add is_provisional
                    (
                        chapter_number,
                        text,
                        raw_text,
                        summary if summary is not None else "",
                        is_provisional, # ****** Pass value ******
                    ),
                )
                logger.debug(
                    f"Saved/Replaced text/summary data for chapter {chapter_number}."
                )

                if (
                    embedding_blob and dtype_str and shape_str
                ):  # Ensure all parts are valid
                    cursor.execute(
                        "INSERT OR REPLACE INTO embeddings (chapter_number, embedding_blob, dtype, shape) VALUES (?, ?, ?, ?)",
                        (chapter_number, embedding_blob, dtype_str, shape_str),
                    )
                    logger.debug(
                        f"Saved/Replaced embedding data for chapter {chapter_number}."
                    )
                else:
                    # If new embedding is invalid/missing, remove any old one for this chapter
                    cursor.execute(
                        "DELETE FROM embeddings WHERE chapter_number = ?",
                        (chapter_number,),
                    )
                    logger.debug(
                        f"Removed any existing embedding entry for chapter {chapter_number} due to invalid new embedding."
                    )
                conn.commit()
            logger.info(
                f"Successfully saved chapter text/embedding data for chapter {chapter_number}."
            )
        except sqlite3.Error as e:
            logger.error(
                f"Database error saving chapter data for chapter {chapter_number}: {e}",
                exc_info=True,
            )
            # Consider re-raising or specific handling if this is critical
        except Exception as e:  # Catch other unexpected errors
            logger.error(
                f"Unexpected error saving chapter data for chapter {chapter_number}: {e}",
                exc_info=True,
            )

    def get_embedding_from_db(self, chapter_number: int) -> Optional[np.ndarray]:
        """Retrieves and deserializes a single chapter's embedding."""
        logger.debug(
            f"Retrieving embedding for chapter {chapter_number} from database."
        )
        embedding = None
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT embedding_blob, dtype, shape FROM embeddings WHERE chapter_number = ?",
                    (chapter_number,),
                )
                result = cursor.fetchone()
            if result and result["embedding_blob"]:
                embedding = self._deserialize_embedding(
                    result["embedding_blob"], result["dtype"], result["shape"]
                )
                if embedding is not None:
                    logger.debug(
                        f"Successfully retrieved and deserialized embedding for chapter {chapter_number}."
                    )
            else:
                logger.debug(
                    f"No embedding found in database for chapter {chapter_number}."
                )
        except sqlite3.Error as e:
            logger.error(
                f"Database error retrieving embedding for chapter {chapter_number}: {e}",
                exc_info=True,
            )
        except Exception as e:
            logger.error(
                f"Unexpected error retrieving embedding for chapter {chapter_number}: {e}",
                exc_info=True,
            )
        return embedding

    def get_all_past_embeddings(
        self, current_chapter_number: int
    ) -> List[Tuple[int, np.ndarray]]:
        """Retrieves all embeddings for chapters before the specified chapter number."""
        logger.debug(
            f"Retrieving all past embeddings before chapter {current_chapter_number}."
        )
        past_embeddings: List[Tuple[int, np.ndarray]] = []
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """SELECT chapter_number, embedding_blob, dtype, shape
                       FROM embeddings
                       WHERE chapter_number < ?
                       ORDER BY chapter_number DESC""",  # Most recent first helps if caller truncates
                    (current_chapter_number,),
                )
                results = cursor.fetchall()
            logger.debug(f"Found {len(results)} past embedding entries in database.")
            for row in results:
                embedding = self._deserialize_embedding(
                    row["embedding_blob"], row["dtype"], row["shape"]
                )
                if embedding is not None:
                    past_embeddings.append((row["chapter_number"], embedding))
                else:
                    logger.warning(
                        f"Failed to load/deserialize embedding for past chapter {row['chapter_number']}. Skipping."
                    )
        except sqlite3.Error as e:
            logger.error(
                f"Database error retrieving past embeddings: {e}", exc_info=True
            )
        except Exception as e:
            logger.error(
                f"Unexpected error retrieving past embeddings: {e}", exc_info=True
            )

        logger.info(
            f"Successfully retrieved and deserialized {len(past_embeddings)} past embeddings."
        )
        return past_embeddings

    def get_chapter_data_from_db(self, chapter_number: int) -> Optional[Dict[str, Any]]:
        """Retrieves the final text and summary for a given chapter number."""
        logger.debug(f"Retrieving text/summary data for chapter {chapter_number}.")
        data = None
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT text, summary FROM chapters WHERE chapter_number = ?",
                    (chapter_number,),
                )
                result = cursor.fetchone()
            if result:
                data = dict(result)  # Convert sqlite3.Row to dict
                logger.debug(f"Found text/summary data for chapter {chapter_number}.")
            else:
                logger.debug(
                    f"No text/summary data found for chapter {chapter_number}."
                )
        except sqlite3.Error as e:
            logger.error(
                f"Database error retrieving text/summary for chapter {chapter_number}: {e}",
                exc_info=True,
            )
        except Exception as e:
            logger.error(
                f"Unexpected error retrieving chapter data for chapter {chapter_number}: {e}",
                exc_info=True,
            )
        return data

    # --- Knowledge Graph Methods ---

    def add_kg_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        chapter_added: int,
        confidence: float = 1.0,
        is_provisional: bool = False # ****** NEW ARGUMENT ******
    ):
        """
        Adds a single (subject, predicate, object) triple to the knowledge graph.
        Avoids adding exact duplicates for the same chapter.
        """
        subj_clean = subject.strip()
        pred_clean = predicate.strip()
        obj_clean = obj.strip()

        if not all([subj_clean, pred_clean, obj_clean]) or chapter_added <= 0:
            logger.warning(
                f"Attempted to add invalid or empty triple: S='{subj_clean}', P='{pred_clean}', O='{obj_clean}', Chap={chapter_added}"
            )
            return

        logger.debug(
            f"Adding KG triple: ({subj_clean}, {pred_clean}, {obj_clean}) from chapter {chapter_added} (Provisional: {is_provisional})"
        )
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # Check for existing now includes provisional status, or you might decide an S,P,O,chapter match is enough
                cursor.execute(
                    """SELECT id FROM knowledge_graph
                       WHERE subject = ? AND predicate = ? AND object = ? AND chapter_added = ?""", # Consider if provisional status makes it "different"
                    (subj_clean, pred_clean, obj_clean, chapter_added),
                )
                exists = cursor.fetchone()

                if exists:
                    logger.debug(
                        f"Triple ({subj_clean}, {pred_clean}, {obj_clean}) already exists for chapter {chapter_added}. Skipping insert."
                        # Potentially update provisional status or confidence if new info is better? For now, skip.
                    )
                else:
                    cursor.execute(
                        """INSERT INTO knowledge_graph (subject, predicate, object, chapter_added, confidence, is_provisional)
                           VALUES (?, ?, ?, ?, ?, ?)""", # Add is_provisional
                        (subj_clean, pred_clean, obj_clean, chapter_added, confidence, is_provisional), # ****** Pass value ******
                    )
                    conn.commit()
                    logger.debug(
                        f"Successfully added triple for chapter {chapter_added}."
                    )
                    
        except sqlite3.Error as e:
            logger.error(
                f"Database error adding KG triple ({subj_clean}, {pred_clean}, {obj_clean}): {e}",
                exc_info=True,
            )
        except Exception as e:
            logger.error(f"Unexpected error adding KG triple: {e}", exc_info=True)

    def query_kg(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
        chapter_limit: Optional[int] = None,
        include_provisional: bool = True # ****** NEW ARGUMENT to control fetching provisional data ******
    ) -> List[Dict[str, Any]]:
        query_parts = [
            "SELECT id, subject, predicate, object, chapter_added, confidence, timestamp, is_provisional FROM knowledge_graph WHERE 1=1" # Add is_provisional
        ]
        params: List[Any] = []

        if subject is not None:
            query_parts.append("AND subject = ?")
            params.append(subject.strip())
        if predicate is not None:
            query_parts.append("AND predicate = ?")
            params.append(predicate.strip())
        if obj is not None:
            query_parts.append("AND object = ?")
            params.append(obj.strip())
        if (
            chapter_limit is not None
            and isinstance(chapter_limit, int)
            and chapter_limit >= 0
        ):
            query_parts.append("AND chapter_added <= ?")
            params.append(chapter_limit)
        
        # ****** Filter out provisional data if not requested ******
        if not include_provisional:
            query_parts.append("AND is_provisional = FALSE")

        query_parts.append("ORDER BY chapter_added DESC, confidence DESC, timestamp DESC") # Prioritize confidence
        final_query = " ".join(query_parts)

        logger.debug(f"Executing KG query: {final_query} with params: {params}")
        results: List[Dict[str, Any]] = []
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(final_query, params)
                rows = cursor.fetchall()
                results = [dict(row) for row in rows]  # Convert sqlite3.Row to dict
            logger.debug(f"KG query returned {len(results)} results.")
        except sqlite3.Error as e:
            logger.error(f"Database error querying KG: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error querying KG: {e}", exc_info=True)
        return results

    def get_most_recent_value(
        self, subject: str, predicate: str, chapter_limit: Optional[int] = None, include_provisional: bool = False
    ) -> Optional[str]: # Add include_provisional
        """
        Helper function to get the most recent object value for a given
        subject and predicate from the KG, up to a specific chapter.
        """
        if not subject.strip() or not predicate.strip():
            logger.warning(
                f"Attempted get_most_recent_value with empty subject or predicate: S='{subject}', P='{predicate}'"
            )
            return None

        logger.debug(
            f"Getting most recent KG value for ({subject}, {predicate}, ?) up to chapter {chapter_limit} (Include Provisional: {include_provisional})"
        )
        results = self.query_kg(
            subject=subject, predicate=predicate, chapter_limit=chapter_limit, include_provisional=include_provisional # Pass flag
        )
        if results:
            most_recent_value = results[0]["object"]
            logger.debug(
                f"Found most recent value: '{most_recent_value}' from chapter {results[0]['chapter_added']}"
            )
            return str(most_recent_value)  # Ensure string
        else:
            logger.debug(
                f"No value found for ({subject}, {predicate}, ?) up to chapter {chapter_limit}"
            )
            return None
