# database_manager.py
"""
Manages all interactions with the SQLite database for the Saga project.
Handles connection, schema initialization, and CRUD operations for
chapter text, summaries, embeddings, and the knowledge graph.
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
                # Consider raising an error depending on desired behavior
                # raise

        # Initialize the database schema upon instantiation
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """
        Establishes and returns a database connection.

        Sets row_factory to sqlite3.Row for dictionary-like row access.
        Includes a timeout and enables foreign keys.

        Returns:
            A sqlite3.Connection object.

        Raises:
            sqlite3.Error: If the connection cannot be established.
        """
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0) # Timeout helps prevent locking issues
            conn.row_factory = sqlite3.Row # Access columns by name
            conn.execute("PRAGMA foreign_keys = ON;") # Enable foreign key support
            logger.debug(f"Database connection established to {self.db_path}")
            return conn
        except sqlite3.Error as e:
            logger.error(f"Database connection error to {self.db_path}: {e}", exc_info=True)
            raise

    def _init_db(self):
        """
        Initializes the SQLite database schema, including the new knowledge_graph table.
        Creates tables and indices if they don't already exist.
        """
        logger.info(f"Initializing/Verifying database schema at {self.db_path}...")
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # --- Chapters Table (No changes) ---
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS chapters (
                        chapter_number INTEGER PRIMARY KEY,
                        text TEXT,
                        raw_text TEXT,
                        summary TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                logger.debug("Verified/Created 'chapters' table.")

                # --- Embeddings Table (No changes) ---
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS embeddings (
                        chapter_number INTEGER PRIMARY KEY,
                        embedding_blob BLOB NOT NULL,
                        dtype TEXT NOT NULL,
                        shape TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (chapter_number) REFERENCES chapters (chapter_number) ON DELETE CASCADE
                    )
                ''')
                logger.debug("Verified/Created 'embeddings' table.")

                # --- NEW: Knowledge Graph Table ---
                # Stores factual triples extracted from chapters.
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS knowledge_graph (
                        id INTEGER PRIMARY KEY AUTOINCREMENT, -- Unique ID for each triple
                        subject TEXT NOT NULL,                 -- The entity the fact is about (e.g., 'Sága', 'Eclipse Spire')
                        predicate TEXT NOT NULL,               -- The relationship or property (e.g., 'is_a', 'located_in', 'has_trait')
                        object TEXT NOT NULL,                  -- The value or target entity (e.g., 'AI', 'Nova-7', 'Analytical')
                        chapter_added INTEGER NOT NULL,        -- Chapter number when this fact was added/asserted
                        confidence REAL DEFAULT 1.0,           -- Optional: Confidence score from LLM extraction (default 1.0)
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP -- When the triple was added
                    )
                ''')
                logger.debug("Verified/Created 'knowledge_graph' table.")

                # --- Indices ---
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_embeddings_chapter ON embeddings (chapter_number);')
                # NEW Indices for knowledge_graph for faster querying
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_kg_subject ON knowledge_graph (subject);')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_kg_predicate ON knowledge_graph (predicate);')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_kg_object ON knowledge_graph (object);')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_kg_spo ON knowledge_graph (subject, predicate, object);') # For full triple lookups
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_kg_chapter ON knowledge_graph (chapter_added);')
                logger.debug("Verified/Created indices on 'embeddings' and 'knowledge_graph' tables.")

                conn.commit()
            logger.info("Database schema initialization/verification complete.")
        except sqlite3.Error as e:
            logger.error(f"Database schema initialization error: {e}", exc_info=True)
            raise

    # --- Chapter Count and Data Saving/Retrieval (Largely Unchanged) ---

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
            logger.error(f"Failed to load chapter count from DB: {e}. Assuming 0.", exc_info=True)
            return 0

    def save_chapter_data(self, chapter_number: int, text: str, raw_text: str, summary: Optional[str], embedding: Optional[np.ndarray]):
        """Saves or updates chapter text, raw log, summary, and embedding."""
        logger.info(f"Attempting to save data for chapter {chapter_number} to database.")
        embedding_blob, dtype_str, shape_str = None, "", ""
        if embedding is not None and isinstance(embedding, np.ndarray) and embedding.size > 0:
            try:
                embedding = embedding.astype(config.EMBEDDING_DTYPE)
                if embedding.ndim == 0: embedding = embedding.reshape(1)
                embedding_blob = embedding.tobytes()
                dtype_str = str(embedding.dtype)
                shape_str = json.dumps(embedding.shape)
                logger.debug(f"Serialized embedding for chapter {chapter_number} (Shape: {shape_str}, Dtype: {dtype_str})")
            except Exception as e:
                logger.error(f"Error serializing embedding for chapter {chapter_number}: {e}", exc_info=True)
                embedding_blob = None
        else:
            logger.warning(f"Chapter {chapter_number} has no valid embedding provided. Embedding will not be saved.")

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # Save/Update Chapter Text and Summary
                cursor.execute(
                    "INSERT OR REPLACE INTO chapters (chapter_number, text, raw_text, summary) VALUES (?, ?, ?, ?)",
                    (chapter_number, text, raw_text, summary if summary is not None else '')
                )
                logger.debug(f"Saved/Replaced text/summary data for chapter {chapter_number}.")

                # Save/Update Embedding
                if embedding_blob:
                    cursor.execute(
                        "INSERT OR REPLACE INTO embeddings (chapter_number, embedding_blob, dtype, shape) VALUES (?, ?, ?, ?)",
                        (chapter_number, embedding_blob, dtype_str, shape_str)
                    )
                    logger.debug(f"Saved/Replaced embedding data for chapter {chapter_number}.")
                else:
                    # Remove old embedding if new one is invalid/missing
                    cursor.execute("DELETE FROM embeddings WHERE chapter_number = ?", (chapter_number,))
                    logger.debug(f"Removed any existing embedding entry for chapter {chapter_number}.")
                conn.commit()
            logger.info(f"Successfully saved chapter text/embedding data for chapter {chapter_number}.")
        except sqlite3.Error as e:
            logger.error(f"Database error saving chapter data for chapter {chapter_number}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error saving chapter data for chapter {chapter_number}: {e}", exc_info=True)

    def get_embedding_from_db(self, chapter_number: int) -> Optional[np.ndarray]:
        """Retrieves and deserializes a single chapter's embedding."""
        logger.debug(f"Retrieving embedding for chapter {chapter_number} from database.")
        embedding = None
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT embedding_blob, dtype, shape FROM embeddings WHERE chapter_number = ?",
                    (chapter_number,)
                )
                result = cursor.fetchone()
            if result and result["embedding_blob"]:
                try:
                    blob, dtype_str, shape_str = result["embedding_blob"], result["dtype"], result["shape"]
                    shape = tuple(json.loads(shape_str))
                    dtype = np.dtype(dtype_str)
                    embedding = np.frombuffer(blob, dtype=dtype).reshape(shape)
                    logger.debug(f"Successfully retrieved and deserialized embedding for chapter {chapter_number}.")
                except Exception as e:
                    logger.error(f"Error deserializing embedding for chapter {chapter_number}: {e}", exc_info=True)
                    embedding = None
            else:
                logger.debug(f"No embedding found in database for chapter {chapter_number}.")
        except sqlite3.Error as e:
            logger.error(f"Database error retrieving embedding for chapter {chapter_number}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error retrieving embedding for chapter {chapter_number}: {e}", exc_info=True)
        return embedding

    def get_all_past_embeddings(self, current_chapter_number: int) -> List[Tuple[int, np.ndarray]]:
        """Retrieves all embeddings for chapters before the specified chapter number."""
        logger.debug(f"Retrieving all past embeddings before chapter {current_chapter_number}.")
        past_embeddings: List[Tuple[int, np.ndarray]] = []
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """SELECT chapter_number, embedding_blob, dtype, shape
                       FROM embeddings
                       WHERE chapter_number < ?
                       ORDER BY chapter_number DESC""",
                    (current_chapter_number,)
                )
                results = cursor.fetchall()
            logger.debug(f"Found {len(results)} past embedding entries in database.")
            for row in results:
                try:
                    chap_num, blob, dtype_str, shape_str = row["chapter_number"], row["embedding_blob"], row["dtype"], row["shape"]
                    shape = tuple(json.loads(shape_str))
                    dtype = np.dtype(dtype_str)
                    embedding = np.frombuffer(blob, dtype=dtype).reshape(shape)
                    past_embeddings.append((chap_num, embedding))
                except Exception as e:
                    logger.warning(f"Failed to load/deserialize embedding for chapter {row['chapter_number']}: {e}")
        except sqlite3.Error as e:
            logger.error(f"Database error retrieving past embeddings: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error retrieving past embeddings: {e}", exc_info=True)
        logger.info(f"Successfully retrieved and deserialized {len(past_embeddings)} past embeddings.")
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
                    (chapter_number,)
                )
                result = cursor.fetchone()
            if result:
                data = dict(result)
                logger.debug(f"Found text/summary data for chapter {chapter_number}.")
            else:
                 logger.debug(f"No text/summary data found for chapter {chapter_number}.")
        except sqlite3.Error as e:
            logger.error(f"Database error retrieving text/summary for chapter {chapter_number}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error retrieving chapter data for chapter {chapter_number}: {e}", exc_info=True)
        return data

    # --- NEW: Knowledge Graph Methods ---

    def add_kg_triple(self, subject: str, predicate: str, obj: str, chapter_added: int, confidence: float = 1.0):
        """
        Adds a single (subject, predicate, object) triple to the knowledge graph.
        Avoids adding exact duplicates for the same chapter.

        Args:
            subject: The subject of the triple.
            predicate: The predicate (relationship).
            obj: The object of the triple.
            chapter_added: The chapter number where this fact was asserted.
            confidence: Optional confidence score (0.0 to 1.0).
        """
        # Basic validation
        if not all([subject, predicate, obj]) or chapter_added <= 0:
            logger.warning(f"Attempted to add invalid triple: S='{subject}', P='{predicate}', O='{obj}', Chap={chapter_added}")
            return

        subj_clean = subject.strip()
        pred_clean = predicate.strip()
        obj_clean = obj.strip()

        if not all([subj_clean, pred_clean, obj_clean]):
             logger.warning(f"Attempted to add triple with empty components after stripping: S='{subj_clean}', P='{pred_clean}', O='{obj_clean}', Chap={chapter_added}")
             return

        logger.debug(f"Adding KG triple: ({subj_clean}, {pred_clean}, {obj_clean}) from chapter {chapter_added}")
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # Check if this exact triple already exists for this chapter to avoid duplicates
                cursor.execute(
                    """SELECT id FROM knowledge_graph
                       WHERE subject = ? AND predicate = ? AND object = ? AND chapter_added = ?""",
                    (subj_clean, pred_clean, obj_clean, chapter_added)
                )
                exists = cursor.fetchone()

                if exists:
                    logger.debug(f"Triple ({subj_clean}, {pred_clean}, {obj_clean}) already exists for chapter {chapter_added}. Skipping insert.")
                else:
                    # Insert the new triple
                    cursor.execute(
                        """INSERT INTO knowledge_graph (subject, predicate, object, chapter_added, confidence)
                           VALUES (?, ?, ?, ?, ?)""",
                        (subj_clean, pred_clean, obj_clean, chapter_added, confidence)
                    )
                    conn.commit()
                    logger.debug(f"Successfully added triple for chapter {chapter_added}.")

        except sqlite3.Error as e:
            logger.error(f"Database error adding KG triple ({subj_clean}, {pred_clean}, {obj_clean}): {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error adding KG triple: {e}", exc_info=True)

    def query_kg(self, subject: Optional[str] = None, predicate: Optional[str] = None, obj: Optional[str] = None, chapter_limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Queries the knowledge graph based on provided components (S, P, O).
        Allows filtering up to a specific chapter number.
        Returns results as a list of dictionaries.

        Args:
            subject: The subject to match (optional).
            predicate: The predicate to match (optional).
            obj: The object to match (optional).
            chapter_limit: If provided, only return triples added up to this chapter number (inclusive).

        Returns:
            A list of dictionaries, where each dictionary represents a matching triple
            (keys: 'id', 'subject', 'predicate', 'object', 'chapter_added', 'confidence', 'timestamp').
            Returns an empty list if no matches are found or on error.
        """
        query = "SELECT id, subject, predicate, object, chapter_added, confidence, timestamp FROM knowledge_graph WHERE 1=1"
        params = []

        # Build the WHERE clause dynamically based on provided arguments
        if subject is not None:
            query += " AND subject = ?"
            params.append(subject.strip())
        if predicate is not None:
            query += " AND predicate = ?"
            params.append(predicate.strip())
        if obj is not None:
            query += " AND object = ?"
            params.append(obj.strip())
        if chapter_limit is not None and isinstance(chapter_limit, int) and chapter_limit >= 0:
            query += " AND chapter_added <= ?"
            params.append(chapter_limit)

        # Order by chapter DESC then timestamp DESC to get most recent facts first
        query += " ORDER BY chapter_added DESC, timestamp DESC"

        logger.debug(f"Executing KG query: {query} with params: {params}")
        results: List[Dict[str, Any]] = []
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()
                # Convert sqlite3.Row objects to standard dictionaries
                results = [dict(row) for row in rows]
            logger.debug(f"KG query returned {len(results)} results.")
        except sqlite3.Error as e:
            logger.error(f"Database error querying KG: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error querying KG: {e}", exc_info=True)

        return results

    def get_most_recent_value(self, subject: str, predicate: str, chapter_limit: Optional[int] = None) -> Optional[str]:
        """
        Helper function to get the most recent object value for a given
        subject and predicate from the KG, up to a specific chapter.

        Args:
            subject: The subject entity.
            predicate: The predicate (relationship/property).
            chapter_limit: The maximum chapter number to consider.

        Returns:
            The object value (string) of the most recent matching triple, or None if not found.
        """
        logger.debug(f"Getting most recent KG value for ({subject}, {predicate}, ?) up to chapter {chapter_limit}")
        results = self.query_kg(subject=subject, predicate=predicate, chapter_limit=chapter_limit)
        if results:
            # Results are ordered by chapter DESC, timestamp DESC, so the first one is the most recent
            most_recent_value = results[0]['object']
            logger.debug(f"Found most recent value: '{most_recent_value}' from chapter {results[0]['chapter_added']}")
            return most_recent_value
        else:
            logger.debug(f"No value found for ({subject}, {predicate}, ?) up to chapter {chapter_limit}")
            return None

