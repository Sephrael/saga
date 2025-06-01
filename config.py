# config.py
"""
Configuration settings for the Saga Novel Generation system.
Centralizes constants for API endpoints, model names, file paths,
generation parameters, validation thresholds, and logging settings.

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

from dotenv import load_dotenv
load_dotenv() # Loads environment variables from .env file into os.environ

import os
import logging
import json
from typing import Optional, List
import numpy as np

# --- Helper function to load lists from JSON ---
def _load_list_from_json(file_path: str, default_if_missing: Optional[List[str]] = None) -> List[str]:
    """Loads a list of strings from a JSON file."""
    if default_if_missing is None:
        default_if_missing = []
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list) and all(isinstance(item, str) for item in data):
                    return data
                else:
                    logging.warning(f"Content of {file_path} is not a list of strings. Using default.")
                    return default_if_missing
        else:
            logging.warning(f"Unhinged data file not found: {file_path}. Using default.")
            return default_if_missing
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {file_path}. Using default.", exc_info=True)
        return default_if_missing
    except Exception:
        logging.error(f"Unexpected error loading {file_path}. Using default.", exc_info=True)
        return default_if_missing

# --- API and Model Configuration ---
# URL for the Ollama service used for generating embeddings.
OLLAMA_EMBED_URL: str = os.getenv("OLLAMA_EMBED_URL", "http://127.0.0.1:11434")
# Base URL for the OpenAI-compatible API used for LLM text generation.
OPENAI_API_BASE: str = os.getenv("OPENAI_API_BASE", "http://127.0.0.1:8080/v1")
# API key for the OpenAI-compatible LLM API.
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "nope")

# Embedding Model Configuration
# Name of the embedding model to use with Ollama (e.g., "nomic-embed-text:latest", "mxbai-embed-large:latest").
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")
# Expected dimension of the embeddings produced by EMBEDDING_MODEL. Critical for Neo4j index and numpy operations.
EXPECTED_EMBEDDING_DIM: int = int(os.getenv("EXPECTED_EMBEDDING_DIM", "768")) # MODIFIED: Allow int from env
# Numpy data type for storing embeddings. float32 is common.
EMBEDDING_DTYPE: np.dtype = np.dtype(np.float32)

# Neo4j Connection Settings
# URI for connecting to the Neo4j database (e.g., "bolt://localhost:7687", "neo4j://localhost:7687").
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
# Username for Neo4j authentication.
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
# Password for Neo4j authentication.
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "saga_password")
# Name of the Neo4j database to use. Set to None or "neo4j" for the default database.
NEO4J_DATABASE: Optional[str] = os.getenv("NEO4J_DATABASE", "neo4j")

# Neo4j Vector Index Configuration
# Name for the vector index in Neo4j.
NEO4J_VECTOR_INDEX_NAME: str = "chapterEmbeddings"
# Node label on which the vector index will be created (e.g., "Chapter", "Chunk").
NEO4J_VECTOR_NODE_LABEL: str = "Chapter"
# Property name on the node that stores the embedding vector (as a list of floats).
NEO4J_VECTOR_PROPERTY_NAME: str = "embedding_vector"
# Dimensions of the vector. Must match EXPECTED_EMBEDDING_DIM.
NEO4J_VECTOR_DIMENSIONS: int = EXPECTED_EMBEDDING_DIM
# Similarity function for the vector index (e.g., 'cosine', 'euclidean', 'dot_product').
NEO4J_VECTOR_SIMILARITY_FUNCTION: str = "cosine"

# Model Aliases & Defaults
# Default model names if not overridden by environment variables.
# These are just examples; actual model names depend on what's available in your LLM server.
LARGE_MODEL_DEFAULT: str = "Qwen3-14B-Q4" # Example: A large model alias
MEDIUM_MODEL_DEFAULT: str = "Qwen3-8B-Q4" # Example: A medium model alias
SMALL_MODEL_DEFAULT: str = "Qwen3-4B-Q4"  # Example: A small model alias
NARRATOR_MODEL_DEFAULT: str = "Qwen3-14B-Q4" # Model primarily used for creative writing/drafting.

# Actual model names to be used by the system, loaded from environment or defaults.
LARGE_MODEL: str = os.getenv("LARGE_MODEL", LARGE_MODEL_DEFAULT)
MEDIUM_MODEL: str = os.getenv("MEDIUM_MODEL", MEDIUM_MODEL_DEFAULT)
SMALL_MODEL: str = os.getenv("SMALL_MODEL", SMALL_MODEL_DEFAULT)
NARRATOR_MODEL: str = os.getenv("NARRATOR_MODEL", NARRATOR_MODEL_DEFAULT) # Often same as LARGE_MODEL or a specialized writing model.

# --- LLM Call Settings & Fallbacks ---
# Number of retry attempts for LLM API calls if they fail.
LLM_RETRY_ATTEMPTS: int = int(os.getenv("LLM_RETRY_ATTEMPTS", "3"))
# Initial delay in seconds before retrying a failed LLM call (uses exponential backoff).
LLM_RETRY_DELAY_SECONDS: float = 3.0
# Model to use as a fallback if the primary LLM call fails and fallback is allowed.
FALLBACK_GENERATION_MODEL: str = MEDIUM_MODEL # Often a smaller, faster model.

# Specific model assignments for different tasks
# Model used for generating text (can be overridden by task-specific models below).
MAIN_GENERATION_MODEL: str = NARRATOR_MODEL
# Model used for knowledge graph updates and information extraction.
KNOWLEDGE_UPDATE_MODEL: str = MEDIUM_MODEL
# Model used for initial setup tasks like generating plot outline, world building, character profiles.
INITIAL_SETUP_MODEL: str = MEDIUM_MODEL
# Model used for agentic planning (e.g., scene planning).
PLANNING_MODEL: str = LARGE_MODEL
# Model used for drafting chapter text.
DRAFTING_MODEL: str = NARRATOR_MODEL
# Model used for revising chapter drafts.
REVISION_MODEL: str = NARRATOR_MODEL
# Model used for evaluating chapter drafts.
EVALUATION_MODEL: str = LARGE_MODEL
# Model used for generating text patches during revision.
PATCH_GENERATION_MODEL: str = MEDIUM_MODEL

# Task-specific Temperatures
# Creative tasks: Higher temperature for more varied output
TEMPERATURE_INITIAL_SETUP: float = float(os.getenv("TEMPERATURE_INITIAL_SETUP", "0.8"))
TEMPERATURE_DRAFTING: float = float(os.getenv("TEMPERATURE_DRAFTING", "0.8"))
TEMPERATURE_REVISION: float = float(os.getenv("TEMPERATURE_REVISION", "0.65")) # Slightly less than pure drafting
TEMPERATURE_PLANNING: float = float(os.getenv("TEMPERATURE_PLANNING", "0.6")) # Needs structure but also ideas

# Analytical/Extraction tasks: Lower temperature for more deterministic and factual output
TEMPERATURE_EVALUATION: float = float(os.getenv("TEMPERATURE_EVALUATION", "0.3"))
TEMPERATURE_CONSISTENCY_CHECK: float = float(os.getenv("TEMPERATURE_CONSISTENCY_CHECK", "0.2")) # Very factual
TEMPERATURE_KG_EXTRACTION: float = float(os.getenv("TEMPERATURE_KG_EXTRACTION", "0.4"))
TEMPERATURE_SUMMARY: float = float(os.getenv("TEMPERATURE_SUMMARY", "0.5"))
TEMPERATURE_PATCH: float = float(os.getenv("TEMPERATURE_PATCH", "0.7")) # Needs to be creative but focused

# Default temperature if a specific one isn't used (fallback)
TEMPERATURE_DEFAULT: float = 0.6

# Top-P sampling parameter for LLM generation (0.0 to 1.0). Higher is more diverse.
LLM_TOP_P: float = float(os.getenv("LLM_TOP_P", "0.8")) # MODIFIED to get from env

# --- LLM Frequency and Presence Penalties ---
# Values typically between -2.0 and 2.0. Positive values penalize, negative values encourage.
# Drafting
FREQUENCY_PENALTY_DRAFTING: float = float(os.getenv("FREQUENCY_PENALTY_DRAFTING", "0.3"))
PRESENCE_PENALTY_DRAFTING: float = float(os.getenv("PRESENCE_PENALTY_DRAFTING", "0.2"))
# Revision
FREQUENCY_PENALTY_REVISION: float = float(os.getenv("FREQUENCY_PENALTY_REVISION", "0.2"))
PRESENCE_PENALTY_REVISION: float = float(os.getenv("PRESENCE_PENALTY_REVISION", "0.1"))
# Patch Generation
FREQUENCY_PENALTY_PATCH: float = float(os.getenv("FREQUENCY_PENALTY_PATCH", "0.2"))
PRESENCE_PENALTY_PATCH: float = float(os.getenv("PRESENCE_PENALTY_PATCH", "0.1"))
# Planning (less likely to need strong penalties, but configurable)
FREQUENCY_PENALTY_PLANNING: float = float(os.getenv("FREQUENCY_PENALTY_PLANNING", "0.0"))
PRESENCE_PENALTY_PLANNING: float = float(os.getenv("PRESENCE_PENALTY_PLANNING", "0.0"))
# Initial Setup (e.g. plot outline)
FREQUENCY_PENALTY_INITIAL_SETUP: float = float(os.getenv("FREQUENCY_PENALTY_INITIAL_SETUP", "0.1"))
PRESENCE_PENALTY_INITIAL_SETUP: float = float(os.getenv("PRESENCE_PENALTY_INITIAL_SETUP", "0.1"))
# Evaluation (least likely to need penalties, as it's analytical)
FREQUENCY_PENALTY_EVALUATION: float = float(os.getenv("FREQUENCY_PENALTY_EVALUATION", "0.0"))
PRESENCE_PENALTY_EVALUATION: float = float(os.getenv("PRESENCE_PENALTY_EVALUATION", "0.0"))
# Knowledge Graph Extraction
FREQUENCY_PENALTY_KG_EXTRACTION: float = float(os.getenv("FREQUENCY_PENALTY_KG_EXTRACTION", "0.0"))
PRESENCE_PENALTY_KG_EXTRACTION: float = float(os.getenv("PRESENCE_PENALTY_KG_EXTRACTION", "0.0"))
# Summarization
FREQUENCY_PENALTY_SUMMARY: float = float(os.getenv("FREQUENCY_PENALTY_SUMMARY", "0.0"))
PRESENCE_PENALTY_SUMMARY: float = float(os.getenv("PRESENCE_PENALTY_SUMMARY", "0.0"))
# Consistency Check
FREQUENCY_PENALTY_CONSISTENCY_CHECK: float = float(os.getenv("FREQUENCY_PENALTY_CONSISTENCY_CHECK", "0.0"))
PRESENCE_PENALTY_CONSISTENCY_CHECK: float = float(os.getenv("PRESENCE_PENALTY_CONSISTENCY_CHECK", "0.0"))


# --- Output and File Paths ---
# Base directory for all generated novel outputs.
BASE_OUTPUT_DIR: str = "novel_output"
# File path for storing the plot outline JSON.
PLOT_OUTLINE_FILE: str = os.path.join(BASE_OUTPUT_DIR, "plot_outline.json")
# File path for storing character profiles JSON.
CHARACTER_PROFILES_FILE: str = os.path.join(BASE_OUTPUT_DIR, "character_profiles.json")
# File path for storing world building data JSON.
WORLD_BUILDER_FILE: str = os.path.join(BASE_OUTPUT_DIR, "world_building.json")
# Directory for storing generated chapter text files.
CHAPTERS_DIR: str = os.path.join(BASE_OUTPUT_DIR, "chapters")
# Directory for storing raw LLM logs for each chapter.
CHAPTER_LOGS_DIR: str = os.path.join(BASE_OUTPUT_DIR, "chapter_logs")
# Directory for storing various debug outputs from different stages.
DEBUG_OUTPUTS_DIR: str = os.path.join(BASE_OUTPUT_DIR, "debug_outputs")

# Path to the user's Markdown input file containing story elements.
USER_STORY_ELEMENTS_FILE_PATH: str = "user_story_elements.md"

# Directory for "unhinged mode" data files (e.g., lists of genres, themes).
UNHINGED_DATA_DIR: str = "unhinged_data"
os.makedirs(UNHINGED_DATA_DIR, exist_ok=True) # Ensure directory exists

# File paths for unhinged mode JSON data.
UNHINGED_GENRES_FILE: str = os.path.join(UNHINGED_DATA_DIR, "unhinged_genres.json")
UNHINGED_THEMES_FILE: str = os.path.join(UNHINGED_DATA_DIR, "unhinged_themes.json")
UNHINGED_SETTINGS_FILE: str = os.path.join(UNHINGED_DATA_DIR, "unhinged_settings_archetypes.json")
UNHINGED_PROTAGONISTS_FILE: str = os.path.join(UNHINGED_DATA_DIR, "unhinged_protagonist_archetypes.json")
UNHINGED_CONFLICTS_FILE: str = os.path.join(UNHINGED_DATA_DIR, "unhinged_conflict_types.json")

# Ensure output directories exist.
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
os.makedirs(CHAPTERS_DIR, exist_ok=True)
os.makedirs(CHAPTER_LOGS_DIR, exist_ok=True)
os.makedirs(DEBUG_OUTPUTS_DIR, exist_ok=True)


# --- Generation Parameters ---
# Maximum context tokens the LLM can handle for prompts (check your model's limits).
MAX_CONTEXT_TOKENS: int = int(os.getenv("MAX_CONTEXT_TOKENS", "40960")) # Example: 40k, common for some models
# Maximum tokens to request for LLM generation (output length). None means let model decide or use client's default.
MAX_GENERATION_TOKENS: int = int(os.getenv("MAX_GENERATION_TOKENS", "16384")) # Example: 16k, often a model max.
# Number of previous chapters to consider for semantic context (used by Neo4j vector search limit and fallback).
CONTEXT_CHAPTER_COUNT: int = 5
# Number of chapters to attempt to generate in a single run of the orchestrator.
CHAPTERS_PER_RUN: int = int(os.getenv("CHAPTERS_PER_RUN", "3"))
# Target number of plot points for the initial plot outline generation.
TARGET_PLOT_POINTS_INITIAL_GENERATION: int = int(os.getenv("TARGET_PLOT_POINTS_INITIAL_GENERATION", "12"))


# --- Caching ---
# Maximum size for the LRU cache storing generated embeddings.
EMBEDDING_CACHE_SIZE: int = 128
# Maximum size for the LRU cache storing chapter summaries.
SUMMARY_CACHE_SIZE: int = 32
# Maximum size for the LRU cache storing KG triple extraction results (if implemented).
KG_TRIPLE_EXTRACTION_CACHE_SIZE: int = 16
# Maximum size for the LRU cache storing tokenizers.
TOKENIZER_CACHE_SIZE: int = 10

# --- Agentic Planning & Prompt Context Snippets ---
# Toggle for enabling/disabling agentic scene planning.
ENABLE_AGENTIC_PLANNING: bool = os.getenv("ENABLE_AGENTIC_PLANNING", "True").lower() == "true"
# Maximum tokens allowed for the planning LLM call's prompt.
MAX_PLANNING_TOKENS: int = int(os.getenv("MAX_PLANNING_TOKENS", "8192"))
# Target minimum number of scenes per chapter for the planner.
TARGET_SCENES_MIN: int = int(os.getenv("TARGET_SCENES_MIN", "3"))
# Target maximum number of scenes per chapter for the planner.
TARGET_SCENES_MAX: int = int(os.getenv("TARGET_SCENES_MAX", "7"))
# Max characters for character descriptions in planning context snippets.
PLANNING_CONTEXT_MAX_CHARS_PER_PROFILE_DESC: int = 80
# Max characters for recent development notes in planning context snippets.
PLANNING_CONTEXT_MAX_RECENT_DEV_PER_PROFILE: int = 120
# Max number of characters to include in planning context snippets.
PLANNING_CONTEXT_MAX_CHARACTERS_IN_SNIPPET: int = 5
# Max number of locations to include in planning context snippets.
PLANNING_CONTEXT_MAX_LOCATIONS_IN_SNIPPET: int = 3
# Max number of factions to include in planning context snippets.
PLANNING_CONTEXT_MAX_FACTIONS_IN_SNIPPET: int = 2
# Max number of systems to include in planning context snippets.
PLANNING_CONTEXT_MAX_SYSTEMS_IN_SNIPPET: int = 2


# --- Revision and Validation ---
# Toggle for enabling/disabling patch-based revision.
ENABLE_PATCH_BASED_REVISION: bool = os.getenv("ENABLE_PATCH_BASED_REVISION", "True").lower() == "true"
# Maximum number of patch instructions to generate per revision cycle.
MAX_PATCH_INSTRUCTIONS_TO_GENERATE: int = int(os.getenv("MAX_PATCH_INSTRUCTIONS_TO_GENERATE", "3"))
# Max character window for the context provided to the patch generation LLM.
MAX_CHARS_FOR_PATCH_CONTEXT_WINDOW: int = 8192
# Coherence score threshold (cosine similarity with previous chapter) below which revision might be triggered.
REVISION_COHERENCE_THRESHOLD: float = float(os.getenv("REVISION_COHERENCE_THRESHOLD", "0.60"))
# If patched/rewritten text is this similar (or more) to the original, log a warning.
REVISION_SIMILARITY_ACCEPTANCE: float = float(os.getenv("REVISION_SIMILARITY_ACCEPTANCE", "0.995"))
# Maximum tokens for the summary generation LLM call's prompt.
MAX_SUMMARY_TOKENS: int = int(os.getenv("MAX_SUMMARY_TOKENS", "4096"))
# Maximum tokens for the KG triple extraction LLM call's prompt.
MAX_KG_TRIPLE_TOKENS: int = int(os.getenv("MAX_KG_TRIPLE_TOKENS", "8192"))
# Maximum tokens for the KG pre-population LLM call's prompt (if LLM is used for this).
MAX_PREPOP_KG_TOKENS: int = int(os.getenv("MAX_PREPOP_KG_TOKENS", "16384"))

# Default minimum acceptable length for a chapter draft (in characters).
MIN_ACCEPTABLE_DRAFT_LENGTH_DEFAULT = 12000
# Actual minimum draft length, loaded from environment or default.
MIN_ACCEPTABLE_DRAFT_LENGTH: int = int(os.getenv("MIN_ACCEPTABLE_DRAFT_LENGTH", str(MIN_ACCEPTABLE_DRAFT_LENGTH_DEFAULT)))
# Warn if user sets a very high minimum draft length.
if MIN_ACCEPTABLE_DRAFT_LENGTH > MIN_ACCEPTABLE_DRAFT_LENGTH_DEFAULT + 2000:
    logging.warning(
        f"MIN_ACCEPTABLE_DRAFT_LENGTH is set to {MIN_ACCEPTABLE_DRAFT_LENGTH}, "
        f"which is significantly higher than the default {MIN_ACCEPTABLE_DRAFT_LENGTH_DEFAULT}. "
        "This may lead to very long chapter drafts (e.g., 30K+ characters)."
    )

# Toggle for enabling dynamic state adaptation (e.g., LLM proposing 'MODIFY key: value' changes).
ENABLE_DYNAMIC_STATE_ADAPTATION: bool = True # Currently not env configurable.
# Chapter number assigned to items from initial setup/prepopulation (e.g., KG facts from initial world/char setup).
KG_PREPOPULATION_CHAPTER_NUM: int = 1


# --- De-duplication Configuration ---
DEDUPLICATION_USE_SEMANTIC: bool = os.getenv("DEDUPLICATION_USE_SEMANTIC", "True").lower() == 'true'
DEDUPLICATION_SEMANTIC_THRESHOLD: float = float(os.getenv("DEDUPLICATION_SEMANTIC_THRESHOLD", "0.90"))
DEDUPLICATION_MIN_SEGMENT_LENGTH: int = int(os.getenv("DEDUPLICATION_MIN_SEGMENT_LENGTH", "150")) # Characters


# --- Logging ---
# Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
LOG_LEVEL_STR: str = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_LEVEL: int = getattr(logging, LOG_LEVEL_STR, logging.INFO)
# Format string for log messages.
LOG_FORMAT: str = '%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s'
# Format string for timestamps in log messages.
LOG_DATE_FORMAT: str = '%Y-%m-%d %H:%M:%S'
# Path to the log file. If None, logs only to console.
LOG_FILE: Optional[str] = os.path.join(BASE_OUTPUT_DIR, "saga_run.log")


# --- Novel Configuration (Defaults / Placeholders) ---
# Toggle for "unhinged plot mode" - generates more random/surprising plot elements if user input is minimal.
UNHINGED_PLOT_MODE: bool = os.getenv("UNHINGED_PLOT_MODE", "False").lower() == "true"
# Default genre if not supplied by user or unhinged mode.
CONFIGURED_GENRE: str = "gritty fantasy"
# Default theme if not supplied by user or unhinged mode.
CONFIGURED_THEME: str = "the cost of historical revisionism"
# Default setting description if not supplied.
CONFIGURED_SETTING_DESCRIPTION: str = "a megastructure constructed around a star to harness its energy, now partially abandoned"
# Default protagonist name.
DEFAULT_PROTAGONIST_NAME: str = "Sága"
# Default title for the plot outline.
DEFAULT_PLOT_OUTLINE_TITLE: str = "Untitled Saga"

# Neo4j Node IDs for main container/info nodes (used for consistent querying).
MAIN_NOVEL_INFO_NODE_ID: str = "saga_main_novel_info"
MAIN_CHARACTERS_CONTAINER_NODE_ID: str = "saga_main_characters_container" # Not currently used but reserved
MAIN_WORLD_CONTAINER_NODE_ID: str = "saga_main_world_container"


# --- Unhinged Mode Data (Loaded from JSON files) ---
# Default lists used if JSON files for unhinged mode are missing or invalid.
_DEFAULT_GENRE_LIST = ["science fiction", "fantasy", "horror"]
UNHINGED_GENRES: List[str] = _load_list_from_json(UNHINGED_GENRES_FILE, _DEFAULT_GENRE_LIST)
UNHINGED_THEMES: List[str] = _load_list_from_json(UNHINGED_THEMES_FILE, ["the nature of reality", "the cost of power"])
UNHINGED_SETTINGS_ARCHETYPES: List[str] = _load_list_from_json(UNHINGED_SETTINGS_FILE, ["a floating city", "a derelict starship"])
UNHINGED_PROTAGONIST_ARCHETYPES: List[str] = _load_list_from_json(UNHINGED_PROTAGONISTS_FILE, ["a reluctant hero", "a cynical detective"])
UNHINGED_CONFLICT_TYPES: List[str] = _load_list_from_json(UNHINGED_CONFLICTS_FILE, ["man vs self", "man vs society"])

# Log a warning if default unhinged genre list is used.
if not UNHINGED_GENRES or UNHINGED_GENRES == _DEFAULT_GENRE_LIST:
    logging.warning("UNHINGED_GENRES might be using default values. Check unhinged_genres.json.")

# --- Tokenizer Fallback Configuration ---
# Default tiktoken encoding model to use if a specific model's tokenizer isn't found.
TIKTOKEN_DEFAULT_ENCODING: str = "cl100k_base" # Common for many OpenAI models
# Estimated average characters per token, used as a fallback if tiktoken fails.
FALLBACK_CHARS_PER_TOKEN: float = 4.0

# --- Rich Progress Display Configuration ---
# Toggle for enabling Rich library for enhanced console progress display.
ENABLE_RICH_PROGRESS: bool = os.getenv("ENABLE_RICH_PROGRESS", "True").lower() == "true"
# How many times per second the Rich Live display should attempt to refresh.
RICH_REFRESH_PER_SECOND: float = 4.0 # Higher values refresh faster but use more CPU.

# --- Markdown Story Parser Configuration ---
# Placeholder string used in user_story_elements.md to indicate SAGA should generate this field.
# This value is checked by initial_setup_logic and other parsers.
MARKDOWN_FILL_IN_PLACEHOLDER: str = "[Fill-in]"