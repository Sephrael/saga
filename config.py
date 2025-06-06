# config_refactored.py
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

import asyncio
import json
import logging as stdlib_logging  # For configuring structlog's underlying logger
import os
from typing import List, Optional

import numpy as np
import structlog
from dotenv import load_dotenv

load_dotenv()  # Loads environment variables from .env file into os.environ

logger = structlog.get_logger()


# --- Helper function to load lists from JSON ---
async def _load_list_from_json_async(
    file_path: str, default_if_missing: Optional[List[str]] = None
) -> List[str]:
    """Loads a list of strings from a JSON file asynchronously."""
    if default_if_missing is None:
        default_if_missing = []
    try:
        # Note: os.path.exists is synchronous. For a truly async app,
        # consider using aiofiles for async file operations, especially if this
        # function were called frequently outside of startup.
        if os.path.exists(file_path):
            # Using sync open within async function; acceptable for startup config.
            # For high-frequency I/O, use aiofiles.open.
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and all(
                    isinstance(item, str) for item in data
                ):
                    return data
                else:
                    logger.warning(
                        "Content of file is not a list of strings. Using default.",
                        file_path=file_path,
                    )
                    return default_if_missing
        else:
            logger.warning(
                "Configuration file not found. Using default.", file_path=file_path
            )
            return default_if_missing
    except json.JSONDecodeError:
        logger.error(
            "Error decoding JSON from file. Using default.",
            file_path=file_path,
            exc_info=True,
        )
        return default_if_missing
    except Exception:
        logger.error(
            "Unexpected error loading file. Using default.",
            file_path=file_path,
            exc_info=True,
        )
        return default_if_missing


# --- API and Model Configuration ---
# URL for the Ollama service used for generating embeddings.
OLLAMA_EMBED_URL: str = os.getenv("OLLAMA_EMBED_URL", "http://127.0.0.1:11434")
# Base URL for the OpenAI-compatible API used for LLM text generation.
OPENAI_API_BASE: str = os.getenv("OPENAI_API_BASE", "http://127.0.0.1:8080/v1")
# API key for the OpenAI-compatible LLM API.
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "nope")

# Embedding Model Configuration
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")
EXPECTED_EMBEDDING_DIM: int = int(os.getenv("EXPECTED_EMBEDDING_DIM", "768"))
EMBEDDING_DTYPE: np.dtype = np.dtype(np.float32)

# Neo4j Connection Settings
NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "saga_password")
NEO4J_DATABASE: Optional[str] = os.getenv("NEO4J_DATABASE", "neo4j")

# Neo4j Vector Index Configuration
NEO4J_VECTOR_INDEX_NAME: str = "chapterEmbeddings"
NEO4J_VECTOR_NODE_LABEL: str = "Chapter"
NEO4J_VECTOR_PROPERTY_NAME: str = "embedding_vector"
NEO4J_VECTOR_DIMENSIONS: int = EXPECTED_EMBEDDING_DIM
NEO4J_VECTOR_SIMILARITY_FUNCTION: str = "cosine"

# Model Aliases & Defaults
LARGE_MODEL_DEFAULT: str = "Qwen3-14B-Q4"
MEDIUM_MODEL_DEFAULT: str = "Qwen3-8B-Q4"
SMALL_MODEL_DEFAULT: str = "Qwen3-4B-Q4"
NARRATOR_MODEL_DEFAULT: str = "Qwen3-14B-Q4"

LARGE_MODEL: str = os.getenv("LARGE_MODEL", LARGE_MODEL_DEFAULT)
MEDIUM_MODEL: str = os.getenv("MEDIUM_MODEL", MEDIUM_MODEL_DEFAULT)
SMALL_MODEL: str = os.getenv("SMALL_MODEL", SMALL_MODEL_DEFAULT)
NARRATOR_MODEL: str = os.getenv("NARRATOR_MODEL", NARRATOR_MODEL_DEFAULT)

# --- LLM Call Settings & Fallbacks ---
LLM_RETRY_ATTEMPTS: int = int(os.getenv("LLM_RETRY_ATTEMPTS", "3"))
LLM_RETRY_DELAY_SECONDS: float = 3.0
HTTPX_TIMEOUT: float = float(os.getenv("HTTPX_TIMEOUT", "600.0"))
FALLBACK_GENERATION_MODEL: str = MEDIUM_MODEL
ENABLE_LLM_NO_THINK_DIRECTIVE: bool = (
    os.getenv("ENABLE_LLM_NO_THINK_DIRECTIVE", "True").lower() == "true"
)

# Specific model assignments for different tasks
MAIN_GENERATION_MODEL: str = NARRATOR_MODEL
KNOWLEDGE_UPDATE_MODEL: str = MEDIUM_MODEL
INITIAL_SETUP_MODEL: str = MEDIUM_MODEL
PLANNING_MODEL: str = LARGE_MODEL
DRAFTING_MODEL: str = NARRATOR_MODEL
REVISION_MODEL: str = NARRATOR_MODEL
EVALUATION_MODEL: str = LARGE_MODEL
PATCH_GENERATION_MODEL: str = MEDIUM_MODEL

# Task-specific Temperatures
TEMPERATURE_INITIAL_SETUP: float = float(os.getenv("TEMPERATURE_INITIAL_SETUP", "0.8"))
TEMPERATURE_DRAFTING: float = float(os.getenv("TEMPERATURE_DRAFTING", "0.8"))
TEMPERATURE_REVISION: float = float(os.getenv("TEMPERATURE_REVISION", "0.65"))
TEMPERATURE_PLANNING: float = float(os.getenv("TEMPERATURE_PLANNING", "0.6"))
TEMPERATURE_EVALUATION: float = float(os.getenv("TEMPERATURE_EVALUATION", "0.3"))
TEMPERATURE_CONSISTENCY_CHECK: float = float(
    os.getenv("TEMPERATURE_CONSISTENCY_CHECK", "0.2")
)
TEMPERATURE_KG_EXTRACTION: float = float(os.getenv("TEMPERATURE_KG_EXTRACTION", "0.4"))
TEMPERATURE_SUMMARY: float = float(os.getenv("TEMPERATURE_SUMMARY", "0.5"))
TEMPERATURE_PATCH: float = float(os.getenv("TEMPERATURE_PATCH", "0.7"))
TEMPERATURE_DEFAULT: float = 0.6
LLM_TOP_P: float = float(os.getenv("LLM_TOP_P", "0.8"))

# --- LLM Frequency and Presence Penalties ---
FREQUENCY_PENALTY_DRAFTING: float = float(
    os.getenv("FREQUENCY_PENALTY_DRAFTING", "0.3")
)
PRESENCE_PENALTY_DRAFTING: float = float(os.getenv("PRESENCE_PENALTY_DRAFTING", "1.5"))
FREQUENCY_PENALTY_REVISION: float = float(
    os.getenv("FREQUENCY_PENALTY_REVISION", "0.2")
)
PRESENCE_PENALTY_REVISION: float = float(os.getenv("PRESENCE_PENALTY_REVISION", "1.5"))
FREQUENCY_PENALTY_PATCH: float = float(os.getenv("FREQUENCY_PENALTY_PATCH", "0.2"))
PRESENCE_PENALTY_PATCH: float = float(os.getenv("PRESENCE_PENALTY_PATCH", "1.5"))
FREQUENCY_PENALTY_PLANNING: float = float(
    os.getenv("FREQUENCY_PENALTY_PLANNING", "0.0")
)
PRESENCE_PENALTY_PLANNING: float = float(os.getenv("PRESENCE_PENALTY_PLANNING", "1.5"))
FREQUENCY_PENALTY_INITIAL_SETUP: float = float(
    os.getenv("FREQUENCY_PENALTY_INITIAL_SETUP", "0.1")
)
PRESENCE_PENALTY_INITIAL_SETUP: float = float(
    os.getenv("PRESENCE_PENALTY_INITIAL_SETUP", "1.5")
)
FREQUENCY_PENALTY_EVALUATION: float = float(
    os.getenv("FREQUENCY_PENALTY_EVALUATION", "0.0")
)
PRESENCE_PENALTY_EVALUATION: float = float(
    os.getenv("PRESENCE_PENALTY_EVALUATION", "1.5")
)
FREQUENCY_PENALTY_KG_EXTRACTION: float = float(
    os.getenv("FREQUENCY_PENALTY_KG_EXTRACTION", "0.0")
)
PRESENCE_PENALTY_KG_EXTRACTION: float = float(
    os.getenv("PRESENCE_PENALTY_KG_EXTRACTION", "1.5")
)
FREQUENCY_PENALTY_SUMMARY: float = float(os.getenv("FREQUENCY_PENALTY_SUMMARY", "0.0"))
PRESENCE_PENALTY_SUMMARY: float = float(os.getenv("PRESENCE_PENALTY_SUMMARY", "1.5"))
FREQUENCY_PENALTY_CONSISTENCY_CHECK: float = float(
    os.getenv("FREQUENCY_PENALTY_CONSISTENCY_CHECK", "0.0")
)
PRESENCE_PENALTY_CONSISTENCY_CHECK: float = float(
    os.getenv("PRESENCE_PENALTY_CONSISTENCY_CHECK", "1.5")
)

# --- Output and File Paths ---
BASE_OUTPUT_DIR: str = "novel_output"
PLOT_OUTLINE_FILE: str = os.path.join(BASE_OUTPUT_DIR, "plot_outline.json")
CHARACTER_PROFILES_FILE: str = os.path.join(
    BASE_OUTPUT_DIR, "character_profiles.json"
)
WORLD_BUILDER_FILE: str = os.path.join(BASE_OUTPUT_DIR, "world_building.json")
CHAPTERS_DIR: str = os.path.join(BASE_OUTPUT_DIR, "chapters")
CHAPTER_LOGS_DIR: str = os.path.join(BASE_OUTPUT_DIR, "chapter_logs")
DEBUG_OUTPUTS_DIR: str = os.path.join(BASE_OUTPUT_DIR, "debug_outputs")

USER_STORY_ELEMENTS_FILE_PATH: str = "user_story_elements.md"

UNHINGED_DATA_DIR: str = "unhinged_data"
os.makedirs(UNHINGED_DATA_DIR, exist_ok=True)

UNHINGED_GENRES_FILE: str = os.path.join(UNHINGED_DATA_DIR, "unhinged_genres.json")
UNHINGED_THEMES_FILE: str = os.path.join(UNHINGED_DATA_DIR, "unhinged_themes.json")
UNHINGED_SETTINGS_FILE: str = os.path.join(
    UNHINGED_DATA_DIR, "unhinged_settings_archetypes.json"
)
UNHINGED_PROTAGONISTS_FILE: str = os.path.join(
    UNHINGED_DATA_DIR, "unhinged_protagonist_archetypes.json"
)
UNHINGED_CONFLICTS_FILE: str = os.path.join(
    UNHINGED_DATA_DIR, "unhinged_conflict_types.json"
)

os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
os.makedirs(CHAPTERS_DIR, exist_ok=True)
os.makedirs(CHAPTER_LOGS_DIR, exist_ok=True)
os.makedirs(DEBUG_OUTPUTS_DIR, exist_ok=True)

# --- Generation Parameters ---
MAX_CONTEXT_TOKENS: int = int(os.getenv("MAX_CONTEXT_TOKENS", "40960"))
MAX_GENERATION_TOKENS: int = int(os.getenv("MAX_GENERATION_TOKENS", "16384"))
CONTEXT_CHAPTER_COUNT: int = 5
CHAPTERS_PER_RUN: int = int(os.getenv("CHAPTERS_PER_RUN", "3"))
TARGET_PLOT_POINTS_INITIAL_GENERATION: int = int(
    os.getenv("TARGET_PLOT_POINTS_INITIAL_GENERATION", "12")
)

# --- Caching ---
EMBEDDING_CACHE_SIZE: int = 128
SUMMARY_CACHE_SIZE: int = 32
KG_TRIPLE_EXTRACTION_CACHE_SIZE: int = 16
TOKENIZER_CACHE_SIZE: int = 10

# --- Agentic Planning & Prompt Context Snippets ---
ENABLE_AGENTIC_PLANNING: bool = (
    os.getenv("ENABLE_AGENTIC_PLANNING", "True").lower() == "true"
)
MAX_PLANNING_TOKENS: int = int(os.getenv("MAX_PLANNING_TOKENS", "8192"))
TARGET_SCENES_MIN: int = int(os.getenv("TARGET_SCENES_MIN", "3"))
TARGET_SCENES_MAX: int = int(os.getenv("TARGET_SCENES_MAX", "7"))
PLANNING_CONTEXT_MAX_CHARS_PER_PROFILE_DESC: int = 80
PLANNING_CONTEXT_MAX_RECENT_DEV_PER_PROFILE: int = 120
PLANNING_CONTEXT_MAX_CHARACTERS_IN_SNIPPET: int = 5
PLANNING_CONTEXT_MAX_LOCATIONS_IN_SNIPPET: int = 3
PLANNING_CONTEXT_MAX_FACTIONS_IN_SNIPPET: int = 2
PLANNING_CONTEXT_MAX_SYSTEMS_IN_SNIPPET: int = 2

# --- Revision and Validation ---
ENABLE_PATCH_BASED_REVISION: bool = (
    os.getenv("ENABLE_PATCH_BASED_REVISION", "True").lower() == "true"
)
MAX_PATCH_INSTRUCTIONS_TO_GENERATE: int = int(
    os.getenv("MAX_PATCH_INSTRUCTIONS_TO_GENERATE", "3")
)
MAX_CHARS_FOR_PATCH_CONTEXT_WINDOW: int = 8192
REVISION_COHERENCE_THRESHOLD: float = float(
    os.getenv("REVISION_COHERENCE_THRESHOLD", "0.60")
)
REVISION_SIMILARITY_ACCEPTANCE: float = float(
    os.getenv("REVISION_SIMILARITY_ACCEPTANCE", "0.995")
)
MAX_SUMMARY_TOKENS: int = int(os.getenv("MAX_SUMMARY_TOKENS", "4096"))
MAX_KG_TRIPLE_TOKENS: int = int(os.getenv("MAX_KG_TRIPLE_TOKENS", "8192"))
MAX_PREPOP_KG_TOKENS: int = int(os.getenv("MAX_PREPOP_KG_TOKENS", "16384"))

MIN_ACCEPTABLE_DRAFT_LENGTH_DEFAULT = 12000
MIN_ACCEPTABLE_DRAFT_LENGTH: int = int(
    os.getenv("MIN_ACCEPTABLE_DRAFT_LENGTH", str(MIN_ACCEPTABLE_DRAFT_LENGTH_DEFAULT))
)
if MIN_ACCEPTABLE_DRAFT_LENGTH > MIN_ACCEPTABLE_DRAFT_LENGTH_DEFAULT + 2000:
    logger.warning(
        "MIN_ACCEPTABLE_DRAFT_LENGTH is set significantly higher than default.",
        min_acceptable_draft_length=MIN_ACCEPTABLE_DRAFT_LENGTH,
        default_length=MIN_ACCEPTABLE_DRAFT_LENGTH_DEFAULT,
        message="This may lead to very long chapter drafts (e.g., 30K+ characters).",
    )

ENABLE_DYNAMIC_STATE_ADAPTATION: bool = True
KG_PREPOPULATION_CHAPTER_NUM: int = 0

# --- De-duplication Configuration ---
DEDUPLICATION_USE_SEMANTIC: bool = (
    os.getenv("DEDUPLICATION_USE_SEMANTIC", "True").lower() == "true"
)
DEDUPLICATION_SEMANTIC_THRESHOLD: float = float(
    os.getenv("DEDUPLICATION_SEMANTIC_THRESHOLD", "0.90")
)
DEDUPLICATION_MIN_SEGMENT_LENGTH: int = int(
    os.getenv("DEDUPLICATION_MIN_SEGMENT_LENGTH", "150")
)

# --- Logging ---
LOG_LEVEL_STR: str = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT: str = (
    "%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s"
)
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
LOG_FILE: Optional[str] = os.path.join(BASE_OUTPUT_DIR, "saga_run.log")

# Configure structlog
structlog.configure(
    processors=[
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

formatter = structlog.stdlib.ProcessorFormatter.from_processors(
    [
        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
        structlog.dev.ConsoleRenderer()  # Or structlog.processors.JSONRenderer()
    ]
)

handler = stdlib_logging.StreamHandler() # Default to console
if LOG_FILE:
    handler = stdlib_logging.FileHandler(LOG_FILE)
handler.setFormatter(formatter)
root_logger = stdlib_logging.getLogger()
root_logger.addHandler(handler)
root_logger.setLevel(LOG_LEVEL_STR)


# --- Novel Configuration (Defaults / Placeholders) ---
UNHINGED_PLOT_MODE: bool = os.getenv("UNHINGED_PLOT_MODE", "False").lower() == "true"
CONFIGURED_GENRE: str = "gritty fantasy"
CONFIGURED_THEME: str = "the cost of historical revisionism"
CONFIGURED_SETTING_DESCRIPTION: str = (
    "a megastructure constructed around a star to harness its energy, "
    "now partially abandoned"
)
DEFAULT_PROTAGONIST_NAME: str = "Sága"
DEFAULT_PLOT_OUTLINE_TITLE: str = "Untitled Saga"

MAIN_NOVEL_INFO_NODE_ID: str = "saga_main_novel_info"
MAIN_CHARACTERS_CONTAINER_NODE_ID: str = "saga_main_characters_container"
MAIN_WORLD_CONTAINER_NODE_ID: str = "saga_main_world_container"

# --- Unhinged Mode Data (Loaded from JSON files) ---
_DEFAULT_GENRE_LIST = ["science fiction", "fantasy", "horror"]
_DEFAULT_THEMES_LIST = ["the nature of reality", "the cost of power"]
_DEFAULT_SETTINGS_LIST = ["a floating city", "a derelict starship"]
_DEFAULT_PROTAGONISTS_LIST = ["a reluctant hero", "a cynical detective"]
_DEFAULT_CONFLICTS_LIST = ["man vs self", "man vs society"]

# These will be populated by load_unhinged_data_async
UNHINGED_GENRES: List[str] = []
UNHINGED_THEMES: List[str] = []
UNHINGED_SETTINGS_ARCHETYPES: List[str] = []
UNHINGED_PROTAGONIST_ARCHETYPES: List[str] = []
UNHINGED_CONFLICT_TYPES: List[str] = []


async def load_unhinged_data_async():
    """Asynchronously loads all unhinged data files."""
    global UNHINGED_GENRES, UNHINGED_THEMES, UNHINGED_SETTINGS_ARCHETYPES
    global UNHINGED_PROTAGONIST_ARCHETYPES, UNHINGED_CONFLICT_TYPES

    UNHINGED_GENRES = await _load_list_from_json_async(
        UNHINGED_GENRES_FILE, _DEFAULT_GENRE_LIST
    )
    UNHINGED_THEMES = await _load_list_from_json_async(
        UNHINGED_THEMES_FILE, _DEFAULT_THEMES_LIST
    )
    UNHINGED_SETTINGS_ARCHETYPES = await _load_list_from_json_async(
        UNHINGED_SETTINGS_FILE, _DEFAULT_SETTINGS_LIST
    )
    UNHINGED_PROTAGONIST_ARCHETYPES = await _load_list_from_json_async(
        UNHINGED_PROTAGONISTS_FILE, _DEFAULT_PROTAGONISTS_LIST
    )
    UNHINGED_CONFLICT_TYPES = await _load_list_from_json_async(
        UNHINGED_CONFLICTS_FILE, _DEFAULT_CONFLICTS_LIST
    )

    if not UNHINGED_GENRES or UNHINGED_GENRES == _DEFAULT_GENRE_LIST:
        logger.warning(
            "UNHINGED_GENRES might be using default values. "
            "Check unhinged_genres.json."
        )

# Example of how to run the async loading (e.g., in your main app setup):
# if __name__ == "__main__":
#     asyncio.run(load_unhinged_data_async())
#     print(f"Loaded genres: {UNHINGED_GENRES}")


# --- Tokenizer Fallback Configuration ---
TIKTOKEN_DEFAULT_ENCODING: str = "cl100k_base"
FALLBACK_CHARS_PER_TOKEN: float = 4.0

# --- Rich Progress Display Configuration ---
ENABLE_RICH_PROGRESS: bool = (
    os.getenv("ENABLE_RICH_PROGRESS", "True").lower() == "true"
)
RICH_REFRESH_PER_SECOND: float = 4.0

# --- Markdown Story Parser Configuration ---
MARKDOWN_FILL_IN_PLACEHOLDER: str = "[Fill-in]"
