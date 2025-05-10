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

import os
import logging
import json # NEW: Import json
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
OLLAMA_EMBED_URL: str = os.getenv("OLLAMA_EMBED_URL", "http://192.168.64.1:11434")
OPENAI_API_BASE: str = os.getenv("OPENAI_API_BASE", "http://192.168.64.1:8080/v1")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "nope")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")

# Model Aliases (consider populating from environment or a more dynamic config if models change frequently)
LARGE_MODEL_DEFAULT: str = "Qwen3-30B-A3B"
MEDIUM_MODEL_DEFAULT: str = "Qwen3-30B-A3B"

LARGE_MODEL: str = os.getenv("LARGE_MODEL", LARGE_MODEL_DEFAULT)
MEDIUM_MODEL: str = os.getenv("MEDIUM_MODEL", MEDIUM_MODEL_DEFAULT)
SUMMARIZATION_MODEL: str = os.getenv("SUMMARIZATION_MODEL", MEDIUM_MODEL_DEFAULT)

MAIN_GENERATION_MODEL: str = LARGE_MODEL
JSON_CORRECTION_MODEL: str = MEDIUM_MODEL
CONSISTENCY_CHECK_MODEL: str = MEDIUM_MODEL
KNOWLEDGE_UPDATE_MODEL: str = MEDIUM_MODEL
INITIAL_SETUP_MODEL: str = MEDIUM_MODEL
PLANNING_MODEL: str = LARGE_MODEL
DRAFTING_MODEL: str = LARGE_MODEL
REVISION_MODEL: str = LARGE_MODEL


# --- Output and File Paths ---
BASE_OUTPUT_DIR: str = "novel_output"
DATABASE_FILE: str = os.path.join(BASE_OUTPUT_DIR, "novel_data.db")
PLOT_OUTLINE_FILE: str = os.path.join(BASE_OUTPUT_DIR, "plot_outline.json")
CHARACTER_PROFILES_FILE: str = os.path.join(BASE_OUTPUT_DIR, "character_profiles.json")
WORLD_BUILDER_FILE: str = os.path.join(BASE_OUTPUT_DIR, "world_building.json")
CHAPTERS_DIR: str = os.path.join(BASE_OUTPUT_DIR, "chapters")
CHAPTER_LOGS_DIR: str = os.path.join(BASE_OUTPUT_DIR, "chapter_logs")
DEBUG_OUTPUTS_DIR: str = os.path.join(BASE_OUTPUT_DIR, "debug_outputs")

# NEW: Define a directory for unhinged data files
UNHINGED_DATA_DIR: str = "unhinged_data" # You can place this where you like, e.g., next to config.py
os.makedirs(UNHINGED_DATA_DIR, exist_ok=True) # Ensure it exists

# NEW: Paths for unhinged data files
UNHINGED_GENRES_FILE: str = os.path.join(UNHINGED_DATA_DIR, "unhinged_genres.json")
UNHINGED_THEMES_FILE: str = os.path.join(UNHINGED_DATA_DIR, "unhinged_themes.json")
UNHINGED_SETTINGS_FILE: str = os.path.join(UNHINGED_DATA_DIR, "unhinged_settings_archetypes.json")
UNHINGED_PROTAGONISTS_FILE: str = os.path.join(UNHINGED_DATA_DIR, "unhinged_protagonist_archetypes.json")
UNHINGED_CONFLICTS_FILE: str = os.path.join(UNHINGED_DATA_DIR, "unhinged_conflict_types.json")


# Ensure output directories exist
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
os.makedirs(CHAPTERS_DIR, exist_ok=True)
os.makedirs(CHAPTER_LOGS_DIR, exist_ok=True)
os.makedirs(DEBUG_OUTPUTS_DIR, exist_ok=True)


# --- Generation Parameters ---
MAX_CONTEXT_LENGTH: int = 40960
MAX_GENERATION_TOKENS: int = 32768
KNOWLEDGE_UPDATE_SNIPPET_SIZE: int = 32768 # Used for text snippets for KG updates, summaries
CONTEXT_CHAPTER_COUNT: int = 5 # Max number of similar past chapters for context
CHAPTERS_PER_RUN: int = 1 # Number of chapters to generate in a single execution


# --- Caching ---
# LRU cache sizes for LLM/Embedding calls
EMBEDDING_CACHE_SIZE: int = 128
SUMMARY_CACHE_SIZE: int = 32
KG_TRIPLE_EXTRACTION_CACHE_SIZE: int = 16


# --- Agentic Planning ---
ENABLE_AGENTIC_PLANNING: bool = True
MAX_PLANNING_TOKENS: int = 32768
PLANNING_CONTEXT_MAX_CHARS_PER_PROFILE_DESC: int = 100
PLANNING_CONTEXT_MAX_RECENT_DEV_PER_PROFILE: int = 150
PLANNING_CONTEXT_MAX_CHARACTERS_IN_SNIPPET: int = 5
PLANNING_CONTEXT_MAX_LOCATIONS_IN_SNIPPET: int = 3
PLANNING_CONTEXT_MAX_FACTIONS_IN_SNIPPET: int = 2
PLANNING_CONTEXT_MAX_SYSTEMS_IN_SNIPPET: int = 2


# --- Revision and Validation ---
REVISION_COHERENCE_THRESHOLD: float = 0.65 # Cosine similarity threshold with previous chapter
REVISION_CONSISTENCY_TRIGGER: bool = True # Whether to run LLM-based consistency check
PLOT_ARC_VALIDATION_TRIGGER: bool = True # Whether to run LLM-based plot arc validation
REVISION_SIMILARITY_ACCEPTANCE: float = 0.99 # If revised draft is this similar to original, reject revision
MAX_SUMMARY_TOKENS: int = 32768
MAX_CONSISTENCY_TOKENS: int = 32768
MAX_PLOT_VALIDATION_TOKENS: int = 32768
MAX_KG_TRIPLE_TOKENS: int = 32768
MAX_PREPOP_KG_TOKENS: int = 32768 # For initial KG population from plot/world

MIN_ACCEPTABLE_DRAFT_LENGTH: int = 10240 # Minimum character length for a chapter draft
ENABLE_DYNAMIC_STATE_ADAPTATION: bool = True # Allow LLM to propose modifications to JSON state
KG_PREPOPULATION_CHAPTER_NUM: int = 0 # Chapter number assigned to pre-populated KG facts


# --- Embedding Configuration ---
EXPECTED_EMBEDDING_DIM: int = 768
EMBEDDING_DTYPE: np.dtype = np.dtype(np.float32)


# --- Logging ---
LOG_LEVEL_STR: str = os.getenv("LOG_LEVEL", "INFO").upper()
# Ensure LOG_LEVEL is a valid logging level integer
LOG_LEVEL: int = getattr(logging, LOG_LEVEL_STR, logging.INFO)
LOG_FORMAT: str = '%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s'
LOG_DATE_FORMAT: str = '%Y-%m-%d %H:%M:%S'
LOG_FILE: Optional[str] = os.path.join(BASE_OUTPUT_DIR, "saga_run.log")


# --- Novel Configuration ---
UNHINGED_PLOT_MODE: bool = False # If true, uses random genre/theme/etc.
CONFIGURED_GENRE: str = "body horror"
CONFIGURED_THEME: str = "the nature of dreams"
CONFIGURED_SETTING_DESCRIPTION: str = "a colony on the edge of a black hole where time dilation creates generational disparities"
DEFAULT_PROTAGONIST_NAME: str = "Sága"
DEFAULT_PLOT_OUTLINE_TITLE: str = "Untitled Saga"

# --- Unhinged Mode Data (Loaded from JSON files) ---
# MODIFIED: Load these lists from files
_DEFAULT_GENRE_LIST = ["science fiction", "fantasy", "horror"] # Fallback if file is missing/corrupt
UNHINGED_GENRES: List[str] = _load_list_from_json(UNHINGED_GENRES_FILE, _DEFAULT_GENRE_LIST)
UNHINGED_THEMES: List[str] = _load_list_from_json(UNHINGED_THEMES_FILE, ["the nature of reality", "the cost of power"])
UNHINGED_SETTINGS_ARCHETYPES: List[str] = _load_list_from_json(UNHINGED_SETTINGS_FILE, ["a floating city", "a derelict starship"])
UNHINGED_PROTAGONIST_ARCHETYPES: List[str] = _load_list_from_json(UNHINGED_PROTAGONISTS_FILE, ["a reluctant hero", "a cynical detective"])
UNHINGED_CONFLICT_TYPES: List[str] = _load_list_from_json(UNHINGED_CONFLICTS_FILE, ["man vs self", "man vs society"])

# Example of how to check if loading was successful (optional)
if not UNHINGED_GENRES or UNHINGED_GENRES == _DEFAULT_GENRE_LIST:
    logging.warning("UNHINGED_GENRES might be using default values. Check unhinged_genres.json.")
# You can add similar checks for other lists if desired.