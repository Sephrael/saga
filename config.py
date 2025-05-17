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
OLLAMA_EMBED_URL: str = os.getenv("OLLAMA_EMBED_URL", "http://192.168.64.1:11434")
OPENAI_API_BASE: str = os.getenv("OPENAI_API_BASE", "http://192.168.64.1:8080/v1")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "nope")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")

# Model Aliases (consider populating from environment or a more dynamic config if models change frequently)
LARGE_MODEL_DEFAULT: str = "Qwen3-8B"
MEDIUM_MODEL_DEFAULT: str = "Qwen3-4B"
SMALL_MODEL_DEFAULT: str = "Qwen3-4B"
NARRATOR_MODEL_DEFAULT: str = "Qwen3-14B" # Primary high-quality model

LARGE_MODEL: str = os.getenv("LARGE_MODEL", LARGE_MODEL_DEFAULT)
MEDIUM_MODEL: str = os.getenv("MEDIUM_MODEL", MEDIUM_MODEL_DEFAULT)
SMALL_MODEL: str = os.getenv("SMALL_MODEL", SMALL_MODEL_DEFAULT)
NARRATOR_MODEL: str = os.getenv("NARRATOR_MODEL", NARRATOR_MODEL_DEFAULT)

# --- LLM Call Settings & Fallbacks ---
LLM_RETRY_ATTEMPTS: int = 3 # Number of retry attempts for LLM calls
LLM_RETRY_DELAY_SECONDS: float = 2.0 # Initial delay for retries (will be increased exponentially)
# Fallback model to use if a primary model fails after retries for critical generation tasks.
# Should be a capable, but potentially faster or more reliable model.
FALLBACK_GENERATION_MODEL: str = MEDIUM_MODEL 


MAIN_GENERATION_MODEL: str = NARRATOR_MODEL # Retained for clarity, but Narrator Model is primary
JSON_CORRECTION_MODEL: str = SMALL_MODEL
# CONSISTENCY_CHECK_MODEL: str = SMALL_MODEL # Superseded by EVALUATION_MODEL
KNOWLEDGE_UPDATE_MODEL: str = MEDIUM_MODEL # Needs good comprehension for unified extraction
INITIAL_SETUP_MODEL: str = MEDIUM_MODEL # Good balance for initial creative tasks
PLANNING_MODEL: str = LARGE_MODEL # Needs strong reasoning
DRAFTING_MODEL: str = NARRATOR_MODEL # High quality needed
REVISION_MODEL: str = NARRATOR_MODEL # High quality needed
EVALUATION_MODEL: str = LARGE_MODEL # For comprehensive chapter evaluation


# --- Output and File Paths ---
BASE_OUTPUT_DIR: str = "novel_output"
DATABASE_FILE: str = os.path.join(BASE_OUTPUT_DIR, "novel_data.db")
PLOT_OUTLINE_FILE: str = os.path.join(BASE_OUTPUT_DIR, "plot_outline.json")
CHARACTER_PROFILES_FILE: str = os.path.join(BASE_OUTPUT_DIR, "character_profiles.json")
WORLD_BUILDER_FILE: str = os.path.join(BASE_OUTPUT_DIR, "world_building.json")
CHAPTERS_DIR: str = os.path.join(BASE_OUTPUT_DIR, "chapters")
CHAPTER_LOGS_DIR: str = os.path.join(BASE_OUTPUT_DIR, "chapter_logs")
DEBUG_OUTPUTS_DIR: str = os.path.join(BASE_OUTPUT_DIR, "debug_outputs")

UNHINGED_DATA_DIR: str = "unhinged_data" 
os.makedirs(UNHINGED_DATA_DIR, exist_ok=True) 

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
MAX_CONTEXT_LENGTH: int = 131072 # Max characters for combined context in prompts
MAX_GENERATION_TOKENS: int = 65536 # Max tokens LLM can generate in one go
# KNOWLEDGE_UPDATE_SNIPPET_SIZE: int = 32768 # Less relevant if using full text for extraction/evaluation
CONTEXT_CHAPTER_COUNT: int = 5 # Default number of past chapters for semantic context
CHAPTERS_PER_RUN: int = 9 # How many chapters to attempt writing in one execution
LLM_TOP_P: float = 0.95 # LLM nucleus sampling parameter


# --- Caching ---
EMBEDDING_CACHE_SIZE: int = 128
SUMMARY_CACHE_SIZE: int = 32 # Summarization might still use full text as cache key
KG_TRIPLE_EXTRACTION_CACHE_SIZE: int = 16 # KG extraction now part of unified call, this specific cache might be less used


# --- Agentic Planning & Prompt Context Snippets ---
ENABLE_AGENTIC_PLANNING: bool = True
MAX_PLANNING_TOKENS: int = 65536
# Reduced for prompt brevity
PLANNING_CONTEXT_MAX_CHARS_PER_PROFILE_DESC: int = 80 
PLANNING_CONTEXT_MAX_RECENT_DEV_PER_PROFILE: int = 120 
PLANNING_CONTEXT_MAX_CHARACTERS_IN_SNIPPET: int = 5
PLANNING_CONTEXT_MAX_LOCATIONS_IN_SNIPPET: int = 3
PLANNING_CONTEXT_MAX_FACTIONS_IN_SNIPPET: int = 2
PLANNING_CONTEXT_MAX_SYSTEMS_IN_SNIPPET: int = 2


# --- Revision and Validation ---
REVISION_COHERENCE_THRESHOLD: float = 0.60 
# REVISION_CONSISTENCY_TRIGGER: bool = True  # Implicitly handled by comprehensive evaluation
# PLOT_ARC_VALIDATION_TRIGGER: bool = True # Implicitly handled by comprehensive evaluation
# Slightly relaxed revision similarity to avoid rejecting minor rephrasing that fixed issues
REVISION_SIMILARITY_ACCEPTANCE: float = 0.985 
MAX_SUMMARY_TOKENS: int = 65536 # For summarization output
# MAX_CONSISTENCY_TOKENS: int = 65536 # Superseded by EVALUATION_MODEL's token limits
# MAX_PLOT_VALIDATION_TOKENS: int = 65536 # Superseded by EVALUATION_MODEL's token limits
MAX_KG_TRIPLE_TOKENS: int = 65536 # Still relevant for the KG part of unified extraction if needed
MAX_PREPOP_KG_TOKENS: int = 65536 
# MAX_THEMATIC_CONSISTENCY_TOKENS: int = 1024 # Superseded by EVALUATION_MODEL

MIN_ACCEPTABLE_DRAFT_LENGTH: int = 5120 
ENABLE_DYNAMIC_STATE_ADAPTATION: bool = True 
KG_PREPOPULATION_CHAPTER_NUM: int = 0 


# --- Embedding Configuration ---
EXPECTED_EMBEDDING_DIM: int = 768 # Keep this matching your embedding model's output dimension
# Using float16 for embeddings to reduce memory.
# Note: This can impact precision of similarity scores. Test thoroughly.
# If issues arise, revert to np.float32.
EMBEDDING_DTYPE: np.dtype = np.dtype(np.float16) 


# --- Logging ---
LOG_LEVEL_STR: str = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_LEVEL: int = getattr(logging, LOG_LEVEL_STR, logging.INFO)
LOG_FORMAT: str = '%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s'
LOG_DATE_FORMAT: str = '%Y-%m-%d %H:%M:%S'
LOG_FILE: Optional[str] = os.path.join(BASE_OUTPUT_DIR, "saga_run.log")


# --- Novel Configuration ---
UNHINGED_PLOT_MODE: bool = False 
CONFIGURED_GENRE: str = "dystopian horror"
CONFIGURED_THEME: str = "the cost of power"
CONFIGURED_SETTING_DESCRIPTION: str = "a walled city where precious memories can be surrendered for an extension to one's lifespan"
DEFAULT_PROTAGONIST_NAME: str = "Sága"
DEFAULT_PLOT_OUTLINE_TITLE: str = "Untitled Saga"
# THEMATIC_CONSISTENCY_CHAPTER_SNIPPET_SIZE: int = 16384 # No longer needed

# --- Unhinged Mode Data (Loaded from JSON files) ---
_DEFAULT_GENRE_LIST = ["science fiction", "fantasy", "horror"] 
UNHINGED_GENRES: List[str] = _load_list_from_json(UNHINGED_GENRES_FILE, _DEFAULT_GENRE_LIST)
UNHINGED_THEMES: List[str] = _load_list_from_json(UNHINGED_THEMES_FILE, ["the nature of reality", "the cost of power"])
UNHINGED_SETTINGS_ARCHETYPES: List[str] = _load_list_from_json(UNHINGED_SETTINGS_FILE, ["a floating city", "a derelict starship"])
UNHINGED_PROTAGONIST_ARCHETYPES: List[str] = _load_list_from_json(UNHINGED_PROTAGONISTS_FILE, ["a reluctant hero", "a cynical detective"])
UNHINGED_CONFLICT_TYPES: List[str] = _load_list_from_json(UNHINGED_CONFLICTS_FILE, ["man vs self", "man vs society"])

if not UNHINGED_GENRES or UNHINGED_GENRES == _DEFAULT_GENRE_LIST:
    logging.warning("UNHINGED_GENRES might be using default values. Check unhinged_genres.json.")