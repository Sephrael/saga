# SAGA: Semantic And Graph-enhanced Authoring

**⚠️ Early Development Stage:** This project is currently in its early stages of development. Features may be incomplete, and the codebase is subject to significant changes.

## Overview

SAGA is a Python-based system designed to autonomously write novels. It leverages Large Language Models (LLMs) for various creative and analytical tasks, including plot generation, world-building, character development, chapter drafting, content evaluation, and knowledge graph construction. The system aims to produce coherent, consistent, and engaging long-form narratives.

## Key Features

*   **Automated Plot & World Generation:** Creates initial plot outlines, character archetypes, and foundational world-building details.
*   **Agentic Chapter Planning:** (Optional) Generates detailed scene-by-scene plans for each chapter.
*   **Iterative Chapter Drafting & Revision:**
    *   Generates initial chapter drafts.
    *   Evaluates drafts for coherence, consistency with established lore, and plot arc alignment.
    *   Revises drafts based on evaluation feedback.
*   **Dynamic Knowledge Management:**
    *   Updates character profiles and world-building documents based on events in each chapter.
    *   Constructs and maintains a knowledge graph (triples) of facts derived from the narrative.
*   **Context-Aware Generation:** Utilizes summaries of previous chapters and semantically relevant content to maintain narrative flow.
*   **Configurable Models & Parameters:** Allows easy configuration of LLM models, API endpoints, and generation parameters.
*   **Persistent State:** Uses SQLAlchemy and SQLite to store novel data, including chapter text, summaries, embeddings, and knowledge graph triples.
*   **Asynchronous Operations:** Built with `asyncio` for efficient handling of LLM API calls and database operations.

## Architecture

The system is composed of several key modules:

*   **`novel_agent.py`:** The main orchestrator (NovelWriterAgent) that manages the overall novel writing process.
*   **`main.py`:** Entry point for running the agent. Handles setup and invokes the agent's writing loop.
*   **`config.py`:** Centralized configuration for all system parameters, API keys, model names, and file paths.
*   **`llm_interface.py`:** Handles all direct interactions with LLMs and embedding models, including API calls, response cleaning, and JSON parsing/correction.
*   **`state_manager.py`:** Manages persistent storage using SQLAlchemy, handling all database reads and writes for plot outlines, character profiles, world-building data, chapter content, and knowledge graph triples.
*   **`*_logic.py` Files:** These contain the core business logic for specific tasks:
    *   `initial_setup_logic.py`: Logic for generating the initial plot outline and world-building.
    *   `chapter_planning_logic.py`: Logic for creating detailed chapter plans.
    *   `context_generation_logic.py`: Logic for assembling contextual information for chapter drafting.
    *   `chapter_drafting_logic.py`: Logic for generating the first draft of a chapter.
    *   `chapter_evaluation_logic.py`: Logic for evaluating chapter drafts against various criteria.
    *   `chapter_revision_logic.py`: Logic for revising chapter drafts based on evaluation feedback.
    *   `knowledge_management_logic.py`: Logic for summarizing chapters, extracting knowledge, and updating character/world profiles and the knowledge graph.
*   **`prompt_data_getters.py`:** Helper functions to retrieve and format data snippets for LLM prompts.
*   **`type.py`:** Defines custom `TypedDict` types used across the project.
*   **`utils.py`:** General utility functions (e.g., cosine similarity).

## Prerequisites

*   Python 3.11+
*   Access to an OpenAI-compatible LLM API endpoint.
*   Access to an Ollama-compatible embedding API endpoint (for `nomic-embed-text` or similar).

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd saga-novel-writer # Or your repository name
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(You'll need to create a `requirements.txt` file based on the imports: `requests`, `httpx`, `numpy`, `sqlalchemy`, `aiosqlite`, `async_lru`)*

4.  **Configure Environment Variables:**
    Set the following environment variables (e.g., in a `.env` file and use a library like `python-dotenv`, or set them directly in your shell):
    *   `OLLAMA_EMBED_URL`: URL for your Ollama embedding service (e.g., `http://localhost:11434`).
    *   `OPENAI_API_BASE`: URL for your OpenAI-compatible API (e.g., `http://localhost:8080/v1`).
    *   `OPENAI_API_KEY`: Your API key (can be a placeholder like "nope" if your local LLM doesn't require one).
    *   `EMBEDDING_MODEL`: Name of the embedding model (e.g., `nomic-embed-text:latest`).
    *   `LARGE_MODEL`, `MEDIUM_MODEL`, `SMALL_MODEL`, `NARRATOR_MODEL`: Names of your LLM models.
    *   `LOG_LEVEL`: (Optional) Logging level, e.g., `INFO`, `DEBUG`. Defaults to `INFO`.

5.  **(Optional) Unhinged Mode Data:**
    If using `UNHINGED_PLOT_MODE = True` in `config.py`, create a directory named `unhinged_data` (or as configured in `config.UNHINGED_DATA_DIR`) and populate it with JSON files containing lists of strings for genres, themes, etc. (see `config.py` for filenames like `unhinged_genres.json`). Example `unhinged_genres.json`:
    ```json
    [
        "space opera",
        "cyberpunk",
        "high fantasy",
        "urban fantasy",
        "cosmic horror"
    ]
    ```

## Configuration

Most operational parameters are controlled via `config.py`. This includes:
*   LLM model names for different tasks (drafting, summarization, planning, etc.).
*   API endpoints and keys (primarily via environment variables).
*   Output directory paths.
*   Generation parameters (e.g., context length, token limits, minimum draft length).
*   Validation thresholds (e.g., coherence scores).
*   Feature flags (e.g., `ENABLE_AGENTIC_PLANNING`, `UNHINGED_PLOT_MODE`).

Review and adjust `config.py` to suit your local LLM setup and desired novel characteristics.

## Running the Agent

To start the novel generation process:

```bash
python main.py
```

The agent will:
1.  Initialize or load existing plot, world, and character data.
2.  If necessary, generate a new plot outline and world-building information.
3.  Pre-populate the knowledge graph from initial data (if it's a new novel).
4.  Proceed to write chapters sequentially, up to `config.CHAPTERS_PER_RUN`.

Output, including chapter text, logs, and the SQLite database (`novel_data.db`), will be saved in the directory specified by `config.BASE_OUTPUT_DIR` (defaults to `novel_output`).

## Future Work / Contributing

This project is actively evolving. Potential areas for future development include:
*   Enhanced LLM prompting strategies for higher quality and more diverse outputs.
*   More sophisticated evaluation metrics.
*   User interface for interaction and guidance.
*   Support for different LLM backends.
*   Advanced character arc tracking and enforcement.
*   Multi-agent collaboration for different aspects of writing.

Contributions, suggestions, and feedback are welcome! Please open an issue or pull request.

## License

This project is licensed under the Apache License, Version 2.0. See the `config.py` header or include a `LICENSE` file for details.