# SAGA: Semantic And Graph-enhanced Authoring 🌌📚
## Let NANA (Next-gen Autonomous Narrative Architecture) tell you a story, just like old times!

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

SAGA is an autonomous, agentic creative-writing system designed to generate entire novels. Powered by the **NANA** engine, SAGA leverages Large Language Models (LLMs) and a Neo4j graph database to create rich, coherent, and evolving narratives.

![SAGA](https://github.com/Lanerra/saga/blob/master/SAGA.png)

![KG-3chapter](https://github.com/Lanerra/saga/blob/master/SAGA-KG.png)

![KG-14chapter](https://github.com/Lanerra/saga/blob/master/SAGA-KG-2.png)

## Overview

Ever dreamt of an AI that could not just write a paragraph, but an entire saga? SAGA, with its NANA engine, aims to do just that. It's more than just a text generator; it's a team of specialized AI agents working together:

*   **PlannerAgent:** Outlines detailed scene-by-scene plans for each chapter.
*   **DraftingAgent:** Writes the initial prose for chapters based on plans and context.
*   **ComprehensiveEvaluatorAgent:** Critically assesses drafts for plot, theme, consistency, and narrative depth.
*   **WorldContinuityAgent:** Specifically checks for and helps maintain consistency with established world-building and character facts.
*   **ChapterRevisionLogic:** Implements revisions based on evaluation feedback, using either targeted patches or full rewrites.
*   **KGMaintainerAgent:** Manages the novel's knowledge graph in Neo4j, extracting new information, summarizing chapters, and pre-populating initial data.

SAGA builds a dynamic understanding of the story's world, characters, and plot, storing this evolving knowledge in a Neo4j graph database. This allows for greater consistency and depth as the narrative unfolds.

## Key Features

*   **Autonomous Novel Generation:** Capable of generating multiple chapters in a single run.
*   **Agentic Architecture:** Utilizes multiple specialized AI agents for different stages of the writing process.
*   **Knowledge Graph Integration:** Employs a Neo4j database to store and retrieve story canon, character development, world-building details, and plot points.
*   **Semantic Context Generation:** Uses embeddings (via Ollama) to create semantically relevant context from previous chapters, ensuring narrative flow.
*   **Iterative Drafting & Revision:** Chapters go through drafting, evaluation, and revision cycles to improve quality.
*   **Dynamic State Adaptation:** The system learns and updates its knowledge graph based on the generated content.
*   **Configurable Generation:**
    *   Supports "Unhinged Mode" for more random and surprising story elements.
    *   Allows configuration of different LLMs for various tasks (planning, drafting, evaluation).
    *   Accepts user-supplied story elements via `user_story_elements.json` for a custom starting point.
*   **Rich Progress Display:** Optional live progress updates in the console using the Rich library.

## Architecture & Pipeline

SAGA's NANA engine orchestrates a pipeline of agents and logic modules for each chapter:

1.  **Initial Setup (First Run):**
    *   If `user_story_elements.json` is present, it's used to bootstrap the plot, characters, and world.
    *   Otherwise, `InitialSetupLogic` generates a plot outline, initial character profiles (including the protagonist), and core world-building elements using LLMs. This can be guided by "Unhinged Mode" or configured defaults.
    *   The `KGMaintainerAgent` pre-populates the Neo4j graph with this initial data.

2.  **Chapter Generation Loop (for each chapter):**
    *   **Planning:** The `PlannerAgent` creates a detailed scene-by-scene plan for the current chapter based on the overall plot point and current story state.
    *   **Context Generation:** `ContextGenerationLogic` assembles a "hybrid context" string. This includes:
        *   Semantic context from previous chapters (retrieved via vector similarity search in Neo4j).
        *   Key reliable facts extracted from the Knowledge Graph by `PromptDataGetters`.
    *   **Drafting:** The `DraftingAgent` (using `ChapterDraftingLogic`) writes the initial draft of the chapter, guided by the scene plan, plot point focus, and the hybrid context. It also receives filtered character and world data.
    *   **Evaluation:**
        *   The `ComprehensiveEvaluatorAgent` assesses the draft for coherence, plot arc adherence, thematic consistency, and narrative depth.
        *   The `WorldContinuityAgent` performs a focused check for consistency against established lore, character profiles, and plot points.
    *   **Revision (if needed):**
        *   If evaluation flags issues, `ChapterRevisionLogic` takes over.
        *   It can generate targeted "patches" to fix specific problems or perform a full chapter rewrite if `ENABLE_PATCH_BASED_REVISION` is true and issues are suitable.
        *   This process can be iterative up to a configured number of attempts.
    *   **Finalization & Knowledge Update:**
        *   The `KGMaintainerAgent` summarizes the final chapter text.
        *   An embedding is generated for the final chapter text.
        *   The chapter data (text, summary, embedding, provisional status) is saved to Neo4j by the `StateManager`.
        *   The `KGMaintainerAgent` extracts new knowledge (character updates, world-building changes, new KG triples) from the final chapter text and updates both the in-memory state and the Neo4j graph.
    *   The orchestrator updates its internal state (character profiles, world-building) based on the KG Maintainer's merges.

## Setup

### Prerequisites

*   Python 3.8+
*   Docker & Docker Compose
*   An Ollama-compatible LLM server (e.g., running via LM Studio, oobabooga's text-generation-webui with OpenAI extension, or a dedicated OpenAI-API compatible server).
*   An Ollama instance (can be the same as the LLM server if it supports embeddings, or a separate one) for generating text embeddings.

### 1. Clone the Repository

```bash
git clone https://github.com/Lanerra/saga.git
cd saga
```

### 2. Install Python Dependencies

It's recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Configure SAGA

Most configuration is done in `config.py`. Key settings to review and potentially set via environment variables:

*   **LLM API:**
    *   `OPENAI_API_BASE`: URL of your OpenAI-compatible LLM API endpoint (e.g., `http://127.0.0.1:8080/v1`).
    *   `OPENAI_API_KEY`: API key for the LLM (can be "nope" or any string if your server doesn't require auth).
*   **Embedding Model (Ollama):**
    *   `OLLAMA_EMBED_URL`: URL of your Ollama API for embeddings (e.g., `http://127.0.0.1:11434`).
    *   `EMBEDDING_MODEL`: Name of the embedding model in Ollama (e.g., `nomic-embed-text:latest`). Ensure this model is pulled in Ollama.
*   **Neo4j Connection:**
    *   `NEO4J_URI`: Bolt URI for Neo4j (defaults to `bolt://localhost:7687`).
    *   `NEO4J_USER`: Neo4j username (defaults to `neo4j`).
    *   `NEO4J_PASSWORD`: Neo4j password (defaults to `saga_password`).
    *   `NEO4J_DATABASE`: Neo4j database name (defaults to `neo4j`).
*   **Model Aliases:**
    *   `LARGE_MODEL`, `MEDIUM_MODEL`, `SMALL_MODEL`, `NARRATOR_MODEL`: Set these to the names of the models you want to use for different tasks, as recognized by your `OPENAI_API_BASE` server.
*   **Output Directory:**
    *   `BASE_OUTPUT_DIR`: Where novel outputs, logs, and debug files will be stored (defaults to `novel_output`).
*   **Logging:**
    *   `LOG_LEVEL`: Set to `DEBUG`, `INFO`, `WARNING`, `ERROR` (defaults to `INFO`).

You can set these as environment variables or modify `config.py` directly.

### 4. Set up Neo4j Database

SAGA uses Neo4j to store its knowledge graph. A `docker-compose.yml` file is provided to run Neo4j in a Docker container.

Use the `manage_neo4j.sh` script:

*   **Start Neo4j:**
    ```bash
    ./manage_neo4j.sh start
    ```
    This will start a Neo4j container. You can access the Neo4j Browser at `http://localhost:7474`.
    Login with `neo4j` / `saga_password` (or your configured credentials).

*   **Stop Neo4j:**
    ```bash
    ./manage_neo4j.sh stop
    ```

*   **Check Status:**
    ```bash
    ./manage_neo4j.sh status
    ```

The first time SAGA runs, it will attempt to create necessary constraints and indexes in the Neo4j database, including a vector index for chapter embeddings.

### 5. (Optional) User-Supplied Story Elements

If you want to provide your own starting point for the novel, a `user_story_elements.json.example` file is provided as a template in the root directory. Simply edit the template as desired and save it as `user_story_elements.json`. SAGA will use this to initialize the plot, characters, and world. If this file is not present, SAGA will generate these elements.

### 6. (Optional) Unhinged Mode Data

For "Unhinged Mode" (randomized initial elements), SAGA looks for JSON files in the `unhinged_data/` directory:
*   `unhinged_genres.json`
*   `unhinged_themes.json`
*   `unhinged_settings_archetypes.json`
*   `unhinged_protagonist_archetypes.json`
*   `unhinged_conflict_types.json`

These should contain lists of strings. If not present, defaults from `config.py` will be used.

## Running SAGA

Once Neo4j is running and your configuration is set:

```bash
python nana_orchestrator.py
```

SAGA will begin the novel generation process.
*   On the first run, it will perform initial setup (plot, world, characters) and pre-populate the Neo4j KG.
*   On subsequent runs, it will load existing state from Neo4j and continue generating chapters.
*   The number of chapters generated per run is controlled by `CHAPTERS_PER_RUN` in `config.py`.

Output files (chapters, logs, debug info) will be saved in the directory specified by `BASE_OUTPUT_DIR` (default: `novel_output`).

## Resetting the Database

If you want to start SAGA from scratch (e.g., after a test run or to try a new story), you can wipe the Neo4j database.

**⚠️ WARNING: This will delete ALL data in the Neo4j database currently pointed to by your configuration.**

1.  Ensure SAGA is not running.
2.  Run the `reset_neo4j.py` script:
    ```bash
    python reset_neo4j.py
    ```
    You will be asked for confirmation.
    To skip confirmation (e.g., in a script):
    ```bash
    python reset_neo4j.py --force
    ```
    Use the `--uri`, `--user`, and `--password` arguments if your Neo4j instance uses different credentials than the defaults in `config.py` or `reset_neo4j.py`.

After resetting, the next run of `python nana_orchestrator.py` will perform the initial setup again.

## License

This project is licensed under the Apache License, Version 2.0. See the `LICENSE` file for details.

---

Let NANA spin you a tale! Happy authoring!
