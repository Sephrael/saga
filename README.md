# SAGA: Semantic And Graph-enhanced Authoring

SAGA is an autonomous, agentic system designed for generating cohesive and coherent narrative fiction, chapter by chapter. It leverages Large Language Models (LLMs), semantic embeddings, a knowledge graph, and an iterative refinement process to produce novel-length stories.

***This project is currently in early development***
## Core Narrative Generation
*   **Autonomous Chapter-by-Chapter Writing:** Generates novel-length stories sequentially, maintaining context and flow.
*   **Flexible Plot Outline Generation:**
    *   **Standard Mode:** Creates a plot based on user-configured genre, theme, and setting descriptions.
    *   **Unhinged Mode:** Generates highly creative and often paradoxical story seeds by randomizing genre, theme, setting archetypes, protagonist archetypes, and conflict types from extensive predefined lists.
*   **Initial World-Building:** Automatically generates foundational world-building details (locations, society, systems, lore, history) based on the plot outline.
*   **Protagonist-Focused:** Centers the narrative around a defined protagonist, tracking their arc and development.

## Agentic Capabilities & Advanced Planning
*   **Detailed Scene-Based Chapter Planning:** Before drafting, SAGA plans each chapter by generating 8-15 detailed scene descriptions. Each scene outline includes:
    *   `scene_number`: Sequential identifier.
    *   `summary`: Overview of events in the scene.
    *   `characters_involved`: Key characters present.
    *   `key_dialogue_points`: Important dialogue snippets or communication intentions.
    *   `setting_details`: Specific location, atmosphere, and environmental elements.
    *   `contribution`: How the scene advances the plot, subplots, or character arcs.
*   **Plan-Driven Drafting:** The LLM uses the detailed scene-by-scene plan as a strong guide for writing the chapter text, ensuring structural integrity.
*   **Dynamic State Adaptation:** The system can allow the LLM to propose modifications to character profiles and world-building elements based on events within a chapter, enabling organic evolution of the story world. (Configurable via `ENABLE_DYNAMIC_STATE_ADAPTATION`)

## Knowledge Management & Consistency
*   **Persistent Knowledge Graph (KG):** Utilizes an SQLite-based knowledge graph to store and query factual triples (Subject-Predicate-Object) about characters, locations, events, and relationships.
    *   **KG Pre-population:** Extracts foundational knowledge from the initial plot outline and world-building data to populate the KG *before* Chapter 1 generation, establishing a baseline canon. (Triples added with `chapter_added = 0`)
    *   **KG Updates from Chapters:** Extracts new facts from each generated chapter to continuously expand and update the KG.
*   **Provisional Data Handling:**
    *   Marks data in the database (chapters, KG triples) and JSON state files (character/world updates) as `is_provisional` if it's derived from an unrevised or flawed draft.
    *   Allows the system to be aware of data quality, preferring non-provisional facts for critical tasks like planning and consistency checks.
*   **JSON-Based State Management:** Maintains detailed character profiles and world-building information in structured JSON files, updated after each chapter.
*   **Semantic Context Retrieval:** Gathers relevant context from past chapters using embeddings and plot point similarity to inform current chapter generation.
*   **Consistency Checking:** Employs an LLM to analyze chapter drafts against the plot outline, character profiles, world-building data, reliable KG facts, and previous chapter context to identify contradictions or deviations.

## Evaluation & Iterative Refinement
*   **Multi-faceted Draft Evaluation:** Each generated chapter draft undergoes rigorous checks:
    *   **Coherence Score:** Cosine similarity of embeddings with the previous chapter.
    *   **Consistency Issues:** LLM-driven analysis for factual and narrative contradictions.
    *   **Plot Arc Validation:** LLM-driven check to ensure the chapter addresses its intended plot point from the outline.
    *   **Minimum Length:** Ensures drafts meet a minimum character count.
*   **Automated Revision Loop:** If a draft fails evaluation, SAGA triggers an LLM-driven revision process, providing specific feedback to address the identified issues.
*   **Revision Similarity Acceptance:** Prevents superficial revisions by rejecting revised drafts that are too similar to the original.

## LLM Interaction & Customization
*   **Flexible LLM Integration:**
    *   Supports Ollama for local embedding models (e.g., `nomic-embed-text`).
    *   Supports OpenAI-compatible API endpoints for generation models.
    *   Configurable model names for both embeddings and generation.
*   **Robust LLM Response Handling:**
    *   Advanced cleaning of LLM outputs (removing "think" tags, boilerplate, markdown).
    *   Intelligent JSON parsing from LLM responses, including an LLM-based correction mechanism for syntax errors.
*   **Embedding Generation & Caching:** Generates text embeddings with an LRU cache for efficiency.
*   **Granular Token Control:** Configurable maximum token limits for various LLM tasks (generation, planning, summarization, analysis).

## Data Persistence & Output
*   **SQLite Database:** Stores all core narrative data:
    *   Chapter text, raw LLM logs, summaries, provisional status.
    *   Chapter embeddings.
    *   Knowledge Graph triples.
*   **Structured File Output:**
    *   Saves plot outline, character profiles, and world-building data as human-readable JSON files.
    *   Outputs each chapter as a separate `.txt` file.
    *   Saves raw LLM generation logs for each chapter for debugging and analysis.
    *   Organized output directory structure.

## Configuration & Logging
*   **Comprehensive Configuration:** Most parameters are tunable via `config.py` (API keys, model names, file paths, generation limits, thresholds, feature toggles).
*   **Detailed Logging:** Extensive logging throughout the generation process with configurable levels (DEBUG, INFO, etc.) and optional output to a log file.
*   **Controlled Runs:** Ability to specify the number of chapters to generate per execution run.

## Architecture Overview

SAGA is composed of several key modules:

*   `main.py`: The main execution script that initializes the system and drives the novel generation loop.
*   `novel_logic.py`: Contains the `NovelWriterAgent`, the core component responsible for all story logic, state management, and interaction with other modules.
*   `llm_interface.py`: Handles all communication with LLMs and embedding models, including API calls, response cleaning, and robust JSON parsing.
*   `database_manager.py`: Manages all interactions with the SQLite database, including schema creation and CRUD operations for chapters, embeddings, and the Knowledge Graph.
*   `config.py`: Centralizes all configuration settings, such as API endpoints, model names, file paths, generation parameters, and validation thresholds.
*   `utils.py`: Provides general utility functions, like cosine similarity calculation.

## Workflow

The system operates with the following general workflow:

1.  **Initialization**:
    *   Sets up logging.
    *   Initializes the `NovelWriterAgent`.
    *   Loads existing state (plot, characters, world-building from JSON files; chapter count and KG from database).
2.  **Pre-computation (if necessary)**:
    *   **Plot Outline**: If no valid plot outline exists (or it's the default), a new one is generated by the LLM based on configurations in `config.py` (either standard or "unhinged" mode).
    *   **World-Building**: If world-building data is default or missing, initial data is generated by the LLM based on the plot outline.
3.  **Chapter Generation Loop** (for a configured number of chapters per run):
    *   For each chapter:
        *   **Planning (Optional)**: If `ENABLE_AGENTIC_PLANNING` is true, the `NovelWriterAgent` prompts an LLM to create a plan (key beats/scenes) for the chapter. This plan considers the overall plot point, recent story context, character/world state (annotated with provisional status), and reliable (non-provisional) KG facts.
        *   **Context Gathering**: Relevant context is assembled from past chapter summaries/text (retrieved via semantic similarity of embeddings and annotated with provisional status), character profiles, world-building data (both annotated with provisional status for LLM awareness), and the chapter plan.
        *   **Drafting**: An LLM generates the initial draft of the chapter text based on the assembled context and the current plot point focus.
        *   **Evaluation**: The draft is evaluated for:
            *   Coherence with the previous chapter (cosine similarity of embeddings).
            *   Consistency with plot, character profiles, world-building, and reliable KG facts.
            *   Adherence to the intended plot arc for the chapter.
            *   Minimum length.
        *   **Revision (if needed)**: If the evaluation flags issues, the LLM is prompted to revise the chapter based on specific feedback. The revised draft is re-evaluated. If the revision is too similar to the original, it may be rejected.
        *   **Finalization**: The (potentially revised) chapter text, a summary, its embedding, and raw LLM interaction logs are saved to the database. The chapter is marked as `is_provisional` if it's based on a draft that failed final evaluation but was proceeded with. Chapter text and raw logs are also saved to `.txt` files.
        *   **Knowledge Base Update**:
            *   The finalized chapter text is analyzed by an LLM to extract updates for character profiles and world-building information (JSON files). Updates derived from provisional chapters are themselves marked with provisional flags (e.g., `source_quality_chapter_X: "provisional_from_unrevised_draft"`) within the JSON structures.
            *   Factual triples (Subject, Predicate, Object) are extracted from the chapter text by an LLM and added to the Knowledge Graph in the database. These triples are also marked as `is_provisional` if the source chapter was provisional.
        *   The agent's internal state (plot, characters, world) is saved to JSON files, cleaning up temporary provisional flags from the top level of these files.
4.  **Completion**: The run finishes after attempting the configured number of chapters.

## Setup and Installation

### Prerequisites

*   Python 3.8+
*   Access to an Ollama server running an embedding model (e.g., `nomic-embed-text`).
*   Access to an OpenAI-compatible API endpoint for LLM text generation.

### Dependencies

Install the required Python packages:

```bash
pip install numpy requests
```
(It's recommended to create a `requirements.txt` file for easier dependency management.)

### Configuration

1.  Copy or rename `config.py.example` to `config.py` (if an example file is provided; otherwise, directly edit `config.py`).
2.  Edit `config.py` to set:
    *   `OLLAMA_EMBED_URL`: URL for your Ollama embedding server.
    *   `OPENAI_API_BASE`: Base URL for your OpenAI-compatible LLM API.
    *   `OPENAI_API_KEY`: Your API key.
    *   `EMBEDDING_MODEL`: Name of the embedding model in Ollama.
    *   `MAIN_GENERATION_MODEL`: Name of the text generation LLM.
    *   File paths (`OUTPUT_DIR`, etc.) if you want to change defaults.
    *   Other generation parameters, thresholds, and logging settings as desired.

## Running the System

Execute the main script from the project's root directory:

```bash
python main.py
```

The system will start generating the novel based on your configuration. Progress will be printed to the console, and detailed logs will be saved to the file specified in `config.LOG_FILE` (if enabled).

## Output

All generated files and data are stored in the directory specified by `config.OUTPUT_DIR` (default: `novel_output/`):

*   `novel_data.db`: SQLite database containing chapter texts, summaries, embeddings, and the knowledge graph.
*   `plot_outline.json`: The generated plot outline.
*   `character_profiles.json`: Character information.
*   `world_building.json`: World-building details.
*   `chapter_N.txt`: Text file for each generated chapter.
*   `chapter_N_raw_log.txt`: Raw LLM interactions for generating/revising chapter N.
*   `saga_run.log`: Log file for the system's operations (if enabled).
*   `debug/`: Subdirectory containing intermediate LLM outputs if issues occur during certain stages (e.g., failed JSON parsing, short revisions).

## Advanced Concepts

*   **Knowledge Graph (KG)**: SAGA builds a KG of (Subject, Predicate, Object) triples extracted from chapters. This KG serves as a structured memory, helping maintain factual consistency. Queries to the KG can retrieve specific information (e.g., a character's last known location).
*   **Provisional Data Handling**: When a chapter draft fails evaluation but the system proceeds (e.g., after a failed revision attempt), the resulting chapter data and any knowledge extracted from it (KG triples, character/world updates) are marked as "provisional." This allows the system to:
    *   Distinguish between high-confidence and potentially lower-confidence information.
    *   Prioritize reliable (non-provisional) KG facts during consistency checks and planning.
    *   Inform the LLM during generation or analysis if context data is from a provisional source (via `prompt_notes` in JSON contexts).
*   **Dynamic State Adaptation**: The system allows the LLM to propose changes (`modification_proposal`) to existing character profiles or world-building elements during the knowledge update phase. This enables the story world and its inhabitants to evolve more organically based on chapter events.
*   **Agentic Planning**: Before writing a chapter, an LLM can be tasked with creating a high-level plan (key beats/scenes). This helps guide the subsequent drafting process, ensuring the chapter stays focused and contributes to the overall narrative arc.

## Configuration Options

The `config.py` file provides extensive options to customize SAGA's behavior:

*   **API and Models**: Specify your LLM and embedding model endpoints and names.
*   **Output Paths**: Define where generated files are stored.
*   **Generation Parameters**: Control context length, max tokens, number of chapters per run, etc.
*   **Agentic Planning**: Enable/disable the planning step and set token limits for it.
*   **Revision & Validation**: Adjust thresholds for coherence, consistency checks, and minimum draft length.
*   **Unhinged Plot Mode**:
    *   Set `UNHINGED_PLOT_MODE = True` for randomized, often paradoxical, story seeds.
    *   If `False`, configure `CONFIGURED_GENRE`, `CONFIGURED_THEME`, and `CONFIGURED_SETTING_DESCRIPTION` for a more directed plot generation.
    *   The lists `UNHINGED_GENRES`, `UNHINGED_THEMES`, etc., can be expanded for more variety in unhinged mode.
*   **Logging**: Set log level and output file.

Review `config.py` thoroughly to tailor SAGA to your needs.

## License

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
