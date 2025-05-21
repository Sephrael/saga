# SAGA - Semantic And Graph-Enhanced Authoring

> ⚠️ WARNING 
> SAGA has recently begnu undergoing a migration from using SQLite exclusively, to a fully native neo4j + Cypher implementation. Because of this, SAGA might be a bit more verbose in the console logging or have some quirks.

SAGA (Semantic And Graph-enhanced Authoring) is a sophisticated AI-powered creative writing system designed to generate full-length novels with consistent characters, coherent world-building, and compelling narratives. Unlike simple prompt-based writing tools, SAGA employs a multi-stage pipeline that mirrors professional writing processes: planning, drafting, evaluation, and revision.

## 🌟 Key Features

- **Multi-Stage Writing Pipeline**: Separate planning, drafting, evaluation, and revision phases with specialized LLM prompts
- **Hybrid Knowledge Management**: Combines JSON-based character/world profiles with a knowledge graph for factual consistency
- **Intelligent Context Generation**: Uses semantic similarity and reliable knowledge facts to provide relevant context for each chapter
- **Comprehensive Quality Control**: Evaluates consistency, plot alignment, thematic coherence, and narrative depth
- **Agentic Planning**: Detailed scene-by-scene planning with focus elements for narrative depth
- **Provisional Data Tracking**: Marks data quality based on source reliability to maintain canon integrity
- **Adaptive Revision**: Targeted revision strategies based on specific evaluation feedback

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Planning      │    │   Drafting      │    │   Evaluation    │
│                 │───▶│                 │───▶│                 │
│ • Scene details │    │ • Context gen   │    │ • Consistency   │
│ • Focus elements│    │ • Target length │    │ • Plot alignment│
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐              │
│ Knowledge Update│◀───│    Revision     │◀─────────────┘
│                 │    │                 │
│ • Char profiles │    │ • Targeted fixes│
│ • World building│    │ • Length expand │
│ • Knowledge KG  │    │ • Quality improv│
└─────────────────┘    └─────────────────┘
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- OpenAI-compatible API endpoint (local LLM server recommended)
- Sufficient disk space for SQLite database and generated content
- neo4j instance to point at

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/saga.git
cd saga
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start up neo4j instance via provided docker-compose.yml file:
```bash
docker-compose up -d
```

### Quick Start

1. Configure your models in `config.py` or via environment variables
2. Run the system:
```bash
python main.py
```

3. The system will:
   - Generate or load a plot outline
   - Check for user-supplied story info
   - If found, processed as source of truth
   - If not found, uses defaults
   - Create initial world-building
   - Pre-populate the knowledge graph
   - Begin writing chapters iteratively
   - Resume from the last chapter it left off on

## ⚙️ Configuration

### Model Configuration

Configure your LLM models in `config.py`:

```python
# Primary models for different tasks
PLANNING_MODEL = "large-model-name"
DRAFTING_MODEL = "narrator-model-name"
EVALUATION_MODEL = "large-model-name"
REVISION_MODEL = "narrator-model-name"

# API endpoints
OPENAI_API_BASE = "http://localhost:8080/v1"
EMBEDDING_MODEL = "nomic-embed-text:latest"

# Neo4j Connection Settings
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "saga_password")
```

### Generation Parameters

Control output quality and length:

```python
# Target chapter length (characters)
MIN_ACCEPTABLE_DRAFT_LENGTH = 16000

# Scene planning targets
TARGET_SCENES_MIN = 10

# Context and generation limits
MAX_CONTEXT_LENGTH = 40960
CHAPTERS_PER_RUN = 3
```

### Novel Settings

Configure your story parameters:

```python
# Direct configuration
CONFIGURED_GENRE = "dystopian horror"
CONFIGURED_THEME = "the cost of power"
CONFIGURED_SETTING_DESCRIPTION = "a walled city where memories are traded for lifespan"

# Or enable "unhinged mode" for random combinations
UNHINGED_PLOT_MODE = True
```

## 📚 Core Components

### NovelWriterAgent
The main orchestrator that coordinates all writing phases and maintains state.

### State Manager
Handles persistence using SQLAlchemy with async support for chapters, embeddings, and knowledge graph data.

### Knowledge Management
- **Character Profiles**: Dynamic JSON structures tracking development across chapters
- **World Building**: Hierarchical organization of locations, systems, lore, and history
- **Knowledge Graph**: Factual triples for consistency and canonicity

### Context Generation
Hybrid system combining:
- Semantic similarity search across previous chapters
- Reliable knowledge graph facts relevant to current chapter
- Filtered character and world data with provisional markers

### Quality Control
Multi-dimensional evaluation:
- **Consistency**: Character behavior, world rules, established facts
- **Plot Alignment**: Advancement of intended plot points
- **Thematic Coherence**: Genre, theme, and character arc alignment
- **Narrative Depth**: Descriptive detail, pacing, and length targets

## 🔧 Advanced Features

### Provisional Data Tracking
The system marks data extracted from potentially flawed drafts as "provisional," allowing it to maintain integrity while working with imperfect intermediate content.

### Scene Focus Elements
Each planned scene includes specific focus elements that guide the drafting LLM toward deeper elaboration of particular aspects, naturally increasing chapter length and narrative richness.

### Unified Knowledge Extraction
A single LLM call extracts character updates, world-building changes, and knowledge graph facts from each chapter, reducing API calls and improving consistency.

### Intelligent Revision
The revision system detects the type of issues found during evaluation and applies targeted strategies, such as explicit expansion instructions for length-related problems.

## 📁 Project Structure

```
saga/
├── main.py                     # Entry point and execution logic
├── novel_agent.py              # Main NovelWriterAgent class
├── state_manager.py            # Database ORM and state persistence
├── config.py                   # Configuration and constants
├── type.py                     # Type definitions
├── chapter_planning_logic.py   # Scene planning and structure
├── chapter_drafting_logic.py   # Chapter text generation
├── chapter_evaluation_logic.py # Quality assessment
├── chapter_revision_logic.py   # Targeted improvements
├── context_generation_logic.py # Context preparation
├── knowledge_management_logic.py # Profile and KG updates
├── initial_setup_logic.py      # Plot and world generation
├── prompt_data_getters.py      # Data formatting for prompts
├── llm_interface.py            # LLM API interactions
├── utils.py                    # Utility functions
└── novel_output/               # Generated content directory
    ├── chapters/               # Final chapter texts
    ├── chapter_logs/           # Raw LLM outputs
    ├── debug_outputs/          # Debugging information
    └── novel_data.db          # SQLite database
```

## 🚦 Current Status

This system is actively developed and has successfully generated multi-chapter works with:
- Consistent character development across chapters
- Coherent world-building that evolves organically
- Plot advancement that follows planned structure
- Chapters regularly exceeding 16,000 characters with rich narrative depth

### Known Limitations

- Heavy reliance on LLM quality and consistency
- Processing time scales with chapter count due to context complexity
- Evaluation is primarily qualitative rather than quantitative

## 🤝 Contributing

This project is in early development. If you're interested in contributing:

1. Fork the repository
2. Create a feature branch
3. Make your changes with appropriate tests
4. Submit a pull request with detailed description

## 📄 License

Licensed under the Apache License, Version 2.0. See `LICENSE` file for details.

## 🙏 Acknowledgments

This system builds upon advances in large language models, async Python programming, and the creative writing process itself. Special thanks to the open-source community for the foundational tools and libraries.

---

For questions, issues, or discussions about creative AI, please open an issue on GitHub.