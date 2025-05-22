# SAGA: Semantic And Graph-enhanced Authoring

An autonomous, agentic creative writing system that generates full-length novels using advanced AI techniques, semantic context understanding, and knowledge graph management.

## 🚀 Key Features

### Core Capabilities
- **Autonomous Novel Generation**: Generates complete novels chapter by chapter with minimal human intervention
- **Knowledge Graph Integration**: Uses Neo4j to maintain rich relationships between characters, locations, plot points, and world elements
- **Semantic Context Generation**: Employs embeddings to retrieve relevant context from previous chapters for narrative consistency
- **Agentic Scene Planning**: Plans detailed scenes with character interactions, dialogue points, and narrative focus before writing
- **Quality Evaluation & Revision**: Automatically evaluates chapter quality and performs targeted revisions when needed
- **Dynamic Knowledge Management**: Continuously updates character profiles and world-building as the story progresses

### Advanced Features
- **Hybrid Context System**: Combines semantic similarity search with knowledge graph facts for optimal context
- **Patch-based Revision**: Makes surgical edits to specific problematic sections rather than rewriting entire chapters
- **Multi-Model Architecture**: Uses different specialized models for planning, drafting, evaluation, and revision tasks
- **Embedding Coherence Checking**: Ensures narrative flow consistency between chapters using cosine similarity
- **Provisional Data Handling**: Tracks and manages potentially unreliable information from unrevised drafts
- **Comprehensive Caching**: LRU caching for embeddings, summaries, and token counting for performance

### Generation Modes
- **Configured Mode**: Uses predefined genre, theme, and setting parameters
- **Unhinged Mode**: Randomly selects from curated lists of genres, themes, protagonists, and conflicts for creative variety
- **User-Supplied Mode**: Accepts detailed story elements via JSON input file

## 🏗️ Architecture Overview

SAGA follows a modular, async-first architecture:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Novel Agent   │───▶│   State Manager  │───▶│     Neo4j       │
│   (Orchestrator)│    │   (Persistence)  │    │  (Knowledge     │
└─────────────────┘    └──────────────────┘    │   Graph)        │
         │                       │              └─────────────────┘
         ▼                       ▼              
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Logic Modules  │    │  LLM Interface   │───▶│  OpenAI-style   │
│  (Planning,     │    │  (API calls,     │    │    API          │
│   Drafting,     │    │   Embeddings)    │    │  + Ollama       │
│   Evaluation)   │    │                  │    │  (Embeddings)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🔄 Pipeline Flow

### 1. Initial Setup
- **Plot Generation**: Creates story outline with protagonist, plot points, and narrative arc
- **World Building**: Establishes locations, societies, systems, lore, and factions
- **Character Creation**: Develops initial character profiles and relationships
- **KG Pre-population**: Seeds Neo4j with initial story elements and relationships

### 2. Chapter Generation Loop
For each chapter:

1. **Scene Planning** (Optional): Plans 10-18 detailed scenes with character interactions and dialogue points
2. **Context Generation**: 
   - Retrieves semantically relevant previous chapters using embedding similarity
   - Fetches reliable knowledge graph facts about characters and world elements
   - Combines into hybrid context for optimal narrative consistency
3. **Chapter Drafting**: Generates initial chapter text using hybrid context and scene plan
4. **Quality Evaluation**: Analyzes chapter for:
   - Consistency with established canon
   - Plot arc advancement
   - Thematic alignment
   - Narrative depth and length
5. **Revision Process** (if needed):
   - **Patch-based**: Makes targeted fixes to specific problematic quotes
   - **Full Rewrite**: Complete chapter regeneration for major issues
6. **Knowledge Updates**: Extracts and updates character developments, world changes, and new relationships
7. **Persistence**: Saves chapter text, embeddings, and metadata to both Neo4j and local files

### 3. Continuous Learning
- Updates character status and relationships based on story events
- Expands world-building with new locations and lore
- Maintains provisional data tracking for quality control
- Builds comprehensive knowledge graph for future context retrieval

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Docker and Docker Compose
- OpenAI-compatible API endpoint (local or remote)
- Ollama for embeddings (recommended)

### Quick Start

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/saga.git
cd saga
```

2. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

3. **Start Neo4j database**:
```bash
chmod +x manage_neo4j.sh
./manage_neo4j.sh start
```

4. **Configure environment variables** (create `.env` file):
```bash
# LLM API Configuration
OPENAI_API_BASE=http://127.0.0.1:8080/v1  # Your LLM API endpoint
OPENAI_API_KEY=your_api_key_here

# Ollama for embeddings
OLLAMA_EMBED_URL=http://127.0.0.1:11434
EMBEDDING_MODEL=nomic-embed-text:latest

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=saga_password

# Model Configuration (adjust for your setup)
LARGE_MODEL=Qwen3-14B
MEDIUM_MODEL=Qwen3-8B
SMALL_MODEL=Qwen3-4B
```

5. **Run SAGA**:
```bash
python main.py
```

## 🎛️ Neo4j Management

Use the provided script to manage your Neo4j instance:

```bash
# Start Neo4j
./manage_neo4j.sh start

# Check status
./manage_neo4j.sh status

# Stop Neo4j
./manage_neo4j.sh stop
```

### Accessing Neo4j Browser
- **URL**: http://localhost:7474
- **Username**: neo4j
- **Password**: saga_password

You can explore the knowledge graph visually, query relationships, and monitor the story's evolving structure in real-time.

### Useful Neo4j Queries

**View all characters and their relationships**:
```cypher
MATCH (c:Character)-[r:DYNAMIC_REL]->(target)
RETURN c.name, r.type, target.name
```

**Explore world elements by category**:
```cypher
MATCH (we:WorldElement)
RETURN we.category, we.name, we.description
ORDER BY we.category, we.name
```

**Track plot progression**:
```cypher
MATCH (ni:NovelInfo)-[:HAS_PLOT_POINT]->(pp:PlotPoint)
RETURN pp.sequence, pp.description
ORDER BY pp.sequence
```

## ⚙️ Configuration

### Model Configuration
SAGA supports multiple models for different tasks:
- **LARGE_MODEL**: Complex reasoning (planning, evaluation)
- **MEDIUM_MODEL**: Balanced tasks (knowledge updates, patch generation)
- **SMALL_MODEL**: Simple tasks (JSON correction, quick summaries)
- **NARRATOR_MODEL**: High-quality text generation (drafting, revision)

### Generation Parameters
Key settings in `config.py`:
- `CHAPTERS_PER_RUN`: Number of chapters to generate per execution
- `MIN_ACCEPTABLE_DRAFT_LENGTH`: Minimum character count for chapters
- `TARGET_SCENES_MIN/MAX`: Scene count range for chapter planning
- `ENABLE_AGENTIC_PLANNING`: Toggle detailed scene planning
- `ENABLE_PATCH_BASED_REVISION`: Enable targeted revision vs. full rewrites
- `UNHINGED_PLOT_MODE`: Enable random story element generation

### Story Input Options

**Option 1: Configured Generation**
Set story parameters in `config.py`:
```python
CONFIGURED_GENRE = "dystopian horror"
CONFIGURED_THEME = "the cost of power"
CONFIGURED_SETTING_DESCRIPTION = "a walled city where memories extend lifespan"
```

**Option 2: User-Supplied Elements**
Create `user_story_elements.json`:
```json
{
  "novel_concept": {
    "title": "The Memory Merchants",
    "genre": "dystopian fiction",
    "theme": "the price of immortality",
    "logline": "In a world where memories can be traded for extended life..."
  },
  "protagonist": {
    "name": "Aria Chen",
    "description": "A memory broker questioning the system",
    "character_arc": "From complicit trader to revolutionary leader"
  },
  "plot_points": [
    "Aria discovers a forbidden memory cache",
    "She uncovers the truth about the memory trade",
    "..."
  ]
}
```

## 📊 Output Structure

SAGA generates organized output:
```
novel_output/
├── chapters/                 # Final chapter text files
│   ├── chapter_0001.txt
│   └── chapter_0002.txt
├── chapter_logs/            # Raw LLM outputs for debugging
│   ├── chapter_0001_raw_llm_log.txt
│   └── chapter_0002_raw_llm_log.txt
├── debug_outputs/           # Debug information and failed attempts
├── plot_outline.json        # Backup of plot structure
├── character_profiles.json  # Backup of character data
├── world_building.json      # Backup of world data
└── saga_run.log            # Comprehensive system logs
```

## 🔍 Advanced Features

### Quality Control
- **Coherence Scoring**: Measures narrative consistency between chapters
- **Evaluation Categories**: Consistency, plot advancement, thematic alignment, narrative depth
- **Provisional Data Tracking**: Marks potentially unreliable information from unrevised drafts
- **Retry Logic**: Automatic fallback models and retry mechanisms for reliability

### Performance Optimization
- **Async Operations**: Non-blocking I/O throughout the system
- **Smart Caching**: LRU caches for embeddings, summaries, and tokenization
- **Token Management**: Intelligent prompt truncation and token counting
- **Batch Operations**: Efficient Neo4j batch transactions

### Extensibility
- **Modular Logic**: Easy to add new generation steps or modify existing ones
- **Plugin Architecture**: Clean separation of concerns allows easy customization
- **Model Agnostic**: Works with any OpenAI-compatible API
- **Format Flexibility**: Plain text parsing reduces dependency on specific model output formats

## 🐛 Troubleshooting

### Common Issues

**Neo4j Connection Errors**:
```bash
./manage_neo4j.sh status  # Check if Neo4j is running
./manage_neo4j.sh start   # Start if stopped
```

**LLM API Timeouts**:
- Check your API endpoint is accessible
- Adjust timeout settings in `config.py`
- Verify model names match your API setup

**Memory Issues**:
- Reduce `MAX_CONTEXT_TOKENS` for smaller models
- Lower `CHAPTERS_PER_RUN` for less memory usage
- Check embedding model requirements

**Generation Quality**:
- Increase `MIN_ACCEPTABLE_DRAFT_LENGTH` for longer chapters
- Enable `ENABLE_AGENTIC_PLANNING` for better structure
- Adjust temperature settings for creativity vs. consistency

## 📝 Logging

SAGA provides comprehensive logging at multiple levels:
- **INFO**: Major operations and progress
- **DEBUG**: Detailed operation traces
- **WARNING**: Non-fatal issues and fallbacks
- **ERROR**: Failures and exceptions

Logs are written to both console and `novel_output/saga_run.log`.

## 🤝 Contributing

SAGA is open source and welcomes contributions! Areas of particular interest:
- Additional LLM provider integrations
- New evaluation metrics and revision strategies
- Performance optimizations
- User interface improvements
- Documentation and examples

## 📄 License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

SAGA builds upon research in autonomous agents, large language models, and knowledge representation. Special thanks to the open source community for foundational tools like Neo4j, Llama.cpp, Ollama, and the various Python libraries that make this system possible.

---

**Ready to write your novel?** Start with `python main.py` and watch SAGA craft your story, one chapter at a time! 📚✨