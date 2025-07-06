# AGENTS Guide - SAGA Novel Generation

This guide provides concise instructions for Codex and other automation agents working with the SAGA codebase.

## Overview
SAGA is an autonomous novel-writing system powered by the NANA engine. It relies on Python 3.10+, Neo4j, Jinja2 templates and OpenAI-compatible LLM APIs.

## Repository Layout
```
/agents              # Specialized AI agents
/chapter_generation  # Chapter generation services
/core                # Core infrastructure
/data_access         # Database access layer
/ingestion           # Text ingestion
/initialization      # Story genesis and bootstrap
/kg_maintainer       # Knowledge graph merge utilities
/models              # Shared pydantic models
/orchestration       # High level orchestration
/processing          # Text processing pipeline
/storage             # File I/O helpers
/ui                  # Optional Rich CLI
/utils               # Misc helpers
/prompts             # Jinja2 templates
main.py              # CLI entry
config.py            # Configuration via Pydantic
```

## Development Setup
- Create a virtual environment: `python -m venv venv && source venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`
- Install development tools: `pip install ruff mypy pytest-cov`
- Copy `.env.example` to `.env` and update credentials
- Start Neo4j: `docker-compose up -d neo4j`

## Coding Standards
- Python 3.10+ required
- Include the relative file path as the first comment in every source file
- Format and lint with `ruff`
- Use type hints for all functions and methods
- Write Google style docstrings
- Organize imports: standard library, third party, local
- Use async/await for I/O bound operations

## Testing Requirements
- Primary framework: `pytest`
- Run `pytest -v --cov=. --cov-report=term-missing` for coverage
- Lint with `ruff check .` and type check with `mypy .`
- Run tests and linters for any change that modifies code
- Documentation or comment-only changes may skip tests and linters
- Complexity analysis runs in CI via `python complexity_report.py`; the workflow
  fails if `radon` exits with a non-zero status

## Pull Request Guidelines
- Title format: `[COMPONENT] Brief description`
- Include in the description:
  1. Summary of changes
  2. Agent modifications
  3. Database or config updates
  4. Testing performed
  5. Performance considerations
- Ensure all tests and quality checks pass before merging

## Useful Commands
- Format and lint: `ruff check . && ruff format .`
- Type check: `mypy .`
- Full quality suite: `ruff check . && ruff format . && mypy . && pytest -v --cov=. --cov-report=term-missing`
- Run SAGA: `docker-compose up -d neo4j && python main.py`

## Environment Variables
- `OPENAI_API_BASE`, `OPENAI_API_KEY`
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
- `OLLAMA_EMBED_URL`, `EMBEDDING_MODEL`
- `AGENT_LOG_LEVEL`, `AGENT_ENABLE_PATCH_VALIDATION`
- `CHAPTERS_PER_RUN`, `MAX_REVISION_CYCLES_PER_CHAPTER`

## Model Configuration Example
```python
MODELS = {
    "planning": LARGE_MODEL,
    "evaluation": LARGE_MODEL,
    "drafting": NARRATOR_MODEL,
    "patches": MEDIUM_MODEL,
    "summaries": SMALL_MODEL,
    "kg_updates": MEDIUM_MODEL,
}
```

---
**Important Notes**
- Keep AGENTS.md updated as the project evolves
- Follow the revision pipeline patterns
- Maintain Neo4j schema consistency
- Respect configuration management for new features
