# AI Devs 4 Project

## Overview

Python project built with `uv` for the AI Devs 4 course. Provides utilities for API interaction, data management, and LLM integration via DSPy.

## Project Structure

```
ai_devs4/
├── src/ai_devs/           # Core package (installable)
│   ├── __init__.py        # Public API exports
│   ├── config.py          # Config class, configure_logging()
│   ├── client.py          # fetch_data, submit_answer
│   └── io.py              # save_to_file, get_data
├── tasks/                 # Task scripts
├── data/                  # Downloaded data (gitignored)
├── pyproject.toml         # Project config
└── uv.lock                # Lock file
```

## Architecture

### Configuration
- `Config` class via `pydantic-settings` with auto-validation
- Dependency injection: pass `config` to functions, instantiate once in `main()`
- Fails fast if required env vars missing
- Auto-loads `.env` file

### Logging
- Call `configure_logging()` once at script entry point
- Format: `LEVEL name: message`
- Default level: `DEBUG`

### Package Layout
- `src/ai_devs/` - installable package, public API exported via `__init__.py`
- `tasks/` - executable scripts for course tasks
- Import: `from ai_devs import Config, fetch_data, ...`

## Dependencies

### Core
- **dspy** (v3.1.3) - Framework for LLM calls and AI pipelines
- **httpx** (v0.28.1) - HTTP client for API requests
- **pydantic-settings** (v2.13.1) - Settings management with env validation

### Development
- **ruff** (v0.15.6) - Linter and formatter

## Setup

### Prerequisites
- Python 3.12+
- `uv` package manager

### Installation
```bash
uv sync
```

### Environment Variables
Create `.env` file:
```
HUB_BASE_URL=https://api.example.com
HUB_API_KEY=your_key_here
MISTRAL_API_KEY=your_key_here  # Optional, for DSPy
```

## Development Workflow

### Lint and Format
```bash
.venv/bin/ruff check src/ tasks/
.venv/bin/ruff format src/ tasks/
.venv/bin/ruff check --fix src/ tasks/
```

### Add Dependencies
```bash
uv add package_name
uv add --dev package_name
```

### Run Tasks
```bash
uv run python tasks/1_people.py
```

## Key Technologies

### DSPy
Framework for composing LLM prompts:
- **Signatures**: Define input/output structure
- **Modules**: Reusable LLM components
- **Prompt Optimization**: Fine-tuning tools

### pydantic-settings
Settings management with validation:
- Auto-loads from `.env`
- Type coercion
- Required field validation at startup

### uv
Fast Python package manager:
- Replaces pip, venv
- Deterministic builds via `uv.lock`

## Documentation Search

When you need to search docs, use `context7` tools.
