# AI Devs 4 Project

## Overview
This is a Python project built with `uv` as the package manager. It provides a foundation for working with large language models and building AI-powered applications.

## Project Structure
```
ai-devs4/
├── pyproject.toml       # Project configuration and dependencies
├── uv.lock             # Lock file for reproducible builds
├── .env                # Environment variables (not committed)
├── .python-version     # Python version specification
└── README.md           # Project documentation
```

## Dependencies

### Core Dependencies
- **dspy** (v3.1.3) - Framework for composing language model calls and building AI pipelines
- **python-dotenv** (v1.2.2) - Load environment variables from .env file

### Development Dependencies
- **ruff** (v0.15.6) - Fast Python linter and formatter

## Setup

### Prerequisites
- Python 3.12+
- `uv` package manager installed

### Installation
1. Clone the repository
2. Sync dependencies:
   ```bash
   uv sync
   ```
3. Create a `.env` file with any required API keys (e.g., MISTRAL_API_KEY, OPENAI_API_KEY)

## Development Workflow

### Format code with Ruff
```bash
.venv/bin/ruff format <file.py>
```

### Check code with Ruff
```bash
.venv/bin/ruff check <file.py>
```

### Fix issues automatically
```bash
.venv/bin/ruff check --fix <file.py>
```

### Add new dependencies
```bash
uv add package_name           # Add production dependency
uv add --dev package_name     # Add development dependency
```

### Update lock file
```bash
uv lock
```

## Key Technologies

### DSPy
DSPy is a framework for composing language model prompts and building AI applications. It provides:
- **Signatures**: Define input/output structure for tasks
- **Modules**: Reusable components that use language models
- **Prompt Optimization**: Tools for improving and fine-tuning prompts

### uv Package Manager
- Fast, reliable Python package and project manager
- Replaces pip, venv, and other tools
- Deterministic builds via `uv.lock` file
- Supports both production and development dependencies

## Environment Variables
Create a `.env` file in the project root for API keys and configuration:
```
API_KEY=your_key_here
```

Never commit `.env` files to version control.
