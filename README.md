# AI Devs 4

Python project for AI Devs 4 course with DSPy and LLM integration.

## Quick Start

```bash
uv sync
uv run python tasks/1_people.py
```

## Project Structure

```
ai_devs4/
├── src/ai_devs/        # Core package
│   ├── config.py       # Config class, configure_logging()
│   ├── client.py       # fetch_data, submit_answer
│   └── io.py           # save_to_file, get_data
├── tasks/              # Task scripts
├── data/               # Downloaded data (gitignored)
└── pyproject.toml
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HUB_BASE_URL` | Yes | API base URL |
| `HUB_API_KEY` | Yes | API key |
| `MISTRAL_API_KEY` | No | For DSPy/Mistral |

## Development

```bash
.venv/bin/ruff check src/ tasks/      # Lint
.venv/bin/ruff format src/ tasks/     # Format
uv add package_name                   # Add dependency
```
