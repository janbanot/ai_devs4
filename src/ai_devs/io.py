from pathlib import Path

from ai_devs.client import fetch_data
from ai_devs.config import Config


def save_to_file(data: bytes, path: str | Path) -> None:
    """Write bytes to file, creating parent dirs if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def get_data(file_name: str, config: Config, save_path: str | None = None) -> None:
    """Convenience: fetch and save data."""
    data = fetch_data(file_name, config)
    save_to_file(data, save_path or file_name)


def ensure_data_file(file_name: str, path: Path, config: Config) -> None:
    """Download data file if it doesn't exist locally."""
    if path.exists():
        return
    data = fetch_data(file_name, config)
    save_to_file(data, path)
