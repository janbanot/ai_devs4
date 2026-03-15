import csv
import logging
import sys
from pathlib import Path

import httpx

from ai_devs import Config, configure_logging, fetch_data, save_to_file

logger = logging.getLogger(__name__)

DATA_FILE = Path("data/people.csv")


def ensure_data_file(config: Config) -> None:
    """Download data file if missing."""
    if DATA_FILE.exists():
        logger.info("Data file already exists")
        return

    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = fetch_data("people.csv", config)
    save_to_file(data, DATA_FILE)
    logger.info("Downloaded data file")


def read_first_row() -> dict[str, str]:
    """Read and return first row from CSV."""
    with DATA_FILE.open(newline="") as f:
        reader = csv.DictReader(f)
        try:
            return next(reader)
        except StopIteration:
            raise ValueError("CSV file is empty")


def main() -> None:
    configure_logging()
    config = Config()

    try:
        ensure_data_file(config)
        first_row = read_first_row()
        logger.info("First row: %s", first_row)
    except httpx.HTTPError as e:
        logger.error("HTTP error: %s", e)
        sys.exit(1)
    except OSError as e:
        logger.error("IO error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
