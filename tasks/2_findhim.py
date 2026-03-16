import logging
import sys
from pathlib import Path

import httpx

from ai_devs import Config, configure_logging, ensure_data_file

logger = logging.getLogger(__name__)

DATA_FILE = Path("data/findhim_locations.json")


def main() -> None:
    configure_logging()
    config = Config()

    try:
        ensure_data_file("findhim_locations.json", DATA_FILE, config)

    except httpx.HTTPStatusError as e:
        logger.error("HTTP error: %s", e)
        logger.error("Response body: %s", e.response.text)
        sys.exit(1)
    except httpx.HTTPError as e:
        logger.error("HTTP error: %s", e)
        sys.exit(1)
    except OSError as e:
        logger.error("IO error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
