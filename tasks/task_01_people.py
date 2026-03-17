import csv
import logging
import sys
from datetime import date
from pathlib import Path

import dspy
import httpx

from ai_devs import Config, configure_logging, ensure_data_file, submit_answer

logger = logging.getLogger(__name__)

DATA_FILE = Path("data/people.csv")

AVAILABLE_TAGS = [
    "IT",
    "transport",
    "edukacja",
    "medycyna",
    "praca z ludźmi",
    "praca z pojazdami",
    "praca fizyczna",
]


def calculate_age(birth_date: str) -> int:
    """Calculate age from YYYY-MM-DD format."""
    born = date.fromisoformat(birth_date)
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def extract_birth_year(birth_date: str) -> int:
    """Extract year from YYYY-MM-DD format."""
    return int(birth_date.split("-")[0])


def filter_people(filepath: Path) -> list[dict]:
    """Filter people: age 20-40, birthPlace='Grudziądz'."""
    filtered = []
    with filepath.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["birthPlace"] != "Grudziądz":
                continue
            age = calculate_age(row["birthDate"])
            if 20 <= age <= 40:
                filtered.append(row)
    logger.info("Filtered %d people from Grudziądz aged 20-40", len(filtered))
    return filtered


class JobTagger(dspy.Signature):
    """Assign at least one job category tag based on Polish job description. If unsure, pick the most relevant tag from the allowed list."""

    job_description: str = dspy.InputField()
    tags: list[str] = dspy.OutputField(
        desc=f"At least one tag from this list (required): {AVAILABLE_TAGS}"
    )


def assign_tags(people: list[dict], predictor: dspy.Predict) -> list[dict]:
    """Add tags to each person using LLM."""
    results = []
    for person in people:
        response = predictor(job_description=person["job"])
        tags = response.tags if response.tags else ["praca z ludźmi"]
        result = {
            "name": person["name"],
            "surname": person["surname"],
            "gender": person["gender"],
            "born": extract_birth_year(person["birthDate"]),
            "city": person["birthPlace"],
            "tags": tags,
        }
        results.append(result)
        logger.debug(
            "Assigned tags %s to %s %s", tags, person["name"], person["surname"]
        )
    return results


def main() -> None:
    configure_logging()
    config = Config()

    try:
        ensure_data_file("people.csv", DATA_FILE, config)

        lm = dspy.LM(
            model="mistral/mistral-large-latest", api_key=config.mistral_api_key
        )
        dspy.configure(lm=lm, adapter=dspy.JSONAdapter())

        filtered_people = filter_people(DATA_FILE)

        tagger = dspy.Predict(JobTagger)
        tagged_people = assign_tags(filtered_people, tagger)

        answer = [p for p in tagged_people if "transport" in p["tags"]]
        logger.info("Filtered to %d people with 'transport' tag", len(answer))

        logger.info("Submitting answer with %d people", len(answer))
        response = submit_answer("people", answer, config)
        logger.info("Response: %s", response)

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
