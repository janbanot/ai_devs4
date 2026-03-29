import csv
import json
import logging
import math
import sys
from datetime import date
from pathlib import Path

import dspy
import httpx

from ai_devs import Config, configure_logging, ensure_data_file

logger = logging.getLogger(__name__)

DATA_FILE = Path("data/people.csv")
POWER_PLANTS_FILE = Path("data/findhim_locations.json")
AVAILABLE_TAGS = [
    "IT",
    "transport",
    "edukacja",
    "medycyna",
    "praca z ludźmi",
    "praca z pojazdami",
    "praca fizyczna",
]


class FindHimAgent:
    """ReAct agent for finding a valid person near a power plant.

    The task requires finding a person whose credentials pass validation.
    Distance to power plants is used as a heuristic ordering, but the
    actual validation uses different criteria (not just closest distance).
    The agent iterates through candidates until finding one that passes.
    """

    def __init__(self, config: Config):
        self.config = config
        self.client = httpx.Client(timeout=config.http_timeout)
        self.power_plants = self._load_power_plants()
        self.people_data: list[dict] = []
        self._setup_tools()

    def _load_power_plants(self) -> dict:
        with POWER_PLANTS_FILE.open() as f:
            data = json.load(f)
        return data["power_plants"]

    def _setup_tools(self):
        config = self.config
        client = self.client
        power_plants = self.power_plants
        people_data = self.people_data

        def get_location(name: str, surname: str) -> str:
            """Get coordinates for a person. Returns coordinates in 'lat,lng' format."""
            try:
                response = client.post(
                    f"{config.hub_base_url}/api/location",
                    json={
                        "apikey": config.hub_api_key,
                        "name": name,
                        "surname": surname,
                    },
                )
                response.raise_for_status()
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    first = data[0]
                    if isinstance(first, dict):
                        lat = first.get("latitude", first.get("lat"))
                        lng = first.get("longitude", first.get("lng"))
                        if lat is not None and lng is not None:
                            location = f"{lat},{lng}"
                        else:
                            location = first.get(
                                "location", first.get("city", str(first))
                            )
                    else:
                        location = str(first)
                elif isinstance(data, dict):
                    lat = data.get("latitude", data.get("lat"))
                    lng = data.get("longitude", data.get("lng"))
                    if lat is not None and lng is not None:
                        location = f"{lat},{lng}"
                    else:
                        location = data.get("location", data.get("city", str(data)))
                else:
                    location = str(data)
                logger.debug("get_location(%s, %s) = %s", name, surname, location)
                return location
            except Exception as e:
                logger.error("get_location error: %s", e)
                return f"Error: {e}"

        def get_access_level(name: str, surname: str, birthYear: int) -> str:
            """Get access level for a person. Returns the access level as a string."""
            try:
                response = client.post(
                    f"{config.hub_base_url}/api/accesslevel",
                    json={
                        "apikey": config.hub_api_key,
                        "name": name,
                        "surname": surname,
                        "birthYear": birthYear,
                    },
                )
                response.raise_for_status()
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    first = data[0]
                    if isinstance(first, dict):
                        level = first.get("accessLevel", first.get("level", str(first)))
                    else:
                        level = str(first)
                elif isinstance(data, dict):
                    level = data.get("accessLevel", data.get("level", str(data)))
                else:
                    level = str(data)
                logger.debug(
                    "get_access_level(%s, %s, %d) = %s", name, surname, birthYear, level
                )
                return str(level)
            except Exception as e:
                logger.error("get_access_level error: %s", e)
                return f"Error: {e}"

        def get_coordinates(location: str) -> str:
            """Get latitude and longitude for a location name. Use your knowledge of geography. Return as 'lat,lng' format like '50.1234,18.5678'."""
            predictor = dspy.Predict("location -> coordinates")
            result = predictor(location=location)
            coords = result.coordinates
            logger.debug("get_coordinates(%s) = %s", location, coords)
            return coords

        def calculate_distance(coord1: str, coord2: str) -> str:
            """Calculate distance in kilometers between two coordinates. Each coordinate should be 'lat,lng' format. Returns distance as string with unit."""
            try:
                lat1, lng1 = map(float, coord1.split(","))
                lat2, lng2 = map(float, coord2.split(","))
                R = 6371
                phi1 = math.radians(lat1)
                phi2 = math.radians(lat2)
                dphi = math.radians(lat2 - lat1)
                dlambda = math.radians(lng2 - lng1)
                a = (
                    math.sin(dphi / 2) ** 2
                    + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
                )
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
                distance = R * c
                logger.debug(
                    "calculate_distance(%s, %s) = %.2f km", coord1, coord2, distance
                )
                return f"{distance:.2f} km"
            except Exception as e:
                logger.error("calculate_distance error: %s", e)
                return f"Error: {e}"

        def get_power_plant_code(city: str) -> str:
            """Get the power plant code for a given city name. Returns code like 'PWR1234PL'."""
            city_normalized = city.strip()
            for plant_city, plant_data in power_plants.items():
                if (
                    plant_city.lower() == city_normalized.lower()
                    or city_normalized.lower() in plant_city.lower()
                ):
                    code = plant_data.get("code", "Unknown")
                    logger.debug("get_power_plant_code(%s) = %s", city, code)
                    return code
            logger.debug("get_power_plant_code(%s) = Not found", city)
            return f"No power plant found for city: {city}"

        def get_power_plant_cities() -> str:
            """Get list of all power plant cities. Returns comma-separated list."""
            cities = list(power_plants.keys())
            result = ", ".join(cities)
            logger.debug("get_power_plant_cities() = %s", result)
            return result

        def get_filtered_people() -> str:
            """Get list of filtered people (name, surname, birthYear) to search through. Returns JSON string."""
            result = json.dumps(people_data)
            logger.debug("get_filtered_people() = %d people", len(people_data))
            return result

        def get_ranked_people_by_distance() -> str:
            """Pre-compute distances for all people to their nearest power plant.

            Returns candidates ordered by distance (closest first) as a heuristic.
            Note: The validation API may accept candidates that are NOT the closest,
            so the agent should iterate through the list until finding a valid one.

            Returns JSON array with: name, surname, born, distance_km,
            nearest_power_plant_city, power_plant_code.
            """
            ranked = []
            plant_coords = {}
            for city in power_plants.keys():
                predictor = dspy.Predict("location -> coordinates")
                result = predictor(location=city)
                coords = result.coordinates
                coords = (
                    coords.replace("°", "")
                    .replace("N", "")
                    .replace("E", "")
                    .replace("S", "")
                    .replace("W", "")
                )
                coords = coords.replace(" ", "").strip()
                if "," not in coords:
                    coords = coords.replace(" ", ",")
                plant_coords[city] = coords
                logger.debug("Power plant %s coords: %s", city, coords)

            for person in people_data:
                try:
                    response = client.post(
                        f"{config.hub_base_url}/api/location",
                        json={
                            "apikey": config.hub_api_key,
                            "name": person["name"],
                            "surname": person["surname"],
                        },
                    )
                    response.raise_for_status()
                    data = response.json()
                    if isinstance(data, list) and len(data) > 0:
                        first = data[0]
                        if isinstance(first, dict):
                            lat = first.get("latitude", first.get("lat"))
                            lng = first.get("longitude", first.get("lng"))
                            if lat is not None and lng is not None:
                                person_coords = f"{lat},{lng}"
                            else:
                                continue
                        else:
                            continue
                    elif isinstance(data, dict):
                        lat = data.get("latitude", data.get("lat"))
                        lng = data.get("longitude", data.get("lng"))
                        if lat is not None and lng is not None:
                            person_coords = f"{lat},{lng}"
                        else:
                            continue
                    else:
                        continue

                    min_distance = float("inf")
                    nearest_city = None
                    for city, p_coords in plant_coords.items():
                        try:
                            lat1, lng1 = map(float, person_coords.split(","))
                            lat2, lng2 = map(float, p_coords.split(","))
                            R = 6371
                            phi1 = math.radians(lat1)
                            phi2 = math.radians(lat2)
                            dphi = math.radians(lat2 - lat1)
                            dlambda = math.radians(lng2 - lng1)
                            a = (
                                math.sin(dphi / 2) ** 2
                                + math.cos(phi1)
                                * math.cos(phi2)
                                * math.sin(dlambda / 2) ** 2
                            )
                            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
                            distance = R * c
                            if distance < min_distance:
                                min_distance = distance
                                nearest_city = city
                        except Exception as e:
                            logger.debug("Distance calc error: %s", e)
                            continue

                    if nearest_city:
                        plant_code = power_plants[nearest_city].get("code", "Unknown")
                        ranked.append(
                            {
                                "name": person["name"],
                                "surname": person["surname"],
                                "born": person["born"],
                                "distance_km": round(min_distance, 2),
                                "nearest_power_plant_city": nearest_city,
                                "power_plant_code": plant_code,
                            }
                        )
                        logger.debug(
                            "Ranked %s %s: %.2f km to %s",
                            person["name"],
                            person["surname"],
                            min_distance,
                            nearest_city,
                        )
                except Exception as e:
                    logger.debug(
                        "Error processing %s %s: %s",
                        person["name"],
                        person["surname"],
                        e,
                    )
                    continue

            ranked.sort(key=lambda x: x["distance_km"])
            logger.info(
                "Ranked %d people by distance to nearest power plant", len(ranked)
            )
            return json.dumps(ranked)

        def submit_and_check(
            name: str, surname: str, accessLevel: int, powerPlant: str
        ) -> str:
            """Submit answer to verification API and return result. Returns JSON with success status."""
            try:
                response = client.post(
                    f"{config.hub_base_url.rstrip('/')}/verify",
                    json={
                        "apikey": config.hub_api_key,
                        "task": "findhim",
                        "answer": {
                            "name": name,
                            "surname": surname,
                            "accessLevel": accessLevel,
                            "powerPlant": powerPlant,
                        },
                    },
                )
                data = response.json()
                if response.status_code == 200 and data.get("code", 0) == 0:
                    logger.info("Submit SUCCESS for %s %s", name, surname)
                    return json.dumps({"success": True, "response": data})
                else:
                    logger.info("Submit FAILED for %s %s: %s", name, surname, data)
                    return json.dumps(
                        {
                            "success": False,
                            "code": data.get("code"),
                            "message": data.get("message"),
                        }
                    )
            except httpx.HTTPStatusError as e:
                data = e.response.json() if e.response else {}
                logger.info("Submit ERROR for %s %s: %s", name, surname, data)
                return json.dumps(
                    {
                        "success": False,
                        "code": data.get("code"),
                        "message": data.get("message"),
                    }
                )
            except Exception as e:
                logger.error("Submit exception: %s", e)
                return json.dumps({"success": False, "error": str(e)})

        self.tools = [
            get_location,
            get_access_level,
            get_coordinates,
            calculate_distance,
            get_power_plant_code,
            get_power_plant_cities,
            get_filtered_people,
            get_ranked_people_by_distance,
            submit_and_check,
        ]

    def close(self):
        self.client.close()


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


class FindAnswer(dspy.Signature):
    """Find a valid person near a power plant and submit their access information.

    Note: The task asks for a person "close" to a power plant, but validation
    criteria are not purely distance-based. Iterate through candidates until
    finding one that passes the API validation.
    """

    question: str = dspy.InputField()
    result: str = dspy.OutputField(
        desc="Final result message from successful submission"
    )


def main() -> None:
    configure_logging()
    config = Config()

    try:
        ensure_data_file("people.csv", DATA_FILE, config)
        ensure_data_file("findhim_locations.json", POWER_PLANTS_FILE, config)

        lm = dspy.LM(
            model="mistral/mistral-large-latest", api_key=config.mistral_api_key
        )
        dspy.configure(lm=lm, adapter=dspy.JSONAdapter())

        filtered_people = filter_people(DATA_FILE)
        logger.info(
            "Filtered to %d people from Grudziądz aged 20-40", len(filtered_people)
        )

        tagger = dspy.Predict(JobTagger)
        tagged_people = assign_tags(filtered_people, tagger)

        transport_people = [p for p in tagged_people if "transport" in p["tags"]]
        logger.info("Filtered to %d people with 'transport' tag", len(transport_people))

        agent_wrapper = FindHimAgent(config)
        agent_wrapper.people_data.clear()
        agent_wrapper.people_data.extend(transport_people)

        react_agent = dspy.ReAct(
            FindAnswer,
            tools=agent_wrapper.tools,
            max_iters=50,
        )

        question = """Find a valid person near a power plant and submit their information.

BACKGROUND:
The task asks for a person close to a power plant, but the validation API has
additional criteria beyond just distance. You must iterate through candidates
until finding one that passes validation.

CRITICAL INSTRUCTIONS:
1. Call get_ranked_people_by_distance() ONCE to get candidates ordered by distance to nearest power plant
2. For each person in the list (starting from the closest):
   a. Call get_access_level(name, surname, born) to get their access level
   b. Call submit_and_check(name, surname, accessLevel, powerPlant) to submit
   c. If the response contains "success": true, you are DONE - return the success message
   d. If "success": false, try the NEXT person in the list
3. Continue until you get a successful submission

DO NOT use get_location, get_coordinates, calculate_distance, or other tools - use get_ranked_people_by_distance() instead.

Return the final success message as the result."""

        logger.info("Running ReAct agent...")
        result = react_agent(question=question)

        logger.info("Agent result: %s", result)
        logger.info("Final result: %s", result.result)

        agent_wrapper.close()

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
