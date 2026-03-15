from typing import Any

import httpx

from ai_devs.config import Config


def fetch_data(file_name: str, config: Config) -> bytes:
    """Fetch data from hub, return raw bytes."""
    url = f"{config.hub_base_url.rstrip('/')}/data/{config.hub_api_key}/{file_name}"
    response = httpx.get(url, timeout=config.http_timeout)
    response.raise_for_status()
    return response.content


def submit_answer(task: str, answer: Any, config: Config) -> dict:
    """Submit answer to verification endpoint, return parsed response."""
    url = f"{config.hub_base_url.rstrip('/')}/verify"
    payload = {"apikey": config.hub_api_key, "task": task, "answer": answer}
    response = httpx.post(url, json=payload, timeout=config.http_timeout)
    response.raise_for_status()
    return response.json()
