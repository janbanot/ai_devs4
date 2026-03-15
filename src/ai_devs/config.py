import logging

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    hub_base_url: str
    hub_api_key: str
    http_timeout: float = 30.0

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


def configure_logging(level: int = logging.DEBUG) -> None:
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
    )
