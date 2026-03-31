"""Configuration for the TTS server. Reads from config.ini."""
import os
from configparser import ConfigParser
from typing import List

from pydantic_settings import BaseSettings


def _load_ini(path: str = "config.ini") -> dict:
    """Load config.ini and return a flat dict of settings."""
    parser = ConfigParser()
    defaults = {}

    if os.path.exists(path):
        parser.read(path)

        section_map = {
            "server": {"server_host": str, "server_port": int},
            "paths": {"voices_dir": str, "embeddings_dir": str, "audio_cache_dir": str},
            "cache": {"cache_limit": int, "cache_cleanup_interval": int},
            "security": {
                "allowed_origins": str, "rate_limit_requests": int,
                "rate_limit_window": int, "enable_high_priority": bool,
                "model_load_timeout": int,
            },
            "validation": {"max_input_length": int, "min_input_length": int},
            "audio": {
                "queue_size": int, "queue_timeout": float,
                "eof_timeout": float, "chunk_size": int,
            },
            "tts": {
                "temperature": float, "lsd_decode_steps": int,
                "top_p": float, "repetition_penalty": float, "model_tier": str,
            },
        }

        for section, fields in section_map.items():
            if not parser.has_section(section):
                continue
            for key, typ in fields.items():
                if not parser.has_option(section, key):
                    continue
                raw = parser.get(section, key).strip()
                # Strip inline comments
                if ";" in raw:
                    raw = raw[:raw.index(";")].strip()
                if not raw:
                    continue
                if typ == bool:
                    defaults[key] = raw.lower() in ("true", "1", "yes")
                elif typ == int:
                    defaults[key] = int(raw)
                elif typ == float:
                    defaults[key] = float(raw)
                else:
                    defaults[key] = raw

        # Parse allowed_origins as list
        if "allowed_origins" in defaults:
            raw = defaults["allowed_origins"]
            if raw == "*":
                defaults["allowed_origins"] = ["*"]
            else:
                defaults["allowed_origins"] = [o.strip() for o in raw.split(",") if o.strip()]

    return defaults


_ini = _load_ini()


class Settings(BaseSettings):
    # Server settings
    server_host: str = _ini.get("server_host", "0.0.0.0")
    server_port: int = _ini.get("server_port", 8005)

    # Path settings
    voices_dir: str = _ini.get("voices_dir", "voices")
    embeddings_dir: str = _ini.get("embeddings_dir", "embeddings")
    audio_cache_dir: str = _ini.get("audio_cache_dir", "audio_cache")

    # Model settings
    default_sample_rate: int = 24000

    # Generation settings
    temperature: float = _ini.get("temperature", 0.7)
    top_p: float = _ini.get("top_p", 0.95)
    repetition_penalty: float = _ini.get("repetition_penalty", 1.1)
    lsd_decode_steps: int = _ini.get("lsd_decode_steps", 2)
    model_tier: str = _ini.get("model_tier", "tts-1")

    # Input validation settings
    max_input_length: int = _ini.get("max_input_length", 4096)
    min_input_length: int = _ini.get("min_input_length", 1)

    # Cache settings
    cache_limit: int = _ini.get("cache_limit", 10)
    cache_cleanup_interval: int = _ini.get("cache_cleanup_interval", 300)

    # Streaming settings
    chunk_size: int = _ini.get("chunk_size", 8192)
    eof_timeout: float = _ini.get("eof_timeout", 30.0)
    queue_size: int = _ini.get("queue_size", 200)

    # Security settings
    allowed_origins: List[str] = _ini.get("allowed_origins", ["*"])
    allowed_origin_regex: str = ""
    rate_limit_requests: int = _ini.get("rate_limit_requests", 120)
    rate_limit_window: int = _ini.get("rate_limit_window", 60)
    enable_high_priority: bool = _ini.get("enable_high_priority", True)
    model_load_timeout: int = _ini.get("model_load_timeout", 300)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
