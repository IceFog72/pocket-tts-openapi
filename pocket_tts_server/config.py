"""Configuration for the TTS server."""
import os
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Server settings
    server_host: str = "0.0.0.0"
    server_port: int = 8005

    # Path settings
    voices_dir: str = "voices"
    embeddings_dir: str = "embeddings"
    audio_cache_dir: str = "audio_cache"

    # Model settings
    default_sample_rate: int = 24000

    # Generation settings
    temperature: float = 0.7
    top_p: float = 0.95
    repetition_penalty: float = 1.1
    lsd_decode_steps: int = 2
    model_tier: str = "tts-1"

    # Input validation settings
    max_input_length: int = 4096
    min_input_length: int = 1

    # Cache settings
    cache_limit: int = 10
    cache_cleanup_interval: int = 300

    # Streaming settings
    chunk_size: int = 8192
    eof_timeout: int = 30
    queue_size: int = 200

    # Generation settings
    temperature: float = 0.7
    top_p: float = 0.95
    repetition_penalty: float = 1.1
    lsd_decode_steps: int = 2
    model_tier: str = "tts-1"

    # Security settings
    allowed_origins: List[str] = ["http://localhost", "http://127.0.0.1"]
    allowed_origin_regex: str = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"
    rate_limit_requests: int = 120
    rate_limit_window: int = 60
    enable_high_priority: bool = True
    model_load_timeout: int = 300

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
