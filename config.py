from pydantic_settings import BaseSettings, SettingsConfigDict
import os
from typing import List

class Settings(BaseSettings):
    model_config = SettingsConfigDict(  # type: ignore
        env_file='.env', 
        env_file_encoding='utf-8', 
        extra='ignore'
    )

    # Server settings
    server_host: str = "0.0.0.0"
    server_port: int = 8001
    
    # Path settings
    voices_dir: str = "voices"
    embeddings_dir: str = "embeddings"
    audio_cache_dir: str = "audio_cache"

    # Cache settings
    cache_limit: int = 10
    cache_cleanup_interval: int = 300  # seconds between cache cleanup runs

    # Audio stream settings
    queue_size: int = 1024
    queue_timeout: float = 20.0
    eof_timeout: float = 1.0
    chunk_size: int = 32 * 1024
    default_sample_rate: int = 24000

    # TTS default settings
    temperature: float = 0.7
    lsd_decode_steps: int = 2
    top_p: float = 0.95
    repetition_penalty: float = 1.1
    model_tier: str = "tts-1"
    
    # Security settings
    allowed_origins: List[str] = ["http://localhost:*", "http://127.0.0.1:*"]
    rate_limit_requests: int = 30  # requests per minute
    rate_limit_window: int = 60  # seconds
    enable_high_priority: bool = True  # Windows high priority (can disable for shared systems)
    model_load_timeout: int = 300  # seconds to wait for model download
    
    # Input validation
    max_input_length: int = 4096
    min_input_length: int = 1


settings = Settings()

# Ensure directories exist
os.makedirs(settings.voices_dir, exist_ok=True)
os.makedirs(settings.embeddings_dir, exist_ok=True)
os.makedirs(settings.audio_cache_dir, exist_ok=True)

if __name__ == "__main__":
    print("Default Settings:")
    print(settings.model_dump_json(indent=2))
    print("To override, create a config.ini or .env file.")
