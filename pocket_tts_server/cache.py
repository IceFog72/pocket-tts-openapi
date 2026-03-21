"""Audio cache manager with SHA-256 keys and periodic cleanup."""
import hashlib
import logging
import os
import threading
import time
from pathlib import Path
from typing import Tuple

from .config import settings
from .constants import CACHE_EXTENSIONS

logger = logging.getLogger(__name__)


class CacheManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._last_cleanup: float = 0

    def generate_cache_key(self, text: str, voice_name: str, format: str,
                           speed: float, temperature: float, lsd_decode_steps: int,
                           top_p: float, repetition_penalty: float, model_tier: str) -> str:
        cache_key = f"{text}|{voice_name}|{format}|{speed}|{temperature}|{lsd_decode_steps}|{top_p}|{repetition_penalty}|{model_tier}"
        return hashlib.sha256(cache_key.encode("utf-8")).hexdigest()

    def get_cache_path(self, cache_hash: str, format: str) -> Tuple[str, str]:
        cache_path = os.path.join(settings.audio_cache_dir, f"{cache_hash}.{format}")
        meta_path = os.path.join(settings.audio_cache_dir, f"{cache_hash}.json")
        return cache_path, meta_path

    def check_cache(self, cache_path: str) -> bool:
        return os.path.exists(cache_path)

    def should_cleanup(self) -> bool:
        return time.time() - self._last_cleanup > settings.cache_cleanup_interval

    def cleanup(self, force: bool = False) -> int:
        with self._lock:
            self._last_cleanup = time.time()

        try:
            audio_files = []
            cache_dir = Path(settings.audio_cache_dir)
            if not cache_dir.exists():
                return 0

            for f in cache_dir.iterdir():
                if f.is_file() and f.suffix in CACHE_EXTENSIONS:
                    audio_files.append((str(f), f.stat().st_mtime))

            if len(audio_files) <= settings.cache_limit:
                return 0

            audio_files.sort(key=lambda x: x[1])
            to_delete = audio_files[:len(audio_files) - settings.cache_limit]
            deleted = 0

            for audio_path, _ in to_delete:
                try:
                    os.remove(audio_path)
                    json_path = os.path.splitext(audio_path)[0] + ".json"
                    if os.path.exists(json_path):
                        os.remove(json_path)
                    deleted += 1
                except OSError:
                    pass
            return deleted
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")
            return 0


cache_manager = CacheManager()
