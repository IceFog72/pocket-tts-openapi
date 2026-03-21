"""Constants and mappings for the TTS server."""
import sys
from typing import Dict, List, Tuple


class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


VOICE_MAPPING: Dict[str, str] = {
    "alloy": "alba",
    "echo": "jean",
    "fable": "fantine",
    "onyx": "cosette",
    "nova": "eponine",
    "shimmer": "azelma",
}

DEFAULT_VOICES: Dict[str, List[str]] = {
    "openai_aliases": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
    "pocket_tts": ["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"]
}

FFMPEG_FORMATS: Dict[str, Tuple[str, str]] = {
    "mp3": ("mp3", "mp3_mf" if sys.platform == "win32" else "libmp3lame"),
    "opus": ("ogg", "opus"),
    "aac": ("adts", "aac"),
    "flac": ("flac", "flac"),
}

MEDIA_TYPES: Dict[str, str] = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "aac": "audio/aac",
    "opus": "audio/opus",
    "flac": "audio/flac",
    "pcm": "audio/pcm",
}

CACHE_EXTENSIONS = tuple("." + ext for ext in list(FFMPEG_FORMATS.keys()) + ["wav", "pcm"])
