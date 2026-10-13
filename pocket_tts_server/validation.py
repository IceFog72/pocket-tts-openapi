"""Input validation utilities."""
import os
import re
import subprocess
from pathlib import Path

from .config import settings
from .errors import InputError

# Unicode typographic characters → ASCII replacements
_UNICODE_REPLACEMENTS = {
    '\u2018': "'",   # '  left single quote
    '\u2019': "'",   # '  right single quote / apostrophe
    '\u201a': "'",   # ‚  single low-9 quote
    '\u201b': "'",   # ‛  single high-reversed-9 quote
    '\u201c': '"',   # "  left double quote
    '\u201d': '"',   # "  right double quote
    '\u201e': '"',   # „  double low-9 quote
    '\u201f': '"',   # ‟  double high-reversed-9 quote
    '\u2013': '-',   # –  en dash
    '\u2014': '-',   # —  em dash
    '\u2015': '-',   # ―  horizontal bar
    '\u2026': '...', # …  ellipsis
    '\u00ab': '"',   # «  left guillemet
    '\u00bb': '"',   # »  right guillemet
    '\u2032': "'",   # ′  prime
    '\u2033': '"',   # ″  double prime
}


def normalize_text(text: str) -> str:
    """Normalize Unicode typographic characters to ASCII and remove problematic symbols.
    Safe to call on any text — no validation, just cleaning."""
    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # Normalize Unicode typographic chars
    for old, new in _UNICODE_REPLACEMENTS.items():
        text = text.replace(old, new)
    # Remove backticks and double quotes entirely (model makes strange sounds)
    text = text.replace('`', '')
    text = text.replace('"', '')
    return text


def is_valid_voice_name(voice: str) -> bool:
    if not isinstance(voice, str) or not voice or len(voice) > 255:
        return False
    if any(c in voice for c in ['/', '\\', '..']):
        return False
    if not re.match(r'^[a-zA-Z0-9_-]+$', voice):
        return False
    return True


def sanitize_text_input(text: str) -> str:
    """Validate + normalize text for TTS. Raises ValueError on invalid input."""
    if not isinstance(text, str):
        raise ValueError("Text input must be a string")
    if not text:
        raise ValueError("Text input cannot be empty")
    text = text.strip()
    if not text:
        raise ValueError("Text input cannot be empty after trimming")
    if len(text) > settings.max_input_length:
        raise ValueError(f"Text exceeds maximum length of {settings.max_input_length} characters")
    if len(text) < settings.min_input_length:
        raise ValueError(f"Text must be at least {settings.min_input_length} characters")
    text = normalize_text(text).strip()
    if not text:
        raise ValueError("Text input contains no speakable text")
    return text


def check_ffmpeg() -> bool:
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


_ffmpeg_available = check_ffmpeg()


def validate_model_name(model: str) -> str:
    from pocket_tts.utils.config import CONFIGS_DIR
    model = model.strip()
    if model in ("tts-1", "tts-1-hd", "tts-1-cuda"):
        return model
    language, separator, device = model.rpartition("-")
    if not separator or device not in ("cpu", "gpu", "cuda") or not any(
        config.stem == language for config in CONFIGS_DIR.glob("*.yaml")
    ):
        raise ValueError("Unknown model. Use /v1/models to list available models")
    return model


def resolve_voice(voice: str) -> str:
    from .constants import DEFAULT_VOICES, VOICE_MAPPING
    from .voices import voice_lock
    if not isinstance(voice, str) or not voice.strip():
        raise InputError("Voice must be a nonempty string", param="voice")
    voice = voice.strip()
    builtins = {name.casefold(): name for name in DEFAULT_VOICES['pocket_tts']}
    with voice_lock:
        mapping = {name.casefold(): target for name, target in VOICE_MAPPING.items()}
    target = mapping.get(voice.casefold(), builtins.get(voice.casefold()))
    if target in builtins.values():
        return target
    if target is not None or os.path.isabs(voice):
        path = Path(target if target is not None else voice)
        if path.suffix.lower() not in (".wav", ".safetensors") or not path.is_file():
            raise InputError("Voice file is unavailable. Check /v1/voices and reload the server after adding voices",
                             code="voice_unavailable", param="voice", status_code=404)
        return str(path.resolve())
    raise InputError(f"Unknown voice '{voice[:80]}'. Use /v1/voices to list available voices",
                     code="unknown_voice", param="voice", status_code=404)
