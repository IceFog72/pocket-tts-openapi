"""Input validation utilities."""
import os
import re
import subprocess
from pathlib import Path

from .config import settings

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
    if not voice or len(voice) > 255:
        return False
    if any(c in voice for c in ['/', '\\', '..']):
        return False
    if not re.match(r'^[a-zA-Z0-9_-]+$', voice):
        return False
    return True


def sanitize_text_input(text: str) -> str:
    """Validate + normalize text for TTS. Raises ValueError on invalid input."""
    if not text:
        raise ValueError("Text input cannot be empty")
    text = text.strip()
    if not text:
        raise ValueError("Text input cannot be empty after trimming")
    if len(text) > settings.max_input_length:
        raise ValueError(f"Text exceeds maximum length of {settings.max_input_length} characters")
    if len(text) < settings.min_input_length:
        raise ValueError(f"Text must be at least {settings.min_input_length} characters")
    text = normalize_text(text)
    return text


def check_ffmpeg() -> bool:
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


_ffmpeg_available = check_ffmpeg()
