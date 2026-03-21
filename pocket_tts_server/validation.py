"""Input validation utilities."""
import os
import re
import subprocess
from pathlib import Path

from .config import settings


def is_valid_voice_name(voice: str) -> bool:
    if not voice or len(voice) > 255:
        return False
    if any(c in voice for c in ['/', '\\', '..']):
        return False
    if not re.match(r'^[a-zA-Z0-9_-]+$', voice):
        return False
    return True


def sanitize_text_input(text: str) -> str:
    if not text:
        raise ValueError("Text input cannot be empty")
    text = text.strip()
    if not text:
        raise ValueError("Text input cannot be empty after trimming")
    if len(text) > settings.max_input_length:
        raise ValueError(f"Text exceeds maximum length of {settings.max_input_length} characters")
    if len(text) < settings.min_input_length:
        raise ValueError(f"Text must be at least {settings.min_input_length} characters")
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    return text


def check_ffmpeg() -> bool:
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


_ffmpeg_available = check_ffmpeg()
