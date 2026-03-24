"""Pydantic models for API requests and responses."""
import logging
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .config import settings

logger = logging.getLogger(__name__)


class SpeechRequest(BaseModel):
    model: Literal["tts-1", "tts-1-hd", "tts-1-cuda", "tts-1-hd-cuda"] = Field("tts-1", description="TTS model to use")
    input: str = Field(..., min_length=1, max_length=4096, description="Text to generate")
    voice: str = Field("alloy", description="Voice identifier (predefined or custom)")
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field("wav")
    speed: float = Field(1.0, ge=0.25, le=4.0)
    temperature: float = Field(default_factory=lambda: settings.temperature, ge=0.0, le=2.0)
    top_p: float = Field(default_factory=lambda: settings.top_p, ge=0.1, le=1.0, description="Nucleus sampling")
    repetition_penalty: float = Field(default_factory=lambda: settings.repetition_penalty, ge=1.0, le=2.0)
    lsd_decode_steps: int = Field(default_factory=lambda: settings.lsd_decode_steps, ge=1, le=50)
    stream: bool = Field(True, description="OpenAI API compatibility parameter (streaming is always enabled in this server)")

    @field_validator("model", mode="before")
    @classmethod
    def validate_model(cls, v: str) -> str:
        if not v:
            logger.debug("Empty model specified, falling back to settings.model_tier")
            return settings.model_tier
        return v

    @field_validator("voice", mode="before")
    @classmethod
    def validate_voice(cls, v: str) -> str:
        return v.strip() if v else v

    @field_validator("response_format", mode="before")
    @classmethod
    def validate_format(cls, v: str) -> str:
        if not v:
            return "wav"
        return v


class ExportVoiceRequest(BaseModel):
    voice: str = Field(..., description="Voice name (WAV file in voices/ directory)")
    truncate: bool = Field(False, description="Truncate audio to 30 seconds")
    temperature: float = Field(default_factory=lambda: settings.temperature, ge=0.0, le=2.0)
    top_p: float = Field(default_factory=lambda: settings.top_p, ge=0.1, le=1.0)
    repetition_penalty: float = Field(default_factory=lambda: settings.repetition_penalty, ge=1.0, le=2.0)
    lsd_decode_steps: int = Field(default_factory=lambda: settings.lsd_decode_steps, ge=1, le=50)
