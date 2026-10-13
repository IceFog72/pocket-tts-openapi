"""Pydantic models for API requests and responses."""
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .config import settings
from .validation import sanitize_text_input, validate_model_name


class SpeechRequest(BaseModel):
    model: str = Field(default="english-cpu", description="TTS model to use (format: language-device)")
    input: str = Field(..., min_length=1, max_length=settings.max_input_length, description="Text to generate")
    voice: str = Field(default="alloy", description="Voice identifier (predefined or custom)")
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm", "webm"] = Field(default="wav")
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    temperature: float = Field(default_factory=lambda: settings.temperature, ge=0.0, le=2.0)
    top_p: float = Field(default_factory=lambda: settings.top_p, ge=0.1, le=1.0, description="Deprecated compatibility field; ignored by Pocket TTS 3.3", deprecated=True)
    repetition_penalty: float = Field(default_factory=lambda: settings.repetition_penalty, ge=1.0, le=2.0, description="Deprecated compatibility field; ignored by Pocket TTS 3.3", deprecated=True)
    lsd_decode_steps: int = Field(default_factory=lambda: settings.lsd_decode_steps, ge=1, le=50)
    stream: bool = Field(default=True, description="OpenAI API compatibility parameter (streaming is always enabled in this server)")

    @field_validator("model", mode="before")
    @classmethod
    def validate_model(cls, v: str) -> str:
        if v is None or v == "":
            return settings.model_tier
        return v

    @field_validator("model")
    @classmethod
    def check_model(cls, v: str) -> str:
        return validate_model_name(v)

    @field_validator("input")
    @classmethod
    def check_text(cls, v: str) -> str:
        return sanitize_text_input(v)

    @field_validator("voice")
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
    truncate: bool = Field(default=False, description="Truncate audio to 30 seconds")
    temperature: float = Field(default_factory=lambda: settings.temperature, ge=0.0, le=2.0)
    top_p: float = Field(default_factory=lambda: settings.top_p, ge=0.1, le=1.0, description="Deprecated compatibility field; ignored by Pocket TTS 3.3", deprecated=True)
    repetition_penalty: float = Field(default_factory=lambda: settings.repetition_penalty, ge=1.0, le=2.0, description="Deprecated compatibility field; ignored by Pocket TTS 3.3", deprecated=True)
    lsd_decode_steps: int = Field(default_factory=lambda: settings.lsd_decode_steps, ge=1, le=50)
