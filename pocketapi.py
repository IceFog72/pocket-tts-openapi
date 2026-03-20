import asyncio
import hashlib
import io
import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Literal, Optional, Set, Tuple

import numpy as np
import safetensors.torch
import soundfile as sf
import torch
import uvicorn
from anyio import Path as AnyioPath, open_file
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pocket_tts import TTSModel
from pocket_tts.data.audio import stream_audio_chunks
from pocket_tts.data.audio_utils import convert_audio
from pocket_tts.modules.stateful_module import init_states
from pydantic import BaseModel, Field, field_validator
from queue import Empty, Full, Queue

# Windows-specific imports
if os.name == 'nt':
    import ctypes
    from ctypes import wintypes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from config import settings

# ============================================================================
# HUGGINGFACE AUTH & VOICE CLONING SETUP
# ============================================================================

def check_hf_auth() -> bool:
    """Check if user is authenticated with HuggingFace."""
    try:
        from huggingface_hub import whoami
        whoami()
        return True
    except Exception:
        return False


def setup_hf_auth() -> bool:
    """Interactive HuggingFace auth setup. Returns True if successful."""
    try:
        from huggingface_hub import login
        
        # Check if already authenticated
        try:
            from huggingface_hub import whoami
            whoami()
            logger.info("HuggingFace: Already authenticated")
            return True
        except Exception:
            pass
        
        # Check for token in environment
        env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if env_token:
            logger.info("HuggingFace: Using token from environment")
            login(token=env_token)
            return True
        
        # Check for .env file in project root
        env_file = Path(".") / ".env"
        if env_file.exists():
            try:
                with open(env_file) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("HF_TOKEN="):
                            token = line.split("=", 1)[1].strip()
                            if token and token.startswith("hf_"):
                                logger.info("HuggingFace: Using token from .env file")
                                login(token=token)
                                return True
            except Exception:
                pass
        
        # Check for saved token file
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.exists():
            logger.info("HuggingFace: Found saved token")
            return True
        
        # No token found - provide instructions
        logger.info(f"\n{'='*60}")
        logger.info(f"{Colors.YELLOW}{Colors.BOLD}VOICE CLONING SETUP{Colors.RESET}")
        logger.info(f"{'='*60}")
        logger.info("")
        logger.info(f"{Colors.CYAN}To enable voice cloning, you need a HuggingFace token:{Colors.RESET}")
        logger.info("")
        logger.info(f"  1. {Colors.GREEN}Get token at:{Colors.RESET}")
        logger.info(f"     https://huggingface.co/settings/tokens")
        logger.info("")
        logger.info(f"  2. {Colors.GREEN}Accept model license:{Colors.RESET}")
        logger.info(f"     https://huggingface.co/kyutai/pocket-tts")
        logger.info("")
        logger.info(f"  3. {Colors.GREEN}Set token in .env file:{Colors.RESET}")
        logger.info(f"     echo 'HF_TOKEN=hf_xxxxxxxxxxxxx' > .env")
        logger.info("")
        logger.info(f"{Colors.YELLOW}Continuing with basic model (no voice cloning)...{Colors.RESET}")
        logger.info(f"{'='*60}\n")
        
        return False
        
    except ImportError:
        logger.warning("huggingface_hub not installed, skipping auth")
        return False
    except Exception as e:
        logger.warning(f"HuggingFace auth setup failed: {e}")
        return False


def has_voice_cloning() -> bool:
    """Check if loaded model supports voice cloning."""
    model = model_manager.model
    if model is None:
        return False
    return getattr(model, 'has_voice_cloning', False)

# Silence chatty library logs
logging.getLogger("pocket_tts").setLevel(logging.WARNING)
logging.getLogger("pocket_tts.models.tts_model").setLevel(logging.WARNING)
logging.getLogger("pocket_tts.utils.utils").setLevel(logging.WARNING)

# ============================================================================
# CONSTANTS
# ============================================================================

# ANSI color codes
class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

# OpenAI voice name mapping
VOICE_MAPPING: Dict[str, str] = {
    "alloy": "alba",
    "echo": "jean",
    "fable": "fantine",
    "onyx": "cosette",
    "nova": "eponine",
    "shimmer": "azelma",
}

# Default voices
DEFAULT_VOICES: Dict[str, List[str]] = {
    "openai_aliases": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
    "pocket_tts": ["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"]
}

# FFmpeg format mappings
FFMPEG_FORMATS: Dict[str, Tuple[str, str]] = {
    "mp3": ("mp3", "mp3_mf" if sys.platform == "win32" else "libmp3lame"),
    "opus": ("ogg", "opus"),
    "aac": ("adts", "aac"),
    "flac": ("flac", "flac"),
}

# Media type mappings
MEDIA_TYPES: Dict[str, str] = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "aac": "audio/aac",
    "opus": "audio/opus",
    "flac": "audio/flac",
    "pcm": "audio/pcm",
}

# Valid file extensions for cache
CACHE_EXTENSIONS = tuple(list(FFMPEG_FORMATS.keys()) + ["wav", "pcm"])

# ============================================================================
# PYDANTIC MODELS (MUST BE DEFINED BEFORE ENDPOINTS)
# ============================================================================

class SpeechRequest(BaseModel):
    model: Literal["tts-1", "tts-1-hd", "tts-1-cuda", "tts-1-hd-cuda"] = Field("tts-1", description="TTS model to use")
    input: str = Field(
        ..., min_length=1, max_length=4096, description="Text to generate"
    )
    voice: str = Field("alloy", description="Voice identifier (predefined or custom)")
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field("wav")
    speed: Optional[float] = Field(1.0, ge=0.25, le=4.0)
    temperature: float = Field(default_factory=lambda: settings.temperature, ge=0.0, le=2.0)
    top_p: float = Field(default_factory=lambda: settings.top_p, ge=0.1, le=1.0, description="Nucleus sampling")
    repetition_penalty: float = Field(default_factory=lambda: settings.repetition_penalty, ge=1.0, le=2.0)
    lsd_decode_steps: int = Field(default_factory=lambda: settings.lsd_decode_steps, ge=1, le=50)
    stream: bool = Field(False, description="Presence of this flag is for compatibility, streaming is always enabled")

    @field_validator("model", mode="before")
    @classmethod
    def validate_model(cls, v: str) -> str:
        if not v:
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


# ============================================================================
# THREAD-SAFE MODEL MANAGER (Fix #3: Replace global mutable state)
# ============================================================================

@dataclass
class ModelManager:
    """Thread-safe manager for the TTS model state."""
    _model: Optional[TTSModel] = field(default=None, repr=False)
    _device: Optional[str] = None
    _sample_rate: Optional[int] = None
    _lock: threading.RLock = field(default_factory=threading.RLock)
    _loading: bool = False
    _load_event: threading.Event = field(default_factory=threading.Event)
    
    @property
    def model(self) -> Optional[TTSModel]:
        with self._lock:
            return self._model
    
    @property
    def device(self) -> Optional[str]:
        with self._lock:
            return self._device
    
    @property
    def sample_rate(self) -> int:
        with self._lock:
            return self._sample_rate or settings.default_sample_rate
    
    @property
    def is_loaded(self) -> bool:
        with self._lock:
            return self._model is not None
    
    def acquire_lock(self):
        """Acquire the model lock for exclusive access during generation."""
        self._lock.acquire()
        
    def release_lock(self):
        """Release the model lock."""
        self._lock.release()
    
    def load(self, timeout: int = settings.model_load_timeout) -> None:
        """Load the TTS model with timeout protection."""
        with self._lock:
            if self._model is not None:
                return
            
            if self._loading:
                self._lock.release()
                try:
                    if not self._load_event.wait(timeout=timeout):
                        raise TimeoutError("Model loading timed out")
                finally:
                    self._lock.acquire()
                return
            
            self._loading = True
            self._load_event.clear()
        
        try:
            logger.info(f"Loading TTS model (timeout: {timeout}s)...")
            
            load_result: Dict[str, Any] = {"model": None, "error": None}
            
            def _do_load():
                try:
                    load_result["model"] = TTSModel.load_model()
                except Exception as e:
                    load_result["error"] = e
            
            load_thread = threading.Thread(target=_do_load, daemon=True)
            load_thread.start()
            load_thread.join(timeout=timeout)
            
            if load_thread.is_alive():
                raise TimeoutError(f"Model loading exceeded {timeout}s timeout")
            
            if load_result["error"]:
                raise load_result["error"]
            
            with self._lock:
                self._model = load_result["model"]
                
                if not hasattr(TTSModel, "_slice_kv_cache"):
                    logger.info("Patching TTSModel with missing _slice_kv_cache method")
                    TTSModel._slice_kv_cache = _slice_kv_cache
                
                self._device = self._model.device
                self._sample_rate = getattr(self._model, "sample_rate", settings.default_sample_rate)
                self._loading = False
                self._load_event.set()
            
            logger.info(f"Pocket TTS loaded | Device: {self._device} | Sample Rate: {self._sample_rate}")
            
        except Exception as e:
            with self._lock:
                self._loading = False
                self._load_event.set()
            logger.error(f"Failed to load TTS model: {e}")
            raise
    
    def move_to_device(self, target_device: str) -> None:
        """Move model to specified device (cpu/cuda)."""
        with self._lock:
            if self._model is None:
                return
            if self._device != target_device:
                logger.info(f"Moving model from {self._device} to {target_device}")
                self._model.to(target_device)
                self._device = target_device
    
    def shutdown(self) -> None:
        """Clean shutdown of the model manager."""
        with self._lock:
            if self._model is not None:
                logger.info("Unloading TTS model...")
                del self._model
                self._model = None
                self._device = None


model_manager = ModelManager()


# ============================================================================
# RATE LIMITER (Fix #4: Add rate limiting middleware)
# ============================================================================

@dataclass
class RateLimiter:
    """Token bucket rate limiter per client IP."""
    requests_per_window: int = settings.rate_limit_requests
    window_seconds: int = settings.rate_limit_window
    _requests: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def is_allowed(self, client_ip: str) -> Tuple[bool, int]:
        """Check if request is allowed. Returns (is_allowed, retry_after_seconds)."""
        now = time.time()
        
        with self._lock:
            cutoff = now - self.window_seconds
            self._requests[client_ip] = [
                t for t in self._requests[client_ip] if t > cutoff
            ]
            
            if len(self._requests[client_ip]) >= self.requests_per_window:
                oldest = min(self._requests[client_ip])
                retry_after = int(oldest + self.window_seconds - now) + 1
                return False, max(retry_after, 1)
            
            self._requests[client_ip].append(now)
            return True, 0
    
    def cleanup(self) -> None:
        """Remove expired entries to prevent memory leak."""
        now = time.time()
        cutoff = now - self.window_seconds
        
        with self._lock:
            keys_to_remove = []
            for ip, timestamps in self._requests.items():
                self._requests[ip] = [t for t in timestamps if t > cutoff]
                if not self._requests[ip]:
                    keys_to_remove.append(ip)
            for ip in keys_to_remove:
                del self._requests[ip]


rate_limiter = RateLimiter()


# ============================================================================
# INPUT VALIDATION (Fix #12, #13: Strengthen path traversal & text sanitization)
# ============================================================================

def is_valid_voice_name(voice: str) -> bool:
    """Validate voice name to prevent path traversal attacks."""
    if not voice or len(voice) > 255:
        return False
    
    if any(c in voice for c in ['/', '\\', '..']):
        return False
    
    if not re.match(r'^[a-zA-Z0-9_-]+$', voice):
        return False
    
    return True


def sanitize_text_input(text: str) -> str:
    """Sanitize and validate text input."""
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


# ============================================================================
# VOICE LOADING
# ============================================================================

def load_custom_voices() -> Set[str]:
    """Scan voices and embeddings directories and update mapping."""
    custom_voices: Set[str] = set()
    tts_model = model_manager.model
    
    embeddings_path = Path(settings.embeddings_dir)
    if embeddings_path.exists():
        for f in embeddings_path.iterdir():
            if f.suffix.lower() == ".safetensors":
                voice_name = f.stem
                full_path = str(f.resolve())
                VOICE_MAPPING[voice_name] = full_path
                custom_voices.add(voice_name)
    
    voices_path = Path(settings.voices_dir)
    if voices_path.exists():
        for f in voices_path.iterdir():
            if f.suffix.lower() == ".wav":
                voice_name = f.stem
                wav_path = str(f)
                st_path = embeddings_path / f"{voice_name}.safetensors"
                
                if voice_name not in custom_voices:
                    if tts_model is not None:
                        try:
                            logger.info(f"Exporting '{voice_name}' to embeddings/ for faster loading...")
                            audio, sr = sf.read(wav_path)
                            audio_pt = torch.from_numpy(audio).float()
                            if len(audio_pt.shape) == 1:
                                audio_pt = audio_pt.unsqueeze(0)
                            audio_resampled = convert_audio(audio_pt, sr, tts_model.config.mimi.sample_rate, 1)
                            
                            with torch.no_grad():
                                prompt = tts_model._encode_audio(audio_resampled.unsqueeze(0).to(tts_model.device))
                            
                            safetensors.torch.save_file({"audio_prompt": prompt.cpu()}, str(st_path))
                            logger.info(f"Exported '{voice_name}' to {st_path}")
                            
                            VOICE_MAPPING[voice_name] = str(st_path.resolve())
                            custom_voices.add(voice_name)
                        except Exception as e:
                            logger.warning(f"Failed to auto-export voice '{voice_name}': {e}")
                            VOICE_MAPPING[voice_name] = str(f.resolve())
                            custom_voices.add(voice_name)
                    else:
                        VOICE_MAPPING[voice_name] = str(f.resolve())
                        custom_voices.add(voice_name)
    
    logger.info(f"{Colors.CYAN}{Colors.BOLD}Default voices available:{Colors.RESET}")
    logger.info(f"{Colors.CYAN}   OpenAI aliases: {', '.join(DEFAULT_VOICES['openai_aliases'])}{Colors.RESET}")
    logger.info(f"{Colors.CYAN}   Pocket TTS: {', '.join(DEFAULT_VOICES['pocket_tts'])}{Colors.RESET}")
    
    if custom_voices:
        logger.info(f"{Colors.GREEN}{Colors.BOLD}Custom voices loaded: {Colors.RESET}{Colors.GREEN}{', '.join(sorted(custom_voices))}{Colors.RESET}")
    else:
        logger.info(f"{Colors.YELLOW}No custom voices found in 'voices/' directory.{Colors.RESET}")
    
    return custom_voices


# ============================================================================
# CACHE MANAGER (Fix #1: SHA-256, Fix #10: Periodic cleanup)
# ============================================================================

class CacheManager:
    """Manages audio caching with SHA-256 hashing and periodic cleanup."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._last_cleanup: float = 0
    
    def generate_cache_key(self, text: str, voice_name: str, format: str, 
                          speed: float, temperature: float, lsd_decode_steps: int,
                          top_p: float, repetition_penalty: float, model_tier: str) -> str:
        """Generate a SHA-256 cache key."""
        cache_key = f"{text}|{voice_name}|{format}|{speed}|{temperature}|{lsd_decode_steps}|{top_p}|{repetition_penalty}|{model_tier}"
        return hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
    
    def get_cache_path(self, cache_hash: str, format: str) -> Tuple[str, str]:
        """Get cache file paths."""
        cache_filename = f"{cache_hash}.{format}"
        cache_path = os.path.join(settings.audio_cache_dir, cache_filename)
        meta_path = os.path.join(settings.audio_cache_dir, f"{cache_hash}.json")
        return cache_path, meta_path
    
    def check_cache(self, cache_path: str) -> bool:
        """Check if cache file exists."""
        return os.path.exists(cache_path)
    
    def should_cleanup(self) -> bool:
        """Check if cleanup should run based on interval."""
        now = time.time()
        if now - self._last_cleanup > settings.cache_cleanup_interval:
            return True
        return False
    
    def cleanup(self, force: bool = False) -> int:
        """Remove oldest audio files if cache exceeds limit."""
        def _do_cleanup() -> int:
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
                        logger.debug(f"Cache cleanup: Removed {os.path.basename(audio_path)}")
                        
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
        
        with self._lock:
            self._last_cleanup = time.time()
        
        return _do_cleanup()


cache_manager = CacheManager()


# ============================================================================
# WINDOWS PRIORITY (Fix #11: Make optional via config)
# ============================================================================

def set_high_priority() -> None:
    """Set the current process to High Priority on Windows if enabled."""
    if os.name != 'nt' or not settings.enable_high_priority:
        return
    
    try:
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.SetPriorityClass.restype = wintypes.BOOL
        
        handle = kernel32.GetCurrentProcess()
        
        if kernel32.SetPriorityClass(handle, 0x00000080):
            logger.info("Process priority set to HIGH")
        else:
            err = ctypes.get_last_error()
            logger.warning(f"Failed to set priority to HIGH (Error: {err}). Trying ABOVE_NORMAL...")
            if kernel32.SetPriorityClass(handle, 0x00008000):
                logger.info("Process priority set to ABOVE_NORMAL")
            else:
                err = ctypes.get_last_error()
                logger.warning(f"Failed to set any elevated priority (Error: {err})")
    except Exception as e:
        logger.warning(f"Could not set process priority: {e}")


# ============================================================================
# MONKEY PATCH
# ============================================================================

def _slice_kv_cache(self, model_state: dict, sequence_length: int) -> None:
    """Memory optimization: Slice KV cache to actual prompt length."""
    for module_name, module_state in model_state.items():
        if "cache" in module_state:
            cache = module_state["cache"]
            if cache.shape[2] > sequence_length:
                module_state["cache"] = cache[:, :, :sequence_length, :, :].contiguous()
            elif cache.shape[3] > sequence_length:
                module_state["cache"] = cache[:, :, :, :sequence_length, :].contiguous()


# ============================================================================
# AUDIO PRODUCER
# ============================================================================

class FileLikeQueueWriter:
    """File-like adapter that writes bytes to a queue with backpressure."""

    def __init__(self, queue: Queue, timeout: float = 30.0):
        self.queue = queue
        self.timeout = timeout
        self._pos = 0

    def tell(self):
        return self._pos

    def seek(self, offset, whence=0):
        # Dummy seek: we can't actually seek in a queue, but we pretend
        pass

    def write(self, data: bytes) -> int:
        if not data:
            return 0
        self._pos += len(data)
        start_time = time.time()
        while True:
            if getattr(self.queue, 'abort', False):
                raise IOError("Generation aborted by client disconnect")
            try:
                self.queue.put(data, timeout=1.0)
                return len(data)
            except Full:
                if time.time() - start_time > self.timeout:
                    logger.warning("Queue timeout: Client disconnected or too slow.")
                    raise IOError("Queue full - aborting generation")

    def flush(self) -> None:
        pass

    def close(self) -> None:
        try:
            self.queue.put(None, timeout=settings.eof_timeout)
        except (Full, Exception):
            try:
                self.queue.put_nowait(None)
            except Full:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.close()
        except Exception:
            logger.exception("Error closing queue writer")
        return False


def _start_audio_producer(
    queue: Queue,
    voice_name: str,
    text: str,
    temperature: float = settings.temperature,
    lsd_decode_steps: int = settings.lsd_decode_steps,
    top_p: float = settings.top_p,
    repetition_penalty: float = settings.repetition_penalty,
    model_tier: str = settings.model_tier,
) -> threading.Thread:
    """Start background thread that generates audio and writes to queue."""

    def producer() -> None:
        logger.debug(f"Starting audio generation for voice: {voice_name} (model={model_tier})")
        try:
            model_manager.acquire_lock()
            try:
                tts_model = model_manager.model
                if tts_model is None:
                    raise RuntimeError("TTS model not loaded")
                
                tts_model.temp = temperature
                tts_model.top_p = top_p
                tts_model.repetition_penalty = repetition_penalty
                
                if "hd" in model_tier:
                    tts_model.lsd_decode_steps = max(lsd_decode_steps, 16)
                else:
                    tts_model.lsd_decode_steps = lsd_decode_steps
                
                current_device = model_manager.device
                if "cuda" in model_tier and torch.cuda.is_available():
                    if current_device != "cuda":
                        logger.debug(f"Moving model to CUDA for generation")
                        tts_model.to("cuda")
                        model_manager._device = "cuda"
                else:
                    if current_device != "cpu":
                        logger.debug(f"Moving model back to CPU")
                        tts_model.to("cpu")
                        model_manager._device = "cpu"
                
                is_safe_file = os.path.isabs(voice_name)
                if os.path.exists(voice_name) and os.path.isfile(voice_name) and is_safe_file:
                    file_ext = os.path.splitext(voice_name)[1].lower()
                    if file_ext == ".safetensors":
                        logger.debug(f"Loading pre-exported voice embedding: {voice_name}")
                        prompt = safetensors.torch.load_file(voice_name)["audio_prompt"]
                        prompt = prompt.to(tts_model.device)
                        
                        model_state = init_states(tts_model.flow_lm, batch_size=1, sequence_length=1000)
                        with torch.no_grad():
                            tts_model._run_flow_lm_and_increment_step(model_state=model_state, audio_conditioning=prompt)
                        
                        num_audio_frames = prompt.shape[1]
                        tts_model._slice_kv_cache(model_state, num_audio_frames)
                    else:
                        logger.debug(f"Cloning voice from file: {voice_name}")
                        model_state = tts_model.get_state_for_audio_prompt(voice_name)
                else:
                    model_state = tts_model.get_state_for_audio_prompt(voice_name)
                
                # Split text into manageable sentences/clauses (threshold ~500 chars)
                import re
                parts = re.split(r'(?<=[.!?])\s+', text.strip())
                
                def combined_chunks():
                    for part in parts:
                        if not part.strip(): continue
                        
                        # Sub-split long sentences by comma/semicolon for safety
                        if len(part) > 500:
                            subtasks = re.split(r'(?<=[,;])\s+', part)
                        else:
                            subtasks = [part]
                            
                        for subtask in subtasks:
                            if not subtask.strip(): continue
                            for chunk in tts_model.generate_audio_stream(
                                model_state=model_state, text_to_generate=subtask
                            ):
                                yield chunk

                with FileLikeQueueWriter(queue) as writer:
                    stream_audio_chunks(
                        writer, combined_chunks(), model_manager.sample_rate
                    )
            finally:
                model_manager.release_lock()
        except Exception as e:
            logger.error(f"Audio generation failed for voice {voice_name}: {e}")
        finally:
            try:
                queue.put(None, timeout=settings.eof_timeout)
            except Full:
                pass

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()
    return thread


# ============================================================================
# AUDIO STREAMING
# ============================================================================

async def _stream_queue_chunks(queue: Queue) -> AsyncIterator[bytes]:
    """Async generator that yields bytes from queue until EOF."""
    while True:
        chunk = await asyncio.to_thread(queue.get)
        if chunk is None:
            logger.debug("Received EOF from producer")
            break
        yield chunk


def _start_ffmpeg_process(format: str, speed: float = 1.0) -> Tuple[subprocess.Popen, int, int]:
    """Start ffmpeg process with OS pipe for stdin."""
    out_fmt, codec = FFMPEG_FORMATS.get(format, ("wav", "pcm_s16le"))
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "wav",
        "-i",
        "pipe:0",
    ]
    
    if speed != 1.0:
        cmd.extend(["-filter:a", f"atempo={speed}"])
    
    if codec == "opus":
        cmd.extend(["-strict", "-2"])
    
    if format == "mp3":
        cmd.extend(["-ar", "44100"])
        cmd.extend(["-q:a", "0"])
    elif format in ("aac", "opus"):
        cmd.extend(["-b:a", "192k"])
    
    cmd.extend([
        "-f",
        out_fmt,
        "-codec:a",
        codec,
        "pipe:1",
    ])
    
    r_fd, w_fd = os.pipe()
    r_file = os.fdopen(r_fd, "rb")
    proc = subprocess.Popen(cmd, stdin=r_file, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    r_file.close()
    return proc, w_fd, r_fd


def _start_pipe_writer(queue: Queue, write_fd: int) -> threading.Thread:
    """Start thread that writes queue chunks to OS pipe."""

    def pipe_writer() -> None:
        try:
            with os.fdopen(write_fd, "wb") as pipe:
                while True:
                    data = queue.get()
                    if data is None:
                        break
                    try:
                        pipe.write(data)
                    except (BrokenPipeError, OSError):
                        break
                pipe.flush()
        except OSError:
            pass
        except Exception:
            logger.exception("Error in pipe writer")

    thread = threading.Thread(target=pipe_writer, daemon=True)
    thread.start()
    return thread


async def _generate_audio_core(
    text: str,
    voice_name: str,
    speed: float,
    format: str,
    chunk_size: int,
    temperature: float = settings.temperature,
    lsd_decode_steps: int = settings.lsd_decode_steps,
    top_p: float = settings.top_p,
    repetition_penalty: float = settings.repetition_penalty,
    model_tier: str = settings.model_tier,
) -> AsyncIterator[bytes]:
    """Internal generator for the actual TTS + FFmpeg logic."""
    queue = Queue(maxsize=settings.queue_size)
    queue.abort = False
    producer_thread = _start_audio_producer(
        queue, voice_name, text, temperature, lsd_decode_steps, top_p, repetition_penalty, model_tier
    )
    
    proc: Optional[subprocess.Popen] = None
    writer_thread: Optional[threading.Thread] = None
    
    try:
        if format in ("wav", "pcm") and speed == 1.0:
            async for chunk in _stream_queue_chunks(queue):
                yield chunk
            producer_thread.join(timeout=30)
            return
        
        if format in FFMPEG_FORMATS or (format in ("wav", "pcm") and speed != 1.0):
            proc, write_fd, _ = _start_ffmpeg_process(format, speed)
            writer_thread = _start_pipe_writer(queue, write_fd)
            
            try:
                while True:
                    chunk = await asyncio.to_thread(proc.stdout.read, chunk_size)
                    if not chunk:
                        break
                    yield chunk
            finally:
                try:
                    if proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
                except Exception as e:
                    logger.warning(f"Error cleaning up FFmpeg process: {e}")
                finally:
                    try:
                        proc.stdout.close()
                    except Exception:
                        pass
                    if proc.stderr:
                        try:
                            proc.stderr.close()
                        except Exception:
                            pass
                
                await asyncio.to_thread(producer_thread.join)
                if writer_thread:
                    await asyncio.to_thread(writer_thread.join)
            return
        
        async for chunk in _stream_queue_chunks(queue):
            yield chunk
        producer_thread.join(timeout=30)

    except Exception as e:
        logger.exception(f"Error streaming audio format {format}: {e}")
        raise
    finally:
        queue.abort = True
        while not queue.empty():
            try: queue.get_nowait()
            except Exception: break


# ============================================================================
# MAIN AUDIO GENERATION WITH CACHING
# ============================================================================

async def generate_audio(
    text: str,
    voice: str = "alloy",
    speed: float = 1.0,
    format: str = "wav",
    chunk_size: int = settings.chunk_size,
    temperature: float = settings.temperature,
    lsd_decode_steps: int = settings.lsd_decode_steps,
    top_p: float = settings.top_p,
    repetition_penalty: float = settings.repetition_penalty,
    model_tier: str = settings.model_tier,
    stream: bool = False,
    background_tasks: Optional[BackgroundTasks] = None,
) -> AsyncIterator[bytes]:
    """Generate and stream audio, with filesystem caching."""
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="TTS model not loaded")
    
    try:
        text = sanitize_text_input(text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    if not is_valid_voice_name(voice) and voice not in VOICE_MAPPING:
        logger.warning(f"Invalid voice name rejected: '{voice}'")
        raise HTTPException(status_code=400, detail="Invalid voice name")
    
    # Case-insensitive lookup for OpenAI aliases or predefined voices
    v_lower = voice.lower()
    if v_lower in VOICE_MAPPING:
        voice_name = VOICE_MAPPING[v_lower]
    else:
        # Fallback to exact match (for paths)
        voice_name = VOICE_MAPPING.get(voice, voice)
    
    cache_hash = cache_manager.generate_cache_key(
        text, voice_name, format, speed, temperature, 
        lsd_decode_steps, top_p, repetition_penalty, model_tier
    )
    cache_path, meta_path = cache_manager.get_cache_path(cache_hash, format)
    
    if cache_manager.check_cache(cache_path):
        logger.debug(f"Cache HIT for {cache_hash[:16]}... ({format})")
        try:
            async with await open_file(cache_path, "rb") as f:
                while True:
                    chunk = await f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
            return
        except (IOError, OSError) as e:
            logger.warning(f"Failed to read cache file, regenerating: {e}")

    logger.debug(f"Cache MISS for {cache_hash[:16]}... ({format}) - Generating...")
    temp_path = f"{cache_path}.{uuid.uuid4().hex}.tmp"
    
    try:
        async with await open_file(temp_path, "wb") as cache_file:
            async for chunk in _generate_audio_core(
                text, voice_name, speed, format, chunk_size, 
                temperature, lsd_decode_steps, top_p, repetition_penalty, model_tier
            ):
                await cache_file.write(chunk)
                yield chunk
        
        if os.path.exists(temp_path):
            os.replace(temp_path, cache_path)
            
            metadata = {
                "text": text,
                "voice": voice_name,
                "speed": speed,
                "format": format,
                "hash": cache_hash,
                "model": model_tier,
                "created_at": time.time()
            }
            try:
                async with await open_file(meta_path, "w") as f:
                    await f.write(json.dumps(metadata, indent=2))
            except (IOError, OSError) as e:
                logger.warning(f"Failed to save metadata: {e}")
            
            if background_tasks and cache_manager.should_cleanup():
                background_tasks.add_task(cache_manager.cleanup)
        
    except (IOError, OSError, RuntimeError) as e:
        for path in [temp_path, meta_path, cache_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
        raise HTTPException(status_code=500, detail=f"Audio generation failed: {str(e)}")
    except (Exception, asyncio.CancelledError) as e:
        logger.exception(f"Unexpected error during audio generation (or client cancelled)")
        for path in [temp_path, meta_path, cache_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
        raise


# ============================================================================
# LIFESPAN & GRACEFUL SHUTDOWN (Fix #6)
# ============================================================================

shutdown_event = asyncio.Event()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load TTS model on startup, cleanup on shutdown."""
    logger.info("Starting TTS API server...")
    set_high_priority()
    
    # Setup HuggingFace auth for voice cloning
    await asyncio.to_thread(setup_hf_auth)
    
    await asyncio.to_thread(model_manager.load, settings.model_load_timeout)
    await asyncio.to_thread(load_custom_voices)
    
    # Show voice cloning status
    if has_voice_cloning():
        logger.info(f"{Colors.GREEN}{Colors.BOLD}Voice cloning: ENABLED{Colors.RESET}")
    else:
        logger.info(f"{Colors.YELLOW}Voice cloning: DISABLED (using preset voices only){Colors.RESET}")
    
    async def periodic_cleanup() -> None:
        while not shutdown_event.is_set():
            try:
                await asyncio.sleep(settings.cache_cleanup_interval)
                if not shutdown_event.is_set():
                    await asyncio.to_thread(cache_manager.cleanup)
                    await asyncio.to_thread(rate_limiter.cleanup)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
    
    cleanup_task = asyncio.create_task(periodic_cleanup())
    
    yield
    
    logger.info("Shutting down TTS server...")
    shutdown_event.set()
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    
    model_manager.shutdown()
    logger.info("TTS server shutdown complete")


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="OpenAI-Compatible TTS API (Cached)",
    description="OpenAI Audio Speech API compatible endpoint using Kyutai TTS with model caching",
    version="2.0.0",
    lifespan=lifespan,
)

# Fix #2: Restrict CORS to specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


# Fix #4: Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Apply rate limiting to all requests."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        client_ip = forwarded.split(',')[0].strip()
    else:
        client_ip = request.client.host if request.client else "unknown"
    
    is_allowed, retry_after = rate_limiter.is_allowed(client_ip)
    if not is_allowed:
        return StreamingResponse(
            content=iter([json.dumps({"error": "Rate limit exceeded", "retry_after": retry_after}).encode()]),
            status_code=429,
            media_type="application/json",
            headers={"Retry-After": str(retry_after)}
        )
    
    response = await call_next(request)
    return response


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/v1/voices")
async def get_voices():
    """Return all available voices (built-in + custom)."""
    voices = set(DEFAULT_VOICES["openai_aliases"] + DEFAULT_VOICES["pocket_tts"])
    voices.update(VOICE_MAPPING.keys())
    return {"voices": sorted(list(voices))}


@app.get("/v1/formats")
async def get_formats():
    """Return supported audio formats."""
    return {"formats": sorted(list(MEDIA_TYPES.keys()))}


@app.post("/v1/audio/speech")
async def text_to_speech(data: SpeechRequest, background_tasks: BackgroundTasks) -> StreamingResponse:
    """Generate speech audio from text with streaming response."""
    try:
        logger.info(f"TTS request: voice='{data.voice}', format='{data.response_format}', len={len(data.input)}")
        
        return StreamingResponse(
            generate_audio(
                text=data.input,
                voice=data.voice,
                speed=data.speed,
                format=data.response_format,
                temperature=data.temperature,
                lsd_decode_steps=data.lsd_decode_steps,
                top_p=data.top_p,
                repetition_penalty=data.repetition_penalty,
                model_tier=data.model,
                stream=data.stream,
                background_tasks=background_tasks,
            ),
            media_type=MEDIA_TYPES.get(data.response_format, "audio/wav"),
            headers={
                "Transfer-Encoding": "chunked",
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Internal Server Error in text_to_speech")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/audio/export-voice")
async def export_voice(request: ExportVoiceRequest):
    """Manually export a WAV voice to safetensors embedding."""
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="TTS model not loaded")
    
    voice_name = request.voice
    
    if not is_valid_voice_name(voice_name):
        logger.warning(f"Invalid voice name rejected in export: '{voice_name}'")
        raise HTTPException(status_code=400, detail="Invalid voice name")

    wav_path = os.path.join(settings.voices_dir, f"{voice_name}.wav")
    st_path = os.path.join(settings.embeddings_dir, f"{voice_name}.safetensors")
    
    if not os.path.exists(wav_path):
        if voice_name.lower().endswith(".wav"):
            wav_path = os.path.join(settings.voices_dir, voice_name)
            st_path = os.path.join(settings.embeddings_dir, f"{os.path.splitext(voice_name)[0]}.safetensors")
        
        if not os.path.exists(wav_path):
            raise HTTPException(status_code=404, detail=f"Voice '{voice_name}' (WAV) not found in {settings.voices_dir}")

    try:
        logger.info(f"Exporting '{voice_name}' to embeddings/...")
        audio, sr = sf.read(wav_path)
        audio_pt = torch.from_numpy(audio).float()
        if len(audio_pt.shape) == 1:
            audio_pt = audio_pt.unsqueeze(0)
            
        if request.truncate:
            max_samples = int(30 * sr)
            if audio_pt.shape[-1] > max_samples:
                audio_pt = audio_pt[..., :max_samples]
                logger.debug(f"Audio truncated to 30s for export")

        audio_resampled = convert_audio(audio_pt, sr, model_manager.sample_rate, 1)
        
        model_manager.acquire_lock()
        try:
            tts_model = model_manager.model
            if tts_model is None:
                raise RuntimeError("Model not loaded")
            
            with torch.no_grad():
                tts_model.temp = request.temperature
                tts_model.lsd_decode_steps = request.lsd_decode_steps
                prompt = tts_model._encode_audio(audio_resampled.unsqueeze(0).to(tts_model.device))
        finally:
            model_manager.release_lock()
        
        safetensors.torch.save_file({"audio_prompt": prompt.cpu()}, st_path)
        logger.info(f"Exported '{voice_name}' to {st_path}")
        
        await asyncio.to_thread(load_custom_voices)
        
        return {"status": "success", "message": f"Exported {voice_name} to safetensors", "path": st_path}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to export voice '{voice_name}'")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Simple healthcheck endpoint."""
    return {
        "status": "ok",
        "model_loaded": model_manager.is_loaded,
        "device": model_manager.device,
        "sample_rate": model_manager.sample_rate,
        "voice_cloning": has_voice_cloning(),
        "hf_authenticated": check_hf_auth(),
        "cache_files": len([f for f in os.listdir(settings.audio_cache_dir) 
                           if f.endswith(CACHE_EXTENSIONS)]) if os.path.exists(settings.audio_cache_dir) else 0,
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["default"]["fmt"] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_config["formatters"]["access"]["fmt"] = '%(asctime)s - %(client_addr)s - "%(request_line)s" %(status_code)s'
    
    try:
        # Support port override from environment (set by start.sh)
        override_port = os.environ.get("OVERRIDE_PORT")
        if override_port and override_port.isdigit():
            port = int(override_port)
        else:
            port = settings.server_port
        host = settings.server_host
        logger.info(f"Starting server with HTTP debug logging enabled")
        logger.info(f"Server binding to: http://{host}:{port}")
        logger.info(f"If you are using SillyTavern, set provider endpoint to: http://127.0.0.1:{port}/v1/audio/speech")
        uvicorn.run(app, host=host, port=port, log_config=log_config, access_log=True)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.exception("Failed to start server")
