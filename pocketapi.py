import asyncio
import hashlib
import io
import json
import logging
import os
import subprocess
import sys
import threading
import uuid

import numpy as np
import soundfile as sf
import torch
import uvicorn
import safetensors.torch
from anyio import open_file, Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from queue import Queue, Full
from typing import Literal, Optional, AsyncIterator

from pocket_tts import TTSModel
from pocket_tts.data.audio import stream_audio_chunks
from pocket_tts.data.audio_utils import convert_audio
from pocket_tts.modules.stateful_module import init_states

# Windows-specific imports
if os.name == 'nt':
    import ctypes
    from ctypes import wintypes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from config import settings

# Silence chatty library logs to keep progress bar clean
logging.getLogger("pocket_tts").setLevel(logging.WARNING)
logging.getLogger("pocket_tts.models.tts_model").setLevel(logging.WARNING)
logging.getLogger("pocket_tts.utils.utils").setLevel(logging.WARNING)

# Constants
# These are now loaded from config.py and config.ini
# QUEUE_SIZE = 1024
# QUEUE_TIMEOUT = 20.0
# EOF_TIMEOUT = 1.0
# CHUNK_SIZE = 32 * 1024
# DEFAULT_SAMPLE_RATE = 24000

MODEL_LOCK = threading.Lock()

# ANSI color codes for terminal output
class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

# map OpenAI voice names to pocket_tts voice names
VOICE_MAPPING = {
    "alloy": "alba",
    "echo": "jean",
    "fable": "fantine",
    "onyx": "cosette",
    "nova": "eponine",
    "shimmer": "azelma",
}

# Store default voices for later display
DEFAULT_VOICES = {
    "openai_aliases": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
    "pocket_tts": ["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"]
}

# VOICES_DIR = "voices"
# EMBEDDINGS_DIR = "embeddings"
# os.makedirs(VOICES_DIR, exist_ok=True)
# os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

def load_custom_voices():
    """Scan voices and embeddings directories and update mapping. Automatically export WAVs if needed."""
    
    custom_voices = set()
    
    # 1. Load existing safetensors from embeddings directory
    if os.path.exists(settings.embeddings_dir):
        for f in os.listdir(settings.embeddings_dir):
            if f.lower().endswith(".safetensors"):
                voice_name = os.path.splitext(f)[0]
                file_path = os.path.join(settings.embeddings_dir, f)
                full_path = os.path.abspath(file_path).replace("\\", "/")
                VOICE_MAPPING[voice_name] = full_path
                custom_voices.add(voice_name)
    
    # 2. Check WAV files in voices directory for conversion
    if os.path.exists(settings.voices_dir):
        for f in os.listdir(settings.voices_dir):
            if f.lower().endswith(".wav"):
                voice_name = os.path.splitext(f)[0]
                wav_path = os.path.join(settings.voices_dir, f)
                st_path = os.path.join(settings.embeddings_dir, voice_name + ".safetensors")
                
                # If safetensors doesn't exist in embeddings/, try to export it
                if voice_name not in custom_voices:
                    if tts_model:
                        try:
                            logger.info(f"✨ Exporting '{voice_name}' to embeddings/ for faster loading...")
                            # we need the model state to get the prompt
                            audio, sr = sf.read(wav_path)
                            audio_pt = torch.from_numpy(audio).float()
                            if len(audio_pt.shape) == 1:
                                audio_pt = audio_pt.unsqueeze(0)
                            audio_resampled = convert_audio(audio_pt, sr, tts_model.config.mimi.sample_rate, 1)
                            
                            with torch.no_grad():
                                prompt = tts_model._encode_audio(audio_resampled.unsqueeze(0).to(tts_model.device))
                            
                            safetensors.torch.save_file({"audio_prompt": prompt.cpu()}, st_path)
                            logger.info(f"✅ Exported '{voice_name}' to {st_path}")
                            
                            full_path = os.path.abspath(st_path).replace("\\", "/")
                            VOICE_MAPPING[voice_name] = full_path
                            custom_voices.add(voice_name)
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to auto-export voice '{voice_name}': {e}")
                            # Fallback to WAV if export failed
                            full_path = os.path.abspath(wav_path).replace("\\", "/")
                            VOICE_MAPPING[voice_name] = full_path
                            custom_voices.add(voice_name)
                    else:
                        # Model not loaded yet, fallback to WAV for now
                        full_path = os.path.abspath(wav_path).replace("\\", "/")
                        VOICE_MAPPING[voice_name] = full_path
                        custom_voices.add(voice_name)

    # Display default voices first
    logger.info(f"{Colors.CYAN}{Colors.BOLD}🔊 Default voices available:{Colors.RESET}")
    logger.info(f"{Colors.CYAN}   OpenAI aliases: {', '.join(DEFAULT_VOICES['openai_aliases'])}{Colors.RESET}")
    logger.info(f"{Colors.CYAN}   Pocket TTS: {', '.join(DEFAULT_VOICES['pocket_tts'])}{Colors.RESET}")
    
    # Then display custom voices
    if custom_voices:
        logger.info(f"{Colors.GREEN}{Colors.BOLD}🎤 Custom voices loaded: {Colors.RESET}{Colors.GREEN}{', '.join(sorted(list(custom_voices)))}{Colors.RESET}")
    else:
        logger.info(f"{Colors.YELLOW}No custom voices found in 'voices/' directory.{Colors.RESET}")



FFMPEG_FORMATS = {
    "mp3": ("mp3", "mp3_mf" if sys.platform == "win32" else "libmp3lame"),
    "opus": ("ogg", "opus"),
    "aac": ("adts", "aac"),
    "flac": ("flac", "flac"),
}

MEDIA_TYPES = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "aac": "audio/aac",
    "opus": "audio/opus",
    "flac": "audio/flac",
    "pcm": "audio/pcm",
}


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


class FileLikeQueueWriter:
    """File-like adapter that writes bytes to a queue with backpressure."""

    def __init__(self, queue: Queue, timeout: float = settings.queue_timeout):
        self.queue = queue
        self.timeout = timeout

    def write(self, data: bytes) -> int:
        if not data:
            return 0
        try:
            self.queue.put(data, timeout=self.timeout)
            return len(data)
        except Full:
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
            except (Full, Exception):
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.close()
        except Exception:
            logger.exception("Error closing queue writer")
        return False


# Global model state
tts_model: Optional[TTSModel] = None
device: Optional[str] = None
sample_rate: Optional[int] = None


def set_high_priority():
    """Set the current process to High Priority on Windows to avoid audio choppiness."""
    if os.name == 'nt':
        try:
            kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
            
            # Explicitly define types to avoid "Invalid Handle" (Error 6) on 64-bit systems
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            kernel32.SetPriorityClass.restype = wintypes.BOOL
            
            handle = kernel32.GetCurrentProcess()
            
            # HIGH_PRIORITY_CLASS = 0x00000080
            # ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000
            
            if kernel32.SetPriorityClass(handle, 0x00000080):
                logger.info("🚀 Process priority set to HIGH")
            else:
                err = ctypes.get_last_error()
                logger.warning(f"⚠️ Failed to set priority to HIGH (Error: {err}). Trying ABOVE_NORMAL...")
                if kernel32.SetPriorityClass(handle, 0x00008000):
                    logger.info("✅ Process priority set to ABOVE_NORMAL")
                else:
                    err = ctypes.get_last_error()
                    logger.warning(f"❌ Failed to set any elevated priority (Error: {err})")
        except Exception as e:
            logger.warning(f"⚠️ Could not set process priority: {e}")

@asynccontextmanager
async def lifespan(app):
    """Load the TTS model on startup."""
    logger.info("🚀 Starting TTS API server...")
    set_high_priority()
    load_tts_model()
    yield


def _slice_kv_cache(self, model_state: dict, sequence_length: int):
    """Memory optimization: Slice KV cache to actual prompt length.
    Monkey-patched because it's missing in some pocket-tts versions.
    """
    for module_name, module_state in model_state.items():
        if "cache" in module_state:
            cache = module_state["cache"]
            # StreamingMultiheadAttention: [2, B, T, H, D] -> T is dim 2
            # MimiStreamingMultiheadAttention: [2, B, H, T, D] -> T is dim 3
            if cache.shape[2] > sequence_length:
                module_state["cache"] = cache[:, :, :sequence_length, :, :].contiguous()
            elif cache.shape[3] > sequence_length:
                module_state["cache"] = cache[:, :, :, :sequence_length, :].contiguous()

def load_tts_model() -> None:
    """Load TTS model once and keep in memory."""
    global tts_model, device, sample_rate

    tts_model = TTSModel.load_model()
    
    # Monkey-patch missing method if needed
    if not hasattr(TTSModel, "_slice_kv_cache"):
        logger.info("🔧 Patching TTSModel with missing _slice_kv_cache method")
        TTSModel._slice_kv_cache = _slice_kv_cache

    device = tts_model.device
    sample_rate = getattr(tts_model, "sample_rate", settings.default_sample_rate)

    logger.info(f"Pocket TTS loaded | Device: {device} | Sample Rate: {sample_rate}")
    load_custom_voices()


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

    def producer():
        logger.info(f"Starting audio generation for voice: {voice_name} (model={model_tier}, temp={temperature}, steps={lsd_decode_steps}, top_p={top_p}, rep_pen={repetition_penalty})")
        try:
            with MODEL_LOCK:
                # Apply quality parameters to the global model instance
                tts_model.temp = temperature
                tts_model.top_p = top_p
                tts_model.repetition_penalty = repetition_penalty
                
                # Override steps if HD model is requested
                if "hd" in model_tier:
                    # Force higher quality for HD, but respect user choice if even higher
                    tts_model.lsd_decode_steps = max(lsd_decode_steps, 16)
                else:
                    tts_model.lsd_decode_steps = lsd_decode_steps
                
                # Dynamic device placement
                if "cuda" in model_tier and torch.cuda.is_available():
                    if tts_model.device != "cuda":
                        logger.info(f"⚡ Moving model to CUDA for generation (currently {tts_model.device})")
                        logger.info(f"FlowLM device before: {next(tts_model.flow_lm.parameters()).device}")
                        logger.info(f"Mimi device before: {next(tts_model.mimi.parameters()).device}")
                        tts_model.to("cuda")
                        logger.info(f"FlowLM device after: {next(tts_model.flow_lm.parameters()).device}")
                        logger.info(f"Mimi device after: {next(tts_model.mimi.parameters()).device}")
                else:
                    if tts_model.device != "cpu":
                        logger.info(f"🔄 Moving model back to CPU (currently {tts_model.device})")
                        tts_model.to("cpu")
                
                # Check if voice_name is a file path (custom voice)
                if os.path.exists(voice_name) and os.path.isfile(voice_name):
                    file_ext = os.path.splitext(voice_name)[1].lower()
                    if file_ext == ".safetensors":
                        logger.info(f"Loading pre-exported voice embedding: {voice_name}")
                        prompt = safetensors.torch.load_file(voice_name)["audio_prompt"]
                        prompt = prompt.to(tts_model.device)
                        
                        # Manually initialize state and prompt
                        model_state = init_states(tts_model.flow_lm, batch_size=1, sequence_length=1000)
                        with torch.no_grad():
                            tts_model._run_flow_lm_and_increment_step(model_state=model_state, audio_conditioning=prompt)
                        
                        # Optimize memory by slicing KV cache
                        num_audio_frames = prompt.shape[1]
                        tts_model._slice_kv_cache(model_state, num_audio_frames)
                    else:
                        logger.info(f"Cloning voice from file: {voice_name}")
                        model_state = tts_model.get_state_for_audio_prompt(voice_name)
                else:
                    # Standard preset voice
                    model_state = tts_model.get_state_for_audio_prompt(voice_name)
                
                audio_chunks = tts_model.generate_audio_stream(
                    model_state=model_state, text_to_generate=text
                )
                with FileLikeQueueWriter(queue) as writer:
                    stream_audio_chunks(
                        writer, audio_chunks, sample_rate or settings.default_sample_rate
                    )
        except Exception:
            logger.exception(f"Audio generation failed for voice: {voice_name}")
        finally:
            try:
                queue.put(None, timeout=settings.eof_timeout)
            except (Full, Exception):
                pass

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()
    return thread


async def _stream_queue_chunks(queue: Queue) -> AsyncIterator[bytes]:
    """Async generator that yields bytes from queue until EOF."""
    while True:
        chunk = await asyncio.to_thread(queue.get)
        if chunk is None:
            logger.debug("Received EOF from producer")
            break
        yield chunk


def _start_ffmpeg_process(format: str, speed: float = 1.0) -> tuple[subprocess.Popen, int, int]:
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
    
    # Apply high-quality speed adjustment if not 1.0
    if speed != 1.0:
        cmd.extend(["-filter:a", f"atempo={speed}"])
    
    # Enable experimental encoders (required for 'opus' in some FFmpeg builds)
    if codec == "opus":
        cmd.extend(["-strict", "-2"])

    # Force 44.1kHz for MP3 to ensure compatibility with Windows MF encoder
    if format == "mp3":
        cmd.extend(["-ar", "44100"])
        cmd.extend(["-q:a", "0"]) # Best VBR quality
    elif format in ("aac", "opus"):
        cmd.extend(["-b:a", "192k"]) # High constant bitrate

    cmd.extend([
        "-f",
        out_fmt,
        "-codec:a",
        codec,
        "pipe:1",
    ])
    r_fd, w_fd = os.pipe()
    r_file = os.fdopen(r_fd, "rb")
    proc = subprocess.Popen(cmd, stdin=r_file, stdout=subprocess.PIPE)
    r_file.close()
    return proc, w_fd, r_fd


def _start_pipe_writer(queue: Queue, write_fd: int) -> threading.Thread:
    """Start thread that writes queue chunks to OS pipe."""

    def pipe_writer():
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
        except Exception:
            try:
                os.close(write_fd)
            except (OSError, Exception):
                pass

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
    # Using the normalized voice_name passed from wrapper
    producer_thread = _start_audio_producer(
        queue, voice_name, text, temperature, lsd_decode_steps, top_p, repetition_penalty, model_tier
    )

    try:
        if format in ("wav", "pcm") and speed == 1.0:
            async for chunk in _stream_queue_chunks(queue):
                yield chunk
            producer_thread.join()
            return

        if format in FFMPEG_FORMATS or (format in ("wav", "pcm") and speed != 1.0):
            # Transcode or adjust speed
            proc, write_fd, _ = _start_ffmpeg_process(format, speed)
            writer_thread = _start_pipe_writer(queue, write_fd)

            try:
                while True:
                    chunk = await asyncio.to_thread(proc.stdout.read, chunk_size)
                    if not chunk:
                        logger.debug(f"FFmpeg output complete for {format}")
                        break
                    yield chunk
            finally:
                proc.kill()
                try:
                    proc.stdout.close()
                except Exception:
                    pass
                await asyncio.to_thread(proc.wait)
                await asyncio.to_thread(producer_thread.join)
                await asyncio.to_thread(writer_thread.join)
            return

        # Fallback
        async for chunk in _stream_queue_chunks(queue):
            yield chunk
        producer_thread.join()

    except Exception:
        logger.exception(f"Error streaming audio format: {format}")
        raise


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
    if tts_model is None:
        raise HTTPException(status_code=503, detail="TTS model not loaded")

    # Security Check: Prevent path traversal if voice is not a known preset
    if voice not in VOICE_MAPPING and ("/" in voice or "\\" in voice or voice == ".."):
        logger.warning(f"🚨 Path traversal attempt blocked: voice='{voice}'")
        raise HTTPException(status_code=400, detail="Invalid voice name")

    # Normalize voice for cache key
    voice_name = VOICE_MAPPING.get(voice, voice)
    
    # Generate Cache Key
    # We include all parameters that affect output quality or audio content
    cache_key = f"{text}|{voice_name}|{format}|{speed}|{temperature}|{lsd_decode_steps}|{top_p}|{repetition_penalty}|{model_tier}"
    cache_hash = hashlib.md5(cache_key.encode("utf-8")).hexdigest()
    cache_filename = f"{cache_hash}.{format}"
    cache_path = os.path.join(settings.audio_cache_dir, cache_filename)
    
    # 1. Check Cache
    if os.path.exists(cache_path):
        logger.info(f"Cache HIT for {cache_hash} ({format})")
        try:
            async with await open_file(cache_path, "rb") as f:
                while True:
                    chunk = await f.read(settings.chunk_size)
                    if not chunk:
                        break
                    yield chunk
            return
        except Exception as e:
            logger.warning(f"Failed to read cache file, regenerating: {e}")

    # 2. Generate and Cache (Cache Miss)
    logger.info(f"Cache MISS for {cache_hash} ({format}) - Generating...")
    temp_path = f"{cache_path}.{uuid.uuid4().hex}.tmp"
    meta_path = os.path.splitext(cache_path)[0] + ".json"
    
    try:
        async with await open_file(temp_path, "wb") as cache_file:
            async for chunk in _generate_audio_core(
                text, voice_name, speed, format, chunk_size, temperature, lsd_decode_steps, top_p, repetition_penalty, model_tier
            ):
                await cache_file.write(chunk)
                yield chunk
        
        # Rename temp to final (atomic on POSIX, usually fine on Windows if not open)
        if os.path.exists(temp_path):
             os.replace(temp_path, cache_path)
             
             # Save Metadata JSON
             metadata = {
                 "text": text,
                 "voice": voice_name,
                 "speed": speed,
                 "format": format,
                 "hash": cache_hash,
                 "model": model_tier,
                 "top_p": top_p,
                 "repetition_penalty": repetition_penalty,
                 "lsd_decode_steps": lsd_decode_steps
             }
             temp_meta_path = f"{meta_path}.{uuid.uuid4().hex}.tmp"
             try:
                 async with await open_file(temp_meta_path, "w") as f:
                     await f.write(json.dumps(metadata, indent=2))
                 if os.path.exists(temp_meta_path):
                     os.replace(temp_meta_path, meta_path)
             except Exception as e:
                 logger.warning(f"Failed to save metadata: {e}")

             logger.info(f"Cached audio with metadata saved to {cache_path}")
             # Trigger cleanup in background
             if background_tasks:
                 background_tasks.add_task(cleanup_cache)
             
    except BaseException:
        # If generation failed or was cancelled, clean up temp files (both audio and json)
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info(f"Cleaned up failed temp file: {temp_path}")
            except OSError:
                pass
        
        # Also clean up JSON if it was partially written
        if os.path.exists(meta_path):
            try:
                os.remove(meta_path)
                logger.info(f"Cleaned up failed metadata: {meta_path}")
            except OSError:
                pass
        
        # Also remove the final cache file if it exists (edge case)
        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
                logger.info(f"Cleaned up corrupted cache file: {cache_path}")
            except OSError:
                pass
        
        raise

CACHE_LIMIT = 10

async def cleanup_cache():
    """Remove oldest audio files (and their json sidecars) if cache exceeds limit."""
    def _do_cleanup():
        try:
            # Group files by extension
            audio_files = []
            extensions = tuple(list(FFMPEG_FORMATS.keys()) + ["wav", "pcm"])
            
            for f in os.listdir(settings.audio_cache_dir):
                path = os.path.join(settings.audio_cache_dir, f)
                if os.path.isfile(path) and f.endswith(extensions):
                    audio_files.append((path, os.path.getmtime(path)))
            
            if len(audio_files) <= settings.cache_limit:
                return

            # Sort by mtime (oldest first)
            audio_files.sort(key=lambda x: x[1])
            
            # Delete oldest
            to_delete = audio_files[:len(audio_files) - settings.cache_limit]
            for audio_path, _ in to_delete:
                try:
                    # Remove audio file
                    os.remove(audio_path)
                    logger.info(f"🗑️ Cache cleanup: Removed {os.path.basename(audio_path)}")
                    
                    # Remove corresponding json file
                    json_path = os.path.splitext(audio_path)[0] + ".json"
                    if os.path.exists(json_path):
                        os.remove(json_path)
                        logger.info(f"🗑️ Cache cleanup: Removed {os.path.basename(json_path)}")
                except OSError:
                    pass
        except Exception:
            logger.exception("Error during cache cleanup")

    await asyncio.to_thread(_do_cleanup)


app = FastAPI(
    title="OpenAI-Compatible TTS API (Cached)",
    description="OpenAI Audio Speech API compatible endpoint using Kyutai TTS with model caching",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware to allow SillyTavern (and other clients) to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/v1/voices")
async def get_voices():
    """Return all available voices (built-in + custom)."""
    # Combine everything: default aliases, pocket_tts names, and custom keys in VOICE_MAPPING
    voices = set(DEFAULT_VOICES["openai_aliases"] + DEFAULT_VOICES["pocket_tts"])
    for v in VOICE_MAPPING.keys():
        voices.add(v)
    return {"voices": sorted(list(voices))}


@app.get("/v1/formats")
async def get_formats():
    """Return supported audio formats."""
    return {"formats": sorted(list(MEDIA_TYPES.keys()))}


@app.post("/v1/audio/speech")
async def text_to_speech(request: SpeechRequest, background_tasks: BackgroundTasks) -> StreamingResponse:
    """Generate speech audio from text with streaming response."""
    try:
        logger.info(f"Received request: voice='{request.voice}', format='{request.response_format}', input_len={len(request.input)}")
        
        return StreamingResponse(
            generate_audio(
                text=request.input,
                voice=request.voice,
                speed=request.speed,
                format=request.response_format,
                temperature=request.temperature,
                lsd_decode_steps=request.lsd_decode_steps,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                model_tier=request.model,
                stream=request.stream,
                background_tasks=background_tasks,
            ),
            media_type=MEDIA_TYPES.get(request.response_format, "audio/wav"),
            headers={
                "Transfer-Encoding": "chunked",
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    except Exception as e:
        logger.exception("Internal Server Error in text_to_speech")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/audio/export-voice")
async def export_voice(request: ExportVoiceRequest):
    """Manually export a WAV voice to safetensors embedding."""
    if not tts_model:
        raise HTTPException(status_code=503, detail="TTS model not loaded")
    
    voice_name = request.voice
    
    # Security Check: Prevent path traversal
    if "/" in voice_name or "\\" in voice_name or voice_name == "..":
        logger.warning(f"🚨 Path traversal attempt blocked in export: voice='{voice_name}'")
        raise HTTPException(status_code=400, detail="Invalid voice name")

    wav_path = os.path.join(settings.voices_dir, f"{voice_name}.wav")
    st_path = os.path.join(settings.embeddings_dir, f"{voice_name}.safetensors")
    
    if not os.path.exists(wav_path):
        # Check if voice_name already has extension
        if voice_name.lower().endswith(".wav"):
             wav_path = os.path.join(settings.voices_dir, voice_name)
             st_path = os.path.join(settings.embeddings_dir, os.path.splitext(voice_name)[0] + ".safetensors")
        
        if not os.path.exists(wav_path):
            raise HTTPException(status_code=404, detail=f"Voice '{voice_name}' (WAV) not found in {settings.voices_dir}")

    try:
        logger.info(f"✨ Manually exporting '{voice_name}' to embeddings/ (temp={request.temperature}, steps={request.lsd_decode_steps})...")
        audio, sr = sf.read(wav_path)
        audio_pt = torch.from_numpy(audio).float()
        if len(audio_pt.shape) == 1:
            audio_pt = audio_pt.unsqueeze(0)
            
        if request.truncate:
            max_samples = int(30 * sr)
            if audio_pt.shape[-1] > max_samples:
                audio_pt = audio_pt[..., :max_samples]
                logger.info(f"Audio truncated to 30s for export")

        audio_resampled = convert_audio(audio_pt, sr, tts_model.config.mimi.sample_rate, 1)
        
        with torch.no_grad():
            with MODEL_LOCK:
                tts_model.temp = request.temperature
                tts_model.lsd_decode_steps = request.lsd_decode_steps
                prompt = tts_model._encode_audio(audio_resampled.unsqueeze(0).to(tts_model.device))
        
        safetensors.torch.save_file({"audio_prompt": prompt.cpu()}, st_path)
        logger.info(f"✅ Exported '{voice_name}' to {st_path}")
        
        # Reload custom voices to update mapping
        await asyncio.to_thread(load_custom_voices)
        
        return {"status": "success", "message": f"Exported {voice_name} to safetensors", "path": st_path}
    except Exception as e:
        logger.exception(f"Failed to export voice '{voice_name}'")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Simple healthcheck endpoint."""
    return {
        "status": "ok",
        "model_loaded": tts_model is not None,
        "device": device,
        "sample_rate": sample_rate,
    }

if __name__ == "__main__":

    # Configure uvicorn logging for HTTP debugging
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["default"][
        "fmt"
    ] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_config["formatters"]["access"][
        "fmt"
    ] = '%(asctime)s - %(client_addr)s - "%(request_line)s" %(status_code)s'

    try:
        port = settings.server_port
        host = settings.server_host
        logger.info(f"Starting server with HTTP debug logging enabled")
        logger.info(f"✅ Server binding to: http://{host}:{port}")
        logger.info(f"ℹ️  If you are using SillyTavern, set provider endpoint to: http://127.0.0.1:{port}/v1/audio/speech")
        uvicorn.run(app, host=host, port=port, log_config=log_config, access_log=True)
    except Exception as e:
        logger.exception("Failed to start server")
        input("Press Enter to exit...")

