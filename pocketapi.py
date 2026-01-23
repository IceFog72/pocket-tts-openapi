import asyncio
import io
import logging
import os
import subprocess
import threading
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pocket_tts import TTSModel
from pocket_tts.data.audio import stream_audio_chunks
from pydantic import BaseModel, Field, field_validator
from queue import Queue, Full
from typing import Literal, Optional, AsyncIterator
import soundfile as sf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
QUEUE_SIZE = 256  # Increased from 32 to handle slower consumers
QUEUE_TIMEOUT = 10.0  # Increased from 2.0 to prevent dropping chunks
EOF_TIMEOUT = 1.0
CHUNK_SIZE = 32 * 1024
DEFAULT_SAMPLE_RATE = 24000

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

VOICES_DIR = "voices"
os.makedirs(VOICES_DIR, exist_ok=True)

def load_custom_voices():
    """Scan voices directory and update mapping."""
    import soundfile as sf
    import numpy as np
    
    custom_voices = []
    if os.path.exists(VOICES_DIR):
        for f in os.listdir(VOICES_DIR):
            if f.lower().endswith(".wav"):
                voice_name = os.path.splitext(f)[0]
                file_path = os.path.join(VOICES_DIR, f)
                
                # Validate and convert WAV file if needed
                try:
                    # Check file format info first (faster than reading full audio)
                    file_info = sf.info(file_path)
                    needs_conversion = file_info.subtype != 'PCM_16'
                    
                    if needs_conversion:
                        logger.info(f"Converting {f} from {file_info.subtype} to PCM_16...")
                        
                        # Only read audio data if conversion is needed
                        audio_data, sample_rate = sf.read(file_path)
                        
                        # Normalize to -1 to 1 if needed
                        if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                            audio_data = np.clip(audio_data, -1.0, 1.0)
                        
                        # Convert to int16
                        audio_data = (audio_data * 32767).astype(np.int16)
                        
                        # Save back to file
                        sf.write(file_path, audio_data, sample_rate, subtype='PCM_16')
                        logger.info(f"✓ Converted {f} to PCM_16 format")
                    
                    # Map name to full absolute path with forward slashes
                    full_path = os.path.abspath(file_path).replace("\\", "/")
                    VOICE_MAPPING[voice_name] = full_path
                    custom_voices.append(voice_name)
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load voice '{voice_name}': {e}")
                    continue
    
    # Display default voices first
    logger.info(f"{Colors.CYAN}{Colors.BOLD}🔊 Default voices available:{Colors.RESET}")
    logger.info(f"{Colors.CYAN}   OpenAI aliases: {', '.join(DEFAULT_VOICES['openai_aliases'])}{Colors.RESET}")
    logger.info(f"{Colors.CYAN}   Pocket TTS: {', '.join(DEFAULT_VOICES['pocket_tts'])}{Colors.RESET}")
    
    # Then display custom voices
    if custom_voices:
        logger.info(f"{Colors.GREEN}{Colors.BOLD}🎤 Custom voices loaded: {Colors.RESET}{Colors.GREEN}{', '.join(custom_voices)}{Colors.RESET}")
    else:
        logger.info(f"{Colors.YELLOW}No custom voices found in 'voices/' directory.{Colors.RESET}")


import sys

FFMPEG_FORMATS = {
    "mp3": ("mp3", "mp3_mf" if sys.platform == "win32" else "libmp3lame"),
    "opus": ("ogg", "libopus"),
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
    model: Literal["tts-1", "tts-1-hd"] = Field("tts-1", description="TTS model to use")
    input: str = Field(
        ..., min_length=1, max_length=4096, description="Text to generate"
    )
    voice: str = Field("alloy", description="Voice identifier (predefined or custom)")
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field("wav")
    speed: Optional[float] = Field(1.0, ge=0.25, le=4.0)

    @field_validator("model", mode="before")
    @classmethod
    def validate_model(cls, v: str) -> str:
        if not v:
            return "tts-1"
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


class FileLikeQueueWriter:
    """File-like adapter that writes bytes to a queue with backpressure."""

    def __init__(self, queue: Queue, timeout: float = QUEUE_TIMEOUT):
        self.queue = queue
        self.timeout = timeout

    def write(self, data: bytes) -> int:
        if not data:
            return 0
        try:
            self.queue.put(data, timeout=self.timeout)
            return len(data)
        except Full:
            logger.warning("Queue full, dropping chunk")
            return 0

    def flush(self) -> None:
        pass

    def close(self) -> None:
        try:
            self.queue.put(None, timeout=EOF_TIMEOUT)
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


@asynccontextmanager
async def lifespan(app):
    """Load the TTS model on startup."""
    logger.info("🚀 Starting TTS API server...")
    load_tts_model()
    yield


def load_tts_model() -> None:
    """Load TTS model once and keep in memory."""
    global tts_model, device, sample_rate

    tts_model = TTSModel.load_model()
    device = tts_model.device
    sample_rate = getattr(tts_model, "sample_rate", DEFAULT_SAMPLE_RATE)

    logger.info(f"Pocket TTS loaded | Device: {device} | Sample Rate: {sample_rate}")
    load_custom_voices()


def _start_audio_producer(queue: Queue, voice_name: str, text: str) -> threading.Thread:
    """Start background thread that generates audio and writes to queue."""

    def producer():
        logger.info(f"Starting audio generation for voice: {voice_name}")
        try:
            # Check if voice_name is a file path (custom voice)
            if os.path.exists(voice_name) and os.path.isfile(voice_name):
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
                    writer, audio_chunks, sample_rate or DEFAULT_SAMPLE_RATE
                )
        except Exception:
            logger.exception(f"Audio generation failed for voice: {voice_name}")
        finally:
            try:
                queue.put(None, timeout=EOF_TIMEOUT)
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
            logger.info("Received EOF")
            break
        yield chunk


def _start_ffmpeg_process(format: str) -> tuple[subprocess.Popen, int, int]:
    """Start ffmpeg process with OS pipe for stdin."""
    out_fmt, codec = FFMPEG_FORMATS.get(format, ("mp3", "libmp3lame"))
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
    
    # Force 44.1kHz for MP3 to ensure compatibility with Windows MF encoder
    if format == "mp3":
        cmd.extend(["-ar", "44100"])

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


import hashlib
from anyio import open_file, Path

# ... (Previous imports)

AUDIO_CACHE_DIR = "audio_cache"
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

async def _generate_audio_core(
    text: str,
    voice_name: str,
    speed: float,
    format: str,
    chunk_size: int,
) -> AsyncIterator[bytes]:
    """Internal generator for the actual TTS + FFmpeg logic."""
    queue = Queue(maxsize=QUEUE_SIZE)
    # Using the normalized voice_name passed from wrapper
    producer_thread = _start_audio_producer(queue, voice_name, text)

    try:
        if format in ("wav", "pcm"):
            async for chunk in _stream_queue_chunks(queue):
                yield chunk
            producer_thread.join()
            return

        if format in FFMPEG_FORMATS:
            proc, write_fd, _ = _start_ffmpeg_process(format)
            writer_thread = _start_pipe_writer(queue, write_fd)

            try:
                while True:
                    chunk = await asyncio.to_thread(proc.stdout.read, chunk_size)
                    if not chunk:
                        logger.info(f"FFmpeg output complete for {format}")
                        break
                    yield chunk
            finally:
                proc.wait()
                producer_thread.join()
                writer_thread.join()
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
    chunk_size: int = CHUNK_SIZE,
) -> AsyncIterator[bytes]:
    """Generate and stream audio, with filesystem caching."""
    if tts_model is None:
        raise HTTPException(status_code=503, detail="TTS model not loaded")

    # Normalize voice for cache key
    voice_name = VOICE_MAPPING.get(voice, voice)
    
    # Generate Cache Key
    # We include text, normalized voice, speed, and format
    cache_key = f"{text}|{voice_name}|{format}|{speed}"
    cache_hash = hashlib.md5(cache_key.encode("utf-8")).hexdigest()
    cache_filename = f"{cache_hash}.{format}"
    cache_path = os.path.join(AUDIO_CACHE_DIR, cache_filename)
    
    # 1. Check Cache
    if os.path.exists(cache_path):
        logger.info(f"Cache HIT for {cache_hash} ({format})")
        try:
            async with await open_file(cache_path, "rb") as f:
                while True:
                    chunk = await f.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    yield chunk
            return
        except Exception as e:
            logger.warning(f"Failed to read cache file, regenerating: {e}")

    # 2. Generate and Cache (Cache Miss)
    logger.info(f"Cache MISS for {cache_hash} ({format}) - Generating...")
    temp_path = cache_path + ".tmp"
    meta_path = os.path.splitext(cache_path)[0] + ".json"
    
    try:
        async with await open_file(temp_path, "wb") as cache_file:
            async for chunk in _generate_audio_core(text, voice_name, speed, format, chunk_size):
                await cache_file.write(chunk)
                yield chunk
        
        # Rename temp to final (atomic on POSIX, usually fine on Windows if not open)
        if os.path.exists(temp_path):
             os.replace(temp_path, cache_path)
             
             # Save Metadata JSON
             import json
             metadata = {
                 "text": text,
                 "voice": voice_name,
                 "speed": speed,
                 "format": format,
                 "hash": cache_hash
             }
             try:
                 async with await open_file(meta_path, "w") as f:
                     await f.write(json.dumps(metadata, indent=2))
             except Exception as e:
                 logger.warning(f"Failed to save metadata: {e}")

             logger.info(f"Cached audio with metadata saved to {cache_path}")
             # Trigger cleanup in background
             asyncio.create_task(cleanup_cache())
             
    except Exception:
        # If generation failed, clean up temp files (both audio and json)
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
            
            for f in os.listdir(AUDIO_CACHE_DIR):
                path = os.path.join(AUDIO_CACHE_DIR, f)
                if os.path.isfile(path) and f.endswith(extensions):
                    audio_files.append((path, os.path.getmtime(path)))
            
            if len(audio_files) <= CACHE_LIMIT:
                return

            # Sort by mtime (oldest first)
            audio_files.sort(key=lambda x: x[1])
            
            # Delete oldest
            to_delete = audio_files[:len(audio_files) - CACHE_LIMIT]
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


@app.post("/v1/audio/speech")
async def text_to_speech(request: SpeechRequest) -> StreamingResponse:
    """Generate speech audio from text with streaming response."""
    try:
        logger.info(f"Received request: voice='{request.voice}', format='{request.response_format}', input_len={len(request.input)}")
        
        return StreamingResponse(
            generate_audio(
                text=request.input,
                voice=request.voice,
                speed=request.speed,
                format=request.response_format,
            ),
            media_type=MEDIA_TYPES.get(request.response_format, "audio/wav"),
        )
    except Exception as e:
        logger.exception("Internal Server Error in text_to_speech")
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

    import socket

    def find_free_port(start_port: int = 8001, max_retries: int = 20) -> int:
        """Find the first available port starting from start_port."""
        for port in range(start_port, start_port + max_retries):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("0.0.0.0", port))
                    return port
                except OSError:
                    continue
        raise RuntimeError(f"Could not find a free port in range {start_port}-{start_port + max_retries}")

    try:
        port = find_free_port(8001)
        logger.info(f"Starting server with HTTP debug logging enabled")
        logger.info(f"✅ Server binding to: http://0.0.0.0:{port}")
        logger.info(f"ℹ️  If you are using SillyTavern, set provider endpoint to: http://127.0.0.1:{port}/v1/audio/speech")
        uvicorn.run(app, host="0.0.0.0", port=port, log_config=log_config, access_log=True)
    except Exception as e:
        logger.exception("Failed to start server")
        input("Press Enter to exit...")

