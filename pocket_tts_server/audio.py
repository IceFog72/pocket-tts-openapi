"""Audio generation, FFmpeg transcoding, and streaming."""
import asyncio
import json
import logging
import os
import re
import subprocess
import threading
import time
import uuid
import wave
from queue import Empty, Full, Queue
from typing import AsyncIterator, Optional, Tuple
from pathlib import Path

# Patch wave.Wave_write.close to handle IOError on client disconnect
_orig_wave_close = wave.Wave_write.close
def _safe_wave_close(self):
    try:
        _orig_wave_close(self)
    except (IOError, OSError):
        pass  # Client disconnected during header patch — ignore
wave.Wave_write.close = _safe_wave_close

import numpy as np
import soundfile as sf
import torch
from anyio import open_file
from fastapi import BackgroundTasks, HTTPException

import safetensors.torch
from pocket_tts.data.audio import stream_audio_chunks
from pocket_tts.data.audio_utils import convert_audio
from pocket_tts.modules.stateful_module import init_states

from .cache import cache_manager
from .config import settings
from .constants import CACHE_EXTENSIONS, FFMPEG_FORMATS, MEDIA_TYPES, VOICE_MAPPING
from .model_manager import model_manager
from .validation import _ffmpeg_available, is_valid_voice_name, sanitize_text_input

logger = logging.getLogger(__name__)


class FileLikeQueueWriter:
    def __init__(self, queue: Queue, timeout: float = 30.0):
        self.queue = queue
        self.timeout = timeout
        self._pos = 0

    def tell(self):
        return self._pos

    def seek(self, offset, whence=0):
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
            pass
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
    def producer() -> None:
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
                        model_manager.move_to_device("cuda")
                else:
                    if current_device != "cpu":
                        model_manager.move_to_device("cpu")

                is_safe_file = os.path.isabs(voice_name)
                if os.path.exists(voice_name) and os.path.isfile(voice_name) and is_safe_file:
                    file_ext = os.path.splitext(voice_name)[1].lower()
                    if file_ext == ".safetensors":
                        prompt = safetensors.torch.load_file(voice_name)["audio_prompt"]
                        prompt = prompt.to(tts_model.device)
                        model_state = init_states(tts_model.flow_lm, batch_size=1, sequence_length=1000)
                        with torch.no_grad():
                            tts_model._run_flow_lm_and_increment_step(model_state=model_state, audio_conditioning=prompt)
                        num_audio_frames = prompt.shape[1]
                        tts_model._slice_kv_cache(model_state, num_audio_frames)
                    else:
                        model_state = tts_model.get_state_for_audio_prompt(voice_name)
                else:
                    model_state = tts_model.get_state_for_audio_prompt(voice_name)

                parts = re.split(r'(?<=[.!?])\s+', text.strip())

                def combined_chunks():
                    for part in parts:
                        if not part.strip():
                            continue
                        if len(part) > 500:
                            subtasks = re.split(r'(?<=[,;])\s+', part)
                        else:
                            subtasks = [part]
                        for subtask in subtasks:
                            if not subtask.strip():
                                continue
                            for chunk in tts_model.generate_audio_stream(
                                model_state=model_state, text_to_generate=subtask
                            ):
                                yield chunk

                with FileLikeQueueWriter(queue) as writer:
                    stream_audio_chunks(writer, combined_chunks(), model_manager.sample_rate)
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


async def _stream_queue_chunks(queue: Queue) -> AsyncIterator[bytes]:
    while True:
        try:
            chunk = await asyncio.to_thread(queue.get, timeout=5.0)
        except Exception:
            if getattr(queue, "abort", False):
                break
            continue
        if chunk is None:
            break
        yield chunk


def _start_ffmpeg_process(format: str, speed: float = 1.0) -> Tuple[subprocess.Popen, int, int]:
    out_fmt, codec = FFMPEG_FORMATS.get(format, ("wav", "pcm_s16le"))
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "wav", "-i", "pipe:0"]

    if speed != 1.0:
        filters = []
        s = speed
        while s < 0.5:
            filters.append("atempo=0.5")
            s *= 2.0
        filters.append(f"atempo={s}")
        cmd.extend(["-filter:a", ",".join(filters)])

    if codec == "opus":
        cmd.extend(["-strict", "-2"])
    if format == "mp3":
        cmd.extend(["-ar", "44100", "-q:a", "0"])
    elif format in ("aac", "opus"):
        cmd.extend(["-b:a", "192k"])

    cmd.extend(["-f", out_fmt, "-codec:a", codec, "pipe:1"])

    r_fd, w_fd = os.pipe()
    r_file = os.fdopen(r_fd, "rb")
    try:
        proc = subprocess.Popen(cmd, stdin=r_file, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except Exception:
        r_file.close()
        os.close(w_fd)
        raise
    r_file.close()
    return proc, w_fd, r_fd


def _start_pipe_writer(queue: Queue, write_fd: int) -> threading.Thread:
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
    needs_ffmpeg = format in FFMPEG_FORMATS or (format in ("wav", "pcm") and speed != 1.0)
    if needs_ffmpeg and not _ffmpeg_available:
        raise HTTPException(status_code=400, detail=f"FFmpeg required for format '{format}' or speed adjustment, but FFmpeg is not installed")

    queue: Queue[Optional[bytes]] = Queue(maxsize=settings.queue_size)
    queue.abort = False  # type: ignore[attr-defined]
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
            ffmpeg_proc, write_fd, _ = _start_ffmpeg_process(format, speed)
            proc = ffmpeg_proc
            writer_thread = _start_pipe_writer(queue, write_fd)

            try:
                while True:
                    chunk = await asyncio.to_thread(proc.stdout.read, chunk_size)  # type: ignore[union-attr]
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
                        if proc.stdout:
                            proc.stdout.close()  # type: ignore[union-attr]
                    except Exception:
                        pass
                    if proc.stderr:
                        try:
                            proc.stderr.close()  # type: ignore[union-attr]
                        except Exception:
                            pass

                await asyncio.to_thread(producer_thread.join)
                if writer_thread:
                    await asyncio.to_thread(writer_thread.join)
            return

        raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")

    except Exception as e:
        logger.exception(f"Error streaming audio format {format}: {e}")
        raise
    finally:
        queue.abort = True  # type: ignore[attr-defined]
        while not queue.empty():
            try:
                queue.get_nowait()
            except Exception:
                break


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
    skip_cache: bool = False,
) -> AsyncIterator[bytes]:
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="TTS model not loaded")

    try:
        text = sanitize_text_input(text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not is_valid_voice_name(voice) and voice not in VOICE_MAPPING:
        raise HTTPException(status_code=400, detail="Invalid voice name")

    v_lower = voice.lower()
    if v_lower in VOICE_MAPPING:
        voice_name = VOICE_MAPPING[v_lower]
    else:
        voice_name = VOICE_MAPPING.get(voice, voice)

    # Skip caching for WebSocket streaming or when explicitly requested
    if skip_cache:
        cache_path = None
        meta_path = None
        temp_path = None
    else:
        cache_hash = cache_manager.generate_cache_key(
            text, voice_name, format, speed, temperature,
            lsd_decode_steps, top_p, repetition_penalty, model_tier
        )
        cache_path, meta_path = cache_manager.get_cache_path(cache_hash, format)

        # Ensure cache directory exists
        cache_dir = Path(settings.audio_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        if cache_manager.check_cache(cache_path):
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
                "text": text, "voice": voice_name, "speed": speed,
                "format": format, "hash": cache_hash, "model": model_tier,
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
        for path in [temp_path, meta_path, cache_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
        raise
