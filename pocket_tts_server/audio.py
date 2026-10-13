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
import io
import struct
from contextlib import aclosing
from queue import Empty, Full, Queue
from typing import AsyncIterator, Optional, Tuple
from pathlib import Path

import torch
from anyio import open_file
from fastapi import BackgroundTasks, HTTPException
from pydantic import ValidationError

import soundfile
import safetensors.torch
from pocket_tts.modules.stateful_module import init_states

from .cache import cache_manager
from .config import settings
from .constants import FFMPEG_FORMATS
from .errors import InputError, error_details
from .models import SpeechRequest
from .model_manager import model_manager
from .rate_limiter import synthesis_capacity
from .validation import _ffmpeg_available, resolve_voice

logger = logging.getLogger(__name__)


def _is_legacy_safetensors(path: str) -> bool:
    """Check if a safetensors file uses the old {"audio_prompt": ...} format."""
    try:
        with safetensors.safe_open(path, framework="pt") as f:
            keys = list(f.keys())
        return "audio_prompt" in keys and not any("/" in k for k in keys)
    except Exception:
        return False


def _load_legacy_safetensors(tts_model, path: str) -> dict:
    """Load old-format safetensors embedding via manual encode pipeline.

    Old embeddings store only the audio_prompt tensor. We need to run it
    through the flow_lm to create a proper model_state with KV caches.
    """
    logger.info(f"Loading legacy-format embedding: {path}")
    prompt = safetensors.torch.load_file(path)["audio_prompt"]
    prompt = prompt.to(tts_model.device)
    
    if getattr(tts_model.flow_lm, "insert_bos_before_voice", False):
        prompt = torch.cat([tts_model.flow_lm.bos_before_voice, prompt], dim=1)
        
    model_state = init_states(tts_model.flow_lm, batch_size=1, sequence_length=prompt.shape[1])
    with torch.no_grad():
        tts_model._run_flow_lm_and_increment_step(
            model_state=model_state, audio_conditioning=prompt
        )
        
    num_audio_frames = prompt.shape[1]
    if hasattr(tts_model, '_slice_kv_cache'):
        tts_model._slice_kv_cache(model_state, num_audio_frames)
    return model_state


class FileLikeQueueWriter:
    def __init__(self, queue: Queue, timeout: float = 30.0):
        self.queue = queue
        self.timeout = timeout
        self._pos = 0

    def tell(self):
        return self._pos

    def seekable(self):
        return False

    def seek(self, offset, whence=0):
        raise io.UnsupportedOperation("Audio queue is not seekable")

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
        # Don't put EOF here — the producer's outer finally block handles it.
        # Putting None here causes a dual EOF sentinel bug.
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.close()
        except Exception:
            pass
        return False


def _stream_wav_chunks(writer, chunks, sample_rate):
    # Unknown lengths are conventional for a nonseekable WAV stream. Writing
    # the header directly avoids wave.Wave_write trying to seek on failures.
    writer.write(struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 0xFFFFFFFF,
                             b"WAVE", b"fmt ", 16, 1, 1, sample_rate,
                             sample_rate * 2, 2, 16, b"data", 0xFFFFFFFF))
    for chunk in chunks:
        writer.write((chunk.clamp(-1, 1) * 32767).short().detach().cpu().numpy().tobytes())
    writer.write(bytes(int(sample_rate * 0.2) * 2))


def _start_audio_producer(
    queue: Queue,
    voice_name: str,
    text: str,
    temperature: float = settings.temperature,
    lsd_decode_steps: int = settings.lsd_decode_steps,
    top_p: float = settings.top_p,
    repetition_penalty: float = settings.repetition_penalty,
    model_tier: str = settings.model_tier,
    raw_pcm: bool = False,
) -> threading.Thread:
    def producer() -> None:
        try:
            target_device = "cuda" if "gpu" in model_tier or "cuda" in model_tier else "cpu"
            target_lang = model_tier.replace("-cpu", "").replace("-gpu", "").replace("-cuda", "")
            if not target_lang or target_lang in ("tts", "tts-1", "tts-1-hd"):
                target_lang = settings.language
            
            while not model_manager.acquire_lock(timeout=0.1):
                if getattr(queue, "abort", False):
                    return
            try:
                if getattr(queue, "abort", False):
                    return
                if model_manager.language != target_lang:
                    model_manager.load(language=target_lang)
                if getattr(queue, "abort", False):
                    return
                tts_model = model_manager.model
                if tts_model is None:
                    raise RuntimeError("TTS model not loaded")

                tts_model.temp = temperature
                tts_model.sampler_decode_steps = lsd_decode_steps

                if target_device == "cuda" and torch.cuda.is_available():
                    if model_manager.device != "cuda":
                        model_manager.move_to_device("cuda")
                else:
                    if model_manager.device != "cpu":
                        model_manager.move_to_device("cpu")

                try:
                    # Backward compat: detect old-format safetensors with {"audio_prompt": ...}
                    if (
                        voice_name.endswith(".safetensors")
                        and os.path.isfile(voice_name)
                        and _is_legacy_safetensors(voice_name)
                    ):
                        model_state = _load_legacy_safetensors(tts_model, voice_name)
                    else:
                        model_state = tts_model.get_state_for_audio_prompt(voice_name)

                except (OSError, soundfile.SoundFileError, safetensors.SafetensorError) as exc:
                    raise InputError("Voice reference could not be read. Choose another voice or replace its WAV/embedding",
                                     code="invalid_voice_file", param="voice", status_code=422) from exc

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
                            if getattr(queue, "abort", False):
                                return
                            for chunk in tts_model.generate_audio_stream(
                                model_state=model_state, text_to_generate=subtask
                            ):
                                yield chunk

                with FileLikeQueueWriter(queue) as writer:
                    if raw_pcm:
                        for chunk in combined_chunks():
                            writer.write((chunk.clamp(-1, 1) * 32767).short().cpu().numpy().tobytes())
                    else:
                        _stream_wav_chunks(writer, combined_chunks(), model_manager.sample_rate)
            finally:
                model_manager.release_lock()
        except Exception as e:
            if getattr(queue, "abort", False):
                return
            queue.error = e
            if isinstance(e, InputError):
                logger.info("Rejected voice reference: %s", e.detail)
            else:
                logger.error(f"Audio generation failed for voice {voice_name}: {e}")
            try:
                queue.put(e, timeout=settings.eof_timeout)
            except Full:
                pass
        finally:
            synthesis_capacity.release()
            try:
                queue.put(None, timeout=settings.eof_timeout)
            except Full:
                pass
            queue.finished = True
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not synthesis_capacity.acquire():
        raise HTTPException(status_code=503, detail="Synthesis capacity exhausted")
    thread = threading.Thread(target=producer, daemon=True)
    try:
        thread.start()
    except BaseException:
        synthesis_capacity.release()
        raise
    return thread


async def _stream_queue_chunks(queue: Queue, producer: threading.Thread) -> AsyncIterator[bytes]:
    while True:
        try:
            chunk = await asyncio.to_thread(queue.get, timeout=5.0)
        except Empty:
            if getattr(queue, "abort", False) or not producer.is_alive():
                break
            continue
        except Exception:
            break
        if chunk is None:
            break
        if isinstance(chunk, Exception):
            raise chunk
        yield chunk
    if getattr(queue, "error", None) is not None:
        raise queue.error


def _start_ffmpeg_process(format: str, speed: float = 1.0) -> Tuple[subprocess.Popen, int]:
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
    elif format in ("aac", "opus", "webm"):
        cmd.extend(["-b:a", "192k"])

    if format == "webm":
        cmd.extend(["-live", "1", "-cluster_time_limit", "100"])
    if format == "pcm":
        out_fmt, codec = "s16le", "pcm_s16le"
    cmd.extend(["-flush_packets", "1", "-f", out_fmt, "-codec:a", codec, "pipe:1"])

    r_fd, w_fd = os.pipe()
    r_file = os.fdopen(r_fd, "rb")
    try:
        proc = subprocess.Popen(cmd, stdin=r_file, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        r_file.close()
        os.close(w_fd)
        raise
    r_file.close()
    return proc, w_fd


def _start_pipe_writer(queue: Queue, write_fd: int) -> threading.Thread:
    def pipe_writer() -> None:
        try:
            with os.fdopen(write_fd, "wb") as pipe:
                while True:
                    try:
                        data = queue.get(timeout=1)
                    except Empty:
                        if getattr(queue, "abort", False):
                            break
                        if getattr(queue, "finished", False):
                            if getattr(queue, "error", None) is not None:
                                raise queue.error
                            break
                        continue
                    if isinstance(data, Exception):
                        raise data
                    if data is None:
                        if getattr(queue, "error", None) is not None:
                            raise queue.error
                        break
                    try:
                        pipe.write(data)
                    except (BrokenPipeError, OSError):
                        raise
                pipe.flush()
        except Exception as exc:
            thread.error = exc

    thread = threading.Thread(target=pipe_writer, daemon=True)
    thread.error = None
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
        queue, voice_name, text, temperature, lsd_decode_steps, top_p, repetition_penalty, model_tier,
        raw_pcm=(format == "pcm" and speed == 1.0),
    )

    proc: Optional[subprocess.Popen] = None
    writer_thread: Optional[threading.Thread] = None

    try:
        if format in ("wav", "pcm") and speed == 1.0:
            async for chunk in _stream_queue_chunks(queue, producer_thread):
                yield chunk
            await asyncio.to_thread(producer_thread.join, 30)
            return

        if format in FFMPEG_FORMATS or (format in ("wav", "pcm") and speed != 1.0):
            ffmpeg_proc, write_fd = _start_ffmpeg_process(format, speed)
            proc = ffmpeg_proc
            writer_thread = _start_pipe_writer(queue, write_fd)

            # Drain stderr concurrently so a failing encoder cannot fill its
            # pipe and deadlock audio delivery. Retain only a bounded tail.
            stderr_tail = bytearray()
            def drain_stderr():
                while True:
                    data = proc.stderr.read(4096)
                    if not data:
                        break
                    stderr_tail.extend(data)
                    del stderr_tail[:-8192]
            stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
            stderr_thread.start()
            try:
                while True:
                    chunk = await asyncio.to_thread(proc.stdout.read1, chunk_size)  # type: ignore[union-attr]
                    if not chunk:
                        break
                    yield chunk
                returncode = await asyncio.to_thread(proc.wait, timeout=5)
                await asyncio.to_thread(writer_thread.join, 5)
                await asyncio.to_thread(stderr_thread.join, 5)
                if writer_thread.is_alive():
                    raise RuntimeError("FFmpeg input writer did not finish")
                if writer_thread.error is not None:
                    raise writer_thread.error
                if returncode != 0:
                    detail = stderr_tail.decode("utf-8", errors="replace").strip()
                    raise RuntimeError(f"FFmpeg exited with status {returncode}: {detail}")
            finally:
                queue.abort = True
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
                    await asyncio.to_thread(stderr_thread.join, 5)
                    if proc.stderr and not stderr_thread.is_alive():
                        proc.stderr.close()

                queue.abort = True
                await asyncio.to_thread(producer_thread.join, 30)
                if writer_thread:
                    await asyncio.to_thread(writer_thread.join, 5)
            return

    except InputError:
        raise
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
        request = SpeechRequest(input=text, voice=voice, model=model_tier,
                                response_format=format, speed=speed, temperature=temperature,
                                lsd_decode_steps=lsd_decode_steps)
    except ValidationError as exc:
        details = error_details(exc)
        raise InputError(details['message'], param=details['param']) from exc
    text, speed, format = request.input, request.speed, request.response_format
    model_tier = request.model
    voice_name = resolve_voice(request.voice)

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
                logger.info(f"Cache hit: {text[:80]}{'...' if len(text) > 80 else ''} ({os.path.getsize(cache_path)} bytes)")
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
        async with aclosing(_generate_audio_core(
            text, voice_name, speed, format, chunk_size,
            temperature, lsd_decode_steps, top_p, repetition_penalty, model_tier,
        )) as source:
            if skip_cache:
                async for chunk in source:
                    yield chunk
            else:
                async with await open_file(temp_path, "wb") as cache_file:
                    async for chunk in source:
                        await cache_file.write(chunk)
                        yield chunk

        if not skip_cache:
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
        raise HTTPException(status_code=500, detail=f"Audio generation failed: {str(e)}")
    finally:
        # Each request owns only its unique temporary file. A concurrent
        # request may already have published the shared audio and metadata.
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
