"""FastAPI app, middleware, and API endpoints."""
import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from contextlib import asynccontextmanager

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .audio import generate_audio
from .cache import cache_manager
from .config import settings
from .constants import CACHE_EXTENSIONS, DEFAULT_VOICES, MEDIA_TYPES, VOICE_MAPPING
from .model_manager import model_manager
from .models import ExportVoiceRequest, SpeechRequest
from .rate_limiter import rate_limiter
from .validation import _ffmpeg_available, check_ffmpeg, is_valid_voice_name, normalize_text
from .voices import voice_lock

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


def check_hf_auth() -> bool:
    try:
        from huggingface_hub import whoami
        whoami()
        return True
    except Exception:
        return False


def setup_hf_auth() -> bool:
    try:
        from huggingface_hub import login, whoami
        try:
            whoami()
            logger.info("HuggingFace: Already authenticated")
            return True
        except Exception:
            pass
        env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if env_token:
            login(token=env_token)
            return True
        return False
    except ImportError:
        return False
    except Exception:
        return False


def has_voice_cloning() -> bool:
    model = model_manager.model
    if model is None:
        return False
    return getattr(model, 'has_voice_cloning', False)


shutdown_event = asyncio.Event()


@asynccontextmanager
async def lifespan(app: FastAPI):
    from .voices import load_custom_voices

    logger.info("Starting TTS API server...")

    # Ensure required directories exist
    for dir_path in [settings.audio_cache_dir, settings.voices_dir, settings.embeddings_dir]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    if not _ffmpeg_available:
        logger.warning("FFmpeg not found. MP3/Opus/AAC/FLAC formats will not work.")
    else:
        logger.info("FFmpeg: Available")

    await asyncio.to_thread(setup_hf_auth)
    await asyncio.to_thread(model_manager.load, settings.model_load_timeout)
    await asyncio.to_thread(load_custom_voices)

    if has_voice_cloning():
        logger.info("Voice cloning: ENABLED")
    else:
        logger.info("Voice cloning: DISABLED (using preset voices only)")

    async def periodic_cleanup():
        while not shutdown_event.is_set():
            try:
                await asyncio.sleep(settings.cache_cleanup_interval)
                if not shutdown_event.is_set():
                    await asyncio.to_thread(cache_manager.cleanup)
                    await asyncio.to_thread(rate_limiter.cleanup)
            except asyncio.CancelledError:
                break

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


app = FastAPI(
    title="OpenAI-Compatible TTS API",
    description="OpenAI Audio Speech API compatible endpoint using Kyutai TTS",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    forwarded = request.headers.get("X-Forwarded-For")
    client_ip = forwarded.split(',')[0].strip() if forwarded else (request.client.host if request.client else "unknown")

    is_allowed, retry_after = rate_limiter.is_allowed(client_ip)
    if not is_allowed:
        return JSONResponse(
            content={"error": "Rate limit exceeded", "retry_after": retry_after},
            status_code=429,
            headers={"Retry-After": str(retry_after)}
        )

    response = await call_next(request)
    return response


@app.get("/v1/voices")
async def get_voices():
    with voice_lock:
        voices = set(DEFAULT_VOICES["openai_aliases"] + DEFAULT_VOICES["pocket_tts"])
        voices.update(VOICE_MAPPING.keys())
    return {"voices": sorted(list(voices))}


@app.get("/speakers")
async def get_speakers():
    """XTTS-compatible voice list endpoint (returns objects with name and voice_id)."""
    with voice_lock:
        voices = set(DEFAULT_VOICES["openai_aliases"] + DEFAULT_VOICES["pocket_tts"])
        voices.update(VOICE_MAPPING.keys())
    return [{"name": v, "voice_id": v} for v in sorted(voices)]


@app.get("/v1/formats")
async def get_formats():
    return {"formats": sorted(list(MEDIA_TYPES.keys()))}


@app.post("/v1/audio/speech")
async def text_to_speech(data: SpeechRequest, background_tasks: BackgroundTasks) -> StreamingResponse:
    try:
        logger.info(f"TTS request: voice='{data.voice}', format='{data.response_format}', len={len(data.input)}")

        v = data.voice
        is_path_voice = os.path.isabs(v) and os.path.isfile(v)
        if not is_path_voice:
            with voice_lock:
                known = is_valid_voice_name(v) or v.lower() in VOICE_MAPPING or v in VOICE_MAPPING
            if not known:
                raise HTTPException(status_code=400, detail="Invalid voice name")

        return StreamingResponse(
            generate_audio(
                text=data.input, voice=data.voice, speed=data.speed,
                format=data.response_format, temperature=data.temperature,
                lsd_decode_steps=data.lsd_decode_steps, top_p=data.top_p,
                repetition_penalty=data.repetition_penalty, model_tier=data.model,
                stream=data.stream, background_tasks=background_tasks,
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
    import torch
    import soundfile as sf
    import safetensors.torch
    from pocket_tts.data.audio_utils import convert_audio
    from .voices import load_custom_voices

    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="TTS model not loaded")

    voice_name = request.voice
    if not is_valid_voice_name(voice_name):
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
        def _export():
            audio, sr = sf.read(wav_path)
            audio_pt = torch.from_numpy(audio).float()
            if len(audio_pt.shape) == 1:
                audio_pt = audio_pt.unsqueeze(0)
            if request.truncate:
                max_samples = int(30 * sr)
                if audio_pt.shape[-1] > max_samples:
                    audio_pt = audio_pt[..., :max_samples]

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

        await asyncio.to_thread(_export)
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
    return {
        "status": "ok",
        "version": app.version,
        "model_loaded": model_manager.is_loaded,
        "device": model_manager.device,
        "sample_rate": model_manager.sample_rate,
        "voice_cloning": has_voice_cloning(),
        "hf_authenticated": check_hf_auth(),
        "cache_files": len([f for f in os.listdir(settings.audio_cache_dir)
                           if f.endswith(CACHE_EXTENSIONS)]) if os.path.exists(settings.audio_cache_dir) else 0,
    }


@app.post("/cache/clear")
async def clear_cache():
    """Delete all cached audio files."""
    cache_dir = Path(settings.audio_cache_dir)
    if not cache_dir.exists():
        return {"status": "ok", "deleted": 0}
    deleted = 0
    for f in cache_dir.iterdir():
        if f.is_file():
            try:
                f.unlink()
                deleted += 1
            except OSError:
                pass
    return {"status": "ok", "deleted": deleted}


@app.websocket("/v1/audio/stream")
async def websocket_tts(websocket: WebSocket):
    """WebSocket endpoint for TTS streaming with sentence merging.

    Protocol:
      Client -> Server: {"type": "text.append", "text": "sentence", "voice": "nova", "format": "mp3", "request_id": "r0", "speed": 1.0}
      Client -> Server: {"type": "text.append", "text": "sentence2", "voice": "nova", "format": "mp3", "request_id": "r1", "speed": 1.0}
      Client -> Server: {"type": "text.done"}
      Server -> Client: binary audio chunks
      Server -> Client: {"status": "done", "audio_duration": 2.1, "gen_time": 1.05, "request_id": ["r0", "r1"]}
      Server -> Client: {"status": "session_ended"}
    """
    await websocket.accept()

    if not model_manager.is_loaded:
        await websocket.send_json({"error": "TTS model not loaded", "status": "error"})
        await websocket.close(code=1011)
        return

    # Bytes per second estimates for duration calculation
    # wav/pcm: 24kHz * 2 bytes = 48000 B/s; mp3: ~220kbps VBR (-q:a 0); opus/aac: 192k CBR (-b:a 192k)
    _bps = {"wav": 48000, "mp3": 27500, "opus": 24000, "aac": 24000, "flac": 32000, "pcm": 48000}

    try:
        # Merge queue — accumulates short sentences, flushed on threshold or text.done
        merge_queue = []  # [{text, request_id, voice, format, speed, temperature, top_p, model}]
        merge_queue_len = 0
        MIN_MERGE = 40

        request_count = 0
        chunk_count_total = 0
        gen_speed = 1.0
        text_done_received = False
        session_ended_sent = False
        flush_task = None
        last_text_time = time.time()
        merge_lock = asyncio.Lock()
        inflight_gens = 0  # number of in-flight audio generations (flush in progress)

        # Defaults for optional per-append params
        default_speed = 1.0
        default_temperature = settings.temperature
        default_top_p = settings.top_p
        default_repetition_penalty = settings.repetition_penalty
        default_lsd_decode_steps = settings.lsd_decode_steps
        default_model_tier = settings.model_tier

        async def try_send_session_ended():
            """Send session_ended exactly once when all text is processed and no audio is in-flight."""
            nonlocal session_ended_sent
            async with merge_lock:
                if session_ended_sent:
                    return
                if text_done_received and not merge_queue and inflight_gens == 0:
                    session_ended_sent = True
                    logger.debug("WS session ended")
                    await websocket.send_json({"status": "session_ended"})

        async def gen_audio(text, v, f, sp, temp, tp, mt):
            """Generate audio for normalized text. Returns (chunks, duration, gen_time)."""
            gen_chunks = []
            t0 = time.time()
            async for chunk in generate_audio(
                text=text, voice=v, speed=sp, format=f,
                temperature=temp, top_p=tp,
                repetition_penalty=default_repetition_penalty,
                lsd_decode_steps=default_lsd_decode_steps, model_tier=mt,
            ):
                gen_chunks.append(chunk)
            gen_time = time.time() - t0
            total_bytes = sum(len(c) for c in gen_chunks)
            bps_val = _bps.get(f, 16000) / max(sp, 0.5)
            audio_duration = total_bytes / bps_val
            return gen_chunks, audio_duration, gen_time

        async def send_audio(chunks, audio_duration, gen_time, req_ids):
            """Send audio chunks and done message."""
            nonlocal chunk_count_total, request_count, gen_speed
            for c in chunks:
                chunk_count_total += 1
                await websocket.send_bytes(c)
            await websocket.send_json({
                "status": "done",
                "audio_duration": round(audio_duration, 3),
                "gen_time": round(gen_time, 3),
                "request_id": req_ids,
            })
            label = f"merged {len(req_ids)}" if len(req_ids) > 1 else "single"
            logger.info(f"WS done: {audio_duration:.1f}s | {label} | ids={req_ids}")
            request_count += 1

            if gen_time > 0:
                current_speed = audio_duration / gen_time
                gen_speed = 0.3 * current_speed + 0.7 * gen_speed

        async def flush_merge_queue():
            """Generate and send audio for buffered short sentences."""
            nonlocal merge_queue_len, inflight_gens
            async with merge_lock:
                if not merge_queue:
                    return
                # Group by params — items with different params must be generated separately
                groups = {}
                for entry in merge_queue:
                    key = (entry["voice"], entry["format"], entry["speed"],
                           entry["temperature"], entry["top_p"], entry["model"])
                    groups.setdefault(key, []).append(entry)
                merge_queue.clear()
                merge_queue_len = 0
                inflight_gens += 1
            try:
                for (v, f, sp, temp, tp, mt), items in groups.items():
                    merged_text = "\n".join(e["text"] for e in items)
                    merged_ids = [e["request_id"] for e in items]
                    try:
                        chunks, dur, gen_t = await gen_audio(merged_text, v, f, sp, temp, tp, mt)
                        await send_audio(chunks, dur, gen_t, merged_ids)
                    except Exception:
                        for rid in merged_ids:
                            try:
                                await websocket.send_json({
                                    "error": "Generation failed", "status": "error",
                                    "request_id": rid,
                                })
                            except Exception:
                                break
                        raise
            finally:
                async with merge_lock:
                    inflight_gens -= 1
                await try_send_session_ended()

        async def inactivity_timeout():
            """Flush merge queue after inactivity timeout (adaptive)."""
            nonlocal flush_task
            try:
                timeout = 15.0 * max(0.5, min(2.0, gen_speed))
                logger.debug(f"WS inactivity timeout {timeout:.1f}s")
                await asyncio.sleep(timeout)
                if merge_queue:
                    logger.debug(f"WS timeout flush: {len(merge_queue)} items")
                    await flush_merge_queue()
                await try_send_session_ended()
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Timeout flush error: {e}")
            flush_task = None

        # Main message loop
        while True:
            try:
                data = await websocket.receive_json()
            except WebSocketDisconnect:
                break
            except Exception:
                break

            last_text_time = time.time()
            msg_type = data.get("type")

            if msg_type == "text.append":
                text = data.get("text", "")
                if not text.strip():
                    continue

                is_retry = data.get("retry", False)
                entry_voice = data.get("voice", "nova")
                entry_fmt = data.get("format", "mp3")
                entry_speed = float(data.get("speed", default_speed))
                entry_temperature = float(data.get("temperature", default_temperature))
                entry_top_p = float(data.get("top_p", default_top_p))
                entry_model = data.get("model", default_model_tier)
                entry_request_id = data.get("request_id")

                normalized = normalize_text(text)

                # Retry requests get priority — flush queue, generate immediately
                if is_retry:
                    if flush_task and not flush_task.done():
                        flush_task.cancel()
                        flush_task = None
                    await flush_merge_queue()
                    logger.info(f"WS retry: req_id={entry_request_id} | voice={entry_voice} | '{text[:50]}'")
                    chunks, dur, gen_t = await gen_audio(
                        normalized, entry_voice, entry_fmt, entry_speed,
                        entry_temperature, entry_top_p, entry_model,
                    )
                    await send_audio(chunks, dur, gen_t, [entry_request_id])
                    continue

                # If voice/format changed, flush previous queue
                if merge_queue and (
                    merge_queue[-1]["voice"] != entry_voice or
                    merge_queue[-1]["format"] != entry_fmt
                ):
                    await flush_merge_queue()

                # Cancel existing inactivity timeout
                if flush_task and not flush_task.done():
                    flush_task.cancel()
                    flush_task = None

                merge_queue.append({
                    "text": normalized,
                    "request_id": entry_request_id,
                    "voice": entry_voice,
                    "format": entry_fmt,
                    "speed": entry_speed,
                    "temperature": entry_temperature,
                    "top_p": entry_top_p,
                    "model": entry_model,
                })
                merge_queue_len += len(normalized)

                # Check merge threshold
                merge_threshold = 50 if any(
                    e["text"].count('"') % 2 != 0 for e in merge_queue
                ) else MIN_MERGE
                if merge_queue_len >= merge_threshold or len(merge_queue) >= 5:
                    await flush_merge_queue()

                # Start inactivity timeout if queue is not empty
                if merge_queue and not flush_task:
                    flush_task = asyncio.create_task(inactivity_timeout())

            elif msg_type == "text.done":
                text_done_received = True
                logger.debug(f"WS text.done: {len(merge_queue)} items in queue")
                if flush_task and not flush_task.done():
                    flush_task.cancel()
                    flush_task = None
                await flush_merge_queue()

            else:
                logger.warning(f"WS unknown message type: {msg_type}")

        # Disconnect — flush remaining
        if flush_task and not flush_task.done():
            flush_task.cancel()
        await flush_merge_queue()

    except WebSocketDisconnect:
        logger.debug("WebSocket client disconnected")
    except Exception as e:
        logger.warning(f"WebSocket error: {e}")

@app.websocket("/v1/realtime")
async def websocket_realtime_tts(websocket: WebSocket):
    """Real-time TTS streaming — OpenAI Realtime API style.

    Protocol (matches OpenAI Realtime API):
      Client → Server: {"type":"session.update","session":{"voice":"nova","format":"mp3","speed":1.0}}
      Client → Server: {"type":"input_text.append","text":"Hello world"}
      Client → Server: {"type":"input_text.append","text":" how are you"}
      Client → Server: {"type":"input_text.done"}
      Server → Client: binary audio chunk
      Server → Client: {"type":"response.audio.done"}
      Server → Client: binary audio chunk
      Server → Client: {"type":"response.audio.done"}
      ...
      Server → Client: {"type":"response.done"}
    """
    await websocket.accept()

    if not model_manager.is_loaded:
        await websocket.send_json({"type": "error", "error": {"message": "TTS model not loaded"}})
        await websocket.close(code=1011)
        return

    try:
        # Session config
        data = await websocket.receive_json()
        session = data.get("session", {}) if data.get("type") == "session.update" else data
        voice = session.get("voice", "nova")
        fmt = session.get("format") or session.get("response_format", "mp3")
        speed = float(session.get("speed", 1.0))
        temperature = float(session.get("temperature", settings.temperature))
        top_p = float(session.get("top_p", settings.top_p))
        repetition_penalty = float(session.get("repetition_penalty", settings.repetition_penalty))
        lsd_decode_steps = int(session.get("lsd_decode_steps", settings.lsd_decode_steps))
        model_tier = session.get("model", settings.model_tier)

        buffer = ""

        async def process_sentence(sentence: str):
            sentence = sentence.strip()
            if not sentence:
                return
            logger.info(f"WS realtime: '{sentence[:60]}'")
            try:
                chunk_count = 0
                async for chunk in generate_audio(
                    text=sentence, voice=voice, speed=speed, format=fmt,
                    temperature=temperature, top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    lsd_decode_steps=lsd_decode_steps, model_tier=model_tier,
                ):
                    chunk_count += 1
                    await websocket.send_bytes(chunk)
                await websocket.send_json({"type": "response.audio.done"})
            except Exception as e:
                if "websocket" in str(e).lower() or "closed" in str(e).lower():
                    raise
                logger.warning(f"[Realtime] Generation error: {e}")
                await websocket.send_json({"type": "error", "error": {"message": str(e)}})

        async def drain_buffer():
            nonlocal buffer
            while True:
                match = re.search(r'([^.!?]*[.!?])\s*', buffer)
                if not match:
                    break
                sentence = match.group(1).strip()
                buffer = buffer[match.end():]
                if sentence:
                    await process_sentence(sentence)

        # Receive text incrementally
        while True:
            try:
                data = await websocket.receive_json()
            except WebSocketDisconnect:
                break
            except Exception:
                break

            msg_type = data.get("type", "")

            if msg_type == "input_text.append":
                text = data.get("text", "")
                if text:
                    buffer += normalize_text(text)
                    await drain_buffer()
            elif msg_type == "input_text.done":
                break
            elif msg_type == "session.update":
                session = data.get("session", {})
                voice = session.get("voice", voice)
                fmt = session.get("format", fmt)
                speed = float(session.get("speed", speed))
                temperature = float(session.get("temperature", temperature))
                top_p = float(session.get("top_p", top_p))
                repetition_penalty = float(session.get("repetition_penalty", repetition_penalty))
                lsd_decode_steps = int(session.get("lsd_decode_steps", lsd_decode_steps))
                model_tier = session.get("model", model_tier)

        # Flush remaining buffer
        if buffer.strip():
            await process_sentence(buffer.strip())

        await websocket.send_json({"type": "response.done"})

    except WebSocketDisconnect:
        logger.debug("[Realtime] Client disconnected")
    except Exception as e:
        logger.warning(f"[Realtime] Error: {e}")
        try:
            await websocket.send_json({"type": "error", "error": {"message": str(e)}})
        except Exception:
            pass


# ============================================================================
# XTTS-COMPATIBLE ENDPOINTS (GET streaming + POST)
# ============================================================================

@app.get("/tts_stream")
@app.get("/v1/audio/speech/tts_stream")
async def xtts_stream_get(
    text: str = "",
    speaker_wav: str = "nova",
    language: str = "en",
    voice: str = "",
    format: str = "mp3",
    speed: float = 1.0,
) -> StreamingResponse:
    """XTTS-compatible GET streaming endpoint. Defaults to MP3 for browser playback."""
    v = voice or speaker_wav
    if not v:
        v = "nova"
    if not text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    return StreamingResponse(
        generate_audio(text=text, voice=v, speed=speed, format=format),
        media_type=MEDIA_TYPES.get(format, "audio/mpeg"),
        headers={"Transfer-Encoding": "chunked", "X-Accel-Buffering": "no"},
    )


@app.post("/tts_to_audio")
@app.post("/tts_to_audio/")
@app.post("/v1/audio/speech/tts_to_audio")
async def xtts_post(request: Request) -> StreamingResponse:
    """XTTS-compatible POST endpoint."""
    data = await request.json()
    text = data.get("text", "")
    voice = data.get("speaker_wav") or data.get("voice", "nova")
    language = data.get("language", "en")
    speed = float(data.get("speed", 1.0))
    fmt = data.get("format", "wav")
    if not text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    return StreamingResponse(
        generate_audio(text=text, voice=voice, speed=speed, format=fmt),
        media_type=MEDIA_TYPES.get(fmt, "audio/wav"),
        headers={"Transfer-Encoding": "chunked", "X-Accel-Buffering": "no"},
    )


@app.post("/set_tts_settings")
@app.post("/v1/audio/speech/set_tts_settings")
@app.options("/set_tts_settings")
@app.options("/v1/audio/speech/set_tts_settings")
async def xtts_set_settings(request: Request):
    """XTTS-compatible settings endpoint (accepts and ignores params)."""
    return {"status": "ok"}


@app.get("/v1/audio/speech/speakers")
async def xtts_speakers_nested():
    """Voice list at nested path (for misconfigured endpoints)."""
    with voice_lock:
        voices = set(DEFAULT_VOICES["openai_aliases"] + DEFAULT_VOICES["pocket_tts"])
        voices.update(VOICE_MAPPING.keys())
    return [{"name": v, "voice_id": v} for v in sorted(voices)]


# ============================================================================
# MAIN
# ============================================================================

def main():
    from uvicorn.config import LOGGING_CONFIG as log_config
    log_config["formatters"]["default"]["fmt"] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_config["formatters"]["access"]["fmt"] = '%(asctime)s - %(client_addr)s - "%(request_line)s" %(status_code)s'

    override_port = os.environ.get("OVERRIDE_PORT")
    if override_port and override_port.isdigit():
        port = int(override_port)
    else:
        port = settings.server_port
    host = settings.server_host

    logger.info(f"Server binding to: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_config=log_config, access_log=True, ws_ping_interval=30, ws_ping_timeout=120)
