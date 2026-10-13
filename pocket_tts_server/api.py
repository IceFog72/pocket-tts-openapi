"""FastAPI app, middleware, and API endpoints."""
import asyncio
import json
import logging
import os
import re
import time
from importlib.metadata import version as package_version
from pathlib import Path
from contextlib import asynccontextmanager
from functools import wraps

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
from .rate_limiter import rate_limiter, websocket_capacity
from .validation import _ffmpeg_available, check_ffmpeg, is_valid_voice_name, normalize_text
from .voices import voice_lock

POCKET_TTS_VERSION = package_version("pocket-tts")

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


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
    return model_manager.voice_cloning


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

    app.state.hf_authenticated = await asyncio.to_thread(setup_hf_auth)
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
    # Uvicorn resolves forwarded headers only for its configured trusted peers.
    # Do not trust a raw client-supplied header here.
    client_ip = request.client.host if request.client else "unknown"

    is_allowed, retry_after = rate_limiter.is_allowed(client_ip)
    if not is_allowed:
        return JSONResponse(
            content={"error": "Rate limit exceeded", "retry_after": retry_after},
            status_code=429,
            headers={"Retry-After": str(retry_after)}
        )

    response = await call_next(request)
    return response


@app.get("/v1/models")
@app.get("/models")
async def list_models():
    """List available TTS models (languages with CPU/GPU variants)."""
    try:
        from pocket_tts.utils.config import CONFIGS_DIR
        known_languages = [f.stem for f in CONFIGS_DIR.glob("*.yaml")]
    except Exception:
        known_languages = [
            "english", "french", "german", "spanish",
            "portuguese", "italian", "dutch"
        ]
        
    models = []
    created_time = int(time.time())
    for lang in sorted(known_languages):
        models.append({"id": f"{lang}-cpu", "object": "model", "created": created_time, "owned_by": "pocket-tts"})
        models.append({"id": f"{lang}-gpu", "object": "model", "created": created_time, "owned_by": "pocket-tts"})
    return {"object": "list", "data": models}

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


class AudioStreamingResponse(StreamingResponse):
    """Prime generation before headers and always close its owned iterator."""

    def __init__(self, source, first, **kwargs):
        self.source = source

        async def body():
            yield first
            async for chunk in source:
                yield chunk

        super().__init__(body(), **kwargs)

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self.source.aclose()


async def audio_response(source, request: Request = None, **kwargs):
    # Validation and admission errors must reach HTTP before success headers.
    # Watch disconnects while priming too, including waits for model ownership.
    async def disconnected():
        while True:
            if (await request.receive())["type"] == "http.disconnect":
                return

    first_task = asyncio.create_task(anext(source))
    disconnect_task = asyncio.create_task(disconnected()) if request is not None else None
    try:
        if disconnect_task is not None:
            done, _ = await asyncio.wait((first_task, disconnect_task),
                                         return_when=asyncio.FIRST_COMPLETED)
            if disconnect_task in done:
                disconnect_task.result()
                raise HTTPException(status_code=499, detail="Client disconnected")
        first = await first_task
    except BaseException as exc:
        first_task.cancel()
        await asyncio.gather(first_task, return_exceptions=True)
        await source.aclose()
        if isinstance(exc, StopAsyncIteration):
            raise HTTPException(status_code=500, detail="Synthesis produced no audio") from exc
        raise
    finally:
        if disconnect_task is not None:
            disconnect_task.cancel()
            await asyncio.gather(disconnect_task, return_exceptions=True)
    return AudioStreamingResponse(source, first, **kwargs)


@app.post("/v1/audio/speech")
async def text_to_speech(data: SpeechRequest, background_tasks: BackgroundTasks, request: Request) -> StreamingResponse:
    try:
        logger.info(f"TTS request: voice='{data.voice}', format='{data.response_format}', len={len(data.input)}")

        if not model_manager.is_loaded:
            raise HTTPException(status_code=503, detail="TTS model not loaded")

        v = data.voice
        is_path_voice = os.path.isabs(v) and os.path.isfile(v)
        if not is_path_voice:
            with voice_lock:
                known = is_valid_voice_name(v) or v.lower() in VOICE_MAPPING or v in VOICE_MAPPING
            if not known:
                raise HTTPException(status_code=400, detail="Invalid voice name")

        return await audio_response(
            generate_audio(
                text=data.input, voice=data.voice, speed=data.speed,
                format=data.response_format, temperature=data.temperature,
                lsd_decode_steps=data.lsd_decode_steps, model_tier=data.model,
                stream=data.stream, background_tasks=background_tasks,
            ),
            request=request,
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
    from pocket_tts import export_model_state
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
            model_manager.acquire_lock()
            try:
                tts_model = model_manager.model
                if tts_model is None:
                    raise RuntimeError("Model not loaded")
                with torch.no_grad():
                    tts_model.temp = request.temperature
                    tts_model.sampler_decode_steps = request.lsd_decode_steps
                    model_state = tts_model.get_state_for_audio_prompt(wav_path, truncate=request.truncate)
            finally:
                model_manager.release_lock()

            export_model_state(model_state, st_path)

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
def health():
    return {
        "status": "ok",
        "version": app.version,
        "pocket_tts_version": POCKET_TTS_VERSION,
        "language": model_manager.language,
        "temperature": settings.temperature,
        "model_loaded": model_manager.is_loaded,
        "device": str(model_manager.device) if model_manager.device else None,
        "sample_rate": model_manager.sample_rate,
        "voice_cloning": has_voice_cloning(),
        "hf_authenticated": getattr(app.state, "hf_authenticated", False),
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


def limited_websocket(handler):
    """Bound connected sockets, including their per-socket phrase queues."""
    @wraps(handler)
    async def wrapped(websocket: WebSocket):
        await websocket.accept()
        client_ip = websocket.client.host if websocket.client else "unknown"
        allowed, retry_after = rate_limiter.is_allowed(client_ip)
        if not allowed:
            await websocket.close(code=1013, reason=f"Rate limit exceeded; retry after {retry_after}s")
            return
        if not websocket_capacity.acquire():
            await websocket.close(code=1013, reason="WebSocket capacity exhausted")
            return
        try:
            await handler(websocket)
        finally:
            websocket_capacity.release()
    return wrapped


def check_websocket_rate(websocket: WebSocket):
    client_ip = websocket.client.host if websocket.client else "unknown"
    allowed, retry_after = rate_limiter.is_allowed(client_ip)
    if not allowed:
        raise RuntimeError(f"Rate limit exceeded; retry after {retry_after}s")


@app.websocket("/v1/audio/live")
@app.websocket("/v1/audio/stream")
@limited_websocket
async def websocket_tts(websocket: WebSocket):
    """Ordered requests. stream=true sends chunks as the model produces them."""
    if not model_manager.is_loaded:
        await websocket.send_json({"error": "TTS model not loaded", "status": "error"})
        await websocket.close(code=1011)
        return

    pending = asyncio.Queue(maxsize=32)

    async def synthesize():
        while True:
            data = await pending.get()
            if data.get("type") == "text.done":
                await websocket.send_json({"status": "session_ended"})
                continue
            request_id = data.get("request_id")
            try:
                request = SpeechRequest(
                    input=data.get("text", ""), voice=data.get("voice", "nova"),
                    model=data.get("model", "english-cpu"),
                    response_format=data.get("format", "mp3"),
                    speed=data.get("speed", 1.0),
                    temperature=data.get("temperature", settings.temperature),
                    top_p=data.get("top_p", settings.top_p),
                    lsd_decode_steps=data.get("lsd_decode_steps", settings.lsd_decode_steps),
                )
                streaming = data.get("stream", False)
                started = time.monotonic()
                first_audio = None
                total_bytes = 0
                chunks = []
                if streaming:
                    await websocket.send_json({
                        "status": "audio_start", "request_id": [request_id],
                        "format": request.response_format,
                        "sample_rate": model_manager.sample_rate, "channels": 1,
                    })
                async for chunk in generate_audio(
                    text=request.input, voice=request.voice, format=request.response_format,
                    speed=request.speed, temperature=request.temperature,
                    lsd_decode_steps=request.lsd_decode_steps,
                    model_tier=request.model, skip_cache=True,
                ):
                    if first_audio is None:
                        first_audio = time.monotonic() - started
                    total_bytes += len(chunk)
                    if streaming:
                        await websocket.send_bytes(chunk)
                    else:
                        chunks.append(chunk)
                if not total_bytes:
                    raise RuntimeError("Synthesis produced no audio")
                for chunk in chunks:
                    await websocket.send_bytes(chunk)
                byte_rate = {"pcm": model_manager.sample_rate * 2, "wav": model_manager.sample_rate * 2,
                             "mp3": 27500, "opus": 24000, "aac": 24000, "flac": 32000, "webm": 24000}
                audio_bytes = max(0, total_bytes - (44 if request.response_format == "wav" else 0))
                await websocket.send_json({
                    "status": "done", "request_id": [request_id],
                    "audio_duration": round(audio_bytes / byte_rate[request.response_format], 3),
                    "gen_time": round(time.monotonic() - started, 3),
                    "first_audio_ms": round(first_audio * 1000, 1) if first_audio is not None else None,
                })
            except Exception as exc:
                logger.warning("WebSocket synthesis failed: %s", exc)
                await websocket.send_json({"status": "error", "request_id": request_id,
                                           "error": str(exc)})

    worker = asyncio.create_task(synthesize())
    receiver = None
    text_buffer = None
    deadline = None

    def enqueue(data):
        if data.get("type") != "text.done":
            check_websocket_rate(websocket)
        if pending.full():
            raise RuntimeError("Synthesis queue full")
        pending.put_nowait(data)

    def flush_text():
        nonlocal text_buffer, deadline
        if text_buffer and text_buffer["text"].strip():
            enqueue(text_buffer)
        text_buffer = None
        deadline = None

    try:
        while True:
            if receiver is None:
                receiver = asyncio.create_task(websocket.receive_json())
            timeout = max(0, deadline - time.monotonic()) if deadline is not None else None
            done, _ = await asyncio.wait((receiver, worker), timeout=timeout,
                                        return_when=asyncio.FIRST_COMPLETED)
            if not done:
                flush_text()
                continue
            if worker in done:
                await worker
                break
            data = receiver.result()
            receiver = None
            kind = data.get("type")
            if kind == "text.delta" or (websocket.url.path.endswith("/live") and kind == "text.append"):
                fragment = data.get("text", "")
                if not isinstance(fragment, str):
                    raise ValueError("text must be a string")
                params = {key: value for key, value in data.items() if key not in ("text", "type")}
                if text_buffer and any(text_buffer.get(key) != value for key, value in params.items()):
                    flush_text()
                if text_buffer is None:
                    text_buffer = {**params, "type": "text.append", "text": "",
                                   "stream": True, "format": data.get("format", "pcm")}
                text_buffer["text"] += fragment
                deadline = time.monotonic() + 0.12
                if len(text_buffer["text"]) >= 120 or re.search(r"[.!?;:]\s*$", text_buffer["text"]):
                    flush_text()
            elif kind in ("text.append", "text.done"):
                flush_text()
                enqueue(data)
    except WebSocketDisconnect:
        pass
    except (RuntimeError, ValueError) as exc:
        await websocket.close(code=1013, reason=str(exc)[:100])
    finally:
        if receiver is not None:
            receiver.cancel()
        worker.cancel()
        await asyncio.gather(worker, *([receiver] if receiver else []), return_exceptions=True)

@app.websocket("/v1/realtime")
@limited_websocket
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
            check_websocket_rate(websocket)
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
                    if len(buffer) > settings.max_input_length:
                        raise ValueError("Text buffer exceeds maximum input length")
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
    request: Request,
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
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="TTS model not loaded")
    return await audio_response(
        generate_audio(text=text, voice=v, speed=speed, format=format),
        request=request,
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
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="TTS model not loaded")
    return await audio_response(
        generate_audio(text=text, voice=voice, speed=speed, format=fmt),
        request=request,
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
