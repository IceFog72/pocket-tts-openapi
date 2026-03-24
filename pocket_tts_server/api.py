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
from fastapi.responses import StreamingResponse

from .audio import generate_audio
from .cache import cache_manager
from .config import settings
from .constants import CACHE_EXTENSIONS, DEFAULT_VOICES, MEDIA_TYPES, VOICE_MAPPING
from .model_manager import model_manager
from .models import ExportVoiceRequest, SpeechRequest
from .rate_limiter import rate_limiter
from .validation import _ffmpeg_available, check_ffmpeg, is_valid_voice_name

logger = logging.getLogger(__name__)


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
        return StreamingResponse(
            content=iter([json.dumps({"error": "Rate limit exceeded", "retry_after": retry_after}).encode()]),
            status_code=429,
            media_type="application/json",
            headers={"Retry-After": str(retry_after)}
        )

    response = await call_next(request)
    return response


@app.get("/v1/voices")
async def get_voices():
    voices = set(DEFAULT_VOICES["openai_aliases"] + DEFAULT_VOICES["pocket_tts"])
    voices.update(VOICE_MAPPING.keys())
    return {"voices": sorted(list(voices))}


@app.get("/speakers")
async def get_speakers():
    """XTTS-compatible voice list endpoint (returns objects with name and voice_id)."""
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

        if not is_valid_voice_name(data.voice) and data.voice.lower() not in VOICE_MAPPING and data.voice not in VOICE_MAPPING:
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
    """WebSocket endpoint for real-time TTS streaming.

    Protocol:
      Client → Server: {"text": "Hello world", "voice": "nova", "format": "wav", "speed": 1.0}
      Server → Client: binary audio chunks
      Server → Client: {"status": "done", "audio_duration": 2.1, "gen_time": 1.05}

    Can be called multiple times per connection for sequential generations.
    """
    await websocket.accept()

    if not model_manager.is_loaded:
        await websocket.send_json({"error": "TTS model not loaded", "status": "error"})
        await websocket.close(code=1011)
        return

    # Bytes per second estimates for duration calculation
    _bps = {"wav": 48000, "mp3": 16000, "opus": 8000, "aac": 16000, "flac": 32000, "pcm": 48000}

    try:
        while True:
            data = await websocket.receive_json()

            text = data.get("input") or data.get("text", "")
            voice = data.get("voice", "alloy")
            fmt = data.get("format") or data.get("response_format", "wav")
            speed = float(data.get("speed", 1.0))
            temperature = float(data.get("temperature", settings.temperature))
            top_p = float(data.get("top_p", settings.top_p))
            repetition_penalty = float(data.get("repetition_penalty", settings.repetition_penalty))
            lsd_decode_steps = int(data.get("lsd_decode_steps", settings.lsd_decode_steps))
            model_tier = data.get("model", settings.model_tier)

            if not text.strip():
                await websocket.send_json({"error": "Empty text", "status": "error"})
                continue

            try:
                t0 = time.time()
                total_bytes = 0

                async for chunk in generate_audio(
                    text=text, voice=voice, speed=speed, format=fmt,
                    temperature=temperature, top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    lsd_decode_steps=lsd_decode_steps, model_tier=model_tier,
                ):
                    total_bytes += len(chunk)
                    await websocket.send_bytes(chunk)

                gen_time = time.time() - t0
                bps = _bps.get(fmt, 16000) / max(speed, 0.5)
                audio_duration = total_bytes / bps

                # Text stats
                sentences = [s.strip() for s in re.split(r'[.!?…]+', text) if s.strip()]
                words = text.split()

                await websocket.send_json({
                    "status": "done",
                    "audio_duration": round(audio_duration, 3),
                    "gen_time": round(gen_time, 3),
                })

                logger.info(
                    "WS: %d sentences, %d words, %d chars | %.1fs audio in %.1fs (%.2fx) | voice=%s",
                    len(sentences), len(words), len(text),
                    audio_duration, gen_time, audio_duration / max(gen_time, 0.01),
                    voice,
                )
            except Exception as e:
                logger.exception(f"WebSocket generation error")
                await websocket.send_json({"error": str(e), "status": "error"})

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
            logger.info(f"[Realtime] Generating: '{sentence[:60]}...'")
            try:
                async for chunk in generate_audio(
                    text=sentence, voice=voice, speed=speed, format=fmt,
                    temperature=temperature, top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    lsd_decode_steps=lsd_decode_steps, model_tier=model_tier,
                ):
                    await websocket.send_bytes(chunk)
                await websocket.send_json({"type": "response.audio.done"})
            except (RuntimeError, Exception) as e:
                if "websocket" in str(e).lower() or "closed" in str(e).lower():
                    raise
                logger.warning(f"[Realtime] Generation error: {e}")

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
            except Exception:
                break

            msg_type = data.get("type", "")

            if msg_type == "input_text.append":
                text = data.get("text", "")
                if text:
                    buffer += text
                    await drain_buffer()
            elif msg_type == "input_text.done":
                break
            elif msg_type == "session.update":
                session = data.get("session", {})
                voice = session.get("voice", voice)
                fmt = session.get("format", fmt)
                speed = float(session.get("speed", speed))

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
    uvicorn.run(app, host=host, port=port, log_config=log_config, access_log=True)
