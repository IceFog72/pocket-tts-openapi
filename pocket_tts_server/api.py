"""FastAPI app, middleware, and API endpoints."""
import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
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
    allow_origin_regex=settings.allowed_origin_regex,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
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


def main():
    log_config = uvicorn.config.LOGGING_CONFIG
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
