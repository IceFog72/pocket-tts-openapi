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
from .validation import _ffmpeg_available, check_ffmpeg, is_valid_voice_name, normalize_text

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
        data.input = normalize_text(data.input)
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
    """WebSocket endpoint for real-time TTS streaming with sentence buffering.

    Protocol (enhanced):
      Client → Server: {"type": "session.update", "voice": "nova", "format": "mp3", "speed": 1.0}
      Client → Server: {"type": "text.append", "text": "Hello "}
      Client → Server: {"type": "text.append", "text": "world."}
      Server → Client: binary audio chunks
      Server → Client: {"status": "done", "audio_duration": 2.1, "gen_time": 1.05}
      Client → Server: {"type": "text.done"}  # optional, flushes remaining buffer

    Legacy protocol (backward compatible):
      Client → Server: {"text": "Hello world", "voice": "nova", "format": "wav", "speed": 1.0}
      Server → Client: binary audio chunks
      Server → Client: {"status": "done", "audio_duration": 2.1, "gen_time": 1.05}
    """
    await websocket.accept()

    if not model_manager.is_loaded:
        await websocket.send_json({"error": "TTS model not loaded", "status": "error"})
        await websocket.close(code=1011)
        return

    # Bytes per second estimates for duration calculation
    _bps = {"wav": 48000, "mp3": 16000, "opus": 8000, "aac": 16000, "flac": 32000, "pcm": 48000}

    # Sentence detection regex (same as realtime endpoint)
    SENTENCE_RE = re.compile(r'([^.!?]*[.!?])\s*')
    MAX_BUFFER_SIZE = 200  # characters

    try:
        # Session state
        voice = "nova"
        fmt = "mp3"
        speed = 1.0
        temperature = settings.temperature
        top_p = settings.top_p
        repetition_penalty = settings.repetition_penalty
        lsd_decode_steps = settings.lsd_decode_steps
        model_tier = settings.model_tier
        
        # Text buffer for sentence accumulation
        buffer = ""
        request_count = 0
        chunk_count_total = 0
        last_text_time = time.time()
        flush_task = None
        gen_speed = 1.0  # Start with assumption of real-time generation
        text_done_received = False
        
        async def flush_buffer_now(force=False):
            """Flush buffer immediately, splitting on punctuation only."""
            nonlocal buffer
            if not buffer.strip():
                return
                
            # Keep all characters for TTS - don't strip quotes
            # The TTS model can handle quotes and punctuation naturally
            
            print(f"[FLUSH] Buffer ({len(buffer)} chars): {buffer[:80]}")
            flush_text = buffer
            
            # Try to split at sentence-ending punctuation first
            sent_match = SENTENCE_RE.search(flush_text)
            if sent_match:
                # Found sentence ending, split there
                split_pos = sent_match.end()
                to_send = flush_text[:split_pos].strip()
                buffer = flush_text[split_pos:].strip()
                if to_send:
                    await generate_and_send_audio(to_send)
                return
            
            # No sentence ending, try to split at comma or colon
            for punct in [', ', ': ', '; ']:
                last_punct = flush_text.rfind(punct)
                if last_punct >= 20 and last_punct < len(flush_text) - 1:
                    to_send = flush_text[:last_punct + 1].strip()
                    buffer = flush_text[last_punct + 1:].strip()
                    print(f"[FLUSH] Split at '{punct.strip()}': '{to_send[:40]}...' / '{buffer[:40]}...'")
                    if to_send:
                        await generate_and_send_audio(to_send)
                    return
            
            # No punctuation found - send whole buffer
            print(f"[FLUSH] No punctuation, sending whole buffer: '{flush_text[:80]}'")
            if flush_text:
                await generate_and_send_audio(flush_text)
            buffer = ""
        
        async def flush_incomplete_after_timeout():
            """Flush buffer after inactivity timeout, adaptive based on generation speed."""
            nonlocal buffer, flush_task, gen_speed, text_done_received, websocket
            try:
                # Base timeout is 2 seconds, but adjust based on generation speed
                # If generating faster than real-time (speed > 1.0), we can wait longer
                # If generating slower (speed < 1.0), send sooner
                base_timeout = 2.0
                speed_factor = max(0.5, min(2.0, gen_speed))  # Clamp between 0.5x and 2.0x
                adaptive_timeout = base_timeout * speed_factor
                
                print(f"[TIMEOUT] Waiting {adaptive_timeout:.1f}s (gen_speed: {gen_speed:.2f}x)")
                await asyncio.sleep(adaptive_timeout)
                
                if buffer.strip():
                    await flush_buffer_now()
                
                # If text.done was received and buffer is now empty, send session_ended
                if text_done_received and not buffer.strip():
                    await websocket.send_json({"status": "session_ended"})
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Timeout flush error: {e}")
            flush_task = None
        
        async def generate_and_send_audio(sentence: str):
            """Generate audio for a complete sentence and send to client."""
            nonlocal request_count, chunk_count_total, gen_speed, text_done_received, buffer
            sentence = sentence.strip()
            if not sentence:
                return
                
            request_count += 1
            print(f"[SENTENCE] {sentence[:80]}")
            logger.info(f"WS sentence #{request_count}: {len(sentence)} chars, voice={voice}, format={fmt}, text='{sentence[:80]}...'")
            
            try:
                t0 = time.time()
                total_bytes = 0
                chunk_count = 0
                sentence_chunks = []  # collect for debug file save

                async for chunk in generate_audio(
                    text=sentence, voice=voice, speed=speed, format=fmt,
                    temperature=temperature, top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    lsd_decode_steps=lsd_decode_steps, model_tier=model_tier,
                ):
                    total_bytes += len(chunk)
                    chunk_count += 1
                    chunk_count_total += 1
                    sentence_chunks.append(chunk)
                    logger.info(f"WS sentence #{request_count} chunk #{chunk_count}: {len(chunk)} bytes, text='{sentence[:80]}...'")
                    await websocket.send_bytes(chunk)

                # Save debug audio file
                try:
                    debug_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'debug_audio')
                    os.makedirs(debug_dir, exist_ok=True)
                    ext = fmt if fmt in ('mp3', 'wav', 'opus', 'flac', 'aac') else 'mp3'
                    fname = f"{request_count:03d}_sentence.{ext}"
                    fpath = os.path.join(debug_dir, fname)
                    with open(fpath, 'wb') as f:
                        for c in sentence_chunks:
                            f.write(c)
                    print(f"[DEBUG] Saved {fname}: {total_bytes} bytes, text='{sentence[:60]}'")
                except Exception as e:
                    print(f"[DEBUG] Failed to save audio: {e}")

                gen_time = time.time() - t0
                bps = _bps.get(fmt, 16000) / max(speed, 0.5)
                audio_duration = total_bytes / bps
                
                # Update generation speed with exponential moving average
                if gen_time > 0:
                    current_speed = audio_duration / gen_time
                    # EMA with alpha=0.3 for smoothing
                    gen_speed = 0.3 * current_speed + 0.7 * gen_speed
                    print(f"[SPEED] Generation speed: {current_speed:.2f}x, EMA: {gen_speed:.2f}x")

                await websocket.send_json({
                    "status": "done",
                    "audio_duration": round(audio_duration, 3),
                    "gen_time": round(gen_time, 3),
                })

                logger.info(
                    "WS sentence #%d done: %d chars | %.1fs audio in %.1fs (%.2fx) | voice=%s",
                    request_count, len(sentence),
                    audio_duration, gen_time, audio_duration / max(gen_time, 0.01),
                    voice,
                )
                
                # If text.done was received and buffer is empty, send session_ended
                if text_done_received and not buffer.strip():
                    print(f"[SESSION] All text processed, sending session_ended")
                    await websocket.send_json({"status": "session_ended"})
            except Exception as e:
                logger.exception(f"WebSocket generation error (sentence #{request_count})")
                await websocket.send_json({"error": str(e), "status": "error"})

        async def drain_buffer():
            """Process complete sentences from buffer."""
            nonlocal buffer, last_text_time, flush_task, text_done_received, websocket
            # Cancel existing timeout
            if flush_task and not flush_task.done():
                flush_task.cancel()
                flush_task = None
            
            # Log buffer state for debugging
            print(f"[DRAIN] Buffer ({len(buffer)} chars): '{buffer[:80]}...'")
            
            while True:
                match = SENTENCE_RE.search(buffer)
                if not match:
                    break
                sentence = match.group(1).strip()
                buffer = buffer[match.end():]
                if sentence:
                    print(f"[DRAIN] Found sentence: '{sentence[:60]}...'")
                    await generate_and_send_audio(sentence)
            
            # If buffer is too large, log warning — only complete sentences are flushed above
            if len(buffer.strip()) >= MAX_BUFFER_SIZE:
                print(f"[DRAIN] Buffer large ({len(buffer)} chars) but no sentence ending yet, waiting...")
            
            # If text.done was received and buffer is empty, send session_ended
            if text_done_received and not buffer.strip():
                print(f"[SESSION] All text processed, sending session_ended")
                await websocket.send_json({"status": "session_ended"})
                return
            
            last_text_time = time.time()
            # Start timeout for any remaining text (will try to split at grammar points)
            if buffer.strip() and not flush_task:
                print(f"[DRAIN] Starting timeout for remaining {len(buffer)} chars")
                flush_task = asyncio.create_task(flush_incomplete_after_timeout())

        # Main message loop
        while True:
            try:
                data = await websocket.receive_json()
            except Exception:
                break

            # Update last text time
            last_text_time = time.time()
            
            # Check message type
            msg_type = data.get("type")
            
            if msg_type == "session.update":
                # Update session parameters
                session = data.get("session", {})
                voice = session.get("voice", voice)
                fmt = session.get("format", fmt)
                speed = float(session.get("speed", speed))
                temperature = float(session.get("temperature", temperature))
                top_p = float(session.get("top_p", top_p))
                repetition_penalty = float(session.get("repetition_penalty", repetition_penalty))
                lsd_decode_steps = int(session.get("lsd_decode_steps", lsd_decode_steps))
                model_tier = session.get("model", model_tier)
                
            elif msg_type == "text.append":
                # Append text to buffer
                text = data.get("text", "")
                if text:
                    buffer += normalize_text(text)
                    await drain_buffer()
                    
            elif msg_type == "text.done":
                # Mark that no more text is coming
                text_done_received = True
                print(f"[TEXT.DONE] {len(buffer)} chars remaining")
                # Check if we can send session_ended (buffer empty after all sentences extracted)
                await drain_buffer()
                
            else:
                # Legacy protocol: treat as immediate text
                text = data.get("input") or data.get("text", "")
                voice = data.get("voice", voice)
                fmt = data.get("format") or data.get("response_format", fmt)
                speed = float(data.get("speed", speed))
                temperature = float(data.get("temperature", temperature))
                top_p = float(data.get("top_p", top_p))
                repetition_penalty = float(data.get("repetition_penalty", repetition_penalty))
                lsd_decode_steps = int(data.get("lsd_decode_steps", lsd_decode_steps))
                model_tier = data.get("model", model_tier)
                text = normalize_text(text)

                if not text.strip():
                    await websocket.send_json({"error": "Empty text", "status": "error"})
                    continue

                # Generate audio for complete text
                print(f"[LEGACY] {text[:80]}")
                logger.info(f"WS legacy #{request_count+1}: {len(text)} chars, voice={voice}, format={fmt}, text='{text[:80]}...'")
                
                try:
                    t0 = time.time()
                    total_bytes = 0
                    chunk_count = 0

                    async for chunk in generate_audio(
                        text=text, voice=voice, speed=speed, format=fmt,
                        temperature=temperature, top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        lsd_decode_steps=lsd_decode_steps, model_tier=model_tier,
                    ):
                        total_bytes += len(chunk)
                        chunk_count += 1
                        chunk_count_total += 1
                        print(f"[CHUNK] {chunk_count_total} {text[:80]}")
                        logger.info(f"WS legacy chunk #{chunk_count}: {len(chunk)} bytes, text='{text[:80]}...'")
                        await websocket.send_bytes(chunk)

                    gen_time = time.time() - t0
                    bps = _bps.get(fmt, 16000) / max(speed, 0.5)
                    audio_duration = total_bytes / bps

                    await websocket.send_json({
                        "status": "done",
                        "audio_duration": round(audio_duration, 3),
                        "gen_time": round(gen_time, 3),
                    })

                    logger.info(
                        "WS legacy done: %d chars | %.1fs audio in %.1fs (%.2fx) | voice=%s",
                        len(text), audio_duration, gen_time, audio_duration / max(gen_time, 0.01),
                        voice,
                    )
                    request_count += 1
                except Exception as e:
                    logger.exception(f"WebSocket generation error (legacy)")
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
            print(f"[REALTIME RECEIVED] {sentence}")
            logger.info(f"[Realtime] Generating: '{sentence[:60]}...'")
            try:
                chunk_count = 0
                async for chunk in generate_audio(
                    text=sentence, voice=voice, speed=speed, format=fmt,
                    temperature=temperature, top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    lsd_decode_steps=lsd_decode_steps, model_tier=model_tier,
                ):
                    chunk_count += 1
                    print(f"[REALTIME CHUNK] {chunk_count} {sentence[:80]}")
                    logger.info(f"[Realtime] chunk #{chunk_count}: {len(chunk)} bytes, sentence='{sentence[:80]}'")
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
                    buffer += normalize_text(text)
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
