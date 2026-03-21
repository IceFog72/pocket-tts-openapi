# What's New

## Multi-Protocol TTS Server

The server now supports three TTS protocols simultaneously:

### OpenAI Standard
```
POST /v1/audio/speech          → {"input": "...", "voice": "nova", "response_format": "mp3"}
```
Works with any OpenAI-compatible client.

### XTTS-Compatible
```
GET  /tts_stream?text=...&speaker_wav=nova&language=en    → streamed MP3
POST /tts_to_audio/           → {"text": "...", "speaker_wav": "nova"}
GET  /speakers                → [{"name": "nova", "voice_id": "nova"}, ...]
POST /set_tts_settings        → accepts params (no-op)
```
Works with SillyTavern's XTTSv2 provider out of the box.

### WebSocket (Agent Use)
```
ws://localhost:8005/v1/audio/stream
→ send {"text": "...", "voice": "nova"}
← receive binary chunks + {"status": "done"}
```
Reusable connection for multiple requests.

## SillyTavern Integration

Two providers work with the same server:

| Provider | Endpoint to set | Auto-discovers voices |
|----------|----------------|----------------------|
| XTTSv2 | `http://host:8005` | Yes (`/speakers`) |
| Pocket TTS (new) | `http://host:8005` | Yes (`/v1/voices`) |

No need to manually type voice names.

## Bug Fixes

- WAV streaming now defaults to MP3 for browser playback
- WAV writer cleanup no longer logs errors on client disconnect
- CORS accepts all origins (was rejecting LAN IPs)
- Nested routes added (`/v1/audio/speech/speakers`, etc.) for misconfigured endpoints
- `/speakers` returns objects with `name` and `voice_id` (XTTS format)
