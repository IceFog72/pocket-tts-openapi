# AGENTS.md - AI Agent Integration Guide

## TTS Server

**URL:** `http://127.0.0.1:8005`

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Server status |
| GET | `/v1/voices` | Voice list (OpenAI format) |
| GET | `/speakers` | Voice list (XTTS format) |
| POST | `/v1/audio/speech` | Generate audio (OpenAI standard) |
| GET | `/tts_stream?text=...&voice=...` | Stream audio (GET, browser-playable) |
| POST | `/tts_to_audio/` | Generate audio (XTTS format) |
| WS | `/v1/audio/stream` | WebSocket streaming |

### Generate Audio (OpenAI)

```bash
curl -s -X POST http://127.0.0.1:8005/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello", "voice": "nova", "response_format": "mp3"}' \
  --output /tmp/hello.mp3
```

### Stream Audio (GET)

```bash
curl -s "http://127.0.0.1:8005/tts_stream?text=Hello&voice=nova&format=mp3" \
  --output /tmp/hello.mp3
```

### WebSocket (reusable connection)

```python
import asyncio, json, websockets

async def stream_tts():
    async with websockets.connect("ws://localhost:8005/v1/audio/stream") as ws:
        await ws.send(json.dumps({"text": "Hello", "voice": "nova", "format": "mp3"}))
        audio = b""
        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                audio += msg
            elif json.loads(msg).get("status") == "done":
                break

        # Second request on same connection
        await ws.send(json.dumps({"text": "World", "voice": "alloy"}))

asyncio.run(stream_tts())
```

## Proxy Server

**URL:** `http://127.0.0.1:8181`

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Proxy status |
| GET | `/voices` | Voice list |
| POST | `/speak` | Play audio |
| POST | `/v1/audio/speech` | Generate audio (OpenAI) |

### Speak (play audio)

```bash
curl -s -X POST http://127.0.0.1:8181/speak \
  -H "Content-Type: application/json" \
  -d '{"text": "Task completed", "voice": "nova"}'
```

### Python Usage

```python
import requests

def get_voices() -> list:
    r = requests.get("http://127.0.0.1:8181/voices", timeout=5)
    return r.json().get("voices", [])

def speak(text: str, voice: str = "nova") -> bool:
    r = requests.post("http://127.0.0.1:8181/speak",
                       json={"text": text, "voice": voice}, timeout=30)
    return r.json().get("status") == "success"
```

## Voices

| Alias | Native | Custom |
|-------|--------|--------|
| alloy, echo, fable, onyx, nova, shimmer | alba, marius, javert, jean, fantine, cosette, eponine, azelma | Aemeath, Carlotta |

## Parameters

| Parameter | Default | Range |
|-----------|---------|-------|
| `voice` | nova | any voice name |
| `speed` | 1.0 | 0.25 - 4.0 |
| `format` | wav/mp3 | wav, mp3, opus, flac |

## Error Handling

| Error | Cause | Fix |
|-------|-------|-----|
| Connection refused | Server not running | Start server |
| 400 | Empty text | Check input |
| 429 | Rate limited | Wait for Retry-After |
| 503 | Model not loaded | Wait for startup |
