# AGENTS.md - AI Agent Integration Guide

## Endpoints

**Proxy:** `http://127.0.0.1:8181`

### Health Check

```bash
curl -s http://127.0.0.1:8181/health
# {"status":"ok","tts_connected":true,"audio_playing":false}
```

### List Voices

```bash
curl -s http://127.0.0.1:8181/voices
# {"voices":["Aemeath","Carlotta","alba","alloy","azelma","cosette","echo","eponine","fable","fantine","javert","jean","marius","nova","onyx","shimmer"]}
```

### Speak (play audio)

```bash
curl -s -X POST http://127.0.0.1:8181/speak \
  -H "Content-Type: application/json" \
  -d '{"text": "Task completed", "voice": "nova"}'
# {"status":"success","message":"Playing: Task completed...","file":"/tmp/xxx.wav"}
```

### Generate Audio File (save to disk)

```bash
curl -s -X POST http://127.0.0.1:8181/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Report ready", "voice": "nova", "response_format": "mp3"}' \
  --output /tmp/report.mp3
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | required | Text to speak (max 4096 chars) |
| `voice` | string | from config | Voice name |
| `speed` | float | from config | 0.25 - 4.0 |
| `format` | string | from config | wav, mp3, opus, flac |

## Voices

Query `GET /voices` for the current list. Example: `nova, alloy, echo, fable, onyx, shimmer, alba, marius, ...`

## Python Usage

```python
import requests

PROXY = "http://127.0.0.1:8181"

def get_voices() -> list:
    """Get available voices from proxy."""
    try:
        r = requests.get(f"{PROXY}/voices", timeout=5)
        return r.json().get("voices", [])
    except:
        return []

def speak(text: str, voice: str = "nova") -> bool:
    try:
        r = requests.post(f"{PROXY}/speak", json={"text": text, "voice": voice}, timeout=30)
        return r.json().get("status") == "success"
    except:
        return False

# Usage
voices = get_voices()
speak("Starting build...")
speak("Build complete")
```

## Error Handling

| Error | Cause | Fix |
|-------|-------|-----|
| Connection refused | Proxy not running | Start proxy |
| 400 | Empty text | Check input |
| Response `"error"` field | TTS server down | Check TTS server |

## Bash Shortcut

```bash
speak() {
    curl -s -X POST http://127.0.0.1:8181/speak \
        -H "Content-Type: application/json" \
        -d "{\"text\": \"$1\", \"voice\": \"${2:-nova}\"}"
}
```
