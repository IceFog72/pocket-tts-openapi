# Pocket TTS OpenAPI

**Fast, local, multi-protocol Text-to-Speech server** powered by [Kyutai's Pocket TTS](https://kyutai.org/blog/2026-01-13-pocket-tts).

- 🚀 **Runs at 1.5x real-time** on older CPUs (tested on Haswell)
- 🎭 **Voice cloning support** - use your own `.wav` files
- ⚡ **Optimized Loading** - converts voices to `.safetensors` for instant startup
- 📦 **Audio caching** - instant response for repeated phrases
- 🛰️ **Streaming Support** - real-time audio generation
- 🛡️ **Stuttering Protection** - runs with High Priority to prevent choppiness under load
- 🌐 **Multi-protocol**: OpenAI standard, XTTS-compatible, WebSocket, and GET streaming
- 🤖 **SillyTavern ready** - works with XTTSv2 and OpenAI Compatible TTS providers

## Installation

### 1. Download Project
```bash
git clone https://github.com/IceFog72/pocket-tts-openapi
cd pocket-tts-openapi
```

### 2. Setup & Run

#### Windows
1. Run `install.bat` - sets up Python venv and installs dependencies
2. Run `start.bat` - starts the server (automatically sets **High Priority**)

#### Linux
1. Run `chmod +x install.sh start.sh update.sh` (first time only)
2. Run `./install.sh` - sets up Python venv and installs dependencies
3. Run `./start.sh` - starts the server

### 3. Updating
To get the latest version of the project:
- **Windows**: Run `update.bat`
- **Linux**: Run `./update.sh`

---

## API Endpoints

**Server:** `http://localhost:8005`

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Server status |
| POST | `/v1/audio/speech` | Generate audio (OpenAI standard) |
| GET | `/tts_stream?text=...&voice=...` | Stream audio via GET |
| POST | `/tts_to_audio/` | Generate audio (XTTS format) |
| GET | `/v1/voices` | Voice list (OpenAI format) |
| GET | `/speakers` | Voice list (XTTS format) |
| WS | `/v1/audio/stream` | WebSocket streaming |

### OpenAI Standard

```bash
curl http://localhost:8005/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello!", "voice": "nova", "response_format": "mp3"}' \
  --output hello.mp3
```

### XTTS-Compatible (GET Streaming)

```bash
curl "http://localhost:8005/tts_stream?text=Hello&voice=nova&format=mp3" \
  --output hello.mp3
```

### WebSocket

```python
import asyncio, json, websockets

async def stream_tts():
    async with websockets.connect("ws://localhost:8005/v1/audio/stream") as ws:
        await ws.send(json.dumps({"text": "Hello", "voice": "nova", "format": "mp3"}))
        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                with open("out.mp3", "ab") as f:
                    f.write(msg)
            elif json.loads(msg).get("status") == "done":
                break

asyncio.run(stream_tts())
```

---

## SillyTavern Integration

The server works with two SillyTavern providers:

| Provider | Set endpoint to | Voices auto-discovered |
|----------|----------------|----------------------|
| **XTTSv2** | `http://host:8005` | Yes |
| **Pocket TTS** | `http://host:8005` | Yes |

Just select the provider, set the URL, and the voices appear automatically.

---

## 🖥️ Ice Open TTS Proxy (GUI & AI Agent Bridge)

For a desktop experience and AI integration, use the **Ice Open TTS Proxy**.

- 🎨 **Desktop GUI**: Text input, voice selection, playback controls.
- ⚡ **Live Mode**: Speaks as you type with real-time setting sync.
- 🤖 **AI Agent Bridge**: OpenAI-compatible API server on port 8181.

### Launching the Proxy
1. Ensure the main TTS server is running (Step 2 above).
2. Go to the `ice-open-tts-test-proxy/` directory.
3. **Windows**: Run `start_ice_gui.bat`
4. **Linux**: Run `./start_ice_gui.sh`

See **[AGENTS.md](ice-open-tts-test-proxy/AGENTS.md)** for detailed AI Agent integration.

---

## Features

### Built-in Voices
- **Pocket TTS**: `alba`, `marius`, `javert`, `jean`, `fantine`, `cosette`, `eponine`, `azelma`
- **OpenAI aliases**: `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`
- **Custom**: place `.wav` files in `voices/` (auto-converted to `.safetensors`)

### Voice Cloning & Embeddings

- `voices/`: Place your source `.wav` files here (~10 seconds for best results).
- `embeddings/`: Optimized `.safetensors` are stored here for instant loading.

#### Setup Authentication
1. Accept license at https://huggingface.co/kyutai/pocket-tts
2. Login: `huggingface-cli login`
3. Restart the server

### Audio Quality & Performance
- **High Priority Mode**: Auto-runs as High Priority on Windows.
- **Quality Parameters**: `temperature` (0.0-2.0), `lsd_decode_steps` (1-50).
- **Large Block Handling**: Auto-splits long text into sentences.
- **Model Tiers**: `tts-1` (fast), `tts-1-hd` (quality), `tts-1-cuda`, `tts-1-hd-cuda`.

### Audio Caching
- Auto-caches generated files (default: 10).
- Cache includes voice, text, and quality parameters.
- Cache hit = instant response.

## Troubleshooting

- **401 Unauthorized** → Run `huggingface-cli login`
- **Port conflict** → Server auto-selects next free port
- **Slow first run** → Downloads ~236MB model

## Technical Notes

- **Platform**: Windows and Linux
- **Dependencies**: Python 3.10+, FFmpeg (for MP3/AAC/etc)
- **Cache**: `./audio_cache/`
- **Model cache**: `~/.cache/huggingface`

## Feedback

Discord: [https://discord.gg/2tJcWeMjFQ](https://discord.gg/2tJcWeMjFQ) • SillyTavern Discord

[Ko-fi](https://ko-fi.com/icefog72) • [Patreon](https://www.patreon.com/cw/IceFog72)

Inspired by [kyutai-tts-openai-api](https://github.com/dwain-barnes/kyutai-tts-openai-api)
