# Pocket TTS OpenAPI

**Fast, local, OpenAI-compatible Text-to-Speech server** powered by [Kyutai's Pocket TTS](https://kyutai.org/blog/2026-01-13-pocket-tts).

- 🚀 **Runs at 1.5x real-time** on older CPUs (tested on Haswell)
- 🎭 **Voice cloning support** - use your own `.wav` files
- ⚡ **Optimized Loading** - converts voices to `.safetensors` for instant startup
- 📦 **Audio caching** - instant response for repeated phrases
- 🛰️ **Streaming Support** - real-time audio generation (OpenAI compatible)
- 🛡️ **Stuttering Protection** - runs with High Priority to prevent choppiness under load
- 🌐 **OpenAI API compatible** - works with existing tools
- 🏡 **Perfect for Home Assistant** via [OpenAI TTS Component](https://github.com/sfortis/openai_tts)

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

## 🖥️ Ice Open TTS Proxy (GUI & AI Agent Bridge)

For a more robust desktop experience and seamless AI integration, use the built-in **Ice Open TTS Proxy**.

- 🎨 **Desktop GUI**: Clean interface with text input, voice selection, and playback controls.
- ⚡ **Live Mode**: Advanced streaming mode that speaks as you type with real-time setting sync.
- 🤖 **AI Agent Bridge**: Built-in OpenAI-compatible API server (Port 8181) for easy integration with tools like **SillyTavern**.

### Launching the Proxy
1. Ensure the main TTS server is running (Step 2 above).
2. Go to the `ice-open-tts-proxy/` directory.
3. **Windows**: Run `start_ice_gui.bat`
4. **Linux**: Run `./start_ice_gui.sh`

See **[AGENTS.md](file:///home/icefog/LLM/pocket-tts-openapi/ice-open-tts-proxy/AGENTS.md)** for detailed AI Agent integration guides.

---

**Server runs at** `http://localhost:8005` (or next available port)

## Features

### Built-in Voices
Use preset voices without any setup:
- **Pocket TTS**: `alba`, `marius`, `javert`, `jean`, `fantine`, `cosette`, `eponine`, `azelma`
- **OpenAI aliases**: `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`

### Voice Cloning & Embeddings

The server uses two folders to manage custom voices:
- `voices/`: Place your source `.wav` files here.
- `embeddings/`: Optimized `.safetensors` embeddings are stored here for instant loading.

#### Setup Authentication
To use custom voices, authenticate with HuggingFace:
1. **Accept License**: Visit https://huggingface.co/kyutai/pocket-tts
2. **Login locally**: `huggingface-cli login` (enter your token from HF settings)
3. **Restart** the server

#### Adding Custom Voices
1. Place `.wav` files in the `voices/` folder.
   - **Tip**: For best results, use a clean voice sample that is approximately **10 seconds** long.
2. Start the server. It will automatically convert WAVs to `.safetensors` in the `embeddings/` folder.
3. From then on, the voice will load nearly instantly from the embedding!
4. Use via API: `"voice": "filename"`

### Audio Quality & Performance
- **High Priority Mode**: On Windows, the server automatically runs as a High Priority process to ensure smooth audio even when the system is under heavy load (e.g., gaming).
- **Quality Parameters**: You can now control the output quality via the API:
  - `temperature`: Control diversity/naturalness (0.0 to 2.0, default 0.7).
  - `lsd_decode_steps`: Control quality (1 to 50, default 2). Higher is better but slower.
- **Large Block Handling**: The server automatically splits large text (over 500 characters) into sentences and streams them sequentially, avoiding "Maximum generation length" errors.
- **Model Tiers**: You can choose between different performance/quality trade-offs:
  - `tts-1`: Standard quality, fastest (default).
  - `tts-1-hd`: High definition quality (automatically increases decode steps).
  - `tts-1-cuda`: Standard quality using GPU acceleration.
  - `tts-1-hd-cuda`: High definition quality using GPU acceleration.
  > [!TIP]
  > You can set the default tier in `config.ini` under the `[tts]` section.

### Audio Caching
- Automatically caches generated files (default limit: 10).
- Cache keys include voice, text, and quality parameters (changing temperature/steps triggers fresh generation).
- Cache hit = instant response.

## API Documentation

### 1. Speech Generation (`/v1/audio/speech`)
OpenAI-compatible endpoint for generating speech.

**Example:**
```bash
curl http://localhost:8005/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello! This is high quality audio.",
    "voice": "nova",
    "response_format": "mp3",
    "speed": 1.0,
    "temperature": 0.5,
    "lsd_decode_steps": 4,
    "stream": true
  }' \
  --output test.mp3
```

**Supported formats:** `mp3`, `wav`, `opus`, `aac`, `flac`, `pcm`

## Troubleshooting

### Voice Cloning Not Working
- **Error: "401 Unauthorized"** → Need to authenticate (see above)

### Server Won't Start
- **Port conflict** → Server auto-selects next free port
- **Model download slow** → First run downloads ~236MB model
- Check console for error messages

## Technical Notes

- **Platform**: Windows and Linux (macOS should work but untested)
- **Dependencies**: Python 3.10+, FFmpeg (for MP3/AAC/etc encoding)
- **Windows MP3**: Uses `mp3_mf` encoder (MediaFoundation) with auto-resampling to 44.1kHz
- **Linux MP3**: Uses `libmp3lame` if available in your FFmpeg build
- **Cache location**: `./audio_cache/` (auto-limited to 10 files by default)
- **Model cache**: Default HuggingFace cache (`~/.cache/huggingface`)

## Feedback

Join my Discord: [https://discord.gg/2tJcWeMjFQ](https://discord.gg/2tJcWeMjFQ)
Or find me on the official SillyTavern Discord server.

Support me:
[Ko-fi](https://ko-fi.com/icefog72) • [Patreon](https://www.patreon.com/cw/IceFog72)
---
Inspired by [kyutai-tts-openai-api](https://github.com/dwain-barnes/kyutai-tts-openai-api)

