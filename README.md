# Pocket TTS OpenAPI

**Fast, local, OpenAI-compatible Text-to-Speech server** powered by [Kyutai's Pocket TTS](https://kyutai.org/blog/2026-01-13-pocket-tts).

- 🚀 **Runs at 1.5x real-time** on older CPUs (tested on Haswell)
- 🎭 **Voice cloning support** - use your own `.wav` files
- 📦 **Audio caching** - instant response for repeated phrases
- 🌐 **OpenAI API compatible** - works with existing tools
- 🏡 **Perfect for Home Assistant** via [OpenAI TTS Component](https://github.com/sfortis/openai_tts)

## Quick Start

### Windows
1. Run `install.bat` - sets up Python venv and installs dependencies
2. Run `start.bat` - starts the server

### Linux
1. Run `chmod +x install.sh start.sh` (first time only)
2. Run `./install.sh` - sets up Python venv and installs dependencies
3. Run `./start.sh` - starts the server

**Server runs at** `http://localhost:8001` (or next available port)

## Features

### Built-in Voices
Use preset voices without any setup:
- **Pocket TTS**: `alba`, `marius`, `javert`, `jean`, `fantine`, `cosette`, `eponine`, `azelma`
- **OpenAI aliases**: `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`

### Voice Cloning (Custom Voices)

#### Setup Authentication
To use custom voices, authenticate with HuggingFace:

1. **Accept License**: Visit https://huggingface.co/kyutai/pocket-tts
2. **Login locally**:
   ```bash
   # Windows
   .\venv\Scripts\activate
   huggingface-cli login
   
   # Linux
   source venv/bin/activate
   huggingface-cli login
   ```
   Enter your token from: https://huggingface.co/settings/tokens
3. **Restart** the server

#### Add Custom Voices
1. Place `.wav` files in the `voices/` folder
2. Restart server - you'll see: `🎤 Custom voices loaded: filename1, filename2`
3. Use via API: `"voice": "filename1"`

**Audio Format Requirements:**
- Mono or stereo `.wav` files
- Any sample rate (auto-converted to PCM if needed)
- Filenames become voice names (without `.wav` extension)

### Audio Caching
- Automatically caches last **10 generated files**
- Cache hit = instant response (no regeneration)
- Saves **both audio + JSON metadata** (text, voice, etc.)

### Smart Port Selection
Server automatically finds free port starting from **8001**. Check console output:
```
✅ Server binding to: http://0.0.0.0:8001
```

## Usage Example

```bash
curl http://localhost:8001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hello! Testing Pocket TTS.",
    "voice": "nova",
    "response_format": "mp3",
    "speed": 1.0
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
- **Cache location**: `./audio_cache/` (auto-limited to 10 files)
- **Model cache**: Default HuggingFace cache (`~/.cache/huggingface`)

---

Inspired by [kyutai-tts-openai-api](https://github.com/dwain-barnes/kyutai-tts-openai-api)