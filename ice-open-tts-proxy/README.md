# Ice Open TTS Proxy

Middleware between apps/AI agents and Pocket TTS server with real-time streaming.

```
[AI Agent / App / GUI] → Ice Open TTS Proxy → Pocket TTS Server
                      (port 8181)              (port 8005)
```

## Features

- **GUI Version**: Desktop app with text input, voice selection, and advanced streaming
- **CLI Version**: Terminal-only API server (no GUI dependencies)
- **Smart Streaming**: Hear it spoken as you type (sentence-based for best quality)
- **GUI UX**: Standard shortcuts (Ctrl+A, Ctrl+Z/Y, Ctrl+C/V) and automatic paste-to-stream
- **Cross-platform**: Linux, macOS, Windows

## Quick Start

```bash
# 1. Install dependencies
./install.sh          # Linux/macOS
install.bat           # Windows

# 2. Start Pocket TTS server (if not running)
cd .. && ./start.sh

# 3. Start Ice Open TTS Proxy
./start_ice_gui.sh    # GUI version
./start_ice_cli.sh    # CLI server
```

## Two Versions

| Version | File | Requirements | Features |
|---------|------|--------------|----------|
| **GUI** | `ice_open_tts_proxy.py` | tkinter | Full GUI + Streaming + API |
| **CLI** | `ice_open_tts_proxy_cli.py` | None | API server only |

## Streaming Mode

Enable "Live Mode (speak as you type)" in the GUI:

1. Check the live mode checkbox.
2. Type words or paste text (Ctrl+V).
3. The app intelligently captures new text and sends it to the server in sentence-chunks.
4. Audio plays sequentially via an internal playback queue.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/speak` | POST | Generate and play speech |
| `/tts_and_play` | POST | Alias for /speak |
| `/voices` | GET | List available voices |
| `/health` | GET | Server health check |

### Request Format

```json
POST /speak
{
  "text": "Hello world",
  "voice": "nova",
  "speed": 1.0,
  "format": "wav"
}
```

### Response Format

```json
{
  "status": "success",
  "message": "Playing: Hello world...",
  "file": "/tmp/xxxxx.wav"
}
```

## CLI Usage

```bash
# Check server status
python ice_open_tts_proxy_cli.py --status

# List voices
python ice_open_tts_proxy_cli.py --voices

# Generate audio file
python ice_open_tts_proxy_cli.py --text "Hello" --voice nova --save output.mp3

# Send to proxy for playback
python ice_open_tts_proxy_cli.py --speak "Hello" --voice Carlotta

# Run as API server
python ice_open_tts_proxy_cli.py --server --port 5000
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_SERVER` | http://localhost:8005 | Pocket TTS server URL |
| `PROXY_SERVER` | http://127.0.0.1:8181 | Proxy server URL |

## Programmatic Usage

### Python Example

```python
import requests

# Generate and play speech
requests.post("http://127.0.0.1:8181/speak", json={
    "text": "Hello from my application!",
    "voice": "Carlotta",
    "speed": 1.0
})
```

### cURL Example

```bash
curl -X POST http://127.0.0.1:5000/speak \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello!", "voice": "nova"}'
```

## Available Voices

**Built-in:** alloy, echo, fable, onyx, nova, shimmer  
**Pocket TTS:** alba, marius, javert, jean, fantine, cosette, eponine, azelma  
**Custom:** Place `.wav` files in `voices/` folder of main TTS server

## Configuration

Settings are saved in `config.ini` in the local directory.

```ini
[proxy]
tts_server_url = http://localhost:8005
api_host = 127.0.0.1
api_port = 8181
default_voice = nova
speed = 1.0
format = wav
```

## Tests

```bash
# Run unit tests
python test_ice_open_tts_proxy.py -v

# Run integration tests (requires TTS server)
RUN_INTEGRATION_TESTS=1 python test_ice_open_tts_proxy.py -v
```

## Files

```
ice-open-tts-proxy/
├── ice_open_tts_proxy.py          # GUI version
├── ice_open_tts_proxy_cli.py      # CLI version
├── test_ice_open_tts_proxy.py     # Unit tests (42 tests)
├── install.sh                     # Linux/Mac installer
├── install.bat                    # Windows installer
├── start_ice_gui.sh               # Start GUI (Linux/Mac)
├── start_ice_gui.bat              # Start GUI (Windows)
├── start_ice_cli.sh               # Start CLI (Linux/Mac)
├── start_ice_cli.bat              # Start CLI (Windows)
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── AGENTS.md                      # AI agent integration guide
```

## Troubleshooting

**No tkinter?** GUI auto-falls back to API-only mode. Install with:
- Arch: `sudo pacman -S tk`
- Debian: `sudo apt-get install python3-tk`

**No audio?** Install: `pip install simpleaudio`

**Port conflict?** Scripts detect and ask for alternative port.
