# Pocket TTS OpenAPI

Local speech generation with [Kyutai Pocket TTS](https://github.com/kyutai-labs/pocket-tts), an OpenAI-compatible HTTP API, XTTS-compatible endpoints, and WebSocket streaming.

Uses Pocket TTS **3.3.x**, with **English** and **temperature 0.3** by default. Stable language names follow the default checkpoints shipped with the installed Pocket TTS version. CPU is the default device; CUDA is optional.

## Install and start

Install Python 3.10 or newer and FFmpeg. Python 3.12 is used in the current local setup. FFmpeg is needed for encoded formats and speed changes.

```bash
git clone https://github.com/IceFog72/pocket-tts-openapi
cd pocket-tts-openapi
bash install.sh
bash start.sh
```

On Windows, run `install.bat`, then `start.bat`.

The installer creates `venv` and installs the project. The first model load downloads weights from Hugging Face. For voice cloning, complete any access requirements on the [Pocket TTS model page](https://huggingface.co/kyutai/pocket-tts) and authenticate with Hugging Face if prompted.

The default address is `http://localhost:8005`. Settings live in `config.ini`. The default bind address is `0.0.0.0`, so the server also listens on network interfaces.

Check the running engine:

```bash
curl http://localhost:8005/health
```

The health response includes `pocket_tts_version`, `language`, `temperature`, `model_loaded`, and `sample_rate`. Its `version` field identifies the wrapper API, not the Pocket TTS engine.

## Choose a playback path

| Path | Use |
| --- | --- |
| WebSocket with PCM | Audio as it is generated, with buffering controlled by the client |
| OpenAI-compatible HTTP | Existing speech clients and audio downloads |
| XTTS-compatible HTTP | Clients expecting XTTS-style endpoints |

HTTP responses stream bytes, but a client may wait for a complete file before playback. For low startup latency in a browser, use WebSocket PCM and Web Audio. Raw PCM does not depend on a browser's MP3, WebM, or MediaSource support.

Adaptive buffering belongs to the playback client. The SillyTavern extension schedules incoming PCM with Web Audio and adjusts its buffer as audio arrives. Browser autoplay still requires a user gesture. Model loading, voice preparation, and generation speed affect time to first audio; streaming does not remove those costs.

## HTTP API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Engine and server status |
| GET | `/v1/models` | Available model identifiers |
| GET | `/v1/voices` | Available voices |
| GET | `/v1/formats` | Supported output formats |
| POST | `/v1/audio/speech` | OpenAI-compatible speech generation |
| GET | `/tts_stream` | XTTS-style streaming with query parameters |
| POST | `/tts_to_audio/` | XTTS-style speech generation |
| GET | `/speakers` | XTTS-style voice list |
| POST | `/v1/audio/export-voice` | Export a local WAV voice to a cached embedding |
| POST | `/cache/clear` | Clear generated audio files |

Generate a WAV file:

```bash
curl http://localhost:8005/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"english-cpu","input":"Hello there.","voice":"alba","response_format":"wav"}' \
  --output hello.wav
```

XTTS-style request:

```bash
curl 'http://localhost:8005/tts_stream?text=Hello&voice=alba&format=mp3' \
  --output hello.mp3
```

Output formats: `wav`, `pcm`, `mp3`, `opus`, `aac`, `flac`, and `webm`. PCM is signed 16-bit little-endian mono without a WAV header. The current engine outputs 24 kHz audio.

## WebSocket streaming

### Complete sentences

Connect to `ws://localhost:8005/v1/audio/stream`. Send each complete phrase with `stream: true` and an explicit format:

```json
{"type":"text.append","request_id":"r1","text":"Hello there.","voice":"alba","model":"english-cpu","format":"pcm","stream":true}
```

For each request, the server sends:

1. JSON with `status: "audio_start"`, `request_id: ["r1"]`, `format`, `sample_rate`, and `channels`.
2. Binary audio chunks in generation order.
3. JSON with `status: "done"`, the request ID array, `audio_duration`, `gen_time`, and `first_audio_ms`.

Use the sample rate from `audio_start` when decoding PCM. Network chunks can split a 16-bit sample; retain a trailing byte for the next chunk. Encoded-format durations are estimates; PCM duration comes from its byte count.

Send this after the last request in a reply:

```json
{"type":"text.done"}
```

The server sends `status: "session_ended"` after queued requests finish. The socket stays open for the next reply. Without `stream: true`, the regular endpoint collects the audio before sending it.

### Text arriving in fragments

Connect to `ws://localhost:8005/v1/audio/live`. Preserve spaces between fragments:

```json
{"type":"text.append","request_id":"r1","text":"Hello ","voice":"alba"}
{"type":"text.append","request_id":"r1","text":"there.","voice":"alba"}
{"type":"text.done"}
```

Live mode defaults to streaming PCM. It collects text until punctuation, 120 characters, or 120 ms of inactivity, then queues a phrase. `text.done` flushes the remainder. Each flushed phrase gets its own `audio_start` and `done`; multiple phrases can share the supplied request ID. The regular endpoint also accepts `text.delta` for this buffering behavior.

Requests are synthesized sequentially. The queue holds up to 32 phrases; overload closes the socket with code 1013. Disconnect cancels active delivery and discards queued requests. Synthesis failures produce `status: "error"` with an error message.

## Models and generation settings

Model identifiers use `<language>-<device>`, such as `english-cpu` or `french-gpu`. Query `/v1/models` for the variants installed on your server.

Stable languages are `english`, `french`, `german`, `spanish`, `portuguese`, `italian`, and `dutch`. Larger `*_24l` variants are optional. In Pocket TTS 3.3.0, the default English configuration uses the September 2026 checkpoint.

| Setting | Default | Purpose |
| --- | --- | --- |
| `temperature` | `0.3` | Sampling randomness; higher values do not reliably increase emotion |
| `lsd_decode_steps` | `2` | Legacy API name mapped to Pocket TTS's `sampler_decode_steps` |
| `speed` | `1.0` | Playback speed transformation; accepted range 0.25–4.0 |

Omit temperature to use the server default. Long text is split for generation. The default maximum input length is 4096 characters.

`top_p` and `repetition_penalty` remain accepted for older clients, but Pocket TTS 3.3 ignores them. They do not affect generation or create separate cache entries.

## Custom voices

Built-in voices: `alba`, `marius`, `javert`, `jean`, `fantine`, `cosette`, `eponine`, and `azelma`. OpenAI names `alloy`, `echo`, `fable`, `onyx`, `nova`, and `shimmer` are aliases for Pocket TTS voices.

Place a reference recording in `voices/`, for example `voices/Mona.wav`, then restart the server. Its filename stem becomes the voice ID: `Mona`. The server exports new WAV voices to `embeddings/Mona.safetensors` and reuses that file on later starts. Existing embeddings take precedence over matching WAV files.

### Recording format and length

Recommended preparation:

- WAV, signed 16-bit PCM, mono, 24 kHz. This matches the engine's processing format; higher-rate source recordings can be converted.
- Start with 10–15 seconds of actual speech. This is a practical starting point, not a required minimum or a guaranteed optimum.
- One speaker, a few complete sentences, and consistent delivery.
- No music, overlapping voices, clipping, strong echo, or long silence. Avoid heavy processing that damages speech.

Automatic export truncates references to the first 30 seconds. Choose a clean excerpt rather than supplying a long recording.

Convert a recording to a new file:

```bash
ffmpeg -i recording.flac -ac 1 -ar 24000 -c:a pcm_s16le voices/Mona.wav
```

To regenerate an embedding after replacing its source WAV:

```bash
curl http://localhost:8005/v1/audio/export-voice \
  -H 'Content-Type: application/json' \
  -d '{"voice":"Mona","truncate":true}'
```

Embeddings avoid re-encoding the reference recording on each use. They do not eliminate speech generation time. See the [upstream voice export documentation](https://kyutai-labs.github.io/pocket-tts/CLI%20Commands/export_voice/).

### Emotion and delivery

Pocket TTS conditions speech on the reference voice's style and prosody. It has no dedicated emotion parameter in the installed version. For selectable delivery, prepare separate references of the same speaker, such as `Mona-neutral.wav`, `Mona-warm.wav`, and `Mona-excited.wav`, then choose the corresponding voice ID.

Keep one consistent style per reference and compare generated samples. Emotion transfer varies; temperature is not an emotion-strength control. Text tags such as `[angry]` are not a supported control interface and may be read aloud. [Kyutai's technical report](https://kyutai.org/pocket-tts-technical-report/) describes reference conditioning.

## SillyTavern

The [PocketTTS WebSocket extension](https://github.com/IceFog72/SillyTavern-PocketTTS-WebSocket) connects to `http://host:8005` and uses a persistent socket for sentence playback.

Its streaming choices are **Auto** and **Off**. Auto prefers PCM with adaptive buffering; Off uses buffered audio. The playback bar supports pause/resume, volume, speed, and seeking within received audio. When another TTS provider is selected, the bar controls SillyTavern's audio element and uses its native queue controls for skip and stop.

For SillyTavern's built-in XTTSv2 provider, use `http://host:8005`. For an OpenAI-compatible provider, configure its speech URL as `http://host:8005/v1/audio/speech`, or its base URL as `http://host:8005/v1` if the client appends `/audio/speech` itself.

After updating the extension, reload SillyTavern to load its new scripts.

## Request limits

Limits in `config.ini` apply across clients:

- `rate_limit_requests = 30` per IP over `rate_limit_window = 60` seconds. HTTP requests, WebSocket connections, and submitted WebSocket phrases consume this budget. Live text fragments are charged when they become a phrase.
- `max_websocket_connections = 8` across both streaming routes and the legacy realtime route. Each regular/live socket queues at most 32 phrases.
- `max_pending_syntheses = 32` bounds active and waiting synthesis producers across HTTP and WebSocket requests. Excess work is rejected rather than creating more waiting threads.

Overloaded or rate-limited sockets close with code 1013. A client should wait before reconnecting. Limits can be increased for longer replies or multiple users.

Rate limiting uses the connection's resolved client address. Raw `X-Forwarded-For` headers do not override it. Behind a reverse proxy, configure Uvicorn's trusted proxy addresses with `FORWARDED_ALLOW_IPS`; trust only the actual proxy.

A generation or encoder failure does not publish a cache entry. Once HTTP audio bytes have started, a later failure terminates the response stream rather than changing its HTTP status. WebSocket requests report an error instead of successful completion.

## Storage and troubleshooting

| Location | Contents |
| --- | --- |
| `config.ini` | Server, generation, and cache settings |
| `voices/` | Source WAV references |
| `embeddings/` | Exported voice states |
| `audio_cache/` | Generated HTTP audio cache; default limit 10 files |
| Hugging Face cache | Downloaded model weights |

WebSocket requests bypass the generated-audio cache.

- **Model download or authorization fails:** check Hugging Face access, credentials, network availability, and startup logs.
- **First speech is slower:** model loading and initial voice preparation add work. Reuse the running server and cached voice embeddings.
- **Encoded audio or speed changes fail:** check that `ffmpeg` is available on `PATH`.
- **A replaced WAV sounds unchanged:** regenerate its embedding using the export endpoint.
- **Browser receives audio but stays silent:** allow playback with a click and check the selected provider and output device.
- **Port already occupied:** `start.sh` offers restart or a different port; check the actual listening address in its logs.

## Verification

Install test dependencies in the project environment and run the server tests:

```bash
venv/bin/python -m pip install pytest httpx
venv/bin/python -m pytest tests
```

Measure WebSocket time to first audio without playing sound:

```bash
venv/bin/python -m tests.benchmark_streaming
```

The benchmark starts a temporary server on a free local port, loads the model, measures three uncached requests, and shuts down. Pass `--voice /absolute/path/to/voice.safetensors` to test a prepared voice. Results depend on hardware, model, voice, and system load.

## Links

[Upstream Pocket TTS](https://github.com/kyutai-labs/pocket-tts) · [Discord](https://discord.gg/2tJcWeMjFQ) · [Ko-fi](https://ko-fi.com/icefog72)

Inspired by [kyutai-tts-openai-api](https://github.com/dwain-barnes/kyutai-tts-openai-api).
