# Changelog

All notable changes to the PocketTTS OpenAPI project will be documented in this file.

## [2.0.0] - 2026-05-04

### Added
- **Multi-Language Support**: Full integration with `pocket-tts` version >= 2.0.0, natively enabling the use of multiple language models (e.g., `english`, `french_24l`, `spanish_24l`, etc.).
- **Dynamic Language & Device Variant Loading**: The server now supports dynamically loading and unloading different language models via the OpenAI-compatible `model` parameter. You can specify both the language and the compute device (e.g., `french_24l-gpu`, `english-cpu`, etc.).
- **`/v1/models` Endpoint**: The server now implements the OpenAI standard `/v1/models` endpoint, dynamically fetching the actually installed language variants from the upstream `pocket-tts` library's `CONFIGS_DIR` and exposing them as selectable models to clients (like SillyTavern or LibreChat).
- **Global `[tts] language` setting**: Added a new configuration option to `config.ini` that defines the default language loaded by the server at startup (default is `english`).
- **Proxy Application Architecture Update**: Fully overhauled the bundled `ice-open-tts-test-proxy` tool (both GUI and CLI variants). The proxy now dynamically parses the upstream `/v1/models` endpoint, injecting a "Model" selection dropdown into the interface, and securely pipes the selected model configuration to the `OpenAITTSStreamingManager` and OpenAI POST payloads.

### Changed
- **Dependencies**: Bumped the required `pocket-tts` version to `>=2.0.0` in `pyproject.toml`.
- **API Schema Update**: Relaxed the `SpeechRequest.model` field schema from a strict literal array to a regular string to support the countless new language-device permutations.
- **Model Loading Lifecycle**: Overhauled the `ModelManager` to securely track the currently loaded language. Requests that mandate a different language will seamlessly swap out the model weights from VRAM/RAM automatically without a server restart.
- **Standardized Export Pipeline**: Upgraded the embeddings auto-export functionality (in `voices.py` and `api.py`) to utilize the new canonical `get_state_for_audio_prompt` and `export_model_state` upstream API methods, removing previous unreliable and brittle manual audio encoding logic.

### Fixed
- **VRAM Memory Leak & Hoarding**: Addressed an aggressive VRAM retention issue where PyTorch's caching allocator would hold onto GPU memory after a generation ended, or even when transferring a model to CPU mode. The server now forcefully triggers Python garbage collection (`gc.collect()`) and purges the CUDA cache (`torch.cuda.empty_cache()`) instantly upon completion of an audio stream block, on language hot-swaps, and on explicit device movements.
- **API Health Endpoint**: Resolved a crashing bug where the `/health` endpoint would throw a `TypeError` if a `torch.device` object was serialized by FastAPI. It is now safely stringified.
- **WAV Floating-Point Support**: Added a robust monkey-patch fallback mechanism to standard Python `wave` loading. If `wave` crashes with `unknown format: 3` (IEEE float WAVs), the server catches it and automatically delegates the file reading to `soundfile`, ensuring 32-bit and 24-bit WAV prompts load smoothly without modifying the upstream library.
- **Legacy Safetensors Injection Bug**: Fixed an issue where old-format (pre-2.0.0) embeddings would not generate speech correctly under the new library version. Legacy safetensors now have the new multi-language `bos_before_voice` token properly injected and patched during loading.

### Removed
- **HD Tiering Flag (`-hd`)**: Removed legacy "HD" audio configurations attached to model tier names (like `tts-1-hd`). Model variants are now strictly `<language>-<device>` to accommodate the diverse range of language options. Fallback behavior has been added for clients still requesting `tts-1` or `tts-1-hd` to ensure zero downtime.
- **Manual KV Cache Slicing Hooks**: Ripped out redundant internal monkey-patching for KV cache sequence expansion logic which has now been fully unified inside the core `pocket-tts` library.
