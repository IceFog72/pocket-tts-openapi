import pytest
import asyncio
import hashlib
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from queue import Queue, Full
from unittest.mock import MagicMock, patch, AsyncMock, PropertyMock

from fastapi.testclient import TestClient

# Mock the TTS model and other heavy dependencies before importing the app
mock_tts_model = MagicMock()
mock_tts_model.device = "cpu"
mock_tts_model.sample_rate = 24000
mock_tts_model_class = MagicMock(return_value=mock_tts_model)

with patch('pocket_tts.TTSModel.load_model', return_value=mock_tts_model):
    with patch('pocket_tts.TTSModel', mock_tts_model_class):
        from pocketapi import (
            app, model_manager, settings, sanitize_text_input, is_valid_voice_name,
            ModelManager, RateLimiter, CacheManager, FileLikeQueueWriter,
            check_hf_auth, has_voice_cloning, check_ffmpeg, _ffmpeg_available,
            VOICE_MAPPING, DEFAULT_VOICES, FFMPEG_FORMATS, MEDIA_TYPES,
            CACHE_EXTENSIONS, Colors, load_custom_voices,
            SpeechRequest, ExportVoiceRequest,
        )

client = TestClient(app)


@pytest.fixture(autouse=True)
def mock_load_model():
    """Ensure the model is 'loaded' in the model manager without actually loading it."""
    with patch.object(model_manager, 'load', return_value=None):
        model_manager._model = mock_tts_model
        model_manager._device = "cpu"
        model_manager._sample_rate = 24000
        yield


# ============================================================================
# Constants Tests
# ============================================================================

class TestConstants:
    """Test module-level constants and mappings."""

    def test_voice_mapping_has_all_openai_aliases(self):
        expected = {"alloy", "echo", "fable", "onyx", "nova", "shimmer"}
        assert set(VOICE_MAPPING.keys()) == expected

    def test_voice_mapping_values_are_valid(self):
        for alias, voice in VOICE_MAPPING.items():
            assert isinstance(alias, str)
            assert isinstance(voice, str)
            assert len(alias) > 0
            assert len(voice) > 0

    def test_default_voices_structure(self):
        assert "openai_aliases" in DEFAULT_VOICES
        assert "pocket_tts" in DEFAULT_VOICES
        assert len(DEFAULT_VOICES["openai_aliases"]) == 6
        assert len(DEFAULT_VOICES["pocket_tts"]) == 8

    def test_ffmpeg_formats_complete(self):
        expected = {"mp3", "opus", "aac", "flac"}
        assert set(FFMPEG_FORMATS.keys()) == expected
        for fmt, (container, codec) in FFMPEG_FORMATS.items():
            assert isinstance(container, str)
            assert isinstance(codec, str)

    def test_media_types_complete(self):
        expected = {"mp3", "wav", "aac", "opus", "flac", "pcm"}
        assert set(MEDIA_TYPES.keys()) == expected

    def test_cache_extensions_includes_all_formats(self):
        for fmt in FFMPEG_FORMATS:
            assert f".{fmt}" in CACHE_EXTENSIONS
        assert ".wav" in CACHE_EXTENSIONS
        assert ".pcm" in CACHE_EXTENSIONS

    def test_colors_class_attributes(self):
        assert hasattr(Colors, 'CYAN')
        assert hasattr(Colors, 'GREEN')
        assert hasattr(Colors, 'YELLOW')
        assert hasattr(Colors, 'RED')
        assert hasattr(Colors, 'RESET')
        assert hasattr(Colors, 'BOLD')


# ============================================================================
# SpeechRequest Model Tests
# ============================================================================

class TestSpeechRequest:
    """Test the SpeechRequest Pydantic model."""

    def test_defaults(self):
        req = SpeechRequest(input="Hello world")
        assert req.model == "tts-1"
        assert req.voice == "alloy"
        assert req.response_format == "wav"
        assert req.speed == 1.0
        assert req.stream is True

    def test_all_fields_specified(self):
        req = SpeechRequest(
            model="tts-1-hd",
            input="Test text",
            voice="nova",
            response_format="mp3",
            speed=1.5,
            temperature=0.5,
            top_p=0.9,
            repetition_penalty=1.2,
            lsd_decode_steps=10,
            stream=False,
        )
        assert req.model == "tts-1-hd"
        assert req.input == "Test text"
        assert req.voice == "nova"
        assert req.response_format == "mp3"
        assert req.speed == 1.5
        assert req.temperature == 0.5
        assert req.top_p == 0.9
        assert req.repetition_penalty == 1.2
        assert req.lsd_decode_steps == 10

    def test_empty_input_rejected(self):
        with pytest.raises(Exception):
            SpeechRequest(input="")

    def test_input_max_length_rejected(self):
        with pytest.raises(Exception):
            SpeechRequest(input="a" * 4097)

    def test_speed_bounds(self):
        with pytest.raises(Exception):
            SpeechRequest(input="test", speed=0.1)
        with pytest.raises(Exception):
            SpeechRequest(input="test", speed=5.0)

    def test_temperature_bounds(self):
        with pytest.raises(Exception):
            SpeechRequest(input="test", temperature=-0.1)
        with pytest.raises(Exception):
            SpeechRequest(input="test", temperature=2.1)

    def test_top_p_bounds(self):
        with pytest.raises(Exception):
            SpeechRequest(input="test", top_p=0.05)
        with pytest.raises(Exception):
            SpeechRequest(input="test", top_p=1.1)

    def test_lsd_decode_steps_bounds(self):
        with pytest.raises(Exception):
            SpeechRequest(input="test", lsd_decode_steps=0)
        with pytest.raises(Exception):
            SpeechRequest(input="test", lsd_decode_steps=51)

    def test_invalid_model_rejected(self):
        with pytest.raises(Exception):
            SpeechRequest(input="test", model="invalid-model")

    def test_valid_models_accepted(self):
        for model in ["tts-1", "tts-1-hd", "tts-1-cuda", "tts-1-hd-cuda"]:
            req = SpeechRequest(input="test", model=model)
            assert req.model == model

    def test_valid_formats_accepted(self):
        for fmt in ["mp3", "opus", "aac", "flac", "wav", "pcm"]:
            req = SpeechRequest(input="test", response_format=fmt)
            assert req.response_format == fmt

    def test_voice_stripped(self):
        req = SpeechRequest(input="test", voice="  nova  ")
        assert req.voice == "nova"

    def test_empty_model_falls_back(self):
        req = SpeechRequest(input="test", model="")
        assert req.model == settings.model_tier


# ============================================================================
# ExportVoiceRequest Model Tests
# ============================================================================

class TestExportVoiceRequest:
    """Test the ExportVoiceRequest Pydantic model."""

    def test_defaults(self):
        req = ExportVoiceRequest(voice="my_voice")
        assert req.voice == "my_voice"
        assert req.truncate is False
        assert req.temperature == settings.temperature
        assert req.top_p == settings.top_p
        assert req.lsd_decode_steps == settings.lsd_decode_steps

    def test_truncate_option(self):
        req = ExportVoiceRequest(voice="test", truncate=True)
        assert req.truncate is True


# ============================================================================
# Input Validation Tests
# ============================================================================

class TestSanitizeTextInput:
    """Test text input sanitization."""

    def test_strips_whitespace(self):
        assert sanitize_text_input("  Hello  ") == "Hello"

    def test_normal_text_passthrough(self):
        assert sanitize_text_input("Hello world") == "Hello world"

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            sanitize_text_input("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="cannot be empty after trimming"):
            sanitize_text_input("   ")

    def test_too_long_raises(self):
        long = "a" * (settings.max_input_length + 1)
        with pytest.raises(ValueError, match="exceeds maximum length"):
            sanitize_text_input(long)

    def test_max_length_accepted(self):
        text = "a" * settings.max_input_length
        assert sanitize_text_input(text) == text

    def test_control_chars_stripped(self):
        text = "hello\x00\x01world"
        result = sanitize_text_input(text)
        assert "\x00" not in result
        assert "\x01" not in result
        assert "helloworld" == result

    def test_newline_preserved(self):
        text = "line1\nline2"
        assert sanitize_text_input(text) == text

    def test_tab_preserved(self):
        text = "col1\tcol2"
        assert sanitize_text_input(text) == text

    def test_unicode_preserved(self):
        text = "héllo wörld"
        assert sanitize_text_input(text) == text


class TestIsValidVoiceName:
    """Test voice name validation (path traversal prevention)."""

    def test_valid_names(self):
        assert is_valid_voice_name("alloy") is True
        assert is_valid_voice_name("my_voice") is True
        assert is_valid_voice_name("voice-123") is True
        assert is_valid_voice_name("VoiceName") is True

    def test_path_traversal_rejected(self):
        assert is_valid_voice_name("../etc/passwd") is False
        assert is_valid_voice_name("..\\windows") is False

    def test_absolute_path_rejected(self):
        assert is_valid_voice_name("/etc/passwd") is False
        assert is_valid_voice_name("C:\\Windows") is False

    def test_dotfile_rejected(self):
        assert is_valid_voice_name(".hidden") is False

    def test_special_chars_rejected(self):
        assert is_valid_voice_name("voice.wav") is False
        assert is_valid_voice_name("voice/name") is False
        assert is_valid_voice_name("voice\\name") is False
        assert is_valid_voice_name("voice name") is False
        assert is_valid_voice_name("voice@name") is False

    def test_empty_rejected(self):
        assert is_valid_voice_name("") is False

    def test_too_long_rejected(self):
        assert is_valid_voice_name("a" * 256) is False

    def test_max_length_accepted(self):
        assert is_valid_voice_name("a" * 255) is True


# ============================================================================
# ModelManager Tests
# ============================================================================

class TestModelManager:
    """Test the ModelManager class."""

    def test_initial_state(self):
        mgr = ModelManager()
        assert mgr.model is None
        assert mgr.device is None
        assert mgr.sample_rate == settings.default_sample_rate
        assert mgr.is_loaded is False

    def test_properties_after_load(self):
        mgr = ModelManager()
        mgr._model = mock_tts_model
        mgr._device = "cuda"
        mgr._sample_rate = 48000
        assert mgr.model is mock_tts_model
        assert mgr.device == "cuda"
        assert mgr.sample_rate == 48000
        assert mgr.is_loaded is True

    def test_shutdown_clears_state(self):
        mgr = ModelManager()
        mgr._model = mock_tts_model
        mgr._device = "cpu"
        mgr.shutdown()
        assert mgr.model is None
        assert mgr.device is None

    def test_shutdown_when_already_unloaded(self):
        mgr = ModelManager()
        mgr.shutdown()  # Should not raise
        assert mgr.model is None

    def test_lock_acquire_release(self):
        mgr = ModelManager()
        mgr.acquire_lock()
        assert mgr._lock.locked()
        mgr.release_lock()
        assert not mgr._lock.locked()

    def test_move_to_device_when_not_loaded(self):
        mgr = ModelManager()
        mgr.move_to_device("cuda")  # Should not raise, just return

    def test_move_to_device_same_device(self):
        mgr = ModelManager()
        mgr._model = mock_tts_model
        mgr._device = "cpu"
        mgr.move_to_device("cpu")
        mock_tts_model.to.assert_not_called()

    def test_move_to_device_different(self):
        mgr = ModelManager()
        mgr._model = mock_tts_model
        mgr._device = "cpu"
        mgr.move_to_device("cuda")
        mock_tts_model.to.assert_called_with("cuda")
        assert mgr._device == "cuda"


# ============================================================================
# RateLimiter Tests
# ============================================================================

class TestRateLimiter:
    """Test the RateLimiter class."""

    def test_allows_requests_under_limit(self):
        limiter = RateLimiter(requests_per_window=5, window_seconds=60)
        for _ in range(5):
            allowed, retry = limiter.is_allowed("127.0.0.1")
            assert allowed is True
            assert retry == 0

    def test_blocks_over_limit(self):
        limiter = RateLimiter(requests_per_window=2, window_seconds=60)
        limiter.is_allowed("127.0.0.1")
        limiter.is_allowed("127.0.0.1")
        allowed, retry = limiter.is_allowed("127.0.0.1")
        assert allowed is False
        assert retry > 0

    def test_separate_ips_independent(self):
        limiter = RateLimiter(requests_per_window=1, window_seconds=60)
        allowed1, _ = limiter.is_allowed("1.1.1.1")
        allowed2, _ = limiter.is_allowed("2.2.2.2")
        assert allowed1 is True
        assert allowed2 is True

    def test_expired_entries_allow_new_requests(self):
        limiter = RateLimiter(requests_per_window=1, window_seconds=1)
        limiter.is_allowed("127.0.0.1")
        time.sleep(1.1)
        allowed, _ = limiter.is_allowed("127.0.0.1")
        assert allowed is True

    def test_cleanup_removes_expired(self):
        limiter = RateLimiter(requests_per_window=5, window_seconds=1)
        limiter.is_allowed("1.1.1.1")
        limiter.is_allowed("2.2.2.2")
        time.sleep(1.1)
        limiter.cleanup()
        assert "1.1.1.1" not in limiter._requests
        assert "2.2.2.2" not in limiter._requests

    def test_cleanup_keeps_active(self):
        limiter = RateLimiter(requests_per_window=5, window_seconds=60)
        limiter.is_allowed("1.1.1.1")
        limiter.cleanup()
        assert "1.1.1.1" in limiter._requests

    def test_retry_after_minimum_one(self):
        limiter = RateLimiter(requests_per_window=1, window_seconds=1)
        limiter.is_allowed("127.0.0.1")
        time.sleep(0.9)
        _, retry = limiter.is_allowed("127.0.0.1")
        assert retry >= 1


# ============================================================================
# CacheManager Tests
# ============================================================================

class TestCacheManager:
    """Test the CacheManager class."""

    def test_generate_cache_key_deterministic(self):
        cm = CacheManager()
        key1 = cm.generate_cache_key("hello", "alloy", "wav", 1.0, 0.7, 2, 0.95, 1.1, "tts-1")
        key2 = cm.generate_cache_key("hello", "alloy", "wav", 1.0, 0.7, 2, 0.95, 1.1, "tts-1")
        assert key1 == key2

    def test_generate_cache_key_different_inputs(self):
        cm = CacheManager()
        key1 = cm.generate_cache_key("hello", "alloy", "wav", 1.0, 0.7, 2, 0.95, 1.1, "tts-1")
        key2 = cm.generate_cache_key("world", "alloy", "wav", 1.0, 0.7, 2, 0.95, 1.1, "tts-1")
        assert key1 != key2

    def test_generate_cache_key_is_sha256(self):
        cm = CacheManager()
        key = cm.generate_cache_key("test", "alloy", "wav", 1.0, 0.7, 2, 0.95, 1.1, "tts-1")
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    def test_get_cache_path(self):
        cm = CacheManager()
        cache_path, meta_path = cm.get_cache_path("abc123", "wav")
        assert cache_path.endswith("abc123.wav")
        assert meta_path.endswith("abc123.json")

    def test_should_cleanup_after_interval(self):
        cm = CacheManager()
        cm._last_cleanup = time.time() - settings.cache_cleanup_interval - 1
        assert cm.should_cleanup() is True

    def test_should_not_cleanup_within_interval(self):
        cm = CacheManager()
        cm._last_cleanup = time.time()
        assert cm.should_cleanup() is False

    def test_cleanup_empty_dir(self):
        cm = CacheManager()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(settings, 'audio_cache_dir', tmpdir):
                deleted = cm.cleanup()
                assert deleted == 0

    def test_cleanup_under_limit(self):
        cm = CacheManager()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(settings, 'audio_cache_dir', tmpdir):
                with patch.object(settings, 'cache_limit', 10):
                    for i in range(3):
                        Path(tmpdir, f"file{i}.wav").write_bytes(b"data")
                    deleted = cm.cleanup()
                    assert deleted == 0

    def test_cleanup_over_limit_removes_oldest(self):
        cm = CacheManager()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(settings, 'audio_cache_dir', tmpdir):
                with patch.object(settings, 'cache_limit', 2):
                    for i in range(5):
                        p = Path(tmpdir, f"file{i}.wav")
                        p.write_bytes(b"data")
                        time.sleep(0.01)  # Ensure different mtimes
                    deleted = cm.cleanup()
                    assert deleted == 3
                    remaining = list(Path(tmpdir).glob("*.wav"))
                    assert len(remaining) == 2

    def test_cleanup_removes_associated_json(self):
        cm = CacheManager()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(settings, 'audio_cache_dir', tmpdir):
                with patch.object(settings, 'cache_limit', 1):
                    Path(tmpdir, "old.wav").write_bytes(b"data")
                    Path(tmpdir, "old.json").write_bytes(b"meta")
                    time.sleep(0.01)
                    Path(tmpdir, "new.wav").write_bytes(b"data")
                    cm.cleanup()
                    assert not Path(tmpdir, "old.json").exists()
                    assert Path(tmpdir, "new.wav").exists()


# ============================================================================
# FileLikeQueueWriter Tests
# ============================================================================

class TestFileLikeQueueWriter:
    """Test the FileLikeQueueWriter class."""

    def test_write_returns_length(self):
        q = Queue(maxsize=10)
        writer = FileLikeQueueWriter(q, timeout=5.0)
        n = writer.write(b"hello")
        assert n == 5

    def test_write_empty_returns_zero(self):
        q = Queue(maxsize=10)
        writer = FileLikeQueueWriter(q, timeout=5.0)
        n = writer.write(b"")
        assert n == 0

    def test_tell_tracks_position(self):
        q = Queue(maxsize=10)
        writer = FileLikeQueueWriter(q, timeout=5.0)
        writer.write(b"abc")
        writer.write(b"def")
        assert writer.tell() == 6

    def test_close_sends_none(self):
        q = Queue(maxsize=10)
        writer = FileLikeQueueWriter(q, timeout=5.0)
        writer.close()
        assert q.get_nowait() is None

    def test_context_manager(self):
        q = Queue(maxsize=10)
        with FileLikeQueueWriter(q) as writer:
            writer.write(b"data")
        assert q.get_nowait() == b"data"
        assert q.get_nowait() is None

    def test_write_raises_on_abort(self):
        q = Queue(maxsize=1)
        q.put(b"fill")
        q.abort = True
        writer = FileLikeQueueWriter(q, timeout=0.1)
        with pytest.raises(IOError, match="aborted"):
            writer.write(b"data")

    def test_write_raises_on_timeout(self):
        q = Queue(maxsize=1)
        q.put(b"fill")
        writer = FileLikeQueueWriter(q, timeout=0.1)
        with pytest.raises(IOError, match="Queue full"):
            writer.write(b"data")

    def test_seek_is_noop(self):
        q = Queue(maxsize=10)
        writer = FileLikeQueueWriter(q, timeout=5.0)
        writer.seek(100)  # Should not raise

    def test_flush_is_noop(self):
        q = Queue(maxsize=10)
        writer = FileLikeQueueWriter(q, timeout=5.0)
        writer.flush()  # Should not raise


# ============================================================================
# FFmpeg Check Tests
# ============================================================================

class TestFFmpegCheck:
    """Test FFmpeg availability checking."""

    def test_check_ffmpeg_returns_bool(self):
        result = check_ffmpeg()
        assert isinstance(result, bool)

    def test_ffmpeg_unavailable_mock(self):
        with patch('pocketapi.subprocess.run', side_effect=FileNotFoundError):
            assert check_ffmpeg() is False

    def test_ffmpeg_timeout_mock(self):
        import subprocess
        with patch('pocketapi.subprocess.run', side_effect=subprocess.TimeoutExpired(cmd="ffmpeg", timeout=5)):
            assert check_ffmpeg() is False

    def test_ffmpeg_nonzero_exit(self):
        mock_result = MagicMock()
        mock_result.returncode = 1
        with patch('pocketapi.subprocess.run', return_value=mock_result):
            assert check_ffmpeg() is False

    def test_ffmpeg_available_mock(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch('pocketapi.subprocess.run', return_value=mock_result):
            assert check_ffmpeg() is True


# ============================================================================
# HuggingFace Auth Tests
# ============================================================================

class TestHuggingFaceAuth:
    """Test HuggingFace authentication functions."""

    def test_check_hf_auth_returns_bool(self):
        result = check_hf_auth()
        assert isinstance(result, bool)

    def test_check_hf_auth_import_error(self):
        with patch.dict('sys.modules', {'huggingface_hub': None}):
            # Re-import would fail, but function catches ImportError
            result = check_hf_auth()
            assert result is False

    def test_has_voice_cloning_false_when_no_model(self):
        with patch.object(model_manager, '_model', None):
            assert has_voice_cloning() is False

    def test_has_voice_cloning_true_when_supported(self):
        mock_model = MagicMock()
        mock_model.has_voice_cloning = True
        with patch.object(model_manager, '_model', mock_model):
            assert has_voice_cloning() is True

    def test_has_voice_cloning_false_when_not_supported(self):
        mock_model = MagicMock(spec=[])
        with patch.object(model_manager, '_model', mock_model):
            assert has_voice_cloning() is False


# ============================================================================
# Voice Loading Tests
# ============================================================================

class TestLoadCustomVoices:
    """Test custom voice loading."""

    def test_load_from_empty_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            embeddings_dir = os.path.join(tmpdir, "embeddings")
            voices_dir = os.path.join(tmpdir, "voices")
            os.makedirs(embeddings_dir)
            os.makedirs(voices_dir)
            with patch.object(settings, 'embeddings_dir', embeddings_dir):
                with patch.object(settings, 'voices_dir', voices_dir):
                    voices = load_custom_voices()
                    assert len(voices) == 0

    def test_load_safetensors_from_embeddings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            embeddings_dir = os.path.join(tmpdir, "embeddings")
            voices_dir = os.path.join(tmpdir, "voices")
            os.makedirs(embeddings_dir)
            os.makedirs(voices_dir)
            # Create a dummy safetensors file
            Path(embeddings_dir, "custom_voice.safetensors").write_bytes(b"dummy")
            original_mapping = dict(VOICE_MAPPING)
            try:
                with patch.object(settings, 'embeddings_dir', embeddings_dir):
                    with patch.object(settings, 'voices_dir', voices_dir):
                        voices = load_custom_voices()
                        assert "custom_voice" in voices
                        assert "custom_voice" in VOICE_MAPPING
            finally:
                VOICE_MAPPING.clear()
                VOICE_MAPPING.update(original_mapping)

    def test_load_wav_without_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            embeddings_dir = os.path.join(tmpdir, "embeddings")
            voices_dir = os.path.join(tmpdir, "voices")
            os.makedirs(embeddings_dir)
            os.makedirs(voices_dir)
            Path(voices_dir, "myvoice.wav").write_bytes(b"dummy")
            original_mapping = dict(VOICE_MAPPING)
            try:
                with patch.object(settings, 'embeddings_dir', embeddings_dir):
                    with patch.object(settings, 'voices_dir', voices_dir):
                        with patch.object(model_manager, '_model', None):
                            voices = load_custom_voices()
                            assert "myvoice" in voices
            finally:
                VOICE_MAPPING.clear()
                VOICE_MAPPING.update(original_mapping)


# ============================================================================
# API Endpoint Tests
# ============================================================================

class TestHealthEndpoint:
    """Test the /health endpoint."""

    def test_returns_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_includes_model_info(self):
        response = client.get("/health")
        data = response.json()
        assert "model_loaded" in data
        assert "device" in data
        assert "sample_rate" in data
        assert "voice_cloning" in data
        assert "hf_authenticated" in data
        assert "cache_files" in data

    def test_model_loaded_true(self):
        response = client.get("/health")
        data = response.json()
        assert data["model_loaded"] is True


class TestVoicesEndpoint:
    """Test the /v1/voices endpoint."""

    def test_returns_all_voices(self):
        response = client.get("/v1/voices")
        assert response.status_code == 200
        data = response.json()
        assert "voices" in data
        voices = data["voices"]
        for alias in DEFAULT_VOICES["openai_aliases"]:
            assert alias in voices
        for voice in DEFAULT_VOICES["pocket_tts"]:
            assert voice in voices

    def test_returns_sorted_list(self):
        response = client.get("/v1/voices")
        voices = response.json()["voices"]
        assert voices == sorted(voices)


class TestFormatsEndpoint:
    """Test the /v1/formats endpoint."""

    def test_returns_all_formats(self):
        response = client.get("/v1/formats")
        assert response.status_code == 200
        data = response.json()
        assert "formats" in data
        for fmt in MEDIA_TYPES:
            assert fmt in data["formats"]

    def test_returns_sorted_list(self):
        response = client.get("/v1/formats")
        formats = response.json()["formats"]
        assert formats == sorted(formats)


class TestTextToSpeechEndpoint:
    """Test the /v1/audio/speech endpoint."""

    @patch('pocketapi.generate_audio')
    def test_basic_request(self, mock_gen):
        async def dummy_gen(*args, **kwargs):
            yield b"audio_data"
        mock_gen.return_value = dummy_gen()

        response = client.post("/v1/audio/speech", json={
            "input": "Hello world",
            "voice": "alloy",
        })
        assert response.status_code == 200
        assert response.content == b"audio_data"

    @patch('pocketapi.generate_audio')
    def test_custom_voice(self, mock_gen):
        async def dummy_gen(*args, **kwargs):
            yield b"data"
        mock_gen.return_value = dummy_gen()

        response = client.post("/v1/audio/speech", json={
            "input": "Test",
            "voice": "nova",
        })
        assert response.status_code == 200

    @patch('pocketapi.generate_audio')
    def test_all_voices_accepted(self, mock_gen):
        async def dummy_gen(*args, **kwargs):
            yield b"data"
        mock_gen.return_value = dummy_gen()

        for voice in ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]:
            response = client.post("/v1/audio/speech", json={
                "input": "Test",
                "voice": voice,
            })
            assert response.status_code == 200, f"Voice {voice} failed"

    @patch('pocketapi.generate_audio')
    def test_all_formats_accepted(self, mock_gen):
        async def dummy_gen(*args, **kwargs):
            yield b"data"
        mock_gen.return_value = dummy_gen()

        for fmt in ["wav", "pcm"]:
            response = client.post("/v1/audio/speech", json={
                "input": "Test",
                "response_format": fmt,
            })
            assert response.status_code == 200, f"Format {fmt} failed"

    def test_empty_input_rejected(self):
        response = client.post("/v1/audio/speech", json={
            "input": "",
            "voice": "alloy",
        })
        assert response.status_code == 422

    def test_invalid_voice_rejected(self):
        response = client.post("/v1/audio/speech", json={
            "input": "test",
            "voice": "../invalid",
        })
        assert response.status_code == 400
        assert "Invalid voice name" in response.json()["detail"]

    def test_path_traversal_voice_rejected(self):
        response = client.post("/v1/audio/speech", json={
            "input": "test",
            "voice": "/etc/passwd",
        })
        assert response.status_code == 400

    def test_speed_too_low_rejected(self):
        response = client.post("/v1/audio/speech", json={
            "input": "test",
            "speed": 0.1,
        })
        assert response.status_code == 422

    def test_speed_too_high_rejected(self):
        response = client.post("/v1/audio/speech", json={
            "input": "test",
            "speed": 5.0,
        })
        assert response.status_code == 422

    @patch('pocketapi.generate_audio')
    def test_custom_parameters(self, mock_gen):
        async def dummy_gen(*args, **kwargs):
            yield b"data"
        mock_gen.return_value = dummy_gen()

        response = client.post("/v1/audio/speech", json={
            "input": "Test with custom params",
            "voice": "alloy",
            "speed": 1.5,
            "temperature": 0.5,
            "top_p": 0.8,
            "repetition_penalty": 1.2,
            "lsd_decode_steps": 4,
        })
        assert response.status_code == 200

    @patch('pocketapi.generate_audio')
    def test_response_headers(self, mock_gen):
        async def dummy_gen(*args, **kwargs):
            yield b"data"
        mock_gen.return_value = dummy_gen()

        response = client.post("/v1/audio/speech", json={"input": "test"})
        assert response.headers.get("Transfer-Encoding") == "chunked"
        assert response.headers.get("X-Accel-Buffering") == "no"
        assert response.headers.get("Cache-Control") == "no-cache"


# ============================================================================
# Rate Limiting Middleware Tests
# ============================================================================

class TestRateLimitMiddleware:
    """Test rate limiting middleware."""

    @patch('pocketapi.rate_limiter.is_allowed')
    def test_returns_429_when_limited(self, mock_allowed):
        mock_allowed.return_value = (False, 30)
        response = client.get("/v1/voices")
        assert response.status_code == 429
        assert response.headers["Retry-After"] == "30"
        data = response.json()
        assert "Rate limit exceeded" in data["error"]

    @patch('pocketapi.rate_limiter.is_allowed')
    def test_passes_when_allowed(self, mock_allowed):
        mock_allowed.return_value = (True, 0)
        response = client.get("/v1/voices")
        assert response.status_code == 200

    @patch('pocketapi.rate_limiter.is_allowed')
    def test_uses_forwarded_for_header(self, mock_allowed):
        mock_allowed.return_value = (True, 0)
        client.get("/v1/voices", headers={"X-Forwarded-For": "1.2.3.4, 5.6.7.8"})
        args = mock_allowed.call_args
        assert args[0][0] == "1.2.3.4"


# ============================================================================
# Export Voice Endpoint Tests
# ============================================================================

class TestExportVoiceEndpoint:
    """Test the /v1/audio/export-voice endpoint."""

    def test_model_not_loaded_returns_503(self):
        with patch.object(model_manager, '_model', None):
            response = client.post("/v1/audio/export-voice", json={
                "voice": "test",
            })
            assert response.status_code == 503

    def test_invalid_voice_rejected(self):
        response = client.post("/v1/audio/export-voice", json={
            "voice": "../etc/passwd",
        })
        assert response.status_code == 400

    def test_missing_wav_returns_404(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(settings, 'voices_dir', tmpdir):
                response = client.post("/v1/audio/export-voice", json={
                    "voice": "nonexistent",
                })
                assert response.status_code == 404


# ============================================================================
# Caching Integration Tests
# ============================================================================

class TestCachingIntegration:
    """Test caching behavior end-to-end."""

    @patch('os.path.exists')
    @patch('pocketapi.open_file', new_callable=AsyncMock)
    def test_cache_hit_returns_cached(self, mock_open, mock_exists):
        mock_exists.return_value = True
        mock_file = AsyncMock()
        mock_file.read.side_effect = [b"cached_audio", b""]
        mock_open.return_value.__aenter__.return_value = mock_file

        from pocketapi import generate_audio
        import asyncio

        async def run_gen():
            chunks = []
            async for chunk in generate_audio(text="Cached text", voice="alloy"):
                chunks.append(chunk)
            return b"".join(chunks)

        result = asyncio.run(run_gen())
        assert result == b"cached_audio"

    @patch('pocketapi.generate_audio')
    def test_different_params_different_cache(self, mock_gen):
        cm = CacheManager()
        key1 = cm.generate_cache_key("text", "alloy", "wav", 1.0, 0.7, 2, 0.95, 1.1, "tts-1")
        key2 = cm.generate_cache_key("text", "alloy", "wav", 1.0, 0.8, 2, 0.95, 1.1, "tts-1")
        assert key1 != key2


# ============================================================================
# Generate Audio Tests
# ============================================================================

class TestGenerateAudio:
    """Test the generate_audio function."""

    def test_raises_when_model_not_loaded(self):
        from pocketapi import generate_audio
        import asyncio

        async def run():
            async for _ in generate_audio(text="test", voice="alloy"):
                pass

        with patch.object(model_manager, '_model', None):
            with pytest.raises(Exception):
                asyncio.run(run())

    def test_raises_on_invalid_voice(self):
        from pocketapi import generate_audio
        import asyncio

        async def run():
            async for _ in generate_audio(text="test", voice="../bad"):
                pass

        with pytest.raises(Exception):
            asyncio.run(run())

    def test_raises_on_empty_text(self):
        from pocketapi import generate_audio
        import asyncio

        async def run():
            async for _ in generate_audio(text="", voice="alloy"):
                pass

        with pytest.raises(Exception):
            asyncio.run(run())

    @patch('pocketapi._ffmpeg_available', False)
    def test_raises_without_ffmpeg_for_mp3(self):
        from pocketapi import _generate_audio_core
        import asyncio

        async def run():
            async for _ in _generate_audio_core("test", "alloy", 1.0, "mp3", 4096):
                pass

        with pytest.raises(Exception, match="FFmpeg required"):
            asyncio.run(run())

    @patch('pocketapi._ffmpeg_available', False)
    def test_raises_without_ffmpeg_for_speed(self):
        from pocketapi import _generate_audio_core
        import asyncio

        async def run():
            async for _ in _generate_audio_core("test", "alloy", 1.5, "wav", 4096):
                pass

        with pytest.raises(Exception, match="FFmpeg required"):
            asyncio.run(run())
