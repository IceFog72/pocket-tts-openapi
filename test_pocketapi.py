import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch, AsyncMock
import json
import os
import io

# Mock the TTS model and other heavy dependencies before importing the app
# This prevents actual model loading during test discovery/execution
mock_tts_model = MagicMock()
mock_tts_model_class = MagicMock(return_value=mock_tts_model)

with patch('pocket_tts.TTSModel.load_model', return_value=mock_tts_model):
    with patch('pocket_tts.TTSModel', mock_tts_model_class):
        from pocketapi import app, model_manager, settings, sanitize_text_input, is_valid_voice_name

client = TestClient(app)

@pytest.fixture(autouse=True)
def mock_load_model():
    """Ensure the model is 'loaded' in the model manager without actually loading it."""
    with patch.object(model_manager, 'load', return_value=None):
        model_manager._model = mock_tts_model
        model_manager._device = "cpu"
        model_manager._sample_rate = 24000
        yield

def test_health_endpoint():
    """Test the healthcheck endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True

def test_get_voices():
    """Test retrieving the list of available voices."""
    response = client.get("/v1/voices")
    assert response.status_code == 200
    data = response.json()
    assert "voices" in data
    # Check for some default voices
    assert "alloy" in data["voices"]
    assert "nova" in data["voices"]

def test_get_formats():
    """Test retrieving supported audio formats."""
    response = client.get("/v1/formats")
    assert response.status_code == 200
    data = response.json()
    assert "formats" in data
    assert "wav" in data["formats"]
    assert "mp3" in data["formats"]

def test_sanitize_text_input():
    """Test text input sanitization logic."""
    # Valid input
    assert sanitize_text_input("  Hello World  ") == "Hello World"
    
    # Invalid: Too short
    with pytest.raises(ValueError, match="at least 1 characters"):
        sanitize_text_input("")
    
    # Invalid: Too long
    long_text = "a" * (settings.max_input_length + 1)
    with pytest.raises(ValueError, match="exceeds maximum length"):
        sanitize_text_input(long_text)

def test_is_valid_voice_name():
    """Test voice name validation (path traversal protection)."""
    assert is_valid_voice_name("valid_voice_123") is True
    assert is_valid_voice_name("voice-name") is True
    assert is_valid_voice_name("../secret") is False
    assert is_valid_voice_name("/etc/passwd") is False
    assert is_valid_voice_name("voice.wav") is False  # Only alphanumeric, underscore, hyphen allowed

@patch('pocketapi.generate_audio')
def test_text_to_speech_endpoint(mock_gen):
    """Test the main TTS endpoint."""
    # Mock the audio generator to yield some dummy bytes
    async def dummy_gen(*args, **kwargs):
        yield b"fake_audio_chunk"

    mock_gen.return_value = dummy_gen()
    
    payload = {
        "input": "Testing 123",
        "voice": "alloy",
        "response_format": "wav"
    }
    
    response = client.post("/v1/audio/speech", json=payload)
    assert response.status_code == 200
    assert response.content == b"fake_audio_chunk"

def test_text_to_speech_invalid_payload():
    """Test TTS endpoint with invalid payload."""
    # Empty input
    response = client.post("/v1/audio/speech", json={"input": "", "voice": "alloy"})
    assert response.status_code == 422 # Pydantic validation error (min_length=1)

    # Invalid voice name
    response = client.post("/v1/audio/speech", json={"input": "test", "voice": "../invalid"})
    assert response.status_code == 400
    assert "Invalid voice name" in response.json()["detail"]

@patch('pocketapi.rate_limiter.is_allowed')
def test_rate_limiting(mock_allowed):
    """Test rate limiting middleware."""
    # Mock rate limiter to reject request
    mock_allowed.return_value = (False, 60)
    
    response = client.get("/v1/voices")
    assert response.status_code == 429
    assert response.headers["Retry-After"] == "60"
    assert response.json()["error"] == "Rate limit exceeded"

@patch('os.path.exists')
@patch('pocketapi.open_file', new_callable=AsyncMock)
def test_caching_logic_hit(mock_open, mock_exists):
    """Test that the server uses cache when available."""
    # Mock cache hit
    mock_exists.return_value = True 
    
    # Mock anyio open_file to return a mock file handle
    mock_file = AsyncMock()
    mock_file.read.side_effect = [b"cached_audio", b""] # chunk, then EOF
    mock_open.return_value.__aenter__.return_value = mock_file
    
    # We need to import the generator directly to test its internal logic
    from pocketapi import generate_audio
    
    # Run the generator
    import asyncio
    async def run_gen():
        chunks = []
        async for chunk in generate_audio(text="Cached text", voice="alloy"):
            chunks.append(chunk)
        return b"".join(chunks)
    
    result = asyncio.run(run_gen())
    assert result == b"cached_audio"
    
    # Verify it checked the cache
    assert mock_exists.called
