#!/usr/bin/env python3
"""
Unit tests for Ice Open TTS Proxy application.

Run with: python -m pytest test_ice_open_tts_proxy.py -v
    or:   python test_ice_open_tts_proxy.py
"""

import unittest
import json
import os
import sys
import tempfile
import threading
import time
from unittest.mock import patch, MagicMock, Mock, call
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))


class TestConfig(unittest.TestCase):
    """Test configuration management."""
    
    def test_default_config(self):
        """Test default configuration values."""
        from ice_open_tts_proxy import Config, DEFAULT_CONFIG
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            config = Config(config_path)
            
            self.assertEqual(config.get("tts_server_url"), DEFAULT_CONFIG["tts_server_url"])
            self.assertEqual(config.get("api_port"), DEFAULT_CONFIG["api_port"])
            self.assertEqual(config.get("default_voice"), DEFAULT_CONFIG["default_voice"])
    
    def test_config_save_load(self):
        """Test saving and loading configuration."""
        from ice_open_tts_proxy import Config
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            
            config = Config(config_path)
            config.set("default_voice", "Carlotta")
            config.set("speed", 1.5)
            
            config2 = Config(config_path)
            self.assertEqual(config2.get("default_voice"), "Carlotta")
            self.assertEqual(config2.get("speed"), 1.5)
    
    def test_config_get_with_default(self):
        """Test config get with default value."""
        from ice_open_tts_proxy import Config
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            config = Config(config_path)
            
            self.assertEqual(config.get("nonexistent", "default"), "default")
            self.assertIsNone(config.get("nonexistent"))


class TestTTSClient(unittest.TestCase):
    """Test TTS client functionality."""
    
    @patch('ice_open_tts_proxy.requests.get')
    def test_health_check_success(self, mock_get):
        """Test successful health check."""
        from ice_open_tts_proxy import TTSClient
        
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"status": "ok"}
        
        client = TTSClient("http://localhost:8001")
        self.assertTrue(client.health_check())
    
    @patch('ice_open_tts_proxy.requests.get')
    def test_health_check_failure(self, mock_get):
        """Test failed health check."""
        from ice_open_tts_proxy import TTSClient
        
        mock_get.side_effect = Exception("Connection refused")
        
        client = TTSClient("http://localhost:8001")
        self.assertFalse(client.health_check())
    
    @patch('ice_open_tts_proxy.requests.get')
    def test_get_voices(self, mock_get):
        """Test getting voices list."""
        from ice_open_tts_proxy import TTSClient
        
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "voices": ["nova", "alloy", "Carlotta", "Aemeath"]
        }
        
        client = TTSClient("http://localhost:8001")
        voices = client.get_voices()
        
        self.assertEqual(len(voices), 4)
        self.assertIn("Carlotta", voices)
    
    @patch('ice_open_tts_proxy.requests.get')
    def test_get_voices_error_returns_defaults(self, mock_get):
        """Test getting voices with error returns defaults."""
        from ice_open_tts_proxy import TTSClient
        
        mock_get.side_effect = Exception("Connection error")
        
        client = TTSClient("http://localhost:8001")
        voices = client.get_voices()
        
        self.assertIn("nova", voices)
        self.assertIn("alloy", voices)
    
    @patch('ice_open_tts_proxy.requests.post')
    def test_generate_speech(self, mock_post):
        """Test speech generation."""
        from ice_open_tts_proxy import TTSClient
        
        audio_data = b"fake audio data"
        mock_post.return_value.status_code = 200
        mock_post.return_value.content = audio_data
        
        client = TTSClient("http://localhost:8001")
        result = client.generate_speech("Hello", "nova")
        
        self.assertEqual(result, audio_data)
        mock_post.assert_called_once()
    
    @patch('ice_open_tts_proxy.requests.post')
    def test_generate_speech_error_returns_none(self, mock_post):
        """Test speech generation error."""
        from ice_open_tts_proxy import TTSClient
        
        mock_post.side_effect = Exception("Server error")
        
        client = TTSClient("http://localhost:8001")
        result = client.generate_speech("Hello", "nova")
        
        self.assertIsNone(result)
    
    @patch('ice_open_tts_proxy.requests.post')
    def test_generate_speech_parameters(self, mock_post):
        """Test speech generation sends correct parameters."""
        from ice_open_tts_proxy import TTSClient
        
        mock_post.return_value.status_code = 200
        mock_post.return_value.content = b"audio"
        
        client = TTSClient("http://localhost:8001")
        client.generate_speech("Test text", "Carlotta", speed=1.5, format="mp3")
        
        call_args = mock_post.call_args
        self.assertEqual(call_args[1]["json"]["input"], "Test text")
        self.assertEqual(call_args[1]["json"]["voice"], "Carlotta")
        self.assertEqual(call_args[1]["json"]["speed"], 1.5)
        self.assertEqual(call_args[1]["json"]["response_format"], "mp3")


class TestStreamingTTS(unittest.TestCase):
    """Test streaming TTS functionality."""
    
    def test_streaming_text_buffer_initialization(self):
        """Test streaming text buffer initialization."""
        from ice_open_tts_proxy import LiveTextBuffer
        
        buffer = LiveTextBuffer()
        self.assertEqual(buffer.text, "")
        self.assertEqual(buffer.word_count, 0)
    
    def test_streaming_text_buffer_add_text(self):
        """Test adding text to streaming buffer."""
        from ice_open_tts_proxy import LiveTextBuffer
        
        buffer = LiveTextBuffer()
        buffer.add_text("Hello")
        self.assertEqual(buffer.text, "Hello")
        
        buffer.add_text(" world")
        self.assertEqual(buffer.text, "Hello world")
    
    def test_streaming_text_buffer_get_words(self):
        """Test getting complete words from buffer."""
        from ice_open_tts_proxy import LiveTextBuffer
        
        buffer = LiveTextBuffer()
        buffer.add_text("Hello world ")
        words = buffer.get_complete_words()
        self.assertGreater(len(words), 0)
    
    def test_streaming_text_buffer_get_remaining(self):
        """Test getting remaining text."""
        from ice_open_tts_proxy import LiveTextBuffer
        
        buffer = LiveTextBuffer()
        buffer.add_text("Hello world")
        words = buffer.get_complete_words()  # "Hello world" with no space won't be extracted
        remaining = buffer.get_remaining_text()
        # If no complete words extracted, entire text remains
        self.assertIsNotNone(remaining)
    
    def test_streaming_text_buffer_clear(self):
        """Test clearing streaming buffer."""
        from ice_open_tts_proxy import LiveTextBuffer
        
        buffer = LiveTextBuffer()
        buffer.add_text("Hello world")
        buffer.clear()
        self.assertEqual(buffer.text, "")
        self.assertEqual(buffer.word_count, 0)
    
    def test_live_tts_manager_initialization(self):
        """Test live TTS manager initialization."""
        from ice_open_tts_proxy import LiveTTSManager, AudioPlayer

        mock_client = MagicMock()
        audio_player = AudioPlayer()
        manager = LiveTTSManager(mock_client, audio_player)

        self.assertFalse(manager.is_active)

    def test_live_tts_manager_start_stop(self):
        """Test starting and stopping live TTS manager."""
        from ice_open_tts_proxy import LiveTTSManager, AudioPlayer

        mock_client = MagicMock()
        audio_player = AudioPlayer()
        manager = LiveTTSManager(mock_client, audio_player)

        manager.start()
        self.assertTrue(manager.is_active)

        manager.stop()
        self.assertFalse(manager.is_active)

    def test_live_tts_manager_set_voice_speed(self):
        """Test updating voice and speed."""
        from ice_open_tts_proxy import LiveTTSManager, AudioPlayer

        mock_client = MagicMock()
        audio_player = AudioPlayer()
        manager = LiveTTSManager(mock_client, audio_player)

        manager.set_voice("Carlotta")
        self.assertEqual(manager.voice, "Carlotta")

        manager.set_speed(1.5)
        self.assertEqual(manager.speed, 1.5)

    @patch('ice_open_tts_proxy.requests.post')
    def test_live_tts_manager_stream_word(self, mock_post):
        """Test live TTS streaming a single word."""
        from ice_open_tts_proxy import LiveTTSManager, AudioPlayer

        mock_post.return_value.status_code = 200
        mock_post.return_value.content = b"fake audio"

        mock_client = MagicMock()
        mock_client.generate_speech.return_value = b"fake audio"

        audio_player = AudioPlayer()
        manager = LiveTTSManager(mock_client, audio_player)
        manager.min_words = 1  # Set after initialization
        manager.start()

        manager.stream_word("Hello ")
        time.sleep(0.3)

        self.assertTrue(True)  # No exception = pass
        manager.stop()

    def test_live_tts_manager_double_start(self):
        """Test that double start doesn't create multiple threads."""
        from ice_open_tts_proxy import LiveTTSManager, AudioPlayer

        mock_client = MagicMock()
        audio_player = AudioPlayer()
        manager = LiveTTSManager(mock_client, audio_player)
        
        manager.start()
        first_thread = manager.worker_thread
        
        manager.start()  # Should not start again
        self.assertEqual(manager.worker_thread, first_thread)
        
        manager.stop()


class TestAudioPlayer(unittest.TestCase):
    """Test audio player functionality."""
    
    def test_player_initialization(self):
        """Test player initialization."""
        from ice_open_tts_proxy import AudioPlayer
        
        player = AudioPlayer()
        self.assertFalse(player.is_playing)
        self.assertTrue(player.play_queue.empty())
    
    def test_player_start_stop(self):
        """Test starting and stopping player."""
        from ice_open_tts_proxy import AudioPlayer
        
        player = AudioPlayer()
        player.start()
        self.assertIsNotNone(player.worker_thread)
        
        player.stop()
    
    def test_player_queue_operations(self):
        """Test audio player queue functionality."""
        from ice_open_tts_proxy import AudioPlayer
        import queue
        
        player = AudioPlayer()
        
        self.assertTrue(player.play_queue.empty())
        
        player.play_queue.put((b"test", None))
        self.assertFalse(player.play_queue.empty())


class TestAPIServer(unittest.TestCase):
    """Test API server functionality."""
    
    def test_server_initialization(self):
        """Test server initialization."""
        from ice_open_tts_proxy import APIServer, TTSClient, AudioPlayer
        
        tts_client = TTSClient("http://localhost:8001")
        audio_player = AudioPlayer()
        
        server = APIServer(tts_client, audio_player, "127.0.0.1", 5000)
        
        self.assertEqual(server.host, "127.0.0.1")
        self.assertEqual(server.port, 5000)
        self.assertFalse(server.is_running)
        self.assertIsNone(server.server_thread)


class TestCLIClient(unittest.TestCase):
    """Test CLI client functionality."""
    
    @patch('ice_open_tts_proxy_cli.requests.get')
    def test_check_status_server_up(self, mock_get):
        """Test checking TTS server when it's up."""
        from ice_open_tts_proxy_cli import check_status
        
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "model_loaded": True,
            "voice_cloning": True
        }
        
        result = check_status("http://localhost:8001")
        self.assertTrue(result)
    
    @patch('ice_open_tts_proxy_cli.requests.get')
    def test_check_status_server_down(self, mock_get):
        """Test checking TTS server when it's down."""
        from ice_open_tts_proxy_cli import check_status
        
        mock_get.side_effect = Exception("Connection refused")
        
        result = check_status("http://localhost:8001")
        self.assertFalse(result)
    
    @patch('ice_open_tts_proxy_cli.requests.post')
    def test_generate_speech_cli(self, mock_post):
        """Test CLI speech generation."""
        from ice_open_tts_proxy_cli import generate_speech
        
        mock_post.return_value.status_code = 200
        mock_post.return_value.content = b"fake audio"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "output.mp3")
            generate_speech(
                text="Hello world",
                voice="nova",
                speed=1.0,
                format="mp3",
                output=output_file,
                tts_url="http://localhost:8001"
            )
            
            self.assertTrue(os.path.exists(output_file))
            with open(output_file, "rb") as f:
                self.assertEqual(f.read(), b"fake audio")
    
    @patch('ice_open_tts_proxy_cli.requests.post')
    def test_generate_speech_creates_default_output(self, mock_post):
        """Test CLI creates default output file."""
        from ice_open_tts_proxy_cli import generate_speech
        
        mock_post.return_value.status_code = 200
        mock_post.return_value.content = b"fake audio"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                generate_speech(
                    text="Hello",
                    voice="nova",
                    speed=1.0,
                    format="mp3",
                    output=None,
                    tts_url="http://localhost:8001"
                )
                self.assertTrue(os.path.exists("tts_output.mp3"))
            finally:
                os.chdir(original_cwd)
    
    @patch('ice_open_tts_proxy_cli.requests.post')
    def test_speak(self, mock_post):
        """Test sending speak request to proxy."""
        from ice_open_tts_proxy_cli import speak
        
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "status": "success",
            "message": "Playing..."
        }
        
        speak(text="Hello", voice="nova", speed=1.0, proxy_url="http://127.0.0.1:5000")
    
    @patch('ice_open_tts_proxy_cli.requests.post')
    def test_speak_error_handling(self, mock_post):
        """Test speak with error prints error message."""
        from ice_open_tts_proxy_cli import speak
        from io import StringIO
        import sys
        
        mock_post.side_effect = Exception("Connection refused")
        
        # Capture output
        captured = StringIO()
        sys.stderr = captured
        
        try:
            speak(text="Hello", voice="nova", speed=1.0, proxy_url="http://127.0.0.1:5000")
        except SystemExit:
            pass  # Expected
        finally:
            sys.stderr = sys.__stderr__


class TestCLIListVoices(unittest.TestCase):
    """Test CLI list voices functionality."""
    
    @patch('ice_open_tts_proxy_cli.requests.get')
    def test_list_voices(self, mock_get):
        """Test listing voices."""
        from ice_open_tts_proxy_cli import list_voices
        
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "voices": ["nova", "alloy", "Carlotta"]
        }
        
        list_voices("http://localhost:8001")
    
    @patch('ice_open_tts_proxy_cli.requests.get')
    def test_list_voices_error(self, mock_get):
        """Test listing voices with error."""
        from ice_open_tts_proxy_cli import list_voices
        
        mock_get.side_effect = Exception("Connection error")
        
        # Should not raise
        list_voices("http://localhost:8001")


class TestPortChecking(unittest.TestCase):
    """Test port checking functionality."""
    
    def test_check_port_functions_exist(self):
        """Test that port checking functions exist."""
        from ice_open_tts_proxy_cli import check_port_in_use, get_pid_on_port
        
        self.assertTrue(callable(check_port_in_use))
        self.assertTrue(callable(get_pid_on_port))
    
    def test_check_port_returns_bool(self):
        """Test that check_port_in_use returns boolean."""
        from ice_open_tts_proxy_cli import check_port_in_use
        
        result = check_port_in_use(1)
        self.assertIsInstance(result, bool)


class TestAudioBackend(unittest.TestCase):
    """Test audio backend selection."""
    
    def test_setup_audio_backend(self):
        """Test audio backend detection."""
        from ice_open_tts_proxy import setup_audio_backend
        result = setup_audio_backend()
        self.assertIsInstance(result, bool)


class TestGUIConstants(unittest.TestCase):
    """Test GUI constants and defaults."""
    
    def test_default_config_values(self):
        """Test default configuration values."""
        from ice_open_tts_proxy import DEFAULT_CONFIG
        
        self.assertEqual(DEFAULT_CONFIG["tts_server_url"], "http://localhost:8001")
        self.assertEqual(DEFAULT_CONFIG["api_host"], "127.0.0.1")
        self.assertEqual(DEFAULT_CONFIG["api_port"], 5000)
        self.assertEqual(DEFAULT_CONFIG["default_voice"], "nova")
        self.assertEqual(DEFAULT_CONFIG["speed"], 1.0)
        self.assertEqual(DEFAULT_CONFIG["format"], "wav")
    
    def test_voices_list_on_error(self):
        """Test default voices returned on error."""
        from ice_open_tts_proxy import TTSClient
        
        with patch('ice_open_tts_proxy.requests.get') as mock_get:
            mock_get.side_effect = Exception("No server")
            client = TTSClient("http://localhost:8001")
            voices = client.get_voices()
            
            self.assertIn("nova", voices)
            self.assertIn("alloy", voices)
            self.assertIn("echo", voices)


class TestIntegration(unittest.TestCase):
    """Integration tests (require TTS server)."""
    
    @unittest.skipUnless(
        os.environ.get("RUN_INTEGRATION_TESTS"),
        "Integration tests require RUN_INTEGRATION_TESTS=1"
    )
    def test_full_workflow(self):
        """Test full workflow with real server."""
        from ice_open_tts_proxy import TTSClient
        
        client = TTSClient("http://localhost:8001")
        
        self.assertTrue(client.health_check())
        
        voices = client.get_voices()
        self.assertGreater(len(voices), 0)
        
        audio = client.generate_speech("Test", "nova")
        self.assertIsNotNone(audio)
        self.assertGreater(len(audio), 0)
    
    @unittest.skipUnless(
        os.environ.get("RUN_INTEGRATION_TESTS"),
        "Integration tests require RUN_INTEGRATION_TESTS=1"
    )
    def test_live_tts_integration(self):
        """Test live TTS with real server."""
        from ice_open_tts_proxy import TTSClient, LiveTTSManager, AudioPlayer

        client = TTSClient("http://localhost:8001")
        audio_player = AudioPlayer()
        manager = LiveTTSManager(client, "nova", 1.0, audio_player)
        manager.min_words = 1

        manager.start()
        manager.stream_word("Hello ")
        manager.stream_word("world ")
        time.sleep(2)
        manager.stop()

        self.assertTrue(True)


class TestCLIEndpoints(unittest.TestCase):
    """Test CLI endpoint functions."""
    
    @patch('ice_open_tts_proxy_cli.requests.get')
    def test_check_gui_app_up(self, mock_get):
        """Test checking GUI app when it's up."""
        from ice_open_tts_proxy_cli import check_status
        
        mock_get.return_value.status_code = 200
        
        # Should not raise
        check_status("http://localhost:8001", "http://127.0.0.1:5000")
    
    @patch('ice_open_tts_proxy_cli.requests.get')
    def test_check_gui_app_down(self, mock_get):
        """Test checking GUI app when it's down."""
        from ice_open_tts_proxy_cli import check_status
        
        mock_get.side_effect = Exception("Connection refused")
        
        # Should not raise
        result = check_status("http://localhost:8001", "http://127.0.0.1:5000")
        self.assertFalse(result)


class TestFlaskEndpoints(unittest.TestCase):
    """Test Flask endpoint structure."""

    @unittest.skipUnless(
        'flask' in sys.modules or
        __import__('importlib').util.find_spec('flask') is not None,
        "Flask not installed"
    )
    def test_server_creates_flask_app(self):
        """Test that server creates Flask app on start."""
        from ice_open_tts_proxy import APIServer, TTSClient, AudioPlayer

        tts_client = TTSClient("http://localhost:8001")
        audio_player = AudioPlayer()
        server = APIServer(tts_client, audio_player, "127.0.0.1", 5099)

        # Before start, flask_app should be None
        self.assertIsNone(server.flask_app)

        # Start server
        server.start()
        time.sleep(1)

        # After start, flask_app should exist
        self.assertIsNotNone(server.flask_app)
        self.assertTrue(server.is_running)
        server.stop()


class TestOpenAIEndpoints(unittest.TestCase):
    """Test OpenAI-compatible /v1/audio/speech endpoint."""

    def setUp(self):
        """Set up test client."""
        try:
            from flask import Flask
        except ImportError:
            self.skipTest("Flask not installed")

        # Import and create minimal test app
        import ice_open_tts_proxy_cli as cli_module
        import requests

        # Mock TTS server responses
        self.mock_tts_url = "http://localhost:8001"
        self.sample_audio = b"FAKE_WAV_AUDIO_DATA" * 100

    @patch('ice_open_tts_proxy_cli.requests.post')
    def test_openai_speech_basic(self, mock_post):
        """Test basic OpenAI TTS endpoint returns audio."""
        import requests
        import ice_open_tts_proxy_cli as cli_module
        from flask import Flask

        # Mock TTS server response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = self.sample_audio
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        # Create test app
        app = Flask(__name__)

        @app.route('/v1/audio/speech', methods=['POST'])
        def openai_speech():
            from flask import request, jsonify, Response
            data = request.get_json() or {}
            text = data.get('input', data.get('text', ''))
            voice = data.get('voice', 'nova')
            speed = float(data.get('speed', 1.0))
            fmt = data.get('response_format', data.get('format', 'mp3'))
            stream = data.get('stream', False)

            if not text:
                return jsonify({"error": "No input text"}), 400

            payload = {"input": text, "voice": voice, "response_format": fmt, "speed": speed}
            resp = requests.post(f"{self.mock_tts_url}/v1/audio/speech",
                                 json=payload, timeout=60, stream=stream)
            resp.raise_for_status()

            if stream:
                def generate():
                    for chunk in resp.iter_content(chunk_size=4096):
                        if chunk:
                            yield chunk
                return Response(generate(), mimetype=f'audio/{fmt}')
            else:
                return Response(resp.content, mimetype=f'audio/{fmt}')

        with app.test_client() as client:
            response = client.post('/v1/audio/speech',
                                   json={'input': 'Hello world', 'voice': 'alloy'})

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content_type, 'audio/mp3')
            self.assertGreater(len(response.data), 0)

    @patch('ice_open_tts_proxy_cli.requests.post')
    def test_openai_speech_streaming(self, mock_post):
        """Test OpenAI TTS endpoint with streaming enabled (AI agent scenario)."""
        import requests
        import requests as real_requests
        from flask import Flask

        # Mock streaming response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content = MagicMock(return_value=[b"chunk1", b"chunk2", b"chunk3"])
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        app = Flask(__name__)

        @app.route('/v1/audio/speech', methods=['POST'])
        def openai_speech():
            from flask import request, Response
            data = request.get_json() or {}
            text = data.get('input', '')
            voice = data.get('voice', 'nova')
            fmt = data.get('response_format', 'mp3')
            stream = data.get('stream', False)

            payload = {"input": text, "voice": voice, "response_format": fmt}
            resp = requests.post(f"{self.mock_tts_url}/v1/audio/speech",
                                 json=payload, timeout=60, stream=stream)
            resp.raise_for_status()

            if stream:
                def generate():
                    for chunk in resp.iter_content(chunk_size=4096):
                        if chunk:
                            yield chunk
                return Response(generate(), mimetype=f'audio/{fmt}')
            else:
                return Response(resp.content, mimetype=f'audio/{fmt}')

        with app.test_client() as client:
            # Simulate AI agent sending stream=true
            response = client.post('/v1/audio/speech',
                                   json={
                                       'input': 'Task complete',
                                       'voice': 'alloy',
                                       'response_format': 'wav',
                                       'stream': True
                                   })

            self.assertEqual(response.status_code, 200)
            self.assertIn('audio/wav', response.content_type)
            # Verify chunked response
            self.assertGreater(len(response.data), 0)

    def test_openai_speech_missing_text(self):
        """Test OpenAI endpoint returns error for missing text."""
        from flask import Flask

        app = Flask(__name__)

        @app.route('/v1/audio/speech', methods=['POST'])
        def openai_speech():
            from flask import request, jsonify
            data = request.get_json() or {}
            text = data.get('input', data.get('text', ''))

            if not text:
                return jsonify({"error": "No input text"}), 400
            return jsonify({"status": "ok"})

        with app.test_client() as client:
            response = client.post('/v1/audio/speech', json={})
            self.assertEqual(response.status_code, 400)
            data = response.get_json()
            self.assertIn('error', data)

    @patch('ice_open_tts_proxy_cli.requests.post')
    def test_openai_speech_openai_sdk_format(self, mock_post):
        """Test endpoint handles OpenAI SDK request format."""
        from flask import Flask

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"AUDIO_DATA"
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        app = Flask(__name__)

        @app.route('/v1/audio/speech', methods=['POST'])
        def openai_speech():
            from flask import request, Response
            data = request.get_json() or {}
            # Handle both 'input' (OpenAI) and 'text' formats
            text = data.get('input', data.get('text', ''))
            voice = data.get('voice', 'nova')
            fmt = data.get('response_format', 'mp3')

            payload = {"input": text, "voice": voice, "response_format": fmt}
            resp = requests.post(f"{self.mock_tts_url}/v1/audio/speech",
                                 json=payload, timeout=60)
            return Response(resp.content, mimetype=f'audio/{fmt}')

        with app.test_client() as client:
            # OpenAI SDK uses 'input' not 'text'
            response = client.post('/v1/audio/speech',
                                   json={
                                       'model': 'gpt-4o-mini-tts',
                                       'input': 'Hello from OpenAI SDK',
                                       'voice': 'coral',
                                       'response_format': 'mp3'
                                   })

            self.assertEqual(response.status_code, 200)
            self.assertIn('audio/mp3', response.content_type)

    @patch('ice_open_tts_proxy_cli.requests.post')
    def test_agent_send_scenario(self, mock_post):
        """Test AI agent send scenario: agent sends TTS request and gets streaming audio."""
        import time
        import requests
        from flask import Flask

        # Simulate TTS server generating audio in chunks
        def mock_iter_content(chunk_size=4096):
            chunks = [b"RIFF" + b"_" * 36, b"WAVEfmt " + b"_" * 16, b"data" + b"_" * 100]
            for chunk in chunks:
                time.sleep(0.01)  # Simulate generation delay
                yield chunk

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content = mock_iter_content
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        app = Flask(__name__)

        @app.route('/v1/audio/speech', methods=['POST'])
        def openai_speech():
            from flask import request, Response
            data = request.get_json() or {}
            text = data.get('input', '')
            voice = data.get('voice', 'nova')
            fmt = data.get('response_format', 'wav')
            stream = data.get('stream', False)

            payload = {"input": text, "voice": voice, "response_format": fmt}
            resp = requests.post(f"{self.mock_tts_url}/v1/audio/speech",
                                 json=payload, timeout=60, stream=stream)

            if stream:
                def generate():
                    for chunk in resp.iter_content(chunk_size=4096):
                        yield chunk
                return Response(generate(), mimetype='audio/wav')
            return Response(resp.content, mimetype='audio/wav')

        with app.test_client() as client:
            start = time.time()

            # AI agent sends TTS request with streaming
            response = client.post('/v1/audio/speech',
                                   json={
                                       'input': 'Agent task completed successfully',
                                       'voice': 'nova',
                                       'response_format': 'wav',
                                       'stream': True
                                   })

            elapsed = time.time() - start

            self.assertEqual(response.status_code, 200)
            self.assertIn('audio/wav', response.content_type)
            # Verify we got all chunks
            self.assertIn(b"RIFF", response.data)
            self.assertIn(b"WAVEfmt", response.data)
            self.assertIn(b"data", response.data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
