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
import requests

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))


class TestConfig(unittest.TestCase):
    """Test configuration management."""
    
    def test_default_config(self):
        """Test default configuration values."""
        from tts_backend import Config
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.ini")
            config = Config(config_path)
            
            self.assertEqual(config.get("tts_server_url"), "http://localhost:8005")
            self.assertEqual(config.get("api_port"), "8181")
            self.assertEqual(config.get("default_voice"), "nova")
    
    def test_config_save_load(self):
        """Test saving and loading configuration."""
        from tts_backend import Config
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.ini")
            
            config = Config(config_path)
            config.set("default_voice", "Carlotta")
            config.set("speed", 1.5)
            
            config2 = Config(config_path)
            self.assertEqual(config2.get("default_voice"), "Carlotta")
            self.assertEqual(float(config2.get("speed")), 1.5)
    
    def test_config_get_with_default(self):
        """Test config get with default value."""
        from tts_backend import Config
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.ini")
            config = Config(config_path)
            
            self.assertEqual(config.get("nonexistent", "default"), "default")
            self.assertIsNone(config.get("nonexistent"))

    def test_config_getint(self):
        """Test config getint."""
        from tts_backend import Config
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.ini")
            config = Config(config_path)
            
            self.assertEqual(config.getint("api_port"), 8181)
            self.assertEqual(config.getint("nonexistent", 42), 42)

    def test_config_getfloat(self):
        """Test config getfloat."""
        from tts_backend import Config
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.ini")
            config = Config(config_path)
            
            self.assertEqual(config.getfloat("speed"), 1.0)
            self.assertEqual(config.getfloat("nonexistent", 2.5), 2.5)

    def test_config_getboolean(self):
        """Test config getboolean."""
        from tts_backend import Config
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.ini")
            config = Config(config_path)
            
            # logging.enabled should be False by default
            self.assertFalse(config.getboolean("nonexistent_key", False))

    def test_config_overwrites_existing(self):
        """Test that config overwrites existing file."""
        from tts_backend import Config
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.ini")
            
            config1 = Config(config_path)
            config1.set("default_voice", "alloy")
            
            config2 = Config(config_path)
            self.assertEqual(config2.get("default_voice"), "alloy")


class TestTTSClient(unittest.TestCase):
    """Test TTS client functionality."""
    
    @patch('requests.get')
    def test_health_check_success(self, mock_get):
        """Test successful health check."""
        from tts_backend import TTSClient
        
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"status": "ok"}
        
        client = TTSClient("http://localhost:8005")
        self.assertTrue(client.health_check())
    
    @patch('requests.get')
    def test_health_check_failure(self, mock_get):
        """Test failed health check."""
        from tts_backend import TTSClient
        
        mock_get.side_effect = Exception("Connection refused")
        
        client = TTSClient("http://localhost:8005")
        self.assertFalse(client.health_check())

    @patch('requests.get')
    def test_health_check_non_200(self, mock_get):
        """Test health check with non-200 status."""
        from tts_backend import TTSClient
        
        mock_get.return_value.status_code = 500
        
        client = TTSClient("http://localhost:8005")
        self.assertFalse(client.health_check())
    
    @patch('requests.get')
    def test_get_voices(self, mock_get):
        """Test getting voices list."""
        from tts_backend import TTSClient
        
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "voices": ["nova", "alloy", "Carlotta", "Aemeath"]
        }
        
        client = TTSClient("http://localhost:8005")
        voices = client.get_voices()
        
        self.assertEqual(len(voices), 4)
        self.assertIn("Carlotta", voices)
    
    @patch('requests.get')
    def test_get_voices_error_returns_defaults(self, mock_get):
        """Test getting voices with error returns defaults."""
        from tts_backend import TTSClient
        
        mock_get.side_effect = Exception("Connection error")
        
        client = TTSClient("http://localhost:8005")
        voices = client.get_voices()
        
        self.assertIn("nova", voices)
        self.assertIn("alloy", voices)
        self.assertEqual(len(voices), 6)  # 6 default voices
    
    @patch('requests.post')
    def test_generate_speech(self, mock_post):
        """Test speech generation."""
        from tts_backend import TTSClient
        
        audio_data = b"fake audio data"
        mock_post.return_value.status_code = 200
        mock_post.return_value.content = audio_data
        
        client = TTSClient("http://localhost:8005")
        result = client.generate_speech("Hello", "nova")
        
        self.assertEqual(result, audio_data)
        mock_post.assert_called_once()
    
    @patch('requests.post')
    def test_generate_speech_error_returns_none(self, mock_post):
        """Test speech generation error."""
        from tts_backend import TTSClient
        
        mock_post.side_effect = Exception("Server error")
        
        client = TTSClient("http://localhost:8005")
        result = client.generate_speech("Hello", "nova")
        
        self.assertIsNone(result)
    
    @patch('requests.post')
    def test_generate_speech_parameters(self, mock_post):
        """Test speech generation sends correct parameters."""
        from tts_backend import TTSClient
        
        mock_post.return_value.status_code = 200
        mock_post.return_value.content = b"audio"
        
        client = TTSClient("http://localhost:8005")
        client.generate_speech("Test text", "Carlotta", speed=1.5, format="mp3")
        
        call_args = mock_post.call_args
        self.assertEqual(call_args[1]["json"]["input"], "Test text")
        self.assertEqual(call_args[1]["json"]["voice"], "Carlotta")
        self.assertEqual(call_args[1]["json"]["speed"], 1.5)
        self.assertEqual(call_args[1]["json"]["response_format"], "mp3")

    def test_base_url_trailing_slash_stripped(self):
        """Test that trailing slash is stripped from base URL."""
        from tts_backend import TTSClient
        
        client = TTSClient("http://localhost:8005/")
        self.assertEqual(client.base_url, "http://localhost:8005")


class TestAudioPlayer(unittest.TestCase):
    """Test audio player functionality."""
    
    def test_player_initialization(self):
        from ice_open_tts_proxy import AudioPlayer
        player = AudioPlayer()
        self.assertFalse(player.is_playing)
        self.assertTrue(player.play_queue.empty())
        self.assertIsNone(player.worker_thread)
    
    def test_player_start_stop(self):
        from ice_open_tts_proxy import AudioPlayer
        player = AudioPlayer()
        player.start()
        self.assertIsNotNone(player.worker_thread)
        self.assertTrue(player.worker_thread.is_alive())
        player.stop()
        self.assertTrue(player.stop_event.is_set())

    def test_player_double_start(self):
        from ice_open_tts_proxy import AudioPlayer
        player = AudioPlayer()
        player.start()
        t1 = player.worker_thread
        player.start()  # should be idempotent or safe
        player.stop()
    
    def test_player_queue_operations(self):
        from ice_open_tts_proxy import AudioPlayer
        player = AudioPlayer()
        self.assertTrue(player.play_queue.empty())
        player.play_queue.put((b"test", None))
        self.assertFalse(player.play_queue.empty())

    def test_player_play_queues_audio(self):
        from ice_open_tts_proxy import AudioPlayer
        player = AudioPlayer()
        player.play(b"audio_data")
        self.assertFalse(player.play_queue.empty())
        item = player.play_queue.get_nowait()
        self.assertEqual(item[0], b"audio_data")


class TestAPIServer(unittest.TestCase):
    """Test API server functionality."""
    
    def test_server_initialization(self):
        """Test server initialization."""
        from ice_open_tts_proxy import APIServer, TTSClient, AudioPlayer
        
        tts_client = TTSClient("http://localhost:8005")
        audio_player = AudioPlayer()
        
        server = APIServer(tts_client, audio_player, "127.0.0.1", 5000)
        
        self.assertEqual(server.host, "127.0.0.1")
        self.assertEqual(server.port, 5000)
        self.assertFalse(server.is_running)
        self.assertIsNone(server.server_thread)

    def test_server_with_feed_queue(self):
        from ice_open_tts_proxy import APIServer, TTSClient, AudioPlayer
        import queue
        
        fq = queue.Queue()
        server = APIServer(
            TTSClient("http://localhost:8005"),
            AudioPlayer(),
            feed_queue=fq
        )
        self.assertIs(server.feed_queue, fq)

    def test_server_stop(self):
        from ice_open_tts_proxy import APIServer, TTSClient, AudioPlayer
        server = APIServer(TTSClient("http://localhost:8005"), AudioPlayer())
        server.is_running = True
        server.stop()
        self.assertFalse(server.is_running)


class TestLiveTextBuffer(unittest.TestCase):
    """Test the LiveTextBuffer class."""
    
    def test_initial_state(self):
        from ice_open_tts_proxy import LiveTextBuffer
        
        buf = LiveTextBuffer()
        self.assertEqual(buf.text, "")
        self.assertEqual(buf.word_count, 0)
    
    def test_add_text(self):
        from ice_open_tts_proxy import LiveTextBuffer
        
        buf = LiveTextBuffer()
        buf.add_text("hello ")
        self.assertEqual(buf.text, "hello ")
    
    def test_add_text_appends(self):
        from ice_open_tts_proxy import LiveTextBuffer
        
        buf = LiveTextBuffer()
        buf.add_text("hello ")
        buf.add_text("world ")
        self.assertEqual(buf.text, "hello world ")
    
    def test_get_complete_words(self):
        from ice_open_tts_proxy import LiveTextBuffer
        
        buf = LiveTextBuffer()
        buf.add_text("hello world ")
        words = buf.get_complete_words()
        
        self.assertIsInstance(words, list)
        self.assertGreater(len(words), 0)
    
    def test_get_remaining_text(self):
        from ice_open_tts_proxy import LiveTextBuffer
        
        buf = LiveTextBuffer()
        buf.add_text("hello")
        remaining = buf.get_remaining_text()
        self.assertEqual(remaining, "hello")
    
    def test_get_remaining_text_strips(self):
        from ice_open_tts_proxy import LiveTextBuffer
        
        buf = LiveTextBuffer()
        buf.add_text("  hello  ")
        remaining = buf.get_remaining_text()
        self.assertEqual(remaining, "hello")
    
    def test_clear(self):
        from ice_open_tts_proxy import LiveTextBuffer
        
        buf = LiveTextBuffer()
        buf.add_text("hello world")
        buf.word_count = 5
        buf.clear()
        
        self.assertEqual(buf.text, "")
        self.assertEqual(buf.word_count, 0)
    
    def test_thread_safety(self):
        from ice_open_tts_proxy import LiveTextBuffer
        
        buf = LiveTextBuffer()
        
        def writer():
            for i in range(100):
                buf.add_text("word ")
        
        threads = [threading.Thread(target=writer) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should not crash and should have text
        self.assertGreater(len(buf.text), 0)


class TestOpenAITTSStreamingManager(unittest.TestCase):
    """Test the streaming TTS manager."""
    
    def test_initialization(self):
        from ice_open_tts_proxy import OpenAITTSStreamingManager
        
        mock_client = MagicMock()
        mock_player = MagicMock()
        
        manager = OpenAITTSStreamingManager(mock_client, mock_player)
        
        self.assertEqual(manager.voice, "nova")
        self.assertEqual(manager.speed, 1.0)
        self.assertEqual(manager.format, "wav")
        self.assertFalse(manager.is_active)
        self.assertEqual(manager.buffer, "")
    
    def test_start(self):
        from ice_open_tts_proxy import OpenAITTSStreamingManager
        
        mock_client = MagicMock()
        mock_player = MagicMock()
        manager = OpenAITTSStreamingManager(mock_client, mock_player)
        
        manager.start(voice="alloy", speed=1.5, format="mp3")
        
        self.assertTrue(manager.is_active)
        self.assertEqual(manager.voice, "alloy")
        self.assertEqual(manager.speed, 1.5)
        self.assertEqual(manager.format, "mp3")
    
    def test_start_idempotent(self):
        from ice_open_tts_proxy import OpenAITTSStreamingManager
        
        mock_client = MagicMock()
        mock_player = MagicMock()
        manager = OpenAITTSStreamingManager(mock_client, mock_player)
        
        manager.start()
        manager.start()  # Should not raise
        
        self.assertTrue(manager.is_active)
    
    def test_stop(self):
        from ice_open_tts_proxy import OpenAITTSStreamingManager
        
        mock_client = MagicMock()
        mock_player = MagicMock()
        manager = OpenAITTSStreamingManager(mock_client, mock_player)
        
        manager.start()
        manager.stop()
        
        self.assertFalse(manager.is_active)
    
    def test_stop_when_not_started(self):
        from ice_open_tts_proxy import OpenAITTSStreamingManager
        
        mock_client = MagicMock()
        mock_player = MagicMock()
        manager = OpenAITTSStreamingManager(mock_client, mock_player)
        
        manager.stop()  # Should not raise
    
    def test_add_text(self):
        from ice_open_tts_proxy import OpenAITTSStreamingManager
        
        mock_client = MagicMock()
        mock_player = MagicMock()
        manager = OpenAITTSStreamingManager(mock_client, mock_player)
        
        manager.add_text("hello ")
        self.assertIn("hello ", manager.buffer)
    
    def test_clear_buffer(self):
        from ice_open_tts_proxy import OpenAITTSStreamingManager
        
        mock_client = MagicMock()
        mock_player = MagicMock()
        manager = OpenAITTSStreamingManager(mock_client, mock_player)
        
        manager.add_text("hello ")
        manager.clear_buffer()
        
        self.assertEqual(manager.buffer, "")
    
    def test_set_voice(self):
        from ice_open_tts_proxy import OpenAITTSStreamingManager
        
        mock_client = MagicMock()
        mock_player = MagicMock()
        manager = OpenAITTSStreamingManager(mock_client, mock_player)
        
        manager.set_voice("alloy")
        self.assertEqual(manager.voice, "alloy")
    
    def test_set_speed(self):
        from ice_open_tts_proxy import OpenAITTSStreamingManager
        
        mock_client = MagicMock()
        mock_player = MagicMock()
        manager = OpenAITTSStreamingManager(mock_client, mock_player)
        
        manager.set_speed(1.5)
        self.assertEqual(manager.speed, 1.5)
    
    def test_set_format(self):
        from ice_open_tts_proxy import OpenAITTSStreamingManager
        
        mock_client = MagicMock()
        mock_player = MagicMock()
        manager = OpenAITTSStreamingManager(mock_client, mock_player)
        
        manager.set_format("mp3")
        self.assertEqual(manager.format, "mp3")
    
    def test_set_client(self):
        from ice_open_tts_proxy import OpenAITTSStreamingManager
        
        mock_client = MagicMock()
        mock_player = MagicMock()
        manager = OpenAITTSStreamingManager(mock_client, mock_player)
        
        new_client = MagicMock()
        manager.set_client(new_client)
        self.assertIs(manager.tts_client, new_client)


class TestCLIClient(unittest.TestCase):
    """Test CLI client functionality."""
    
    @patch('requests.get')
    def test_check_status_server_up(self, mock_get):
        """Test checking TTS server when it's up."""
        from ice_open_tts_proxy_cli import check_status
        
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "model_loaded": True,
            "voice_cloning": True
        }
        
        result = check_status("http://localhost:8005")
        self.assertTrue(result)
    
    @patch('requests.get')
    def test_check_status_server_down(self, mock_get):
        """Test checking TTS server when it's down."""
        from ice_open_tts_proxy_cli import check_status
        
        mock_get.side_effect = Exception("Connection refused")
        
        result = check_status("http://localhost:8005")
        self.assertFalse(result)
    
    @patch('requests.post')
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
                tts_url="http://localhost:8005"
            )
            
            self.assertTrue(os.path.exists(output_file))
            with open(output_file, "rb") as f:
                self.assertEqual(f.read(), b"fake audio")
    
    @patch('requests.post')
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
                    tts_url="http://localhost:8005"
                )
                self.assertTrue(os.path.exists("tts_output.mp3"))
            finally:
                os.chdir(original_cwd)
    
    @patch('requests.post')
    def test_speak(self, mock_post):
        """Test sending speak request to proxy."""
        from ice_open_tts_proxy_cli import speak
        
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "status": "success",
            "message": "Playing..."
        }
        
        speak(text="Hello", voice="nova", speed=1.0, proxy_url="http://127.0.0.1:5000")
    
    @patch('requests.post')
    def test_speak_error_handling(self, mock_post):
        """Test speak with error prints error message."""
        from ice_open_tts_proxy_cli import speak
        
        mock_post.side_effect = Exception("Connection refused")
        
        # Should not raise (prints error)
        speak(text="Hello", voice="nova", speed=1.0, proxy_url="http://127.0.0.1:5000")


class TestCLIListVoices(unittest.TestCase):
    """Test CLI list voices functionality."""
    
    @patch('requests.get')
    def test_list_voices(self, mock_get):
        """Test listing voices."""
        from ice_open_tts_proxy_cli import list_voices
        
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "voices": ["nova", "alloy", "Carlotta"]
        }
        
        list_voices("http://localhost:8005")
    
    @patch('requests.get')
    def test_list_voices_error(self, mock_get):
        """Test listing voices with error."""
        from ice_open_tts_proxy_cli import list_voices
        
        mock_get.side_effect = Exception("Connection error")
        
        # Should not raise
        list_voices("http://localhost:8005")


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

    def test_backend_port_check(self):
        """Test port checking from backend."""
        from tts_backend import check_port_in_use
        
        result = check_port_in_use(1)
        self.assertIsInstance(result, bool)

    def test_backend_get_pid_returns_none_on_free_port(self):
        """Test get_pid_on_port returns None for unused port."""
        from tts_backend import get_pid_on_port
        
        # Use a very unlikely port
        result = get_pid_on_port(59999)
        self.assertIsNone(result)


class TestAudioBackend(unittest.TestCase):
    """Test audio backend selection."""
    
    def test_setup_audio_backend(self):
        """Test audio backend detection."""
        from ice_open_tts_proxy import setup_audio_backend
        result = setup_audio_backend()
        self.assertIsInstance(result, bool)


class TestIntegration(unittest.TestCase):
    """Integration tests (require TTS server)."""
    
    @unittest.skipUnless(
        os.environ.get("RUN_INTEGRATION_TESTS"),
        "Integration tests require RUN_INTEGRATION_TESTS=1"
    )
    def test_full_workflow(self):
        """Test full workflow with real server."""
        from tts_backend import TTSClient
        
        client = TTSClient("http://localhost:8005")
        
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
        from tts_backend import TTSClient
        from ice_open_tts_proxy import OpenAITTSStreamingManager, AudioPlayer

        client = TTSClient("http://localhost:8005")
        audio_player = AudioPlayer()
        manager = OpenAITTSStreamingManager(client, audio_player)
        manager.start(voice="nova", speed=1.0)

        manager.add_text("Hello world. ")
        time.sleep(2)
        manager.stop()


class TestCLIEndpoints(unittest.TestCase):
    """Test CLI endpoint functions."""
    
    @patch('requests.get')
    def test_check_gui_app_up(self, mock_get):
        """Test checking GUI app when it's up."""
        from ice_open_tts_proxy_cli import check_status
        
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"model_loaded": True, "voice_cloning": False}
        
        # Should not raise
        check_status("http://localhost:8005", "http://127.0.0.1:5000")
    
    @patch('requests.get')
    def test_check_gui_app_down(self, mock_get):
        """Test checking GUI app when it's down."""
        from ice_open_tts_proxy_cli import check_status
        
        mock_get.side_effect = Exception("Connection refused")
        
        # Should not raise
        result = check_status("http://localhost:8005", "http://127.0.0.1:5000")
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

        tts_client = TTSClient("http://localhost:8005")
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

        self.mock_tts_url = "http://localhost:8005"
        self.sample_audio = b"FAKE_WAV_AUDIO_DATA" * 100

    @patch('requests.post')
    def test_openai_speech_basic(self, mock_post):
        """Test basic OpenAI TTS endpoint returns audio."""
        from flask import Flask

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = self.sample_audio
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

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

    @patch('requests.post')
    def test_openai_speech_streaming(self, mock_post):
        """Test OpenAI TTS endpoint with streaming enabled (AI agent scenario)."""
        from flask import Flask

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
            response = client.post('/v1/audio/speech',
                                   json={
                                       'input': 'Task complete',
                                       'voice': 'alloy',
                                       'response_format': 'wav',
                                       'stream': True
                                   })

            self.assertEqual(response.status_code, 200)
            self.assertIn('audio/wav', response.content_type)
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

    @patch('requests.post')
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
            text = data.get('input', data.get('text', ''))
            voice = data.get('voice', 'nova')
            fmt = data.get('response_format', 'mp3')

            payload = {"input": text, "voice": voice, "response_format": fmt}
            resp = requests.post(f"{self.mock_tts_url}/v1/audio/speech",
                                 json=payload, timeout=60)
            return Response(resp.content, mimetype=f'audio/{fmt}')

        with app.test_client() as client:
            response = client.post('/v1/audio/speech',
                                   json={
                                       'model': 'gpt-4o-mini-tts',
                                       'input': 'Hello from OpenAI SDK',
                                       'voice': 'coral',
                                       'response_format': 'mp3'
                                   })

            self.assertEqual(response.status_code, 200)
            self.assertIn('audio/mp3', response.content_type)

    @patch('requests.post')
    def test_agent_send_scenario(self, mock_post):
        """Test AI agent send scenario: agent sends TTS request and gets streaming audio."""
        from flask import Flask

        def mock_iter_content(chunk_size=4096):
            chunks = [b"RIFF" + b"_" * 36, b"WAVEfmt " + b"_" * 16, b"data" + b"_" * 100]
            for chunk in chunks:
                time.sleep(0.01)
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
            response = client.post('/v1/audio/speech',
                                   json={
                                       'input': 'Agent task completed successfully',
                                       'voice': 'nova',
                                       'response_format': 'wav',
                                       'stream': True
                                   })

            self.assertEqual(response.status_code, 200)
            self.assertIn('audio/wav', response.content_type)
            self.assertIn(b"RIFF", response.data)
            self.assertIn(b"WAVEfmt", response.data)
            self.assertIn(b"data", response.data)


class TestTTSProxyClient(unittest.TestCase):
    """Test tts_proxy_client.py functions."""
    
    @patch('requests.get')
    def test_check_tts_server_up(self, mock_get):
        from tts_proxy_client import check_tts_server
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"model_loaded": True, "voice_cloning": True}
        self.assertTrue(check_tts_server("http://localhost:8005"))
    
    @patch('requests.get')
    def test_check_tts_server_down(self, mock_get):
        from tts_proxy_client import check_tts_server
        mock_get.side_effect = Exception("refused")
        self.assertFalse(check_tts_server("http://localhost:8005"))
    
    @patch('requests.get')
    def test_check_tts_server_non_200(self, mock_get):
        from tts_proxy_client import check_tts_server
        mock_get.return_value.status_code = 500
        self.assertFalse(check_tts_server("http://localhost:8005"))
    
    @patch('requests.get')
    def test_check_gui_app_up(self, mock_get):
        from tts_proxy_client import check_gui_app
        mock_get.return_value.status_code = 200
        self.assertTrue(check_gui_app("http://127.0.0.1:8181"))
    
    @patch('requests.get')
    def test_check_gui_app_down(self, mock_get):
        from tts_proxy_client import check_gui_app
        mock_get.side_effect = Exception("refused")
        self.assertFalse(check_gui_app("http://127.0.0.1:8181"))

    @patch('requests.get')
    def test_get_voices_client(self, mock_get):
        from tts_proxy_client import get_voices
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"voices": ["a", "b"]}
        mock_get.return_value.raise_for_status = MagicMock()
        get_voices("http://localhost:8005")  # Should not raise

    @patch('requests.get')
    def test_get_voices_client_error(self, mock_get):
        from tts_proxy_client import get_voices
        mock_get.side_effect = Exception("error")
        get_voices("http://localhost:8005")  # Should not raise

    @patch('requests.post')
    def test_generate_speech_to_file(self, mock_post):
        from tts_proxy_client import generate_speech
        mock_post.return_value.status_code = 200
        mock_post.return_value.content = b"audio"
        mock_post.return_value.raise_for_status = MagicMock()
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "test.wav")
            generate_speech("http://localhost:8005", "hello", output_file=out)
            self.assertTrue(os.path.exists(out))

    @patch('requests.post')
    def test_generate_speech_play_flag(self, mock_post):
        from tts_proxy_client import generate_speech
        mock_post.return_value.status_code = 200
        mock_post.return_value.content = b"audio"
        mock_post.return_value.raise_for_status = MagicMock()
        with tempfile.TemporaryDirectory() as d:
            orig = os.getcwd()
            os.chdir(d)
            try:
                generate_speech("http://localhost:8005", "hello", play=True)
            finally:
                os.chdir(orig)

    @patch('requests.post')
    def test_generate_speech_error_exits(self, mock_post):
        from tts_proxy_client import generate_speech
        mock_post.side_effect = Exception("fail")
        with self.assertRaises(SystemExit):
            generate_speech("http://localhost:8005", "hello")

    @patch('requests.post')
    def test_speak_to_gui(self, mock_post):
        from tts_proxy_client import speak_to_gui
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"status": "success"}
        speak_to_gui("hello", gui_url="http://127.0.0.1:8181")

    @patch('requests.post')
    def test_speak_to_gui_error(self, mock_post):
        from tts_proxy_client import speak_to_gui
        mock_post.side_effect = Exception("refused")
        with self.assertRaises(SystemExit):
            speak_to_gui("hello", gui_url="http://127.0.0.1:8181")


if __name__ == "__main__":
    unittest.main(verbosity=2)
