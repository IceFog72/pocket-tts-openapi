#!/usr/bin/env python3
"""
Ice Open TTS Proxy GUI - Desktop app for TTS with built-in API server

Features:
- GUI with text input, voice selection, playback controls
- Built-in API server for programmatic access
- Audio playback (Windows, Linux, macOS)

For CLI-only (no GUI/tkinter), use: tts_proxy_cli.py

Architecture:
    [GUI / AI Agent] → Ice Open TTS Proxy (port 8181) → Pocket TTS (port 8005)

Usage:
    python tts_proxy.py
    python tts_proxy.py --port 5001 --tts-url http://localhost:8005
"""

import os
import sys
import json
import threading
import tempfile
import argparse
import queue
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Callable

# Import the shared backend
from tts_backend import Config, TTSClient, check_port_in_use, get_pid_on_port, get_process_name, handle_port_conflict, setup_logging

logger = logging.getLogger(__name__)

# GUI imports - tkinter is built-in
HAS_TKINTER = False
try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox
    HAS_TKINTER = True
except ImportError:
    tk = None
    ttk = None
    scrolledtext = None
    messagebox = None

# HTTP requests
try:
    import requests
except ImportError:
    print("Error: requests package required. Install with: pip install requests")
    sys.exit(1)


# Audio playback - cross-platform
AUDIO_BACKEND = None
play_sound = None

def setup_audio_backend():
    """Setup the appropriate audio backend for the platform."""
    global play_sound, AUDIO_BACKEND
    
    # Try simpleaudio first (cross-platform, but skip on Linux due to segfault issues)
    if sys.platform != "linux":
        try:
            import simpleaudio as sa
            def play_sound_simpleaudio(filepath):
                wave_obj = sa.WaveObject.from_wave_file(filepath)
                play_obj = wave_obj.play()
                play_obj.wait_done()
            play_sound = play_sound_simpleaudio
            AUDIO_BACKEND = "simpleaudio"
            return True
        except ImportError:
            pass
    
    # Try playsound (works on most platforms)
    try:
        from playsound import playsound as ps
        def play_sound_playsound(filepath):
            ps(filepath, block=True)
        play_sound = play_sound_playsound
        AUDIO_BACKEND = "playsound"
        return True
    except ImportError:
        pass
    
    # Fallback to platform-specific
    if sys.platform == "win32":
        try:
            import winsound
            def play_sound_winsound(filepath):
                winsound.PlaySound(filepath, winsound.SND_FILENAME | winsound.SND_NODEFAULT)
            play_sound = play_sound_winsound
            AUDIO_BACKEND = "winsound"
            return True
        except ImportError:
            pass
    
    # Linux fallback using paplay, aplay, or ffplay
    if sys.platform == "linux":
        import subprocess
        def play_sound_linux(filepath):
            for cmd in [
                ["paplay", filepath],
                ["aplay", filepath],
                ["ffplay", "-nodisp", "-autoexit", filepath],
                ["mpv", "--no-video", filepath],
            ]:
                try:
                    subprocess.run(cmd, check=True, capture_output=True)
                    return
                except (FileNotFoundError, subprocess.CalledProcessError):
                    continue
            raise RuntimeError("No audio player found. Install paplay, aplay, ffplay, or mpv.")
        play_sound = play_sound_linux
        AUDIO_BACKEND = "system"
        return True
    
    # macOS fallback
    if sys.platform == "darwin":
        import subprocess
        def play_sound_macos(filepath):
            subprocess.run(["afplay", filepath], check=True)
        play_sound = play_sound_macos
        AUDIO_BACKEND = "afplay"
        return True
    
    return False
    


# =============================================================================
# Live TTS Classes (speaks as you type)
# =============================================================================

class LiveTextBuffer:
    """Manages text input and extracts complete words for live TTS."""

    def __init__(self):
        self.text = ""
        self.word_count = 0
        self._lock = threading.Lock()

    def add_text(self, text: str) -> None:
        """Add text to the buffer."""
        with self._lock:
            self.text += text
    
    def get_complete_words(self) -> list:
        """Extract and return complete words (ending with space/punctuation)."""
        with self._lock:
            words = []
            # Find word boundaries (space or punctuation)
            import re
            pattern = r'[\w\']+(?:\s+|$|[.,!?;:])'
            matches = re.findall(pattern, self.text)
            
            if matches:
                consumed_length = 0
                for match in matches:
                    word = match.strip()
                    if word:
                        words.append(word)
                        consumed_length += len(match)
                
                # Remove consumed text
                self.text = self.text[consumed_length:]
                self.word_count += len(words)
            
            return words
    
    def get_remaining_text(self) -> str:
        """Get any remaining text in buffer."""
        with self._lock:
            return self.text.strip()
    
    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self.text = ""
            self.word_count = 0


class OpenAITTSStreamingManager:
    """Manages streaming TTS using OpenAI-compatible endpoint with debounce."""

    def __init__(self, tts_client, audio_player):
        self.tts_client = tts_client
        self.audio_player = audio_player
        self.buffer = ""
        self.timer = None
        self.debounce_delay = 0.4  # Slightly increased to reduce 429 bursts
        self.lock = threading.Lock()
        self.is_active = False
        self.on_audio_callback = None
        self.voice = "nova"
        self.speed = 1.0
        self.format = "wav"
        self.abort_flag = threading.Event()
        self.request_serial = 0 # Track requests for cancellation
        
        # Max text size before we split into multiple requests
        self.chunk_threshold = 200 # characters

    def clear_buffer(self) -> None:
        """Clear the current text buffer and abort ongoing processing."""
        self.abort_flag.set()
        with self.lock:
            self.buffer = ""
            self.request_serial += 1 # Invalidate in-flight chunks
        self.abort_flag.clear()

    def add_text(self, text: str) -> None:
        """Add text to the buffer and reset the debounce timer."""
        with self.lock:
            self.buffer += text
        self._restart_timer()

    def _restart_timer(self) -> None:
        """Restart the debounce timer."""
        if self.timer is not None:
            self.timer.cancel()
        self.timer = threading.Timer(self.debounce_delay, self._on_timeout)
        self.timer.daemon = True
        self.timer.start()

    def _on_timeout(self) -> None:
        """Called when the debounce timer expires. Process the current buffer."""
        with self.lock:
            if not self.buffer.strip():
                return
            text = self.buffer
            self.buffer = ""
        # Outside the lock to avoid holding lock during network call
        self._process_text_with_splitting(text)

    def _process_text_with_splitting(self, text: str) -> None:
        """Split large text into sentences and process each."""
        import re
        # Split by sentence-final punctuation followed by space or newline
        # Using positive lookbehinds to keep the punctuation
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        
        for part in parts:
            if not part.strip():
                continue
            if self.abort_flag.is_set():
                break
            
            # If a single part is still too long, break it by commas
            if len(part) > self.chunk_threshold:
                subparts = re.split(r'(?<=[,;])\s+', part)
                for sub in subparts:
                    if self.abort_flag.is_set():
                        break
                    if sub.strip():
                        self._process_single_chunk(sub)
            else:
                self._process_single_chunk(part)

    def _process_single_chunk(self, text: str) -> None:
        """Send a single chunk of text to TTS server."""
        if not self.is_active:
            return
            
        # Capture current serial to check validity later
        with self.lock:
            initial_serial = self.request_serial
            
        log = logging.getLogger("tts_proxy.stream")
        log.info(f"Streaming chunk: '{text[:50]}...' (length: {len(text)})")
        try:
            payload = {
                "input": text,
                "voice": self.voice,
                "response_format": self.format,
                "speed": self.speed
            }
            log.info(f"LIVE STREAM REQUEST: voice='{self.voice}', speed={self.speed}, payload_voice='{payload['voice']}'")
            # Use stream=True to get chunked response
            for attempt in range(2):
                # Check for cancellation before network call
                with self.lock:
                    if initial_serial != self.request_serial or self.abort_flag.is_set():
                        log.info("Aborting stream request before it started (cancelled)")
                        return

                resp = requests.post(
                    f"{self.tts_client.base_url}/v1/audio/speech",
                    json=payload,
                    timeout=60,
                    stream=True
                )
                
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 1))
                    log.warning(f"Rate limited (429). Retrying after {retry_after}s... (Attempt {attempt+1}/2)")
                    time.sleep(retry_after)
                    continue
                
                resp.raise_for_status()
                break
            else:
                log.error("Failed to stream chunk after retries due to rate limit")
                return

            # Collect audio chunks
            audio_chunks = []
            for chunk in resp.iter_content(chunk_size=4096):
                # Check for cancellation during download
                with self.lock:
                    if initial_serial != self.request_serial or self.abort_flag.is_set():
                        log.info("Aborting stream download mid-way (cancelled)")
                        return
                        
                if chunk:
                    audio_chunks.append(chunk)

            if audio_chunks:
                # Final check before playing
                with self.lock:
                    if initial_serial != self.request_serial or self.abort_flag.is_set():
                        log.info("Discarding downloaded stream chunk (cancelled)")
                        return
                        
                audio_data = b"".join(audio_chunks)
                log.info(f"Received streamed audio: {len(audio_data)} bytes")
                # Play the complete audio data
                if getattr(self, "on_audio_callback", None):
                    self.on_audio_callback(audio_data)
                elif self.audio_player:
                    self.audio_player.play(audio_data)
            else:
                log.warning("Received empty audio stream")
        except Exception as e:
            log.error(f"Streaming TTS error: {e}")

    def start(self, voice: str = "nova", speed: float = 1.0, format: str = "wav") -> None:
        """Start the streaming manager."""
        self.voice = voice
        self.speed = speed
        self.format = format
        if self.is_active:
            return
        self.is_active = True

    def stop(self) -> None:
        """Stop the streaming manager and process any remaining text."""
        if not self.is_active:
            return
        self.is_active = False
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None
        # Process any remaining text in the buffer
        with self.lock:
            if self.buffer.strip():
                text = self.buffer
                self.buffer = ""
                # Process outside lock to avoid deadlock if _process_text tries to acquire lock
                # But _process_text checks is_active, which we just set to False, so we need to adjust
                # We'll set a flag or just process if we are stopping? Let's process anyway for the last chunk.
                # We'll temporarily set is_active to True for this last call.
                was_active = self.is_active
                self.is_active = True
                self._process_text_with_splitting(text)
                self.is_active = was_active

    def set_voice(self, voice: str) -> None:
        """Update the voice for streaming."""
        self.voice = voice

    def set_speed(self, speed: float) -> None:
        """Update the speed for streaming."""
        self.speed = speed

    def set_format(self, fmt: str) -> None:
        """Update the audio format for streaming."""
        self.format = fmt

    def set_client(self, tts_client) -> None:
        """Update the TTS client for streaming."""
        self.tts_client = tts_client


# =============================================================================
# Audio Player (Thread-safe)
# =============================================================================

class AudioPlayer:
    """Thread-safe audio player with queue."""

    def __init__(self):
        self.play_queue = queue.Queue()
        self.is_playing = False
        self.worker_thread = None
        self.stop_event = threading.Event()
        self.current_proc = None
        self._lock = threading.Lock()
        self.log = logging.getLogger("tts_proxy.audio")

    def start(self):
        """Start the audio player worker thread."""
        self.stop_event.clear()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        self.log.info("Audio player started")

    def stop(self):
        """Stop current playback and player thread."""
        self.stop_event.set()
        # Kill current process
        with self._lock:
            if self.current_proc:
                try:
                    if os.name == 'nt':
                        # Use taskkill for Windows to ensure process-tree termination
                        subprocess.run(['taskkill', '/F', '/T', '/PID', str(self.current_proc.pid)], 
                                       capture_output=True)
                    else:
                        self.current_proc.kill() # Force kill (SIGKILL on Linux)
                    self.current_proc = None
                except:
                    pass
        
        # Clear the queue
        while not self.play_queue.empty():
            try:
                self.play_queue.get_nowait()
            except queue.Empty:
                break
        
        # Put sentinel
        self.play_queue.put(None)
        
        if self.worker_thread:
            self.worker_thread.join(timeout=1)
        self.log.info("Audio player stopped")

    def play(self, audio_data: bytes, callback=None):
        """Add audio to play queue."""
        self.log.debug(f"Queueing audio: {len(audio_data)} bytes")
        self.play_queue.put((audio_data, callback))

    def _worker(self):
        """Worker thread for playing audio."""
        while not self.stop_event.is_set():
            try:
                item = self.play_queue.get(timeout=0.5)
                if item is None:
                    break

                audio_data, callback = item
                self.is_playing = True
                self.log.debug(f"Playing audio: {len(audio_data)} bytes")

                # Detect format from magic bytes
                if audio_data[:3] == b'ID3' or audio_data[:2] == b'\xff\xfb' or audio_data[:2] == b'\xff\xf3':
                    suffix = ".mp3"
                elif audio_data[:4] == b'OggS':
                    suffix = ".ogg"
                elif audio_data[:4] == b'fLaC':
                    suffix = ".flac"
                elif audio_data[:4] == b'RIFF':
                    suffix = ".wav"
                else:
                    suffix = ".wav"

                # Save to temp file
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                    f.write(audio_data)
                    temp_path = f.name
                
                try:
                    # Multi-platform process-based playback (required for interruption)
                    success = False
                    import subprocess

                    # Wake up audio device (PulseAudio/PipeWire goes to sleep between plays)
                    if sys.platform == "linux":
                        try:
                            subprocess.run(["pw-cat", "--playback", "/dev/null"],
                                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL, timeout=1)
                        except Exception:
                            pass
                        time.sleep(0.15)

                    if sys.platform == "win32":
                        # Windows: Try common process-based players
                        for cmd_name in ["ffplay", "mpv", "powershell"]:
                            try:
                                if cmd_name == "powershell":
                                    # Use a lightweight PowerShell command to play sound while keeping process control
                                    cmd = ["powershell", "-c", f"(New-Object Media.SoundPlayer '{temp_path}').PlaySync()"]
                                elif cmd_name == "ffplay":
                                    cmd = ["ffplay", "-nodisp", "-autoexit", temp_path]
                                elif cmd_name == "mpv":
                                    cmd = ["mpv", "--no-video", temp_path]
                                else:
                                    cmd = [cmd_name, temp_path]
                                
                                proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                                with self._lock:
                                    self.current_proc = proc
                                
                                proc.wait()
                                with self._lock:
                                    self.current_proc = None
                                success = True
                                break
                            except:
                                continue
                                
                    elif sys.platform == "linux" and AUDIO_BACKEND == "system":
                        # Linux: Standard system command logic
                        for cmd_name in ["ffplay", "mpv", "paplay", "aplay"]:
                            try:
                                cmd = [cmd_name, temp_path]
                                if cmd_name == "ffplay": cmd = ["ffplay", "-nodisp", "-autoexit", temp_path]
                                if cmd_name == "mpv": cmd = ["mpv", "--no-video", temp_path]

                                proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                                with self._lock:
                                    self.current_proc = proc

                                proc.wait()
                                with self._lock:
                                    self.current_proc = None
                                if proc.returncode == 0:
                                    success = True
                                    break
                            except:
                                continue
                    
                    # Fallback to blocking libraries if no process-based player worked
                    if not success and play_sound:
                        self.log.warning("No process-based player found; using fallback (Stop button may not work for this sentence)")
                        play_sound(temp_path)
                    
                    if callback:
                        callback("done")
                finally:
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
                    self.is_playing = False
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Audio playback error: {e}")
                if callback:
                    callback(f"error: {e}")
                self.is_playing = False


# =============================================================================
# API Server (for programmatic access)
# =============================================================================

class APIServer:
    """Flask-based API server for programmatic access."""
    
    def __init__(self, tts_client: TTSClient, audio_player: AudioPlayer,
                 host: str = "127.0.0.1", port: int = 8181,
                 feed_queue: Optional[queue.Queue] = None):
        self.tts_client = tts_client
        self.audio_player = audio_player
        self.host = host
        self.port = port
        self.feed_queue = feed_queue
        self.server_thread = None
        self.flask_app = None
        self.is_running = False
    
    def start(self):
        """Start the API server in a background thread."""
        from flask import Flask, request, jsonify, Response
        log = logging.getLogger("tts_proxy.api")

        self.flask_app = Flask(__name__)

        @self.flask_app.route('/health', methods=['GET'])
        def health():
            return jsonify({
                "status": "ok",
                "tts_connected": self.tts_client.health_check(),
                "audio_playing": self.audio_player.is_playing
            })

        @self.flask_app.route('/shutdown', methods=['POST'])
        def shutdown():
            func = request.environ.get('werkzeug.server.shutdown')
            if func is None:
                self.is_running = False
                return jsonify({"status": "shutting_down"})
            func()
            return jsonify({"status": "shutting_down"})

        @self.flask_app.route('/speak', methods=['POST'])
        @self.flask_app.route('/tts_and_play', methods=['POST'])
        def speak():
            data = request.get_json() or {}
            text = data.get('text', '')
            voice = data.get('voice', 'nova')
            speed = float(data.get('speed', 1.0))
            format = data.get('format', 'wav')

            if not text:
                log.warning("Speak request with empty text")
                return jsonify({"error": "No text provided"}), 400

            log.info(f"Speak request: voice={voice}, speed={speed}, text='{text[:50]}...'")
            audio_data = self.tts_client.generate_speech(text, voice, speed, format)
            if audio_data is None:
                log.error(f"Failed to generate speech for text='{text[:50]}...'")
                return jsonify({"error": "Failed to generate speech"}), 500

            # Log to feed queue if available
            if self.feed_queue is not None:
                self.feed_queue.put({
                    'type': 'speak',
                    'text': text[:100] + ('...' if len(text) > 100 else ''),
                    'voice': voice,
                    'speed': speed,
                    'format': format,
                    'timestamp': time.time()
                })

            self.audio_player.play(audio_data)
            log.info(f"Playing audio: {len(audio_data)} bytes")
            return jsonify({"status": "success", "message": f"Playing: {text[:50]}..."})

        @self.flask_app.route('/voices', methods=['GET'])
        def voices():
            return jsonify({"voices": self.tts_client.get_voices()})

        @self.flask_app.route('/v1/audio/speech', methods=['POST'])
        def openai_speech():
            """OpenAI-compatible TTS endpoint with streaming support."""
            data = request.get_json() or {}
            text = data.get('input', data.get('text', ''))
            voice = data.get('voice', 'nova')
            speed = float(data.get('speed', 1.0))
            fmt = data.get('response_format', data.get('format', 'mp3'))
            stream = data.get('stream', False)

            if not text:
                return jsonify({"error": "No input text"}), 400

            log.info(f"OpenAI TTS: voice={voice}, speed={speed}, stream={stream}, text='{text[:50]}...'")

            # Generate audio from TTS server
            payload = {"input": text, "voice": voice, "response_format": fmt, "speed": speed}
            try:
                resp = requests.post(f"{self.tts_client.base_url}/v1/audio/speech",
                                     json=payload, timeout=60, stream=stream)
                resp.raise_for_status()
            except Exception as e:
                log.error(f"TTS generation failed: {e}")
                return jsonify({"error": str(e)}), 500

            # Log to feed queue if available
            if self.feed_queue is not None:
                self.feed_queue.put({
                    'type': 'openai_speech',
                    'text': text[:100] + ('...' if len(text) > 100 else ''),
                    'voice': voice,
                    'speed': speed,
                    'format': fmt,
                    'stream': stream,
                    'timestamp': time.time()
                })

            if stream:
                # Return chunked response for streaming
                def generate():
                    for chunk in resp.iter_content(chunk_size=4096):
                        if chunk:
                            yield chunk

                content_type = {
                    'mp3': 'audio/mpeg',
                    'wav': 'audio/wav',
                    'opus': 'audio/opus',
                    'flac': 'audio/flac',
                    'pcm': 'audio/pcm'
                }.get(fmt, 'audio/mpeg')

                return Response(generate(), mimetype=content_type)
            else:
                # Return complete audio
                return Response(resp.content, mimetype=f'audio/{fmt}')

        def run():
            log.info(f"API server starting on {self.host}:{self.port}")
            self.flask_app.run(host=self.host, port=self.port,
                              debug=False, use_reloader=False)

        self.server_thread = threading.Thread(target=run, daemon=True)
        self.server_thread.start()
        self.is_running = True
        log.info(f"API server thread started")
    
    def stop(self):
        """Stop the API server."""
        self.is_running = False
        if self.flask_app:
            try:
                # Use Flask's built-in shutdown endpoint
                import requests
                requests.post(f"http://{self.host}:{self.port}/shutdown", timeout=2)
            except Exception:
                pass
            self.flask_app = None


# =============================================================================
# GUI Application
# =============================================================================

class TTSApp:
    """Main GUI application."""
    
    def __init__(self, root: tk.Tk, config: Config, tts_url: str, 
                  api_port: int, api_host: str):
        self.root = root
        self.config = config
        self.tts_url = tts_url or config.get("tts_server_url")
        
        # Prevent blocking CLI input in GUI mode: check port manually
        desired_port = int(api_port or config.getint("api_port"))
        if check_port_in_use(desired_port):
            pid = get_pid_on_port(desired_port)
            proc_name = get_process_name(pid) if pid else "Unknown"
            
            is_our_app = False
            if proc_name and any(kw in proc_name.lower() for kw in ["python", "pocket", "ice"]):
                is_our_app = True
                
            if is_our_app and messagebox:
                msg = f"Port {desired_port} is in use by same app/script ({proc_name}).\nDo you want to kill it and restart?"
                if messagebox.askyesno("Restart Server?", msg):
                    import subprocess
                    import os as sys_os
                    if sys.platform == "win32":
                        subprocess.run(["taskkill", "/PID", pid, "/F"], check=True)
                    else:
                        sys_os.kill(int(pid), 9)
                else:
                    sys.exit(1)
            else:
                if messagebox:
                    messagebox.showerror("Port in use", f"API port {desired_port} is occupied by another app: {proc_name}\nPlease shut it down or change the port.")
                sys.exit(1)
        final_port = desired_port
        
        # Live TTS tracking
        self.last_live_text_len = 0
        self.live_tts_enabled = False
        
        # Initialize components
        self.tts_client = TTSClient(self.tts_url)
        self.audio_player = AudioPlayer()
        self.live_tts_manager = None  # Created when needed
        self.live_tts_enabled = False
        self.feed_queue = queue.Queue()
        self.api_server = APIServer(
            self.tts_client, self.audio_player,
            host=api_host or config.get("api_host"),
            port=final_port,
            feed_queue=self.feed_queue
        )
        
        # Status queue for thread-safe GUI updates
        self.status_queue = queue.Queue()
        
        # Generation tracking for cancellation
        self.gen_serial = 0
        self.serial_lock = threading.Lock()
        
        # Setup GUI
        self._setup_gui()
        
        # Start background services
        self._start_services()
        
        # Start status poller
        self._poll_status()
    
    def _setup_gui(self):
        """Setup the GUI layout."""
        self.root.title("TTS Speaker")
        self.root.geometry("500x600")
        self.root.minsize(400, 500)
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # --- TTS Server Section ---
        server_frame = ttk.LabelFrame(main_frame, text="TTS Server", padding="5")
        server_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(server_frame, text="URL:").pack(side=tk.LEFT)
        self.server_url_var = tk.StringVar(value=self.tts_url)
        self.server_entry = ttk.Entry(server_frame, textvariable=self.server_url_var)
        self.server_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.connect_btn = ttk.Button(server_frame, text="Connect", 
                                       command=self._connect_server)
        self.connect_btn.pack(side=tk.LEFT)
        
        self.server_status = ttk.Label(server_frame, text="●", foreground="red")
        self.server_status.pack(side=tk.LEFT, padx=5)
        
        # --- Voice Selection ---
        voice_frame = ttk.LabelFrame(main_frame, text="Voice", padding="5")
        voice_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(voice_frame, text="Voice:").pack(side=tk.LEFT)
        self.voice_var = tk.StringVar(value=self.config.get("default_voice", "nova"))
        self.voice_combo = ttk.Combobox(voice_frame, textvariable=self.voice_var,
                                         state="readonly", width=20)
        self.voice_combo.pack(side=tk.LEFT, padx=5)
        self.voice_combo['values'] = ["nova", "alloy", "echo", "fable", "onyx", "shimmer"]
        
        ttk.Label(voice_frame, text="Speed:").pack(side=tk.LEFT, padx=(10, 0))
        self.speed_var = tk.DoubleVar(value=self.config.get("speed", 1.0))
        self.speed_spin = ttk.Spinbox(voice_frame, from_=0.25, to=4.0, increment=0.1,
                                       textvariable=self.speed_var, width=5)
        self.speed_spin.pack(side=tk.LEFT, padx=5)
        
        # Sync changes to live manager if it exists
        self.voice_var.trace_add("write", lambda *args: self._sync_live_settings())
        self.speed_var.trace_add("write", lambda *args: self._sync_live_settings())
        
        refresh_btn = ttk.Button(voice_frame, text="↻", width=3,
                                  command=self._refresh_voices)
        refresh_btn.pack(side=tk.LEFT)

        # --- API Mode ---
        mode_frame = ttk.LabelFrame(main_frame, text="API Mode", padding="5")
        mode_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(mode_frame, text="Mode:").pack(side=tk.LEFT)
        self.api_mode_var = tk.StringVar(value="OpenAI POST")
        self.api_mode_combo = ttk.Combobox(mode_frame, textvariable=self.api_mode_var,
                                            state="readonly", width=18,
                                            values=["OpenAI POST", "XTTS GET Stream", "XTTS POST", "WebSocket"])
        self.api_mode_combo.pack(side=tk.LEFT, padx=5)

        clear_cache_btn = ttk.Button(mode_frame, text="Clear Cache",
                                      command=self._clear_cache)
        clear_cache_btn.pack(side=tk.RIGHT)

        # --- Text Input ---
        text_frame = ttk.LabelFrame(main_frame, text="Text", padding="5")
        text_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.text_input = scrolledtext.ScrolledText(text_frame, height=6, wrap=tk.WORD, undo=True)
        self.text_input.pack(fill=tk.BOTH, expand=True)
        self.text_input.insert("1.0", "Hello! This is a test of the TTS speaker.")
        
        # Live TTS checkbox
        live_frame = ttk.Frame(text_frame)
        live_frame.pack(fill=tk.X, pady=(5, 0))

        self.live_tts_var = tk.BooleanVar(value=False)
        self.live_tts_check = ttk.Checkbutton(
            live_frame,
            text="Live Mode (speak as you type)",
            variable=self.live_tts_var,
            command=self._toggle_live_tts
        )
        self.live_tts_check.pack(side=tk.LEFT)

        # Bind standard shortcuts (especially for Linux)
        self.text_input.bind("<Control-a>", self._select_all)
        self.text_input.bind("<Control-A>", self._select_all)
        self.text_input.bind("<<Paste>>", self._on_paste)
        
        # Undo / Redo
        self.text_input.bind("<Control-z>", lambda e: self.text_input.event_generate("<<Undo>>"))
        self.text_input.bind("<Control-y>", lambda e: self.text_input.event_generate("<<Redo>>"))
        self.text_input.bind("<Control-Shift-Z>", lambda e: self.text_input.event_generate("<<Redo>>"))
        
        # Bind key release for live TTS
        self.text_input.bind("<KeyRelease>", self._on_text_keypress)
        
        # --- Controls ---
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.speak_btn = ttk.Button(control_frame, text="🔊 Speak", 
                                     command=self._speak, style="Accent.TButton")
        self.speak_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.stop_btn = ttk.Button(control_frame, text="⏹ Stop", 
                                    command=self._stop_speaking, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        save_btn = ttk.Button(control_frame, text="💾 Save", command=self._save_audio)
        save_btn.pack(side=tk.LEFT)
        
        # --- Status Bar ---
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X)
        
        self.playing_dot = ttk.Label(status_frame, text="●", foreground="gray")
        self.playing_dot.pack(side=tk.LEFT)
        self.playing_label = ttk.Label(status_frame, text="Ready")
        self.playing_label.pack(side=tk.LEFT, padx=5)
        
        self.api_label = ttk.Label(status_frame, text="API: --")
        self.api_label.pack(side=tk.RIGHT)
        
        # Bind Enter key to speak
        self.text_input.bind("<Control-Return>", lambda e: self._speak())
        
        # Common shortcut bindings to ensures they work
        self.text_input.bind("<Control-c>", lambda e: self.text_input.event_generate("<<Copy>>"))
        self.text_input.bind("<Control-v>", lambda e: self.text_input.event_generate("<<Paste>>"))
        self.text_input.bind("<Control-x>", lambda e: self.text_input.event_generate("<<Cut>>"))
        
        # --- Request Feed ---
        feed_frame = ttk.LabelFrame(main_frame, text="Request Feed", padding="5")
        feed_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        self.feed_text = scrolledtext.ScrolledText(feed_frame, height=8, wrap=tk.WORD, state=tk.DISABLED)
        self.feed_text.pack(fill=tk.BOTH, expand=True)
        
        # Clear feed button
        clear_feed_btn = ttk.Button(feed_frame, text="Clear Feed", command=self._clear_feed)
        clear_feed_btn.pack(side=tk.RIGHT, pady=(2, 0))
    
    def _start_services(self):
        """Start background services."""
        self.audio_player.start()
        self.api_server.start()
        self.status_queue.put(f"API server running on http://{self.api_server.host}:{self.api_server.port}")
        self.api_label.config(text=f"API: :{self.api_server.port}")
        
        # Try to connect to TTS server
        self._connect_server()
    
    def _connect_server(self):
        """Connect to TTS server and fetch voices."""
        self.tts_url = self.server_url_var.get()
        self.tts_client = TTSClient(self.tts_url)
        self.config.set("tts_server_url", self.tts_url)
        
        def do_connect():
            if self.tts_client.health_check():
                self.status_queue.put("Connected to TTS server")
                self.server_status.config(foreground="green")
                # Update live manager if it exists
                if self.live_tts_manager:
                    self.live_tts_manager.set_client(self.tts_client)
                self._refresh_voices()
            else:
                self.status_queue.put("Cannot connect to TTS server")
                self.server_status.config(foreground="red")
        
        threading.Thread(target=do_connect, daemon=True).start()
    
    def _refresh_voices(self):
        """Fetch available voices from TTS server."""
        def do_refresh():
            voices = self.tts_client.get_voices()
            if voices:
                self.voice_combo['values'] = voices
                self.status_queue.put(f"Loaded {len(voices)} voices")
        
        threading.Thread(target=do_refresh, daemon=True).start()
    
    def _toggle_live_tts(self):
        """Toggle live TTS mode on/off."""
        self.live_tts_enabled = self.live_tts_var.get()

        if self.live_tts_enabled:
            if not self.live_tts_manager:
                self.live_tts_manager = OpenAITTSStreamingManager(
                    self.tts_client,
                    self.audio_player
                )
            self.live_tts_manager.start(
                voice=self.voice_var.get(),
                speed=self.speed_var.get(),
                format="wav"
            )
            self.last_live_text_len = len(self.text_input.get("1.0", tk.INSERT))
            self.stop_btn.config(state=tk.NORMAL)
            self.speak_btn.config(state=tk.DISABLED)
            self.status_queue.put("Live mode ON - type and it will speak (streaming)")
        else:
            if self.live_tts_manager:
                self.live_tts_manager.stop()
            self.speak_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.status_queue.put("Live mode OFF")
            
    def _sync_live_settings(self):
        """Sync current UI settings to the live TTS manager."""
        if self.live_tts_manager:
            try:
                v = self.voice_var.get()
                s = self.speed_var.get()
                self.live_tts_manager.set_voice(v)
                self.live_tts_manager.set_speed(s)
                print(f"DEBUG: Synced live settings: voice={v}, speed={s}")
            except Exception as e:
                print(f"DEBUG: Failed to sync live settings: {e}")

    def _on_text_keypress(self, event):
        """Handle key release for live TTS."""
        if not self.live_tts_enabled or not self.live_tts_manager:
            return

        current_text = self.text_input.get("1.0", tk.INSERT)

        # If text decreased (deletion) or we are before the last offset, reset tracking
        if len(current_text) < self.last_live_text_len:
            self.last_live_text_len = len(current_text)
            return

        # Only process trigger keys for sending to TTS
        if event.keysym in ('space', 'Return', 'period', 'exclam', 'question', 'comma', 'semicolon', 'colon'):
            # Extract only the NEW portion since last trigger
            if len(current_text) > self.last_live_text_len:
                new_chunk = current_text[self.last_live_text_len:]
                self.live_tts_manager.add_text(new_chunk)
                self.last_live_text_len = len(current_text)
                self.playing_label.config(text=f"Live: {new_chunk.strip()[:30]}...")
    
    def _on_paste(self, event=None):
        """Handle paste events for live mode."""
        if not self.live_tts_enabled or not self.live_tts_manager:
            return
        
        # Need a small delay to let the paste complete before we read the text
        self.root.after(50, self._process_after_paste)

    def _process_after_paste(self):
        """Read text after paste completed."""
        if not self.live_tts_enabled or not self.live_tts_manager:
            return
            
        current_text = self.text_input.get("1.0", tk.INSERT)
        if len(current_text) > self.last_live_text_len:
            new_chunk = current_text[self.last_live_text_len:]
            self.live_tts_manager.add_text(new_chunk)
            self.last_live_text_len = len(current_text)
            self.playing_label.config(text=f"Live (Pasted): {new_chunk.strip()[:30]}...")
        else:
            self.last_live_text_len = len(current_text)

    def _select_all(self, event=None):
        """Select all text in input widget."""
        self.text_input.tag_add(tk.SEL, "1.0", tk.END)
        self.text_input.mark_set(tk.INSERT, tk.END)
        self.text_input.see(tk.INSERT)
        return "break" # Prevent default behavior
    
    def _speak(self):
        """Generate and play speech."""
        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("No Text", "Please enter some text to speak.")
            return
        
        voice = self.voice_var.get()
        speed = self.speed_var.get()
        
        self.speak_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.playing_label.config(text=f"Generating speech with {voice}...")
        
        # Increment and capture serial for cancellation
        with self.serial_lock:
            self.gen_serial += 1
            request_serial = self.gen_serial
        
        def do_speak():
            audio_data = self.tts_client.generate_speech(text, voice, speed, "wav")
            
            # Check if this request is still valid
            with self.serial_lock:
                if request_serial != self.gen_serial:
                    self.log.info(f"Discarding obsolete audio request (ID {request_serial})")
                    return

            if audio_data:
                self.status_queue.put(f"Playing audio ({len(audio_data)} bytes)")
                self.audio_player.play(audio_data, callback=self._on_playback_done)
            else:
                self.status_queue.put("Failed to generate speech")
                self.speak_btn.config(state=tk.NORMAL)
                self.stop_btn.config(state=tk.DISABLED)
        
        threading.Thread(target=do_speak, daemon=True).start()

    def _generate_by_mode(self, text, voice, speed, fmt, mode):
        """Generate audio using the selected API mode."""
        import requests

        base = self.tts_url.rstrip("/")

        # Log to feed
        self.feed_queue.put({
            "timestamp": time.time(),
            "type": "tts",
            "text": text[:100] + ("..." if len(text) > 100 else ""),
            "voice": voice,
            "format": fmt,
            "speed": speed,
            "mode": mode,
        })
        
        if mode == "OpenAI POST":
            try:
                r = requests.post(f"{base}/v1/audio/speech",
                    json={"input": text, "voice": voice, "response_format": fmt, "speed": speed},
                    timeout=60)
                r.raise_for_status()
                return r.content
            except Exception as e:
                self.log.error(f"OpenAI POST error: {e}")
                return None
        
        elif mode == "XTTS GET Stream":
            try:
                from urllib.parse import urlencode
                params = urlencode({"text": text, "voice": voice, "format": fmt, "speed": speed})
                r = requests.get(f"{base}/tts_stream?{params}", timeout=60, stream=True)
                r.raise_for_status()
                return r.content
            except Exception as e:
                self.log.error(f"XTTS GET error: {e}")
                return None
        
        elif mode == "XTTS POST":
            try:
                r = requests.post(f"{base}/tts_to_audio",
                    json={"text": text, "speaker_wav": voice, "speed": speed, "format": fmt},
                    timeout=60)
                r.raise_for_status()
                return r.content
            except Exception as e:
                self.log.error(f"XTTS POST error: {e}")
                return None
        
        elif mode == "WebSocket":
            try:
                import asyncio, json as json_mod, websockets
                return asyncio.run(self._ws_generate(base, text, voice, speed, fmt))
            except Exception as e:
                self.log.error(f"WebSocket error: {e}")
                return None
        
        return None

    async def _ws_generate(self, base, text, voice, speed, fmt):
        """Generate audio via WebSocket."""
        import json as json_mod, websockets
        ws_url = base.replace("http://", "ws://").replace("https://", "wss://") + "/v1/audio/stream"
        audio = b""
        async with websockets.connect(ws_url) as ws:
            await ws.send(json_mod.dumps({"text": text, "voice": voice, "format": fmt, "speed": speed}))
            while True:
                msg = await ws.recv()
                if isinstance(msg, bytes):
                    audio += msg
                else:
                    data = json_mod.loads(msg)
                    if data.get("status") == "done":
                        break
                    elif data.get("status") == "error":
                        raise Exception(data.get("error", "Unknown error"))
        return audio
    
    def _on_playback_done(self, result):
        """Callback when playback finishes."""
        if not self.live_tts_enabled:
            self.root.after(0, lambda: self.speak_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.stop_btn.config(state=tk.DISABLED))
        
        if result == "done":
            self.root.after(0, lambda: self.playing_label.config(text="Playback complete"))
        else:
            self.root.after(0, lambda: self.playing_label.config(text=f"Playback: {result}"))
    
    def _stop_speaking(self):
        """Stop current playback."""
        # Increment serial to invalidate pending requests
        with self.serial_lock:
             self.gen_serial += 1

        # Stop and clear the live manager if it exists
        if self.live_tts_manager:
            self.live_tts_manager.clear_buffer()
        
        # Stop the player (kills current process and clears queue)
        self.audio_player.stop()
        
        # Restart audio player thread for next use
        self.audio_player.start()
        
        if not self.live_tts_enabled:
            self.speak_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
        else:
            self.speak_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
        self.playing_dot.config(foreground="gray")
        self.playing_label.config(text="Stopped")
    
    def _save_audio(self):
        """Generate and save audio to file."""
        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("No Text", "Please enter some text.")
            return
        
        from tkinter import filedialog
        filepath = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("MP3 files", "*.mp3"), ("All files", "*.*")]
        )
        if not filepath:
            return
        
        voice = self.voice_var.get()
        speed = self.speed_var.get()
        fmt = "mp3" if filepath.endswith(".mp3") else "wav"
        
        def do_save():
            self.status_queue.put("Generating audio...")
            audio_data = self.tts_client.generate_speech(text, voice, speed, fmt)
            if audio_data:
                with open(filepath, 'wb') as f:
                    f.write(audio_data)
                self.status_queue.put(f"Saved to {filepath}")
            else:
                self.status_queue.put("Failed to generate audio")
        
        threading.Thread(target=do_save, daemon=True).start()
    
    def _poll_status(self):
        """Poll status and feed queues and update GUI."""
        # Status queue
        try:
            while True:
                msg = self.status_queue.get_nowait()
                self.playing_label.config(text=msg)
        except queue.Empty:
            pass
        
        # Feed queue
        try:
            while True:
                feed_msg = self.feed_queue.get_nowait()
                self._update_feed(feed_msg)
        except queue.Empty:
            pass
            
        # Update playing indicator
        if self.audio_player.is_playing:
            self.playing_dot.config(foreground="limegreen")
            self.playing_label.config(text="🔊 Playing...")
        else:
            self.playing_dot.config(foreground="gray")
            self.playing_label.config(text="Ready")
            
        self.root.after(100, self._poll_status)
    
    def _update_feed(self, msg: dict):
        """Add a message to the feed display."""
        timestamp = time.strftime("%H:%M:%S", time.localtime(msg.get('timestamp', time.time())))
        text = msg.get('text', '')
        voice = msg.get('voice', 'unknown')
        stream = msg.get('stream', False)
        fmt = msg.get('format', 'wav')
        mode = msg.get('mode', '')
        
        line = f"[{timestamp}] {voice} ({fmt}"
        if mode:
            line += f", {mode}"
        elif stream:
            line += ", stream"
        line += f"): {text}\n"
        
        self.feed_text.config(state=tk.NORMAL)
        self.feed_text.insert(tk.END, line)
        self.feed_text.see(tk.END)  # Scroll to bottom
        self.feed_text.config(state=tk.DISABLED)
    
    def _clear_cache(self):
        """Clear server audio cache."""
        import requests
        try:
            r = requests.post(f"{self.tts_url.rstrip('/')}/cache/clear", timeout=10)
            data = r.json()
            deleted = data.get("deleted", 0)
            self.status_queue.put(f"Cache cleared: {deleted} files deleted")
            self._update_feed({
                "timestamp": time.time(),
                "type": "system",
                "text": f"Cache cleared ({deleted} files)",
                "voice": "",
                "format": "",
            })
        except Exception as e:
            self.status_queue.put(f"Cache clear failed: {e}")

    def _clear_feed(self):
        """Clear the feed display."""
        self.feed_text.config(state=tk.NORMAL)
        self.feed_text.delete(1.0, tk.END)
        self.feed_text.config(state=tk.DISABLED)
    
    def on_close(self):
        """Handle window close."""
        # Invalidate any in-flight requests
        with self.serial_lock:
            self.gen_serial += 1
            
        if self.live_tts_manager:
            self.live_tts_manager.stop()
        self.audio_player.stop()
        self.api_server.stop()
        self.root.destroy()


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    _config = Config()
    
    parser = argparse.ArgumentParser(description="TTS Speaker - GUI and API for Pocket TTS")
    parser.add_argument("--tts-url", default=_config.get("tts_server_url"),
                        help="Pocket TTS server URL")
    parser.add_argument("--port", type=int, default=_config.getint("api_port"),
                        help="API server port")
    parser.add_argument("--host", default=_config.get("api_host"),
                        help="API server host")
    parser.add_argument("--no-gui", action="store_true",
                        help="Run in headless mode (API server only)")
    parser.add_argument("--log", action="store_true",
                        help="Enable logging to logs/ folder")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level (default: INFO)")
    args = parser.parse_args()

    # Setup logging only if --log flag is provided
    if args.log:
        log_level = getattr(logging, args.log_level.upper())
        log_file = setup_logging(log_level)
        log = logging.getLogger("tts_proxy.main")
        log.info(f"Starting Ice Open TTS Proxy (log file: {log_file})")
    else:
        logging.disable(logging.CRITICAL)  # Disable all logging
        log = logging.getLogger("tts_proxy.main")
    
    # Setup audio backend
    if not setup_audio_backend():
        log.warning("No audio playback backend found. Install simpleaudio or playsound.")
        print("Warning: No audio playback backend found.")
        print("Install one of: simpleaudio, playsound")
        print("Audio will be saved to files instead of played.")
    
    log.info(f"Audio backend: {AUDIO_BACKEND or 'None (file save only)'}")
    print(f"Audio backend: {AUDIO_BACKEND or 'None (file save only)'}")
    
    config = Config()
    
    if args.no_gui or not HAS_TKINTER:
        if not HAS_TKINTER and not args.no_gui:
            print("=" * 50)
            print("  GUI not available (tkinter not installed)")
            print("  Running in headless/API mode")
            print("  Install tkinter: sudo pacman -S tk (Arch)")
            print("                  sudo apt-get install python3-tk (Debian)")
            print("=" * 50)
            print()
        
        # Headless mode - lightweight API server only (no audio player)
        print(f"Starting API server on {args.host}:{args.port}")
        print(f"API ready at http://{args.host}:{args.port}/speak")
        print("Press Ctrl+C to stop")
        log.info(f"Starting headless mode on {args.host}:{args.port}")

        from flask import Flask, request, jsonify
        headless_app = Flask(__name__)

        @headless_app.route('/health')
        def health():
            return jsonify({"status": "ok", "mode": "headless"})

        @headless_app.route('/speak', methods=['POST'])
        @headless_app.route('/tts_and_play', methods=['POST'])
        def speak():
            data = request.get_json() or {}
            text = data.get('text', '')
            voice = data.get('voice', 'nova')
            speed = float(data.get('speed', 1.0))
            fmt = data.get('format', 'wav')

            if not text:
                log.warning("Speak request with empty text")
                return jsonify({"error": "No text"}), 400

            log.info(f"Speak request: voice={voice}, speed={speed}, text='{text[:50]}...'")

            # Generate audio via TTS server
            payload = {"input": text, "voice": voice, "response_format": fmt, "speed": speed}
            try:
                start_time = time.time()
                resp = requests.post(f"{args.tts_url}/v1/audio/speech", json=payload, timeout=60)
                resp.raise_for_status()
                elapsed = time.time() - start_time
                log.info(f"Generated speech: {len(resp.content)} bytes in {elapsed:.2f}s")

                # Save to temp file
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as f:
                    f.write(resp.content)
                    temp_path = f.name

                # Play using system command (non-blocking)
                import subprocess, platform
                system = platform.system()
                try:
                    if system == "Linux":
                        subprocess.Popen(["paplay", temp_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    elif system == "Darwin":
                        subprocess.Popen(["afplay", temp_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    elif system == "Windows":
                        os.startfile(temp_path)
                    log.info(f"Playing audio via {system}")
                except Exception as e:
                    log.warning(f"Audio playback failed: {e}")

                return jsonify({"status": "success", "message": f"Playing: {text[:50]}...", "file": temp_path})
            except Exception as e:
                log.error(f"Speak error: {e}")
                return jsonify({"error": str(e)}), 500

        @headless_app.route('/voices')
        def voices():
            try:
                resp = requests.get(f"{args.tts_url}/v1/voices", timeout=5)
                return resp.json()
            except Exception as e:
                log.error(f"Failed to fetch voices: {e}")
                return jsonify({"voices": []})

        log.info(f"Headless server running on {args.host}:{args.port}")
        headless_app.run(host=args.host, port=args.port, debug=False)
    else:
        # GUI mode
        root = tk.Tk()
        
        # Try to set icon
        try:
            # You can add an icon file here
            pass
        except:
            pass
        
        app = TTSApp(root, config, args.tts_url, args.port, args.host)
        root.protocol("WM_DELETE_WINDOW", app.on_close)
        
        try:
            root.mainloop()
        except KeyboardInterrupt:
            app.on_close()


if __name__ == "__main__":
    main()
