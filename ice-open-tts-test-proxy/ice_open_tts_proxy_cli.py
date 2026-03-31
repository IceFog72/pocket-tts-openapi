#!/usr/bin/env python3
"""
Ice Open Ice Open TTS Proxy - Command-line only TTS client and server

No GUI dependencies (no tkinter needed).

Features:
- Send text and get audio files
- Trigger playback via Ice Open TTS Proxy
- Check server status
- List voices
- Run as lightweight API server

Usage:
    python tts_proxy_cli.py --text "Hello" --save output.mp3
    python tts_proxy_cli.py --speak "Hello" --voice Carlotta
    python tts_proxy_cli.py --status
    python tts_proxy_cli.py --server  # Start API-only server
"""

import sys
import os
import argparse
import tempfile
import logging
import time
import queue
import threading
from pathlib import Path
from typing import Optional

try:
    import requests
except ImportError:
    print("Error: pip install requests")
    sys.exit(1)

# Global feed queue for --feed option
feed_queue = None


def setup_logging(log_level=logging.INFO):
    """Configure logging with file and console handlers."""
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    log_file = log_dir / f"tts_proxy_cli_{time.strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger("tts_proxy_cli")
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return log_file


def check_port_in_use(port: int) -> bool:
    """Check if a port is already in use."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def get_pid_on_port(port: int):
    """Get PID using a port (cross-platform)."""
    import subprocess
    try:
        if sys.platform == "win32":
            result = subprocess.run(
                ["netstat", "-ano"], capture_output=True, text=True
            )
            for line in result.stdout.split('\n'):
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.strip().split()
                    return parts[-1]
        else:
            # Linux/Mac
            for cmd in [["lsof", "-ti", f":{port}"], ["ss", "-tlnp"]]:
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if cmd[0] == "lsof" and result.stdout.strip():
                        return result.stdout.strip()
                except FileNotFoundError:
                    continue
    except Exception:
        pass
    return None


def handle_port_conflict(port: int, script_name: str = "server") -> int:
    """Handle port conflict - ask user for new port or exit."""
    if not check_port_in_use(port):
        return port
    
    pid = get_pid_on_port(port)
    print(f"\n⚠️  Port {port} already in use" + (f" (PID: {pid})" if pid else ""))
    
    try:
        choice = input("\n1) Use different port\n2) Exit\n\nChoice [1-2]: ").strip()
        if choice == "1":
            new_port = input("Enter new port: ").strip()
            if new_port.isdigit():
                new_port = int(new_port)
                if check_port_in_use(new_port):
                    print(f"Port {new_port} also in use")
                    sys.exit(1)
                return new_port
            print("Invalid port")
        sys.exit(1)
    except (KeyboardInterrupt, EOFError):
        sys.exit(1)


from tts_backend import Config
_config = Config()

# Read from config.ini (created by Config with defaults if missing)
TTS_SERVER = os.environ.get("TTS_SERVER", _config.get("tts_server_url"))
PROXY_SERVER = os.environ.get("PROXY_SERVER", f"http://{_config.get('api_host')}:{_config.get('api_port')}")


def check_status(tts_url: str, proxy_url: str = None) -> bool:
    """Check server status. Returns True if TTS server is up."""
    print("Checking servers...\n")
    tts_ok = False
    
    # TTS Server
    try:
        resp = requests.get(f"{tts_url}/health", timeout=3)
        data = resp.json()
        print(f"✓ TTS Server: {tts_url}")
        print(f"  Model: {'loaded' if data.get('model_loaded') else 'not loaded'}")
        print(f"  Voice cloning: {'yes' if data.get('voice_cloning') else 'no'}")
        tts_ok = True
    except Exception as e:
        print(f"✗ TTS Server: {tts_url} - {e}")
    
    # Proxy Server
    if proxy_url:
        try:
            resp = requests.get(f"{proxy_url}/health", timeout=3)
            print(f"\n✓ Ice Open TTS Proxy: {proxy_url}")
        except:
            print(f"\n✗ Ice Open TTS Proxy: {proxy_url} - not running")
    
    return tts_ok


def list_voices(tts_url: str):
    """List available voices."""
    try:
        resp = requests.get(f"{tts_url}/v1/voices", timeout=5)
        voices = resp.json().get("voices", [])
        print(f"\nAvailable voices ({len(voices)}):")
        for v in voices:
            print(f"  • {v}")
    except Exception as e:
        print(f"Error: {e}")


def generate_speech(text: str, voice: str, speed: float, format: str,
                    output: str, tts_url: str):
    """Generate speech and save to file."""
    log = logging.getLogger("tts_proxy_cli")
    payload = {
        "input": text,
        "voice": voice,
        "response_format": format,
        "speed": speed
    }

    print(f"Generating speech (voice={voice}, format={format})")
    log.info(f"Generate speech: voice={voice}, speed={speed}, format={format}, text_len={len(text)}")

    try:
        start_time = time.time()
        resp = requests.post(f"{tts_url}/v1/audio/speech", json=payload, timeout=60)
        resp.raise_for_status()
        elapsed = time.time() - start_time

        if not output:
            output = f"tts_output.{format}"

        with open(output, 'wb') as f:
            f.write(resp.content)

        print(f"✓ Saved: {output} ({len(resp.content):,} bytes)")
        log.info(f"Saved: {output} ({len(resp.content)} bytes, {elapsed:.2f}s)")

    except Exception as e:
        print(f"Error: {e}")
        log.error(f"Generate speech error: {e}")
        sys.exit(1)


def speak(text: str, voice: str, speed: float, proxy_url: str):
    """Send text to proxy for playback."""
    log = logging.getLogger("tts_proxy_cli")
    payload = {"text": text, "voice": voice, "speed": speed}
    log.info(f"Speak: voice={voice}, speed={speed}, text='{text}'")

    try:
        resp = requests.post(f"{proxy_url}/speak", json=payload, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        print(f"✓ {result.get('message', 'OK')}")
        log.info(f"Speak success: {result.get('message', 'OK')}")
    except Exception as e:
        print(f"Error: {e}")
        log.error(f"Speak error: {e}")
        print("Is Ice Open TTS Proxy running? Start with: python tts_proxy.py")


def run_server(tts_url: str, host: str, port: int, feed_queue: Optional[queue.Queue] = None):
    """Run lightweight API server (no GUI)."""
    log = logging.getLogger("tts_proxy_cli")
    # Check for port conflict first
    port = handle_port_conflict(port, "Ice Open TTS Proxy")
    
    try:
        from flask import Flask, request, jsonify, Response
    except ImportError:
        print("Error: pip install flask")
        sys.exit(1)
    
    app = Flask(__name__)
    
    @app.route('/health')
    def health():
        return jsonify({"status": "ok", "server": "tts_proxy_cli"})
    
    @app.route('/speak', methods=['POST'])
    @app.route('/tts_and_play', methods=['POST'])
    def speak_endpoint():
        data = request.get_json() or {}
        text = data.get('text', '')
        voice = data.get('voice', 'nova')
        speed = float(data.get('speed', 1.0))
        fmt = data.get('format', 'wav')
        
        if not text:
            return jsonify({"error": "No text"}), 400
        
        # Generate audio
        payload = {"input": text, "voice": voice, "response_format": fmt, "speed": speed}
        try:
            resp = requests.post(f"{tts_url}/v1/audio/speech", json=payload, timeout=60)
            resp.raise_for_status()
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as f:
            f.write(resp.content)
            temp_path = f.name
        
        # Play audio
        play_audio(temp_path)
        
        # Log to feed if enabled
        if feed_queue is not None:
            feed_queue.put({
                'type': 'speak',
                'text': text[:100] + ('...' if len(text) > 100 else ''),
                'voice': voice,
                'speed': speed,
                'format': fmt,
                'timestamp': time.time()
            })
        
        return jsonify({"status": "success", "message": f"Playing: {text}"})
    
    @app.route('/voices')
    def voices():
        try:
            resp = requests.get(f"{tts_url}/v1/voices", timeout=5)
            return resp.json()
        except:
            return jsonify({"voices": []})
    
    @app.route('/v1/audio/speech', methods=['POST'])
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
        
        log.info(f"OpenAI TTS: voice={voice}, speed={speed}, stream={stream}")
        
        payload = {"input": text, "voice": voice, "response_format": fmt, "speed": speed}
        try:
            resp = requests.post(f"{tts_url}/v1/audio/speech",
                                 json=payload, timeout=60, stream=stream)
            resp.raise_for_status()
        except Exception as e:
            log.error(f"TTS generation failed: {e}")
            return jsonify({"error": str(e)}), 500
        
        # Log to feed if enabled
        if feed_queue is not None:
            feed_queue.put({
                'type': 'openai_speech',
                'text': text[:100] + ('...' if len(text) > 100 else ''),
                'voice': voice,
                'speed': speed,
                'format': fmt,
                'stream': stream,
                'timestamp': time.time()
            })
        
        if stream:
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
            return Response(resp.content, mimetype=f'audio/{fmt}')
    
    print(f"Starting Ice Open TTS Proxy Server")
    print(f"  TTS Server: {tts_url}")
    print(f"  Listening: http://{host}:{port}")
    print(f"  Endpoint: POST /speak")
    if feed_queue is not None:
        print(f"  Feed: ENABLED (showing requests in real-time)")
    print()
    log.info(f"Server starting on {host}:{port}, TTS: {tts_url}")
    
    app.run(host=host, port=port, debug=False)


def play_audio(filepath: str):
    """Play audio file cross-platform."""
    import subprocess
    import platform
    
    system = platform.system()
    
    try:
        if system == "Windows":
            os.startfile(filepath)
        elif system == "Darwin":
            subprocess.run(["afplay", filepath], check=True)
        else:  # Linux
            for cmd in [["paplay", filepath], ["aplay", filepath], 
                        ["ffplay", "-nodisp", "-autoexit", filepath]]:
                try:
                    subprocess.run(cmd, check=True, capture_output=True)
                    return
                except (FileNotFoundError, subprocess.CalledProcessError):
                    continue
    except Exception as e:
        print(f"Audio playback error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Ice Open Ice Open TTS Proxy - Command-line TTS client/server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --text "Hello world" --save hello.mp3
  %(prog)s --speak "Task complete" --voice Carlotta
  %(prog)s --status
  %(prog)s --voices
  %(prog)s --server --port 8181
        """
    )
    
    # Input
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--text", help="Text to convert to speech")
    group.add_argument("--speak", help="Send text to proxy for playback")
    group.add_argument("--file", help="Read text from file")
    
    # Options
    parser.add_argument("--voice", default=_config.get("default_voice", "nova"), help="Voice name (default: nova)")
    parser.add_argument("--speed", type=float, default=_config.getfloat("speed", 1.0), help="Speed (0.25-4.0)")
    parser.add_argument("--format", default=_config.get("format", "mp3"), choices=["wav", "mp3", "opus", "flac"])
    parser.add_argument("--save", help="Output file path")
    
    # Commands
    parser.add_argument("--status", action="store_true", help="Check server status")
    parser.add_argument("--voices", action="store_true", help="List voices")
    parser.add_argument("--server", action="store_true", help="Run as API server")
    
    # Server config
    parser.add_argument("--tts-url", default=TTS_SERVER, help="TTS server URL")
    parser.add_argument("--proxy-url", default=PROXY_SERVER, help="Proxy server URL")
    parser.add_argument("--host", default=_config.get("api_host", "127.0.0.1"), help="Server host")
    parser.add_argument("--port", type=int, default=_config.getint("api_port", 8181), help="Server port")
    parser.add_argument("--feed", action="store_true", help="Show request feed in real-time (server mode only)")
    
    # Logging
    parser.add_argument("--log", action="store_true", help="Enable logging to logs/ folder")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level (default: INFO)")

    args = parser.parse_args()

    # Setup logging only if --log flag is provided
    if args.log:
        log_level = getattr(logging, args.log_level.upper())
        log_file = setup_logging(log_level)
        log = logging.getLogger("tts_proxy_cli")
        log.info(f"Starting CLI (log file: {log_file})")
    else:
        logging.disable(logging.CRITICAL)
        log = logging.getLogger("tts_proxy_cli")

    # Commands
    if args.status:
        check_status(args.tts_url, args.proxy_url)
        return

    if args.voices:
        list_voices(args.tts_url)
        return

    if args.server:
        if args.feed:
            # Server mode with feed: run server in background thread and show feed
            feed_queue = queue.Queue()
            server_thread = threading.Thread(target=run_server, args=(args.tts_url, args.host, args.port, feed_queue), daemon=True)
            server_thread.start()
            print("\n📡 Request Feed (press Ctrl+C to stop):")
            print("-" * 50)
            try:
                while True:
                    try:
                        msg = feed_queue.get(timeout=0.1)
                        timestamp = time.strftime("%H:%M:%S", time.localtime(msg.get('timestamp', time.time())))
                        msg_type = msg.get('type', 'unknown')
                        text = msg.get('text', '')
                        voice = msg.get('voice', 'unknown')
                        fmt = msg.get('format', 'wav')
                        stream = msg.get('stream', False)
                        
                        line = f"[{timestamp}] {voice} ({fmt}"
                        if stream:
                            line += ", stream"
                        line += f"): {text}"
                        print(line)
                    except queue.Empty:
                        continue
            except KeyboardInterrupt:
                print("\nFeed stopped.")
            # Wait a bit for server thread to finish (it's daemon, so will exit when main exits)
            time.sleep(0.5)
        else:
            run_server(args.tts_url, args.host, args.port)
        return
    
    # Get text
    text = None
    if args.text:
        text = args.text
    elif args.speak:
        text = args.speak
    elif args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            sys.exit(1)
    
    if not text:
        parser.print_help()
        print("\nError: No text provided. Use --text, --speak, or --file.")
        sys.exit(1)
    
    # Execute
    if args.speak:
        speak(text, args.voice, args.speed, args.proxy_url)
    else:
        generate_speech(text, args.voice, args.speed, args.format, 
                       args.save, args.tts_url)


if __name__ == "__main__":
    main()
