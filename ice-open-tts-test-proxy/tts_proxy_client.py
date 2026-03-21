#!/usr/bin/env python3
"""
TTS Proxy Client - CLI for interacting with TTS Proxy

Usage:
    # Generate speech and save to file
    python tts_proxy_client.py --text "Hello world" --save output.mp3
    
    # Send to TTS Proxy for immediate playback
    python tts_proxy_client.py --speak "Hello from AI agent"
    
    # Check server status
    python tts_proxy_client.py --status

This client works with:
- TTS Proxy (port 8181) - for playback via proxy
- Pocket TTS Server (port 8005) - direct generation
"""

import sys
import os
import argparse
import tempfile
import json
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error: requests package required. Install with: pip install requests")
    sys.exit(1)


from tts_backend import Config
_config = Config()

# Read from config.ini (created by Config with defaults if missing)
TTS_SERVER_URL = os.environ.get("TTS_SERVER_URL", _config.get("tts_server_url"))
GUI_APP_URL = os.environ.get("GUI_APP_URL", f"http://{_config.get('api_host')}:{_config.get('api_port')}")


def check_tts_server(url: str) -> bool:
    """Check if TTS server is running."""
    try:
        resp = requests.get(f"{url}/health", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            print(f"✓ TTS Server: {url}")
            print(f"  Model loaded: {data.get('model_loaded', False)}")
            print(f"  Voice cloning: {data.get('voice_cloning', False)}")
            return True
        return False
    except Exception as e:
        print(f"✗ Cannot connect to TTS server: {e}")
        return False


def check_gui_app(url: str) -> bool:
    """Check if GUI app is running."""
    try:
        resp = requests.get(f"{url}/health", timeout=3)
        if resp.status_code == 200:
            print(f"✓ GUI App: {url}")
            return True
        return False
    except:
        print(f"✗ GUI App not running at {url}")
        return False


def get_voices(url: str):
    """List available voices."""
    try:
        resp = requests.get(f"{url}/v1/voices", timeout=5)
        resp.raise_for_status()
        voices = resp.json().get("voices", [])
        print(f"\nAvailable voices ({len(voices)}):")
        for v in voices:
            print(f"  - {v}")
    except Exception as e:
        print(f"Error getting voices: {e}")


def generate_speech(url: str, text: str, voice: str = "nova", 
                    speed: float = 1.0, format: str = "wav",
                    output_file: str = None, play: bool = False):
    """Generate speech from text."""
    payload = {
        "input": text,
        "voice": voice,
        "response_format": format,
        "speed": speed
    }
    
    print(f"Generating speech with voice '{voice}'...")
    
    try:
        resp = requests.post(f"{url}/v1/audio/speech", json=payload, timeout=60)
        resp.raise_for_status()
        audio_data = resp.content
        
        if output_file:
            with open(output_file, 'wb') as f:
                f.write(audio_data)
            print(f"✓ Saved to: {output_file} ({len(audio_data)} bytes)")
        elif play:
            # Save to temp and play
            suffix = f".{format}"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(audio_data)
                temp_path = f.name
            print(f"Audio saved to: {temp_path}")
            print("(Use a media player to play this file)")
        else:
            # Default: save to current directory
            output_file = f"tts_output.{format}"
            with open(output_file, 'wb') as f:
                f.write(audio_data)
            print(f"✓ Saved to: {output_file} ({len(audio_data)} bytes)")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def speak_to_gui(text: str, voice: str = "nova", speed: float = 1.0,
                 gui_url: str = GUI_APP_URL):
    """Send text to GUI app for playback."""
    payload = {
        "text": text,
        "voice": voice,
        "speed": speed
    }
    
    print(f"Sending to GUI app: '{text[:50]}...'")
    
    try:
        resp = requests.post(f"{gui_url}/speak", json=payload, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        print(f"✓ {result.get('message', 'Success')}")
    except Exception as e:
        print(f"Error: {e}")
        print(f"Make sure the GUI app is running: python tts_gui_app.py")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="TTS Client - Generate or play speech",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate speech and save to file
  %(prog)s --text "Hello world" --voice nova --save output.mp3
  
  # List available voices
  %(prog)s --voices
  
  # Send to GUI app for playback
  %(prog)s --speak "Hello from AI agent"
  
  # Check server status
  %(prog)s --status
  
  # Use from another script
  python -c "from tts_client import speak_to_gui; speak_to_gui('Hello')"
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--text", help="Text to convert to speech")
    input_group.add_argument("--file", help="File containing text to convert")
    input_group.add_argument("--speak", help="Send text to GUI app for playback")
    
    # Options
    parser.add_argument("--voice", default=_config.get("default_voice", "nova"), help="Voice to use (default: nova)")
    parser.add_argument("--speed", type=float, default=_config.getfloat("speed", 1.0), help="Speech speed (0.25-4.0)")
    parser.add_argument("--format", default=_config.get("format", "wav"), choices=["wav", "mp3", "opus", "flac"],
                        help="Audio format (default: wav)")
    parser.add_argument("--save", help="Save output to file")
    parser.add_argument("--play", action="store_true", help="Show temp file path for playback")
    
    # Server options
    parser.add_argument("--tts-url", default=TTS_SERVER_URL,
                        help=f"TTS server URL (default: {TTS_SERVER_URL})")
    parser.add_argument("--gui-url", default=GUI_APP_URL,
                        help=f"GUI app URL (default: {GUI_APP_URL})")
    
    # Info options
    parser.add_argument("--voices", action="store_true", help="List available voices")
    parser.add_argument("--status", action="store_true", help="Check server status")
    
    args = parser.parse_args()
    
    # Info commands
    if args.status:
        print("Checking servers...\n")
        check_tts_server(args.tts_url)
        print()
        check_gui_app(args.gui_url)
        return
    
    if args.voices:
        get_voices(args.tts_url)
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
        print("\nError: No text provided. Use --text, --file, or --speak.")
        sys.exit(1)
    
    # Execute command
    if args.speak:
        speak_to_gui(text, args.voice, args.speed, args.gui_url)
    else:
        generate_speech(args.tts_url, text, args.voice, args.speed,
                       args.format, args.save, args.play)


if __name__ == "__main__":
    main()
