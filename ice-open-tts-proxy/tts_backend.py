"""
Shared backend for Ice Open TTS Proxy (GUI and CLI).
Contains common classes and utilities.
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

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CONFIG = {
    "tts_server_url": "http://localhost:8001",
    "api_port": 5000,
    "api_host": "127.0.0.1",
    "default_voice": "nova",
    "default_speed": 1.0,
}

class Config:
    """Configuration management."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
    
    def _get_default_config_path(self) -> str:
        """Get default configuration file path."""
        return str(Path.home() / ".ice_open_tts_proxy_config.json")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                # Merge with defaults to ensure all keys exist
                merged_config = DEFAULT_CONFIG.copy()
                merged_config.update(config)
                return merged_config
            except Exception as e:
                print(f"Warning: Could not load config file: {e}")
                print("Using default configuration.")
                return DEFAULT_CONFIG.copy()
        else:
            # No config file, use defaults
            return DEFAULT_CONFIG.copy()
    
    def save(self):
        """Save current configuration to file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save config file: {e}")
    
    def get(self, key: str, default=None):
        """Get configuration value."""
        return self.config.get(key, default)
    
    def set(self, key: str, value):
        """Set configuration value."""
        self.config[key] = value

# =============================================================================
# TTS Client
# =============================================================================

logger = logging.getLogger("tts_proxy")

class TTSClient:
    """Client for communicating with Pocket TTS server."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
    
    def get_voices(self) -> list:
        """Get available voices from TTS server."""
        log = logging.getLogger("tts_proxy.client")
        try:
            log.debug(f"Fetching voices from {self.base_url}")
            import requests
            resp = requests.get(f"{self.base_url}/v1/voices", timeout=5)
            resp.raise_for_status()
            voices = resp.json().get("voices", [])
            log.info(f"Fetched {len(voices)} voices")
            return voices
        except Exception as e:
            log.error(f"Error fetching voices: {e}")
            # Fallback voices
            return ["nova", "alloy", "echo", "fable", "onyx", "shimmer"]
    
    def generate_speech(self, text: str, voice: str, speed: float = 1.0, 
                        format: str = "wav") -> Optional[bytes]:
        """Generate speech and return audio bytes."""
        log = logging.getLogger("tts_proxy.client")
        payload = {
            "input": text,
            "voice": voice,
            "response_format": format,
            "speed": speed
        }
        try:
            log.info(f"Generating speech: voice={voice}, speed={speed}, text_len={len(text)}")
            start_time = time.time()
            import requests
            resp = requests.post(
                f"{self.base_url}/v1/audio/speech",
                json=payload,
                timeout=60
            )
            resp.raise_for_status()
            elapsed = time.time() - start_time
            log.info(f"Speech generated: {len(resp.content)} bytes in {elapsed:.2f}s")
            return resp.content
        except Exception as e:
            log.error(f"Error generating speech: {e}")
            return None
    
    def health_check(self) -> bool:
        """Check if TTS server is running."""
        log = logging.getLogger("tts_proxy.client")
        try:
            import requests
            resp = requests.get(f"{self.base_url}/health", timeout=3)
            healthy = resp.status_code == 200
            log.debug(f"Health check: {'OK' if healthy else 'FAILED'}")
            return healthy
        except Exception as e:
            log.warning(f"Health check failed: {e}")
            return False

# =============================================================================
# Port Checking Utilities
# =============================================================================

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
                    if parts:
                        return parts[-1]
        else:
            # Linux/macOS
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"], capture_output=True, text=True
            )
            if result.stdout.strip():
                return result.stdout.strip().split()[0]
    except Exception:
        pass
    return None

def handle_port_conflict(port: int, service_name: str = "Service") -> int:
    """Handle port conflict by prompting user for action."""
    if not check_port_in_use(port):
        return port
    
    pid = get_pid_on_port(port)
    print(f"Port {port} is already in use by {service_name}.")
    if pid:
        print(f"  PID: {pid}")
    
    print("Options:")
    print("  1. Kill the existing process and use this port")
    print("  2. Use a different port")
    print("  3. Exit")
    
    while True:
        choice = input("Enter choice (1-3): ").strip()
        if choice == "1":
            if pid:
                try:
                    if sys.platform == "win32":
                        subprocess.run(["taskkill", "/PID", pid, "/F"], check=True)
                    else:
                        os.kill(int(pid), 9)
                    print(f"Killed process {pid}")
                    return port
                except Exception as e:
                    print(f"Failed to kill process: {e}")
                    print("Trying different port...")
            else:
                print("Could not determine PID, trying different port...")
        elif choice == "2":
            while True:
                try:
                    new_port = int(input("Enter new port: ").strip())
                    if 1 <= new_port <= 65535:
                        if not check_port_in_use(new_port):
                            return new_port
                        else:
                            print(f"Port {new_port} is also in use. Try again.")
                    else:
                        print("Port must be between 1 and 65535.")
                except ValueError:
                    print("Please enter a valid port number.")
        elif choice == "3":
            print("Exiting.")
            sys.exit(0)
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

# =============================================================================
# Logging Configuration
# =============================================================================

def setup_logging(log_level=logging.INFO) -> Path:
    """Configure logging with file and console handlers.
    Returns the log file path.
    """
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # File handler - daily log file
    log_file = log_dir / f"tts_proxy_{time.strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Configure root logger for our modules
    root_logger = logging.getLogger("tts_proxy")
    root_logger.setLevel(logging.DEBUG)
    # Clear any existing handlers
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return log_file

# Prevent duplicate logging if module is imported multiple times
logging.getLogger("tts_proxy").handlers = []