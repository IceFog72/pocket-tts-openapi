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
import configparser
from pathlib import Path
from typing import Optional, Dict, Any, Callable

# =============================================================================
# Configuration
# =============================================================================

class Config:
    """Configuration manager using INI file in the same directory as the script."""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            # Determine the directory of this script
            try:
                script_dir = os.path.dirname(os.path.abspath(__file__))
            except NameError:
                # Fallback to current working directory
                script_dir = os.getcwd()
            config_path = os.path.join(script_dir, "config.ini")
        self.config_path = config_path
        self.parser = configparser.ConfigParser()
        if not os.path.exists(self.config_path):
            self._create_default()
        self.parser.read(self.config_path)
    
    def _create_default(self):
        """Create a default configuration file."""
        self.parser['server'] = {
            'tts_server_url': 'http://localhost:8005',
            'api_host': '127.0.0.1',
            'api_port': '8181',
            'default_voice': 'nova',
            'speed': '1.0',
            'format': 'wav'
        }
        # Also add a section for logging if needed
        self.parser['logging'] = {
            'enabled': 'False',
            'level': 'INFO'
        }
        with open(self.config_path, 'w') as f:
            self.parser.write(f)
    
    def get(self, key: str, default=None):
        """Get a configuration value as a string."""
        try:
            return self.parser.get('server', key)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return default
    
    def getint(self, key: str, default=0):
        """Get a configuration value as an integer."""
        try:
            return self.parser.getint('server', key)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            return default
    
    def getfloat(self, key: str, default=0.0):
        """Get a configuration value as a float."""
        try:
            return self.parser.getfloat('server', key)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            return default
    
    def getboolean(self, key: str, default=False):
        """Get a configuration value as a boolean."""
        try:
            return self.parser.getboolean('server', key)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            return default
    
    def set(self, key: str, value):
        """Set a configuration value and save to file."""
        self.parser.set('server', key, str(value))
        self.save()
    
    def save(self):
        """Save the configuration to the INI file."""
        with open(self.config_path, 'w') as f:
            self.parser.write(f)

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

def get_process_name(pid: str) -> Optional[str]:
    """Get process name given a PID."""
    import subprocess
    try:
        if sys.platform == "win32":
            # Use tasklist to get the image name
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True
            )
            lines = result.stdout.strip().split('\n')
            # Skip the header lines
            if len(lines) >= 3:
                # The process name is in the first column of the third line
                parts = lines[2].split()
                if parts:
                    return parts[0]
        else:
            # Linux/macOS: use ps
            result = subprocess.run(
                ["ps", "-p", pid, "-o", "comm="], capture_output=True, text=True
            )
            name = result.stdout.strip()
            if name:
                return name
    except Exception:
        pass
    return None

def handle_port_conflict(port: int, service_name: str = "Service") -> int:
    """Handle port conflict by prompting user for action."""
    if not check_port_in_use(port):
        return port
    
    pid = get_pid_on_port(port)
    proc_name = get_process_name(pid) if pid else None
    display_name = proc_name if proc_name else service_name
    print(f"Port {port} is already in use by {display_name}.")
    if pid:
        print(f"  PID: {pid}")
    
    is_our_app = False
    if proc_name and any(kw in proc_name.lower() for kw in ["python", "pocket", "ice"]):
        is_our_app = True

    print("Options:")
    if is_our_app:
        print("  1. Kill the existing process and use this port (Restart Server)")
    print("  2. Use a different port")
    print("  3. Exit")
    
    while True:
        choice = input("Enter choice: ").strip()
        if choice == "1":
            if not is_our_app:
                print("Invalid choice (kill restricted for external processes).")
                continue
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