#!/bin/bash
# Start Ice Open TTS Proxy GUI (Linux/macOS)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
[ -d "../venv" ] && source ../venv/bin/activate
python ice_open_tts_proxy.py ${1:+--port "$1"}
