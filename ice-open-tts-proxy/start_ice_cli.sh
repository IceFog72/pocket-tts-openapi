#!/bin/bash
# Start Ice Open TTS Proxy CLI Server (Linux/macOS)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
[ -d "../venv" ] && source ../venv/bin/activate
python ice_open_tts_proxy_cli.py --server ${1:+--port "$1"}
