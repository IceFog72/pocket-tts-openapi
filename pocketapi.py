#!/usr/bin/env python3
"""Pocket TTS API Server - Entry point.

This is a thin wrapper that imports from the pocket_tts_server package.
For direct use: python pocketapi.py
For import: from pocket_tts_server import app
"""
import logging
import os
import sys

logging.basicConfig(level=logging.INFO)

from pocket_tts_server import app
from pocket_tts_server.api import main

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Server stopped by user")
    except Exception as e:
        logging.exception("Failed to start server")
