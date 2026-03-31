"""Voice loading and management."""
import logging
import os
import threading
from pathlib import Path
from typing import Set

import safetensors.torch
import soundfile as sf
import torch

from .config import settings
from .constants import Colors, DEFAULT_VOICES, VOICE_MAPPING
from .model_manager import model_manager

logger = logging.getLogger(__name__)

voice_lock = threading.Lock()


def load_custom_voices() -> Set[str]:
    custom_voices: Set[str] = set()
    tts_model = model_manager.model

    embeddings_path = Path(settings.embeddings_dir)
    if embeddings_path.exists():
        for f in embeddings_path.iterdir():
            if f.suffix.lower() == ".safetensors":
                voice_name = f.stem
                with voice_lock:
                    VOICE_MAPPING[voice_name] = str(f.resolve())
                custom_voices.add(voice_name)

    voices_path = Path(settings.voices_dir)
    if voices_path.exists():
        for f in voices_path.iterdir():
            if f.suffix.lower() == ".wav":
                voice_name = f.stem
                wav_path = str(f)
                st_path = embeddings_path / f"{voice_name}.safetensors"

                if voice_name not in custom_voices:
                    if tts_model is not None:
                        try:
                            logger.info(f"Exporting '{voice_name}' to embeddings/ for faster loading...")
                            audio, sr = sf.read(wav_path)
                            audio_pt = torch.from_numpy(audio).float()
                            if len(audio_pt.shape) == 1:
                                audio_pt = audio_pt.unsqueeze(0)
                            from pocket_tts.data.audio_utils import convert_audio
                            audio_resampled = convert_audio(audio_pt, sr, tts_model.config.mimi.sample_rate, 1)
                            with torch.no_grad():
                                prompt = tts_model._encode_audio(audio_resampled.unsqueeze(0).to(tts_model.device))
                            safetensors.torch.save_file({"audio_prompt": prompt.cpu()}, str(st_path))
                            logger.info(f"Exported '{voice_name}' to {st_path}")
                            with voice_lock:
                                VOICE_MAPPING[voice_name] = str(st_path.resolve())
                            custom_voices.add(voice_name)
                        except Exception as e:
                            logger.warning(f"Failed to auto-export voice '{voice_name}': {e}")
                            with voice_lock:
                                VOICE_MAPPING[voice_name] = str(f.resolve())
                            custom_voices.add(voice_name)
                    else:
                        with voice_lock:
                            VOICE_MAPPING[voice_name] = str(f.resolve())
                        custom_voices.add(voice_name)

    logger.info(f"{Colors.CYAN}{Colors.BOLD}Default voices available:{Colors.RESET}")
    logger.info(f"{Colors.CYAN}   OpenAI aliases: {', '.join(DEFAULT_VOICES['openai_aliases'])}{Colors.RESET}")
    logger.info(f"{Colors.CYAN}   Pocket TTS: {', '.join(DEFAULT_VOICES['pocket_tts'])}{Colors.RESET}")

    if custom_voices:
        logger.info(f"{Colors.GREEN}{Colors.BOLD}Custom voices loaded: {Colors.RESET}{Colors.GREEN}{', '.join(sorted(custom_voices))}{Colors.RESET}")
    else:
        logger.info(f"{Colors.YELLOW}No custom voices found in 'voices/' directory.{Colors.RESET}")

    return custom_voices
