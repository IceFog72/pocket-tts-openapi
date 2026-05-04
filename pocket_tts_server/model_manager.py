"""Thread-safe model manager."""
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

from .config import settings

logger = logging.getLogger(__name__)


@dataclass
class ModelManager:
    _model: Any = field(default=None, repr=False)
    _device: Optional[str] = None
    _sample_rate: Optional[int] = None
    _language: Optional[str] = None
    _lock: threading.RLock = field(default_factory=threading.RLock)
    _loading: bool = False
    _load_event: threading.Event = field(default_factory=threading.Event)

    @property
    def model(self):
        with self._lock:
            return self._model

    @property
    def device(self) -> Optional[str]:
        with self._lock:
            return self._device

    @property
    def sample_rate(self) -> int:
        with self._lock:
            return self._sample_rate or settings.default_sample_rate

    @property
    def language(self) -> Optional[str]:
        with self._lock:
            return self._language

    @property
    def is_loaded(self) -> bool:
        with self._lock:
            return self._model is not None

    def acquire_lock(self):
        self._lock.acquire()

    def release_lock(self):
        self._lock.release()

    def load(self, timeout: int = settings.model_load_timeout, language: Optional[str] = None) -> None:
        target_language = language or settings.language
        
        import pocket_tts.data.audio as pt_audio
        if not hasattr(pt_audio, "_patched_audio_read"):
            orig_audio_read = pt_audio.audio_read
            def patched_audio_read(filepath):
                from pathlib import Path
                filepath = Path(filepath)
                if filepath.suffix.lower() == ".wav":
                    try:
                        import wave
                        with wave.open(str(filepath), "rb") as f:
                            pass # Just test if it opens
                    except Exception as e:
                        if "unknown format: 3" in str(e) or "wave" in str(e).lower():
                            import soundfile as sf
                            import torch
                            data, sample_rate = sf.read(str(filepath), dtype="float32")
                            if data.ndim == 1:
                                wav = torch.from_numpy(data).unsqueeze(0)
                            else:
                                wav = torch.from_numpy(data.mean(axis=1)).unsqueeze(0)
                            return wav, sample_rate
                return orig_audio_read(filepath)
            
            pt_audio.audio_read = patched_audio_read
            pt_audio._patched_audio_read = True
            
            # Also patch tts_model's namespace if already imported
            import sys
            if "pocket_tts.models.tts_model" in sys.modules:
                sys.modules["pocket_tts.models.tts_model"].audio_read = patched_audio_read

        from pocket_tts import TTSModel

        self._lock.acquire()
        try:
            if self._model is not None:
                if self._language == target_language:
                    return
                else:
                    logger.info(f"Switching language from {self._language} to {target_language}")
                    del self._model
                    self._model = None
                    if self._device == "cuda" and torch.cuda.is_available():
                        torch.cuda.empty_cache()
            if self._loading:
                # Another thread is loading — wait outside the lock
                self._lock.release()
                try:
                    if not self._load_event.wait(timeout=timeout):
                        raise TimeoutError("Model loading timed out")
                finally:
                    self._lock.acquire()
                return
            self._loading = True
            self._load_event.clear()
        finally:
            self._lock.release()

        try:
            logger.info(f"Loading TTS model (timeout: {timeout}s)...")
            load_result: Dict[str, Any] = {"model": None, "error": None}

            def _do_load():
                try:
                    # Patch again just in case it was imported during another thread
                    import sys
                    if "pocket_tts.models.tts_model" in sys.modules:
                        sys.modules["pocket_tts.models.tts_model"].audio_read = pt_audio.audio_read
                        
                    load_result["model"] = TTSModel.load_model(language=target_language)
                except Exception as e:
                    load_result["error"] = e

            load_thread = threading.Thread(target=_do_load, daemon=True)
            load_thread.start()
            load_thread.join(timeout=timeout)

            if load_thread.is_alive():
                raise TimeoutError(f"Model loading exceeded {timeout}s timeout")
            if load_result["error"]:
                raise load_result["error"]

            with self._lock:
                self._model = load_result["model"]

                self._device = self._model.device
                self._sample_rate = getattr(self._model, "sample_rate", settings.default_sample_rate)
                self._language = target_language
                self._loading = False
                self._load_event.set()

            logger.info(f"Pocket TTS loaded | Device: {self._device} | Sample Rate: {self._sample_rate} | Language: {self._language}")

        except Exception as e:
            with self._lock:
                self._loading = False
                self._load_event.set()
            logger.error(f"Failed to load TTS model: {e}")
            raise

    def move_to_device(self, target_device: str) -> None:
        with self._lock:
            if self._model is None:
                return
            if self._device != target_device:
                logger.info(f"Moving model from {self._device} to {target_device}")
                self._model.to(target_device)
                self._device = target_device
                if target_device == "cpu":
                    torch.cuda.empty_cache()

    def shutdown(self) -> None:
        with self._lock:
            if self._model is not None:
                logger.info("Unloading TTS model...")
                del self._model
                self._model = None
                self._device = None
                self._language = None


model_manager = ModelManager()
