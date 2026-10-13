"""Thread-safe model manager."""
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from pathlib import Path

import torch

from .config import settings

logger = logging.getLogger(__name__)

_audio_read_patch_lock = threading.Lock()
_audio_read_patched = False


@dataclass
class ModelManager:
    _model: Any = field(default=None, repr=False)
    _device: Optional[str] = None
    _sample_rate: Optional[int] = None
    _language: Optional[str] = None
    _lock: threading.RLock = field(default_factory=threading.RLock)

    @property
    def model(self):
        with self._lock:
            return self._model

    @property
    def device(self) -> Optional[str]:
        return self._device

    @property
    def sample_rate(self) -> int:
        return self._sample_rate or settings.default_sample_rate

    @property
    def language(self) -> Optional[str]:
        return self._language

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def voice_cloning(self) -> bool:
        # Metadata reads must not wait behind an entire synthesis. A local
        # reference stays valid if another thread replaces the model.
        model = self._model
        return bool(getattr(model, "has_voice_cloning", False))

    def acquire_lock(self, timeout: float = -1) -> bool:
        return self._lock.acquire(timeout=timeout)

    def release_lock(self):
        self._lock.release()

    def load(self, timeout: int = settings.model_load_timeout, language: Optional[str] = None) -> None:
        target_language = language or settings.language

        global _audio_read_patched
        import pocket_tts.data.audio as pt_audio
        import pocket_tts.models.tts_model as pt_model
        with _audio_read_patch_lock:
            if not _audio_read_patched:
                orig_audio_read = pt_audio.audio_read
                def patched_audio_read(filepath: str | Path) -> tuple[torch.Tensor, int]:
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
                pt_model.audio_read = patched_audio_read
                _audio_read_patched = True

        from pocket_tts import TTSModel

        # Keep selection, loading and publication under the same ownership as
        # synthesis. The loader thread never accesses this manager's lock.
        with self._lock:
            if self._model is not None and self._language == target_language:
                return
            logger.info("Loading TTS model %s (timeout: %ss)...", target_language, timeout)
            load_result: Dict[str, Any] = {"model": None, "error": None}

            def _do_load():
                try:
                    load_result["model"] = TTSModel.load_model(language=target_language)
                except Exception as exc:
                    load_result["error"] = exc

            load_thread = threading.Thread(target=_do_load, daemon=True)
            load_thread.start()
            load_thread.join(timeout=timeout)
            if load_thread.is_alive():
                raise TimeoutError(f"Model loading exceeded {timeout}s timeout")
            if load_result["error"] is not None:
                raise load_result["error"]

            # Validate before replacing the previous working model. A failed or
            # timed-out loader cannot publish a result after this method exits.
            new_model = load_result["model"]
            device = new_model.device
            sample_rate = getattr(new_model, "sample_rate", settings.default_sample_rate)
            self._model = new_model
            self._device = device
            self._sample_rate = sample_rate
            self._language = target_language
            logger.info("Pocket TTS loaded | Device: %s | Sample Rate: %s | Language: %s",
                        device, sample_rate, target_language)

    def move_to_device(self, target_device: str) -> None:
        with self._lock:
            if self._model is None:
                return
            if self._device != target_device:
                logger.info(f"Moving model from {self._device} to {target_device}")
                self._model.to(target_device)
                self._device = target_device
                if target_device == "cpu":
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    def shutdown(self) -> None:
        with self._lock:
            if self._model is not None:
                logger.info("Unloading TTS model...")
                del self._model
                self._model = None
                self._device = None
                self._language = None
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


model_manager = ModelManager()
