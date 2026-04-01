"""Thread-safe model manager."""
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .config import settings

logger = logging.getLogger(__name__)


@dataclass
class ModelManager:
    _model: Any = field(default=None, repr=False)
    _device: Optional[str] = None
    _sample_rate: Optional[int] = None
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
    def is_loaded(self) -> bool:
        with self._lock:
            return self._model is not None

    def acquire_lock(self):
        self._lock.acquire()

    def release_lock(self):
        self._lock.release()

    def load(self, timeout: int = settings.model_load_timeout) -> None:
        from pocket_tts import TTSModel

        self._lock.acquire()
        try:
            if self._model is not None:
                return
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
                    load_result["model"] = TTSModel.load_model()
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

                if not hasattr(TTSModel, "_slice_kv_cache"):
                    logger.info("Patching TTSModel with missing _slice_kv_cache method")
                    setattr(TTSModel, "_slice_kv_cache", _slice_kv_cache)

                self._device = self._model.device
                self._sample_rate = getattr(self._model, "sample_rate", settings.default_sample_rate)
                self._loading = False
                self._load_event.set()

            logger.info(f"Pocket TTS loaded | Device: {self._device} | Sample Rate: {self._sample_rate}")

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

    def shutdown(self) -> None:
        with self._lock:
            if self._model is not None:
                logger.info("Unloading TTS model...")
                del self._model
                self._model = None
                self._device = None


def _slice_kv_cache(self, model_state: dict, sequence_length: int) -> None:
    for module_name, module_state in model_state.items():
        if "cache" in module_state:
            cache = module_state["cache"]
            if cache.shape[2] > sequence_length:
                cache = cache[:, :, :sequence_length, :, :].contiguous()
            if cache.shape[3] > sequence_length:
                cache = cache[:, :, :, :sequence_length, :].contiguous()
            module_state["cache"] = cache


model_manager = ModelManager()
