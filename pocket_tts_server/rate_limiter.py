"""Token bucket rate limiter per client IP."""
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .config import settings


@dataclass
class RateLimiter:
    requests_per_window: int = settings.rate_limit_requests
    window_seconds: int = settings.rate_limit_window
    _requests: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def is_allowed(self, client_ip: str) -> Tuple[bool, int]:
        now = time.time()
        with self._lock:
            cutoff = now - self.window_seconds
            self._requests[client_ip] = [t for t in self._requests[client_ip] if t > cutoff]
            if len(self._requests[client_ip]) >= self.requests_per_window:
                oldest = min(self._requests[client_ip])
                retry_after = int(oldest + self.window_seconds - now) + 1
                return False, max(retry_after, 1)
            self._requests[client_ip].append(now)
            return True, 0

    def cleanup(self) -> None:
        now = time.time()
        cutoff = now - self.window_seconds
        with self._lock:
            keys_to_remove = []
            for ip, timestamps in self._requests.items():
                self._requests[ip] = [t for t in timestamps if t > cutoff]
                if not self._requests[ip]:
                    keys_to_remove.append(ip)
            for ip in keys_to_remove:
                del self._requests[ip]


rate_limiter = RateLimiter()
