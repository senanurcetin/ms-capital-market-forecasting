"""Request guards and metrics for the serving API - standard library only.

The API image is kept slim on purpose (see requirements-serve.txt), so rate limiting and the
Prometheus text format are written out here rather than pulling in slowapi or
prometheus_client for forty lines of logic.

Both are per-process: behind several workers or replicas each keeps its own counts. That is
the right trade for a single-container research service; a shared limit belongs in the
reverse proxy in front of it.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    """Sliding-window limit of `per_minute` requests per client key. 0 disables it."""

    # Bounds memory when many distinct clients show up: past this, idle keys are dropped.
    MAX_KEYS = 10_000

    def __init__(self, per_minute: int, window: float = 60.0, clock=time.monotonic):
        self.per_minute = per_minute
        self.window = window
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> float | None:
        """None if the request is allowed, else the seconds until it would be."""
        if self.per_minute <= 0:
            return None
        now = self._clock()
        with self._lock:
            if len(self._hits) >= self.MAX_KEYS and key not in self._hits:
                self._evict(now)
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= self.per_minute:
                return max(self.window - (now - hits[0]), 0.0)
            hits.append(now)
            return None

    def _evict(self, now: float) -> None:
        for k in [k for k, h in self._hits.items() if not h or now - h[-1] >= self.window]:
            del self._hits[k]


class Metrics:
    """Request counts, latency sums and scored rows, rendered as Prometheus text."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests: dict[tuple[str, str, int], int] = defaultdict(int)
        self._latency_sum: dict[str, float] = defaultdict(float)
        self._latency_count: dict[str, int] = defaultdict(int)
        self.rows_scored = 0
        self._out_of_range: dict[str, int] = defaultdict(int)

    def observe(self, method: str, path: str, status: int, seconds: float) -> None:
        with self._lock:
            self._requests[(method, path, status)] += 1
            self._latency_sum[path] += seconds
            self._latency_count[path] += 1

    def add_rows(self, n: int) -> None:
        with self._lock:
            self.rows_scored += n

    def add_out_of_range(self, counts: dict[str, int]) -> None:
        with self._lock:
            for feature, n in counts.items():
                self._out_of_range[feature] += n

    def render(self) -> str:
        with self._lock:
            lines = [
                "# HELP mscapital_http_requests_total HTTP requests by route and status.",
                "# TYPE mscapital_http_requests_total counter",
            ]
            for (method, path, status), n in sorted(self._requests.items()):
                lines.append(
                    f'mscapital_http_requests_total{{method="{method}",path="{path}",'
                    f'status="{status}"}} {n}'
                )
            lines += [
                "# HELP mscapital_http_request_seconds Time spent handling a request.",
                "# TYPE mscapital_http_request_seconds summary",
            ]
            for path in sorted(self._latency_count):
                lines.append(
                    f'mscapital_http_request_seconds_sum{{path="{path}"}} '
                    f"{self._latency_sum[path]:.6f}"
                )
                lines.append(
                    f'mscapital_http_request_seconds_count{{path="{path}"}} '
                    f"{self._latency_count[path]}"
                )
            lines += [
                "# HELP mscapital_rows_scored_total Feature rows scored by the model.",
                "# TYPE mscapital_rows_scored_total counter",
                f"mscapital_rows_scored_total {self.rows_scored}",
                "# HELP mscapital_out_of_range_values_total Values outside the band the model "
                "was trained on, by feature.",
                "# TYPE mscapital_out_of_range_values_total counter",
            ]
            for feature, n in sorted(self._out_of_range.items()):
                lines.append(f'mscapital_out_of_range_values_total{{feature="{feature}"}} {n}')
        return "\n".join(lines) + "\n"
