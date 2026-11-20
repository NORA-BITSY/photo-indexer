"""Prometheus metrics for photo indexer CLI."""

from __future__ import annotations

from collections.abc import Generator
import time

from contextlib import contextmanager

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

    # No-op stubs so code compiles without prometheus-client installed
    class _NoOp:
        def __init__(self, *a, **k): pass

        def __getattr__(self, name): return self

        def __call__(self, *a, **k): return self

        def __enter__(self): return self

        def __exit__(self, *a): return False

        def observe(self, *a): pass

        def inc(self, *a): pass

        def dec(self, *a): pass

        def set(self, *a): pass

        def labels(self, *a): return self

    Counter = _NoOp  # type: ignore
    Gauge = _NoOp  # type: ignore
    Histogram = _NoOp  # type: ignore

    def start_http_server(*a, **k):  # type: ignore
        raise RuntimeError("prometheus_client not installed")


# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------
INDEXED_TOTAL = Counter(
    "photo_indexer_indexed_total",
    "Total assets indexed",
    ["status"],
)
VLM_ANALYZED_TOTAL = Counter(
    "photo_indexer_vlm_analyzed_total",
    "Total VLM analyses performed",
    ["status"],
)
SIDECAR_WRITTEN_TOTAL = Counter(
    "photo_indexer_sidecar_written_total",
    "Total XMP sidecars written",
    ["status"],
)
FACES_DETECTED_TOTAL = Counter(
    "photo_indexer_faces_detected_total",
    "Total faces detected",
    ["model"],
)
ERRORS_TOTAL = Counter(
    "photo_indexer_errors_total",
    "Total errors encountered",
    ["command"],
)

# ---------------------------------------------------------------------------
# Gauges
# ---------------------------------------------------------------------------
ASSETS_INDEXED = Gauge(
    "photo_indexer_assets_indexed",
    "Number of assets currently indexed",
)
DB_SIZE_BYTES = Gauge(
    "photo_indexer_db_size_bytes",
    "Size of index database in bytes",
)

# ---------------------------------------------------------------------------
# Histograms
# ---------------------------------------------------------------------------
INDEX_DURATION = Histogram(
    "photo_indexer_index_duration_seconds",
    "Time spent indexing assets",
)
VLM_ANALYZE_DURATION = Histogram(
    "photo_indexer_vlm_analyze_duration_seconds",
    "Time spent on VLM analysis",
)


@contextmanager
def index_timer() -> Generator[None, None, None]:
    start = time.monotonic()
    try:
        yield
    finally:
        INDEX_DURATION.observe(time.monotonic() - start)


@contextmanager
def vlm_timer() -> Generator[None, None, None]:
    start = time.monotonic()
    try:
        yield
    finally:
        VLM_ANALYZE_DURATION.observe(time.monotonic() - start)


def start_metrics_server(port: int = 9090) -> None:
    """Start Prometheus HTTP metrics server. No-op if prometheus_client missing."""
    if PROMETHEUS_AVAILABLE:
        start_http_server(port)


def set_assets_indexed(count: int) -> None:
    ASSETS_INDEXED.set(count)


def set_db_size_bytes(size: int) -> None:
    DB_SIZE_BYTES.set(size)
