"""Tests for Prometheus metrics."""

import time
from unittest.mock import patch
# Test that metrics module can be imported without prometheus_client
from photo_indexer import metrics


def test_metrics_import_without_prometheus():
    """Test that metrics module works without prometheus_client installed."""
    # The module should import successfully with no-op stubs
    assert metrics.PROMETHEUS_AVAILABLE is False
    assert metrics.INDEXED_TOTAL is not None
    assert metrics.INDEX_DURATION is not None


def test_noop_counters():
    """Test that no-op counters don't raise errors."""
    metrics.INDEXED_TOTAL.inc()
    metrics.INDEXED_TOTAL.inc(5)
    # No-op labels() takes *args not **kwargs
    metrics.INDEXED_TOTAL.labels("success").inc()
    metrics.ERRORS_TOTAL.inc()


def test_noop_gauges():
    """Test that no-op gauges don't raise errors."""
    metrics.set_assets_indexed(100)
    metrics.ASSETS_INDEXED.set(50)
    metrics.set_db_size_bytes(1024 * 1024)
    metrics.DB_SIZE_BYTES.set(2048)


def test_noop_histograms():
    """Test that no-op histograms don't raise errors."""
    metrics.INDEX_DURATION.observe(1.5)
    metrics.VLM_ANALYZE_DURATION.observe(2.3)


def test_index_timer_context_manager():
    """Test index_timer context manager."""
    with metrics.index_timer():
        time.sleep(0.01)
    # Should not raise


def test_vlm_timer_context_manager():
    """Test vlm_timer context manager."""
    with metrics.vlm_timer():
        time.sleep(0.01)
    # Should not raise


def test_start_metrics_server_noop():
    """Test start_metrics_server is no-op when prometheus_client missing."""
    # When PROMETHEUS_AVAILABLE is False, it should just return without error
    metrics.start_metrics_server(9090)
    # Should not raise


@patch("photo_indexer.metrics.PROMETHEUS_AVAILABLE", True)
@patch("photo_indexer.metrics.start_http_server")
def test_start_metrics_server_with_prometheus(mock_start_server):
    """Test start_metrics_server calls start_http_server when available."""
    metrics.start_metrics_server(9090)
    mock_start_server.assert_called_once_with(9090)