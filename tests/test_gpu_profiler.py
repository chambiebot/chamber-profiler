"""Tests for src.profiler.gpu_profiler."""

import time
import threading
from unittest.mock import patch, MagicMock

import pytest

from src.profiler.gpu_profiler import (
    GPUProfiler,
    GPUMetricSnapshot,
    ProfileResult,
    _safe_float,
)


# ---------------------------------------------------------------------------
# GPUMetricSnapshot
# ---------------------------------------------------------------------------

class TestGPUMetricSnapshot:
    def test_creation(self):
        snap = GPUMetricSnapshot(
            timestamp=1000.0,
            gpu_index=0,
            utilization_pct=85.0,
            memory_used_mb=4096.0,
            memory_total_mb=8192.0,
            power_watts=250.0,
            temperature_c=72.0,
            sm_activity_pct=80.0,
            memory_bandwidth_pct=60.0,
            clock_speed_mhz=1500.0,
        )
        assert snap.gpu_index == 0
        assert snap.utilization_pct == 85.0
        assert snap.memory_used_mb == 4096.0

    def test_to_dict_from_dict_roundtrip(self):
        snap = GPUMetricSnapshot(
            timestamp=1000.0,
            gpu_index=1,
            utilization_pct=50.0,
            memory_used_mb=2048.0,
            memory_total_mb=8192.0,
            power_watts=200.0,
            temperature_c=65.0,
            sm_activity_pct=45.0,
            memory_bandwidth_pct=30.0,
            clock_speed_mhz=1200.0,
        )
        d = snap.to_dict()
        restored = GPUMetricSnapshot.from_dict(d)
        assert restored == snap

    def test_frozen(self):
        snap = GPUMetricSnapshot(
            timestamp=1.0, gpu_index=0, utilization_pct=0.0,
            memory_used_mb=0.0, memory_total_mb=0.0, power_watts=0.0,
            temperature_c=0.0, sm_activity_pct=0.0,
            memory_bandwidth_pct=0.0, clock_speed_mhz=0.0,
        )
        with pytest.raises(AttributeError):
            snap.gpu_index = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ProfileResult
# ---------------------------------------------------------------------------

class TestProfileResult:
    def test_empty_snapshots(self):
        result = ProfileResult(duration_s=1.0, snapshots=[], gpu_count=0)
        assert result.avg_utilization == 0.0
        assert result.peak_memory_mb == 0.0
        assert result.avg_power_watts == 0.0

    def test_computed_stats(self):
        snaps = [
            GPUMetricSnapshot(
                timestamp=i, gpu_index=0, utilization_pct=float(i * 20),
                memory_used_mb=float(i * 1000), memory_total_mb=8000.0,
                power_watts=float(100 + i * 10), temperature_c=60.0,
                sm_activity_pct=0.0, memory_bandwidth_pct=0.0,
                clock_speed_mhz=1500.0,
            )
            for i in range(1, 4)
        ]
        result = ProfileResult(duration_s=3.0, snapshots=snaps, gpu_count=1)
        assert result.avg_utilization == pytest.approx(40.0)  # (20+40+60)/3
        assert result.peak_memory_mb == 3000.0
        assert result.avg_power_watts == pytest.approx(120.0)

    def test_to_dict_from_dict_roundtrip(self):
        snaps = [
            GPUMetricSnapshot(
                timestamp=1.0, gpu_index=0, utilization_pct=50.0,
                memory_used_mb=2000.0, memory_total_mb=8000.0,
                power_watts=200.0, temperature_c=70.0,
                sm_activity_pct=45.0, memory_bandwidth_pct=30.0,
                clock_speed_mhz=1500.0,
            ),
        ]
        result = ProfileResult(duration_s=5.0, snapshots=snaps, gpu_count=1)
        d = result.to_dict()
        restored = ProfileResult.from_dict(d)
        assert restored.duration_s == 5.0
        assert restored.gpu_count == 1
        assert len(restored.snapshots) == 1


# ---------------------------------------------------------------------------
# GPUProfiler
# ---------------------------------------------------------------------------

class TestGPUProfiler:
    def test_start_stop_no_backend(self):
        """Profiler works even without GPU backends - returns empty results."""
        profiler = GPUProfiler(interval_ms=50)
        profiler.start()
        time.sleep(0.1)
        result = profiler.stop()
        assert isinstance(result, ProfileResult)
        assert result.duration_s > 0

    def test_duplicate_start_ignored(self):
        profiler = GPUProfiler(interval_ms=50)
        profiler.start()
        profiler.start()  # Should log warning, not crash
        profiler.stop()

    def test_get_live_metrics_empty(self):
        profiler = GPUProfiler(interval_ms=50)
        profiler.start()
        # Immediately get metrics - may be empty
        metrics = profiler.get_live_metrics()
        assert isinstance(metrics, dict)
        profiler.stop()

    def test_resolve_indices_all(self):
        profiler = GPUProfiler()
        indices = profiler._resolve_indices(4)
        assert indices == [0, 1, 2, 3]

    def test_resolve_indices_specific(self):
        profiler = GPUProfiler(gpu_indices=[0, 2])
        indices = profiler._resolve_indices(4)
        assert indices == [0, 2]

    def test_resolve_indices_out_of_range(self):
        profiler = GPUProfiler(gpu_indices=[0, 5, 10])
        indices = profiler._resolve_indices(4)
        assert indices == [0]

    def test_read_gpu_metrics_no_backend(self):
        profiler = GPUProfiler()
        profiler._backend = None
        result = profiler._read_gpu_metrics(0)
        assert result is None


# ---------------------------------------------------------------------------
# _safe_float
# ---------------------------------------------------------------------------

class TestSafeFloat:
    def test_valid_float(self):
        assert _safe_float("42.5") == 42.5

    def test_invalid_string(self):
        assert _safe_float("[N/A]") == 0.0

    def test_empty_string(self):
        assert _safe_float("") == 0.0

    def test_none(self):
        assert _safe_float(None) == 0.0
