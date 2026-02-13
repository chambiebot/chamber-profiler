"""Tests for src.profiler.memory_leak_detector."""

import pytest

from src.profiler.memory_analyzer import MemorySnapshot
from src.profiler.memory_leak_detector import (
    LeakCandidate,
    LeakDetectionResult,
    MemoryLeakDetector,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_snapshot(ts: float, allocated_mb: float) -> MemorySnapshot:
    return MemorySnapshot(
        timestamp=ts,
        allocated_mb=allocated_mb,
        reserved_mb=allocated_mb * 1.5,
        peak_allocated_mb=allocated_mb,
        peak_reserved_mb=allocated_mb * 1.5,
        num_tensors=0,
        device_index=0,
    )


def _make_snapshots(allocs: list[float], dt: float = 1.0) -> list[MemorySnapshot]:
    return [_make_snapshot(i * dt, a) for i, a in enumerate(allocs)]


# ---------------------------------------------------------------------------
# LeakCandidate
# ---------------------------------------------------------------------------

class TestLeakCandidate:
    def test_creation(self):
        lc = LeakCandidate(
            start_timestamp=0.0, end_timestamp=10.0,
            start_allocated_mb=100.0, end_allocated_mb=200.0,
            growth_mb=100.0, growth_rate_mb_per_sec=10.0,
            num_snapshots=10, confidence=0.8,
        )
        assert lc.growth_mb == 100.0
        assert lc.confidence == 0.8

    def test_to_dict_from_dict_roundtrip(self):
        lc = LeakCandidate(
            start_timestamp=1.0, end_timestamp=5.0,
            start_allocated_mb=50.0, end_allocated_mb=80.0,
            growth_mb=30.0, growth_rate_mb_per_sec=7.5,
            num_snapshots=5, confidence=0.6,
        )
        d = lc.to_dict()
        restored = LeakCandidate.from_dict(d)
        assert restored.growth_mb == lc.growth_mb
        assert restored.num_snapshots == lc.num_snapshots

    def test_frozen(self):
        lc = LeakCandidate(
            start_timestamp=0.0, end_timestamp=1.0,
            start_allocated_mb=0.0, end_allocated_mb=1.0,
            growth_mb=1.0, growth_rate_mb_per_sec=1.0,
            num_snapshots=2, confidence=0.5,
        )
        with pytest.raises(AttributeError):
            lc.growth_mb = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# LeakDetectionResult
# ---------------------------------------------------------------------------

class TestLeakDetectionResult:
    def test_to_dict_from_dict_roundtrip(self):
        lc = LeakCandidate(
            start_timestamp=0.0, end_timestamp=5.0,
            start_allocated_mb=100.0, end_allocated_mb=150.0,
            growth_mb=50.0, growth_rate_mb_per_sec=10.0,
            num_snapshots=6, confidence=0.7,
        )
        result = LeakDetectionResult(
            has_leak=True,
            leak_candidates=[lc],
            overall_growth_mb=50.0,
            overall_growth_rate_mb_per_sec=10.0,
            trend_slope=10.0,
            trend_r_squared=0.95,
            recommendations=["Fix it"],
            snapshots_analyzed=20,
        )
        d = result.to_dict()
        restored = LeakDetectionResult.from_dict(d)
        assert restored.has_leak is True
        assert len(restored.leak_candidates) == 1
        assert restored.trend_r_squared == 0.95
        assert restored.snapshots_analyzed == 20

    def test_empty_roundtrip(self):
        result = LeakDetectionResult(
            has_leak=False, leak_candidates=[],
            overall_growth_mb=0.0, overall_growth_rate_mb_per_sec=0.0,
            trend_slope=0.0, trend_r_squared=0.0,
            recommendations=[], snapshots_analyzed=0,
        )
        d = result.to_dict()
        restored = LeakDetectionResult.from_dict(d)
        assert restored.has_leak is False
        assert restored.leak_candidates == []


# ---------------------------------------------------------------------------
# MemoryLeakDetector
# ---------------------------------------------------------------------------

class TestMemoryLeakDetector:
    def test_too_few_snapshots(self):
        detector = MemoryLeakDetector()
        result = detector.analyze([_make_snapshot(0, 100)])
        assert result.has_leak is False
        assert result.snapshots_analyzed == 1

    def test_flat_allocation_no_leak(self):
        detector = MemoryLeakDetector()
        snapshots = _make_snapshots([100.0] * 20)
        result = detector.analyze(snapshots)
        assert result.has_leak is False
        assert result.leak_candidates == []

    def test_monotonic_increase_leak(self):
        detector = MemoryLeakDetector()
        # Strictly increasing by 10 MB each step over 20 steps
        allocs = [100.0 + i * 10.0 for i in range(20)]
        snapshots = _make_snapshots(allocs)
        result = detector.analyze(snapshots)
        assert result.has_leak is True
        assert len(result.leak_candidates) > 0
        assert result.overall_growth_mb > 0

    def test_temporary_spike_no_leak(self):
        detector = MemoryLeakDetector()
        # Flat, then spike, then back to flat
        allocs = [100.0] * 10 + [200.0] * 3 + [100.0] * 10
        snapshots = _make_snapshots(allocs)
        result = detector.analyze(snapshots)
        # The spike is only 3 points, not enough for a monotonic run
        assert result.has_leak is False

    def test_gradual_leak_with_noise(self):
        detector = MemoryLeakDetector(min_run_length=3, min_growth_mb=0.5)
        # Gradual uptrend with some dips
        allocs = []
        base = 100.0
        for i in range(30):
            base += 2.0  # leak
            if i % 7 == 0:
                base -= 0.5  # minor GC
            allocs.append(base)
        snapshots = _make_snapshots(allocs)
        result = detector.analyze(snapshots)
        # Should detect via trend analysis (high R²)
        assert result.trend_slope > 0
        assert result.overall_growth_mb > 0

    def test_sawtooth_no_leak(self):
        detector = MemoryLeakDetector()
        # Repeating pattern: up 3, down to baseline
        allocs = []
        for _ in range(5):
            allocs.extend([100.0, 110.0, 120.0, 100.0])
        snapshots = _make_snapshots(allocs)
        result = detector.analyze(snapshots)
        # No sustained monotonic run >= 5
        assert len(result.leak_candidates) == 0

    def test_multiple_leak_episodes(self):
        detector = MemoryLeakDetector(min_run_length=5, min_growth_mb=1.0)
        # Two separate leak episodes
        allocs = (
            [100 + i * 5 for i in range(8)]   # first leak: 100 -> 135
            + [100.0] * 5                       # flat
            + [200 + i * 5 for i in range(8)]  # second leak: 200 -> 235
        )
        snapshots = _make_snapshots(allocs)
        result = detector.analyze(snapshots)
        assert result.has_leak is True
        assert len(result.leak_candidates) == 2

    def test_linear_regression_perfect_line(self):
        slope, r2 = MemoryLeakDetector._linear_regression(
            [0, 1, 2, 3, 4], [0, 2, 4, 6, 8],
        )
        assert abs(slope - 2.0) < 1e-10
        assert abs(r2 - 1.0) < 1e-10

    def test_linear_regression_flat(self):
        slope, r2 = MemoryLeakDetector._linear_regression(
            [0, 1, 2, 3], [5, 5, 5, 5],
        )
        assert slope == 0.0

    def test_linear_regression_single_point(self):
        slope, r2 = MemoryLeakDetector._linear_regression([1], [1])
        assert slope == 0.0
        assert r2 == 0.0

    def test_recommendations_for_leak(self):
        detector = MemoryLeakDetector()
        allocs = [100.0 + i * 10.0 for i in range(20)]
        snapshots = _make_snapshots(allocs)
        result = detector.analyze(snapshots)
        assert len(result.recommendations) > 0
        # Should mention detach or computation graph
        any_detach = any("detach" in r.lower() for r in result.recommendations)
        assert any_detach

    def test_recommendations_no_leak(self):
        detector = MemoryLeakDetector()
        snapshots = _make_snapshots([100.0] * 20)
        result = detector.analyze(snapshots)
        assert result.recommendations == []

    def test_high_r_squared_detection(self):
        detector = MemoryLeakDetector(r_squared_threshold=0.9)
        # Perfectly linear growth — should trigger R² based detection
        # even if we set min_run_length very high
        allocs = [100.0 + i * 0.1 for i in range(50)]
        snapshots = _make_snapshots(allocs)
        result = detector.analyze(snapshots)
        assert result.trend_r_squared > 0.99
        assert result.has_leak is True

    def test_custom_thresholds(self):
        detector = MemoryLeakDetector(
            min_run_length=3,
            min_growth_mb=0.1,
            r_squared_threshold=0.5,
        )
        allocs = [100.0, 100.5, 101.0, 100.0, 100.0, 100.0]
        snapshots = _make_snapshots(allocs)
        result = detector.analyze(snapshots)
        assert result.snapshots_analyzed == 6

    def test_growth_rate_calculation(self):
        detector = MemoryLeakDetector()
        # 10 MB growth over 19 seconds (20 snapshots, dt=1)
        allocs = [100.0 + i * (10.0 / 19.0) for i in range(20)]
        snapshots = _make_snapshots(allocs, dt=1.0)
        result = detector.analyze(snapshots)
        assert result.overall_growth_mb > 0
        assert result.overall_growth_rate_mb_per_sec > 0
