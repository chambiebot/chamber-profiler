"""Tests for src.profiler.memory_analyzer."""

import time

import pytest

from src.profiler.memory_analyzer import (
    MemoryAnalyzer,
    MemorySnapshot,
    MemoryBreakdown,
    OOMRiskAssessment,
    MemoryAnalysisResult,
)


# ---------------------------------------------------------------------------
# MemorySnapshot
# ---------------------------------------------------------------------------

class TestMemorySnapshot:
    def test_creation(self):
        snap = MemorySnapshot(
            timestamp=1000.0,
            allocated_mb=2048.0,
            reserved_mb=4096.0,
            peak_allocated_mb=3000.0,
            peak_reserved_mb=5000.0,
            num_tensors=100,
            device_index=0,
        )
        assert snap.allocated_mb == 2048.0
        assert snap.device_index == 0

    def test_to_dict_from_dict_roundtrip(self):
        snap = MemorySnapshot(
            timestamp=1.0, allocated_mb=100.0, reserved_mb=200.0,
            peak_allocated_mb=150.0, peak_reserved_mb=250.0,
            num_tensors=10, device_index=0,
        )
        d = snap.to_dict()
        restored = MemorySnapshot.from_dict(d)
        assert restored == snap

    def test_frozen(self):
        snap = MemorySnapshot(
            timestamp=1.0, allocated_mb=0.0, reserved_mb=0.0,
            peak_allocated_mb=0.0, peak_reserved_mb=0.0,
            num_tensors=0, device_index=0,
        )
        with pytest.raises(AttributeError):
            snap.allocated_mb = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# MemoryBreakdown
# ---------------------------------------------------------------------------

class TestMemoryBreakdown:
    def test_defaults(self):
        bd = MemoryBreakdown()
        assert bd.parameter_memory_mb == 0.0

    def test_to_dict_from_dict_roundtrip(self):
        bd = MemoryBreakdown(
            parameter_memory_mb=100.0,
            gradient_memory_mb=100.0,
            optimizer_state_mb=200.0,
            activation_memory_mb=300.0,
            other_mb=50.0,
        )
        d = bd.to_dict()
        restored = MemoryBreakdown.from_dict(d)
        assert restored.parameter_memory_mb == bd.parameter_memory_mb
        assert restored.optimizer_state_mb == bd.optimizer_state_mb


# ---------------------------------------------------------------------------
# OOMRiskAssessment
# ---------------------------------------------------------------------------

class TestOOMRiskAssessment:
    def test_defaults(self):
        oom = OOMRiskAssessment()
        assert oom.risk_score == 0.0
        assert oom.suggestions == []

    def test_to_dict_from_dict_roundtrip(self):
        oom = OOMRiskAssessment(
            risk_score=0.8, peak_usage_pct=92.0,
            headroom_mb=640.0, suggestions=["Reduce batch size"],
        )
        d = oom.to_dict()
        restored = OOMRiskAssessment.from_dict(d)
        assert restored.risk_score == 0.8
        assert len(restored.suggestions) == 1


# ---------------------------------------------------------------------------
# MemoryAnalysisResult
# ---------------------------------------------------------------------------

class TestMemoryAnalysisResult:
    def test_to_dict_from_dict_roundtrip(self):
        snap = MemorySnapshot(
            timestamp=1.0, allocated_mb=100.0, reserved_mb=200.0,
            peak_allocated_mb=150.0, peak_reserved_mb=250.0,
            num_tensors=5, device_index=0,
        )
        result = MemoryAnalysisResult(
            snapshots=[snap],
            breakdown=MemoryBreakdown(parameter_memory_mb=50.0),
            leak_detected=True,
            leak_points=[(1.0, 100.0)],
            oom_risk=OOMRiskAssessment(risk_score=0.5),
            peak_snapshot=snap,
        )
        d = result.to_dict()
        restored = MemoryAnalysisResult.from_dict(d)
        assert len(restored.snapshots) == 1
        assert restored.leak_detected is True
        assert len(restored.leak_points) == 1


# ---------------------------------------------------------------------------
# MemoryAnalyzer
# ---------------------------------------------------------------------------

class TestMemoryAnalyzer:
    def test_start_stop_without_cuda(self):
        """Analyzer should work without CUDA - returns empty results."""
        analyzer = MemoryAnalyzer(device_index=0, interval_ms=50)
        analyzer.start()
        time.sleep(0.1)
        result = analyzer.stop()
        assert isinstance(result, MemoryAnalysisResult)

    def test_duplicate_start(self):
        analyzer = MemoryAnalyzer()
        analyzer.start()
        analyzer.start()  # Should warn, not crash
        analyzer.stop()

    def test_take_snapshot_without_cuda(self):
        analyzer = MemoryAnalyzer()
        snap = analyzer.take_snapshot()
        assert isinstance(snap, MemorySnapshot)
        assert snap.allocated_mb == 0.0

    def test_detect_memory_leaks_no_leak(self):
        analyzer = MemoryAnalyzer()
        # Flat allocation pattern - no leak
        snapshots = [
            MemorySnapshot(
                timestamp=float(i), allocated_mb=100.0, reserved_mb=200.0,
                peak_allocated_mb=100.0, peak_reserved_mb=200.0,
                num_tensors=0, device_index=0,
            )
            for i in range(10)
        ]
        leaks = analyzer.detect_memory_leaks(snapshots)
        assert leaks == []

    def test_detect_memory_leaks_with_leak(self):
        analyzer = MemoryAnalyzer()
        # Monotonically increasing - should detect leak
        snapshots = [
            MemorySnapshot(
                timestamp=float(i), allocated_mb=float(100 + i * 10),
                reserved_mb=200.0, peak_allocated_mb=float(100 + i * 10),
                peak_reserved_mb=200.0, num_tensors=0, device_index=0,
            )
            for i in range(10)
        ]
        leaks = analyzer.detect_memory_leaks(snapshots)
        assert len(leaks) > 0

    def test_detect_memory_leaks_too_few_snapshots(self):
        analyzer = MemoryAnalyzer()
        snapshots = [
            MemorySnapshot(
                timestamp=1.0, allocated_mb=100.0, reserved_mb=200.0,
                peak_allocated_mb=100.0, peak_reserved_mb=200.0,
                num_tensors=0, device_index=0,
            ),
        ]
        leaks = analyzer.detect_memory_leaks(snapshots)
        assert leaks == []

    def test_get_peak_analysis_empty(self):
        analyzer = MemoryAnalyzer()
        assert analyzer.get_peak_analysis([]) is None

    def test_get_peak_analysis(self):
        analyzer = MemoryAnalyzer()
        snapshots = [
            MemorySnapshot(
                timestamp=float(i), allocated_mb=float(i * 100),
                reserved_mb=200.0, peak_allocated_mb=0.0,
                peak_reserved_mb=0.0, num_tensors=0, device_index=0,
            )
            for i in range(5)
        ]
        peak = analyzer.get_peak_analysis(snapshots)
        assert peak is not None
        assert peak.allocated_mb == 400.0

    def test_assess_oom_risk_low(self):
        analyzer = MemoryAnalyzer()
        snapshots = [
            MemorySnapshot(
                timestamp=1.0, allocated_mb=100.0, reserved_mb=200.0,
                peak_allocated_mb=100.0, peak_reserved_mb=200.0,
                num_tensors=0, device_index=0,
            ),
        ]
        risk = analyzer.assess_oom_risk(
            total_gpu_memory_mb=10000.0, snapshots=snapshots,
        )
        assert risk.risk_score == 0.0
        assert risk.peak_usage_pct < 50.0

    def test_assess_oom_risk_high(self):
        analyzer = MemoryAnalyzer()
        snapshots = [
            MemorySnapshot(
                timestamp=1.0, allocated_mb=9000.0, reserved_mb=9500.0,
                peak_allocated_mb=9200.0, peak_reserved_mb=9500.0,
                num_tensors=0, device_index=0,
            ),
        ]
        risk = analyzer.assess_oom_risk(
            total_gpu_memory_mb=10000.0, snapshots=snapshots,
        )
        assert risk.risk_score > 0.5
        assert risk.peak_usage_pct > 80.0

    def test_assess_oom_risk_no_memory_info(self):
        analyzer = MemoryAnalyzer()
        risk = analyzer.assess_oom_risk(total_gpu_memory_mb=0.0, snapshots=[])
        assert risk.risk_score == 0.0

    def test_generate_suggestions_low_usage(self):
        suggestions = MemoryAnalyzer._generate_suggestions(30.0)
        assert suggestions == []

    def test_generate_suggestions_high_usage(self):
        suggestions = MemoryAnalyzer._generate_suggestions(85.0)
        assert len(suggestions) > 0

    def test_get_memory_breakdown_without_cuda(self):
        analyzer = MemoryAnalyzer()
        result = analyzer.get_memory_breakdown()
        assert result is None
