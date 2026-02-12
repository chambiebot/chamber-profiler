"""Tests for src.analysis.bottleneck_detector."""

import pytest

from src.analysis.bottleneck_detector import (
    BottleneckDetector,
    Bottleneck,
    PerformanceSummary,
    TimeCategory,
)
from src.profiler.gpu_profiler import GPUMetricSnapshot, ProfileResult as GPUProfileResult
from src.profiler.kernel_tracer import KernelRecord, KernelTraceResult
from src.profiler.memory_analyzer import (
    MemorySnapshot, MemoryAnalysisResult, MemoryBreakdown, OOMRiskAssessment,
)
from src.profiler.communication_profiler import CollectiveOperation, CommProfileResult
from src.profiler.data_loading_profiler import DataLoadingMetrics, DataLoadingResult


# ---------------------------------------------------------------------------
# TimeCategory
# ---------------------------------------------------------------------------

class TestTimeCategory:
    def test_values(self):
        assert TimeCategory.compute.value == "compute"
        assert TimeCategory.memory.value == "memory"
        assert TimeCategory.communication.value == "communication"
        assert TimeCategory.data_loading.value == "data_loading"
        assert TimeCategory.idle.value == "idle"


# ---------------------------------------------------------------------------
# Bottleneck
# ---------------------------------------------------------------------------

class TestBottleneck:
    def test_impact_clamped(self):
        b = Bottleneck(
            category=TimeCategory.compute,
            time_pct=50.0,
            description="test",
            impact_score=1.5,
            recommendations=["do something"],
            expected_speedup="2x",
        )
        assert b.impact_score == 1.0

    def test_to_dict_from_dict_roundtrip(self):
        b = Bottleneck(
            category=TimeCategory.memory,
            time_pct=30.0,
            description="High memory pressure",
            impact_score=0.7,
            recommendations=["Reduce batch size"],
            expected_speedup="1.3x",
        )
        d = b.to_dict()
        restored = Bottleneck.from_dict(d)
        assert restored.category == TimeCategory.memory
        assert restored.impact_score == 0.7


# ---------------------------------------------------------------------------
# PerformanceSummary
# ---------------------------------------------------------------------------

class TestPerformanceSummary:
    def test_to_dict_from_dict_roundtrip(self):
        summary = PerformanceSummary(
            total_time_s=10.0,
            compute_pct=70.0,
            memory_pct=10.0,
            communication_pct=10.0,
            data_loading_pct=5.0,
            idle_pct=5.0,
            primary_bottleneck="compute",
            bottlenecks=[],
            overall_efficiency=0.7,
            summary_text="Test summary.",
        )
        d = summary.to_dict()
        restored = PerformanceSummary.from_dict(d)
        assert restored.total_time_s == 10.0
        assert restored.compute_pct == 70.0
        assert restored.primary_bottleneck == "compute"


# ---------------------------------------------------------------------------
# BottleneckDetector
# ---------------------------------------------------------------------------

def _make_gpu_snapshots(n=10, util=80.0, mem_bw=30.0, sm=80.0):
    """Helper to create GPU metric snapshots."""
    return [
        GPUMetricSnapshot(
            timestamp=float(i), gpu_index=0,
            utilization_pct=util, memory_used_mb=4000.0,
            memory_total_mb=8000.0, power_watts=200.0,
            temperature_c=70.0, sm_activity_pct=sm,
            memory_bandwidth_pct=mem_bw, clock_speed_mhz=1500.0,
        )
        for i in range(n)
    ]


class TestBottleneckDetector:
    def test_analyze_no_data(self):
        """Detector handles no input data gracefully."""
        detector = BottleneckDetector()
        summary = detector.analyze()
        assert isinstance(summary, PerformanceSummary)
        assert summary.compute_pct == 100.0  # default when no data

    def test_analyze_gpu_only(self):
        snaps = _make_gpu_snapshots(n=10, util=80.0, sm=80.0, mem_bw=20.0)
        gpu_result = GPUProfileResult(
            duration_s=5.0, snapshots=snaps, gpu_count=1,
        )
        detector = BottleneckDetector(gpu_result=gpu_result)
        summary = detector.analyze()
        assert summary.total_time_s == 5.0
        assert summary.compute_pct > 0

    def test_analyze_low_sm_detects_bottleneck(self):
        """Very low SM utilization should flag a compute bottleneck."""
        snaps = _make_gpu_snapshots(n=10, util=20.0, sm=20.0, mem_bw=10.0)
        gpu_result = GPUProfileResult(
            duration_s=5.0, snapshots=snaps, gpu_count=1,
        )
        detector = BottleneckDetector(gpu_result=gpu_result)
        summary = detector.analyze()
        compute_bottlenecks = [
            b for b in summary.bottlenecks
            if b.category == TimeCategory.compute
        ]
        assert len(compute_bottlenecks) > 0

    def test_analyze_kernel_data(self):
        kernels = [
            KernelRecord(
                name="gemm_kernel", duration_us=500.0,
                grid_size=(100, 1, 1), block_size=(256, 1, 1),
                shared_memory_bytes=0, device_index=0, category="gemm",
            ),
            KernelRecord(
                name="memcpy_kernel", duration_us=100.0,
                grid_size=(1, 1, 1), block_size=(1, 1, 1),
                shared_memory_bytes=0, device_index=0, category="memory",
            ),
        ]
        kernel_result = KernelTraceResult(
            kernels=kernels,
            total_kernel_time_us=600.0,
            top_kernels=kernels,
            inefficient_kernels=[],
            category_breakdown={"gemm": 500.0, "memory": 100.0},
        )
        detector = BottleneckDetector(kernel_result=kernel_result)
        summary = detector.analyze()
        assert summary.compute_pct > 0

    def test_analyze_memory_leak(self):
        snaps = [
            MemorySnapshot(
                timestamp=float(i), allocated_mb=float(100 + i * 50),
                reserved_mb=200.0, peak_allocated_mb=float(100 + i * 50),
                peak_reserved_mb=200.0, num_tensors=0, device_index=0,
            )
            for i in range(10)
        ]
        memory_result = MemoryAnalysisResult(
            snapshots=snaps,
            breakdown=None,
            leak_detected=True,
            leak_points=[(9.0, 550.0)],
            oom_risk=OOMRiskAssessment(
                risk_score=0.9, peak_usage_pct=92.0,
                headroom_mb=640.0, suggestions=[],
            ),
            peak_snapshot=snaps[-1],
        )
        detector = BottleneckDetector(memory_result=memory_result)
        summary = detector.analyze()
        memory_bottlenecks = [
            b for b in summary.bottlenecks
            if b.category == TimeCategory.memory
        ]
        assert len(memory_bottlenecks) > 0

    def test_analyze_communication_bottleneck(self):
        ops = [
            CollectiveOperation(
                name="all_reduce", duration_us=50000.0,
                data_size_bytes=10_000_000, src_rank=0, dst_rank=None,
                timestamp=1.0,
            ),
        ]
        gpu_result = GPUProfileResult(
            duration_s=0.1, snapshots=_make_gpu_snapshots(5),
            gpu_count=1,
        )
        comm_result = CommProfileResult(
            operations=ops,
            total_comm_time_us=50000.0,
            total_data_transferred_bytes=10_000_000,
            avg_bandwidth_gbps=1.6,
            comm_compute_overlap_pct=0.0,
            bottleneck_ops=ops,
            straggler_ranks=[],
        )
        detector = BottleneckDetector(
            gpu_result=gpu_result, comm_result=comm_result,
        )
        summary = detector.analyze()
        comm_bottlenecks = [
            b for b in summary.bottlenecks
            if b.category == TimeCategory.communication
        ]
        assert len(comm_bottlenecks) > 0

    def test_analyze_data_loading_bottleneck(self):
        metrics = [
            DataLoadingMetrics(
                batch_load_time_ms=100.0, batch_process_time_ms=50.0,
                gpu_idle_time_ms=80.0, timestamp=1.0, batch_index=i,
                num_workers_active=0, prefetch_buffer_size=0,
                io_throughput_mbps=30.0,
            )
            for i in range(5)
        ]
        data_result = DataLoadingResult(
            metrics=metrics,
            avg_load_time_ms=100.0,
            avg_gpu_idle_time_ms=80.0,
            is_bottleneck=True,
            bottleneck_severity="severe",
            recommendations=["Increase num_workers"],
            io_throughput_mbps=30.0,
        )
        detector = BottleneckDetector(data_result=data_result)
        summary = detector.analyze()
        data_bottlenecks = [
            b for b in summary.bottlenecks
            if b.category == TimeCategory.data_loading
        ]
        assert len(data_bottlenecks) > 0

    def test_bottlenecks_sorted_by_impact(self):
        snaps = _make_gpu_snapshots(n=10, util=20.0, sm=20.0, mem_bw=10.0)
        gpu_result = GPUProfileResult(
            duration_s=5.0, snapshots=snaps, gpu_count=1,
        )
        detector = BottleneckDetector(gpu_result=gpu_result)
        summary = detector.analyze()
        if len(summary.bottlenecks) >= 2:
            for i in range(len(summary.bottlenecks) - 1):
                assert (
                    summary.bottlenecks[i].impact_score
                    >= summary.bottlenecks[i + 1].impact_score
                )

    def test_overall_efficiency_bounded(self):
        detector = BottleneckDetector()
        summary = detector.analyze()
        assert 0.0 <= summary.overall_efficiency <= 1.0

    def test_summary_text_not_empty(self):
        detector = BottleneckDetector()
        summary = detector.analyze()
        assert len(summary.summary_text) > 0
