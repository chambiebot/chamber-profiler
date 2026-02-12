"""Tests for src.profiler.comparison_profiler."""

import pytest

from src.profiler.comparison_profiler import (
    MetricDelta,
    KernelDelta,
    ComparisonResult,
    ComparisonProfiler,
    _REGRESSION_THRESHOLD_PCT,
    _IMPROVEMENT_THRESHOLD_PCT,
)
from src.profiler.gpu_profiler import ProfileResult as GPUProfileResult, GPUMetricSnapshot
from src.profiler.kernel_tracer import KernelRecord, KernelTraceResult
from src.analysis.bottleneck_detector import PerformanceSummary


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_gpu_result(
    duration_s: float = 10.0,
    avg_util: float = 70.0,
    peak_mem: float = 4000.0,
) -> GPUProfileResult:
    snap = GPUMetricSnapshot(
        timestamp=1.0, gpu_index=0, utilization_pct=avg_util,
        memory_used_mb=peak_mem, memory_total_mb=80000.0,
        power_watts=200.0, temperature_c=70.0,
        sm_activity_pct=avg_util, memory_bandwidth_pct=30.0,
        clock_speed_mhz=1400.0,
    )
    return GPUProfileResult(
        duration_s=duration_s,
        snapshots=[snap],
        gpu_count=1,
    )


def _make_kernel(
    name: str, duration_us: float, category: str = "gemm",
) -> KernelRecord:
    return KernelRecord(
        name=name, duration_us=duration_us,
        grid_size=(128, 1, 1), block_size=(256, 1, 1),
        shared_memory_bytes=0, device_index=0,
        category=category,
    )


def _make_kernel_result(kernels=None) -> KernelTraceResult:
    if kernels is None:
        kernels = [
            _make_kernel("sgemm", 500.0),
            _make_kernel("attention", 300.0, "attention"),
        ]
    total = sum(k.duration_us for k in kernels)
    return KernelTraceResult(
        kernels=kernels,
        total_kernel_time_us=total,
        top_kernels=kernels[:5],
        inefficient_kernels=[],
        category_breakdown={},
    )


def _make_summary(
    total_time_s: float = 10.0,
    compute_pct: float = 70.0,
    efficiency: float = 0.7,
    idle_pct: float = 5.0,
    comm_pct: float = 10.0,
    mem_pct: float = 10.0,
    data_pct: float = 5.0,
    bottleneck_count: int = 2,
) -> PerformanceSummary:
    return PerformanceSummary(
        total_time_s=total_time_s,
        compute_pct=compute_pct,
        memory_pct=mem_pct,
        communication_pct=comm_pct,
        data_loading_pct=data_pct,
        idle_pct=idle_pct,
        primary_bottleneck="compute",
        bottlenecks=[],  # simplified for tests
        overall_efficiency=efficiency,
        summary_text="Test summary.",
    )


# ---------------------------------------------------------------------------
# MetricDelta
# ---------------------------------------------------------------------------


class TestMetricDelta:
    def test_creation(self):
        d = MetricDelta(
            name="Duration", value_a=10.0, value_b=8.0,
            delta=-2.0, delta_pct=-20.0, unit="s",
            status="improved", lower_is_better=True,
        )
        assert d.name == "Duration"
        assert d.status == "improved"

    def test_to_dict_from_dict_roundtrip(self):
        d = MetricDelta(
            name="Efficiency", value_a=70.0, value_b=80.0,
            delta=10.0, delta_pct=14.3, unit="%",
            status="improved", lower_is_better=False,
        )
        data = d.to_dict()
        restored = MetricDelta.from_dict(data)
        assert restored.name == d.name
        assert restored.delta_pct == d.delta_pct
        assert restored.status == d.status

    def test_from_dict_defaults(self):
        d = MetricDelta.from_dict({})
        assert d.name == ""
        assert d.status == "unchanged"


# ---------------------------------------------------------------------------
# KernelDelta
# ---------------------------------------------------------------------------


class TestKernelDelta:
    def test_creation(self):
        kd = KernelDelta(
            kernel_name="sgemm", category="gemm",
            duration_a_us=500.0, duration_b_us=400.0,
            delta_us=-100.0, delta_pct=-20.0,
            status="improved",
        )
        assert kd.kernel_name == "sgemm"
        assert kd.status == "improved"

    def test_to_dict_from_dict_roundtrip(self):
        kd = KernelDelta(
            kernel_name="test", category="other",
            duration_a_us=100.0, duration_b_us=200.0,
            delta_us=100.0, delta_pct=100.0,
            status="regressed",
        )
        data = kd.to_dict()
        restored = KernelDelta.from_dict(data)
        assert restored.kernel_name == kd.kernel_name
        assert restored.status == kd.status


# ---------------------------------------------------------------------------
# ComparisonResult
# ---------------------------------------------------------------------------


class TestComparisonResult:
    def test_to_dict_from_dict_roundtrip(self):
        result = ComparisonResult(
            label_a="Baseline", label_b="Current",
            metric_deltas=[], kernel_deltas=[],
            regressions=[], improvements=[],
            summary_a=None, summary_b=None,
            speedup=1.5, verdict="Faster",
        )
        data = result.to_dict()
        restored = ComparisonResult.from_dict(data)
        assert restored.label_a == "Baseline"
        assert restored.speedup == 1.5
        assert restored.verdict == "Faster"


# ---------------------------------------------------------------------------
# ComparisonProfiler — compare metrics
# ---------------------------------------------------------------------------


class TestComparisonProfilerMetrics:
    def test_compare_with_summaries(self):
        summary_a = _make_summary(total_time_s=10.0, efficiency=0.6)
        summary_b = _make_summary(total_time_s=8.0, efficiency=0.8)

        profiler = ComparisonProfiler(
            summary_a=summary_a, summary_b=summary_b,
        )
        result = profiler.compare()

        assert result.speedup > 1.0
        assert len(result.metric_deltas) > 0

    def test_duration_improvement_detected(self):
        summary_a = _make_summary(total_time_s=10.0)
        summary_b = _make_summary(total_time_s=5.0)

        profiler = ComparisonProfiler(
            summary_a=summary_a, summary_b=summary_b,
        )
        result = profiler.compare()

        duration_delta = next(
            (m for m in result.metric_deltas if m.name == "Total Duration"),
            None,
        )
        assert duration_delta is not None
        assert duration_delta.status == "improved"
        assert duration_delta.delta < 0

    def test_duration_regression_detected(self):
        summary_a = _make_summary(total_time_s=5.0)
        summary_b = _make_summary(total_time_s=10.0)

        profiler = ComparisonProfiler(
            summary_a=summary_a, summary_b=summary_b,
        )
        result = profiler.compare()

        duration_delta = next(
            (m for m in result.metric_deltas if m.name == "Total Duration"),
            None,
        )
        assert duration_delta is not None
        assert duration_delta.status == "regressed"

    def test_unchanged_when_similar(self):
        summary_a = _make_summary(total_time_s=10.0)
        summary_b = _make_summary(total_time_s=10.1)

        profiler = ComparisonProfiler(
            summary_a=summary_a, summary_b=summary_b,
        )
        result = profiler.compare()

        duration_delta = next(
            (m for m in result.metric_deltas if m.name == "Total Duration"),
            None,
        )
        assert duration_delta is not None
        assert duration_delta.status == "unchanged"

    def test_gpu_utilization_included_with_gpu_results(self):
        gpu_a = _make_gpu_result(avg_util=50.0)
        gpu_b = _make_gpu_result(avg_util=80.0)
        summary_a = _make_summary()
        summary_b = _make_summary()

        profiler = ComparisonProfiler(
            gpu_result_a=gpu_a, gpu_result_b=gpu_b,
            summary_a=summary_a, summary_b=summary_b,
        )
        result = profiler.compare()

        util_delta = next(
            (m for m in result.metric_deltas if "Utilization" in m.name),
            None,
        )
        assert util_delta is not None
        assert util_delta.status == "improved"


# ---------------------------------------------------------------------------
# ComparisonProfiler — compare kernels
# ---------------------------------------------------------------------------


class TestComparisonProfilerKernels:
    def test_kernel_comparison(self):
        kernels_a = [_make_kernel("sgemm", 500.0)]
        kernels_b = [_make_kernel("sgemm", 300.0)]

        profiler = ComparisonProfiler(
            kernel_result_a=_make_kernel_result(kernels_a),
            kernel_result_b=_make_kernel_result(kernels_b),
            summary_a=_make_summary(),
            summary_b=_make_summary(),
        )
        result = profiler.compare()

        assert len(result.kernel_deltas) > 0
        kd = result.kernel_deltas[0]
        assert kd.kernel_name == "sgemm"
        assert kd.status == "improved"

    def test_new_kernel_detected(self):
        kernels_a = [_make_kernel("sgemm", 500.0)]
        kernels_b = [
            _make_kernel("sgemm", 500.0),
            _make_kernel("new_kernel", 200.0, "other"),
        ]

        profiler = ComparisonProfiler(
            kernel_result_a=_make_kernel_result(kernels_a),
            kernel_result_b=_make_kernel_result(kernels_b),
            summary_a=_make_summary(),
            summary_b=_make_summary(),
        )
        result = profiler.compare()

        new_kd = next(
            (k for k in result.kernel_deltas if k.status == "new"), None,
        )
        assert new_kd is not None
        assert new_kd.kernel_name == "new_kernel"

    def test_removed_kernel_detected(self):
        kernels_a = [
            _make_kernel("sgemm", 500.0),
            _make_kernel("old_kernel", 200.0, "other"),
        ]
        kernels_b = [_make_kernel("sgemm", 500.0)]

        profiler = ComparisonProfiler(
            kernel_result_a=_make_kernel_result(kernels_a),
            kernel_result_b=_make_kernel_result(kernels_b),
            summary_a=_make_summary(),
            summary_b=_make_summary(),
        )
        result = profiler.compare()

        removed_kd = next(
            (k for k in result.kernel_deltas if k.status == "removed"), None,
        )
        assert removed_kd is not None
        assert removed_kd.kernel_name == "old_kernel"

    def test_no_kernel_comparison_without_data(self):
        profiler = ComparisonProfiler(
            summary_a=_make_summary(),
            summary_b=_make_summary(),
        )
        result = profiler.compare()
        assert result.kernel_deltas == []


# ---------------------------------------------------------------------------
# ComparisonProfiler — verdict
# ---------------------------------------------------------------------------


class TestComparisonProfilerVerdict:
    def test_faster_verdict(self):
        profiler = ComparisonProfiler(
            summary_a=_make_summary(total_time_s=10.0),
            summary_b=_make_summary(total_time_s=5.0),
        )
        result = profiler.compare()
        assert "improved" in result.verdict.lower() or "faster" in result.verdict.lower()

    def test_slower_verdict(self):
        profiler = ComparisonProfiler(
            summary_a=_make_summary(total_time_s=5.0),
            summary_b=_make_summary(total_time_s=10.0),
        )
        result = profiler.compare()
        assert "regressed" in result.verdict.lower() or "slower" in result.verdict.lower()

    def test_equivalent_verdict(self):
        profiler = ComparisonProfiler(
            summary_a=_make_summary(total_time_s=10.0),
            summary_b=_make_summary(total_time_s=10.0),
        )
        result = profiler.compare()
        assert "equivalent" in result.verdict.lower()

    def test_bottleneck_change_noted(self):
        sum_a = _make_summary()
        sum_b = PerformanceSummary(
            total_time_s=10.0, compute_pct=50.0, memory_pct=30.0,
            communication_pct=10.0, data_loading_pct=5.0, idle_pct=5.0,
            primary_bottleneck="memory",
            bottlenecks=[], overall_efficiency=0.7,
            summary_text="Test",
        )
        profiler = ComparisonProfiler(summary_a=sum_a, summary_b=sum_b)
        result = profiler.compare()
        assert "bottleneck changed" in result.verdict.lower()


# ---------------------------------------------------------------------------
# ComparisonProfiler — auto-build summaries
# ---------------------------------------------------------------------------


class TestComparisonProfilerAutoSummary:
    def test_builds_summaries_from_gpu_results(self):
        gpu_a = _make_gpu_result(duration_s=10.0, avg_util=60.0)
        gpu_b = _make_gpu_result(duration_s=8.0, avg_util=80.0)

        profiler = ComparisonProfiler(
            gpu_result_a=gpu_a, gpu_result_b=gpu_b,
        )
        result = profiler.compare()
        assert result.summary_a is not None
        assert result.summary_b is not None
        assert result.speedup > 1.0

    def test_labels(self):
        profiler = ComparisonProfiler(
            summary_a=_make_summary(),
            summary_b=_make_summary(),
            label_a="Before",
            label_b="After",
        )
        result = profiler.compare()
        assert result.label_a == "Before"
        assert result.label_b == "After"


# ---------------------------------------------------------------------------
# ComparisonProfiler — speedup
# ---------------------------------------------------------------------------


class TestComparisonProfilerSpeedup:
    def test_speedup_2x(self):
        profiler = ComparisonProfiler(
            summary_a=_make_summary(total_time_s=10.0),
            summary_b=_make_summary(total_time_s=5.0),
        )
        result = profiler.compare()
        assert abs(result.speedup - 2.0) < 0.01

    def test_speedup_half(self):
        profiler = ComparisonProfiler(
            summary_a=_make_summary(total_time_s=5.0),
            summary_b=_make_summary(total_time_s=10.0),
        )
        result = profiler.compare()
        assert abs(result.speedup - 0.5) < 0.01

    def test_speedup_equal(self):
        profiler = ComparisonProfiler(
            summary_a=_make_summary(total_time_s=10.0),
            summary_b=_make_summary(total_time_s=10.0),
        )
        result = profiler.compare()
        assert abs(result.speedup - 1.0) < 0.01
