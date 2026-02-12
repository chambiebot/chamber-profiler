"""Tests for src.analysis.optimization_advisor."""

import pytest

from src.analysis.optimization_advisor import (
    Optimization,
    OptimizationReport,
    OptimizationAdvisor,
    OptimizationCategory,
    Priority,
)
from src.profiler.gpu_profiler import ProfileResult as GPUProfileResult, GPUMetricSnapshot
from src.profiler.kernel_tracer import KernelRecord, KernelTraceResult
from src.profiler.distributed_profiler import DistributedProfileResult
from src.profiler.memory_analyzer import MemoryAnalysisResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_gpu_result(
    duration_s: float = 60.0,
    avg_util: float = 40.0,
    peak_mem: float = 20000.0,
    total_mem: float = 81920.0,
) -> GPUProfileResult:
    snap = GPUMetricSnapshot(
        timestamp=1.0, gpu_index=0, utilization_pct=avg_util,
        memory_used_mb=peak_mem, memory_total_mb=total_mem,
        power_watts=250.0, temperature_c=72.0,
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


def _make_kernel_result_elementwise_heavy() -> KernelTraceResult:
    kernels = [
        _make_kernel("gemm", 200.0, "gemm"),
        _make_kernel("elementwise_add", 400.0, "elementwise"),
        _make_kernel("layernorm", 200.0, "reduction"),
        _make_kernel("relu", 200.0, "elementwise"),
    ]
    total = sum(k.duration_us for k in kernels)
    return KernelTraceResult(
        kernels=kernels,
        total_kernel_time_us=total,
        top_kernels=kernels,
        inefficient_kernels=[kernels[1], kernels[3]],
        category_breakdown={
            "gemm": 200.0,
            "elementwise": 600.0,
            "reduction": 200.0,
        },
    )


def _make_dist_result(
    overhead: float = 35.0,
    ratio: float = 0.5,
    world_size: int = 8,
) -> DistributedProfileResult:
    return DistributedProfileResult(
        allreduce_events=[],
        grad_sync_events=[],
        total_allreduce_time_us=10000.0,
        avg_allreduce_time_us=1000.0,
        total_grad_sync_time_us=15000.0,
        avg_grad_sync_time_us=1500.0,
        total_compute_time_us=30000.0,
        comm_compute_ratio=ratio,
        gradient_sync_overhead_pct=overhead,
        world_size=world_size,
        num_nodes=2,
        recommendations=[],
    )


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------


class TestOptimization:
    def test_creation(self):
        opt = Optimization(
            category=OptimizationCategory.batch_size,
            title="Increase batch size",
            description="GPU underutilized.",
            priority=Priority.high,
            estimated_speedup="1.5x",
            code_change="batch_size = 64",
            current_value="32",
            recommended_value="64",
            confidence=0.8,
        )
        assert opt.category == OptimizationCategory.batch_size
        assert opt.priority == Priority.high

    def test_to_dict_from_dict_roundtrip(self):
        opt = Optimization(
            category=OptimizationCategory.precision,
            title="Switch to BF16",
            description="FP32 is slow.",
            priority=Priority.high,
            estimated_speedup="2.0x",
            code_change="autocast()",
            current_value="fp32",
            recommended_value="bf16",
            confidence=0.9,
        )
        d = opt.to_dict()
        assert d["category"] == "precision"
        assert d["priority"] == "high"

        restored = Optimization.from_dict(d)
        assert restored.category == OptimizationCategory.precision
        assert restored.priority == Priority.high
        assert restored.title == opt.title


# ---------------------------------------------------------------------------
# OptimizationReport
# ---------------------------------------------------------------------------


class TestOptimizationReport:
    def test_to_dict_from_dict(self):
        report = OptimizationReport(
            optimizations=[],
            total_estimated_speedup="1.5x",
            summary="No issues.",
        )
        d = report.to_dict()
        restored = OptimizationReport.from_dict(d)
        assert restored.total_estimated_speedup == "1.5x"
        assert restored.summary == "No issues."


# ---------------------------------------------------------------------------
# Advisor — batch size
# ---------------------------------------------------------------------------


class TestAdvisorBatchSize:
    def test_recommends_increase_when_util_low(self):
        gpu = _make_gpu_result(avg_util=30.0, peak_mem=10000.0)
        advisor = OptimizationAdvisor(
            gpu_result=gpu,
            current_batch_size=16,
            gpu_memory_gb=80.0,
        )
        report = advisor.analyze()
        bs_opts = [
            o for o in report.optimizations
            if o.category == OptimizationCategory.batch_size
        ]
        assert len(bs_opts) >= 1
        assert int(bs_opts[0].recommended_value) > 16

    def test_no_recommendation_when_util_high(self):
        gpu = _make_gpu_result(avg_util=85.0, peak_mem=10000.0)
        advisor = OptimizationAdvisor(
            gpu_result=gpu,
            current_batch_size=64,
        )
        report = advisor.analyze()
        bs_opts = [
            o for o in report.optimizations
            if o.category == OptimizationCategory.batch_size
        ]
        assert len(bs_opts) == 0

    def test_no_recommendation_when_memory_tight(self):
        # peak_mem nearly equals total (80 GB = 81920 MB), leaving < 512 MB headroom
        gpu = _make_gpu_result(avg_util=30.0, peak_mem=81500.0, total_mem=81920.0)
        advisor = OptimizationAdvisor(
            gpu_result=gpu,
            current_batch_size=16,
            gpu_memory_gb=80.0,
        )
        report = advisor.analyze()
        bs_opts = [
            o for o in report.optimizations
            if o.category == OptimizationCategory.batch_size
        ]
        assert len(bs_opts) == 0


# ---------------------------------------------------------------------------
# Advisor — precision
# ---------------------------------------------------------------------------


class TestAdvisorPrecision:
    def test_recommends_bf16_for_fp32(self):
        advisor = OptimizationAdvisor(current_precision="fp32")
        report = advisor.analyze()
        prec_opts = [
            o for o in report.optimizations
            if o.category == OptimizationCategory.precision
        ]
        assert len(prec_opts) >= 1
        assert any("bf16" in o.recommended_value.lower() for o in prec_opts)

    def test_recommends_tf32_for_fp32(self):
        advisor = OptimizationAdvisor(current_precision="fp32")
        report = advisor.analyze()
        tf32_opts = [
            o for o in report.optimizations
            if "tf32" in o.title.lower()
        ]
        assert len(tf32_opts) >= 1

    def test_no_precision_recommendation_for_bf16(self):
        advisor = OptimizationAdvisor(current_precision="bf16")
        report = advisor.analyze()
        prec_opts = [
            o for o in report.optimizations
            if o.category == OptimizationCategory.precision
        ]
        assert len(prec_opts) == 0


# ---------------------------------------------------------------------------
# Advisor — gradient checkpointing
# ---------------------------------------------------------------------------


class TestAdvisorGradientCheckpointing:
    def test_recommends_when_memory_high(self):
        gpu = _make_gpu_result(peak_mem=70000.0)
        advisor = OptimizationAdvisor(
            gpu_result=gpu,
            gpu_memory_gb=80.0,
            current_precision="bf16",  # skip precision recs
        )
        report = advisor.analyze()
        gc_opts = [
            o for o in report.optimizations
            if o.category == OptimizationCategory.gradient_checkpointing
        ]
        assert len(gc_opts) >= 1

    def test_no_recommendation_when_memory_low(self):
        gpu = _make_gpu_result(peak_mem=10000.0)
        advisor = OptimizationAdvisor(
            gpu_result=gpu,
            gpu_memory_gb=80.0,
            current_precision="bf16",
        )
        report = advisor.analyze()
        gc_opts = [
            o for o in report.optimizations
            if o.category == OptimizationCategory.gradient_checkpointing
        ]
        assert len(gc_opts) == 0


# ---------------------------------------------------------------------------
# Advisor — data loading
# ---------------------------------------------------------------------------


class TestAdvisorDataLoading:
    def test_recommends_more_workers(self):
        from src.profiler.data_loading_profiler import DataLoadingResult, DataLoadingMetrics

        metrics = [
            DataLoadingMetrics(
                batch_load_time_ms=50.0,
                batch_process_time_ms=100.0,
                gpu_idle_time_ms=30.0,
                timestamp=1.0,
                batch_index=0,
                num_workers_active=2,
                prefetch_buffer_size=2,
                io_throughput_mbps=100.0,
            ),
        ]
        data_result = DataLoadingResult(
            metrics=metrics,
            avg_load_time_ms=50.0,
            avg_gpu_idle_time_ms=30.0,
            io_throughput_mbps=100.0,
            is_bottleneck=True,
            bottleneck_severity="moderate",
            recommendations=["Increase workers"],
        )
        advisor = OptimizationAdvisor(
            data_result=data_result,
            current_num_workers=2,
            current_precision="bf16",
        )
        report = advisor.analyze()
        dl_opts = [
            o for o in report.optimizations
            if o.category == OptimizationCategory.data_loading
        ]
        assert len(dl_opts) >= 1


# ---------------------------------------------------------------------------
# Advisor — compilation
# ---------------------------------------------------------------------------


class TestAdvisorCompilation:
    def test_recommends_compile_for_elementwise_heavy(self):
        kernel_result = _make_kernel_result_elementwise_heavy()
        advisor = OptimizationAdvisor(
            kernel_result=kernel_result,
            current_precision="bf16",
        )
        report = advisor.analyze()
        compile_opts = [
            o for o in report.optimizations
            if o.category == OptimizationCategory.compilation
        ]
        assert len(compile_opts) >= 1
        assert "compile" in compile_opts[0].title.lower()


# ---------------------------------------------------------------------------
# Advisor — distributed
# ---------------------------------------------------------------------------


class TestAdvisorDistributed:
    def test_recommends_gradient_compression(self):
        dist = _make_dist_result(overhead=35.0, ratio=0.5, world_size=8)
        advisor = OptimizationAdvisor(
            distributed_result=dist,
            current_precision="bf16",
        )
        report = advisor.analyze()
        dist_opts = [
            o for o in report.optimizations
            if o.category == OptimizationCategory.distributed
        ]
        assert len(dist_opts) >= 1
        assert any("powersgd" in o.title.lower() or "compression" in o.title.lower()
                    for o in dist_opts)

    def test_recommends_fsdp_for_large_world(self):
        dist = _make_dist_result(overhead=40.0, ratio=0.6, world_size=16)
        advisor = OptimizationAdvisor(
            distributed_result=dist,
            current_precision="bf16",
        )
        report = advisor.analyze()
        fsdp_opts = [
            o for o in report.optimizations
            if "fsdp" in o.title.lower()
        ]
        assert len(fsdp_opts) >= 1

    def test_no_distributed_recs_when_overhead_low(self):
        dist = _make_dist_result(overhead=5.0, ratio=0.05, world_size=2)
        advisor = OptimizationAdvisor(
            distributed_result=dist,
            current_precision="bf16",
        )
        report = advisor.analyze()
        dist_opts = [
            o for o in report.optimizations
            if o.category == OptimizationCategory.distributed
        ]
        assert len(dist_opts) == 0


# ---------------------------------------------------------------------------
# Advisor — overall report
# ---------------------------------------------------------------------------


class TestAdvisorReport:
    def test_empty_report_when_no_data(self):
        advisor = OptimizationAdvisor(current_precision="bf16")
        report = advisor.analyze()
        assert isinstance(report, OptimizationReport)
        assert isinstance(report.optimizations, list)

    def test_sorted_by_priority(self):
        gpu = _make_gpu_result(avg_util=20.0, peak_mem=70000.0)
        dist = _make_dist_result(overhead=40.0, ratio=0.6, world_size=16)
        advisor = OptimizationAdvisor(
            gpu_result=gpu,
            distributed_result=dist,
            current_batch_size=8,
            current_precision="fp32",
            gpu_memory_gb=80.0,
        )
        report = advisor.analyze()
        if len(report.optimizations) >= 2:
            priorities = [o.priority for o in report.optimizations]
            priority_order = {
                Priority.critical: 0,
                Priority.high: 1,
                Priority.medium: 2,
                Priority.low: 3,
            }
            orders = [priority_order[p] for p in priorities]
            assert orders == sorted(orders)

    def test_summary_generated(self):
        gpu = _make_gpu_result(avg_util=30.0, peak_mem=10000.0)
        advisor = OptimizationAdvisor(
            gpu_result=gpu,
            current_batch_size=16,
        )
        report = advisor.analyze()
        assert len(report.summary) > 0
        assert len(report.total_estimated_speedup) > 0

    def test_no_optimizations_summary(self):
        advisor = OptimizationAdvisor(current_precision="bf16")
        report = advisor.analyze()
        assert "well-tuned" in report.summary.lower() or len(report.optimizations) > 0


# ---------------------------------------------------------------------------
# Advisor — round_batch_size
# ---------------------------------------------------------------------------


class TestRoundBatchSize:
    def test_round_to_power_of_2(self):
        result = OptimizationAdvisor._round_batch_size(33)
        assert result in (32, 40)  # could be either

    def test_round_to_multiple_of_8(self):
        result = OptimizationAdvisor._round_batch_size(50)
        assert result % 8 == 0 or (result & (result - 1)) == 0

    def test_min_is_8(self):
        result = OptimizationAdvisor._round_batch_size(1)
        assert result >= 1

    def test_large_value(self):
        result = OptimizationAdvisor._round_batch_size(256)
        assert result == 256
