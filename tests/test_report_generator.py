"""Tests for src.analysis.report_generator."""

import json
import os
import tempfile

import pytest

from src.analysis.report_generator import (
    ReportGenerator,
    _efficiency_color,
    _efficiency_hex,
    _format_duration,
    _truncate_kernel_name,
)
from src.analysis.bottleneck_detector import (
    Bottleneck,
    PerformanceSummary,
    TimeCategory,
)
from src.profiler.gpu_profiler import GPUMetricSnapshot, ProfileResult as GPUProfileResult
from src.profiler.kernel_tracer import KernelRecord, KernelTraceResult
from src.profiler.memory_analyzer import (
    MemorySnapshot, MemoryAnalysisResult, MemoryBreakdown, OOMRiskAssessment,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_summary(**kwargs):
    defaults = {
        "total_time_s": 10.0,
        "compute_pct": 70.0,
        "memory_pct": 10.0,
        "communication_pct": 10.0,
        "data_loading_pct": 5.0,
        "idle_pct": 5.0,
        "primary_bottleneck": "compute",
        "bottlenecks": [],
        "overall_efficiency": 0.7,
        "summary_text": "Test summary.",
    }
    defaults.update(kwargs)
    return PerformanceSummary(**defaults)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

class TestUtilityFunctions:
    def test_efficiency_color(self):
        assert _efficiency_color(80.0) == "green"
        assert _efficiency_color(50.0) == "yellow"
        assert _efficiency_color(20.0) == "red"

    def test_efficiency_hex(self):
        assert _efficiency_hex(80.0) == "#4ade80"
        assert _efficiency_hex(50.0) == "#facc15"
        assert _efficiency_hex(20.0) == "#f87171"

    def test_format_duration_ms(self):
        assert "ms" in _format_duration(0.5)

    def test_format_duration_seconds(self):
        result = _format_duration(30.0)
        assert "s" in result

    def test_format_duration_minutes(self):
        result = _format_duration(120.0)
        assert "m" in result

    def test_format_duration_hours(self):
        result = _format_duration(7200.0)
        assert "h" in result

    def test_truncate_kernel_name_short(self):
        name = "short_kernel"
        assert _truncate_kernel_name(name) == name

    def test_truncate_kernel_name_long(self):
        name = "a" * 100
        result = _truncate_kernel_name(name, max_len=60)
        assert len(result) == 60
        assert result.startswith("...")


# ---------------------------------------------------------------------------
# ReportGenerator
# ---------------------------------------------------------------------------

class TestReportGenerator:
    def test_generate_terminal_report(self, capsys):
        summary = _make_summary()
        gen = ReportGenerator(performance_summary=summary)
        gen.generate_report(format="terminal")
        # Should print something to stdout (Rich output)
        # No assertion on content, just ensure no exception

    def test_generate_json_report(self):
        summary = _make_summary()
        gen = ReportGenerator(performance_summary=summary)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            result_path = gen.generate_report(format="json", output_path=path)
            assert result_path == path
            assert os.path.exists(path)

            with open(path) as f:
                data = json.load(f)
            assert "summary" in data
            assert "metadata" in data
        finally:
            os.unlink(path)

    def test_generate_html_report(self):
        summary = _make_summary()
        gen = ReportGenerator(performance_summary=summary)

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name

        try:
            result_path = gen.generate_report(format="html", output_path=path)
            assert result_path == path
            assert os.path.exists(path)

            with open(path) as f:
                content = f.read()
            assert "Chamber Profiler" in content
            assert "<canvas" in content
        finally:
            os.unlink(path)

    def test_generate_report_invalid_format(self):
        summary = _make_summary()
        gen = ReportGenerator(performance_summary=summary)
        with pytest.raises(ValueError, match="Unknown report format"):
            gen.generate_report(format="pdf")

    def test_generate_report_default_paths(self):
        summary = _make_summary()
        gen = ReportGenerator(performance_summary=summary)

        # HTML default path
        result = gen.generate_report(format="html")
        assert result == "chamber_profile_report.html"
        if os.path.exists(result):
            os.unlink(result)

        # JSON default path
        result = gen.generate_report(format="json")
        assert result == "chamber_profile_report.json"
        if os.path.exists(result):
            os.unlink(result)

    def test_json_report_with_all_results(self):
        summary = _make_summary(
            bottlenecks=[
                Bottleneck(
                    category=TimeCategory.compute,
                    time_pct=70.0,
                    description="Low SM utilization",
                    impact_score=0.8,
                    recommendations=["Increase batch size"],
                    expected_speedup="2x",
                )
            ]
        )

        gpu_snaps = [
            GPUMetricSnapshot(
                timestamp=float(i), gpu_index=0, utilization_pct=80.0,
                memory_used_mb=4000.0, memory_total_mb=8000.0,
                power_watts=200.0, temperature_c=70.0,
                sm_activity_pct=75.0, memory_bandwidth_pct=30.0,
                clock_speed_mhz=1500.0,
            )
            for i in range(3)
        ]
        gpu_result = GPUProfileResult(
            duration_s=3.0, snapshots=gpu_snaps, gpu_count=1,
        )

        kernel_result = KernelTraceResult(
            kernels=[
                KernelRecord(
                    name="gemm", duration_us=100.0,
                    grid_size=(1, 1, 1), block_size=(256, 1, 1),
                    shared_memory_bytes=0, device_index=0,
                    category="gemm", occupancy=0.7,
                ),
            ],
            total_kernel_time_us=100.0,
            top_kernels=[],
            inefficient_kernels=[],
            category_breakdown={"gemm": 100.0},
        )

        mem_snaps = [
            MemorySnapshot(
                timestamp=1.0, allocated_mb=2000.0, reserved_mb=4000.0,
                peak_allocated_mb=2500.0, peak_reserved_mb=4000.0,
                num_tensors=50, device_index=0,
            ),
        ]
        memory_result = MemoryAnalysisResult(
            snapshots=mem_snaps,
            breakdown=MemoryBreakdown(parameter_memory_mb=500.0),
            leak_detected=False,
            leak_points=[],
            oom_risk=OOMRiskAssessment(
                risk_score=0.3, peak_usage_pct=50.0,
                headroom_mb=4000.0, suggestions=[],
            ),
            peak_snapshot=mem_snaps[0],
        )

        gen = ReportGenerator(
            performance_summary=summary,
            gpu_result=gpu_result,
            kernel_result=kernel_result,
            memory_result=memory_result,
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            gen.generate_report(format="json", output_path=path)
            with open(path) as f:
                data = json.load(f)
            assert "gpu_profile" in data
            assert "kernel_trace" in data
            assert "memory_analysis" in data
        finally:
            os.unlink(path)

    def test_html_report_with_gpu_and_memory(self):
        summary = _make_summary()

        gpu_snaps = [
            GPUMetricSnapshot(
                timestamp=float(i), gpu_index=0, utilization_pct=60.0,
                memory_used_mb=3000.0, memory_total_mb=8000.0,
                power_watts=180.0, temperature_c=65.0,
                sm_activity_pct=55.0, memory_bandwidth_pct=40.0,
                clock_speed_mhz=1400.0,
            )
            for i in range(5)
        ]
        gpu_result = GPUProfileResult(
            duration_s=5.0, snapshots=gpu_snaps, gpu_count=1,
        )

        mem_snaps = [
            MemorySnapshot(
                timestamp=float(i), allocated_mb=float(1000 + i * 100),
                reserved_mb=3000.0, peak_allocated_mb=1400.0,
                peak_reserved_mb=3000.0, num_tensors=0, device_index=0,
            )
            for i in range(5)
        ]
        memory_result = MemoryAnalysisResult(
            snapshots=mem_snaps,
            breakdown=MemoryBreakdown(
                parameter_memory_mb=200.0,
                gradient_memory_mb=200.0,
                optimizer_state_mb=400.0,
                activation_memory_mb=200.0,
            ),
            leak_detected=True,
            leak_points=[(4.0, 1400.0)],
            oom_risk=OOMRiskAssessment(
                risk_score=0.4, peak_usage_pct=60.0,
                headroom_mb=3200.0, suggestions=["Use mixed precision"],
            ),
            peak_snapshot=mem_snaps[-1],
        )

        gen = ReportGenerator(
            performance_summary=summary,
            gpu_result=gpu_result,
            memory_result=memory_result,
        )

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name

        try:
            gen.generate_report(format="html", output_path=path)
            with open(path) as f:
                content = f.read()
            assert "GPU Utilization Over Time" in content
            assert "Memory Usage" in content
            assert "Memory leak detected" in content
        finally:
            os.unlink(path)
