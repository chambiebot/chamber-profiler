"""Tests for src.analysis.executive_summary."""

import pytest

from src.profiler.gpu_profiler import ProfileResult, GPUMetricSnapshot
from src.profiler.memory_analyzer import (
    MemoryAnalysisResult,
    MemorySnapshot,
    OOMRiskAssessment,
)
from src.analysis.cost_analyzer import (
    CostAnalysisResult,
    CostEstimate,
    GPURecommendation,
)
from src.analysis.executive_summary import (
    Recommendation,
    ExecutiveSummary,
    ExecutiveSummaryGenerator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gpu_result(
    avg_util: float = 70.0,
    peak_mem: float = 8000.0,
    duration_s: float = 3600.0,
) -> ProfileResult:
    snap = GPUMetricSnapshot(
        timestamp=1000.0, gpu_index=0,
        utilization_pct=avg_util,
        memory_used_mb=peak_mem,
        memory_total_mb=16000.0,
        power_watts=250.0,
        temperature_c=72.0,
        sm_activity_pct=avg_util,
        memory_bandwidth_pct=50.0,
        clock_speed_mhz=1500.0,
    )
    return ProfileResult(
        duration_s=duration_s,
        snapshots=[snap],
        gpu_count=1,
    )


def _make_cost_result(
    hourly: float = 4.10,
    savings: float = 1500.0,
) -> CostAnalysisResult:
    est = CostEstimate(
        gpu_name="A100 80GB", num_gpus=4,
        cost_per_gpu_hour=hourly / 4,
        total_cost_per_hour=hourly,
        profiled_duration_hours=1.0,
        estimated_session_cost=hourly,
        gpu_utilization_pct=70.0,
        effective_cost_per_util_hour=hourly / 0.7,
    )
    rec = GPURecommendation(
        gpu_key="l4", gpu_name="NVIDIA L4",
        cost_per_gpu_hour=0.81,
        estimated_cost_per_hour=3.24,
        num_gpus=4, memory_gb=24,
        fp16_tflops=121.0,
        estimated_savings_pct=21.0,
        reason="Cheaper GPU",
    )
    return CostAnalysisResult(
        current_estimate=est,
        recommendations=[rec],
        potential_monthly_savings=savings,
        summary="Test summary",
    )


def _make_memory_result(
    leak: bool = False,
    oom_risk: float = 0.2,
) -> MemoryAnalysisResult:
    snap = MemorySnapshot(
        timestamp=1.0, allocated_mb=5000.0, reserved_mb=8000.0,
        peak_allocated_mb=6000.0, peak_reserved_mb=8000.0,
        num_tensors=100, device_index=0,
    )
    return MemoryAnalysisResult(
        snapshots=[snap],
        breakdown=None,
        leak_detected=leak,
        leak_points=[(1.0, 5000.0)] if leak else [],
        oom_risk=OOMRiskAssessment(risk_score=oom_risk),
        peak_snapshot=snap,
    )


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

class TestRecommendation:
    def test_creation(self):
        rec = Recommendation(
            title="Increase batch size",
            description="Double the batch size for 2x throughput.",
            estimated_savings_pct=15.0,
            effort="low",
            priority=1,
        )
        assert rec.title == "Increase batch size"
        assert rec.priority == 1

    def test_to_dict_from_dict_roundtrip(self):
        rec = Recommendation(
            title="Switch GPU", description="Use L4",
            estimated_savings_pct=20.0, effort="high", priority=2,
        )
        d = rec.to_dict()
        restored = Recommendation.from_dict(d)
        assert restored.title == rec.title
        assert restored.estimated_savings_pct == 20.0


# ---------------------------------------------------------------------------
# ExecutiveSummary
# ---------------------------------------------------------------------------

class TestExecutiveSummary:
    def test_to_dict_from_dict_roundtrip(self):
        rec = Recommendation(
            title="Test", description="Do things",
            estimated_savings_pct=10.0, effort="low", priority=1,
        )
        summary = ExecutiveSummary(
            gpu_utilization_pct=75.0,
            monthly_gpu_cost=3000.0,
            potential_monthly_savings=500.0,
            potential_savings_pct=16.7,
            memory_health="healthy",
            io_health="healthy",
            overall_health="healthy",
            top_recommendations=[rec],
            summary_text="All good",
            key_findings=["GPU util is 75%"],
        )
        d = summary.to_dict()
        restored = ExecutiveSummary.from_dict(d)
        assert restored.gpu_utilization_pct == 75.0
        assert len(restored.top_recommendations) == 1
        assert len(restored.key_findings) == 1

    def test_empty_roundtrip(self):
        summary = ExecutiveSummary(
            gpu_utilization_pct=0.0, monthly_gpu_cost=0.0,
            potential_monthly_savings=0.0, potential_savings_pct=0.0,
            memory_health="unknown", io_health="unknown",
            overall_health="unknown", top_recommendations=[],
            summary_text="", key_findings=[],
        )
        d = summary.to_dict()
        restored = ExecutiveSummary.from_dict(d)
        assert restored.memory_health == "unknown"


# ---------------------------------------------------------------------------
# ExecutiveSummaryGenerator
# ---------------------------------------------------------------------------

class TestExecutiveSummaryGenerator:
    def test_generate_with_all_data(self):
        gen = ExecutiveSummaryGenerator(
            gpu_result=_make_gpu_result(avg_util=70.0),
            cost_result=_make_cost_result(hourly=4.10, savings=1500.0),
            memory_result=_make_memory_result(leak=False, oom_risk=0.2),
            num_gpus=4,
            gpu_name="A100 80GB",
        )
        summary = gen.generate()
        assert summary.gpu_utilization_pct == 70.0
        assert summary.monthly_gpu_cost > 0
        assert summary.potential_monthly_savings > 0
        assert summary.memory_health == "healthy"
        assert summary.overall_health == "healthy"
        assert len(summary.summary_text) > 0
        assert len(summary.key_findings) > 0

    def test_generate_no_data(self):
        gen = ExecutiveSummaryGenerator()
        summary = gen.generate()
        assert summary.gpu_utilization_pct == 0.0
        assert summary.monthly_gpu_cost == 0.0
        assert summary.memory_health == "unknown"

    def test_low_utilization_warning(self):
        gen = ExecutiveSummaryGenerator(
            gpu_result=_make_gpu_result(avg_util=25.0),
        )
        summary = gen.generate()
        assert summary.overall_health == "warning"
        # Should recommend increasing utilization
        any_util_rec = any(
            "utilization" in r.title.lower()
            for r in summary.top_recommendations
        )
        assert any_util_rec

    def test_memory_leak_critical(self):
        gen = ExecutiveSummaryGenerator(
            gpu_result=_make_gpu_result(avg_util=80.0),
            memory_result=_make_memory_result(leak=True, oom_risk=0.9),
        )
        summary = gen.generate()
        assert summary.memory_health == "critical"
        assert summary.overall_health == "critical"
        # Should have memory recommendation
        any_mem_rec = any(
            "memory" in r.title.lower()
            for r in summary.top_recommendations
        )
        assert any_mem_rec

    def test_high_oom_risk_critical(self):
        gen = ExecutiveSummaryGenerator(
            memory_result=_make_memory_result(leak=False, oom_risk=0.85),
        )
        summary = gen.generate()
        assert summary.memory_health == "critical"

    def test_moderate_oom_risk_warning(self):
        gen = ExecutiveSummaryGenerator(
            memory_result=_make_memory_result(leak=False, oom_risk=0.6),
        )
        summary = gen.generate()
        assert summary.memory_health == "warning"

    def test_cost_recommendation(self):
        gen = ExecutiveSummaryGenerator(
            gpu_result=_make_gpu_result(avg_util=80.0),
            cost_result=_make_cost_result(hourly=4.10, savings=1500.0),
        )
        summary = gen.generate()
        assert summary.potential_monthly_savings > 0
        # Should have GPU switch recommendation
        any_gpu_rec = any(
            "switch" in r.title.lower() or "l4" in r.title.lower()
            for r in summary.top_recommendations
        )
        assert any_gpu_rec

    def test_max_three_recommendations(self):
        gen = ExecutiveSummaryGenerator(
            gpu_result=_make_gpu_result(avg_util=25.0),
            cost_result=_make_cost_result(hourly=4.10, savings=1500.0),
            memory_result=_make_memory_result(leak=True, oom_risk=0.9),
        )
        summary = gen.generate()
        assert len(summary.top_recommendations) <= 3

    def test_summary_text_contains_key_info(self):
        gen = ExecutiveSummaryGenerator(
            gpu_result=_make_gpu_result(avg_util=60.0),
            cost_result=_make_cost_result(hourly=4.10, savings=1500.0),
            num_gpus=4,
            gpu_name="A100",
        )
        summary = gen.generate()
        assert "EXECUTIVE SUMMARY" in summary.summary_text
        assert "KEY METRICS" in summary.summary_text
        assert "$" in summary.summary_text

    def test_savings_percentage_calculation(self):
        gen = ExecutiveSummaryGenerator(
            cost_result=_make_cost_result(hourly=10.0, savings=3650.0),
        )
        summary = gen.generate()
        # monthly_cost = 10.0 * 730 = 7300
        # savings_pct = 3650 / 7300 * 100 = 50%
        assert summary.potential_savings_pct == 50.0

    def test_findings_include_utilization(self):
        gen = ExecutiveSummaryGenerator(
            gpu_result=_make_gpu_result(avg_util=45.0),
            num_gpus=2,
            gpu_name="V100",
        )
        summary = gen.generate()
        any_util = any("45%" in f or "utilization" in f.lower() for f in summary.key_findings)
        assert any_util

    def test_findings_include_cost(self):
        gen = ExecutiveSummaryGenerator(
            cost_result=_make_cost_result(hourly=4.10, savings=1500.0),
        )
        summary = gen.generate()
        any_cost = any("$" in f for f in summary.key_findings)
        assert any_cost

    def test_health_assessment_healthy(self):
        assert ExecutiveSummaryGenerator._assess_overall_health(80.0, "healthy") == "healthy"

    def test_health_assessment_low_util(self):
        assert ExecutiveSummaryGenerator._assess_overall_health(20.0, "healthy") == "warning"

    def test_health_assessment_critical_memory(self):
        assert ExecutiveSummaryGenerator._assess_overall_health(90.0, "critical") == "critical"
