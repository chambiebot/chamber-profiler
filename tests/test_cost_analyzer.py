"""Tests for src.analysis.cost_analyzer."""

import pytest

from src.analysis.cost_analyzer import (
    GPU_CATALOG,
    CostEstimate,
    GPURecommendation,
    CostAnalysisResult,
    CostAnalyzer,
)


# ---------------------------------------------------------------------------
# CostEstimate
# ---------------------------------------------------------------------------

class TestCostEstimate:
    def test_creation(self):
        est = CostEstimate(
            gpu_name="NVIDIA A100 80GB",
            num_gpus=4,
            cost_per_gpu_hour=4.10,
            total_cost_per_hour=16.40,
            profiled_duration_hours=0.5,
            estimated_session_cost=8.20,
            gpu_utilization_pct=75.0,
            effective_cost_per_util_hour=21.87,
        )
        assert est.gpu_name == "NVIDIA A100 80GB"
        assert est.num_gpus == 4
        assert est.total_cost_per_hour == 16.40

    def test_to_dict_from_dict_roundtrip(self):
        est = CostEstimate(
            gpu_name="A100", num_gpus=2, cost_per_gpu_hour=3.40,
            total_cost_per_hour=6.80, profiled_duration_hours=1.0,
            estimated_session_cost=6.80, gpu_utilization_pct=80.0,
            effective_cost_per_util_hour=8.50,
        )
        d = est.to_dict()
        restored = CostEstimate.from_dict(d)
        assert restored.gpu_name == est.gpu_name
        assert restored.total_cost_per_hour == est.total_cost_per_hour

    def test_from_dict_defaults(self):
        est = CostEstimate.from_dict({})
        assert est.gpu_name == ""
        assert est.num_gpus == 0


# ---------------------------------------------------------------------------
# GPURecommendation
# ---------------------------------------------------------------------------

class TestGPURecommendation:
    def test_creation(self):
        rec = GPURecommendation(
            gpu_key="t4", gpu_name="NVIDIA T4",
            cost_per_gpu_hour=0.53, estimated_cost_per_hour=1.06,
            num_gpus=2, memory_gb=16, fp16_tflops=65.0,
            estimated_savings_pct=50.0, reason="Cheaper option",
        )
        assert rec.gpu_key == "t4"
        assert rec.estimated_savings_pct == 50.0

    def test_to_dict_from_dict_roundtrip(self):
        rec = GPURecommendation(
            gpu_key="l4", gpu_name="NVIDIA L4",
            cost_per_gpu_hour=0.81, estimated_cost_per_hour=1.62,
            num_gpus=2, memory_gb=24, fp16_tflops=121.0,
            estimated_savings_pct=40.0, reason="Good value",
        )
        d = rec.to_dict()
        restored = GPURecommendation.from_dict(d)
        assert restored.gpu_key == rec.gpu_key
        assert restored.estimated_savings_pct == rec.estimated_savings_pct


# ---------------------------------------------------------------------------
# CostAnalysisResult
# ---------------------------------------------------------------------------

class TestCostAnalysisResult:
    def test_to_dict_from_dict_roundtrip(self):
        est = CostEstimate(
            gpu_name="A100", num_gpus=1, cost_per_gpu_hour=4.10,
            total_cost_per_hour=4.10, profiled_duration_hours=1.0,
            estimated_session_cost=4.10, gpu_utilization_pct=80.0,
            effective_cost_per_util_hour=5.125,
        )
        rec = GPURecommendation(
            gpu_key="t4", gpu_name="NVIDIA T4",
            cost_per_gpu_hour=0.53, estimated_cost_per_hour=1.06,
            num_gpus=2, memory_gb=16, fp16_tflops=65.0,
            estimated_savings_pct=50.0, reason="Cheaper",
        )
        result = CostAnalysisResult(
            current_estimate=est, recommendations=[rec],
            potential_monthly_savings=2200.0, summary="Test summary",
        )
        d = result.to_dict()
        restored = CostAnalysisResult.from_dict(d)
        assert restored.potential_monthly_savings == 2200.0
        assert len(restored.recommendations) == 1
        assert restored.summary == "Test summary"

    def test_from_dict_empty(self):
        result = CostAnalysisResult.from_dict({})
        assert result.current_estimate.gpu_name == ""
        assert result.recommendations == []


# ---------------------------------------------------------------------------
# GPU_CATALOG
# ---------------------------------------------------------------------------

class TestGPUCatalog:
    def test_has_expected_gpus(self):
        expected_keys = ["a100_40gb", "a100_80gb", "h100_80gb", "t4", "l4"]
        for key in expected_keys:
            assert key in GPU_CATALOG, f"{key} missing from GPU_CATALOG"

    def test_catalog_entries_have_required_fields(self):
        required_fields = ["name", "memory_gb", "fp16_tflops", "price_per_hour", "provider"]
        for key, info in GPU_CATALOG.items():
            for field in required_fields:
                assert field in info, f"{key} missing field {field}"

    def test_prices_are_positive(self):
        for key, info in GPU_CATALOG.items():
            assert info["price_per_hour"] > 0, f"{key} has non-positive price"


# ---------------------------------------------------------------------------
# CostAnalyzer — basic analysis
# ---------------------------------------------------------------------------

class TestCostAnalyzerBasic:
    def test_analyze_without_gpu_result(self):
        analyzer = CostAnalyzer(gpu_name="a100_80gb", num_gpus=4)
        result = analyzer.analyze()
        assert isinstance(result, CostAnalysisResult)
        assert result.current_estimate.gpu_name == "NVIDIA A100 80GB"
        assert result.current_estimate.num_gpus == 4

    def test_estimate_cost_defaults(self):
        analyzer = CostAnalyzer(gpu_name="h100_80gb", num_gpus=2)
        est = analyzer.estimate_cost()
        assert est.cost_per_gpu_hour == 5.50
        assert est.total_cost_per_hour == pytest.approx(11.0)

    def test_estimate_cost_custom_config(self):
        analyzer = CostAnalyzer(gpu_name="a100_80gb", num_gpus=1)
        est = analyzer.estimate_cost(gpu_name="t4", num_gpus=4, duration_hours=2.0)
        assert est.cost_per_gpu_hour == 0.53
        assert est.total_cost_per_hour == pytest.approx(2.12)
        assert est.estimated_session_cost == pytest.approx(4.24)

    def test_estimate_cost_custom_price(self):
        analyzer = CostAnalyzer(
            gpu_name="a100_80gb", num_gpus=1, cost_per_gpu_hour=5.00,
        )
        est = analyzer.estimate_cost()
        assert est.cost_per_gpu_hour == 5.00

    def test_analyze_num_gpus_minimum_one(self):
        analyzer = CostAnalyzer(gpu_name="t4", num_gpus=0)
        result = analyzer.analyze()
        assert result.current_estimate.num_gpus == 1


# ---------------------------------------------------------------------------
# CostAnalyzer — recommendations
# ---------------------------------------------------------------------------

class TestCostAnalyzerRecommendations:
    def test_finds_cheaper_alternatives(self):
        analyzer = CostAnalyzer(gpu_name="h100_80gb", num_gpus=1)
        result = analyzer.analyze()
        # H100 is expensive; there should be cheaper options
        assert len(result.recommendations) > 0

    def test_recommendations_sorted_by_savings(self):
        analyzer = CostAnalyzer(gpu_name="h100_80gb", num_gpus=4)
        result = analyzer.analyze()
        if len(result.recommendations) >= 2:
            for i in range(len(result.recommendations) - 1):
                assert (
                    result.recommendations[i].estimated_savings_pct
                    >= result.recommendations[i + 1].estimated_savings_pct
                )

    def test_no_self_recommendation(self):
        analyzer = CostAnalyzer(gpu_name="a100_80gb", num_gpus=1)
        result = analyzer.analyze()
        for rec in result.recommendations:
            assert rec.gpu_key != "a100_80gb"

    def test_memory_constraint_filters_recommendations(self):
        # If we need 70GB, only GPUs with >= 70GB memory should be recommended
        analyzer = CostAnalyzer(
            gpu_name="h100_80gb", num_gpus=1, peak_memory_mb=70_000,
        )
        result = analyzer.analyze()
        for rec in result.recommendations:
            assert rec.memory_gb >= 70000 / 1024.0

    def test_cheap_gpu_has_fewer_recommendations(self):
        # T4 is already cheap — fewer alternatives should be cheaper
        analyzer = CostAnalyzer(gpu_name="rtx_3090", num_gpus=1)
        result_cheap = analyzer.analyze()
        analyzer2 = CostAnalyzer(gpu_name="h100_80gb", num_gpus=1)
        result_expensive = analyzer2.analyze()
        assert len(result_expensive.recommendations) >= len(result_cheap.recommendations)


# ---------------------------------------------------------------------------
# CostAnalyzer — monthly savings
# ---------------------------------------------------------------------------

class TestCostAnalyzerMonthlySavings:
    def test_monthly_savings_calculation(self):
        current = CostEstimate(
            gpu_name="H100", num_gpus=1, cost_per_gpu_hour=5.50,
            total_cost_per_hour=5.50, profiled_duration_hours=1.0,
            estimated_session_cost=5.50, gpu_utilization_pct=80.0,
            effective_cost_per_util_hour=6.875,
        )
        rec = GPURecommendation(
            gpu_key="t4", gpu_name="NVIDIA T4",
            cost_per_gpu_hour=0.53, estimated_cost_per_hour=1.06,
            num_gpus=2, memory_gb=16, fp16_tflops=65.0,
            estimated_savings_pct=80.0, reason="Much cheaper",
        )
        savings = CostAnalyzer._estimate_monthly_savings(current, [rec])
        expected = (5.50 - 1.06) * 730.0
        assert savings == pytest.approx(expected)

    def test_monthly_savings_empty_recs(self):
        current = CostEstimate(
            gpu_name="T4", num_gpus=1, cost_per_gpu_hour=0.53,
            total_cost_per_hour=0.53, profiled_duration_hours=1.0,
            estimated_session_cost=0.53, gpu_utilization_pct=80.0,
            effective_cost_per_util_hour=0.6625,
        )
        savings = CostAnalyzer._estimate_monthly_savings(current, [])
        assert savings == 0.0


# ---------------------------------------------------------------------------
# CostAnalyzer — summary
# ---------------------------------------------------------------------------

class TestCostAnalyzerSummary:
    def test_summary_with_recommendations(self):
        analyzer = CostAnalyzer(gpu_name="h100_80gb", num_gpus=4)
        result = analyzer.analyze()
        assert "Current config" in result.summary
        assert "$" in result.summary

    def test_summary_without_recommendations(self):
        summary = CostAnalyzer._generate_summary(
            CostEstimate(
                gpu_name="T4", num_gpus=1, cost_per_gpu_hour=0.53,
                total_cost_per_hour=0.53, profiled_duration_hours=0.0,
                estimated_session_cost=0.0, gpu_utilization_pct=0.0,
                effective_cost_per_util_hour=0.53,
            ),
            [],
            0.0,
        )
        assert "No cheaper alternatives" in summary

    def test_unknown_gpu_returns_zero_price(self):
        analyzer = CostAnalyzer(gpu_name="unknown_gpu_xyz", num_gpus=1)
        est = analyzer.estimate_cost()
        assert est.cost_per_gpu_hour == 0.0
