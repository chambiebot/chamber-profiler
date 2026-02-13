"""Tests for src.analysis.roi_calculator."""

import pytest

from src.analysis.roi_calculator import (
    OptimizationScenario,
    ROICalculator,
    ROIEstimate,
    ROIReport,
    GPU_HOURLY_COST,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scenario(
    name: str = "increase_batch_size",
    before_time: float = 0.1,
    after_time: float = 0.05,
    confidence: float = 0.8,
) -> OptimizationScenario:
    return OptimizationScenario(
        name=name,
        category="batch_size",
        before_value="batch_size=32",
        after_value="batch_size=64",
        before_step_time_s=before_time,
        after_step_time_s=after_time,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Tests: OptimizationScenario
# ---------------------------------------------------------------------------


class TestOptimizationScenario:
    def test_round_trip(self):
        s = _make_scenario()
        d = s.to_dict()
        s2 = OptimizationScenario.from_dict(d)
        assert s2.name == "increase_batch_size"
        assert s2.before_step_time_s == pytest.approx(0.1)
        assert s2.after_step_time_s == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Tests: ROICalculator.estimate
# ---------------------------------------------------------------------------


class TestROIEstimate:
    def test_basic_estimate(self):
        calc = ROICalculator(gpu_type="a100_80gb", gpu_count=1, steps_per_epoch=10000)
        scenario = _make_scenario(before_time=0.1, after_time=0.05)
        est = calc.estimate(scenario)
        assert est.speedup_factor == pytest.approx(2.0)
        assert est.time_saved_per_step_s == pytest.approx(0.05)
        assert est.cost_saved_per_month > 0
        assert "Strongly recommended" in est.recommendation

    def test_minor_speedup(self):
        calc = ROICalculator(gpu_type="a100_80gb", gpu_count=1)
        scenario = _make_scenario(before_time=0.1, after_time=0.095)
        est = calc.estimate(scenario)
        assert est.speedup_factor == pytest.approx(0.1 / 0.095, rel=1e-3)
        assert "Minor" in est.recommendation

    def test_regression(self):
        calc = ROICalculator()
        scenario = _make_scenario(before_time=0.1, after_time=0.15)
        est = calc.estimate(scenario)
        assert est.speedup_factor < 1.0
        assert "Not recommended" in est.recommendation

    def test_zero_step_time(self):
        calc = ROICalculator()
        scenario = _make_scenario(before_time=0.0, after_time=0.05)
        est = calc.estimate(scenario)
        assert est.speedup_factor == 1.0  # empty estimate

    def test_multi_gpu_cost_scaling(self):
        calc1 = ROICalculator(gpu_type="a100_80gb", gpu_count=1)
        calc8 = ROICalculator(gpu_type="a100_80gb", gpu_count=8)
        scenario = _make_scenario(before_time=0.1, after_time=0.05)
        est1 = calc1.estimate(scenario)
        est8 = calc8.estimate(scenario)
        assert est8.cost_saved_per_month == pytest.approx(est1.cost_saved_per_month * 8, rel=1e-2)

    def test_time_saved_per_epoch(self):
        calc = ROICalculator(gpu_type="h100_80gb", steps_per_epoch=10000)
        scenario = _make_scenario(before_time=0.1, after_time=0.05)
        est = calc.estimate(scenario)
        # Before: 10000 * 0.1 / 3600 = 0.278h, After: 10000 * 0.05 / 3600 = 0.139h
        assert est.time_saved_per_epoch_hours == pytest.approx(0.1389, rel=0.01)

    def test_estimate_round_trip(self):
        calc = ROICalculator()
        est = calc.estimate(_make_scenario())
        d = est.to_dict()
        est2 = ROIEstimate.from_dict(d)
        assert est2.scenario_name == est.scenario_name
        assert est2.speedup_factor == pytest.approx(est.speedup_factor)

    def test_memory_delta(self):
        scenario = OptimizationScenario(
            name="test", category="batch_size",
            before_value="bs=32", after_value="bs=64",
            before_step_time_s=0.1, after_step_time_s=0.06,
            before_memory_mb=30000, after_memory_mb=55000,
        )
        calc = ROICalculator()
        est = calc.estimate(scenario)
        assert est.memory_delta_mb == pytest.approx(25000.0)


# ---------------------------------------------------------------------------
# Tests: ROICalculator.analyze
# ---------------------------------------------------------------------------


class TestROIAnalyze:
    def test_empty_scenarios(self):
        calc = ROICalculator()
        report = calc.analyze([])
        assert report.total_speedup_factor == 1.0
        assert report.total_cost_saved_per_month == 0.0
        assert "No optimization" in report.summary

    def test_single_scenario(self):
        calc = ROICalculator(gpu_type="a100_80gb", steps_per_epoch=10000)
        report = calc.analyze([_make_scenario()])
        assert len(report.estimates) == 1
        assert report.total_speedup_factor > 1.0
        assert report.total_cost_saved_per_month > 0
        assert "1/1" in report.summary

    def test_multiple_scenarios(self):
        calc = ROICalculator()
        scenarios = [
            _make_scenario("opt1", 0.1, 0.05),
            _make_scenario("opt2", 0.05, 0.04),
            _make_scenario("opt3", 0.04, 0.06),  # regression
        ]
        report = calc.analyze(scenarios)
        assert len(report.estimates) == 3
        # 2 beneficial
        assert "2/3" in report.summary

    def test_compound_speedup(self):
        calc = ROICalculator()
        scenarios = [
            _make_scenario("opt1", 0.1, 0.05),   # 2x
            _make_scenario("opt2", 0.05, 0.025),  # 2x
        ]
        report = calc.analyze(scenarios)
        assert report.total_speedup_factor == pytest.approx(4.0)

    def test_report_round_trip(self):
        calc = ROICalculator()
        report = calc.analyze([_make_scenario()])
        d = report.to_dict()
        r2 = ROIReport.from_dict(d)
        assert r2.gpu_type == report.gpu_type
        assert len(r2.estimates) == len(report.estimates)


# ---------------------------------------------------------------------------
# Tests: batch_size_roi helper
# ---------------------------------------------------------------------------


class TestBatchSizeROI:
    def test_double_batch_size(self):
        est = ROICalculator.batch_size_roi(
            current_batch_size=32, proposed_batch_size=64,
            current_step_time_s=0.1, gpu_type="a100_80gb",
        )
        assert est.speedup_factor > 1.0
        assert est.cost_saved_per_month > 0
        assert "batch_size_32_to_64" in est.scenario_name

    def test_zero_batch_size(self):
        est = ROICalculator.batch_size_roi(
            current_batch_size=0, proposed_batch_size=64,
            current_step_time_s=0.1,
        )
        assert est.speedup_factor == 1.0

    def test_quadruple_batch_size(self):
        est = ROICalculator.batch_size_roi(
            current_batch_size=16, proposed_batch_size=64,
            current_step_time_s=0.2, gpu_type="h100_80gb", gpu_count=8,
        )
        assert est.cost_saved_per_month > 0


# ---------------------------------------------------------------------------
# Tests: GPU pricing
# ---------------------------------------------------------------------------


class TestGPUPricing:
    def test_known_gpus(self):
        assert "a100_80gb" in GPU_HOURLY_COST
        assert "h100_80gb" in GPU_HOURLY_COST
        assert all(v > 0 for v in GPU_HOURLY_COST.values())

    def test_unknown_gpu_fallback(self):
        calc = ROICalculator(gpu_type="unknown_gpu")
        est = calc.estimate(_make_scenario())
        # Should still work with fallback pricing
        assert est.cost_saved_per_month > 0
