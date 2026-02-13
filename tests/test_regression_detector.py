"""Tests for src.profiler.regression_detector."""

import json
import pytest
from pathlib import Path

from src.profiler.regression_detector import (
    Baseline,
    CurrentRunMetrics,
    RegressionDetector,
    RegressionFlag,
    RegressionReport,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_baseline(
    throughput: float = 1000.0,
    step_time: float = 0.1,
    peak_mem: float = 40000.0,
    util: float = 80.0,
    comm: float = 5.0,
) -> Baseline:
    return Baseline(
        name="v1.0-baseline",
        timestamp="2025-01-15T00:00:00Z",
        avg_step_time_s=step_time,
        throughput_samples_per_sec=throughput,
        peak_memory_mb=peak_mem,
        avg_gpu_utilization_pct=util,
        avg_communication_overhead_pct=comm,
    )


def _make_current(
    throughput: float = 1000.0,
    step_time: float = 0.1,
    peak_mem: float = 40000.0,
    util: float = 80.0,
    comm: float = 5.0,
) -> CurrentRunMetrics:
    return CurrentRunMetrics(
        avg_step_time_s=step_time,
        throughput_samples_per_sec=throughput,
        peak_memory_mb=peak_mem,
        avg_gpu_utilization_pct=util,
        avg_communication_overhead_pct=comm,
    )


# ---------------------------------------------------------------------------
# Tests: Baseline
# ---------------------------------------------------------------------------


class TestBaseline:
    def test_round_trip(self):
        b = _make_baseline()
        d = b.to_dict()
        b2 = Baseline.from_dict(d)
        assert b2.name == "v1.0-baseline"
        assert b2.throughput_samples_per_sec == pytest.approx(1000.0)

    def test_save_and_load(self, tmp_path: Path):
        b = _make_baseline()
        p = tmp_path / "baseline.json"
        b.save(p)
        b2 = Baseline.load(p)
        assert b2.name == b.name
        assert b2.avg_step_time_s == pytest.approx(b.avg_step_time_s)
        assert b2.peak_memory_mb == pytest.approx(b.peak_memory_mb)

    def test_metadata(self):
        b = Baseline(
            name="test", timestamp="2025-01-01",
            avg_step_time_s=0.1, throughput_samples_per_sec=500.0,
            peak_memory_mb=20000, avg_gpu_utilization_pct=70.0,
            metadata={"model": "llama-7b", "framework": "pytorch"},
        )
        d = b.to_dict()
        assert d["metadata"]["model"] == "llama-7b"


# ---------------------------------------------------------------------------
# Tests: CurrentRunMetrics
# ---------------------------------------------------------------------------


class TestCurrentRunMetrics:
    def test_round_trip(self):
        c = _make_current()
        d = c.to_dict()
        c2 = CurrentRunMetrics.from_dict(d)
        assert c2.throughput_samples_per_sec == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# Tests: RegressionDetector
# ---------------------------------------------------------------------------


class TestRegressionDetector:
    def test_no_change(self):
        det = RegressionDetector()
        report = det.compare(_make_baseline(), _make_current())
        assert not report.has_regressions
        assert report.regression_count == 0

    def test_throughput_regression(self):
        det = RegressionDetector()
        baseline = _make_baseline(throughput=1000.0)
        current = _make_current(throughput=900.0)  # 10% drop
        report = det.compare(baseline, current)
        assert report.has_regressions
        tp_flags = [f for f in report.flags if f.metric == "throughput"]
        assert tp_flags[0].severity == "regression"
        assert tp_flags[0].delta_pct == pytest.approx(-10.0)

    def test_throughput_improvement(self):
        det = RegressionDetector()
        baseline = _make_baseline(throughput=1000.0)
        current = _make_current(throughput=1200.0)  # 20% increase
        report = det.compare(baseline, current)
        tp_flags = [f for f in report.flags if f.metric == "throughput"]
        assert tp_flags[0].severity == "improvement"

    def test_memory_regression(self):
        det = RegressionDetector()
        baseline = _make_baseline(peak_mem=40000.0)
        current = _make_current(peak_mem=48000.0)  # 20% increase
        report = det.compare(baseline, current)
        mem_flags = [f for f in report.flags if f.metric == "peak_memory"]
        assert mem_flags[0].severity == "regression"

    def test_step_time_regression(self):
        det = RegressionDetector()
        baseline = _make_baseline(step_time=0.1)
        current = _make_current(step_time=0.12)  # 20% slower
        report = det.compare(baseline, current)
        st_flags = [f for f in report.flags if f.metric == "avg_step_time"]
        assert st_flags[0].severity == "regression"

    def test_utilization_regression(self):
        det = RegressionDetector()
        baseline = _make_baseline(util=80.0)
        current = _make_current(util=60.0)  # 25% drop
        report = det.compare(baseline, current)
        u_flags = [f for f in report.flags if f.metric == "avg_gpu_utilization"]
        assert u_flags[0].severity == "regression"

    def test_communication_overhead_regression(self):
        det = RegressionDetector()
        baseline = _make_baseline(comm=5.0)
        current = _make_current(comm=10.0)  # doubled
        report = det.compare(baseline, current)
        c_flags = [f for f in report.flags if f.metric == "communication_overhead"]
        assert c_flags[0].severity == "regression"

    def test_custom_thresholds(self):
        det = RegressionDetector(throughput_threshold_pct=20.0)
        baseline = _make_baseline(throughput=1000.0)
        current = _make_current(throughput=850.0)  # 15% drop, under 20% threshold
        report = det.compare(baseline, current)
        tp_flags = [f for f in report.flags if f.metric == "throughput"]
        assert tp_flags[0].severity == "unchanged"

    def test_zero_baseline_unchanged(self):
        det = RegressionDetector()
        baseline = _make_baseline(throughput=0.0)
        current = _make_current(throughput=500.0)
        report = det.compare(baseline, current)
        tp_flags = [f for f in report.flags if f.metric == "throughput"]
        assert tp_flags[0].severity == "unchanged"

    def test_multiple_regressions(self):
        det = RegressionDetector()
        baseline = _make_baseline(throughput=1000, step_time=0.1, peak_mem=40000)
        current = _make_current(throughput=800, step_time=0.15, peak_mem=50000)
        report = det.compare(baseline, current)
        assert report.regression_count >= 3
        assert "REGRESSIONS" in report.summary

    def test_compare_from_file(self, tmp_path: Path):
        baseline = _make_baseline()
        p = tmp_path / "baseline.json"
        baseline.save(p)
        current = _make_current(throughput=800.0)
        det = RegressionDetector()
        report = det.compare_from_file(p, current)
        assert report.has_regressions

    def test_report_round_trip(self):
        det = RegressionDetector()
        report = det.compare(_make_baseline(), _make_current(throughput=800))
        d = report.to_dict()
        r2 = RegressionReport.from_dict(d)
        assert r2.baseline_name == "v1.0-baseline"
        assert len(r2.flags) == len(report.flags)

    def test_no_comm_flag_when_both_zero(self):
        det = RegressionDetector()
        baseline = _make_baseline(comm=0.0)
        current = _make_current(comm=0.0)
        report = det.compare(baseline, current)
        c_flags = [f for f in report.flags if f.metric == "communication_overhead"]
        assert len(c_flags) == 0
